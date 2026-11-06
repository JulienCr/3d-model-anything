import json

import numpy as np
import pytest

from room_model import shell
from room_model.frame import world_to_room
from room_model.lidar import load_samples
from helpers import make_case as _case, truth_to_room as _truth_to_room, wall_labels as _labels, wall_line as _line


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("shell")


@pytest.fixture(scope="module")
def default(tmp_root):
    ctx, truth = _case(tmp_root, "default")
    path = shell.run(ctx)
    return ctx, truth, json.loads(path.read_text()), path


@pytest.fixture(scope="module")
def no_panel(tmp_root):
    ctx, _ = _case(tmp_root, "no-panel", other_room_panel=False)
    return shell.build(ctx)


@pytest.fixture(scope="module")
def no_wing(tmp_root):
    ctx, truth = _case(tmp_root, "no-wing", drop_wall_faces="wing_east")
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def split(tmp_root):
    ctx, truth = _case(tmp_root, "split", split_south_wall=True)
    return ctx, truth, shell.build(ctx)


def _distance(doc, truth, name_a, name_b):
    """Distance between two fitted walls, measured at the middle of their overlap."""
    a, b = (_line(doc, _labels(doc, truth, n)[0]) for n in (name_a, name_b))
    d = np.array([-a["normal"][1], a["normal"][0]])
    spans = [sorted([(w["start"] - a["normal"] * a["offset"]) @ d, (w["end"] - a["normal"] * a["offset"]) @ d]) for w in (a, b)]
    mid = (max(s[0] for s in spans) + min(s[1] for s in spans)) / 2
    point = a["normal"] * a["offset"] + mid * d
    return abs(b["normal"] @ point - b["offset"])


def _sigma(doc, label, what):
    return doc["parameters"][f"wall.{label}.{what}"]["sigma"]


def test_widths_recovered(default):
    ctx, truth, doc, _ = default
    for (a, b), true_width in truth.widths.items():
        la, lb = _labels(doc, truth, a)[0], _labels(doc, truth, b)[0]
        err = abs(_distance(doc, truth, a, b) - true_width)
        bound = 3 * np.hypot(_sigma(doc, la, "offset"), _sigma(doc, lb, "offset"))
        assert err < 0.01 and err < bound, (a, b, err)


def test_well_supported_wall_sigma_is_tight(default):
    _, _, doc, _ = default
    strong = [k for k, d in doc["diagnostics"]["walls"].items() if d["support_m2"] > 2]
    assert strong
    for label in strong:
        assert doc["parameters"][f"wall.{label}.offset"]["source"] == "lidar"
        assert _sigma(doc, label, "offset") < 0.015


def test_floor_level(default):
    ctx, truth, doc, _ = default
    assert abs(doc["parameters"]["floor.level"]["value"] - _truth_to_room(ctx, truth)[1, 3]) < 0.005
    assert doc["diagnostics"]["floor"]["tilt_deg"] < 0.5


def test_clutter_does_not_bias_west_wall(default):
    ctx, truth, doc, _ = default
    T = _truth_to_room(ctx, truth)
    n = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    n_xz = n[[0, 2]] / np.linalg.norm(n[[0, 2]])
    true_offset = n_xz @ T[[0, 2], 3]
    west = _line(doc, _labels(doc, truth, "west")[0])
    assert west["normal"] @ n_xz > 0
    assert abs(west["offset"] - true_offset) < 0.005


def test_missing_wall_falls_back_to_roomplan(no_wing):
    _, truth, doc = no_wing
    label = _labels(doc, truth, "wing_east")[0]
    assert doc["parameters"][f"wall.{label}.offset"]["source"] == "roomplan"
    assert _sigma(doc, label, "offset") >= 0.05
    entry = doc["diagnostics"]["walls"][label]
    assert entry["n_points"] == 0 and len(entry["centroid"]) == 2
    assert set(doc["diagnostics"]["walls"]) == {e["label"] for e in doc["elements"] if e["kind"] == "wall"}
    assert "roomplan_fallback" in next(e for e in doc["elements"] if e.get("label") == label)["flags"]
    for (a, b), true_width in truth.widths.items():
        if "wing_east" not in (a, b):
            assert abs(_distance(doc, truth, a, b) - true_width) < 0.01, (a, b)


def test_split_wall_corner_is_stable(split):
    ctx, truth, doc = split
    walls = [e for e in doc["elements"] if e["kind"] == "wall"]
    assert len(walls) == 7
    first, second = (next(w for w in walls if w["label"] == _labels(doc, truth, "south")[i]) for i in (0, 1))
    if walls.index(first) > walls.index(second):
        first, second = second, first
    corner = np.array(first["end"])
    assert np.allclose(corner, second["start"])
    T = _truth_to_room(ctx, truth)
    south_n = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    south_n = south_n[[0, 2]] / np.linalg.norm(south_n[[0, 2]])
    assert abs(south_n @ (corner - T[[0, 2], 3])) < 0.02
    assert np.linalg.norm(corner - (T @ [*truth.split_point[:1], 0.0, truth.split_point[1], 1.0])[[0, 2]]) < 0.05
    assert "near_parallel_corner" in first["flags"] and "near_parallel_corner" in second["flags"]


def test_other_room_faces_ignored(default, no_panel):
    _, truth, doc, _ = default
    label = _labels(doc, truth, "south")[0]
    a = doc["parameters"][f"wall.{label}.offset"]["value"]
    b = no_panel["parameters"][f"wall.{label}.offset"]["value"]
    assert abs(a - b) < 0.001


def test_shell_document_format(default):
    ctx, _, doc, path = default
    assert path == ctx.out / "shell.json"
    assert doc["schema"] == "room-model/shell-1"
    assert len(doc["frame"]["world_to_room"]) == 16
    kinds = [e["kind"] for e in doc["elements"]]
    assert kinds == ["floor"] + ["wall"] * 6 + ["ceiling"] * 2 + ["beam"]
    for el in doc["elements"]:
        for name in el["params"].values():
            assert name in doc["parameters"]
    assert all(p["source"] in ("lidar", "roomplan", "assumed") and p["unit"] in ("m", "rad") for p in doc["parameters"].values())
    assert len(doc["elements"][0]["polygon"]) == 6
    assert set(doc["diagnostics"]) >= {"walls", "floor", "ceilings", "unexplained_vertical", "off_outline_walls", "flags"}
    diag = next(iter(doc["diagnostics"]["walls"].values()))
    assert {"rms_cm", "support_m2", "offset_vs_roomplan_cm", "angle_vs_roomplan_deg", "tilt_deg", "n_points", "centroid"} <= set(diag)


def test_samples_drop_other_room_faces(default):
    ctx, truth, _, _ = default
    w2r = world_to_room(ctx.capture, ctx.room, ctx.turn)
    samples = load_samples(ctx, w2r)
    local = (samples.P - (w2r @ truth.world_from_local)[:3, 3]) @ (w2r @ truth.world_from_local)[:3, :3]
    panel = (np.abs(local[:, 2] + 0.05) < 0.03) & (local[:, 0] > 1) & (local[:, 0] < 3) & (local[:, 1] < 2)
    assert not panel.any()
    assert len(samples.P) > 100_000
