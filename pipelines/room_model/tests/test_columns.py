import numpy as np
import pytest
import shapely

from room_model import shell
from helpers import make_case, plan_point, truth_to_room, wall_labels, wall_line


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    ctx, truth = make_case(tmp_path_factory.mktemp("columns"), "default")
    return ctx, truth, shell.build(ctx)


def _columns(doc):
    return [e for e in doc["elements"] if e["kind"] == "column"]


def _p(doc, column, name):
    return doc["parameters"][column["params"][name]]


def _dist(doc, truth, name, point):
    w = wall_line(doc, wall_labels(doc, truth, name)[0])
    return abs(w["normal"] @ point - w["offset"])


def _footprint(scene, x, z):
    ctx, truth, _ = scene
    T = truth_to_room(ctx, truth)
    return shapely.Polygon([plan_point(T, a, b) for a, b in ((x[0], z[0]), (x[1], z[0]), (x[1], z[1]), (x[0], z[1]))])


def _bbox(entry):
    return shapely.box(*entry["bbox"][0], *entry["bbox"][1])


def _center(doc, column):
    return np.array([_p(doc, column, "center_x")["value"], _p(doc, column, "center_z")["value"]])


def test_free_standing_column(scene):
    _, truth, doc = scene
    (col,) = [c for c in _columns(doc) if c["engaged_wall"] is None]
    assert [_p(doc, col, n)["value"] for n in ("size_x", "size_z")] == pytest.approx([0.40, 0.40], abs=0.01)
    c = _center(doc, col)
    assert _dist(doc, truth, "west", c) == pytest.approx(1.50, abs=0.02)
    assert _dist(doc, truth, "south", c) == pytest.approx(1.20, abs=0.02)
    top = _p(doc, col, "top")
    assert top["value"] - doc["parameters"]["floor.level"]["value"] == pytest.approx(2.50, abs=0.01)
    assert all(_p(doc, col, n)["source"] == "lidar" and _p(doc, col, n)["unit"] == "m" for n in col["params"])


def test_engaged_column(scene):
    _, truth, doc = scene
    south = wall_labels(doc, truth, "south")[0]
    (col,) = [c for c in _columns(doc) if c["engaged_wall"] is not None]
    assert col["engaged_wall"] == south
    w = wall_line(doc, south)
    along = np.array([-w["normal"][1], w["normal"][0]])
    size = np.array([_p(doc, col, "size_x")["value"], _p(doc, col, "size_z")["value"]])
    assert abs(along @ size) == pytest.approx(0.50, abs=0.01)
    assert abs(w["normal"] @ size) == pytest.approx(0.25, abs=0.01)
    c = _center(doc, col)
    assert _dist(doc, truth, "south", c) == pytest.approx(0.125, abs=0.02)
    assert _dist(doc, truth, "west", c) == pytest.approx(5.25, abs=0.02)


def test_tall_cabinet_is_not_a_column(scene):
    doc = scene[2]
    assert len(_columns(doc)) == 2
    cabinet = _footprint(scene, (0.0, 0.6), (3.0, 4.0))
    hits = [u for u in doc["diagnostics"]["unexplained_vertical"] if _bbox(u).intersects(cabinet)]
    assert len(hits) == 1
    assert hits[0]["height_max"] == pytest.approx(1.90, abs=0.05)
    assert hits[0]["reason"] == "not_full_height"


def test_beam_is_not_a_column(scene):
    doc = scene[2]
    beam = _footprint(scene, (3.85, 4.15), (0.0, 2.5))
    assert not [c for c in _columns(doc) if shapely.Point(_center(doc, c)).intersects(beam)]
    assert not [u for u in doc["diagnostics"]["unexplained_vertical"] if _bbox(u).intersects(beam)]


def test_no_ceiling_level_skips_detection(scene):
    ctx, _, doc = scene
    from room_model import columns
    from room_model.frame import world_to_room
    from room_model.lidar import load_samples

    samples = load_samples(ctx, world_to_room(ctx.capture, ctx.room, ctx.turn))
    polygon = shapely.Polygon(next(e for e in doc["elements"] if e["kind"] == "floor")["polygon"])
    result = columns.detect(samples, shell.wall_lines(doc), polygon, doc["parameters"]["floor.level"]["value"], [])
    assert result.columns == [] and result.unexplained == []
