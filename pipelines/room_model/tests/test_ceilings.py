import numpy as np
import pytest
import shapely

from room_model import ceilings, shell
from room_model.frame import world_to_room
from room_model.lidar import Samples, load_samples
from helpers import make_case, plan_point, truth_to_room, wall_labels, wall_line


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("ceilings")


@pytest.fixture(scope="module")
def default(tmp_root):
    ctx, truth = make_case(tmp_root, "default")
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def hole(tmp_root):
    ctx, truth = make_case(tmp_root, "hole", ceiling_hole=True)
    return ctx, truth, shell.build(ctx)


def _floor(doc):
    return doc["parameters"]["floor.level"]["value"]


def _elements(doc, kind):
    return [e for e in doc["elements"] if e["kind"] == kind]


def _value(doc, name):
    return doc["parameters"][name]["value"]


def _region(element):
    return shapely.MultiPolygon([shapely.Polygon(r["exterior"], r["holes"]) for r in element["region"]])


def test_two_ceiling_levels(default):
    ctx, truth, doc = default
    ceilings = _elements(doc, "ceiling")
    assert [c["id"] for c in ceilings] == ["Ceiling_1", "Ceiling_2"]
    heights = [_value(doc, c["params"]["level"]) - _floor(doc) for c in ceilings]
    assert heights == pytest.approx([2.50, 2.30], abs=0.01)
    T = truth_to_room(ctx, truth)
    wing = shapely.Polygon([plan_point(T, x, z) for x, z in ((0, 2.5), (2.5, 2.5), (2.5, 4.5), (0, 4.5))])
    low, high = _region(ceilings[1]), _region(ceilings[0])
    assert low.intersection(wing).area / wing.area >= 0.90
    assert high.intersection(wing).area / wing.area < 0.02


def test_beam_recovered(default):
    ctx, truth, doc = default
    (beam,) = _elements(doc, "beam")
    floor = _floor(doc)
    p = {k: doc["parameters"][name] for k, name in beam["params"].items()}
    assert p["underside"]["value"] - floor == pytest.approx(2.20, abs=0.01)
    assert p["top"]["value"] - floor == pytest.approx(2.50, abs=0.01)
    assert p["width"]["value"] == pytest.approx(0.30, abs=0.02)
    assert all(v["source"] == "lidar" and v["unit"] == "m" and v["sigma"] > 0 for v in p.values())
    a, b = (np.array(pt) for pt in beam["axis"])
    east = wall_line(doc, wall_labels(doc, truth, "east")[0])
    direction = np.array([-east["normal"][1], east["normal"][0]])
    assert np.degrees(np.arccos(abs((b - a) @ direction) / np.linalg.norm(b - a))) < 1.0
    assert np.linalg.norm(b - a) >= 2.3
    west = wall_line(doc, wall_labels(doc, truth, "west")[0])
    assert abs(west["normal"] @ ((a + b) / 2) - west["offset"]) == pytest.approx(4.00, abs=0.02)


def test_wall_tops_follow_ceilings(default):
    _, truth, doc = default
    expected = {"north": 2.30, "wing_east": 2.30, "south": 2.50, "east": 2.50}
    for name, height in expected.items():
        top = doc["parameters"][f"wall.{wall_labels(doc, truth, name)[0]}.top"]
        assert top["source"] == "lidar", name
        assert top["value"] - _floor(doc) == pytest.approx(height, abs=0.01), name


def test_ceiling_hole_creates_no_beam(default, hole):
    doc = hole[2]
    assert len(_elements(doc, "beam")) == 1
    assert len(_elements(doc, "ceiling")) == 2
    unobserved = doc["diagnostics"]["ceiling_unobserved_fraction"]
    assert unobserved > 0.04
    assert unobserved > default[2]["diagnostics"]["ceiling_unobserved_fraction"] + 0.03


def _main_area(doc):
    return _region(_elements(doc, "ceiling")[0]).area


def test_unobserved_fraction_is_small_on_a_scanned_ceiling(default):
    assert default[2]["diagnostics"]["ceiling_unobserved_fraction"] < 0.03


def test_ceiling_hole_is_measured(default, hole):
    gain = hole[2]["diagnostics"]["ceiling_unobserved_fraction"] - default[2]["diagnostics"]["ceiling_unobserved_fraction"]
    assert 0.035 <= gain <= 0.065
    assert 0.8 <= _main_area(default[2]) - _main_area(hole[2]) <= 1.2


def test_no_ceiling_samples_gives_no_levels(default):
    ctx, _, doc = default
    samples = load_samples(ctx, world_to_room(ctx.capture, ctx.room, ctx.turn))
    keep = samples.N[:, 1] >= -0.5
    stripped = Samples(samples.P[keep], samples.N[keep], samples.cls[keep])
    polygon = shapely.Polygon(next(e for e in doc["elements"] if e["kind"] == "floor")["polygon"])
    result = ceilings.detect(stripped, polygon, doc["parameters"]["floor.level"]["value"])
    assert result.levels == [] and result.beams == []
    assert result.unobserved_fraction == 1.0
