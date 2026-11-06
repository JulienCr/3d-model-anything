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


@pytest.fixture(scope="module")
def short(tmp_root):
    ctx, truth = make_case(tmp_root, "short", short_beam=True)
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def patch(tmp_root):
    ctx, truth = make_case(tmp_root, "patch", ceiling_patch=True, patch_drop=0.07, short_beam=True, short_beam_underside=2.16)
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def stout(tmp_root):
    ctx, truth = make_case(tmp_root, "stout", short_beam=True, short_beam_width=0.55)
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def panel(tmp_root):
    ctx, truth = make_case(tmp_root, "panel", ceiling_patch=True, patch_drop=0.07, patch_length=1.62)
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def raised(tmp_root):
    ctx, truth = make_case(tmp_root, "raised", ceiling_patch=True, patch_drop=-0.07, patch_length=3.0, patch_width=0.6)
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def ripple(tmp_root):
    ctx, truth = make_case(tmp_root, "ripple", ceiling_patch=True, patch_drop=0.035, patch_length=1.62)
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def shallow(tmp_root):
    ctx, truth = make_case(tmp_root, "shallow", short_beam=True, short_beam_underside=2.40)
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def fixtures(tmp_root):
    ctx, truth = make_case(tmp_root, "fixtures", suspended_tubes=True, wall_duct=True)
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
    assert np.linalg.norm(b - a) == pytest.approx(2.50, abs=0.05)
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


def _beam_near(doc, truth, ctx, x, z):
    point = plan_point(truth_to_room(ctx, truth), x, z)
    return min(_elements(doc, "beam"), key=lambda b: np.linalg.norm(np.mean(b["axis"], axis=0) - point))


def test_beam_length_is_its_own_extent(short):
    ctx, truth, doc = short
    assert len(_elements(doc, "beam")) == 2
    beam = _beam_near(doc, truth, ctx, 5.05, 1.20)
    a, b = (np.array(pt) for pt in beam["axis"])
    assert np.linalg.norm(b - a) == pytest.approx(1.40, abs=0.05)
    assert np.linalg.norm((a + b) / 2 - plan_point(truth_to_room(ctx, truth), 5.05, 1.20)) < 0.05
    assert doc["parameters"][beam["params"]["underside"]]["value"] - _floor(doc) == pytest.approx(2.00, abs=0.01)
    full = _beam_near(doc, truth, ctx, 4.0, 1.25)
    assert np.linalg.norm(np.array(full["axis"][1]) - full["axis"][0]) == pytest.approx(2.50, abs=0.05)


def test_levels_seven_centimetres_apart_stay_separate(patch):
    ctx, truth, doc = patch
    heights = sorted(_value(doc, c["params"]["level"]) - _floor(doc) for c in _elements(doc, "ceiling"))
    assert heights == pytest.approx([2.30, 2.43, 2.50], abs=0.008)
    T = truth_to_room(ctx, truth)
    panel = shapely.Polygon([plan_point(T, x, z) for x, z in ((2.0, 1.5), (3.0, 1.5), (3.0, 2.4), (2.0, 2.4))])
    region = _region(next(c for c in _elements(doc, "ceiling") if abs(_value(doc, c["params"]["level"]) - _floor(doc) - 2.43) < 0.01))
    assert region.intersection(panel).area / panel.area >= 0.85
    assert region.area < 1.3
    undersides = sorted(_value(doc, b["params"]["underside"]) - _floor(doc) for b in _elements(doc, "beam"))
    assert undersides == pytest.approx([2.16, 2.20], abs=0.01)


def test_stout_beam_is_still_a_beam(stout):
    ctx, truth, doc = stout
    assert len(_elements(doc, "beam")) == 2
    beam = _beam_near(doc, truth, ctx, 5.175, 1.20)
    a, b = (np.array(pt) for pt in beam["axis"])
    assert np.linalg.norm(b - a) == pytest.approx(1.40, abs=0.05)
    assert doc["parameters"][beam["params"]["width"]]["value"] == pytest.approx(0.55, abs=0.03)
    assert len(_elements(doc, "ceiling")) == 2


def test_fixtures_and_ducts_are_suspended(fixtures):
    ctx, truth, doc = fixtures
    (beam,) = _elements(doc, "beam")
    assert beam["flags"] == []
    assert doc["parameters"][beam["params"]["width"]]["value"] == pytest.approx(0.30, abs=0.02)
    items = doc["diagnostics"]["unexplained_horizontal"]
    assert len(items) == 6
    assert all(set(i) == {"bbox", "underside", "width", "length", "reason"} and i["reason"] == "suspended" for i in items)
    T = truth_to_room(ctx, truth)
    duct = plan_point(T, 0.27, 1.30)
    hits = [i for i in items if shapely.box(*i["bbox"][0], *i["bbox"][1]).distance(shapely.Point(duct)) < 0.1]
    assert len(hits) == 1 and hits[0]["length"] == pytest.approx(2.0, abs=0.15) and hits[0]["width"] >= 0.15
    tubes = [i for i in items if i not in hits]
    assert all(i["length"] == pytest.approx(1.2, abs=0.1) and i["underside"] - _floor(doc) == pytest.approx(2.25, abs=0.02) for i in tubes)


def test_default_scene_has_no_unexplained_horizontal(default):
    assert default[2]["diagnostics"]["unexplained_horizontal"] == []


def test_shallow_strip_is_not_a_beam(shallow):
    _, _, doc = shallow
    assert len(_elements(doc, "beam")) == 1
    (item,) = doc["diagnostics"]["unexplained_horizontal"]
    assert item["reason"] == "too_shallow"



def test_ripple_of_a_few_centimetres_is_one_level(ripple, default):
    _, _, doc = ripple
    levels = _elements(doc, "ceiling")
    assert len(levels) == 2
    assert not _elements(doc, "beam")[1:] and not doc["diagnostics"]["unexplained_horizontal"]
    top = _value(doc, levels[0]["params"]["level"]) - _floor(doc)
    assert top == pytest.approx(2.50 - 0.035 * 0.1, abs=0.015)
    assert len(_elements(default[2], "ceiling")) == 2


def test_noisy_ceiling_stays_one_level_per_height(tmp_root):
    ctx, _ = make_case(tmp_root, "noisy", noise_m=0.01)
    assert len(_elements(shell.build(ctx), "ceiling")) == 2


def test_lowered_panel_of_ratio_1_8_is_not_a_beam(panel):
    _, _, doc = panel
    assert len(_elements(doc, "beam")) == 1
    assert doc["diagnostics"]["unexplained_horizontal"] == []
    assert len(_elements(doc, "ceiling")) == 3


def test_strip_of_the_highest_level_is_ceiling_not_unexplained(raised):
    _, _, doc = raised
    assert doc["diagnostics"]["unexplained_horizontal"] == []
    assert len(_elements(doc, "beam")) == 1
    assert len(_elements(doc, "ceiling")) == 3


def _verdict(tmp_root, name, **options):
    ctx, truth = make_case(tmp_root, name, short_beam=True, **options)
    doc = shell.build(ctx)
    beams = [b for b in _elements(doc, "beam") if abs(_value(doc, b["params"]["underside"]) - _floor(doc) - options.get("short_beam_underside", 2.0)) < 0.02]
    return doc, beams, doc["diagnostics"]["unexplained_horizontal"]


def test_hanging_strip_with_visible_ceiling_is_suspended(tmp_root):
    ctx, _ = make_case(tmp_root, "hanging", hanging_strip=True)
    doc = shell.build(ctx)
    (item,) = doc["diagnostics"]["unexplained_horizontal"]
    assert item["reason"] == "suspended" and item["width"] >= 0.2
    assert len(_elements(doc, "beam")) == 1


def test_beam_with_an_unseen_side_is_to_verify(tmp_root, default):
    doc, beams, unexplained = _verdict(tmp_root, "oneside", short_beam_sides=1)
    (beam,) = beams
    assert beam["flags"] == ["to_verify"] and unexplained == []
    assert all(doc["parameters"][v]["source"] == "lidar" for v in beam["params"].values())
    assert default[2]["diagnostics"]["unexplained_horizontal"] == [] and _elements(default[2], "beam")[0]["flags"] == []


def test_narrow_strip_without_ceiling_above_is_too_narrow(tmp_root):
    _, beams, unexplained = _verdict(tmp_root, "narrow", short_beam_width=0.13, short_beam_long=True)
    assert not beams and [u["reason"] for u in unexplained] == ["too_narrow"]


@pytest.mark.parametrize("angle, flags", [(25.0, []), (55.0, ["to_verify"])])
def test_bevelled_faces_bracket_the_steepness_limit(tmp_root, angle, flags):
    _, beams, unexplained = _verdict(tmp_root, f"bevel{int(angle)}", short_beam_underside=2.35, short_beam_bevel_deg=angle)
    assert unexplained == [] and [b["flags"] for b in beams] == [flags]


def test_chain_of_small_steps_does_not_merge_into_one_level(tmp_root):
    ctx, _ = make_case(tmp_root, "stairs", staircase=True)
    doc = shell.build(ctx)
    assert len(_elements(doc, "ceiling")) == 4
    assert max(abs(_value(doc, c["params"]["level"]) - _floor(doc) - 2.37) for c in _elements(doc, "ceiling")) < 0.2


@pytest.mark.parametrize("length, expected", [(1.5, 2.57), (1.0, 2.57), (0.7, 2.50)])
def test_wall_rises_to_a_higher_ceiling_over_30_cm_but_not_a_clip(tmp_root, length, expected):
    ctx, truth = make_case(tmp_root, f"clip-{length}", ceiling_patch=True, patch_drop=-0.07, patch_length=length)
    doc = shell.build(ctx)
    top = doc["parameters"][f"wall.{wall_labels(doc, truth, 'step')[0]}.top"]
    assert top["value"] - _floor(doc) == pytest.approx(expected, abs=0.01)
