"""Plan geometry of the shell (blender/arch/plan.py) without Blender: joins, gaps, fallbacks, partition trims."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender"))
from arch import clip, openings, plan  # noqa: E402

from room_model import meshes  # noqa: E402

THICKNESS = 0.2


def _param(arch, pid, value):
    arch["parameters"][pid] = {"value": value}
    return pid


def _wall(arch, kind, id_, start, end, thickness, normal, offset=None):
    n = np.asarray(normal, float)
    a = np.asarray(start, float)
    offset = float(n @ a) if offset is None else offset
    arch["elements"].append({"id": id_, "kind": kind, "start": list(start), "end": list(end), "params": {
        "azimuth": _param(arch, f"{id_}.az", math.atan2(n[1], n[0])), "offset": _param(arch, f"{id_}.off", offset),
        "thickness": _param(arch, f"{id_}.t", thickness), "top": _param(arch, f"{id_}.top", 2.5)}})


def _ring(points, thickness=THICKNESS, offsets=None):
    """Walls round a counter-clockwise polygon of (x, z) points, interior on the left; `offsets` moves a face line."""
    arch = {"parameters": {}, "elements": []}
    for k, a in enumerate(points):
        b = points[(k + 1) % len(points)]
        u = (np.array(b) - np.array(a)) / math.dist(a, b)
        n = np.array([-u[1], u[0]])
        _wall(arch, "wall", f"W{k}", a, b, thickness, n, float(n @ np.array(a)) + (offsets or {}).get(k, 0.0))
    return arch


def _twin(arch):
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    inputs = [{"start": w["start"], "end": w["end"], "normal": [math.cos(arch["parameters"][w["params"]["azimuth"]]["value"]),
                                                                  math.sin(arch["parameters"][w["params"]["azimuth"]]["value"])],
               "offset": arch["parameters"][w["params"]["offset"]]["value"], "thickness": arch["parameters"][w["params"]["thickness"]]["value"]}
              for w in walls]
    return {w["id"]: f for w, f in zip(walls, meshes.mitred_wall_footprints(inputs))}


def _same_as_twin(arch):
    own = plan.wall_footprints(arch)
    for wid, expected in _twin(arch).items():
        assert np.array(own[wid]) == pytest.approx(expected, abs=1e-9), wid
    return own


RECT = [(0, 0), (4, 0), (4, 3), (0, 3)]
L_SHAPE = [(0, 0), (4, 0), (4, 1.5), (2, 1.5), (2, 3), (0, 3)]


@pytest.mark.parametrize("points", [RECT, L_SHAPE])
def test_right_angle_rings_are_cut_along_one_line_per_corner(points):
    own = _same_as_twin(_ring(points))
    polygons = [shapely.Polygon(f) for f in own.values()]
    assert all(p.is_valid for p in polygons)
    assert sum(p.area for p in polygons) == pytest.approx(shapely.union_all(polygons).area, abs=1e-9), "no overlap"
    assert shapely.union_all(polygons).geom_type == "Polygon", "no gap"


def test_near_parallel_corner_with_a_jog_has_no_overlap_and_no_gap():
    arch = _ring([(0, 0), (4, 0), (4, 3), (2.0, 3.0), (0.0, 3.0)], offsets={2: -0.1})  # W2 and W3 are collinear, W2 stepped back
    own = _same_as_twin(arch)
    polygons = [shapely.Polygon(f) for f in own.values()]
    assert sum(p.area for p in polygons) == pytest.approx(shapely.union_all(polygons).area, abs=1e-9)
    assert shapely.union_all(polygons).geom_type == "Polygon"


def test_a_jog_larger_than_the_wall_is_closed_by_one_shared_edge():
    arch = _ring([(0, 0), (4, 0), (4, 3), (2.0, 3.0), (0.0, 3.0)], offsets={2: -0.35})
    own = _same_as_twin(arch)
    a, b = shapely.Polygon(own["W2"]), shapely.Polygon(own["W3"])
    assert a.is_valid and b.is_valid
    assert a.intersection(b).area == pytest.approx(0.0, abs=1e-9)
    assert shapely.union_all([a, b]).geom_type == "Polygon", "no hole between the two bodies"
    assert a.intersection(b).length > THICKNESS, "they meet along an edge spanning the jog"


def test_a_wall_too_short_for_its_mitres_keeps_square_ends():
    arch = _ring([(0, 0), (4, 0), (4, 3), (2.05, 3), (2.05, 2.9), (1.95, 2.9), (1.95, 3), (0, 3)])  # a 10 cm pilaster
    own = _same_as_twin(arch)
    short = shapely.Polygon(own["W4"])
    assert short.is_valid and short.area == pytest.approx(0.1 * THICKNESS, abs=1e-9)
    assert np.array(own["W4"])[:, 0] == pytest.approx([2.05, 1.95, 1.95, 2.05], abs=1e-9)


def _partitions(order):
    arch = {"parameters": {}, "elements": []}
    specs = {"Bar": ((0.5, 2.0), (3.5, 2.0), (0.0, 1.0)),  # runs along x, its body behind the face z = 2
             "Stem": ((2.0, 0.5), (2.0, 2.04), (1.0, 0.0))}  # comes from below and goes 4 cm through the bar
    for key in order:
        start, end, normal = specs[key]
        _wall(arch, "partition", f"Partition_{key}", start, end, 0.1, normal)
    return arch


@pytest.mark.parametrize("order", [("Bar", "Stem"), ("Stem", "Bar")])
def test_a_t_junction_is_trimmed_whatever_the_order(order):
    arch = _partitions(order)
    out = plan.partition_footprints(arch, {})
    stem, bar = np.array(out["Partition_Stem"]), np.array(out["Partition_Bar"])
    assert stem[:, 1].max() == pytest.approx(1.9, abs=1e-9), "the stem ends on the near face of the bar"
    assert bar[:, 0].min() == pytest.approx(0.5) and bar[:, 0].max() == pytest.approx(3.5), "the bar is not touched"
    assert shapely.Polygon(stem).intersection(shapely.Polygon(bar)).area == pytest.approx(0.0, abs=1e-9)


def test_crossing_partitions_do_not_trim_each_other():
    arch = {"parameters": {}, "elements": []}
    _wall(arch, "partition", "Partition_A", (0.5, 2.0), (3.5, 2.0), 0.1, (0.0, 1.0))
    _wall(arch, "partition", "Partition_B", (2.0, 0.5), (2.0, 3.5), 0.1, (1.0, 0.0))
    out = plan.partition_footprints(arch, {})
    assert np.array(out["Partition_A"])[:, 0].min() == pytest.approx(0.5) and np.array(out["Partition_B"])[:, 1].max() == pytest.approx(3.5)


def test_a_free_partition_end_stays():
    arch = {"parameters": {}, "elements": []}
    _wall(arch, "partition", "Partition_A", (0.5, 2.0), (3.5, 2.0), 0.1, (0.0, 1.0))
    assert np.array(plan.partition_footprints(arch, {})["Partition_A"])[:, 0].tolist() == pytest.approx([0.5, 3.5, 3.5, 0.5])


def _resolved(arch):
    """Shapely polygons of the walls and of the partitions after the boolean the Blender build applies."""
    walls = {k: shapely.Polygon(v) for k, v in plan.wall_footprints(arch).items()}
    plans = plan.partition_plans(arch, {k: list(v.exterior.coords)[:-1] for k, v in walls.items()})
    finals = {}
    for pid, p in plans.items():
        cut = shapely.union_all([shapely.Polygon(c) for c in p["cut"]]) if p["cut"] else shapely.Polygon()
        finals[pid] = shapely.Polygon(p["solid"]).difference(cut)
    return walls, finals, plans


def _slots(arch, walls, finals, partition_lines):
    """Number of times a line along a partition, going out through one of its ends, finds air between two solids."""
    solid = shapely.union_all([*walls.values(), *finals.values()])
    bad = 0
    for pid, (start, end, thickness) in partition_lines.items():
        u = np.array(end) - np.array(start)
        u /= np.linalg.norm(u)
        normal = np.array([-u[1], u[0]])
        for tip, direction in ((end, u), (start, -u)):
            for lateral in np.linspace(-0.4, 0.4, 5) * thickness:
                inside_run, seen_air = False, False
                for k in range(-30, int(4 * THICKNESS / 0.005)):
                    p = np.array(tip) + normal * lateral + direction * (k * 0.005)
                    here = solid.contains(shapely.Point(p))
                    if here and seen_air and inside_run:
                        bad += 1
                        break
                    inside_run = inside_run or here
                    seen_air = seen_air or (inside_run and not here)
    return bad


def _junction_case(name):
    """(arch, partition lines {id: (start, end, thickness)}) of a few outlines with partitions, a jog on either side of
    the partition line, a reflex corner, a collinear partition, a partition meeting a partition."""
    if name == "jog_back":
        arch = _ring([(0, 0), (6, 0), (6, 4), (0, 4), (0, 2.1)])
        # the west wall is split at z = 2.1 into two near-parallel walls, the lower one 11 cm further out
        lower = next(e for e in arch["elements"] if e["id"] == "W4")
        arch["parameters"][lower["params"]["offset"]]["value"] -= 0.11
        lines = {"Partition_P1": ((5.9, 2.0), (0.05, 2.0), 0.2)}
    elif name == "jog_forward":
        arch = _ring([(0, 0), (6, 0), (6, 4), (0, 4), (0, 2.1)])
        lower = next(e for e in arch["elements"] if e["id"] == "W4")
        arch["parameters"][lower["params"]["offset"]]["value"] += 0.1
        lines = {"Partition_P1": ((5.9, 2.0), (0.05, 2.0), 0.2)}
    elif name == "reflex_collinear":
        arch = _ring([(0, 0), (6, 0), (6, 2), (3, 2), (3, 4), (0, 4)])
        lines = {"Partition_P1": ((3.2, 1.96), (0.0, 1.96), 0.2)}  # carries on the step wall, 4 cm off its face line
    elif name == "t_and_partition":
        arch = _ring([(0, 0), (6, 0), (6, 4), (0, 4)])
        lines = {"Partition_P1": ((3.0, -0.1), (3.0, 3.85), 0.1), "Partition_P2": ((0.0, 2.0), (3.0, 2.0), 0.1),
                 "Partition_P3": ((6.2, 1.0), (3.0, 1.0), 0.1)}
    else:
        raise ValueError(name)
    for pid, (start, end, thickness) in lines.items():
        u = (np.array(end) - np.array(start)) / math.dist(start, end)
        _wall(arch, "partition", pid, start, end, thickness, (-u[1], u[0]))
    return arch, lines


CASES = ["jog_back", "jog_forward", "reflex_collinear", "t_and_partition"]


@pytest.mark.parametrize("name", CASES)
def test_partitions_fit_the_walls_without_overlap_or_slot(name):
    arch, lines = _junction_case(name)
    walls, finals, _ = _resolved(arch)
    for pid, final in finals.items():
        assert final.area > 0.5 * shapely.Polygon(plan.partition_plans(arch, {k: list(v.exterior.coords)[:-1] for k, v in walls.items()})[pid]["face"]).area
        for wid, wall in walls.items():
            assert final.intersection(wall).area == pytest.approx(0.0, abs=1e-9), f"{pid} overlaps {wid}"
    names = list(finals)
    for k, a in enumerate(names):
        for b in names[k + 1:]:
            assert finals[a].intersection(finals[b]).area < 0.02 * THICKNESS * 0.1, f"{a} overlaps {b}"
    assert _slots(arch, walls, finals, lines) == 0


def test_a_partition_collinear_with_a_wall_ends_flush_with_it():
    arch, _ = _junction_case("reflex_collinear")
    _, finals, _ = _resolved(arch)
    at_corner = sorted(z for x, z in finals["Partition_P1"].exterior.coords if abs(x - 3.0) < 1e-6)
    assert at_corner[0] == pytest.approx(2.0, abs=0.005) and at_corner[-1] == pytest.approx(2.2, abs=0.005), "both faces meet the wall's"


@pytest.mark.parametrize("fraction,snapped", [(0.45, True), (0.55, False)])
def test_snap_threshold_is_half_the_thinner_thickness(fraction, snapped):
    """Off the step wall's face line by 0.9 x half the thickness the partition is flush with it; by 1.1 x it keeps its step."""
    offset = fraction * THICKNESS
    arch = _ring([(0, 0), (6, 0), (6, 2), (3, 2), (3, 4), (0, 4)])
    line = ((3.2, 2.0 - offset), (0.0, 2.0 - offset), THICKNESS)
    _wall(arch, "partition", "Partition_P1", line[0], line[1], THICKNESS, (0.0, -1.0))
    walls, finals, _ = _resolved(arch)
    lowest = min(z for x, z in finals["Partition_P1"].exterior.coords if x > 2.9)  # the face towards the room, at the step wall
    assert lowest == pytest.approx(2.0 if snapped else 2.0 - offset, abs=0.01 if snapped else 1e-6), "flush, or the step stays"
    for wall in walls.values():
        assert finals["Partition_P1"].intersection(wall).area == pytest.approx(0.0, abs=1e-9)
    assert _slots(arch, walls, finals, {"Partition_P1": (line[0], line[1], THICKNESS)}) == 0


def _free(arch, wall_id, near, side=1):
    bodies, solids = plan.built(arch)
    host = next(e for e in arch["elements"] if e["id"] == wall_id)
    return clip.free_span(host, side, plan.host_geometry(arch, host, bodies), bodies[wall_id], solids, near)


def test_the_free_face_of_a_wall_ends_where_a_partition_stands_in_front_of_it():
    """The jog case: the partition's face meets the lower west wall 10 cm after its start, and the upper one 1.8 m after it."""
    arch, _ = _junction_case("jog_back")
    lower, upper = _free(arch, "W4", 0.5), _free(arch, "W3", 0.5)
    assert lower == pytest.approx((0.1, 2.1), abs=0.006), "the first 10 cm of the face are behind the partition"
    assert upper[1] == pytest.approx(1.8, abs=0.006) and upper[0] < 0.2, "the partition's far face stops the upper wall"


def test_a_mitred_corner_covers_no_face():
    arch = _ring(RECT)
    for wid, length in (("W0", 4.0), ("W1", 3.0)):
        lo, hi = _free(arch, wid, length / 2)
        assert (lo, hi) == pytest.approx((0.0, length), abs=0.21), wid


def test_a_face_beside_an_abutting_partition_is_split_in_two():
    arch = _ring(RECT)
    _wall(arch, "partition", "Partition_P1", (2.0, -0.02), (2.0, 2.0), 0.1, (-1.0, 0.0))
    assert _free(arch, "W0", 0.5)[1] == pytest.approx(2.0, abs=0.006) and _free(arch, "W0", 3.0)[0] == pytest.approx(2.1, abs=0.006)


def _layout(centre, casings=(0.07, 0.07, 0.07, 0.0)):
    return openings.Layout(cut_w=0.76, cut_h=2.0, bottom=0.0, centre=centre, casings=casings)


@pytest.mark.parametrize("room,kept,flags", [(0.5, 0.07, ()), (0.04, 0.04, ()), (0.01, 0.0, ("to_verify",)), (-0.03, 0.0, ("to_verify",))])
def test_a_casing_is_cut_back_to_the_free_stretch_and_dropped_below_the_minimum(room, kept, flags):
    """`room` is the free length between the cut and the body on its right; the left is wide open."""
    lay = _layout(centre=0.8)
    result = clip.fit(lay, (0.0, lay.span[1] + room))
    assert result.casings == pytest.approx((0.07, 0.07, kept, 0.0)) and result.flags == flags
    assert result.bars == pytest.approx((lay.span[0], min(lay.span[1], lay.span[1] + room)))
