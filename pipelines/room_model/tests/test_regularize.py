import copy
import json

import numpy as np
import pytest
import shapely

from room_model import model, regularize, report, shell
from room_model.context import output_path
from helpers import add_hand_beam, add_hand_partition, ceiling_samples, hand_shell, make_case, wall_labels

RECT = [(0, 0), (5, 0), (5, 4), (0, 4)]
LSHAPE = [(0, 0), (6, 0), (6, 2.5), (2.5, 2.5), (2.5, 4.5), (0, 4.5)]
NO_SAMPLES = ceiling_samples([])


def _walls(doc):
    return {e["label"]: e for e in doc["elements"] if e["kind"] == "wall"}


def _param(doc, el, name):
    return doc["parameters"][el["params"][name]]


def _corner_angles(doc):
    """Angle in degrees between the normals of consecutive walls."""
    lines = shell.wall_lines(doc)
    return [float(np.degrees(np.arccos(np.clip(a["normal"] @ b["normal"], -1, 1)))) for a, b in zip(lines, lines[1:] + lines[:1])]


def _ceilings(doc):
    return {e["id"]: e for e in doc["elements"] if e["kind"] == "ceiling"}


def _region(el):
    return shapely.MultiPolygon([shapely.Polygon(r["exterior"], r["holes"]) for r in el["region"]])


def test_walls_within_their_uncertainty_are_squared_to_exact_right_angles():
    doc, outline = hand_shell(RECT, [1.5, -1.5, 1.0, -1.2], sigma_deg=0.5, offsets=[0.03, -0.02, 0.01, 0.02])
    before = copy.deepcopy(doc)
    out = regularize.apply(doc, outline, NO_SAMPLES)
    assert doc == before
    assert _corner_angles(out) == pytest.approx([90.0] * 4, abs=1e-9)
    for label, el in _walls(out).items():
        old, new = _param(before, _walls(before)[label], "azimuth"), _param(out, el, "azimuth")
        assert "squared" in el["flags"] and new["source"] == old["source"] and new["sigma"] == old["sigma"]
        assert new["measured_value"] == old["value"] and new["value"] != old["value"]
        n = np.array([np.cos(new["value"]), np.sin(new["value"])])
        centroid = out["diagnostics"]["walls"][label]["centroid"]  # the evidence stays where it was measured
        assert n @ centroid == pytest.approx(_param(out, el, "offset")["value"], abs=1e-12)
    floor = next(e for e in out["elements"] if e["kind"] == "floor")["polygon"]
    assert [w["start"] for w in _walls(out).values()] == floor
    assert shapely.Polygon(floor).is_valid


def test_a_wall_beyond_its_uncertainty_keeps_its_measured_angle_and_is_flagged():
    doc, outline = hand_shell(RECT, [0.0, 0.0, 6.0, 0.0], sigma_deg=0.3)
    out = regularize.apply(doc, outline, NO_SAMPLES)
    walls = _walls(out)
    odd = walls["M3"]
    assert "out_of_square" in odd["flags"] and "squared" not in odd["flags"]
    assert _param(out, odd, "azimuth") == _param(doc, _walls(doc)["M3"], "azimuth")
    entry = out["diagnostics"]["squaring"]["walls"]["M3"]
    assert entry["state"] == "out_of_square" and entry["deviation_deg"] == pytest.approx(6.0, abs=1e-6)
    for label in ("M1", "M2", "M4"):
        assert "squared" in walls[label]["flags"]
    # the outlier does not drag the frame: the squared walls sit exactly on the axes
    assert _param(out, walls["M1"], "azimuth")["value"] == pytest.approx(np.pi / 2, abs=1e-12)
    angles = _corner_angles(out)
    assert angles[0] == pytest.approx(90.0, abs=1e-9) and abs(angles[1] - 90.0) == pytest.approx(6.0, abs=1e-6)
    assert "M3:out_of_square" in out["diagnostics"]["flags"]


def test_the_squaring_tolerance_grows_with_the_azimuth_uncertainty():
    tight, outline = hand_shell(RECT, [0.0, 0.0, 3.0, 0.0], sigma_deg=0.3)
    loose, _ = hand_shell(RECT, [0.0, 0.0, 3.0, 0.0], sigma_deg=0.3)
    loose["parameters"]["wall.M3.azimuth"]["sigma"] = float(np.radians(1.5))
    assert "out_of_square" in _walls(regularize.apply(tight, outline, NO_SAMPLES))["M3"]["flags"]
    assert "squared" in _walls(regularize.apply(loose, outline, NO_SAMPLES))["M3"]["flags"]


def test_corner_report_gives_measured_and_retained_angles():
    doc, outline = hand_shell(RECT, [1.5, -1.5, 6.0, -1.2], sigma_deg=0.3)
    out = regularize.apply(doc, outline, NO_SAMPLES)
    corners = out["diagnostics"]["squaring"]["corners"]
    assert [c["between"] for c in corners] == [["M4", "M1"], ["M1", "M2"], ["M2", "M3"], ["M3", "M4"]]
    first, second = corners[1], corners[2]
    assert first["measured_deg"] == pytest.approx(90.0 - 3.0, abs=1e-6) and first["retained_deg"] == pytest.approx(90.0, abs=1e-9)
    assert second["measured_deg"] == pytest.approx(90.0 + 7.5, abs=1e-6)  # a corner next to a wall that keeps its angle
    assert second["retained_deg"] == pytest.approx(_corner_angles(out)[1], abs=1e-9) != second["measured_deg"]


def test_a_near_parallel_jog_is_kept_not_merged():
    outline_pts = [(0, 0), (3, 0), (6, 0), (6, 4), (0, 4)]  # the south wall is cut in two at (3, 0)
    doc, outline = hand_shell(outline_pts, [1.0, -1.0, 0.0, 0.0, 0.0], sigma_deg=0.5, offsets=[0.0, 0.08, 0.0, 0.0, 0.0])
    out = regularize.apply(doc, outline, NO_SAMPLES)
    walls = _walls(out)
    a, b = (_param(out, walls[k], "azimuth")["value"] for k in ("M1", "M2"))
    assert a == b and a == pytest.approx(np.pi / 2, abs=1e-9)
    gap = _param(out, walls["M2"], "offset")["value"] - _param(out, walls["M1"], "offset")["value"]
    assert gap == pytest.approx(0.08, abs=0.002)
    assert "near_parallel_corner" in walls["M1"]["flags"] and "near_parallel_corner" in walls["M2"]["flags"]
    floor = next(e for e in out["elements"] if e["kind"] == "floor")["polygon"]
    assert len(floor) == 5 and shapely.Polygon(floor).is_valid


def test_a_partition_is_squared_and_stays_wall_to_wall():
    doc, outline = hand_shell(RECT, [0.0] * 4)
    pid = add_hand_partition(doc, (2.0, 4.0), (2.0, 0.0), turn_deg=1.0, sigma_deg=0.3)
    out = regularize.apply(doc, outline, NO_SAMPLES)
    part = next(e for e in out["elements"] if e["id"] == pid)
    az = _param(out, part, "azimuth")
    assert "squared" in part["flags"] and az["measured_value"] == _param(doc, next(e for e in doc["elements"] if e["id"] == pid), "azimuth")["value"]
    assert az["value"] == pytest.approx(0.0, abs=1e-9)
    boundary = shapely.Polygon(next(e for e in out["elements"] if e["kind"] == "floor")["polygon"]).exterior
    for end in (part["start"], part["end"]):
        assert boundary.distance(shapely.Point(end)) < 1e-6
    assert np.linalg.norm(np.subtract(part["end"], part["start"])) == pytest.approx(4.0, abs=0.01)


def test_ceiling_zones_follow_the_walls_and_the_partition_not_the_scan_outline():
    doc, outline = hand_shell(LSHAPE, [0.0] * 6)
    add_hand_partition(doc, (1.0, 0.0), (1.0, 4.5))
    ragged = ceiling_samples([(0.1, 0.8, 0.2, 4.0, 2.3), (1.3, 2.4, 0.4, 2.1, 2.5), (2.8, 5.7, 0.3, 2.2, 2.5), (1.2, 2.3, 2.9, 4.2, 2.5)])
    out = regularize.apply(doc, outline, ragged)
    ceilings = _ceilings(out)
    assert sorted(ceilings) == ["Ceiling_1", "Ceiling_2"]
    levels = {k: out["parameters"][e["params"]["level"]] for k, e in ceilings.items()}
    assert levels["Ceiling_1"]["value"] == pytest.approx(2.5, abs=0.01) and levels["Ceiling_2"]["value"] == pytest.approx(2.3, abs=0.01)
    floor = shapely.Polygon(next(e for e in out["elements"] if e["kind"] == "floor")["polygon"])
    strip = shapely.box(0, 0, 1, 4.5)
    assert _region(ceilings["Ceiling_2"]).symmetric_difference(strip).area < 1e-6
    assert _region(ceilings["Ceiling_1"]).symmetric_difference(floor.difference(strip)).area < 1e-6
    assert all(not e.get("flags") for e in ceilings.values())
    tops = {k: out["parameters"][e["params"]["top"]]["value"] for k, e in _walls(out).items()}
    assert tops["M6"] == pytest.approx(2.3, abs=0.01) and tops["M1"] == pytest.approx(2.5, abs=0.01)


def test_a_zone_without_ceiling_samples_takes_its_largest_neighbour_level_and_is_assumed():
    doc, outline = hand_shell(LSHAPE, [0.0] * 6)
    add_hand_partition(doc, (1.0, 0.0), (1.0, 4.5))
    out = regularize.apply(doc, outline, ceiling_samples([(1.3, 2.4, 0.4, 2.1, 2.5), (2.8, 5.7, 0.3, 2.2, 2.5), (1.2, 2.3, 2.9, 4.2, 2.5)]))
    ceilings = _ceilings(out)
    assert sorted(ceilings) == ["Ceiling_1", "Ceiling_2"]
    seen, assumed = ceilings["Ceiling_1"], ceilings["Ceiling_2"]
    assert "assumed" not in seen.get("flags", []) and "assumed" in assumed["flags"]
    level = out["parameters"][assumed["params"]["level"]]
    assert level["source"] == "assumed" and level["value"] == pytest.approx(out["parameters"][seen["params"]["level"]]["value"], abs=1e-9)
    assert _region(assumed).symmetric_difference(shapely.box(0, 0, 1, 4.5)).area < 1e-6


def test_a_partition_almost_on_the_extension_of_a_jog_leaves_no_sliver_zone():
    doc, outline = hand_shell(LSHAPE, [0.0] * 6)
    add_hand_partition(doc, (0.0, 2.53), (2.5, 2.53))
    out = regularize.apply(doc, outline, ceiling_samples([(0.1, 5.9, 0.1, 2.4, 2.5), (0.1, 2.4, 2.6, 4.4, 2.5)]))
    (ceiling,) = _ceilings(out).values()
    assert _region(ceiling).symmetric_difference(shapely.Polygon(LSHAPE)).area < 1e-6


def test_a_beam_splits_the_zones_and_is_cut_out_of_the_ceilings():
    doc, outline = hand_shell([(0, 0), (6, 0), (6, 3), (0, 3)], [0.0] * 4)
    add_hand_beam(doc, [(3.0, 0.0), (3.0, 3.0)], width=0.3)
    out = regularize.apply(doc, outline, ceiling_samples([(0.1, 2.7, 0.1, 2.9, 2.5), (3.3, 5.9, 0.1, 2.9, 2.3)]))
    ceilings = _ceilings(out)
    assert sorted(ceilings) == ["Ceiling_1", "Ceiling_2"]
    assert _region(ceilings["Ceiling_1"]).symmetric_difference(shapely.box(0, 0, 2.85, 3)).area < 1e-6
    assert _region(ceilings["Ceiling_2"]).symmetric_difference(shapely.box(3.15, 0, 6, 3)).area < 1e-6


def test_zones_at_the_same_level_on_both_sides_of_a_beam_are_one_ceiling():
    doc, outline = hand_shell([(0, 0), (6, 0), (6, 3), (0, 3)], [0.0] * 4)
    add_hand_beam(doc, [(2.99, 0.0), (3.01, 3.0)], width=0.3)  # not exactly square to the walls it spans
    out = regularize.apply(doc, outline, ceiling_samples([(0.1, 2.7, 0.1, 2.9, 2.5), (3.3, 5.9, 0.1, 2.9, 2.5)]))
    (ceiling,) = _ceilings(out).values()
    assert len(ceiling["region"]) == 2 and _region(ceiling).is_valid
    assert _region(ceiling).area == pytest.approx(18 - 0.9, abs=0.01)


def test_a_beam_square_to_the_room_axes_but_not_to_the_walls_still_splits_the_ceiling_in_two_parts():
    th = np.radians(0.4)
    turn = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    doc, outline = hand_shell([tuple(turn @ p) for p in [(0, 0), (6, 0), (6, 3), (0, 3)]], [0.0] * 4)
    floor = shapely.Polygon(next(e for e in doc["elements"] if e["kind"] == "floor")["polygon"])
    axis = list(floor.intersection(shapely.LineString([(3, -5), (3, 8)])).coords)
    add_hand_beam(doc, axis, width=0.3)
    out = regularize.apply(doc, outline, ceiling_samples([(0.1, 2.7, 0.3, 2.7, 2.5), (3.3, 5.9, 0.3, 2.7, 2.5)]))
    (ceiling,) = _ceilings(out).values()
    assert len(ceiling["region"]) == 2 and _region(ceiling).is_valid


def test_no_ceiling_samples_at_all_leaves_no_ceiling():
    doc, outline = hand_shell(RECT, [0.0] * 4)
    assert not _ceilings(regularize.apply(doc, outline, NO_SAMPLES))


# --- on a synthetic capture -----------------------------------------------------------------------------------------

def _model(root, name, **options):
    ctx, truth = make_case(root, name, **options)
    shell.run(ctx)
    model.run(ctx)
    return json.loads(output_path(ctx, "model").read_text(encoding="utf-8")), truth


@pytest.fixture(scope="module")
def default(tmp_path_factory):
    return _model(tmp_path_factory.mktemp("regularize"), "default")


def test_model_walls_are_square_and_the_floor_follows_them(default):
    doc, _ = default
    assert _corner_angles(doc) == pytest.approx([90.0, 90.0, 90.0, 90.0, 90.0, 90.0], abs=1e-9)
    floor = next(e for e in doc["elements"] if e["kind"] == "floor")["polygon"]
    assert [w["start"] for w in _walls(doc).values()] == floor


def test_model_ceilings_are_the_l_split_at_its_step_whatever_the_scan_looks_like(default, tmp_path_factory):
    doc, _ = default
    ragged, _ = _model(tmp_path_factory.mktemp("regularize-hole"), "hole", ceiling_hole=True)
    for d in (doc, ragged):
        ceilings = _ceilings(d)
        assert sorted(ceilings) == ["Ceiling_1", "Ceiling_2"]
        levels = [d["parameters"][e["params"]["level"]]["value"] - d["parameters"]["floor.level"]["value"] for e in ceilings.values()]
        assert levels == pytest.approx([2.5, 2.3], abs=0.01)
    areas = [[_region(e).area for e in _ceilings(d).values()] for d in (doc, ragged)]
    assert areas[0] == pytest.approx(areas[1], abs=0.01)  # the walls are fitted on two different meshes
    assert areas[0] == pytest.approx([15.0 - 0.75, 5.0], abs=0.15)


def test_model_wing_without_ceiling_is_assumed_at_the_main_level(tmp_path_factory):
    doc, _ = _model(tmp_path_factory.mktemp("regularize-wing"), "wing", no_wing_ceiling=True)
    ceilings = _ceilings(doc)
    assert sorted(ceilings) == ["Ceiling_1", "Ceiling_2"]
    assert "assumed" in ceilings["Ceiling_2"]["flags"]
    levels = [doc["parameters"][e["params"]["level"]] for e in ceilings.values()]
    assert levels[1]["source"] == "assumed" and levels[1]["value"] == pytest.approx(levels[0]["value"], abs=1e-9)
    heights = {d["id"]: d for d in doc["dimensions"] if d["kind"] == "ceiling_height"}
    assert heights["H2"]["source"] == "assumed"


def test_stations_are_measured_between_the_squared_walls(tmp_path_factory):
    doc, truth = _model(tmp_path_factory.mktemp("regularize-thin"), "thin", thin_wall="wing_east")
    west, wing = wall_labels(doc, truth, "west")[0], wall_labels(doc, truth, "wing_east")[0]
    assert "squared" in _walls(doc)[wing]["flags"]
    assert _corner_angles(doc) == pytest.approx([90.0] * 6, abs=1e-9)
    pair = [d["value"] for d in doc["dimensions"] if d["kind"] == "width" and set(d["elements"]) == {f"Wall_{west}", f"Wall_{wing}"}]
    assert len(pair) >= 2 and max(pair) - min(pair) < 1e-9


def _section(text, title):
    return text.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


def test_report_gives_each_corner_measured_and_retained(default):
    doc, _ = default
    text = report.render(doc)
    rows = [line for line in _section(text, "Équerrage").splitlines() if line.startswith("| M") and "→" in line]
    assert len(rows) == 6
    for row, corner in zip(rows, doc["diagnostics"]["squaring"]["corners"]):
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[0] == " → ".join(corner["between"]) and cells[4] == "équerré"
        assert cells[2] == "90,0°"
    assert "squared" not in _section(text, "Alertes") and "équerr" not in _section(text, "Alertes")


def test_report_alerts_name_the_wall_out_of_square_and_the_assumed_ceiling(default, tmp_path_factory):
    doc = copy.deepcopy(default[0])
    wall = next(e for e in doc["elements"] if e["kind"] == "wall")
    wall["flags"] = ["out_of_square"]
    doc["diagnostics"]["squaring"]["walls"][wall["label"]].update(state="out_of_square", deviation_deg=4.2, tolerance_deg=2.0)
    alerts = _section(report.render(doc), "Alertes")
    assert f"Mur {wall['label']} : hors équerre" in alerts and "4,2°" in alerts and "2,0°" in alerts
    assumed, _ = _model(tmp_path_factory.mktemp("regularize-report"), "wing", no_wing_ceiling=True)
    assert "Plafond 2 : niveau supposé" in _section(report.render(assumed), "Alertes")
