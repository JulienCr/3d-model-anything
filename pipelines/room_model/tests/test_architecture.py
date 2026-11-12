import copy
import dataclasses
import json
import re
import shutil

import numpy as np
import pytest

from room_model import architecture, model, openings, photo_geom, report, shell, survey_page, tape, vision
from helpers import make_case, plan_point, truth_to_room
from synthetic import DOOR, WINDOW


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    root = tmp_path_factory.mktemp("architecture")
    ctx, truth = make_case(root, "full", window=True, partition=True, partition_opening=True, photos=8)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    path = architecture.run(ctx)
    return ctx, truth, json.loads(path.read_text(encoding="utf-8"))


def _dim(arch, id_):
    return next(d for d in arch["dimensions"] if d["id"] == id_)


def _element(arch, id_):
    return next(e for e in arch["elements"] if e["id"] == id_)


def test_architecture_dimensions(full):
    ctx, truth, arch = full
    assert arch["schema"] == "room-model/arch-1"
    assert _dim(arch, "Door_1-w")["value"] == pytest.approx(DOOR["w"], abs=0.02)
    assert _dim(arch, "Door_1-h")["value"] == pytest.approx(DOOR["h"], abs=0.02)
    assert _dim(arch, "Window_1-w")["value"] == pytest.approx(WINDOW["w"], abs=0.02)
    assert _dim(arch, "Window_1-h")["value"] == pytest.approx(WINDOW["h"], abs=0.02)
    assert _dim(arch, "Window_1-sill")["value"] == pytest.approx(WINDOW["sill"], abs=0.02)
    assert "Door_1-sill" not in {d["id"] for d in arch["dimensions"]}
    south = next(e for e in arch["elements"] if e["kind"] == "wall" and e["roomplan_id"] == truth.wall_ident["south"][0])
    length = _dim(arch, f"L-{south['label']}")
    assert length["value"] == pytest.approx(6.0, abs=0.02) and length["sigma"] > 0
    assert {d["id"] for d in json.loads((ctx.out / "model.json").read_text(encoding="utf-8"))["dimensions"]} <= {
        d["id"] for d in arch["dimensions"]}


def test_opening_elements_and_parameters(full):
    ctx, _, arch = full
    openings_doc = json.loads((ctx.out / "openings.json").read_text(encoding="utf-8"))
    for o in openings_doc["openings"]:
        el = _element(arch, o["id"])
        assert el == {"id": o["id"], "kind": o["kind"], "host": o["host"], "face": o["face"], "params": o["params"],
                      "type": None, "flags": o["flags"]}
        assert all(pid in arch["parameters"] for pid in el["params"].values())
    assert _element(arch, "Door_1")["host"] == "Wall_M2"
    assert arch["diagnostics"]["openings"] == openings_doc["diagnostics"]


def test_dimension_endpoints_lie_on_the_element(full):
    ctx, truth, arch = full
    T = truth_to_room(ctx, truth)
    floor = arch["parameters"]["floor.level"]["value"]

    p0, p1 = architecture.dimension_endpoints(arch, "Door_1-w")
    assert np.linalg.norm(np.subtract(p0, p1)) == pytest.approx(DOOR["w"], abs=0.02)
    lo, hi = sorted([p0, p1], key=lambda p: (p[0], p[2]))
    wanted = sorted([plan_point(T, 6.0, DOOR["z"] - DOOR["w"] / 2), plan_point(T, 6.0, DOOR["z"] + DOOR["w"] / 2)], key=lambda p: (p[0], p[1]))
    assert np.allclose([lo[0], lo[2]], wanted[0], atol=0.03) and np.allclose([hi[0], hi[2]], wanted[1], atol=0.03)
    assert p0[1] == pytest.approx(floor + DOOR["h"] / 2, abs=0.03) == pytest.approx(p1[1], abs=0.03)

    p0, p1 = architecture.dimension_endpoints(arch, "Door_1-h")
    assert (p0[0], p0[2]) == pytest.approx((p1[0], p1[2])) and abs(p1[1] - p0[1]) == pytest.approx(DOOR["h"], abs=0.02)
    assert min(p0[1], p1[1]) == pytest.approx(floor, abs=0.02)

    p0, p1 = architecture.dimension_endpoints(arch, "Window_1-sill")
    assert min(p0[1], p1[1]) == pytest.approx(floor, abs=0.02) and max(p0[1], p1[1]) == pytest.approx(floor + WINDOW["sill"], abs=0.02)

    s = _dim(arch, "S01")
    p0, p1 = architecture.dimension_endpoints(arch, "S01")
    assert np.linalg.norm(np.subtract(p0, p1)) == pytest.approx(s["value"], abs=1e-6) and p0[1] == p1[1]
    h = next(d for d in arch["dimensions"] if d["kind"] == "ceiling_height")
    p0, p1 = architecture.dimension_endpoints(arch, h["id"])
    assert abs(p1[1] - p0[1]) == pytest.approx(h["value"], abs=1e-6) and (p0[0], p0[2]) == pytest.approx((p1[0], p1[2]))
    assert architecture.dimension_endpoints(arch, "nope") is None


def test_every_dimension_carries_its_endpoints(full):
    _, _, arch = full
    for dim in arch["dimensions"]:
        ends = architecture.dimension_endpoints(arch, dim["id"])
        assert dim["endpoints"] == (None if ends is None else [list(ends[0]), list(ends[1])])
    assert sum(d["endpoints"] is not None for d in arch["dimensions"]) >= 5


def _sections(html):
    svg = re.search(r"<svg.*?</svg>", html, re.S).group(0)
    rows = {int(m.group(1)): m.group(0) for m in re.finditer(r'<article class="row[^"]*" id="row-(\d+)".*?</article>', html, re.S)}
    return svg, rows


def _priority(d):
    return d["sigma"] >= 0.02 or d["source"] in ("roomplan", "llm", "assumed") or bool(d["flags"])


def test_survey_page(full):
    ctx, _, arch = full
    html = (ctx.out / "measurements" / "index.html").read_text(encoding="utf-8")
    numbers = {int(n): d for n, d in json.loads((ctx.out / "measurements" / "plan_numbers.json").read_text(encoding="utf-8")).items()}
    svg, rows = _sections(html)
    assert sorted(numbers) == list(range(1, len(numbers) + 1)) and set(rows) == set(numbers)
    for n in numbers:
        assert f'id="span-{n}"' in svg
    assert html.count("<input") == len(numbers) and "Door_1-w" in numbers.values()
    assert "http" not in html and "<script src" not in html and "<link" not in html

    dims = {n: _dim(arch, d) for n, d in numbers.items()}
    flags = [_priority(dims[n]) for n in sorted(numbers)]
    assert flags == sorted(flags, reverse=True) and any(flags) and not all(flags)  # priority rows come first
    for n, row in rows.items():
        assert ("à reprendre en priorité" in row) == _priority(dims[n]), numbers[n]
    assert any(dims[n]["source"] == "roomplan" for n in numbers)
    assert not {"L-M1", "B1-u"} & set(numbers.values())  # not on the sheet
    assert "Largeur de la porte Door_1" in rows[next(n for n, d in numbers.items() if d == "Door_1-w")]
    assert all(d["kind"] in {"opening_width", "opening_height", "opening_sill", "width", "length", "ceiling_height"} for d in dims.values())

    cropped = {numbers[n] for n, r in rows.items() if "data:image/jpeg;base64," in r}
    assert {"Door_1-w", "Door_1-h", "Window_1-w", "Window_1-sill"} <= cropped
    assert _element(arch, "Opening_1")["face"] == "B" and not {"Opening_1-w", "Opening_1-h"} & cropped  # hidden behind the partition
    assert len(html) < 3_000_000
    assert "Télécharger measurements.json" in html and "Copier" in html and "localStorage" in html


def test_survey_page_shows_both_values_of_a_corrected_dimension(full):
    _, _, arch = full
    arch = copy.deepcopy(arch)
    dim = _dim(arch, "Door_1-w")
    dim.update(scan_value=dim["value"] - 0.03, scan_sigma=0.01, source="llm", flags=["llm_corrected"])
    html = survey_page.render(arch, survey_page.numbered_rows(arch), {})
    row = next(m.group(0) for m in re.finditer(r'<article class="row.*?</article>', html, re.S) if 'data-dim="Door_1-w"' in m.group(0))
    assert "Scan" in row and "LLM" in row and "img" not in row


def test_tape_report(full):
    _, _, arch = full
    door = _dim(arch, "Door_1-w")
    report = tape.compare(arch, {"Door_1-w": door["value"] + 0.02, "Nope-w": 1.0})
    line = next(l for l in report.splitlines() if "Door_1-w" in l)
    assert "+2,0" in line and line.count("|") >= 5
    assert "Nope-w" in report and "Door_1-h" not in report

    corrected = copy.deepcopy(arch)
    d = _dim(corrected, "Door_1-w")
    d.update(scan_value=door["value"] - 0.05, source="llm")
    line = next(l for l in tape.compare(corrected, {"Door_1-w": door["value"] + 0.02}).splitlines() if "Door_1-w" in l)
    assert "+7,0" in line and "+2,0" in line  # tape minus scan, tape minus LLM


def test_build_ignores_absent_vision(full):
    ctx, _, arch = full
    model_doc = json.loads((ctx.out / "model.json").read_text(encoding="utf-8"))
    openings_doc = json.loads((ctx.out / "openings.json").read_text(encoding="utf-8"))
    assert architecture.build(model_doc, openings_doc, None) == arch


def test_survey_alternatives(full):
    ctx, _, _ = full
    html = (ctx.out / "measurements" / "index.html").read_text(encoding="utf-8")
    _, rows = _sections(html)
    alts = {n: len(re.findall(r'class="alt"', r)) for n, r in rows.items()}
    assert max(alts.values()) in (1, 2, 3) and all(a <= 3 for a in alts.values())
    assert all(r.count('class="main"') == 1 for r in rows.values() if "data:image/jpeg" in r)
    assert "swap" in html and "http" not in html and "<script src" not in html


def test_survey_masked_note(full):
    _, _, arch = full
    rows = survey_page.numbered_rows(arch)[:2]
    html = survey_page.render(arch, rows, {1: ["AAAA", "BBBB"], 2: ["CCCC"]}, frozenset({1}))
    sections = {int(m.group(1)): m.group(0) for m in re.finditer(r'<article class="row[^"]*" id="row-(\d+)".*?</article>', html, re.S)}
    assert "vue partiellement masquée par un meuble" in sections[1] and "meuble" not in sections[2]


# --- merging the LLM answers and the tape measurements ---------------------------------------------------------

@dataclasses.dataclass
class Env:
    ctx: object
    arch: dict
    model_doc: dict
    openings_doc: dict
    crops: dict
    photos: dict

    def merge(self, answers=None, tape_values=None, overrides=None):
        return architecture.build(self.model_doc, self.openings_doc, answers, tape_values, self.ctx, overrides)


@pytest.fixture(scope="module")
def env(full):
    ctx, _, arch = full
    pending = vision.prepare(ctx)
    read = lambda name: json.loads((ctx.out / name).read_text(encoding="utf-8"))  # noqa: E731
    return Env(ctx, arch, read("model.json"), read("openings.json"), json.loads((pending / "crops.json").read_text(encoding="utf-8")),
               {p.name: p for p in photo_geom.load_posed_photos(ctx)})


def _param(arch, element_id, name):
    return arch["parameters"][_element(arch, element_id)["params"][name]]


def _crop_uv(env, crop_id, pts):
    """Crop pixels of room-frame points: the inverse of the mapping `architecture` applies to the LLM corners."""
    info = env.crops[crop_id]
    photo = env.photos[info["photo"]]
    up = photo_geom.to_upright(photo, photo_geom.project(photo, np.asarray(pts, float))[0])
    x0, y0, x1, y1 = info["box"]
    w, h = info["size"]
    return ((up - [x0, y0]) * [w / (x1 - x0), h / (y1 - y0)]).tolist()


def _opening_rect(arch, element_id, widen=0.0, shift=0.0):
    """Room-frame corners TL, TR, BR, BL of an opening's scan rectangle, `widen` m wider, `shift` m along the host."""
    (x0, _, z0), (x1, _, z1) = architecture.dimension_endpoints(arch, f"{element_id}-w")
    ends = architecture.dimension_endpoints(arch, f"{element_id}-h")
    low, high = sorted((ends[0][1], ends[1][1]))
    u = np.array([x1 - x0, z1 - z0]) / np.hypot(x1 - x0, z1 - z0)
    a, b = np.array([x0, z0]) - u * (widen / 2 - shift), np.array([x1, z1]) + u * (widen / 2 + shift)
    return [[a[0], high, a[1]], [b[0], high, b[1]], [b[0], low, b[1]], [a[0], low, a[1]]]


def _face_rect(env, host_id, face, s, y):
    """TL, TR, BR, BL of the rectangle s = (s0, s1) m along the host face, y = (low, high) m above the floor."""
    host = _element(env.arch, host_id)
    floor = env.arch["parameters"]["floor.level"]["value"]
    (xa, za), (xb, zb) = (architecture._face_point(env.arch, host, face, v) for v in s)
    return [[xa, floor + y[1], za], [xb, floor + y[1], zb], [xb, floor + y[0], zb], [xa, floor + y[0], za]]


def _corners(env, crop_ids, rect):
    return [{"crop": c, "uv": _crop_uv(env, c, rect)} for c in crop_ids]


def _answers(env, **parts):
    return {"elements": [], "new_elements": [], "cornice": None, "skirting": None, "panel_frames": [], "crops": env.crops, **parts}


def _vision(arch):
    return arch["diagnostics"]["vision"]


def _number(cell):
    return float(cell.replace(",", "."))


def test_llm_correction_applied(env):
    rect = _opening_rect(env.arch, "Door_1", widen=0.04)
    arch = env.merge(_answers(env, elements=[{"id": "Door_1", "corners": _corners(env, ["Door_1-a", "Door_1-b"], rect)}]))
    scan = _param(env.arch, "Door_1", "width")
    width = _param(arch, "Door_1", "width")
    assert width["source"] == "llm" and width["value"] == pytest.approx(scan["value"] + 0.04, abs=0.002)
    assert width["scan_value"] == scan["value"] and width["scan_sigma"] == scan["sigma"]
    assert width["sigma"] >= 0.01
    assert _param(arch, "Door_1", "center_s")["value"] == pytest.approx(_param(env.arch, "Door_1", "center_s")["value"], abs=0.002)
    assert _param(arch, "Door_1", "height")["value"] == pytest.approx(_param(env.arch, "Door_1", "height")["value"], abs=0.002)
    assert _param(arch, "Door_1", "sill") == _param(env.arch, "Door_1", "sill")  # a door reaches the floor: its sill is not measured
    assert "llm_corrected" in _element(arch, "Door_1")["flags"]
    dim = _dim(arch, "Door_1-w")
    assert dim["source"] == "llm" and dim["scan_value"] == scan["value"] and "llm_corrected" in dim["flags"]
    assert _vision(arch)["corrected"] == ["Door_1"] and _vision(arch)["not_applied"] == []
    assert env.openings_doc["parameters"][_element(env.arch, "Door_1")["params"]["width"]]["source"] == "lidar"  # inputs untouched


def test_llm_correction_of_an_elevated_opening_moves_its_sill(env):
    rect = [[x, y + 0.05, z] for x, y, z in _opening_rect(env.arch, "Window_1", widen=-0.04)]
    arch = env.merge(_answers(env, elements=[{"id": "Window_1", "corners": _corners(env, ["Window_1-a", "Window_1-b"], rect)}]))
    scan = {n: _param(env.arch, "Window_1", n)["value"] for n in ("width", "height", "sill")}
    got = {n: _param(arch, "Window_1", n)["value"] for n in scan}
    assert got["width"] == pytest.approx(scan["width"] - 0.04, abs=0.002)
    assert got["sill"] == pytest.approx(scan["sill"] + 0.05, abs=0.002) and got["height"] == pytest.approx(scan["height"], abs=0.002)
    assert "llm_corrected" in _element(arch, "Window_1")["flags"]


def test_llm_correction_uses_the_plane_of_the_partition_face(env):
    rect = _opening_rect(env.arch, "Opening_1", widen=0.04)
    arch = env.merge(_answers(env, elements=[{"id": "Opening_1", "corners": _corners(env, ["Opening_1-a", "Opening_1-b"], rect)}]))
    assert _param(arch, "Opening_1", "width")["value"] == pytest.approx(_param(env.arch, "Opening_1", "width")["value"] + 0.04, abs=0.002)
    assert _param(arch, "Opening_1", "center_s")["value"] == pytest.approx(_param(env.arch, "Opening_1", "center_s")["value"], abs=0.002)


def test_llm_disagreeing_crops(env):
    wide, narrow = (_opening_rect(env.arch, "Door_1", widen=w) for w in (0.04, -0.04))
    corners = [*_corners(env, ["Door_1-a"], wide), *_corners(env, ["Door_1-b"], narrow)]
    arch = env.merge(_answers(env, elements=[{"id": "Door_1", "corners": corners}]))
    assert _param(arch, "Door_1", "width") == _param(env.arch, "Door_1", "width")
    flags = _element(arch, "Door_1")["flags"]
    assert "llm_inconsistent" in flags and "llm_corrected" not in flags
    assert _vision(arch)["inconsistent"] == ["Door_1"] and _vision(arch)["corrected"] == []


def test_llm_two_crops_out_of_three_agree(env):
    wide, narrow = (_opening_rect(env.arch, "Door_1", widen=w) for w in (0.04, -0.08))
    corners = [*_corners(env, ["Door_1-a", "Door_1-c"], wide), *_corners(env, ["Door_1-b"], narrow)]
    arch = env.merge(_answers(env, elements=[{"id": "Door_1", "corners": corners}]))
    assert _param(arch, "Door_1", "width")["value"] == pytest.approx(_param(env.arch, "Door_1", "width")["value"] + 0.04, abs=0.002)
    assert "llm_corrected" in _element(arch, "Door_1")["flags"]


def test_llm_single_view_is_applied_and_flagged(env):
    rect = _opening_rect(env.arch, "Door_1", widen=0.04)
    arch = env.merge(_answers(env, elements=[{"id": "Door_1", "corners": _corners(env, ["Door_1-a"], rect)}]))
    width = _param(arch, "Door_1", "width")
    assert width["source"] == "llm" and width["sigma"] == pytest.approx(0.01)
    flags = _element(arch, "Door_1")["flags"]
    assert "llm_single_view" in flags and "llm_corrected" not in flags
    assert _vision(arch)["single_view"] == ["Door_1"]


def test_llm_wild_correction_is_refused(env):
    rect = _opening_rect(env.arch, "Door_1", widen=0.6)
    arch = env.merge(_answers(env, elements=[{"id": "Door_1", "corners": _corners(env, ["Door_1-a", "Door_1-b"], rect)}]))
    assert _param(arch, "Door_1", "width") == _param(env.arch, "Door_1", "width")
    assert [n["id"] for n in _vision(arch)["not_applied"]] == ["Door_1"] and "cadre" in _vision(arch)["not_applied"][0]["reason"]


def test_llm_typing_fields_are_copied_not_parameters(env):
    item = {"id": "Door_1", "type": "hinged_double", "hinge": "right", "swing": "out", "panels": 4, "handle": True,
            "frame": {"profile": "moulded", "width_cm": 9}, "confidence": 0.7}
    arch = env.merge(_answers(env, elements=[item, {"id": "Window_1", "type": None, "frame": None}]))
    door = _element(arch, "Door_1")
    assert (door["type"], door["hinge"], door["swing"], door["panels"], door["handle"]) == ("hinged_double", "right", "out", 4, True)
    assert door["frame"] == {"profile": "moulded", "width": pytest.approx(0.09)}
    assert _element(arch, "Window_1")["type"] is None and _element(arch, "Window_1")["frame"] is None
    assert arch["parameters"] == env.arch["parameters"] and "llm_corrected" not in door["flags"]
    fields = {(t["element"], t["field"]): t["value"] for t in _vision(arch)["typing"]}
    assert fields[("Door_1", "type")] == "hinged_double" and fields[("Door_1", "hinge")] == "right"


def test_new_niche_hosted(env):
    rect = _face_rect(env, "Partition_P1", "A", (0.1, 0.45), (0.8, 1.6))
    item = {"kind": "niche", "host_hint": "Partition_P1", "through": True, "dividers": 2, "shelves_count": 3,
            "frame": {"profile": "flat", "width_cm": 6}, "corners": _corners(env, ["Partition_P1-A-ov"], rect)}
    arch = env.merge(_answers(env, new_elements=[item]))
    niche = _element(arch, "Niche_1")
    assert (niche["kind"], niche["host"], niche["face"], niche["through"], niche["dividers"]) == ("niche", "Partition_P1", "A", True, 2)
    assert niche["frame"] == {"profile": "flat", "width": pytest.approx(0.06)}
    values = {n: arch["parameters"][pid] for n, pid in niche["params"].items()}
    assert set(values) == {"center_s", "sill", "width", "height", "depth"}
    assert values["width"]["value"] == pytest.approx(0.35, abs=0.005) and values["sill"]["value"] == pytest.approx(0.8, abs=0.005)
    assert values["height"]["value"] == pytest.approx(0.8, abs=0.005) and values["center_s"]["value"] == pytest.approx(0.275, abs=0.005)
    assert all(values[n]["source"] == "llm" and "scan_value" not in values[n] for n in ("center_s", "sill", "width", "height"))
    assert values["depth"]["value"] == pytest.approx(_param(arch, "Partition_P1", "thickness")["value"])  # through: the whole thickness
    assert niche["shelves"] == pytest.approx([0.2, 0.4, 0.6], abs=0.005)  # evenly spaced above the sill, inside the height
    assert "llm_single_view" in niche["flags"]
    assert {"Niche_1-w", "Niche_1-h", "Niche_1-sill"} <= {d["id"] for d in arch["dimensions"]}
    assert architecture.dimension_endpoints(arch, "Niche_1-w") is not None
    assert _vision(arch)["declared"] == ["Niche_1"]


def test_new_niche_checks_through_against_the_lidar(env):
    rect = _face_rect(env, "Partition_P1", "A", (0.1, 0.45), (0.8, 1.6))
    corners = _corners(env, ["Partition_P1-A-ov"], rect)
    through = env.merge(_answers(env, new_elements=[{"kind": "niche", "host_hint": "Partition_P1", "through": True, "corners": corners}]))
    closed = env.merge(_answers(env, new_elements=[{"kind": "niche", "host_hint": "Partition_P1", "through": False, "corners": corners}]))
    assert "through_disputed" in _element(through, "Niche_1")["flags"]  # the mesh has a face B behind the rectangle
    assert "through_disputed" not in _element(closed, "Niche_1")["flags"] and _element(closed, "Niche_1")["through"] is False
    depth = closed["parameters"][_element(closed, "Niche_1")["params"]["depth"]]
    assert depth["source"] == "assumed" and 0 < depth["value"] < _param(closed, "Partition_P1", "thickness")["value"]


def test_new_niche_in_a_wall_has_no_lidar_check(env):
    rect = _face_rect(env, "Wall_M2", None, (0.1, 0.6), (0.9, 1.5))
    item = {"kind": "niche", "host_hint": "Wall_M2", "through": False, "shelves_count": 0, "corners": _corners(env, ["Wall_M2-ov"], rect)}
    arch = env.merge(_answers(env, new_elements=[item]))
    niche = _element(arch, "Niche_1")
    assert niche["host"] == "Wall_M2" and niche["face"] is None and niche["shelves"] == [] and niche["frame"] is None
    assert "through_disputed" not in niche["flags"]


def test_new_niche_overlapping_an_opening_is_refused(env):
    rect = _face_rect(env, "Wall_M2", None, (0.9, 1.6), (0.2, 1.2))  # inside the door
    item = {"kind": "niche", "host_hint": "Wall_M2", "corners": _corners(env, ["Wall_M2-ov"], rect)}
    arch = env.merge(_answers(env, new_elements=[item]))
    assert not any(e["kind"] == "niche" for e in arch["elements"]) and "Door_1" in _vision(arch)["not_applied"][0]["reason"]


def test_mouldings_and_panel_frames_are_copied_as_assumed(env):
    rect = _face_rect(env, "Wall_M2", None, (0.1, 0.7), (0.9, 1.9))
    answers = _answers(env, cornice={"profile": "cove", "height_cm": 12, "projection_cm": 8}, skirting={"profile": "bevel", "height_cm": 10},
                       panel_frames=[{"host_hint": "Wall_M2", "profile": "quarter_round", "corners": _corners(env, ["Wall_M2-ov"], rect)}])
    arch = env.merge(answers)
    cornice, skirting, frame = (_element(arch, i) for i in ("Cornice", "Skirting", "PanelFrame_1"))
    assert cornice["profile"] == "cove" and skirting["profile"] == "bevel"
    for el, name, value in ((cornice, "height", 0.12), (cornice, "projection", 0.08), (skirting, "height", 0.10)):
        p = arch["parameters"][el["params"][name]]
        assert p["value"] == pytest.approx(value) and p["source"] == "assumed" and p["unit"] == "m"
    assert (frame["kind"], frame["host"], frame["face"], frame["profile"]) == ("panel_frame", "Wall_M2", None, "quarter_round")
    s0, y0, s1, y1 = frame["rect"]
    assert (s1 - s0, y1 - y0) == pytest.approx((0.6, 1.0), abs=0.01) and y0 == pytest.approx(0.9, abs=0.01)
    assert "to_verify" in frame["flags"]
    assert env.merge(_answers(env))["elements"] == env.arch["elements"]  # nothing declared, nothing added


@pytest.mark.parametrize("answers", [{"skipped": True}, {"error": "claude timed out", "model": "opus", "duration_s": 1.0}])
def test_placeholders_mean_no_answers(env, answers):
    assert env.merge(answers) == env.arch


def test_unusable_answers_are_listed_never_raised(env):
    crops = env.crops
    garbage = _answers(
        env,
        elements=[{"id": "Nope_9", "type": "fixed"},
                  {"type": "fixed"}, "not an object",
                  {"id": "Door_1", "corners": [{"crop": "nowhere", "uv": [[0, 0], [1, 0], [1, 1], [0, 1]]}]},
                  {"id": "Window_1", "corners": [{"crop": "Door_1-a", "uv": "garbage"}]},
                  {"id": "Opening_1", "corners": [{"crop": "Opening_1-a", "uv": [[0, 0], [crops["Opening_1-a"]["size"][0], 0], [0, 1], [1, 1]]}]}],
        new_elements=[{"kind": "niche", "host_hint": "Nowhere", "corners": _corners(env, ["Wall_M2-ov"], _face_rect(env, "Wall_M2", None, (0.2, 0.5), (1, 1.5)))},
                      {"kind": "niche", "host_hint": "Wall_M2", "corners": []}],
        panel_frames=[{"host_hint": "Nowhere", "profile": "ogee", "corners": []}])
    arch = env.merge(garbage)
    assert arch["parameters"] == env.arch["parameters"]
    assert arch["elements"] == env.arch["elements"]
    reasons = {n["id"]: n["reason"] for n in _vision(arch)["not_applied"]}
    assert {"Nope_9", "Door_1", "Window_1"} <= set(reasons) and all(reasons.values())
    assert len(_vision(arch)["not_applied"]) >= 8 and {"elements[1]", "elements[2]"} <= set(reasons)


def test_orphan_opening_cannot_be_corrected(env):
    openings_doc = copy.deepcopy(env.openings_doc)
    door = next(o for o in openings_doc["openings"] if o["id"] == "Door_1")
    door["host"], door["flags"] = None, ["orphan"]
    arch = architecture.build(env.model_doc, openings_doc, _answers(env, elements=[
        {"id": "Door_1", "corners": _corners(env, ["Door_1-a"], _opening_rect(env.arch, "Door_1", widen=0.04))}]), None, env.ctx)
    assert "hôte" in next(n["reason"] for n in _vision(arch)["not_applied"] if n["id"] == "Door_1")


def _run_in_copy(env, tmp_path, answers=None, tape_values=None):
    ctx = dataclasses.replace(env.ctx, out=tmp_path / "out")
    shutil.copytree(env.ctx.out, ctx.out, ignore=shutil.ignore_patterns("pending"))
    (ctx.out / "vision").mkdir(exist_ok=True)
    if answers is not None:
        (ctx.out / "vision" / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    if tape_values is not None:
        (ctx.out / "measurements.json").write_text(json.dumps(tape_values), encoding="utf-8")
    return ctx, json.loads(architecture.run(ctx).read_text(encoding="utf-8"))


def test_run_reads_answers_and_measurements(env, tmp_path):
    rect = _opening_rect(env.arch, "Door_1", widen=0.04)
    answers = _answers(env, elements=[{"id": "Door_1", "corners": _corners(env, ["Door_1-a", "Door_1-b"], rect)}])
    ctx, arch = _run_in_copy(env, tmp_path, answers, {"Window_1-h": 1.03})
    assert _param(arch, "Door_1", "width")["source"] == "llm" and _param(arch, "Window_1", "height")["source"] == "tape"
    assert (ctx.out / "measurements" / "index.html").exists()


def test_report_corrections_table(env, tmp_path):
    rect = _opening_rect(env.arch, "Door_1", widen=0.04)
    answers = _answers(env, elements=[{"id": "Door_1", "type": "hinged_double", "corners": _corners(env, ["Door_1-a", "Door_1-b"], rect)},
                                      {"id": "Nope_9"}])
    ctx, arch = _run_in_copy(env, tmp_path, answers, {"Door_1-w": 0.95, "Nope-w": 1.0})
    text = (ctx.out / "report.md").read_text(encoding="utf-8")
    section = text.split("## Corrections LLM")[1].split("\n## ")[0]
    header = next(l for l in section.splitlines() if l.startswith("|"))
    assert [c.strip() for c in header.strip("|").split("|")] == ["Élément", "Paramètre", "Scan (cm)", "LLM (cm)", "Écart (cm)"]
    width = next(l for l in section.splitlines() if l.startswith("| Door_1 | largeur"))
    scan = _param(env.arch, "Door_1", "width")["value"] * 100
    cells = [c.strip() for c in width.strip("|").split("|")]
    assert _number(cells[2]) == pytest.approx(scan, abs=0.05) and _number(cells[3]) == pytest.approx(scan + 4, abs=0.1) and cells[4] == "+4,0"
    assert any("type" in l and "hinged_double" in l for l in section.splitlines())
    assert "Nope_9" in section and "inconnu" in section  # listed as not applied
    tape_section = text.split("## Mesures au mètre")[1].split("\n## ")[0]
    line = next(l for l in tape_section.splitlines() if l.startswith("| Door_1-w"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert [_number(c) for c in cells[1:5]] == pytest.approx([scan, scan + 4, 95.0, 95.0], abs=0.1)
    assert "Nope-w" in tape_section
    assert report.render_corrections(arch)[0][0] == "Corrections LLM"


def test_report_sections_replace_previous_ones_and_keep_the_rest(env, tmp_path):
    ctx, _ = _run_in_copy(env, tmp_path)
    path = ctx.out / "report.md"
    path.write_text("# Rapport\n\n## Murs\n\nUn mur.\n", encoding="utf-8")
    for _ in range(2):
        architecture.run(ctx)
    text = path.read_text(encoding="utf-8")
    assert text.count("## Corrections LLM") == 1 and text.count("## Mesures au mètre") == 1 and "## Murs\n\nUn mur." in text


def test_tape_replaces_an_opening_parameter(env):
    arch = env.merge(None, {"Window_1-sill": 0.93, "Door_1-h": 2.07})
    sill = _param(arch, "Window_1", "sill")
    assert sill["value"] == 0.93 and sill["sigma"] == 0.005 and sill["source"] == "tape"
    assert sill["scan_value"] == _param(env.arch, "Window_1", "sill")["value"]
    assert "tape_corrected" in _element(arch, "Window_1")["flags"]
    assert _dim(arch, "Window_1-sill")["value"] == 0.93 and _dim(arch, "Window_1-sill")["source"] == "tape"
    assert _dim(arch, "Door_1-h")["value"] == 2.07
    assert [a["id"] for a in arch["diagnostics"]["tape"]["applied"]] == ["Window_1-sill", "Door_1-h"]


def test_tape_overrides_llm(env):
    rect = _opening_rect(env.arch, "Door_1", widen=0.04)
    answers = _answers(env, elements=[{"id": "Door_1", "corners": _corners(env, ["Door_1-a", "Door_1-b"], rect)}])
    scan = _param(env.arch, "Door_1", "width")["value"]
    arch = env.merge(answers, {"Door_1-w": 0.93})
    width = _param(arch, "Door_1", "width")
    assert width["source"] == "tape" and width["value"] == 0.93 and width["sigma"] == 0.005
    assert width["scan_value"] == scan and width["llm_value"] == pytest.approx(scan + 0.04, abs=0.002)
    flags = _element(arch, "Door_1")["flags"]
    assert "tape_corrected" in flags and "llm_corrected" in flags
    assert _param(arch, "Door_1", "height")["source"] == "llm"  # the tape measured the width only
    line = next(l for l in tape.compare(arch, {"Door_1-w": 0.93}).splitlines() if "Door_1-w" in l)
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert [_number(c) for c in cells[2:4]] == pytest.approx([scan * 100, scan * 100 + 4], abs=0.1)
    row = next(m.group(0) for m in re.finditer(r'<article class="row.*?</article>', survey_page.render(arch, survey_page.numbered_rows(arch), {}), re.S)
               if 'data-dim="Door_1-w"' in m.group(0))
    assert "Mètre : 93,0 cm" in row and "à reprendre en priorité" not in row


def _offset(arch, wall):
    return arch["parameters"][_element(arch, wall)["params"]["offset"]]


def test_tape_moves_wall(env):
    station = _dim(env.arch, "S01")
    walls = station["elements"]
    sigmas = [_offset(env.arch, w)["sigma"] for w in walls]
    assert sigmas[0] != sigmas[1]
    mover, still = (walls[0], walls[1]) if sigmas[0] > sigmas[1] else (walls[1], walls[0])
    arch = env.merge(None, {"S01": station["value"] + 0.03})
    assert _dim(arch, "S01")["value"] == pytest.approx(station["value"] + 0.03, abs=1e-6)
    assert _dim(arch, "S01")["source"] == "tape" and _dim(arch, "S01")["sigma"] == 0.005
    assert _dim(arch, "S01")["scan_value"] == pytest.approx(station["value"])
    moved, kept = _offset(arch, mover), _offset(arch, still)
    assert abs(moved["value"] - _offset(env.arch, mover)["value"]) == pytest.approx(0.03, abs=1e-6)
    assert kept == _offset(env.arch, still)
    assert moved["source"] == "tape" and moved["scan_value"] == _offset(env.arch, mover)["value"]
    assert "tape_corrected" in _element(arch, mover)["flags"] and "tape_corrected" not in _element(arch, still)["flags"]
    assert arch["diagnostics"]["tape"]["moved"][0]["wall"] == mover


def test_tape_moves_the_second_wall_when_sigmas_are_equal(env):
    model_doc = copy.deepcopy(env.model_doc)
    first, second = _dim(env.arch, "S01")["elements"]
    sigmas = [model_doc["parameters"][_element(env.arch, w)["params"]["offset"]] for w in (first, second)]
    sigmas[1]["sigma"] = sigmas[0]["sigma"]
    arch = architecture.build(model_doc, env.openings_doc, None, {"S01": _dim(env.arch, "S01")["value"] - 0.02}, env.ctx)
    assert _offset(arch, second)["value"] != _offset(env.arch, second)["value"] and _offset(arch, first)["value"] == _offset(env.arch, first)["value"]


def test_moving_a_wall_keeps_the_geometry_closed(env):
    station = _dim(env.arch, "S01")
    model_doc = copy.deepcopy(env.model_doc)
    model_doc["parameters"][_element(env.arch, "Wall_M1")["params"]["offset"]]["sigma"] = 0.1  # M1 is the wall that moves; M2 starts on it
    arch = architecture.build(model_doc, env.openings_doc, None, {"S01": station["value"] - 0.04}, env.ctx)
    assert arch["diagnostics"]["tape"]["moved"][0]["wall"] == "Wall_M1"
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    for w in walls:
        n = np.array([np.cos(arch["parameters"][w["params"]["azimuth"]]["value"]), np.sin(arch["parameters"][w["params"]["azimuth"]]["value"])])
        off = arch["parameters"][w["params"]["offset"]]["value"]
        for end in ("start", "end"):
            assert n @ w[end] == pytest.approx(off, abs=1e-6)  # still on its own line
    floor = _element(arch, "Floor")["polygon"]
    assert np.allclose([w["start"] for w in walls], floor, atol=1e-9)
    for old, new in zip(_element(env.arch, "Ceiling_1")["region"][0]["exterior"], _element(arch, "Ceiling_1")["region"][0]["exterior"]):
        assert np.linalg.norm(np.subtract(old, new)) < 0.05
    assert _param(arch, "Window_1", "center_s")["value"] == pytest.approx(_param(env.arch, "Window_1", "center_s")["value"], abs=1e-4)  # moves with M1
    for el in arch["elements"]:
        if el["kind"] in ("door", "window") and el["host"] != "Wall_M1":  # the other openings stay where they are in the room
            host_old, host_new = _element(env.arch, el["host"]), _element(arch, el["host"])
            u = np.subtract(host_old["end"], host_old["start"]) / np.linalg.norm(np.subtract(host_old["end"], host_old["start"]))
            before = np.array(host_old["start"]) + u * _param(env.arch, el["id"], "center_s")["value"]
            after = np.array(host_new["start"]) + u * _param(arch, el["id"], "center_s")["value"]
            assert np.linalg.norm(before - after) < 0.002
    assert _dim(arch, "S02")["value"] == pytest.approx(station["value"] - 0.04, abs=0.001)  # the other station across the same walls follows
    assert _dim(arch, "S02")["scan_value"] == pytest.approx(station["value"]) and "S02" in {d["id"] for d in arch["diagnostics"]["tape"]["derived"]}
    assert _dim(arch, "L-M2")["value"] == pytest.approx(_dim(env.arch, "L-M2")["value"] - 0.04, abs=0.001)


def test_tape_moves_the_ceiling(env):
    height = _dim(env.arch, "H1")
    arch = env.merge(None, {"H1": height["value"] + 0.02})
    floor = arch["parameters"]["floor.level"]["value"]
    level = arch["parameters"][_element(arch, "Ceiling_1")["params"]["level"]]
    assert level["value"] == pytest.approx(floor + height["value"] + 0.02) and level["source"] == "tape"
    assert _dim(arch, "H1")["value"] == pytest.approx(height["value"] + 0.02) and _dim(arch, "H1")["source"] == "tape"
    tops = [(w["id"], arch["parameters"][w["params"]["top"]]["value"]) for w in arch["elements"] if w["kind"] == "wall"]
    old_level = env.arch["parameters"][_element(env.arch, "Ceiling_1")["params"]["level"]]["value"]
    followed = [i for i, t in tops if abs(t - level["value"]) < 1e-9]
    assert followed and all(abs(env.arch["parameters"][_element(env.arch, i)["params"]["top"]]["value"] - old_level) < 1e-9 for i in followed)


def test_tape_unknown_id(env):
    arch = env.merge(None, {"Nope-w": 1.0, "Door_1-w": "abc", "Door_1-h": -2.0, "L-M1": 6.0, "Window_1-w": 1.25})
    listed = {n["id"]: n["reason"] for n in arch["diagnostics"]["tape"]["not_applied"]}
    assert set(listed) == {"Nope-w", "Door_1-w", "Door_1-h", "L-M1"} and all(listed.values())
    assert "inconnue" in listed["Nope-w"]
    assert _param(arch, "Door_1", "width") == _param(env.arch, "Door_1", "width")
    assert [a["id"] for a in arch["diagnostics"]["tape"]["applied"]] == ["Window_1-w"]


def test_tape_wall_move_is_bounded(env):
    arch = env.merge(None, {"S01": _dim(env.arch, "S01")["value"] + 2.0})
    assert arch["diagnostics"]["tape"]["applied"] == [] and "S01" in {n["id"] for n in arch["diagnostics"]["tape"]["not_applied"]}
    assert _dim(arch, "S01")["value"] == _dim(env.arch, "S01")["value"]


def test_corners_without_photos_are_listed(env):
    rect = _opening_rect(env.arch, "Door_1", widen=0.04)
    arch = architecture.build(env.model_doc, env.openings_doc, _answers(env, elements=[{"id": "Door_1", "corners": _corners(env, ["Door_1-a"], rect)}]))
    assert _param(arch, "Door_1", "width") == _param(env.arch, "Door_1", "width")
    assert "indisponible" in _vision(arch)["not_applied"][0]["reason"]


def test_placeholder_answers_leave_a_plain_report(env, tmp_path):
    ctx, arch = _run_in_copy(env, tmp_path, {"skipped": True})
    assert "vision" not in arch["diagnostics"] and "tape" not in arch["diagnostics"]
    text = (ctx.out / "report.md").read_text(encoding="utf-8")
    assert "Aucune réponse du LLM appliquée" in text and "Aucune mesure au mètre fournie" in text


def test_moving_a_wall_carries_the_panel_frames_and_niches_of_its_neighbour(env):
    niche_rect = _face_rect(env, "Wall_M2", None, (0.1, 0.6), (0.9, 1.5))
    frame_rect = _face_rect(env, "Wall_M2", None, (1.8, 2.3), (0.9, 1.9))
    moving_rect = _face_rect(env, "Wall_M1", None, (3.7, 4.1), (0.9, 1.5))
    answers = _answers(env, new_elements=[{"kind": "niche", "host_hint": "Wall_M2", "corners": _corners(env, ["Wall_M2-ov"], niche_rect)},
                                          {"kind": "niche", "host_hint": "Wall_M1", "corners": _corners(env, ["Wall_M1-ov"], moving_rect)}],
                       panel_frames=[{"host_hint": "Wall_M2", "profile": "ogee", "corners": _corners(env, ["Wall_M2-ov"], frame_rect)}])
    model_doc = copy.deepcopy(env.model_doc)
    model_doc["parameters"][_element(env.arch, "Wall_M1")["params"]["offset"]]["sigma"] = 0.1  # M1 moves, M2 starts on it
    before = architecture.build(model_doc, env.openings_doc, answers, None, env.ctx)
    after = architecture.build(model_doc, env.openings_doc, answers, {"S01": _dim(env.arch, "S01")["value"] - 0.04}, env.ctx)
    assert after["diagnostics"]["tape"]["moved"][0]["wall"] == "Wall_M1"

    def absolute(arch, s):
        host = _element(arch, "Wall_M2")
        u = np.subtract(host["end"], host["start"]) / np.linalg.norm(np.subtract(host["end"], host["start"]))
        return np.array(host["start"]) + u * s

    shift = np.linalg.norm(np.subtract(_element(after, "Wall_M2")["start"], _element(before, "Wall_M2")["start"]))
    assert shift > 0.03
    rect_before, rect_after = _element(before, "PanelFrame_1")["rect"], _element(after, "PanelFrame_1")["rect"]
    assert rect_after[0] != rect_before[0] and rect_after[1:2] == rect_before[1:2]
    for k in (0, 2):
        assert np.linalg.norm(absolute(before, rect_before[k]) - absolute(after, rect_after[k])) < 0.002
    centre_before, centre_after = (np.array(_element(a, "Niche_1")["center"]) for a in (before, after))
    assert np.linalg.norm(centre_before - centre_after) < 0.002
    on_moved_wall = [np.array(_element(a, "Niche_2")["center"]) for a in (before, after)]
    assert np.linalg.norm(on_moved_wall[1] - on_moved_wall[0]) == pytest.approx(0.04, abs=0.002)  # a niche goes with its wall


def test_running_the_stage_twice_gives_the_same_file(env, tmp_path):
    rect = _opening_rect(env.arch, "Door_1", widen=0.04)
    answers = _answers(env, elements=[{"id": "Door_1", "type": "fixed", "corners": _corners(env, ["Door_1-a", "Door_1-b"], rect)}])
    ctx, _ = _run_in_copy(env, tmp_path, answers, {"Window_1-h": 1.03, "S01": _dim(env.arch, "S01")["value"] + 0.02})
    path = ctx.out / "architecture.json"
    first, report_first = path.read_bytes(), (ctx.out / "report.md").read_bytes()
    architecture.run(ctx)
    assert path.read_bytes() == first and (ctx.out / "report.md").read_bytes() == report_first


@pytest.mark.parametrize("content", ['{"elements": [', "[1, 2]", ""])
def test_unreadable_answers_are_no_answers_with_a_note(env, tmp_path, content):
    ctx = dataclasses.replace(env.ctx, out=tmp_path / "out")
    shutil.copytree(env.ctx.out, ctx.out, ignore=shutil.ignore_patterns("pending"))
    (ctx.out / "vision").mkdir(exist_ok=True)
    (ctx.out / "vision" / "answers.json").write_text(content, encoding="utf-8")
    arch = json.loads(architecture.run(ctx).read_text(encoding="utf-8"))
    assert "vision" not in arch["diagnostics"] and arch["diagnostics"]["vision_unreadable"]
    assert "inutilisable" in (ctx.out / "report.md").read_text(encoding="utf-8")


def test_opening_dimensions_follow_the_final_position(env):
    rect = _opening_rect(env.arch, "Window_1", shift=0.05)
    arch = env.merge(_answers(env, elements=[{"id": "Window_1", "corners": _corners(env, ["Window_1-a", "Window_1-b"], rect)}]))
    host = _element(arch, "Window_1")["host"]
    expected = architecture._face_point(arch, _element(arch, host), None, _param(arch, "Window_1", "center_s")["value"])
    assert _dim(arch, "Window_1-w")["at"] == pytest.approx(list(expected))
    assert _dim(arch, "Window_1-w")["at"] != _dim(env.arch, "Window_1-w")["at"]


# --- user overrides ---------------------------------------------------------------------------------------------

def _niche_answers(env, through=False):
    rect = _face_rect(env, "Partition_P1", "A", (0.1, 0.45), (0.8, 1.6))
    item = {"kind": "niche", "host_hint": "Partition_P1", "through": through, "dividers": 0, "shelves_count": 1,
            "corners": _corners(env, ["Partition_P1-A-ov"], rect)}
    return _answers(env, new_elements=[item])


def _overrides(arch):
    return arch["diagnostics"]["overrides"]


def test_override_changes_a_niche(env):
    answers = _niche_answers(env, through=False)
    arch = env.merge(answers, None, {"Niche_1": {"through": True, "dividers": 2, "shelves_count": 3, "frame": {"profile": "moulded", "width_cm": 5}}})
    niche = _element(arch, "Niche_1")
    assert niche["through"] is True and niche["dividers"] == 2
    height = _param(arch, "Niche_1", "height")["value"]
    assert niche["shelves"] == pytest.approx([height * k / 4 for k in (1, 2, 3)])
    assert niche["frame"] == {"profile": "moulded", "width": pytest.approx(0.05)}
    assert "user_override" in niche["flags"] and _overrides(arch)["rejected"] == []
    assert {(a["element"], a["field"]) for a in _overrides(arch)["applied"]} >= {("Niche_1", "through"), ("Niche_1", "shelves_count")}
    back = env.merge(_niche_answers(env, through=True), None, {"Niche_1": {"through": False}})
    depth = back["parameters"][_element(back, "Niche_1")["params"]["depth"]]["value"]
    assert 0 < depth < _param(back, "Partition_P1", "thickness")["value"]  # a recess needs a back inside the host


def test_override_wins_over_the_llm_typing(env):
    answers = _answers(env, elements=[{"id": "Door_1", "type": "hinged_double", "hinge": "right"}])
    arch = env.merge(answers, None, {"Door_1": {"type": "sliding", "hinge": "left", "swing": "out", "panels": 2, "handle": False}})
    door = _element(arch, "Door_1")
    assert (door["type"], door["hinge"], door["swing"], door["panels"], door["handle"]) == ("sliding", "left", "out", 2, False)
    assert "user_override" in door["flags"]


def test_override_retypes_between_door_niche_and_opening(env):
    arch = env.merge(_answers(env, elements=[{"id": "Door_1", "type": "hinged_double", "hinge": "right"}]), None, {"Door_1": {"kind": "niche"}})
    niche = _element(arch, "Door_1")
    assert niche["kind"] == "niche" and niche["through"] is False and niche["dividers"] == 0 and niche["shelves"] == []
    assert "hinge" not in niche and "type" not in niche
    assert "Door_1-sill" in {d["id"] for d in arch["dimensions"]}  # a niche has a sill
    depth = arch["parameters"][niche["params"]["depth"]]["value"]
    assert depth < _param(arch, "Wall_M2", "thickness")["value"]  # a recess, not a hole through the wall
    arch = env.merge(None, None, {"Door_1": {"kind": "opening"}, "Opening_1": {"type": "sliding", "kind": "door"}, "Window_1": {"frame": None}})
    assert _element(arch, "Door_1")["kind"] == "opening" and _element(arch, "Door_1")["type"] is None
    assert _element(arch, "Opening_1")["kind"] == "door" and _element(arch, "Opening_1")["type"] == "sliding"  # checked against the new kind
    assert _element(arch, "Window_1")["frame"] is None
    assert _overrides(arch)["rejected"] == []


def test_override_removes_an_element_with_its_parameters_and_dimensions(env):
    arch = env.merge(_niche_answers(env), None, {"Niche_1": {"remove": True}, "Window_1": {"remove": True}})
    ids = {e["id"] for e in arch["elements"]}
    assert "Niche_1" not in ids and "Window_1" not in ids and "Door_1" in ids
    assert not any(k.startswith(("niche.1.", "opening.Window_1.")) for k in arch["parameters"])
    assert not any(d["id"].startswith(("Niche_1-", "Window_1-")) for d in arch["dimensions"])
    assert _overrides(arch)["removed"] == ["Niche_1", "Window_1"]


def test_override_rejects_what_the_catalogue_does_not_allow(env):
    bad = {"Door_1": {"type": "hinged_triple", "hinge": "up", "swing": "sideways", "panels": 99, "handle": "yes", "through": True, "kind": "window"},
           "Window_1": {"type": "hinged_single", "frame": {"profile": "baroque", "width_cm": 5}, "kind": "door"},
           "Opening_1": {"type": "fixed"}, "Wall_M1": {"remove": True}, "Nope_9": {"remove": True}, "Door_2": "text",
           "Door_1x": {}, }
    arch = env.merge(None, None, bad)
    assert arch["elements"] == env.arch["elements"] and arch["parameters"] == env.arch["parameters"]
    rejected = {(r["id"], r.get("field")) for r in _overrides(arch)["rejected"]}
    expected = {("Door_1", f) for f in ("type", "hinge", "swing", "panels", "handle", "through", "kind")} | {
        ("Window_1", "type"), ("Window_1", "frame"), ("Window_1", "kind"), ("Opening_1", "type"), ("Wall_M1", "remove"), ("Nope_9", None), ("Door_2", None)}
    assert expected <= rejected and all(r["reason"] for r in _overrides(arch)["rejected"])
    assert _overrides(arch)["applied"] == []
    door = _element(arch, "Door_1")
    assert door["kind"] == "door" and door.get("type") == _element(env.arch, "Door_1").get("type") and "user_override" not in door["flags"]
    unknown = env.merge(None, None, {"Door_1": {"colour": "red"}})
    assert ("Door_1", "colour") in {(r["id"], r["field"]) for r in _overrides(unknown)["rejected"]}


def test_override_report_section_and_run(env, tmp_path):
    ctx = dataclasses.replace(env.ctx, out=tmp_path / "out")
    shutil.copytree(env.ctx.out, ctx.out, ignore=shutil.ignore_patterns("pending"))
    (ctx.out / "overrides.json").write_text(json.dumps({"Door_1": {"swing": "out", "type": "nope"}, "Window_1": {"remove": True}}), encoding="utf-8")
    arch = json.loads(architecture.run(ctx).read_text(encoding="utf-8"))
    assert _element(arch, "Door_1")["swing"] == "out" and not any(e["id"] == "Window_1" for e in arch["elements"])
    text = (ctx.out / "report.md").read_text(encoding="utf-8")
    section = text.split("## Corrections manuelles")[1].split("\n## ")[0]
    assert "Door_1" in section and "out" in section and "Window_1" in section and "nope" in section
