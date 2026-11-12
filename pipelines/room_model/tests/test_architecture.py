import copy
import json
import re

import numpy as np
import pytest

from room_model import architecture, model, openings, shell, survey_page, tape
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
