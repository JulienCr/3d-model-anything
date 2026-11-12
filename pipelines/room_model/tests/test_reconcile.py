import copy
import dataclasses
import json
import shutil

import pytest

from room_model import architecture, model, openings, reconcile, shell
from helpers import make_case


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    root = tmp_path_factory.mktemp("reconcile")
    ctx, _ = make_case(root, "cap", window=True)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    return ctx


@pytest.fixture
def ctx(scanned, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(scanned.out, out)
    return dataclasses.replace(scanned, out=out)


def _arch(ctx):
    return json.loads((ctx.out / "architecture.json").read_text(encoding="utf-8"))


def _run(ctx, tape=None):
    if tape is not None:
        (ctx.out / "measurements.json").write_text(json.dumps(tape), encoding="utf-8")
    architecture.run(ctx)
    return json.loads((ctx.out / "reconcile.json").read_text(encoding="utf-8")), (ctx.out / "reconcile.md").read_text(encoding="utf-8")


def _survey(arch):
    """A tape survey agreeing with the model: every station and wall length at its model value."""
    return {d["id"]: round(d["value"], 4) for d in arch["dimensions"] if d["kind"] in ("width", "length", "wall_length")}


def _chained_wall(arch, doc):
    """A wall length that some station loop uses."""
    loops = [c for c in doc["checks"] if c["kind"] == "tape_loop" and not c["id"].startswith("loop:closure")]
    return next(d for c in loops for d in c["dims"] if d.startswith("L-"))


def test_the_architecture_stage_writes_the_reconciliation(ctx):
    doc, md = _run(ctx)
    assert doc["schema"] == reconcile.SCHEMA and "# Réconciliation du relevé" in md
    kinds = {c["kind"] for c in doc["checks"]}
    assert {"opening_inside_host", "corner_square", "wall_top_vs_ceiling"} <= kinds
    assert all(c["verdict"] in ("ok", "square_within_noise") for c in doc["checks"] if c["kind"] == "opening_inside_host")


def test_a_consistent_tape_survey_passes_every_loop(ctx):
    doc, _ = _run(ctx, _survey(_arch(ctx)))
    loops = [c for c in doc["checks"] if c["kind"] == "tape_loop"]
    assert loops and all(c["verdict"] == "ok" for c in loops)
    assert not [c for c in doc["checks"] if c["kind"] == "tape_consistency"]


def test_a_tape_typo_is_named_as_the_probable_error(ctx):
    arch = _arch(ctx)
    survey = _survey(arch)
    doc, _ = _run(ctx, survey)
    typo = _chained_wall(arch, doc)
    survey[typo] = round(survey[typo] + 0.10, 4)  # 10 cm misread on the tape
    doc, md = _run(ctx, survey)
    culprit = next(c for c in doc["checks"] if c["kind"] == "tape_consistency")
    assert culprit["verdict"] == "probable_tape_error" and culprit["dims"] == [typo]
    assert culprit["residual"] == pytest.approx(0.10, abs=0.01) and f"measure:{typo}" in doc["actions"]
    assert "erreur de mètre probable" in md and typo in md


def test_parameters_list_every_source_and_flag_a_conflict(ctx):
    arch = _arch(ctx)
    width = next(d for d in arch["dimensions"] if d["id"] == "Door_1-w")
    doc, md = _run(ctx, {"Door_1-w": round(width["value"] + 0.20, 4)})
    door = next(e for e in _arch(ctx)["elements"] if e["id"] == "Door_1")
    entry = doc["parameters"][door["params"]["width"]]
    assert [s["source"] for s in entry["sources"]][-1] == "tape" and entry["retained"]["source"] == "tape"
    assert len(entry["sources"]) == 2 and "conflict" in entry["flags"] and "measure:Door_1-w" in entry["actions"]
    assert "Paramètres en désaccord" in md


def _element(arch, ident):
    return next(e for e in arch["elements"] if e["id"] == ident)


def _set(arch, el, name, value):
    arch["parameters"][el["params"][name]]["value"] = value


def test_an_opening_outside_its_host_or_overlapping_another_is_reported(ctx):
    arch = _arch(ctx)
    door = _element(arch, "Door_1")
    host = _element(arch, door["host"])
    moved = copy.deepcopy(arch)
    _set(moved, _element(moved, "Door_1"), "center_s", reconcile._length(host) + 0.2)
    check = next(c for c in reconcile.element_checks(moved) if c["id"] == "inside:Door_1")
    assert check["verdict"] == "conflict" and check["residual"] > 0.5
    twin = copy.deepcopy(door)
    twin["id"] = "Door_9"
    moved = copy.deepcopy(arch)
    moved["elements"].append(twin)
    overlap = next(c for c in reconcile.element_checks(moved) if c["kind"] == "opening_overlap")
    assert overlap["verdict"] == "conflict" and {"override:Door_1:remove", "override:Door_9:remove"} & set(overlap["actions"])


def test_a_leaf_larger_than_its_hole_fails_the_leaf_check(ctx):
    arch = _arch(ctx)
    door = _element(arch, "Door_1")
    width = arch["parameters"][door["params"]["width"]]
    door["params"]["leaf_width"] = "opening.Door_1.leaf_width"
    arch["parameters"]["opening.Door_1.leaf_width"] = {"value": width["value"] + 0.05, "sigma": 0.005, "unit": "m", "source": "photo"}
    check = next(c for c in reconcile.element_checks(arch) if c["id"] == "leaf:Door_1:width")
    assert check["verdict"] == "conflict" and check["residual"] == pytest.approx(0.05 + 2 * 0.05)


def test_a_wall_top_far_from_every_ceiling_is_reported(ctx):
    arch = _arch(ctx)
    wall = next(e for e in arch["elements"] if e["kind"] == "wall")
    arch["parameters"][wall["params"]["top"]]["value"] += 0.30
    check = next(c for c in reconcile.element_checks(arch) if c["id"] == f"top:{wall['id']}")
    assert check["verdict"] == "conflict" and check["residual"] == pytest.approx(0.30, abs=0.06)


@pytest.mark.parametrize("shift, verdict", [(0.0, "square_within_noise"), (0.15, "out_of_square")])
def test_a_corner_is_out_of_square_only_beyond_its_sigma(ctx, shift, verdict):
    arch = _arch(ctx)
    outline = reconcile._outline(arch)
    a, b = outline[0], outline[1]
    corner = next(c for c in reconcile.element_checks(arch) if c["id"] == f"corner:{a['id']}:{b['id']}")
    if shift:
        b_dir = reconcile._direction(b)
        normal = [-b_dir[1], b_dir[0]]
        b["end"] = [b["end"][0] + shift * normal[0], b["end"][1] + shift * normal[1]]
        outline_next = outline[2]
        outline_next["start"] = list(b["end"])
        corner = next(c for c in reconcile.element_checks(arch) if c["id"] == f"corner:{a['id']}:{b['id']}")
    assert corner["verdict"] == verdict and corner["sigma"] > 0


def test_an_llm_value_against_a_measured_source_proposes_to_prefer_the_measure():
    layers = [("scan", {"parameters": {"p.h": {"value": 2.10, "sigma": 0.016, "source": "lidar"}}}),
              ("llm", {"parameters": {"p.h": {"value": 2.02, "sigma": 0.01, "source": "llm"}}})]
    entry = reconcile.parameters(layers)["p.h"]
    assert "conflict" in entry["flags"] and entry["residuals"]["lidar"]["delta"] == pytest.approx(0.08)
