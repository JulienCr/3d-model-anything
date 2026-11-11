import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from room_model import anchor, architecture, edge_pick, model, openings, ortho, shell
from helpers import colour_stub, make_case
from synthetic import DOOR, PHOTO_POSES, WINDOW

FAKE = str(Path(__file__).with_name("fake_claude.py"))
WINDOW_POSES = (((3.0, 1.4, 2.2), (3.0, 1.4, 0.0), 90.0), ((2.2, 1.4, 2.2), (3.0, 1.4, 0.0), 90.0),
                ((3.8, 1.4, 2.2), (3.0, 1.4, 0.0), 90.0))
DOOR_POSES = (PHOTO_POSES[0], PHOTO_POSES[8], PHOTO_POSES[9])


@pytest.fixture(scope="module")
def anchored(tmp_path_factory):
    root = tmp_path_factory.mktemp("edge-pick")
    poses = DOOR_POSES + WINDOW_POSES
    ctx, _ = make_case(root, "cap", window=True, photos=len(poses), photo_scale=3, photo_poses=poses)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    anchor.run(ctx, colour_stub)
    return ctx


@pytest.fixture
def ctx(anchored, tmp_path):
    """A private copy of the anchored outputs, so that each test writes its own answers."""
    out = tmp_path / "out"
    shutil.copytree(anchored.out, out)
    return dataclasses.replace(anchored, out=out)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOM_MODEL_CLAUDE", FAKE)
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(tmp_path / "calls.jsonl"))

    def set_answer(answer):
        path = tmp_path / "answer.txt"
        path.write_text(json.dumps(answer), encoding="utf-8")
        monkeypatch.setenv("FAKE_CLAUDE_ANSWER", str(path))

    return set_answer


def _candidates(ctx):
    pending = edge_pick.prepare(ctx)
    return pending, json.loads((pending / "candidates.json").read_text(encoding="utf-8"))


def _letter(cands, ident, side, origin="mask"):
    return next(c["letter"] for c in cands[ident]["sides"][side] if origin in c["origin"])


def _param(arch, ident, name):
    el = next(e for e in arch["elements"] if e["id"] == ident)
    return arch["parameters"][el["params"][name]]


def test_prepare_letters_the_candidate_lines_of_each_edge(ctx):
    pending, cands = _candidates(ctx)
    door = cands["Door_1"]
    assert set(door["sides"]) == {"left", "right", "top"}  # a door's bottom is the floor, never asked
    letters = [c["letter"] for side in door["sides"].values() for c in side]
    assert len(letters) == len(set(letters)) >= len(door["sides"])
    for side, entries in door["sides"].items():
        assert {"mask", "model"} <= {o for c in entries for o in c["origin"]}
        assert len(entries) <= edge_pick.MAX_LETTERS
        for a, b in ((a, b) for a in entries for b in entries if a["letter"] < b["letter"]):  # one physical line, one letter
            overlap = a["depths"][0] <= b["depths"][1] and b["depths"][0] <= a["depths"][1]
            assert abs(a["pos"] - b["pos"]) >= edge_pick.PICK_RESOLUTION or not overlap
        with Image.open(pending / door["images"][side]) as im:  # one image per edge, two views side by side
            assert max(im.size) <= edge_pick.SHEET_SIDE
    assert set(cands["Window_1"]["sides"]) == {"left", "right", "bottom", "top"}
    prompt = (pending / "prompt.md").read_text(encoding="utf-8")
    assert "vantail" in prompt and all(f"`{x}`" in prompt for x in letters)
    assert all(f"`{name}`" in prompt for name in door["images"].values()) and "outer edge of the left leaf" in prompt


def test_picked_edges_become_anchored_photo_edges(ctx, fake):
    _, cands = _candidates(ctx)
    picks = {"Door_1": {side: _letter(cands, "Door_1", side) for side in cands["Door_1"]["sides"]},
             "Window_1": {side: _letter(cands, "Window_1", side, "model") for side in cands["Window_1"]["sides"]}}
    fake({"elements": [{"id": ident, "picks": p} for ident, p in picks.items()]})
    path = edge_pick.run(ctx)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["elements"]["Door_1"]["edges"]["left"]["letter"] == picks["Door_1"]["left"]
    arch = json.loads(architecture.run(ctx).read_text(encoding="utf-8"))
    sash = _param(arch, "Window_1", "leaf_width")  # a window is measured only once its edges are picked
    assert sash["source"] == "photo" and abs(sash["value"] - WINDOW["w"]) < 0.015
    assert abs(_param(arch, "Window_1", "leaf_height")["value"] - WINDOW["h"]) < 0.015
    assert abs(_param(arch, "Door_1", "leaf_width")["value"] - DOOR["w"]) < 0.015
    assert "edge_picked" in next(e for e in arch["elements"] if e["id"] == "Window_1")["flags"]


def test_without_picks_a_window_keeps_its_scan_values(ctx):
    arch = json.loads(architecture.run(ctx).read_text(encoding="utf-8"))
    assert "leaf_width" not in next(e for e in arch["elements"] if e["id"] == "Window_1")["params"]


def test_an_unknown_letter_rejects_that_element_only(ctx, fake):
    _, cands = _candidates(ctx)
    door = {side: _letter(cands, "Door_1", side) for side in cands["Door_1"]["sides"]}
    fake({"elements": [{"id": "Door_1", "picks": door}, {"id": "Window_1", "picks": {"left": "ZZ"}}]})
    doc = json.loads(edge_pick.run(ctx).read_text(encoding="utf-8"))
    rejected = json.loads((ctx.out / "edge_pick" / "rejected.json").read_text(encoding="utf-8"))
    assert set(doc["elements"]) == {"Door_1"} and [r["where"] for r in rejected] == ["Window_1"]


def test_a_null_pick_leaves_that_edge_to_the_mask(ctx, fake):
    _, cands = _candidates(ctx)
    fake({"elements": [{"id": "Door_1", "picks": {"left": None, "right": _letter(cands, "Door_1", "right", "model"), "top": None}}]})
    edge_pick.run(ctx)
    arch = json.loads(architecture.run(ctx).read_text(encoding="utf-8"))
    door = next(e for e in arch["elements"] if e["id"] == "Door_1")
    assert "edge_picked" in door["flags"] and _param(arch, "Door_1", "leaf_width")["source"] == "photo"


def test_no_llm_writes_the_skip_placeholder(ctx):
    path = edge_pick.run(dataclasses.replace(ctx, no_llm=True))
    assert json.loads(path.read_text(encoding="utf-8")) == {"skipped": True}
    arch = json.loads(architecture.run(ctx).read_text(encoding="utf-8"))
    assert "leaf_width" not in next(e for e in arch["elements"] if e["id"] == "Window_1")["params"]


def test_a_failed_call_is_recorded_as_an_error(ctx, fake, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_EXIT", "3")
    fake({})
    doc = json.loads(edge_pick.run(ctx).read_text(encoding="utf-8"))
    assert "error" in doc and "elements" not in doc


def test_each_line_is_drawn_on_its_own_visible_edge_in_every_view():
    from test_anchor import FRAME, LEAF_DELTA, LEAF_S, _door_patches, _views

    views = _views(_door_patches())
    entries = [{"letter": "A", "pos": 0.50, "delta": 0.015}, {"letter": "B", "pos": LEAF_S[0], "delta": LEAF_DELTA}]  # casing, leaf
    rect = (0.42, 1.38, 0.0, 2.15)
    for reference in (0.0, LEAF_DELTA):
        box, res = edge_pick._strip_box(views, FRAME, "left", entries, rect[0], rect, reference)
        for photo, image in views:
            _, o, drawn = edge_pick.panel(photo, image, FRAME, "left", entries, rect[0], box, res, reference)
            for letter, at in drawn.items():
                seen = ortho.snap(o, "s", (box[2], box[3]), at, 4 * res)
                # 2 strip pixels: the strip is finer than the photo (about a quarter of a photo pixel each)
                assert seen is not None and abs(seen - at) <= 2 * res, (reference, photo.name, letter)


def test_a_gradient_letter_corrects_a_mask_that_took_the_casing():
    leaf = {"pos": 0.55, "delta": -0.04, "sigma": 0.002, "views": 3, "rms": 0.001, "depths": [-0.045, -0.035], "distance": 2.2}
    casing = {**leaf, "pos": 0.40, "delta": 0.015, "depths": [0.01, 0.02]}
    edges = {"left": casing, "right": {**leaf, "pos": 1.25}, "top": {**leaf, "pos": 2.08}}
    edges, measures = anchor.measures_from(edges, True, 0.01)
    anchors = {"elements": {"Door_1": {"kind": "door", "status": "anchored", "edges": edges, "measures": measures}}}
    picks = {"elements": {"Door_1": {"kind": "door", "floor_based": True, "edges": {"left": {**leaf, "letter": "B", "origin": ["gradient"]}}}}}
    merged = edge_pick.merge(anchors, picks, 0.01)["elements"]["Door_1"]
    assert merged["measures"]["width"]["value"] == pytest.approx(0.70) and merged["picked"] == ["left"]


@pytest.mark.parametrize("answer", [{"elements": [{"id": ["Door_1"], "picks": {}}]},
                                    {"elements": [{"id": "Door_1", "picks": {"left": ["A"]}}]},
                                    {"elements": [{"id": {"x": 1}, "picks": {"left": "A"}}]}])
def test_malformed_picks_reject_the_element_without_raising(answer):
    cands = {"Door_1": {"kind": "door", "sides": {"left": [{"letter": "A"}]}}}
    clean, rejected = edge_pick.validate(answer, cands)
    assert clean == {} and len(rejected) == 1


def test_a_null_on_an_edge_that_was_not_asked_is_ignored():
    cands = {"Door_1": {"kind": "door", "sides": {"left": [{"letter": "A"}], "top": [{"letter": "B"}]}}}
    clean, rejected = edge_pick.validate({"elements": [{"id": "Door_1", "picks": {"left": "A", "bottom": None}}]}, cands)
    assert clean == {"Door_1": {"left": "A"}} and rejected == []


def test_every_edge_of_an_element_gets_its_own_letters():
    assert len(edge_pick.SIDES) * edge_pick.MAX_LETTERS <= 26
