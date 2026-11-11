import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from room_model import __main__ as cli
from room_model import architecture, model, openings, shell, vision
from room_model.context import ClaudeMissing, NoPhotos, StageContext
from room_model.photo_geom import load_posed_photos, project, to_sensor, to_upright
from blender.arch import catalogue
from helpers import make_case

FAKE = str(Path(__file__).resolve().parent / "fake_claude.py")
FLOOR = np.array([120, 100, 80])  # synthetic floor colour (class 2)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("vision")
    ctx, truth = make_case(root, "full", window=True, partition=True, partition_opening=True, photos=8)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    return ctx


@pytest.fixture(scope="module")
def pending(case):
    return vision.prepare(case)


def _load(path: Path, name: str):
    return json.loads((path / name).read_text(encoding="utf-8"))


def _corner(crops: dict, crop_id: str, lo: float = 0.2, hi: float = 0.8) -> dict:
    w, h = crops[crop_id]["size"]
    return {"crop": crop_id, "uv": [[w * lo, h * lo], [w * hi, h * lo], [w * hi, h * hi], [w * lo, h * hi]]}


def _host_ids(pending: Path) -> list[str]:
    return [f["host"] for f in _load(pending, "elements.json")["faces"]]


def _overview_crop(pending: Path) -> str:
    crops = _load(pending, "crops.json")
    return next(c for c, v in crops.items() if v["kind"] == "overview")


def _canned(pending: Path) -> dict:
    crops = _load(pending, "crops.json")
    door = [c for c in crops if c.startswith("Door_1-")]
    window = [c for c in crops if c.startswith("Window_1-")]
    host = _host_ids(pending)[0]
    overview = _overview_crop(pending)
    return {
        "elements": [
            {"id": "Door_1", "type": "hinged_double", "hinge": "right", "swing": "out", "panels": 4, "handle": True,
             "frame": {"profile": "moulded", "width_cm": 9}, "corners": [_corner(crops, c) for c in door[:2]], "confidence": 0.8},
            {"id": "Window_1", "type": "casement_double", "swing": "in", "corners": [_corner(crops, c) for c in window[:1]]},
            {"id": "Opening_1", "frame": {"profile": "flat", "width_cm": 6}},
        ],
        "new_elements": [{"kind": "niche", "host_hint": host, "through": True, "dividers": 2, "shelves_count": 3,
                          "corners": [_corner(crops, overview)]}],
        "cornice": {"profile": "cove", "height_cm": 12, "projection_cm": 8},
        "skirting": {"profile": "bevel", "height_cm": 10},
        "panel_frames": [{"host_hint": host, "corners": [_corner(crops, overview)], "profile": "quarter_round"}],
    }


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Point ROOM_MODEL_CLAUDE at the fake binary; returns set(answer, **env) and the call log path."""
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("ROOM_MODEL_CLAUDE", FAKE)
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))

    def set_answer(answer, raw=False):
        path = tmp_path / "answer.txt"
        path.write_text(answer if isinstance(answer, str) else json.dumps(answer), encoding="utf-8")
        monkeypatch.setenv("FAKE_CLAUDE_ANSWER", str(path))
        if raw:
            monkeypatch.setenv("FAKE_CLAUDE_RAW", "1")

    set_answer({})
    return set_answer, log


def _stage(ctx, **changes):
    return replace(ctx, **changes)


def test_prepare_sheets(case, pending):
    elements, crops = _load(pending, "elements.json"), _load(pending, "crops.json")
    by_id = {e["id"]: e for e in elements["elements"]}
    assert {"Door_1", "Window_1"} <= {i for i, e in by_id.items() if e["sheet"]}
    sheets = sorted(p.name for p in pending.glob("*.jpg"))
    assert 1 <= len(sheets) <= 12
    assert sorted(s["file"] for s in elements["sheets"]) == sheets
    for name in sheets:
        with Image.open(pending / name) as im:
            assert im.format == "JPEG" and max(im.size) <= 1600
    for e in by_id.values():
        assert not e["sheet"] or e["sheet"] == f"{e['id']}.jpg"
    assert (pending / "prompt.md").read_text(encoding="utf-8").strip() and (pending / "schema.json").exists()
    assert all(f["crop"] or f["sheet"] is None for f in elements["faces"]) and any(f["crop"] for f in elements["faces"])
    assert all(c["id"] in crops for e in by_id.values() for c in e["crops"])


def test_sheet_images_are_upright(case, pending):
    crops = _load(pending, "crops.json")
    assert crops
    checked = 0
    for crop_id, c in crops.items():
        with Image.open(pending / c["sheet"]) as sheet:
            x, y = c["sheet_origin"]
            w, h = c["size"]
            panel = np.asarray(sheet.convert("RGB").crop((x, y, x + w, y + h)), float)
        near = (np.abs(panel - FLOOR).sum(2) < 60)
        third = h // 3
        top, bottom = near[:third].mean(), near[-third:].mean()
        if top + bottom > 0.02:  # a panel showing some floor
            assert bottom > top, crop_id
            checked += 1
    assert checked >= 3


def _panel(pending, crops, crop_id):
    c = crops[crop_id]
    with Image.open(pending / c["sheet"]) as sheet:
        x, y = c["sheet_origin"]
        return np.asarray(sheet.convert("RGB").crop((x, y, x + c["size"][0], y + c["size"][1])), float)


@pytest.mark.parametrize("element, colour", [("Door_1", (60, 90, 170)), ("Window_1", (135, 206, 235))])  # the door leaf, the sky behind the glass
def test_element_centre_lands_on_the_element_in_every_crop(case, pending, element, colour):
    arch = json.loads((case.out / "architecture.json").read_text(encoding="utf-8"))
    centre = np.mean(architecture.dimension_endpoints(arch, f"{element}-w"), axis=0)
    crops = _load(pending, "crops.json")
    photos = {p.name: p for p in load_posed_photos(case)}
    ids = [c for c in crops if c.startswith(f"{element}-")]
    assert ids
    for crop_id in ids:
        c, photo = crops[crop_id], photos[crops[crop_id]["photo"]]
        x0, y0, x1, y1 = c["box"]
        uv = (to_upright(photo, project(photo, centre)[0][0]) - [x0, y0]) * [c["size"][0] / (x1 - x0), c["size"][1] / (y1 - y0)]
        panel = _panel(pending, crops, crop_id)
        u, v = int(uv[0]), int(uv[1])
        assert 0 <= u < panel.shape[1] and 0 <= v < panel.shape[0], crop_id
        near = panel[max(v - 12, 0):v + 13, max(u - 12, 0):u + 13]  # the synthetic window hole is half wall, half sky
        assert (np.abs(near - colour).sum(2) < 70).any(), crop_id


def test_crop_pixels_map_back_to_the_sensor_image(case, pending):
    crops = _load(pending, "crops.json")
    photos = {p.name: p for p in load_posed_photos(case)}
    rng = np.random.default_rng(3)
    for crop_id in [c for c in crops if c.startswith(("Door_1-", "Window_1-"))][:4]:
        c, photo = crops[crop_id], photos[crops[crop_id]["photo"]]
        assert c["up_turns"] == photo.up_turns
        x0, y0, x1, y1 = c["box"]
        w, h = c["size"]
        with Image.open(pending / c["sheet"]) as sheet, Image.open(photo.path) as raw:
            sheet, raw = sheet.convert("RGB"), raw.convert("RGB")
            hits = 0
            for _ in range(12):
                u, v = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
                su, sv = to_sensor(photo, np.array([x0 + u * (x1 - x0) / w, y0 + v * (y1 - y0) / h]))
                got = np.array(sheet.getpixel((int(c["sheet_origin"][0] + u), int(c["sheet_origin"][1] + v))), float)
                want = np.array(raw.getpixel((min(int(su), raw.width - 1), min(int(sv), raw.height - 1))), float)
                hits += np.abs(got - want).sum() < 90
        assert hits >= 9, (crop_id, hits)


def test_overview_sheets_cover_every_face_that_is_seen(case, pending):
    elements, crops = _load(pending, "elements.json"), _load(pending, "crops.json")
    arch = json.loads((case.out / "architecture.json").read_text(encoding="utf-8"))
    wanted = {(e["id"], f) for e in arch["elements"] if e["kind"] in ("wall", "partition") for f in (("A", "B") if e["kind"] == "partition" else ("A",))}
    assert {(f["host"], f["face"]) for f in elements["faces"]} == wanted
    covered = [f for f in elements["faces"] if f["crop"]]
    assert len(covered) >= 5 and all(crops[f["crop"]]["kind"] == "overview" for f in covered)
    assert {f["host"] for f in covered} >= {"Partition_P1"}
    assert all(f["crop"] is None for f in elements["faces"] if f not in covered)


def test_prepare_without_architecture_json_builds_the_same_list(case, pending, tmp_path):
    ctx = _stage(case, out=tmp_path / "out")
    for name in ("model.json", "openings.json"):
        (ctx.out).mkdir(exist_ok=True)
        (ctx.out / name).write_text((case.out / name).read_text(encoding="utf-8"), encoding="utf-8")
    other = vision.prepare(ctx)
    assert not (ctx.out / "architecture.json").exists()
    assert _load(other, "elements.json") == _load(pending, "elements.json")


def test_sheet_count_stays_within_twelve(case, tmp_path):
    arch = json.loads((case.out / "architecture.json").read_text(encoding="utf-8"))
    door = next(e for e in arch["elements"] if e["id"] == "Door_1")
    dims = [d for d in arch["dimensions"] if d["id"].startswith("Door_1-")]
    for k in range(20, 34):
        arch["elements"].append({**door, "id": f"Door_{k}"})
        arch["dimensions"] += [{**d, "id": d["id"].replace("Door_1", f"Door_{k}"), "elements": [f"Door_{k}", door["host"]]} for d in dims]
    out = tmp_path / "out"
    out.mkdir()
    (out / "architecture.json").write_text(json.dumps(arch), encoding="utf-8")
    (out / "model.json").write_text((case.out / "model.json").read_text(encoding="utf-8"), encoding="utf-8")
    pending = vision.prepare(replace(case, out=out))
    elements = _load(pending, "elements.json")
    assert len(elements["sheets"]) == len(list(pending.glob("*.jpg"))) <= 12
    assert elements["skipped"] and all(e["sheet"] is None for e in elements["elements"] if e["id"] in elements["skipped"])
    assert any(f["crop"] for f in elements["faces"])  # the overview sheets are kept


def test_prepare_without_photos_asks_for_photos(build, tmp_path):
    root, _ = build()
    ctx = StageContext(root, tmp_path / "out")
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    with pytest.raises(NoPhotos):
        vision.prepare(ctx)


def test_prompt_states_the_hinge_convention(pending):
    text = (pending / "prompt.md").read_text(encoding="utf-8")
    for word in ("gonds", "gauche", "droite", "hinge", "swing", "corners", "uv"):
        assert word in text
    assert "{{" not in text and all(t in text for t in catalogue.CATALOGUE["door"]) and all(t in text for t in catalogue.CATALOGUE["window"])
    assert "Door_1.jpg" in text


def test_schema_enums_match_the_catalogue(pending):
    schema = _load(pending, "schema.json")
    props = schema["properties"]["elements"]["items"]["properties"]
    wanted = set(catalogue.CATALOGUE["door"]) | set(catalogue.CATALOGUE["window"])
    assert set(props["type"]["enum"]) - {None} == wanted
    assert set(props["hinge"]["enum"]) - {None} == set(catalogue.SIDES) and set(props["swing"]["enum"]) - {None} == set(catalogue.SWINGS)


def test_run_with_fake_claude(case, pending, fake):
    set_answer, log = fake
    set_answer(_canned(pending))
    path = vision.run(_stage(case, force=True))
    assert path == case.out / "vision" / "answers.json"
    answers = _load(path.parent, "answers.json")
    types = {e["id"]: e.get("type") for e in answers["elements"]}
    assert types == {"Door_1": "hinged_double", "Window_1": "casement_double", "Opening_1": None}
    door = next(e for e in answers["elements"] if e["id"] == "Door_1")
    assert (door["hinge"], door["swing"], door["panels"], door["frame"]["profile"]) == ("right", "out", 4, "moulded")
    assert len(door["corners"]) == 2 and answers["new_elements"][0]["kind"] == "niche"
    assert answers["cornice"]["profile"] == "cove" and answers["skirting"]["height_cm"] == 10
    assert answers["model"] == "opus" and answers["duration_s"] >= 0 and set(answers["crops"]) == set(_load(pending, "crops.json"))
    assert _load(path.parent, "rejected.json") == []

    call = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert call["argv"] == ["-p", "--model", "opus", "--output-format", "json", "--allowedTools", "Read"]
    assert call["cwd"] == str(case.out / "vision" / "pending") and "elements.json" in call["files"]
    assert "gonds" in call["stdin"] and "Door_1.jpg" in call["stdin"]


def test_llm_model_flag_reaches_claude(build, tmp_path, fake):
    set_answer, log = fake
    root, _ = build(photos=4)
    args = ["--capture", str(root), "--out", str(tmp_path / "out")]
    for stage in (shell, model, openings):
        stage.run(StageContext(root, tmp_path / "out"))
    set_answer({"elements": []})
    assert cli.main(["vision", *args, "--force", "--llm-model", "sonnet"]) == 0
    argv = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["argv"]
    assert argv[argv.index("--model") + 1] == "sonnet"


def test_fenced_json_is_accepted(case, pending, fake):
    set_answer, _ = fake
    set_answer("Voici :\n```json\n" + json.dumps({"skirting": {"profile": "flat", "height_cm": 8}}) + "\n```")
    answers = _load(vision.run(_stage(case, force=True)).parent, "answers.json")
    assert answers["skirting"] == {"profile": "flat", "height_cm": 8}


def _reasons(case):
    return _load(case.out / "vision", "rejected.json")


@pytest.mark.parametrize("answer", [
    {"elements": [{"id": "Door_9", "type": "hinged_single"}]},
    {"elements": [{"id": "Door_1", "type": "trapdoor"}]},
    {"elements": [{"id": "Door_1", "type": "casement_single"}]},  # a window type on a door
    {"elements": [{"id": "Opening_1", "type": "hinged_single"}]},  # a bare opening carries no type
    {"elements": [{"id": "Door_1", "hinge": "gauche"}]},
    {"elements": [{"id": "Door_1", "swing": "inwards"}]},
    {"elements": [{"id": "Door_1", "panels": -1}]},
    {"elements": [{"id": "Door_1", "confidence": 1.5}]},
    {"elements": [{"id": "Door_1", "handle": "yes"}]},
    {"elements": [{"id": "Door_1", "frame": {"profile": "baroque", "width_cm": 5}}]},
    {"elements": [{"id": "Door_1", "colour": "red"}]},
    {"elements": [{"id": "Door_1", "type": "sliding"}, {"id": "Door_1", "type": "pocket"}]},
    {"elements": "Door_1"},
    {"cornice": {"profile": "cove", "height_cm": -3, "projection_cm": 5}},
    {"skirting": {"profile": "ogee"}},
    {"panel_frames": [{"host_hint": "Wall_M99", "corners": "CORNERS", "profile": "ogee"}]},
    {"new_elements": [{"kind": "niche", "host_hint": "Wall_M99", "through": True, "dividers": 0, "shelves_count": 0, "corners": "CORNERS"}]},
    {"new_elements": [{"kind": "niche", "host_hint": "HOST", "through": True, "dividers": 0, "shelves_count": 0, "corners": []}]},
    {"new_elements": [{"kind": "column", "host_hint": "HOST", "corners": "CORNERS"}]},
    {"new_elements": [{"kind": "niche", "host_hint": "HOST", "through": True, "dividers": 0, "shelves_count": 0, "corners": "DOORCROP"}]},
    {"surprise": 1},
])
def test_invalid_answers_are_rejected_item_by_item(case, pending, fake, answer):
    set_answer, _ = fake
    crops = _load(pending, "crops.json")
    corners = [_corner(crops, _overview_crop(pending))]
    door_crop = next(c for c in crops if c.startswith("Door_1-"))
    text = json.dumps(answer).replace('"CORNERS"', json.dumps(corners)).replace('"HOST"', json.dumps(_host_ids(pending)[0]))
    text = text.replace('"DOORCROP"', json.dumps([_corner(crops, door_crop)]))
    set_answer(text)
    before = (case.out / "architecture.json").read_bytes()
    path = vision.run(_stage(case, force=True))
    answers = _load(path.parent, "answers.json")
    assert [answers["elements"], answers["new_elements"], answers["panel_frames"], answers["cornice"], answers["skirting"]] \
        in ([[], [], [], None, None], [[{"id": "Door_1", "type": "sliding"}], [], [], None, None])
    assert _reasons(case) and all({"where", "reason"} <= set(r) for r in _reasons(case))
    assert (case.out / "architecture.json").read_bytes() == before


@pytest.mark.parametrize("uv", [
    [[10, 10], [50, 10], [50, 50]],  # three points
    [[10, 10], [50, 10], [50, 50], [10, 50], [20, 20]],
    [[10, 10], [50, 10], [50, "50"], [10, 50]],
    [[10, 10], [50, 10], [50, 50], [10]],
    [[10, 10], [5000, 10], [5000, 50], [10, 50]],  # beyond the crop
    [[-20, 10], [50, 10], [50, 50], [-20, 50]],
    [[10, 10], [10, 10], [10, 10], [10, 10]],  # degenerate
    [[10, 10], [50, 10], [50, float("nan")], [10, 50]],
])
def test_malformed_corners_are_rejected(case, pending, fake, uv):
    set_answer, _ = fake
    crop = next(c for c in _load(pending, "crops.json") if c.startswith("Door_1-"))
    answer = {"elements": [{"id": "Door_1", "type": "hinged_single", "corners": [{"crop": crop, "uv": uv}]},
                           {"id": "Window_1", "corners": [{"crop": "Door_1-zz", "uv": [[1, 1], [9, 1], [9, 9], [1, 9]]}]}]}
    set_answer(json.dumps(answer))
    answers = _load(vision.run(_stage(case, force=True)).parent, "answers.json")
    assert answers["elements"] == [] and len(_reasons(case)) == 2


def test_a_good_item_survives_a_bad_neighbour(case, pending, fake):
    set_answer, _ = fake
    set_answer({"elements": [{"id": "Door_1", "type": "pocket"}, {"id": "Door_2", "type": "nonsense"}, {"id": "Nope", "type": "sliding"}],
                "skirting": {"profile": "flat", "height_cm": 9}, "cornice": {"profile": "nope", "height_cm": 9, "projection_cm": 3}})
    path = vision.run(_stage(case, force=True))
    answers = _load(path.parent, "answers.json")
    assert [e["id"] for e in answers["elements"]] == ["Door_1"] and answers["skirting"]["height_cm"] == 9 and answers["cornice"] is None
    assert {r["where"] for r in _reasons(case)} == {"elements[1]", "elements[2]", "cornice"}
    assert next(r for r in _reasons(case) if r["where"] == "elements[2]")["id"] == "Nope"


@pytest.mark.parametrize("text, raw", [("Je ne sais pas.", False), ("", False), ('["not", "an", "object"]', False),
                                      ("{\"elements\": [", False), ("plain text, no envelope", True), ("{\"result\": 5}", True)])
def test_unparseable_output_is_rejected_not_fatal(case, pending, fake, text, raw):
    set_answer, _ = fake
    set_answer(text, raw=raw)
    path = vision.run(_stage(case, force=True))
    answers = _load(path.parent, "answers.json")
    assert answers["elements"] == [] and answers["cornice"] is None and answers["model"] == "opus"
    assert len(_reasons(case)) == 1 and _reasons(case)[0]["where"] == "result"


def test_failing_claude_is_rejected_not_fatal(case, pending, fake, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_CLAUDE_EXIT", "3")
    path = vision.run(_stage(case, force=True))
    answers = _load(path.parent, "answers.json")
    assert answers["elements"] == [] and "authentication failed" in _reasons(case)[0]["reason"]
    assert "authentication failed" in capsys.readouterr().err


def test_missing_claude_binary(case, pending, monkeypatch, capsys):
    monkeypatch.setenv("ROOM_MODEL_CLAUDE", "/nonexistent/claude")
    with pytest.raises(ClaudeMissing):
        vision.run(_stage(case, force=True))
    args = ["--capture", str(case.capture), "--out", str(case.out), "--force"]
    assert cli.main(["vision", *args]) == 1
    assert "claude not found" in capsys.readouterr().err
    assert cli.main(["vision", *args, "--no-llm"]) == 0  # --no-llm never needs the binary


def test_no_llm(build, tmp_path):
    root, _ = build()
    args = ["--capture", str(root), "--out", str(tmp_path / "out")]
    assert cli.main(["all", *args, "--no-llm"]) == 0
    answers = tmp_path / "out" / "vision" / "answers.json"
    assert json.loads(answers.read_text(encoding="utf-8")) == {"skipped": True}
    assert not (tmp_path / "out" / "vision" / "pending").exists()
    assert (tmp_path / "out" / "architecture.json").exists()
