import copy
import html as htmllib
import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from room_model import architecture, survey_page, tape_merge
from test_architecture import full  # noqa: F401  (module fixture: the synthetic room with door, window and partition opening)


def _element(arch, id_):
    return next(e for e in arch["elements"] if e["id"] == id_)


def _dim(arch, id_):
    return next(d for d in arch["dimensions"] if d["id"] == id_)


def _rows(arch):
    return {r.dim["id"]: r for r in survey_page.numbered_rows(arch)}


def _page(arch, **kwargs):
    return survey_page.render(arch, survey_page.numbered_rows(arch), {}, **kwargs)


def _row_html(html, dim_id):
    return htmllib.unescape(next(m.group(0) for m in re.finditer(r'<article class="row.*?</article>', html, re.S) if f'data-dim="{dim_id}"' in m.group(0)))


def _as_niche(arch, element_id, sill):
    """Turns an opening into a niche with the given sill (metres), with its sill parameter and dimension."""
    arch = copy.deepcopy(arch)
    el = _element(arch, element_id)
    el["kind"] = "niche"
    pid = f"opening.{element_id}.sill"
    arch["parameters"][pid] = {"value": sill, "sigma": 0.02, "unit": "m", "source": "llm"}
    el["params"]["sill"] = pid
    arch["dimensions"] += architecture._opening_dimensions(el, arch["parameters"], (0.0, 0.0))[2:]
    return arch


def test_rows_per_kind(full):  # noqa: F811
    _, _, arch = full
    rows = _rows(arch)
    assert {"Door_1-leaf-w", "Door_1-leaf-h", "Window_1-leaf-w", "Window_1-leaf-h", "Window_1-sill", "Opening_1-w", "Opening_1-h"} <= set(rows)
    assert not {"Door_1-w", "Door_1-h", "Window_1-w", "Window_1-h", "Opening_1-sill", "Door_1-sill"} & set(rows)
    html = _page(arch)
    door_w, door_h = _row_html(html, "Door_1-leaf-w"), _row_html(html, "Door_1-leaf-h")
    assert "Largeur du vantail (le panneau qui pivote, sans le bâti ni le chambranle)" in door_w
    assert "Hauteur du vantail, du sol au haut du panneau" in door_h
    assert 'class="kind">vantail<' in door_w
    window_w = _row_html(html, "Window_1-leaf-w")
    assert "ouvrant" in window_w and 'class="kind">ouvrant<' in window_w
    passage = _row_html(html, "Opening_1-w")
    assert "Largeur du passage, d'un tableau à l'autre" in passage and 'class="kind">passage<' in passage
    assert "Hauteur du passage, du sol au dessous du linteau" in _row_html(html, "Opening_1-h")


@pytest.mark.parametrize("element_id, kind_type", [("Door_1", "hinged_double"), ("Window_1", "casement_double")])
def test_double_leaves(full, element_id, kind_type):  # noqa: F811
    _, _, arch = full
    arch = copy.deepcopy(arch)
    _element(arch, element_id)["type"] = kind_type
    html = _page(arch)
    width, height = _row_html(html, f"{element_id}-leaf-w"), _row_html(html, f"{element_id}-leaf-h")
    assert "Largeur des deux vantaux ensemble, du bord extérieur de l'un au bord extérieur de l'autre" in width
    assert "Hauteur d'un vantail" in height and 'class="kind">deux vantaux<' in width


def test_niche_rows(full):  # noqa: F811
    _, _, arch = full
    floor_based = _rows(_as_niche(arch, "Opening_1", 0.0))
    assert {"Opening_1-w", "Opening_1-h"} <= set(floor_based) and "Opening_1-sill" not in floor_based
    raised = _as_niche(arch, "Opening_1", 0.9)
    assert "Opening_1-sill" in _rows(raised)
    html = _page(raised)
    assert "Largeur de la niche, d'un montant à l'autre" in _row_html(html, "Opening_1-w")
    assert "Hauteur, du sol (ou de l'appui) au dessous du linteau" in _row_html(html, "Opening_1-h")
    assert 'class="kind">niche<' in _row_html(html, "Opening_1-w")


def test_leaf_endpoints_are_the_opening_inset_by_casing_and_jamb(full):  # noqa: F811
    _, _, arch = full
    arch = copy.deepcopy(arch)
    rows = _rows(arch)
    p0, p1 = architecture.dimension_endpoints(arch, "Door_1-w")
    leaf = rows["Door_1-leaf-w"]
    assert rows["Door_1-leaf-w"].estimated
    assert np.linalg.norm(np.subtract(leaf.p1, leaf.p0)) == pytest.approx(np.linalg.norm(np.subtract(p1, p0)) - 2 * survey_page.JAMB_VISIBLE)
    assert "position estimée" in _row_html(_page(arch), "Door_1-leaf-w")
    _element(arch, "Door_1")["frame"] = {"profile": "flat", "width": 0.08}
    wide = _rows(arch)["Door_1-leaf-w"]
    assert np.linalg.norm(np.subtract(wide.p1, wide.p0)) == pytest.approx(np.linalg.norm(np.subtract(p1, p0)) - 2 * (0.08 + survey_page.JAMB_VISIBLE))
    low, high = architecture.dimension_endpoints(arch, "Door_1-h")
    height = _rows(arch)["Door_1-leaf-h"]
    assert height.p0[1] == pytest.approx(low[1]) and height.p1[1] == pytest.approx(high[1] - 0.08 - survey_page.JAMB_VISIBLE)


# --- tape merge: leaf ids and both file formats ---------------------------------------------------------------------

def test_tape_merge_applies_leaf_measures(full):  # noqa: F811
    ctx, _, arch = full
    model_doc = json.loads((ctx.out / "model.json").read_text(encoding="utf-8"))
    openings_doc = json.loads((ctx.out / "openings.json").read_text(encoding="utf-8"))
    built = architecture.build(model_doc, openings_doc, None, {"Door_1-leaf-w": 0.73, "Door_1-leaf-h": 2.01, "Window_1-leaf-w": 1.1, "Opening_1-leaf-w": 0.8})
    door = _element(built, "Door_1")
    assert built["parameters"][door["params"]["leaf_width"]] == {"value": 0.73, "sigma": tape_merge.TAPE_SIGMA, "unit": "m", "source": "tape"}
    assert built["parameters"][door["params"]["leaf_height"]]["value"] == 2.01 and "tape_corrected" in door["flags"]
    dim = _dim(built, "Door_1-leaf-w")
    assert dim["kind"] == "leaf_width" and dim["value"] == 0.73 and dim["source"] == "tape" and dim["elements"][0] == "Door_1"
    assert _dim(built, "Door_1-w")["value"] == pytest.approx(_dim(arch, "Door_1-w")["value"])  # the opening itself is not touched
    applied = {a["id"] for a in built["diagnostics"]["tape"]["applied"]}
    rejected = {a["id"] for a in built["diagnostics"]["tape"]["not_applied"]}
    assert applied == {"Door_1-leaf-w", "Door_1-leaf-h", "Window_1-leaf-w"} and rejected == {"Opening_1-leaf-w"}  # a passage has no leaf
    row = _row_html(survey_page.render(built, survey_page.numbered_rows(built), {}), "Door_1-leaf-w")
    assert "Mètre : 73,0 cm" in row and "à reprendre en priorité" not in row
    again = architecture.build(model_doc, openings_doc, None, {"Door_1-leaf-w": 0.74})
    assert _dim(again, "Door_1-leaf-w")["value"] == 0.74 and sum(d["id"] == "Door_1-leaf-w" for d in again["dimensions"]) == 1


V2 = {"version": 2, "capture": "synthetic", "values": {
    "Window_1-sill": {"value_m": 0.9, "date": "2026-11-11T10:00:00Z", "note": "au niveau du rebord"},
    "Door_1-leaf-w": {"value_m": 0.734, "date": "2026-11-11T10:01:00Z", "note": ""}}}
LEGACY = {"Window_1-sill": 0.9, "Door_1-leaf-w": 0.734}


@pytest.mark.parametrize("doc", [V2, LEGACY], ids=["v2", "legacy"])
def test_tape_merge_reads_both_formats(full, doc):  # noqa: F811
    ctx, _, _ = full
    model_doc = json.loads((ctx.out / "model.json").read_text(encoding="utf-8"))
    openings_doc = json.loads((ctx.out / "openings.json").read_text(encoding="utf-8"))
    built = architecture.build(model_doc, openings_doc, None, doc)
    assert {a["id"] for a in built["diagnostics"]["tape"]["applied"]} == {"Window_1-sill", "Door_1-leaf-w"}
    assert _dim(built, "Door_1-leaf-w")["value"] == 0.734


def test_normalize_skips_nothing_silently():
    assert tape_merge.normalize(LEGACY) == LEGACY
    assert tape_merge.normalize(V2) == {"Window_1-sill": 0.9, "Door_1-leaf-w": 0.734}
    broken = tape_merge.normalize({"version": 2, "values": {"A-w": {"note": "x"}, "B-w": "abc", "C-w": {"value_m": 0.5}}})
    assert broken["C-w"] == 0.5 and broken["A-w"] is None and broken["B-w"] == "abc"  # left for apply() to report as invalid
    assert tape_merge.read_store(V2)["Window_1-sill"] == {"value_m": 0.9, "date": "2026-11-11T10:00:00Z", "note": "au niveau du rebord"}
    assert tape_merge.read_store(LEGACY)["Door_1-leaf-w"]["value_m"] == 0.734


def test_compare_reads_a_v2_file(full):  # noqa: F811
    from room_model import tape
    _, _, arch = full
    report = tape.compare(arch, V2)
    assert "| Window_1-sill" in report and "Door_1-leaf-w" in report.split("ignorées")[1]  # unknown to this architecture


# --- durable store ---------------------------------------------------------------------------------------------------

def _prefill(html):
    return json.loads(re.search(r"var PREFILL = (.*?);\n", html).group(1))


@pytest.mark.parametrize("doc", [V2, LEGACY], ids=["v2", "legacy"])
def test_page_is_prefilled_from_the_existing_file(full, doc):  # noqa: F811
    ctx, _, arch = full
    ctx = type(ctx)(ctx.capture, ctx.out.with_name("prefill-" + ("v2" if doc is V2 else "legacy")))
    shutil.copytree(full[0].out, ctx.out)
    (ctx.out / "measurements.json").write_text(json.dumps(doc), encoding="utf-8")
    survey_page.write(ctx, arch)
    html = (ctx.out / "measurements" / "index.html").read_text(encoding="utf-8")
    store = _prefill(html)
    assert store["Door_1-leaf-w"]["value_m"] == 0.734 and store["Window_1-sill"]["value_m"] == 0.9
    assert re.search(r'data-dim="Door_1-leaf-w"[^>]*value="73,4"', html) or re.search(r'value="73,4"[^>]*data-dim="Door_1-leaf-w"', html)
    note = "au niveau du rebord" if doc is V2 else ""
    assert (f'value="{note}"' in _row_html(html, "Window_1-sill")) if note else True


def test_prefill_keeps_values_without_a_row(full):  # noqa: F811
    ctx, _, arch = full
    legacy_door = {"Door_1-w": 0.9}
    out = ctx.out.with_name("prefill-orphan")
    shutil.copytree(ctx.out, out)
    (out / "measurements.json").write_text(json.dumps(legacy_door), encoding="utf-8")
    survey_page.write(type(ctx)(ctx.capture, out), arch)
    html = (out / "measurements" / "index.html").read_text(encoding="utf-8")
    assert _prefill(html)["Door_1-w"]["value_m"] == 0.9 and 'data-dim="Door_1-w"' not in html


def test_page_has_the_file_access_path_and_its_fallback(full):  # noqa: F811
    _, _, arch = full
    html = _page(arch, capture_id="synthetic")
    for needle in ("Lier le fichier de mesures", "showSaveFilePicker", "showOpenFilePicker", "indexedDB", "createWritable",
                   "requestPermission", "Importer", "Télécharger measurements.json", "localStorage", "room-model-tape:", 'var CAPTURE = "synthetic"'):
        assert needle in html, needle
    assert re.search(r"'showSaveFilePicker' in window", html)  # feature detection hides the link button, the downloads stay
    assert 'id="link" type="button" hidden' in html and 'id="download"' in html
    assert html.count('<input class="note"') == html.count("<article") > 0  # one note field per row


def test_page_is_self_contained(full):  # noqa: F811
    _, _, arch = full
    html = _page(arch)
    bare = re.sub(r"data:[^\"')]*", "", html)
    assert "http" not in bare and "<script src" not in html and "<link" not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_script_parses(full, tmp_path):  # noqa: F811
    _, _, arch = full
    script = re.findall(r"<script>(.*?)</script>", _page(arch), re.S)
    assert script
    for i, text in enumerate(script):
        path = tmp_path / f"s{i}.js"
        path.write_text(text, encoding="utf-8")
        done = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr


# --- legacy migration and notes without a value (page script run under node with a minimal DOM) -------------------------

HARNESS = r"""
const [script, setup] = [process.argv[2], JSON.parse(process.argv[3])];
const store = Object.assign({}, setup.local);
global.localStorage = { get length() { return Object.keys(store).length; }, key: i => Object.keys(store)[i],
  getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
const el = (extra) => Object.assign({ listeners: {}, hidden: false, textContent: '', children: [], value: '',
  addEventListener(n, f) { this.listeners[n] = f; }, appendChild(c) { this.children.push(c); },
  setAttribute() {}, getAttribute(n) { return this.attrs[n]; } }, extra);
const inputs = setup.dims.map(d => el({ attrs: { 'data-dim': d } }));
const notes = setup.dims.map(d => el({ attrs: { 'data-note': d } }));
const byId = {};
global.document = { querySelectorAll: s => (s === 'input[data-dim]' ? inputs : s === 'input[data-note]' ? notes : []),
  querySelector: s => { const m = /\[(data-\w+)="(.*)"\]/.exec(s); return (m[1] === 'data-dim' ? inputs : notes).find(i => i.attrs[m[1]] === m[2]); },
  getElementById: id => (byId[id] = byId[id] || el({})), createElement: () => el({}) };
global.window = global; global.navigator = {};
global.PREFILL = setup.prefill; global.CAPTURE = 'cap';
eval(script);
const type = (id, value, note) => { const i = inputs.find(x => x.attrs['data-dim'] === id), n = notes.find(x => x.attrs['data-note'] === id);
  if (value !== undefined) i.value = value; if (note !== undefined) n.value = note; i.listeners.input(); };
const steps = setup.steps || [];
const out = [];
const snapshot = () => ({ inputs: Object.fromEntries(inputs.map(i => [i.attrs['data-dim'], i.value])),
  stored: JSON.parse(store['room-model-tape:cap'] || '{}'), legacy: store, orphans: byId.legacylist.children.map(c => c.textContent), boxHidden: byId.legacy.hidden });
setTimeout(() => {
  for (const s of steps) { type(...s); }
  setTimeout(() => console.log(JSON.stringify(snapshot())), 700);
}, 10);
"""


def _run_page(arch, tmp_path, setup):
    script = re.search(r"<script>(.*?)</script>", _page(arch), re.S).group(1)
    script = re.sub(r"^var PREFILL.*\nvar CAPTURE.*\n", "", script)
    (tmp_path / "harness.js").write_text(HARNESS, encoding="utf-8")
    done = subprocess.run(["node", str(tmp_path / "harness.js"), script, json.dumps(setup)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
def test_legacy_localstorage_is_migrated_without_loss(full, tmp_path):  # noqa: F811
    _, _, arch = full
    legacy = {"Opening_1-w": "81,5", "Door_1-w": "92", "Window_1-h": "120.5", "Cornice-x": "3", "Window_1-sill": "oops"}
    old_key = "room-model-tape:Relevé au mètre ruban"
    setup = {"dims": ["Opening_1-w", "Window_1-sill"], "local": {old_key: json.dumps(legacy)},
             "prefill": {"Opening_1-w": {"value_m": 0.9, "date": "2026-11-11T09:00:00Z", "note": ""}}}
    out = _run_page(arch, tmp_path, setup)
    stored = out["stored"]
    assert stored["Opening_1-w"]["value_m"] == 0.9  # the newer entry is not overwritten
    assert stored["Door_1-w"] == {"value_m": 0.92, "date": "", "note": "ancienne page — mesure bâti compris"}
    assert stored["Window_1-h"]["value_m"] == 1.205 and stored["Window_1-h"]["note"] == "ancienne page — mesure bâti compris"
    assert stored["Cornice-x"]["note"] == "ancienne page" and "Window_1-sill" not in stored
    assert not out["boxHidden"] and len(out["orphans"]) == 3 and any("Door_1-w" in o and "92" in o for o in out["orphans"])
    assert json.loads(out["legacy"][old_key]) == legacy  # the legacy key is never deleted

    only_rows = _run_page(arch, tmp_path, {**setup, "prefill": {}, "local": {"room-model-tape:other": json.dumps({"Opening_1-w": "70"})}})
    assert only_rows["inputs"]["Opening_1-w"] == "70" and only_rows["stored"]["Opening_1-w"]["note"] == ""
    current_format = {"room-model-tape:other": json.dumps({"Opening_1-w": {"value_m": 0.7, "date": "x", "note": ""}})}
    assert _run_page(arch, tmp_path, {**setup, "prefill": {}, "local": current_format})["inputs"]["Opening_1-w"] == ""


@needs_node
def test_note_without_value_is_kept_and_not_redated(full, tmp_path):  # noqa: F811
    _, _, arch = full
    first = {"value_m": 0.9, "date": "2026-01-01T00:00:00Z", "note": "n"}
    setup = {"dims": ["Opening_1-w", "Window_1-sill"], "local": {}, "prefill": {"Opening_1-w": first},
             "steps": [["Opening_1-w", "90", "n"], ["Window_1-sill", "", "bord de fenêtre"]]}
    stored = _run_page(arch, tmp_path, setup)["stored"]
    assert stored["Opening_1-w"] == first  # same value and note: same date
    assert stored["Window_1-sill"]["value_m"] is None and stored["Window_1-sill"]["note"] == "bord de fenêtre"
    reloaded = _run_page(arch, tmp_path, {**setup, "steps": [], "prefill": {"Window_1-sill": stored["Window_1-sill"]}})
    assert reloaded["stored"]["Window_1-sill"]["note"] == "bord de fenêtre"
    changed = _run_page(arch, tmp_path, {**setup, "steps": [["Opening_1-w", "91", "n"]]})["stored"]["Opening_1-w"]
    assert changed["value_m"] == 0.91 and changed["date"] > first["date"]


def test_python_side_accepts_a_note_without_a_value():
    doc = {"version": 2, "values": {"A-w": {"value_m": None, "date": "d", "note": "x"}, "B-w": {"value_m": 0.5, "date": "", "note": ""}}}
    assert tape_merge.normalize(doc) == {"B-w": 0.5}
    assert tape_merge.read_store(doc)["A-w"] == {"value_m": None, "date": "d", "note": "x"}
    assert tape_merge.read_store({"version": 2, "values": {"A-w": {"value_m": None, "note": " "}}}) == {}
