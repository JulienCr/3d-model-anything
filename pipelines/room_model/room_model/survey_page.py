"""Tape measurement sheet: a self-contained local HTML page with a numbered plan and one row per dimension."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from room_model import plan_svg
from room_model.architecture import KIND_WORDS, STATION_HEIGHT, dimension_endpoints
from room_model.capture import colmajor, load_roomplan
from room_model import tape_merge
from room_model.context import StageContext
from room_model.fmt import fr
from room_model.photo_geom import LOOSE_TOLERANCE, SURVEY_FACING_COS, DepthMaps, PosedPhoto, best_views, load_posed_photos, load_room_mesh, occluded_fraction, project, room_boxes, to_upright, upright, upright_size

GROUP = {"opening_width": 0, "opening_height": 0, "opening_sill": 0, "leaf_width": 0, "leaf_height": 0, "width": 1, "length": 1,
         "wall_length": 1, "ceiling_height": 2}
SOURCES = {"lidar": "LiDAR", "roomplan": "RoomPlan", "assumed": "valeur supposée", "llm": "LLM", "photo": "photos", "tape": "mètre ruban"}
PRIORITY_SIGMA = 0.02  # m
PRIORITY_SOURCES = ("roomplan", "llm", "assumed")
BADGE = "à reprendre en priorité"
MASKED_NOTE = "vue partiellement masquée par un meuble"
CROP_SIDE = 480  # px on the long side of a crop
MAX_ALTERNATIVES = 3
# Visible part of the frame between the hole in the wall and the leaf (jamb and rebate): a door or window frame is
# 2 to 4 cm thick. The casing width (`frame.width`) comes on top when the element has one.
JAMB_VISIBLE = 0.03  # m
ESTIMATE_SIGMA = 0.05  # m; the opening's own error plus the spread of casing and jamb widths
FLOOR_BASED = 0.03  # m; a niche whose sill is below this stands on the floor (RoomPlan's sigma is 3 cm)
DOUBLE_TYPES = ("hinged_double", "casement_double", "french_window")
LEAF_IDS = {"leaf-w": ("leaf_width", "w", "Largeur du vantail de"), "leaf-h": ("leaf_height", "h", "Hauteur du vantail de")}
MARKER_RADIUS = 11
MARKER_GAP = 26.0
OPENING_COLOURS = {"door": "#8a5a2b", "window": "#1f78d1", "opening": "#6b7280", "niche": "#6b7280"}
SPAN_STYLE = (".span{stroke:#d1242f;stroke-width:3;stroke-linecap:round;fill:none}.marker{fill:#d1242f;stroke:#fff;stroke-width:2}"
              ".markertext{fill:#fff;stroke:none;font-weight:bold;font-size:12px}.leader{stroke:#d1242f;stroke-width:1;stroke-dasharray:3 2}"
              ".joinery{stroke-width:6;stroke-linecap:butt;fill:none}.joinerytext{font-size:11px;fill:#444}")


@dataclass
class Row:
    number: int
    dim: dict
    p0: list[float]
    p1: list[float]
    priority: bool
    word: str = ""
    estimated: bool = False


def is_priority(dim: dict) -> bool:
    return dim["source"] != "tape" and (dim["sigma"] >= PRIORITY_SIGMA or dim["source"] in PRIORITY_SOURCES or bool(dim["flags"]))


def _element(arch: dict, dim: dict) -> dict | None:
    return next((e for e in arch["elements"] if e["id"] == dim["elements"][0]), None)


def _suffix(dim: dict) -> str:
    return dim["id"].removeprefix(dim["elements"][0] + "-")


def _is_double(el: dict) -> bool:
    return isinstance(el.get("type"), str) and el["type"] in DOUBLE_TYPES


def _word(el: dict, suffix: str) -> str:
    """The kind word shown in the row badge: what the user is about to tape."""
    if el["kind"] == "door":
        return "deux vantaux" if _is_double(el) else "vantail"
    if el["kind"] == "window":
        return "allège" if suffix == "sill" else "deux vantaux" if _is_double(el) else "ouvrant"
    return {"opening": "passage", "niche": "niche"}.get(el["kind"], "")


def _estimated_leaf(arch: dict, el: dict, suffix: str, base: dict):
    """Leaf span as the opening rectangle inset by casing and jamb; the door leaf stands on the floor."""
    ends = dimension_endpoints(arch, base["id"])
    if ends is None:
        return None
    p0, p1 = np.array(ends[0], float), np.array(ends[1], float)
    span = float(np.linalg.norm(p1 - p0))
    inset = min(((el.get("frame") or {}).get("width") or 0.0) + JAMB_VISIBLE, 0.25 * span)
    low = 0.0 if el["kind"] == "door" and suffix == "leaf-h" else inset
    u = (p1 - p0) / span
    return [float(c) for c in p0 + u * low], [float(c) for c in p1 - u * inset], span - low - inset


def _leaf_rows(arch: dict, el: dict, dims: dict) -> list[tuple[dict, list[float], list[float], bool]]:
    out = []
    for suffix, (kind, base_suffix, label) in LEAF_IDS.items():
        base, own = dims.get(f"{el['id']}-{base_suffix}"), dims.get(f"{el['id']}-{suffix}")
        guess = None if base is None else _estimated_leaf(arch, el, suffix, base)
        ends = dimension_endpoints(arch, own["id"]) if own is not None else None
        if own is None and guess is not None:
            own = {"id": f"{el['id']}-{suffix}", "kind": kind, "label": f"{label} {el['id']}", "value": guess[2], "sigma": ESTIMATE_SIGMA,
                   "elements": base["elements"], "at": base["at"], "source": "assumed", "flags": [], "estimate": True}
        if own is not None and (ends or guess):
            out.append((own, *(ends if ends else guess[:2]), ends is None))
    return out


def numbered_rows(arch: dict) -> list[Row]:
    """Rows of the sheet: priority first, then openings, stations, ceiling heights; dimensions without a span are left out.

    A door or window is measured by its leaf (`<id>-leaf-w`, `-leaf-h`), a niche or passage jamb to jamb; a floor-based
    niche has no sill row.
    """
    dims = {d["id"]: d for d in arch["dimensions"]}
    found = []
    for dim in arch["dimensions"]:
        el = _element(arch, dim) if dim["kind"].startswith(("opening_", "leaf_")) else None
        if el is not None and (dim["kind"].startswith("leaf_") or el["kind"] in ("door", "window") and dim["kind"] != "opening_sill"):
            continue
        if el is not None and el["kind"] == "niche" and dim["kind"] == "opening_sill" and dim["value"] < FLOOR_BASED:
            continue
        ends = dimension_endpoints(arch, dim["id"]) if dim["kind"] in GROUP else None
        if ends is not None:
            found.append((dim, *ends, False))
    for el in arch["elements"]:
        if el["kind"] in ("door", "window"):
            found += _leaf_rows(arch, el, dims)
    found.sort(key=lambda f: (not is_priority(f[0]), GROUP[f[0]["kind"]]))
    rows = []
    for i, (dim, p0, p1, estimated) in enumerate(found, start=1):
        el = _element(arch, dim) if dim["kind"].startswith(("opening_", "leaf_")) else None
        rows.append(Row(i, dim, p0, p1, is_priority(dim), "" if el is None else _word(el, _suffix(dim)), estimated))
    return rows


class _SurveyPlan(plan_svg._Plan):
    def draw_survey(self, rows: list[Row], arch: dict) -> str:
        self.draw_elements()
        for el in arch["elements"]:
            if el["kind"] == "partition":
                self._partition(el)
        for el in arch["elements"]:
            ends = dimension_endpoints(arch, f"{el['id']}-w") if el["kind"] in OPENING_COLOURS else None
            if ends is not None:
                self._joinery(el, ends)
        placed: list[tuple[float, float]] = []
        spans, markers = [], []
        for row in rows:
            a, b = self.px((row.p0[0], row.p0[2])), self.px((row.p1[0], row.p1[2]))
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if np.hypot(a[0] - b[0], a[1] - b[1]) > 2.0:
                spans.append(f'<path class="span" d="M{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f}"/>')
            c = self._free_spot(mid, placed)
            placed.append(c)
            leader = "" if c == mid else f'<path class="leader" d="M{mid[0]:.1f},{mid[1]:.1f} L{c[0]:.1f},{c[1]:.1f}"/>'
            markers.append(f'<g id="span-{row.number}">{leader}<circle class="marker" cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{MARKER_RADIUS}"/>'
                           f'<text class="markertext" x="{c[0]:.1f}" y="{c[1] + 4:.1f}" text-anchor="middle">{row.number}</text></g>')
        self.out += spans + markers
        self._scale_bar()
        return self.document(standalone=False, extra_style=SPAN_STYLE)

    def _partition(self, el: dict) -> None:
        az = self.value(el["params"]["azimuth"])
        n = np.array([np.cos(az), np.sin(az)])
        a, b = np.array(el["start"], float), np.array(el["end"], float)
        t = self.value(el["params"]["thickness"])
        self.out.append(f'<polygon id="{el["id"]}" class="wall" points="{self.points([a, b, b - n * t, a - n * t])}"/>')
        self.text((a + b) / 2 + n * 0.25, el["label"], "walllabel")

    def _joinery(self, el: dict, ends) -> None:
        (x0, _, z0), (x1, _, z1) = ends
        (u0, v0), (u1, v1) = self.px((x0, z0)), self.px((x1, z1))
        self.out.append(f'<path id="{el["id"]}" class="joinery" stroke="{OPENING_COLOURS[el["kind"]]}" d="M{u0:.1f},{v0:.1f} L{u1:.1f},{v1:.1f}"/>')
        self.out.append(f'<text class="joinerytext" x="{(u0 + u1) / 2 + 8:.1f}" y="{(v0 + v1) / 2 - 8:.1f}">{escape(el["id"])}</text>')

    @staticmethod
    def _free_spot(mid, placed) -> tuple[float, float]:
        offsets = [(0.0, 0.0)] + [(r * np.cos(k * np.pi / 4), r * np.sin(k * np.pi / 4)) for r in (26.0, 52.0, 78.0) for k in range(8)]
        spots = [(mid[0] + dx, mid[1] + dy) for dx, dy in offsets]
        return next((c for c in spots if all(np.hypot(c[0] - q[0], c[1] - q[1]) >= MARKER_GAP for q in placed)), spots[-1])


LEAF_TEXT = {
    ("door", "leaf-w"): "Largeur du vantail (le panneau qui pivote, sans le bâti ni le chambranle)",
    ("door", "leaf-h"): "Hauteur du vantail, du sol au haut du panneau",
    ("window", "leaf-w"): "Largeur de l'ouvrant (le battant qui s'ouvre, sans le dormant)",
    ("window", "leaf-h"): "Hauteur de l'ouvrant, du bas au haut du battant",
    ("double", "leaf-w"): "Largeur des deux vantaux ensemble, du bord extérieur de l'un au bord extérieur de l'autre",
    ("double", "leaf-h"): "Hauteur d'un vantail",
    ("opening", "w"): "Largeur du passage, d'un tableau à l'autre",
    ("opening", "h"): "Hauteur du passage, du sol au dessous du linteau",
    ("niche", "w"): "Largeur de la niche, d'un montant à l'autre",
    ("niche", "h"): "Hauteur, du sol (ou de l'appui) au dessous du linteau",
}


def _describe(dim: dict, arch: dict) -> str:
    kind, label = dim["kind"], dim["label"]
    if kind.startswith(("opening_", "leaf_")):
        el = _element(arch, dim)
        host = next((e for e in arch["elements"] if e["id"] == dim["elements"][-1] and e["kind"] in ("wall", "partition")), None)
        where = "" if host is None else f", sur {'le mur' if host['kind'] == 'wall' else 'la cloison'} {host['label']}"
        if kind == "opening_sill":
            how = "du sol au bord bas de la vitre, au milieu de la largeur" if el["kind"] == "window" else "du sol au bas de la niche, au milieu de la largeur"
            return f"{label}{where}, {how}"
        group = "double" if _is_double(el) else el["kind"]
        text = LEAF_TEXT.get((group, _suffix(dim)))
        if text is not None:
            return f"{text} — {KIND_WORDS.get(el['kind'], el['kind'])} {el['id']}{where}"
        return f"{label}{where}"
    if kind == "ceiling_height":
        return f"{label}, du sol au plafond, à l'endroit marqué sur le plan"
    if kind == "wall_length":  # above the skirtings, which shorten a reading taken at the floor
        return f"{label}, le long du mur à environ {fr(STATION_HEIGHT, 2)} m du sol, au-dessus des plinthes"
    return f"{label}, à environ {fr(STATION_HEIGHT, 2)} m du sol, entre les deux murs"


def _values(dim: dict) -> str:
    """Scan value, then the LLM and tape values when they were applied."""
    if dim.get("estimate"):
        return f"Estimation : {fr(dim['value'] * 100)} cm (ouverture moins cadre)"
    llm = dim.get("llm_value", dim["value"] if "scan_value" in dim and dim["source"] != "tape" else None)
    parts = [f"Scan : {fr(dim.get('scan_value', dim['value']) * 100)} cm"]
    if llm is not None:
        parts.append(f"LLM : {fr(llm * 100)} cm")
    if dim["source"] == "tape":
        parts.append(f"Mètre : {fr(dim['value'] * 100)} cm")
    return " · ".join(parts)


def _input_value(entry: dict | None) -> str:
    return "" if not entry or entry.get("value_m") is None else f"{round(entry['value_m'] * 100, 2):g}".replace(".", ",")


def _row_html(row: Row, arch: dict, crops: list[str], masked: bool = False, entry: dict | None = None) -> str:
    d = row.dim
    kind = f'<span class="kind">{escape(row.word)}</span> ' if row.word else ""
    guess = ' <span class="estimated">position estimée</span>' if row.estimated else ""
    badge = f' <span class="badge">{BADGE}</span>' if row.priority else ""
    alts = "".join(f'<img class="alt" alt="Autre vue" title="Autre vue" src="data:image/jpeg;base64,{c}">' for c in crops[1:])
    note = f'<p class="masked">{MASKED_NOTE}</p>' if masked else ""
    image = (f'<figure><img class="main" alt="Photo de {escape(d["id"])}" src="data:image/jpeg;base64,{crops[0]}">'
             f'<div class="alts">{alts}</div>{note}</figure>'
             if crops else '<figure class="nophoto">Pas de photo où les deux extrémités sont visibles.</figure>')
    return (f'<article class="row{" priority" if row.priority else ""}" id="row-{row.number}" data-dim="{escape(d["id"])}">'
            f'<div class="num">{row.number}</div><div class="body"><h3>{kind}{escape(_describe(d, arch))}</h3>'
            f'<p class="meta">{escape(_values(d))} · ± {fr(d["sigma"] * 100)} cm · source : {SOURCES.get(d["source"], escape(d["source"]))}{badge}{guess}</p>'
            f'<label class="entry">Mesure au mètre (cm) <input type="text" inputmode="decimal" autocomplete="off" '
            f'data-dim="{escape(d["id"])}" id="in-{row.number}" placeholder="ex. 90,5" value="{_input_value(entry)}"></label>'
            f'<input class="note" type="text" autocomplete="off" data-note="{escape(d["id"])}" placeholder="Note (facultatif)" '
            f'aria-label="Note" value="{escape((entry or {}).get("note", ""))}"></div>{image}</article>')


STYLE = """
:root{color-scheme:light}
body{font-family:system-ui,sans-serif;margin:0 auto;max-width:960px;padding:16px;color:#1f2328;background:#fafafa}
h1{font-size:1.5rem;margin:0 0 .5rem}h3{font-size:1rem;margin:0 0 .35rem}
.actions{position:sticky;top:0;z-index:2;display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:#fafafa;padding:8px 0}
button{font:inherit;padding:.5rem .9rem;border:1px solid #888;border-radius:6px;background:#fff;cursor:pointer}
button.main{background:#1f6feb;color:#fff;border-color:#1f6feb}
.plan{overflow-x:auto;background:#fff;border:1px solid #ddd;border-radius:6px;margin:12px 0}
.plan svg{max-width:100%;height:auto;display:block}
.row{display:grid;grid-template-columns:36px 1fr minmax(0,300px);gap:12px;background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}
.row.priority{border-left:5px solid #d1242f}
.num{font-weight:bold;background:#d1242f;color:#fff;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center}
.meta{margin:.2rem 0 .6rem;color:#57606a;font-size:.9rem}
.badge{display:inline-block;background:#ffebe9;color:#a40e26;border-radius:10px;padding:1px 8px;font-size:.8rem;margin-left:6px}
.entry{display:block;font-size:.9rem}.entry input{font:inherit;width:8rem;padding:.4rem;margin-left:6px}
figure{margin:0}figure img{width:100%;height:auto;border-radius:4px;display:block}.nophoto{color:#6e7781;font-size:.85rem}
figure .alts{display:flex;gap:6px;margin-top:6px}.masked{color:#9a6700;font-size:.8rem;margin:4px 0 0}.alts img{width:72px;height:54px;object-fit:cover;cursor:pointer;border:1px solid #bbb}
#status,#saved{color:#57606a;font-size:.9rem}
.kind{display:inline-block;background:#e6edf7;color:#0b3d8c;border-radius:10px;padding:1px 8px;font-size:.8rem;font-weight:normal;margin-right:4px}
.estimated{display:inline-block;background:#fff4d6;color:#7a5200;border-radius:10px;padding:1px 8px;font-size:.8rem;margin-left:6px}
.note{display:block;font:inherit;font-size:.85rem;width:100%;box-sizing:border-box;margin-top:6px;padding:.3rem}
@media(max-width:640px){.row{grid-template-columns:36px 1fr}.row figure{grid-column:1 / -1}}"""

SCRIPT = """
(function () {
  function swap(alt) {
    var main = alt.parentNode.parentNode.querySelector('img.main'), src = main.src;
    main.src = alt.src;
    alt.src = src;
  }
  Array.prototype.forEach.call(document.querySelectorAll('img.alt'), function (a) { a.addEventListener('click', function () { swap(a); }); });
  var inputs = Array.prototype.slice.call(document.querySelectorAll('input[data-dim]'));
  var notes = Array.prototype.slice.call(document.querySelectorAll('input[data-note]'));
  var storageKey = 'room-model-tape:' + CAPTURE;
  var status = document.getElementById('status'), saved = document.getElementById('saved');
  var state = {};
  var handle = null, pending = null, timer = null;

  function parse(text) { var v = parseFloat(String(text).replace(',', '.')); return isFinite(v) && v > 0 ? v : null; }
  function cm(v) { return String(Math.round(v * 10000) / 100).replace('.', ','); }
  function clock() { var d = new Date(); return ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2); }
  function say(text) { saved.textContent = text; }

  function normalize(doc) {
    var out = {}, src = doc && doc.version === 2 && doc.values ? doc.values : doc;
    Object.keys(src || {}).forEach(function (id) {
      var raw = src[id], e = raw !== null && typeof raw === 'object' ? raw : { value_m: raw };
      var v = e.value_m === null || e.value_m === undefined ? null : Number(e.value_m);
      if (v === null ? String(e.note || '').trim() : isFinite(v) && v > 0) out[id] = { value_m: v, date: String(e.date || ''), note: String(e.note || '') };
    });
    return out;
  }
  function merge(entries, force) {
    Object.keys(entries).forEach(function (id) {
      var mine = state[id];
      if (force || !mine || String(entries[id].date || '') > String(mine.date || '')) state[id] = entries[id];
    });
  }
  function collect() {
    var values = {};
    Object.keys(state).forEach(function (id) { values[id] = { value_m: state[id].value_m, date: state[id].date, note: state[id].note }; });
    return { version: 2, capture: CAPTURE, values: values };
  }
  function show() {
    var rows = {};
    inputs.forEach(function (i) { rows[i.getAttribute('data-dim')] = true; });
    var box = document.getElementById('legacy'), list = document.getElementById('legacylist');
    var orphans = Object.keys(state).filter(function (id) { return !rows[id]; });
    list.textContent = '';
    orphans.forEach(function (id) {
      var li = document.createElement('li'), e = state[id];
      li.textContent = id + ' : ' + (e.value_m === null ? 'sans valeur' : cm(e.value_m) + ' cm') + (e.note ? ' (' + e.note + ')' : '');
      list.appendChild(li);
    });
    box.hidden = !orphans.length;
    inputs.forEach(function (i) { var e = state[i.getAttribute('data-dim')]; i.value = e && e.value_m !== null ? cm(e.value_m) : ''; });
    notes.forEach(function (n) { var e = state[n.getAttribute('data-note')]; n.value = e ? e.note : ''; });
    count();
  }
  function edit(id) {
    var value = parse(document.querySelector('input[data-dim="' + id + '"]').value);
    var note = document.querySelector('input[data-note="' + id + '"]').value;
    var metres = value === null ? null : Math.round(value * 100) / 10000, before = state[id];
    if (value === null && !note.trim()) { delete state[id]; }
    else if (!before || before.value_m !== metres || before.note !== note) state[id] = { value_m: metres, date: new Date().toISOString(), note: note };
    count();
    clearTimeout(timer);
    timer = setTimeout(persist, 600);
  }
  function count() { status.textContent = Object.keys(state).filter(function (id) { return state[id].value_m !== null; }).length + ' cote(s) renseignée(s), ' + inputs.length + ' lignes'; }

  function saveLocal() { try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch (e) {} }
  async function writeFile() {
    if (!handle) return;
    try {
      var w = await handle.createWritable();
      await w.write(JSON.stringify(collect(), null, 2));
      await w.close();
      say('enregistré à ' + clock() + ' dans ' + handle.name);
    } catch (e) { say('écriture impossible dans le fichier lié (' + e.name + ') : la copie du navigateur est conservée'); }
  }
  function persist() { saveLocal(); writeFile(); }

  try { merge(normalize(PREFILL), true); } catch (e) {}
  try {
    var local = JSON.parse(localStorage.getItem(storageKey) || '{}');
    Object.keys(local).forEach(function (id) { if (local[id] && typeof local[id] === 'object') { var one = {}; one[id] = local[id]; merge(one, false); } });
  } catch (e) {}
  function migrate() {
    var rows = {}, current = storageKey;
    inputs.forEach(function (i) { rows[i.getAttribute('data-dim')] = true; });
    var keys = ['room-model-tape:Relevé au mètre ruban'];
    for (var k = 0; k < localStorage.length; k++) { var name = localStorage.key(k); if (name.indexOf('room-model-tape:') === 0) keys.push(name); }
    keys.forEach(function (key) {
      var old = key === current ? null : localStorage.getItem(key);
      if (!old) return;
      try {
      var doc = JSON.parse(old);
      var plain = function (id) { return typeof doc[id] === 'string' || typeof doc[id] === 'number'; };
      var ids = doc && typeof doc === 'object' ? Object.keys(doc) : [];
      if (!ids.some(plain)) return;
      ids.filter(plain).forEach(function (id) {
        var v = parse(doc[id]), mine = state[id];
        if (v === null || (mine && mine.value_m !== null)) return;
        var frame = /^(Door|Window)_[0-9]+-[wh]$/.test(id);
        state[id] = { value_m: Math.round(v * 100) / 10000, date: '', note: mine ? mine.note : rows[id] ? '' : frame ? 'ancienne page — mesure bâti compris' : 'ancienne page' };
      });
      } catch (e) {}
    });
  }
  try { migrate(); saveLocal(); } catch (e) {}
  show();
  inputs.forEach(function (i) { i.addEventListener('input', function () { edit(i.getAttribute('data-dim')); }); });
  notes.forEach(function (n) { n.addEventListener('input', function () { edit(n.getAttribute('data-note')); }); });

  function download() {
    var blob = new Blob([JSON.stringify(collect(), null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'measurements.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
  document.getElementById('download').addEventListener('click', download);
  document.getElementById('copy').addEventListener('click', function () {
    var text = JSON.stringify(collect(), null, 2);
    function done(ok) { say(ok ? 'copié dans le presse-papiers' : 'copie impossible : utilisez le téléchargement'); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
      return;
    }
    var area = document.createElement('textarea');
    area.value = text;
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(area);
    done(ok);
  });

  var picker = document.getElementById('importfile');
  document.getElementById('import').addEventListener('click', function () { picker.click(); });
  picker.addEventListener('change', function () {
    var file = picker.files && picker.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      try { merge(normalize(JSON.parse(String(reader.result))), true); show(); persist(); say('fichier importé'); }
      catch (e) { say('fichier illisible : ' + e.message); }
    };
    reader.readAsText(file);
    picker.value = '';
  });

  var supported = 'showSaveFilePicker' in window && 'showOpenFilePicker' in window && 'indexedDB' in window;
  var linkButton = document.getElementById('link'), regrant = document.getElementById('regrant');
  function store(mode, work) {
    return new Promise(function (resolve, reject) {
      var open = indexedDB.open('room-model-tape', 1);
      open.onupgradeneeded = function () { open.result.createObjectStore('handles'); };
      open.onerror = function () { reject(open.error); };
      open.onsuccess = function () {
        try {
          var tx = open.result.transaction('handles', mode), req = work(tx.objectStore('handles'));
          tx.oncomplete = function () { resolve(req.result); };
          tx.onerror = function () { reject(tx.error); };
        } catch (e) { reject(e); }
      };
    });
  }
  async function attach(h) {
    handle = h;
    try {
      var text = await (await h.getFile()).text();
      if (text.trim()) merge(normalize(JSON.parse(text)), false);
    } catch (e) {}
    show();
    saveLocal();
    await writeFile();
    linkButton.textContent = 'Fichier lié : ' + h.name;
  }
  if (supported) {
    linkButton.hidden = false;
    linkButton.addEventListener('click', async function () {
      try {
        var h = await window.showSaveFilePicker({ suggestedName: 'measurements.json', types: [{ description: 'Mesures', accept: { 'application/json': ['.json'] } }] });
        try { await store('readwrite', function (s) { return s.put(h, storageKey); }); } catch (e) { say('lien non mémorisé pour la prochaine ouverture'); }
        await attach(h);
      } catch (e) { if (e.name !== 'AbortError') say('liaison impossible : ' + e.message); }
    });
    regrant.addEventListener('click', async function () {
      if (!pending) return;
      try {
        if ((await pending.requestPermission({ mode: 'readwrite' })) === 'granted') { regrant.hidden = true; await attach(pending); pending = null; }
      } catch (e) { say('autorisation refusée : ' + e.message); }
    });
    store('readonly', function (s) { return s.get(storageKey); }).then(async function (h) {
      if (!h) return;
      if ((await h.queryPermission({ mode: 'readwrite' })) === 'granted') { await attach(h); return; }
      pending = h;
      regrant.hidden = false;
      say('fichier lié : ' + h.name + ', autorisation à renouveler');
    }).catch(function () {});
  }
})();"""


def render(arch: dict, rows: list[Row], crops: dict[int, list[str]], masked: frozenset[int] = frozenset(),
           store: dict | None = None, capture_id: str = "") -> str:
    """The sheet as one HTML document; `crops` maps a row number to base64 JPEGs, the chosen view first.

    `store` is {dimension id: {"value_m", "date", "note"}}, the measurements already taken: it pre-fills the inputs and
    travels whole in the page, so a value without a row is written back by the next save. `capture_id` keys the browser copy.
    """
    store = store or {}
    body = "\n".join(_row_html(r, arch, crops.get(r.number, []), r.number in masked, store.get(r.dim["id"])) for r in rows)
    data = f"var PREFILL = {_js(store)};\nvar CAPTURE = {_js(capture_id)};\n"
    return (f'<!doctype html>\n<html lang="fr"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>Relevé au mètre ruban</title>'
            f'<style>{STYLE}</style></head><body>\n<h1>Relevé au mètre ruban</h1>\n'
            f'<p>Mesurez chaque cote entre les deux points rouges de la photo, au repère numéroté du plan, et saisissez-la en centimètres. '
            f'Les cotes marquées « {BADGE} » sont les moins sûres. Les saisies sont gardées dans ce navigateur ; '
            f'« Lier le fichier de mesures » les enregistre en continu dans un fichier à donner à l\'option <code>--tape</code>.</p>\n'
            f'<div class="actions"><button class="main" id="link" type="button" hidden>Lier le fichier de mesures</button>'
            f'<button id="regrant" type="button" hidden>Autoriser à nouveau l\'accès au fichier</button>'
            f'<button id="download" type="button">Télécharger measurements.json</button>'
            f'<button id="import" type="button">Importer</button><input id="importfile" type="file" accept=".json,application/json" hidden>'
            f'<button id="copy" type="button">Copier</button><span id="status"></span><span id="saved"></span></div>\n'
            f'<section id="legacy" hidden><h2>Anciennes valeurs</h2><p>Saisies d\'une ancienne version de la page, sans ligne correspondante : elles restent dans le fichier de mesures.</p><ul id="legacylist"></ul></section>\n'
            f'<div class="plan">{_SurveyPlan(arch).draw_survey(rows, arch)}</div>\n{body}\n<script>{data}{SCRIPT}</script>\n</body></html>\n')


def _js(value) -> str:
    """JSON safe inside a <script> element."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _crop(photo: PosedPhoto, p0: list[float], p1: list[float], quality: int = 75) -> str:
    """Base64 JPEG of the upright photo around a span, with the span and its two ends drawn."""
    uv = to_upright(photo, project(photo, np.array([p0, p1]))[0])
    width, height = upright_size(photo)
    lo, hi = uv.min(0), uv.max(0)
    half = (hi - lo) / 2 + max(0.35 * float((hi - lo).max()), 0.08 * max(width, height))
    box0, box1 = np.maximum((lo + hi) / 2 - half, 0), np.minimum((lo + hi) / 2 + half, [width, height])
    scale = min(1.0, CROP_SIDE / float((box1 - box0).max()))
    image = upright(photo).crop((*map(int, box0), *map(int, np.ceil(box1))))
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    draw = ImageDraw.Draw(image)
    a, b = ((uv[i] - np.floor(box0)) * scale for i in (0, 1))
    draw.line([tuple(a), tuple(b)], fill=(230, 40, 40), width=3)
    for x, y in (a, b):
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(230, 40, 40), outline=(255, 255, 255), width=2)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=quality)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _host_normal(arch: dict, dim: dict) -> np.ndarray | None:
    host = next((e for e in arch["elements"] if dim["kind"].startswith(("opening_", "leaf_")) and e["id"] == dim["elements"][-1]
                 and e["kind"] in ("wall", "partition")), None)
    if host is None:
        return None
    az = arch["parameters"][host["params"]["azimuth"]]["value"]
    return np.array([np.cos(az), 0.0, np.sin(az)])


def _crops(ctx: StageContext, arch: dict, rows: list[Row]) -> tuple[dict[int, list[str]], frozenset[int]]:
    w2r = colmajor(arch["frame"]["world_to_room"], 4)
    photos = load_posed_photos(ctx, w2r)
    if not photos or not rows:
        return {}, frozenset()
    maps = DepthMaps(*load_room_mesh(ctx, w2r))
    boxes = room_boxes(load_roomplan(ctx.capture, ctx.room), w2r)
    crops, masked = {}, set()
    for row in rows:
        ends, normal = np.array([row.p0, row.p1]), _host_normal(arch, row.dim)
        k = 1 + MAX_ALTERNATIVES
        views = (best_views(photos, ends, maps, k, normal, boxes=boxes, min_cos=SURVEY_FACING_COS)
                 or best_views(photos, ends, maps, k, normal, LOOSE_TOLERANCE, boxes, SURVEY_FACING_COS))
        if views:
            crops[row.number] = [_crop(v, row.p0, row.p1) for v in views]
            if boxes and occluded_fraction(views[0].centre, ends, boxes) > 0:
                masked.add(row.number)
    return crops, frozenset(masked)


def _existing_store(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return tape_merge.read_store(doc) if isinstance(doc, dict) else {}


def write(ctx: StageContext, arch: dict) -> Path:
    """measurements/index.html and measurements/plan_numbers.json (number -> dimension id) under the output folder."""
    rows = numbered_rows(arch)
    folder = ctx.out / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plan_numbers.json").write_text(json.dumps({str(r.number): r.dim["id"] for r in rows}, indent=2), encoding="utf-8")
    page = folder / "index.html"
    store = _existing_store(ctx.out / "measurements.json")
    page.write_text(render(arch, rows, *_crops(ctx, arch, rows), store, ctx.out.name), encoding="utf-8")
    return page
