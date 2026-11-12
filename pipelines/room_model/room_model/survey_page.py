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
from room_model.architecture import STATION_HEIGHT, dimension_endpoints
from room_model.capture import colmajor
from room_model.context import StageContext
from room_model.fmt import fr
from room_model.photo_geom import DepthMaps, PosedPhoto, best_views, load_posed_photos, load_room_mesh, project, to_upright, upright, upright_size

GROUP = {"opening_width": 0, "opening_height": 0, "opening_sill": 0, "width": 1, "length": 1, "ceiling_height": 2}
SOURCES = {"lidar": "LiDAR", "roomplan": "RoomPlan", "assumed": "valeur supposée", "llm": "LLM"}
PRIORITY_SIGMA = 0.02  # m
PRIORITY_SOURCES = ("roomplan", "llm", "assumed")
BADGE = "à reprendre en priorité"
LOOSE_TOLERANCE = 0.06  # m; second try when no photo passes the 3 cm test (a skirting or a frame hides an end)
CROP_SIDE = 480  # px on the long side of a crop
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


def is_priority(dim: dict) -> bool:
    return dim["sigma"] >= PRIORITY_SIGMA or dim["source"] in PRIORITY_SOURCES or bool(dim["flags"])


def numbered_rows(arch: dict) -> list[Row]:
    """Rows of the sheet: priority first, then openings, stations, ceiling heights; dimensions without a span are left out."""
    found = []
    for dim in arch["dimensions"]:
        ends = dimension_endpoints(arch, dim["id"]) if dim["kind"] in GROUP else None
        if ends is not None:
            found.append((dim, *ends))
    found.sort(key=lambda f: (not is_priority(f[0]), GROUP[f[0]["kind"]]))
    return [Row(i, dim, p0, p1, is_priority(dim)) for i, (dim, p0, p1) in enumerate(found, start=1)]


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


def _describe(dim: dict, arch: dict) -> str:
    kind, label = dim["kind"], dim["label"]
    if kind.startswith("opening_"):
        host = next((e for e in arch["elements"] if e["id"] == dim["elements"][-1] and e["kind"] in ("wall", "partition")), None)
        where = "" if host is None else f", sur {'le mur' if host['kind'] == 'wall' else 'la cloison'} {host['label']}"
        how = {"opening_width": "entre les tableaux, à mi-hauteur",
               "opening_height": "du bas au haut du passage, au milieu de la largeur",
               "opening_sill": "du sol au bord bas de la vitre, au milieu de la largeur"}[kind]
        return f"{label}{where}, {how}"
    if kind == "ceiling_height":
        return f"{label}, du sol au plafond, à l'endroit marqué sur le plan"
    return f"{label}, à environ {fr(STATION_HEIGHT, 2)} m du sol, entre les deux murs"


def _values(dim: dict) -> str:
    if "scan_value" in dim:
        return f"Scan : {fr(dim['scan_value'] * 100)} cm · LLM : {fr(dim['value'] * 100)} cm"
    return f"Scan : {fr(dim['value'] * 100)} cm"


def _row_html(row: Row, arch: dict, crop: str | None) -> str:
    d = row.dim
    badge = f' <span class="badge">{BADGE}</span>' if row.priority else ""
    image = (f'<figure><img alt="Photo de {escape(d["id"])}" src="data:image/jpeg;base64,{crop}"></figure>'
             if crop else '<figure class="nophoto">Pas de photo où les deux extrémités sont visibles.</figure>')
    return (f'<article class="row{" priority" if row.priority else ""}" id="row-{row.number}" data-dim="{escape(d["id"])}">'
            f'<div class="num">{row.number}</div><div class="body"><h3>{escape(_describe(d, arch))}</h3>'
            f'<p class="meta">{escape(_values(d))} · ± {fr(d["sigma"] * 100)} cm · source : {SOURCES.get(d["source"], escape(d["source"]))}{badge}</p>'
            f'<label class="entry">Mesure au mètre (cm) <input type="text" inputmode="decimal" autocomplete="off" '
            f'data-dim="{escape(d["id"])}" id="in-{row.number}" placeholder="ex. 90,5"></label></div>{image}</article>')


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
#status{color:#57606a;font-size:.9rem}
@media(max-width:640px){.row{grid-template-columns:36px 1fr}.row figure{grid-column:1 / -1}}
"""

SCRIPT = """
(function () {
  var inputs = Array.prototype.slice.call(document.querySelectorAll('input[data-dim]'));
  var key = 'room-model-tape:' + document.title;
  var status = document.getElementById('status');
  function parse(text) { var v = parseFloat(String(text).replace(',', '.')); return isFinite(v) && v > 0 ? v : null; }
  function collect() {
    var out = {};
    inputs.forEach(function (i) { var v = parse(i.value); if (v !== null) out[i.getAttribute('data-dim')] = Math.round(v * 100) / 10000; });
    return out;
  }
  function save() {
    var raw = {};
    inputs.forEach(function (i) { if (i.value.trim()) raw[i.getAttribute('data-dim')] = i.value; });
    try { localStorage.setItem(key, JSON.stringify(raw)); } catch (e) {}
    status.textContent = Object.keys(collect()).length + ' cote(s) renseignée(s) sur ' + inputs.length;
  }
  try {
    var saved = JSON.parse(localStorage.getItem(key) || '{}');
    inputs.forEach(function (i) { var v = saved[i.getAttribute('data-dim')]; if (v) i.value = v; });
  } catch (e) {}
  inputs.forEach(function (i) { i.addEventListener('input', save); });
  save();
  document.getElementById('download').addEventListener('click', function () {
    var blob = new Blob([JSON.stringify(collect(), null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'measurements.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });
  document.getElementById('copy').addEventListener('click', function () {
    var text = JSON.stringify(collect(), null, 2);
    function done(ok) { status.textContent = ok ? 'Copié dans le presse-papiers.' : 'Copie impossible : utilisez le téléchargement.'; }
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
})();
"""


def render(arch: dict, rows: list[Row], crops: dict[int, str]) -> str:
    """The sheet as one HTML document; `crops` maps a row number to a base64 JPEG."""
    body = "\n".join(_row_html(r, arch, crops.get(r.number)) for r in rows)
    return (f'<!doctype html>\n<html lang="fr"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>Relevé au mètre ruban</title>'
            f'<style>{STYLE}</style></head><body>\n<h1>Relevé au mètre ruban</h1>\n'
            f'<p>Mesurez chaque cote entre les deux points rouges de la photo, au repère numéroté du plan, et saisissez-la en centimètres. '
            f'Les cotes marquées « {BADGE} » sont les moins sûres. Les saisies restent dans ce navigateur ; '
            f'le fichier téléchargé est à donner à la commande <code>compare-tape</code>.</p>\n'
            f'<div class="actions"><button class="main" id="download" type="button">Télécharger measurements.json</button>'
            f'<button id="copy" type="button">Copier</button><span id="status"></span></div>\n'
            f'<div class="plan">{_SurveyPlan(arch).draw_survey(rows, arch)}</div>\n{body}\n<script>{SCRIPT}</script>\n</body></html>\n')


def _crop(photo: PosedPhoto, p0: list[float], p1: list[float]) -> str:
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
    image.save(out, "JPEG", quality=72)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _host_normal(arch: dict, dim: dict) -> np.ndarray | None:
    host = next((e for e in arch["elements"] if dim["kind"].startswith("opening_") and e["id"] == dim["elements"][-1]
                 and e["kind"] in ("wall", "partition")), None)
    if host is None:
        return None
    az = arch["parameters"][host["params"]["azimuth"]]["value"]
    return np.array([np.cos(az), 0.0, np.sin(az)])


def _crops(ctx: StageContext, arch: dict, rows: list[Row]) -> dict[int, str]:
    w2r = colmajor(arch["frame"]["world_to_room"], 4)
    photos = load_posed_photos(ctx, w2r)
    if not photos or not rows:
        return {}
    maps = DepthMaps(*load_room_mesh(ctx, w2r))
    crops = {}
    for row in rows:
        ends, normal = np.array([row.p0, row.p1]), _host_normal(arch, row.dim)
        views = best_views(photos, ends, maps, 1, normal) or best_views(photos, ends, maps, 1, normal, LOOSE_TOLERANCE)
        if views:
            crops[row.number] = _crop(views[0], row.p0, row.p1)
    return crops


def write(ctx: StageContext, arch: dict) -> Path:
    """measurements/index.html and measurements/plan_numbers.json (number -> dimension id) under the output folder."""
    rows = numbered_rows(arch)
    folder = ctx.out / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plan_numbers.json").write_text(json.dumps({str(r.number): r.dim["id"] for r in rows}, indent=2), encoding="utf-8")
    page = folder / "index.html"
    page.write_text(render(arch, rows, _crops(ctx, arch, rows)), encoding="utf-8")
    return page
