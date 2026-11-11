"""Edge pick stage: the LLM picks, for each edge of an element, one of the lettered candidate lines drawn on
rectified photos; it never gives coordinates.

`prepare` writes edge_pick/pending/ (one sheet per element, candidates.json, prompt.md); `run` calls `claude -p`
there, validates the letters and writes edge_pick/answers.json (each picked edge with its triangulated position,
depth and sigma) and edge_pick/rejected.json. `merge` turns the picks into anchored elements for `anchor.apply`.
"""
from __future__ import annotations

import copy
import json
import shutil
import string
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from room_model import anchor, ortho, vision
from room_model.anchor import MARGIN, SIDES, SNAP
from room_model.context import NoPhotos, StageContext, output_path, require
from room_model.photo_geom import LOOSE_TOLERANCE, load_posed_photos

SCHEMA = "room-model/edge-pick-1"
PICK_STATUSES = ("anchored", "candidates")
MAX_LETTERS = 6  # per edge: more lines crowd a strip and a model tells them apart less reliably
SAME_LINE = anchor.SAME_LINE
UNSEEN_SIGMA = 0.10  # m; a line no view could snap to keeps its position unconfirmed, far above the apply gates
SHEET_SIDE = vision.SHEET_SIDE
OVERVIEW_H = 1000  # px of the element overview on a sheet
STRIP, ZOOM_LEN, ZOOM = 0.12, 0.40, 2  # zoom strips: m across the edge each side, m along it, upscaling
COLOURS = {"left": (230, 40, 40), "right": (30, 120, 255), "bottom": (240, 190, 0), "top": (20, 190, 70)}
COLOUR_NAMES = {"left": "red", "right": "blue", "bottom": "yellow", "top": "green"}
QUESTIONS = {
    "door": {"left": "the left edge of the door leaf (vantail), the moving panel itself, not the frame (huisserie) nor the casing (chambranle); for a double door, the outer edge of the left leaf",
             "right": "the right edge of the door leaf (vantail), same rule; for a double door, the outer edge of the right leaf",
             "top": "the top edge of the door leaf (vantail), not the frame head nor the casing"},
    "window": {"left": "the outer left edge of the opening sash (ouvrant), both sashes together: not the fixed frame (dormant), the casing or a curtain",
               "right": "the outer right edge of the opening sash (ouvrant), both sashes together",
               "bottom": "the bottom edge of the sash (ouvrant)", "top": "the top edge of the sash (ouvrant)"},
    "niche": {"left": "the left jamb face (tableau) where the recess starts, not a casing or frame around it",
              "right": "the right jamb face (tableau) where the recess starts",
              "bottom": "the top of the recess sill (appui)", "top": "the underside of the recess head (linteau)"},
    "opening": {"left": "the left jamb face (tableau) of the passage, not a casing around it",
                "right": "the right jamb face (tableau) of the passage", "top": "the underside of the lintel (linteau)",
                "bottom": "the floor of the passage"},
}


def _known_edges(item: dict) -> dict:
    return item.get("edges") or (item.get("proposal") or {}).get("edges") or {}


def _footprint(pos: float, delta: float, cams) -> np.ndarray:
    """Where a line at (pos, delta) shows on the delta = 0 ortho of each camera (pos, delta)."""
    return np.array([pos + (c - pos) * -delta / (cd - delta) for c, cd in cams])


def _distinct(lines: list[dict], cams) -> list[dict]:
    """One entry per line (footprints within SAME_LINE in every view, origins merged), mask and scan lines first."""
    out = []
    for line in lines:
        at = _footprint(line["pos"], line["delta"], cams)
        same = next((o for o in out if np.abs(o["at"] - at).max() <= SAME_LINE), None)
        if same is None:
            out.append({**line, "at": at})
        else:
            same["origin"] += [o for o in line["origin"] if o not in same["origin"]]
    keep = [ln for ln in out if {"mask", "model"} & set(ln["origin"])]
    keep += [ln for ln in out if ln not in keep][:MAX_LETTERS - len(keep)]
    return sorted(keep, key=lambda ln: ln["pos"])


def _lines(side: str, item: dict, rect, cams) -> list[dict]:
    """The mask edge, the anchor's candidate lines and the scan edge of one side, as drawn (before confirmation)."""
    known = _known_edges(item).get(side)
    raw = [] if known is None or known.get("rule") else [(known["pos"], known["delta"], "mask")]
    raw += [(c["pos"], c.get("delta", 0.0), "gradient") for c in item.get("candidates", {}).get(side, [])]
    raw.append((dict(zip(SIDES, rect))[side], 0.0, "model"))
    return [{"pos": pos, "delta": delta, "origin": [origin]} for pos, delta, origin in raw]


def _confirm(line: dict, side: str, faces, cams, bands, bounds) -> dict:
    """The line triangulated from its snapped position in each view; UNSEEN_SIGMA when no view shows it."""
    axis = anchor._axis(side)
    obs = []
    for o, cam, at in zip(faces, cams, line["at"]):
        snapped = ortho.snap(o, axis, bands[axis], float(at), SNAP)
        if snapped is not None:
            obs.append((snapped, cam[0], cam[1]))
    fix = ortho.triangulate(obs, bounds)
    if fix is None:
        return {"pos": line["pos"], "delta": line["delta"], "sigma": UNSEEN_SIGMA, "views": 0, "rms": 0.0,
                "depths": [line["delta"], line["delta"]], "obs": []}
    return {**anchor.edge_entry(fix), "obs": [list(o) for o in obs]}


def _element(arch: dict, el: dict, item: dict, photos: list) -> dict | None:
    """Candidates of one element: per asked side, lettered lines with their confirmation, and the sheet drawing data
    (the view showing most of the element)."""
    frame, rect = anchor.face_frame(arch, el), anchor.scan_rect(arch, el)
    host = next(e for e in arch["elements"] if e["id"] == el["host"])
    bounds = (-arch["parameters"][host["params"]["thickness"]]["value"], LOOSE_TOLERANCE)
    known = _known_edges(item)
    vertical = [known[s] for s in ("left", "right") if s in known]
    reference = float(np.mean([e["delta"] for e in vertical])) if vertical else 0.0
    plane = (min(e["depths"][0] for e in vertical), max(e["depths"][1] for e in vertical)) if vertical else bounds
    s0, s1, y0, y1 = rect
    box = (s0 - MARGIN, s1 + MARGIN, y0 - MARGIN, y1 + MARGIN)
    views = [(p, anchor._image(p.path)) for p in photos]
    faces = [ortho.ortho(p, image, frame, box) for p, image in views]
    cams_full = [ortho.camera_coords(frame, p) for p, _ in views]
    bands = anchor._bands(rect)
    floor_based = item.get("floor_based", anchor.is_floor_based(el["kind"], rect))
    sides = [s for s in SIDES if not (floor_based and s == "bottom")]
    letters, out = iter(string.ascii_uppercase), {}
    for side in sides:
        axis = anchor._axis(side)
        cams = [(c[0] if axis == "s" else c[1], c[2]) for c in cams_full]
        drawn = [{**line, "at": _footprint(line["pos"], line["delta"], cams)} for line in _lines(side, item, rect, cams)]
        confirmed = [{**_confirm(line, side, faces, cams, bands, bounds if axis == "s" else plane), "origin": line["origin"]}
                     for line in drawn]
        out[side] = [{"letter": next(letters), "shown": c["pos"], **{k: v for k, v in c.items() if k != "at"}}
                     for c in _distinct(confirmed, cams)]
    return {"kind": el["kind"], "floor_based": floor_based, "sides": out, "sheet": f"{el['id']}.jpg",
            "_draw": (views[int(np.argmax([anchor._rect_pixels(o, rect) for o in faces]))], frame, box, reference)}


# --- sheets --------------------------------------------------------------------------------------------------------

def _draw_lines(image: Image.Image, o: ortho.Ortho, sides: dict, scale: float, offset=(0.0, 0.0)) -> None:
    draw, font = ImageDraw.Draw(image), vision._font(max(14, int(9 * scale)))
    for side, entries in sides.items():
        for k, c in enumerate(entries):
            if anchor._axis(side) == "s":
                x = (float(o.col(c["shown"])) + 0.5 - offset[0]) * scale
                draw.line([(x, 0), (x, image.height)], fill=COLOURS[side], width=2)
                at = (x + 3, 6 + (k % 3) * (font.size + 4))
            else:
                y = (float(o.row(c["shown"])) + 0.5 - offset[1]) * scale
                draw.line([(0, y), (image.width, y)], fill=COLOURS[side], width=2)
                at = (6 + (k % 3) * (2 * font.size), y + 3)
            box = draw.textbbox(at, c["letter"], font=font)
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text(at, c["letter"], font=font, fill=COLOURS[side])


def _sheet(path: Path, ident: str, entry: dict) -> None:
    """Overview of the element on its rectified face with every candidate line, and a zoom strip per side."""
    (photo, image), frame, box, reference = entry["_draw"]
    o = ortho.ortho(photo, image, frame, box, reference)
    rgb = Image.fromarray(o.rgb)
    scale = OVERVIEW_H / rgb.height
    overview = rgb.resize((max(1, int(rgb.width * scale)), OVERVIEW_H), Image.BILINEAR)
    _draw_lines(overview, o, entry["sides"], scale)
    strips = []
    for side, entries in entry["sides"].items():
        mid = float(np.median([c["shown"] for c in entries]))
        s0, s1, y0, y1 = box
        if anchor._axis(side) == "s":
            ym = (y0 + y1) / 2
            crop = (o.col(mid - STRIP), o.row(ym + ZOOM_LEN / 2), o.col(mid + STRIP), o.row(ym - ZOOM_LEN / 2))
        else:
            sm = (s0 + s1) / 2
            crop = (o.col(sm - ZOOM_LEN / 2), o.row(mid + STRIP), o.col(sm + ZOOM_LEN / 2), o.row(mid - STRIP))
        crop = tuple(int(round(v)) for v in crop)
        strip = rgb.crop(crop)
        strip = strip.resize((strip.width * ZOOM, strip.height * ZOOM), Image.BILINEAR)
        _draw_lines(strip, o, {side: entries}, ZOOM, offset=(crop[0], crop[1]))
        strips.append(strip)
    width = overview.width + sum(s.width for s in strips) + vision.GAP * (len(strips) + 1)
    height = vision.HEADER + max([overview.height, *(s.height for s in strips)])
    sheet = Image.new("RGB", (width, height), (40, 40, 48))
    ImageDraw.Draw(sheet).text((10, 8), f"{ident}  ({entry['kind']})", font=vision._font(26), fill=(255, 255, 255))
    x = 0
    for part in (overview, *strips):
        sheet.paste(part, (x, vision.HEADER))
        x += part.width + vision.GAP
    if max(sheet.size) > SHEET_SIDE:
        f = SHEET_SIDE / max(sheet.size)
        sheet = sheet.resize((int(sheet.width * f), int(sheet.height * f)), Image.BILINEAR)
    sheet.save(path, quality=vision.JPEG_QUALITY)


def _prompt(candidates: dict) -> str:
    lines = ["# Picking the edges of doors, windows, niches and openings", "",
             "Each JPEG in this folder is one element, to open with the Read tool. On the left, a straightened view of the "
             "element on its wall (vertical lines are vertical, distances are true); on the right, zoomed strips of each edge. "
             "Coloured lines with a letter are candidate positions: "
             + ", ".join(f"{COLOUR_NAMES[s]} = {s} edge" for s in SIDES) + ".",
             "", "For every element and every edge listed below, answer the letter of the line lying exactly on the edge "
             "described, or null when no line does. Never guess: null is a good answer.", "", "## Elements", ""]
    for ident, entry in candidates.items():
        lines.append(f"### {ident} ({entry['kind']}), sheet `{entry['sheet']}`")
        for side, entries in entry["sides"].items():
            letters = ", ".join(f"`{c['letter']}`" for c in entries)
            lines.append(f"- {side}: {QUESTIONS[entry['kind']][side]}. Letters: {letters}.")
        lines.append("")
    example = {"elements": [{"id": "Door_1", "picks": {"left": "B", "right": None, "top": "H"}}]}
    lines += ["## Answer", "", "Only one JSON object, nothing else, for example:", "", json.dumps(example)]
    return "\n".join(lines) + "\n"


def prepare(ctx: StageContext) -> Path:
    """Writes edge_pick/pending/ for the context's capture and returns it."""
    anchors = json.loads(require(ctx, "anchor").read_text(encoding="utf-8"))
    arch = anchor.capture_arch(ctx)
    elements = {e["id"]: e for e in arch["elements"]}
    photos = {p.name: p for p in load_posed_photos(ctx)}
    pending = ctx.out / "edge_pick" / "pending"
    shutil.rmtree(pending, ignore_errors=True)
    pending.mkdir(parents=True)
    candidates = {}
    for ident, item in sorted(anchors.get("elements", {}).items()):
        el, seen = elements.get(ident), [photos[n] for n in item.get("photos", []) if n in photos]
        if item.get("status") not in PICK_STATUSES or el is None or el["kind"] != item.get("kind") or not seen:
            continue
        entry = _element(arch, el, item, seen)
        _sheet(pending / entry["sheet"], ident, entry)
        candidates[ident] = {k: v for k, v in entry.items() if k != "_draw"}
    (pending / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    (pending / "prompt.md").write_text(_prompt(candidates), encoding="utf-8")
    return pending


def validate(answer: object, candidates: dict) -> tuple[dict, list[dict]]:
    """({element id: {side: letter or None}}, rejected items); an element with any invalid pick is rejected whole."""
    if not isinstance(answer, dict) or not isinstance(answer.get("elements"), list):
        return {}, [{"where": "result", "reason": "expected an object with an `elements` list"}]
    clean, rejected = {}, []
    for i, item in enumerate(answer["elements"]):
        ident = item.get("id") if isinstance(item, dict) else None
        where = ident if isinstance(ident, str) else f"elements[{i}]"
        entry, picks = candidates.get(ident), item.get("picks") if isinstance(item, dict) else None
        if entry is None or not isinstance(picks, dict):
            rejected.append({"where": where, "reason": "unknown element or no `picks` object"})
            continue
        letters = {side: {c["letter"] for c in entries} for side, entries in entry["sides"].items()}
        bad = [f"{side}={value!r}" for side, value in picks.items()
               if side not in letters or not (value is None or value in letters[side])]
        if bad or ident in clean:
            rejected.append({"where": where, "reason": f"invalid picks: {', '.join(bad) or 'element given twice'}"})
            continue
        clean[ident] = dict(picks)
    return clean, rejected


def _resolve(picks: dict, candidates: dict) -> dict:
    out = {}
    for ident, sides in picks.items():
        entry = candidates[ident]
        edges = {side: {**next(c for c in entry["sides"][side] if c["letter"] == letter), "letter": letter}
                 for side, letter in sides.items() if letter is not None}
        if edges:
            out[ident] = {"kind": entry["kind"], "floor_based": entry["floor_based"], "edges": edges}
    return out


def run(ctx: StageContext) -> Path:
    """edge_pick/answers.json and edge_pick/rejected.json; `--no-llm` writes the skip placeholder."""
    path = output_path(ctx, "edge_pick")
    path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = path.parent / "rejected.json"
    if ctx.no_llm:
        rejected_path.unlink(missing_ok=True)
        path.write_text(json.dumps({"skipped": True}), encoding="utf-8")
        return path
    if not load_posed_photos(ctx):
        raise NoPhotos(f"no photos for room {ctx.room} in {ctx.capture}")
    vision._command(ctx.llm_model)  # fail before the sheets are drawn when claude is missing
    pending = prepare(ctx)
    candidates = json.loads((pending / "candidates.json").read_text(encoding="utf-8"))
    meta, rejected = {"model": ctx.llm_model}, []
    if not candidates:
        document = {"schema": SCHEMA, "elements": {}, **meta, "duration_s": 0.0}
    else:
        text, error, seconds = vision.call_claude(pending, ctx.llm_model)
        meta["duration_s"] = seconds
        picks = {}
        if error:
            rejected = [{"where": "claude", "reason": error}]
        else:
            try:
                picks, rejected = validate(vision._parse_answer(text), candidates)
            except ValueError as exc:
                rejected = [{"where": "result", "reason": f"not a JSON answer: {exc}"}]
        document = {"error": error, **meta} if error else {"schema": SCHEMA, "elements": _resolve(picks, candidates), **meta}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    for reject in rejected:
        print(f"edge_pick: rejected {reject['where']}: {reject['reason']}", file=sys.stderr)
    return path


def usable(picks: object) -> bool:
    return isinstance(picks, dict) and isinstance(picks.get("elements"), dict)


def _on_plane(edges: dict, picked: dict) -> None:
    """Picked horizontal edges solved again within the depths their element's vertical edges allow (a leaf, a sash
    or a recess head is one plane parallel to the face, as in `anchor.measure`)."""
    vertical = [edges[s]["depths"] for s in ("left", "right") if s in edges and "depths" in edges[s]]
    if not vertical:
        return
    plane = (min(d[0] for d in vertical), max(d[1] for d in vertical))
    for side in ("bottom", "top"):
        obs = picked.get(side, {}).get("obs")
        fix = ortho.triangulate([tuple(o) for o in obs], plane) if obs else None
        if fix is not None:
            edges[side] = {**anchor.edge_entry(fix), "obs": obs}


def merge(anchors: dict, picks: dict, floor_sigma: float) -> dict:
    """The anchors with the picks applied: a picked edge replaces the mask's, and an element whose needed edges are
    all there becomes `anchored` (marked `picked`). A window takes picked edges only: its mask alone is not trusted."""
    out = copy.deepcopy(anchors)
    for ident, pick in picks.get("elements", {}).items():
        item = out.get("elements", {}).get(ident)
        if item is None or item.get("kind") != pick.get("kind") or not pick.get("edges"):
            continue
        edges = {} if item["kind"] == "window" else {s: e for s, e in (item.get("edges") or {}).items() if not e.get("rule")}
        edges.update({side: {k: v for k, v in e.items() if k not in ("letter", "origin", "shown")} for side, e in pick["edges"].items()})
        _on_plane(edges, pick["edges"])
        found = anchor.measures_from(edges, pick["floor_based"], floor_sigma)
        if found is not None:
            item.update(status="anchored", edges=found[0], measures=found[1], picked=sorted(pick["edges"]))
    return out
