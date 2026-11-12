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
MAX_LETTERS = 5  # per edge: more lines crowd a strip and a model tells them apart less reliably (4 edges <= 26 letters)
PICK_RESOLUTION = 0.01  # m; lines closer than this are one choice: below what a pick on a zoomed photo can separate
SAME_LINE = anchor.SAME_LINE
UNSEEN_SIGMA = 0.10  # m; a line no view could snap to keeps its position unconfirmed, far above the apply gates
SHEET_SIDE = vision.SHEET_SIDE
PANEL_ACROSS, PANEL_ALONG = 560, 420  # px of one view's panel across and along the edge
PAD = 0.03  # m shown beyond the outermost lines: the context a model needs to tell a leaf from its frame
LABEL_BAND = 70  # px outside the photo where the letters sit, joined to their line by a leader
LETTER_PX = 28  # px; letters a model reads without zooming (>= 24 px on the final image)
VIEWS_PER_IMAGE = 2
COVERED = 0.98  # share of the strip a view must show to be the main, most frontal one
COLOURS = {"left": (230, 40, 40), "right": (30, 120, 255), "bottom": (240, 190, 0), "top": (20, 190, 70)}
QUESTIONS = {
    "door": {"left": "the outer edge of the left leaf (vantail): the moving panel itself, not the frame (huisserie) nor the casing (chambranle)",
             "right": "the outer edge of the right leaf (vantail): the moving panel itself, not the frame nor the casing",
             "top": "the top edge of the leaf (vantail), not the frame head nor the casing"},
    "window": {"left": "the outer edge of the left opening sash (ouvrant), not the fixed frame (dormant), the casing or a curtain",
               "right": "the outer edge of the right opening sash (ouvrant), not the fixed frame, the casing or a curtain",
               "bottom": "the bottom edge of the opening sash (ouvrant)", "top": "the top edge of the opening sash (ouvrant)"},
    "niche": {"left": "the inner face of the left jamb (tableau) of the niche, where the recess starts, not a frame around it",
              "right": "the inner face of the right jamb (tableau) of the niche, where the recess starts",
              "bottom": "the top of the niche sill (appui)", "top": "the underside of the niche head (linteau)"},
    "opening": {"left": "the arris where the left jamb (tableau) of the passage meets the wall face on this side (the near corner), not a casing nor the far side",
                "right": "the arris where the right jamb (tableau) meets the wall face on this side (the near corner), not a casing nor the far side",
                "top": "the arris where the lintel (linteau) meets the wall face on this side",
                "bottom": "the floor of the passage"},
}


def _known_edges(item: dict) -> dict:
    return item.get("edges") or (item.get("proposal") or {}).get("edges") or {}


def _footprint(pos: float, delta: float, cams) -> np.ndarray:
    """Where a line at (pos, delta) shows on the delta = 0 ortho of each camera (pos, delta)."""
    return np.array([pos + (c - pos) * -delta / (cd - delta) for c, cd in cams])


def _distinct(lines: list[dict], cams) -> list[dict]:
    """One entry per physical line, origins merged: two lines are one when their footprints agree within SAME_LINE in
    every view, or when they lie within PICK_RESOLUTION with overlapping depth ranges (one line found at several
    depth hypotheses). At most MAX_LETTERS, mask and scan lines kept first."""
    def same(a, b):
        if np.abs(a["at"] - b["at"]).max() <= SAME_LINE:
            return True
        (a0, a1), (b0, b1) = a["depths"], b["depths"]
        return abs(a["pos"] - b["pos"]) < PICK_RESOLUTION and a0 <= b1 + SAME_LINE and b0 <= a1 + SAME_LINE

    out = []
    for line in lines:
        line = {**line, "at": _footprint(line["pos"], line["delta"], cams)}
        twin = next((o for o in out if same(o, line)), None)
        if twin is None:
            out.append(line)
        else:
            twin["origin"] += [o for o in line["origin"] if o not in twin["origin"]]
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


def _distance(obs, fix: ortho.EdgeFix) -> float:
    return float(np.mean([o[2] for o in obs])) - fix.delta


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
                "depths": [line["delta"], line["delta"]], "distance": 0.0, "obs": []}
    return {**anchor.edge_entry(fix, _distance(obs, fix)), "obs": [list(o) for o in obs]}


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
    return {"kind": el["kind"], "floor_based": floor_based, "sides": out,
            "images": {side: f"{el['id']}-{side}.jpg" for side in out}, "_draw": (views, frame, rect, reference)}


# --- sheets --------------------------------------------------------------------------------------------------------

def drawn_at(c: dict, cam: np.ndarray, axis: str, reference: float) -> float:
    """Where the line (pos, delta) shows on a photo rectified at depth `reference`, for a camera at `cam` (s, y, delta)."""
    a = cam[0] if axis == "s" else cam[1]
    return c["pos"] + (a - c["pos"]) * (reference - c["delta"]) / (cam[2] - c["delta"])


def _strip_box(views, frame, side: str, entries: list[dict], model: float, rect, reference: float):
    """(box, res) of the strip: from the first to the last line drawn in any view, plus PAD, PANEL_ACROSS px across."""
    axis = anchor._axis(side)
    lines = entries + [{"pos": model, "delta": 0.0}]
    drawn = [drawn_at(c, ortho.camera_coords(frame, p), axis, reference) for p, _ in views for c in lines]
    lo, hi = min(drawn) - PAD, max(drawn) + PAD
    res = (hi - lo) / PANEL_ACROSS
    along = anchor._bands(rect)["s" if axis == "s" else "y"]
    mid, half = (along[0] + along[1]) / 2, PANEL_ALONG * res / 2
    return ((lo, hi, mid - half, mid + half) if axis == "s" else (mid - half, mid + half, lo, hi)), res


def panel(photo, image, frame, side: str, entries: list[dict], model: float, box, res: float, reference: float):
    """(panel image, ortho, {letter: drawn position}): the strip rendered from the photo at `res` (no upscaling),
    each line drawn at its own footprint in this view, the scan edge dashed, letters in a band outside the photo."""
    vertical = anchor._axis(side) == "s"
    o = ortho.ortho(photo, image, frame, box, reference, res)
    cam = ortho.camera_coords(frame, photo)
    rgb = Image.fromarray(o.rgb)
    w, h = rgb.size
    canvas = Image.new("RGB", (w, h + LABEL_BAND) if vertical else (w + LABEL_BAND, h), (25, 25, 30))
    canvas.paste(rgb, (0, LABEL_BAND) if vertical else (LABEL_BAND, 0))
    draw, font = ImageDraw.Draw(canvas), vision._font(LETTER_PX)
    px = lambda v: float(o.col(v)) + 0.5 if vertical else float(o.row(v)) + 0.5  # noqa: E731
    m = px(drawn_at({"pos": model, "delta": 0.0}, cam, "s" if vertical else "y", reference))
    for k in range(0, int(h if vertical else w), 16):  # the scan edge, dashed, as context
        seg = [(m, LABEL_BAND + k), (m, LABEL_BAND + k + 8)] if vertical else [(LABEL_BAND + k, m), (LABEL_BAND + k + 8, m)]
        draw.line(seg, fill=(255, 255, 255), width=2)
    drawn = {c["letter"]: drawn_at(c, cam, "s" if vertical else "y", reference) for c in entries}
    order = sorted(entries, key=lambda c: drawn[c["letter"]])
    for n, c in enumerate(order):
        x = px(drawn[c["letter"]])
        slot = (n + 0.5) * (w if vertical else h) / len(order)  # letters spread evenly, in line order
        if vertical:
            draw.line([(x, LABEL_BAND), (x, LABEL_BAND + h)], fill=COLOURS[side], width=2)
            draw.line([(slot, LABEL_BAND - 26), (x, LABEL_BAND)], fill=COLOURS[side], width=1)
            draw.text((slot, LABEL_BAND / 2 - 8), c["letter"], font=font, fill=COLOURS[side], anchor="mm")
        else:
            draw.line([(LABEL_BAND, x), (LABEL_BAND + w, x)], fill=COLOURS[side], width=2)
            draw.line([(LABEL_BAND - 26, slot), (LABEL_BAND, x)], fill=COLOURS[side], width=1)
            draw.text((LABEL_BAND / 2 - 8, slot), c["letter"], font=font, fill=COLOURS[side], anchor="mm")
    return canvas, o, drawn


def _edge_image(path: Path, ident: str, side: str, entry: dict) -> None:
    """One image per edge: the most frontal view covering the whole strip, then the view covering most of it."""
    views, frame, rect, reference = entry["_draw"]
    model = dict(zip(SIDES, rect))[side]
    entries = entry["sides"][side]
    box, res = _strip_box(views, frame, side, entries, model, rect, reference)
    panels = []
    for p, image in views:
        canvas, o, _ = panel(p, image, frame, side, entries, model, box, res, reference)
        cam = ortho.camera_coords(frame, p)
        offset = np.array([cam[0] - np.mean(box[:2]), cam[1] - np.mean(box[2:]), cam[2]])
        panels.append((canvas, float(o.valid.mean()), float(cam[2] / np.linalg.norm(offset))))  # cosine: how frontal
    covering = [t for t in panels if t[1] >= COVERED] or panels
    first = max(covering, key=lambda t: (t[2], t[1]))
    rest = sorted((t for t in panels if t is not first and t[1] > 0), key=lambda t: -t[1])
    best = [first[0]] + [t[0] for t in rest[:VIEWS_PER_IMAGE - 1]]
    width = sum(pn.width for pn in best) + vision.GAP * (len(best) + 1)
    sheet = Image.new("RGB", (width, vision.HEADER + max(pn.height for pn in best)), (40, 40, 48))
    ImageDraw.Draw(sheet).text((10, 8), f"{ident}: {side} edge", font=vision._font(26), fill=(255, 255, 255))
    x = vision.GAP
    for pn in best:
        sheet.paste(pn, (x, vision.HEADER))
        x += pn.width + vision.GAP
    if max(sheet.size) > SHEET_SIDE:  # never reached at PANEL_ACROSS x VIEWS_PER_IMAGE: letters stay LETTER_PX
        f = SHEET_SIDE / max(sheet.size)
        sheet = sheet.resize((int(sheet.width * f), int(sheet.height * f)), Image.BILINEAR)
    sheet.save(path, quality=vision.JPEG_QUALITY)


def _prompt(candidates: dict) -> str:
    lines = ["# Picking the edges of doors, windows, niches and openings", "",
             "Each JPEG in this folder shows ONE edge of one element, to open with the Read tool. The photo is "
             "straightened onto the wall (vertical lines are vertical) and zoomed on the edge; two panels are the same "
             "edge seen from two photos. Coloured lines are candidate positions, each joined by a thin leader to its "
             "letter outside the photo. The dashed white line is where the scan put the edge: context only, not a choice.",
             "", "For every image below, answer the letter of the line closest to the edge described, within about 1 cm "
             "(a few pixels). Answer null only when the edge is hidden or no line is near it.", "",
             "What to look for: a door leaf (vantail) is the flat panel that moves, usually with mouldings and the handle; "
             "around it the frame (huisserie) and the casing (chambranle) are narrow strips, often lighter or in relief, "
             "standing proud of the wall. A window sash (ouvrant) holds the glass; when a curtain hides its outer edge, the "
             "visible outline of the glazing bead next to it is acceptable. A jamb (tableau) is the side face of the hole "
             "in the wall, usually plain plaster.", "", "## Images", ""]
    for ident, entry in candidates.items():
        for side, entries in entry["sides"].items():
            letters = ", ".join(f"`{c['letter']}`" for c in entries)
            lines.append(f"- `{entry['images'][side]}` ({ident}, {side}): {QUESTIONS[entry['kind']][side]}. Letters: {letters}.")
    example = {"elements": [{"id": "Door_1", "picks": {"left": "B", "right": None, "top": "H"}}]}
    lines += ["", "## Answer", "", "Only one JSON object, one entry per element, `picks` keyed by edge (left, right, "
              "bottom, top), for example:", "", json.dumps(example)]
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
        for side in entry["sides"]:
            _edge_image(pending / entry["images"][side], ident, side, entry)
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
        ident = ident if isinstance(ident, str) else None
        where = ident or f"elements[{i}]"
        entry, picks = candidates.get(ident) if ident else None, item.get("picks") if isinstance(item, dict) else None
        if entry is None or not isinstance(picks, dict):
            rejected.append({"where": where, "reason": "unknown element or no `picks` object"})
            continue
        letters = {side: {c["letter"] for c in entries} for side, entries in entry["sides"].items()}
        given = {side: value for side, value in picks.items() if not (value is None and side not in letters)}  # null on an unasked edge: nothing said
        bad = [f"{side}={value!r}" for side, value in given.items()
               if not isinstance(side, str) or side not in letters or not (value is None or (isinstance(value, str) and value in letters[side]))]
        if bad or ident in clean:
            rejected.append({"where": where, "reason": f"invalid picks: {', '.join(bad) or 'element given twice'}"})
            continue
        clean[ident] = given
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
            edges[side] = {**anchor.edge_entry(fix, _distance(obs, fix)), "obs": obs}


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
