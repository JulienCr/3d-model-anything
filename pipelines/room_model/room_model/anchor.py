"""Anchor stage: the edges of doors, windows, openings and niches measured in the posed photos.

Each host face is rectified from every good view, SAM segments the element (the leaf of a door, the clear opening
of a niche) prompted by the scan rectangle, each mask edge is snapped to a gradient peak and its depth behind the
face is solved across views (`ortho.triangulate`). Floor-based bottoms are the floor plane by rule.
"""
from __future__ import annotations

import json
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from room_model import ortho
from room_model.capture import colmajor, load_roomplan
from room_model.context import NoPhotos, StageContext, StageInputMissing, output_path, require
from room_model.model import SOURCE_RANK
from room_model.ortho import FaceFrame
from room_model.photo_geom import (FACING_COS, LOOSE_TOLERANCE, MIN_BAND_SEEN, DepthMaps, PosedPhoto, load_posed_photos,
                                   load_room_mesh, occluded_mask, project, rect_grid, room_boxes)

SCHEMA = "room-model/anchors-1"
KINDS = ("door", "window", "opening", "niche")
FLOOR_KINDS = ("door", "opening")
SIDES = ("left", "right", "bottom", "top")
MARGIN = 0.15  # m around the scan rectangle: a casing (up to 12 cm) plus RoomPlan's 3 cm sigma
SNAP = 0.015  # m from a mask edge to its gradient peak: SAM's boundary error, about ten photo pixels at 3 m
MIN_SCORE = 0.6  # SAM's own predicted IoU: below it SAM itself expects to miss or spill over 40 % of the region
# Share of the scan rectangle a leaf or a clear opening fills: the hole minus two casings (<= 15 cm) on an opening
# >= 0.5 m wide leaves at least 35 %; SAM may spill past the scan by its 3 cm sigma.
AREA = (0.35, 1.1)
CENTRAL = (0.3, 0.7)  # share of a side where the mask edge is read: clear of corners, mitres and handles
MIN_LINES = 10  # rows or columns (2 cm at 2 mm) a mask edge needs
MEETING_GAP = 0.03  # m between the two leaves of a double door: the meeting stiles and their rebate
SKIRTING_TOP = 0.15  # m; a niche starting below a skirting's height stands on the floor
CLOSED_SHARE = 0.5  # share of the doorway where the mesh lies within the wall slab (casings included) for a closed leaf
MAX_VIEWS = 6  # SAM and the profiles cost per view; 6 leave baseline after `ortho.triangulate` drops an outlier
CANDIDATES = 4  # candidate lines kept per side for a later pick
CANDIDATE_STEP = 0.005  # m between depth hypotheses: half a step of depth error shifts a line < 5 mm at 60 degrees
SAME_LINE = 0.005  # m; two candidate lines closer than this are one line (2.5 ortho pixels)
STRENGTH_TIE = 0.05  # gradient strengths within 5 % are the same line seen at depths the views cannot separate


def _px_box(o: ortho.Ortho, s0: float, s1: float, y0: float, y1: float) -> list[float]:
    return [float(o.col(s0)), float(o.row(y1)), float(o.col(s1)), float(o.row(y0))]


def _rect_pixels(o: ortho.Ortho, rect) -> int:
    s0, s1, y0, y1 = rect
    inside = ((o.ss >= s0) & (o.ss <= s1))[None, :] & ((o.ys >= y0) & (o.ys <= y1))[:, None]
    return int((inside & o.valid).sum())


def _best(candidates, o: ortho.Ortho, rect) -> np.ndarray | None:
    """Largest component of the best-scored mask filling AREA of the scan rectangle's visible part; None if none does."""
    ref = max(_rect_pixels(o, rect), 1)
    kept = []
    for mask, score in candidates:
        labels, n = ndimage.label(mask & o.valid)
        if score < MIN_SCORE or not n:
            continue
        main = labels == 1 + int(np.argmax(ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))))
        if AREA[0] <= main.sum() / ref <= AREA[1]:
            kept.append((score, main))
    return max(kept, key=lambda k: k[0])[1] if kept else None


def _s_extent(mask: np.ndarray, o: ortho.Ortho) -> tuple[float, float]:
    cols = np.flatnonzero(mask.any(axis=0))
    return float(o.ss[cols[0]]), float(o.ss[cols[-1]])


def _prompted(o: ortho.Ortho, rect, segment) -> np.ndarray | None:
    """Best mask of the box prompt and of the box plus its centre point: a box drawn on a casing's outer edge
    alone tends to return the casing ring, the centre point pulls SAM onto the leaf or the opening inside it."""
    s0, s1, y0, y1 = rect
    box = _px_box(o, s0, s1, y0, y1)
    centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return _best(segment(o.rgb, box) + segment(o.rgb, box, [centre]), o, rect)


def leaf_mask(o: ortho.Ortho, rect, segment) -> np.ndarray | None:
    """The element's region (`_prompted`), or the union of the two half-box masks when they are two distinct
    leaves meeting within MEETING_GAP and together wider than it (a double door, a pair of sashes)."""
    s0, s1, y0, y1 = rect
    mid = (s0 + s1) / 2
    whole = _prompted(o, rect, segment)
    halves = [_prompted(o, (a, b, y0, y1), segment) for a, b in ((s0, mid), (mid, s1))]
    if any(h is None for h in halves) or (halves[0] & halves[1]).sum() > 0.5 * min(h.sum() for h in halves):
        return whole
    (l0, l1), (r0, r1) = (_s_extent(h, o) for h in halves)
    union = halves[0] | halves[1]
    if r0 - l1 > MEETING_GAP or (whole is not None and (r1 - l0) - np.subtract(*_s_extent(whole, o)[::-1]) <= MEETING_GAP):
        return whole
    return union


def mask_edges(mask: np.ndarray, o: ortho.Ortho, rect) -> dict[str, float | None]:
    """Median mask extremes over the central rows (left, right) and columns (bottom, top); an extreme touching the
    edge of the photo is not an edge of the element and is left out."""
    s0, s1, y0, y1 = rect
    res = o.ss[1] - o.ss[0]
    rows = np.flatnonzero((o.ys > y0 + CENTRAL[0] * (y1 - y0)) & (o.ys < y0 + CENTRAL[1] * (y1 - y0)))
    cols = np.flatnonzero((o.ss > s0 + CENTRAL[0] * (s1 - s0)) & (o.ss < s0 + CENTRAL[1] * (s1 - s0)))
    found = {side: [] for side in SIDES}
    h, w = mask.shape
    for r in rows:
        c = np.flatnonzero(mask[r])
        if len(c):
            if c[0] > 0 and o.valid[r, max(c[0] - 4, 0)]:
                found["left"].append(o.ss[c[0]] - res / 2)
            if c[-1] < w - 1 and o.valid[r, min(c[-1] + 4, w - 1)]:
                found["right"].append(o.ss[c[-1]] + res / 2)
    for c in cols:
        r = np.flatnonzero(mask[:, c])
        if len(r):
            if r[0] > 0 and o.valid[max(r[0] - 4, 0), c]:
                found["top"].append(o.ys[r[0]] + res / 2)
            if r[-1] < h - 1 and o.valid[min(r[-1] + 4, h - 1), c]:
                found["bottom"].append(o.ys[r[-1]] - res / 2)
    return {side: float(np.median(v)) if len(v) >= MIN_LINES else None for side, v in found.items()}


def _bands(rect):
    s0, s1, y0, y1 = rect
    return {"s": (y0 + CENTRAL[0] * (y1 - y0), y0 + CENTRAL[1] * (y1 - y0)), "y": (s0 + CENTRAL[0] * (s1 - s0), s0 + CENTRAL[1] * (s1 - s0))}


def _axis(side: str) -> str:
    return "s" if side in ("left", "right") else "y"


def candidates(orthos: list[ortho.Ortho], cams: list[np.ndarray], rect, bounds) -> dict[str, list[dict]]:
    """Per side, the strongest lines within MARGIN of the scan edge, each with the depth where the views agree.

    Each view's gradient profile is read once at delta 0, then sheared to every depth hypothesis in `bounds` (a line
    at depth d shows at x + (cam - x) * -d / (cam_delta - d)) and fused by median: a recessed sash or leaf edge
    stands out at its own depth, where fusing at delta 0 would blur it away.
    """
    bands, edge, out = _bands(rect), dict(zip(SIDES, rect)), {}
    depths = np.arange(bounds[0], bounds[1] + CANDIDATE_STEP / 2, CANDIDATE_STEP)
    for side in SIDES:
        axis = _axis(side)
        profs = []
        for o, cam in zip(orthos, cams):
            x, p = ortho.profile(o, axis, bands[axis])
            scale = np.nanpercentile(p, 95) if np.isfinite(p).any() else np.nan
            if scale > 0:
                profs.append((p / scale, cam[0] if axis == "s" else cam[1], cam[2]))
        found = []
        for d in depths if profs else ():
            order = np.argsort(x)  # y profiles run top-down
            sheared = [np.interp(x + (c - x) * -d / (cd - d), x[order], p[order], left=np.nan, right=np.nan) for p, c, cd in profs]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
                fused = np.nanmedian(np.array(sheared), axis=0)
            found += [(pos, float(d), strength) for pos, strength in ortho.peaks(x, fused, edge[side] - MARGIN, edge[side] + MARGIN, CANDIDATES)]
        out[side] = [{"pos": round(p, 4), "delta": round(d, 4), "strength": round(st, 3)}
                     for p, d, st in sorted(_distinct_lines(found, [(c, cd) for _, c, cd in profs]))]
    return out


def _distinct_lines(found, cams) -> list[tuple[float, float, float]]:
    """The CANDIDATES strongest of (pos, depth, strength), one per line: two hypotheses are one line when they fall
    within SAME_LINE of each other in every view at delta 0. Among hypotheses of one line within STRENGTH_TIE of its
    best (no baseline across it), the depth nearest the face is kept, where the scan puts the element."""
    def footprint(pos, d):
        return np.array([pos + (c - pos) * -d / (cd - d) for c, cd in cams])

    groups = []
    for pos, d, strength in sorted(found, key=lambda t: -t[2]):
        group = next((g for g in groups if np.abs(footprint(pos, d) - g["at"]).max() <= SAME_LINE), None)
        if group is None:
            groups.append({"at": footprint(pos, d), "best": strength, "members": [(pos, d, strength)]})
        elif strength >= (1 - STRENGTH_TIE) * group["best"]:
            group["members"].append((pos, d, strength))
    return [min(g["members"], key=lambda m: abs(m[1])) for g in groups[:CANDIDATES]]


def _value(v: float, sigma: float) -> dict:
    return {"value": float(v), "sigma": float(sigma)}


def measure(kind: str, frame: FaceFrame, rect, views: list[tuple[PosedPhoto, np.ndarray]], segment, bounds,
            floor_sigma: float) -> dict:
    """Anchor one element seen in `views` [(photo, sensor RGB)]; rect = scan (s0, s1, y0, y1) on the face.

    `segment(rgb, box) -> [(mask, score)]`; `bounds` = (min, max) depth of an edge along the face normal. Status
    `anchored` (with `measures` and `edges`) or `candidates` (no usable mask, or an edge missing in every view).
    """
    s0, s1, y0, y1 = rect
    box = (s0 - MARGIN, s1 + MARGIN, y0 - MARGIN, y1 + MARGIN)
    floor_based = is_floor_based(kind, rect)
    bands = _bands(rect)
    obs, used, orthos, cams = {side: [] for side in SIDES}, [], [], []
    for photo, image in views:
        o = ortho.ortho(photo, image, frame, box)
        orthos.append(o)
        cams.append(ortho.camera_coords(frame, photo))
        mask = leaf_mask(o, rect, segment)
        if mask is None:
            continue
        used.append(photo.name)
        cam = ortho.camera_coords(frame, photo)
        for side, pos in mask_edges(mask, o, rect).items():
            if pos is None:
                continue
            axis = _axis(side)
            snapped = ortho.snap(o, axis, bands[axis], pos, SNAP)
            obs[side].append((pos if snapped is None else snapped, cam[0] if axis == "s" else cam[1], cam[2]))
    out = {"kind": kind, "views": used, "photos": [p.name for p, _ in views], "floor_based": floor_based,
           "candidates": candidates(orthos, cams, rect, bounds)}
    fix = {side: ortho.triangulate(obs[side], bounds) for side in ("left", "right")}
    # A leaf, a sash or a niche head is one plane parallel to the face: its horizontal edges take the depths its
    # vertical edges allow, as photos taken at one height give no baseline across a horizontal edge.
    plane = [f.depths for f in fix.values() if f] or [bounds]
    fix.update({side: ortho.triangulate(obs[side], (min(d[0] for d in plane), max(d[1] for d in plane))) for side in ("bottom", "top")})
    edges = {side: edge_entry(f) for side, f in fix.items() if f}
    found = measures_from(edges, floor_based, floor_sigma)
    if found is None:
        return {**out, "status": "candidates"}
    edges, measures = found
    if kind == "window":  # curtains and frosted glass fool the mask: a window waits for an edge pick
        return {**out, "status": "candidates", "proposal": {"edges": edges, "measures": measures}}
    return {**out, "status": "anchored", "edges": edges, "measures": measures}


def is_floor_based(kind: str, rect) -> bool:
    return kind in FLOOR_KINDS or (kind == "niche" and rect[2] < SKIRTING_TOP)


def edge_entry(fix: ortho.EdgeFix) -> dict:
    return {"pos": fix.pos, "delta": fix.delta, "sigma": fix.sigma, "views": fix.views, "depths": list(fix.depths),
            "rms": fix.rms}


def measures_from(edges: dict, floor_based: bool, floor_sigma: float) -> tuple[dict, dict] | None:
    """(edges with the floor rule applied, measures), or None when a needed edge is missing."""
    needed = ("left", "right", "top") if floor_based else SIDES
    if any(side not in edges for side in needed):
        return None
    edges = dict(edges)
    if floor_based:
        edges["bottom"] = {"pos": 0.0, "delta": edges["top"]["delta"], "sigma": floor_sigma, "views": 0, "rule": "floor"}
    left, right, bottom, top = (edges[side] for side in SIDES)
    return edges, {"center_s": _value((left["pos"] + right["pos"]) / 2, np.hypot(left["sigma"], right["sigma"]) / 2),
                   "width": _value(right["pos"] - left["pos"], np.hypot(left["sigma"], right["sigma"])),
                   "sill": _value(bottom["pos"], bottom["sigma"]),
                   "height": _value(top["pos"] - bottom["pos"], np.hypot(top["sigma"], bottom["sigma"]))}


# --- the stage -----------------------------------------------------------------------------------------------------

def face_frame(arch: dict, el: dict) -> FaceFrame:
    """Frame of the host face carrying an element: origin at the host start on the floor, normal into the room."""
    from room_model.architecture import _face_point, _value as param

    host = next(e for e in arch["elements"] if e["id"] == el["host"])
    a, b = (_face_point(arch, host, el.get("face"), s) for s in (0.0, 1.0))
    az = param(arch, host["params"]["azimuth"])
    normal = np.array([np.cos(az), 0.0, np.sin(az)]) * (-1.0 if el.get("face") == "B" else 1.0)
    return FaceFrame(np.array([a[0], param(arch, "floor.level"), a[1]]), np.array([b[0] - a[0], 0.0, b[1] - a[1]]), normal)


def scan_rect(arch: dict, el: dict) -> tuple[float, float, float, float]:
    p = {name: arch["parameters"][pid]["value"] for name, pid in el["params"].items()}
    return p["center_s"] - p["width"] / 2, p["center_s"] + p["width"] / 2, p["sill"], p["sill"] + p["height"]


def _corners(frame: FaceFrame, rect) -> np.ndarray:
    s0, s1, y0, y1 = rect
    return frame.point(np.array([s0, s1, s1, s0]), np.array([y0, y0, y1, y1]))


def _side_samples(frame: FaceFrame, rect, n: int = 5) -> np.ndarray:
    """(4, n, 3): points along the central part (CENTRAL) of each side, left, right, bottom, top, on the face."""
    s0, s1, y0, y1 = rect
    t = np.linspace(*CENTRAL, n)
    along_y, along_s = y0 + t * (y1 - y0), s0 + t * (s1 - s0)
    return np.stack([frame.point(np.full(n, s0), along_y), frame.point(np.full(n, s1), along_y),
                     frame.point(along_s, np.full(n, y0)), frame.point(along_s, np.full(n, y1))])


def select_views(photos, maps: DepthMaps, boxes, frame: FaceFrame, rect) -> list[PosedPhoto]:
    """Up to MAX_VIEWS photos facing the face (FACING_COS) that see at least two sides of the rectangle (MIN_BAND_SEEN
    of each side's central samples: depth test with LOOSE_TOLERANCE, furniture boxes), best first: sides seen x
    cosine x projected area. Edges are triangulated side by side, so a view showing only part of a door still counts."""
    corners = _corners(frame, rect)
    sides, centre = _side_samples(frame, rect), corners.mean(0)
    scored = []
    for photo in photos:
        to_cam = photo.centre - centre
        cos = float(frame.normal @ to_cam) / max(float(np.linalg.norm(to_cam)), 1e-9)
        if cos < FACING_COS:
            continue
        pts = sides.reshape(-1, 3)
        seen = (maps.visible(photo, pts, LOOSE_TOLERANCE) & ~occluded_mask(photo.centre, pts, boxes)).reshape(4, -1)
        count = int((seen.mean(axis=1) >= MIN_BAND_SEEN - 1e-9).sum())
        if count < 2:
            continue
        uv, _ = project(photo, corners)
        area = 0.5 * abs(float(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1))))
        scored.append((count * cos * area, photo))
    scored.sort(key=lambda t: -t[0])
    return [p for _, p in scored[:MAX_VIEWS]]


def leaf_closed(maps: DepthMaps, photo: PosedPhoto, frame: FaceFrame, rect, thickness: float) -> bool:
    """The mesh seen through the doorway lies within the wall slab, casings included, on CLOSED_SHARE of it: a closed
    leaf. Empty space behind, or a surface in front of the casings (the leaf swung into the room), is an open door."""
    pts = rect_grid(_corners(frame, rect))
    uv, depth = project(photo, pts)
    zbuf, scale = maps.get(photo)
    col, row = (uv[:, 0] * scale).astype(int), (uv[:, 1] * scale).astype(int)
    inside = (col >= 0) & (row >= 0) & (col < zbuf.shape[1]) & (row < zbuf.shape[0]) & (depth > 0)
    if not inside.any():
        return True
    surface = zbuf[row[inside], col[inside]]
    hits = photo.centre + (pts[inside] - photo.centre) * (surface / depth[inside])[:, None]
    with np.errstate(invalid="ignore"):
        delta = frame.coords(hits)[:, 2]
        closed = np.isfinite(delta) & (delta >= -(thickness + LOOSE_TOLERANCE)) & (delta <= LOOSE_TOLERANCE)
    return float(np.mean(closed)) >= CLOSED_SHARE


@lru_cache(maxsize=MAX_VIEWS)
def _image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _read(path: Path) -> dict | None:
    """The JSON object at `path`; None when absent or unreadable (architecture reports an unreadable file itself)."""
    from room_model import architecture

    try:
        return architecture._read(path, path.name) if path.exists() else None
    except StageInputMissing:
        return None


def capture_arch(ctx: StageContext) -> dict:
    """The architecture before any photo anchoring: model, openings, LLM answers (declared niches) and overrides."""
    from room_model import architecture, llm_merge

    model_doc = json.loads(require(ctx, "model").read_text(encoding="utf-8"))
    openings_doc = json.loads(require(ctx, "openings").read_text(encoding="utf-8"))
    vision = _read(ctx.out / architecture.VISION_PATH)
    return architecture.build(model_doc, openings_doc, vision if llm_merge.usable(vision) else None, None, ctx,
                              _read(ctx.out / architecture.OVERRIDES_PATH))


def build(ctx: StageContext, segment) -> dict:
    """The anchors document for the context's capture; `segment` as in `measure` (`segment.Segmenter` in the stage)."""
    t0 = time.monotonic()
    arch = capture_arch(ctx)
    photos = load_posed_photos(ctx)
    if not photos:
        raise NoPhotos(f"no photos for room {ctx.room} in {ctx.capture}")
    w2r = colmajor(arch["frame"]["world_to_room"], 4)
    maps, boxes = DepthMaps(*load_room_mesh(ctx, w2r)), room_boxes(load_roomplan(ctx.capture, ctx.room), w2r)
    floor_sigma = arch["parameters"]["floor.level"]["sigma"]
    elements = {}
    for el in arch["elements"]:
        if el["kind"] not in KINDS:
            continue
        if not el.get("host"):
            elements[el["id"]] = {"kind": el["kind"], "status": "no_host", "views": []}
            continue
        frame, rect = face_frame(arch, el), scan_rect(arch, el)
        host = next(e for e in arch["elements"] if e["id"] == el["host"])
        thickness = arch["parameters"][host["params"]["thickness"]]["value"]
        views = select_views(photos, maps, boxes, frame, rect)
        if not views:
            elements[el["id"]] = {"kind": el["kind"], "status": "no_view", "views": []}
        elif el["kind"] == "door" and not leaf_closed(maps, views[0], frame, rect, thickness):
            elements[el["id"]] = {"kind": el["kind"], "status": "open", "views": [p.name for p in views]}
        else:
            elements[el["id"]] = measure(el["kind"], frame, rect, [(p, _image(p.path)) for p in views], segment,
                                         (-thickness, LOOSE_TOLERANCE), floor_sigma)
    diagnostics = {"seconds": round(time.monotonic() - t0, 1)}
    if hasattr(segment, "diagnostics"):
        diagnostics.update(segment.diagnostics())
    return {"schema": SCHEMA, "elements": elements, "diagnostics": diagnostics}


def run(ctx: StageContext, segment=None) -> Path:
    if segment is None:
        from room_model.segment import Segmenter

        segment = Segmenter()
    doc = build(ctx, segment)
    path = output_path(ctx, "anchor")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


LEAF_KINDS = ("door", "window")  # the photo measures the leaf or sash, another quantity than the hole: own parameters
PAIRS = {("left", "right"): ("center_s", "width"), ("bottom", "top"): ("sill", "height")}  # decided together
MIN_EDGE_VIEWS = 2  # an edge seen in one view has no depth, only the whole depth bounds
MAX_RMS = 2 * ortho.LOCALISATION  # m; views agreeing within two peak localisations: a 95 % bound for that noise
# m; RoomPlan's own stated accuracy: a photo measure looser than the coarsest scan source is a depth ambiguity
# (two views without baseline), not a measurement.
PLAUSIBLE_SIGMA = 0.03
DISAGREE = 3.0  # combined sigmas beyond which photo and scan disagree: flagged to_verify, never settled silently
MEASURED = ("roomplan", "lidar")  # sources that measured the hole; `assumed` and `llm` (clicks) did not


def _edge_ok(edge: dict | None) -> bool:
    if edge is None:
        return False
    return edge.get("rule") == "floor" or (edge["views"] >= MIN_EDGE_VIEWS and edge.get("rms", np.inf) <= MAX_RMS
                                           and edge["sigma"] <= PLAUSIBLE_SIGMA)


def _param_doc(m: dict) -> dict:
    return {"value": float(m["value"]), "sigma": float(m["sigma"]), "unit": "m", "source": "photo"}


def _set_leaf(arch: dict, el: dict, names, item: dict) -> bool:
    """leaf_<name> parameters beside the hole's; True when the leaf is larger than its hole beyond DISAGREE sigmas."""
    prefix = el["params"]["width"].rsplit(".", 1)[0]
    wrong = False
    for name in names:
        m = item["measures"][name]
        pid = el["params"][f"leaf_{name}"] = f"{prefix}.leaf_{name}"
        arch["parameters"][pid] = _param_doc(m)
        if name in ("width", "height"):
            hole = arch["parameters"][el["params"][name]]
            wrong |= m["value"] - hole["value"] > DISAGREE * np.hypot(m["sigma"], hole["sigma"])
    return wrong


def _replace(arch: dict, el: dict, names, item: dict) -> bool:
    """Photo values in place of the hole's (scan value kept); True when they disagree with a measured scan source."""
    wrong = False
    for name in names:
        pid = el["params"].get(name)
        if pid is None or SOURCE_RANK[arch["parameters"][pid]["source"]] >= SOURCE_RANK["photo"]:
            continue
        old, m = arch["parameters"][pid], item["measures"][name]
        if old["source"] in MEASURED:
            wrong |= abs(m["value"] - old["value"]) > DISAGREE * np.hypot(m["sigma"], old["sigma"])
        new = _param_doc(m)
        if old["source"] != "llm" or "scan_value" in old:
            new.update(scan_value=old.get("scan_value", old["value"]), scan_sigma=old.get("scan_sigma", old["sigma"]))
        if old["source"] == "llm" or "llm_value" in old:
            new["llm_value"] = old.get("llm_value", old["value"])
        arch["parameters"][pid] = new
    return wrong


def apply(arch: dict, doc: dict) -> dict:
    """Anchored elements take their photo measures, side pair by side pair (PAIRS), when every edge of the pair passes
    `_edge_ok` (MIN_EDGE_VIEWS, MAX_RMS, PLAUSIBLE_SIGMA; a floor-ruled bottom always does).

    Doors and windows get leaf_* parameters and `leaf_edges` depths, the hole's width and height stay; niches and
    openings get their own parameters replaced (below tape) and `edges` depths. A disagreement beyond DISAGREE
    sigmas with a measured source, or a leaf larger than its hole, flags the element `to_verify`.
    """
    elements = {e["id"]: e for e in arch["elements"]}
    report = {"anchored": [], "candidates": [], "open": [], "no_view": [], "no_host": [], "unknown": [], "kept": [],
              "to_verify": []}
    for ident, item in sorted(doc.get("elements", {}).items()):
        el = elements.get(ident)
        if el is None or el["kind"] != item.get("kind"):
            report["unknown"].append(ident)
            continue
        report.setdefault(item.get("status"), []).append(ident)
        if item.get("status") != "anchored":
            continue
        leaf, applied, wrong = el["kind"] in LEAF_KINDS, set(), False
        for sides, names in PAIRS.items():
            if not all(_edge_ok(item["edges"].get(side)) for side in sides):
                report["kept"] += [f"{ident}.{name}" for name in names if name in el["params"]]
                continue
            wrong |= (_set_leaf if leaf else _replace)(arch, el, names, item)
            applied |= set(sides)
        el["leaf_edges" if leaf else "edges"] = {side: float(e["delta"]) for side, e in item["edges"].items() if side in applied}
        new_flags = (["photo_anchored"] if applied else []) + (["edge_picked"] if applied and item.get("picked") else [])
        new_flags += ["to_verify"] if wrong else []
        el["flags"] += [f for f in new_flags if f not in el["flags"]]
        if wrong:
            report["to_verify"].append(ident)
    return report
