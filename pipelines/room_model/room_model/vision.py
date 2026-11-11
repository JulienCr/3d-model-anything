"""Vision stage: photo sheets for `claude -p`, and the validation of its answer.

`prepare` writes vision/pending/ (sheets, elements.json, crops.json, prompt.md, schema.json); `run_llm` runs the
model there and validates what comes back item by item; `run` writes vision/answers.json and vision/rejected.json.
crops.json maps a crop id to its photo, its box in upright pixels, `up_turns` and its size: a crop pixel (u, v) is
the upright pixel box[:2] + (u, v) * box_size / size, then `photo_geom.to_sensor`. An element crop also lists
`visible_corners`, in the answer order (top-left, top-right, bottom-right, bottom-left): for the pipeline, not the model.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from room_model import architecture
from room_model.architecture import _face_point, dimension_endpoints
from room_model.capture import colmajor, load_photos, load_roomplan
from room_model.context import ClaudeMissing, NoPhotos, StageContext, output_path, require
from room_model.photo_geom import DEPTH_TOLERANCE, LOOSE_TOLERANCE, MIN_CORNERS, MIN_DEPTH, DepthMaps, PosedPhoto, best_views, load_posed_photos, load_room_mesh, project, rect_views, room_boxes, to_upright, upright, upright_size

PACKAGE = Path(__file__).resolve().parent
CATALOGUE_FILE = PACKAGE.parent / "blender" / "arch" / "catalogue.py"
MAX_SHEETS = 12
SHEET_SIDE = 1568  # px; larger images are shrunk before the model sees them
GAP, HEADER, CELL_MAX_H = 12, 44, 880
VIEWS_PER_ELEMENT = 3
OVERVIEW_PER_SHEET = (4, 6, 9)
TYPED_KINDS = ("door", "window", "opening", "niche")
HOST_KINDS = ("wall", "partition")
PAD_RATIO, MIN_PAD = 0.35, 0.04  # crop margin: share of the framed extent, and of the photo's long side
MIN_CORNER_AREA = 25.0  # px^2 of a quadrilateral that can be a real answer
LLM_TIMEOUT_S = 900
JPEG_QUALITY = 85
YELLOW, ORANGE = (255, 215, 0), (255, 140, 0)


@functools.cache
def catalogue():
    """The Blender catalogue module, loaded from its file: it is pure Python and must stay the single source."""
    spec = importlib.util.spec_from_file_location("room_model_catalogue", CATALOGUE_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except (TypeError, OSError):
        return ImageFont.load_default()


# --- geometry of what is shown ---------------------------------------------------------------------------------

def _value(arch: dict, pid: str) -> float:
    return arch["parameters"][pid]["value"]


def _opening_rect(arch: dict, el: dict) -> np.ndarray | None:
    """Room-frame corners (4, 3) of an opening's scan rectangle, from the spans of its width and height dimensions."""
    wide, tall = (dimension_endpoints(arch, f"{el['id']}-{k}") for k in ("w", "h"))
    if wide is None or tall is None:
        return None
    (x0, _, z0), (x1, _, z1) = wide
    low, high = sorted((tall[0][1], tall[1][1]))
    return np.array([[x0, low, z0], [x1, low, z1], [x1, high, z1], [x0, high, z0]])


def _face_rect(arch: dict, host: dict, face: str, s: tuple[float, float], y: tuple[float, float]) -> np.ndarray:
    a, b = (_face_point(arch, host, face, v) for v in s)
    return np.array([[a[0], y[0], a[1]], [b[0], y[0], b[1]], [b[0], y[1], b[1]], [a[0], y[1], a[1]]])


def _face_normal(arch: dict, host: dict, face: str) -> np.ndarray:
    az = _value(arch, host["params"]["azimuth"])
    n = np.array([np.cos(az), 0.0, np.sin(az)])
    return -n if face == "B" else n


def _frame_box(photo: PosedPhoto, pts: np.ndarray) -> tuple[int, int, int, int] | None:
    """Box in upright pixels around the projected points that are in front of the camera, with a margin."""
    uv, depth = project(photo, pts)
    uv = to_upright(photo, uv[depth > MIN_DEPTH])
    if not len(uv):
        return None
    width, height = upright_size(photo)
    lo, hi = uv.min(0), uv.max(0)
    half = (hi - lo) / 2 + max(PAD_RATIO * float((hi - lo).max()), MIN_PAD * max(width, height))
    b0 = np.floor(np.maximum((lo + hi) / 2 - half, 0)).astype(int)
    b1 = np.ceil(np.minimum((lo + hi) / 2 + half, [width, height])).astype(int)
    b1 = np.maximum(b1, np.minimum(b0 + 16, [width, height]))
    b0 = np.maximum(np.minimum(b0, b1 - 16), 0)
    return (int(b0[0]), int(b0[1]), int(b1[0]), int(b1[1]))


@dataclass
class _Scene:
    arch: dict
    photos: list[PosedPhoto]
    maps: DepthMaps
    boxes: list

    def views(self, pts: np.ndarray, normal: np.ndarray | None, k: int, photos: list[PosedPhoto] | None = None):
        """Up to k photos: those seeing all the points (3 cm, then 6 cm), then those with all points in the image, then the centre."""
        found: list[PosedPhoto] = []
        centre = pts.mean(0, keepdims=True)
        for tolerance, sel in ((DEPTH_TOLERANCE, pts), (LOOSE_TOLERANCE, pts), (math.inf, pts), (LOOSE_TOLERANCE, centre)):
            names = {p.name for p in found}
            pool = [p for p in (self.photos if photos is None else photos) if p.name not in names]
            found += best_views(pool, sel, self.maps, k - len(found), normal, tolerance, self.boxes)
            if len(found) >= k:
                break
        return found


@dataclass
class _Crop:
    id: str
    kind: str  # "element" or "overview"
    target: str
    photo: PosedPhoto
    box: tuple[int, int, int, int]
    overlays: list[tuple[np.ndarray, tuple[int, int, int], str | None]] = field(default_factory=list)
    fallback: str | None = None  # "overview" for an element shown on a wall overview because no photo qualified
    corners: tuple[bool, ...] = ()  # which corners of the element rectangle are visible on this crop


def _element_crops(scene: _Scene, el: dict, rect: np.ndarray) -> list[_Crop]:
    """Up to three eligible views of the element, together covering as many corners as possible.

    No eligible view: overviews of its host face from photos that pass every rule but the size (the element is
    small in them, never grazing or hidden). A door, window, opening or through niche is a hole: it is seen from
    either side of its host.
    """
    host = next((e for e in scene.arch["elements"] if e["id"] == el["host"]), None)
    face = el.get("face") or "A"
    partition = host is not None and host["kind"] == "partition"
    normal = None if host is None else _face_normal(scene.arch, host, face)
    through = el["kind"] != "niche" or bool(el.get("through"))
    args = (scene.photos, rect, scene.maps, normal, through, scene.boxes)
    crops = []
    for k, view in enumerate(rect_views(*args, VIEWS_PER_ELEMENT)):
        box = _frame_box(view.photo, rect)
        if box:
            crops.append(_Crop(f"{el['id']}-{'abc'[k]}", "element", el["id"], view.photo, box, [(rect, YELLOW, None)], corners=view.corners))
    if crops or host is None:
        return crops
    for k, view in enumerate(rect_views(*args, VIEWS_PER_ELEMENT, min_side=0)):
        seen_from = face
        if partition and _face_normal(scene.arch, host, face) @ (view.photo.centre - rect.mean(0)) < 0:
            seen_from = "B" if face == "A" else "A"
        box = _frame_box(view.photo, np.vstack([_whole_face(scene.arch, host, seen_from), rect]))
        if box:
            crops.append(_Crop(f"{el['id']}-{'abc'[k]}", "element", el["id"], view.photo, box, [(rect, YELLOW, None)], "overview", view.corners))
    return crops


def _whole_face(arch: dict, host: dict, face: str) -> np.ndarray:
    """The rectangle of the overview crop of a host face: nearly its whole length, 10 cm clear of floor and top."""
    floor, top = _value(arch, "floor.level"), _value(arch, host["params"]["top"])
    length = float(np.linalg.norm(np.subtract(host["end"], host["start"])))
    return _face_rect(arch, host, face, (0.04 * length, 0.96 * length), (floor + 0.1, max(top - 0.1, floor + 0.2)))


def _face_crop(scene: _Scene, host: dict, face: str, others: dict) -> _Crop | None:
    arch = scene.arch
    floor, top = _value(arch, "floor.level"), _value(arch, host["params"]["top"])
    length = float(np.linalg.norm(np.subtract(host["end"], host["start"])))
    normal = _face_normal(arch, host, face)
    for lo, hi, inset in ((0.04, 0.96, 0.1), (0.25, 0.75, 0.3), (0.45, 0.55, 0.6)):
        rect = _face_rect(arch, host, face, (lo * length, hi * length), (floor + inset, max(top - inset, floor + inset + 0.1)))
        facing = [p for p in scene.photos if float(normal @ (p.centre - rect.mean(0))) > 0.1]
        for photo in scene.views(rect, normal, 1, facing):
            box = _frame_box(photo, rect)
            if box:
                overlays = [(r, ORANGE, i) for i, r in others.get(host["id"], [])]
                suffix = f"-{face}" if host["kind"] == "partition" else ""
                return _Crop(f"{host['id']}{suffix}-ov", "overview", host["id"], photo, box, overlays)
    return None


# --- sheets --------------------------------------------------------------------------------------------------------

def _ruler(image: Image.Image) -> Image.Image:
    """Ticks every 50 px, numbers every 100 px on the top and left edges, a light grid every 100 px."""
    w, h = image.size
    over = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw, font = ImageDraw.Draw(over), _font(13)
    for pos in range(0, max(w, h), 50):
        major, length = pos % 100 == 0, 14 if pos % 100 == 0 else 7
        for axis, limit in ((0, w), (1, h)):
            if pos >= limit:
                continue
            line = lambda a, b: [(pos, a), (pos, b)] if axis == 0 else [(a, pos), (b, pos)]  # noqa: E731
            if major:
                draw.line(line(0, h if axis == 0 else w), fill=(0, 255, 255, 60))
            draw.line(line(0, length), fill=(255, 255, 255, 255), width=2)
            if major and pos:
                at = (pos + 3, 14) if axis == 0 else (16, pos + 1)
                draw.text(at, str(pos), font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(image.convert("RGBA"), over).convert("RGB")


def _render(crop: _Crop, cell: tuple[int, int]) -> Image.Image:
    x0, y0, x1, y1 = crop.box
    scale = min(1.0, cell[0] / (x1 - x0), cell[1] / (y1 - y0))
    size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
    image = upright(crop.photo).crop(crop.box).resize(size)
    draw, font = ImageDraw.Draw(image), _font(14)
    sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)
    for pts, colour, label in crop.overlays:
        uv, depth = project(crop.photo, pts)
        if (depth <= MIN_DEPTH).any():
            continue
        uv = (to_upright(crop.photo, uv) - [x0, y0]) * [sx, sy]
        poly = [tuple(p) for p in uv]
        draw.line(poly + poly[:1], fill=(0, 0, 0), width=5)
        draw.line(poly + poly[:1], fill=colour, width=3)
        if label:
            draw.text((uv[3][0] + 4, uv[3][1] - 18), label, font=font, fill=colour, stroke_width=2, stroke_fill=(0, 0, 0))
    return _ruler(image)


def _layout(count: int, max_cols: int) -> tuple[int, int, tuple[int, int]]:
    cols = min(count, max_cols)
    rows = math.ceil(count / cols)
    return cols, rows, ((SHEET_SIDE - GAP * (cols - 1)) // cols, min(CELL_MAX_H, (SHEET_SIDE - rows * (HEADER + GAP)) // rows))


def _sheet(path: Path, crops: list[_Crop], max_cols: int) -> dict[str, dict]:
    """Writes a sheet of panels (title band with the crop id, then the upright crop); returns each crop's size and origin."""
    cols, _, cell = _layout(len(crops), max_cols)
    title, small = _font(30), _font(16)
    placed, x, y, row_h, panels = {}, 0, 0, 0, []
    for crop in crops:
        image = _render(crop, cell)
        text_w = ImageDraw.Draw(image).textlength
        band = max(image.width, int(text_w(crop.id, font=title) + text_w(f"{image.width}x{image.height}px", font=small)) + 36)
        panels.append((crop, image, band))
    for k, (crop, image, band) in enumerate(panels):
        if k and k % cols == 0:
            x, y, row_h = 0, y + row_h + GAP, 0
        placed[crop.id] = {"size": [image.width, image.height], "sheet_origin": [x, y + HEADER], "sheet": path.name, "slot": (x, y, band)}
        x, row_h = x + band + GAP, max(row_h, HEADER + image.height)
    width = max(p["slot"][0] + p["slot"][2] for p in placed.values())
    sheet = Image.new("RGB", (width, y + row_h), (28, 28, 30))
    draw = ImageDraw.Draw(sheet)
    for crop, image, band in panels:
        sx, sy, _ = placed[crop.id]["slot"]
        draw.rectangle([sx, sy, sx + band, sy + HEADER - 1], fill=(48, 48, 56))
        draw.text((sx + 10, sy + 4), crop.id, font=title, fill=(255, 255, 255))
        draw.text((sx + band - 4, sy + HEADER - 3), f"{image.width}x{image.height}px", font=small, fill=(190, 190, 200), anchor="rd")
        sheet.paste(image, (sx, sy + HEADER))
    sheet.save(path, "JPEG", quality=JPEG_QUALITY)
    return {cid: {k: v for k, v in info.items() if k != "slot"} for cid, info in placed.items()}


def _plan_overview(n_elements: int, n_faces: int) -> tuple[int, int]:
    """(crops per overview sheet, element sheets kept) so that everything fits in MAX_SHEETS."""
    for per in OVERVIEW_PER_SHEET:
        if n_elements + math.ceil(n_faces / per) <= MAX_SHEETS:
            return per, n_elements
    per = OVERVIEW_PER_SHEET[-1]
    return per, MAX_SHEETS - math.ceil(n_faces / per)


# --- prepare ---------------------------------------------------------------------------------------------------------

def _architecture(ctx: StageContext) -> dict:
    """architecture.json when it exists and holds scan values only, else built from model.json and openings.json.

    A document that merged LLM answers or tape measurements is not shown back to the model as the scan.
    """
    path = output_path(ctx, "architecture")
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not {"vision", "tape"} & set(doc.get("diagnostics", {})):
            return doc
    model_doc = json.loads(require(ctx, "model").read_text(encoding="utf-8"))
    return architecture.build(model_doc, json.loads(require(ctx, "openings").read_text(encoding="utf-8")))


def _scan_summary(arch: dict, el: dict) -> dict:
    keys = [k for k in ("width", "height", "sill") if k in el["params"]]
    return {"scan_cm": {k: round(_value(arch, el["params"][k]) * 100, 1) for k in keys},
            "sources": {k: arch["parameters"][el["params"][k]]["source"] for k in keys}}


def _elements_doc(arch: dict, typed: list[dict], crops: dict[str, list[_Crop]], shown: set[str], faces: list, sheets: list[dict], info: dict) -> dict:
    """elements.json: the scan elements with their crops, the wall faces with their overview crop, the sheets."""
    sheet_of = {cid: s["file"] for s in sheets for cid in s["crops"]}
    ceilings = [_value(arch, e["params"]["level"]) - _value(arch, "floor.level") for e in arch["elements"] if e["kind"] == "ceiling"]
    return {
        "room": {"ceiling_height_cm": sorted({round(c * 100) for c in ceilings}), "hosts": [h["id"] for h in arch["elements"] if h["kind"] in HOST_KINDS]},
        "elements": [{"id": e["id"], "kind": e["kind"], "host": e.get("host"), "face": e.get("face"), "flags": e.get("flags", []), **_scan_summary(arch, e),
                      "sheet": f"{e['id']}.jpg" if e["id"] in shown else None,
                      **({"fallback": "overview"} if any(c.fallback for c in crops[e["id"]]) else {}),
                      "crops": [{"id": c.id, "size": info[c.id]["size"]} for c in crops[e["id"]] if c.id in info]} for e in typed],
        "faces": [{"host": h["id"], "face": f, "kind": h["kind"], "length_cm": round(float(np.linalg.norm(np.subtract(h["end"], h["start"]))) * 100),
                   "crop": c.id if c and c.id in info else None, "sheet": sheet_of.get(c.id) if c else None} for h, f, c in faces],
        "sheets": sheets,
        "skipped": [e["id"] for e in typed if crops[e["id"]] and e["id"] not in shown],
    }


def prepare(ctx: StageContext) -> Path:
    """Writes vision/pending/ for the context's capture and returns it; NoPhotos when the room has no photos."""
    arch = _architecture(ctx)
    w2r = colmajor(arch["frame"]["world_to_room"], 4)
    photos = load_posed_photos(ctx, w2r)
    if not photos:
        raise NoPhotos(f"no photos for room {ctx.room} in {ctx.capture}")
    scene = _Scene(arch, photos, DepthMaps(*load_room_mesh(ctx, w2r)), room_boxes(load_roomplan(ctx.capture, ctx.room), w2r))
    pending = ctx.out / "vision" / "pending"
    shutil.rmtree(pending, ignore_errors=True)
    pending.mkdir(parents=True)

    typed = [e for e in arch["elements"] if e["kind"] in TYPED_KINDS]
    rects = {e["id"]: _opening_rect(arch, e) for e in typed}
    others: dict[str, list] = {}
    for e in typed:
        if rects[e["id"]] is not None and e.get("host"):
            others.setdefault(e["host"], []).append((e["id"], rects[e["id"]]))
    crops = {e["id"]: [] if rects[e["id"]] is None else _element_crops(scene, e, rects[e["id"]]) for e in typed}
    faces = [(h, f, _face_crop(scene, h, f, others)) for h in arch["elements"] if h["kind"] in HOST_KINDS
             for f in (("A", "B") if h["kind"] == "partition" else ("A",))]

    covered = [c for _, _, c in faces if c]
    per_sheet, kept = _plan_overview(sum(bool(crops[e["id"]]) for e in typed), len(covered))
    shown = [e for e in typed if crops[e["id"]]][:kept]
    info: dict[str, dict] = {}
    sheets = []
    for e in shown:
        info.update(_sheet(pending / f"{e['id']}.jpg", crops[e["id"]], VIEWS_PER_ELEMENT))
        sheets.append({"file": f"{e['id']}.jpg", "crops": [c.id for c in crops[e["id"]]]})
    for k in range(math.ceil(len(covered) / per_sheet)):
        group = covered[k * per_sheet:(k + 1) * per_sheet]
        info.update(_sheet(pending / f"walls_{k + 1}.jpg", group, 2 if per_sheet == 4 else 3))
        sheets.append({"file": f"walls_{k + 1}.jpg", "crops": [c.id for c in group]})
    for c in [c for cs in crops.values() for c in cs] + covered:
        if c.id in info:
            info[c.id].update(photo=c.photo.name, box=list(c.box), up_turns=c.photo.up_turns, kind=c.kind, target=c.target,
                              **({"fallback": c.fallback} if c.fallback else {}), **({"visible_corners": list(reversed(c.corners))} if c.corners else {}))
    elements = _elements_doc(arch, typed, crops, {e["id"] for e in shown}, faces, sheets, info)
    (pending / "crops.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (pending / "elements.json").write_text(json.dumps(elements, indent=1), encoding="utf-8")
    (pending / "schema.json").write_text((PACKAGE / "vision_schema.json").read_text(encoding="utf-8"), encoding="utf-8")
    (pending / "prompt.md").write_text(_prompt(sheets), encoding="utf-8")
    return pending


def _prompt(sheets: list[dict]) -> str:
    schema = json.loads((PACKAGE / "vision_schema.json").read_text(encoding="utf-8"))
    props = schema["properties"]
    profiles = lambda node: " | ".join(f'"{p}"' for p in node["properties"]["profile"]["enum"])  # noqa: E731
    fields = {
        "sheets": "\n".join(f"  - `{s['file']}`: crops {', '.join(f'`{c}`' for c in s['crops'])}" for s in sheets),
        "door_types": ", ".join(f'"{t}"' for t in catalogue().CATALOGUE["door"]),
        "window_types": ", ".join(f'"{t}"' for t in catalogue().CATALOGUE["window"]),
        "cornice_profiles": profiles(props["cornice"]),
        "skirting_profiles": profiles(props["skirting"]),
        "panel_frame_profiles": profiles(props["panel_frames"]["items"]),
        "architrave_profiles": profiles(schema["definitions"]["frame"]),
    }
    text = (PACKAGE / "vision_prompt.md").read_text(encoding="utf-8")
    for key, value in fields.items():
        assert "{{" + key + "}}" in text, key
        text = text.replace("{{" + key + "}}", value)
    assert "{{" not in text
    return text


# --- validation ----------------------------------------------------------------------------------------------------

def _is(value, kind: str) -> bool:
    return {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list), "string": lambda v: isinstance(v, str),
            "boolean": lambda v: isinstance(v, bool), "null": lambda v: v is None,
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}[kind](value)


def check(value, schema: dict, root: dict, path: str = "value") -> list[str]:
    """Errors of `value` against the JSON-Schema subset used by vision_schema.json (type, enum, required, bounds, $ref)."""
    if "$ref" in schema:
        schema = root["definitions"][schema["$ref"].rsplit("/", 1)[-1]]
    if "enum" in schema and not any(value == e and type(value) is type(e) for e in schema["enum"]):
        return [f"{path}: {value!r} is not one of {schema['enum']}"]
    kinds = schema.get("type")
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if kinds and not any(_is(value, k) for k in kinds):
        return [f"{path}: expected {' or '.join(kinds)}, got {value!r}"]
    errors = []
    if isinstance(value, dict):
        errors += [f"{path}: missing {k!r}" for k in schema.get("required", []) if k not in value]
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unknown field {k!r}" for k in value if k not in props]
        for k, sub in props.items():
            if k in value:
                errors += check(value[k], sub, root, f"{path}.{k}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            errors.append(f"{path}: expected {schema.get('minItems', 0)} to {schema.get('maxItems', 'any number of')} items, got {len(value)}")
        for i, item in enumerate(value):
            if "items" in schema:
                errors += check(item, schema["items"], root, f"{path}[{i}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            errors.append(f"{path}: {value} outside [{schema.get('minimum')}, {schema.get('maximum')}]")
    return errors


def _corner_errors(corners: list, crops: dict, kind: str, target: str, needed: int = 4) -> list[str]:
    """Corners must sit on crops of `kind` that show `target` (an element id, or the host wall for overview crops).

    A point may be null (hidden or outside the crop) as long as `needed` are given: an element is cast corner by
    corner, a declared niche or panel frame needs its whole rectangle.
    """
    errors = []
    for i, c in enumerate(corners):
        info = crops.get(c["crop"])
        if info is None:
            errors.append(f"corners[{i}]: unknown crop {c['crop']!r}")
            continue
        if info["kind"] != kind or info["target"] != target:
            errors.append(f"corners[{i}]: crop {c['crop']!r} is not a {kind} crop of {target!r}")
            continue
        w, h = info["size"]
        given = [p for p in c["uv"] if p is not None]
        uv = np.array(given, float).reshape(-1, 2)
        if len(given) < needed:
            errors.append(f"corners[{i}]: {len(given)} points given on {c['crop']!r}, at least {needed} needed")
        elif not ((uv >= 0).all() and (uv[:, 0] <= w).all() and (uv[:, 1] <= h).all()):
            errors.append(f"corners[{i}]: points outside the {w}x{h} crop {c['crop']!r}")
        elif len(given) == 4 and abs(float(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1)))) / 2 < MIN_CORNER_AREA:
            errors.append(f"corners[{i}]: degenerate quadrilateral")
    return errors


def _semantic_errors(name: str, item: dict, known: dict, crops: dict, seen: set) -> list[str]:
    cat = catalogue()
    errors = []
    if name == "elements":
        kind = known["kinds"].get(item["id"])
        if kind is None:
            return [f"unknown element id {item['id']!r}"]
        if item["id"] in seen:
            errors.append(f"element {item['id']!r} answered twice")
        if item.get("type") is not None and item["type"] not in cat.CATALOGUE.get(kind, {}):
            errors.append(f"type {item['type']!r} does not exist for a {kind}" if kind in cat.CATALOGUE else f"a {kind} carries no type")
        if item.get("hinge") not in (None, *cat.SIDES) or item.get("swing") not in (None, *cat.SWINGS):
            errors.append("hinge or swing outside the catalogue")
    else:
        if item["host_hint"] not in known["hosts"]:
            errors.append(f"unknown host {item['host_hint']!r}")
    if "corners" in item:
        errors += _corner_errors(item["corners"], crops, *(("element", item["id"], MIN_CORNERS) if name == "elements" else ("overview", item["host_hint"])))
    return errors


def validate(answer: object, schema: dict, known: dict, crops: dict) -> tuple[dict, list[dict]]:
    """(clean answer, rejects): items are kept or dropped one by one; nothing invalid reaches the clean answer.

    `known` = {"kinds": {element id: kind}, "hosts": [wall and partition ids]}; `crops` is crops.json.
    """
    clean = {"elements": [], "new_elements": [], "cornice": None, "skirting": None, "panel_frames": []}
    if not isinstance(answer, dict):
        return clean, [{"where": "result", "reason": "the answer is not a JSON object"}]
    rejects = []
    props = schema["properties"]
    for key, value in answer.items():
        if key not in props:
            rejects.append({"where": key, "reason": "unknown key"})
        elif props[key].get("type") == "array":
            if not isinstance(value, list):
                rejects.append({"where": key, "reason": "expected a list"})
                continue
            seen: set = set()
            for i, item in enumerate(value):
                errors = check(item, props[key]["items"], schema, key)
                errors = errors or _semantic_errors(key, item, known, crops, seen)
                if errors:
                    rejects.append({"where": f"{key}[{i}]", **({"id": item["id"]} if isinstance(item, dict) and isinstance(item.get("id"), str) else {}),
                                    "reason": "; ".join(errors)})
                else:
                    clean[key].append(item)
                    seen.add(item.get("id"))
        else:
            errors = check(value, props[key], schema, key)
            if errors:
                rejects.append({"where": key, "reason": "; ".join(errors)})
            else:
                clean[key] = value
    return clean, rejects


# --- running claude ---------------------------------------------------------------------------------------------------

def _command(model: str) -> list[str]:
    exe = os.environ.get("ROOM_MODEL_CLAUDE") or "claude"
    found = shutil.which(exe)
    if found is None:
        raise ClaudeMissing(f"claude not found: {exe} (set ROOM_MODEL_CLAUDE)")
    prefix = [sys.executable] if found.endswith(".py") else []
    return [*prefix, found, "-p", "--model", model, "--output-format", "json", "--allowedTools", "Read"]


class _CallFailed(Exception):
    """The call itself failed (process, envelope, claude error): nothing was answered, so nothing may be cached."""


def _result_text(stdout: str) -> str:
    """The `result` field of the JSON envelope printed by `claude -p --output-format json`; _CallFailed otherwise."""
    try:
        envelope = json.loads(stdout)
    except ValueError:
        raise _CallFailed("output is not the claude JSON envelope") from None
    if not isinstance(envelope, dict) or not isinstance(envelope.get("result"), str):
        raise _CallFailed("the claude JSON envelope has no text `result`")
    if envelope.get("is_error"):
        raise _CallFailed(f"claude reported an error: {envelope['result'][:300]}")
    return envelope["result"]


def _parse_answer(text: str) -> object:
    """The JSON object in the model's text: bare, inside a code fence, or between its first `{` and last `}`."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = (fenced.group(1) if fenced else text).strip()
    try:
        return json.loads(body)
    except ValueError:
        first, last = body.find("{"), body.rfind("}")
        if 0 <= first < last:
            return json.loads(body[first:last + 1])
        raise


def call_claude(pending: Path, model: str) -> tuple[str | None, str | None, float]:
    """(result text, error, seconds) of `claude -p` run in `pending` on its prompt.md; `error` is a message when the
    call itself failed (timeout, exit code, OSError, bad envelope), the text None then. ClaudeMissing if absent."""
    command = _command(model)
    prompt = (pending / "prompt.md").read_text(encoding="utf-8")
    started, text, error = time.monotonic(), None, None
    try:
        proc = subprocess.run(command, input=prompt, capture_output=True, text=True, cwd=pending, timeout=LLM_TIMEOUT_S, errors="replace")
        if proc.returncode != 0:
            raise _CallFailed(f"exit code {proc.returncode}: {(proc.stderr.strip() or proc.stdout.strip())[-400:]}")
        text = _result_text(proc.stdout)
    except subprocess.TimeoutExpired:
        error = f"timed out after {LLM_TIMEOUT_S} s"
    except OSError as exc:
        error = f"could not run claude: {exc}"
    except _CallFailed as exc:
        error = str(exc)
    return text, error, round(time.monotonic() - started, 1)


def run_llm(pending: Path, model: str) -> dict:
    """Runs `claude -p` in `pending` on prompt.md and validates the answer; never raises on a bad answer.

    Returns {"answer", "rejected", "model", "duration_s", "error"}; `error` is a message when the call itself failed
    (timeout, exit code, OSError, bad envelope) and None otherwise. ClaudeMissing when the executable does not exist.
    """
    schema = json.loads((pending / "schema.json").read_text(encoding="utf-8"))
    elements = json.loads((pending / "elements.json").read_text(encoding="utf-8"))
    crops = json.loads((pending / "crops.json").read_text(encoding="utf-8"))
    known = {"kinds": {e["id"]: e["kind"] for e in elements["elements"]}, "hosts": elements["room"]["hosts"]}
    answer, rejected = validate({}, schema, known, crops)[0], []  # an empty clean answer
    text, error, seconds = call_claude(pending, model)
    if error:
        rejected = [{"where": "claude", "reason": error}]
    else:
        try:
            answer, rejected = validate(_parse_answer(text), schema, known, crops)
        except ValueError as exc:
            rejected = [{"where": "result", "reason": f"not a JSON answer: {exc}"}]
    return {"answer": answer, "rejected": rejected, "model": model, "duration_s": seconds, "error": error}


def run(ctx: StageContext) -> Path:
    """vision/answers.json (validated answer, model, duration, crops) and vision/rejected.json; `--no-llm` skips the model.

    When the call itself fails, answers.json holds only {"error", "model", "duration_s"} and the CLI treats it as not done.
    """
    path = output_path(ctx, "vision")
    path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = path.parent / "rejected.json"
    if ctx.no_llm:
        rejected_path.unlink(missing_ok=True)
        path.write_text(json.dumps({"skipped": True}), encoding="utf-8")
        return path
    if not load_photos(ctx.capture, ctx.room):
        raise NoPhotos(f"no photos for room {ctx.room} in {ctx.capture}")
    _command(ctx.llm_model)  # fail before the sheets are made when claude is missing
    pending = prepare(ctx)
    result = run_llm(pending, ctx.llm_model)
    crops = json.loads((pending / "crops.json").read_text(encoding="utf-8"))
    meta = {"model": result["model"], "duration_s": result["duration_s"]}
    document = {"error": result["error"], **meta} if result["error"] else {**result["answer"], **meta, "crops": crops}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    rejected_path.write_text(json.dumps(result["rejected"], indent=2), encoding="utf-8")
    for reject in result["rejected"]:
        print(f"vision: rejected {reject['where']}: {reject['reason']}", file=sys.stderr)
    return path
