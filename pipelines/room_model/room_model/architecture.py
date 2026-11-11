"""Architecture stage: model.json plus the openings as architecture.json, and the tape measurement sheet."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from room_model.context import StageContext, StageInputMissing, output_path, require
from room_model.model import _Doc

SCHEMA = "room-model/arch-1"
VERSION = "v1-arch"
KIND_WORDS = {"door": "la porte", "window": "la fenêtre", "opening": "l'ouverture", "niche": "la niche"}
SILL_KINDS = ("window", "niche")
STATION_HEIGHT = 1.2  # m above the floor at which a width or length is read
VISION_PATH = "vision/answers.json"
TAPE_PATH = "measurements.json"
OVERRIDES_PATH = "overrides.json"
OPENING_KINDS = ("door", "window", "opening", "niche")


def _opening_element(o: dict) -> dict:
    return {"id": o["id"], "kind": o["kind"], "host": o["host"], "face": o["face"], "params": dict(o["params"]),
            "type": None, "flags": list(o["flags"])}


def _dimension(id_, kind, label, param, elements, at, flags=()) -> dict:
    """Dimension whose value is one parameter; a corrected parameter also carries its scan and LLM values."""
    out = {"id": id_, "kind": kind, "label": label, "value": param["value"], "sigma": param["sigma"],
           "elements": elements, "at": None if at is None else [float(at[0]), float(at[1])],
           "source": param["source"], "flags": list(flags)}
    if "scan_value" in param:
        out.update(scan_value=param["scan_value"], scan_sigma=param["scan_sigma"])
    if "llm_value" in param:
        out["llm_value"] = param["llm_value"]
    return out


def _opening_dimensions(o: dict, params: dict, center) -> list[dict]:
    word = KIND_WORDS.get(o["kind"], o["kind"])
    elements = [o["id"]] + ([o["host"]] if o["host"] else [])

    def make(suffix, kind, label, name):
        return _dimension(f"{o['id']}-{suffix}", kind, f"{label} {word} {o['id']}", params[o["params"][name]],
                          elements, center, o["flags"])

    out = [make("w", "opening_width", "Largeur de", "width"), make("h", "opening_height", "Hauteur de", "height")]
    if o["kind"] in SILL_KINDS:
        out.append(make("sill", "opening_sill", "Allège de", "sill"))
    return out


def _wall_length_dimensions(model_doc: dict) -> list[dict]:
    """L-M<k>: corner to corner along the room face of each wall; the corners move with the neighbouring walls."""
    d = _Doc(model_doc)
    walls = [e for e in model_doc["elements"] if e["kind"] == "wall"]

    def neighbour_at(point, other_end):
        for w in walls:
            if np.linalg.norm(np.subtract(w[other_end], point)) < 1e-6:
                return w
        return None

    out = []
    for el in walls:
        a, b = np.array(el["start"]), np.array(el["end"])
        prev, nxt = neighbour_at(a, "end"), neighbour_at(b, "start")
        sigma = np.hypot(d.line_sigma_at((prev or el)["label"], a), d.line_sigma_at((nxt or el)["label"], b))
        used = [el, *(w for w in (prev, nxt) if w)]
        out.append({"id": f"L-{el['label']}", "kind": "wall_length", "label": f"Longueur du mur {el['label']}, d'angle à angle",
                    "value": float(np.linalg.norm(b - a)), "sigma": float(sigma), "elements": [el["id"]],
                    "at": [float(v) for v in (a + b) / 2], "source": d.weakest([p for w in used for p in d.walls[w["label"]]["params"]]),
                    "flags": list(el.get("flags", []))})
    return out


def build(model_doc: dict, openings_doc: dict, vision: dict | None = None, tape: dict | None = None, ctx: StageContext | None = None,
          overrides: dict | None = None) -> dict:
    """architecture.json: the model, the openings as elements, and their dimensions plus the wall lengths.

    `vision` (the answers document), `tape` ({dimension id: metres}) and `overrides` ({element id: fields}) are merged
    in that order when given: see `llm_merge`, `tape_merge` and `overrides`. `ctx` gives the LLM corners their photos and the niche check its LiDAR samples.
    """
    from room_model import llm_merge, overrides as user_overrides, tape_merge

    arch = {"schema": SCHEMA, "version": VERSION, "source": model_doc.get("source"), "frame": copy.deepcopy(model_doc["frame"]),
            "parameters": copy.deepcopy({**model_doc["parameters"], **openings_doc["parameters"]}),
            "elements": copy.deepcopy([*model_doc["elements"], *map(_opening_element, openings_doc["openings"])]),
            "diagnostics": copy.deepcopy({**model_doc["diagnostics"], "openings": openings_doc["diagnostics"]}), "dimensions": []}
    if llm_merge.usable(vision):
        arch["diagnostics"]["vision"] = llm_merge.apply(arch, vision, ctx)
    arch["dimensions"] = copy.deepcopy(model_doc["dimensions"])
    if tape:
        arch["diagnostics"]["tape"] = tape_merge.apply(arch, tape)
    if overrides:
        arch["diagnostics"]["overrides"] = user_overrides.apply(arch, overrides)
    arch["dimensions"] += _wall_length_dimensions(arch)
    hosts, centres = {e["id"]: e for e in arch["elements"]}, {o["id"]: o["center"] for o in openings_doc["openings"]}
    for el in arch["elements"]:
        if el["kind"] in OPENING_KINDS:
            host = hosts.get(el["host"])
            centre = centres.get(el["id"]) if host is None else _face_point(arch, host, el.get("face"), _value(arch, el["params"]["center_s"]))
            arch["dimensions"] += _opening_dimensions(el, arch["parameters"], centre)
    for dim in arch["dimensions"]:
        ends = dimension_endpoints(arch, dim["id"])
        dim["endpoints"] = None if ends is None else [[float(c) for c in ends[0]], [float(c) for c in ends[1]]]
    return arch


def _value(arch: dict, pid: str) -> float:
    return arch["parameters"][pid]["value"]


def _face_point(arch: dict, host: dict, face: str | None, s: float) -> np.ndarray:
    """Plan point (x, z) at `s` m from the host's start, on its face A or B."""
    az = _value(arch, host["params"]["azimuth"])
    n = np.array([np.cos(az), np.sin(az)])
    start, end = np.array(host["start"], float), np.array(host["end"], float)
    p = start + (end - start) / np.linalg.norm(end - start) * s
    shift = 0.0 if face != "B" else _value(arch, host["params"]["thickness"])
    return p + n * (_value(arch, host["params"]["offset"]) - shift - n @ p)


def _opening_endpoints(arch: dict, dim: dict, element: dict):
    host = next((e for e in arch["elements"] if e["id"] == element["host"]), None)
    if host is None:
        return None
    floor = _value(arch, "floor.level")
    p = {name: _value(arch, pid) for name, pid in element["params"].items()}
    if dim["kind"] == "opening_width":
        y = floor + p["sill"] + p["height"] / 2
        (x0, z0), (x1, z1) = (_face_point(arch, host, element["face"], p["center_s"] + k * p["width"] / 2) for k in (-1, 1))
        return [x0, y, z0], [x1, y, z1]
    x, z = _face_point(arch, host, element["face"], p["center_s"])
    low, high = {"opening_height": (p["sill"], p["sill"] + p["height"]), "opening_sill": (0.0, p["sill"])}[dim["kind"]]
    return [x, floor + low, z], [x, floor + high, z]


def dimension_endpoints(arch: dict, dim_id: str) -> tuple[list[float], list[float]] | None:
    """The two ends [x, y, z] (room frame, metres) of the span a dimension measures; None if it has no span."""
    dim = next((d for d in arch["dimensions"] if d["id"] == dim_id), None)
    if dim is None:
        return None
    elements = {e["id"]: e for e in arch["elements"]}
    floor = _value(arch, "floor.level")
    kind = dim["kind"]
    if kind in ("width", "length"):
        wall = elements[dim["elements"][0]]
        az = _value(arch, wall["params"]["azimuth"])
        half = np.array([np.cos(az), np.sin(az)]) * dim["value"] / 2
        at = np.array(dim["at"])
        (x0, z0), (x1, z1) = at - half, at + half
        return [x0, floor + STATION_HEIGHT, z0], [x1, floor + STATION_HEIGHT, z1]
    if kind == "ceiling_height":
        x, z = dim["at"]
        return [x, floor, z], [x, floor + dim["value"], z]
    if kind == "wall_length":
        wall = elements[dim["elements"][0]]
        (x0, z0), (x1, z1) = wall["start"], wall["end"]
        return [x0, floor + STATION_HEIGHT, z0], [x1, floor + STATION_HEIGHT, z1]
    if kind in ("opening_width", "opening_height", "opening_sill"):
        return _opening_endpoints(arch, dim, elements[dim["elements"][0]])
    return None


def _read(path: Path, label: str) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageInputMissing(f"{label} illisible ({path}) : {exc}") from None
    if not isinstance(doc, dict):
        raise StageInputMissing(f"{label} illisible ({path}) : un objet JSON est attendu")
    return doc


def run(ctx: StageContext) -> Path:
    from room_model import report, survey_page

    model_doc = json.loads(require(ctx, "model").read_text(encoding="utf-8"))
    openings_doc = json.loads(require(ctx, "openings").read_text(encoding="utf-8"))
    vision_path, tape_path = ctx.out / VISION_PATH, ctx.out / TAPE_PATH
    vision, unreadable = None, None
    if vision_path.exists():
        try:
            vision = _read(vision_path, "réponses du LLM")
        except StageInputMissing as exc:
            unreadable = str(exc)
    tape = _read(tape_path, "mesures au mètre") if tape_path.exists() else None
    overrides = _read(ctx.out / OVERRIDES_PATH, "corrections manuelles") if (ctx.out / OVERRIDES_PATH).exists() else None
    arch = build(model_doc, openings_doc, vision, tape, ctx, overrides)
    if unreadable:
        arch["diagnostics"]["vision_unreadable"] = unreadable
    path = output_path(ctx, "architecture")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(arch, indent=2), encoding="utf-8")
    survey_page.write(ctx, arch)
    report.write_sections(ctx.out / "report.md", arch)
    return path
