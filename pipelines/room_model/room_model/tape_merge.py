"""Apply the user's tape measurements (measurements.json, metres per dimension id) to the architecture.

The file is either `{"<id>": metres}` or `{"version": 2, "capture": ..., "values": {"<id>": {"value_m", "date", "note"}}}`.
A leaf measure (`<id>-leaf-w`, `-leaf-h`: the moving panel of a door or window) creates its parameter and dimension.

An opening dimension replaces the parameter it measures; a ceiling height moves the ceiling level; a station moves
the wall of its pair with the larger offset sigma along its normal. Whatever cannot be applied is listed, never applied.
"""
from __future__ import annotations

import math
import re

import numpy as np

from room_model import regularize, shell
from room_model.architecture import _face_point
from room_model.llm_merge import RECT_KINDS, _elements, _flag

TAPE_SIGMA = 0.005  # m
MOVE_SIGMAS = 10  # a wall moves by at most this many station sigmas or one wall thickness, whichever is larger:
# beyond it the tape and the scan describe different places (wrong face, typing mistake)
ON_LINE = 1e-4  # m; a plan point this close to a wall line lies on it
OPENING_PARAM = {"w": "width", "h": "height", "sill": "sill", "leaf-w": "leaf_width", "leaf-h": "leaf_height"}
LEAF_KINDS = ("door", "window")
LEAF_DIMENSION = {"leaf_width": ("leaf-w", "Largeur du vantail de"), "leaf_height": ("leaf-h", "Hauteur du vantail de")}
STATION_KINDS = ("width", "length")


def _lines(arch: dict) -> dict:
    return {w["label"]: w for w in shell.wall_lines(arch)}


def _scan(p: dict) -> tuple[float, float]:
    return p.get("scan_value", p["value"]), p.get("scan_sigma", p["sigma"])


def _set_tape(p: dict, value: float) -> None:
    scan, scan_sigma = _scan(p)
    p.update(value=float(value), sigma=TAPE_SIGMA, source="tape", scan_value=scan, scan_sigma=scan_sigma)


def _opening(arch: dict, dim_id: str) -> tuple[dict, str] | None:
    found = re.fullmatch(r"(.+?)-(leaf-w|leaf-h|w|h|sill)", dim_id)
    el = _elements(arch).get(found.group(1)) if found else None
    if el is None or el["kind"] not in RECT_KINDS or (found.group(2) == "sill" and el["kind"] not in ("window", "niche")):
        return None
    if found.group(2).startswith("leaf") and el["kind"] not in LEAF_KINDS:
        return None
    return el, OPENING_PARAM[found.group(2)]


def _set_leaf(arch: dict, el: dict, name: str, value: float) -> None:
    """Creates or updates the leaf parameter of a door or window and its dimension (source tape)."""
    pid = el["params"].get(name) or f"{el['params']['width'].rsplit('.', 1)[0]}.{name}"
    arch["parameters"][pid] = {"value": float(value), "sigma": TAPE_SIGMA, "unit": "m", "source": "tape"}
    el["params"][name] = pid
    dims = {d["id"]: d for d in arch["dimensions"]}
    suffix, label = LEAF_DIMENSION[name]
    dim_id = f"{el['id']}-{suffix}"
    dim = {"id": dim_id, "kind": name, "label": f"{label} {el['id']}", "value": float(value), "sigma": TAPE_SIGMA,
           "elements": [el["id"]] + ([el["host"]] if el.get("host") else []), "at": None, "source": "tape", "flags": []}
    if dim_id in dims:
        dims[dim_id].update(value=dim["value"], sigma=TAPE_SIGMA, source="tape")
    else:
        arch["dimensions"].append(dim)


# --- walls --------------------------------------------------------------------------------------------------------

def _anchors(arch: dict) -> dict:
    """Per station: the point of its first wall's line the span starts from, and that line's offset, before any move."""
    lines, out = _lines(arch), {}
    for dim in arch["dimensions"]:
        if dim["kind"] not in STATION_KINDS or not all(e.removeprefix("Wall_") in lines for e in dim["elements"]):
            continue
        a = lines[dim["elements"][0].removeprefix("Wall_")]
        at = np.array(dim["at"])
        out[dim["id"]] = (at - (a["normal"] @ at - a["offset"]) * a["normal"], a["offset"])
    return out


def _span(lines: dict, dim: dict, anchor: tuple, shift_a: float = 0.0, shift_b: float = 0.0) -> float:
    """Signed distance along the first wall's normal from its line to the second wall's, with optional offset shifts."""
    a, b = (lines[e.removeprefix("Wall_")] for e in dim["elements"])
    point = anchor[0] + (a["offset"] + shift_a - anchor[1]) * a["normal"]
    return float((b["offset"] + shift_b - b["normal"] @ point) / (b["normal"] @ a["normal"]))


def _remap(point, before: dict, after: dict, moved: str):
    """New position of a plan point: moved with the wall it lies on, or at the new corner when it lies on a neighbour too."""
    p = np.asarray(point, float)
    if abs(before[moved]["normal"] @ p - before[moved]["offset"]) > ON_LINE:
        return p
    n, offset = after[moved]["normal"], after[moved]["offset"]
    for label, line in before.items():
        if label != moved and abs(line["normal"] @ p - line["offset"]) <= ON_LINE and abs(n[0] * line["normal"][1] - n[1] * line["normal"][0]) > 0.1:
            return np.linalg.solve(np.stack([n, line["normal"]]), [offset, line["offset"]])
    return p + (offset - before[moved]["offset"]) * n


def _move_wall(arch: dict, wall: dict, delta: float) -> None:
    """Moves a wall along its normal; corners, floor, ceilings and openings follow so that nothing leaves its place."""
    before = _lines(arch)
    p = arch["parameters"][wall["params"]["offset"]]
    _set_tape(p, p["value"] + delta)
    _flag(wall, "tape_corrected")
    after = _lines(arch)
    moved = wall["label"]
    moves = []
    for el in arch["elements"]:
        if "start" in el:
            start, end = (_remap(el[k], before, after, moved) for k in ("start", "end"))
            u = np.subtract(el["end"], el["start"]) / np.linalg.norm(np.subtract(el["end"], el["start"]))
            moves.append((el["id"], float((start - el["start"]) @ u)))
            el["start"], el["end"] = [float(c) for c in start], [float(c) for c in end]
        elif el["kind"] == "floor":
            el["polygon"] = [[float(c) for c in _remap(v, before, after, moved)] for v in el["polygon"]]
        elif el["kind"] == "ceiling":
            for region in el["region"]:
                region["exterior"] = [[float(c) for c in _remap(v, before, after, moved)] for v in region["exterior"]]
                region["holes"] = [[[float(c) for c in _remap(v, before, after, moved)] for v in hole] for hole in region["holes"]]
    if any("extent" in el for el in arch["elements"]):
        regularize.underlay(arch)
    hosts = _elements(arch)
    for host_id, shift in moves:
        for el in arch["elements"]:
            if el.get("host") != host_id:
                continue
            if "center_s" in el.get("params", {}):
                arch["parameters"][el["params"]["center_s"]]["value"] -= shift
            elif el["kind"] == "panel_frame":
                el["rect"] = [el["rect"][0] - shift, el["rect"][1], el["rect"][2] - shift, el["rect"][3]]
    for el in arch["elements"]:
        if "center" in el and el.get("host") in hosts:
            x, z = _face_point(arch, hosts[el["host"]], el.get("face"), arch["parameters"][el["params"]["center_s"]]["value"])
            el["center"] = [float(x), float(z)]


def _refresh_stations(arch: dict, anchors: dict, changed: dict) -> None:
    lines = _lines(arch)
    for dim in arch["dimensions"]:
        if dim["id"] not in anchors:
            continue
        t = _span(lines, dim, anchors[dim["id"]])
        if abs(abs(t) - dim["value"]) > 1e-9:
            changed.setdefault(dim["id"], (dim["value"], dim["sigma"]))
            a = lines[dim["elements"][0].removeprefix("Wall_")]
            point = anchors[dim["id"]][0] + (a["offset"] - anchors[dim["id"]][1]) * a["normal"]
            dim["value"], dim["at"] = float(abs(t)), [float(c) for c in point + t / 2 * a["normal"]]


def _apply_station(arch: dict, dim: dict, value: float, anchors: dict, changed: dict, rep: dict) -> dict:
    lines, elements = _lines(arch), _elements(arch)
    first, second = (elements[e] for e in dim["elements"])
    sigma = [arch["parameters"][w["params"]["offset"]]["sigma"] for w in (first, second)]
    mover = first if sigma[0] > sigma[1] else second
    t0 = _span(lines, dim, anchors[dim["id"]])
    shifts = (1.0, 0.0) if mover is first else (0.0, 1.0)
    slope = _span(lines, dim, anchors[dim["id"]], *shifts) - t0
    if abs(slope) < 1e-9:
        raise ValueError("murs non parallèles, la cote ne dépend pas d'un déplacement")
    delta = (math.copysign(value, t0) - t0) / slope
    limit = max(MOVE_SIGMAS * dim["sigma"], arch["parameters"][mover["params"]["thickness"]]["value"])
    if abs(delta) > limit:
        raise ValueError(f"déplacement du mur {mover['label']} de {abs(delta) * 100:.0f} cm (au-delà de {limit * 100:.0f} cm), saisie écartée".replace(".", ","))
    if abs(delta) > 1e-9:
        _move_wall(arch, mover, delta)
        rep["moved"].append({"wall": mover["id"], "delta": float(delta), "station": dim["id"]})
    _refresh_stations(arch, anchors, changed)
    return {"effect": f"mur {mover['id']} déplacé de {'+' if delta >= 0 else '-'}{abs(delta) * 100:.1f} cm".replace(".", ",")}


def _corner(a: dict, b: dict) -> np.ndarray:
    return np.linalg.solve(np.stack([a["normal"], b["normal"]]), [a["offset"], b["offset"]])


def _apply_wall_length(arch: dict, dim_id: str, value: float, anchors: dict, changed: dict, rep: dict) -> dict:
    """`L-<label>`: the wall's corner-to-corner length, set by moving whichever of its two end walls has the larger
    offset sigma along its normal (bounded like a station: MOVE_SIGMAS sigmas or one wall thickness)."""
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    wall = next((w for w in walls if f"L-{w['label']}" == dim_id), None)
    if wall is None:
        raise ValueError("cote inconnue de l'architecture")
    near = lambda p, q: np.linalg.norm(np.subtract(p, q)) < ON_LINE  # noqa: E731
    prev = next((w for w in walls if w is not wall and near(w["end"], wall["start"])), None)
    nxt = next((w for w in walls if w is not wall and near(w["start"], wall["end"])), None)
    if prev is None or nxt is None:
        raise ValueError(f"le mur {wall['label']} n'a pas deux murs voisins, sa longueur ne se règle pas")
    sigma = {w["id"]: arch["parameters"][w["params"]["offset"]]["sigma"] for w in (prev, nxt)}
    mover = prev if sigma[prev["id"]] > sigma[nxt["id"]] else nxt

    def length(shift: float) -> float:
        lines = {k: dict(v) for k, v in _lines(arch).items()}
        lines[mover["label"]]["offset"] += shift
        return float(np.linalg.norm(_corner(lines[wall["label"]], lines[nxt["label"]]) - _corner(lines[prev["label"]], lines[wall["label"]])))

    slope = length(1e-3) / 1e-3 - length(0.0) / 1e-3
    if abs(slope) < 1e-6:
        raise ValueError("murs voisins parallèles au mur, sa longueur ne dépend pas d'un déplacement")
    before = length(0.0)
    delta = (value - before) / slope
    limit = max(MOVE_SIGMAS * float(np.hypot(*sigma.values())), arch["parameters"][mover["params"]["thickness"]]["value"])
    if abs(delta) > limit:
        raise ValueError(f"déplacement du mur {mover['label']} de {abs(delta) * 100:.0f} cm (au-delà de {limit * 100:.0f} cm), saisie écartée".replace(".", ","))
    if abs(delta) > 1e-9:
        _move_wall(arch, mover, delta)
        rep["moved"].append({"wall": mover["id"], "delta": float(delta), "station": dim_id})
    _refresh_stations(arch, anchors, changed)
    return {"effect": f"mur {mover['id']} déplacé de {'+' if delta >= 0 else '-'}{abs(delta) * 100:.1f} cm".replace(".", ","),
            "scan": before}


def _apply_ceiling(arch: dict, dim: dict, value: float) -> dict:
    ceiling = _elements(arch)[dim["elements"][0]]
    pid = ceiling["params"]["level"]
    p = arch["parameters"][pid]
    old, new = p["value"], arch["parameters"]["floor.level"]["value"] + value
    for key, q in arch["parameters"].items():
        if key.endswith(".top") and abs(q["value"] - old) < 1e-9:
            q["value"] = float(new)
    _set_tape(p, new)
    _flag(ceiling, "tape_corrected")
    return {"effect": f"plafond {ceiling['id']} déplacé de {'+' if new >= old else '-'}{abs(new - old) * 100:.1f} cm".replace(".", ",")}


# --- entry point ----------------------------------------------------------------------------------------------------

def _number(raw) -> float | None:
    ok = isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw) and raw > 0
    return float(raw) if ok else None


def normalize(doc: dict) -> dict:
    """{id: metres} from either file format; an unusable v2 entry becomes None or stays as is, for `apply` to report."""
    if doc.get("version") != 2 or not isinstance(doc.get("values"), dict):
        return doc
    return {k: (v.get("value_m") if isinstance(v, dict) else v) for k, v in doc["values"].items()
            if not (isinstance(v, dict) and "value_m" in v and v["value_m"] is None)}  # a note without a value: nothing to apply


def read_store(doc: dict) -> dict:
    """{id: {"value_m", "date", "note"}} from either file format, for pre-filling the survey page; unusable entries are dropped.

    `value_m` is None for a note without a value.
    """
    out = {}
    for key, raw in (doc["values"] if doc.get("version") == 2 and isinstance(doc.get("values"), dict) else doc).items():
        entry = raw if isinstance(raw, dict) else {"value_m": raw}
        if _number(entry.get("value_m")) is not None or (entry.get("value_m") is None and str(entry.get("note") or "").strip()):
            out[key] = {"value_m": None if entry.get("value_m") is None else float(entry["value_m"]), "date": str(entry.get("date") or ""), "note": str(entry.get("note") or "")}
    return out


def apply(arch: dict, tape: dict) -> dict:
    """Applies the measurements to `arch` in place (its model dimensions must already be in `arch["dimensions"]`).

    Returns {"applied": [{id, scan, llm, tape, applied, effect}], "not_applied": [{id, reason}], "moved": [...],
    "derived": [{id, scan, applied}]}; `derived` lists station values that changed because a wall moved.
    """
    rep = {"applied": [], "not_applied": [], "moved": [], "derived": []}
    tape = normalize(tape)
    dims = {d["id"]: d for d in arch["dimensions"]}
    anchors, changed = _anchors(arch), {}
    for dim_id, raw in tape.items():
        value = _number(raw)
        if value is None:
            rep["not_applied"].append({"id": dim_id, "reason": "valeur invalide (un nombre de mètres positif est attendu)"})
            continue
        row = {"id": dim_id, "scan": None, "llm": None, "tape": value, "applied": value, "effect": None}
        try:
            if (target := _opening(arch, dim_id)) is not None:
                el, name = target
                if name.startswith("leaf"):
                    _set_leaf(arch, el, name, value)
                else:
                    p = arch["parameters"][el["params"][name]]
                    row.update(scan=_scan(p)[0], llm=p.get("llm_value"))
                    _set_tape(p, value)
                _flag(el, "tape_corrected")
            elif (dim := dims.get(dim_id)) is not None and (dim_id in anchors or dim["kind"] == "ceiling_height"):
                row.update(scan=dim.get("scan_value", changed.get(dim_id, (dim["value"],))[0]))
                if dim["kind"] == "ceiling_height":
                    row["effect"] = _apply_ceiling(arch, dim, value)["effect"]
                else:
                    row["effect"] = _apply_station(arch, dim, value, anchors, changed, rep)["effect"]
                dim.update(scan_value=row["scan"], scan_sigma=changed.get(dim_id, (None, dim["sigma"]))[1], value=value, sigma=TAPE_SIGMA, source="tape")
            elif dim_id.startswith("L-"):
                done = _apply_wall_length(arch, dim_id, value, anchors, changed, rep)
                row.update(effect=done["effect"], scan=done["scan"])
            elif dim_id in dims:
                raise ValueError("cote non mesurable au mètre sur le relevé")
            else:
                raise ValueError("cote inconnue de l'architecture")
        except ValueError as exc:
            rep["not_applied"].append({"id": dim_id, "reason": str(exc)})
            continue
        rep["applied"].append(row)
    done = {r["id"] for r in rep["applied"]}
    rep["derived"] = [{"id": i, "scan": scan, "applied": dims[i]["value"]} for i, (scan, _) in changed.items() if i not in done]
    for i, (scan, sigma) in changed.items():
        if i not in done:
            dims[i].update(scan_value=scan, scan_sigma=sigma)
    return rep
