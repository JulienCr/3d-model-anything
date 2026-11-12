"""Regularised model: walls squared within their measured uncertainty, floor and ceilings rebuilt from them."""
from __future__ import annotations

import copy

import numpy as np
import shapely

from room_model import ceilings, columns, shell, zones
from room_model.lidar import Samples

SQUARING_FLOOR_DEG = 2.0  # user requirement: an architectural model is drawn square unless the scan proves more than 2 deg of deviation
SQUARING_SIGMAS = 3.0  # a deviation within 3 sigma_azimuth cannot be told from the fit noise
FRAME_PASSES = 4
FRAME_STEP_DEG = 0.05  # resolution of the frame scan, far below the squaring tolerance
QUARTER = np.pi / 2
FLOOR_SILL = 0.03  # m: an opening whose sill is this low (the parameters' own sigma is of that order) is cut down to the floor
HIDDEN_DROP = 0.001  # m, see `underlay`
ON_BOUNDARY = 1e-6  # m: a segment end this close to the old floor boundary touched a wall


def _wrap_quarter(x):
    """Signed deviation (rad) from the nearest multiple of a quarter turn."""
    return (np.asarray(x) + QUARTER / 2) % QUARTER - QUARTER / 2


def _normal(az: float) -> np.ndarray:
    return np.array([np.cos(az), np.sin(az)])


def _records(doc: dict, kind: str) -> list[dict]:
    """Wall or partition lines of the document: parameters, fit centroid and fit support."""
    diag = doc["diagnostics"]
    out = []
    for el in doc["elements"]:
        if el["kind"] != kind:
            continue
        az, off = (doc["parameters"][el["params"][n]] for n in ("azimuth", "offset"))
        info = diag["walls"][el["label"]] if kind == "wall" else next(d for d in diag["partitions"] if d["id"] == el["id"])
        if info.get("centroid") is not None:
            centroid = np.array(info["centroid"], float)
        else:
            mid = (np.array(el["start"]) + np.array(el["end"])) / 2
            n = _normal(az["value"])
            centroid = mid - (n @ mid - off["value"]) * n
        support = info.get("support_m2") if kind == "wall" else None
        out.append({"el": el, "az": az, "off": off, "label": el["label"], "centroid": centroid,
                    "weight": float(support or 0.0), "tol": max(np.radians(SQUARING_FLOOR_DEG), SQUARING_SIGMAS * az["sigma"])})
    return out


def _mean_azimuth(az: np.ndarray, weights: np.ndarray) -> float:
    return float(np.angle(np.sum(weights * np.exp(4j * az))) / 4)


def _frame(walls: list[dict]) -> float:
    """Azimuth (rad) of the dominant orthogonal frame: support-weighted circular mean of the wall azimuths mod 90 degrees.

    The mean runs over the walls that can be squared onto it. They are found by scanning frames every FRAME_STEP_DEG
    for the one that takes in the most support, so a wall far off the axes (a chamfer) cannot drag the frame.
    """
    az = np.array([w["az"]["value"] for w in walls])
    weights = np.array([w["weight"] for w in walls])
    weights = weights if weights.sum() > 0 else np.ones(len(walls))
    tol = np.array([w["tol"] for w in walls])
    grid = np.radians(np.arange(-45.0, 45.0, FRAME_STEP_DEG))
    inside = np.abs(_wrap_quarter(az[None, :] - grid[:, None])) <= tol[None, :]
    score = inside @ weights
    best = grid[score >= score.max() - 1e-12]
    frame = _mean_azimuth(best, np.ones(len(best)))
    for _ in range(FRAME_PASSES):
        near = np.abs(_wrap_quarter(az - frame)) <= tol
        if not near.any():
            break
        refined = _mean_azimuth(az[near], weights[near])
        if abs(float(_wrap_quarter(refined - frame))) < 1e-12:
            break
        frame = refined
    return frame


def _decide(lines: list[dict], frame: float) -> None:
    """Sets `state`, `azimuth` (retained) and `deviation` on each line record."""
    for ln in lines:
        dev = float(_wrap_quarter(ln["az"]["value"] - frame))
        ln["deviation"] = abs(dev)
        if abs(dev) <= ln["tol"]:
            axis = frame + round((ln["az"]["value"] - frame - dev) / QUARTER) * QUARTER
            ln["state"], ln["azimuth"] = "squared", float(np.arctan2(np.sin(axis), np.cos(axis)))
        else:
            ln["state"], ln["azimuth"] = "out_of_square", ln["az"]["value"]
        ln["offset"] = float(_normal(ln["azimuth"]) @ ln["centroid"]) if ln["state"] == "squared" else ln["off"]["value"]


def _refit(start, end, old_ring: shapely.LinearRing, polygon: shapely.Polygon) -> tuple[list[float], list[float]]:
    """The segment clipped to the new polygon, an end that touched the old boundary extended to reach the new one."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    d = (b - a) / np.linalg.norm(b - a)
    a2 = a - zones.CUT_REACH * d if old_ring.distance(shapely.Point(a)) < ON_BOUNDARY else a
    b2 = b + zones.CUT_REACH * d if old_ring.distance(shapely.Point(b)) < ON_BOUNDARY else b
    pieces = [g for g in shapely.get_parts(polygon.intersection(shapely.LineString([a2, b2]))) if g.geom_type == "LineString" and not g.is_empty]
    if not pieces:
        return list(map(float, start)), list(map(float, end))
    coords = sorted((np.array(c) for c in max(pieces, key=lambda g: g.length).coords), key=lambda c: c @ d)
    return [float(v) for v in coords[0]], [float(v) for v in coords[-1]]


def _footprint(axis: list, width: float, polygon: shapely.Polygon) -> shapely.Geometry:
    """Plan footprint of a beam, an end on the wall running past it so that the flat end leaves no sliver along the wall."""
    a, b = np.array(axis, float)
    d = (b - a) / np.linalg.norm(b - a)
    reach = [zones.CUT_REACH if polygon.exterior.distance(shapely.Point(p)) < ON_BOUNDARY else 0.0 for p in (a, b)]
    return ceilings.beam_footprint({"axis": [a - reach[0] * d, b + reach[1] * d], "width": width}, polygon)


def _snap_columns(out: dict, walls: list[dict]) -> None:
    """Move the engaged side of each column onto its squared wall, the opposite side staying where it was measured."""
    params, lines = out["parameters"], {w["label"]: w for w in walls}
    engaged = {c["id"]: c["engaged_walls"] for c in out["diagnostics"].get("columns", [])}
    for el in (e for e in out["elements"] if e["kind"] == "column"):
        p = {k: params[v] for k, v in el["params"].items()}
        for label in engaged.get(el["id"], [el["engaged_wall"]] if el.get("engaged_wall") else []):
            n, off = _normal(lines[label]["azimuth"]), lines[label]["offset"]
            axis = 0 if abs(n[0]) > abs(n[1]) else 1
            name = "xz"[axis]
            centre, size, other = p[f"center_{name}"], p[f"size_{name}"], p[f"center_{'zx'[axis]}"]["value"]
            wall = (off - n[1 - axis] * other) / n[axis]
            faces = [centre["value"] - size["value"] / 2, centre["value"] + size["value"] / 2]
            side = int(np.argmin([abs(f - wall) for f in faces]))
            new = faces[:]
            new[side] = float(wall)
            if new[1] - new[0] >= columns.SIZE_RANGE[0]:
                centre["value"], size["value"] = (new[0] + new[1]) / 2, new[1] - new[0]


def _set_param(params: dict, pid: str, level: dict, assumed: bool) -> None:
    params[pid] = {"value": float(level["level"]), "sigma": float(level["sigma"]), "unit": "m",
                   "source": "assumed" if assumed else "lidar"}


def _follow(params: dict, pid: str, levels: list[dict]) -> None:
    """A beam or column top is copied from the ceiling it meets: move it to the new level that replaces its old one."""
    p = params[pid]
    near = min(levels, key=lambda lv: abs(lv["level"] - p["value"]))
    if abs(near["level"] - p["value"]) < ceilings.MERGE_SIGMAS * max(near["sigma"], p["sigma"]):
        _set_param(params, pid, near, near["assumed"])


def _rings(region: shapely.MultiPolygon) -> list[dict]:
    return [{"exterior": [[float(x), float(z)] for x, z in g.exterior.coords[:-1]],
             "holes": [[[float(x), float(z)] for x, z in h.coords[:-1]] for h in g.interiors]} for g in region.geoms]


def _thresholds(doc: dict, outer: shapely.Polygon, notes: list[dict]) -> list[shapely.Polygon]:
    """Plan strip of each floor-based cut (a door, a passage, a through niche down to the floor) in a wall, from the
    interior face to the outer face: the threshold is floor too. A strip that is degenerate or off the walls is noted."""
    params, by_id, out = doc["parameters"], {e["id"]: e for e in doc["elements"]}, []
    for el in doc["elements"]:
        host = by_id.get(el.get("host"))
        if el["kind"] not in ("door", "opening", "niche") or host is None or host["kind"] != "wall":
            continue
        if params[el["params"]["sill"]]["value"] > FLOOR_SILL or (el["kind"] == "niche" and not el.get("through")):
            continue
        az = params[host["params"]["azimuth"]]["value"]
        n, thickness = _normal(az), params[host["params"]["thickness"]]["value"]
        start, end = np.array(host["start"]), np.array(host["end"])
        u = (end - start) / np.linalg.norm(end - start)
        centre, half = params[el["params"]["center_s"]]["value"], params[el["params"]["width"]]["value"] / 2
        a, b = (start + u * (centre + s * half) for s in (-1.0, 1.0))
        a, b = (q + n * (params[host["params"]["offset"]]["value"] - n @ q) for q in (a, b))
        strip = shapely.Polygon([a, b, b - n * thickness, a - n * thickness]).buffer(0)
        if strip.is_empty or not strip.intersects(outer):
            notes.append({"reason": "threshold_dropped", "element": el["id"]})
        else:
            out.append(strip)
    return out


def _parts(geom: shapely.Geometry, drop: float) -> list[dict]:
    """Rings of the polygons of `geom` as `{"exterior", "holes", "drop"}`; builders draw a part `drop` m below its level."""
    return [{**r, "drop": drop} for r in _rings(ceilings._as_multipolygon(geom))]


def underlay(doc: dict) -> None:
    """Give the floor and each ceiling an `extent`, a list of parts `{"exterior", "holes", "drop"}`.

    The interior (`polygon`, `region`) is drawn at its level. Pushed under (over) the walls to their mid-thickness, so
    that its edge is hidden inside the bodies, the rim is drawn HIDDEN_DROP lower: a rim at the very height of a wall base
    or top would have a face coplanar with it and facing the other way. The floor also runs at its level to the outer face
    through floor-based cuts. The walls' lines move outward by half their thickness and are joined by the shell's corner
    rule. No extent is written, and `diagnostics.underlay_notes` says so, when the outer lines do not close.
    """
    notes: list[dict] = []
    doc["diagnostics"]["underlay_notes"] = notes
    elements = doc["elements"]
    floor = next(e for e in elements if e["kind"] == "floor")
    inner = shapely.Polygon(floor["polygon"])
    walls = [e for e in elements if e["kind"] == "wall"]
    lines = [{"n": w["normal"], "offset": w["offset"] - doc["parameters"][el["params"]["thickness"]]["value"] / 2, "flags": []}
             for w, el in zip(shell.wall_lines(doc), walls)]
    outer = shapely.Polygon(shell._corners(np.array([[x, 0.0, z] for x, z in floor["polygon"]]), lines))
    if not outer.is_valid or not outer.contains(inner):
        notes.append({"reason": "no_extent"})
        for el in elements:
            el.pop("extent", None)
        return
    sills = zones._opened(shapely.union_all([inner, *_thresholds(doc, outer, notes)])).difference(inner)
    floor["extent"] = (_parts(inner, 0.0) + _parts(sills, 0.0)
                       + _parts(zones._opened(outer.difference(shapely.union_all([inner, sills]))), HIDDEN_DROP))
    band = outer.difference(inner)
    reach = max(doc["parameters"][w["params"]["thickness"]]["value"] for w in walls) / 2 + zones.MIN_FEATURE
    for el in (e for e in elements if e["kind"] == "ceiling"):
        region = shapely.MultiPolygon([shapely.Polygon(r["exterior"], r["holes"]) for r in el["region"]])
        grown = shapely.union_all([region, region.buffer(reach, join_style="mitre").intersection(band)])
        closed = grown.buffer(2 * zones.MIN_FEATURE, join_style="mitre").buffer(-2 * zones.MIN_FEATURE, join_style="mitre")
        el["extent"] = _parts(region, 0.0) + _parts(zones._opened(closed.difference(region)), HIDDEN_DROP)


def _rebuild_ceilings(out: dict, polygon: shapely.Polygon, samples: Samples, segments: list, beams: shapely.Geometry) -> list[dict]:
    params = out["parameters"]
    cand = ceilings.ceiling_points(samples, polygon, params["floor.level"]["value"])
    found, notes = zones.build(polygon, zones.split(polygon, segments), cand, beams)
    out["diagnostics"]["ceiling_notes"] = notes
    for pid in [p for p in params if p.startswith("ceiling.")]:
        del params[pid]
    elements, diag = [], []
    for k, lv in enumerate(found, 1):
        _set_param(params, f"ceiling.{k}.level", lv, lv["assumed"])
        elements.append({"id": f"Ceiling_{k}", "kind": "ceiling", "params": {"level": f"ceiling.{k}.level"}, "region": _rings(lv["region"]),
                         "flags": ["assumed"] if lv["assumed"] else []})
        diag.append({"id": f"Ceiling_{k}", "support_m2": None if lv["assumed"] else lv["support_m2"], "area_m2": float(lv["region"].area),
                     "other_levels": [{"level": o["level"], "support_m2": o["support_m2"]} for o in lv["others"]]})
    rest = [e for e in out["elements"] if e["kind"] != "ceiling"]
    at = next((i for i, e in enumerate(rest) if e["kind"] in ("beam", "column")), len(rest))
    out["elements"] = rest[:at] + elements + rest[at:]
    out["diagnostics"]["ceilings"] = diag
    return found


def apply(doc: dict, outline: np.ndarray, samples: Samples) -> dict:
    """The shell document with squared walls, the floor and ceilings rebuilt from them, and `diagnostics.squaring`.

    `outline` is RoomPlan's floor outline (xyz) that places near-parallel corners. A wall or partition whose azimuth is
    within max(2 degrees, 3 sigma) of the dominant frame is turned onto it about its fit centroid, the measured azimuth
    kept as `measured_value`. Others keep their measurement and get the flag `out_of_square`.
    """
    walls, parts = _records(doc, "wall"), _records(doc, "partition")
    frame = _frame(walls)
    _decide(walls + parts, frame)
    lines = [{"n": _normal(w["azimuth"]), "offset": w["offset"], "flags": []} for w in walls]
    corners = shell._corners(outline, lines)
    polygon = [[float(c[0]), float(c[1])] for c in corners]
    new = shapely.Polygon(polygon)
    if not new.is_valid or new.is_empty:
        skipped = copy.deepcopy(doc)
        skipped["diagnostics"]["flags"].append("squaring:invalid_polygon")
        return skipped

    out = copy.deepcopy(doc)
    params, diag = out["parameters"], out["diagnostics"]
    elements = {e["id"]: e for e in out["elements"]}
    old_ring = shapely.LinearRing(next(e for e in out["elements"] if e["kind"] == "floor")["polygon"])
    stale = {f"{w['label']}:near_parallel_corner" for w in walls}
    diag["flags"] = [f for f in diag["flags"] if f not in stale]
    for ln in walls + parts:
        el = elements[ln["el"]["id"]]
        if ln["state"] == "squared":
            az_p, off_p = params[el["params"]["azimuth"]], params[el["params"]["offset"]]
            az_p.update(measured_value=ln["az"]["value"], value=ln["azimuth"])
            off_p["value"] = ln["offset"]
            el["flags"].append("squared")
        else:
            el["flags"].append("out_of_square")
            diag["flags"].append(f"{ln['label']}:out_of_square")
    for k, (w, line) in enumerate(zip(walls, lines)):
        el = elements[w["el"]["id"]]
        el["start"], el["end"] = polygon[k], polygon[(k + 1) % len(polygon)]
        el["flags"] = [f for f in el["flags"] if f != "near_parallel_corner"] + line["flags"]
        diag["flags"] += [f"{w['label']}:{f}" for f in line["flags"]]
    next(e for e in out["elements"] if e["kind"] == "floor")["polygon"] = polygon

    segments = []
    for p in parts:
        el = elements[p["el"]["id"]]
        el["start"], el["end"] = _refit(el["start"], el["end"], old_ring, new)
        segments.append((el["start"], el["end"]))
    beam_shapes = []
    for el in (e for e in out["elements"] if e["kind"] == "beam"):
        el["axis"] = list(_refit(*el["axis"], old_ring, new))
        segments.append(tuple(el["axis"]))
        beam_shapes.append(_footprint(el["axis"], params[el["params"]["width"]]["value"], new))
    beams = shapely.union_all(beam_shapes) if beam_shapes else shapely.Polygon()
    _snap_columns(out, walls)
    levels = _rebuild_ceilings(out, new, samples, segments, beams)

    if levels:
        for ln in walls + parts:
            el = elements[ln["el"]["id"]]
            top = shell._wall_top({"n": _normal(ln["azimuth"])}, np.array(el["start"]), np.array(el["end"]), levels)
            if top is not None:
                _set_param(params, ln["el"]["params"]["top"], top, top["assumed"])
        for el in out["elements"]:
            if el["kind"] in ("beam", "column"):
                _follow(params, el["params"]["top"], levels)

    diag["squaring"] = _report(walls, parts, frame)
    underlay(out)
    return out


def _report(walls: list[dict], parts: list[dict], frame: float) -> dict:
    def entry(ln):
        return {"measured_deg": float(np.degrees(ln["az"]["value"])), "retained_deg": float(np.degrees(ln["azimuth"])),
                "deviation_deg": float(np.degrees(ln["deviation"])), "tolerance_deg": float(np.degrees(ln["tol"])), "state": ln["state"],
                "sigma_deg": float(np.degrees(ln["az"]["sigma"])),
                "forced": ln["state"] == "squared" and ln["deviation"] > SQUARING_SIGMAS * ln["az"]["sigma"]}

    def angle(a, b):
        return float(np.degrees(np.arccos(np.clip(_normal(a) @ _normal(b), -1.0, 1.0))))

    corners = [{"between": [a["label"], b["label"]], "measured_deg": angle(a["az"]["value"], b["az"]["value"]),
                "retained_deg": angle(a["azimuth"], b["azimuth"])} for a, b in zip(walls[-1:] + walls[:-1], walls)]
    return {"frame_deg": float(np.degrees(_wrap_quarter(frame))), "floor_deg": SQUARING_FLOOR_DEG, "sigmas": SQUARING_SIGMAS,
            "walls": {w["label"]: entry(w) for w in walls}, "partitions": {p["label"]: entry(p) for p in parts}, "corners": corners}
