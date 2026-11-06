"""Shell stage: refit RoomPlan walls and floor on LiDAR and close the floor polygon."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely

from room_model import ceilings
from room_model.capture import load_roomplan
from room_model.context import StageContext, output_path
from room_model.frame import world_to_room
from room_model.lidar import Samples, load_samples
from room_model.planes import PlaneFit, azimuth_sigma, fit_plane, offset_sigma
from room_model.roomplan import Room, Wall

SCHEMA = "room-model/shell-1"
FIT_WINDOW = 0.08  # m around the seed plane
MIN_CANDIDATES = 200
LOW_SUPPORT_M2 = 0.5
NEAR_PARALLEL_DEG = 10.0
THICKNESS = 0.20
FALLBACK_OFFSET_SIGMA = 0.05
WALL_TOP_STEP = 0.10  # m between the points sampling a wall top
WALL_TOP_INSET = 0.15  # m from the wall into the room
FALLBACK_AZIMUTH_SIGMA = np.radians(2.0)
EXCLUDED_CLASSES = (4, 5, 6, 7)  # table, seat, window, door


def wall_candidates(s: Samples, room: Room, wall: Wall, margin_s: float = 0.10, margin_t: float = 0.20) -> np.ndarray:
    """Indices of samples on the wall rectangle (doors excluded) within the window of the RoomPlan plane."""
    along, up, dist = wall.local(s.P)
    t_floor = room.floor_y - wall.c[1]
    ok = (np.abs(dist) < FIT_WINDOW) & (np.abs(along) < wall.w / 2 - margin_s) & (up > t_floor + margin_t)
    ok &= (np.abs(s.N @ wall.n) > 0.75) & ~np.isin(s.cls, EXCLUDED_CLASSES)
    idx = np.nonzero(ok)[0]
    keep = up[idx] < wall.top_t(np.clip(along[idx], -wall.w / 2, wall.w / 2)) - margin_t
    for dr in wall.doors:
        keep &= ~((np.abs(along[idx] - dr.s) < dr.w / 2 + 0.08) & (up[idx] < dr.t + dr.h / 2 + 0.08))
    return idx[keep]


def _param(value: float, sigma: float, unit: str, source: str) -> dict:
    return {"value": float(value), "sigma": float(sigma), "unit": unit, "source": source}


def _horizontal(n: np.ndarray) -> np.ndarray:
    h = np.asarray(n, float)[[0, 2]]
    return h / np.linalg.norm(h)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def _fit_wall(samples: Samples, room: Room, wall: Wall) -> PlaneFit | None:
    idx = wall_candidates(samples, room, wall)
    if len(idx) < MIN_CANDIDATES:
        return None
    return fit_plane(samples.P[idx], wall.n, wall.c, FIT_WINDOW)


def _wall_lines(room: Room, samples: Samples) -> tuple[list[dict], dict, dict]:
    """Per wall: plan line (normal, offset), parameters, and diagnostics."""
    lines, params, diag = [], {}, {}
    for wall in room.walls:
        seed_h = _horizontal(wall.n)
        fit = _fit_wall(samples, room, wall)
        flags = []
        if fit is None:
            n_h, offset, source = seed_h, float(seed_h @ wall.c[[0, 2]]), "roomplan"
            sig_off, sig_az = FALLBACK_OFFSET_SIGMA, FALLBACK_AZIMUTH_SIGMA
            flags.append("roomplan_fallback")
            diag[wall.label] = {
                "rms_cm": None, "support_m2": None, "offset_vs_roomplan_cm": 0.0, "angle_vs_roomplan_deg": 0.0,
                "tilt_deg": None, "n_points": 0, "centroid": [float(wall.c[0]), float(wall.c[2])],
            }
        else:
            n_h = _horizontal(fit.normal)
            offset, source = float(n_h @ fit.point[[0, 2]]), "lidar"
            sig_off, sig_az = offset_sigma(fit), azimuth_sigma(fit)
            if fit.support_m2 < LOW_SUPPORT_M2:
                flags.append("low_support")
            diag[wall.label] = {
                "rms_cm": fit.rms * 100, "support_m2": fit.support_m2,
                "offset_vs_roomplan_cm": float((fit.point - wall.c) @ wall.n * 100),
                "angle_vs_roomplan_deg": _angle_deg(n_h, seed_h),
                "tilt_deg": float(np.degrees(np.arcsin(min(abs(fit.normal[1]), 1.0)))),
                "n_points": fit.n_inliers, "centroid": [float(fit.centroid[0]), float(fit.centroid[2])],
            }
        if wall.ident in room.seeded_from_outline:
            flags.append("seeded_from_outline")
        key = f"wall.{wall.label}"
        params[f"{key}.azimuth"] = _param(np.arctan2(n_h[1], n_h[0]), sig_az, "rad", source)
        params[f"{key}.offset"] = _param(offset, sig_off, "m", source)
        params[f"{key}.thickness"] = _param(THICKNESS, 0.05, "m", "assumed")
        params[f"{key}.top"] = _param(wall.c[1] + wall.top_t(0.0), 0.05, "m", "roomplan")
        lines.append({"wall": wall, "n": n_h, "offset": offset, "flags": flags})
    return lines, params, diag


def _corners(room: Room, lines: list[dict]) -> list[np.ndarray]:
    """Plan corner k joins wall k-1 and wall k; flags the walls around a near-parallel one."""
    sin_limit = np.sin(np.radians(NEAR_PARALLEL_DEG))
    out = []
    for k in range(len(lines)):
        a, b = lines[k - 1], lines[k]
        cross = a["n"][0] * b["n"][1] - a["n"][1] * b["n"][0]
        if abs(cross) > sin_limit:
            out.append(np.linalg.solve(np.stack([a["n"], b["n"]]), [a["offset"], b["offset"]]))
            continue
        rp = room.outline[k][[0, 2]]
        pa = rp - (a["n"] @ rp - a["offset"]) * a["n"]
        pb = rp - (b["n"] @ rp - b["offset"]) * b["n"]
        out.append((pa + pb) / 2)
        a["flags"].append("near_parallel_corner")
        b["flags"].append("near_parallel_corner")
    return out


def _fit_floor(samples: Samples, room: Room) -> PlaneFit | None:
    P = samples.P
    ok = (samples.N[:, 1] > 0.9) & ~np.isin(samples.cls, (4, 5)) & (np.abs(P[:, 1] - room.floor_y) < FIT_WINDOW)
    idx = np.nonzero(ok)[0]
    idx = idx[room.footprint_signed_margin(P[idx][:, [0, 2]]) > 0.12]
    if len(idx) < MIN_CANDIDATES:
        return None
    return fit_plane(P[idx], np.array([0.0, 1.0, 0.0]), np.array([0.0, room.floor_y, 0.0]), FIT_WINDOW)


def _floor(room: Room, samples: Samples, polygon: list) -> tuple[dict, dict, list]:
    fit = _fit_floor(samples, room)
    if fit is None:
        level = _param(room.floor_y, FALLBACK_OFFSET_SIGMA, "m", "roomplan")
        return level, {"source": "roomplan"}, ["floor:roomplan_fallback"]
    c = shapely.Polygon(polygon).centroid
    n, p = fit.normal, fit.point
    y = p[1] - (n[0] * (c.x - p[0]) + n[2] * (c.y - p[2])) / n[1]
    diag = {"rms_cm": fit.rms * 100, "support_m2": fit.support_m2, "n_points": fit.n_inliers,
            "tilt_deg": float(np.degrees(np.arccos(min(n[1], 1.0)))),
            "offset_vs_roomplan_cm": float((y - room.floor_y) * 100)}
    return _param(y, offset_sigma(fit), "m", "lidar"), diag, []


def _region_json(region: shapely.MultiPolygon) -> list[dict]:
    def ring(r):
        return [[float(x), float(z)] for x, z in r.coords[:-1]]

    return [{"exterior": ring(g.exterior), "holes": [ring(h) for h in g.interiors]} for g in region.geoms]


def _wall_top(line: dict, start: np.ndarray, end: np.ndarray, levels: list[dict]) -> dict | None:
    """Highest ceiling level over points every WALL_TOP_STEP along the wall, WALL_TOP_INSET inside the room."""
    length = float(np.linalg.norm(end - start))
    count = max(int(length / WALL_TOP_STEP), 1)
    t = (np.arange(count) + 0.5) / count
    pts = start + np.outer(t, end - start) + WALL_TOP_INSET * line["n"]
    hit = [lv for lv in levels if shapely.contains_xy(lv["region"], pts[:, 0], pts[:, 1]).any()]
    return max(hit, key=lambda lv: lv["level"]) if hit else None


def _ceiling_elements(result: ceilings.CeilingResult) -> tuple[list[dict], dict, list[dict]]:
    elements, params, diag = [], {}, []
    for k, lv in enumerate(result.levels, 1):
        params[f"ceiling.{k}.level"] = _param(lv["level"], lv["sigma"], "m", "lidar")
        elements.append({"id": f"Ceiling_{k}", "kind": "ceiling", "params": {"level": f"ceiling.{k}.level"},
                         "region": _region_json(lv["region"])})
        diag.append({"id": f"Ceiling_{k}", "support_m2": lv["support_m2"], "area_m2": float(lv["region"].area)})
    for k, b in enumerate(result.beams, 1):
        for name in ("underside", "width", "top"):
            params[f"beam.{k}.{name}"] = _param(b[name], b[f"{name}_sigma"], "m", "lidar")
        elements.append({"id": f"Beam_{k}", "kind": "beam", "axis": b["axis"],
                         "params": {n: f"beam.{k}.{n}" for n in ("underside", "width", "top")}})
    return elements, params, diag


def build(ctx: StageContext) -> dict:
    """The shell document (shell-1 format) for the context's capture; Tasks adding elements extend this dict."""
    w2r = world_to_room(ctx.capture, ctx.room, ctx.turn)
    room = Room(load_roomplan(ctx.capture, ctx.room), w2r)
    samples = load_samples(ctx, w2r)
    lines, parameters, walls_diag = _wall_lines(room, samples)
    corners = _corners(room, lines)
    polygon = [[float(c[0]), float(c[1])] for c in corners]
    level, floor_diag, floor_flags = _floor(room, samples, polygon)
    parameters["floor.level"] = level
    ceiling_result = ceilings.detect(samples, shapely.Polygon(polygon), level["value"])

    elements = [{"id": "Floor", "kind": "floor", "params": {"level": "floor.level"}, "polygon": polygon}]
    flags = list(floor_flags)
    for k, line in enumerate(lines):
        wall = line["wall"]
        key = f"wall.{wall.label}"
        elements.append({
            "id": f"Wall_{wall.label}", "kind": "wall", "label": wall.label, "roomplan_id": wall.ident,
            "start": polygon[k], "end": polygon[(k + 1) % len(polygon)], "flags": line["flags"],
            "params": {name: f"{key}.{name}" for name in ("azimuth", "offset", "thickness", "top")},
        })
        flags += [f"{wall.label}:{f}" for f in line["flags"]]
        top = _wall_top(line, np.array(polygon[k]), np.array(polygon[(k + 1) % len(polygon)]), ceiling_result.levels)
        if top is not None:
            parameters[f"{key}.top"] = _param(top["level"], top["sigma"], "m", "lidar")
    ceiling_elements, ceiling_params, ceiling_diag = _ceiling_elements(ceiling_result)
    elements += ceiling_elements
    parameters.update(ceiling_params)
    return {
        "schema": SCHEMA,
        "source": {"capture": Path(ctx.capture).name, "room": ctx.room},
        "frame": {"world_to_room": [float(v) for v in w2r.T.ravel()], "turn": ctx.turn},
        "parameters": parameters,
        "elements": elements,
        "diagnostics": {"walls": walls_diag, "floor": floor_diag, "ceilings": ceiling_diag, "ceiling_unobserved_fraction": ceiling_result.unobserved_fraction, "unexplained_vertical": [],
                        "off_outline_walls": list(room.off_outline), "flags": flags},
    }


def run(ctx: StageContext) -> Path:
    doc = build(ctx)
    path = output_path(ctx, "shell")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def wall_lines(doc: dict) -> list[dict]:
    """Plan line of each wall: label, unit interior normal (x, z), offset (n . p = offset), start, end."""
    params = doc["parameters"]
    out = []
    for el in doc["elements"]:
        if el["kind"] != "wall":
            continue
        az = params[el["params"]["azimuth"]]["value"]
        out.append({"label": el["label"], "normal": np.array([np.cos(az), np.sin(az)]),
                    "offset": params[el["params"]["offset"]]["value"],
                    "start": np.array(el["start"]), "end": np.array(el["end"])})
    return out
