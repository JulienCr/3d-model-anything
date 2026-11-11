"""Openings stage: RoomPlan doors, windows and openings placed on the model's walls and partitions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from room_model import columns
from room_model.capture import colmajor, load_roomplan
from room_model.context import StageContext, output_path, require
from room_model.lidar import Samples, load_samples

SCHEMA = "room-model/openings-1"
ROOMPLAN_SIGMA = 0.03
HOST_DISTANCE = 0.30  # m, an opening this far from the slab of a host (and no further) still belongs to it
HOST_PARALLEL = np.cos(np.radians(10.0))
CLAMP_TOLERANCE = 0.005  # m an opening may overhang its host before it counts as clamped
MIN_SPAN = 0.10  # m of an opening that must overlap its host, else it is an orphan
BIN = 0.01
EDGE_RANGE = 0.15  # m searched each side of a RoomPlan edge
EDGE_BAND_INSET = 0.10  # m kept clear of the opening corners when counting along an edge
SLAB_MARGIN = 0.04  # m around a host face that still counts as wall surface
KINDS = (("doors", "door", "Door"), ("windows", "window", "Window"), ("openings", "opening", "Opening"))


def hosts_from_model(doc: dict) -> list[dict]:
    """Walls and partitions of a model document as plan slabs: unit normal, face offset, thickness, start, end."""
    params = doc["parameters"]
    out = []
    for el in doc["elements"]:
        if el["kind"] not in ("wall", "partition"):
            continue
        az = params[el["params"]["azimuth"]]["value"]
        out.append({"id": el["id"], "kind": el["kind"], "roomplan_id": el.get("roomplan_id"), "start": el["start"],
                    "end": el["end"], "normal": [float(np.cos(az)), float(np.sin(az))],
                    "offset": params[el["params"]["offset"]]["value"], "thickness": params[el["params"]["thickness"]]["value"]})
    return out


def _slab_distance(host: dict, xz: np.ndarray) -> float:
    """Signed gap (m) from a plan point to the host slab: 0 inside, + on the normal side, - behind."""
    q = float(np.asarray(host["normal"]) @ xz - host["offset"])
    return q if q > 0 else (q + host["thickness"] if q < -host["thickness"] else 0.0)


def _along(host: dict, xz: np.ndarray) -> tuple[float | np.ndarray, float]:
    """Coordinate(s) of plan point(s) (last axis x, z) along start to end, and the host length."""
    start, end = np.asarray(host["start"], float), np.asarray(host["end"], float)
    length = float(np.linalg.norm(end - start))
    return (xz - start) @ (end - start) / length, length


def host_of(opening_tf: np.ndarray, rp_parent: str | None, hosts: list[dict]) -> tuple[str, float] | None:
    """(host id, signed slab distance) of an opening: its RoomPlan parent, else the nearest parallel host within 0.30 m."""
    xz = np.asarray(opening_tf, float)[[0, 2], 3]
    for host in hosts:
        if rp_parent is not None and host.get("roomplan_id") == rp_parent:
            return host["id"], _slab_distance(host, xz)
    x_axis = np.asarray(opening_tf, float)[[0, 2], 0]
    best = None
    for host in hosts:
        direction = np.subtract(host["end"], host["start"])
        if abs(x_axis @ direction) / (np.linalg.norm(x_axis) * np.linalg.norm(direction) + 1e-12) < HOST_PARALLEL:
            continue
        dist = _slab_distance(host, xz)
        s, length = _along(host, xz)
        if abs(dist) <= HOST_DISTANCE and -HOST_DISTANCE <= s <= length + HOST_DISTANCE and (best is None or abs(dist) < abs(best[1])):
            best = (host["id"], dist)
    return best


def _face(host: dict, tf: np.ndarray) -> str:
    """Partition face (A = along the normal) the opening's +z axis points to; the nearest face when it lies in the plane."""
    z_axis = np.asarray(tf, float)[[0, 2], 2]
    n = np.asarray(host["normal"])
    along_normal = float(z_axis @ n) / (np.linalg.norm(z_axis) + 1e-12)
    if abs(along_normal) < 0.5:
        q = float(n @ np.asarray(tf, float)[[0, 2], 3]) - host["offset"]
        return "A" if abs(q) <= abs(q + host["thickness"]) else "B"
    return "A" if along_normal > 0 else "B"


def _wall_samples(samples: Samples, host: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(s, y, rms) of the LiDAR samples lying on the host faces, s metres from `start`."""
    n, start = np.asarray(host["normal"]), np.asarray(host["start"], float)
    xz = samples.P[:, [0, 2]]
    q = xz @ n - host["offset"]
    near = np.abs(q) < SLAB_MARGIN
    if host["kind"] == "partition":
        near |= np.abs(q + host["thickness"]) < SLAB_MARGIN
    s, length = _along(host, xz)
    ok = near & (s > -0.01) & (s < length + 0.01) & (np.abs(samples.N[:, [0, 2]] @ n) > 0.75) & ~np.isin(samples.cls, columns.EXCLUDED_CLASSES)
    q_ok = q[ok]
    resid = np.minimum(np.abs(q_ok), np.abs(q_ok + host["thickness"])) if host["kind"] == "partition" else q_ok
    return s[ok], samples.P[ok, 1], float(np.sqrt(np.mean(resid**2))) if ok.any() else 0.0


def _step(values: np.ndarray, centre: float, wall_below: bool) -> float | None:
    """Coordinate where wall support (1 cm bins within EDGE_RANGE of `centre`) switches to none; None if not clean."""
    lo = centre - EDGE_RANGE
    counts = np.histogram(values, bins=np.arange(lo, centre + EDGE_RANGE + BIN / 2, BIN))[0].astype(float)
    ref = np.percentile(counts, 90)
    if ref <= 0:
        return None
    wall = counts > 0.25 * ref
    if not wall_below:
        wall = wall[::-1]
    best, score = None, -1
    for k in range(3, len(wall) - 2):
        left, right = wall[:k].mean(), (~wall[k:]).mean()
        if left >= 0.9 and right >= 0.9 and left + right > score:
            best, score = k, left + right
    if best is None:
        return None
    return float(lo + (best if wall_below else len(wall) - best) * BIN)


def _refine(samples: Samples, host: dict, c: float, w: float, sill: float, h: float, floor: float, is_door: bool):
    """Edges (left, right, bottom, top) found in the LiDAR, bottom None for a door; None when one edge is missing."""
    s, y, rms = _wall_samples(samples, host)
    y = y - floor
    left, right, top = c - w / 2, c + w / 2, sill + h
    in_y = (y > sill + EDGE_BAND_INSET) & (y < top - EDGE_BAND_INSET)
    in_s = (s > left + EDGE_BAND_INSET) & (s < right - EDGE_BAND_INSET)
    edges = (_step(s[in_y], left, True), _step(s[in_y], right, False), None if is_door else _step(y[in_s], sill, True),
             _step(y[in_s], top, False))
    if any(e is None for i, e in enumerate(edges) if not (is_door and i == 2)):
        return None
    return edges, float(np.hypot(BIN / np.sqrt(12), rms))


def _param(value: float, sigma: float, source: str) -> dict:
    return {"value": float(value), "sigma": float(sigma), "unit": "m", "source": source}


def build(ctx: StageContext, model_doc: dict) -> dict:
    """The openings document for the context's capture and its model document."""
    w2r = np.array(model_doc["frame"]["world_to_room"], float).reshape(4, 4).T
    data = load_roomplan(ctx.capture, ctx.room)
    hosts = hosts_from_model(model_doc)
    by_id = {h["id"]: h for h in hosts}
    floor = model_doc["parameters"]["floor.level"]
    samples = load_samples(ctx, w2r)
    parameters, out, orphans, clamped = {}, [], [], []
    for key, kind, prefix in KINDS:
        for k, raw in enumerate(data[key], 1):
            tf = w2r @ colmajor(raw["transform"], 4)
            ident = f"{prefix}_{k}"
            width, height = float(raw["dimensions"][0]), float(raw["dimensions"][1])
            sill = max(float(tf[1, 3]) - height / 2 - floor["value"], 0.0)
            vals = {"width": _param(width, ROOMPLAN_SIGMA, "roomplan"), "height": _param(height, ROOMPLAN_SIGMA, "roomplan"),
                    "sill": _param(sill, ROOMPLAN_SIGMA, "roomplan")}
            flags, found = [], host_of(tf, raw.get("parentIdentifier"), hosts)
            host = by_id[found[0]] if found else None
            if host is not None:
                s, length = _along(host, tf[[0, 2], 3])
                s = float(s)
                lo, hi = max(s - width / 2, 0.0), min(s + width / 2, length)
                if hi - lo < MIN_SPAN:
                    host = None
            if host is None:
                flags.append("orphan")
                orphans.append(ident)
            else:
                if s - width / 2 < -CLAMP_TOLERANCE or s + width / 2 > length + CLAMP_TOLERANCE:
                    flags.append("clamped")
                    clamped.append(ident)
                    s, width = (lo + hi) / 2, hi - lo
                    vals["width"] = _param(width, ROOMPLAN_SIGMA, "roomplan")
                vals["center_s"] = _param(s, ROOMPLAN_SIGMA, "roomplan")
                thickness = model_doc["parameters"][next(e for e in model_doc["elements"] if e["id"] == host["id"])["params"]["thickness"]]
                vals["depth"] = {**thickness}
                refined = None if "clamped" in flags else _refine(samples, host, s, width, sill, height, floor["value"], kind == "door")
                if refined is not None:
                    (left, right, bottom, top), sigma = refined
                    bottom_value, bottom_sigma, bottom_source = (0.0, floor["sigma"], "lidar") if bottom is None else (bottom, sigma, "lidar")
                    vals["center_s"] = _param((left + right) / 2, sigma / np.sqrt(2), "lidar")
                    vals["width"] = _param(right - left, sigma * np.sqrt(2), "lidar")
                    vals["sill"] = _param(bottom_value, bottom_sigma, bottom_source)
                    vals["height"] = _param(top - bottom_value, np.hypot(sigma, bottom_sigma), "lidar")
            for name, p in vals.items():
                parameters[f"opening.{ident}.{name}"] = p
            xz = tf[[0, 2], 3]
            out.append({"id": ident, "kind": kind, "roomplan_id": raw["identifier"], "host": host["id"] if host else None,
                        "face": _face(host, tf) if host and host["kind"] == "partition" else None,
                        "center": [float(xz[0]), float(xz[1])], "flags": flags,
                        "params": {name: f"opening.{ident}.{name}" for name in vals}})
    return {"schema": SCHEMA, "parameters": parameters, "openings": out,
            "diagnostics": {"orphans": orphans, "clamped": clamped, "orphan_doors": model_doc["diagnostics"].get("orphan_doors", [])}}


def run(ctx: StageContext) -> Path:
    model_doc = json.loads(require(ctx, "model").read_text(encoding="utf-8"))
    path = output_path(ctx, "openings")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(ctx, model_doc), indent=2), encoding="utf-8")
    return path
