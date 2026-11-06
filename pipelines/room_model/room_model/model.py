"""Model stage: assemble model.json and derive the key dimensions with propagated sigmas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely

from room_model import shell
from room_model.capture import load_roomplan
from room_model.context import StageContext, output_path, require
from room_model.fmt import fr
from room_model.lidar import Samples, load_samples
from room_model.roomplan import Room, Station

SCHEMA = "room-model/1"
VERSION = "v1-scan"
SOURCE_RANK = {"assumed": 0, "roomplan": 1, "lidar": 2}
PARALLEL_COS = np.cos(np.radians(20.0))
EXTENT_MARGIN = 0.10  # m past a wall end still counted as facing it
AXES = {"x": np.array([1.0, 0.0]), "z": np.array([0.0, 1.0])}


def line_sigma(sigma_offset: float, sigma_azimuth: float, delta_s: float) -> float:
    """Sigma of a fitted wall line at distance delta_s (m) along the wall from its fit centroid."""
    return float(np.hypot(sigma_offset, sigma_azimuth * delta_s))


def mode_median(vals: np.ndarray, lo: float, hi: float, bin_w=0.005, half=0.015, min_n=8):
    """Median of the values around the histogram mode inside [lo, hi]; (value, n) or (None, n)."""
    v = vals[(vals >= lo) & (vals <= hi)]
    if len(v) < min_n:
        return None, int(len(v))
    edges = np.arange(lo, hi + bin_w, bin_w)
    h, e = np.histogram(v, bins=edges)
    j = int(np.argmax(np.convolve(h, np.ones(3), mode="same")))
    core = v[np.abs(v - (e[j] + bin_w / 2)) < half]
    if len(core) < min_n:
        return None, int(len(core))
    return float(np.median(core)), int(len(core))


def measure_station(samples: Samples, room: Room, st: Station, heights=(0.8, 1.2, 1.6), lateral=None) -> dict:
    """Local wall-to-wall distance of a station at three heights; `lateral` = expected (a, b) offsets along wall A's normal."""
    a_at, b_at = (0.0, st.d_roomplan) if lateral is None else lateral
    wa = st.wall_a
    rel = samples.P - (wa.c + wa.x * st.along)
    al, lat = rel @ wa.x, rel @ wa.n
    slab = (np.abs(al) < 0.06) & (np.abs(samples.N @ wa.n) > 0.7)
    per_h = []
    for h in heights:
        band = slab & (np.abs(samples.P[:, 1] - (room.floor_y + h)) < 0.08)
        a_pos, _ = mode_median(lat[band], a_at - 0.12, a_at + 0.12)
        b_pos, _ = mode_median(lat[band], b_at - 0.12, b_at + 0.12)
        per_h.append(None if a_pos is None or b_pos is None else b_pos - a_pos)
    vals = [v for v in per_h if v is not None]
    return {"width_by_height": per_h,
            "width_m": float(np.median(vals)) if len(vals) >= 2 else None,
            "width_spread_m": float(np.ptp(vals)) if len(vals) >= 2 else None}


def _fitted_lateral(d, st: Station) -> tuple[float, float] | None:
    """Offsets of the fitted lines of both walls along RoomPlan's normal of wall A at the station; None if a wall is unfitted."""
    if st.wall_a.label not in d.walls or st.wall_b.label not in d.walls:
        return None
    q, n = (st.wall_a.c + st.wall_a.x * st.along)[[0, 2]], st.wall_a.n[[0, 2]]
    a, b = (d.walls[w.label] for w in (st.wall_a, st.wall_b))
    return tuple(float((w["offset"] - w["normal"] @ q) / (w["normal"] @ n)) for w in (a, b))


def build_stations(room: Room, samples: Samples, d=None) -> list[Station]:
    """Candidate stations shifted (max +-0.75 m) onto wall sections the LiDAR sees as clean and consistent with the
    fitted walls `d` (RoomPlan's width when absent). A station in a door is moved out of it, or dropped."""
    stations, number = [], 0
    for wa, wb, kind, positions, (lo, hi) in room.station_candidates():
        tol = 0.07 if kind == "width" else 0.12
        for s0 in positions:
            number += 1  # a dropped candidate leaves a gap: ids keep the numbering of the unfiltered candidates
            chosen = None
            for sh in [0.0] + [sg * k * 0.15 for k in range(1, 6) for sg in (1, -1)]:
                s = s0 + sh
                if sh != 0 and not (lo + 0.3 <= s <= hi - 0.3):
                    continue
                st = room.station_at(wa, wb, s, kind)
                if st is None or st.door_labels:
                    continue
                lateral = None if d is None else _fitted_lateral(d, st)
                r = measure_station(samples, room, st, lateral=lateral)
                expected = st.d_roomplan if lateral is None else lateral[1] - lateral[0]
                if (r["width_m"] is not None and all(v is not None for v in r["width_by_height"])
                        and r["width_spread_m"] < 0.03 and abs(r["width_m"] - expected) < tol):
                    chosen = st
                    break
            if chosen is None:
                chosen = room.station_at(wa, wb, s0, kind)
                if chosen is None or chosen.door_labels:
                    continue
                chosen.flag = "obstacle_or_uneven_wall"
            chosen.sid = f"S{number:02d}"
            stations.append(chosen)
    return stations


class _Doc:
    """Read access to a shell document: parameters, wall lines with their fit sigmas."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.params = doc["parameters"]
        self.walls = {w["label"]: w for w in shell.wall_lines(doc)}
        for el in doc["elements"]:
            if el["kind"] == "wall":
                w = self.walls[el["label"]]
                w["params"] = [el["params"]["azimuth"], el["params"]["offset"]]
                w["sigma_azimuth"] = self.params[w["params"][0]]["sigma"]
                w["sigma_offset"] = self.params[w["params"][1]]["sigma"]
                w["centroid"] = np.array(doc["diagnostics"]["walls"][el["label"]]["centroid"])

    def value(self, pid: str) -> float:
        return self.params[pid]["value"]

    def sigma(self, pid: str) -> float:
        return self.params[pid]["sigma"]

    def weakest(self, pids: list[str]) -> str:
        return min((self.params[p]["source"] for p in pids), key=SOURCE_RANK.__getitem__)

    def line_sigma_at(self, label: str, point: np.ndarray) -> float:
        w = self.walls[label]
        along = (point - w["centroid"]) @ np.array([-w["normal"][1], w["normal"][0]])
        return line_sigma(w["sigma_offset"], w["sigma_azimuth"], float(along))

    def project(self, label: str, point: np.ndarray) -> np.ndarray:
        w = self.walls[label]
        return point - (w["normal"] @ point - w["offset"]) * w["normal"]


def _dimension(id_, kind, label, value, sigma, elements, at, source, flags=()) -> dict:
    return {"id": id_, "kind": kind, "label": label, "value": float(value), "sigma": float(sigma),
            "elements": elements, "at": None if at is None else [float(at[0]), float(at[1])],
            "source": source, "flags": list(flags)}


def _station_dimension(d: _Doc, st: Station) -> dict | None:
    la, lb = st.wall_a.label, st.wall_b.label
    if la not in d.walls or lb not in d.walls:
        return None
    wa, wb = d.walls[la], d.walls[lb]
    p = d.project(la, (st.wall_a.c + st.wall_a.x * st.along)[[0, 2]])
    denom = wb["normal"] @ wa["normal"]
    t = (wb["offset"] - wb["normal"] @ p) / denom
    q = p + t * wa["normal"]
    sigma = np.hypot(d.line_sigma_at(la, p), d.line_sigma_at(lb, q))
    word = "Largeur" if st.kind == "width" else "Longueur"
    label = f"{word} {la}–{lb} à {fr(st.corner_dist, 2)} m de {st.corner_ref}"
    flags = ["obstacle_or_uneven_wall"] if st.flag else []
    return _dimension(st.sid, st.kind, label, abs(t), sigma, [f"Wall_{la}", f"Wall_{lb}"], (p + q) / 2,
                      d.weakest(wa["params"] + wb["params"]), flags)


def _height_dimensions(d: _Doc) -> list[dict]:
    floor = "floor.level"
    out = []
    for el in d.doc["elements"]:
        if el["kind"] != "ceiling":
            continue
        k = el["id"].split("_")[1]
        pid = el["params"]["level"]
        region = shapely.MultiPolygon([shapely.Polygon(r["exterior"], r["holes"]) for r in el["region"]])
        centre = region.representative_point()
        out.append(_dimension(f"H{k}", "ceiling_height", f"Hauteur sous plafond C{k}", d.value(pid) - d.value(floor),
                              np.hypot(d.sigma(pid), d.sigma(floor)), [el["id"], "Floor"], (centre.x, centre.y),
                              d.weakest([pid, floor])))
    return out


def _beam_dimensions(d: _Doc) -> list[dict]:
    out = []
    for el in d.doc["elements"]:
        if el["kind"] != "beam":
            continue
        k = el["id"].split("_")[1]
        under, width = el["params"]["underside"], el["params"]["width"]
        mid = np.mean(el["axis"], axis=0)
        out.append(_dimension(f"B{k}-u", "beam_underside", f"Hauteur sous poutre B{k}", d.value(under) - d.value("floor.level"),
                              np.hypot(d.sigma(under), d.sigma("floor.level")), [el["id"], "Floor"], mid,
                              d.weakest([under, "floor.level"]), el.get("flags", [])))
        out.append(_dimension(f"B{k}-w", "beam_width", f"Largeur de la poutre B{k}", d.value(width), d.sigma(width),
                              [el["id"]], mid, d.weakest([width]), el.get("flags", [])))
    return out


def _column_offset(d: _Doc, axis: str, centre: np.ndarray, half: float, sig_c: float, sig_h: float):
    """Nearest face-to-wall distance along an axis: (value, sigma, wall label, face point) or None."""
    e = AXES[axis]
    best = None
    for pass_extent in (True, False):
        for label, w in d.walls.items():
            if abs(w["normal"] @ e) < PARALLEL_COS:
                continue
            for sign in (1.0, -1.0):
                face = centre + sign * half * e
                dist = w["normal"] @ face - w["offset"]
                if dist < -0.05:
                    continue
                tangent = w["end"] - w["start"]
                u = ((face - w["start"]) @ tangent) / (tangent @ tangent)
                margin = EXTENT_MARGIN / np.linalg.norm(tangent)
                if pass_extent and not -margin <= u <= 1 + margin:
                    continue
                if best is None or abs(dist) < best[0]:
                    sigma = np.hypot(np.hypot(sig_c, sig_h), d.line_sigma_at(label, face))
                    best = (abs(dist), sigma, label)
        if best:
            return best
    return None


def _column_dimensions(d: _Doc) -> list[dict]:
    out = []
    engaged = {c["id"]: c["engaged_walls"] for c in d.doc["diagnostics"].get("columns", [])}
    for el in d.doc["elements"]:
        if el["kind"] != "column":
            continue
        k = el["id"].split("_")[1]
        p = el["params"]
        centre = np.array([d.value(p["center_x"]), d.value(p["center_z"])])
        for axis in ("x", "z"):
            size = p[f"size_{axis}"]
            out.append(_dimension(f"COL{k}-{axis}", "column_size", f"Colonne {k} : dimension selon {axis}", d.value(size),
                                  d.sigma(size), [el["id"]], centre, d.weakest([size])))
        for axis in ("x", "z"):
            size, c = p[f"size_{axis}"], p[f"center_{axis}"]
            hit = _column_offset(d, axis, centre, d.value(size) / 2, d.sigma(c), d.sigma(size) / 2)
            if hit is None:
                continue
            value, sigma, label = hit
            if label in engaged.get(el["id"], [el["engaged_wall"]]):
                continue
            wall = d.walls[label]
            out.append(_dimension(f"COL{k}-d{axis}", "column_offset", f"Colonne {k} : écart au mur {label} selon {axis}", value,
                                  sigma, [el["id"], f"Wall_{label}"], centre, d.weakest([size, c] + wall["params"])))
    return out


def build(ctx: StageContext, doc: dict) -> dict:
    """Model document: the shell document plus schema/version and the derived dimensions."""
    w2r = np.array(doc["frame"]["world_to_room"], float).reshape(4, 4).T
    room = Room(load_roomplan(ctx.capture, ctx.room), w2r)
    samples = load_samples(ctx, w2r)
    d = _Doc(doc)
    dimensions = [x for x in (_station_dimension(d, st) for st in build_stations(room, samples, d)) if x]
    dimensions += _height_dimensions(d) + _beam_dimensions(d) + _column_dimensions(d)
    return {**doc, "schema": SCHEMA, "version": VERSION, "dimensions": dimensions}


def run(ctx: StageContext) -> Path:
    doc = json.loads(require(ctx, "shell").read_text(encoding="utf-8"))
    path = output_path(ctx, "model")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(ctx, doc), indent=2), encoding="utf-8")
    return path
