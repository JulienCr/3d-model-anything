"""Interior partitions: RoomPlan walls off the floor outline whose midpoint lies inside the floor polygon."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import shapely

from room_model import shell
from room_model.lidar import Samples
from room_model.planes import PlaneFit, azimuth_sigma, fit_plane, offset_sigma
from room_model.roomplan import Room, Wall

ASSUMED_THICKNESS = 0.10
THICKNESS_SIGMA = 0.03
FACE_MIN_AREA = 0.3  # m2 of support a face needs to count in the thickness
THICKNESS_RANGE = (0.03, 0.40)  # m, measured thicknesses outside are discarded
PARALLEL_TO_Z = np.sin(np.radians(5.0))  # |normal_z| below this: the partition runs along z


def _face_a_normal(n: np.ndarray) -> np.ndarray:
    """Unit plan normal (x, z) of face A: towards +z, or towards +x for a partition running along z."""
    h = shell._horizontal(n)
    flip = h[1] < -PARALLEL_TO_Z or (abs(h[1]) <= PARALLEL_TO_Z and h[0] < 0)
    return -h if flip else h


def _fit_face(samples: Samples, room: Room, wall: Wall, n: np.ndarray, centre: float) -> PlaneFit | None:
    """Fit of the face looking along `n` (plan), centred `centre` m from the wall's RoomPlan plane along `n`."""
    n3 = np.array([n[0], 0.0, n[1]])
    probe = replace(wall, n=n3)
    idx = shell.wall_candidates(samples, room, probe, centre=centre)
    idx = idx[samples.N[idx] @ n3 > 0.0]
    if len(idx) < shell.MIN_CANDIDATES:
        return None
    return fit_plane(samples.P[idx], n3, wall.c + centre * n3, shell.FIT_WINDOW)


def _face_line(wall: Wall, n: np.ndarray, fit_a: PlaneFit | None, fit_b: PlaneFit | None) -> tuple:
    """Face A line and thickness: (normal, offset, offset sigma, azimuth sigma, thickness, thickness sigma, source, flags)."""
    both = fit_a is not None and fit_b is not None and min(fit_a.support_m2, fit_b.support_m2) >= FACE_MIN_AREA
    if fit_a is not None:
        n_a = shell._horizontal(fit_a.normal)
        offset, sig_off, sig_az = float(n_a @ fit_a.centroid[[0, 2]]), offset_sigma(fit_a), azimuth_sigma(fit_a)
        if both:
            t = float(n_a @ (fit_a.centroid - fit_b.centroid)[[0, 2]])
            if THICKNESS_RANGE[0] < t < THICKNESS_RANGE[1]:
                return n_a, offset, sig_off, sig_az, t, float(np.hypot(sig_off, offset_sigma(fit_b))), "lidar", []
        return n_a, offset, sig_off, sig_az, ASSUMED_THICKNESS, THICKNESS_SIGMA, "assumed", ["one_face_seen"]
    if fit_b is not None:
        n_b = shell._horizontal(fit_b.normal)
        offset = float(-n_b @ fit_b.centroid[[0, 2]]) + ASSUMED_THICKNESS
        sig = float(np.hypot(offset_sigma(fit_b), THICKNESS_SIGMA))
        return -n_b, offset, sig, azimuth_sigma(fit_b), ASSUMED_THICKNESS, THICKNESS_SIGMA, "assumed", ["one_face_seen"]
    offset = float(n @ wall.c[[0, 2]]) + ASSUMED_THICKNESS / 2
    return (n, offset, shell.FALLBACK_OFFSET_SIGMA, shell.FALLBACK_AZIMUTH_SIGMA, ASSUMED_THICKNESS, THICKNESS_SIGMA,
            "assumed", ["roomplan_fallback"])


def _extent(wall: Wall, n: np.ndarray, offset: float, polygon: shapely.Polygon) -> tuple[np.ndarray, np.ndarray]:
    """RoomPlan extent projected on face A's line, clipped to the floor polygon (longest piece)."""
    ends = []
    for p in wall.edge:
        q = np.asarray(p, float)[[0, 2]]
        ends.append(q - (n @ q - offset) * n)
    clipped = polygon.intersection(shapely.LineString(ends))
    pieces = [g for g in shapely.get_parts(clipped) if g.geom_type == "LineString" and not g.is_empty]
    if not pieces:
        return ends[0], ends[1]
    best = max(pieces, key=lambda g: g.length).coords
    return np.array(best[0]), np.array(best[-1])


def find_partitions(room: Room, samples: Samples, polygon: list, levels: list[dict] | None = None) -> list[dict]:
    """Per partition `{"element": ..., "parameters": ..., "diagnostic": ...}`; `levels` are the detected ceilings."""
    floor = shapely.Polygon(polygon)
    out = []
    for wall in room.off_outline_walls:
        if not shapely.contains_xy(floor, wall.c[0], wall.c[2]):
            continue
        k = len(out) + 1
        n_seed = _face_a_normal(wall.n)
        half = ASSUMED_THICKNESS / 2
        fit_a = _fit_face(samples, room, wall, n_seed, half)
        fit_b = _fit_face(samples, room, wall, -n_seed, half)
        n, offset, sig_off, sig_az, thickness, sig_t, t_source, flags = _face_line(wall, n_seed, fit_a, fit_b)
        start, end = _extent(wall, n, offset, floor)
        top = shell._wall_top({"n": n}, start, end, levels or [])
        key = f"partition.P{k}"
        src = "roomplan" if "roomplan_fallback" in flags else "lidar"
        params = {
            f"{key}.azimuth": shell._param(np.arctan2(n[1], n[0]), sig_az, "rad", src),
            f"{key}.offset": shell._param(offset, sig_off, "m", src),
            f"{key}.thickness": shell._param(thickness, sig_t, "m", t_source),
            f"{key}.top": shell._param(top["level"], top["sigma"], "m", "lidar") if top is not None
            else shell._param(wall.c[1] + wall.top_t(0.0), 0.05, "m", "roomplan"),
        }
        element = {"id": f"Partition_P{k}", "kind": "partition", "label": f"P{k}", "roomplan_id": wall.ident,
                   "start": [float(v) for v in start], "end": [float(v) for v in end], "flags": flags,
                   "params": {name: f"{key}.{name}" for name in ("azimuth", "offset", "thickness", "top")}}
        diagnostic = {"id": element["id"], "support_a_m2": None if fit_a is None else fit_a.support_m2,
                      "support_b_m2": None if fit_b is None else fit_b.support_m2}
        out.append({"element": element, "parameters": params, "diagnostic": diagnostic})
    return out
