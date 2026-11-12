"""Ceiling levels, their plan regions, and beams, from the upward-facing underside of the LiDAR mesh."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage

from room_model.lidar import SAMPLE_DENSITY, Samples
from room_model.planes import fit_plane, offset_sigma

MIN_HEIGHT = 1.6  # m above the floor
POLYGON_MARGIN = 0.05
LEVEL_WINDOW = 0.03
MIN_LEVEL_AREA = 0.3  # m2
MAX_LEVELS = 8
MAX_MERGE_SPAN = 0.04  # m, total height spread of one merged level: a chain of small steps must not walk across a real step
MERGE_SIGMAS = 3.0  # neighbouring levels closer than this many sigmas (the larger one) are one rough surface
GRID = 0.02  # m, plan raster cell
CLOSING_ITERATIONS = 2
MIN_COMPONENT_AREA = 0.1  # m2
BEAM_MINOR = (0.15, 0.80)
BEAM_RATIO = 2.0
ADJACENT = 0.15  # m, reach of the "higher level next to a long side" test
TOP_REACH = 0.30  # m, band beside the long sides in which a higher level's area ranks it as the beam top
MIN_ADJACENT_CELLS = 25
FACE_NORMAL_Y = 0.65  # a side face is any surface steeper than this: meshes bevel the edge of a beam
MIN_BEAM_DROP = 0.14  # m between the underside and the level it hangs from: shallower reliefs are not beams
REASONS = {"suspended": "suspended", "narrow": "too_narrow", "shallow": "too_shallow"}
SEEN_ABOVE = 0.30  # share of a strip's cells with ceiling samples above them: it hangs below a visible ceiling
MIN_FACE_AREA = 0.05  # m2 of vertical samples a beam side face needs
FACE_TOP_TOLERANCE = 0.05  # m, between the top of a side face and the higher level next to it
WALL_SIDE = 0.06  # m, a long side this close to a wall needs no face of its own
SIDE_GROW = 0.10
MIN_SIDE_AREA = 0.05  # m2 of beam side face needed to measure the width
FALLBACK_WIDTH_SIGMA = 0.02
WIDTH_AGREEMENT = 0.10  # m, a side-plane width further than this from the footprint is not trusted
SIMPLIFY = 0.02
BEAM_SNAP = 0.10  # m, an end this close to the floor outline is extended onto it


@dataclass
class CeilingResult:
    levels: list[dict]  # {"level", "sigma", "region": MultiPolygon, "support_m2"}, highest first
    beams: list[dict]  # {"axis", "underside", "underside_sigma", "width", "width_sigma", "top", "top_sigma"}
    unobserved_fraction: float
    unexplained: list[dict]  # {"bbox", "underside", "width", "length", "reason"}: strips that are not beams


def _histogram_levels(y: np.ndarray) -> list[float]:
    """Dominant heights, greedy over a 1 cm histogram."""
    levels = []
    for _ in range(MAX_LEVELS):
        if len(y) < 100:
            break
        bins = np.arange(y.min() - 0.01, y.max() + 0.02, 0.01)
        h, e = np.histogram(y, bins=bins)
        k = np.convolve(h, np.ones(2 * round(LEVEL_WINDOW / 0.01) + 1), mode="same")
        j = int(np.argmax(k))
        yc = float(np.median(y[np.abs(y - (e[j] + 0.005)) < LEVEL_WINDOW]))
        if (np.abs(y - yc) < LEVEL_WINDOW).sum() / SAMPLE_DENSITY < MIN_LEVEL_AREA:
            break
        levels.append(yc)
        y = y[np.abs(y - yc) > LEVEL_WINDOW]
    return levels


class _Grid:
    def __init__(self, polygon: shapely.Polygon):
        x0, z0, x1, z1 = polygon.bounds
        pad = 6 * GRID
        self.x0, self.z0 = x0 - pad, z0 - pad
        self.shape = (int(np.ceil((x1 - x0 + 2 * pad) / GRID)) + 1, int(np.ceil((z1 - z0 + 2 * pad) / GRID)) + 1)

    def occupancy(self, xz: np.ndarray) -> np.ndarray:
        occ = np.zeros(self.shape, bool)
        ij = np.floor((xz - [self.x0, self.z0]) / GRID).astype(int)
        occ[ij[:, 0], ij[:, 1]] = True
        return ndimage.binary_closing(occ, iterations=CLOSING_ITERATIONS)

    def centres(self, mask: np.ndarray) -> np.ndarray:
        ij = np.argwhere(mask)
        return (ij + 0.5) * GRID + [self.x0, self.z0]

    def polygon(self, mask: np.ndarray) -> shapely.MultiPolygon:
        """Union of the mask's cells, merged along z into runs first to keep the union cheap."""
        padded = np.pad(mask, ((0, 0), (1, 1)))
        edges = np.diff(padded.astype(np.int8), axis=1)
        (ri, rs), (_, re) = np.nonzero(edges == 1), np.nonzero(edges == -1)
        x0, x1 = self.x0 + ri * GRID, self.x0 + (ri + 1) * GRID  # same expression per edge: abutting boxes must share bits
        geom = shapely.union_all(shapely.box(x0, self.z0 + rs * GRID, x1, self.z0 + re * GRID))
        return _as_multipolygon(geom)


def _as_multipolygon(geom) -> shapely.MultiPolygon:
    parts = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, shapely.Polygon) and not g.is_empty]
    return shapely.MultiPolygon(parts)


def _component_frame(centres: np.ndarray) -> dict:
    """Principal axes of a component and its 2nd-98th percentile spans along them."""
    centre = centres.mean(0)
    _, vecs = np.linalg.eigh(np.cov((centres - centre).T))
    minor_axis, major_axis = vecs[:, 0], vecs[:, 1]
    a, b = (centres - centre) @ major_axis, (centres - centre) @ minor_axis
    a_lo, a_hi = np.percentile(a, [2, 98])
    b_lo, b_hi = np.percentile(b, [2, 98])
    tail = (a_hi - a_lo) * 0.02 / 0.96  # the trimmed 2 % on each end of an evenly filled strip
    return {"centre": centre, "u": major_axis, "v": minor_axis, "a": (a_lo, a_hi), "b": (b_lo, b_hi),
            "extent": (float(a_lo - tail - GRID / 2), float(a_hi + tail + GRID / 2)),
            "cells": centres, "major": a_hi - a_lo, "minor": b_hi - b_lo}


def _band(frame: dict, cells: np.ndarray, sign: float, reach: float) -> int:
    """Number of cells within `reach` of the long side of the component on the `sign` side."""
    rel = cells - frame["centre"]
    ra, rb = rel @ frame["u"], rel @ frame["v"]
    edge = frame["b"][1] if sign > 0 else frame["b"][0]
    along = (ra >= frame["a"][0]) & (ra <= frame["a"][1])
    return int((along & (sign * (rb - edge) >= 0) & (sign * (rb - edge) <= reach)).sum())


def _is_strip(frame: dict) -> bool:
    return frame["minor"] <= BEAM_MINOR[1] and frame["major"] / max(frame["minor"], GRID) >= BEAM_RATIO


def _side_evidence(samples: Samples, polygon: shapely.Polygon, frame: dict, underside: float,
                   higher: list[tuple[float, np.ndarray]], sign: float) -> dict:
    """What stands beside one long side: vertical face area and top, neighbouring higher levels, wall distance."""
    P = samples.P
    rel = P[:, [0, 2]] - frame["centre"]
    ra, rb = rel @ frame["u"], rel @ frame["v"]
    edge = frame["b"][1] if sign > 0 else frame["b"][0]
    mid = (frame["b"][0] + frame["b"][1]) / 2
    face = (np.abs(samples.N[:, 1]) < FACE_NORMAL_Y) & (P[:, 1] > underside + 0.03) & (ra >= frame["a"][0]) & (ra <= frame["a"][1])
    face &= (sign * (rb - mid) > 0) & (sign * (rb - edge) < SIDE_GROW)
    rel_c = frame["cells"] - frame["centre"]
    span = frame["a"][1] - frame["a"][0]
    along_c = rel_c @ frame["u"]  # the ends of a strip often touch a wall, its long side does not
    middle = (along_c > frame["a"][0] + 0.15 * span) & (along_c < frame["a"][1] - 0.15 * span)
    near = frame["cells"][middle & (sign * (rel_c @ frame["v"] - edge) > -0.04)]
    wall_gap = float(shapely.distance(shapely.points(near), polygon.exterior).min()) - GRID / 2 if len(near) else np.inf
    return {"area": float(face.sum() / SAMPLE_DENSITY), "top": float(np.percentile(P[face, 1], 95)) if face.any() else None,
            "levels": {lv: _band(frame, cells, sign, ADJACENT) for lv, cells in higher},
            "reach": {lv: _band(frame, cells, sign, TOP_REACH) for lv, cells in higher},
            "wall": wall_gap <= WALL_SIDE}


def _confirmed_tops(sides: list[dict]) -> set[float]:
    """Higher levels matched by the side faces when both long sides have a face (or one a wall); empty otherwise."""
    matched, walls = set(), 0
    for side in sides:
        near = [lv for lv, count in side["levels"].items() if count >= MIN_ADJACENT_CELLS]
        hit = [lv for lv in near if side["top"] is not None and abs(side["top"] - lv) <= FACE_TOP_TOLERANCE]
        if side["area"] >= MIN_FACE_AREA and hit:
            matched.update(hit)
        elif side["wall"]:
            walls += 1
        else:
            return set()
    return matched if walls < 2 else set()


def _classify_strip(samples, polygon, frame, underside, higher, seen_above) -> tuple[str, float | None]:
    """(verdict, top level): "suspended" and "narrow" are unexplained, "shallow" is a relief under 14 cm, "beam" has side
    faces (or a wall) on both long sides, "verify" is a probable beam whose sides are not both seen."""
    if seen_above >= SEEN_ABOVE:
        return "suspended", None
    if frame["minor"] < BEAM_MINOR[0]:
        return "narrow", None
    sides = [_side_evidence(samples, polygon, frame, underside, higher, sign) for sign in (1.0, -1.0)]
    matched = _confirmed_tops(sides)
    reach = {lv: sum(side["reach"][lv] for side in sides) for lv, _ in higher}
    pool = matched or [lv for lv, count in reach.items() if count] or [min(lv for lv, _ in higher)]
    top = max(pool, key=lambda lv: reach.get(lv, 0))
    if top - underside < MIN_BEAM_DROP:
        return "shallow", None
    return ("beam" if matched else "verify"), top


def _side_planes(samples: Samples, frame: dict, underside: float, top: float):
    """Fits of the two vertical side faces; (pos, neg) with None where the face is not seen."""
    P = samples.P
    rel = P[:, [0, 2]] - frame["centre"]
    ra, rb = rel @ frame["u"], rel @ frame["v"]
    ok = (np.abs(samples.N[:, 1]) < 0.2) & (P[:, 1] > underside + 0.03) & (P[:, 1] < top - 0.03)
    ok &= (ra >= frame["a"][0]) & (ra <= frame["a"][1])
    ok &= (rb >= frame["b"][0] - SIDE_GROW) & (rb <= frame["b"][1] + SIDE_GROW)
    v3 = np.array([frame["v"][0], 0.0, frame["v"][1]])
    mid = (frame["b"][0] + frame["b"][1]) / 2
    out = []
    for sign in (1.0, -1.0):
        idx = np.nonzero(ok & (sign * (rb - mid) > 0))[0]
        fit = None
        if len(idx) >= 30:
            fit = fit_plane(P[idx], sign * v3, P[idx].mean(0), SIDE_GROW)
        if fit is not None and (fit.support_m2 < MIN_SIDE_AREA or abs(fit.normal @ v3) < 0.9):
            fit = None
        out.append(fit)
    return out


def _beam_axis(frame: dict, centre: np.ndarray, polygon: shapely.Polygon) -> list[list[float]]:
    """Axis endpoints at the component's own extent; an end within BEAM_SNAP of the outline moves onto it."""
    u, (lo, hi) = frame["u"], frame["extent"]
    probe = shapely.LineString([centre + (lo - BEAM_SNAP) * u, centre + (hi + BEAM_SNAP) * u])
    inside = shapely.intersection(probe, polygon)
    pieces = [g for g in getattr(inside, "geoms", [inside]) if isinstance(g, shapely.LineString) and not g.is_empty]
    start, end = lo, hi
    if pieces:
        piece = min(pieces, key=lambda g: g.distance(shapely.Point(centre)))
        t = sorted(float((np.array(c) - centre) @ u) for c in (piece.coords[0], piece.coords[-1]))
        start = t[0] if t[0] > lo - BEAM_SNAP + 1e-6 else lo
        end = t[1] if t[1] < hi + BEAM_SNAP - 1e-6 else hi
    return [[float(x), float(z)] for x, z in (centre + start * u, centre + end * u)]


def can_merge(a: dict, b: dict) -> bool:
    """Two levels (a above b) are one rough surface: closer than MERGE_SIGMAS sigmas, merged span within MAX_MERGE_SPAN."""
    gap = a["level"] - b["level"]
    return gap < MERGE_SIGMAS * max(a["sigma"], b["sigma"]) and a.get("high", a["level"]) - b.get("low", b["level"]) <= MAX_MERGE_SPAN


def combine(a: dict, b: dict) -> dict:
    """Support-weighted level of two mergeable levels; the caller merges the other fields."""
    support = a["support_m2"] + b["support_m2"]
    gap = a["level"] - b["level"]
    return {"level": (a["level"] * a["support_m2"] + b["level"] * b["support_m2"]) / support,
            "sigma": float(np.hypot(max(a["sigma"], b["sigma"]), gap / 2)), "support_m2": support,
            "low": b.get("low", b["level"]), "high": a.get("high", a["level"])}


def _merge_rough_levels(levels: list[dict]) -> list[dict]:
    """Merge touching levels (highest first) that `can_merge`."""
    levels = [{**lv, "low": lv["level"], "high": lv["level"]} for lv in levels]
    k = 0
    while k < len(levels) - 1:
        a, b = levels[k], levels[k + 1]
        touch = a["region"].buffer(CLOSING_ITERATIONS * GRID + 1e-6).intersects(b["region"])
        if not touch or not can_merge(a, b):
            k += 1
            continue
        levels[k:k + 2] = [{**combine(a, b), "region": _as_multipolygon(shapely.union_all([a["region"], b["region"]]))}]
    return levels


def _beam(samples: Samples, polygon: shapely.Polygon, frame: dict, level: dict, top: dict) -> dict:
    v, centre = frame["v"], frame["centre"].copy()
    width, width_sigma = frame["minor"], FALLBACK_WIDTH_SIGMA
    pos, neg = _side_planes(samples, frame, level["level"], top["level"])
    if pos is not None and neg is not None:
        w = float((pos.centroid[[0, 2]] - neg.centroid[[0, 2]]) @ v)
        if abs(w - frame["minor"]) < WIDTH_AGREEMENT:
            width, width_sigma = w, float(np.hypot(offset_sigma(pos), offset_sigma(neg)))
            centre += v * (((pos.centroid[[0, 2]] + neg.centroid[[0, 2]]) / 2 - centre) @ v)
    return {"axis": _beam_axis(frame, centre, polygon),
            "underside": level["level"], "underside_sigma": level["sigma"],
            "width": width, "width_sigma": width_sigma, "top": top["level"], "top_sigma": top["sigma"]}


def _unexplained(frame: dict, underside: float, reason: str) -> dict:
    lo, hi = frame["cells"].min(0) - GRID / 2, frame["cells"].max(0) + GRID / 2
    return {"bbox": [[float(lo[0]), float(lo[1])], [float(hi[0]), float(hi[1])]], "underside": float(underside),
            "width": float(frame["minor"]), "length": float(frame["major"]), "reason": reason}


def beam_footprint(beam: dict, polygon: shapely.Polygon):
    return shapely.LineString(beam["axis"]).buffer(beam["width"] / 2, cap_style="flat").intersection(polygon)


def ceiling_points(samples: Samples, polygon: shapely.Polygon, floor_level: float) -> np.ndarray:
    """Ceiling-facing samples above MIN_HEIGHT, inside the polygon shrunk by POLYGON_MARGIN."""
    P = samples.P
    inner = polygon.buffer(-POLYGON_MARGIN)
    ok = (samples.N[:, 1] < -0.9) & (P[:, 1] > floor_level + MIN_HEIGHT) & shapely.contains_xy(inner, P[:, 0], P[:, 2])
    return P[ok]


def level_fits(cand: np.ndarray) -> list[dict]:
    """Dominant levels of ceiling points as `{"level", "sigma", "support_m2"}`, highest first."""
    found = []
    for yc in _histogram_levels(cand[:, 1]):
        near = cand[np.abs(cand[:, 1] - yc) < LEVEL_WINDOW]
        fit = fit_plane(near, [0.0, -1.0, 0.0], near.mean(0), LEVEL_WINDOW)
        if fit is not None:
            found.append({"level": float(fit.point[1]), "sigma": offset_sigma(fit), "support_m2": fit.support_m2})
    found.sort(key=lambda lv: -lv["level"])
    return found


def detect(samples: Samples, polygon: shapely.Polygon, floor_level: float) -> CeilingResult:
    """Ceiling levels with plan regions, beams, and the fraction of the plan neither covers."""
    inner = polygon.buffer(-POLYGON_MARGIN)
    cand = ceiling_points(samples, polygon, floor_level)
    found = level_fits(cand)

    grid = _Grid(polygon)
    gap = np.abs(cand[:, 1:2] - np.array([lv["level"] for lv in found])) if found else np.zeros((len(cand), 0))
    for k, lv in enumerate(found):
        own = (gap.argmin(axis=1) == k) & (gap[:, k] < LEVEL_WINDOW)  # a sample serves its nearest level only
        lv["mask"] = grid.occupancy(cand[own][:, [0, 2]])
        lv["cells"] = grid.centres(lv["mask"])
    levels, beams, unexplained = [], [], []
    for k, lv in enumerate(found):
        higher = [(h["level"], h["cells"]) for h in found[:k]]
        labels, count = ndimage.label(lv["mask"], structure=np.ones((3, 3)))
        keep = np.zeros(lv["mask"].shape, bool)
        for comp in range(1, count + 1):
            mask = labels == comp
            if mask.sum() * GRID**2 < MIN_COMPONENT_AREA:
                continue
            frame = _component_frame(grid.centres(mask))
            if not higher or not _is_strip(frame):  # nothing is above the highest level: its strips are ceiling
                keep |= mask
                continue
            above = np.logical_or.reduce([h["mask"] for h in found[:k]])
            verdict, top_level = _classify_strip(samples, polygon, frame, lv["level"], higher, float(above[mask].mean()))
            if verdict in ("beam", "verify"):
                beam = _beam(samples, polygon, frame, lv, next(h for h in found if h["level"] == top_level))
                beams.append({**beam, "to_verify": verdict == "verify"})
            else:
                unexplained.append(_unexplained(frame, lv["level"], REASONS[verdict]))
        region = _as_multipolygon(grid.polygon(keep).simplify(SIMPLIFY).intersection(polygon)) if keep.any() else None
        if region is not None and not region.is_empty:
            levels.append({"level": lv["level"], "sigma": lv["sigma"], "region": region, "support_m2": lv["support_m2"]})
    levels = _merge_rough_levels(levels)
    beams.sort(key=lambda b: (-b["underside"], b["axis"][0]))
    covered = shapely.union_all([lv["region"] for lv in levels] + [beam_footprint(b, polygon) for b in beams])
    return CeilingResult(levels, beams, float(1.0 - covered.intersection(inner).area / inner.area), unexplained)
