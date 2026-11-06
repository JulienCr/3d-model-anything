"""Free-standing and wall-engaged columns from a plan-view density map of vertical LiDAR surfaces."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage

from room_model.lidar import SAMPLE_DENSITY, Samples

EXCLUDED_CLASSES = (4, 5, 6, 7)  # table, seat, window, door
VERTICAL_N_Y = 0.2
BAND_MARGIN = 0.3  # m above the floor and below the lowest ceiling
WALL_BAND = 0.04  # m, vertical samples this close to a fitted wall belong to the wall
WALL_EXTENT_PAD = 0.10
CELL = 0.01
OCCUPANCY_FRACTION = 0.5
MIN_CELLS = 40
ENGAGED_TOLERANCE = 0.06
SIZE_RANGE = (0.10, 1.50)
SIDE_DEPTH = 3  # cells inspected inside each side of the bounding rectangle
SIDE_FILL = 0.7  # fraction of a side's length that must be occupied
CEILING_REACH = 0.15
FACE_SIGMA = 0.01
BBOX_PAD = 0.03
AXIS_TOLERANCE_DEG = 20.0
FACE_WINDOW = 0.03  # m around a raster side searched for its face samples
FACE_NORMAL = 0.7
MIN_FACE_SAMPLES = 30
UNMEASURED_FACE_SIGMA = 0.02


@dataclass
class ColumnResult:
    columns: list[dict]  # {"center", "size", "center_sigma", "size_sigma", "engaged_wall", "engaged_walls", "top", "top_sigma"}, by x then z
    unexplained: list[dict]  # {"bbox", "height_max", "reason"}


def _vertical_mask(samples: Samples, walls: list[dict], polygon: shapely.Polygon) -> np.ndarray:
    P = samples.P
    ok = (np.abs(samples.N[:, 1]) < VERTICAL_N_Y) & ~np.isin(samples.cls, EXCLUDED_CLASSES)
    ok &= shapely.contains_xy(polygon, P[:, 0], P[:, 2])
    xz = P[:, [0, 2]]
    for w in walls:
        d = w["end"] - w["start"]
        length = float(np.linalg.norm(d))
        along = (xz - w["start"]) @ (d / length)
        near = np.abs(xz @ w["normal"] - w["offset"]) < WALL_BAND
        ok &= ~(near & (along > -WALL_EXTENT_PAD) & (along < length + WALL_EXTENT_PAD))
    return ok


def _side_filled(mask: np.ndarray, box: tuple[int, int, int, int], side: str) -> bool:
    i0, i1, j0, j1 = box
    strips = {"x0": mask[i0:i0 + SIDE_DEPTH, j0:j1 + 1].any(axis=0), "x1": mask[i1 - SIDE_DEPTH + 1:i1 + 1, j0:j1 + 1].any(axis=0),
              "z0": mask[i0:i1 + 1, j0:j0 + SIDE_DEPTH].any(axis=1), "z1": mask[i0:i1 + 1, j1 - SIDE_DEPTH + 1:j1 + 1].any(axis=1)}
    return bool(strips[side].mean() >= SIDE_FILL)


def _engaged_walls(bbox: np.ndarray, walls: list[dict]) -> list[tuple[int, dict, str, float]]:
    """Walls whose line passes within ENGAGED_TOLERANCE of a bbox side, one per axis: (index, wall, side, snapped coordinate)."""
    best: dict[int, tuple] = {}
    for k, w in enumerate(walls):
        n = w["normal"]
        axis = 0 if abs(n[0]) > abs(n[1]) else 1
        if np.degrees(np.arccos(min(abs(n[axis]), 1.0))) > AXIS_TOLERANCE_DEG:
            continue
        other = 1 - axis
        mid = (bbox[0, other] + bbox[1, other]) / 2
        d = w["end"] - w["start"]
        length = float(np.linalg.norm(d))
        along = bbox @ (d / length) - w["start"] @ (d / length)
        if not (along.max() > -WALL_EXTENT_PAD and along.min() < length + WALL_EXTENT_PAD):
            continue
        for hi in (0, 1):
            point = np.zeros(2)
            point[axis], point[other] = bbox[hi, axis], mid
            dist = abs(n @ point - w["offset"])
            if dist < ENGAGED_TOLERANCE and (axis not in best or dist < best[axis][0]):
                side = ("x" if axis == 0 else "z") + str(hi)
                best[axis] = (dist, k, w, side, float((w["offset"] - n[other] * mid) / n[axis]))
    return sorted((b[1:] for b in best.values()), key=lambda b: b[0])


def _measure_faces(P: np.ndarray, N: np.ndarray, in_band: np.ndarray, bbox: np.ndarray, fixed: list[str]):
    """Median face position per bbox side, and per-side sigma; raster sides without enough samples keep the raster value."""
    bbox = bbox.copy()
    sigma = {}
    for side in ("x0", "x1", "z0", "z1"):
        axis, hi = (0 if side[0] == "x" else 1), side.endswith("1")
        if side in fixed:
            sigma[side] = FACE_SIGMA
            continue
        other = 1 - axis
        near = in_band & (np.abs(P[:, 2 * axis] - bbox[int(hi), axis]) < FACE_WINDOW) & (np.abs(N[:, 2 * axis]) > FACE_NORMAL)
        near &= (P[:, 2 * other] >= bbox[0, other]) & (P[:, 2 * other] <= bbox[1, other])
        if near.sum() < MIN_FACE_SAMPLES:
            sigma[side] = UNMEASURED_FACE_SIGMA
        else:
            bbox[int(hi), axis] = np.median(P[near, 2 * axis])
            sigma[side] = FACE_SIGMA
    return bbox, sigma


def _ceiling_above(centre: np.ndarray, ceilings: list[dict]) -> dict:
    point = shapely.Point(centre)
    return min(ceilings, key=lambda lv: (lv["region"].distance(point), lv["level"]))


def detect(samples: Samples, walls: list[dict], polygon: shapely.Polygon, floor_level: float,
           ceilings: list[dict]) -> ColumnResult:
    """Columns and unexplained vertical blobs; `walls` is shell.wall_lines output, `ceilings` excludes beams."""
    if not ceilings:
        return ColumnResult([], [])
    low, high = floor_level + BAND_MARGIN, min(lv["level"] for lv in ceilings) - BAND_MARGIN
    if high <= low:
        return ColumnResult([], [])
    vertical = _vertical_mask(samples, walls, polygon)
    P = samples.P
    band = vertical & (P[:, 1] > low) & (P[:, 1] < high)

    x0, z0, x1, z1 = polygon.bounds
    shape = (int(np.ceil((x1 - x0) / CELL)) + 2, int(np.ceil((z1 - z0) / CELL)) + 2)
    origin = np.array([x0, z0])
    ij = np.floor((P[band][:, [0, 2]] - origin) / CELL).astype(int)
    ij = ij[(ij >= 0).all(axis=1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])]
    counts = np.bincount(ij[:, 0] * shape[1] + ij[:, 1], minlength=shape[0] * shape[1]).reshape(shape)
    # A 1 cm wide face often straddles two cells and halves its count, so each window pools 2x2 cells.
    pooled = (counts[:-1, :-1] + counts[1:, :-1] + counts[:-1, 1:] + counts[1:, 1:]) / 2
    occupied = pooled >= OCCUPANCY_FRACTION * SAMPLE_DENSITY * CELL * (high - low)
    occupied = ndimage.binary_closing(occupied, structure=np.ones((3, 3)), iterations=1)
    labels, count = ndimage.label(occupied, structure=np.ones((3, 3)))

    columns, unexplained = [], []
    for k in range(1, count + 1):
        cells = np.argwhere(labels == k)
        if len(cells) < MIN_CELLS:
            continue
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        bbox = np.array([origin + (lo + 1) * CELL, origin + (hi + 1) * CELL])
        engaged = _engaged_walls(bbox, walls)
        box = (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]))
        required = ["x0", "x1", "z0", "z1"]
        for _, _, side, coord in engaged:
            required.remove(side)
            bbox[0 if side.endswith("0") else 1, 0 if side[0] == "x" else 1] = coord
        bbox, faces = _measure_faces(P, samples.N, band, bbox, [side for _, _, side, _ in engaged])
        filled = [_side_filled(occupied, box, s) for s in required]
        size = bbox[1] - bbox[0]
        centre = bbox.mean(axis=0)

        inside = vertical & (P[:, 0] > bbox[0, 0] - BBOX_PAD) & (P[:, 0] < bbox[1, 0] + BBOX_PAD)
        inside &= (P[:, 2] > bbox[0, 1] - BBOX_PAD) & (P[:, 2] < bbox[1, 1] + BBOX_PAD)
        height_max = float(P[inside, 1].max() - floor_level) if inside.any() else 0.0
        above = _ceiling_above(centre, ceilings)

        if not all(SIZE_RANGE[0] <= s <= SIZE_RANGE[1] for s in size):
            reason = "size_out_of_range"
        elif sum(filled) < min(len(required), 3):
            reason = "open_sides"
        elif height_max + floor_level < above["level"] - CEILING_REACH:
            reason = "not_full_height"
        else:
            columns.append({"center": [float(c) for c in centre], "size": [float(s) for s in size],
                            "engaged_wall": engaged[0][1]["label"] if engaged else None,
                            "engaged_walls": [wall["label"] for _, wall, _, _ in engaged],
                            "top": float(above["level"]), "top_sigma": float(above["sigma"]),
                            "center_sigma": [float(np.hypot(faces["x0"], faces["x1"]) / 2), float(np.hypot(faces["z0"], faces["z1"]) / 2)],
                            "size_sigma": [float(np.hypot(faces["x0"], faces["x1"])), float(np.hypot(faces["z0"], faces["z1"]))]})
            continue
        unexplained.append({"bbox": [[float(v) for v in bbox[0]], [float(v) for v in bbox[1]]],
                            "height_max": height_max, "reason": reason})
    columns.sort(key=lambda c: (c["center"][0], c["center"][1]))
    return ColumnResult(columns, unexplained)
