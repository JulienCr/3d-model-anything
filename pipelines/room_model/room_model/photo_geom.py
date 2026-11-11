"""Posed photos in the room frame: projection, back-projection, upright orientation and visibility."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from room_model import raster
from room_model.capture import colmajor, invert_rigid, load_mesh_ply, load_photos, mesh_path
from room_model.context import StageContext, output_path
from room_model.frame import world_to_room

DEPTH_SIDE = 320  # px on the long side of the depth maps
DEPTH_TOLERANCE = 0.03  # m; a point this much nearer than the nearest surface is still visible
LOOSE_TOLERANCE = 0.06  # m; a skirting or a casing stands out of its wall plane by up to this much and may hide an end
EDGE_MARGIN = 0.02  # fraction of the image kept clear of a point that must be visible
MIN_DEPTH = 0.3
PARALLEL_EPS = 1e-3  # |cos| between a ray and the plane normal below which the ray counts as parallel
BOX_MARGIN = 0.03  # m; furniture boxes are grown by this much before the sight test
SIGHT_SAMPLES = 9
# A planar element is a usable view only if the camera faces it: beyond 60 degrees from its normal the
# foreshortening makes a corner click unreliable (a 1 px error grows by 1/cos, so more than 2 px).
FACING_COS = math.cos(math.radians(60))
# A measured span is only read by a person, who can still tell its two ends apart up to 70 degrees (1/cos < 3).
SURVEY_FACING_COS = math.cos(math.radians(70))
GRID_SIDE = 5  # points per side of the grid laid on a rectangle
# The grid stops 3 cm inside the edges: an opening's interior is empty space in the mesh, where nothing is "visible"
# by default, so only its border ring, against the frame (the frame band), is tested.
FRAME_BAND = 0.03
# Curtains or furniture often hide part of a frame, so half of the band is enough: what is clicked is the corners.
MIN_BAND_SEEN = 0.5
# Each corner is cast on the host plane on its own, so two visible corners per crop are enough and other crops add the rest.
MIN_CORNERS = 2
CORNER_RADIUS = 0.02  # m; a corner is tested on its grid point and four samples this far around it
MIN_CORNER_SEEN = 2  # of the 5 samples: a casing or a leaf edge can hide most of a 2 cm disc, 0 or 1 left means a hidden corner
MIN_SIDE_PX = 120  # shorter side of the projected rectangle: below it a corner click is within a few percent of the size
ROTATE = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}


@dataclass
class PosedPhoto:
    name: str
    path: Path
    K: np.ndarray  # 3x3, pixels of the raw sensor image
    c2w_room: np.ndarray  # 4x4, OpenCV camera (x right, y down, z forward) to room frame
    size: tuple[int, int]  # (width, height) of the sensor image
    up_turns: int  # counter-clockwise quarter turns of the sensor image that bring room +Y up

    @property
    def w2c(self) -> np.ndarray:
        return invert_rigid(self.c2w_room)

    @property
    def centre(self) -> np.ndarray:
        return self.c2w_room[:3, 3]


def up_turns_of(c2w_room: np.ndarray) -> int:
    """Quarter turns (counter-clockwise) that bring room +Y up in the sensor image of a camera."""
    up_cam = invert_rigid(c2w_room)[:3, 1]  # room +Y in camera axes: x = image right, y = image down
    if abs(up_cam[1]) >= abs(up_cam[0]):
        return 0 if up_cam[1] < 0 else 2
    return 1 if up_cam[0] > 0 else 3


def load_posed_photos(ctx: StageContext, w2r: np.ndarray | None = None) -> list[PosedPhoto]:
    """Photos of the room with room-frame poses: refined ones from registration/poses.json, else ARKit."""
    photos = load_photos(ctx.capture, ctx.room)
    if not photos:
        return []
    if w2r is None:
        model_path = output_path(ctx, "model")
        if model_path.exists():
            w2r = colmajor(json.loads(model_path.read_text(encoding="utf-8"))["frame"]["world_to_room"], 4)
        else:
            w2r = world_to_room(ctx.capture, ctx.room, ctx.turn)
    refined = {}
    poses_path = output_path(ctx, "registration")
    if poses_path.exists():
        refined = {p["name"]: colmajor(p["c2w_arkit"], 4) for p in json.loads(poses_path.read_text(encoding="utf-8"))["photos"]}
    out = []
    for p in photos:
        c2w_arkit = refined.get(p.name, p.c2w_arkit)
        c2w_room = w2r @ c2w_arkit @ np.diag([1.0, -1.0, -1.0, 1.0])
        out.append(PosedPhoto(p.name, ctx.capture / p.rel_file, p.K, c2w_room, (p.width, p.height), up_turns_of(c2w_room)))
    return out


def load_room_mesh(ctx: StageContext, w2r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of the in-room faces of the LiDAR mesh, vertices in the room frame."""
    mesh = load_mesh_ply(mesh_path(ctx.capture, ctx.room))
    faces = mesh["faces"]
    if mesh["inroom"] is not None:
        faces = faces[(mesh["inroom"] == 1)[faces].all(axis=1)]
    return mesh["vertices"] @ w2r[:3, :3].T + w2r[:3, 3], faces


def upright_size(photo: PosedPhoto) -> tuple[int, int]:
    w, h = photo.size
    return (h, w) if photo.up_turns % 2 else (w, h)


def upright(photo: PosedPhoto) -> Image.Image:
    """The sensor image turned so that room +Y is up."""
    with Image.open(photo.path) as im:
        im = im.convert("RGB")
        return im.transpose(ROTATE[photo.up_turns % 4]) if photo.up_turns % 4 else im.copy()


def to_upright(photo: PosedPhoto, uv_sensor: np.ndarray) -> np.ndarray:
    """Continuous sensor pixel coordinates (pixel i covers [i, i + 1)) to upright-image coordinates."""
    uv = np.asarray(uv_sensor, float)
    u, v = uv[..., 0], uv[..., 1]
    w, h = photo.size
    k = photo.up_turns % 4
    out = {0: (u, v), 1: (v, w - u), 2: (w - u, h - v), 3: (h - v, u)}[k]
    return np.stack(out, axis=-1)


def to_sensor(photo: PosedPhoto, uv_upright: np.ndarray) -> np.ndarray:
    """Inverse of `to_upright`: upright-image coordinates to sensor pixel coordinates."""
    uv = np.asarray(uv_upright, float)
    a, b = uv[..., 0], uv[..., 1]
    w, h = photo.size
    k = photo.up_turns % 4
    out = {0: (a, b), 1: (w - b, a), 2: (w - a, h - b), 3: (b, h - a)}[k]
    return np.stack(out, axis=-1)


def project(photo: PosedPhoto, pts_room: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(uv, depth): sensor pixel coordinates (n, 2) and distance along the optical axis (n,) of room-frame points."""
    pts = np.atleast_2d(np.asarray(pts_room, float))
    w2c = photo.w2c
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = cam @ photo.K.T
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, depth


def backproject(photo: PosedPhoto, uv_sensor: np.ndarray, plane: tuple[np.ndarray, float]) -> np.ndarray | None:
    """Room-frame points (n, 3) where the rays through sensor pixels meet the plane n . x = offset.

    None when any ray is parallel to the plane or meets it behind the camera: no point is a better answer than a
    point at infinity or on the wrong side.
    """
    uv = np.atleast_2d(np.asarray(uv_sensor, float))
    normal, offset = np.asarray(plane[0], float), float(plane[1])
    rays_cam = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(photo.K).T
    rays = rays_cam @ photo.c2w_room[:3, :3].T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    slope = rays @ normal
    if (np.abs(slope) < PARALLEL_EPS * np.linalg.norm(normal)).any():
        return None
    t = (offset - normal @ photo.centre) / slope
    if not (np.isfinite(t).all() and (t > 0).all()):
        return None
    return photo.centre + t[:, None] * rays


@dataclass
class RoomBox:
    centre: np.ndarray  # (3,), room frame
    axes: np.ndarray  # 3x3, columns are the unit box axes
    half: np.ndarray  # (3,), half sizes along the axes


def room_boxes(roomplan: dict, w2r: np.ndarray) -> list[RoomBox]:
    """Oriented boxes of the RoomPlan objects (furniture), in the room frame."""
    out = []
    for obj in roomplan.get("objects", []):
        T = w2r @ colmajor(obj["transform"], 4)
        out.append(RoomBox(T[:3, 3].copy(), T[:3, :3] / np.linalg.norm(T[:3, :3], axis=0), np.asarray(obj["dimensions"], float) / 2))
    return out


def occluded_mask(centre: np.ndarray, samples: np.ndarray, boxes: list[RoomBox], margin: float = BOX_MARGIN) -> np.ndarray:
    """Bool (n,): the sight line from `centre` to the sample crosses a box grown by `margin`.

    A box holding the sample, or the camera centre, is ignored for that sample.
    """
    samples, centre = np.atleast_2d(np.asarray(samples, float)), np.asarray(centre, float)
    blocked = np.zeros(len(samples), bool)
    for box in boxes:
        half = box.half + margin
        a, p = (centre - box.centre) @ box.axes, (samples - box.centre) @ box.axes
        if (np.abs(a) <= half).all():
            continue
        d = p - a
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (-half - a) / d, (half - a) / d
        parallel = np.abs(d) < 1e-12
        lo = np.where(parallel, np.where(np.abs(a) <= half, -np.inf, np.inf), np.minimum(t0, t1)).max(1)
        hi = np.where(parallel, np.inf, np.maximum(t0, t1)).min(1)
        inside = (np.abs(p) <= half).all(1)
        blocked |= ~inside & (np.maximum(lo, 0.0) <= np.minimum(hi, 1.0))
    return blocked


def occluded_fraction(centre: np.ndarray, span: np.ndarray, boxes: list[RoomBox], margin: float = BOX_MARGIN) -> float:
    """Fraction of SIGHT_SAMPLES points along the span (ends included) hidden from `centre` by a box (see `occluded_mask`)."""
    span = np.asarray(span, float)
    samples = span[0] + np.linspace(0.0, 1.0, SIGHT_SAMPLES)[:, None] * (span[-1] - span[0])
    return float(occluded_mask(centre, samples, boxes, margin).mean())


class DepthMaps:
    """Z-buffers of a room-frame mesh seen by photos, at most DEPTH_SIDE px on the long side, computed on demand."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices, self.faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
        self._maps: dict[str, tuple[np.ndarray, float]] = {}

    def get(self, photo: PosedPhoto) -> tuple[np.ndarray, float]:
        """(depth map, scale from sensor pixels to map pixels)."""
        if photo.name not in self._maps:
            w, h = photo.size
            scale = min(1.0, DEPTH_SIDE / max(w, h))
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            K = np.diag([size[0] / w, size[1] / h, 1.0]) @ photo.K
            faces = self._in_frustum(photo, K, size)
            self._maps[photo.name] = (raster.depth_map(self.vertices, faces, K, photo.w2c, size), size[0] / w)
        return self._maps[photo.name]

    def _in_frustum(self, photo: PosedPhoto, K: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Faces that may reach the image: the culling is conservative, a face crossing the image border is kept."""
        w2c = photo.w2c
        cam = self.vertices @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (cam @ K.T)[:, :2] / z[:, None]
        f = self.faces
        left, right = (uv[f, 0] < 0).all(1), (uv[f, 0] > size[0]).all(1)
        top, bottom = (uv[f, 1] < 0).all(1), (uv[f, 1] > size[1]).all(1)
        behind = (z[f] <= raster.NEAR).all(1)
        return f[~(left | right | top | bottom | behind)]

    def visible(self, photo: PosedPhoto, pts_room: np.ndarray, tolerance: float = DEPTH_TOLERANCE) -> np.ndarray:
        """Bool (n,): inside the image, in front of the camera and not hidden by a surface more than `tolerance` m nearer.

        The farthest depth of the 3x3 pixels around a point counts, so a point on the edge of an opening is not
        hidden by the frame next to it.
        """
        pts = np.atleast_2d(np.asarray(pts_room, float))
        uv, depth = project(photo, pts)
        zbuf, scale = self.get(photo)
        w, h = photo.size
        inside = ((uv[:, 0] > EDGE_MARGIN * w) & (uv[:, 0] < (1 - EDGE_MARGIN) * w)
                  & (uv[:, 1] > EDGE_MARGIN * h) & (uv[:, 1] < (1 - EDGE_MARGIN) * h) & (depth > MIN_DEPTH))
        out = np.zeros(len(pts), bool)
        i = np.nonzero(inside)[0]
        col = np.clip((uv[i, 0] * scale).astype(int), 0, zbuf.shape[1] - 1)
        row = np.clip((uv[i, 1] * scale).astype(int), 0, zbuf.shape[0] - 1)
        nearest = np.max([zbuf[np.clip(row + dr, 0, zbuf.shape[0] - 1), np.clip(col + dc, 0, zbuf.shape[1] - 1)]
                          for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)
        out[i] = depth[i] <= nearest + tolerance
        return out


def best_views(photos: list[PosedPhoto], pts_room: np.ndarray, mesh: DepthMaps | tuple[np.ndarray, np.ndarray],
               k: int = 3, normal: np.ndarray | None = None, tolerance: float = DEPTH_TOLERANCE, boxes: list[RoomBox] | None = None,
               min_cos: float = 0.0) -> list[PosedPhoto]:
    """Up to k photos that see every point (depth test, 3 cm by default), best first.

    Score: squared pixel diagonal of the projected points, times the cosine between the optical axis and the
    direction to the points' centroid, times |cos| between `normal` and the direction to the camera when given.
    With `boxes`, photos are ranked by the fraction of the span hidden by furniture first (see `occluded_fraction`).
    With `normal` and `min_cos`, photos seeing the points at less than `min_cos` from the normal are dropped.
    """
    maps = mesh if isinstance(mesh, DepthMaps) else DepthMaps(*mesh)
    pts = np.atleast_2d(np.asarray(pts_room, float))
    centroid = pts.mean(0)
    scored = []
    for photo in photos:
        uv, depth = project(photo, pts)
        if not (depth > MIN_DEPTH).all() or not maps.visible(photo, pts, tolerance).all():
            continue
        hidden = occluded_fraction(photo.centre, pts, boxes) if boxes else 0.0
        diag2 = float(((uv.max(0) - uv.min(0)) ** 2).sum()) + 1.0
        to_point = centroid - photo.centre
        distance = np.linalg.norm(to_point)
        cos_axis = float(to_point @ photo.c2w_room[:3, 2]) / max(distance, 1e-9)
        cos_surface = 1.0 if normal is None else abs(float(np.asarray(normal, float) @ to_point)) / max(distance, 1e-9)
        if cos_surface < min_cos:
            continue
        scored.append(((round(hidden, 6), -diag2 * max(cos_axis, 0.0) * cos_surface), photo))
    scored.sort(key=lambda s: s[0])
    return [p for _, p in scored[:k]]


@dataclass
class RectView:
    photo: PosedPhoto
    cos: float  # between the face normal and the direction from the rectangle centre to the camera
    area_px: float  # of the projected rectangle
    side_px: float  # its shorter side
    seen: float  # share of the frame band passing the depth test and the furniture box test
    corners: tuple[bool, ...]  # which of the four corners are visible


def rect_grid(rect: np.ndarray) -> np.ndarray:
    """GRID_SIDE x GRID_SIDE points (n, 3) over the rectangle (4 corners, in order) shrunk by FRAME_BAND on every side."""
    rect = np.asarray(rect, float)
    u, v = rect[1] - rect[0], rect[3] - rect[0]
    lo = [min(FRAME_BAND, 0.25 * float(np.linalg.norm(e))) / max(float(np.linalg.norm(e)), 1e-9) for e in (u, v)]
    s, t = (np.linspace(a, 1.0 - a, GRID_SIDE) for a in lo)
    return (rect[0] + s[None, :, None] * u + t[:, None, None] * v).reshape(-1, 3)


def rect_band(rect: np.ndarray) -> np.ndarray:
    """The border ring of `rect_grid` (4 x (GRID_SIDE - 1) points): the frame band, 3 cm inside the edges."""
    ring = np.zeros((GRID_SIDE, GRID_SIDE), bool)
    ring[[0, -1], :] = ring[:, [0, -1]] = True
    return rect_grid(rect)[ring.ravel()]


def rect_corner_samples(rect: np.ndarray) -> np.ndarray:
    """(4, 5, 3): per corner, its grid point (3 cm inside along both edges) and four samples CORNER_RADIUS around it."""
    rect = np.asarray(rect, float)
    centres = rect_grid(rect).reshape(GRID_SIDE, GRID_SIDE, 3)[[0, 0, -1, -1], [0, -1, -1, 0]]
    u, v = ((e / max(float(np.linalg.norm(e)), 1e-9)) * CORNER_RADIUS for e in (rect[1] - rect[0], rect[3] - rect[0]))
    steps = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    return centres[:, None, :] + steps[None, :, 0, None] * u + steps[None, :, 1, None] * v


def rect_views(photos: list[PosedPhoto], rect: np.ndarray, mesh: DepthMaps | tuple[np.ndarray, np.ndarray], normal: np.ndarray | None = None,
               two_sided: bool = False, boxes: list[RoomBox] | None = None, k: int = 3, tolerance: float = LOOSE_TOLERANCE,
               min_side: float = MIN_SIDE_PX) -> list[RectView]:
    """Up to k eligible photos of a planar rectangle (4 corners in order), best first; none when no photo qualifies.

    Eligible: the camera faces the rectangle (cosine of `normal`, the face normal towards the viewing side, at least
    FACING_COS; |cos| when `two_sided`, as for a hole through its host, or when no normal is given), the projected
    rectangle is at least `min_side` px on its shorter side, at least MIN_CORNERS corners are visible (MIN_CORNER_SEEN
    of their samples) and MIN_BAND_SEEN of the frame band is. Visible = depth test (LOOSE_TOLERANCE: casings stand out
    of the wall plane, an occluder is a wall thickness or more) and furniture box test. Ranked by visible corners, then
    seen share x area x cos; the k kept views cover as many corners as possible (`cover`).
    """
    maps = mesh if isinstance(mesh, DepthMaps) else DepthMaps(*mesh)
    rect = np.asarray(rect, float)
    if normal is None:
        normal, two_sided = np.cross(rect[1] - rect[0], rect[3] - rect[0]), True
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    band, corners, centre = rect_band(rect), rect_corner_samples(rect), rect.mean(0)
    points = np.vstack([band, corners.reshape(-1, 3)])
    found = []
    for photo in photos:
        uv, depth = project(photo, rect)
        to_camera = photo.centre - centre
        cos = float(normal @ to_camera) / max(float(np.linalg.norm(to_camera)), 1e-9)
        cos = abs(cos) if two_sided else cos
        wide, tall = ((np.linalg.norm(uv[a] - uv[b]) + np.linalg.norm(uv[c] - uv[d])) / 2 for a, b, c, d in ((1, 0, 2, 3), (3, 0, 2, 1)))
        if not (depth > MIN_DEPTH).all() or cos < FACING_COS or min(wide, tall) < min_side:
            continue
        seen = maps.visible(photo, points, tolerance)
        if boxes:
            seen &= ~occluded_mask(photo.centre, points, boxes)
        band_seen, corner_seen = seen[:len(band)], seen[len(band):].reshape(4, 5)
        visible = tuple(bool(n >= MIN_CORNER_SEEN) for n in corner_seen.sum(1))
        if sum(visible) < MIN_CORNERS or band_seen.mean() < MIN_BAND_SEEN - 1e-9:
            continue
        area = 0.5 * abs(float(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1))))
        found.append(RectView(photo, cos, area, min(wide, tall), float(band_seen.mean()), visible))
    found.sort(key=lambda v: (-sum(v.corners), -v.seen * v.area_px * v.cos))
    return cover(found, k)


def cover(ranked: list[RectView], k: int) -> list[RectView]:
    """k views from a ranked list: the best, then each time the one adding most uncovered corners (rank breaks ties)."""
    pool, chosen, covered = list(ranked), [], [False] * 4
    while pool and len(chosen) < k:
        gains = [sum(c and not s for c, s in zip(v.corners, covered)) for v in pool]
        best = pool.pop(gains.index(max(gains)))
        chosen.append(best)
        covered = [s or c for s, c in zip(covered, best.corners)]
    return chosen
