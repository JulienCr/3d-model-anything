"""Posed photos in the room frame: projection, back-projection, upright orientation and visibility."""
from __future__ import annotations

import json
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
EDGE_MARGIN = 0.02  # fraction of the image kept clear of a point that must be visible
MIN_DEPTH = 0.3
PARALLEL_EPS = 1e-3  # |cos| between a ray and the plane normal below which the ray counts as parallel
BOX_MARGIN = 0.03  # m; furniture boxes are grown by this much before the sight test
SIGHT_SAMPLES = 9
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


def occluded_fraction(centre: np.ndarray, span: np.ndarray, boxes: list[RoomBox], margin: float = BOX_MARGIN) -> float:
    """Fraction of the span samples (ends included) whose sight line from `centre` crosses a box grown by `margin`.

    A box holding the sample, or the camera centre, is ignored for that sample.
    """
    span, centre = np.asarray(span, float), np.asarray(centre, float)
    t = np.linspace(0.0, 1.0, SIGHT_SAMPLES)[:, None]
    samples = span[0] + t * (span[-1] - span[0])
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
    return float(blocked.mean())


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
               k: int = 3, normal: np.ndarray | None = None, tolerance: float = DEPTH_TOLERANCE, boxes: list[RoomBox] | None = None) -> list[PosedPhoto]:
    """Up to k photos that see every point (depth test, 3 cm by default), best first.

    Score: squared pixel diagonal of the projected points, times the cosine between the optical axis and the
    direction to the points' centroid, times |cos| between `normal` and the direction to the camera when given.
    With `boxes`, photos are ranked by the fraction of the span hidden by furniture first (see `occluded_fraction`).
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
        scored.append(((round(hidden, 6), -diag2 * max(cos_axis, 0.0) * cos_surface), photo))
    scored.sort(key=lambda s: s[0])
    return [p for _, p in scored[:k]]
