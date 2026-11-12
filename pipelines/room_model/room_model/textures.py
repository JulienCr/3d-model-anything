"""Textures stage: posed photos projected onto walls, floor and ceilings, with an occlusion mask and a CC0 fill for the holes.

Every surface is a planar grid of TEXEL-sized texels (origin, u_axis, v_axis, normal, size). A texel is seen by a photo
when it falls inside the image, faces the camera, passes the depth test against the LiDAR mesh and lies in no furniture
box; its colour is the weighted median of the best candidates once the photos are exposure-matched. Holes take a
tinted fill texture. Output: textures/<element>_<face>.png and textures/textures.json.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from PIL import Image
from scipy import ndimage

from room_model import raster
from room_model.capture import colmajor, load_mesh_ply, load_roomplan, mesh_path
from room_model.context import NoPhotos, StageContext, output_path, require
from room_model.photo_geom import (DEPTH_TOLERANCE, EDGE_MARGIN, MIN_DEPTH, PosedPhoto, RoomBox, load_posed_photos, occluded_mask,
                                   project, room_boxes)

TEXEL = 0.02  # m per texel: finer than the ~1 cm LiDAR noise would justify, coarse enough to stay small
# Depth test: the mesh must lie within `Surface.tolerance` of the plane. Nearer is an occluder, farther a hole in the
# surface (opening, recess): both are rejected. DEPTH_TOLERANCE (photo_geom, 3 cm) is the floor, for a perfectly fitted plane.
LIDAR_NOISE = 0.01  # m, 1 sigma of the ARKit LiDAR mesh against a flat wall at room range
TOLERANCE_K = 3.0  # sigmas: a texel on the right plane then passes 99.7% of the time
TOLERANCE_CAP = 0.10  # m: furniture and radiators stand 10 cm or more proud of a wall; frames, casings and skirtings (< 10 cm) belong to its surface
TEX_BOX_MARGIN = 0.05  # m RoomPlan boxes are grown by: their sigma is of that order
TEX_DEPTH_SIDE = 640  # px on the long side of the depth maps: about 8 mm per pixel at 3 m, finer than a texel
SMALL_HOLE = 9  # texels: an isolated hole this small is speckle (3 x 3 texels = 6 cm), closed from its neighbours instead of feathered
GRAZING_COS = math.cos(math.radians(75.0))  # beyond 75 degrees a texel is stretched 4 times along the view: too coarse to trust
POOL = 6  # best candidates kept per texel; the 3 best colour it, all of them serve to match the exposures
BEST = 3
GAMMA = 2.2  # photos are sRGB: gains and medians are computed in linear light
GAIN_ROUNDS = 3
GAIN_RANGE = (0.5, 2.0)  # a stop either way: auto-exposure drift within one capture stays inside it
MAX_LEVEL = 5  # deepest reduction of a photo used to sample a texel (32 times smaller)
FEATHER = 8  # texels over which the observed colour fades into the fill next to a hole
FILL_TILE_M = 1.0  # m covered by one fill tile
DEFAULT_FILL = {"wall": "polyhaven:white_plaster_02", "ceiling": "polyhaven:white_plaster_02", "floor": "polyhaven:oak_veneer_01"}
FETCH_SCRIPT = Path(__file__).resolve().parents[3] / ".agents" / "skills" / "scan-to-threejs" / "scripts" / "fetch_textures.py"
BLENDER_DIR = Path(__file__).resolve().parents[1] / "blender"
FETCH_TIMEOUT = 30  # s per asset: three small maps; a slower link means no usable network


@dataclass
class Surface:
    """A planar surface sampled on a texel grid: texel (i, j) is centred at origin + (i + .5) TEXEL u + (j + .5) TEXEL v."""
    element: str
    face: str
    kind: str  # "wall", "floor" or "ceiling"
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray  # towards the side the surface is seen from
    size: tuple[int, int]  # (nu, nv) texels
    mask: np.ndarray | None = None  # (nv, nu) bool: texels inside the outline; all of them when None
    sigma: float = 0.0  # m, 1 sigma of the fitted plane position over the surface (offset or level, plus the tilt over half its length)

    @property
    def tolerance(self) -> float:
        """Depth-test tolerance in m: TOLERANCE_K sigmas of the plane error and the LiDAR noise, between DEPTH_TOLERANCE and TOLERANCE_CAP."""
        return float(np.clip(TOLERANCE_K * math.hypot(self.sigma, LIDAR_NOISE), DEPTH_TOLERANCE, TOLERANCE_CAP))

    @property
    def key(self) -> tuple[str, str]:
        return (self.element, self.face)

    @property
    def inside(self) -> np.ndarray:
        return np.ones(self.size[::-1], bool) if self.mask is None else self.mask

    def points(self) -> np.ndarray:
        """Texel centres (nu * nv, 3) in the room frame, ordered row by row (index j * nu + i)."""
        nu, nv = self.size
        i, j = np.meshgrid(np.arange(nu) + 0.5, np.arange(nv) + 0.5)
        return self.origin + i.reshape(-1, 1) * TEXEL * self.u_axis + j.reshape(-1, 1) * TEXEL * self.v_axis


def _arch_modules():
    """The pure-Python modules of blender/arch: the footprints the textures must follow are computed there."""
    if str(BLENDER_DIR) not in sys.path:
        sys.path.insert(0, str(BLENDER_DIR))
    from arch import frame, plan
    return frame, plan


def _upright(p0, p1, y0: float, y1: float, normal, element: str, face: str, sigma: float = 0.0) -> Surface:
    """A vertical surface along the plan segment p0 -> p1 between heights y0 and y1; u runs start to end, v up."""
    length = math.dist(p0, p1)
    u = np.array([(p1[0] - p0[0]) / length, 0.0, (p1[1] - p0[1]) / length])
    perpendicular = np.array([-u[2], 0.0, u[0]])  # the face is the segment itself, which a snapped partition end can turn away from its azimuth
    side = perpendicular if perpendicular @ np.array([normal[0], 0.0, normal[1]]) > 0 else -perpendicular
    return Surface(element, face, "wall", np.array([p0[0], y0, p0[1]]), u, np.array([0.0, 1.0, 0.0]), side,
                   (math.ceil(length / TEXEL - 1e-9), math.ceil((y1 - y0) / TEXEL - 1e-9)), sigma=sigma)


def _horizontal(element: str, kind: str, rings: list, level: float, normal_y: float, sigma: float = 0.0) -> Surface:
    """A floor or ceiling over the union of (exterior, holes) polygons; u is x, v is z."""
    shape = shapely.union_all([shapely.Polygon(ext, holes) for ext, holes in rings])
    x0, z0, x1, z1 = shape.bounds
    nu, nv = math.ceil((x1 - x0) / TEXEL - 1e-9), math.ceil((z1 - z0) / TEXEL - 1e-9)
    xs, zs = np.meshgrid(x0 + (np.arange(nu) + 0.5) * TEXEL, z0 + (np.arange(nv) + 0.5) * TEXEL)
    return Surface(element, "A", kind, np.array([x0, level, z0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                   np.array([0.0, normal_y, 0.0]), (nu, nv), shapely.contains_xy(shape, xs, zs), sigma)


def _sigma(arch: dict, pid: str) -> float:
    return float(arch["parameters"][pid].get("sigma") or 0.0)


def surfaces_of(arch: dict) -> list[Surface]:
    """The textured surfaces: the interior face of each wall, both faces of each partition, the floor, each ceiling.

    Walls and partitions follow the mitred footprints of blender/arch/plan.py, so a texture lands on the face the
    build produces. Face A is the front face (interior for a wall), B the back face of a partition.
    """
    frame, plan = _arch_modules()
    y_floor = frame.value(arch, "floor.level")
    walls = plan.wall_footprints(arch)
    partitions = plan.partition_plans(arch, walls)
    out = []
    for el in arch["elements"]:
        kind = el["kind"]
        if kind in ("wall", "partition"):
            geom = frame.wall_geometry(arch, el)
            n, top = geom["n"], geom["top"]
            sigma = math.hypot(_sigma(arch, el["params"]["offset"]), _sigma(arch, el["params"]["azimuth"]) * geom["length"] / 2)
            quad = walls[el["id"]] if kind == "wall" else partitions[el["id"]]["face"]
            out.append(_upright(quad[0], quad[1], y_floor, top, n, el["id"], "A", sigma))
            if kind == "partition":
                out.append(_upright(quad[3], quad[2], y_floor, top, (-n[0], -n[1]), el["id"], "B", sigma))
        elif kind == "floor":
            out.append(_horizontal(el["id"], "floor", [(el["polygon"], [])], frame.value(arch, el["params"]["level"]), 1.0,
                                    _sigma(arch, el["params"]["level"])))
        elif kind == "ceiling":
            rings = [(r["exterior"], r["holes"]) for r in el["region"]]
            out.append(_horizontal(el["id"], "ceiling", rings, frame.value(arch, el["params"]["level"]), -1.0,
                                    _sigma(arch, el["params"]["level"])))
    return out


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, np.float32) / 255.0) ** GAMMA


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(255.0 * np.clip(linear, 0.0, 1.0) ** (1.0 / GAMMA)), 0, 255).astype(np.uint8)


def _in_grown_box(points: np.ndarray, boxes: list[RoomBox]) -> np.ndarray:
    """Bool (n,): the point lies inside a RoomPlan box grown by TEX_BOX_MARGIN (a wall behind furniture is not there to see)."""
    hit = np.zeros(len(points), bool)
    for box in boxes:
        hit |= (np.abs((points - box.centre) @ box.axes) <= box.half + TEX_BOX_MARGIN).all(axis=1)
    return hit


class Observation:
    """The best candidate photos of every texel of a surface: weight, colour (sRGB bytes) and photo index, POOL slots each."""

    def __init__(self, surface: Surface, boxes: list[RoomBox]):
        self.surface = surface
        all_points = surface.points()
        usable = surface.inside.reshape(-1) & ~_in_grown_box(all_points, boxes)
        self.active = np.nonzero(usable)[0]
        self.points = all_points[self.active]
        n = len(all_points)
        self.weight = np.zeros((n, POOL), np.float32)
        self.colour = np.zeros((n, POOL, 3), np.uint8)
        self.photo = np.full((n, POOL), -1, np.int32)

    def offer(self, photo: int, texels: np.ndarray, weight: np.ndarray, colour: np.ndarray) -> None:
        """Gives each texel the candidate when it beats the weakest of its slots (each texel at most once per call)."""
        slot = self.weight[texels].argmin(axis=1)
        better = weight > self.weight[texels, slot]
        texels, slot = texels[better], slot[better]
        self.weight[texels, slot], self.colour[texels, slot], self.photo[texels, slot] = weight[better], colour[better], photo

    def colours(self) -> tuple[np.ndarray, np.ndarray]:
        """(linear colour (n, 3), seen (n,)) after matching the exposure of the photos on the texels they share."""
        filled = self.photo >= 0
        linear = _to_linear(self.colour)
        gains = _match_exposure(linear, self.photo, filled & ~(self.colour == 255).any(axis=2))
        linear = linear * gains[np.maximum(self.photo, 0)]
        order = np.argsort(-self.weight, axis=1)[:, :BEST]
        weights = np.take_along_axis(self.weight, order, axis=1)
        values = np.take_along_axis(linear, order[..., None], axis=1)
        return _weighted_median(values, weights), filled.any(axis=1)


def _match_exposure(linear: np.ndarray, photo: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Per photo and channel gain (linear light, shape (photos, 3)) that brings it to the per-texel median of the photos sharing its texels.

    Only `usable` slots count (filled and not clipped at 255, whose true value is unknown). The gains are normalised to a
    median of 1, so the overall exposure stays that of the photos, then clipped to GAIN_RANGE.
    """
    count = int(photo.max()) + 1
    gains = np.ones((max(count, 1), 3), np.float32)
    shared = usable.sum(axis=1) >= 2
    if not shared.any():
        return gains
    lin, ids, ok = linear[shared], photo[shared], usable[shared]
    for _ in range(GAIN_ROUNDS):
        adjusted = np.maximum(lin * gains[np.maximum(ids, 0)], 1e-6)
        reference = np.nanmedian(np.where(ok[..., None], adjusted, np.nan), axis=1)
        ratio = (reference[:, None, :] / adjusted)[ok]
        flat = ids[ok]
        order = np.argsort(flat, kind="stable")
        flat, ratio = flat[order], ratio[order]
        names, starts = np.unique(flat, return_index=True)
        for name, chunk in zip(names, np.split(ratio, starts[1:])):
            gains[name] = np.clip(gains[name] * np.median(chunk, axis=0), *GAIN_RANGE)
        gains[names] = np.clip(gains[names] / np.median(gains[names], axis=0), *GAIN_RANGE)
    return gains


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per channel weighted median of (n, k, 3) values with (n, k) weights; weight 0 means absent."""
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    w = np.take_along_axis(np.repeat(weights[..., None], 3, axis=2), order, axis=1)
    cumulative = np.cumsum(w, axis=1)
    pick = (cumulative >= 0.5 * cumulative[:, -1:, :] - 1e-12).argmax(axis=1)
    return np.take_along_axis(v, pick[:, None, :], axis=1)[:, 0, :]


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples (n, 3) at continuous pixel coordinates (pixel i covers [i, i + 1))."""
    h, w = image.shape[:2]
    x, y = np.clip(x - 0.5, 0, w - 1), np.clip(y - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


class _Pyramid:
    """A photo and its box-filtered reductions, loaded when first used."""

    def __init__(self, photo: PosedPhoto):
        self.photo, self.levels, self._image, self._scale = photo, {}, None, (1.0, 1.0)

    def sample(self, uv: np.ndarray, level: np.ndarray) -> np.ndarray:
        out = np.zeros((len(uv), 3), np.float32)
        for k in np.unique(level):
            if k not in self.levels:
                if self._image is None:
                    with Image.open(self.photo.path) as im:
                        self._image = im.convert("RGB")
                    self._scale = (self._image.width / self.photo.size[0], self._image.height / self.photo.size[1])  # K is in manifest pixels
                self.levels[k] = np.asarray(self._image.reduce(2 ** int(k)) if k else self._image, np.float32)
            sel = level == k
            out[sel] = _bilinear(self.levels[k], uv[sel, 0] * self._scale[0] / 2 ** k, uv[sel, 1] * self._scale[1] / 2 ** k)
        return out


class _DepthView:
    """Z-buffer of the LiDAR mesh seen by a photo."""

    def __init__(self, photo: PosedPhoto, vertices: np.ndarray, faces: np.ndarray):
        w, h = photo.size
        scale = min(1.0, TEX_DEPTH_SIDE / max(w, h))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        self.photo, self.sx, self.sy = photo, size[0] / w, size[1] / h
        K = np.diag([self.sx, self.sy, 1.0]) @ photo.K
        self.depth = raster.depth_map(vertices, faces, K, photo.w2c, size)

    def passes(self, surface: Surface, uv: np.ndarray) -> np.ndarray:
        """Bool: the mesh agrees with the surface plane at the texel's pixel and no pixel next to it shows something nearer.

        Each pixel is compared with the plane depth along its own ray, so the slope of an oblique surface (about 2 cm
        per pixel at 70 degrees) does not count as a disagreement. A pixel without mesh is no evidence against the texel.
        """
        photo, tolerance = self.photo, surface.tolerance
        h, w = self.depth.shape
        col, row = np.floor(uv[:, 0] * self.sx).astype(int), np.floor(uv[:, 1] * self.sy).astype(int)
        normal_cam = photo.w2c[:3, :3] @ surface.normal
        reach = float(surface.normal @ (surface.origin - photo.centre))
        ok = np.ones(len(uv), bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                c, r = np.clip(col + dc, 0, w - 1), np.clip(row + dr, 0, h - 1)
                x, y = (c + 0.5) / self.sx, (r + 0.5) / self.sy
                slope = normal_cam[0] * (x - photo.K[0, 2]) / photo.K[0, 0] + normal_cam[1] * (y - photo.K[1, 2]) / photo.K[1, 1] + normal_cam[2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    plane = reach / slope
                mesh = self.depth[r, c]
                nearer = mesh < plane - tolerance
                ok &= ~nearer if (dr or dc) else (np.isinf(mesh) | (np.abs(mesh - plane) < tolerance))
        return ok


def observe(surfaces: list[Surface], photos: list[PosedPhoto], mesh: tuple[np.ndarray, np.ndarray],
            boxes: list[RoomBox]) -> dict[tuple[str, str], Observation]:
    """Candidate photos of every texel of `surfaces` (see the module docstring); `mesh` is (vertices, faces) in the room frame."""
    vertices, faces = np.asarray(mesh[0], float), np.asarray(mesh[1], np.int64)
    observations = {s.key: Observation(s, boxes) for s in surfaces}
    for index, photo in enumerate(photos):
        w, h = photo.size
        views = {}
        for key, obs in observations.items():
            uv, depth = project(photo, obs.points)
            to_camera = photo.centre - obs.points
            distance = np.maximum(np.linalg.norm(to_camera, axis=1), 1e-9)
            cos = (to_camera @ obs.surface.normal) / distance
            ok = ((depth > MIN_DEPTH) & (cos > GRAZING_COS) & (uv[:, 0] > EDGE_MARGIN * w) & (uv[:, 0] < (1 - EDGE_MARGIN) * w)
                  & (uv[:, 1] > EDGE_MARGIN * h) & (uv[:, 1] < (1 - EDGE_MARGIN) * h))
            if ok.any():
                views[key] = (np.nonzero(ok)[0], uv, depth, cos, distance)
        if not views:
            continue
        zbuffer, pyramid = _DepthView(photo, vertices, faces), _Pyramid(photo)
        for key, (sel, uv, depth, cos, distance) in views.items():
            obs = observations[key]
            keep = zbuffer.passes(obs.surface, uv[sel])
            sel = sel[keep]
            if boxes and len(sel):
                sel = sel[~occluded_mask(photo.centre, obs.points[sel], boxes, TEX_BOX_MARGIN)]
            if not len(sel):
                continue
            footprint = TEXEL * photo.K[0, 0] / depth[sel]
            level = np.clip(np.rint(np.log2(np.maximum(footprint, 1.0))), 0, MAX_LEVEL).astype(int)
            obs.offer(index, obs.active[sel], cos[sel] / distance[sel] ** 2, _round(pyramid.sample(uv[sel], level)))
    return observations


def _round(colour: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(colour), 0, 255).astype(np.uint8)


@dataclass
class Fill:
    """CC0 fill tiles per role (wall, floor, ceiling), as linear RGB; None means flat colour."""
    tiles: dict[str, np.ndarray | None]
    sources: dict[str, str]
    note: str | None = None

    @classmethod
    def flat(cls, note: str | None = None) -> "Fill":
        return cls({}, {role: "flat" for role in DEFAULT_FILL}, note)

    @classmethod
    def from_image(cls, path: Path) -> "Fill":
        tile = _load_tile(Path(path))
        return cls({role: tile for role in DEFAULT_FILL}, {role: Path(path).name for role in DEFAULT_FILL})

    @classmethod
    def load(cls, arch: dict, directory: Path) -> "Fill":
        """The fill for each role, from ROOM_MODEL_FILL ("none", or an image) or downloaded once into `directory`.

        Assets come from arch["materials"] ({role: "site:asset"}) over DEFAULT_FILL. A failed download leaves that
        role and every role after it flat, with the reason in `note`.
        """
        setting = os.environ.get("ROOM_MODEL_FILL")
        if setting == "none":
            return cls.flat("fill disabled by ROOM_MODEL_FILL")
        if setting:
            return cls.from_image(Path(setting))
        wanted = {**DEFAULT_FILL, **{k: v for k, v in (arch.get("materials") or {}).items() if k in DEFAULT_FILL}}
        tiles, sources, notes = {}, {}, []
        for asset in dict.fromkeys(wanted.values()):
            slug = re.sub(r"\W+", "_", asset)
            tile = None
            if not notes:  # one failure means no network: the other assets are not tried
                try:
                    tile = _fetched(asset, slug, Path(directory))
                except Exception as exc:  # noqa: BLE001 - any failure of the download leaves the surface a flat colour
                    notes.append(f"{asset}: {exc}")
            for role, wanted_asset in wanted.items():
                if wanted_asset == asset:
                    tiles[role], sources[role] = tile, asset if tile is not None else "flat"
        return cls(tiles, sources, "download failed: " + "; ".join(notes) if notes else None)

    def render(self, role: str, size: tuple[int, int], target: np.ndarray) -> np.ndarray:
        """Linear fill (nv, nu, 3) for a surface of `size` = (nu, nv): the tile, tinted so that its mean is `target`."""
        nu, nv = size
        tile = self.tiles.get(role)
        if tile is None:
            return np.broadcast_to(target.astype(np.float32), (nv, nu, 3)).copy()
        tinted = np.clip(tile * (target / np.maximum(tile.mean(axis=(0, 1)), 1e-6)), 0.0, 1.0)
        side = len(tinted)
        return tinted[np.arange(nv)[:, None] % side, np.arange(nu)[None, :] % side]


def _load_tile(path: Path) -> np.ndarray:
    """A fill image reduced to FILL_TILE_M / TEXEL texels per side, as linear RGB."""
    side = round(FILL_TILE_M / TEXEL)
    with Image.open(path) as im:
        return _to_linear(np.asarray(im.convert("RGB").resize((side, side), Image.Resampling.BOX)))


def _fetched(asset: str, slug: str, directory: Path) -> np.ndarray:
    """The diffuse map of `asset`, downloaded into `directory` by the project's fetch script unless already there."""
    path = directory / f"{slug}_diff.jpg"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        script = os.environ.get("ROOM_MODEL_FETCH_TEXTURES", str(FETCH_SCRIPT))
        proc = subprocess.run([sys.executable, script, "get", str(directory), f"{slug}={asset}"], capture_output=True, text=True,
                              timeout=FETCH_TIMEOUT)
        if proc.returncode != 0 or not path.exists():
            lines = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError(lines[-1] if lines else "no file written")
    return _load_tile(path)


@dataclass
class Painting:
    rgb: np.ndarray  # (nv, nu, 3) uint8, row j along v from the origin
    coverage: float  # share of the surface texels that some photo sees
    fill: str  # source of the fill ("flat" when none)
    note: str | None = None


def _close_small_holes(colour: np.ndarray, observed: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(colour, observed) with every hole of at most SMALL_HOLE texels taking the colour of its nearest observed texel."""
    labels, count = ndimage.label(inside & ~observed)
    if not count:
        return colour, observed
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, np.arange(1, count + 1))
    small = np.isin(labels, np.nonzero(sizes <= SMALL_HOLE)[0] + 1)
    if not small.any():
        return colour, observed
    rows, cols = ndimage.distance_transform_edt(~observed, return_distances=False, return_indices=True)
    return np.where(small[..., None], colour[rows, cols], colour), observed | small


def paint(surface: Surface, observation: Observation, fill: Fill) -> Painting | None:
    """The texture: observed colours, fill in the holes, faded over FEATHER texels; None when no photo saw the surface."""
    nu, nv = surface.size
    colour, seen = observation.colours()
    inside = surface.inside
    observed = (seen.reshape(nv, nu)) & inside
    if not observed.any():
        return None
    colour = colour.reshape(nv, nu, 3)
    median = np.median(colour[observed], axis=0)
    role = surface.kind
    colour, closed = _close_small_holes(colour, observed, inside)
    holes = inside & ~closed
    alpha = closed.astype(np.float32)
    if holes.any():
        alpha = np.where(closed, np.clip(ndimage.distance_transform_edt(~holes) / FEATHER, 0.0, 1.0), 0.0).astype(np.float32)
    mixed = alpha[..., None] * colour + (1 - alpha[..., None]) * fill.render(role, surface.size, median)
    return Painting(_to_srgb(mixed), float(observed.sum() / inside.sum()), fill.sources.get(role, "flat"), fill.note if fill.sources.get(role) == "flat" else None)


def _entry(surface: Surface, painting: Painting, relative: str) -> dict:
    out = {"image": relative, "origin": [float(v) for v in surface.origin], "u_axis": [float(v) for v in surface.u_axis],
           "v_axis": [float(v) for v in surface.v_axis], "normal": [float(v) for v in surface.normal], "texel": TEXEL, "size": list(surface.size),
           "coverage": round(painting.coverage, 4), "fill": painting.fill}
    if painting.note:
        out["fill_note"] = painting.note
    return out


def run(ctx: StageContext) -> Path:
    arch = json.loads(require(ctx, "architecture").read_text(encoding="utf-8"))
    w2r = colmajor(arch["frame"]["world_to_room"], 4)
    photos = load_posed_photos(ctx, w2r)
    if not photos:
        raise NoPhotos(f"room {ctx.room} has no photos to project")
    mesh = load_mesh_ply(mesh_path(ctx.capture, ctx.room))
    vertices = mesh["vertices"] @ w2r[:3, :3].T + w2r[:3, 3]
    boxes = room_boxes(load_roomplan(ctx.capture, ctx.room), w2r)
    surfaces = surfaces_of(arch)
    observed = observe(surfaces, photos, (vertices, mesh["faces"]), boxes)
    path = output_path(ctx, "textures")
    folder = path.parent
    folder.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # first: an interrupted run must not leave an index naming deleted images
    for old in folder.glob("*.png"):
        old.unlink()
    fill = Fill.load(arch, folder / "fill")
    doc, skipped = {}, []
    for surface in surfaces:
        painting = paint(surface, observed[surface.key], fill)
        if painting is None:
            skipped.append(f"{surface.element}/{surface.face}")
            continue
        name = f"{surface.element}_{surface.face}.png"
        Image.fromarray(np.ascontiguousarray(painting.rgb[::-1])).save(folder / name, optimize=True)
        doc.setdefault(surface.element, {})[surface.face] = _entry(surface, painting, f"textures/{name}")
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    coverage = sorted(e["coverage"] for faces in doc.values() for e in faces.values())
    print(f"textures: {len(coverage)} surfaces, coverage min {coverage[0]:.0%} median {coverage[len(coverage) // 2]:.0%}" if coverage
          else "textures: no surface was seen by a photo")
    if skipped:
        print(f"textures: no photo sees {', '.join(skipped)} (named material kept)")
    return path
