"""Synthetic house-capture/5 folder with known truth (L-shaped room, beam, columns, clutter)."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

UP = np.array([0.0, 1.0, 0.0])
OUTLINE = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.5), (2.5, 2.5), (2.5, 4.5), (0.0, 4.5)]
WALL_NAMES = ["south", "east", "step", "wing_east", "north", "west"]
WALL_OFFSETS = [0.02, -0.015, 0.03, -0.02, 0.01, -0.025]
WALL_TURN_DEG = 0.8
WALL_HEIGHT = 2.45
ROOMPLAN_OFFSET = 0.20  # m an `offset_wall` is moved into the room in RoomPlan, away from the mesh
THIN_WALL_TURN_DEG = 3.0  # a thin wall keeps one narrow mesh patch, turned by this angle
THIN_WALL_HALF_WIDTH = 0.125
FLOOR_OFFSET = 0.01
H_MAIN, H_WING = 2.5, 2.3
YAW, SHIFT = 0.5, (1.2, -1.45, -0.8)
CLASS_COLORS = {0: (90, 90, 90), 1: (200, 200, 190), 2: (120, 100, 80), 3: (230, 230, 230),
                4: (150, 90, 40), 7: (60, 90, 170)}


@dataclass
class SyntheticOptions:
    seed: int = 0
    noise_m: float = 0.004
    cell_m: float = 0.05
    split_south_wall: bool = False
    extra_roomplan_wall: bool = False
    drop_wall_faces: str | None = None
    ceiling_hole: bool = False
    other_room_panel: bool = True
    thin_wall: str | None = None
    corner_column: bool = False
    step_column: bool = False
    short_beam: bool = False
    ceiling_patch: bool = False
    patch_drop: float = 0.032
    patch_length: float = 1.0
    short_beam_underside: float = 2.00
    short_beam_width: float = 0.30
    suspended_tubes: bool = False
    wall_duct: bool = False
    offset_wall: str | None = None


@dataclass
class SyntheticTruth:
    world_from_local: np.ndarray
    wall_ident: dict[str, list[str]]
    split_point: list[float]
    widths: dict[tuple[str, str], float]
    ceilings: list[float]
    beam: dict
    columns: list[dict]
    outline: list[tuple[float, float]] = field(default_factory=lambda: list(OUTLINE))
    extra_ident: str | None = None


BEAM = {"x": (3.85, 4.15), "z": (0.0, 2.5), "underside": 2.20, "top": 2.50, "width": 0.30}
COLUMNS = [
    {"x": (1.30, 1.70), "z": (1.00, 1.40), "top": 2.50, "engaged": None},
    {"x": (5.00, 5.50), "z": (0.00, 0.25), "top": 2.50, "engaged": "south"},
]
CORNER_COLUMN = {"x": (0.0, 0.35), "z": (0.0, 0.30), "top": 2.50, "engaged": "south", "flush": ("x0",)}
CEILING_PATCH = {"x0": 2.00, "z": (1.50, 2.40)}  # a slightly lower flat ceiling panel
SHORT_BEAM = {"x": (4.90, 5.20), "z": (0.50, 1.90), "top": 2.50}  # hangs free from the main ceiling
TUBES = {"z": [2.10, 2.10, 2.10, 0.40, 0.40], "x0": [0.30, 1.80, 4.40, 0.50, 2.00], "length": 1.20, "width": 0.08,
         "drop": 0.25, "height": 0.06}  # light tubes hanging on rods: no side face reaches the ceiling
DUCT = {"x": 0.145, "z": (0.30, 2.30), "diameter": 0.25, "gap": 0.05}  # octagonal stand-in for a round duct along the west wall
STEP_COLUMN = {"x": (2.15, 2.50), "z": (2.50, 2.85), "top": H_WING, "engaged": "wing_east", "flush": ("x1",)}
TABLE = {"x": (0.0, 0.6), "z": (0.5, 1.7), "h": 0.75}
CABINET = {"x": (0.0, 0.6), "z": (3.0, 4.0), "h": 1.90}
DOOR = {"z": 1.25, "w": 0.90, "h": 2.05}


def _short_beam_x(opt: SyntheticOptions) -> tuple[float, float]:
    return (SHORT_BEAM["x"][0], SHORT_BEAM["x"][0] + opt.short_beam_width)


def _patch_x(opt: SyntheticOptions) -> tuple[float, float]:
    return (CEILING_PATCH["x0"], CEILING_PATCH["x0"] + opt.patch_length)


def _columns(opt: SyntheticOptions) -> list[dict]:
    return COLUMNS + ([CORNER_COLUMN] if opt.corner_column else []) + ([STEP_COLUMN] if opt.step_column else [])


def _world_from_local() -> np.ndarray:
    c, s = np.cos(YAW), np.sin(YAW)
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = SHIFT
    return m


def _rot_y(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _ident(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-room/{name}")).upper()


def _flat(m: np.ndarray) -> list[float]:
    return [float(v) for v in m.T.ravel()]


def _in_rects(u, v, rects):
    out = np.zeros(np.shape(u), bool)
    for (u0, u1), (v0, v1) in rects:
        out |= (u > u0) & (u < u1) & (v > v0) & (v < v1)
    return out


class _Mesh:
    def __init__(self, cell: float, noise: float, rng):
        self.cell, self.noise, self.rng = cell, noise, rng
        self.verts, self.normals, self.inroom, self.faces, self.cls = [], [], [], [], []
        self.n_verts = 0

    def rect(self, origin, U, V, normal, cls, u_range, v_range, keep=None, inroom=1):
        """Grid of two-triangle cells on the plane origin + u*U + v*V, cells aligned to multiples of the cell size."""
        c = self.cell
        i0, i1 = round(u_range[0] / c), round(u_range[1] / c)
        j0, j1 = round(v_range[0] / c), round(v_range[1] / c)
        ci, cj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        mask = np.ones(ci.shape, bool) if keep is None else keep((ci + 0.5) * c, (cj + 0.5) * c)
        if not mask.any():
            return
        used = np.zeros((i1 - i0 + 1, j1 - j0 + 1), bool)
        for di in (0, 1):
            for dj in (0, 1):
                used[di:di + mask.shape[0], dj:dj + mask.shape[1]] |= mask
        index = np.full(used.shape, -1, np.int64)
        index[used] = np.arange(used.sum()) + self.n_verts
        gi, gj = np.nonzero(used)
        uv = np.stack([(gi + i0) * c, (gj + j0) * c], 1)
        normal = np.asarray(normal, float)
        pts = origin + uv[:, :1] * U + uv[:, 1:] * V
        pts = pts + self.rng.normal(0.0, self.noise, len(pts))[:, None] * normal
        a = index[:-1, :-1][mask]
        b = index[1:, :-1][mask]
        d = index[1:, 1:][mask]
        e = index[:-1, 1:][mask]
        flip = np.dot(np.cross(U, V), normal) < 0
        t1, t2 = ((a, d, b), (a, e, d)) if flip else ((a, b, d), (a, d, e))
        tri = np.concatenate([np.stack(t1, 1), np.stack(t2, 1)])
        self.verts.append(pts)
        self.normals.append(np.tile(normal, (len(pts), 1)))
        self.inroom.append(np.full(len(pts), inroom, np.uint8))
        self.faces.append(tri)
        self.cls.append(np.full(len(tri), cls, np.uint8))
        self.n_verts += len(pts)


def _build_mesh(opt: SyntheticOptions) -> _Mesh:
    m = _Mesh(opt.cell_m, opt.noise_m, np.random.default_rng(opt.seed))
    X, Y, Z = np.eye(3)
    o = np.zeros(3)
    columns = _columns(opt)
    col_rects = [(c["x"], c["z"]) for c in columns]
    furniture = [(TABLE["x"], TABLE["z"]), (CABINET["x"], CABINET["z"])]
    beam_xy = (BEAM["x"], (BEAM["underside"], BEAM["top"]))
    drop = opt.drop_wall_faces

    def wall(name, origin, U, normal, u_range, v_top, excl=()):
        if drop == name:
            return
        if opt.thin_wall == name:
            mid = (u_range[0] + u_range[1]) / 2
            R = _rot_y(np.radians(THIN_WALL_TURN_DEG))
            m.rect(origin + mid * U, R @ U, Y, R @ normal, 1, (-THIN_WALL_HALF_WIDTH, THIN_WALL_HALF_WIDTH), (0, v_top))
            return
        m.rect(origin, U, Y, normal, 1, u_range, (0, v_top), lambda u, v: ~_in_rects(u, v, excl))

    wall("south", o, X, Z, (0, 6), H_MAIN, [(c["x"], (0, 9)) for c in columns if c["engaged"] == "south"] + [beam_xy])
    door_z = (DOOR["z"] - DOOR["w"] / 2, DOOR["z"] + DOOR["w"] / 2)
    if drop != "east":
        m.rect(np.array([6.0, 0, 0]), Z, Y, -X, 1, (0, 2.5), (0, H_MAIN),
               lambda u, v: ~_in_rects(u, v, [(door_z, (0, DOOR["h"]))]))
    m.rect(np.array([6.05, 0, 0]), Z, Y, -X, 7, door_z, (0, DOOR["h"]))
    wall("step", np.array([0, 0, 2.5]), X, -Z, (2.5, 6), H_MAIN, [beam_xy])
    m.rect(np.array([0, 0, 2.5]), X, Y, -Z, 1, (0, 2.5), (H_WING, H_MAIN))
    wall("wing_east", np.array([2.5, 0, 0]), Z, -X, (2.5, 4.5), H_WING)
    wall("north", np.array([0, 0, 4.5]), X, -Z, (0, 2.5), H_WING)
    if drop != "west":
        behind = [(TABLE["z"], (0, TABLE["h"])), (CABINET["z"], (0, CABINET["h"]))]
        behind += [(c["z"], (0, 9)) for c in columns if "x0" in c.get("flush", ())]
        m.rect(o, Z, Y, X, 1, (0, 4.5), (0, H_MAIN),
               lambda u, v: (v < np.where(u < 2.5, H_MAIN, H_WING)) & ~_in_rects(u, v, behind))

    for c in columns:
        (x0, x1), (z0, z1) = c["x"], c["z"]
        h = c["top"]
        if "x0" not in c.get("flush", ()):
            m.rect(np.array([x0, 0, 0]), Z, Y, -X, 1, (z0, z1), (0, h))
        if "x1" not in c.get("flush", ()):
            m.rect(np.array([x1, 0, 0]), Z, Y, X, 1, (z0, z1), (0, h))
        m.rect(np.array([0, 0, z1]), X, Y, Z, 1, (x0, x1), (0, h))
        if c["engaged"] is None:
            m.rect(np.array([0, 0, z0]), X, Y, -Z, 1, (x0, x1), (0, h))

    (bx0, bx1), (bz0, bz1) = BEAM["x"], BEAM["z"]
    m.rect(np.array([bx0, 0, 0]), Z, Y, -X, 1, (bz0, bz1), (BEAM["underside"], BEAM["top"]))
    m.rect(np.array([bx1, 0, 0]), Z, Y, X, 1, (bz0, bz1), (BEAM["underside"], BEAM["top"]))
    m.rect(np.array([0, BEAM["underside"], 0]), X, Z, -Y, 3, (bx0, bx1), (bz0, bz1))

    if opt.short_beam:
        (sx0, sx1), (sz0, sz1), under = _short_beam_x(opt), SHORT_BEAM["z"], opt.short_beam_underside
        m.rect(np.array([sx0, 0, 0]), Z, Y, -X, 1, (sz0, sz1), (under, SHORT_BEAM["top"]))
        m.rect(np.array([sx1, 0, 0]), Z, Y, X, 1, (sz0, sz1), (under, SHORT_BEAM["top"]))
        m.rect(np.array([0, under, 0]), X, Z, -Y, 3, (sx0, sx1), (sz0, sz1))

    if opt.suspended_tubes:
        for z, x0 in zip(TUBES["z"], TUBES["x0"]):
            top = H_MAIN - TUBES["drop"] + TUBES["height"]
            under = H_MAIN - TUBES["drop"]
            ux, vz = (x0, x0 + TUBES["length"]), (z, z + TUBES["width"])
            m.rect(np.array([0, under, 0]), X, Z, -Y, 0, ux, vz)
            m.rect(np.array([0, top, 0]), X, Z, Y, 0, ux, vz)
            m.rect(np.array([0, 0, vz[0]]), X, Y, -Z, 0, ux, (under, top))
            m.rect(np.array([0, 0, vz[1]]), X, Y, Z, 0, ux, (under, top))

    if opt.wall_duct:
        r = DUCT["diameter"] / 2
        cy = H_MAIN - DUCT["gap"] - r
        side = DUCT["diameter"] * np.tan(np.pi / 8)
        for k in range(8):
            phi = k * np.pi / 4
            n = np.array([np.cos(phi), np.sin(phi), 0.0])
            t = np.array([-np.sin(phi), np.cos(phi), 0.0])
            centre = np.array([DUCT["x"], cy, 0.0]) + r * n
            m.rect(centre - side / 2 * t, Z, t, n, 0, DUCT["z"], (0, side))

    for box, cls in ((TABLE, 4), (CABINET, 1)):
        (x0, x1), (z0, z1), h = box["x"], box["z"], box["h"]
        m.rect(np.array([0, h, 0]), X, Z, Y, cls, (x0, x1), (z0, z1))
        m.rect(np.array([x1, 0, 0]), Z, Y, X, cls, (z0, z1), (0, h))
        m.rect(np.array([0, 0, z0]), X, Y, -Z, cls, (x0, x1), (0, h))
        m.rect(np.array([0, 0, z1]), X, Y, Z, cls, (x0, x1), (0, h))

    if opt.other_room_panel:
        m.rect(np.array([0, 0, -0.05]), X, Y, -Z, 1, (1, 3), (0, 2), inroom=0)

    solid = col_rects + furniture
    l_shape = [((0, 6), (0, 2.5)), ((0, 2.5), (2.5, 4.5))]
    m.rect(o, X, Z, Y, 2, (0, 6), (0, 4.5),
           lambda u, v: _in_rects(u, v, l_shape) & ~_in_rects(u, v, solid))
    holes = col_rects + ([(_patch_x(opt), CEILING_PATCH["z"])] if opt.ceiling_patch else []) + [(BEAM["x"], BEAM["z"])] + ([(_short_beam_x(opt), SHORT_BEAM["z"])] if opt.short_beam else [])
    if opt.ceiling_patch:
        m.rect(np.array([0, H_MAIN - opt.patch_drop, 0]), X, Z, -Y, 3, _patch_x(opt), CEILING_PATCH["z"])
    if opt.ceiling_hole:
        holes.append(((4.6, 5.6), (1.0, 2.0)))
    m.rect(np.array([0, H_MAIN, 0]), X, Z, -Y, 3, (0, 6), (0, 2.5), lambda u, v: ~_in_rects(u, v, holes))
    m.rect(np.array([0, H_WING, 0]), X, Z, -Y, 3, (0, 2.5), (2.5, 4.5))
    return m


def _write_ply(mesh: _Mesh, world: np.ndarray, path: Path) -> np.ndarray:
    verts = np.concatenate(mesh.verts) @ world[:3, :3].T + world[:3, 3]
    normals = np.concatenate(mesh.normals) @ world[:3, :3].T
    faces = np.concatenate(mesh.faces)
    face_cls = np.concatenate(mesh.cls)
    inroom = np.concatenate(mesh.inroom)
    vert_cls = np.zeros(len(verts), np.uint8)
    vert_cls[faces.ravel()] = np.repeat(face_cls, 3)
    lut = np.zeros((8, 3), np.uint8)
    for k, v in CLASS_COLORS.items():
        lut[k] = v
    rgb = lut[vert_cls]
    vdt = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
           ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("coverage", "f4"), ("inroom", "u1")]
    v = np.zeros(len(verts), vdt)
    for k, name in enumerate("xyz"):
        v[name], v["n" + name] = verts[:, k], normals[:, k]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    v["coverage"], v["inroom"] = 1.0, inroom
    f = np.zeros(len(faces), [("vertex_indices", "O"), ("class", "u1")])
    f["vertex_indices"] = list(faces.astype(np.uint32))
    f["class"] = face_cls
    PlyData(
        [PlyElement.describe(v, "vertex"),
         PlyElement.describe(f, "face", len_types={"vertex_indices": "u1"}, val_types={"vertex_indices": "u4"})],
        text=False, byte_order="<", comments=["synthetic house-capture LiDAR mesh"],
    ).write(str(path))
    path.with_name(path.stem + "-colors.bin").write_bytes(rgb.tobytes())
    return np.array([len(verts), len(faces)])


def _intersect(p1, d1, p2, d2):
    a = np.array([d1, -np.asarray(d2)]).T
    t = np.linalg.solve(a, np.asarray(p2) - np.asarray(p1))
    return np.asarray(p1) + t[0] * np.asarray(d1)


def _roomplan(opt: SyntheticOptions, world: np.ndarray) -> tuple[dict, dict[str, list[str]], str | None]:
    n_edges = len(OUTLINE)
    lines, info = [], []
    for k, name in enumerate(WALL_NAMES):
        a, b = np.array(OUTLINE[k]), np.array(OUTLINE[(k + 1) % n_edges])
        length = float(np.linalg.norm(b - a))
        d = (b - a) / length
        n = np.array([-d[1], d[0]])
        phi = np.radians(WALL_TURN_DEG) * (1 if k % 2 == 0 else -1)
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        c = (a + b) / 2 + (WALL_OFFSETS[k] + (ROOMPLAN_OFFSET if opt.offset_wall == WALL_NAMES[k] else 0.0)) * n
        lines.append((c, rot @ d))
        info.append((name, c, rot @ d, rot @ n, length))
    corners = [_intersect(*lines[k - 1], *lines[k]) for k in range(n_edges)]
    if opt.split_south_wall:
        corners.insert(1, lines[0][0])  # south centre = split point, on the shared perturbed line

    def frame_of(c_plan, d_plan, n_plan, y):
        m = np.eye(4)
        m[:3, 0] = [d_plan[0], 0, d_plan[1]]
        m[:3, 1] = UP
        m[:3, 2] = [n_plan[0], 0, n_plan[1]]
        m[:3, 3] = [c_plan[0], y, c_plan[1]]
        return world @ m

    def wall_json(ident, c, d, n, length):
        return {
            "category": {"wall": {}}, "completedEdges": [], "confidence": {"high": {}}, "curve": None,
            "dimensions": [float(length), WALL_HEIGHT, 0],
            "identifier": ident, "parentIdentifier": None,
            "polygonCorners": [[-length / 2, WALL_HEIGHT / 2, 0.0], [length / 2, WALL_HEIGHT / 2, 0.0],
                               [length / 2, -WALL_HEIGHT / 2, 0.0], [-length / 2, -WALL_HEIGHT / 2, 0.0]],
            "story": 0, "transform": _flat(frame_of(c, d, n, WALL_HEIGHT / 2)),
        }

    walls, idents = [], {}
    for name, c, d, n, length in info:
        if name == "south" and opt.split_south_wall:
            pieces = [(-length / 4, length / 2, name + "-a"), (length / 4, length / 2, name + "-b")]
            idents[name] = [_ident(p[2]) for p in pieces]
            walls += [wall_json(_ident(tag), c + off * d, d, n, ln) for off, ln, tag in pieces]
        else:
            idents[name] = [_ident(name)]
            walls.append(wall_json(_ident(name), c, d, n, length))
    extra = None
    if opt.extra_roomplan_wall:
        extra = _ident("extra")
        walls.append(wall_json(extra, np.array([1.0, 3.5]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]), 1.8))

    east = info[WALL_NAMES.index("east")]
    door_c = east[1] + (DOOR["z"] - 1.25) * east[2]
    door = {
        "category": {"door": {"isOpen": False}}, "completedEdges": [], "confidence": {"high": {}}, "curve": None,
        "dimensions": [DOOR["w"], DOOR["h"], 0], "identifier": _ident("door"),
        "parentIdentifier": idents["east"][0], "polygonCorners": [], "story": 0,
        "transform": _flat(frame_of(door_c, east[2], east[3], DOOR["h"] / 2)),
    }
    floor_local = np.array([[1, 0, 0, 0], [0, 0, 1, FLOOR_OFFSET], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    floor = {
        "category": {"floor": {}}, "completedEdges": [], "confidence": {"high": {}}, "curve": None,
        "dimensions": [6.0, 4.5, 0], "identifier": _ident("floor"), "parentIdentifier": None,
        "polygonCorners": [[float(p[0]), float(-p[1]), 0.0] for p in corners],
        "story": 0, "transform": _flat(world @ floor_local),
    }
    data = {
        "coreModel": "", "doors": [door], "floors": [floor], "objects": [], "openings": [],
        "referenceOriginTransform": _flat(np.eye(4)),
        "sections": [{"center": [0.0, 0.0, 0.0], "label": "unidentified", "story": 0}],
        "story": 0, "version": 2, "walls": walls, "windows": [],
    }
    return data, idents, extra


def make_capture(root: Path, options: SyntheticOptions | None = None) -> SyntheticTruth:
    opt = options or SyntheticOptions()
    root = Path(root)
    for sub in ("rooms", "mesh", "photos"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    world = _world_from_local()
    n_verts, n_faces = _write_ply(_build_mesh(opt), world, root / "mesh" / "room-0.ply")
    data, idents, extra = _roomplan(opt, world)
    (root / "rooms" / "room-0.json").write_text(json.dumps(data), encoding="utf-8")
    (root / "photos" / "photos.json").write_text("[]", encoding="utf-8")
    mesh_entry = {"anchors": 0, "faces": int(n_faces), "file": "mesh/room-0.ply", "vertices": int(n_verts)}
    manifest = {
        "format": "house-capture/5", "mesh": mesh_entry, "photos": "photos/photos.json", "photoCount": 0,
        "rooms": [{"index": 0, "name": "Synthetic", "file": "rooms/room-0.json", "mesh": mesh_entry,
                   "walls": len(data["walls"]), "doors": 1}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return SyntheticTruth(
        world_from_local=world, wall_ident=idents, split_point=[3.0, 0.0],
        widths={("south", "step"): 2.50, ("west", "east"): 6.00, ("west", "wing_east"): 2.50, ("south", "north"): 4.50},
        ceilings=[H_MAIN, H_WING], beam=dict(BEAM),
        columns=[{**c, "center": [sum(c["x"]) / 2, sum(c["z"]) / 2], "size": [c["x"][1] - c["x"][0], c["z"][1] - c["z"][0]]}
                 for c in _columns(opt)],
        extra_ident=extra,
    )
