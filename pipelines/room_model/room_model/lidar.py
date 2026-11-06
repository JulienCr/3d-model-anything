"""LiDAR mesh to area-proportional point samples in the room frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from room_model.capture import load_mesh_ply, mesh_path
from room_model.context import StageContext

MIN_COMPONENT_AREA = 0.25  # m2, after welding
SAMPLE_DENSITY = 4000.0  # samples per m2 of mesh
WELD = 0.01  # m; ARKit mesh anchors are separate chunks whose borders do not share vertices


@dataclass
class Samples:
    P: np.ndarray  # (n, 3) float64, room frame
    N: np.ndarray  # (n, 3) unit face normals, room frame
    cls: np.ndarray  # (n,) ARKit face class


def triangle_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    t = v[f]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def weld_components(v: np.ndarray, f: np.ndarray, weld: float = WELD) -> tuple[np.ndarray, np.ndarray]:
    """Per-face component id and per-component area; vertices closer than `weld` count as connected."""
    used = np.unique(f.ravel())
    pairs = cKDTree(v[used]).query_pairs(weld, output_type="ndarray")
    pairs = used[pairs] if len(pairs) else np.zeros((0, 2), int)
    rows = np.r_[f[:, 0], f[:, 1], f[:, 2], pairs[:, 0]]
    cols = np.r_[f[:, 1], f[:, 2], f[:, 0], pairs[:, 1]]
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(v), len(v)))
    _, lab = connected_components(graph, directed=False)
    _, comp = np.unique(lab[f[:, 0]], return_inverse=True)
    return comp, np.bincount(comp, weights=triangle_areas(v, f))


def load_samples(ctx: StageContext, world_to_room: np.ndarray, density: float = SAMPLE_DENSITY, seed: int = 0) -> Samples:
    """Samples (about `density` per m2) of the in-room faces of the room mesh, minus small floating components."""
    mesh = load_mesh_ply(mesh_path(ctx.capture, ctx.room))
    w2r = np.asarray(world_to_room, float)
    v = mesh["vertices"] @ w2r[:3, :3].T + w2r[:3, 3]
    f, cls = mesh["faces"], mesh["face_class"]
    if mesh["inroom"] is not None:
        keep = (mesh["inroom"] == 1)[f].all(axis=1)
        f, cls = f[keep], cls[keep]
    comp, areas = weld_components(v, f)
    ok = areas[comp] >= MIN_COMPONENT_AREA
    f, cls = f[ok], cls[ok]

    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(fn, axis=1)
    fn /= np.maximum(2 * area, 1e-12)[:, None]
    vn = (mesh["normals"] @ w2r[:3, :3].T)[f].sum(axis=1)
    fn[np.einsum("ij,ij->i", fn, vn) < 0] *= -1  # winding is not trusted over the vertex normals

    rng = np.random.default_rng(seed)
    counts = np.floor(area * density + rng.random(len(f))).astype(np.int64)
    face = np.repeat(np.arange(len(f)), counts)
    r1, r2 = rng.random(len(face)), rng.random(len(face))
    fold = r1 + r2 > 1
    r1, r2 = np.where(fold, 1 - r1, r1), np.where(fold, 1 - r2, r2)
    t = tri[face]
    P = t[:, 0] + r1[:, None] * (t[:, 1] - t[:, 0]) + r2[:, None] * (t[:, 2] - t[:, 0])
    return Samples(P=P, N=fn[face], cls=cls[face].astype(np.int64))
