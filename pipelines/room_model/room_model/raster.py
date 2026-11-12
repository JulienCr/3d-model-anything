"""Numpy z-buffer rasteriser: depth and face id of a triangle mesh seen by a pinhole camera."""
from __future__ import annotations

import numpy as np

NEAR = 0.05  # m; a triangle with a vertex nearer than this (or behind the camera) is dropped, not clipped
CHUNK = 2_000_000  # pixel candidates evaluated at once


def rasterize(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, w2c: np.ndarray,
              size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(depth, face id) images of shape (h, w): depth in metres along the optical axis (inf where empty), id -1 where empty.

    K is in continuous pixel coordinates (pixel i covers [i, i + 1)) of an image of `size` = (w, h).
    """
    w, h = size
    cam = np.asarray(vertices, float) @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe_z = np.where(z > NEAR, z, 1.0)
    uv = (cam @ np.asarray(K, float).T)[:, :2] / safe_z[:, None]
    faces = np.asarray(faces, np.int64)
    ok = (z[faces] > NEAR).all(axis=1)
    tri_u, tri_v, tri_z = uv[faces, 0], uv[faces, 1], z[faces]
    # pixel centres sit at i + 0.5: a pixel can be inside only if its centre is in the bounding box
    x0, x1 = np.ceil(tri_u.min(1) - 0.5), np.floor(tri_u.max(1) - 0.5)
    y0, y1 = np.ceil(tri_v.min(1) - 0.5), np.floor(tri_v.max(1) - 0.5)
    x0, y0, x1, y1 = np.maximum(x0, 0), np.maximum(y0, 0), np.minimum(x1, w - 1), np.minimum(y1, h - 1)
    area2 = (tri_u[:, 1] - tri_u[:, 0]) * (tri_v[:, 2] - tri_v[:, 0]) - (tri_u[:, 2] - tri_u[:, 0]) * (tri_v[:, 1] - tri_v[:, 0])
    ok &= (x1 >= x0) & (y1 >= y0) & (np.abs(area2) > 1e-12)
    index = np.nonzero(ok)[0]
    x0, y0 = x0[index].astype(np.int64), y0[index].astype(np.int64)
    bw, bh = (x1[index] - x0 + 1).astype(np.int64), (y1[index] - y0 + 1).astype(np.int64)

    depth = np.full(w * h, np.inf)
    ids = np.full(w * h, -1, np.int64)
    counts = bw * bh
    ends = np.cumsum(counts)
    start = 0
    while start < len(index):
        base = ends[start - 1] if start else 0
        stop = max(int(np.searchsorted(ends, base + CHUNK, side="right")), start + 1)
        sl = slice(start, stop)
        n = counts[sl]
        tri = np.repeat(np.arange(start, stop), n)
        local = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        px = x0[tri] + local % bw[tri]
        py = y0[tri] + local // bw[tri]
        f = index[tri]
        cu, cv = px + 0.5, py + 0.5
        au, av, bu, bv, du, dv = (tri_u[f, 0], tri_v[f, 0], tri_u[f, 1], tri_v[f, 1], tri_u[f, 2], tri_v[f, 2])
        l0 = ((bu - cu) * (dv - cv) - (du - cu) * (bv - cv)) / area2[f]
        l1 = ((du - cu) * (av - cv) - (au - cu) * (dv - cv)) / area2[f]
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        f, px, py, l0, l1, l2 = f[inside], px[inside], py[inside], l0[inside], l1[inside], l2[inside]
        d = 1.0 / (l0 / tri_z[f, 0] + l1 / tri_z[f, 1] + l2 / tri_z[f, 2])
        pix = py * w + px
        np.minimum.at(depth, pix, d)
        won = d == depth[pix]
        ids[pix[won]] = f[won]
        start = stop
    return depth.reshape(h, w), ids.reshape(h, w)


def depth_map(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, w2c: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Z-buffer in metres of shape (h, w), `inf` where no triangle is seen."""
    return rasterize(vertices, faces, K, w2c, size)[0]
