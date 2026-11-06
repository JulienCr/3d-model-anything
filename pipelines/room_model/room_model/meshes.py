"""Triangle meshes of the model elements, in the room frame (x, y up, z); plan points are (x, z)."""
from __future__ import annotations

import numpy as np
import shapely

Mesh = tuple[np.ndarray, np.ndarray]


def _horizontal(polygon: shapely.Polygon, y: float, up: bool) -> Mesh:
    """Polygon (holes allowed) triangulated at height y; faces point up or down."""
    if not polygon.is_valid:
        polygon = shapely.make_valid(polygon)
    verts: dict[tuple[float, float], int] = {}
    tris = []
    for part in shapely.get_parts(polygon):
        if part.geom_type != "Polygon" or part.is_empty:
            continue
        for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
            pts = np.array(tri.exterior.coords[:3])
            if (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0]) < 0:
                pts = pts[::-1]  # positive (x, z) area: the normal points to -y
            if up:
                pts = pts[::-1]
            tris.append([verts.setdefault((round(float(p[0]), 9), round(float(p[1]), 9)), len(verts)) for p in pts])
    xz = np.array(list(verts), float).reshape(-1, 2)
    return np.column_stack([xz[:, 0], np.full(len(xz), y), xz[:, 1]]), np.array(tris, np.uint32).reshape(-1, 3)


def floor_mesh(polygon: list, y: float) -> Mesh:
    return _horizontal(shapely.Polygon(polygon), y, up=True)


def ceiling_mesh(region: list[dict], y: float) -> Mesh:
    parts = [shapely.Polygon(r["exterior"], r["holes"]) for r in region]
    return _horizontal(shapely.MultiPolygon(parts) if len(parts) > 1 else parts[0], y, up=False)


def prism(quad: np.ndarray, y0: float, y1: float) -> Mesh:
    """Vertical box over a convex plan quad (4 points), faces pointing outwards."""
    quad = np.asarray(quad, float)
    pos = np.vstack([np.column_stack([quad[:, 0], np.full(4, y0), quad[:, 1]]),
                     np.column_stack([quad[:, 0], np.full(4, y1), quad[:, 1]])])
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)]
    for k in range(4):
        a, b = k, (k + 1) % 4
        faces += [(a, b, b + 4), (a, b + 4, a + 4)]
    centre = pos.mean(0)
    tris = []
    for f in faces:
        p = pos[list(f)]
        if np.cross(p[1] - p[0], p[2] - p[0]) @ (p.mean(0) - centre) < 0:
            f = (f[0], f[2], f[1])
        tris.append(f)
    return pos, np.array(tris, np.uint32)


def wall_mesh(start, end, normal, thickness: float, y0: float, y1: float) -> Mesh:
    """Wall body behind the interior face, extended by its thickness at both ends."""
    start, end, normal = (np.asarray(v, float) for v in (start, end, normal))
    t = (end - start) / np.linalg.norm(end - start)
    a, b = start - t * thickness, end + t * thickness
    return prism(np.array([a, b, b - normal * thickness, a - normal * thickness]), y0, y1)


def beam_mesh(axis: list, width: float, y0: float, y1: float) -> Mesh:
    a, b = np.asarray(axis, float)
    t = (b - a) / np.linalg.norm(b - a)
    side = np.array([-t[1], t[0]]) * width / 2
    return prism(np.array([a - side, b - side, b + side, a + side]), y0, y1)


def column_mesh(cx: float, cz: float, sx: float, sz: float, y0: float, y1: float) -> Mesh:
    hx, hz = sx / 2, sz / 2
    return prism(np.array([[cx - hx, cz - hz], [cx + hx, cz - hz], [cx + hx, cz + hz], [cx - hx, cz + hz]]), y0, y1)
