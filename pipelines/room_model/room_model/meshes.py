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


def sheet_mesh(parts: list[dict], y: float, up: bool) -> Mesh:
    """Horizontal sheets of `parts` (`{"exterior", "holes", "drop"}`), each `drop` m below the level y."""
    meshes = [_horizontal(shapely.Polygon(p["exterior"], p["holes"]), y - p["drop"], up) for p in parts]
    sizes = np.cumsum([0] + [len(m[0]) for m in meshes])
    return np.vstack([m[0] for m in meshes]), np.vstack([m[1] + int(o) for m, o in zip(meshes, sizes)]).astype(np.uint32)


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


def wall_mesh(start, end, normal, thickness: float, y0: float, y1: float, extend_end: bool = True) -> Mesh:
    """Wall body behind the interior face, extended by its thickness past `end` (exterior corners meet without overlap)
    unless `extend_end` is False (a free-standing partition)."""
    start, end, normal = (np.asarray(v, float) for v in (start, end, normal))
    t = (end - start) / np.linalg.norm(end - start)
    a, b = start, end + t * thickness if extend_end else end
    return prism(np.array([a, b, b - normal * thickness, a - normal * thickness]), y0, y1)


NEAR_PARALLEL = float(np.sin(np.radians(10.0)))  # |sin| of the turn under which two walls count as parallel
JOIN_TOL, CORNER_SHIFT = 0.02, 2.0  # walls join when one's end is within JOIN_TOL m of the next start; an intersection further than CORNER_SHIFT thicknesses off is cut at the bisector


def _convex(poly: np.ndarray) -> bool:
    edges = np.roll(poly, -1, axis=0) - poly
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    cross = cross[np.abs(cross) > 1e-12]
    return len(cross) > 0 and bool((cross > 0).all() or (cross < 0).all())


def mitred_wall_footprints(walls: list[dict]) -> list[np.ndarray]:
    """Plan footprints [start face, end face, end back, start back] of walls mitred where one ends on the next one's start.

    A wall is {"start", "end", "normal", "offset", "thickness"}: raw corners, unit interior normal, face line n.p = offset.
    Test-side twin of the Blender scene builder (blender/arch/plan.py); the GLB export keeps `wall_mesh`.
    """
    lines = []
    for w in walls:
        n, t = np.asarray(w["normal"], float), float(w["thickness"])
        u = np.array([-n[1], n[0]])
        lines.append({"n": n, "u": u if u @ (np.asarray(w["end"]) - np.asarray(w["start"])) >= 0 else -u,
                      "o": float(w["offset"]), "t": t})

    def meet(a, b, back):
        oa, ob = a["o"] - back * a["t"], b["o"] - back * b["t"]
        return np.linalg.solve(np.array([a["n"], b["n"]]), [oa, ob])

    def along(line, back, point, d):
        if abs(line["n"] @ d) < 0.2 * np.linalg.norm(d):
            return None
        return point + d * ((line["o"] - back * line["t"] - line["n"] @ point) / (line["n"] @ d))

    def joint(la, lb, corner):
        if abs(la["u"][0] * lb["u"][1] - la["u"][1] * lb["u"][0]) >= NEAR_PARALLEL:
            inner, outer = meet(la, lb, 0), meet(la, lb, 1)
            limit = CORNER_SHIFT * max(la["t"], lb["t"])
            if np.linalg.norm(inner - corner) <= limit and np.linalg.norm(outer - corner) <= limit + la["t"] + lb["t"]:
                return (inner, outer), (inner, outer)
        if la["n"] @ lb["n"] <= 0:
            return None
        d = la["n"] + lb["n"]
        cuts = [tuple(along(line, back, corner, d) for back in (0, 1)) for line in (la, lb)]
        if any(p is None for cut in cuts for p in cut):
            return None
        spans = [sorted(float(la["n"] @ p) for p in cut) for cut in cuts]
        if max(spans[0][0], spans[1][0]) > min(spans[0][1], spans[1][1]):  # a gap between the two bodies: they meet along one edge
            points = [p for cut in cuts for p in cut]
            edge = (max(points, key=lambda p: la["n"] @ p), min(points, key=lambda p: la["n"] @ p))
            return edge, edge
        return tuple(cuts)

    cuts = [{} for _ in walls]
    for i, a in enumerate(walls):
        for j, b in enumerate(walls):
            if i == j or np.linalg.norm(np.asarray(a["end"]) - np.asarray(b["start"])) > JOIN_TOL:
                continue
            corner = (np.asarray(a["end"], float) + np.asarray(b["start"], float)) / 2
            found = joint(lines[i], lines[j], corner)
            if found is not None:
                cuts[i]["end"], cuts[j]["start"] = found
    out = []
    for w, line, cut in zip(walls, lines, cuts):
        def free(point, line=line, back=False):
            p = np.asarray(point, float)
            return p + line["n"] * (line["o"] - back * line["t"] - line["n"] @ p)

        start, end = cut.get("start", (free(w["start"]), free(w["start"], back=True))), cut.get("end", (free(w["end"]), free(w["end"], back=True)))
        quad = np.array([start[0], end[0], end[1], start[1]])
        if not _convex(quad):  # a wall too short for its mitres: square ends
            quad = np.array([free(w["start"]), free(w["end"]), free(w["end"], back=True), free(w["start"], back=True)])
        out.append(quad)
    return out


def beam_mesh(axis: list, width: float, y0: float, y1: float) -> Mesh:
    a, b = np.asarray(axis, float)
    t = (b - a) / np.linalg.norm(b - a)
    side = np.array([-t[1], t[0]]) * width / 2
    return prism(np.array([a - side, b - side, b + side, a + side]), y0, y1)


def column_mesh(cx: float, cz: float, sx: float, sz: float, y0: float, y1: float) -> Mesh:
    hx, hz = sx / 2, sz / 2
    return prism(np.array([[cx - hx, cz - hz], [cx + hx, cz - hz], [cx + hx, cz + hz], [cx - hx, cz + hz]]), y0, y1)
