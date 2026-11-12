"""Ceiling zones: the interior polygon cut along partitions, beams and interior jogs, one LiDAR level per zone."""
from __future__ import annotations

import numpy as np
import shapely
from scipy import ndimage

from room_model import ceilings, shell

CUT_REACH = shell.THICKNESS  # m: a cut ending this close to the boundary reaches it (RoomPlan ends stop inside the wall)
MIN_SHARED_EDGE = 1e-3  # m of common boundary that makes two zones neighbours; a shared corner point does not
GRID = 1e-9  # m, precision at which cuts and walls are noded
MIN_FEATURE = 1e-3  # m: the LiDAR does not resolve a ceiling feature thinner than 2 mm
FAR = 1e3  # m, length of the ray drawn from a reflex vertex before it is clipped to the polygon


def _extended(a: np.ndarray, b: np.ndarray, reach: float) -> shapely.LineString:
    d = (b - a) / np.linalg.norm(b - a)
    return shapely.LineString([a - reach * d, b + reach * d])


def jog_cuts(polygon: shapely.Polygon) -> list[shapely.LineString]:
    """Both edges meeting at each reflex vertex, extended into the interior up to the opposite boundary.

    A vertex that turns by less than shell.NEAR_PARALLEL_DEG is a near-straight continuation, not a jog.
    """
    ring = shapely.LinearRing(polygon.exterior.coords)
    pts = np.array(ring.coords)[:-1]
    sign = 1.0 if ring.is_ccw else -1.0
    cuts = []
    for k, v in enumerate(pts):
        a, b = v - pts[k - 1], pts[(k + 1) % len(pts)] - v
        cross = (a[0] * b[1] - a[1] * b[0]) * sign
        if cross >= -np.linalg.norm(a) * np.linalg.norm(b) * np.sin(np.radians(shell.NEAR_PARALLEL_DEG)):
            continue
        for d in (a / np.linalg.norm(a), -b / np.linalg.norm(b)):
            inside = polygon.intersection(shapely.LineString([v, v + FAR * d]))
            touching = [g for g in shapely.get_parts(inside) if g.geom_type == "LineString" and g.distance(shapely.Point(v)) < 1e-9]
            cuts += touching
    return cuts


def split(polygon: shapely.Polygon, segments: list[tuple[np.ndarray, np.ndarray]]) -> list[shapely.Polygon]:
    """The polygon cut by the segments (each reaching the boundary when it stops within CUT_REACH of it) and by the
    extensions of its reflex edges; a cut that does not run boundary to boundary splits nothing."""
    lines = list(jog_cuts(polygon))
    for a, b in segments:
        lines += [g for g in shapely.get_parts(polygon.intersection(_extended(np.asarray(a), np.asarray(b), CUT_REACH)))
                  if g.geom_type == "LineString" and not g.is_empty]
    noded = shapely.union_all([polygon.exterior, *lines], grid_size=GRID)  # float noise must not leave a cut short of the wall
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(noded)))
    return [f for f in faces if polygon.contains(f.representative_point())]


def _neighbours(a: shapely.Geometry, b: shapely.Geometry) -> bool:
    return a.intersection(b).length > MIN_SHARED_EDGE


def _absorb_small(cells: list[shapely.Polygon]) -> list[shapely.Polygon]:
    """Zones under MIN_LEVEL_AREA cannot carry a level (the detection floor): each joins the neighbour it shares the longest
    edge with. This closes the slivers between two almost coincident cuts, such as a partition on a jog's extension."""
    cells = list(cells)
    while len(cells) > 1:
        small = min(cells, key=lambda c: c.area)
        if small.area >= ceilings.MIN_LEVEL_AREA:
            break
        shared = [(small.intersection(c).length, c) for c in cells if c is not small]
        length, host = max(shared, key=lambda lc: lc[0])
        if length <= MIN_SHARED_EDGE:
            cells.remove(small)
            continue
        cells = [c for c in cells if c is not small and c is not host] + [shapely.union_all([small, host])]
    return cells


def _dominant(points: np.ndarray) -> tuple[dict | None, list[dict]]:
    """The best supported level of the zone's ceiling points, and the other levels found in them."""
    fits = ceilings.level_fits(points)
    if not fits:
        return None, []
    best = max(fits, key=lambda f: f["support_m2"])
    return best, [f for f in fits if f is not best]


def _merge_measured(groups: list[dict]) -> list[dict]:
    """Merge neighbouring measured groups whose levels `ceilings.can_merge`, closest pair first."""
    while True:
        pairs = []
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                hi, lo = (a, b) if a["level"] >= b["level"] else (b, a)
                if ceilings.can_merge(hi, lo) and _neighbours(a["geom"], b["geom"]):
                    pairs.append((hi["level"] - lo["level"], a, b, hi, lo))
        if not pairs:
            return groups
        _, a, b, hi, lo = min(pairs, key=lambda p: p[0])
        merged = {**ceilings.combine(hi, lo), "geom": shapely.union_all([a["geom"], b["geom"]]), "assumed": False,
                  "others": a["others"] + b["others"]}
        groups = [g for g in groups if g is not a and g is not b] + [merged]


def _opened(geom: shapely.Geometry) -> shapely.Geometry:
    """Without the necks and spikes thinner than 2 * MIN_FEATURE that a beam's flat end leaves along a slanted wall."""
    return geom.buffer(-MIN_FEATURE, join_style="mitre").buffer(MIN_FEATURE, join_style="mitre").simplify(GRID * 1e3)


def _merge_assumed(groups: list[dict]) -> list[dict]:
    """Neighbouring assumed zones that took the same level are one ceiling."""
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(groups):
            twin = next((b for b in groups[i + 1:] if a["assumed"] and b["assumed"] and a["level"] == b["level"]
                         and _neighbours(a["geom"], b["geom"])), None)
            if twin is not None:
                a["geom"] = shapely.union_all([a["geom"], twin["geom"]])
                groups = [g for g in groups if g is not twin]
                merged = True
                break
    return groups


def _minority_regions(cell: shapely.Polygon, pts: np.ndarray, best: dict, other: dict) -> list[shapely.Polygon]:
    """Plan regions of the samples that sit on `other` rather than on the zone's level `best`, as clean polygons.

    The occupied cells are closed like the scan regions, their strips (ducts, light tubes) are left out, holes are
    filled, and an edge that stops short of the zone boundary by the sampling margin is carried onto it.
    """
    dy_other, dy_best = np.abs(pts[:, 1] - other["level"]), np.abs(pts[:, 1] - best["level"])
    grid = ceilings._Grid(cell)
    labels, count = ndimage.label(grid.occupancy(pts[(dy_other < dy_best) & (dy_other < ceilings.LEVEL_WINDOW)][:, [0, 2]]),
                                  structure=np.ones((3, 3)))
    reach = ceilings.POLYGON_MARGIN + 2 * ceilings.GRID
    out = []
    for comp in range(1, count + 1):
        mask = labels == comp
        if mask.sum() * ceilings.GRID**2 < ceilings.MIN_COMPONENT_AREA or ceilings._is_strip(ceilings._component_frame(grid.centres(mask))):
            continue
        biggest = max(ceilings._as_multipolygon(grid.polygon(mask)).geoms, key=lambda g: g.area)
        poly = shapely.Polygon(biggest.exterior).simplify(ceilings.SIMPLIFY)
        poly = poly.union(poly.buffer(reach).intersection(cell.exterior.buffer(reach)))
        poly = ceilings._as_multipolygon(_opened(poly.intersection(cell)))
        out += [g for g in poly.geoms if g.area >= ceilings.MIN_COMPONENT_AREA and not g.interiors]
    return out


def _split_zone(cell: shapely.Polygon, pts: np.ndarray, best: dict, others: list[dict], notes: list[dict]) -> list[shapely.Polygon]:
    """The zone cut along each other level that is a ceiling of its own (MIN_LEVEL_AREA of support, beyond
    MAX_MERGE_SPAN of the zone level); a level that cannot be made a clean region is noted and the zone keeps one level."""
    pieces = [cell]
    for other in others:
        if other["support_m2"] < ceilings.MIN_LEVEL_AREA or abs(other["level"] - best["level"]) <= ceilings.MAX_MERGE_SPAN:
            continue
        regions = _minority_regions(cell, pts, best, other)
        if not regions:
            notes.append({"reason": "unsplit_level", "level": other["level"], "support_m2": other["support_m2"],
                          "at": [float(v) for v in cell.representative_point().coords[0]]})
            continue
        cut = shapely.union_all(regions)
        pieces = [g for piece in pieces for g in shapely.get_parts(piece.difference(cut)) if g.geom_type == "Polygon"] + regions
    return pieces


def _distinct(others: list[dict], level: float) -> list[dict]:
    """Other levels seen in a ceiling's zones, those within LEVEL_WINDOW of each other (or of the ceiling) counted once."""
    kept: list[dict] = []
    for o in sorted(others, key=lambda o: -o["support_m2"]):
        if abs(o["level"] - level) > ceilings.LEVEL_WINDOW and all(abs(o["level"] - k["level"]) > ceilings.LEVEL_WINDOW for k in kept):
            kept.append(o)
    return kept


def build(polygon: shapely.Polygon, cells: list[shapely.Polygon], points: np.ndarray, beams: shapely.Geometry) -> tuple[list[dict], list[dict]]:
    """(ceilings, notes). Ceilings, highest first: `{"level", "sigma", "support_m2", "region", "assumed", "others"}`.

    A zone takes the dominant level of the ceiling `points` over it (beams excluded); another level with a ceiling's worth
    of support is cut out of it as a zone of its own. The sigma of a level is that of its fit over the whole room, so the
    merge rule of `ceilings` gives the same verdict however the room is cut: neighbouring zones that merge form one ceiling.
    A zone no point supports takes the level of its largest neighbour and stays a ceiling of its own, `assumed`, unless
    that neighbour is itself assumed. `notes` lists the levels that could not be cut out.
    """
    room = ceilings.level_fits(points)
    notes: list[dict] = []
    sampled = beams.buffer(ceilings.POLYGON_MARGIN)

    def measure(region):
        pts = points[shapely.contains_xy(region, points[:, 0], points[:, 2])]
        best, others = _dominant(pts)
        if best is not None and room:
            seen = min(room, key=lambda f: abs(f["level"] - best["level"]))
            if abs(seen["level"] - best["level"]) < ceilings.LEVEL_WINDOW:
                best = {**best, "sigma": seen["sigma"]}
        return best, others, pts

    groups, unsupported = [], []
    for cell in _absorb_small(cells):
        best, others, pts = measure(cell.difference(sampled))
        if best is None:
            unsupported.append(cell)
            continue
        for piece in _split_zone(cell, pts, best, others, notes):
            level, rest, _ = measure(piece.difference(sampled))
            if level is None:
                unsupported.append(piece)
            else:
                groups.append({**level, "geom": piece, "assumed": False, "others": rest})
    groups = _merge_measured(groups)
    if not groups:
        return [], notes
    while unsupported:
        known = [(i, c) for i, c in enumerate(unsupported) if any(_neighbours(c, g["geom"]) for g in groups)]
        if not known:
            break
        i, cell = max(known, key=lambda ic: ic[1].area)
        unsupported.pop(i)
        host = max((g for g in groups if _neighbours(cell, g["geom"])), key=lambda g: g["geom"].area)
        if host["assumed"]:
            host["geom"] = shapely.union_all([host["geom"], cell])
            continue
        groups.append({"level": host["level"], "sigma": float(np.hypot(host["sigma"], shell.FALLBACK_OFFSET_SIGMA)),
                       "support_m2": 0.0, "geom": cell, "assumed": True, "others": []})
    groups = _merge_assumed(groups)
    out = []
    for g in groups:
        region = ceilings._as_multipolygon(_opened(g["geom"].difference(beams)))
        region = shapely.MultiPolygon([p for p in region.geoms if p.area >= ceilings.MIN_COMPONENT_AREA])
        if not region.is_empty:
            out.append({k: g[k] for k in ("level", "sigma", "support_m2", "assumed")} | {"region": region, "others": _distinct(g["others"], g["level"])})
    return sorted(out, key=lambda c: (-c["level"], c["assumed"], c["region"].bounds)), notes
