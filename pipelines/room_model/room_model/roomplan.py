"""RoomPlan room (walls, doors, floor outline) in the room frame, plus measurement stations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely

from room_model.capture import colmajor

UP = np.array([0.0, 1.0, 0.0])
MATCH_DISTANCE = 0.4
DEFAULT_WALL_HEIGHT = 2.5


@dataclass
class Wall:
    label: str
    ident: str
    c: np.ndarray
    x: np.ndarray
    y: np.ndarray
    n: np.ndarray  # unit normal, points into the room
    w: float
    h: float
    poly: list  # (s, t) outline in the wall frame, may be empty
    edge: tuple  # (start, end) of the matching floor-outline edge, room frame xyz
    doors: list = field(default_factory=list)

    def local(self, p: np.ndarray):
        rel = np.atleast_2d(p) - self.c
        return rel @ self.x, rel @ self.y, rel @ self.n

    def top_t(self, s):
        """Top edge of the wall outline at along-coordinate s (scalar or array, wall frame, relative to the centre)."""
        s = np.asarray(s, float)
        scalar = s.ndim == 0
        s = np.atleast_1d(s)
        if not self.poly:
            out = np.full(s.shape, self.h / 2)
        else:
            best = np.full(s.shape, -np.inf)
            for (s0, t0), (s1, t1) in zip(self.poly, self.poly[1:] + self.poly[:1]):
                inside = (s >= min(s0, s1) - 1e-6) & (s <= max(s0, s1) + 1e-6)
                val = max(t0, t1) if abs(s1 - s0) < 1e-9 else t0 + (t1 - t0) * (s - s0) / (s1 - s0)
                best = np.where(inside, np.maximum(best, val), best)
            out = np.where(np.isfinite(best), best, self.h / 2)
        return float(out[0]) if scalar else out

    @property
    def plane(self):
        """(n, d) with n.p = d on the wall."""
        return self.n, float(self.n @ self.c)


@dataclass
class Door:
    label: str
    ident: str
    wall: Wall
    s: float  # centre, wall frame
    t: float
    w: float
    h: float

    def centre(self) -> np.ndarray:
        return self.wall.c + self.wall.x * self.s + self.wall.y * self.t


@dataclass
class Station:
    sid: str
    kind: str  # "largeur" | "longueur"
    wall_a: Wall
    wall_b: Wall
    along: float  # coordinate along wall_a.x relative to wall_a.c
    mid: np.ndarray  # room-frame point at floor level, halfway between the wall planes
    d_roomplan: float
    door_labels: list
    corner_ref: str
    corner_dist: float
    rp_height: float
    flag: str = ""


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


class Room:
    def __init__(self, data: dict, world_to_room: np.ndarray):
        w2r = np.asarray(world_to_room, float)
        floor = data["floors"][0]
        tf = w2r @ colmajor(floor["transform"], 4)
        poly = (tf[:3, :3] @ np.array(floor["polygonCorners"]).T).T + tf[:3, 3]
        self.floor_y = float(poly[:, 1].mean())
        self.outline = poly  # room frame xyz, closed implicitly
        self._polygon = shapely.Polygon(poly[:, [0, 2]])
        raw = data["walls"]
        wall_t = [w2r @ colmajor(w["transform"], 4) for w in raw]
        n_edges = len(poly)
        self.walls: list[Wall] = []
        self.off_outline: list[str] = []
        self.seeded_from_outline: list[str] = []
        self.orphan_doors: list[str] = []
        matched = self._match_walls(raw, wall_t)
        height = float(np.median([raw[i]["dimensions"][1] for i in matched.values()])) if matched else DEFAULT_WALL_HEIGHT
        for k in range(n_edges):
            a_pt, b_pt = poly[k], poly[(k + 1) % n_edges]
            if k in matched:
                w, T = raw[matched[k]], wall_t[matched[k]]
                self.walls.append(Wall(
                    label=f"M{k + 1}", ident=w["identifier"], c=T[:3, 3].copy(), x=T[:3, 0].copy(), y=T[:3, 1].copy(),
                    n=T[:3, 2].copy(), w=float(w["dimensions"][0]), h=float(w["dimensions"][1]),
                    poly=[(float(p[0]), float(p[1])) for p in w["polygonCorners"]], edge=(a_pt, b_pt)))
            else:
                self.walls.append(self._seed_wall(k, a_pt, b_pt, height))
                self.seeded_from_outline.append(self.walls[-1].ident)
        used = {raw[i]["identifier"] for i in matched.values()}
        self.off_outline = [w["identifier"] for w in raw if w["identifier"] not in used]
        self._attach_doors(data["doors"], w2r)
        self.ceiling_top = max(w.c[1] + w.top_t(0.0) for w in self.walls)

    def _match_walls(self, raw: list, wall_t: list) -> dict[int, int]:
        """Outline edge index -> index of the RoomPlan wall standing on it (parallel, within MATCH_DISTANCE)."""
        matched: dict[int, int] = {}
        poly = self.outline
        for k in range(len(poly)):
            a_pt, b_pt = poly[k], poly[(k + 1) % len(poly)]
            edge_len = np.linalg.norm(b_pt - a_pt)
            candidates = []
            for i, T in enumerate(wall_t):
                if i in matched.values() or abs(T[:3, 0] @ (b_pt - a_pt)) / edge_len < 0.98:
                    continue
                dd = _seg_dist(T[[0, 2], 3][None], a_pt[[0, 2]], b_pt[[0, 2]])[0]
                if dd < MATCH_DISTANCE:
                    candidates.append((round(dd / 0.05), -raw[i]["dimensions"][0], dd, i))
            if candidates:
                matched[k] = min(candidates)[3]
        return matched

    def _seed_wall(self, k: int, a_pt: np.ndarray, b_pt: np.ndarray, height: float) -> Wall:
        d = (b_pt - a_pt)[[0, 2]]
        length = float(np.linalg.norm(d))
        d = d / length
        left = np.array([-d[1], d[0]])
        n2 = left if self._polygon.exterior.is_ccw else -left
        n = np.array([n2[0], 0.0, n2[1]])
        c = (a_pt + b_pt) / 2
        c[1] = self.floor_y + height / 2
        return Wall(label=f"M{k + 1}", ident=f"outline-{k}", c=c, x=np.array([n[2], 0.0, -n[0]]), y=UP.copy(), n=n,
                    w=length, h=height, poly=[], edge=(a_pt, b_pt))

    def _attach_doors(self, doors_raw: list, w2r: np.ndarray) -> None:
        by_ident = {}
        for wall in self.walls:
            by_ident[wall.ident] = wall
        found = []
        for d in doors_raw:
            T = w2r @ colmajor(d["transform"], 4)
            wall = by_ident.get(d["parentIdentifier"])
            if wall is None:
                self.orphan_doors.append(d["identifier"])
                continue
            rel = T[:3, 3] - wall.c
            found.append((float(T[0, 3]), float(T[2, 3]), d, wall, float(rel @ wall.x), float(rel @ wall.y)))
        found.sort(key=lambda r: r[:2])
        self.doors: list[Door] = []
        for k, (_, _, d, wall, s, t) in enumerate(found):
            door = Door(f"P{k + 1}", d["identifier"], wall, s, t, float(d["dimensions"][0]), float(d["dimensions"][1]))
            wall.doors.append(door)
            self.doors.append(door)

    def plan(self, p: np.ndarray) -> np.ndarray:
        """Plan coordinates [x, z] of room-frame points."""
        return np.atleast_2d(p)[:, [0, 2]]

    def outline_plan(self) -> np.ndarray:
        return self.plan(self.outline)

    def _inside(self, pts_xz: np.ndarray) -> np.ndarray:
        return shapely.contains_xy(self._polygon, pts_xz[:, 0], pts_xz[:, 1])

    def footprint_signed_margin(self, pts_xz: np.ndarray) -> np.ndarray:
        """Signed distance to the outline in plan: positive inside, negative outside."""
        inside = self._inside(pts_xz)
        o = self.outline[:, [0, 2]]
        d = np.min([_seg_dist(pts_xz, o[k], o[(k + 1) % len(o)]) for k in range(len(o))], axis=0)
        return np.where(inside, d, -d)

    def in_roi(self, pts: np.ndarray, margin: float = 0.12, y_pad: float = 0.25) -> np.ndarray:
        m = self.footprint_signed_margin(pts[:, [0, 2]]) > -margin
        return m & (pts[:, 1] > self.floor_y - y_pad) & (pts[:, 1] < self.ceiling_top + y_pad)

    def corner_reference(self, wall: Wall, point: np.ndarray) -> tuple[str, float]:
        """Nearest end of the wall's outline edge to a point on the wall: ('M3/M4', distance along the wall)."""
        k = self.walls.index(wall)
        a_pt, b_pt = wall.edge
        da, db = abs((point - a_pt) @ wall.x), abs((point - b_pt) @ wall.x)
        if da <= db:
            other = self.walls[(k - 1) % len(self.walls)]
            return f"{wall.label}/{other.label}", float(da)
        other = self.walls[(k + 1) % len(self.walls)]
        return f"{wall.label}/{other.label}", float(db)

    def station_at(self, wa: Wall, wb: Wall, s: float, kind: str) -> Station | None:
        """Station between two facing walls at along-coordinate s (relative to wa.c, along wa.x)."""
        d = float((wb.c - wa.c) @ wa.n)
        base = wa.c + wa.x * s
        seg = np.linspace(0.05, d - 0.05, 9)
        pts = base[None] + seg[:, None] * wa.n[None]
        if np.any(self.footprint_signed_margin(pts[:, [0, 2]]) < -0.05):
            return None
        mid = base + wa.n * d / 2
        mid[1] = self.floor_y
        doors = [dr.label for dr in self.doors
                 if dr.wall in (wa, wb) and abs((dr.centre() - wa.c) @ wa.x - s) < dr.w / 2 + 0.1]
        corner, cd = self.corner_reference(wa, base)
        s_w, _, _ = wa.local(base)
        return Station(sid="", kind=kind, wall_a=wa, wall_b=wb, along=float(s), mid=mid, d_roomplan=d, door_labels=doors,
                       corner_ref=corner, corner_dist=cd, rp_height=float(wa.c[1] + wa.top_t(float(s_w[0])) - self.floor_y))

    def station_candidates(self, spacing: float = 1.0, margin: float = 0.45, min_overlap: float = 0.6, max_width: float = 4.5):
        """Opposite wall pairs facing each other across the room: (wa, wb, kind, [along positions], (lo, hi))."""
        out = []
        for i, wa in enumerate(self.walls):
            for j, wb in enumerate(self.walls):
                if j <= i or wa.n @ wb.n > -0.98:
                    continue
                d = float((wb.c - wa.c) @ wa.n)
                if d < 0.3 or float((wa.c - wb.c) @ wb.n) < 0.3:
                    continue
                ia = sorted([(wa.edge[0] - wa.c) @ wa.x, (wa.edge[1] - wa.c) @ wa.x])
                ib = sorted([(wb.edge[0] - wa.c) @ wa.x, (wb.edge[1] - wa.c) @ wa.x])
                lo, hi = max(ia[0], ib[0]), min(ia[1], ib[1])
                if hi - lo < min_overlap:
                    continue
                kind = "largeur" if d <= max_width else "longueur"
                if kind == "longueur" or hi - lo < 2 * margin + 0.1:
                    positions = [(lo + hi) / 2]
                else:
                    n_st = max(2, int(np.floor((hi - lo - 2 * margin) / spacing)) + 1)
                    positions = list(np.linspace(lo + margin, hi - margin, n_st))
                out.append((wa, wb, kind, positions, (lo, hi)))
        out.sort(key=lambda r: (r[2] != "largeur", r[0].label, r[1].label))
        return out
