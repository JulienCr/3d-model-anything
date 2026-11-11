"""Plan geometry of the shell in the room frame ([x, z] points): mitred wall footprints, trimmed partitions, overlap tests.

A footprint is a convex polygon. Walls get a mitre at every corner they share, so no two footprints overlap;
partitions are clipped to the face of the wall (or earlier partition) their ends run into.
"""
import math

from . import frame

JOIN_TOL = 0.02  # m: a wall ends where the next one starts when these points are this close
NEAR_PARALLEL = math.sin(math.radians(10.0))  # |sin| of the turn under which two walls count as parallel
MAX_SHIFT = 0.4  # m: an intersection this far from the shared corner is a blow-up, so the corner is cut at the bisector
CROSS_DEPTH = 0.002  # m a partition end must go into a host before it is trimmed
TOUCH_TOL = 0.001  # m: polygons overlapping by less than this only touch


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _line(el: dict, arch: dict) -> dict:
    """The face line n.p = o of a wall, its back line n.p = o - t, and its direction along start -> end."""
    p = el["params"]
    az = frame.value(arch, p["azimuth"])
    n = (math.cos(az), math.sin(az))
    u = (-n[1], n[0])
    if _dot(u, _sub(el["end"], el["start"])) < 0:
        u = (n[1], -n[0])
    t = frame.value(arch, p["thickness"])
    return {"n": n, "u": u, "o": frame.value(arch, p["offset"]), "t": t, "start": el["start"], "end": el["end"]}


def _on(line: dict, point, back: bool = False) -> tuple[float, float]:
    """`point` moved along the normal onto the face line (or the back line)."""
    d = line["o"] - (0.0 if not back else line["t"]) - _dot(line["n"], point)
    return (point[0] + line["n"][0] * d, point[1] + line["n"][1] * d)


def _meet(a: dict, b: dict, back_a: bool, back_b: bool):
    """Intersection of the face (or back) lines of two walls, None when they are parallel."""
    (nax, naz), (nbx, nbz) = a["n"], b["n"]
    oa, ob = a["o"] - (a["t"] if back_a else 0.0), b["o"] - (b["t"] if back_b else 0.0)
    det = nax * nbz - naz * nbx
    if abs(det) < 1e-9:
        return None
    return ((oa * nbz - naz * ob) / det, (nax * ob - oa * nbx) / det)


def _along(line: dict, back: bool, point, d):
    """Where the line through `point` with direction `d` crosses the face (or back) line of a wall, None when parallel."""
    nd = _dot(line["n"], d)
    if abs(nd) < 0.2 * math.hypot(*d):
        return None
    s = (line["o"] - (line["t"] if back else 0.0) - _dot(line["n"], point)) / nd
    return (point[0] + d[0] * s, point[1] + d[1] * s)


def _joint(a: dict, b: dict):
    """((face point, back point) cut of a's end, the same for b's start) at the corner of consecutive walls, or None.

    Both cuts lie on one line, so the two bodies meet along it without overlap or gap: through the two inner and the
    two outer line intersections for an ordinary corner, along the bisector of the normals through the shared corner
    for near-parallel walls.
    """
    corner = ((a["end"][0] + b["start"][0]) / 2, (a["end"][1] + b["start"][1]) / 2)
    if abs(a["u"][0] * b["u"][1] - a["u"][1] * b["u"][0]) >= NEAR_PARALLEL:
        inner, outer = _meet(a, b, False, False), _meet(a, b, True, True)
        if inner and outer and math.dist(inner, corner) <= MAX_SHIFT and math.dist(outer, corner) <= MAX_SHIFT + a["t"] + b["t"]:
            return (inner, outer), (inner, outer)
    if _dot(a["n"], b["n"]) <= 0:  # a wall folding back on itself has no bisector to cut along
        return None
    d = (a["n"][0] + b["n"][0], a["n"][1] + b["n"][1])
    cuts = [(_along(w, False, corner, d), _along(w, True, corner, d)) for w in (a, b)]
    return None if any(p is None for cut in cuts for p in cut) else tuple(cuts)


def _convex(poly: list) -> bool:
    signs = set()
    for k in range(len(poly)):
        a, b, c = poly[k], poly[(k + 1) % len(poly)], poly[(k + 2) % len(poly)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-12:
            signs.add(cross > 0)
    return len(signs) == 1


def wall_footprints(arch: dict) -> dict[str, list]:
    """{wall id: [start face, end face, end back, start back]}: perimeter walls mitred where one ends on the next one's start."""
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    lines = {e["id"]: _line(e, arch) for e in walls}
    cuts = {e["id"]: {} for e in walls}
    for a in walls:
        for b in walls:
            if a is b or math.dist(a["end"], b["start"]) > JOIN_TOL:
                continue
            joint = _joint(lines[a["id"]], lines[b["id"]])
            if joint is not None:
                cuts[a["id"]]["end"], cuts[b["id"]]["start"] = joint
    out = {}
    for e in walls:
        line = lines[e["id"]]
        ends = {}
        for key in ("start", "end"):
            free = _on(line, e[key])
            ends[key] = cuts[e["id"]].get(key) or (free, _on(line, e[key], back=True))
        quad = [ends["start"][0], ends["end"][0], ends["end"][1], ends["start"][1]]
        if not _convex(quad):  # a wall too short for its mitres: square ends
            quad = [_on(line, e["start"]), _on(line, e["end"]), _on(line, e["end"], back=True), _on(line, e["start"], back=True)]
        out[e["id"]] = quad
    return out


def _clip(poly: list, point, normal) -> list:
    """The part of a convex polygon on the side of the line through `point` that `normal` points to."""
    out = []
    for k, cur in enumerate(poly):
        prev = poly[k - 1]
        dp, dc = _dot(_sub(prev, point), normal), _dot(_sub(cur, point), normal)
        if (dp >= 0) != (dc >= 0):
            r = dp / (dp - dc)
            out.append((prev[0] + (cur[0] - prev[0]) * r, prev[1] + (cur[1] - prev[1]) * r))
        if dc >= 0:
            out.append(cur)
    return out


def _entry(a, b, poly: list):
    """(parameter, outward normal, point) where the segment a -> b enters the convex polygon, None when it starts inside or misses."""
    area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1]))
    d = _sub(b, a)
    enter, edge, leave = -math.inf, None, math.inf
    for k, p in enumerate(poly):
        q = poly[(k + 1) % len(poly)]
        out = (q[1] - p[1], p[0] - q[0]) if area > 0 else (p[1] - q[1], q[0] - p[0])
        num, den = _dot(_sub(a, p), out), _dot(d, out)
        if abs(den) < 1e-12:
            if num > 0:
                return None
        elif den < 0 and -num / den > enter:
            enter, edge = -num / den, (out, p)
        elif den > 0:
            leave = min(leave, -num / den)
    if edge is None or not 0.0 < enter < leave:
        return None
    return enter, edge[0], edge[1]


def _trim(quad: list, hosts: list) -> list:
    """The partition `quad` [face start, face end, back end, back start] clipped to the nearest host face each end crosses.

    Each end is followed from the middle of the partition, so a start drawn out through a wall is cut at its first face.
    """
    ends = [((quad[0][0] + quad[3][0]) / 2, (quad[0][1] + quad[3][1]) / 2), ((quad[1][0] + quad[2][0]) / 2, (quad[1][1] + quad[2][1]) / 2)]
    middle = ((ends[0][0] + ends[1][0]) / 2, (ends[0][1] + ends[1][1]) / 2)
    reach = math.dist(middle, ends[0])
    for end in ends:
        hits = [h for h in (_entry(middle, end, host) for host in hosts) if h and (1.0 - h[0]) * reach > CROSS_DEPTH]
        if hits:
            _, normal, point = min(hits, key=lambda h: h[0])
            quad = _clip(quad, point, normal)
    return quad


def partition_footprints(arch: dict, walls: dict[str, list]) -> dict[str, list]:
    """{partition id: polygon}: each end trimmed to the face of the nearest wall or earlier partition it runs into."""
    hosts = list(walls.values())
    out = {}
    for e in arch["elements"]:
        if e["kind"] != "partition":
            continue
        line = _line(e, arch)
        quad = [_on(line, e["start"]), _on(line, e["end"]), _on(line, e["end"], back=True), _on(line, e["start"], back=True)]
        out[e["id"]] = _trim(quad, hosts)
        hosts.append(out[e["id"]])
    return out


def overlap(a: list, b: list) -> bool:
    """Whether two convex polygons overlap by more than a touch (separating-axis test)."""
    for poly in (a, b):
        for k, p in enumerate(poly):
            q = poly[(k + 1) % len(poly)]
            axis = (q[1] - p[1], p[0] - q[0])
            norm = math.hypot(*axis)
            if norm < 1e-12:
                continue
            ra = [_dot(v, axis) / norm for v in a]
            rb = [_dot(v, axis) / norm for v in b]
            if min(max(ra), max(rb)) - max(min(ra), min(rb)) <= TOUCH_TOL:
                return False
    return True
