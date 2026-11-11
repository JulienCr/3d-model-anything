"""Plan geometry of the shell in the room frame ([x, z] points): mitred wall footprints, trimmed partitions, overlap tests.

A footprint is a convex polygon [start face, end face, end back, start back]. Walls get a mitre at every corner they
share, so no two footprints overlap. A partition end that comes within REACH thicknesses of a wall (or another partition) is
planned in three forms: `face` (each face line stopped on the host face, for mouldings and cutter tests), `solid`
(extended through the host) and `cut` (what the Blender boolean subtracts from the solid, so it fits any jog or corner).
"""
import math

from . import frame

# Every length below is a multiple of a wall thickness t, or a numeric tolerance, so nothing depends on the room.
JOIN_TOL = 0.02  # m: RoomPlan walls share their corner point to well under this
NEAR_PARALLEL = math.sin(math.radians(10.0))  # |sin| of the turn under which two walls count as parallel (brief)
CORNER_SHIFT = 2.0  # x t: a line intersection further than this from the shared corner is a blow-up (a 10 degree turn with a jog)
REACH = 1.5  # x t: a RoomPlan partition ends within about a wall thickness of the face it meets, plus its own error
# Snap a collinear partition flush when its body still overlaps the wall's by more than half the thinner of the two
# (offset < SNAP x min thickness); below that it is a different plane and keeps its step.
SNAP = 0.5
SHADOW = 5.0  # x t a host is swept along a partition line to remove what sticks out behind it
TOUCH_TOL = 0.001  # m: polygons overlapping by less than this only touch (mesh precision)


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
        limit = CORNER_SHIFT * max(a["t"], b["t"])
        if inner and outer and math.dist(inner, corner) <= limit and math.dist(outer, corner) <= limit + a["t"] + b["t"]:
            return (inner, outer), (inner, outer)
    if _dot(a["n"], b["n"]) <= 0:  # a wall folding back on itself has no bisector to cut along
        return None
    d = (a["n"][0] + b["n"][0], a["n"][1] + b["n"][1])
    cuts = [(_along(w, False, corner, d), _along(w, True, corner, d)) for w in (a, b)]
    if any(p is None for cut in cuts for p in cut):
        return None
    spans = [sorted(_dot(a["n"], p) for p in cut) for cut in cuts]
    if max(spans[0][0], spans[1][0]) > min(spans[0][1], spans[1][1]):  # the faces are further apart than the wall is thick: no wall in between
        points = [p for cut in cuts for p in cut]
        edge = (max(points, key=lambda p: _dot(a["n"], p)), min(points, key=lambda p: _dot(a["n"], p)))
        return edge, edge
    return tuple(cuts)


def _convex(poly: list) -> bool:
    signs = set()
    for k in range(len(poly)):
        a, b, c = poly[k], poly[(k + 1) % len(poly)], poly[(k + 2) % len(poly)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-12:
            signs.add(cross > 0)
    return len(signs) == 1


def _joined(arch: dict):
    """(walls, lines, cuts, links): the wall elements, their face lines, the cuts at their joints and {id: id of the next wall}."""
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    lines = {e["id"]: _line(e, arch) for e in walls}
    cuts, links = {e["id"]: {} for e in walls}, {}
    for a in walls:
        for b in walls:
            if a is b or math.dist(a["end"], b["start"]) > JOIN_TOL:
                continue
            joint = _joint(lines[a["id"]], lines[b["id"]])
            if joint is not None:
                cuts[a["id"]]["end"], cuts[b["id"]]["start"] = joint
                links[a["id"]] = b["id"]
    return walls, lines, cuts, links


def wall_links(arch: dict) -> dict[str, str]:
    """{wall id: id of the wall joined to its end}."""
    return _joined(arch)[3]


def wall_footprints(arch: dict) -> dict[str, list]:
    """{wall id: [start face, end face, end back, start back]}: perimeter walls mitred where one ends on the next one's start."""
    walls, lines, cuts, _ = _joined(arch)
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


def _span(a, b, poly: list):
    """(enter, leave): parameters (0 at a, 1 at b) where the line a -> b enters and leaves a convex polygon, None if it misses."""
    area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1]))
    d = _sub(b, a)
    enter, leave = -math.inf, math.inf
    for k, p in enumerate(poly):
        q = poly[(k + 1) % len(poly)]
        out = (q[1] - p[1], p[0] - q[0]) if area > 0 else (p[1] - q[1], q[0] - p[0])
        size = math.hypot(*out)
        num, den = _dot(_sub(a, p), out) / size, _dot(d, out) / size
        if abs(den) < 1e-12:
            if num > 1e-6:
                return None
        elif den < 0:
            enter = max(enter, -num / den)
        else:
            leave = min(leave, -num / den)
    return (enter, leave) if enter < leave else None


def _direction(poly: list):
    """Unit direction of the longest edge of a polygon."""
    p, q = max(zip(poly, poly[1:] + poly[:1]), key=lambda e: math.dist(*e))
    return ((q[0] - p[0]) / math.dist(p, q), (q[1] - p[1]) / math.dist(p, q))


def _contains(poly: list, point) -> bool:
    """Whether a convex polygon holds the point."""
    signs = {((q[0] - p[0]) * (point[1] - p[1]) - (q[1] - p[1]) * (point[0] - p[0])) > 0 for p, q in zip(poly, poly[1:] + poly[:1])}
    return len(signs) == 1


def _hull(points: list) -> list:
    """Convex hull (counter-clockwise) of plan points."""
    pts = sorted(set(points))

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return half(pts) + half(pts[::-1])


def _width(poly: list) -> float:
    """Thickness of a wall or partition footprint: its area over its longest edge."""
    area = abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1]))) / 2
    return area / max(math.dist(p, q) for p, q in zip(poly, poly[1:] + poly[:1]))


def _shadow(poly: list, d) -> list:
    """The polygon swept SHADOW thicknesses along `d`: the host and everything behind it."""
    reach = SHADOW * _width(poly)
    return _hull([*poly, *((x + d[0] * reach, z + d[1] * reach) for x, z in poly)])


def _face_lines(quad: list):
    """The two face lines of a partition quad as (vertex, vertex) index pairs: A (0, 1) and B (3, 2)."""
    return ((0, 1), (3, 2))


def _hits(quad: list, hosts: list) -> dict:
    """{vertex index: (enter point, leave point, host, direction)} for each end vertex within REACH thicknesses of a non-collinear host.

    The line of each face is followed from its middle towards the vertex and that far beyond it.
    """
    out = {}
    for first, second in _face_lines(quad):
        mid = ((quad[first][0] + quad[second][0]) / 2, (quad[first][1] + quad[second][1]) / 2)
        for i in (first, second):
            v = quad[i]
            d = _sub(v, mid)
            length = math.hypot(*d)
            if length < 1e-9:
                continue
            unit = (d[0] / length, d[1] / length)
            best = None
            for host in hosts:
                u = _direction(host)
                if abs(unit[0] * u[1] - unit[1] * u[0]) < NEAR_PARALLEL:
                    continue
                span = _span(mid, v, host)
                if span and 0.0 < span[0] <= 1.0 + REACH * _width(host) / length and (best is None or span[0] < best[0]):
                    best = (*span, host)
            if best is not None:
                point = lambda t: (mid[0] + d[0] * t, mid[1] + d[1] * t)  # noqa: E731
                out[i] = (point(best[0]), point(best[1]), best[2], unit)
    return out


def _snap(quad: list, walls: dict, lines: dict) -> list:
    """The quad with an end moved onto the face and back lines of a collinear wall that is within reach of it.

    The n of a wall points to its room, so a partition facing the same way has its face on the wall face line.
    """
    quad = list(quad)
    thickness = math.dist(quad[0], quad[3])
    n = _sub(quad[0], quad[3])  # from the back line to the face line
    size = math.hypot(*n) or 1.0
    n = (n[0] / size, n[1] / size)
    for faces in ((0, 3), (1, 2)):
        end = quad[faces[0]]
        for wid, poly in walls.items():
            line = lines[wid]
            if abs(n[0] * line["n"][1] - n[1] * line["n"][0]) >= NEAR_PARALLEL or _distance(poly, end) > REACH * line["t"]:
                continue
            same = _dot(n, line["n"]) > 0
            targets = (line["o"], line["o"] - line["t"]) if same else (line["o"] - line["t"], line["o"])
            moved = []
            for i, target in zip(faces, targets):
                step = (target - _dot(line["n"], quad[i])) / _dot(line["n"], n)
                moved.append((quad[i][0] + n[0] * step, quad[i][1] + n[1] * step, step))
            if all(abs(m[2]) < SNAP * min(line["t"], thickness) for m in moved):
                for i, m in zip(faces, moved):
                    quad[i] = (m[0], m[1])
                break
    return quad


def _distance(poly: list, point) -> float:
    """Distance from a point to a convex polygon (0 inside)."""
    if _contains(poly, point):
        return 0.0
    best = math.inf
    for p, q in zip(poly, poly[1:] + poly[:1]):
        e = _sub(q, p)
        t = max(0.0, min(1.0, _dot(_sub(point, p), e) / (_dot(e, e) or 1.0)))
        best = min(best, math.dist(point, (p[0] + e[0] * t, p[1] + e[1] * t)))
    return best


def _middle(quad: list):
    return tuple(sum(quad[i][k] for i in range(4)) / 4 for k in (0, 1))


def partition_plans(arch: dict, walls: dict[str, list]) -> dict[str, dict]:
    """{partition id: {"face", "solid", "cut"}}: the partition quad with each face line stopped on the host face it meets,
    the quad extended through those hosts, and the polygons (host shadows, other overlapped hosts) to subtract from it.

    A partition whose body holds the middle of this one is a crossing, not a host; of two partitions that run into
    each other only the later one extends. Hosts are taken untrimmed, so the order of the elements does not matter.
    """
    lines = _joined(arch)[1]
    quads = {}
    for e in arch["elements"]:
        if e["kind"] == "partition":
            line = _line(e, arch)
            quads[e["id"]] = _snap([_on(line, e["start"]), _on(line, e["end"]), _on(line, e["end"], back=True),
                                    _on(line, e["start"], back=True)], walls, lines)
    reach = {k: {j for j in quads if j != k and _hits(quad, [quads[j]])} for k, quad in quads.items()}
    out = {}
    for pid, quad in quads.items():
        hosts = [*walls.values(), *(q for j, q in quads.items() if j != pid and not _contains(q, _middle(quad))
                                    and not (pid in reach[j] and j in reach[pid] and j > pid))]
        face, solid, cut = list(quad), list(quad), []
        hits = _hits(quad, hosts)
        for i, (enter, leave, host, unit) in hits.items():
            face[i], solid[i] = enter, leave
            if not any(host is c[0] for c in cut):
                cut.append((host, unit))
        shadows = [_shadow(host, unit) for host, unit in cut]
        others = [h for h in hosts if not any(h is c[0] for c in cut) and overlap(h, solid, 1e-9)]
        out[pid] = {"face": face, "solid": solid, "cut": [*shadows, *others]}
    return out


def partition_footprints(arch: dict, walls: dict[str, list]) -> dict[str, list]:
    """{partition id: quad [face start, face end, back end, back start]} with each face line stopped on the host it meets."""
    return {pid: plan["face"] for pid, plan in partition_plans(arch, walls).items()}


def overlap(a: list, b: list, tol: float = TOUCH_TOL) -> bool:
    """Whether two convex polygons overlap by more than `tol` m (separating-axis test)."""
    for poly in (a, b):
        for k, p in enumerate(poly):
            q = poly[(k + 1) % len(poly)]
            axis = (q[1] - p[1], p[0] - q[0])
            norm = math.hypot(*axis)
            if norm < 1e-12:
                continue
            ra = [_dot(v, axis) / norm for v in a]
            rb = [_dot(v, axis) / norm for v in b]
            if min(max(ra), max(rb)) - max(min(ra), min(rb)) <= tol:
                return False
    return True
