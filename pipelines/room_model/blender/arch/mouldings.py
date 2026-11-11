"""Cornices, skirting, architraves and panel frames: 2D profiles swept along wall faces, mitred at the corners.

A chain is `{"points": plan points, "normals": one inward normal per segment, "tags": (host id, side, s0, s1) per
segment, "closed": bool}`: the runs of a ring (floor polygon, ceiling outline) that lie on wall faces.
"""
import math

import bmesh
import bpy

from . import frame, profiles, shell

BITE = 0.001  # m a moulding sinks into its host, so no two faces are coplanar
DEFAULT_PROJECTION = 0.015
WALL_TOL = 0.25  # m: a ring vertex this close to a wall face lies on it (scanned ceilings stop short of the walls)
END_TOL = 0.08  # m: how far past either end of a wall a vertex may lie and still belong to it
MIN_RUN = 0.3  # m: shorter stretches along a wall are noise
JOIN_TOL = 0.35  # m: two stretches meet at a corner when the face lines cross this close to both ends
PARALLEL = 0.17  # sin of the angle under which two face lines count as parallel
MIN_PIECE = 0.01  # m
MAX_MITRE = 4.0  # floor of 1 / (1 + cos) so a near fold-back does not shoot off
EPS = 1e-9
ARCHITRAVE_PROJECTION = 0.015
PANEL_WIDTH, PANEL_PROJECTION = 0.03, 0.015
SILL_TOL = 0.02  # m: an opening whose sill is below this reaches the floor, so its architrave has no bottom bar
HOST_KINDS = ("wall", "partition")


def _mitre(normals: list, closed: bool) -> list[tuple[float, float]]:
    """Per-vertex 2D offset vector that moves a point by 1 m perpendicular to both adjacent segments."""
    n = len(normals)
    out = []
    for i in range(n if closed else n + 1):
        before = normals[(i - 1) % n] if (closed or i > 0) else None
        after = normals[i % n] if (closed or i < n) else None
        if before is None or after is None:
            out.append(before or after)
            continue
        k = max(1.0 + before[0] * after[0] + before[1] * after[1], 1.0 / MAX_MITRE)
        out.append(((before[0] + after[0]) / k, (before[1] + after[1]) / k))
    return out


def _sweep(bm: bmesh.types.BMesh, points: list, across: list, fixed: tuple, profile: list, scale: tuple, origin: tuple,
           closed: bool) -> None:
    """Adds the solid of `profile` swept along room-frame `points`: a runs along the per-vertex `across`, b along `fixed`."""
    rings = []
    for point, vec in zip(points, across):
        ring = []
        for a, b in profile:
            da, db = a * scale[0] + origin[0], b * scale[1] + origin[1]
            ring.append(bm.verts.new(frame.to_blender(*(point[i] + vec[i] * da + fixed[i] * db for i in range(3)))))
        rings.append(ring)
    size = len(profile)
    for ring, nxt in zip(rings, rings[1:] + ([rings[0]] if closed else [])):
        for j in range(size):
            bm.faces.new((ring[j], ring[(j + 1) % size], nxt[(j + 1) % size], nxt[j]))
    if not closed:
        bm.faces.new(rings[0])
        bm.faces.new(rings[-1])


def _finish(name: str, bm: bmesh.types.BMesh, collection, flags: list[str]) -> bpy.types.Object:
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(bpy.data.materials["Moulure"])
    obj = bpy.data.objects.new(name, mesh)
    if flags:
        obj["flags"] = ",".join(flags)
    collection.objects.link(obj)
    return obj


def _flags(el: dict, known: bool) -> list[str]:
    return [*el.get("flags", []), *([] if known else ["unknown_profile"])]


def _dedupe(points: list, closed: bool) -> list:
    out = [p for k, p in enumerate(points) if k == 0 or math.dist(p, points[k - 1]) > 1e-6]
    if closed and len(out) > 1 and math.dist(out[0], out[-1]) <= 1e-6:
        out.pop()
    return out


def _stretches(arch: dict, ring: list) -> list[dict]:
    """The runs of consecutive ring vertices lying on a wall face, in ring order."""
    ring = _dedupe([tuple(p) for p in ring], closed=True)
    n = len(ring)
    found = []
    for el in arch["elements"]:
        if el["kind"] != "wall":
            continue
        geom = frame.wall_geometry(arch, el)
        (ox, oz), (ux, uz), (nx, nz) = geom["point"](0.0), geom["u"], geom["n"]
        along = []
        for x, z in ring:
            s, d = (x - ox) * ux + (z - oz) * uz, (x - ox) * nx + (z - oz) * nz
            along.append(s if abs(d) <= WALL_TOL and -END_TOL <= s <= geom["length"] + END_TOL else None)
        gaps = [i for i, s in enumerate(along) if s is None]
        order = list(range(n)) if not gaps else [(gaps[0] + 1 + k) % n for k in range(n)]
        group = []
        for i in [*order, None]:
            if i is not None and along[i] is not None:
                group.append(i)
                continue
            values = [along[j] for j in group]
            if group and max(values) - min(values) >= MIN_RUN:
                found.append({"key": group[0], "id": el["id"], "geom": geom, "s0": along[group[0]], "s1": along[group[-1]]})
            group = []
    return sorted(found, key=lambda f: f["key"])


def _crossing(a: dict, b: dict) -> tuple[float, float] | None:
    """Where the face lines of two consecutive stretches meet, if that is a corner both stretches really reach."""
    ga, gb = a["geom"], b["geom"]
    if a["id"] == b["id"] or abs(ga["u"][0] * gb["u"][1] - ga["u"][1] * gb["u"][0]) < PARALLEL:
        return None
    (ax, az), (bx, bz) = ga["point"](0.0), gb["point"](0.0)
    det = ga["u"][0] * -gb["u"][1] + gb["u"][0] * ga["u"][1]
    t = ((bx - ax) * -gb["u"][1] + gb["u"][0] * (bz - az)) / det
    corner = (ax + ga["u"][0] * t, az + ga["u"][1] * t)
    if math.dist(corner, ga["point"](a["s1"])) > JOIN_TOL or math.dist(corner, gb["point"](b["s0"])) > JOIN_TOL:
        return None
    return corner


def _chains(arch: dict, ring: list) -> list[dict]:
    """Chains along the walls a ring follows; stretches that meet at a corner share its vertex."""
    items = _stretches(arch, ring)
    count = len(items)
    if not count:
        return []
    corners = [_crossing(items[k], items[(k + 1) % count]) if count > 1 else None for k in range(count)]
    starts = [k for k in range(count) if corners[k - 1] is None]
    closed = not starts
    out = []
    for first in starts or [0]:
        chain = {"points": [], "normals": [], "tags": [], "closed": closed}
        k = first
        while True:
            item, geom = items[k], items[k]["geom"]
            a = corners[k - 1] if corners[k - 1] is not None else geom["point"](item["s0"])
            b = corners[k] if corners[k] is not None else geom["point"](item["s1"])
            (ox, oz), (ux, uz) = geom["point"](0.0), geom["u"]
            chain["points"].append(a)
            chain["normals"].append(geom["n"])
            chain["tags"].append((item["id"], 1, (a[0] - ox) * ux + (a[1] - oz) * uz, (b[0] - ox) * ux + (b[1] - oz) * uz))
            k = (k + 1) % count
            if corners[(k - 1) % count] is None or k == first:
                break
        if not closed:
            chain["points"].append(b)
        out.append(chain)
    return out


def _kept(s0: float, s1: float, spans: list) -> list[tuple[float, float]]:
    """Parameters (ta, tb) of the parts of the segment s0 -> s1 left once the (lo, hi) spans are removed."""
    lo, hi = sorted((s0, s1))
    pieces, cursor = [], lo
    for a, b in sorted(spans):
        if a > cursor:
            pieces.append((cursor, min(a, hi)))
        cursor = max(cursor, b)
    if cursor < hi:
        pieces.append((cursor, hi))
    span = s1 - s0
    return sorted(tuple(sorted(((a - s0) / span, (b - s0) / span))) for a, b in pieces if b - a >= MIN_PIECE)


def _cut(chain: dict, gaps: dict) -> list[dict]:
    """The chain with the host spans in `gaps` {(host id, side): [(lo, hi)]} removed; each remainder is an open chain."""
    pts, tags, closed = chain["points"], chain["tags"], chain["closed"]
    kept = [_kept(s0, s1, gaps.get((hid, side), [])) for hid, side, s0, s1 in tags]
    if all(k == [(0.0, 1.0)] for k in kept):
        return [chain]
    out, cur, joined = [], None, False
    for k, intervals in enumerate(kept):
        p0, p1 = pts[k], pts[(k + 1) % len(pts)]
        if cur is not None and (not intervals or intervals[0][0] > EPS):
            out.append(cur)
            cur = None
        for ta, tb in intervals:
            lerp = lambda t: (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            if ta <= EPS and cur is not None:
                cur["points"].append(lerp(tb))
                cur["normals"].append(chain["normals"][k])
            else:
                if cur is not None:
                    out.append(cur)
                joined = joined or (k == 0 and ta <= EPS)
                cur = {"points": [lerp(ta), lerp(tb)], "normals": [chain["normals"][k]], "closed": False}
            if tb < 1 - EPS:
                out.append(cur)
                cur = None
    if cur is not None:
        if closed and joined and out:
            out[0] = {"points": cur["points"] + out[0]["points"][1:], "normals": cur["normals"] + out[0]["normals"], "closed": False}
        else:
            out.append(cur)
    return out


def _horizontal(bm, chains: list, y: float, profile: list, projection: float, height: float) -> None:
    for chain in chains:
        pts = chain["points"]
        across = [(mx, 0.0, mz) for mx, mz in _mitre(chain["normals"], chain["closed"])]
        _sweep(bm, [(x, y, z) for x, z in pts], across, (0.0, 1.0, 0.0), profile, (projection, height), (-BITE, 0.0),
               chain["closed"])


def _spec(arch: dict, el: dict) -> tuple[float, float]:
    p = el["params"]
    projection = frame.value(arch, p["projection"]) if "projection" in p else DEFAULT_PROJECTION
    return frame.value(arch, p["height"]), projection


def _faces(arch: dict, el: dict, host: dict, geom: dict) -> tuple[int, ...]:
    """The sides of `host` on which `el` is visible: both faces of a partition it goes through, else one."""
    if host["kind"] == "wall":
        return (1,)
    return (1, -1) if shell.recess_depth(arch, geom, el) is None else (frame.side(host, el.get("face")),)


def _skirting_gaps(arch: dict, hosts: dict, height: float) -> dict:
    """{(host id, side): [(lo, hi)]}: the spans of every cut that reaches into the skirting band."""
    gaps = {}
    for el in arch["elements"]:
        host = hosts.get(el.get("host"))
        if el["kind"] not in shell.CUT_KINDS or host is None or host["kind"] not in HOST_KINDS:
            continue
        w, centre, sill = (frame.value(arch, el["params"][k]) for k in ("width", "center_s", "sill"))
        if sill >= height:
            continue
        for side in _faces(arch, el, host, frame.wall_geometry(arch, host)):
            gaps.setdefault((host["id"], side), []).append((centre - w / 2, centre + w / 2))
    return gaps


def _partition_chains(arch: dict) -> list[dict]:
    out = []
    for el in arch["elements"]:
        if el["kind"] != "partition":
            continue
        geom = frame.wall_geometry(arch, el)
        for side in (1, -1):
            ends = [frame.face_point(geom, s, side) for s in (0.0, geom["length"])]
            out.append({"points": ends, "normals": [(geom["n"][0] * side, geom["n"][1] * side)],
                        "tags": [(el["id"], side, 0.0, geom["length"])], "closed": False})
    return out


def _cornice(arch: dict, el: dict, collection) -> bpy.types.Object | None:
    height, projection = _spec(arch, el)
    profile, known = profiles.resolve("cornice", el.get("profile"))
    bm = bmesh.new()
    for ceiling in (e for e in arch["elements"] if e["kind"] == "ceiling"):
        level = frame.value(arch, ceiling["params"]["level"])
        for region in ceiling["region"]:
            _horizontal(bm, _chains(arch, region["exterior"]), level - height, profile, projection, height)
    if not bm.faces:
        bm.free()
        return None
    return _finish(el["id"], bm, collection, _flags(el, known))


def _skirting(arch: dict, el: dict, hosts: dict, collection) -> bpy.types.Object | None:
    height, projection = _spec(arch, el)
    profile, known = profiles.resolve("skirting", el.get("profile"))
    floor = next((e for e in arch["elements"] if e["kind"] == "floor"), None)
    chains = _chains(arch, floor["polygon"]) if floor else []
    gaps = _skirting_gaps(arch, hosts, height)
    bm = bmesh.new()
    for chain in [*chains, *_partition_chains(arch)]:
        _horizontal(bm, _cut(chain, gaps), frame.value(arch, "floor.level"), profile, projection, height)
    if not bm.faces:
        bm.free()
        return None
    return _finish(el["id"], bm, collection, _flags(el, known))


def _on_face(geom: dict, side: int, path: list, normals: list, closed: bool) -> tuple[list, list, tuple]:
    """Lifts a path in the (s, y) plane of a host face to the room frame: points, mitre vectors, face normal."""
    ends = [frame.face_point(geom, s, side) for s in (0.0, 1.0)]
    u = (ends[1][0] - ends[0][0], ends[1][1] - ends[0][1])
    points = []
    for s, y in path:
        x, z = frame.face_point(geom, s, side)
        points.append((x, y, z))
    across = [(u[0] * ds, dy, u[1] * ds) for ds, dy in _mitre(normals, closed)]
    return points, across, (geom["n"][0] * side, 0.0, geom["n"][1] * side)


def _band(arch: dict, name: str, host: dict, side: int, path: list, normals: list, closed: bool, spec: tuple,
          collection) -> bpy.types.Object:
    """One profile swept along a rectangle (or three sides of one) drawn on a face of `host`.

    `spec` is (kind, profile name, width, projection, element flags).
    """
    kind, profile_name, width, projection, flags = spec
    profile, known = profiles.resolve(kind, profile_name)
    points, across, fixed = _on_face(frame.wall_geometry(arch, host), side, path, normals, closed)
    bm = bmesh.new()
    _sweep(bm, points, across, fixed, profile, (width, projection), (0.0, -BITE), closed)
    return _finish(name, bm, collection, [*flags, *([] if known else ["unknown_profile"])])


def _architraves(arch: dict, el: dict, hosts: dict, y_floor: float, collection) -> list[bpy.types.Object]:
    host = hosts.get(el.get("host"))
    spec = el.get("frame")
    if not spec or host is None or host["kind"] not in HOST_KINDS:
        return []
    geom = frame.wall_geometry(arch, host)
    w, h, sill, centre = (frame.value(arch, el["params"][k]) for k in ("width", "height", "sill", "center_s"))
    s0, s1, y0 = centre - w / 2, centre + w / 2, y_floor + sill
    closed = sill > SILL_TOL
    path = [(s0, y0), (s0, y0 + h), (s1, y0 + h), (s1, y0)]
    normals = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)] + ([(0.0, -1.0)] if closed else [])
    sides = _faces(arch, el, host, geom)
    out = []
    for side in sides:
        suffix = "" if len(sides) == 1 else ("_A" if side == 1 else "_B")
        band = ("architrave", spec.get("profile"), spec["width"], ARCHITRAVE_PROJECTION, el.get("flags", []))
        out.append(_band(arch, f"{el['id']}_Architrave{suffix}", host, side, path, normals, closed, band, collection))
    return out


def _panel_frame(arch: dict, el: dict, hosts: dict, y_floor: float, collection) -> bpy.types.Object | None:
    host = hosts.get(el.get("host"))
    if host is None or host["kind"] not in HOST_KINDS:
        return None
    s0, y0, s1, y1 = el["rect"]
    path = [(s0, y_floor + y0), (s0, y_floor + y1), (s1, y_floor + y1), (s1, y_floor + y0)]
    normals = [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)]
    band = ("panel_frame", el.get("profile"), PANEL_WIDTH, PANEL_PROJECTION, el.get("flags", []))
    return _band(arch, el["id"], host, frame.side(host, el.get("face")), path, normals, True, band, collection)


def add(arch: dict, collections: dict, ctx: dict) -> list[bpy.types.Object]:
    """Cornice, skirting and architraves in `Moulures`, panel frames in `À vérifier`."""
    hosts = {e["id"]: e for e in arch["elements"]}
    y_floor = frame.value(arch, "floor.level")
    moulures, out = collections["Moulures"], []
    for el in arch["elements"]:
        kind = el["kind"]
        if kind == "cornice":
            out.append(_cornice(arch, el, moulures))
        elif kind == "skirting":
            out.append(_skirting(arch, el, hosts, moulures))
        elif kind == "panel_frame":
            out.append(_panel_frame(arch, el, hosts, y_floor, collections["À vérifier"]))
        elif kind in shell.CUT_KINDS:
            out += _architraves(arch, el, hosts, y_floor, moulures)
    return [o for o in out if o is not None]
