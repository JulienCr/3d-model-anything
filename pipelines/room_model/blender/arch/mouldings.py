"""Cornices, skirting, architraves and panel frames: 2D profiles swept along wall faces, mitred at the corners.

A chain is `{"points": plan points, "normals": one inward normal per segment, "tops": top of the host at each point,
"closed": bool}`: consecutive wall faces, then each face of a partition.
"""
import itertools
import math

import bmesh
import bpy

from . import clip, frame, openings, plan, profiles, shell

BITE = 0.001  # m a moulding sinks into its host, so no two faces are coplanar
DEFAULT_PROJECTION = 0.015
TOP_STEP = 0.01  # m: a cornice stops where the wall top changes by more than this from one wall to the next
LEDGE_MIN = 0.001  # m: the faces of two joined walls are this far apart where they meet, so a step joins them
MIN_PIECE = 0.01  # m
MAX_MITRE = 4.0  # floor of 1 / (1 + cos) so a near fold-back does not shoot off
EPS = 1e-9
JOIN_EPS = 1e-4  # m: chain ends this close are the same point (a tenth of the mesh precision)
ARCHITRAVE_PROJECTION = 0.015
PANEL_WIDTH, PANEL_PROJECTION = 0.03, 0.015
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


def _mitre_widths(normals: list, widths: list, closed: bool) -> list[tuple[float, float]]:
    """Per-vertex 2D offset that lands on the offset line of each adjacent segment, at that segment's own width."""
    n = len(normals)
    out = []
    for i in range(n if closed else n + 1):
        before = (i - 1) % n if (closed or i > 0) else None
        after = i % n if (closed or i < n) else None
        if before is None or after is None:
            k = after if before is None else before
            out.append((normals[k][0] * widths[k], normals[k][1] * widths[k]))
            continue
        (bx, by), (ax, ay), db, da = normals[before], normals[after], widths[before], widths[after]
        det = bx * ay - by * ax
        if abs(det) < 0.25:  # nearly straight: the equal-width mitre at the mean width
            k = max(1.0 + bx * ax + by * ay, 1.0 / MAX_MITRE)
            out.append(((bx + ax) / k * (db + da) / 2, (by + ay) / k * (db + da) / 2))
        else:
            out.append(((db * ay - da * by) / det, (bx * da - ax * db) / det))
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


def _segment(arch: dict, host: dict, side: int, a, b) -> dict:
    geom = frame.wall_geometry(arch, host)
    normal = (geom["n"][0] * side, geom["n"][1] * side)
    return {"a": a, "b": b, "normal": normal, "top": geom["top"], "u": geom["u"]}


def _step(prev: dict, a, b) -> dict:
    """The end face that sticks out where two joined walls are not flush, from the end of `prev` to the start of the next wall."""
    jog, u = (b[0] - a[0], b[1] - a[1]), prev["u"]
    perp = (-jog[1], jog[0])
    forward = -1.0 if jog[0] * prev["normal"][0] + jog[1] * prev["normal"][1] > 0 else 1.0  # the free side is ahead if the next face is behind
    if (perp[0] * u[0] + perp[1] * u[1]) * forward < 0:
        perp = (-perp[0], -perp[1])
    size = math.hypot(*jog)
    return {"a": a, "b": b, "normal": (perp[0] / size, perp[1] / size), "top": prev["top"], "u": u}


def _chain(segments: list, closed: bool) -> dict:
    """Chain of consecutive segments; the top at a point shared by two walls is the mean of theirs."""
    tops = [s["top"] for s in segments]
    shared = [(tops[k - 1] + tops[k]) / 2 for k in range(len(tops))]
    return {"points": [s["a"] for s in segments] + ([] if closed else [segments[-1]["b"]]),
            "normals": [s["normal"] for s in segments], "closed": closed,
            "tops": shared if closed else [tops[0], *shared[1:], tops[-1]]}


def _split(segments: list, closed: bool, step: float | None) -> list[dict]:
    """Chains of `segments`, cut wherever the top changes by more than `step` (never when it is None)."""
    breaks = [k for k in range(len(segments)) if step is not None
              and abs(segments[k - 1]["top"] - segments[k]["top"]) > step and (closed or k > 0)]
    if not breaks:
        return [_chain(segments, closed)]
    first = breaks[0] if closed else 0
    runs, cur = [], []
    for k in [*range(first, len(segments)), *range(first)] if closed else range(len(segments)):
        if k in breaks and cur:
            runs.append(cur)
            cur = []
        cur.append(segments[k])
    return [_chain(run, False) for run in [*runs, cur]]


def _face_chains(arch: dict, step: float | None) -> list[dict]:
    """Chains along the interior face of every wall (consecutive joined walls share a chain) and of both sides of every partition."""
    hosts = {e["id"]: e for e in arch["elements"]}
    walls = plan.wall_footprints(arch)
    links = plan.wall_links(arch)
    order = [e["id"] for e in arch["elements"] if e["kind"] == "wall"]
    seen, out = set(), []
    for first in [i for i in order if i not in links.values()] + order:
        if first in seen:
            continue
        ids, cur = [], first
        while cur is not None and cur not in seen:
            seen.add(cur)
            ids.append(cur)
            cur = links.get(cur)
        segments = []
        for k, wid in enumerate(ids):
            quad = walls[wid]
            segments.append(_segment(arch, hosts[wid], 1, quad[0], quad[1]))
            following = ids[(k + 1) % len(ids)] if k + 1 < len(ids) or cur == first else None
            if following is not None and math.dist(quad[1], walls[following][0]) > LEDGE_MIN:
                segments.append(_step(segments[-1], quad[1], walls[following][0]))
        out += _split(segments, cur == first, step)
    for pid, quad in plan.partition_footprints(arch, walls).items():
        for side, a, b in ((1, quad[0], quad[1]), (-1, quad[3], quad[2])):
            out.append(_chain([_segment(arch, hosts[pid], side, a, b)], False))
    return out


def _kept(length: float, spans: list) -> list[tuple[float, float]]:
    """Parameter ranges (ta, tb) of a segment of `length` m left once the (enter, leave) spans are removed; short pieces go."""
    pieces, cursor = [], 0.0
    for a, b in sorted(spans):
        if cursor >= 1.0:
            break
        if a > cursor:
            pieces.append((cursor, min(a, 1.0)))
        cursor = max(cursor, b)
    if cursor < 1.0:
        pieces.append((cursor, 1.0))
    return [(a, b) for a, b in pieces if (b - a) * length >= MIN_PIECE]


def _cut(chain: dict, cuts: list) -> list[dict]:
    """The chain with the parts inside the plan quads `cuts` removed; each remainder is an open chain."""
    pts, closed = chain["points"], chain["closed"]
    ends = [(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(chain["normals"]))]
    kept = [_kept(math.dist(a, b), [c for c in (plan.span(a, b, quad) for quad in cuts) if c]) for a, b in ends]
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


def _horizontal(bm, chains: list, levels: list | float, profile: list, projection: float, height: float) -> None:
    """Sweeps the profile along each chain, at `levels` (one per point of a chain, or one for all)."""
    for chain in chains:
        pts = chain["points"]
        ys = levels if isinstance(levels, list) else [levels] * len(pts)
        across = [(mx, 0.0, mz) for mx, mz in _mitre(chain["normals"], chain["closed"])]
        _sweep(bm, [(x, y, z) for (x, z), y in zip(pts, ys)], across, (0.0, 1.0, 0.0), profile, (projection, height), (-BITE, 0.0),
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


def _floor_cuts(arch: dict, hosts: dict, bodies: dict, solids: dict, height: float) -> list[tuple]:
    """(element, host, geometry, fits) of the cuts that reach into the skirting band, on any wall: a cut is geometry,
    not a host's property, and a niche recessed from one face leaves the other face alone."""
    out = []
    for el in arch["elements"]:
        host = hosts.get(el.get("host"))
        if el["kind"] not in shell.CUT_KINDS or host is None or host["kind"] not in HOST_KINDS:
            continue
        if openings.resolve(arch, el).lowest >= height:
            continue
        geom = plan.host_geometry(arch, host, bodies)
        out.append((el, host, geom, clip.fits(arch, el, host, _faces(arch, el, host, geom), bodies, solids)))
    return out


def _gaps(arch: dict, floor_cuts: list[tuple]) -> list[list]:
    """Plan quads the skirting stops at: each cut out to the widest casing drawn on any face."""
    return [shell.cut_quad(arch, geom, el, host, *clip.outer_span(openings.resolve(arch, el), fitted))[0]
            for el, host, geom, fitted in floor_cuts]


def _reveals(arch: dict, floor_cuts: list[tuple]) -> list[dict]:
    """A chain across the thickness along each jamb of a passage (an opening, or a niche that goes through): its skirting
    returns into the cut, a door keeps its gap."""
    out = []
    for el, host, geom, _ in floor_cuts:
        if el["kind"] == "opening" or (el["kind"] == "niche" and shell.recess_depth(arch, geom, el) is None):
            for s, toward in zip(openings.resolve(arch, el).span, (1.0, -1.0)):
                points = [frame.face_point(geom, s, side) for side in (1, -1)]
                out.append({"points": points, "normals": [(geom["u"][0] * toward, geom["u"][1] * toward)], "closed": False})
    return out


def _turned(chain: dict) -> dict:
    return {**chain, "points": chain["points"][::-1], "normals": chain["normals"][::-1]}


def _join(chains: list[dict]) -> list[dict]:
    """The open chains whose ends meet merged into one, so the mitre at the meeting point is swept."""
    chains = list(chains)
    while True:
        for i, j in itertools.permutations(range(len(chains)), 2):
            if chains[i]["closed"] or chains[j]["closed"]:
                continue
            for a, b in itertools.product((chains[i], _turned(chains[i])), (chains[j], _turned(chains[j]))):
                if math.dist(a["points"][-1], b["points"][0]) < JOIN_EPS:
                    chains[i] = {"points": a["points"] + b["points"][1:], "normals": a["normals"] + b["normals"], "closed": False}
                    del chains[j]
                    break
            else:
                continue
            break
        else:
            return chains


def _cornice(arch: dict, el: dict, collection) -> bpy.types.Object | None:
    height, projection = _spec(arch, el)
    profile, known = profiles.resolve("cornice", el.get("profile"))
    bm = bmesh.new()
    for chain in _face_chains(arch, TOP_STEP):
        _horizontal(bm, [chain], [top - height for top in chain["tops"]], profile, projection, height)
    if not bm.faces:
        bm.free()
        return None
    return _finish(el["id"], bm, collection, _flags(el, known))


def _skirting(arch: dict, el: dict, hosts: dict, bodies: dict, solids: dict, collection) -> bpy.types.Object | None:
    height, projection = _spec(arch, el)
    profile, known = profiles.resolve("skirting", el.get("profile"))
    floor_cuts = _floor_cuts(arch, hosts, bodies, solids, height)
    gaps = _gaps(arch, floor_cuts)
    runs = [piece for chain in _face_chains(arch, None) for piece in _cut(chain, gaps)]
    bm = bmesh.new()
    for chain in _join([*runs, *_reveals(arch, floor_cuts)]):
        _horizontal(bm, [chain], frame.value(arch, "floor.level"), profile, projection, height)
    if not bm.faces:
        bm.free()
        return None
    return _finish(el["id"], bm, collection, _flags(el, known))


def _on_face(geom: dict, side: int, path: list, normals: list, widths: list, closed: bool) -> tuple[list, list, tuple]:
    """Lifts a path in the (s, y) plane of a host face to the room frame: points, mitre vectors, face normal.

    The mitre vectors carry the width of each segment, so the profile is swept with a unit width.
    """
    ends = [frame.face_point(geom, s, side) for s in (0.0, 1.0)]
    u = (ends[1][0] - ends[0][0], ends[1][1] - ends[0][1])
    points = []
    for s, y in path:
        x, z = frame.face_point(geom, s, side)
        points.append((x, y, z))
    across = [(u[0] * ds, dy, u[1] * ds) for ds, dy in _mitre_widths(normals, widths, closed)]
    return points, across, (geom["n"][0] * side, 0.0, geom["n"][1] * side)


def _runs(points: list, normals: list, widths: list, closed: bool) -> list[tuple[list, list, list, bool]]:
    """The polyline cut into runs of consecutive segments that have a width (a dropped casing leaves a gap)."""
    n = len(normals)
    dead = next((i for i in range(n) if not widths[i] > 0), None)
    if dead is None:
        return [(points, normals, widths, closed)]
    runs, cur = [], []
    for i in ([(dead + 1 + k) % n for k in range(n)] if closed else range(n)):
        if widths[i] > 0:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    runs = [*runs, cur] if cur else runs
    return [([points[i % len(points)] for i in (*run, run[-1] + 1)], [normals[i] for i in run], [widths[i] for i in run], False) for run in runs]


def _band(geom: dict, name: str, side: int, path: list, normals: list, widths: list, closed: bool, spec: tuple,
          collection) -> bpy.types.Object:
    """One profile swept along a rectangle (or three sides of one) drawn on a face of a host, `widths[k]` wide along segment k.

    A segment of width 0 is left out. `spec` is (kind, profile name, projection, element flags).
    """
    kind, profile_name, projection, flags = spec
    profile, known = profiles.resolve(kind, profile_name)
    bm = bmesh.new()
    for run in _runs(path, normals, widths, closed):
        points, across, fixed = _on_face(geom, side, *run[:3], run[3])
        _sweep(bm, points, across, fixed, profile, (1.0, projection), (0.0, -BITE), run[3])
    return _finish(name, bm, collection, [*flags, *([] if known else ["unknown_profile"])])


def _architraves(arch: dict, el: dict, hosts: dict, bodies: dict, solids: dict, y_floor: float, collection, verify) -> list[bpy.types.Object]:
    host = hosts.get(el.get("host"))
    spec = el.get("frame")
    if not spec or host is None or host["kind"] not in HOST_KINDS:
        return []
    if "width" not in spec:
        raise ValueError(f"{el['id']}: frame.width is missing")
    geom = plan.host_geometry(arch, host, bodies)
    lay = openings.resolve(arch, el)
    y0 = y_floor + lay.bottom
    sides = _faces(arch, el, host, geom)
    fitted = clip.fits(arch, el, host, sides, bodies, solids)
    flags = [*el.get("flags", []), *(f for f in (*lay.flags, *clip.flags(fitted)) if f == "to_verify")]
    target = verify if "to_verify" in flags else collection
    out = []
    for side in sides:
        (left, top, right, bottom), (s0, s1) = fitted[side].casings, fitted[side].bars
        closed = bottom > 0
        path = [(s0, y0), (s0, y0 + lay.cut_h), (s1, y0 + lay.cut_h), (s1, y0)]
        normals = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)] + ([(0.0, -1.0)] if closed else [])
        widths = [left, top, right] + ([bottom] if closed else [])
        suffix = "" if len(sides) == 1 else ("_A" if side == 1 else "_B")
        band = ("architrave", spec.get("profile"), ARCHITRAVE_PROJECTION, flags)
        out.append(_band(geom, f"{el['id']}_Architrave{suffix}", side, path, normals, widths, closed, band, target))
    return out


def _panel_frame(arch: dict, el: dict, hosts: dict, bodies: dict, y_floor: float, collection) -> bpy.types.Object | None:
    host = hosts.get(el.get("host"))
    if host is None or host["kind"] not in HOST_KINDS:
        return None
    s0, y0, s1, y1 = el["rect"]
    path = [(s0, y_floor + y0), (s0, y_floor + y1), (s1, y_floor + y1), (s1, y_floor + y0)]
    normals = [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)]
    band = ("panel_frame", el.get("profile"), PANEL_PROJECTION, el.get("flags", []))
    geom = plan.host_geometry(arch, host, bodies)
    return _band(geom, el["id"], frame.side(host, el.get("face")), path, normals, [PANEL_WIDTH] * 4, True, band, collection)


def add(arch: dict, collections: dict, ctx: dict) -> list[bpy.types.Object]:
    """Cornice, skirting and architraves in `Moulures`, panel frames in `À vérifier`."""
    hosts = {e["id"]: e for e in arch["elements"]}
    y_floor = frame.value(arch, "floor.level")
    bodies, solids = plan.built(arch)
    moulures, out = collections["Moulures"], []
    for el in arch["elements"]:
        kind = el["kind"]
        if kind == "cornice":
            out.append(_cornice(arch, el, moulures))
        elif kind == "skirting":
            out.append(_skirting(arch, el, hosts, bodies, solids, moulures))
        elif kind == "panel_frame":
            out.append(_panel_frame(arch, el, hosts, bodies, y_floor, collections["À vérifier"]))
        elif kind in shell.CUT_KINDS:
            out += _architraves(arch, el, hosts, bodies, solids, y_floor, moulures, collections["À vérifier"])
    return [o for o in out if o is not None]
