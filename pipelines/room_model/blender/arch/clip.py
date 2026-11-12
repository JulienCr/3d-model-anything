"""Where the casings of a cut can stand: the free length of the host face around it (pure Python).

A body that stands in front of the face (a partition ending on the wall, a neighbour sticking out) covers that stretch of
the face: a casing there would run into it. The casings are cut back to the free stretch around the cut.
"""
import math
from dataclasses import dataclass

from . import openings, plan

FACE_LIFT = 0.001  # m: bodies are looked for this far in front of the face, so one that only touches it does not count
MIN_COVER = 0.001  # m: a covered run shorter than this is mesh precision


@dataclass(frozen=True)
class Fit:
    """What can be drawn on one face: `casings` (left, top, right, bottom) in m, `bars` the stretch (low, high) along the host
    the top and bottom bars run over, `outside` the length of the cut that lies beyond the free stretch, `flags` ("to_verify"
    when a casing was dropped or the cut runs into another body)."""
    casings: tuple[float, float, float, float]
    bars: tuple[float, float]
    flags: tuple[str, ...] = ()
    outside: float = 0.0


def _inside(a, b, solid: list, cuts: list) -> list[tuple[float, float]]:
    """Parameter ranges (0 at a, 1 at b) of the segment inside `solid` and outside every one of `cuts`."""
    span = plan.span(a, b, solid)
    if span is None:
        return []
    pieces = [(max(span[0], 0.0), min(span[1], 1.0))]
    for cut in cuts:
        gap = plan.span(a, b, cut)
        if gap:
            pieces = [p for lo, hi in pieces for p in ((lo, min(hi, gap[0])), (max(lo, gap[1]), hi)) if p[1] > p[0]]
    return [p for p in pieces if p[1] > p[0]]


def free_span(host: dict, side: int, geom: dict, quad: list, solids: dict, near: float) -> tuple[float, float]:
    """(low, high) m along the host: the stretch of its face of `side` that holds `near` (or is the closest to it)
    and has no other body in front of it. `quad` is the host's built body, `solids` those of the shell."""
    a, b = (quad[0], quad[1]) if side > 0 else (quad[3], quad[2])
    lift = (geom["n"][0] * side * FACE_LIFT, geom["n"][1] * side * FACE_LIFT)
    a, b = (a[0] + lift[0], a[1] + lift[1]), (b[0] + lift[0], b[1] + lift[1])
    length = math.dist(a, b)
    origin = (a[0] - host["start"][0]) * geom["u"][0] + (a[1] - host["start"][1]) * geom["u"][1]
    covered = sorted(run for oid, (solid, cuts) in solids.items() if oid != host["id"] for run in _inside(a, b, solid, cuts)
                     if (run[1] - run[0]) * length >= MIN_COVER)
    free, cursor = [], 0.0
    for lo, hi in covered:
        if lo > cursor:
            free.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < 1.0:
        free.append((cursor, 1.0))
    t = (near - origin) / length if length else 0.0
    lo, hi = min(free, key=lambda r: 0.0 if r[0] <= t <= r[1] else min(abs(t - r[0]), abs(t - r[1])), default=(0.0, 1.0))
    return origin + lo * length, origin + hi * length


def fit(lay: openings.Layout, free: tuple[float, float]) -> Fit:
    """The casings of `lay` cut back to the free stretch `free`; one left narrower than CASING_MIN is dropped.

    A cut with no overlap with the free stretch gets no casing at all.
    """
    (c_lo, c_hi), (lo, hi) = lay.span, free
    shown = min(c_hi, hi) - max(c_lo, lo)
    outside = (c_hi - c_lo) - max(shown, 0.0)
    flags = ["to_verify"] if outside > FACE_LIFT else []
    if shown <= 0:
        return Fit((0.0, 0.0, 0.0, 0.0), (c_lo, c_lo), tuple(flags), outside)
    left, top, right, bottom = lay.casings
    sides = []
    for width, room in ((left, c_lo - lo), (right, hi - c_hi)):
        kept = min(width, max(room, 0.0))
        if width > 0 and kept < openings.CASING_MIN:
            kept = 0.0
            flags.append("to_verify")
        sides.append(kept)
    return Fit((sides[0], top, sides[1], bottom), (max(c_lo, lo), min(c_hi, hi)), tuple(dict.fromkeys(flags)), outside)


def fits(arch: dict, el: dict, host: dict, sides: tuple[int, ...], bodies: dict, solids: dict) -> dict[int, Fit]:
    """{side: Fit} of a cut on each face of its host in `sides`, from the built bodies and solids (`plan.built`)."""
    geom = plan.host_geometry(arch, host, bodies)
    lay = openings.resolve(arch, el)
    return {side: fit(lay, free_span(host, side, geom, bodies[host["id"]], solids, lay.centre)) for side in sides}


def flags(fitted: dict[int, Fit]) -> tuple[str, ...]:
    """The flags of all faces together."""
    return tuple(dict.fromkeys(f for one in fitted.values() for f in one.flags))


def outer_span(lay: openings.Layout, fitted: dict[int, Fit]) -> tuple[float, float]:
    """The cut out to the widest casing drawn on any face."""
    left, right = (max(one.casings[k] for one in fitted.values()) for k in (0, 2))
    return lay.span[0] - left, lay.span[1] + right


def overlaps(arch: dict) -> list[tuple[str, float]]:
    """(element id, m of its cut beyond the free face of its host) for every cut that runs into another body or past its host."""
    bodies, solids = plan.built(arch)
    hosts = {e["id"]: e for e in arch["elements"]}
    out = []
    for el in arch["elements"]:
        host = hosts.get(el.get("host"))
        if el["kind"] not in ("door", "window", "opening", "niche") or host is None or host["kind"] not in ("wall", "partition"):
            continue
        sides = (1, -1) if host["kind"] == "partition" else (1,)
        worst = max(f.outside for f in fits(arch, el, host, sides, bodies, solids).values())
        if worst > FACE_LIFT:
            out.append((el["id"], worst))
    return out
