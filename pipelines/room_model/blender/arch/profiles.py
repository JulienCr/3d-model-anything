"""Moulding profiles: closed 2D polygons in unit coordinates (a, b), scaled by the caller.

For cornices and skirting, a runs out from the wall and b up the height (b = 1 touches the ceiling for a cornice).
For architraves, a runs outward across the band from the opening edge and b out of the wall face.
"""
import math

RECT = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _arc(cx: float, cy: float, rx: float, ry: float, start: float, end: float, steps: int = 8) -> list[tuple[float, float]]:
    """Points of an elliptical arc from angle `start` to `end` (degrees)."""
    angles = [math.radians(start + (end - start) * k / steps) for k in range(steps + 1)]
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in angles]


def _s_curve(p0: tuple, p1: tuple, steps: int = 8) -> list[tuple[float, float]]:
    """S-shaped curve from p0 to p1, vertical at both ends."""
    out = []
    for k in range(steps + 1):
        u = k / steps
        smooth = u * u * (3 - 2 * u)
        out.append((p0[0] + (p1[0] - p0[0]) * smooth, p0[1] + (p1[1] - p0[1]) * u))
    return out


PROFILES = {
    "cornice": {
        "flat": RECT,
        "cove": _arc(1.0, 0.0, 1.0, 1.0, 180.0, 90.0) + [(0.0, 1.0)],
        "ogee": _s_curve((0.0, 0.0), (1.0, 1.0)) + [(0.0, 1.0)],
        "stepped": [(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.7, 0.4), (0.7, 0.7), (1.0, 0.7), (1.0, 1.0), (0.0, 1.0)],
    },
    "skirting": {
        "flat": RECT,
        "bevel": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.75), (0.4, 1.0), (0.0, 1.0)],
        "ogee": [(0.0, 0.0), (1.0, 0.0), *_s_curve((1.0, 0.45), (0.25, 1.0)), (0.0, 1.0)],
    },
    "architrave": {
        "flat": RECT,
        "moulded": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.9, 0.75), (0.7, 0.95), (0.45, 1.0), (0.3, 1.0), (0.2, 0.7),
                    (0.0, 0.7)],
    },
    "panel_frame": {
        "quarter_round": [(0.0, 0.0), *_arc(0.0, 0.0, 1.0, 1.0, 0.0, 90.0)],
        "ogee": [(0.0, 0.0), (1.0, 0.0), *_s_curve((1.0, 0.3), (0.1, 1.0)), (0.0, 1.0)],
    },
}


def resolve(kind: str, name: str | None) -> tuple[list[tuple[float, float]], bool]:
    """(polygon, known) for a profile of `kind`; an unknown or missing name gives a plain rectangle and known=False."""
    table = PROFILES[kind]
    if name in table:
        return table[name], True
    return RECT, False
