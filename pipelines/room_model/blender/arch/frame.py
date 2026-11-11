"""Room frame (metres, +Y up, plan [x, z]) to the Blender scene (Z up), and parameter access."""
from __future__ import annotations

import math


def to_blender(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x, -z, y)


def value(arch: dict, pid: str) -> float:
    return arch["parameters"][pid]["value"]


def element(arch: dict, id_: str) -> dict | None:
    return next((e for e in arch["elements"] if e["id"] == id_), None)


def wall_geometry(arch: dict, el: dict) -> dict:
    """Plan geometry of a wall or partition. `point(s)` is the point `s` m from `start` along start->end, on the face
    line (offset, azimuth); the body lies behind it, in the direction of -`n`."""
    p = el["params"]
    az, offset = value(arch, p["azimuth"]), value(arch, p["offset"])
    n = (math.cos(az), math.sin(az))
    start, end = el["start"], el["end"]
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    u = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)

    def point(s: float) -> tuple[float, float]:
        x, z = start[0] + u[0] * s, start[1] + u[1] * s
        d = offset - (n[0] * x + n[1] * z)
        return (x + n[0] * d, z + n[1] * d)

    return {"point": point, "length": length, "u": u, "n": n, "thickness": value(arch, p["thickness"]),
            "top": value(arch, p["top"])}


def side(host: dict, face: str | None) -> int:
    """+1 for the face the interior normal points out of (a wall's interior face, a partition's face A), -1 for a partition's face B."""
    return -1 if host["kind"] == "partition" and face == "B" else 1


def face_point(geom: dict, s: float, side_: int) -> tuple[float, float]:
    """Plan point `s` m along the host on the face line of `side_` (+1: the front face, -1: the back one)."""
    x, z = geom["point"](s)
    if side_ < 0:
        x, z = x - geom["n"][0] * geom["thickness"], z - geom["n"][1] * geom["thickness"]
    return x, z
