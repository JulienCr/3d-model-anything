"""Niche contents: the dividers and shelves inside the cut that `shell` makes in the host (frames are in `mouldings`)."""
import math

import bpy

from . import frame, joinery, shell

BOARD = 0.02  # m, thickness of dividers and shelves


def _boards(width: float, height: float, depth: float, dividers: int, shelves: list) -> list[tuple[str, tuple]]:
    """(name, box) in the niche frame: x across (centred), y into the wall from the cut face, z up from the sill."""
    out = []
    for k in range(1, dividers + 1):
        x = -width / 2 + width * k / (dividers + 1)
        out.append((f"Divider_{k}", (x - BOARD / 2, x + BOARD / 2, 0.0, depth, 0.0, height)))
    for k, z in enumerate(sorted(shelves), start=1):
        if BOARD / 2 <= z <= height - BOARD / 2:
            out.append((f"Shelf_{k}", (-width / 2, width / 2, 0.0, depth, z - BOARD / 2, z + BOARD / 2)))
    return out


def _niche(arch: dict, el: dict, host: dict, collection) -> list[bpy.types.Object]:
    geom = frame.wall_geometry(arch, host)
    w, h, sill, centre = (frame.value(arch, el["params"][k]) for k in ("width", "height", "sill", "center_s"))
    side = frame.side(host, el.get("face"))
    recess = shell.recess_depth(arch, geom, el)
    depth = geom["thickness"] if recess is None else recess
    x, z = frame.face_point(geom, centre, side)
    ux, uz = geom["u"]
    angle = math.atan2(-uz, ux)  # local x -> the host direction, in the Blender plan
    into = (-geom["n"][0] * side, geom["n"][1] * side)  # from the cut face into the wall, in the Blender plan
    flip = 1.0 if -math.sin(angle) * into[0] + math.cos(angle) * into[1] > 0 else -1.0
    out = []
    for name, (x0, x1, y0, y1, z0, z1) in _boards(w, h, depth, int(el.get("dividers") or 0), el.get("shelves") or []):
        box = (x0, x1, *sorted((flip * y0, flip * y1)), z0, z1)
        obj = joinery._object(f"{el['id']}_{name}", [box], "Menuiserie", None, collection,
                              frame.to_blender(x, frame.value(arch, "floor.level") + sill, z))
        obj.rotation_euler = (0.0, 0.0, angle)
        out.append(obj)
    return out


def add(arch: dict, collections: dict, ctx: dict) -> list[bpy.types.Object]:
    """Dividers and shelves of every hosted niche, in `Menuiseries`."""
    hosts = {e["id"]: e for e in arch["elements"]}
    out = []
    for el in arch["elements"]:
        host = hosts.get(el.get("host"))
        if el["kind"] == "niche" and host is not None and host["kind"] in ("wall", "partition"):
            out += _niche(arch, el, host, collections["Menuiseries"])
    return out
