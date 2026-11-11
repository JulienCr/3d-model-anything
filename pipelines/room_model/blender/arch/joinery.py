"""Doors and windows: a frame in the cut, leaves and glass, rigged to the custom properties of one empty per element.

Local axes of the empty (origin: opening centre, bottom, mid-thickness): +Y is the swing direction, +Z up, +X is the
viewer's right when facing +Y. Leaf `k` (1-based) is `<id>_Leaf_<k>`, its pane `<id>_Glass_<k>` is parented to it.
"""
import math

import bmesh
import bpy

from . import catalogue, frame, openings, plan
from .catalogue import FIXED, HINGED, POCKET, SLIDING

FRAME_DEPTH = 0.10  # m, frame depth when the wall is thicker than this
BITE = 0.002  # m the frame sinks into the reveal, so no two faces are coplanar
LEAF_T = 0.04
RAIL = 0.06  # m, width of the rails around a pane
GLASS_T = 0.006
PANEL_RAISE, PANEL_MARGIN, PANEL_GAP = 0.01, 0.07, 0.05
HANDLE = (0.12, 0.05, 0.025)  # m: along the leaf, off the face, height
HANDLE_INSET = 0.05  # m from the closing edge
HANDLE_HEIGHT = 1.0  # m above the sill on a door
OVERLAP = 0.03  # m a surface sliding door covers on each side of the opening
MIN_SIZE = 0.2  # m: a smaller opening gets no joinery
DEG = math.pi / 180
RIG_PROPS = {  # name: (default, description, (min, max) or None for a checkbox)
    "Ouverture": (0.0, "Ouverture : 0 fermé, 1 ouvert", (0.0, 1.0)),
    "Inverser sens": (False, "Coché : le sens d'ouverture est l'inverse de celui détecté ; sans effet sur une porte ou une fenêtre coulissante", None),
    "Inverser charnière": (False, "Coché : charnières (ou coulisse) de l'autre côté ; sans effet sur deux vantaux", None),
    "AngleMax": (90.0, "Angle maximal des vantaux battants, en degrés", (0.0, 180.0)),
}
VARIABLES = {"ouv": "Ouverture", "inv_sens": "Inverser sens", "inv_hinge": "Inverser charnière", "amax": "AngleMax"}
SWING_SIGN, HINGE_SIGN = "(1-2*inv_sens)", "(1-2*inv_hinge)"  # 1 unchecked, -1 checked


def _box_mesh(name: str, boxes: list[tuple]) -> bpy.types.Mesh:
    """One closed box per (x0, x1, y0, y1, z0, z1)."""
    bm = bmesh.new()
    for x0, x1, y0, y1, z0, z1 in boxes:
        v = [bm.verts.new((x, y, z)) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        for ring in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            bm.faces.new([v[i] for i in ring])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _object(name, boxes, material, parent, collection, location=(0.0, 0.0, 0.0)) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, _box_mesh(name, boxes))
    obj.data.materials.append(bpy.data.materials[material])
    obj.parent, obj.location = parent, location
    collection.objects.link(obj)
    return obj


def _driver(obj: bpy.types.Object, path: str, index: int, expression: str, root: bpy.types.Object) -> None:
    driver = obj.driver_add(path, index).driver
    driver.type, driver.expression = "SCRIPTED", expression
    for name, prop in VARIABLES.items():
        if name in expression:
            var = driver.variables.new()
            var.name, var.type = name, "SINGLE_PROP"
            var.targets[0].id, var.targets[0].data_path = root, f'["{prop}"]'


def _frame_boxes(kind: str, lay: openings.Layout, depth: float) -> list[tuple]:
    """The fixed frame around the leaf: a jamb on each side, a head, and a sill bar on a window."""
    w, h, leaf, jamb = lay.cut_w, lay.cut_h, lay.leaf_w / 2, lay.jamb
    hw, hd, z0 = w / 2 + BITE, depth / 2, (0.0 if kind == "door" else -BITE)
    boxes = [(-hw, -leaf, -hd, hd, z0, h + BITE), (leaf, hw, -hd, hd, z0, h + BITE), (-leaf, leaf, -hd, hd, h - jamb, h + BITE)]
    if kind == "window":
        boxes.append((-leaf, leaf, -hd, hd, z0, jamb))
    return boxes


def _panels(lo: float, hi: float, height: float, count: int) -> list[tuple]:
    """`count` raised rectangles on each face of a leaf spanning x in [lo, hi], evenly split along its height."""
    if count <= 0 or hi - lo < 2 * PANEL_MARGIN + 0.02:
        return []
    each = (height - 2 * PANEL_MARGIN - (count - 1) * PANEL_GAP) / count
    out = []
    for k in range(count):
        z0 = PANEL_MARGIN + k * (each + PANEL_GAP)
        for y0, y1 in ((LEAF_T / 2, LEAF_T / 2 + PANEL_RAISE), (-LEAF_T / 2 - PANEL_RAISE, -LEAF_T / 2)):
            out.append((lo + PANEL_MARGIN, hi - PANEL_MARGIN, y0, y1, z0, z0 + each))
    return out


def _handles(closing_x: float, inward: float, z: float) -> list[tuple]:
    """A lever on each face near the closing edge at `closing_x`; `inward` (+1/-1) points into the leaf."""
    a, b = sorted((closing_x + inward * HANDLE_INSET, closing_x + inward * (HANDLE_INSET + HANDLE[0])))
    return [(a, b, y0, y1, z, z + HANDLE[2]) for y0, y1 in ((LEAF_T / 2, LEAF_T / 2 + HANDLE[1]),
                                                           (-LEAF_T / 2 - HANDLE[1], -LEAF_T / 2))]


def _leaf_boxes(lo: float, hi: float, height: float, glazed: bool) -> tuple[list[tuple], tuple | None]:
    """(leaf boxes, glass box) of a leaf spanning x in [lo, hi] and z in [0, height]."""
    h = LEAF_T / 2
    if not glazed:
        return [(lo, hi, -h, h, 0.0, height)], None
    boxes = [(lo, lo + RAIL, -h, h, 0.0, height), (hi - RAIL, hi, -h, h, 0.0, height),
             (lo + RAIL, hi - RAIL, -h, h, 0.0, RAIL), (lo + RAIL, hi - RAIL, -h, h, height - RAIL, height)]
    return boxes, (lo + RAIL, hi - RAIL, -GLASS_T / 2, GLASS_T / 2, RAIL, height - RAIL)


def _leaves(j: catalogue.Joinery, inner: float, wall: float) -> list[dict]:
    """Leaves in the frame of the empty: `pivot` (x, y), `x` extent around the pivot, closing edge and the motion.

    `mirror` leaves switch jamb with `Inverser charnière`: everything but fixed panes and pairs of hinged leaves.
    """
    spec, left = j.spec, j.hinge == "left"
    if spec.motion == HINGED:
        width = inner / spec.leaves
        hinges = [(-inner / 2, 1), (inner / 2, -1)] if spec.leaves == 2 else [(-inner / 2, 1) if left else (inner / 2, -1)]
        return [{"pivot": (x, 0.0), "x": sorted((0.0, e * width)), "closing": e * width, "inward": -e, "rotation": e,
                 "mirror": spec.leaves == 1} for x, e in hinges]
    toward = -1 if left else 1  # the stack side
    if spec.motion == FIXED:
        return [{"pivot": (0.0, 0.0), "x": [-inner / 2, inner / 2], "closing": 0.0, "inward": 0, "handle": False, "mirror": False}]
    if spec.leaves == 2:  # sliding window: a fixed pane on the stack side, a sliding one on the other rail
        width, rail = inner / 2, LEAF_T / 2 + 0.002
        return [{"pivot": (toward * width / 2, -rail), "x": [-width / 2, width / 2], "closing": 0.0, "inward": 0, "handle": False,
                 "slide": 0.0, "mirror": True},
                {"pivot": (-toward * width / 2, rail), "x": [-width / 2, width / 2], "slide": toward * width,
                 "closing": -toward * width / 2, "inward": toward, "mirror": True}]
    width = inner + (2 * OVERLAP if spec.motion == SLIDING else 0.0)
    y = wall / 2 + LEAF_T / 2 + 0.01 if spec.motion == SLIDING else 0.0  # a surface door runs in front of the wall
    return [{"pivot": (0.0, y), "x": [-width / 2, width / 2], "slide": toward * width, "closing": -toward * width / 2,
             "inward": toward, "mirror": True}]


def _rig(obj: bpy.types.Object, leaf: dict, root: bpy.types.Object) -> None:
    """Drivers of a leaf; a `mirror` leaf is flipped about the centre of the opening by `Inverser charnière`.

    `Inverser sens` only reverses a rotation: a sliding leaf has no swing.
    """
    flip = HINGE_SIGN if leaf["mirror"] else "1"
    px = leaf["pivot"][0]
    if "rotation" in leaf:
        _driver(obj, "rotation_euler", 2, f"({leaf['rotation']})*ouv*amax*{DEG!r}*{SWING_SIGN}*{flip}", root)
    if leaf["mirror"]:
        _driver(obj, "location", 0, f"({px:.6f}+({leaf.get('slide', 0.0):.6f})*ouv)*{flip}", root)
        _driver(obj, "scale", 0, flip, root)


def _root(arch: dict, el: dict, j: catalogue.Joinery, lay: openings.Layout, host: dict, geom: dict, collection) -> bpy.types.Object:
    sill, (nx, nz) = lay.bottom, geom["n"]
    (xa, za), (xb, zb) = (frame.face_point(geom, lay.centre, side) for side in (1, -1))
    x, z = (xa + xb) / 2, (za + zb) / 2  # the middle of the built thickness
    away = (-1 if j.swing == "out" else 1) * (-1 if host["kind"] == "partition" and el.get("face") == "B" else 1)
    root = bpy.data.objects.new(el["id"], None)
    root.empty_display_type, root.empty_display_size = "PLAIN_AXES", 0.15
    root.location = frame.to_blender(x, frame.value(arch, "floor.level") + sill, z)
    root.rotation_euler = (0.0, 0.0, math.atan2(-away * nx, -away * nz))  # local +Y = swing direction
    for name, (default, text, span) in RIG_PROPS.items():
        root[name] = default
        ui = root.id_properties_ui(name)
        if span is None:
            ui.update(default=default, description=text)
        else:
            ui.update(min=span[0], max=span[1], soft_min=span[0], soft_max=span[1], description=text)
    if lay.flags:
        root["flags"] = ",".join(lay.flags)
    collection.objects.link(root)
    return root


def _build(arch: dict, el: dict, host: dict, geom: dict, j: catalogue.Joinery, collection) -> list[bpy.types.Object]:
    lay = openings.resolve(arch, el)
    t = frame.value(arch, host["params"]["thickness"])
    depth = FRAME_DEPTH if t >= FRAME_DEPTH + 0.02 else t + 0.01
    bottom = 0.0 if j.kind == "door" else lay.jamb
    root = _root(arch, el, j, lay, host, geom, collection)
    out = [root, _object(f"{el['id']}_Frame", _frame_boxes(j.kind, lay, depth), "Menuiserie", root, collection)]
    height = lay.leaf_h
    for k, leaf in enumerate(_leaves(j, lay.leaf_w, t), start=1):
        lo, hi = leaf["x"]
        boxes, pane = _leaf_boxes(lo, hi, height, j.spec.glazed)
        if not j.spec.glazed:
            boxes += _panels(lo, hi, height, j.panels)
        if j.handle and leaf.get("handle", True):
            boxes += _handles(leaf["closing"], leaf["inward"], HANDLE_HEIGHT - bottom if j.kind == "door" else height / 2)
        obj = _object(f"{el['id']}_Leaf_{k}", boxes, "Menuiserie", root, collection, (*leaf["pivot"], bottom))
        _rig(obj, leaf, root)
        out.append(obj)
        if pane is not None:
            out.append(_object(f"{el['id']}_Glass_{k}", [pane], "Verre", obj, collection))
    return out


def add(arch: dict, collections: dict, ctx: dict) -> list[bpy.types.Object]:
    """Frame, leaves and glass of every hosted door and window, in `Menuiseries`."""
    hosts = {e["id"]: e for e in arch["elements"]}
    bodies = plan.built_bodies(arch)
    out = []
    for el in arch["elements"]:
        j = catalogue.resolve(el)
        host = hosts.get(el.get("host"))
        if j is None or host is None or host["kind"] not in ("wall", "partition"):
            continue
        if min(frame.value(arch, el["params"]["width"]), frame.value(arch, el["params"]["height"])) < MIN_SIZE:
            continue
        out += _build(arch, el, host, plan.host_geometry(arch, host, bodies), j, collections["Menuiseries"])
    return out
