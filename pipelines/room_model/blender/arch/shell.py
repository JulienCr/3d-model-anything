"""The architectural shell: floor, ceilings, walls and partitions (cut by their openings), beams and columns."""
import bmesh
import bpy
from mathutils import Vector, geometry

from . import frame, plan

CUT_KINDS = ("door", "window", "opening", "niche")
CUT_OVERSHOOT = 0.01  # m a cutter sticks out of each face of its host
COPLANAR_TOL = 0.005  # m: a cutter edge closer than this to a wall edge is pushed past it


def _object(name: str, bm: bmesh.types.BMesh, solid: bool) -> bpy.types.Object:
    if solid:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return bpy.data.objects.new(name, mesh)


def _prism(name: str, poly: list, y0: float, y1: float) -> bpy.types.Object:
    """Closed box over a convex plan polygon (room-frame [x, z] points) between heights y0 and y1."""
    bm = bmesh.new()
    n = len(poly)
    verts = [bm.verts.new(frame.to_blender(x, y, z)) for y in (y0, y1) for x, z in poly]
    for ring in (range(n), range(n, 2 * n), *((k, (k + 1) % n, (k + 1) % n + n, k + n) for k in range(n))):
        bm.faces.new([verts[i] for i in ring])
    return _object(name, bm, solid=True)


def _sheet(name: str, regions: list[tuple[list, list]], y: float, up: bool) -> bpy.types.Object:
    """Flat triangulated polygons (exterior, holes) at height y; the faces point up or down."""
    bm = bmesh.new()
    for exterior, holes in regions:
        rings = [r[:-1] if len(r) > 1 and r[0] == r[-1] else r for r in (exterior, *holes)]
        verts = [bm.verts.new(frame.to_blender(x, y, z)) for ring in rings for x, z in ring]
        flat = [[Vector((x, -z, 0.0)) for x, z in ring] for ring in rings]
        for tri in geometry.tessellate_polygon(flat):
            a, b, c = (verts[i] for i in tri)
            area = (b.co.x - a.co.x) * (c.co.y - a.co.y) - (b.co.y - a.co.y) * (c.co.x - a.co.x)
            if area == 0.0:
                continue
            bm.faces.new((a, b, c) if (area > 0) == up else (a, c, b))
    return _object(name, bm, solid=False)


def recess_depth(arch: dict, geom: dict, el: dict) -> float | None:
    """Depth of a niche that stops short of the far face; None when it goes right through (or `el` is not a niche)."""
    if el["kind"] != "niche" or el.get("through"):
        return None
    depth = frame.value(arch, el["params"]["depth"])
    return depth if depth < geom["thickness"] - COPLANAR_TOL else None


def _cutter(arch: dict, geom: dict, opening: dict, y_floor: float, host: dict) -> tuple[bpy.types.Object, list] | None:
    """The cutting box of an opening and its plan quad; None for an opening of no size."""
    p = opening["params"]
    w, h, sill, centre = (frame.value(arch, p[k]) for k in ("width", "height", "sill", "center_s"))
    if w <= 0 or h <= 0:
        return None
    side = frame.side(host, opening.get("face"))
    a, b = (frame.face_point(geom, centre + d, side) for d in (-w / 2, w / 2))
    inward = (-geom["n"][0] * side, -geom["n"][1] * side)  # from the cut face into the wall
    recess = recess_depth(arch, geom, opening)
    front, back = CUT_OVERSHOOT, (geom["thickness"] + CUT_OVERSHOOT if recess is None else recess)
    quad = [(a[0] - inward[0] * front, a[1] - inward[1] * front), (b[0] - inward[0] * front, b[1] - inward[1] * front),
            (b[0] + inward[0] * back, b[1] + inward[1] * back), (a[0] + inward[0] * back, a[1] + inward[1] * back)]
    y0, y1 = y_floor + sill, y_floor + sill + h
    if sill <= COPLANAR_TOL:
        y0 = y_floor - 2 * CUT_OVERSHOOT
    if y1 >= geom["top"] - COPLANAR_TOL:
        y1 = geom["top"] + 2 * CUT_OVERSHOOT
    return _prism(f"Cutter_{opening['id']}", quad, y0, y1), quad


def _apply_cuts(obj: bpy.types.Object, cutters: list[bpy.types.Object]) -> None:
    """Boolean-difference each cutter from `obj` and bake the result (no modifier left)."""
    for cutter in cutters:
        mod = obj.modifiers.new(cutter.name, "BOOLEAN")
        mod.operation, mod.operand_type, mod.solver, mod.object = "DIFFERENCE", "OBJECT", "EXACT", cutter
    bpy.context.view_layer.update()
    baked = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    old = obj.data
    obj.data = baked
    obj.modifiers.clear()
    bpy.data.meshes.remove(old)
    baked.name = obj.name


def _beam(arch: dict, el: dict) -> bpy.types.Object:
    p = el["params"]
    (ax, az), (bx, bz) = el["axis"]
    length = ((bx - ax) ** 2 + (bz - az) ** 2) ** 0.5
    sx, sz = -(bz - az) / length * frame.value(arch, p["width"]) / 2, (bx - ax) / length * frame.value(arch, p["width"]) / 2
    quad = [(ax - sx, az - sz), (bx - sx, bz - sz), (bx + sx, bz + sz), (ax + sx, az + sz)]
    return _prism(el["id"], quad, frame.value(arch, p["underside"]), frame.value(arch, p["top"]))


def _column(arch: dict, el: dict, y_floor: float) -> bpy.types.Object:
    cx, cz, sx, sz = (frame.value(arch, el["params"][k]) for k in ("center_x", "center_z", "size_x", "size_z"))
    quad = [(cx - sx / 2, cz - sz / 2), (cx + sx / 2, cz - sz / 2), (cx + sx / 2, cz + sz / 2), (cx - sx / 2, cz + sz / 2)]
    return _prism(el["id"], quad, y_floor, frame.value(arch, el["params"]["top"]))


def _cutters(arch: dict, y_floor: float, hosts: dict, scene_collection) -> list[tuple[bpy.types.Object, list]]:
    """A cutter for every opening on a wall or partition, linked to the scene until `_remove` drops it."""
    out = []
    for o in arch["elements"]:
        host = hosts.get(o.get("host"))
        if o["kind"] not in CUT_KINDS or host is None:
            continue
        cutter = _cutter(arch, frame.wall_geometry(arch, host), o, y_floor, host)
        if cutter is not None:
            scene_collection.objects.link(cutter[0])
            out.append(cutter)
    return out


def _remove(obj: bpy.types.Object) -> None:
    mesh = obj.data
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)


def build(arch: dict, collections: dict, materials: dict) -> list[bpy.types.Object]:
    """One object per shell element, named by element id, in `Coque` (`À vérifier` when flagged to_verify).

    Walls are mitred at their corners, partitions end on the face they meet, and every opening cuts
    each wall or partition its box crosses.
    """
    scene_collection = bpy.context.scene.collection
    y_floor = frame.value(arch, "floor.level")
    walls = plan.wall_footprints(arch)
    footprints = {**walls, **plan.partition_footprints(arch, walls)}
    hosts = {e["id"]: e for e in arch["elements"] if e["kind"] in ("wall", "partition")}
    cutters = _cutters(arch, y_floor, hosts, scene_collection)
    out = []
    for el in arch["elements"]:
        kind = el["kind"]
        target = collections["À vérifier" if "to_verify" in el.get("flags", []) else "Coque"]
        if kind in ("wall", "partition"):
            obj, material = _prism(el["id"], footprints[el["id"]], y_floor, frame.wall_geometry(arch, el)["top"]), "Mur"
            crossing = [c for c, quad in cutters if plan.overlap(footprints[el["id"]], quad)]
            if crossing:
                target.objects.link(obj)
                _apply_cuts(obj, crossing)
        elif kind == "floor":
            obj, material = _sheet(el["id"], [(el["polygon"], [])], frame.value(arch, el["params"]["level"]), up=True), "Sol"
        elif kind == "ceiling":
            regions = [(r["exterior"], r["holes"]) for r in el["region"]]
            obj, material = _sheet(el["id"], regions, frame.value(arch, el["params"]["level"]), up=False), "Plafond"
        elif kind == "beam":
            obj, material = _beam(arch, el), "Mur"
        elif kind == "column":
            obj, material = _column(arch, el, y_floor), "Mur"
        else:
            continue
        if obj.name not in target.objects:
            target.objects.link(obj)
        obj.data.materials.append(materials[material])
        out.append(obj)
    for cutter, _ in cutters:
        _remove(cutter)
    return out
