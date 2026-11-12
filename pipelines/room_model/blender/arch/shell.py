"""The architectural shell: floor, ceilings, walls and partitions (cut by their openings), beams and columns."""
import bmesh
import bpy
from mathutils import Vector, geometry

from . import frame, openings, plan

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


def _sheet(name: str, regions: list[tuple], y: float, up: bool) -> bpy.types.Object:
    """Flat triangulated polygons (exterior, holes[, drop]) at height y, a drop m lower; the faces point up or down."""
    bm = bmesh.new()
    for exterior, holes, *drop in regions:
        rings = [r[:-1] if len(r) > 1 and r[0] == r[-1] else r for r in (exterior, *holes)]
        verts = [bm.verts.new(frame.to_blender(x, y - (drop[0] if drop else 0.0), z)) for ring in rings for x, z in ring]
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


def cut_quad(arch: dict, geom: dict, opening: dict, host: dict, lo: float, hi: float) -> tuple[list, list]:
    """(quad, reach): the plan quad cutting the host between `lo` and `hi` m along it, and the same without the overshoots.

    The quad starts CUT_OVERSHOOT in front of the built face the opening is on and ends CUT_OVERSHOOT behind the
    built far one (or after the recess depth), so it pierces a partition snapped off its nominal lines.
    """
    side = frame.side(host, opening.get("face"))
    near = [frame.face_point(geom, s, side) for s in (lo, hi)]
    inward = (-geom["n"][0] * side, -geom["n"][1] * side)  # from the cut face into the wall
    recess = recess_depth(arch, geom, opening)
    if recess is None:
        far = [frame.face_point(geom, s, -side) for s in (lo, hi)]
        back = [(x + inward[0] * CUT_OVERSHOOT, z + inward[1] * CUT_OVERSHOOT) for x, z in far]
    else:
        far = back = [(x + inward[0] * recess, z + inward[1] * recess) for x, z in near]
    front = [(x - inward[0] * CUT_OVERSHOOT, z - inward[1] * CUT_OVERSHOOT) for x, z in near]
    return [front[0], front[1], back[1], back[0]], [near[0], near[1], far[1], far[0]]


def _cutter(arch: dict, geom: dict, opening: dict, y_floor: float, host: dict) -> tuple[bpy.types.Object, list] | None:
    """The cutting box of an opening and the plan quad it cuts through the host (without the overshoots); None for no size."""
    lay = openings.resolve(arch, opening)
    w, h, sill = lay.cut_w, lay.cut_h, lay.bottom
    if w <= 0 or h <= 0:
        return None
    quad, reach = cut_quad(arch, geom, opening, host, *lay.span)
    y0, y1 = y_floor + sill, y_floor + sill + h
    if sill <= COPLANAR_TOL:
        y0 = y_floor - 2 * CUT_OVERSHOOT
    if y1 >= geom["top"] - COPLANAR_TOL:
        y1 = geom["top"] + 2 * CUT_OVERSHOOT
    return _prism(f"Cutter_{opening['id']}", quad, y0, y1), reach


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


def _cutters(arch: dict, y_floor: float, hosts: dict, bodies: dict, scene_collection) -> list[tuple[bpy.types.Object, list, str]]:
    """A cutter for every opening on a wall or partition (`bodies`: built footprint of each), linked to the scene until `_remove` drops it."""
    out = []
    for o in arch["elements"]:
        host = hosts.get(o.get("host"))
        if o["kind"] not in CUT_KINDS or host is None:
            continue
        cutter = _cutter(arch, plan.host_geometry(arch, host, bodies), o, y_floor, host)
        if cutter is not None:
            scene_collection.objects.link(cutter[0])
            out.append((*cutter, host["id"]))
    return out


def _remove(obj: bpy.types.Object) -> None:
    mesh = obj.data
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)


def build(arch: dict, collections: dict, materials: dict) -> list[bpy.types.Object]:
    """One object per shell element, named by element id, in `Coque` (`À vérifier` when flagged to_verify).

    Walls are mitred at their corners, partitions are extended to the walls they come near and cut by them, and every
    opening cuts each wall or partition its box crosses.
    """
    scene_collection = bpy.context.scene.collection
    y_floor = frame.value(arch, "floor.level")
    walls = plan.wall_footprints(arch)
    partitions = plan.partition_plans(arch, walls)
    footprints = {**walls, **{pid: p["face"] for pid, p in partitions.items()}}
    hosts = {e["id"]: e for e in arch["elements"] if e["kind"] in ("wall", "partition")}
    cutters = _cutters(arch, y_floor, hosts, footprints, scene_collection)
    out = []
    for el in arch["elements"]:
        kind = el["kind"]
        target = collections["À vérifier" if "to_verify" in el.get("flags", []) else "Coque"]
        if kind in ("wall", "partition"):
            top = frame.wall_geometry(arch, el)["top"]
            solid = partitions[el["id"]]["solid"] if kind == "partition" else footprints[el["id"]]
            obj, material = _prism(el["id"], solid, y_floor, top), "Mur"
            crossing = [c for c, quad, host in cutters if host == el["id"] or plan.overlap(footprints[el["id"]], quad)]
            trims = [_prism(f"Trim_{el['id']}_{k}", poly, y_floor - 0.5, top + 0.5) for k, poly in enumerate(
                partitions[el["id"]]["cut"] if kind == "partition" else [])]
            for trim in trims:
                scene_collection.objects.link(trim)
            if crossing or trims:
                target.objects.link(obj)
                _apply_cuts(obj, [*trims, *crossing])
            for trim in trims:
                _remove(trim)
        elif kind == "floor":
            obj, material = _sheet(el["id"], [(r["exterior"], r["holes"], r["drop"]) for r in el["extent"]] if "extent" in el else [(el["polygon"], [])], frame.value(arch, el["params"]["level"]), up=True), "Sol"
        elif kind == "ceiling":
            regions = [(r["exterior"], r["holes"], r.get("drop", 0.0)) for r in el.get("extent", el["region"])]
            obj, material = _sheet(el["id"], regions, frame.value(arch, el["params"]["level"]), up=False), "Plafond"
        elif kind == "beam":
            obj, material = _beam(arch, el), "Mur"
        elif kind == "column":
            obj, material = _column(arch, el, y_floor), "Mur"
        else:
            continue
        if obj.name not in target.objects:
            target.objects.link(obj)
        obj.data.materials.clear()  # a boolean result carries the empty slots of its cutters, which would take slot 0
        obj.data.materials.append(materials[material])
        obj.data.polygons.foreach_set("material_index", [0] * len(obj.data.polygons))
        out.append(obj)
    for cutter, *_ in cutters:
        _remove(cutter)
    return out
