"""Runs inside Blender: `blender -b architecture.blend --python blender_probe.py -- spec.json facts.json`.

Dumps scene facts the tests assert on, ray-casts rays against single objects, evaluates the rigs under given
custom properties, then exercises the build_scene hook contract (which clears the scene, hence last).
"""
import json
import math
import sys
import types

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

RIG_PROPS = ("Ouverture", "Inverser sens", "Inverser charni\u00e8re", "AngleMax")


def _object_facts(obj):
    facts = {"type": obj.type, "collections": sorted(c.name for c in obj.users_collection), "modifiers": len(obj.modifiers),
             "hide_render": obj.hide_render, "hide_viewport": obj.hide_viewport, "location": list(obj.location),
             "parent": obj.parent.name if obj.parent else None, "dims": list(obj.dimensions),
             "props": {k: obj[k] for k in RIG_PROPS if k in obj},
             "flags": obj.get("flags"),
             "drivers": len(obj.animation_data.drivers) if obj.animation_data else 0,
             "drivers_simple": all(d.driver.is_simple_expression for d in obj.animation_data.drivers) if obj.animation_data else True}
    if obj.type == "FONT":
        facts["text"] = obj.data.body
        facts["normal"] = list(obj.matrix_world.to_3x3() @ Vector((0, 0, 1)))
    segments = getattr(obj, "MeasureGenerator", None)
    if segments is not None and len(segments):
        facts["measureit"] = [{"a": m.glpointa, "b": m.glpointb, "txt": m.gltxt, "dist": m.gldist, "names": m.glnames,
                               "view": m.glview} for m in segments[0].measureit_segments]
    if obj.type == "MESH":
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        facts.update(verts=len(bm.verts), faces=len(bm.faces), non_manifold=sum(1 for e in bm.edges if not e.is_manifold),
                     materials=[m.name for m in obj.data.materials if m],
                     zs=sorted({round(v.co.z, 4) for v in bm.verts}))
        corners = [obj.matrix_world @ v.co for v in bm.verts]
        bm.free()
        if corners:
            facts["bbox"] = [[min(c[i] for c in corners) for i in range(3)], [max(c[i] for c in corners) for i in range(3)]]
    return facts


def _cast(ray):
    """Nearest hit of the ray against the object (or the list of objects) it names."""
    names = ray["object"] if isinstance(ray["object"], list) else [ray["object"]]
    best = {"hit": False, "location": None, "object": None}
    for name in names:
        obj = bpy.data.objects[name]
        inv = obj.matrix_world.inverted()
        origin = inv @ Vector(ray["origin"])
        direction = (inv.to_3x3() @ Vector(ray["direction"])).normalized()
        hit, location, _, _ = obj.ray_cast(origin, direction, distance=ray.get("distance", 1e4))
        if hit:
            world = obj.matrix_world @ location
            if not best["hit"] or (world - Vector(ray["origin"])).length < (Vector(best["location"]) - Vector(ray["origin"])).length:
                best = {"hit": True, "location": list(world), "object": name}
    return best


def _descendants(obj):
    for child in obj.children:
        yield child
        yield from _descendants(child)


def _rig(spec):
    """Pose of every descendant of the empty (relative to it) after setting its custom properties."""
    empty = bpy.data.objects[spec["empty"]]
    saved = {k: empty[k] for k in spec["props"]}
    for key, value in spec["props"].items():
        empty[key] = value
    empty.update_tag()
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    frame_inv = empty.evaluated_get(deps).matrix_world.inverted()
    out = {}
    for obj in _descendants(empty):
        ev = obj.evaluated_get(deps)
        rel = frame_inv @ ev.matrix_world
        entry = {"location": list(rel.translation), "rotation_z_deg": math.degrees(rel.to_euler().z),
                 "driven_rotation_z_deg": math.degrees(ev.rotation_euler.z),
                 "world_origin": list(ev.matrix_world.translation)}
        if obj.type == "MESH":
            entry["world_centre"] = list(ev.matrix_world @ (sum((Vector(c) for c in ev.bound_box), Vector()) / 8))
            xs = [(rel @ Vector(c)).x for c in ev.bound_box]
            entry["extent_x"] = [min(xs), max(xs)]
        out[obj.name] = entry
    for key, value in saved.items():
        empty[key] = value
    empty.update_tag()
    return out


def _inside(bvh, point):
    """Whether the point lies inside a closed mesh: odd number of surface crossings along a skew ray."""
    direction, origin, hits = Vector((0.37, 0.53, 0.76)).normalized(), point.copy(), 0
    while True:
        location = bvh.ray_cast(origin, direction)[0]
        if location is None:
            return hits % 2 == 1
        hits, origin = hits + 1, location + direction * 1e-5


def _overlaps(step=0.01, levels=(0.1, 0.5, 0.9, 1.3, 1.7, 2.1)):
    """{"A|B": area in m2} of the plan overlap of each pair of walls and partitions, sampled on a `step` grid at several heights."""
    deps = bpy.context.evaluated_depsgraph_get()
    objects = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith(("Wall_", "Partition_"))]
    bvh = {o.name: BVHTree.FromObject(o, deps) for o in objects}
    boxes = {o.name: [[f(v.co[i] for v in o.data.vertices) for i in range(2)] for f in (min, max)] for o in objects}
    out = {}
    for k, a in enumerate(objects):
        for b in objects[k + 1:]:
            lo = [max(boxes[a.name][0][i], boxes[b.name][0][i]) for i in range(2)]
            hi = [min(boxes[a.name][1][i], boxes[b.name][1][i]) for i in range(2)]
            xs = [lo[0] + 0.0037 + step * i for i in range(max(0, math.ceil((hi[0] - lo[0]) / step)))]
            ys = [lo[1] + 0.0041 + step * i for i in range(max(0, math.ceil((hi[1] - lo[1]) / step)))]
            count = sum(1 for x in xs for y in ys for z in levels
                        if x < hi[0] and y < hi[1] and _inside(bvh[a.name], Vector((x, y, z))) and _inside(bvh[b.name], Vector((x, y, z))))
            if count:
                out[f"{a.name}|{b.name}"] = count * step * step / len(levels)
    return out


def _hook_contract(spec):
    """build_scene.build passes ctx to a hook and rejects returned objects linked to no collection."""
    sys.path.insert(0, spec["blender_dir"])
    import build_scene
    with open(spec["arch"], encoding="utf-8") as fh:
        arch = json.load(fh)
    seen = {}

    def run(link):
        def add(arch_, collections, ctx):
            seen.update(ctx=ctx, collections=sorted(collections))
            obj = bpy.data.objects.new("ProbeLoose", None)
            if link:
                collections["Menuiseries"].objects.link(obj)
            return [obj]
        module = types.ModuleType("arch.zz_probe")
        module.add = add
        sys.modules["arch.zz_probe"] = module
        build_scene.HOOKS = (*build_scene.HOOKS, "zz_probe")
        try:
            build_scene.build(arch, None, None, "/probe/out")
            return {"raised": None}
        except RuntimeError as exc:
            return {"raised": str(exc)}
        finally:
            build_scene.HOOKS = build_scene.HOOKS[:-1]
    return {"linked": run(True), "unlinked": run(False), "seen": seen}


def _measureit_failure(spec):
    """A MeasureIt API failure during the build leaves a native scene and records the error."""
    sys.path.insert(0, spec["blender_dir"])
    import build_scene
    from arch import dimensions
    with open(spec["arch"], encoding="utf-8") as fh:
        arch = json.load(fh)

    def broken(*_):
        raise AttributeError("measureit API changed")
    saved, dimensions._measureit = dimensions._measureit, broken
    try:
        build_scene.build(arch, None, None, "/probe/out")
    finally:
        dimensions._measureit = saved
    scene = bpy.context.scene
    return {"backend": scene.get("dimensions_backend"), "error": scene.get("measureit_error"),
            "dims": sorted(o.name for o in bpy.data.objects if o.name.startswith("Dim_") and o.parent is None)}


def main():
    spec_path, out_path = sys.argv[sys.argv.index("--") + 1:][:2]
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    bpy.context.view_layer.update()
    facts = {
        "collections": {c.name: {"objects": sorted(o.name for o in c.all_objects), "hide_viewport": c.hide_viewport,
                                 "hide_render": c.hide_render} for c in bpy.data.collections},
        "objects": {o.name: _object_facts(o) for o in bpy.data.objects},
        "scene": {"dimensions_backend": bpy.context.scene.get("dimensions_backend"),
                  "show_names": getattr(bpy.context.scene, "measureit_gl_show_n", None),
                  "show_all": getattr(bpy.context.scene, "measureit_gl_ghost", None),
                  "measureit_available": hasattr(bpy.types.Scene, "measureit_gl_ghost")},
        "materials": sorted(m.name for m in bpy.data.materials),
        "scene_objects": sorted(o.name for o in bpy.context.scene.collection.objects),
        "rays": {r["name"]: _cast(r) for r in spec["rays"]},
        "rigs": {r["name"]: _rig(r) for r in spec["rigs"]},
        "overlaps": _overlaps(),
    }
    facts["measureit_failure"] = _measureit_failure(spec)
    facts["hook"] = _hook_contract(spec)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(facts, fh)


main()
