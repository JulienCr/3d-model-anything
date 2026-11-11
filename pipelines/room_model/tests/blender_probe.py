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

RIG_PROPS = ("Ouverture", "Sens", "AngleMax")


def _object_facts(obj):
    facts = {"type": obj.type, "collections": sorted(c.name for c in obj.users_collection), "modifiers": len(obj.modifiers),
             "hide_render": obj.hide_render, "hide_viewport": obj.hide_viewport, "location": list(obj.location),
             "parent": obj.parent.name if obj.parent else None, "dims": list(obj.dimensions),
             "props": {k: obj[k] for k in RIG_PROPS if k in obj},
             "drivers": len(obj.animation_data.drivers) if obj.animation_data else 0}
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
    obj = bpy.data.objects[ray["object"]]
    inv = obj.matrix_world.inverted()
    origin = inv @ Vector(ray["origin"])
    direction = (inv.to_3x3() @ Vector(ray["direction"])).normalized()
    hit, location, _, _ = obj.ray_cast(origin, direction, distance=ray.get("distance", 1e4))
    return {"hit": bool(hit), "location": list(obj.matrix_world @ location) if hit else None}


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
                 "world_origin": list(ev.matrix_world.translation)}
        if obj.type == "MESH":
            entry["world_centre"] = list(ev.matrix_world @ (sum((Vector(c) for c in ev.bound_box), Vector()) / 8))
        out[obj.name] = entry
    for key, value in saved.items():
        empty[key] = value
    empty.update_tag()
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


def main():
    spec_path, out_path = sys.argv[sys.argv.index("--") + 1:][:2]
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    bpy.context.view_layer.update()
    facts = {
        "collections": {c.name: {"objects": sorted(o.name for o in c.all_objects), "hide_viewport": c.hide_viewport,
                                 "hide_render": c.hide_render} for c in bpy.data.collections},
        "objects": {o.name: _object_facts(o) for o in bpy.data.objects},
        "materials": sorted(m.name for m in bpy.data.materials),
        "scene_objects": sorted(o.name for o in bpy.context.scene.collection.objects),
        "rays": {r["name"]: _cast(r) for r in spec["rays"]},
        "rigs": {r["name"]: _rig(r) for r in spec["rigs"]},
    }
    facts["hook"] = _hook_contract(spec)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(facts, fh)


main()
