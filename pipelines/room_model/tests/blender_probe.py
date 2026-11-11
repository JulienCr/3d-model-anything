"""Runs inside Blender: `blender -b architecture.blend --python blender_probe.py -- rays.json facts.json`.

Dumps scene facts the tests assert on, and ray-casts the requested rays against single objects.
"""
import json
import sys

import bmesh
import bpy
from mathutils import Vector


def _object_facts(obj):
    facts = {"type": obj.type, "collections": sorted(c.name for c in obj.users_collection), "modifiers": len(obj.modifiers),
             "hide_render": obj.hide_render, "hide_viewport": obj.hide_viewport, "location": list(obj.location)}
    if obj.type == "MESH":
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        facts.update(verts=len(bm.verts), faces=len(bm.faces), non_manifold=sum(1 for e in bm.edges if not e.is_manifold),
                     materials=[m.name for m in obj.data.materials if m])
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


def main():
    rays_path, out_path = sys.argv[sys.argv.index("--") + 1:][:2]
    with open(rays_path, encoding="utf-8") as fh:
        rays = json.load(fh)
    bpy.context.view_layer.update()
    facts = {
        "collections": {c.name: {"objects": sorted(o.name for o in c.all_objects), "hide_viewport": c.hide_viewport,
                                 "hide_render": c.hide_render} for c in bpy.data.collections},
        "objects": {o.name: _object_facts(o) for o in bpy.data.objects},
        "materials": sorted(m.name for m in bpy.data.materials),
        "scene_objects": sorted(o.name for o in bpy.context.scene.collection.objects),
        "rays": {r["name"]: _cast(r) for r in rays},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(facts, fh)


main()
