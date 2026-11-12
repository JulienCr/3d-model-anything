"""Projected photo textures: one image material per textured surface, planar UVs from the surface's own axes.

ctx["textures"] is textures.json: {element id: {face: {"image", "origin", "u_axis", "v_axis", "normal", "texel", "size"}}}
in the room frame; "image" is relative to ctx["out_dir"]. A surface cut by an opening keeps the UVs of the uncut plane,
because every face of the object lying on that plane is mapped from the same origin and axes.
"""
import os

import bpy
from mathutils import Vector

from . import frame

PLANE_TOL = 0.002  # m: a face this close to the textured plane lies on it (booleans leave the cut faces on the plane to float error)
FACING = 0.99  # cosine between a face normal and the surface normal: the face looks the same way


def _vector(room_xyz) -> Vector:
    return Vector(frame.to_blender(*room_xyz))


def _material(name: str, image: bpy.types.Image) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "EXTEND"  # a sheet runs past the surface under the walls: clamp to the edge texel, do not tile
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    nodes.active = tex  # the Workbench "Texture" colour reads the active image node
    return mat


def _apply(obj: bpy.types.Object, face: str, entry: dict, out_dir: str) -> int:
    """Gives every face of `obj` on the surface plane the texture material and its UVs; returns how many faces."""
    image = bpy.data.images.load(os.path.join(out_dir, entry["image"]), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image.pack()
    mesh = obj.data
    mesh.materials.append(_material(f"Tex_{obj.name}_{face}", image))
    slot = len(mesh.materials) - 1
    uv_layer = mesh.uv_layers.active or mesh.uv_layers.new(name="UVMap")
    origin, u_axis, v_axis, normal = (_vector(entry[k]) for k in ("origin", "u_axis", "v_axis", "normal"))
    extent_u, extent_v = entry["size"][0] * entry["texel"], entry["size"][1] * entry["texel"]
    count = 0
    for poly in mesh.polygons:
        if poly.normal.dot(normal) < FACING or abs((poly.center - origin).dot(normal)) > PLANE_TOL:
            continue
        poly.material_index = slot
        for loop in poly.loop_indices:
            offset = mesh.vertices[mesh.loops[loop].vertex_index].co - origin
            uv_layer.data[loop].uv = (offset.dot(u_axis) / extent_u, offset.dot(v_axis) / extent_v)
        count += 1
    return count


def add(arch: dict, collections: dict, ctx: dict) -> list:
    """Textures the shell objects named in ctx["textures"]; creates no object."""
    textures = ctx.get("textures")
    if not textures:
        return []
    for element_id, faces in textures.items():
        obj = bpy.data.objects.get(element_id)
        if obj is None or obj.type != "MESH":
            continue
        for face, entry in faces.items():
            _apply(obj, face, entry, ctx["out_dir"])
    return []
