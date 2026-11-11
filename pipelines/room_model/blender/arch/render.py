"""Workbench renders: a plan view and two perspective views across the longest diagonal, ceilings hidden."""
import math
import os

import bpy
from mathutils import Vector

from . import frame

SIZE = (1280, 720)
EYE_HEIGHT = 1.6  # m above the floor
VIEW_MARGIN = 0.6  # m a perspective camera keeps from the walls
FOCAL_MM = 16.0
BACKGROUND = (0.62, 0.72, 0.82)
OPEN_RATIO = 0.6  # Ouverture of every door and window in renders/open.png


def _inside(poly: list, p: tuple, margin: float) -> bool:
    """Even-odd point-in-polygon test, with a clearance of `margin` from every edge."""
    x, z = p
    inside = False
    for (x0, z0), (x1, z1) in zip(poly, poly[1:] + poly[:1]):
        dx, dz = x1 - x0, z1 - z0
        t = max(0.0, min(1.0, ((x - x0) * dx + (z - z0) * dz) / (dx * dx + dz * dz or 1.0)))
        if math.hypot(x - (x0 + t * dx), z - (z0 + t * dz)) < margin:
            return False
        if (z0 > z) != (z1 > z) and x < x0 + (z - z0) / (z1 - z0) * dx:
            inside = not inside
    return inside


def viewpoints(polygon: list) -> list[tuple[tuple, tuple]]:
    """(eye, target) plan points looking across the longest diagonal, once each way; eyes are inside the polygon."""
    pts = [tuple(p) for p in polygon]
    a, b = max(((p, q) for p in pts for q in pts), key=lambda pq: math.dist(*pq))
    out = []
    for src, dst in ((a, b), (b, a)):
        eye = src
        for k in range(1, 100):
            eye = (src[0] + (dst[0] - src[0]) * k / 100, src[1] + (dst[1] - src[1]) * k / 100)
            if _inside(pts, eye, VIEW_MARGIN):
                break
        out.append((eye, dst))
    return out


def _camera(name: str, location: Vector, look_at: Vector, ortho_scale: float | None = None) -> bpy.types.Object:
    cam = bpy.data.cameras.new(name)
    cam.clip_start, cam.clip_end = 0.05, 200.0
    if ortho_scale is None:
        cam.lens = FOCAL_MM
    else:
        cam.type, cam.ortho_scale = "ORTHO", ortho_scale
    obj = bpy.data.objects.new(name, cam)
    obj.location = location
    if ortho_scale is None:  # the plan camera looks straight down, which has no defined "up" to track
        obj.rotation_euler = (look_at - location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _setup(scene: bpy.types.Scene) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    shading = scene.display.shading
    shading.light, shading.color_type, shading.show_object_outline = "STUDIO", "MATERIAL", True
    scene.display.render_aa = "8"
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = BACKGROUND


def render_views(arch: dict, out_dir: str) -> None:
    """Writes renders/{top,persp_1,persp_2,open}.png (open: plan view, doors and windows at OPEN_RATIO); ceilings hidden meanwhile."""
    scene = bpy.context.scene
    _setup(scene)
    floor = frame.element(arch, "Floor")
    polygon = floor["polygon"]
    y_floor = frame.value(arch, "floor.level")
    ceilings = [bpy.data.objects[e["id"]] for e in arch["elements"] if e["kind"] == "ceiling"]
    saved = [(c, c.hide_render) for c in ceilings]
    for c in ceilings:
        c.hide_render = True
    xs, zs = [p[0] for p in polygon], [p[1] for p in polygon]
    centre = Vector(frame.to_blender((min(xs) + max(xs)) / 2, y_floor, (min(zs) + max(zs)) / 2))
    span = max(max(xs) - min(xs), (max(zs) - min(zs)) * SIZE[0] / SIZE[1]) * 1.1
    views = [("top", _camera("RenderTop", centre + Vector((0, 0, 20.0)), centre, ortho_scale=span))]
    for k, (eye, target) in enumerate(viewpoints(polygon), start=1):
        views.append((f"persp_{k}", _camera(f"RenderPersp{k}", Vector(frame.to_blender(eye[0], y_floor + EYE_HEIGHT, eye[1])),
                                            Vector(frame.to_blender(target[0], y_floor + EYE_HEIGHT - 0.4, target[1])))))
    os.makedirs(os.path.join(out_dir, "renders"), exist_ok=True)
    for name, cam in views:
        scene.camera = cam
        scene.render.filepath = os.path.join(out_dir, "renders", f"{name}.png")
        bpy.ops.render.render(write_still=True)
    rigged = [o for o in bpy.data.objects if o.type == "EMPTY" and "Ouverture" in o]
    for o in rigged:
        o["Ouverture"] = OPEN_RATIO
        o.update_tag()
    scene.camera = views[0][1]
    scene.render.filepath = os.path.join(out_dir, "renders", "open.png")
    bpy.ops.render.render(write_still=True)
    for o in rigged:
        o["Ouverture"] = 0.0
        o.update_tag()
    for c, hidden in saved:
        c.hide_render = hidden
    for _, cam in views:
        data = cam.data
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(data)
