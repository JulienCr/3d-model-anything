"""Dimensions of the architecture: MeasureIt segments when the extension is enabled, native geometry otherwise."""
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

TEXT_SIZE = 0.06  # m
ARROW_LENGTH, ARROW_RADIUS, LINE_RADIUS = 0.05, 0.015, 0.002  # m
TEXT_LIFT = 0.04  # m between the line and its text
OPT_OUT_ENV = "ROOM_MODEL_NO_MEASUREIT"
ARROW_TRIANGLE = "2"  # MeasureIt enum value of the triangle arrow head


def format_text(dim: dict) -> str:
    """`286,0 cm ± 1,8`: value and sigma in cm, one decimal, French comma."""
    return f"{dim['value'] * 100:.1f} cm ± {dim['sigma'] * 100:.1f}".replace(".", ",")


def _point(p) -> Vector:
    """Room frame (x, y up, z) to Blender (x, -z, y)."""
    return Vector((p[0], -p[2], p[1]))


def _spanned(arch: dict) -> list[dict]:
    return [d for d in arch["dimensions"] if d.get("endpoints")]


def measureit_available() -> bool:
    return not os.environ.get(OPT_OUT_ENV) and hasattr(bpy.types.Scene, "measureit_gl_ghost")


def _measureit(arch: dict, cotes: bpy.types.Collection) -> list:
    """One two-vertex helper mesh per dimension carrying a MeasureIt segment whose label is the French text."""
    scene = bpy.context.scene
    scene.measureit_gl_show_n, scene.measureit_gl_ghost = True, True
    objects = []
    for dim in _spanned(arch):
        mesh = bpy.data.meshes.new(f"Dim_{dim['id']}")
        mesh.from_pydata([_point(p) for p in dim["endpoints"]], [(0, 1)], [])
        obj = bpy.data.objects.new(f"Dim_{dim['id']}", mesh)
        cotes.objects.link(obj)
        objects.append(obj)
        obj.MeasureGenerator.add()
        container = obj.MeasureGenerator[0]
        segment = container.measureit_segments.add()
        segment.gltype, segment.glpointa, segment.glpointb = 1, 0, 1
        segment.glarrow_a = segment.glarrow_b = ARROW_TRIANGLE
        segment.glarrow_s = scene.measureit_glarrow_s
        segment.glcolor, segment.glspace = scene.measureit_default_color, scene.measureit_hint_space
        segment.glfont_size = scene.measureit_font_size
        segment.gldist, segment.glnames, segment.gltxt = False, True, format_text(dim)
        container.measureit_num += 1
    return objects


def _room_centre(arch: dict) -> Vector:
    pts = [_point((w[k][0], 0.0, w[k][1])) for w in arch["elements"] if w["kind"] == "wall" for k in ("start", "end")]
    return sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts) if pts else Vector((0.0, 0.0, 0.0))


def _facing(normal: Vector, up: Vector) -> tuple:
    """Euler rotation of a text object whose face looks along `normal` with its top towards `up`."""
    z = normal.normalized()
    x = up.cross(z).normalized()
    return Matrix((x, z.cross(x), z)).transposed().to_euler()


def _material(name: str, color) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def _arrow(name: str, tip: Vector, axis: Vector, mat) -> bpy.types.Object:
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=ARROW_RADIUS, radius2=0.0, depth=ARROW_LENGTH)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = tip - axis * ARROW_LENGTH / 2
    obj.rotation_euler = axis.to_track_quat("Z", "Y").to_euler()
    return obj


def _native_one(dim: dict, centre: Vector, cotes: bpy.types.Collection, mat) -> list:
    a, b = (_point(p) for p in dim["endpoints"])
    name = f"Dim_{dim['id']}"
    empty = bpy.data.objects.new(name, None)
    curve = bpy.data.curves.new(f"{name}_Line", "CURVE")
    curve.dimensions, curve.bevel_depth = "3D", LINE_RADIUS
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    for point, p in zip(spline.points, (a, b)):
        point.co = (*p, 1.0)
    curve.materials.append(mat)
    line = bpy.data.objects.new(f"{name}_Line", curve)
    direction = (b - a).normalized()
    arrows = [_arrow(f"{name}_ArrowA", a, -direction, mat), _arrow(f"{name}_ArrowB", b, direction, mat)]
    font = bpy.data.curves.new(f"{name}_Text", "FONT")
    font.body, font.size, font.align_x, font.align_y = format_text(dim), TEXT_SIZE, "CENTER", "BOTTOM"
    font.materials.append(mat)
    text = bpy.data.objects.new(f"{name}_Text", font)
    mid = (a + b) / 2
    if abs(direction.z) < 0.5:  # horizontal span: flat text, readable from above
        along = Vector((direction.x, direction.y, 0.0)).normalized()
        if along.x < -1e-6 or (abs(along.x) <= 1e-6 and along.y < 0):
            along = -along
        top = Vector((-along.y, along.x, 0.0))
        text.rotation_euler = _facing(Vector((0, 0, 1)), top)
        text.location = mid + top * TEXT_LIFT + Vector((0, 0, 0.005))
    else:  # vertical span: upright text facing the room
        towards = Vector((centre.x - mid.x, centre.y - mid.y, 0.0))
        towards = towards.normalized() if towards.length > 1e-6 else Vector((1.0, 0.0, 0.0))
        text.rotation_euler = _facing(towards, Vector((0, 0, 1)))
        text.location = mid + towards * (LINE_RADIUS + 0.01)
    objects = [empty, line, *arrows, text]
    for obj in objects:
        cotes.objects.link(obj)
        if obj is not empty:
            obj.parent = empty
    return objects


def _native(arch: dict, cotes: bpy.types.Collection) -> list:
    mat = _material("Cotes", (0.9, 0.1, 0.1, 1.0))
    centre = _room_centre(arch)
    return [o for dim in _spanned(arch) for o in _native_one(dim, centre, cotes, mat)]


def _discard(objects: list) -> None:
    for obj in objects:
        data = obj.data
        bpy.data.objects.remove(obj)
        if data is not None and data.users == 0:
            (bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.curves).remove(data)


def add(arch, collections, ctx):
    """Creates one `Dim_<id>` per dimension that has endpoints; records the backend on the scene."""
    scene, cotes = bpy.context.scene, collections["Cotes"]
    objects, backend, error = [], "native", None
    if measureit_available():
        try:
            objects = _measureit(arch, cotes)
            backend = "measureit"
        except Exception as exc:  # any MeasureIt API mismatch must leave the build intact
            error = f"{type(exc).__name__}: {exc}"
            _discard([o for o in cotes.objects if o.name.startswith("Dim_")])
            objects = []
    if backend == "native":
        objects = _native(arch, cotes)
    scene["dimensions_backend"], scene["dimensions_count"] = backend, len(_spanned(arch))
    if error:
        scene["measureit_error"] = error
    return objects
