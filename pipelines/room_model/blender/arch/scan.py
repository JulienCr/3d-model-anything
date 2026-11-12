"""Imports the decimated LiDAR mesh written by room_model.blender_stage (binary PLY, room frame)."""
import bpy
import numpy as np


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) and triangles (m, 3) of the fixed layout blender_stage.write_ply produces."""
    with open(path, "rb") as fh:
        data = fh.read()
    head, _, body = data.partition(b"end_header\n")
    lines = head.decode("ascii").split("\n")
    assert lines[1] == "format binary_little_endian 1.0", lines[1]
    n_vert = int(next(l for l in lines if l.startswith("element vertex")).split()[2])
    n_face = int(next(l for l in lines if l.startswith("element face")).split()[2])
    vertices = np.frombuffer(body, "<f4", count=3 * n_vert).reshape(-1, 3)
    faces = np.frombuffer(body, np.dtype([("n", "u1"), ("i", "<i4", 3)]), count=n_face, offset=12 * n_vert)["i"]
    return vertices.astype(np.float64), faces.astype(np.int64)


def add(path: str, collection: bpy.types.Collection) -> bpy.types.Object:
    vertices, faces = read_ply(path)
    mesh = bpy.data.meshes.new("Scan")
    mesh.from_pydata([(x, -z, y) for x, y, z in vertices.tolist()], [], faces.tolist())
    mesh.validate()
    obj = bpy.data.objects.new("Scan", mesh)
    collection.objects.link(obj)
    return obj
