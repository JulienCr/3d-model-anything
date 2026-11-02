#!/usr/bin/env python3
"""Shared helpers for a room's make-assets script: scan GLB for Scan/Overlay, photo-toned CC0 maps.

usage: scan_assets.py glb <scan_room.ply> <out.glb> [--faces 400000] [--offset x,y,z]
Needs NumPy, Open3D and Pillow (a project environment, not system Python).
"""
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageEnhance


def scan_glb(src, out, faces=400000, offset=(0.0, 0.0, 0.0)):
    """Decimate a photo-coloured room-space PLY to a GLB; offset shifts it into the modelled frame."""
    mesh = o3d.io.read_triangle_mesh(src)
    mesh = mesh.simplify_quadric_decimation(faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if any(offset):
        mesh.translate(offset)
    mesh.compute_vertex_normals()
    c = np.asarray(mesh.vertex_colors)  # glTF vertex colors are linear, the photos are sRGB
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))
    o3d.io.write_triangle_mesh(out, mesh)


def tone(src, out, mean, saturation, contrast=1.0):
    """Desaturate a CC0 colour map and shift it to a mean sRGB tone taken from the photos."""
    arr = np.asarray(ImageEnhance.Color(Image.open(src).convert('RGB')).enhance(saturation), np.float32)
    arr = arr / arr.mean((0, 1))
    arr = (1 + (arr - 1) * contrast) * np.array(mean, np.float32)
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(out, quality=90)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('glb')
    g.add_argument('src'); g.add_argument('out')
    g.add_argument('--faces', type=int, default=400000)
    g.add_argument('--offset', type=lambda v: tuple(map(float, v.split(','))), default=(0.0, 0.0, 0.0))
    a = ap.parse_args()
    scan_glb(a.src, a.out, a.faces, a.offset)
