#!/usr/bin/env python3
"""Build the git-ignored Couloir assets in data/web-assets/couloir (run from the repo root).

Prerequisites, from the raw export in data/captures-in/couloir:
  bundle_to_room.py data/captures-in/couloir --room 0 -o data/demos/couloir/room.json
  prepare_photos.py data/captures-in/couloir data/demos/couloir/room.json data/web-assets/couloir/photos
  reconstruct_bundle.py data/captures-in/couloir --mesh data/captures-in/couloir/mesh/room-0.ply \
      --room data/demos/couloir/room.json -o data/scans/couloir/recon
Uses only the user's own photos and LiDAR mesh.
"""
import os
import numpy as np
import open3d as o3d
from PIL import Image, ImageEnhance, ImageFilter

OUT = 'data/web-assets/couloir'
os.makedirs(f'{OUT}/textures', exist_ok=True)


def scan_glb(faces=300000):
    mesh = o3d.io.read_triangle_mesh('data/scans/couloir/recon/scan_room.ply')
    mesh = mesh.simplify_quadric_decimation(faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    c = np.asarray(mesh.vertex_colors)  # glTF vertex colors are linear, the photos are sRGB
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))
    o3d.io.write_triangle_mesh(f'{OUT}/scan.glb', mesh)


def flat_tile(photo, box, out, size, mean=None, saturation=1.0, mirror=True):
    """Seamless tile from a photo crop: mirrored 2x2, lighting gradient divided out, optional tone shift."""
    crop = Image.open(f'{OUT}/photos/{photo:04d}.jpg').convert('RGB').crop(box).resize(size, Image.LANCZOS)
    if mirror:
        w, h = size
        tile = Image.new('RGB', (2 * w, 2 * h))
        tile.paste(crop, (0, 0))
        tile.paste(crop.transpose(Image.FLIP_LEFT_RIGHT), (w, 0))
        tile.paste(crop.transpose(Image.FLIP_TOP_BOTTOM), (0, h))
        tile.paste(crop.transpose(Image.ROTATE_180), (w, h))
    else:
        tile = crop
    low = np.asarray(tile.filter(ImageFilter.GaussianBlur(max(tile.size) / 6)), np.float32).clip(1)
    arr = np.asarray(tile, np.float32) / low * low.mean((0, 1))
    tile = ImageEnhance.Color(Image.fromarray(arr.clip(0, 255).astype(np.uint8))).enhance(saturation)
    if mean is not None:
        a = np.asarray(tile, np.float32)
        tile = Image.fromarray((a / a.mean((0, 1)) * np.array(mean, np.float32)).clip(0, 255).astype(np.uint8))
    tile.save(out, quality=90)


def textures():
    """Crops picked on gridded contact sheets; one tile = the repeat length set in materials.ts."""
    T = f'{OUT}/textures'
    flat_tile(23, (80, 0, 560, 250), f'{T}/ceiling.jpg', (512, 512), mean=(172, 172, 168), saturation=0.3)
    flat_tile(64, (250, 950, 750, 1280), f'{T}/floor.jpg', (512, 512), mean=(178, 175, 168), saturation=0.4)
    flat_tile(9, (420, 20, 880, 360), f'{T}/woodwool.jpg', (512, 512), mean=(172, 136, 88), saturation=0.9)
    flat_tile(148, (420, 640, 800, 1180), f'{T}/blocks.jpg', (512, 512), mean=(150, 148, 142), saturation=0.4, mirror=False)
    flat_tile(124, (565, 350, 680, 950), f'{T}/raw-concrete.jpg', (256, 512), mean=(168, 162, 150), saturation=0.6)


if __name__ == '__main__':
    scan_glb()
    textures()
