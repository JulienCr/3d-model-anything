#!/usr/bin/env python3
"""Build the git-ignored Atrium assets in data/web-assets/atrium (run from the repo root).

Prerequisites: prepare_photos.py (photos/), reconstruct_bundle.py -o data/scans/atrium (scan_room.ply).
Uses only the user's own photos plus the repo's CC0 laminate texture.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, '.agents/skills/scan-to-threejs/scripts')
from scan_assets import scan_glb, tone  # noqa: E402

OUT = 'data/web-assets/atrium'
os.makedirs(f'{OUT}/textures', exist_ok=True)


def ceiling_tile(photo=100, box=(0, 0, 640, 330)):
    """Wood-wool panel (1.2 x 0.62 m): a seam-free crop of an upward photo, mirrored 2x2, with joint lines on its edges."""
    crop = Image.open(f'{OUT}/photos/{photo:04d}.jpg').convert('RGB').crop(box).resize((600, 310), Image.LANCZOS)
    tile = Image.new('RGB', (1200, 620))
    tile.paste(crop, (0, 0))
    tile.paste(crop.transpose(Image.FLIP_LEFT_RIGHT), (600, 0))
    tile.paste(crop.transpose(Image.FLIP_TOP_BOTTOM), (0, 310))
    tile.paste(crop.transpose(Image.ROTATE_180), (600, 310))
    # flatten the lighting gradient of the source photo
    low = np.asarray(tile.filter(ImageFilter.GaussianBlur(90)), np.float32).clip(1)
    arr = np.asarray(tile, np.float32) / low * low.mean((0, 1))
    tile = ImageEnhance.Color(Image.fromarray(arr.clip(0, 255).astype(np.uint8))).enhance(0.3)
    tile = Image.fromarray((np.asarray(tile, np.float32) / np.asarray(tile, np.float32).mean((0, 1)) * np.array([182, 168, 148])).clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(tile)
    d.line([(0, 1), (1200, 1)], fill=(70, 56, 42), width=4)
    d.line([(1, 0), (1, 620)], fill=(70, 56, 42), width=4)
    tile.save(f'{OUT}/textures/ceiling.jpg', quality=90)


def floor_tile():
    """Light greige laminate: CC0 laminate planks at the photographed floor tone (about 140, 131, 120 before lighting)."""
    tone('data/web-assets/textures/laminate_diff.jpg', f'{OUT}/textures/floor.jpg', (138, 132, 124), 0.1, 0.55)


def oak_light():
    """Pale grey-oak table top (photo 124)."""
    tone('data/web-assets/textures/oak_diff.jpg', f'{OUT}/textures/oak-light.jpg', (194, 176, 148), 0.3)


if __name__ == '__main__':
    scan_glb('data/scans/atrium/scan_room.ply', f'{OUT}/scan.glb')
    ceiling_tile()
    floor_tile()
    oak_light()
