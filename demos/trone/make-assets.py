#!/usr/bin/env python3
"""Build the git-ignored Trone assets in data/web-assets/trone (run from the repo root).

Prerequisites: prepare_photos.py (photos/), reconstruct_bundle.py --mesh ... -o data/scans/trone (scan_room.ply).
Uses only the user's own photos.
"""
import json
import os
import sys
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, '.agents/skills/scan-to-threejs/scripts')
from scan_assets import scan_glb  # noqa: E402

OUT = 'data/web-assets/trone'
FLOOR_Y = json.load(open('demos/trone/geometry.json'))['floorY']  # shared with shell.ts
os.makedirs(f'{OUT}/textures', exist_ok=True)


def photo(i, box):
    return Image.open(f'{OUT}/photos/{i:04d}.jpg').convert('RGB').crop(box)


def flatten(img, radius, mean):
    """Divide out the photo's lighting gradient and set a neutral mean sRGB tone."""
    low = np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), np.float32).clip(1)
    arr = np.asarray(img, np.float32) / low
    arr = arr / arr.mean((0, 1)) * np.array(mean, np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def seamless(img):
    """Cross-fade the crop with its half-period shift so the tile wraps without mirror symmetry."""
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    b = np.roll(a, (h // 2, w // 2), (0, 1))
    wy = np.sin(np.linspace(0, np.pi, h)) ** 0.7
    wx = np.sin(np.linspace(0, np.pi, w)) ** 0.7
    m = (wy[:, None] * wx[None, :])[..., None]
    return Image.fromarray((a * m + b * (1 - m)).clip(0, 255).astype(np.uint8))


def ceiling():
    """Sprayed fibre insulation, photo 39 looking up; one tile is about 1.2 x 0.7 m."""
    img = ImageEnhance.Color(photo(39, (0, 0, 960, 540))).enhance(0.15)
    seamless(flatten(img, 60, (200, 200, 200))).save(f'{OUT}/textures/ceiling.jpg', quality=88)


def floor():
    """Dusty concrete slab with grit, photo 253 looking down; neutral grey, tinted in the material."""
    img = ImageEnhance.Color(photo(253, (0, 640, 960, 1280))).enhance(0.1)
    img = ImageEnhance.Contrast(flatten(img, 40, (200, 200, 200))).enhance(0.6)
    seamless(img).save(f'{OUT}/textures/floor.jpg', quality=88)


def wall():
    """Painted concrete with the flaking, damp lower band (photo 44); floor line at the bottom edge."""
    img = ImageEnhance.Color(photo(44, (210, 150, 960, 905))).enhance(0.0)
    img = ImageEnhance.Contrast(flatten(img, 220, (205, 205, 205))).enhance(1.45)
    w, h = img.size
    tile = Image.new('RGB', (w * 2, h))
    tile.paste(img, (0, 0))
    tile.paste(img.transpose(Image.FLIP_LEFT_RIGHT), (w, 0))
    tile.save(f'{OUT}/textures/wall.jpg', quality=88)


def pilaster():
    """Patchy plaster on a pilaster face (photo 220): beige render over grey concrete, floor at the bottom."""
    img = ImageEnhance.Color(flatten(photo(220, (700, 0, 860, 1120)), 120, (190, 185, 175))).enhance(0.6)
    img.resize((128, 640), Image.LANCZOS).save(f'{OUT}/textures/pilaster.jpg', quality=88)


if __name__ == '__main__':
    scan_glb('data/scans/trone/scan_room.ply', f'{OUT}/scan.glb', offset=(0.0, -FLOOR_Y, 0.0))
    ceiling()
    floor()
    wall()
    pilaster()
