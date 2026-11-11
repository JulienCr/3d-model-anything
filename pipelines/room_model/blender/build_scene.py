"""Builds the architecture scene from architecture.json, headless.

blender --background --python build_scene.py -- --arch A --out-dir D [--textures T] [--scan S]
Writes D/architecture.blend, D/architecture.glb and D/renders/{top,persp_1,persp_2}.png.
"""
import argparse
import importlib
import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch import collections as collection_names  # noqa: E402
from arch import frame, materials, render, scan, shell  # noqa: E402

HOOKS = ("joinery", "niches", "mouldings", "dimensions", "textures")  # later tasks: arch/<name>.py with add(arch, collections)


def _hook(name: str):
    """The optional module arch.<name>, or None when it does not exist (a missing import inside it still raises)."""
    try:
        return importlib.import_module(f"arch.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"arch.{name}":
            return None
        raise


def _clear() -> None:
    for datablocks in (bpy.data.objects, bpy.data.collections, bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
        for item in list(datablocks):
            datablocks.remove(item)


def build(arch: dict, textures: dict | None, scan_path: str | None) -> None:
    """Builds the whole scene in the current file; `textures` is reserved for arch/textures.py."""
    _clear()
    scene = bpy.context.scene
    scene.unit_settings.system, scene.unit_settings.scale_length = "METRIC", 1.0
    colls = collection_names.create(scene)
    mats = materials.ensure_materials()
    shell.build(arch, colls, mats)
    for name in HOOKS:
        module = _hook(name)
        if module is not None:
            module.add(arch, colls)
    if scan_path:
        scan.add(scan_path, colls["Scan"])
    bpy.context.view_layer.update()


def export_glb(path: str, colls: dict) -> None:
    """Static GLB of the exported collections (no dimensions, no scan)."""
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    for name in collection_names.EXPORTED:
        for obj in colls[name].all_objects:
            obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_extras=False,
                              export_cameras=False, export_lights=False, export_yup=True)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--textures")
    parser.add_argument("--scan")
    args = parser.parse_args(argv)
    with open(args.arch, encoding="utf-8") as fh:
        arch = json.load(fh)
    textures = None
    if args.textures:
        with open(args.textures, encoding="utf-8") as fh:
            textures = json.load(fh)
    build(arch, textures, args.scan)
    colls = {c.name: c for c in bpy.data.collections}
    os.makedirs(args.out_dir, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.out_dir, "architecture.blend"))
    export_glb(os.path.join(args.out_dir, "architecture.glb"), colls)
    render.render_views(arch, args.out_dir)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:])
