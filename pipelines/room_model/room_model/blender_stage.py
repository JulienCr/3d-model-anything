"""Blender stage: decimates the LiDAR scan, then builds the architecture scene headless (blender/build_scene.py)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from room_model.capture import load_mesh_ply, mesh_path
from room_model.context import BlenderMissing, StageContext, output_path, require
from room_model.frame import world_to_room

DEFAULT_BLENDER = "/mnt/c/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe"
BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "blender" / "build_scene.py"
SCAN_FILE = "scan_decimated.ply"
OUTPUTS = ("architecture.blend", "architecture.glb", "renders/top.png", "renders/persp_1.png", "renders/persp_2.png",
           "renders/open.png")
BACKUPS = ("architecture.blend1",)  # Blender's own copy of the file it overwrites


def blender_executable() -> str:
    return os.environ.get("BLENDER", DEFAULT_BLENDER)


def _native(arg) -> str:
    """Paths go to a Windows executable as Windows paths; other arguments pass through."""
    if not isinstance(arg, Path):
        return str(arg)
    if blender_executable().lower().endswith(".exe"):
        return subprocess.run(["wslpath", "-w", str(arg)], capture_output=True, text=True, check=True).stdout.strip()
    return str(arg)


def _resolve() -> str:
    exe = shutil.which(blender_executable())
    if exe is None:
        raise BlenderMissing(f"Blender not found: {blender_executable()} (set BLENDER)")
    return exe


def run_blender(args: list, log: Path) -> None:
    """Runs Blender with `args` (Path items are converted for a Windows exe); the output goes to `log`."""
    exe = _resolve()
    proc = subprocess.run([exe, *map(_native, args)], capture_output=True, text=True, errors="replace")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"Blender failed ({proc.returncode}), see {log}:\n{tail}")


def decimate(vertices: np.ndarray, faces: np.ndarray, cell: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: vertices in the same `cell`-sized voxel merge into their mean; collapsed and duplicate faces go."""
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    keys = np.floor(vertices / cell).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    merged = np.zeros((len(counts), 3))
    np.add.at(merged, inverse, vertices)
    merged /= counts[:, None]
    tris = inverse[faces]
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
    _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    return merged, tris[np.sort(first)].astype(np.int64)


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Binary little-endian PLY: float32 x y z, then `uchar int` face lists (what blender/arch/scan.py reads)."""
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\n"
              f"property float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    records = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<i4", 3)])
    records["n"], records["i"] = 3, faces
    path.write_bytes(header.encode("ascii") + np.asarray(vertices, "<f4").tobytes() + records.tobytes())


def _write_scan(ctx: StageContext) -> Path | None:
    """The LiDAR mesh in the room frame, decimated; None when the capture has no mesh file."""
    ply = mesh_path(ctx.capture, ctx.room)
    if not ply.exists():
        return None
    mesh = load_mesh_ply(ply)
    T = world_to_room(ctx.capture, ctx.room, ctx.turn)
    vertices, faces = decimate(mesh["vertices"] @ T[:3, :3].T + T[:3, 3], mesh["faces"])
    path = ctx.out / SCAN_FILE
    write_ply(path, vertices, faces)
    return path


def run(ctx: StageContext) -> Path:
    arch = require(ctx, "architecture")
    _resolve()
    args = ["--background", "--python-exit-code", "1", "--python", BUILD_SCRIPT, "--", "--arch", arch, "--out-dir", ctx.out]
    textures = ctx.out / "textures" / "textures.json"
    if textures.exists():
        args += ["--textures", textures]
    scan = _write_scan(ctx)
    if scan is not None:
        args += ["--scan", scan]
    for name in (*OUTPUTS, *BACKUPS):  # so a run that writes nothing cannot pass on the previous run's files
        (ctx.out / name).unlink(missing_ok=True)
    run_blender(args, ctx.out / "blender.log")
    missing = [name for name in OUTPUTS if not (ctx.out / name).exists()]
    if missing:
        raise RuntimeError(f"Blender finished without writing {', '.join(missing)}, see {ctx.out / 'blender.log'}")
    return output_path(ctx, "blender")
