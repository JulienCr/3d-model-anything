"""Readers for house-capture/5 folders: manifest, photo poses, LiDAR mesh PLY, RoomPlan JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from room_model.context import InvalidCapture

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def colmajor(values, n: int) -> np.ndarray:
    """Column-major flat list (n*n floats) to an (n, n) matrix."""
    return np.asarray(values, dtype=np.float64).reshape(n, n).T


def arkit_to_cv(c2w_arkit: np.ndarray) -> np.ndarray:
    """ARKit camera (x right, y up, looks down -Z) to OpenCV camera (y down, looks down +Z)."""
    return c2w_arkit @ ARKIT_TO_CV


def invert_rigid(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


@dataclass
class Photo:
    name: str  # basename, e.g. "0000.jpg"
    rel_file: str
    width: int
    height: int
    K: np.ndarray  # 3x3, pixels of the raw sensor image
    c2w_arkit: np.ndarray
    timestamp: float
    room: int

    @property
    def c2w_cv(self) -> np.ndarray:
        return arkit_to_cv(self.c2w_arkit)

    @property
    def w2c_cv(self) -> np.ndarray:
        return invert_rigid(self.c2w_cv)


def load_manifest(capture: Path) -> dict:
    path = Path(capture) / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InvalidCapture(f"capture unreadable ({path}): {exc}") from exc


def load_photos(capture: Path, room: int | None = None) -> list[Photo]:
    capture = Path(capture)
    manifest = load_manifest(capture)
    entries = json.loads((capture / manifest["photos"]).read_text(encoding="utf-8"))
    photos = [
        Photo(
            name=Path(e["file"]).name,
            rel_file=e["file"],
            width=e["width"],
            height=e["height"],
            K=colmajor(e["intrinsics"], 3),
            c2w_arkit=colmajor(e["transform"], 4),
            timestamp=e["timestamp"],
            room=e["room"],
        )
        for e in entries
        if room is None or e["room"] == room
    ]
    return sorted(photos, key=lambda p: p.name)


def room_entry(capture: Path, room: int) -> dict:
    for r in load_manifest(capture)["rooms"]:
        if r["index"] == room:
            return r
    raise InvalidCapture(f"room {room} not in the manifest of {capture}")


def mesh_path(capture: Path, room: int) -> Path:
    """LiDAR mesh of one room."""
    return Path(capture) / room_entry(capture, room)["mesh"]["file"]


def load_mesh_ply(path: Path) -> dict:
    """LiDAR mesh arrays: vertices, normals, colors (uint8), coverage, inroom, faces, face_class."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v, f = ply["vertex"], ply["face"]
    return {
        "vertices": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64),
        "normals": np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64),
        "colors": np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8),
        "coverage": np.asarray(v["coverage"]),
        "inroom": np.asarray(v["inroom"]) if "inroom" in v.data.dtype.names else None,
        "faces": np.stack(f["vertex_indices"]).astype(np.int64),
        "face_class": np.asarray(f["class"]) if "class" in f.data.dtype.names else None,
    }


def load_roomplan(capture: Path, room: int = 0) -> dict:
    capture = Path(capture)
    return json.loads((capture / room_entry(capture, room)["file"]).read_text(encoding="utf-8"))
