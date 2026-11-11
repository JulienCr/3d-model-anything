"""Minimal deterministic glTF 2.0 binary (GLB) writer."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MATERIALS = {  # fixed base colours (linear RGBA)
    "floor": [0.55, 0.45, 0.35, 1.0],
    "wall": [0.85, 0.85, 0.80, 1.0],
    "partition": [0.80, 0.80, 0.75, 1.0],
    "ceiling": [0.92, 0.92, 0.95, 1.0],
    "beam": [0.60, 0.40, 0.25, 1.0],
    "column": [0.45, 0.50, 0.60, 1.0],
}
FLOAT, UINT = 5126, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


@dataclass
class GltfNode:
    name: str
    positions: np.ndarray | None = None
    indices: np.ndarray | None = None
    material: str | None = None
    extras: dict = field(default_factory=dict)
    children: list[str] = field(default_factory=list)


def _flat(positions: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle duplicated vertices and their flat normals, as float32."""
    tri = np.asarray(positions, np.float64)[np.asarray(indices, np.int64).reshape(-1, 3)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, length, out=np.zeros_like(n), where=length > 0)
    return tri.reshape(-1, 3).astype(np.float32), np.repeat(n, 3, axis=0).astype(np.float32)


def _round(obj):
    """Round floats so the JSON text does not depend on float noise."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    return obj


def _chunk(kind: bytes, data: bytes, pad: bytes) -> bytes:
    data += pad * (-len(data) % 4)
    return struct.pack("<I4s", len(data), kind) + data


def build_glb(nodes: list[GltfNode]) -> bytes:
    blob = bytearray()
    views, accessors, meshes, gltf_nodes = [], [], [], []
    index = {n.name: i for i, n in enumerate(nodes)}
    for node in nodes:
        entry: dict = {"name": node.name, "extras": _round(node.extras)}
        if node.positions is not None and len(node.indices):
            pos, nor = _flat(node.positions, node.indices)
            idx = np.arange(len(pos), dtype=np.uint32)
            attrs = {}
            for key, arr, comp, target in (("POSITION", pos, "VEC3", ARRAY_BUFFER), ("NORMAL", nor, "VEC3", ARRAY_BUFFER),
                                           (None, idx, "SCALAR", ELEMENT_ARRAY_BUFFER)):
                views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes, "target": target})
                blob += arr.tobytes()
                acc = {"bufferView": len(views) - 1, "componentType": UINT if key is None else FLOAT,
                       "count": len(arr), "type": comp}
                if key == "POSITION":
                    acc["min"], acc["max"] = [float(v) for v in pos.min(0)], [float(v) for v in pos.max(0)]
                accessors.append(acc)
                if key:
                    attrs[key] = len(accessors) - 1
            prim = {"attributes": attrs, "indices": len(accessors) - 1, "mode": 4}
            if node.material:
                prim["material"] = list(MATERIALS).index(node.material)
            meshes.append({"name": node.name, "primitives": [prim]})
            entry["mesh"] = len(meshes) - 1
        if node.children:
            entry["children"] = [index[c] for c in node.children]
        gltf_nodes.append(entry)
    children = {c for n in nodes for c in n.children}
    doc = {
        "asset": {"version": "2.0", "generator": "room_model"},
        "scene": 0,
        "scenes": [{"nodes": [index[n.name] for n in nodes if n.name not in children]}],
        "nodes": gltf_nodes,
        "meshes": meshes,
        "materials": [{"name": k, "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorFactor": v, "metallicFactor": 0.0, "roughnessFactor": 1.0}}
                      for k, v in MATERIALS.items()],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    body = _chunk(b"JSON", text, b" ") + _chunk(b"BIN\0", bytes(blob), b"\0")
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def write_glb(path: Path, nodes: list[GltfNode]) -> None:
    Path(path).write_bytes(build_glb(nodes))
