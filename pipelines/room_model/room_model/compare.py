"""Compare a model.json against a reference (polygon, edge heights, dimensions) and report in Markdown."""
from __future__ import annotations

import numpy as np

from room_model.shell import wall_lines

MAX_ANGLE_DEG = 5.0
MAX_OFFSET_M = 0.5


def _edges(polygon) -> list[dict]:
    pts = np.asarray(polygon, dtype=float)
    out = []
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        d = b - a
        length = float(np.linalg.norm(d))
        if length > 1e-9:
            out.append({"index": i, "mid": (a + b) / 2, "angle": float(np.arctan2(d[1], d[0]))})
    return out


def _angle_delta_deg(wall_angle: float, edge_angle: float) -> float:
    """Signed angle between two undirected lines, in (-90, 90] degrees."""
    d = np.degrees(edge_angle - wall_angle) % 180.0
    return d - 180.0 if d > 90.0 else d


def _cm(value: float) -> str:
    return f"{round(value * 100, 1) + 0.0:+.1f}"


def _match(wall: dict, edges: list[dict]) -> tuple[dict, float, float] | None:
    n = wall["normal"]
    wall_angle = float(np.arctan2(n[0], -n[1]))
    mid = (wall["start"] + wall["end"]) / 2
    best = None
    for e in edges:
        angle = _angle_delta_deg(wall_angle, e["angle"])
        offset = float(n @ e["mid"]) - wall["offset"]
        if abs(angle) >= MAX_ANGLE_DEG or abs(offset) >= MAX_OFFSET_M:
            continue
        dist = float(np.linalg.norm(e["mid"] - mid))
        if best is None or dist < best[0]:
            best = (dist, e, offset, angle)
    return None if best is None else (best[1], best[2], best[3])


def _walls_section(model: dict, reference: dict) -> list[str]:
    edges = _edges(reference["polygon"])
    heights = reference.get("edge_heights")
    floor = model["parameters"].get("floor.level", {}).get("value", 0.0)
    tops = {e["label"]: model["parameters"][e["params"]["top"]]["value"]
            for e in model["elements"] if e["kind"] == "wall"}
    head = "| Mur | Arête de référence | Δ décalage (cm) | Δ angle (°) |" + (" Δ haut de mur (cm) |" if heights else "")
    lines = ["## Murs", "", "Δ = modèle − référence.", "", head, "|" + "---|" * (head.count("|") - 1)]
    used = set()
    for wall in wall_lines(model):
        found = _match(wall, edges)
        if found is None:
            lines.append(f"| {wall['label']} | aucune | | |" + (" |" if heights else ""))
            continue
        edge, offset, angle = found
        used.add(edge["index"])
        row = f"| {wall['label']} | {edge['index']} | {_cm(-offset)} | {round(-angle, 1) + 0.0:+.1f} |"
        if heights:
            ref_h = heights[edge["index"]]
            row += f" {_cm(tops[wall['label']] - floor - ref_h)} |"
        lines.append(row)
    missing = [e["index"] for e in edges if e["index"] not in used]
    if missing:
        lines += ["", "Arêtes de référence non appariées : " + ", ".join(map(str, missing)) + "."]
    return lines + [""]


def _dimensions_section(model: dict, reference: dict) -> list[str]:
    by_id = {d["id"]: d for d in model.get("dimensions", [])}
    lines = ["## Dimensions", "", "| Id | Modèle (m) | Référence (m) | Δ cm | Δ/σ |", "|---|---|---|---|---|"]
    for dim_id, ref in reference["dimensions"].items():
        d = by_id.get(dim_id)
        if d is None:
            lines.append(f"| {dim_id} | absente | {ref:.3f} | | |")
            continue
        delta = d["value"] - ref
        lines.append(f"| {dim_id} | {d['value']:.3f} | {ref:.3f} | {_cm(delta)} | {delta / d['sigma']:+.1f} |")
    return lines + [""]


def compare(model: dict, reference: dict) -> str:
    """Markdown report of model minus reference; every reference key is optional (metres, room frame)."""
    lines = ["# Comparaison modèle / référence", ""]
    if reference.get("polygon"):
        lines += _walls_section(model, reference)
    if reference.get("dimensions"):
        lines += _dimensions_section(model, reference)
    return "\n".join(lines)
