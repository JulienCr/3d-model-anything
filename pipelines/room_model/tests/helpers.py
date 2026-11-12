"""Shared helpers for tests that read a shell document built from a synthetic capture."""
from __future__ import annotations

import numpy as np

from room_model import shell
from room_model.context import StageContext
from room_model.frame import world_to_room
from synthetic import SyntheticOptions, make_capture


def make_case(root, name, **options):
    capture = root / name
    truth = make_capture(capture, SyntheticOptions(**options))
    return StageContext(capture, root / f"{name}-out"), truth


def wall_labels(doc, truth, name):
    ids = truth.wall_ident[name]
    return [e["label"] for e in doc["elements"] if e["kind"] == "wall" and e["roomplan_id"] in ids]


def wall_line(doc, label):
    return next(w for w in shell.wall_lines(doc) if w["label"] == label)


def truth_to_room(ctx, truth):
    return world_to_room(ctx.capture, ctx.room, ctx.turn) @ truth.world_from_local


def plan_point(T, x, z):
    """Truth-local (x, z) on the floor plane mapped to the room frame (x, z)."""
    return (T @ [x, 0.0, z, 1.0])[[0, 2]]


def colour_stub(rgb, box, points=None):
    """Stand-in for SAM: the region of uniform colour under the box centre (a sliver when the centre is a gap)."""
    from scipy import ndimage

    x0, y0, x1, y1 = (int(round(v)) for v in box)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    ref = np.median(rgb[cy - 2:cy + 3, cx - 2:cx + 3].reshape(-1, 3), axis=0)
    labels, _ = ndimage.label(np.abs(rgb.astype(float) - ref).max(axis=2) < 20)
    return [(labels == labels[cy, cx], 0.9)]


def hand_shell(polygon, turns_deg, sigma_deg=0.5, support_m2=4.0, offsets=None):
    """Shell document of a hand-made outline: wall k runs polygon[k] -> polygon[k+1], its fitted line pivots about its
    midpoint by `turns_deg[k]` (and sits `offsets[k]` m further into the room). Returns (document, outline xyz)."""
    from room_model import shell

    pts = np.array(polygon, float)
    ccw = shapely_ccw(pts)
    lines, params, walls_diag, elements = [], {}, {}, []
    for k, a in enumerate(pts):
        b = pts[(k + 1) % len(pts)]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])
        phi = np.radians(turns_deg[k])
        n = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]) @ n
        centre = (a + b) / 2 + (0.0 if offsets is None else offsets[k]) * n
        label = f"M{k + 1}"
        lines.append({"n": n, "offset": float(n @ centre), "flags": [], "label": label})
        key = f"wall.{label}"
        params[f"{key}.azimuth"] = {"value": float(np.arctan2(n[1], n[0])), "sigma": float(np.radians(sigma_deg)), "unit": "rad", "source": "lidar"}
        params[f"{key}.offset"] = {"value": float(n @ centre), "sigma": 0.01, "unit": "m", "source": "lidar"}
        params[f"{key}.thickness"] = {"value": 0.2, "sigma": 0.05, "unit": "m", "source": "assumed"}
        params[f"{key}.top"] = {"value": 2.5, "sigma": 0.05, "unit": "m", "source": "roomplan"}
        walls_diag[label] = {"centroid": [float(v) for v in centre], "support_m2": support_m2}
    outline = np.column_stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]])
    corners = shell._corners(outline, lines)
    poly = [[float(c[0]), float(c[1])] for c in corners]
    params["floor.level"] = {"value": 0.0, "sigma": 0.005, "unit": "m", "source": "lidar"}
    elements.append({"id": "Floor", "kind": "floor", "params": {"level": "floor.level"}, "polygon": poly})
    for k, line in enumerate(lines):
        label = line["label"]
        elements.append({"id": f"Wall_{label}", "kind": "wall", "label": label, "roomplan_id": label, "start": poly[k],
                         "end": poly[(k + 1) % len(poly)], "flags": list(line["flags"]),
                         "params": {n: f"wall.{label}.{n}" for n in ("azimuth", "offset", "thickness", "top")}})
    doc = {"schema": "room-model/shell-1", "parameters": params, "elements": elements,
           "diagnostics": {"walls": walls_diag, "floor": {}, "ceilings": [], "partitions": [], "flags": [], "columns": []}}
    return doc, outline


def shapely_ccw(pts) -> bool:
    import shapely

    return bool(shapely.LinearRing(pts).is_ccw)


def add_hand_partition(doc, start, end, turn_deg=0.0, sigma_deg=0.3, thickness=0.1):
    """Adds `Partition_P<k>` along start -> end (face A on the left of the direction); its fitted line pivots about the
    midpoint by `turn_deg` and runs to the floor polygon. Returns the element id."""
    import shapely

    k = 1 + sum(e["kind"] == "partition" for e in doc["elements"])
    a, b = np.array(start, float), np.array(end, float)
    d = (b - a) / np.linalg.norm(b - a)
    phi = np.radians(turn_deg)
    rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    d, n = rot @ d, rot @ np.array([-d[1], d[0]])
    centre = (a + b) / 2
    floor = shapely.Polygon(next(e for e in doc["elements"] if e["kind"] == "floor")["polygon"])
    piece = floor.intersection(shapely.LineString([centre - 50 * d, centre + 50 * d]))
    ends = [list(map(float, c)) for c in piece.coords]
    key = f"partition.P{k}"
    doc["parameters"].update({
        f"{key}.azimuth": {"value": float(np.arctan2(n[1], n[0])), "sigma": float(np.radians(sigma_deg)), "unit": "rad", "source": "lidar"},
        f"{key}.offset": {"value": float(n @ centre), "sigma": 0.01, "unit": "m", "source": "lidar"},
        f"{key}.thickness": {"value": thickness, "sigma": 0.03, "unit": "m", "source": "lidar"},
        f"{key}.top": {"value": 2.5, "sigma": 0.05, "unit": "m", "source": "roomplan"}})
    doc["elements"].append({"id": f"Partition_P{k}", "kind": "partition", "label": f"P{k}", "roomplan_id": f"P{k}",
                            "start": ends[0], "end": ends[-1], "flags": [],
                            "params": {n_: f"{key}.{n_}" for n_ in ("azimuth", "offset", "thickness", "top")}})
    doc["diagnostics"]["partitions"].append({"id": f"Partition_P{k}", "support_a_m2": 2.0, "support_b_m2": 2.0,
                                             "centroid": [float(v) for v in centre]})
    return f"Partition_P{k}"


def add_hand_beam(doc, axis, width=0.3, underside=2.2, top=2.5):
    """Adds `Beam_<k>` along `axis` (two plan points)."""
    k = 1 + sum(e["kind"] == "beam" for e in doc["elements"])
    values = {"underside": underside, "width": width, "top": top}
    for name, v in values.items():
        doc["parameters"][f"beam.{k}.{name}"] = {"value": v, "sigma": 0.01, "unit": "m", "source": "lidar"}
    doc["elements"].append({"id": f"Beam_{k}", "kind": "beam", "axis": [list(map(float, p)) for p in axis], "flags": [],
                            "params": {n: f"beam.{k}.{n}" for n in values}})
    return f"Beam_{k}"


def add_hand_column(doc, x0, z0, size=0.5, engaged="M1"):
    """Adds `Column_<k>` with its lower-z face at `z0` (a face flush with the wall it is engaged on), `size` square."""
    k = 1 + sum(e["kind"] == "column" for e in doc["elements"])
    values = {"center_x": x0 + size / 2, "center_z": z0 + size / 2, "size_x": size, "size_z": size, "top": 2.5}
    for name, v in values.items():
        doc["parameters"][f"column.{k}.{name}"] = {"value": v, "sigma": 0.01, "unit": "m", "source": "lidar"}
    doc["elements"].append({"id": f"Column_{k}", "kind": "column", "engaged_wall": engaged,
                            "params": {n: f"column.{k}.{n}" for n in values}})
    doc["diagnostics"]["columns"].append({"id": f"Column_{k}", "engaged_walls": [engaged]})
    return f"Column_{k}"


def ceiling_samples(zones, spacing=0.0158, noise=0.003, seed=0):
    """Ceiling-facing samples (about 4000 per m2): `zones` is a list of (x0, x1, z0, z1, level) rectangles."""
    from room_model.lidar import Samples

    rng = np.random.default_rng(seed)
    pts = []
    for x0, x1, z0, z1, y in zones:
        xs, zs = np.meshgrid(np.arange(x0 + spacing / 2, x1, spacing), np.arange(z0 + spacing / 2, z1, spacing))
        pts.append(np.column_stack([xs.ravel(), np.full(xs.size, y), zs.ravel()]))
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    P[:, 1] += rng.normal(0.0, noise, len(P))
    N = np.tile([0.0, -1.0, 0.0], (len(P), 1))
    return Samples(P=P, N=N, cls=np.zeros(len(P), np.int64))
