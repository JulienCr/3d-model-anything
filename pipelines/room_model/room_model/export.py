"""Export stage: measured GLB, dimensioned plan, dimensions table and French report."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from room_model import meshes, plan_svg, report
from room_model.context import StageContext, output_path, require
from room_model.gltf import GltfNode, write_glb

KIND_ORDER = ["floor", "ceiling", "wall", "partition", "beam", "column"]
CSV_HEADER = ["id", "kind", "label", "value_cm", "sigma_cm", "source", "elements", "flags"]


def _extras(doc: dict, el: dict) -> dict:
    params = {name: {"id": pid, **{k: doc["parameters"][pid][k] for k in ("value", "sigma", "source")}}
              for name, pid in el["params"].items()}
    return {"kind": el["kind"], "params": params, "flags": list(el.get("flags", []))}


def build_nodes(doc: dict) -> list[GltfNode]:
    value = lambda pid: doc["parameters"][pid]["value"]  # noqa: E731
    y0 = value("floor.level")
    nodes = []
    for el in sorted(doc["elements"], key=lambda e: KIND_ORDER.index(e["kind"])):
        p, kind = el["params"], el["kind"]
        if kind == "floor":
            mesh = meshes.floor_mesh(el["polygon"], y0)
        elif kind == "ceiling":
            mesh = meshes.ceiling_mesh(el["region"], value(p["level"]))
        elif kind in ("wall", "partition"):
            az = value(p["azimuth"])
            normal = (math.cos(az), math.sin(az))
            mesh = meshes.wall_mesh(el["start"], el["end"], normal, value(p["thickness"]), y0, value(p["top"]),
                                    extend_end=kind == "wall")
        elif kind == "beam":
            mesh = meshes.beam_mesh(el["axis"], value(p["width"]), value(p["underside"]), value(p["top"]))
        else:
            mesh = meshes.column_mesh(*(value(p[n]) for n in ("center_x", "center_z", "size_x", "size_z")), y0, value(p["top"]))
        nodes.append(GltfNode(el["id"], mesh[0], mesh[1], kind, _extras(doc, el)))
    unsure = [n.name for n in nodes if "to_verify" in n.extras.get("flags", [])]
    root = GltfNode("Coque", extras={"kind": "shell", "schema": doc["schema"], "version": doc["version"]},
                    children=[n.name for n in nodes if n.name not in unsure])
    groups = [GltfNode("À vérifier", extras={"kind": "to_verify"}, children=unsure)] if unsure else []
    return [root, *groups, *nodes]


def dimensions_csv(doc: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for d in doc["dimensions"]:
        writer.writerow([d["id"], d["kind"], d["label"], f"{d['value'] * 100:.1f}", f"{d['sigma'] * 100:.1f}",
                         d["source"], ";".join(d["elements"]), ";".join(d["flags"])])
    return buf.getvalue()


def run(ctx: StageContext) -> Path:
    doc = json.loads(require(ctx, "model").read_text(encoding="utf-8"))
    poses_path = ctx.out / "registration" / "poses.json"
    poses = json.loads(poses_path.read_text(encoding="utf-8")) if poses_path.exists() else None
    path = output_path(ctx, "export")
    path.parent.mkdir(parents=True, exist_ok=True)
    (ctx.out / "plan.svg").write_text(plan_svg.render(doc), encoding="utf-8")
    (ctx.out / "dimensions.csv").write_text(dimensions_csv(doc), encoding="utf-8")
    (ctx.out / "report.md").write_text(report.render(doc, poses), encoding="utf-8")
    write_glb(path, build_nodes(doc))
    return path
