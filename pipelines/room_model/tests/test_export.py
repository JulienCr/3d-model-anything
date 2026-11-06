import csv
import json
import struct
import xml.etree.ElementTree as ET

import pytest

from room_model import export, model, shell
from room_model.context import output_path
from helpers import make_case

HEADER = ["id", "kind", "label", "value_cm", "sigma_cm", "source", "elements"]


def read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    n, kind = struct.unpack_from("<I4s", data, 12)
    assert kind == b"JSON"
    return json.loads(data[20:20 + n])


def section(text, title):
    body = text.split(f"## {title}\n", 1)[1]
    return body.split("\n## ", 1)[0]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("export"), "default")
    shell.run(ctx)
    model.run(ctx)
    export.run(ctx)
    doc = json.loads(output_path(ctx, "model").read_text(encoding="utf-8"))
    return ctx, doc


def accessor(gltf, node_name, attribute="POSITION"):
    node = next(n for n in gltf["nodes"] if n["name"] == node_name)
    return gltf["accessors"][gltf["meshes"][node["mesh"]]["primitives"][0]["attributes"][attribute]]


def test_glb_nodes_and_extras(scene):
    ctx, _ = scene
    gltf = read_glb(ctx.out / "model_measured.glb")
    names = [n["name"] for n in gltf["nodes"]]
    expected = ["Coque", "Floor", "Ceiling_1", "Ceiling_2", *[f"Wall_M{i}" for i in range(1, 7)], "Beam_1", "Column_1", "Column_2"]
    assert sorted(names) == sorted(expected)
    root = next(n for n in gltf["nodes"] if n["name"] == "Coque")
    assert len(root["children"]) == len(expected) - 1
    for node in gltf["nodes"]:
        if node["name"] == "Coque":
            continue
        params = node["extras"]["params"]
        assert params and all(p["sigma"] > 0 and p["id"] and p["source"] for p in params.values())
        assert gltf["materials"][gltf["meshes"][node["mesh"]]["primitives"][0]["material"]]["doubleSided"]


def test_glb_geometry_matches_model(scene):
    ctx, doc = scene
    gltf = read_glb(ctx.out / "model_measured.glb")
    value = lambda pid: doc["parameters"][pid]["value"]  # noqa: E731
    assert accessor(gltf, "Beam_1")["min"][1] == pytest.approx(value("beam.1.underside"), abs=1e-3)
    for el in doc["elements"]:
        if el["kind"] == "wall":
            assert accessor(gltf, el["id"])["max"][1] == pytest.approx(value(el["params"]["top"]), abs=1e-3)
    floor = accessor(gltf, "Floor")
    assert floor["min"][1] == floor["max"][1] == pytest.approx(value("floor.level"), abs=1e-3)
    assert accessor(gltf, "Ceiling_1")["min"][1] == pytest.approx(value("ceiling.1.level"), abs=1e-3)


def test_export_is_deterministic(scene):
    ctx, _ = scene
    names = ["model_measured.glb", "plan.svg", "dimensions.csv", "report.md"]
    first = {n: (ctx.out / n).read_bytes() for n in names}
    export.run(ctx)
    assert first == {n: (ctx.out / n).read_bytes() for n in names}


def test_plan_svg(scene):
    ctx, doc = scene
    root = ET.fromstring((ctx.out / "plan.svg").read_text(encoding="utf-8"))
    ids = {e.get("id") for e in root.iter() if e.get("id")}
    assert {el["id"] for el in doc["elements"]} <= ids
    texts = " ".join("".join(e.itertext()) for e in root.iter() if e.tag.endswith("text"))
    for d in doc["dimensions"]:
        if d["id"].startswith("S"):
            assert d["id"] in texts
    assert "±" in texts and "1 m" in texts


def test_dimensions_csv(scene):
    ctx, doc = scene
    rows = list(csv.reader((ctx.out / "dimensions.csv").open(encoding="utf-8")))
    assert rows[0] == HEADER
    assert len(rows) == len(doc["dimensions"]) + 1
    s01 = next(r for r in rows if r[0] == "S01")
    assert s01[3] == f"{doc['dimensions'][0]['value'] * 100:.1f}"
    assert s01[5] in ("assumed", "roomplan", "lidar") and ";" in s01[6]


def test_report_sections(scene):
    ctx, doc = scene
    text = (ctx.out / "report.md").read_text(encoding="utf-8")
    for title in ("Entrées", "Repère", "Recalage", "Murs", "Sol", "Plafonds", "Poutres", "Poteaux",
                  "Surfaces verticales non expliquées", "Orthogonalité", "Alertes", "Épaisseurs"):
        assert f"## {title}\n" in text
    assert "non exécuté" in section(text, "Recalage")
    assert "supposées à 20 cm" in section(text, "Épaisseurs")
    assert "%" in section(text, "Plafonds")


def test_report_shows_registration_summary(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("export-poses"), "default")
    shell.run(ctx)
    model.run(ctx)
    summary = {"reprojection_px_arkit": 2.5, "reprojection_px_refined": 0.75, "sim3_scale": 1.0123,
               "points3d": 4321, "refined": 40, "weak": 3, "seconds": 12.5}
    (ctx.out / "registration").mkdir()
    (ctx.out / "registration" / "poses.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")
    export.run(ctx)
    recalage = section((ctx.out / "report.md").read_text(encoding="utf-8"), "Recalage")
    assert "non exécuté" not in recalage and "0,75" in recalage and "4321" in recalage


def test_report_flags_fallback(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("export-fallback"), "no-wing", drop_wall_faces="wing_east")
    shell.run(ctx)
    model.run(ctx)
    export.run(ctx)
    doc = json.loads(output_path(ctx, "model").read_text(encoding="utf-8"))
    label = next(e["label"] for e in doc["elements"] if e["kind"] == "wall" and "roomplan_fallback" in e["flags"])
    alerts = section((ctx.out / "report.md").read_text(encoding="utf-8"), "Alertes")
    line = next(ln for ln in alerts.splitlines() if f"Mur {label}" in ln)
    assert "non vu par le LiDAR" in line
