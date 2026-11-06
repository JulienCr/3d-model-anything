import copy
import csv
import io
import json
import struct
import xml.etree.ElementTree as ET

import pytest

from room_model import export, model, plan_svg, report, shell
from room_model.context import output_path
from helpers import make_case

HEADER = ["id", "kind", "label", "value_cm", "sigma_cm", "source", "elements", "flags"]


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
        if d["kind"] in ("width", "length"):
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
                  "Surfaces verticales non expliquées", "Éléments horizontaux non expliqués (réseaux, luminaires ?)", "Orthogonalité", "Alertes", "Épaisseurs"):
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


def decode_glb(path):
    """(gltf, bin bytes, {node name: (positions Nx3, indices)}) read from the BIN chunk."""
    import numpy as np

    data = path.read_bytes()
    n, _ = struct.unpack_from("<I4s", data, 12)
    gltf = json.loads(data[20:20 + n])
    start = 20 + n
    m, kind = struct.unpack_from("<I4s", data, start)
    assert kind == b"BIN\0"
    blob = data[start + 8:start + 8 + m]
    sizes = {5126: 4, 5125: 4}
    comps = {"VEC3": 3, "SCALAR": 1}

    def read(acc_i):
        acc = gltf["accessors"][acc_i]
        view = gltf["bufferViews"][acc["bufferView"]]
        nbytes = acc["count"] * comps[acc["type"]] * sizes[acc["componentType"]]
        assert nbytes <= view["byteLength"]
        dtype = np.float32 if acc["componentType"] == 5126 else np.uint32
        arr = np.frombuffer(blob, dtype, acc["count"] * comps[acc["type"]], view["byteOffset"])
        return arr.reshape(-1, comps[acc["type"]]) if comps[acc["type"]] > 1 else arr

    meshes = {}
    for node in gltf["nodes"]:
        if "mesh" in node:
            prim = gltf["meshes"][node["mesh"]]["primitives"][0]
            meshes[node["name"]] = (read(prim["attributes"]["POSITION"]), read(prim["indices"]), prim["attributes"]["POSITION"])
    return gltf, blob, meshes


def test_wall_bodies_lie_behind_interior_face(scene):
    import numpy as np

    ctx, doc = scene
    _, _, meshes = decode_glb(ctx.out / "model_measured.glb")
    value = lambda pid: doc["parameters"][pid]["value"]  # noqa: E731
    for el in doc["elements"]:
        if el["kind"] != "wall":
            continue
        az, offset, thick = (value(el["params"][k]) for k in ("azimuth", "offset", "thickness"))
        depth = meshes[el["id"]][0][:, [0, 2]] @ np.array([np.cos(az), np.sin(az)])
        assert depth.max() <= offset + 1e-6
        assert depth.min() == pytest.approx(offset - thick, abs=1e-3)
        start, end = (np.array(el[k]) for k in ("start", "end"))
        t = (end - start) / np.linalg.norm(end - start)
        along = (meshes[el["id"]][0][:, [0, 2]] - start) @ t
        assert along.min() == pytest.approx(0.0, abs=1e-3)
        assert along.max() == pytest.approx(np.linalg.norm(end - start) + thick, abs=1e-3)


def test_glb_binary_integrity_and_winding(scene):
    import numpy as np

    ctx, _ = scene
    gltf, blob, meshes = decode_glb(ctx.out / "model_measured.glb")
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] % 4 == 0 and view["byteOffset"] + view["byteLength"] <= len(blob)
    for name, (pos, idx, acc_i) in meshes.items():
        acc = gltf["accessors"][acc_i]
        assert np.allclose(pos.min(0), acc["min"]) and np.allclose(pos.max(0), acc["max"])
        assert idx.max() < len(pos)
        if name.split("_")[0] not in ("Wall", "Beam", "Column"):
            continue
        tri = pos[idx.reshape(-1, 3)].astype(np.float64)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert ((tri.mean(1) - pos.mean(0)) * normals).sum(1).min() > 0


def test_beam_to_verify_is_grouped_labelled_and_reported(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("verify"), "verify", short_beam=True, short_beam_sides=1)
    shell.run(ctx)
    model.run(ctx)
    export.run(ctx)
    doc = json.loads(output_path(ctx, "model").read_text(encoding="utf-8"))
    (unsure,) = [e["id"] for e in doc["elements"] if e["kind"] == "beam" and e["flags"]]
    gltf = read_glb(ctx.out / "model_measured.glb")
    names = [n["name"] for n in gltf["nodes"]]
    group = next(n for n in gltf["nodes"] if n["name"] == "À vérifier")
    shell_root = next(n for n in gltf["nodes"] if n["name"] == "Coque")
    assert [names[i] for i in group["children"]] == [unsure]
    assert unsure not in [names[i] for i in shell_root["children"]]
    k = unsure.split("_")[1]
    assert f"B{k} s/p" in (ctx.out / "plan.svg").read_text(encoding="utf-8")
    assert any(t.endswith("?") for t in ET.fromstring((ctx.out / "plan.svg").read_text(encoding="utf-8")).itertext())
    text = (ctx.out / "report.md").read_text(encoding="utf-8")
    assert "à vérifier" in section(text, "Poutres") and "poutre probable, à confirmer" in section(text, "Alertes")
    assert all(d["flags"] == ["to_verify"] for d in doc["dimensions"] if d["id"] in (f"B{k}-u", f"B{k}-w"))
    assert "Station B" not in section(text, "Alertes")


def _alerts(doc):
    return section(report.render(doc), "Alertes")


def test_obstructed_station_is_reported_but_not_a_beam_dimension(scene):
    _, doc = scene
    doc = copy.deepcopy(doc)
    station = next(d for d in doc["dimensions"] if d["kind"] == "width")
    station["flags"] = ["obstacle_or_uneven_wall"]
    assert f"Station {station['id']}" in _alerts(doc)
    assert "Station" not in _alerts(scene[1])


def test_csv_and_plan_show_the_station_flags(scene):
    _, doc = scene
    doc = copy.deepcopy(doc)
    station = next(d for d in doc["dimensions"] if d["kind"] == "width")
    station["flags"] = ["obstacle_or_uneven_wall"]
    rows = {r[0]: r for r in csv.reader(io.StringIO(export.dimensions_csv(doc)))}
    assert rows[station["id"]][-1] == "obstacle_or_uneven_wall"
    assert next(r for k, r in rows.items() if k != station["id"] and k != "id")[-1] == ""
    texts = [t for t in ET.fromstring(plan_svg.render(doc)).itertext() if t.startswith(station["id"] + " ")]
    assert len(texts) == 1 and texts[0].endswith(" ~")
    assert not any(t.endswith("~") for t in ET.fromstring(plan_svg.render(scene[1])).itertext())


def test_report_wording_and_thresholds(scene):
    _, doc = scene
    text = report.render(doc)
    assert "LiDAR" in section(text, "Murs") and "lidar" not in section(text, "Murs")
    assert "Écart à l'angle droit ou à l'alignement" in section(text, "Orthogonalité")
    doc = copy.deepcopy(doc)
    doc["diagnostics"]["ceiling_unobserved_fraction"] = 0.04
    assert "du plafond non observé" not in _alerts(doc)
    doc["diagnostics"]["ceiling_unobserved_fraction"] = 0.2
    assert "20,0 % du plafond non observé" in _alerts(doc)


def test_report_lists_every_engaged_wall_of_a_column(scene):
    _, doc = scene
    doc = copy.deepcopy(doc)
    engaged = {e["id"] for e in doc["elements"] if e["kind"] == "column" and e["engaged_wall"]}
    column = next(c for c in doc["diagnostics"]["columns"] if c["id"] in engaged)
    column["engaged_walls"] = ["M1", "M2"]
    row = next(r for r in section(report.render(doc), "Poteaux").splitlines() if r.startswith(f"| {column['id'].split('_')[1]} |"))
    assert "M1, M2" in row


def test_report_shows_registration_fallback_reason(scene):
    _, doc = scene
    summary = {"reprojection_px_arkit": 2.5, "reprojection_px_refined": 2.5, "sim3_scale": 1.0,
               "points3d": 10, "refined": 0, "weak": 3, "seconds": 1.0}
    text = report.render(doc, {"summary": summary, "fallback": "not all images registered"})
    assert "not all images registered" in section(text, "Recalage")


def test_door_on_a_wall_off_the_outline_is_reported(tmp_path_factory):
    ctx, truth = make_case(tmp_path_factory.mktemp("orphan"), "orphan", extra_roomplan_wall=True)
    path = ctx.capture / "rooms" / "room-0.json"
    data = json.loads(path.read_text())
    data["doors"][0]["parentIdentifier"] = truth.extra_ident
    path.write_text(json.dumps(data))
    shell.run(ctx)
    model.run(ctx)
    export.run(ctx)
    alerts = section((ctx.out / "report.md").read_text(encoding="utf-8"), "Alertes")
    assert "Porte rattachée à un mur hors contour, ignorée" in alerts
