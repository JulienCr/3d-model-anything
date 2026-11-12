import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
from PIL import Image

from room_model import architecture, blender_stage, meshes, model, openings, shell
from room_model.context import BlenderMissing, output_path
from helpers import make_case

PROBE = Path(__file__).with_name("blender_probe.py")
BLENDER_DIR = Path(__file__).resolve().parents[1] / "blender"
sys.path.insert(0, str(BLENDER_DIR))  # the pure-Python modules of blender/arch are importable as `arch.*`
from arch import plan as wall_plan  # noqa: E402
OPEN_RATIO = 0.6
needs_blender = pytest.mark.skipif(shutil.which(blender_stage.blender_executable()) is None, reason="Blender is not installed")


def _plane(side: float, n: int):
    """n x n quads over a side x side square at z = 0, two triangles each."""
    xs = np.linspace(0.0, side, n + 1)
    gx, gy = np.meshgrid(xs, xs)
    vertices = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    return vertices, np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])


def test_decimate_keeps_the_shape_with_far_fewer_faces():
    vertices, faces = _plane(1.2, 224)
    assert len(faces) > 100_000
    v, f = blender_stage.decimate(vertices, faces)
    assert 0 < len(f) < 5_000
    assert f.max() < len(v) and (f[:, 0] != f[:, 1]).all() and (f[:, 1] != f[:, 2]).all() and (f[:, 0] != f[:, 2]).all()
    assert np.abs(v.min(0) - vertices.min(0)).max() < 0.03 and np.abs(v.max(0) - vertices.max(0)).max() < 0.03
    assert np.abs(v[:, 2]).max() < 1e-9


def test_decimate_keeps_orientation():
    vertices, faces = _plane(1.2, 100)
    v, f = blender_stage.decimate(vertices, faces)
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    assert (normals[:, 2] > 0).all()


def read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    n, kind = struct.unpack_from("<I4s", data, 12)
    assert kind == b"JSON"
    return json.loads(data[20:20 + n])


def _value(arch, pid):
    return arch["parameters"][pid]["value"]


def _element(arch, id_):
    return next(e for e in arch["elements"] if e["id"] == id_)


def _blender_point(x, y, z):
    return [x, -z, y]


def _on_face(arch, opening, along=0.0):
    """(host, point on the host face line, interior normal) in the plan of the room frame, `along` m from the centre."""
    host = _element(arch, opening["host"])
    az = _value(arch, host["params"]["azimuth"])
    n = np.array([math.cos(az), math.sin(az)])
    start, end = np.array(host["start"]), np.array(host["end"])
    u = (end - start) / np.linalg.norm(end - start)
    p = start + u * (_value(arch, opening["params"]["center_s"]) + along)
    return host, p + n * (_value(arch, host["params"]["offset"]) - n @ p), n


def _ray(arch, opening_id, name, along, height, side_distance=0.6, target=None):
    """Ray crossing the host of `opening_id` (or `target`) at `along` m from the centre of the opening, `height` m above the floor."""
    opening = _element(arch, opening_id)
    host, p, n = _on_face(arch, opening, along)
    floor = _value(arch, "floor.level")
    origin = p + n * side_distance
    return {"name": name, "object": target or host["id"], "origin": _blender_point(origin[0], floor + height, origin[1]),
            "direction": _blender_point(-n[0], 0.0, -n[1])}


# (source, id, host, centre along the host in m, width, fields): typed copies of the synthetic openings, as the
# vision stage will type them; the originals keep their defaults.
JOINERY_VARIANTS = [
    ("Door_1", "Door_2", "Wall_M3", 0.7, 0.8, {"type": "hinged_double"}),
    ("Door_1", "Door_3", "Wall_M3", 1.7, 0.8, {"type": "sliding", "hinge": "right"}),
    ("Door_1", "Door_4", "Wall_M3", 2.7, 0.8, {"type": "pocket"}),
    ("Door_1", "Door_5", "Wall_M5", 1.25, 1.2, {"type": "french_window"}),
    ("Door_1", "Door_6", "Wall_M4", 1.0, 0.8,
     {"type": "hinged_single", "hinge": "right", "swing": "out", "panels": 3, "handle": False}),
    ("Door_1", "Door_7", "Partition_P1", 0.3, 0.4, {"face": "B"}),
    ("Door_1", "Door_8", "Partition_P1", 1.82, 0.3, {"face": "A", "swing": "out"}),
    ("Window_1", "Window_2", "Wall_M1", 0.8, 0.8, {"type": "casement_single", "hinge": "right"}),
    ("Window_1", "Window_3", "Wall_M1", 1.8, 0.8, {"type": "sliding"}),
    ("Window_1", "Window_4", "Wall_M1", 4.6, 0.8, {"type": "fixed"}),
    ("Window_1", "Window_5", "Wall_M1", 5.5, 0.6, {"type": "tilt_turn"}),
]


def _add_variants(arch, variants=JOINERY_VARIANTS):
    for source, new_id, host, centre, width, fields in variants:
        element = json.loads(json.dumps(_element(arch, source)))
        element.update(id=new_id, host=host, **fields)
        for name, pid in element["params"].items():
            element["params"][name] = f"opening.{new_id}.{name}"
            arch["parameters"][element["params"][name]] = dict(arch["parameters"][pid])
        arch["parameters"][element["params"]["center_s"]]["value"] = centre
        arch["parameters"][element["params"]["width"]]["value"] = width
        arch["elements"].append(element)


RIGS = [
    ("closed", "Door_1", {}),
    ("open", "Door_1", {"Ouverture": 1}),
    ("open-flipped", "Door_1", {"Ouverture": 1, "Inverser sens": True}),
    ("open-mirrored", "Door_1", {"Ouverture": 1, "Inverser charni\u00e8re": True}),
    ("open-both", "Door_1", {"Ouverture": 1, "Inverser sens": True, "Inverser charni\u00e8re": True}),
    ("half", "Door_1", {"Ouverture": 0.5}),
    ("short-angle", "Door_1", {"Ouverture": 1, "AngleMax": 60}),
]
for _id in (*(f"Door_{k}" for k in range(1, 9)),
            "Window_1", "Window_2", "Window_3", "Window_4", "Window_5"):
    RIGS += [(f"closed-{_id}", _id, {}), (f"open-{_id}", _id, {"Ouverture": 1}),
             (f"flipped-{_id}", _id, {"Ouverture": 1, "Inverser sens": True}),
             (f"mirrored-{_id}", _id, {"Ouverture": 1, "Inverser charni\u00e8re": True}),
             (f"closed-mirrored-{_id}", _id, {"Inverser charni\u00e8re": True}),
             (f"closed-flipped-{_id}", _id, {"Inverser sens": True})]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("blender")
    ctx, _ = make_case(root, "scene", window=True, partition=True, partition_opening=True, partition_opening_width=0.6,
                       orphan_opening=True)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    arch_path = output_path(ctx, "architecture")
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    _add_variants(arch)
    arch_path.write_text(json.dumps(arch), encoding="utf-8")
    blender_stage.run(ctx)

    rays = []
    for oid in ("Door_1", "Window_1", "Opening_1"):
        el = _element(arch, oid)
        w, h = (_value(arch, el["params"][k]) for k in ("width", "height"))
        sill = _value(arch, el["params"]["sill"])
        rays += [
            _ray(arch, oid, f"{oid}-centre", 0.0, sill + h / 2),
            _ray(arch, oid, f"{oid}-beside", w / 2 + 0.3, sill + h / 2),
            _ray(arch, oid, f"{oid}-above", 0.0, sill + h + 0.15),
        ]
    window = _element(arch, "Window_1")
    rays.append(_ray(arch, "Window_1", "Window_1-below", 0.0, _value(arch, window["params"]["sill"]) / 2))
    rays.append(_ray(arch, "Door_1", "Door_1-leaf", 0.0, 1.0, target="Door_1_Leaf_1"))
    spec = {"rays": rays, "rigs": [{"name": n, "empty": e, "props": p} for n, e, p in RIGS], "arch": str(arch_path),
            "blender_dir": str(BLENDER_DIR)}
    spec_path, facts_path = ctx.out / "spec.json", ctx.out / "facts.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    blender_stage.run_blender(["--background", ctx.out / "architecture.blend", "--python-exit-code", "1", "--python", PROBE,
                               "--", spec_path, facts_path], ctx.out / "probe.log")
    return ctx, arch, json.loads(facts_path.read_text(encoding="utf-8"))


def _probe(ctx, arch_path, rays=(), spans=(), texel_rays=()):
    spec = {"rays": list(rays), "rigs": [], "spans": list(spans), "texel_rays": list(texel_rays), "arch": str(arch_path), "blender_dir": str(BLENDER_DIR)}
    spec_path, facts_path = ctx.out / "spec.json", ctx.out / "facts.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    blender_stage.run_blender(["--background", ctx.out / "architecture.blend", "--python-exit-code", "1", "--python", PROBE,
                               "--", spec_path, facts_path], ctx.out / "probe.log")
    return json.loads(facts_path.read_text(encoding="utf-8"))


GRID_U, GRID_V = (0.12, 0.37, 0.63, 0.88), (0.2, 0.5, 0.8)


@pytest.fixture(scope="module")
def textured_scene(tmp_path_factory):
    """A scene whose textures encode their own texel coordinates (red: u, green: v), with rays cast at known points of every surface."""
    from room_model import textures
    ctx, _ = make_case(tmp_path_factory.mktemp("textured"), "textured", window=True, partition=True, partition_opening=True,
                       partition_opening_width=0.6)
    for stage in (shell, model, openings, architecture):
        stage.run(ctx)
    arch = json.loads(output_path(ctx, "architecture").read_text(encoding="utf-8"))
    folder = ctx.out / "textures"
    folder.mkdir()
    doc, rays = {}, []
    for s in textures.surfaces_of(arch):
        nu, nv = s.size
        pixels = np.zeros((nv, nu, 3), np.uint8)
        pixels[..., 0] = ((np.arange(nu) + 0.5) / nu * 255).astype(np.uint8)
        pixels[..., 1] = (((np.arange(nv) + 0.5) / nv * 255).astype(np.uint8))[:, None]
        name = f"{s.element}_{s.face}.png"
        Image.fromarray(pixels[::-1]).save(folder / name)
        doc.setdefault(s.element, {})[s.face] = {
            "image": f"textures/{name}", "origin": list(s.origin), "u_axis": list(s.u_axis), "v_axis": list(s.v_axis),
            "normal": list(s.normal), "texel": textures.TEXEL, "size": [nu, nv], "coverage": 1.0}
        for fu in GRID_U:
            for fv in GRID_V:
                p = s.origin + fu * nu * textures.TEXEL * s.u_axis + fv * nv * textures.TEXEL * s.v_axis
                rays.append({"name": f"{s.element}/{s.face}/{fu}/{fv}", "object": s.element,
                             "origin": _blender_point(*(p + 0.5 * s.normal)), "direction": _blender_point(*(-s.normal))})
    (folder / "textures.json").write_text(json.dumps(doc), encoding="utf-8")
    blender_stage.run(ctx)
    return ctx, arch, doc, _probe(ctx, output_path(ctx, "architecture"), texel_rays=rays)


@needs_blender
def test_blender_uv(textured_scene):
    _, _, doc, facts = textured_scene
    hits = {}
    for name, texel in facts["texels"].items():
        element, face, fu, fv = name.split("/")
        if not texel["hit"]:
            continue
        nu, nv = doc[element][face]["size"]
        hits[(element, face)] = hits.get((element, face), 0) + 1
        assert texel["material"] == f"Tex_{element}_{face}", name
        assert texel["rgb"][0] / 255 == pytest.approx(float(fu), abs=1.5 / nu + 0.01), name
        assert texel["rgb"][1] / 255 == pytest.approx(float(fv), abs=1.5 / nv + 0.01), name
    expected = {(e, f) for e, faces in doc.items() for f in faces}
    assert set(hits) == expected and min(hits.values()) >= 4, "every surface is hit by several rays, cuts left aside"
    assert {"Floor", "Ceiling_1", "Partition_P1"} <= set(doc) and set(doc["Partition_P1"]) == {"A", "B"}


@needs_blender
def test_textures_leave_the_other_faces_on_the_named_material(textured_scene):
    _, arch, doc, facts = textured_scene
    for element, faces in doc.items():
        assert facts["objects"][element]["materials"][0] in ("Mur", "Sol", "Plafond")
        assert facts["objects"][element]["materials"][1:] == [f"Tex_{element}_{face}" for face in faces]
    assert {"Mur", "Sol", "Plafond", "Menuiserie", "Verre", "Moulure"} <= set(facts["materials"])


@needs_blender
def test_textured_glb_embeds_small_jpegs(textured_scene):
    ctx, _, doc, _ = textured_scene
    path = ctx.out / "architecture.glb"
    gltf = read_glb(path)
    surfaces = sum(len(faces) for faces in doc.values())
    assert len(gltf["images"]) >= surfaces and {i["mimeType"] for i in gltf["images"]} == {"image/jpeg"}
    assert path.stat().st_size < 2_000_000


@needs_blender
def test_perspective_render_shows_the_textures(textured_scene):
    ctx, _, _, _ = textured_scene
    rgb = np.asarray(Image.open(ctx.out / "renders" / "persp_1.png").convert("RGB"), int)
    coloured = ((rgb[..., :2].max(axis=2) - rgb[..., 2]) > 30).mean()  # the encoded textures have no blue, the plain materials are nearly grey
    assert coloured > 0.3


@pytest.fixture(scope="module")
def native_scene(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("native"), "native", window=True, partition=True, partition_opening=True)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    arch_path = output_path(ctx, "architecture")
    mp = pytest.MonkeyPatch()
    mp.setenv("ROOM_MODEL_NO_MEASUREIT", "1")
    try:
        blender_stage.run(ctx)
    finally:
        mp.undo()
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    return ctx, arch, _probe(ctx, arch_path)


def _expected_text(dim):
    return f"{dim['value'] * 100:.1f} cm \u00b1 {dim['sigma'] * 100:.1f}".replace(".", ",")


def _spanned(arch):
    return [d for d in arch["dimensions"] if d["endpoints"] is not None]


@needs_blender
def test_dimensions_native(native_scene):
    _, arch, facts = native_scene
    spanned = _spanned(arch)
    assert len(spanned) >= 5
    assert all(facts["objects"][f"Dim_{d['id']}"]["type"] == "EMPTY" for d in spanned)
    for dim in spanned:
        text = facts["objects"][f"Dim_{dim['id']}_Text"]
        assert re.fullmatch(r"\d+,\d cm \u00b1 \d+,\d", text["text"]) and text["text"] == _expected_text(dim)
        assert text["parent"] == f"Dim_{dim['id']}" and text["collections"] == ["Cotes"]
        assert facts["objects"][f"Dim_{dim['id']}"]["collections"] == ["Cotes"]


@needs_blender
def test_dimensions_fallback(native_scene):
    ctx, arch, facts = native_scene
    assert facts["scene"]["dimensions_backend"] == "native"
    report = json.loads((ctx.out / "blender_report.json").read_text(encoding="utf-8"))
    assert report["dimensions_backend"] == "native" and report["dimensions"] == len(_spanned(arch))


@needs_blender
def test_measureit_failure_falls_back_to_native(scene):
    _, arch, facts = scene
    if not facts["scene"]["measureit_available"]:
        pytest.skip("MeasureIt is not installed")
    failure = facts["measureit_failure"]
    assert failure["backend"] == "native" and "measureit API changed" in failure["error"]
    assert failure["dims"] == sorted(f"Dim_{d['id']}" for d in _spanned(arch))


@needs_blender
def test_native_vertical_text_faces_the_room_and_horizontal_text_faces_up(native_scene):
    _, arch, facts = native_scene
    horizontal = next(d for d in _spanned(arch) if d["endpoints"][0][1] == d["endpoints"][1][1])
    vertical = next(d for d in _spanned(arch) if d["endpoints"][0][1] != d["endpoints"][1][1])
    up = facts["objects"][f"Dim_{horizontal['id']}_Text"]["normal"]
    side = facts["objects"][f"Dim_{vertical['id']}_Text"]["normal"]
    assert up[2] == pytest.approx(1.0, abs=1e-3) and abs(side[2]) < 1e-3


@needs_blender
def test_dimensions_measureit(scene):
    ctx, arch, facts = scene
    if not facts["scene"]["measureit_available"]:
        pytest.skip("MeasureIt is not installed")
    assert facts["scene"]["dimensions_backend"] == "measureit"
    for dim in _spanned(arch):
        obj = facts["objects"][f"Dim_{dim['id']}"]
        assert obj["collections"] == ["Cotes"] and obj["type"] == "MESH" and obj["verts"] == 2
        assert obj["measureit"] == [{"a": 0, "b": 1, "txt": _expected_text(dim), "dist": False, "names": True, "view": True}]
        lo, hi = (np.array(_blender_point(*p)) for p in dim["endpoints"])
        assert np.array(obj["bbox"][0]) == pytest.approx(np.minimum(lo, hi), abs=1e-4)
        assert np.array(obj["bbox"][1]) == pytest.approx(np.maximum(lo, hi), abs=1e-4)
    assert facts["scene"]["show_names"] is True and facts["scene"]["show_all"] is True
    report = json.loads((ctx.out / "blender_report.json").read_text(encoding="utf-8"))
    assert report["dimensions_backend"] == "measureit" and report["dimensions"] == len(_spanned(arch))


@needs_blender
def test_scene_objects(scene):
    _, arch, facts = scene
    shell_ids = {e["id"] for e in arch["elements"] if e["kind"] in ("floor", "ceiling", "wall", "partition", "beam", "column")}
    assert {"Floor", "Partition_P1"} <= shell_ids and any(i.startswith("Wall_M") for i in shell_ids)
    placed = set(facts["collections"]["Coque"]["objects"]) | set(facts["collections"]["À vérifier"]["objects"])
    assert placed == shell_ids
    for name in ("Coque", "Menuiseries", "Moulures", "Cotes", "À vérifier", "Scan"):
        assert name in facts["collections"]
    scan = facts["collections"]["Scan"]
    assert scan["objects"] and scan["hide_viewport"] and scan["hide_render"]
    assert not facts["collections"]["Coque"]["hide_viewport"]
    assert {"Mur", "Sol", "Plafond", "Menuiserie", "Verre", "Moulure"} <= set(facts["materials"])
    walls = [facts["objects"][e["id"]] for e in arch["elements"] if e["kind"] in ("wall", "partition")]
    assert all(o["modifiers"] == 0 and o["non_manifold"] == 0 and o["materials"] == ["Mur"] for o in walls)
    assert all(o["bare_faces"] == 0 for o in walls), "a boolean cut must not leave faces on an empty material slot"
    assert facts["objects"]["Floor"]["materials"] == ["Sol"] and facts["objects"]["Ceiling_1"]["materials"] == ["Plafond"]
    assert not any("utter" in name for name in facts["objects"]) and not facts["scene_objects"]
    orphans = [e["id"] for e in arch["elements"] if "host" in e and e["host"] is None]
    assert orphans and not set(orphans) & set(facts["objects"])


@needs_blender
@pytest.mark.parametrize("opening", ["Door_1", "Window_1", "Opening_1"])
def test_wall_cut(scene, opening):
    _, _, facts = scene
    rays = facts["rays"]
    assert not rays[f"{opening}-centre"]["hit"]
    assert rays[f"{opening}-beside"]["hit"]
    assert rays[f"{opening}-above"]["hit"]


@needs_blender
def test_window_keeps_the_wall_under_the_sill(scene):
    assert scene[2]["rays"]["Window_1-below"]["hit"]


@needs_blender
def test_scene_is_saved_with_ceilings_visible(scene):
    ctx, arch, facts = scene
    ceilings = [e["id"] for e in arch["elements"] if e["kind"] == "ceiling"]
    assert ceilings and not any(facts["objects"][c]["hide_render"] for c in ceilings)
    assert (ctx.out / "architecture.blend").stat().st_size > 0
    assert (ctx.out / "scan_decimated.ply").stat().st_size > 0


@needs_blender
def test_glb_written(scene):
    ctx, arch, _ = scene
    gltf = read_glb(ctx.out / "architecture.glb")
    names = {n["name"] for n in gltf["nodes"]}
    walls = {e["id"] for e in arch["elements"] if e["kind"] in ("wall", "partition")}
    assert walls <= names and {"Floor", "Ceiling_1"} <= names
    assert not any(n.startswith("Scan") for n in names)
    assert gltf["asset"]["version"] == "2.0"


@needs_blender
def test_renders(scene):
    ctx, _, _ = scene
    for name in ("top", "persp_1", "persp_2", "open"):
        image = Image.open(ctx.out / "renders" / f"{name}.png")
        assert image.size == (1280, 720)
        assert np.asarray(image.convert("L")).std() > 10, name


@needs_blender
def test_open_render_shows_the_leaves_swung(scene):
    ctx, _, _ = scene
    closed, opened = (np.asarray(Image.open(ctx.out / "renders" / f"{n}.png").convert("L"), int) for n in ("top", "open"))
    assert (np.abs(closed - opened) > 20).sum() > 300


def _plan_dir(vec):
    """Room-frame plan vector (x, z) as a Blender plan vector."""
    return np.array([vec[0], -vec[1]])


def _swing_direction(arch, el):
    """Blender plan direction the leaves swing to with "Inverser sens" unchecked: the interior normal, reversed by `swing: out`; on a partition,
    face B faces the other way."""
    host = _element(arch, el["host"])
    az = _value(arch, host["params"]["azimuth"])
    sign = (-1 if el.get("swing") == "out" else 1) * (-1 if host["kind"] == "partition" and el["face"] == "B" else 1)
    return _plan_dir(sign * np.array([math.cos(az), math.sin(az)]))


def _pose(facts, rig, name):
    return facts["rigs"][rig][name]


JOINERY = tuple(f"Door_{k}" for k in range(1, 9)) + tuple(f"Window_{k}" for k in range(1, 6))


@needs_blender
def test_joinery_objects(scene):
    _, arch, facts = scene
    objects = facts["objects"]
    for oid in JOINERY:
        root = objects[oid]
        assert root["type"] == "EMPTY" and root["collections"] == ["Menuiseries"]
        assert root["props"] == {"Ouverture": 0.0, "Inverser sens": False, "Inverser charni\u00e8re": False, "AngleMax": 90.0}, oid
        assert all(type(root["props"][k]) is bool for k in ("Inverser sens", "Inverser charni\u00e8re")), "checkboxes, not integers"
        frame = objects[f"{oid}_Frame"]
        assert frame["parent"] == oid and frame["collections"] == ["Menuiseries"] and frame["materials"] == ["Menuiserie"]
        leaves = [n for n in objects if n.startswith(f"{oid}_Leaf_")]
        assert leaves and all(objects[n]["parent"] == oid and objects[n]["collections"] == ["Menuiseries"] for n in leaves)
        glass = [n for n in objects if n.startswith(f"{oid}_Glass_")]
        glazed = oid.startswith("Window") or oid == "Door_5"
        assert len(glass) == (len(leaves) if glazed else 0), oid
        assert all(objects[g]["materials"] == ["Verre"] for g in glass)
    assert not [n for n in objects if n.startswith(("Opening_1_", "Opening_2_"))] and "Opening_1" not in objects
    assert {n for n in objects if n.endswith("_Frame")} == {f"{o}_Frame" for o in JOINERY}
    assert set(facts["collections"]["Menuiseries"]["objects"]) == {n for n in objects if n.startswith(tuple(JOINERY))}


@needs_blender
@pytest.mark.parametrize("oid", JOINERY)
def test_frame_sits_in_the_cut(scene, oid):
    _, arch, facts = scene
    el = _element(arch, oid)
    host, p, n = _on_face(arch, el)
    t = _value(arch, host["params"]["thickness"])
    w, h, sill = (_value(arch, el["params"][k]) for k in ("width", "height", "sill"))
    lo, hi = facts["objects"][f"{oid}_Frame"]["bbox"]
    centre = (np.array(lo) + np.array(hi)) / 2
    mid = p - n * t / 2
    expected = np.array(_blender_point(mid[0], _value(arch, "floor.level") + sill + h / 2, mid[1]))
    delta = centre - expected
    n_b = _plan_dir(n)
    assert abs(delta[:2] @ n_b) < 0.001, "centred in the host thickness"
    assert abs(delta[:2] @ np.array([-n_b[1], n_b[0]])) < 0.005 and abs(delta[2]) < 0.01
    assert abs(np.array(hi)[2] - np.array(lo)[2] - h) < 0.01
    assert (np.array(hi) - np.array(lo))[:2] @ np.abs(np.array([-n_b[1], n_b[0]])) == pytest.approx(w, abs=0.01)
    assert (np.array(hi) - np.array(lo))[:2] @ np.abs(n_b) <= t + 0.02


@needs_blender
def test_closed_door_leaf_blocks_the_cut(scene):
    assert scene[2]["rays"]["Door_1-leaf"]["hit"]


@needs_blender
@pytest.mark.parametrize("oid", ["Door_1", "Door_6", "Door_7", "Door_8", "Window_2"])
def test_hinge_side_and_swing_direction(scene, oid):
    _, arch, facts = scene
    el, leaf = _element(arch, oid), f"{oid}_Leaf_1"
    f = _swing_direction(arch, el)
    left = np.array([-f[1], f[0]])  # the viewer's left when facing the swing direction
    rig = facts["rigs"]
    closed, opened, flipped, mirrored = (rig[f"{kind}-{oid}"][leaf] for kind in ("closed", "open", "flipped", "mirrored"))
    centre = np.array(facts["objects"][oid]["location"])[:2]
    hinge, other = (np.array(pose["world_origin"])[:2] for pose in (closed, rig[f"closed-mirrored-{oid}"][leaf]))
    side = 1 if el.get("hinge", "left") == "left" else -1
    assert np.sign((hinge - centre) @ left) == side
    assert np.sign((other - centre) @ left) == -side, "Inverser charni\u00e8re moves the hinge to the other jamb"
    assert (np.array(opened["world_centre"])[:2] - hinge) @ f > 0.05, "opens towards the swing direction"
    assert (np.array(flipped["world_centre"])[:2] - hinge) @ f < -0.05, "Inverser sens flips it"
    assert (np.array(mirrored["world_centre"])[:2] - np.array(mirrored["world_origin"])[:2]) @ f > 0.05, "still opens towards the swing"


@needs_blender
def test_rig_hinged(scene):
    _, _, facts = scene
    leaf = "Door_1_Leaf_1"
    assert _pose(facts, "closed", leaf)["rotation_z_deg"] == pytest.approx(0, abs=0.01)
    assert _pose(facts, "open", leaf)["rotation_z_deg"] == pytest.approx(90, abs=0.5)
    assert _pose(facts, "open-flipped", leaf)["rotation_z_deg"] == pytest.approx(-90, abs=0.5)
    assert _pose(facts, "open-mirrored", leaf)["driven_rotation_z_deg"] == pytest.approx(-90, abs=0.5), "the sign follows the mirrored hinge"
    assert _pose(facts, "open-both", leaf)["driven_rotation_z_deg"] == pytest.approx(90, abs=0.5)
    assert _pose(facts, "half", leaf)["rotation_z_deg"] == pytest.approx(45, abs=0.5)
    assert _pose(facts, "short-angle", leaf)["rotation_z_deg"] == pytest.approx(60, abs=0.5)
    for key in ("open", "half"):  # the hinge stays where it is
        assert _pose(facts, key, leaf)["location"] == pytest.approx(_pose(facts, "closed", leaf)["location"], abs=1e-4)
    assert _pose(facts, "open-flipped", leaf)["location"] == pytest.approx(_pose(facts, "closed", leaf)["location"], abs=1e-4)
    closed, mirrored = _pose(facts, "closed", leaf)["location"], _pose(facts, "open-mirrored", leaf)["location"]
    assert mirrored[0] == pytest.approx(-closed[0], abs=1e-4) and mirrored[1:] == pytest.approx(closed[1:], abs=1e-4)
    assert facts["objects"][leaf]["drivers"] > 0 and facts["objects"]["Door_1"]["props"]["Ouverture"] == 0


@needs_blender
@pytest.mark.parametrize("oid", ["Door_1", "Window_2"])
def test_leaf_is_mirrored_with_the_hinge(scene, oid):
    _, _, facts = scene
    for name in (f"{oid}_Leaf_1", f"{oid}_Glass_1") if oid.startswith("Window") else (f"{oid}_Leaf_1",):
        lo, hi = facts["rigs"][f"closed-{oid}"][name]["extent_x"]
        mirrored = facts["rigs"][f"closed-mirrored-{oid}"][name]["extent_x"]
        assert mirrored == pytest.approx([-hi, -lo], abs=1e-4), name
        assert facts["objects"][name]["drivers_simple"], "drivers use the simple-expression subset"


@needs_blender
@pytest.mark.parametrize("oid,glass", [("Door_2", False), ("Door_5", True), ("Window_1", True)])
def test_rig_double(scene, oid, glass):
    _, _, facts = scene
    one, two = f"{oid}_Leaf_1", f"{oid}_Leaf_2"
    opened = facts["rigs"][f"open-{oid}"]
    assert opened[one]["rotation_z_deg"] == pytest.approx(90, abs=0.5)
    assert opened[two]["rotation_z_deg"] == pytest.approx(-90, abs=0.5)
    flipped = facts["rigs"][f"flipped-{oid}"]
    assert flipped[one]["rotation_z_deg"] == pytest.approx(-90, abs=0.5) and flipped[two]["rotation_z_deg"] == pytest.approx(90, abs=0.5)
    closed = facts["rigs"][f"closed-{oid}"]
    assert closed[one]["location"][0] < 0 < closed[two]["location"][0], "hinges at the two jambs"
    for name in (one, two):
        assert facts["rigs"][f"mirrored-{oid}"][name] == opened[name], "no hinge side to switch on two leaves"
    if glass:
        for k in (1, 2):  # the glass follows its leaf
            assert opened[f"{oid}_Glass_{k}"]["rotation_z_deg"] == pytest.approx(opened[f"{oid}_Leaf_{k}"]["rotation_z_deg"], abs=0.5)


@needs_blender
@pytest.mark.parametrize("oid,rotation", [("Window_2", -90), ("Window_5", 90), ("Door_6", -90)])
def test_rig_single_and_tilt_turn(scene, oid, rotation):
    _, _, facts = scene
    assert facts["rigs"][f"open-{oid}"][f"{oid}_Leaf_1"]["rotation_z_deg"] == pytest.approx(rotation, abs=0.5)


def _width(facts, leaf):
    return facts["objects"][leaf]["dims"][0]


@needs_blender
@pytest.mark.parametrize("oid,leaf,direction", [("Door_3", "Door_3_Leaf_1", 1), ("Door_4", "Door_4_Leaf_1", -1)])
def test_rig_sliding(scene, oid, leaf, direction):
    _, _, facts = scene
    closed, opened = facts["rigs"][f"closed-{oid}"][leaf], facts["rigs"][f"open-{oid}"][leaf]
    mirrored = facts["rigs"][f"mirrored-{oid}"][leaf]
    travel = np.array(opened["location"]) - np.array(closed["location"])
    assert travel[0] == pytest.approx(direction * _width(facts, leaf), abs=0.002)
    assert travel[1:] == pytest.approx([0, 0], abs=1e-4) and opened["rotation_z_deg"] == pytest.approx(0, abs=0.01)
    assert mirrored["location"][0] == pytest.approx(-closed["location"][0] - direction * _width(facts, leaf), abs=0.002), "slides to the other side"
    assert mirrored["location"][1] == pytest.approx(closed["location"][1], abs=1e-4)
    flipped = facts["rigs"][f"flipped-{oid}"][leaf]
    assert flipped["location"] == pytest.approx(opened["location"], abs=1e-4), "a sliding leaf has no swing to invert"
    assert flipped["extent_x"] == pytest.approx(opened["extent_x"], abs=1e-4)


@needs_blender
def test_pocket_door_slides_inside_the_wall(scene):
    _, arch, facts = scene
    el = _element(arch, "Door_4")
    wall_t = _value(arch, _element(arch, el["host"])["params"]["thickness"])
    for rig in ("closed-Door_4", "open-Door_4"):
        assert abs(facts["rigs"][rig]["Door_4_Leaf_1"]["location"][1]) < wall_t / 2 - 0.03


@needs_blender
def test_rig_sliding_window_moves_one_leaf(scene):
    _, _, facts = scene
    closed, opened = facts["rigs"]["closed-Window_3"], facts["rigs"]["open-Window_3"]
    moved = {k: opened[k]["location"][0] - closed[k]["location"][0] for k in ("Window_3_Leaf_1", "Window_3_Leaf_2")}
    still = [k for k, v in moved.items() if abs(v) < 1e-4]
    assert len(still) == 1
    mover = next(k for k in moved if k not in still)
    assert abs(moved[mover]) == pytest.approx(_width(facts, mover), abs=0.002)
    assert opened["Window_3_Glass_1"]["location"][1] != opened["Window_3_Glass_2"]["location"][1], "two rails"
    mirrored = facts["rigs"]["mirrored-Window_3"]
    for k in moved:
        assert mirrored[k]["location"][0] == pytest.approx(-opened[k]["location"][0], abs=1e-4), "both panes swap sides"


@needs_blender
def test_fixed_window_does_not_move(scene):
    _, _, facts = scene
    closed, opened = facts["rigs"]["closed-Window_4"], facts["rigs"]["open-Window_4"]
    assert closed == opened and "Window_4_Glass_1" in opened
    assert facts["objects"]["Window_4_Leaf_1"]["drivers"] == 0


@needs_blender
def test_panels_are_evenly_split_and_handle_optional(scene):
    _, _, facts = scene
    leaf = facts["objects"]["Door_6_Leaf_1"]  # three panels, no handle
    zs = np.array(leaf["zs"])
    assert zs[0] == pytest.approx(0, abs=1e-3)
    inner = zs[1:-1]
    assert len(inner) == 6
    heights, gaps = inner[1::2] - inner[0::2], inner[2::2] - inner[1:-1:2]
    assert np.ptp(heights) < 0.001 and np.ptp(gaps) < 0.001 and heights[0] > gaps[0]
    assert leaf["dims"][1] > facts["objects"]["Door_1_Leaf_1"]["dims"][1] - 0.2, "raised on both faces"
    plain = facts["objects"]["Door_1_Leaf_1"]
    assert plain["verts"] == 8 + 2 * 8, "a plain leaf has one box and two handle boxes"
    assert leaf["verts"] == 8 + 2 * 3 * 8, "no handle, three raised panels on each face"


@needs_blender
def test_static_glb_closed(scene):
    ctx, _, facts = scene
    raw = (ctx.out / "architecture.glb").read_bytes()
    assert b"Ouverture" not in raw and b"Inverser" not in raw
    gltf = read_glb(ctx.out / "architecture.glb")
    nodes = {n["name"]: n for n in gltf["nodes"]}
    assert {"Door_1", "Door_1_Frame", "Door_1_Leaf_1", "Window_1_Glass_2"} <= set(nodes)
    leaves = [n for name, n in nodes.items() if "_Leaf_" in name]
    assert len(leaves) == sum(1 for name in facts["objects"] if "_Leaf_" in name)
    for name, node in nodes.items():
        if "_Leaf_" in name or "_Glass_" in name:
            assert node.get("rotation", [0, 0, 0, 1]) == pytest.approx([0, 0, 0, 1], abs=1e-6), name
            assert node.get("scale", [1, 1, 1]) == pytest.approx([1, 1, 1], abs=1e-6), name
    x, y, z = facts["rigs"]["closed-Door_3"]["Door_3_Leaf_1"]["location"]
    assert nodes["Door_3_Leaf_1"].get("translation", [0, 0, 0]) == pytest.approx([x, z, -y], abs=1e-4)


def _footprint_input(arch, wall):
    az = _value(arch, wall["params"]["azimuth"])
    return {"start": wall["start"], "end": wall["end"], "normal": [math.cos(az), math.sin(az)],
            "offset": _value(arch, wall["params"]["offset"]), "thickness": _value(arch, wall["params"]["thickness"])}


@needs_blender
@pytest.mark.parametrize("fixture", ["scene", "joined", "junctions"])
def test_footprints_match_the_test_side_twin(request, fixture):
    arch = request.getfixturevalue(fixture)[1]
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    twin = meshes.mitred_wall_footprints([_footprint_input(arch, w) for w in walls])
    own = wall_plan.wall_footprints(arch)
    for wall, expected in zip(walls, twin):
        assert np.array(own[wall["id"]]) == pytest.approx(expected, abs=1e-9), wall["id"]


@needs_blender
def test_wall_matches_the_mitred_footprint(scene):
    _, arch, facts = scene
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    footprints = meshes.mitred_wall_footprints([_footprint_input(arch, w) for w in walls])
    for index in (2, 3):  # a convex corner on one side, a reflex one on the other
        vertices, _ = meshes.prism(footprints[index], _value(arch, "floor.level"), _value(arch, walls[index]["params"]["top"]))
        mapped = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
        lo, hi = facts["objects"][walls[index]["id"]]["bbox"]
        assert np.array(lo) == pytest.approx(mapped.min(0), abs=0.001)
        assert np.array(hi) == pytest.approx(mapped.max(0), abs=0.001)


@needs_blender
def test_build_passes_ctx_and_rejects_unlinked_hook_objects(scene):
    _, _, facts = scene
    hook = facts["hook"]
    assert hook["linked"]["raised"] is None
    assert hook["seen"]["ctx"] == {"textures": None, "scan_path": None, "out_dir": "/probe/out"}
    assert "Menuiseries" in hook["seen"]["collections"]
    assert "zz_probe" in hook["unlinked"]["raised"] and "ProbeLoose" in hook["unlinked"]["raised"]


JOIN_VARIANTS = [  # a door close to a near-parallel corner, to a right-angle corner and to a reflex corner
    ("Door_1", "Door_9", "Wall_M7", 0.18, 0.3, {}),
    ("Door_1", "Door_10", "Wall_M2", 0.2, 0.3, {}),
    ("Door_1", "Door_11", "Wall_M4", 0.18, 0.3, {}),
    ("Door_1", "Door_12", "Wall_M1", 1.05, 0.8, {}),  # where Partition_P1 ends on the wall
]
SPLIT_AT, SPLIT_TURN_DEG, SPLIT_JOG = 2.5, -3.0, 0.06  # Wall_M1 splits into itself and Wall_M7, a near-parallel step
PARTITION_OVERSHOOT = 0.3  # m the partition is drawn into Wall_M1


def _face_z(arch, wall, x):
    """z of the face line of a wall (not parallel to the z axis) at plan abscissa x."""
    az = _value(arch, wall["params"]["azimuth"])
    return (_value(arch, wall["params"]["offset"]) - math.cos(az) * x) / math.sin(az)


def _split_wall(arch, wall_id, new_id, at, turn_deg, jog):
    """Splits a wall `at` m from its start; the second part turns by `turn_deg` and its face sits `jog` m behind the first."""
    wall = _element(arch, wall_id)
    start, end = np.array(wall["start"]), np.array(wall["end"])
    cut = start + (end - start) / np.linalg.norm(end - start) * at
    new = json.loads(json.dumps(wall))
    new.update(id=new_id, start=list(cut), end=list(end))
    wall["end"] = list(cut)
    key = new_id.split("_")[1]
    for name, pid in new["params"].items():
        new["params"][name] = f"wall.{key}.{name}"
        arch["parameters"][new["params"][name]] = dict(arch["parameters"][pid])
    az = _value(arch, wall["params"]["azimuth"]) + math.radians(turn_deg)
    arch["parameters"][new["params"]["azimuth"]]["value"] = az
    arch["parameters"][new["params"]["offset"]]["value"] = float(np.array([math.cos(az), math.sin(az)]) @ cut - jog)
    arch["elements"].insert(arch["elements"].index(wall) + 1, new)


def _join_arch(arch):
    """The synthetic room with a near-parallel step in the south wall and a partition drawn through it."""
    _split_wall(arch, "Wall_M1", "Wall_M7", SPLIT_AT, SPLIT_TURN_DEG, SPLIT_JOG)
    wall = _element(arch, "Wall_M1")
    window = _element(arch, "Window_1")  # it stays in the same place, now on the second part
    window["host"] = "Wall_M7"
    arch["parameters"][window["params"]["center_s"]]["value"] -= SPLIT_AT
    partition = _element(arch, "Partition_P1")
    partition["start"][1] = _face_z(arch, wall, partition["start"][0]) - PARTITION_OVERSHOOT
    _add_variants(arch, JOIN_VARIANTS)
    for pid in ("cornice.height", "cornice.projection", "skirting.height", "skirting.projection"):
        arch["parameters"][pid] = {"value": MOULDING_PARAMS[pid][0], "sigma": 0.0, "unit": "m", "source": "llm"}
    arch["elements"] += [json.loads(json.dumps(_element_of(MOULDING_ELEMENTS, i))) for i in ("Cornice", "Skirting")]


def _element_of(elements, id_):
    return next(e for e in elements if e["id"] == id_)


def _partition_x(arch):
    """x of the middle of Partition_P1 (it runs along z)."""
    partition = _element(arch, "Partition_P1")
    az = _value(arch, partition["params"]["azimuth"])
    n = np.array([math.cos(az), math.sin(az)])
    face = np.array(partition["start"]) + n * (_value(arch, partition["params"]["offset"]) - n @ np.array(partition["start"]))
    return float((face - n * _value(arch, partition["params"]["thickness"]) / 2)[0])


def _shell_ids(arch):
    return [e["id"] for e in arch["elements"] if e["kind"] in ("wall", "partition")]


@pytest.fixture(scope="module")
def joined(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("joined"), "joined", window=True, partition=True, partition_opening=True,
                       partition_opening_width=0.6)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    arch_path = output_path(ctx, "architecture")
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    _join_arch(arch)
    arch_path.write_text(json.dumps(arch), encoding="utf-8")
    blender_stage.run(ctx)
    rays = []
    for oid in ("Door_9", "Door_10"):
        rays += [_ray(arch, oid, f"{oid}-{k}", along, 1.0, target=_shell_ids(arch)) for k, along in enumerate((-0.12, 0.0, 0.12))]
    rays.append(_ray(arch, "Door_11", "Door_11-neighbour", -0.08, 1.0, target="Wall_M3"))
    for ray in rays:
        ray["distance"] = 1.0
    face_y = -_face_z(arch, _element(arch, "Wall_M1"), _element(arch, "Partition_P1")["start"][0])
    rays.append({"name": "partition-end", "object": "Partition_P1", "origin": [_partition_x(arch), face_y + 0.1, 1.0],
                 "direction": [0.0, -1.0, 0.0]})
    rays += _cornice_rays(arch)
    rays += _face_rays(arch, "skirting", "Skirting", 0.01, lambda host: _value(arch, "floor.level") + _value(arch, "skirting.height"))
    return ctx, arch, _probe(ctx, arch_path, rays), {r["name"]: r for r in rays}


@needs_blender
@pytest.mark.parametrize("fixture", ["scene", "joined"])
def test_walls_do_not_overlap(request, fixture):
    facts = request.getfixturevalue(fixture)[2]
    assert facts["overlaps"] == {}


@needs_blender
@pytest.mark.parametrize("door", ["Door_9", "Door_10"])
def test_door_near_corner_is_clear(joined, door):
    _, _, facts, _ = joined
    for k in range(3):
        assert not facts["rays"][f"{door}-{k}"]["hit"], f"{door} ray {k} hits {facts['rays'][f'{door}-{k}']['object']}"


@needs_blender
def test_cutter_cuts_the_neighbouring_wall(joined):
    _, arch, facts, rays = joined
    hit = facts["rays"]["Door_11-neighbour"]
    assert hit["hit"], "Wall_M3 is still there behind the doorway"
    distance = np.linalg.norm(np.array(hit["location"]) - np.array(rays["Door_11-neighbour"]["origin"]))
    assert distance == pytest.approx(0.6 + _value(arch, "wall.M4.thickness") + 0.01, abs=0.005)


@needs_blender
def test_partition_trimmed_at_wall(joined):
    _, arch, facts, _ = joined
    partition = _element(arch, "Partition_P1")
    face_z = _face_z(arch, _element(arch, "Wall_M1"), partition["start"][0])
    lo, hi = facts["objects"]["Partition_P1"]["bbox"]
    assert hi[1] == pytest.approx(-face_z, abs=0.002), "ends on the face of the wall, not inside it"
    assert lo[1] == pytest.approx(-partition["end"][1], abs=0.005), "the free end stays where it is"


@needs_blender
def test_opening_does_not_notch_a_partition_that_abuts_it(joined):
    _, arch, facts, _ = joined
    face_z = _face_z(arch, _element(arch, "Wall_M1"), _element(arch, "Partition_P1")["start"][0])
    hit = facts["rays"]["partition-end"]
    assert hit["hit"] and hit["location"][1] == pytest.approx(-face_z, abs=0.002), "the cutter overshoot stays out of the partition"


@needs_blender
def test_mouldings_follow_the_step_between_near_parallel_walls(joined):
    _, arch, facts, rays = joined
    level = {"cornice": lambda host: _value(arch, host["params"]["top"]),
             "skirting": lambda host: _value(arch, "floor.level") + _value(arch, "skirting.height")}
    names = [n for n in rays if n.startswith("cornice-") and not n.startswith("cornice-Partition_P1-") or n.endswith(("-mid", "-end15", "-end3")) and n.startswith("cornice-Partition_P1-")] + [n for n in rays if n.startswith("skirting-Wall_M1-") and n.endswith(("end3", "end15"))]
    assert any(n.endswith("Wall_M1-1-end3") for n in names)
    for name in names:
        hit = facts["rays"][name]
        assert hit["hit"], f"{name}: no moulding"
        assert hit["location"][2] == pytest.approx(level[name.split("-")[0]](_element(arch, name.split("-")[1])), abs=0.001), name


def _junction_arch(arch):
    """The L-shaped room with Partition_P1 carried on from the step wall (3.5 cm off its face line) to a west wall that
    has a near-parallel jog across the partition, and Partition_P2 ending 15 cm short of Partition_P1."""
    step = _element(arch, "Wall_M3")
    az = _value(arch, step["params"]["azimuth"])
    n = np.array([math.cos(az), math.sin(az)])
    t = 0.2
    east, west = np.array(step["end"]) + [0.14, 0.0], np.array([-2.98, step["end"][1]])
    partition = _element(arch, "Partition_P1")
    partition.update(start=list(east + n * 0.035), end=list(west + n * 0.035))
    for name, value in (("azimuth", az), ("offset", float(n @ (east + n * 0.035))), ("thickness", t)):
        arch["parameters"][partition["params"][name]]["value"] = value
    band_z = float(east[1] + n[1] * 0.035 - t / 2 * n[1])
    wall = _element(arch, "Wall_M6")
    at = math.dist(wall["start"], [wall["start"][0], band_z]) / 1.0
    _split_wall(arch, "Wall_M6", "Wall_M7", at, -0.5, 0.11)
    second = json.loads(json.dumps(partition))
    second.update(id="Partition_P2", start=[-1.5, -1.0], end=[-1.5, float(partition["start"][1]) - 0.15 * 1.0 * np.sign(n[1])])
    second["params"] = {k: f"partition.P2.{k}" for k in partition["params"]}
    for k, pid in partition["params"].items():
        arch["parameters"][second["params"][k]] = dict(arch["parameters"][pid])
    arch["parameters"][second["params"]["azimuth"]]["value"] = 0.0
    arch["parameters"][second["params"]["offset"]]["value"] = -1.5 + 0.05
    arch["parameters"][second["params"]["thickness"]]["value"] = 0.1
    arch["elements"].append(second)
    arch["elements"] = [e for e in arch["elements"] if e.get("host") != "Partition_P1"]


def _span_segments(arch):
    """Segments along the partition lines, from inside the partition out through each end and past the wall behind it."""
    out = []
    for name, x0, x1, zs in (("west", -2.2, -3.7, (0.24, 0.28, 0.36, 0.40)), ("east", -1.5, 0.6, (0.28, 0.33, 0.40))):
        for z in zs:
            out.append({"name": f"{name}-{z}", "p0": _blender_point(x0, 1.0, z), "p1": _blender_point(x1, 1.0, z)})
    out.append({"name": "face-east", "p0": _blender_point(-0.55, 1.0, 0.15), "p1": _blender_point(-0.55, 1.0, 0.5)})
    return out


@pytest.fixture(scope="module")
def junctions(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("junctions"), "junctions", window=True, partition=True)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    arch_path = output_path(ctx, "architecture")
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    _junction_arch(arch)
    arch_path.write_text(json.dumps(arch), encoding="utf-8")
    blender_stage.run(ctx)
    return ctx, arch, _probe(ctx, arch_path, spans=_span_segments(arch))


@needs_blender
def test_partitions_fit_jogs_and_reflex_corners_without_a_slot(junctions):
    facts = junctions[2]
    for name, runs in ((n, r) for n, r in facts["spans"].items() if n != "face-east"):
        kinds = [k for k, _ in runs]
        assert runs[0][0] == "solid", f"{name} starts inside the partition"
        assert "air" not in kinds[:-1] or kinds.index("air") == len(kinds) - 1 or "solid" not in kinds[kinds.index("air"):], \
            f"{name}: air between the partition and the wall behind it: {runs}"


@needs_blender
def test_partition_collinear_with_a_wall_comes_flush_with_it(junctions):
    runs = junctions[2]["spans"]["face-east"]
    assert runs[0][0] == "air" and runs[0][1] >= 0.09, "its 3.5 cm of protrusion has faded out at the corner"


@needs_blender
def test_junction_walls_do_not_overlap(junctions):
    assert junctions[2]["overlaps"] == {}


MOULDING_PARAMS = {  # id: (value, unit)
    "niche.1.center_s": (0.3, "m"), "niche.1.sill": (0.8, "m"), "niche.1.width": (0.4, "m"), "niche.1.height": (0.8, "m"),
    "niche.1.depth": (0.05, "m"),  # ignored: the niche is through
    "niche.2.center_s": (1.5, "m"), "niche.2.sill": (0.9, "m"), "niche.2.width": (0.5, "m"), "niche.2.height": (0.6, "m"),
    "niche.2.depth": (0.1, "m"),
    "cornice.height": (0.08, "m"), "cornice.projection": (0.1, "m"), "skirting.height": (0.1, "m"),
    "skirting.projection": (0.02, "m"),
}
NICHE_PARAMS = ("center_s", "sill", "width", "height", "depth")
MOULDING_ELEMENTS = [
    {"id": "Niche_1", "kind": "niche", "host": "Partition_P1", "face": "A", "through": True, "dividers": 2, "shelves": [],
     "frame": {"profile": "flat", "width": 0.06}, "flags": []},
    {"id": "Niche_2", "kind": "niche", "host": "Wall_M3", "face": None, "through": False, "dividers": 0, "shelves": [-0.5, 0.3],
     "frame": None, "flags": []},
    {"id": "Cornice", "kind": "cornice", "profile": "cove", "params": {"height": "cornice.height", "projection": "cornice.projection"}},
    {"id": "Skirting", "kind": "skirting", "profile": "baroque",
     "params": {"height": "skirting.height", "projection": "skirting.projection"}},
    {"id": "PanelFrame_1", "kind": "panel_frame", "host": "Wall_M3", "face": None, "rect": [2.5, 0.9, 3.3, 1.9],
     "profile": "quarter_round", "flags": ["to_verify"]},
    {"id": "PanelFrame_2", "kind": "panel_frame", "host": "Wall_M3", "face": None, "rect": [0.5, 0.9, 1.3, 1.9],
     "profile": "flat", "flags": ["to_verify"]},
]
ARCHITRAVES = {"Door_1": {"profile": "moulded", "width": 0.07}, "Opening_1": {"profile": "flat", "width": 0.05}}
SKIRTING_HEIGHT, NICHE_1_COMPARTMENTS = 0.1, 3


def _add_mouldings(arch):
    for pid, (value, unit) in MOULDING_PARAMS.items():
        arch["parameters"][pid] = {"value": value, "sigma": 0.0, "unit": unit, "source": "llm"}
    for element in json.loads(json.dumps(MOULDING_ELEMENTS)):
        if element["kind"] == "niche":
            n = element["id"].split("_")[1]
            element["params"] = {name: f"niche.{n}.{name}" for name in NICHE_PARAMS}
        arch["elements"].append(element)
    for oid, spec in ARCHITRAVES.items():
        _element(arch, oid)["frame"] = spec


STOP_SHORT = 0.4  # m the floor and the ceilings stop short of the walls, as on a real scan
CORNICE_REACH = 0.05  # m from the wall face where the cornice rays come down


def _stop_short(arch):
    """Insets the floor and every ceiling region by STOP_SHORT, so no mesh outline lies on a wall face."""
    inset = lambda ring, holes=(): [list(p) for p in shapely.Polygon(ring, holes).buffer(-STOP_SHORT).exterior.coords][:-1]  # noqa: E731
    floor = _element(arch, "Floor")
    floor["polygon"] = inset(floor["polygon"])
    for ceiling in (e for e in arch["elements"] if e["kind"] == "ceiling"):
        for region in ceiling["region"]:
            region["exterior"], region["holes"] = inset(region["exterior"], region["holes"]), []


def _face_rays(arch, prefix, target, reach, level):
    """Rays coming down on each face of every wall and partition (`level(host)` is the height they must hit `target` at):
    at the middle and 15 cm from the ends, and 3 cm from an end that no taller or shorter wall meets (the moulding must
    reach the corner there)."""
    rays = []
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    for host in (e for e in arch["elements"] if e["kind"] in ("wall", "partition")):
        az = _value(arch, host["params"]["azimuth"])
        n, t = np.array([math.cos(az), math.sin(az)]), _value(arch, host["params"]["thickness"])
        start, end = np.array(host["start"]), np.array(host["end"])
        length = np.linalg.norm(end - start)
        top = _value(arch, host["params"]["top"])
        positions = {"mid": length / 2, "start15": 0.15, "end15": length - 0.15}
        for key, other, name in (("start", "end", "start3"), ("end", "start", "end3")):
            neighbour = next((w for w in walls if w is not host and np.allclose(w[other], host[key], atol=1e-6)), None)
            if neighbour is None or abs(_value(arch, neighbour["params"]["top"]) - top) < 0.01:
                positions[name] = 0.03 if key == "start" else length - 0.03
        for side in (1, -1) if host["kind"] == "partition" else (1,):
            for label, s in positions.items():
                p = start + (end - start) / length * s
                p = p + n * (_value(arch, host["params"]["offset"]) - n @ p) - (n * t if side < 0 else 0)
                origin = p + side * n * reach
                rays.append({"name": f"{prefix}-{host['id']}-{side}-{label}", "object": target,
                             "origin": _blender_point(origin[0], level(host) + 1.0, origin[1]), "direction": [0.0, 0.0, -1.0]})
    return rays


def _cornice_rays(arch):
    return _face_rays(arch, "cornice", "Cornice", CORNICE_REACH, lambda host: _value(arch, host["params"]["top"]))


def _across(ray, name, distance):
    """The same ray started `distance` m further along, travelling back (the view from the other face of a host)."""
    direction = np.array(ray["direction"])
    return {**ray, "name": name, "origin": list(np.array(ray["origin"]) + direction * distance), "direction": list(-direction)}


def _corner_ray(arch):
    """A ray down the diagonal into the corner of Wall_M1 and Wall_M2, 2 mm off the tip of the skirting mitre."""
    lines = []
    for wall in (_element(arch, "Wall_M1"), _element(arch, "Wall_M2")):
        az = _value(arch, wall["params"]["azimuth"])
        lines.append((np.array([math.cos(az), math.sin(az)]), _value(arch, wall["params"]["offset"])))
    (n1, o1), (n2, o2) = lines
    corner = np.linalg.solve(np.array([n1, n2]), np.array([o1, o2]))
    inward = (n1 + n2) / np.linalg.norm(n1 + n2)
    origin = corner + (n1 + n2) * 0.5 + (n1 - n2) * 0.002 / np.sqrt(2)
    floor = _value(arch, "floor.level")
    return {"name": "skirt-corner", "object": "Skirting", "origin": _blender_point(origin[0], floor + 0.05, origin[1]),
            "direction": _blender_point(-inward[0], 0.0, -inward[1])}


@pytest.fixture(scope="module")
def moulded(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("moulded"), "moulded", window=True, partition=True, partition_opening=True,
                       partition_opening_width=0.6)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    arch_path = output_path(ctx, "architecture")
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    _add_mouldings(arch)
    _stop_short(arch)
    arch_path.write_text(json.dumps(arch), encoding="utf-8")
    blender_stage.run(ctx)

    niche = _element(arch, "Niche_1")
    width, thickness = _value(arch, niche["params"]["width"]), _value(arch, "partition.P1.thickness")
    boards = ["Partition_P1", "Niche_1_Divider_1", "Niche_1_Divider_2"]
    mid = _value(arch, niche["params"]["sill"]) + 0.4
    rays = []
    for k in range(NICHE_1_COMPARTMENTS):
        rays.append(_ray(arch, "Niche_1", f"compartment-{k}", -width / 2 + width * (2 * k + 1) / 6, mid, target=boards))
    for k in (1, 2):
        rays.append(_ray(arch, "Niche_1", f"divider-{k}", -width / 2 + width * k / 3, mid, target=boards))
    rays += [_ray(arch, "Niche_1", "beside-low", 0.0, 0.3, target="Partition_P1"),
             _ray(arch, "Niche_1", "beside", width / 2 + 0.1, mid, target="Partition_P1"),
             _ray(arch, "Niche_1", "above", 0.0, mid + 0.6, target="Partition_P1"),
             _ray(arch, "Niche_1", "centre", 0.0, mid, target="Partition_P1")]
    rays.append(_across(rays[-1], "centre-back", 1.2 + thickness))
    sill = _value(arch, _element(arch, "Niche_2")["params"]["sill"])
    rays += [_ray(arch, "Niche_2", "recess-back", 0.0, sill + 0.15, target="Wall_M3"),
             _ray(arch, "Niche_2", "recess-above", 0.0, sill + 0.7, target="Wall_M3"),
             _ray(arch, "Niche_2", "shelf", 0.0, sill + 0.3, target="Niche_2_Shelf_1")]
    low = SKIRTING_HEIGHT / 2
    for oid in ("Door_1", "Window_1", "Opening_1"):  # the rays stop short of the next wall
        w = _value(arch, _element(arch, oid)["params"]["width"])
        rays += [_ray(arch, oid, f"skirt-{oid}-centre", 0.0, low, target="Skirting"),
                 _ray(arch, oid, f"skirt-{oid}-beside", w / 2 + 0.2, low, target="Skirting"),
                 _ray(arch, oid, f"skirt-{oid}-beside-minus", -(w / 2 + 0.2), low, target="Skirting")]
    for name in ("centre", "beside"):
        rays.append(_across(next(r for r in rays if r["name"] == f"skirt-Opening_1-{name}"), f"skirt-Opening_1-{name}-back",
                            1.2 + thickness))
    for ray in rays:
        if ray["name"].startswith("skirt-") and "corner" not in ray["name"]:
            ray["distance"] = 0.65
    rays.append(_corner_ray(arch))
    rays += _cornice_rays(arch)
    return ctx, arch, _probe(ctx, arch_path, rays), {r["name"]: r for r in rays}


@needs_blender
def test_niche_through(moulded):
    _, arch, facts, _ = moulded
    rays = facts["rays"]
    for k in range(NICHE_1_COMPARTMENTS):
        assert not rays[f"compartment-{k}"]["hit"], "light passes through every compartment"
    assert [rays[f"divider-{k}"]["object"] for k in (1, 2)] == ["Niche_1_Divider_1", "Niche_1_Divider_2"]
    assert not rays["centre"]["hit"] and not rays["centre-back"]["hit"], "open on both faces"
    assert all(rays[n]["hit"] for n in ("beside", "beside-low", "above")), "the partition is still there around it"
    assert sorted(n for n in facts["objects"] if n.startswith("Niche_1_")) == [
        "Niche_1_Architrave_A", "Niche_1_Architrave_B", "Niche_1_Divider_1", "Niche_1_Divider_2"]
    for n in (1, 2):
        board = facts["objects"][f"Niche_1_Divider_{n}"]
        assert board["collections"] == ["Menuiseries"] and board["materials"] == ["Menuiserie"]
        assert board["dims"] == pytest.approx([0.02, _value(arch, "partition.P1.thickness"), 0.8], abs=1e-3)


@needs_blender
def test_niche_recess_has_a_back(moulded):
    _, arch, facts, rays = moulded
    for name, expected in (("recess-back", 0.6 + 0.1), ("recess-above", 0.6)):
        assert facts["rays"][name]["hit"], name
        assert np.linalg.norm(np.array(facts["rays"][name]["location"]) - np.array(rays[name]["origin"])) == pytest.approx(expected, abs=0.002)
    assert facts["rays"]["shelf"]["hit"]
    assert [n for n in facts["objects"] if n.startswith("Niche_2_Shelf")] == ["Niche_2_Shelf_1"], "numbered after filtering"
    wall = facts["objects"]["Wall_M3"]
    assert wall["modifiers"] == 0 and wall["non_manifold"] == 0
    assert not any("utter" in name for name in facts["objects"])


@needs_blender
def test_skirting_gaps(moulded):
    _, _, facts, _ = moulded
    rays = facts["rays"]
    for oid in ("Door_1", "Opening_1"):
        assert not rays[f"skirt-{oid}-centre"]["hit"], oid
        assert rays[f"skirt-{oid}-beside"]["hit"] and rays[f"skirt-{oid}-beside-minus"]["hit"], oid
    assert rays["skirt-Window_1-centre"]["hit"], "a window sill does not interrupt the skirting"
    assert not rays["skirt-Opening_1-centre-back"]["hit"] and rays["skirt-Opening_1-beside-back"]["hit"], "both faces of a partition"


@needs_blender
def test_skirting_corner_is_mitred(moulded):
    _, arch, facts, rays = moulded
    hit = facts["rays"]["skirt-corner"]
    assert hit["hit"]
    distance = np.linalg.norm(np.array(hit["location"]) - np.array(rays["skirt-corner"]["origin"]))
    assert distance == pytest.approx(math.sqrt(2) * (0.5 - _value(arch, "skirting.projection")), abs=0.0045)


@needs_blender
def test_cornice_height(moulded):
    _, arch, facts, _ = moulded
    cornice = facts["objects"]["Cornice"]
    levels = [_value(arch, e["params"]["level"]) for e in arch["elements"] if e["kind"] == "ceiling"]
    height = _value(arch, "cornice.height")
    assert cornice["collections"] == ["Moulures"] and cornice["materials"] == ["Moulure"]
    assert cornice["bbox"][1][2] == pytest.approx(max(levels), abs=0.001)
    for level in levels:
        zs = np.array(cornice["zs"])
        assert np.abs(zs - level).min() < 0.001 and np.abs(zs - (level - height)).min() < 0.001, level
    assert cornice["bbox"][0][2] == pytest.approx(min(levels) - height, abs=0.001)
    assert cornice["flags"] is None


@needs_blender
def test_cornice_follows_walls(moulded):
    _, arch, facts, rays = moulded
    walls = [e for e in arch["elements"] if e["kind"] in ("wall", "partition")]
    assert len({_value(arch, e["params"]["top"]) for e in walls}) > 1, "two ceiling levels"
    for name in (n for n in rays if n.startswith("cornice-")):
        host = _element(arch, name.split("-")[1])
        hit = facts["rays"][name]
        assert hit["hit"], f"{name}: the cornice stops short of the wall"
        assert hit["location"][2] == pytest.approx(_value(arch, host["params"]["top"]), abs=0.001), name


@needs_blender
def test_unknown_profile(moulded):
    _, _, facts, _ = moulded
    skirting = facts["objects"]["Skirting"]
    assert skirting["flags"] == "unknown_profile" and skirting["verts"] > 0
    assert skirting["collections"] == ["Moulures"] and skirting["materials"] == ["Moulure"]


@needs_blender
def test_architraves_frame_the_openings(moulded):
    _, arch, facts, _ = moulded
    objects = facts["objects"]
    assert "Window_1_Architrave" not in objects, "no frame, no architrave"
    for oid, spec in ARCHITRAVES.items():
        el = _element(arch, oid)
        host, p, n = _on_face(arch, el)
        w, h = (_value(arch, el["params"][k]) for k in ("width", "height"))
        names = [f"{oid}_Architrave_A", f"{oid}_Architrave_B"] if host["kind"] == "partition" else [f"{oid}_Architrave"]
        assert [k for k in objects if k.startswith(f"{oid}_Architrave")] == names
        floor = _value(arch, "floor.level") + _value(arch, el["params"]["sill"])
        n_b = _plan_dir(n)
        for name in names:
            obj = objects[name]
            lo, hi = np.array(obj["bbox"])
            assert obj["collections"] == ["Moulures"] and obj["materials"] == ["Moulure"]
            assert hi[2] - floor == pytest.approx(h + spec["width"], abs=0.003) and lo[2] == pytest.approx(floor, abs=0.003)
            assert (hi - lo)[:2] @ np.abs(np.array([-n_b[1], n_b[0]])) == pytest.approx(w + 2 * spec["width"], abs=0.012)
    t = _value(arch, "partition.P1.thickness")
    a, b = (np.array(objects[f"Opening_1_Architrave_{f}"]["bbox"]).mean(0) for f in "AB")
    assert abs((a - b)[:2] @ _plan_dir(_on_face(arch, _element(arch, "Opening_1"))[2])) == pytest.approx(t + 0.015, abs=0.004)


@needs_blender
def test_panel_frame_goes_to_the_verify_collection(moulded):
    _, arch, facts, _ = moulded
    frame_ = facts["objects"]["PanelFrame_1"]
    assert frame_["collections"] == ["\u00c0 v\u00e9rifier"] and frame_["flags"] == "to_verify"
    assert facts["objects"]["PanelFrame_2"]["flags"] == "to_verify,unknown_profile", "a panel frame has no flat profile"
    lo, hi = np.array(frame_["bbox"])
    floor = _value(arch, "floor.level")
    assert (lo[2], hi[2]) == pytest.approx((floor + 0.9, floor + 1.9), abs=0.004)
    assert np.hypot(*(hi - lo)[:2]) == pytest.approx(0.8, abs=0.02)


@needs_blender
def test_architrave_without_a_width_says_which_element(tmp_path):
    ctx, _ = make_case(tmp_path, "case")
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    arch_path = output_path(ctx, "architecture")
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    _element(arch, "Door_1")["frame"] = {"profile": "flat"}
    arch_path.write_text(json.dumps(arch), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"Door_1: frame\.width is missing"):
        blender_stage.run(ctx)


def test_stage_deletes_stale_outputs_before_launching_blender(tmp_path, monkeypatch):
    ctx, _ = make_case(tmp_path, "case")
    ctx.out.mkdir(parents=True)
    (ctx.out / "architecture.json").write_text("{}", encoding="utf-8")
    stale = [ctx.out / name for name in blender_stage.OUTPUTS]
    for path in stale:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"stale")
    fake = tmp_path / "blender-that-writes-nothing"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("BLENDER", str(fake))
    with pytest.raises(RuntimeError, match="finished without writing"):
        blender_stage.run(ctx)
    assert not any(path.exists() for path in stale)


def test_stage_without_blender_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", str(tmp_path / "no-such-blender"))
    ctx, _ = make_case(tmp_path, "case")
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    with pytest.raises(BlenderMissing, match="no-such-blender"):
        blender_stage.run(ctx)
