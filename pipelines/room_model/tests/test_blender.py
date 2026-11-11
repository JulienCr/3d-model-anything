import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from room_model import architecture, blender_stage, model, openings, shell
from room_model.context import BlenderMissing, output_path
from helpers import make_case

PROBE = Path(__file__).with_name("blender_probe.py")
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


def _ray(arch, opening_id, name, along, height, side_distance=0.6):
    """Ray crossing the host of `opening_id` at `along` m from the centre of the opening, at `height` m above the floor."""
    opening = _element(arch, opening_id)
    host = _element(arch, opening["host"])
    az = _value(arch, host["params"]["azimuth"])
    n = np.array([math.cos(az), math.sin(az)])
    start, end = np.array(host["start"]), np.array(host["end"])
    u = (end - start) / np.linalg.norm(end - start)
    p = start + u * (_value(arch, opening["params"]["center_s"]) + along)
    p = p + n * (_value(arch, host["params"]["offset"]) - n @ p)  # on the face the normal points away from
    floor = _value(arch, "floor.level")
    origin = p + n * side_distance
    return {"name": name, "object": host["id"], "origin": _blender_point(origin[0], floor + height, origin[1]),
            "direction": _blender_point(-n[0], 0.0, -n[1])}


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("blender")
    ctx, _ = make_case(root, "scene", window=True, partition=True, partition_opening=True, orphan_opening=True)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    blender_stage.run(ctx)
    arch = json.loads(output_path(ctx, "architecture").read_text(encoding="utf-8"))

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
    rays_path, facts_path = ctx.out / "rays.json", ctx.out / "facts.json"
    rays_path.write_text(json.dumps(rays), encoding="utf-8")
    blender_stage.run_blender(["--background", ctx.out / "architecture.blend", "--python-exit-code", "1", "--python", PROBE,
                               "--", rays_path, facts_path], ctx.out / "probe.log")
    return ctx, arch, json.loads(facts_path.read_text(encoding="utf-8"))


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
    for name in ("top", "persp_1", "persp_2"):
        image = Image.open(ctx.out / "renders" / f"{name}.png")
        assert image.size == (1280, 720)
        assert np.asarray(image.convert("L")).std() > 10, name


def test_stage_without_blender_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", str(tmp_path / "no-such-blender"))
    ctx, _ = make_case(tmp_path, "case")
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    with pytest.raises(BlenderMissing, match="no-such-blender"):
        blender_stage.run(ctx)
