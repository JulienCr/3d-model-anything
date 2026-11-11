import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from room_model import architecture, blender_stage, meshes, model, openings, shell
from room_model.context import BlenderMissing, output_path
from helpers import make_case

PROBE = Path(__file__).with_name("blender_probe.py")
BLENDER_DIR = Path(__file__).resolve().parents[1] / "blender"
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


def _add_variants(arch):
    for source, new_id, host, centre, width, fields in JOINERY_VARIANTS:
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
    ("open-flipped", "Door_1", {"Ouverture": 1, "Sens": 1}),
    ("half", "Door_1", {"Ouverture": 0.5}),
    ("short-angle", "Door_1", {"Ouverture": 1, "AngleMax": 60}),
]
for _id in (*(f"Door_{k}" for k in range(1, 9)),
            "Window_1", "Window_2", "Window_3", "Window_4", "Window_5"):
    RIGS += [(f"closed-{_id}", _id, {}), (f"open-{_id}", _id, {"Ouverture": 1}),
             (f"flipped-{_id}", _id, {"Ouverture": 1, "Sens": 1})]


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
    """Blender plan direction the leaves swing to at Sens = 0: the interior normal, reversed by `swing: out`; on a partition,
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
        assert root["props"] == {"Ouverture": 0.0, "Sens": 0, "AngleMax": 90.0}, oid
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
    closed, opened, flipped = (facts["rigs"][f"{kind}-{oid}"][leaf] for kind in ("closed", "open", "flipped"))
    hinge = np.array(closed["world_origin"])[:2]
    assert np.sign((hinge - np.array(facts["objects"][oid]["location"])[:2]) @ left) == (1 if el.get("hinge", "left") == "left" else -1)
    assert (np.array(opened["world_centre"])[:2] - hinge) @ f > 0.05, "opens towards the swing direction"
    assert (np.array(flipped["world_centre"])[:2] - hinge) @ f < -0.05, "Sens flips it"


@needs_blender
def test_rig_hinged(scene):
    _, _, facts = scene
    leaf = "Door_1_Leaf_1"
    assert _pose(facts, "closed", leaf)["rotation_z_deg"] == pytest.approx(0, abs=0.01)
    assert _pose(facts, "open", leaf)["rotation_z_deg"] == pytest.approx(90, abs=0.5)
    assert _pose(facts, "open-flipped", leaf)["rotation_z_deg"] == pytest.approx(-90, abs=0.5)
    assert _pose(facts, "half", leaf)["rotation_z_deg"] == pytest.approx(45, abs=0.5)
    assert _pose(facts, "short-angle", leaf)["rotation_z_deg"] == pytest.approx(60, abs=0.5)
    for key in ("open", "half"):  # the hinge stays where it is
        assert _pose(facts, key, leaf)["location"] == pytest.approx(_pose(facts, "closed", leaf)["location"], abs=1e-4)
    assert facts["objects"][leaf]["drivers"] > 0 and facts["objects"]["Door_1"]["props"]["Ouverture"] == 0


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
    flipped = facts["rigs"][f"flipped-{oid}"][leaf]
    travel = np.array(opened["location"]) - np.array(closed["location"])
    assert travel[0] == pytest.approx(direction * _width(facts, leaf), abs=0.002)
    assert travel[1:] == pytest.approx([0, 0], abs=1e-4) and opened["rotation_z_deg"] == pytest.approx(0, abs=0.01)
    assert flipped["location"][0] - closed["location"][0] == pytest.approx(-direction * _width(facts, leaf), abs=0.002)


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
    assert b"Ouverture" not in raw and b"Sens" not in raw
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


@needs_blender
def test_wall_matches_the_m1_mesh(scene):
    _, arch, facts = scene
    wall = _element(arch, "Wall_M3")
    az = _value(arch, wall["params"]["azimuth"])
    n = np.array([math.cos(az), math.sin(az)])
    start, end = np.array(wall["start"]), np.array(wall["end"])
    offset = _value(arch, wall["params"]["offset"])
    start, end = start + n * (offset - n @ start), end + n * (offset - n @ end)  # onto the fitted plane, as the scene does
    vertices, _ = meshes.wall_mesh(start, end, n, _value(arch, wall["params"]["thickness"]), _value(arch, "floor.level"),
                                   _value(arch, wall["params"]["top"]))
    mapped = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
    lo, hi = facts["objects"]["Wall_M3"]["bbox"]
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
