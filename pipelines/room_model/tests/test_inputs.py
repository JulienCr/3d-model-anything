import json

import numpy as np

from room_model.capture import colmajor, load_mesh_ply, load_roomplan, mesh_path
from room_model.frame import world_to_room
from room_model.roomplan import Room


def _room(capture_dir):
    return Room(load_roomplan(capture_dir, 0), world_to_room(capture_dir, 0, 0))


def test_room_frame_aligns_walls(capture):
    root, _ = capture
    w2r = world_to_room(root, 0, 0)
    data = load_roomplan(root, 0)
    for wall in data["walls"]:
        n = w2r[:3, :3] @ colmajor(wall["transform"], 4)[:3, 2]
        assert max(abs(n[0]), abs(n[2])) > 0.999
    floor_origin = colmajor(data["floors"][0]["transform"], 4)[:3, 3]
    assert abs((w2r[:3, :3] @ floor_origin + w2r[:3, 3])[1]) < 0.03


def test_walls_follow_outline_order(capture):
    root, truth = capture
    room = _room(root)
    assert [w.label for w in room.walls] == [f"M{k}" for k in range(1, 7)]
    labels = {name: [w.label for w in room.walls if w.ident in ids] for name, ids in truth.wall_ident.items()}
    assert all(len(v) == 1 for v in labels.values())
    assert len({v[0] for v in labels.values()}) == 6
    assert [d.label for d in room.doors] == ["P1"]
    assert room.doors[0].wall.label == labels["east"][0]
    assert room.off_outline == [] and room.seeded_from_outline == []


def test_off_outline_wall_is_reported(build):
    base_root, _ = build()
    root, truth = build(extra_roomplan_wall=True)
    room, base = _room(root), _room(base_root)
    assert len(room.walls) == 6
    assert [w.label for w in room.walls] == [w.label for w in base.walls]
    assert [w.ident for w in room.walls] == [w.ident for w in base.walls]
    assert room.off_outline == [truth.extra_ident]


def test_split_wall_gets_one_label_per_piece(build):
    root, truth = build(split_south_wall=True)
    room = _room(root)
    assert len(room.walls) == 7 and room.off_outline == []
    assert len(truth.wall_ident["south"]) == 2
    labels = [[w.label for w in room.walls if w.ident == i] for i in truth.wall_ident["south"]]
    assert all(len(v) == 1 for v in labels) and labels[0] != labels[1]


def test_missing_wall_is_seeded_from_outline(build):
    root, truth = build()
    path = root / "rooms" / "room-0.json"
    data = json.loads(path.read_text())
    west = truth.wall_ident["west"][0]
    data["walls"] = [w for w in data["walls"] if w["identifier"] != west]
    path.write_text(json.dumps(data))
    room = _room(root)
    seeded = [w for w in room.walls if w.ident.startswith("outline-")]
    assert len(room.walls) == 6 and len(seeded) == 1
    assert room.seeded_from_outline == [seeded[0].ident]
    inward = seeded[0].n[[0, 2]]
    centre = room.outline[:, [0, 2]].mean(axis=0)
    assert inward @ (centre - seeded[0].c[[0, 2]]) > 0


def test_mesh_roundtrip(capture):
    root, truth = capture
    mesh = load_mesh_ply(mesh_path(root, 0))
    assert {1, 2, 3, 4, 7} <= set(np.unique(mesh["face_class"]).tolist())
    face_inroom = mesh["inroom"][mesh["faces"]].min(axis=1)
    panel = mesh["faces"][face_inroom == 0]
    assert len(panel) > 0
    inverse = np.linalg.inv(truth.world_from_local)
    local = mesh["vertices"][np.unique(panel)] @ inverse[:3, :3].T + inverse[:3, 3]
    assert np.abs(local[:, 2] + 0.05).max() < 0.03
    assert np.all(mesh["face_class"][face_inroom == 0] == 1)
    assert mesh["colors"].dtype == np.uint8 and len(mesh["vertices"]) == len(mesh["normals"])
