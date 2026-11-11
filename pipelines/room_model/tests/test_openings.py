import json

import numpy as np
import pytest

from room_model import model, openings, shell
from helpers import make_case, plan_point, truth_to_room
from synthetic import DOOR, PARTITION, PARTITION_OPENING, WINDOW


def _run(root, name, **options):
    ctx, truth = make_case(root, name, **options)
    shell.run(ctx)
    model.run(ctx)
    path = openings.run(ctx)
    return ctx, truth, json.loads(path.read_text(encoding="utf-8")), json.loads((ctx.out / "model.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("openings")


@pytest.fixture(scope="module")
def full(tmp_root):
    return _run(tmp_root, "full", window=True, partition=True, partition_opening=True, orphan_opening=True)


def _opening(doc, id_):
    return next(o for o in doc["openings"] if o["id"] == id_)


def _value(doc, o, name):
    return doc["parameters"][o["params"][name]]["value"]


def _element(model_doc, id_):
    return next(e for e in model_doc["elements"] if e["id"] == id_)


def _s_of(model_doc, host_id, point):
    host = _element(model_doc, host_id)
    start, end = np.array(host["start"]), np.array(host["end"])
    d = (end - start) / np.linalg.norm(end - start)
    return float((np.asarray(point) - start) @ d)


def test_door_on_perimeter(full):
    ctx, truth, doc, model_doc = full
    assert doc["schema"] == "room-model/openings-1"
    door = _opening(doc, "Door_1")
    assert door["kind"] == "door" and door["face"] is None and door["flags"] == []
    host = _element(model_doc, door["host"])
    assert host["kind"] == "wall" and host["roomplan_id"] == truth.wall_ident["east"][0]
    T = truth_to_room(ctx, truth)
    assert _value(doc, door, "center_s") == pytest.approx(_s_of(model_doc, door["host"], plan_point(T, 6.0, DOOR["z"])), abs=0.02)
    assert _value(doc, door, "width") == pytest.approx(DOOR["w"], abs=0.02)
    assert _value(doc, door, "height") == pytest.approx(DOOR["h"], abs=0.02)
    assert _value(doc, door, "sill") == pytest.approx(0.0, abs=0.02)
    assert _value(doc, door, "depth") == pytest.approx(model_doc["parameters"][host["params"]["thickness"]]["value"])


def test_window_sill(full):
    ctx, truth, doc, model_doc = full
    win = _opening(doc, "Window_1")
    assert win["kind"] == "window" and win["roomplan_id"] == truth.opening_idents["window"]
    assert _value(doc, win, "sill") == pytest.approx(WINDOW["sill"], abs=0.02)
    assert _value(doc, win, "width") == pytest.approx(WINDOW["w"], abs=0.02)
    assert _value(doc, win, "height") == pytest.approx(WINDOW["h"], abs=0.02)


def test_opening_on_partition(full):
    ctx, truth, doc, model_doc = full
    op = _opening(doc, "Opening_1")
    assert op["kind"] == "opening" and op["host"] == "Partition_P1" and op["face"] in ("A", "B")
    T = truth_to_room(ctx, truth)
    s = _s_of(model_doc, "Partition_P1", plan_point(T, PARTITION["x"], PARTITION_OPENING["z"]))
    assert _value(doc, op, "center_s") == pytest.approx(s, abs=0.03)
    thickness = model_doc["parameters"][_element(model_doc, "Partition_P1")["params"]["thickness"]]["value"]
    assert _value(doc, op, "depth") == pytest.approx(thickness)
    assert doc["parameters"][op["params"]["width"]]["source"] == "roomplan"


def test_opening_clamped_to_host(tmp_root):
    ctx, truth, doc, model_doc = _run(tmp_root, "wide", partition=True, partition_opening=True, partition_opening_width=2.3)
    op = _opening(doc, "Opening_1")
    host = _element(model_doc, "Partition_P1")
    length = float(np.linalg.norm(np.subtract(host["end"], host["start"])))
    assert "clamped" in op["flags"]
    assert _value(doc, op, "width") == pytest.approx(length, abs=0.01)
    assert _value(doc, op, "center_s") == pytest.approx(length / 2, abs=0.05)


def test_orphan_opening(full):
    _, truth, doc, _ = full
    orphan = next(o for o in doc["openings"] if o["roomplan_id"] == truth.opening_idents["orphan-opening"])
    assert orphan["host"] is None and orphan["face"] is None and "orphan" in orphan["flags"]
    assert orphan["id"] in doc["diagnostics"]["orphans"]
    assert "center_s" not in orphan["params"]


def test_lidar_refines_misplaced_window(tmp_root):
    ctx, truth, doc, _ = _run(tmp_root, "shifted", window=True, window_rp_error=0.06)
    win = _opening(doc, "Window_1")
    width, centre = doc["parameters"][win["params"]["width"]], doc["parameters"][win["params"]["center_s"]]
    assert width["source"] == centre["source"] == "lidar" and 0.0 < width["sigma"] < 0.03
    assert width["value"] == pytest.approx(WINDOW["w"], abs=0.02)
    assert _value(doc, win, "sill") == pytest.approx(WINDOW["sill"], abs=0.02)
    plain = _run(tmp_root, "unshifted", window=True)[2]
    assert centre["value"] == pytest.approx(_value(plain, _opening(plain, "Window_1"), "center_s"), abs=0.02)


def test_roomplan_kept_when_mesh_has_no_hole(tmp_root):
    _, _, doc, _ = _run(tmp_root, "no-lidar", partition=True, partition_opening=True)
    assert {doc["parameters"][pid]["source"] for pid in _opening(doc, "Opening_1")["params"].values()} <= {"roomplan", "lidar"}
    assert doc["parameters"][_opening(doc, "Opening_1")["params"]["width"]]["source"] == "roomplan"


def test_host_of_prefers_parent_then_nearest_parallel():
    hosts = [{"id": "A", "roomplan_id": "ra", "start": [0, 0], "end": [4, 0], "normal": [0, 1], "offset": 0.0, "thickness": 0.2},
             {"id": "B", "roomplan_id": "rb", "start": [0, 3], "end": [4, 3], "normal": [0, -1], "offset": -3.0, "thickness": 0.2}]
    tf = np.eye(4)
    tf[:3, 3] = [2.0, 1.0, 0.05]
    assert openings.host_of(tf, "rb", hosts)[0] == "B"
    host, dist = openings.host_of(tf, "unknown", hosts)
    assert host == "A" and dist == pytest.approx(0.05)
    tf[:3, 3] = [2.0, 1.0, 1.5]
    assert openings.host_of(tf, None, hosts) is None
    turned = np.eye(4)
    turned[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # x axis along z: not parallel to either host
    turned[:3, 3] = [2.0, 1.0, 0.05]
    assert openings.host_of(turned, None, hosts) is None
