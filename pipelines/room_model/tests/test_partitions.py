import numpy as np
import pytest

from room_model import shell
from helpers import make_case, plan_point, truth_to_room


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("partitions")


@pytest.fixture(scope="module")
def base(tmp_root):
    ctx, truth = make_case(tmp_root, "base")
    return shell.build(ctx)


@pytest.fixture(scope="module")
def both(tmp_root):
    ctx, truth = make_case(tmp_root, "both", partition=True)
    return ctx, truth, shell.build(ctx)


@pytest.fixture(scope="module")
def one_face(tmp_root):
    ctx, truth = make_case(tmp_root, "one-face", partition=True, partition_faces="a")
    return ctx, truth, shell.build(ctx)


def _partitions(doc):
    return [e for e in doc["elements"] if e["kind"] == "partition"]


def _param(doc, el, name):
    return doc["parameters"][el["params"][name]]


def test_partition_detected(both):
    ctx, truth, doc = both
    (el,) = _partitions(doc)
    assert el["id"] == "Partition_P1" and el["label"] == "P1" and el["roomplan_id"] == truth.partition["ident"]
    thickness = _param(doc, el, "thickness")
    assert thickness["value"] == pytest.approx(truth.partition["thickness"], abs=0.02) and thickness["source"] == "lidar"
    T = truth_to_room(ctx, truth)
    expected = sorted(tuple(plan_point(T, *p)) for p in truth.partition["ends"])
    found = sorted(tuple(p) for p in (el["start"], el["end"]))
    for got, want in zip(found, expected):
        assert np.linalg.norm(np.subtract(got, want)) < 0.03
    assert doc["diagnostics"]["off_outline_walls"] == []
    assert el["id"] in {e["id"] for e in doc["diagnostics"]["partitions"]}


def test_partition_face_a_points_to_plus_x(both):
    _, truth, doc = both
    (el,) = _partitions(doc)
    az = _param(doc, el, "azimuth")["value"]
    # the room frame is the truth frame turned by the registration; face A keeps the +x convention in the room frame
    n = np.array([np.cos(az), np.sin(az)])
    assert n[1] > 0 or (abs(n[1]) < 0.1 and n[0] > 0)


def test_partition_one_face_seen(one_face):
    _, truth, doc = one_face
    (el,) = _partitions(doc)
    thickness = _param(doc, el, "thickness")
    assert thickness["value"] == pytest.approx(0.10) and thickness["source"] == "assumed"
    assert "one_face_seen" in el["flags"]


def test_partition_outside_floor_ignored(tmp_root):
    ctx, truth = make_case(tmp_root, "outside", outside_roomplan_wall=True)
    doc = shell.build(ctx)
    assert _partitions(doc) == []
    assert doc["diagnostics"]["off_outline_walls"] == [truth.outside_ident]


def test_perimeter_unchanged(base, both):
    _, _, doc = both
    walls = lambda d: {e["label"]: e for e in d["elements"] if e["kind"] == "wall"}  # noqa: E731
    assert walls(doc).keys() == walls(base).keys()
    for label, el in walls(doc).items():
        for name, pid in el["params"].items():
            assert doc["parameters"][pid]["value"] == pytest.approx(base["parameters"][walls(base)[label]["params"][name]]["value"], abs=1e-3)
        assert el["start"] == pytest.approx(walls(base)[label]["start"], abs=1e-3)


def test_partition_without_lidar_keeps_roomplan_extent(tmp_root):
    ctx, truth = make_case(tmp_root, "no-lidar", partition=True, partition_faces="")
    doc = shell.build(ctx)
    (el,) = _partitions(doc)
    assert "roomplan_fallback" in el["flags"]
    assert _param(doc, el, "thickness")["source"] == "assumed"
    T = truth_to_room(ctx, truth)
    expected = sorted(tuple(plan_point(T, *p)) for p in truth.partition["ends"])
    for got, want in zip(sorted(tuple(p) for p in (el["start"], el["end"])), expected):
        assert np.linalg.norm(np.subtract(got, want)) < 0.05


def test_partition_only_face_b_seen_still_places_face_a(tmp_root, both):
    ctx, truth = make_case(tmp_root, "face-b", partition=True, partition_faces="b")
    doc = shell.build(ctx)
    (el,) = _partitions(doc)
    (ref,) = _partitions(both[2])
    assert "one_face_seen" in el["flags"]
    assert _param(doc, el, "offset")["value"] == pytest.approx(_param(both[2], ref, "offset")["value"], abs=0.02)


def test_partition_reaches_glb_and_report(tmp_root):
    from room_model import export, model
    from test_export import read_glb, section

    ctx, _ = make_case(tmp_root, "exported", partition=True, partition_faces="a")
    shell.run(ctx)
    model.run(ctx)
    glb = read_glb(export.run(ctx))
    node = next(n for n in glb["nodes"] if n["name"] == "Partition_P1")
    assert node["extras"]["kind"] == "partition" and "mesh" in node
    report_text = (ctx.out / "report.md").read_text(encoding="utf-8")
    assert "| P1 |" in section(report_text, "Cloisons")
    assert "Cloison P1 : une seule face vue" in section(report_text, "Alertes")
    assert "Mur RoomPlan hors du contour" not in report_text
