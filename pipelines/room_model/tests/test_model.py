import json

import numpy as np
import pytest

from room_model import model, shell
from room_model.context import output_path
from helpers import make_case, plan_point, truth_to_room, wall_labels


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    ctx, truth = make_case(tmp_path_factory.mktemp("model"), "default")
    shell.run(ctx)
    model.run(ctx)
    doc = json.loads(output_path(ctx, "model").read_text(encoding="utf-8"))
    return doc, {d["id"]: d for d in doc["dimensions"]}, truth


def test_line_sigma():
    assert model.line_sigma(0.01, 0.0, 5.0) == 0.01
    assert model.line_sigma(0.0, 0.002, 3.0) == pytest.approx(0.006)


def test_station_widths(scene):
    doc, dims, truth = scene
    south, step = wall_labels(doc, truth, "south")[0], wall_labels(doc, truth, "step")[0]
    stations = [d for d in doc["dimensions"] if d["kind"] == "width" and set(d["elements"]) == {f"Wall_{south}", f"Wall_{step}"}]
    assert len(stations) >= 2
    for d in stations:
        assert d["value"] == pytest.approx(2.50, abs=0.01)
        assert abs(d["value"] - 2.50) < 3 * d["sigma"]
        assert len(d["at"]) == 2
    assert [d["id"] for d in doc["dimensions"] if d["id"].startswith("S")][0] == "S01"


def test_heights_and_structures(scene):
    doc, dims, _ = scene
    assert dims["H1"]["value"] == pytest.approx(2.50, abs=0.01)
    assert dims["H2"]["value"] == pytest.approx(2.30, abs=0.01)
    assert dims["B1-u"]["value"] == pytest.approx(2.20, abs=0.01)
    assert dims["B1-w"]["value"] == pytest.approx(0.30, abs=0.02)
    columns = [e for e in doc["elements"] if e["kind"] == "column"]
    free, engaged = (next(c for c in columns if (c["engaged_wall"] is None) == want) for want in (True, False))
    for col, count in ((free, 4), (engaged, 3)):
        assert len([d for d in doc["dimensions"] if col["id"] in d["elements"]]) == count
    assert not [d for d in doc["dimensions"] if d["kind"] == "column_offset" and engaged["id"] in d["elements"] and f"Wall_{engaged['engaged_wall']}" in d["elements"]]
    assert dims["COL1-x"]["value"] == pytest.approx(0.40, abs=0.01)
    assert "COL2-dz" not in dims
    assert dims["COL2-dx"]["value"] == pytest.approx(0.50, abs=0.02)


def test_model_json_roundtrip(scene):
    doc, _, _ = scene
    assert doc["schema"] == "room-model/1" and doc["version"] == "v1-scan"
    for el in doc["elements"]:
        assert all(p in doc["parameters"] for p in el["params"].values())
    ids = {e["id"] for e in doc["elements"]}
    for d in doc["dimensions"]:
        assert d["sigma"] > 0 and d["source"] in ("assumed", "roomplan", "lidar")
        assert set(d["elements"]) <= ids and isinstance(d["flags"], list)


def _run(root, name, **options):
    ctx, truth = make_case(root, name, **options)
    shell.run(ctx)
    model.run(ctx)
    return json.loads(output_path(ctx, "model").read_text(encoding="utf-8")), ctx, truth


def _pair(doc, truth, a, b):
    ids = {f"Wall_{wall_labels(doc, truth, n)[0]}" for n in (a, b)}
    return [d for d in doc["dimensions"] if d["id"].startswith("S") and set(d["elements"]) == ids]


def test_stations_follow_the_fitted_walls_not_roomplan(tmp_path_factory):
    doc, _, truth = _run(tmp_path_factory.mktemp("offset"), "offset", offset_wall="wing_east")
    stations = _pair(doc, truth, "west", "wing_east")
    assert stations
    for d in stations:
        assert d["flags"] == [] and d["value"] == pytest.approx(2.50, abs=0.02)


def test_no_station_stands_in_a_door(tmp_path_factory):
    doc, ctx, truth = _run(tmp_path_factory.mktemp("door"), "door")
    T = truth_to_room(ctx, truth)
    door, along = plan_point(T, 6.0, 1.25), plan_point(T, 6.0, 2.25) - plan_point(T, 6.0, 1.25)
    stations = _pair(doc, truth, "west", "east")
    assert stations
    assert all(abs((np.array(d["at"]) - door) @ along) > 0.45 + 0.1 for d in stations)
