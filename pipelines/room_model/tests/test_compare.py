import json

import numpy as np
import pytest

from room_model import __main__ as cli, compare, model, shell
from room_model.context import output_path
from helpers import make_case


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    ctx, _ = make_case(tmp_path_factory.mktemp("compare"), "default")
    shell.run(ctx)
    model.run(ctx)
    return json.loads(output_path(ctx, "model").read_text(encoding="utf-8"))


def moved_polygon(poly, edge, shift):
    """Polygon with edge `edge` translated by `shift` along its normal, neighbours keeping their lines."""
    pts = [np.array(p) for p in poly]
    n = len(pts)
    lines = []
    for i in range(n):
        d = pts[(i + 1) % n] - pts[i]
        d /= np.linalg.norm(d)
        normal = np.array([-d[1], d[0]])
        lines.append((pts[i] + (shift * normal if i == edge else 0.0), d))
    out = []
    for i in range(n):
        (p0, d0), (p1, d1) = lines[i - 1], lines[i]
        t = np.linalg.solve(np.array([d0, -d1]).T, p1 - p0)[0]
        out.append((p0 + t * d0).tolist())
    return out


def rows(report, header):
    lines = report.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(header))
    start = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
    table = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        table.append([c.strip() for c in line.strip("|").split("|")])
    return table


def test_compare_detects_offset(doc):
    poly = next(e for e in doc["elements"] if e["kind"] == "floor")["polygon"]
    reference = {"polygon": moved_polygon(poly, 2, -0.03), "edge_heights": [2.5] * len(poly)}
    walls = rows(compare.compare(doc, reference), "## Murs")
    deltas = {r[0]: float(r[2]) for r in walls}
    assert abs(deltas["M3"]) == pytest.approx(3.0, abs=0.2)
    assert all(abs(v) < 0.2 for k, v in deltas.items() if k != "M3")
    assert all(abs(float(r[3])) < 0.1 for r in walls)


def test_compare_heights_dimensions_and_unmatched(doc):
    poly = next(e for e in doc["elements"] if e["kind"] == "floor")["polygon"]
    dim = next(d for d in doc["dimensions"] if d["id"] == "S01")
    pillar = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    reference = {"polygon": poly, "edge_heights": [2.5, 2.5, 2.5, 2.3, 2.3, 2.5],
                 "dimensions": {"S01": dim["value"] - 0.02, "ZZ": 1.0}}
    report = compare.compare(doc, reference)
    assert {r[0]: abs(float(r[4])) < 3 for r in rows(report, "## Murs")} == {f"M{i}": True for i in range(1, 7)}
    s01 = next(r for r in rows(report, "## Dimensions") if r[0] == "S01")
    assert float(s01[3]) == pytest.approx(2.0, abs=0.01)
    assert float(s01[4]) == pytest.approx(0.02 / dim["sigma"], abs=0.05)
    with_pillar = compare.compare(doc, {"polygon": poly + pillar})
    assert "non appariées" in with_pillar


def test_compare_cli_writes_file(doc, tmp_path):
    (tmp_path / "model.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "ref.json").write_text(json.dumps({"dimensions": {"S01": 2.5}}), encoding="utf-8")
    out = tmp_path / "compare.md"
    assert cli.main(["compare", "--model", str(tmp_path / "model.json"), "--reference", str(tmp_path / "ref.json"), "-o", str(out)]) == 0
    assert "S01" in out.read_text(encoding="utf-8")


def run_cli(tmp_path, doc, reference, capsys):
    (tmp_path / "model.json").write_text(json.dumps(doc), encoding="utf-8")
    ref = tmp_path / "ref.json"
    if isinstance(reference, str):
        ref.write_text(reference, encoding="utf-8")
    elif reference is not None:
        ref.write_text(json.dumps(reference), encoding="utf-8")
    code = cli.main(["compare", "--model", str(tmp_path / "model.json"), "--reference", str(ref)])
    return code, capsys.readouterr().err


@pytest.mark.parametrize("reference", [None, "{not json"])
def test_compare_cli_unreadable_reference(doc, tmp_path, capsys, reference):
    code, err = run_cli(tmp_path, doc, reference, capsys)
    assert code == 1 and "référence illisible" in err and "Traceback" not in err


@pytest.mark.parametrize("polygon", [None, [1, 2, 3], [[0, 0], [1, "a"], [2, 2]], [[0, 0], [1, 0]]])
def test_compare_cli_bad_polygon(doc, tmp_path, capsys, polygon):
    code, err = run_cli(tmp_path, doc, {"polygon": polygon}, capsys)
    assert code == 1 and "polygon" in err


def test_compare_cli_edge_heights_length(doc, tmp_path, capsys):
    ref = {"polygon": [[0, 0], [1, 0], [1, 1]], "edge_heights": [2.5, 2.5]}
    code, err = run_cli(tmp_path, doc, ref, capsys)
    assert code == 1 and "edge_heights" in err


def test_compare_cli_non_numeric_dimension(doc, tmp_path, capsys):
    code, err = run_cli(tmp_path, doc, {"dimensions": {"S01": "2.5m"}}, capsys)
    assert code == 1 and "dimensions.S01" in err


def test_compare_zero_sigma_shows_na(doc):
    flat = json.loads(json.dumps(doc))
    for d in flat["dimensions"]:
        d["sigma"] = 0.0
    s01 = next(r for r in rows(compare.compare(flat, {"dimensions": {"S01": 2.4}}), "## Dimensions") if r[0] == "S01")
    assert s01[4] == "n/a"
