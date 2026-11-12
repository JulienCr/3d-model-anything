import json
import sys
import types

import pytest

from room_model import __main__ as cli
from room_model.context import STAGE_OUTPUTS, require
from synthetic import make_capture


@pytest.fixture(autouse=True)
def _capture_folder(tmp_path):
    make_capture(tmp_path / "cap")


def _run(tmp_path, *args):
    return cli.main([*args, "--capture", str(tmp_path / "cap"), "--out", str(tmp_path / "out")])


def _fake_stage(monkeypatch, stage, body):
    module = types.ModuleType(f"fake_{stage}")
    module.run = body
    monkeypatch.setitem(cli.STAGE_MODULES, stage, module.__name__)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_model_before_shell_asks_for_shell(tmp_path, monkeypatch, capsys):
    _fake_stage(monkeypatch, "model", lambda ctx: require(ctx, "shell"))
    assert _run(tmp_path, "model") != 0
    assert "run shell first" in capsys.readouterr().err


def test_stage_skipped_when_output_exists_and_rerun_with_force(tmp_path, monkeypatch):
    calls = []

    def run(ctx):
        calls.append(ctx.force)
        out = ctx.out / STAGE_OUTPUTS["shell"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return out

    _fake_stage(monkeypatch, "shell", run)
    assert _run(tmp_path, "shell") == 0 and calls == [False]
    assert _run(tmp_path, "shell") == 0 and calls == [False]
    assert _run(tmp_path, "shell", "--force") == 0 and calls == [False, True]


def test_all_runs_stages_in_order(tmp_path, monkeypatch):
    order = []
    for stage in STAGE_OUTPUTS:
        def run(ctx, stage=stage):
            order.append(stage)
            out = ctx.out / STAGE_OUTPUTS[stage]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}")
            return out

        _fake_stage(monkeypatch, stage, run)
    assert _run(tmp_path, "all") == 0
    assert order == ["registration", "shell", "model", "export", "openings", "architecture"]


def _synthetic(build, tmp_path):
    root, _ = build()
    return ["--capture", str(root), "--out", str(tmp_path / "out")]


def test_all_skips_registration_of_a_capture_without_photos(build, tmp_path, capsys):
    assert cli.main(["all", *_synthetic(build, tmp_path)]) == 0
    assert "registration skipped: no photos for room 0" in capsys.readouterr().out
    for stage in ("shell", "model", "export", "openings", "architecture"):
        assert (tmp_path / "out" / STAGE_OUTPUTS[stage]).exists()
    assert (tmp_path / "out" / "measurements" / "index.html").exists()
    assert not (tmp_path / "out" / STAGE_OUTPUTS["registration"]).exists()


def test_registration_alone_without_photos_exits_with_one_line(build, tmp_path, capsys):
    assert cli.main(["registration", *_synthetic(build, tmp_path)]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "no photos for room 0" in err[0]


@pytest.mark.parametrize("stage", ["registration", "shell", "all"])
def test_missing_capture_or_room_exits_with_one_line(build, tmp_path, capsys, stage):
    args = _synthetic(build, tmp_path)
    assert cli.main([stage, "--capture", str(tmp_path / "nowhere"), "--out", args[3]]) == 1
    assert len(capsys.readouterr().err.strip().splitlines()) == 1
    assert cli.main([stage, *args, "--room", "7"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "room 7" in err[0]


def test_unknown_subcommand_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "nope")


def test_compare_tape_writes_a_markdown_report(build, tmp_path, capsys):
    args = _synthetic(build, tmp_path)
    assert cli.main(["architecture", *args]) == 1  # needs the model first
    assert "run model first" in capsys.readouterr().err
    assert cli.main(["all", *args]) == 0
    arch_path = tmp_path / "out" / STAGE_OUTPUTS["architecture"]
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    dim = next(d for d in arch["dimensions"] if d["id"] == "Door_1-w")
    tape_path = tmp_path / "measurements.json"
    tape_path.write_text(json.dumps({"Door_1-w": dim["value"] + 0.02}), encoding="utf-8")
    report = tmp_path / "tape.md"
    assert cli.main(["compare-tape", "--arch", str(arch_path), "--tape", str(tape_path), "-o", str(report)]) == 0
    assert "+2,0" in report.read_text(encoding="utf-8")
    assert cli.main(["compare-tape", "--arch", str(arch_path), "--tape", str(tmp_path / "nowhere.json")]) == 1
    assert len(capsys.readouterr().err.strip().splitlines()) == 1
