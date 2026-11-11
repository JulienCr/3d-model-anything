import json
import os
import sys
import time
import types

import pytest

from room_model import __main__ as cli
from room_model.context import STAGE_OUTPUTS, require
from synthetic import make_capture


@pytest.fixture(autouse=True)
def _capture_folder(tmp_path, monkeypatch):
    make_capture(tmp_path / "cap")
    monkeypatch.setenv("BLENDER", str(tmp_path / "no-such-blender"))  # `all` skips the Blender stage instead of launching it


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
    assert order == ["registration", "shell", "model", "export", "openings", "vision", "anchor", "edge_pick", "architecture", "textures", "blender"]


def test_vision_placeholder_and_stale_architecture_rerun(tmp_path, monkeypatch):
    calls = []
    for stage in ("vision", "architecture"):
        def run(ctx, stage=stage):
            calls.append((stage, ctx.no_llm))
            out = ctx.out / STAGE_OUTPUTS[stage]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text('{"skipped": true}' if stage == "vision" and ctx.no_llm else "{}")
            return out

        _fake_stage(monkeypatch, stage, run)
    assert _run(tmp_path, "vision", "--no-llm") == 0 and _run(tmp_path, "vision", "--no-llm") == 0
    assert calls == [("vision", True)]  # the placeholder satisfies a --no-llm rerun
    assert _run(tmp_path, "vision") == 0 and calls[-1] == ("vision", False)  # but not a real one
    assert _run(tmp_path, "architecture") == 0 and _run(tmp_path, "architecture") == 0
    assert [c[0] for c in calls].count("architecture") == 1
    answers = tmp_path / "out" / STAGE_OUTPUTS["vision"]
    os.utime(answers, ns=(time.time_ns() + 10**10,) * 2)  # answers newer than architecture.json
    assert _run(tmp_path, "architecture") == 0  # merge them again
    assert [c[0] for c in calls].count("architecture") == 2


def test_textures_and_blender_rerun_when_their_inputs_are_newer(tmp_path, monkeypatch):
    calls = []
    for stage in ("architecture", "textures", "blender"):
        def run(ctx, stage=stage):
            calls.append(stage)
            out = ctx.out / STAGE_OUTPUTS[stage]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}")
            return out

        _fake_stage(monkeypatch, stage, run)
    assert _run(tmp_path, "all") == 0 and _run(tmp_path, "all") == 0
    assert calls.count("textures") == calls.count("blender") == 1
    out = tmp_path / "out"
    os.utime(out / STAGE_OUTPUTS["architecture"], ns=(time.time_ns() + 10**10,) * 2)
    assert _run(tmp_path, "all") == 0
    assert calls.count("textures") == calls.count("blender") == 2  # the new architecture reaches the textures and the scene
    os.utime(out / STAGE_OUTPUTS["textures"], ns=(time.time_ns() + 2 * 10**10,) * 2)
    assert _run(tmp_path, "all") == 0
    assert calls.count("textures") == 2 and calls.count("blender") == 3


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


def test_all_skips_the_blender_stage_without_blender(build, tmp_path, capsys):
    assert cli.main(["all", *_synthetic(build, tmp_path)]) == 0
    assert "blender skipped: Blender not found" in capsys.readouterr().out
    assert (tmp_path / "out" / STAGE_OUTPUTS["architecture"]).exists()
    assert not (tmp_path / "out" / STAGE_OUTPUTS["blender"]).exists()
    assert cli.main(["blender", *_synthetic(build, tmp_path)]) == 1  # asked for alone, it is an error
    assert "Blender not found" in capsys.readouterr().err


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


def test_tape_flag_installs_the_file_and_reruns_the_architecture(build, tmp_path, capsys):
    args = _synthetic(build, tmp_path)
    out = tmp_path / "out"
    assert cli.main(["all", *args]) == 0
    first = json.loads((out / STAGE_OUTPUTS["architecture"]).read_text(encoding="utf-8"))
    door = next(d for d in first["dimensions"] if d["id"] == "Door_1-w")
    mine = tmp_path / "mine.json"
    mine.write_text(json.dumps({"Door_1-w": door["value"] + 0.03, "Door_1-leaf-w": 0.8}), encoding="utf-8")
    capsys.readouterr()
    assert cli.main(["architecture", *args, "--tape", str(mine)]) == 0  # no --force: the new file makes the output stale
    assert "skipped" not in capsys.readouterr().out
    assert json.loads((out / "measurements.json").read_text(encoding="utf-8")) == json.loads(mine.read_text(encoding="utf-8"))
    arch = json.loads((out / STAGE_OUTPUTS["architecture"]).read_text(encoding="utf-8"))
    applied = next(d for d in arch["dimensions"] if d["id"] == "Door_1-w")
    assert applied["source"] == "tape" and applied["value"] == pytest.approx(door["value"] + 0.03)
    assert "| Door_1-w" in (out / "report.md").read_text(encoding="utf-8").split("## Mesures au mètre")[1]
    assert "Mètre : " in (out / "measurements" / "index.html").read_text(encoding="utf-8")
    assert cli.main(["architecture", *args]) == 0 and "skipped" in capsys.readouterr().out  # nothing new: left alone
    assert cli.main(["all", *args, "--tape", str(out / "measurements.json")]) == 0  # the installed file may be given back


def test_tape_flag_rejects_an_unusable_file(build, tmp_path, capsys):
    args = _synthetic(build, tmp_path)
    for text in ("not json", "[1, 2]"):
        bad = tmp_path / "bad.json"
        bad.write_text(text, encoding="utf-8")
        assert cli.main(["all", *args, "--tape", str(bad)]) == 1
        assert len(capsys.readouterr().err.strip().splitlines()) == 1
    assert cli.main(["all", *args, "--tape", str(tmp_path / "nowhere.json")]) == 1
    assert not (tmp_path / "out" / "measurements.json").exists()
    with pytest.raises(SystemExit):
        cli.main(["shell", *args, "--tape", str(bad)])  # only `architecture` and `all` take it


def test_overrides_flag_installs_the_file_and_reruns_the_architecture(build, tmp_path, capsys):
    args = _synthetic(build, tmp_path)
    out = tmp_path / "out"
    assert cli.main(["all", *args]) == 0
    mine = tmp_path / "mine.json"
    mine.write_text(json.dumps({"Door_1": {"swing": "out", "hinge": "right"}, "Door_9": {"remove": True}}), encoding="utf-8")
    capsys.readouterr()
    assert cli.main(["architecture", *args, "--overrides", str(mine)]) == 0  # no --force: the new file makes the output stale
    assert "skipped" not in capsys.readouterr().out
    assert json.loads((out / "overrides.json").read_text(encoding="utf-8")) == json.loads(mine.read_text(encoding="utf-8"))
    arch = json.loads((out / STAGE_OUTPUTS["architecture"]).read_text(encoding="utf-8"))
    door = next(e for e in arch["elements"] if e["id"] == "Door_1")
    assert (door["swing"], door["hinge"], "user_override" in door["flags"]) == ("out", "right", True)
    section = (out / "report.md").read_text(encoding="utf-8").split("## Corrections manuelles")[1]
    assert "Door_1" in section and "Door_9" in section  # applied and refused ones are both listed
    assert cli.main(["architecture", *args]) == 0 and "skipped" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    assert cli.main(["all", *args, "--overrides", str(bad)]) == 1 and len(capsys.readouterr().err.strip().splitlines()) == 1
