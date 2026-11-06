import sys
import types

import pytest

from room_model import __main__ as cli
from room_model.context import STAGE_OUTPUTS, require


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


def test_unknown_stage_module_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(cli.STAGE_MODULES, "export", "room_model._not_written")
    assert _run(tmp_path, "export") == 2
    assert "not implemented" in capsys.readouterr().err


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


def test_all_runs_stages_in_order_and_stops_at_first_missing(tmp_path, monkeypatch, capsys):
    order = []
    for stage in ("registration", "shell"):
        def run(ctx, stage=stage):
            order.append(stage)
            out = ctx.out / STAGE_OUTPUTS[stage]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}")
            return out

        _fake_stage(monkeypatch, stage, run)
    monkeypatch.setitem(cli.STAGE_MODULES, "model", "room_model._not_written")
    assert _run(tmp_path, "all") == 2
    assert order == ["registration", "shell"]
    assert "stage model not implemented" in capsys.readouterr().err


def test_unknown_subcommand_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "nope")
