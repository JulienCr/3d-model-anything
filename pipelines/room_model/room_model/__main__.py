"""Command line: one subcommand per stage plus `all`."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from room_model.context import STAGE_OUTPUTS, StageContext, StageInputMissing

STAGE_MODULES = {
    "registration": "room_model.registration",
    "shell": "room_model.shell",
    "model": "room_model.model",
    "export": "room_model.export",
}


def _stage_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--capture", type=Path, required=True, help="house-capture/5 folder")
    p.add_argument("--out", type=Path, required=True, help="output folder (data/room-model/<room>/)")
    p.add_argument("--room", type=int, default=0)
    p.add_argument("--turn", type=int, default=0, help="extra quarter turns of the room frame")
    p.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    p.add_argument("--keep-work", action="store_true", help="keep intermediate work files")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="room_model")
    sub = parser.add_subparsers(dest="command", required=True)
    parent = _stage_parent()
    for name in (*STAGE_MODULES, "all"):
        sub.add_parser(name, parents=[parent], help=f"run {name}")
    return parser


def _run_stage(name: str, ctx: StageContext) -> int:
    out = ctx.out / STAGE_OUTPUTS[name]
    if out.exists() and not ctx.force:
        print(f"stage {name}: {out} exists, skipped (use --force)")
        return 0
    module_name = STAGE_MODULES[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        print(f"stage {name} not implemented", file=sys.stderr)
        return 2
    try:
        result = module.run(ctx)
    except StageInputMissing as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"stage {name}: wrote {result}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = StageContext(args.capture, args.out, args.room, args.turn, args.force, args.keep_work)
    for name in STAGE_MODULES if args.command == "all" else [args.command]:
        code = _run_stage(name, ctx)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
