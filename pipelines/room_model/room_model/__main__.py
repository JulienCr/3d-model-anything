"""Command line: one subcommand per stage plus `all`."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from room_model.capture import room_entry
from room_model.context import STAGE_OUTPUTS, BlenderMissing, InvalidCapture, NoPhotos, StageContext, StageInputMissing

STAGE_MODULES = {
    "registration": "room_model.registration",
    "shell": "room_model.shell",
    "model": "room_model.model",
    "export": "room_model.export",
    "openings": "room_model.openings",
    "architecture": "room_model.architecture",
    "blender": "room_model.blender_stage",
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
    cmp = sub.add_parser("compare", help="compare a model.json against a reference")
    cmp.add_argument("--model", type=Path, required=True)
    cmp.add_argument("--reference", type=Path, required=True)
    cmp.add_argument("-o", "--output", type=Path, help="write Markdown here instead of stdout")
    tp = sub.add_parser("compare-tape", help="compare tape measurements against an architecture.json")
    tp.add_argument("--arch", type=Path, required=True)
    tp.add_argument("--tape", type=Path, required=True, help='JSON {"<dimension id>": <metres>}')
    tp.add_argument("-o", "--output", type=Path, help="write Markdown here instead of stdout")
    return parser


def _run_stage(name: str, ctx: StageContext, in_all: bool = False) -> int:
    out = ctx.out / STAGE_OUTPUTS[name]
    if out.exists() and not ctx.force:
        print(f"stage {name}: {out} exists, skipped (use --force)")
        return 0
    module = importlib.import_module(STAGE_MODULES[name])
    try:
        result = module.run(ctx)
    except NoPhotos as exc:
        if in_all:
            print(f"registration skipped: no photos for room {ctx.room}")
            return 0
        print(str(exc), file=sys.stderr)
        return 1
    except BlenderMissing as exc:
        if in_all:
            print(f"blender skipped: {exc}")
            return 0
        print(str(exc), file=sys.stderr)
        return 1
    except StageInputMissing as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"stage {name}: wrote {result}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from room_model.compare import InvalidReference, compare

    docs = []
    for label, path in (("modèle", args.model), ("référence", args.reference)):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"{label} illisible ({path}) : {exc}", file=sys.stderr)
            return 1
    try:
        report = compare(*docs)
    except InvalidReference as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"compare: wrote {args.output}")
    else:
        print(report)
    return 0


def _compare_tape(args: argparse.Namespace) -> int:
    from room_model.tape import compare

    docs = []
    for label, path in (("architecture", args.arch), ("mesures", args.tape)):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"{label} illisible ({path}) : {exc}", file=sys.stderr)
            return 1
    report = compare(*docs)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"compare-tape: wrote {args.output}")
    else:
        print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        return _compare(args)
    if args.command == "compare-tape":
        return _compare_tape(args)
    ctx = StageContext(args.capture, args.out, args.room, args.turn, args.force, args.keep_work)
    try:
        room_entry(ctx.capture, ctx.room)
    except InvalidCapture as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for name in STAGE_MODULES if args.command == "all" else [args.command]:
        code = _run_stage(name, ctx, in_all=args.command == "all")
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
