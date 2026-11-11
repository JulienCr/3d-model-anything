"""Command line: one subcommand per stage plus `all`."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from room_model.capture import room_entry
from room_model.context import STAGE_OUTPUTS, BlenderMissing, ClaudeMissing, InvalidCapture, NoPhotos, StageContext, StageInputMissing

STAGE_MODULES = {
    "registration": "room_model.registration",
    "shell": "room_model.shell",
    "model": "room_model.model",
    "export": "room_model.export",
    "openings": "room_model.openings",
    "vision": "room_model.vision",
    "architecture": "room_model.architecture",
    "blender": "room_model.blender_stage",
}


STALE_AFTER = {"architecture": "vision"}  # the stage reruns when the named stage's output is newer than its own


def _skipped(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("skipped") is True
    except (OSError, ValueError, AttributeError):
        return False


def _current(ctx: StageContext, name: str) -> bool:
    """The stage output exists, is not a `--no-llm` placeholder and is not older than the output it merges."""
    out = ctx.out / STAGE_OUTPUTS[name]
    if not out.exists() or (name == "vision" and not ctx.no_llm and _skipped(out)):
        return False
    if name in STALE_AFTER:
        newer = ctx.out / STAGE_OUTPUTS[STALE_AFTER[name]]
        return not (newer.is_file() and newer.stat().st_mtime_ns > out.stat().st_mtime_ns)
    return True


def _stage_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--capture", type=Path, required=True, help="house-capture/5 folder")
    p.add_argument("--out", type=Path, required=True, help="output folder (data/room-model/<room>/)")
    p.add_argument("--room", type=int, default=0)
    p.add_argument("--turn", type=int, default=0, help="extra quarter turns of the room frame")
    p.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    p.add_argument("--keep-work", action="store_true", help="keep intermediate work files")
    p.add_argument("--no-llm", action="store_true", help="skip the vision stage (no claude call)")
    p.add_argument("--llm-model", default="opus", help="model given to claude -p by the vision stage")
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
    if _current(ctx, name) and not ctx.force:
        print(f"stage {name}: {ctx.out / STAGE_OUTPUTS[name]} exists, skipped (use --force)")
        return 0
    module = importlib.import_module(STAGE_MODULES[name])
    try:
        result = module.run(ctx)
    except NoPhotos as exc:
        if in_all:
            print(f"{name} skipped: no photos for room {ctx.room}")
            return 0
        print(str(exc), file=sys.stderr)
        return 1
    except (BlenderMissing, ClaudeMissing) as exc:
        if in_all:
            print(f"{name} skipped: {exc}")
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
    ctx = StageContext(args.capture, args.out, args.room, args.turn, args.force, args.keep_work, args.no_llm, args.llm_model)
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
