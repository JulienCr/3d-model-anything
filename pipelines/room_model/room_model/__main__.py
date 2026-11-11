"""Command line: one subcommand per stage plus `all`."""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
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
    "anchor": "room_model.anchor",
    "architecture": "room_model.architecture",
    "textures": "room_model.textures",
    "blender": "room_model.blender_stage",
}


STALE_AFTER = {"anchor": ("openings.json", "vision/answers.json", "overrides.json"),
               "architecture": ("anchors.json", "vision/answers.json", "measurements.json", "overrides.json"), "textures": ("architecture.json",),
               "blender": ("architecture.json", "textures/textures.json")}  # the stage reruns when one of these is newer than its output
TAPE_FILE, OVERRIDES_FILE = "measurements.json", "overrides.json"


def _placeholder(path: Path) -> str | None:
    """`skipped` for the `--no-llm` placeholder, `error` for the record of a failed call, else None."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    return "skipped" if doc.get("skipped") is True else "error" if "error" in doc else None


def _current(ctx: StageContext, name: str) -> bool:
    """The stage output exists, holds an answer (`--no-llm` accepts its own placeholder) and is not older than the output it merges."""
    out = ctx.out / STAGE_OUTPUTS[name]
    if not out.exists():
        return False
    if name == "vision" and (kind := _placeholder(out)) and (kind == "error" or not ctx.no_llm):
        return False
    newer = [ctx.out / rel for rel in STALE_AFTER.get(name, ())]
    return not any(p.is_file() and p.stat().st_mtime_ns > out.stat().st_mtime_ns for p in newer)


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
        stage = sub.add_parser(name, parents=[parent], help=f"run {name}")
        if name in ("architecture", "all"):
            stage.add_argument("--tape", type=Path, help='tape measurements {"<dimension id>": <metres>}, copied to <out>/measurements.json')
            stage.add_argument("--overrides", type=Path, help='manual corrections {"<element id>": {field: value}}, copied to <out>/overrides.json')
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


def _install_json(source: Path, out: Path, name: str, label: str) -> str | None:
    """Copies a user JSON object to <out>/<name>; an error message when it cannot be used."""
    try:
        doc = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"{label} illisibles ({source}) : {exc}"
    if not isinstance(doc, dict):
        return f"{label} illisibles ({source}) : un objet JSON est attendu"
    target = out / name
    out.mkdir(parents=True, exist_ok=True)
    if not (target.exists() and target.samefile(source)):
        shutil.copyfile(source, target)
    return None


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
    for option, name, label in (("tape", TAPE_FILE, "mesures"), ("overrides", OVERRIDES_FILE, "corrections")):
        if getattr(args, option, None) and (problem := _install_json(getattr(args, option), ctx.out, name, label)):
            print(problem, file=sys.stderr)
            return 1
    for name in STAGE_MODULES if args.command == "all" else [args.command]:
        code = _run_stage(name, ctx, in_all=args.command == "all")
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
