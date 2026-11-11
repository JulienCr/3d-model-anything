"""Shared stage context and output locations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STAGE_OUTPUTS = {
    "registration": "registration/poses.json",
    "shell": "shell.json",
    "model": "model.json",
    "export": "model_measured.glb",
    "openings": "openings.json",
}


class StageInputMissing(RuntimeError):
    """A stage was asked to run before the stage producing its input."""


class NoPhotos(StageInputMissing, ValueError):
    """The room has no photos, so registration has nothing to refine."""


class InvalidCapture(StageInputMissing):
    """The capture folder or the requested room cannot be read."""


@dataclass(frozen=True)
class StageContext:
    capture: Path
    out: Path
    room: int = 0
    turn: int = 0
    force: bool = False
    keep_work: bool = False


def output_path(ctx: StageContext, stage: str) -> Path:
    return ctx.out / STAGE_OUTPUTS[stage]


def require(ctx: StageContext, stage: str) -> Path:
    """Output of an earlier stage; raises StageInputMissing('run <stage> first: <path>') if absent."""
    path = output_path(ctx, stage)
    if not path.exists():
        raise StageInputMissing(f"run {stage} first: {path}")
    return path
