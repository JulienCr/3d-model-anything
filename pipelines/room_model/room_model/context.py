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
    "vision": "vision/answers.json",
    "anchor": "anchors.json",
    "edge_pick": "edge_pick/answers.json",
    "architecture": "architecture.json",
    "textures": "textures/textures.json",
    "blender": "architecture.blend",
}


class StageInputMissing(RuntimeError):
    """A stage was asked to run before the stage producing its input."""


class BlenderMissing(StageInputMissing):
    """The Blender executable named by BLENDER does not exist."""


class NoPhotos(StageInputMissing, ValueError):
    """The room has no photos, so a photo stage (registration, vision) has nothing to work on."""


class ClaudeMissing(StageInputMissing):
    """The `claude` executable (or $ROOM_MODEL_CLAUDE) does not exist."""


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
    no_llm: bool = False
    llm_model: str = "opus"


def output_path(ctx: StageContext, stage: str) -> Path:
    return ctx.out / STAGE_OUTPUTS[stage]


def require(ctx: StageContext, stage: str) -> Path:
    """Output of an earlier stage; raises StageInputMissing('run <stage> first: <path>') if absent."""
    path = output_path(ctx, stage)
    if not path.exists():
        raise StageInputMissing(f"run {stage} first: {path}")
    return path
