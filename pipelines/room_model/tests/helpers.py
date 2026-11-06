"""Shared helpers for tests that read a shell document built from a synthetic capture."""
from __future__ import annotations

import numpy as np

from room_model import shell
from room_model.context import StageContext
from room_model.frame import world_to_room
from synthetic import SyntheticOptions, make_capture


def make_case(root, name, **options):
    capture = root / name
    truth = make_capture(capture, SyntheticOptions(**options))
    return StageContext(capture, root / f"{name}-out"), truth


def wall_labels(doc, truth, name):
    ids = truth.wall_ident[name]
    return [e["label"] for e in doc["elements"] if e["kind"] == "wall" and e["roomplan_id"] in ids]


def wall_line(doc, label):
    return next(w for w in shell.wall_lines(doc) if w["label"] == label)


def truth_to_room(ctx, truth):
    return world_to_room(ctx.capture, ctx.room, ctx.turn) @ truth.world_from_local


def plan_point(T, x, z):
    """Truth-local (x, z) on the floor plane mapped to the room frame (x, z)."""
    return (T @ [x, 0.0, z, 1.0])[[0, 2]]
