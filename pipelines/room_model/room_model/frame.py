"""Room frame (transform.worldToRoom) computed by the scan-to-threejs bundle_to_room script."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from room_model.capture import colmajor

BUNDLE_TO_ROOM = Path(__file__).resolve().parents[3] / ".agents/skills/scan-to-threejs/scripts/bundle_to_room.py"


def world_to_room(capture: Path, room: int, turn: int) -> np.ndarray:
    """4x4 ARKit-world to room-frame matrix; the heading logic stays in bundle_to_room.py."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "room.json"
        cmd = [sys.executable, str(BUNDLE_TO_ROOM), str(capture), "--room", str(room), "--turn", str(turn), "-o", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"bundle_to_room.py failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")
        return colmajor(json.loads(out.read_text())["transform"]["worldToRoom"], 4)
