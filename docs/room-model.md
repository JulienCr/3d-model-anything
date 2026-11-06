# Room model pipeline (CPU, Python)

Turns a House Capture export (`house-capture/5`) into a measured architectural model of one room: walls, floor, ceilings, beams and columns, with a sigma on every measurement. No GPU, no torch, no open3d. Milestone M1.

## Setup

```bash
uv venv --python 3.12 pipelines/room_model/.venv
uv pip install --python pipelines/room_model/.venv/bin/python -r pipelines/room_model/requirements.txt
```

The venv is about 0.4 GB and git-ignored. Tests:

```bash
cd pipelines/room_model && .venv/bin/python -m pytest -q
```

## Commands

Run from `pipelines/room_model/`:

```bash
.venv/bin/python -m room_model <registration|shell|model|export|all> \
    --capture <capture dir> --out <output dir> [--room 0] [--turn 0] [--force] [--keep-work]
.venv/bin/python -m room_model compare --model <model.json> --reference <reference.json> [-o compare.md]
```

| Stage | Reads | Writes (under `--out`) |
|---|---|---|
| `registration` | photos, ARKit poses | `registration/poses.json` (ARKit frame) |
| `shell` | RoomPlan, LiDAR PLY | `shell.json` |
| `model` | `shell.json` | `model.json` (shell plus `dimensions`) |
| `export` | `model.json`, optional poses | `model_measured.glb`, `plan.svg`, `dimensions.csv`, `report.md` |

`all` runs the four stages in order. A stage whose output exists is skipped unless `--force`. `--room` picks the room of a multi-room capture, `--turn` adds quarter turns to the room frame. Keep outputs under `data/room-model/<room>/` (git-ignored, below 500 MB per room).

## Frame

Shell, model and export use the room frame of `bundle_to_room.py` (`transform.worldToRoom`): metres, +Y up, plan coordinates `[x, z]`. A wall's interior normal is `(cos azimuth, sin azimuth)` and its line is `n . (x, z) = offset`. Registration stays in the ARKit frame.

## Sigma model

Every parameter carries `value`, `sigma`, `unit` and `source` (`lidar`, `roomplan` or `assumed`).

| Quantity | Sigma |
|---|---|
| Plane offset | `sqrt((rms / sqrt(support_m2 / 0.01))^2 + SIGMA_SYS_OFFSET^2 + (SIGMA_SUPPORT / sqrt(support_m2))^2)` |
| Plane azimuth | `hypot(rms / (sqrt(support_m2 / 0.01) * max(along_std, 0.05)), SIGMA_SYS_AZIMUTH)` |
| Wall line at distance `s` from its fit centroid | `hypot(sigma_offset, sigma_azimuth * s)` |
| Derived dimension | Propagated from the parameters it uses |

Constants: `SIGMA_SYS_OFFSET = 0.01 m`, `SIGMA_SYS_AZIMUTH = 0.25 deg`, `SIGMA_SUPPORT = 0.014 m` (the support term: 2 cm at 0.5 m2, 4.5 cm at 0.1 m2, below 0.5 cm from 8 m2). When a wall has too little LiDAR support, the RoomPlan plane is kept with a fallback sigma of 5 cm and 2 deg (flag `roomplan_fallback`). Wall thickness is assumed: 20 cm, sigma 5 cm.

Low-support walls (support under 0.5 m2, flag `low_support`): the azimuth of a patch that small is not trustworthy, so the wall keeps the RoomPlan azimuth (source `roomplan`, sigma 2 deg). Its offset is the LiDAR fit evaluated at the fit centroid along the RoomPlan normal (source `lidar`), with the support term above dominating the offset sigma. `diagnostics.walls[*].angle_vs_roomplan_deg` still reports the fitted angle.

## Detection rules

- Ceilings: height histogram of upward-facing LiDAR samples above 1.6 m; every level covering at least 0.3 m2 becomes a ceiling, so a patch the LiDAR never saw makes none.
- Beams: a lower-level region that is long and narrow (minor side 8-80 cm, ratio at least 3) with a higher level next to a long side. The axis spans the component's own extent along its long axis (not the room chord); an end within 10 cm of the floor outline is extended onto it.
- Columns: vertical wall-class surface that reaches the ceiling, outside the fitted wall bands, with a plausible size (10-150 cm). A rectangle within 6 cm of a wall line (whichever side the wall faces) is engaged on it, on up to two non-parallel walls (a corner column); engaged sides are snapped to the wall and only the remaining sides must be seen. `engaged_wall` is the first engaged wall in outline order and `diagnostics.columns` lists all of them; a cabinet not reaching the ceiling is listed as an unexplained vertical surface.
- Walls are never moved by faces of an earlier room (`inroom == 0`); RoomPlan walls off the floor outline are listed in `diagnostics.off_outline_walls`.

## Comparing against a reference

`compare` checks a `model.json` against a reference from any source (a hand-built model, an earlier evaluation, tape measurements). Every key is optional, metres, room frame:

```json
{"polygon": [[x, z], ...], "edge_heights": [2.5, ...], "dimensions": {"S01": 2.5}}
```

Edge `i` runs from `polygon[i]` to `polygon[i+1]`; `edge_heights[i]` is its height above the floor. Each model wall is paired with the reference edge within 5 deg and 0.5 m whose midpoint is nearest. The Markdown report (French, differences are model minus reference) holds:

- a wall table: wall, reference edge, offset difference (cm), angle difference (deg), wall-top difference (cm, with `edge_heights`);
- the reference edges no wall was paired with (a pillar the model lacks, for example);
- a dimension table for shared ids: model, reference, difference (cm) and difference in sigmas.

## Limits of M1

- No openings (doors, windows) and no wall thickness from jambs: thickness is assumed.
- Registration poses are written but not yet consumed by the other stages.
- One room per run, no networks (pipes, ducts), no squared variant, no `room.json`.
