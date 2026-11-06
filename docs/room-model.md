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

Misplaced walls (flag `shifted_seed`): a RoomPlan wall can sit away from the real surface. The wall-facing vertical samples inside the wall rectangle within 30 cm of the RoomPlan plane are histogrammed in 1 cm bins; the mode with the most support (at least 0.3 m2 within 4 cm; modes within 10 % of the best tie and the outer one wins, walls being behind furniture) is refitted inside 8 cm. It replaces the fit on the RoomPlan plane only when it is more than 8 cm away and has twice the support, or when the RoomPlan plane has no fit. Walls shorter than about 0.5 m keep the RoomPlan plane: there is nothing to measure.

Low-support walls (support under 0.5 m2, flag `low_support`): the azimuth of a patch that small is not trustworthy, so the wall keeps the RoomPlan azimuth (source `roomplan`, sigma 2 deg). Its offset is the LiDAR fit evaluated at the fit centroid along the RoomPlan normal (source `lidar`), with the support term above dominating the offset sigma. `diagnostics.walls[*].angle_vs_roomplan_deg` still reports the fitted angle.

## Detection rules

- Ceilings: height histogram (window 3 cm) of ceiling-facing LiDAR samples (normal pointing down) above 1.6 m; every level covering at least 0.3 m2 becomes a ceiling, so a patch the LiDAR never saw makes none. Once a level is picked, samples within 3 cm of it are removed, and a sample belongs to its nearest level only. Beams are detected on these unmerged levels. Then touching levels closer than 3 times the larger of their two sigmas merge into one support-weighted level, with a merged span of at most 4 cm: steps of about 3-4 cm or less are absorbed, larger steps stay separate.
- Strip-shaped lower-level components (minor side up to 80 cm, ratio at least 2) get one of five verdicts, in this order:
  - `suspended` (unexplained): ceiling samples are seen above at least 30 % of the strip's cells, so the ceiling continues over it: fixtures, trays and ducts hang below a visible ceiling, a beam hides it.
  - `too_narrow` (unexplained): under 15 cm wide.
  - `too_shallow` (unexplained): it hangs less than 14 cm below its top.
  - Confirmed beam `Beam_<k>`: each long side carries a face (at least 0.05 m2 of steep samples, |normal y| < 0.65 because meshes bevel beam edges, inside the strip grown by 10 cm) whose 95th-percentile top is within 5 cm of a higher level beside that side, or, for one side, a wall within 6 cm.
  - Beam to verify `Beam_<k>` with `flags: ["to_verify"]`: none of the above, so a probable beam whose sides are not both seen. Parameters keep `source: lidar`. `report.md` lists it in `Poutres` (« à vérifier ») and in `Alertes`, `plan.svg` labels it with a trailing « ? », the GLB puts it under the root node `À vérifier` instead of `Coque`, and its dimensions carry the flag.
  The beam top is the adjacent higher level with the most cells within 30 cm of the long sides. The axis spans the 2nd-98th percentile extent along the long axis, padded for the trimmed tails (not the room chord); an end within 10 cm of the floor outline is extended onto it.
- Unexplained horizontal elements go to `diagnostics.unexplained_horizontal` as `{bbox, underside, width, length, reason}` (`suspended`, `too_narrow`, `too_shallow`) and are left out of the ceiling regions; the list is the input of milestone M2 (networks). A strip of the highest level has nothing above it and stays in its region. A duct pressed against the ceiling hides it and is not told from a beam by this test.
- Columns: vertical surface of any class except table, seat, window and door that reaches the ceiling, outside the fitted wall bands, with a plausible size (10-150 cm). A rectangle within 6 cm of a wall line (whichever side the wall faces) is engaged on it, on up to two non-parallel walls (a corner column); engaged sides are snapped to the wall and only the remaining sides must be seen. `engaged_wall` is the first engaged wall in outline order and `diagnostics.columns` lists all of them; a cabinet not reaching the ceiling is listed as an unexplained vertical surface.
- Wall top: the highest ceiling level lying over at least 0.30 m of the wall, sampled every 10 cm and 15 cm inside the room (a wall rises to the highest ceiling it meets; a shorter corner clip does not count); otherwise the wall keeps its RoomPlan top.
- Stations: opposite wall pairs measured every metre. A station is accepted when the LiDAR width at three heights agrees within 3 cm and is within 7 cm (12 cm for lengths) of the width between the fitted lines at that position, searched within +-0.75 m. Positions whose line crosses a door span (door width + 10 cm) on either wall are skipped, and a station that finds no position outside the door is dropped; ids keep the numbering of the unfiltered candidates, so a dropped station leaves a gap. A candidate with no clean position stays, flagged `obstacle_or_uneven_wall`.
- Walls are never moved by faces of an earlier room (`inroom == 0`); RoomPlan walls off the floor outline are listed in `diagnostics.off_outline_walls`.

## Outputs

Everything below is written under `--out`. Coordinates are in metres in the room frame (+Y up, plan `[x, z]`).

### `model_measured.glb`

- Root nodes: `Coque` (extras `kind: "shell"`, `schema`, `version`) holds the elements of the shell; `À vérifier` (extras `kind: "to_verify"`) exists only when an element carries the flag `to_verify`, and holds those elements instead.
- One mesh node per element, named by its element id: `Floor`, `Wall_M1`, `Ceiling_1`, `Beam_1`, `Column_1`. Walls are prisms behind the interior face, extended past their end by their thickness. A beam or column `top` is the ceiling level it meets, copied from the ceiling, not measured on its own.
- Node `extras`: `kind` (`floor`, `wall`, `ceiling`, `beam`, `column`), `params` (per parameter name: `{id, value, sigma, source}`) and `flags` (list).

### `model.json`

`model.json` is `shell.json` (schema `room-model/shell-1`) with `schema: "room-model/1"`, `version` and a `dimensions` list. Each entry: `id`, `kind`, `label` (French), `value` and `sigma` (m), `elements` (element ids such as `Wall_M1`), `at` (`[x, z]` or `null`), `source` (the weakest source among the parameters used: `assumed` < `roomplan` < `lidar`) and `flags` (list).

| Id | Kind | Meaning |
|---|---|---|
| `S01`, `S02`, ... | `width`, `length` | Station between two opposite walls; numbering follows the unfiltered candidates, so a dropped station leaves a gap |
| `H1`, `H2`, ... | `ceiling_height` | Floor to ceiling level `k` |
| `B1-u`, `B1-w` | `beam_underside`, `beam_width` | Floor to beam underside; beam width |
| `COL1-x`, `COL1-z` | `column_size` | Column size along x or z |
| `COL1-dx`, `COL1-dz` | `column_offset` | Gap between a column face and the nearest wall that does not engage it |

`engaged_wall` of a column holds a wall label (`M1`), and so does `diagnostics.columns[*].engaged_walls`; `elements` always hold element ids (`Wall_M1`).

### `dimensions.csv`

Columns: `id`, `kind`, `label`, `value_cm`, `sigma_cm`, `source`, `elements` (ids joined with `;`), `flags` (flags joined with `;`, empty when there are none).

### Flags

| Where | Flag | Meaning |
|---|---|---|
| Wall `flags` | `roomplan_fallback` | No LiDAR fit, RoomPlan plane kept |
| | `shifted_seed` | Surface found away from the RoomPlan plane |
| | `low_support` | Support under 0.5 m2, RoomPlan azimuth kept |
| | `seeded_from_outline` | Wall absent from RoomPlan, built from the floor outline |
| | `near_parallel_corner` | Corner with a near-parallel neighbour, obtained by projection |
| Beam `flags` and its dimensions | `to_verify` | Probable beam whose sides are not both seen |
| Station dimension `flags` | `obstacle_or_uneven_wall` | No clean wall section found for the station; measure with caution (`~` in `plan.svg`) |
| `diagnostics.flags` | `floor:roomplan_fallback`, `columns:no_ceiling` | Floor level not seen; no ceiling, so no columns |
| | `<label>:<wall flag>` | Wall flag repeated per wall, for example `M3:shifted_seed` |

`plan.svg` is the dimensioned top view and `report.md` the French report: inputs, registration, per-wall table, floor, ceilings, beams, columns, unexplained surfaces, orthogonality, alerts.

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
- Ducts or trays pressed against the ceiling hide it, so they look like beams.
- Soffits that drop less than 14 cm below the neighbouring ceiling are not reported as beams.
- A narrow ceiling region can be classed as an unexplained strip and lower a wall top.
- Photo silhouettes (M3) and the site survey (M4) are expected to settle these cases.
