# Room model pipeline (CPU, Python)

Turns a House Capture export (`house-capture/5`) into a measured architectural model of one room: walls, floor, ceilings, beams and columns, with a sigma on every measurement. Milestone M1 measures the shell; milestone M1.5 adds openings, photo-anchored fixture edges, a vision pass, an architecture description and a Blender scene. `docs/specs/room-model-pipeline.md` holds the decisions.

## Setup

```bash
uv venv --python 3.12 pipelines/room_model/.venv
uv pip install --python pipelines/room_model/.venv/bin/python -r pipelines/room_model/requirements.txt
```

The pipeline needs an NVIDIA GPU: torch and SAM 2.1 (photo anchoring) are regular dependencies of `requirements.txt`, so the venv is several GB and git-ignored. Stages up to `export` run on CPU. Tests:

```bash
cd pipelines/room_model && .venv/bin/python -m pytest -q
```

Other requirements:

- Blender 5.1+ for the `blender` stage. `BLENDER` names the executable (default: the Steam install). From WSL it can be a Windows `.exe`; paths are converted with `wslpath`.
- The `claude` CLI for the `vision` stage (`claude -p`, no API key). Without it, or with `--no-llm`, the stage is skipped.

## Commands

Run from `pipelines/room_model/`:

```bash
.venv/bin/python -m room_model <stage|all> --capture <capture dir> --out <output dir> \
    [--room 0] [--turn 0] [--force] [--keep-work] [--no-llm] [--llm-model opus] [--tape measurements.json] [--overrides overrides.json]
.venv/bin/python -m room_model compare --model <model.json> --reference <reference.json> [-o compare.md]
.venv/bin/python -m room_model compare-tape --arch <architecture.json> --tape <measurements.json> [-o tape.md]
```

Stages, in `all` order:

| Stage | Reads | Writes (under `--out`) |
|---|---|---|
| `registration` | photos, ARKit poses | `registration/poses.json` (ARKit frame) |
| `shell` | RoomPlan, LiDAR PLY | `shell.json` |
| `model` | `shell.json` | `model.json` (shell plus `dimensions`) |
| `export` | `model.json`, optional poses | `model_measured.glb`, `plan.svg`, `dimensions.csv`, `report.md` |
| `openings` | `model.json`, RoomPlan, LiDAR | `openings.json`: doors, windows, openings hosted on walls and partitions |
| `vision` | `openings.json`, photos | `vision/pending/` (crops, prompt, schema), `vision/answers.json` |
| `anchor` | `openings.json`, `vision/answers.json`, poses, photos | `anchors.json`: fixture edges anchored in the photos (SAM 2.1) |
| `architecture` | model, openings, anchors, vision answers, `measurements.json`, `overrides.json` | `architecture.json` (`room-model/arch-1`), `measurements/index.html`, tape report |
| `textures` | `architecture.json`, photos, LiDAR mesh | `textures/*.png`, `textures/textures.json` |
| `blender` | `architecture.json`, textures, LiDAR mesh | `architecture.blend`, `architecture.glb`, `renders/*.png`, `blender_report.json` |

A stage whose output exists is skipped unless `--force`. `anchor`, `architecture`, `textures` and `blender` also rerun when an input they merge (`vision/answers.json`, `measurements.json`, `overrides.json`, `architecture.json`, textures) is newer than their output. In `all`, a stage with no photos, or with `claude` or Blender missing, is skipped with a message; run alone it fails.

- `--room` picks the room of a multi-room capture, `--turn` adds quarter turns to the room frame.
- `--no-llm` writes a placeholder `vision/answers.json`: the architecture is built from RoomPlan types and scan values. A later run without the flag redoes the stage. `--llm-model` is passed to `claude -p`.
- `--tape` and `--overrides` (on `architecture` and `all`) copy a file to `<out>/measurements.json` or `<out>/overrides.json`.
- `compare-tape` compares tape values with an `architecture.json`.

Keep outputs under `data/room-model/<room>/` (git-ignored, below 500 MB per room).

## Files you edit

Both live in `--out` and are merged by `architecture` (priority: tape > LLM > LiDAR > RoomPlan > assumed). Editing one reruns the stages that read it.

- `measurements.json`: tape values in metres, per dimension id. Written by the survey page (legacy and v2 formats: see the survey page); a value moves the scan: an opening dimension replaces the parameter it measures, a ceiling height moves the ceiling level, a station moves the wall of its pair with the larger offset sigma (bounded). Values that cannot be applied are listed in the report, never applied.
- `overrides.json`: `{"<element id>": {field: value}}`, applied last. Fields: `kind` (door, niche and opening convert between each other), `type`, `hinge`, `swing`, `panels`, `handle`, `frame`, `through`, `dividers`, `shelves_count` (niches only), `remove` (`true` deletes an opening, panel frame, cornice or skirting). Values are checked against the vision answer schema and the Blender catalogue; a rejected field is listed, the rest of the element is applied.

## Survey page

`measurements/index.html` is a local, self-contained page, never published. A numbered SVG plan, one row per measure (what and where to tape, photo crop with the endpoints drawn, current value with source and sigma, priority badge, input in cm; the badge says leaf, both leaves together, passage or niche), and a save of `measurements.json` (linked file with autosave when the browser allows it, download and import otherwise). Rerun `architecture` (or `all`) after saving. `compare-tape` and the tape report compare scan, LLM and tape values.

## Blender scene

`blender/build_scene.py` builds the scene from `architecture.json` only. Walls are mitred at corners, partitions end on the face they meet, an opening cuts every wall it crosses. Cornice, skirting and architraves come from a small profile catalogue (heights from the LLM, source `assumed`).

- Collections: `Coque`, `Menuiseries`, `Moulures`, `Cotes`, `À vérifier`, `Scan` (decimated LiDAR mesh, hidden).
- Each door and window has an empty parent with custom properties: `Ouverture` (0-1), `Inverser sens` (flips the swing), `Inverser charnière` (hinges on the other jamb), `AngleMax` (degrees, default 90). Checkboxes unchecked = as detected. A double door has one slider for both leaves. Sliding leaves translate.
- Dimensions are MeasureIt segments in `Cotes`; in the Blender UI, press the MeasureIt "Show" button to display them. If MeasureIt is unavailable, native objects are used (`blender_report.json` names the backend).
- `architecture.glb` is static, doors and windows closed; `Cotes` and `Scan` are not exported. `model_measured.glb` stays the M1 output.

## Textures

Walls, floor and ceilings are textured from the posed photos at 2 cm per texel. A texel is kept when a photo sees it facing the camera, its depth agrees with the LiDAR mesh and it lies in no RoomPlan object box, so furniture is not printed. The colour is the weighted median of the best views after exposure matching. Texels no photo covers (coverage holes) are filled with a CC0 texture (`fetch_textures.py`) tinted to the measured colour, feathered into the observed colour. Joinery and mouldings use plain named materials. `textures/textures.json` lists the surfaces and their coverage.

## Contributing: generic for any capture

Nothing in code, tests or docs may depend on one capture: no wall labels, counts, dimensions or room names from a real room. Behaviour comes from the capture data, the catalogue and the schema; synthetic fixtures (`tests/synthetic.py`) stand in for rooms. Real-capture values stay in git-ignored outputs.

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

## Limits

- Wall thickness is assumed unless a jamb is measured.
- Open doors are not anchored.
- A curtained or frosted window defeats the mask: it needs the pick step (the LLM picks a lettered candidate edge), or a tape value.
- Anchoring and textures need photos; without them the scan and RoomPlan values stand.
- The LLM can only choose within the catalogue; a fixture it lacks is typed by `overrides.json`.
- One room per run, no networks (pipes, ducts), no squared variant, no `room.json`.
- Ducts or trays pressed against the ceiling hide it, so they look like beams.
- Soffits that drop less than 14 cm below the neighbouring ceiling are not reported as beams.
- A narrow ceiling region can be classed as an unexplained strip and lower a wall top.
- Networks (M2), photo silhouettes (M3) and the site survey (M4) are expected to settle the remaining cases.
