Guide de l'opérateur (HTML, du scan à Blender) : [room-model-guide.html](room-model-guide.html)

# Room model pipeline (CPU, Python, GPU for photo anchoring)

Turns a House Capture export (`house-capture/5`) into a measured architectural model of one room: squared walls and partitions, floor, ceilings, beams and columns, then doors, windows, openings, niches and mouldings, with a sigma on every measurement. It also writes a Blender scene, a tape-measure survey page and a reconciliation of the survey with the model. Milestone M1 measures the shell; milestone M1.5 adds the openings, photo-anchored fixture edges, the vision passes, the architecture description and the Blender scene. `docs/specs/room-model-pipeline.md` holds the decisions.

## Setup

```bash
uv venv --python 3.12 pipelines/room_model/.venv
uv pip install --python pipelines/room_model/.venv/bin/python -r pipelines/room_model/requirements.txt
```

The pipeline wants an NVIDIA GPU: torch (CUDA wheels) and SAM 2.1 (`facebook/sam2.1-hiera-large`, photo anchoring in `anchor`) are regular dependencies of `requirements.txt`, so the venv is several GB and git-ignored. The first `anchor` run downloads the SAM weights from Hugging Face (network needed once). Without CUDA `anchor` still runs, on the CPU, far slower. Stages up to `export` need neither GPU nor network. Tests:

```bash
cd pipelines/room_model && .venv/bin/python -m pytest -q
```

Other requirements:

- Blender 5.1+ for the `blender` stage. `BLENDER` names the executable; default: the Steam install under `/mnt/c/Program Files (x86)/Steam/`. From WSL it can be a Windows `.exe`: the paths given to it are converted with `wslpath -w`. A Windows `.exe` only sees the environment variables listed in `WSLENV`: the stage adds `ROOM_MODEL_NO_MEASUREIT` and nothing else, so any other variable meant for Blender has to be added to `WSLENV` by hand. `ROOM_MODEL_NO_MEASUREIT=1` forces native dimension objects instead of MeasureIt.
- The `claude` CLI for `vision` and `edge_pick` (`claude -p --model <llm-model> --output-format json --allowedTools Read`, no API key; a call is cut after 15 minutes). `ROOM_MODEL_CLAUDE` names another executable. Without it, or with `--no-llm`, those stages write a placeholder (or are skipped in `all`) and the architecture is built without LLM answers.
- Photos for `registration`, `vision`, `anchor`, `edge_pick` and `textures`: a room without photos fails these stages when run alone and skips them in `all`.

## Commands

Run from `pipelines/room_model/`:

```bash
.venv/bin/python -m room_model <stage|all> --capture <capture dir> --out <output dir> \
    [--room 0] [--turn 0] [--force] [--keep-work] [--no-llm] [--llm-model opus] [--tape measurements.json] [--overrides overrides.json]
.venv/bin/python -m room_model compare --model <model.json> --reference <reference.json> [-o compare.md]
.venv/bin/python -m room_model compare-tape --arch <architecture.json> --tape <measurements.json> [-o tape.md]
```

Stages, in `all` order (`STAGE_OUTPUTS` in `context.py` names the file that says a stage is done, the first of its outputs below):

| Stage | Reads | Writes (under `--out`) |
|---|---|---|
| `registration` | photos, ARKit poses | `registration/poses.json` (ARKit frame) |
| `shell` | RoomPlan, LiDAR PLY | `shell.json`: scan walls, partitions, floor, scan ceiling regions, beams, columns |
| `model` | `shell.json`, RoomPlan outline, LiDAR | `model.json`: the shell squared, floor and ceilings rebuilt from it, plus `dimensions` |
| `export` | `model.json`, optional poses | `model_measured.glb`, `plan.svg`, `dimensions.csv`, `report.md` |
| `openings` | `model.json`, RoomPlan, LiDAR | `openings.json`: doors, windows, openings hosted on walls and partitions |
| `vision` | `openings.json` (or `architecture.json` when present), photos | `vision/pending/` (crops, prompt, schema), `vision/answers.json`, `vision/rejected.json` |
| `anchor` | `openings.json`, `vision/answers.json`, `overrides.json`, poses, photos | `anchors.json`: fixture edges anchored in the photos (SAM 2.1) |
| `edge_pick` | `anchors.json`, photos | `edge_pick/pending/`, `edge_pick/answers.json`, `edge_pick/rejected.json`: edges the LLM picked among lettered candidates |
| `architecture` | model, openings, anchors, edge picks, vision answers, `measurements.json`, `overrides.json` | `architecture.json` (`room-model/arch-1`), `reconcile.json`, `reconcile.md`, `measurements/index.html`, the sections `Corrections LLM`, `Mesures au mètre` and `Corrections manuelles` of `report.md` |
| `textures` | `architecture.json`, photos, LiDAR mesh | `textures/*.png`, `textures/textures.json` |
| `blender` | `architecture.json`, textures, LiDAR mesh | `architecture.blend`, `architecture.glb`, `renders/{top,persp_1,persp_2,open}.png`, `blender_report.json`, `blender.log`, `scan_decimated.ply` |

A stage whose output exists is skipped unless `--force`. Some stages also rerun when an input they merge is newer than their output (`STALE_AFTER` in `__main__.py`, by file modification time):

| Stage | Reruns when newer |
|---|---|
| `anchor` | `openings.json`, `vision/answers.json`, `overrides.json` |
| `edge_pick` | `anchors.json` |
| `architecture` | `anchors.json`, `edge_pick/answers.json`, `vision/answers.json`, `measurements.json`, `overrides.json` |
| `textures` | `architecture.json` |
| `blender` | `architecture.json`, `textures/textures.json` |

Neither a changed capture nor changed code makes anything stale: use `--force` on the stage. `vision` and `edge_pick` are also redone, without `--force`, when their answer file holds only the record of a failed call (`error`), or the `--no-llm` placeholder (`{"skipped": true}`) and the flag is now off. In `all`, a stage with no photos, or with `claude` or Blender missing, is skipped with a message; run alone it fails.

- `--room` picks the room of a multi-room capture, `--turn` adds quarter turns to the room frame, `--keep-work` keeps the work files of `registration` (deleted at the end otherwise).
- `--no-llm` writes the placeholder `{"skipped": true}` in `vision/answers.json` and `edge_pick/answers.json`: the architecture is built from RoomPlan types, scan values and the photo anchors. `--llm-model` (default `opus`) is passed to `claude -p`.
- `--tape` and `--overrides` (on `architecture` and `all` only) check that the file is a JSON object and copy it to `<out>/measurements.json` or `<out>/overrides.json`; a file already there is left alone.
- `compare-tape` compares tape values with an `architecture.json`; it reads both tape file formats.

Keep outputs under `data/room-model/<room>/` (git-ignored, below 500 MB per room).

## Architecture

```
registration -> shell -> model -> export          (scan: M1, no photo needed except registration)
                           \-> openings -> vision -> anchor -> edge_pick -> architecture -> textures -> blender
                                                                  ^ measurements.json, overrides.json (you)
```

`architecture` is the only place sources are merged; `blender` reads `architecture.json` and nothing else (the Blender package runs in Blender's Python and does not import `room_model`). Every parameter carries `value`, `sigma`, `unit`, `source`.

Source priority, the higher replaces the lower: `tape` > `photo` > `llm` > `lidar` > `roomplan` > `assumed` (`SOURCE_RANK` in `model.py`). Merge order in `architecture.build`: scan values, LLM answers (`llm_merge`), photo anchors (`anchor`, completed by `edge_pick`), tape (`tape_merge`), overrides (`overrides`, which can send a parameter back to its scan value with `prefer`). Rules that are not a plain replacement:

- A door or window gets its photo measure in separate `leaf_width` / `leaf_height` / `leaf_center_s` parameters (the moving panel); `width` / `height` stay the extent of the hole. A niche or an opening takes its photo measure in place of the lower sources.
- A photo value that disagrees with a measured source (`roomplan`, `lidar`) by more than 3 combined sigmas is kept but the element gets `to_verify`. A photo leaf that cannot fit the scanned hole (leaf plus its visible jambs) is refused: its pair of edges is kept out, the reason is in `diagnostics.anchor.reasons` and the element gets `to_verify`. A tape leaf is applied; `reconcile.json` then reports a `leaf_fits` conflict.
- An element the photos measured whole ignores the corners the LLM clicked; its typing (type, hinge, swing...) is still copied.
- The tape sigma is fixed at 0.5 cm. A tape value that cannot be applied is listed in `diagnostics.tape.not_applied`, never applied.

Sigma propagation: wall planes get their sigma from the LiDAR fit (see Sigma model); a derived dimension takes its sigma from the parameters it uses (a station or a wall length: the sigmas of the wall lines it spans, at the measuring points or corners); a dimension's `source` is the weakest source among the parameters it uses. The LLM's corner clicks get the spread of the crops that agree (at least 1 cm). A moved wall keeps its scan value and sigma in `scan_value` / `scan_sigma` next to the tape ones.

Where each rule lives (`pipelines/room_model/room_model/`):

| Module | Rule |
|---|---|
| `capture.py`, `context.py`, `frame.py`, `lidar.py`, `roomplan.py` | Readers of the capture, stage context and output names, room frame, LiDAR samples, RoomPlan room |
| `registration.py` | Photo poses (SIFT, bundle adjustment) |
| `planes.py`, `shell.py`, `partitions.py` | Plane fits and the sigma formulas, walls (shifted seed, low support, fallback), interior partitions |
| `ceilings.py`, `zones.py`, `columns.py` | Ceiling levels, beams and unexplained strips, ceiling zones, columns |
| `regularize.py` | Squaring, floor polygon, ceilings rebuilt from the squared walls, floor and ceiling `extent` under the walls |
| `model.py`, `export.py`, `meshes.py`, `gltf.py`, `plan_svg.py`, `report.py`, `fmt.py` | Stations and dimensions, `model_measured.glb`, `plan.svg`, `dimensions.csv`, `report.md` |
| `openings.py` | Doors, windows and openings on their hosts |
| `vision.py` (+ `vision_prompt.md`, `vision_schema.json`) | Photo sheets, the `claude -p` call, validation of the answer |
| `llm_merge.py` | LLM answers into the architecture: corners to metres, typing, declared niches, mouldings |
| `photo_geom.py`, `ortho.py`, `raster.py`, `segment.py`, `anchor.py` | Posed photos, rectified faces, SAM 2.1, edge depth across views, the photo gates |
| `edge_pick.py` | Lettered candidate lines, the pick, its validation |
| `tape_merge.py`, `tape.py` | Tape values applied to the model; `compare-tape` |
| `overrides.py` | `overrides.json`: fields, validation, `prefer` |
| `architecture.py` | The merge, wall lengths, dimension endpoints, the reconciliation hook |
| `reconcile.py` | `reconcile.json` / `.md`, the checks, the actions, the set-aside tape errors |
| `survey_page.py` | The survey page |
| `textures.py` | Photo textures |
| `blender_stage.py`, `../blender/` | Launches Blender; the scene (`build_scene.py`, `arch/*`: shell, joinery, niches, mouldings, clipping of casings, dimensions, render) |
| `compare.py` | `compare` |

## Files you edit

Both live in `--out` and are merged by `architecture` (priority: tape > photo > LLM > LiDAR > RoomPlan > assumed). Editing one reruns the stages that read it (`STALE_AFTER`); a file copied with an old modification time triggers nothing: `touch` it or use `--force`.

### `measurements.json`

Tape values in metres, per dimension id (`Door_1-leaf-w`, `S01`, `H1`...). Written by the survey page, or given with `--tape`. Two formats are read:

- v1: `{"<dimension id>": <metres>}`.
- v2, written by the page: `{"version": 2, "capture": "<capture id>", "values": {"<dimension id>": {"value_m": <metres or null>, "date": "<ISO date>", "note": "<free text>"}}}`.

A `note` is kept with or without a value; `"value_m": null` is a note alone (or a value the person withdrew) and is never applied. Not a number of metres above zero: listed in `diagnostics.tape.not_applied`, never applied. Effects:

- An opening dimension (`<id>-w`, `-h`, `-sill`) replaces the parameter it measures; a leaf dimension (`<id>-leaf-w`, `-leaf-h`, doors and windows) creates the leaf parameter and its dimension.
- A ceiling height (`H<k>`) moves the ceiling level, and the wall, beam and column tops that were at that level.
- A station (`S..`) moves the wall of its pair with the larger offset sigma along its normal, by at most the larger of 10 station sigmas and one wall thickness; floor, ceilings, openings, mouldings and the other stations follow. Beyond the bound the value is refused (« saisie écartée »).
- A wall length (`L-<label>`, corner to corner, read above the skirtings) moves whichever of the wall's two end walls has the larger offset sigma along its normal, with the same bound as a station.
- An id the architecture does not know is refused.
- Probable tape errors found by the reconciliation are kept out of the model, in memory only: see Reconciliation. The file is never rewritten by the pipeline.

### `overrides.json`

`{"<element id>": {field: value}}`, applied last (after the tape). Fields (`FIELDS` in `overrides.py`):

| Field | Applies to | Meaning |
|---|---|---|
| `kind` | door, niche, opening | Converts between these three; fields of the old kind the new one lacks are dropped; a niche gets `through: false`, no dividers or shelves and an assumed recess depth. Ids are kept. Applied first. |
| `type`, `hinge`, `swing`, `panels`, `handle` | door, window | Typing, validated against `vision_schema.json` and the Blender catalogue (`blender/arch/catalogue.py`) |
| `frame` | door, window, opening, niche | `{"profile": "flat" or "moulded", "width_cm": ...}` or `null` |
| `through`, `dividers`, `shelves_count` | niche only | Goes through its host; number of dividers; number of evenly spaced shelves |
| `prefer` | any element with parameters | `{"<parameter name>": "scan"}` (or the scan's own source, `lidar`, `roomplan`): the parameter goes back to the value, sigma and source the scan gave, over an LLM, photo or tape value |
| `remove` | door, window, opening, niche, `panel_frame`, cornice, skirting | `true` deletes the element and the parameters nothing else uses. Walls, floor, ceilings cannot be removed |

A rejected field (unknown element, unknown field, value outside the schema or the catalogue, field that does not apply to the kind) is listed in `diagnostics.overrides.rejected` and under `Refusées` in `report.md`; the rest of the element is applied. An element with an applied field gets the flag `user_override`.

## Survey page

`measurements/index.html` is a local, self-contained page, never published, written by `architecture` (it is rebuilt at each run and pre-filled from `measurements.json`). Open it from disk; Chrome or Edge give autosave. It shows a numbered SVG plan and one row per measure:

- what and where to tape, a photo crop with the endpoints drawn, the current value with source and sigma, a field in cm, a note field;
- the badge says what to tape: `vantail` (the leaf alone), `deux vantaux` (both leaves together; the pipeline halves it), `ouvrant` (window sash), `allège` (window sill), `passage` or `niche` (jamb to jamb). A leaf end the photos have not found yet is marked « position estimée »;
- the badge « à reprendre en priorité » marks a value that is not from the tape and comes from RoomPlan, the LLM or an assumption, has a sigma of 2 cm or more, or carries a flag;
- rows are: leaves (doors, windows), openings and niches, window sills (and niche sills above the floor), stations `S..`, ceiling heights `H..`. Wall lengths, beams and columns have no row.

Saving: « Lier le fichier de mesures » links `measurements.json` in `--out` and writes it at each edit (File System Access API); otherwise the page downloads `measurements.json` and imports one. A copy is also kept in the browser, per capture id; legacy browser keys are migrated, and values whose row no longer exists are shown apart. The file is the durable copy. Rerun `architecture` (or `all`) after saving. `compare-tape` and the tape report compare scan, LLM and tape values.

## Reconciliation

`architecture` writes `reconcile.json` (`room-model/reconcile-1`) and `reconcile.md` (French) next to `architecture.json`, from the same inputs. It rebuilds the model once per source it has (scan, LLM, photo, tape, override) to see what each source said. Nothing in it changes the model, except the set-aside tape errors below.

- `parameters[<parameter id>]`: `sources` (value and sigma per source), `retained`, `residuals` (per other source: difference in metres and in combined sigmas), `flags` (`conflict` when a measured source disagrees with the retained value by more than 3 combined sigmas; LLM clicks and assumed values never raise it), `actions`.
- `checks[]`: `id`, `kind`, `verdict` (`ok`, `conflict`, `probable_tape_error`, `out_of_square`, `square_within_noise`), `residual`, `sigma` (metres), `elements` or `dims`, `actions`. Kinds: `opening_inside_host`, `opening_overlap`, `leaf_fits`, `corner_square` (corners within 30 degrees of a right angle: the offset the departure makes at the end of the shorter wall against the azimuth sigmas), `wall_top_vs_ceiling`, `outline_closed`, and for the tape: `tape_loop` (relations the geometry imposes: the outline closes along two axes; a station equals the walls between its two ends projected on its direction, both ways round; two stations between the same walls agree), `tape_vs_scan` (one tape value alone against the scan: the tape may be right, to be confirmed), `tape_consistency` (several entries: `probable_tape_error` when removing one entry, or one pair, clears every failing relation, else a ranked `conflict`).
- `elements`: the failing check ids per element. `actions`: the union of the actions, in the grammar `measure:<dimension id>` (tape it), `prefer:<parameter id>:<source>` (the `prefer` override above) and `override:<element id>:<field>` (only `remove`, proposed for two overlapping openings). `reconcile.md` reads them in French.
- A `probable_tape_error` is kept out of the model: `architecture` rebuilds once without the entry. It is listed under `diagnostics.tape.not_applied` (« écartée, à remesurer : erreur de mètre probable ») and `diagnostics.tape.set_aside`, its dimension gets the flag `tape_set_aside`, and `measurements.json` is not touched.
- If the reconciliation fails, `architecture.json` is still written, with `diagnostics.reconcile_error`, and both files hold only an error stub.

The `room-reconcile` skill drives this from the survey to a settled model.

## Blender scene

`blender/build_scene.py` builds the scene from `architecture.json` only (plus the optional textures and the decimated scan), headless. It writes `architecture.blend`, `architecture.glb`, `renders/{top,persp_1,persp_2,open}.png` and `blender_report.json`; `blender.log` holds Blender's own output, whose end names the real failure when it fails.

- Walls are mitred at corners, partitions end on the face they meet (extended through it and cut by an exact boolean, so a jog or a collinear wall stays watertight), an opening cuts every wall or partition its box crosses. The tops of bodies that meet and agree within their sigmas (at most 5 cm apart) are levelled to one height.
- The floor and each ceiling are drawn from their `extent`: the interior at its level, the rim pushed under (over) the walls to their mid-thickness 1 mm lower, and the floor through the thresholds of floor-based cuts, so no strip of void opens between them and the wall bodies or under a doorway.
- A door or window is built from its leaf: the cut is the leaf plus the visible jamb (5 cm by default) on each side; the casing is what remains up to the LiDAR extent of the hole. A passage or niche is cut jamb to jamb with its casing, if any, outside the cut.
- Casings are clipped to the free length of the host face (a partition standing in front of the face covers it); a casing that would be narrower than 2 cm is dropped and the casings of that cut go to `À vérifier`.
- Cornice, skirting and architraves come from a small profile catalogue (heights from the LLM, source `assumed`). They run along the interior wall and partition faces, mitred at the corners. The skirting stops at floor-based cuts; along the jambs of a passage (an opening, or a niche that goes through) it returns into the cut; a door keeps its gap.
- Collections: `Coque` (shell, partitions, beams, columns), `Menuiseries`, `Moulures`, `Cotes`, `À vérifier` (elements flagged `to_verify`, panel frames), `Scan` (decimated LiDAR mesh, hidden).
- Each door and window has an empty parent with custom properties: `Ouverture` (0-1), `Inverser sens` (flips the swing; no effect on a sliding leaf), `Inverser charnière` (hinges on the other jamb; no effect on two leaves), `AngleMax` (degrees, default 90). Checkboxes unchecked = as detected. A double door has one slider for both leaves. Sliding leaves translate.
- Dimensions are MeasureIt segments in `Cotes`, labelled « 286,0 cm ± 1,8 »; in the Blender UI, press the MeasureIt "Show" button (N sidebar) to display them. If MeasureIt is unavailable, or `ROOM_MODEL_NO_MEASUREIT` is set, native objects are used. `blender_report.json` holds `dimensions_backend`, `dimensions` (count), `measureit_error` when MeasureIt failed, and `avertissements`: one French line per cut that runs into a neighbouring wall or partition (« la découpe empiète de ... cm ... à vérifier »); the cut is kept as measured.
- `architecture.glb` is static, doors and windows closed, `Coque`, `Menuiseries`, `Moulures` and `À vérifier` only; `Cotes` and `Scan` are not exported. `model_measured.glb` stays the M1 output.

## Textures

Walls, floor and ceilings are textured from the posed photos at 2 cm per texel. A texel is kept when a photo sees it facing the camera, its depth agrees with the LiDAR mesh (tolerance from the plane error, 10 cm at most, so casings and skirtings belong to the surface) and it lies in no RoomPlan object box, so furniture is not printed. The colour is the weighted median of the best views after exposure matching. Texels no photo covers (coverage holes) are filled with a CC0 texture (`.agents/skills/scan-to-threejs/scripts/fetch_textures.py`, network on the first fill) tinted to the measured colour, feathered into the observed colour. Joinery and mouldings use plain named materials. `textures/textures.json` lists the surfaces and their coverage.

## Contributing: generic for any capture

Nothing in code, tests or docs may depend on one capture: no wall labels, counts, dimensions or room names from a real room. Behaviour comes from the capture data, the catalogue and the schema; synthetic fixtures (`tests/synthetic.py`) stand in for rooms. Real-capture values stay in git-ignored outputs.

## Frame

Shell, model and export use the room frame of `bundle_to_room.py` (`transform.worldToRoom`): metres, +Y up, plan coordinates `[x, z]`. A wall's interior normal is `(cos azimuth, sin azimuth)` and its line is `n . (x, z) = offset`. Registration stays in the ARKit frame.

## Sigma model

Every parameter carries `value`, `sigma`, `unit` and `source`: `lidar`, `roomplan` or `assumed` from the scan stages; `llm`, `photo` and `tape` once `architecture` merges them (priority in Architecture). A corrected parameter keeps its scan values beside it (`scan_value`, `scan_sigma`, and `llm_value` when the LLM set it).

| Quantity | Sigma |
|---|---|
| Plane offset | `sqrt((rms / n_eff)^2 + SIGMA_SYS_OFFSET^2 + (SIGMA_SUPPORT / sqrt(support_m2))^2)`, `n_eff = sqrt(max(support_m2 / 0.01, 1))`, `support_m2` floored at 0.01 |
| Plane azimuth | `hypot(rms / (n_eff * max(along_std, 0.05)), SIGMA_SYS_AZIMUTH)` |
| Wall line at distance `s` from its fit centroid | `hypot(sigma_offset, sigma_azimuth * s)` |
| Derived dimension | Propagated from the parameters it uses |
| Tape value | 0.5 cm, fixed |
| LLM corner clicks | Spread of the crops that agree within 3 cm, at least 1 cm |

Constants: `SIGMA_SYS_OFFSET = 0.01 m`, `SIGMA_SYS_AZIMUTH = 0.25 deg`, `SIGMA_SUPPORT = 0.014 m` (the support term: 2 cm at 0.5 m2, 4.5 cm at 0.1 m2, below 0.5 cm from 8 m2). When a wall has too little LiDAR support, the RoomPlan plane is kept with a fallback sigma of 5 cm and 2 deg (flag `roomplan_fallback`). Wall thickness is assumed: 20 cm, sigma 5 cm.

Misplaced walls (flag `shifted_seed`): a RoomPlan wall can sit away from the real surface. The wall-facing vertical samples inside the wall rectangle within 30 cm of the RoomPlan plane are histogrammed in 1 cm bins; the mode with the most support (at least 0.3 m2 within 4 cm; modes within 10 % of the best tie and the outer one wins, walls being behind furniture) is refitted inside 8 cm. It replaces the fit on the RoomPlan plane only when it is more than 8 cm away and has twice the support, or when the RoomPlan plane has no fit. Walls shorter than about 0.5 m keep the RoomPlan plane: there is nothing to measure.

Low-support walls (support under 0.5 m2, flag `low_support`): the azimuth of a patch that small is not trustworthy, so the wall keeps the RoomPlan azimuth (source `roomplan`, sigma 2 deg). Its offset is the LiDAR fit evaluated at the fit centroid along the RoomPlan normal (source `lidar`), with the support term above dominating the offset sigma. `diagnostics.walls[*].angle_vs_roomplan_deg` still reports the fitted angle.

Squaring (model stage, `regularize.py`): the walls and partitions are turned onto the dominant orthogonal frame, the support-weighted circular mean of the wall azimuths modulo 90 degrees (found among the frames that take in the most support, so a chamfer does not drag it). A line whose azimuth is within `max(2 deg, 3 sigma_azimuth)` of its nearest axis cannot be told from fit noise: it is rotated onto the axis about the centroid of its supporting samples, which keeps the measured position where the evidence is. The measured azimuth is kept as `measured_value` on the parameter (source unchanged) and the element gets `squared`; the 2 degree floor is a user requirement (the model is drawn square unless the scan proves a larger deviation), so a squaring beyond 3 sigma is recorded as `forced` in `diagnostics.squaring` and listed in the report (« équerre imposée »). The engaged side of a column is moved onto its squared wall. A line further off keeps its measurement and gets `out_of_square`. Corners are recomputed with the shell's corner rule, so near-parallel jogs stay jogs and nothing is merged. `diagnostics.squaring` lists, per line, the measured and retained azimuth, the deviation and the tolerance, and per corner the measured and retained angle.

## Detection rules

- Ceilings: height histogram (window 3 cm) of ceiling-facing LiDAR samples (normal pointing down) above 1.6 m; every level covering at least 0.3 m2 becomes a ceiling, so a patch the LiDAR never saw makes none. Once a level is picked, samples within 3 cm of it are removed, and a sample belongs to its nearest level only. Beams are detected on these unmerged levels. Then touching levels closer than 3 times the larger of their two sigmas merge into one support-weighted level, with a merged span of at most 4 cm: steps of about 3-4 cm or less are absorbed, larger steps stay separate.
- Ceilings of the model (`zones.py`, built after squaring; `shell.json` keeps the scan regions): the interior polygon of the squared walls is cut along partitions, beams and the extensions of the edges meeting at each reflex corner (cuts ending within 20 cm of a wall reach it; a cut that does not run wall to wall splits nothing). Each zone takes the best supported level of the ceiling samples inside it, beams excluded; another level with at least 0.3 m2 of support and more than 4 cm away is cut out of the zone as a region of its own (clean polygon, strips such as ducts left out), and when it cannot be made one it is listed in `diagnostics.ceiling_notes` and in the report alerts. The sigma used by the merge rule is that of the level's fit over the whole room, so the verdict does not depend on how the room is cut. Neighbouring zones whose levels merge by the rule above form one `Ceiling_<k>` (possibly several polygons); beam footprints are cut out of the regions. A zone with no level takes the level of its largest neighbour: from 0.3 m2 up it stays a ceiling of its own with source `assumed` (sigma widened by 5 cm) and the element flag `assumed`, below it joins that neighbour. Other levels found in a zone are kept in `diagnostics.ceilings[*].other_levels`. The floor polygon is the one of the squared corners. `Floor.polygon` and `Ceiling_<k>.region` stay the interior (areas are interior areas); each also carries `extent`, a list of parts `{exterior, holes, drop}` that the GLB and the Blender scene draw so that no void opens between the floor and the wall bodies or under a doorway: the interior at its level, then the same shape pushed under (over) the walls to their mid-thickness with mitred corners, drawn `drop` (1 mm) lower so that no face is coplanar with a wall base or top. The floor also runs at its level through the whole thickness of floor-based cuts (doors, passages, through niches down to the floor). When the outer lines do not close no `extent` is written, and `diagnostics.underlay_notes` says why (`no_extent`; `threshold_dropped` for a threshold strip that is degenerate or off the walls).
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
- Walls are never moved by faces of an earlier room (`inroom == 0`); RoomPlan walls off the floor outline whose midpoint lies inside the floor polygon are partitions `Partition_P<k>` (two faces, each fitted when the LiDAR sees it; thickness measured when both faces have at least 0.3 m2 of support and the gap is 3-40 cm, else assumed 10 cm, sigma 3 cm, flag `one_face_seen`), the others are listed in `diagnostics.off_outline_walls`.

## Outputs

Everything below is written under `--out`. Coordinates are in metres in the room frame (+Y up, plan `[x, z]`).

### `model_measured.glb`

- Root nodes: `Coque` (extras `kind: "shell"`, `schema`, `version`) holds the elements of the shell; `À vérifier` (extras `kind: "to_verify"`) exists only when an element carries the flag `to_verify`, and holds those elements instead.
- One mesh node per element, named by its element id: `Floor`, `Wall_M1`, `Partition_P1`, `Ceiling_1`, `Beam_1`, `Column_1`. Walls are prisms behind the interior face, extended past their end by their thickness; floor and ceilings are drawn from their `extent` (under the walls) when the model has one. A beam or column `top` is the ceiling level it meets, copied from the ceiling, not measured on its own.
- Node `extras`: `kind` (`floor`, `wall`, `partition`, `ceiling`, `beam`, `column`), `params` (per parameter name: `{id, value, sigma, source}`) and `flags` (list).

### `model.json`

`model.json` is `shell.json` (schema `room-model/shell-1`) squared and rebuilt (see Sigma model and Detection rules), with `schema: "room-model/1"`, `version`, `diagnostics.squaring` and a `dimensions` list. Each entry: `id`, `kind`, `label` (French), `value` and `sigma` (m), `elements` (element ids such as `Wall_M1`), `at` (`[x, z]` or `null`), `source` (the weakest source among the parameters used: `assumed` < `roomplan` < `lidar` < `llm` < `photo` < `tape`) and `flags` (list).

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

### `architecture.json`

Schema `room-model/arch-1`: the single input of `blender`. `parameters` (every value with sigma and source), `elements` (the model's elements plus `Door_k`, `Window_k`, `Opening_k`, `Niche_k`, the single `Cornice` and `Skirting` the LLM declares, `PanelFrame_k`; an opening has `host`, `face`, `params`, `flags`, typing fields, and `edges` / `leaf_edges` when photo-anchored), `dimensions` and `diagnostics`. Ids are stable across reruns, so `measurements.json` and `overrides.json` keys keep matching (an opening retyped by an override keeps its id).

Dimension ids added by `architecture` (each with `endpoints`, the two ends [x, y, z] it spans, or `null`):

| Id | Kind | Meaning |
|---|---|---|
| `L-M1`, ... | `wall_length` | Corner to corner along the room face of a wall; a survey row, applied by moving an end wall |
| `Door_1-w`, `-h` (`-sill` for windows and niches) | `opening_width`, `opening_height`, `opening_sill` | The hole: width, height, height of the bottom above the floor |
| `Door_1-leaf-w`, `-leaf-h` | `leaf_width`, `leaf_height` | The moving panel of a door or window, once measured or anchored |

`diagnostics` keys: `squaring`, `ceilings`, `ceiling_notes`, `underlay_notes`, `columns`, `unexplained_horizontal`, `off_outline_walls` and the other scan diagnostics; `openings` (`orphans`, `clamped`); `vision` (`corrections`, `typing`, `corrected`, `single_view`, `inconsistent`, `declared`, `superseded`, `not_applied`); `anchor` (`anchored`, `candidates`, `open`, `no_view`, `no_host`, `unknown`, `kept`, `reasons`, `to_verify`, `llm_not_applied`); `tape` (`applied`, `not_applied`, `moved`, `derived`, `set_aside`); `overrides` (`applied`, `rejected`, `removed`); `vision_unreadable` and `reconcile_error` when a file could not be used.

### Flags

| Where | Flag | Meaning |
|---|---|---|
| Wall `flags` | `roomplan_fallback` | No LiDAR fit, RoomPlan plane kept |
| | `shifted_seed` | Surface found away from the RoomPlan plane |
| | `low_support` | Support under 0.5 m2, RoomPlan azimuth kept |
| | `seeded_from_outline` | Wall absent from RoomPlan, built from the floor outline |
| | `near_parallel_corner` | Corner with a near-parallel neighbour, obtained by projection |
| Partition `flags` | `one_face_seen` | One face seen by the LiDAR, thickness assumed |
| | `roomplan_fallback` | Not seen by the LiDAR, RoomPlan position kept |
| Wall and partition `flags` | `squared` | Azimuth turned onto the room axes (measured value in `measured_value`) |
| | `out_of_square` | Azimuth further than its tolerance from the axes, measurement kept |
| Ceiling `flags` | `assumed` | No LiDAR level in the zone, level taken from the largest neighbour |
| Beam `flags` and its dimensions | `to_verify` | Probable beam whose sides are not both seen |
| Opening `flags` | `orphan`, `clamped` | No host to stand on; overhung its host and was clamped to it |
| | `llm_corrected`, `llm_single_view`, `llm_inconsistent` | Corners clicked by the LLM were applied (two or more views agree), applied from one view only, or disagreed and the scan stayed |
| | `photo_anchored`, `edge_picked` | Measured in the photos by the mask; with edges picked by the LLM |
| | `to_verify` | Photo and scan disagree beyond 3 sigmas, or a photo leaf cannot fit its hole; also set on every panel frame the LLM declares |
| Any element | `tape_corrected`, `user_override` | A tape value or an override was applied (a moved wall, a moved ceiling, an opening) |
| Station dimension `flags` | `obstacle_or_uneven_wall` | No clean wall section found for the station; measure with caution (`~` in `plan.svg`) |
| Dimension `flags` | `tape_set_aside` | The tape value was kept out of the model as a probable tape error |
| `diagnostics.flags` | `floor:roomplan_fallback`, `columns:no_ceiling` | Floor level not seen; no ceiling, so no columns |
| | `squaring:invalid_polygon` | The squared outline was not a valid polygon, the model stays unsquared |
| | `<label>:<wall flag>` | Wall flag repeated per wall, for example `M3:shifted_seed` |

`plan.svg` is the dimensioned top view. `report.md` is the French report: inputs, frame, registration, per-wall and per-partition tables, floor, ceilings, beams, columns, unexplained surfaces and horizontal elements, squaring (measured and retained corner angles, « équerre imposée » where the squaring went beyond 3 sigma), alerts (flags, ceiling levels that could not be cut out, share of the ceiling the LiDAR never saw), wall thicknesses; after `architecture` also `Corrections LLM`, `Mesures au mètre` (with the values not applied and the ones set aside) and `Corrections manuelles` (with `Refusées`).

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
- One room per run, no networks (pipes, ducts), no `room.json`.
- Squaring is a user requirement: a wall within 2 degrees of the axes is drawn square even when the scan proves the deviation (recorded as « équerre imposée » beyond 3 sigma); only a larger deviation stays `out_of_square`.
- A real ceiling step is cut out only when it has 0.3 m2 of LiDAR support and a clean outline; otherwise it is reported in the alerts and the zone keeps one level.
- Ducts or trays pressed against the ceiling hide it, so they look like beams.
- Soffits that drop less than 14 cm below the neighbouring ceiling are not reported as beams.
- A narrow ceiling region can be classed as an unexplained strip and lower a wall top.
- `L-<label>` wall lengths appear in `architecture.json`, in the Blender dimensions, on the survey page and in the reconciliation, whose corner checks propose `measure:L-<label>`.
- Networks (M2), photo silhouettes (M3) and the site survey (M4) are expected to settle the remaining cases.
