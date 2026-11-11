# Room model M1.5 (architecture → Blender) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a House Capture and the M1 `model.json`, build `architecture.json` (walls, partitions, doors, windows, openings, niches, mouldings, dimensions, optional LLM corrections) and a headless-built Blender scene with rigged joinery, MeasureIt dimensions, projected wall textures and a static GLB.

**Architecture:** Five new stages after `export`, same CLI and skip rules: `openings` (RoomPlan openings hosted on fitted walls and partitions → `openings.json`), `vision` (photo crops → `claude -p` → `vision/answers.json`), `architecture` (merge → `architecture.json` + tape measurement sheet), `textures` (photo projection → `textures/textures.json` + PNGs), `blender` (runs `blender/build_scene.py` headless → `.blend`, `.glb`, renders). The Blender side is a separate package of `bpy` modules that only reads JSON.

**Tech Stack:** Python 3.12 (existing venv and pins, nothing added), Blender 5.1+ `bpy` (headless 5.3.0), `claude` CLI, MeasureIt extension (optional).

**Spec:** `docs/specs/room-model-pipeline.md`, section "M1.5 Architecture → Blender" (overrides the rest). Current pipeline doc: `docs/room-model.md`.

## Global Constraints

- Worktree `~/dev/3d-model-anything-room-model`, branch `feat/room-model-architecture`. Never touch `~/dev/3d-model-anything`.
- Venv `pipelines/room_model/.venv`; pins unchanged (`numpy scipy plyfile shapely pycolmap pillow pytest`). No new pip dependency. Blender-side code uses only `bpy`, `mathutils` and Blender's bundled `numpy`.
- CLI from `pipelines/room_model/`: `.venv/bin/python -m room_model <stage|all> --capture <dir> --out <dir> [--room 0] [--turn 0] [--force] [--keep-work] [--no-llm]`. `STAGE_OUTPUTS` gains `openings: openings.json`, `vision: vision/answers.json`, `architecture: architecture.json`, `textures: textures/textures.json`, `blender: architecture.blend`. `all` runs registration, shell, model, export, openings, vision, architecture, textures, blender.
- `BLENDER` env var, default `/mnt/c/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe`. When the executable ends in `.exe`, every path passed to it goes through `wslpath -w`. Blender runs with `--background --python-exit-code 1`; user extensions stay enabled (no `--factory-startup`) so MeasureIt is found.
- Frame: room frame of `bundle_to_room.py`, metres, +Y up, plan `[x, z]`. Blender scene: Z-up, room `(x, y, z)` → Blender `(x, -z, y)`.
- Parameter format unchanged (`value`, `sigma`, `unit`, `source`); `source` gains `llm`. An LLM-corrected parameter also has `scan_value` and `scan_sigma`, and its element carries flag `llm_corrected`.
- Element ids: `Partition_P<k>`, `Door_<k>`, `Window_<k>`, `Opening_<k>`, `Niche_<k>`, `Cornice`, `Skirting`, `PanelFrame_<k>`. Numbering follows RoomPlan order, then LLM-declared elements.
- Outputs only under `--out`; < 500 MB per room; < 30 min compute per room excluding the LLM.
- **Private data:** nothing derived from a real capture in code, tests, docs or commits. Tests use `tests/synthetic.py` only.
- Language: code, identifiers, JSON keys, node and object names in English, except the Blender collection names and custom properties fixed by the spec (`Coque`, `Menuiseries`, `Moulures`, `Cotes`, `À vérifier`, `Scan`, `Ouverture`, `Sens`). User-visible text (labels, reports, dimension texts) in French with accents.
- Comments ≤ 3 lines, docstrings ≤ 10 lines; `~/.claude/scripts/comment-budget.sh 2a6b5db`.
- Behavioural tests, no mocks of this package's own modules. External processes (`claude`, Blender) are replaced only through their env var (`ROOM_MODEL_CLAUDE`, `BLENDER`). Blender tests are skipped when the executable is missing.
- Shell: `rm`/`cp`/`mv` aliased to `-i`: use `\rm`, `command cp`, `mv -f`.
- Commit only your task's files (`git add <paths>`, never `-A`, never `docs/specs/` or `docs/plans/`). Conventional Commits ending with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_011azxumEuhEcPUjGAeasscQ`.

## Review Focus

1. An opening whose RoomPlan `parentIdentifier` is a wall the outline dropped (interior partition) must land on the partition, not be lost or snapped to the nearest perimeter wall (Task 2 test `test_opening_on_partition`).
2. Photos stored in sensor orientation (landscape with the room sideways): crops shown to the LLM must be gravity-upright, and pixel answers must be mapped back to sensor pixels before back-projection (Task 3 test `test_upright_roundtrip`).
3. Two openings overlapping on one wall, or an opening wider than its host: the Blender cut must not produce non-manifold or missing walls; the pipeline clamps to the host and flags `clamped` (Task 2 test `test_opening_clamped_to_host`).
4. LLM answer missing, malformed, or naming an unknown element or catalogue type: the stage keeps the scan values, lists the rejects in `vision/rejected.json`, and never crashes (Task 8 test `test_bad_answers_rejected`).
5. MeasureIt absent or its API changed: the scene still gets native dimensions and the report says which path ran (Task 6 test `test_dimensions_fallback`).

---

### Task 1: Interior partitions in the shell — `sonnet-medium`

**Files:**
- Create: `room_model/partitions.py`
- Modify: `room_model/shell.py` (`build` appends partitions), `room_model/meshes.py` (reuse `wall_mesh`), `room_model/export.py` (`KIND_ORDER` gains `partition`), `room_model/report.py` (partition rows), `tests/synthetic.py` (option `partition: bool`)
- Test: `tests/test_partitions.py`

**Interfaces:**
- Produces: element `{"id": "Partition_P1", "kind": "partition", "label": "P1", "roomplan_id", "start": [x, z], "end": [x, z], "flags", "params": {"azimuth", "offset", "thickness", "top"}}` with parameter ids `partition.P1.<name>`. Face A is the face whose normal `(cos az, sin az)` points to +z or, if parallel to z, to +x; `offset` is face A's line. `diagnostics.off_outline_walls` keeps only walls that are neither on the outline nor partitions.
- `find_partitions(room: Room, samples: Samples, polygon: list) -> list[dict]` in `partitions.py`.

- [ ] **Step 1: tests.** Extend `SyntheticOptions` with `partition: bool = False` (a 2.0 m long, 0.10 m thick, full-height partition inside the floor polygon, both faces in the mesh, plus its RoomPlan wall off the outline). Tests: `test_partition_detected` (one `Partition_P1`, thickness 0.10 ± 0.02 with source `lidar`, start/end within 3 cm of truth); `test_partition_one_face_seen` (only face A in the mesh → thickness 0.10, source `assumed`); `test_partition_outside_floor_ignored` (a RoomPlan wall outside the floor polygon stays in `off_outline_walls`); `test_perimeter_unchanged` (with `partition=True`, all perimeter walls equal those of `partition=False` within 1 mm).
- [ ] **Step 2:** run `.venv/bin/python -m pytest -q tests/test_partitions.py` → FAIL.
- [ ] **Step 3: implement.** A partition is an off-outline RoomPlan wall whose segment midpoint lies inside the floor polygon. Fit each face with the existing wall-candidate/plane-fit code on the two half-spaces (`FIT_WINDOW` around RoomPlan plane ± thickness/2); thickness = distance between the two fits when both have ≥ 0.3 m² support, else 0.10 `assumed` sigma 0.03. Top = same rule as walls (`_wall_top`). Extent = RoomPlan width, ends clipped to the floor polygon.
- [ ] **Step 4:** full suite passes (`.venv/bin/python -m pytest -q`).
- [ ] **Step 5: commit** `feat(room-model): keep interior partitions as two-faced elements`.

### Task 2: `openings` stage — `sonnet-medium`

**Files:**
- Create: `room_model/openings.py`
- Modify: `room_model/context.py`, `room_model/__main__.py` (stage registration), `tests/synthetic.py` (options `window: bool`, `partition_opening: bool`; synthetic door already exists)
- Test: `tests/test_openings.py`

**Interfaces:**
- Consumes: `model.json` elements (walls, partitions), `load_roomplan`, `world_to_room` (stored in `model.json["frame"]["world_to_room"]`).
- Produces `openings.json`: `{"schema": "room-model/openings-1", "parameters": {...}, "openings": [{"id", "kind": "door"|"window"|"opening", "roomplan_id", "host": "<element id>", "face": "A"|"B"|null, "params": {"center_s", "sill", "width", "height", "depth"}, "flags": []}]}`. `center_s` metres along host from `start`; `sill` metres above the floor level; `depth` = host thickness. Parameter ids `opening.<id>.<name>`, source `roomplan` (sigma 0.03) or `lidar` when refined.
- `host_of(opening_tf: np.ndarray, rp_parent: str, hosts: list[dict]) -> tuple[str, float] | None` (host id, signed distance).

- [ ] **Step 1: tests.** `test_door_on_perimeter` (synthetic door → `Door_1` on its wall, `center_s`, `width`, `height` within 2 cm, `sill` within 2 cm of 0); `test_window_sill` (window at 0.9 m → sill 0.90 ± 0.02); `test_opening_on_partition` (opening whose parent is the partition → host `Partition_P1`); `test_opening_clamped_to_host` (opening 0.3 m wider than its host → width clipped to host, flag `clamped`); `test_orphan_opening` (no host within 0.3 m and 10° → kept with `host: null`, flag `orphan`, listed in report).
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implement.** Host by `parentIdentifier` first, else nearest parallel (≤ 10°) host within 0.30 m. Project the RoomPlan centre onto the fitted host line; keep RoomPlan width/height. LiDAR refinement: on the host's samples, the opening edges are where wall support drops to zero within ± 15 cm of RoomPlan's edges (1 cm bins along `s` and `y`); if all four edges are found, they replace RoomPlan's (source `lidar`, sigma from bin width and fit RMS), else keep RoomPlan.
- [ ] **Step 4:** full suite passes.
- [ ] **Step 5: commit** `feat(room-model): add the openings stage`.

### Task 3: Photo geometry and the `architecture` stage (no LLM yet) — `sonnet-medium`

**Files:**
- Create: `room_model/photo_geom.py`, `room_model/raster.py`, `room_model/architecture.py`, `room_model/survey_page.py`, `room_model/tape.py`
- Modify: `room_model/context.py`, `room_model/__main__.py` (stage + subcommand `compare-tape --arch <architecture.json> --tape <measurements.json> [-o tape.md]`), `tests/synthetic.py` (option `photos: int = 0`: posed photos rendered with `raster.py`, flat colour per surface, stored in sensor orientation rotated 90° like the iPhone)
- Test: `tests/test_photo_geom.py`, `tests/test_architecture.py`

**Interfaces:**
- `raster.depth_map(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, w2c: np.ndarray, size: tuple[int, int]) -> np.ndarray` (z-buffer in metres, `inf` where empty; numpy only).
- `photo_geom.load_posed_photos(ctx) -> list[PosedPhoto]` (poses from `registration/poses.json` when present, else ARKit; room frame). `PosedPhoto` has `name, path, K, c2w_room, size, up_turns`.
- `photo_geom.upright(photo) -> PIL.Image` and `to_sensor(photo, uv_upright) -> np.ndarray` (inverse); `up_turns` = quarter turns that bring room +Y up in the image.
- `photo_geom.project(photo, pts_room) -> (uv: np.ndarray, depth: np.ndarray)`; `photo_geom.backproject(photo, uv_sensor, plane: (normal3, offset)) -> np.ndarray`.
- `photo_geom.best_views(photos, pts_room, mesh, k=3) -> list[PosedPhoto]` (visible by depth test within 3 cm, sorted by projected area × cos of view angle).
- `architecture.build(model: dict, openings: dict, vision: dict | None) -> dict` → `architecture.json` (schema `room-model/arch-1`): model `parameters`/`elements`/`diagnostics` plus opening elements (from `openings.json`, `type: null` until Task 9), `dimensions` = model dimensions + `L-M<k>` (wall length between corners), `<id>-w`, `<id>-h`, `<id>-sill` (windows and niches).
- `architecture.run` also writes the survey page `measurements/index.html` (self-contained: images inlined as base64, no external script; French) via `room_model/survey_page.py`:
  - a top-view SVG plan (reuse `plan_svg.py` geometry) with a numbered marker on each measured span (1, 2, 3…), the span drawn between its two endpoints;
  - a form, one row per number: number, French description of what to measure and where (« Largeur de la porte Door_1, entre les tableaux, à mi-hauteur »), the upright photo crop with both endpoints drawn, the current value (scan value, and LLM value when `llm_corrected`), its sigma and source, a priority badge « à reprendre en priorité » when sigma ≥ 2 cm or source is `roomplan`/`llm`/`assumed` or the dimension has flags, and an input in cm;
  - rows: every opening `-w`/`-h`, every `-sill`, every width/length station, every ceiling height; priority rows first;
  - buttons « Télécharger measurements.json » (`{"<dimension id>": <metres>}`, only filled inputs) and « Copier »; inputs kept in `localStorage` (wrapped in try/catch).
  `measurements/plan_numbers.json` maps number → dimension id. Each dimension used in the page needs its two endpoints in the room frame: `architecture.dimension_endpoints(arch, dim_id) -> (p0: [x, y, z], p1: [x, y, z]) | None`.
- `tape.compare(arch: dict, tape: dict) -> str` (Markdown, French): per tape id, scan value, LLM value (if any), tape value, both differences in cm. `measurements.json` = `{"<dimension id>": <metres>}`.

- [ ] **Step 1: tests.** `test_upright_roundtrip` (random sensor pixels → upright → `to_sensor` identity for each of the 4 `up_turns`); `test_backproject_hits_wall` (pixel of a known synthetic wall corner back-projects within 1 cm); `test_depth_map_occlusion` (a box in front of a wall: wall points behind it fail the 3 cm depth test); `test_architecture_dimensions` (synthetic with door + window: `Door_1-w`, `Door_1-h`, `Window_1-sill`, `L-M1` present and within 2 cm of truth); `test_survey_page` (synthetic with door + window: page exists, every number in `plan_numbers.json` appears in the SVG and has one input row, `Door_1-w` row present, a dimension with source `roomplan` carries the priority badge, no `http` URL in the file); `test_tape_report` (tape value 2 cm off → difference column shows `+2,0`).
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implement** per interfaces. Rasterise at most 320 px on the long side.
- [ ] **Step 4:** full suite passes; run on the real capture: `.venv/bin/python -m room_model architecture --capture ../../data/captures-in/bureau/Room --out ../../data/room-model/bureau` (with `--no-llm`) succeeds and writes the sheet.
- [ ] **Step 5: commit** `feat(room-model): add the architecture stage and the tape measurement sheet`.

### Task 4: Blender scene builder: shell, cuts, collections, scan, GLB — `sonnet-high`

**Files:**
- Create: `blender/build_scene.py` (entry: `blender --background --python build_scene.py -- --arch A --out-dir D [--textures T] [--scan S]`), `blender/arch/__init__.py`, `blender/arch/frame.py`, `blender/arch/shell.py`, `blender/arch/collections.py`, `blender/arch/materials.py`, `blender/arch/render.py`; `room_model/blender_stage.py` (runner + scan decimation)
- Modify: `room_model/context.py`, `room_model/__main__.py`
- Test: `tests/test_blender.py` (skipped without Blender), `tests/blender_probe.py` (runs inside Blender, dumps scene facts as JSON)

**Interfaces:**
- `blender_stage.decimate(vertices, faces, cell: float = 0.03) -> (v, f)` (vertex clustering, numpy) → `scan_decimated.ply`.
- `blender_stage.run(ctx)` calls Blender; outputs `architecture.blend`, `architecture.glb` (static: rigs at `Ouverture = 0`, MeasureIt and `Scan` excluded), `renders/{top,persp_1,persp_2}.png` (Workbench, 1280×720).
- In Blender: one object per element named by element id; walls and partitions are prisms cut by one boolean cutter per hosted opening (cutter = opening rect, depth host thickness + 2 cm), modifiers applied before saving so the `.blend` and GLB carry clean meshes; cutters deleted. Collections per spec; `Scan` hidden in viewport and render. Plain materials named `Mur`, `Sol`, `Plafond`, `Menuiserie`, `Verre`, `Moulure`.
- `build_scene.build(arch: dict, textures: dict | None, scan_path: str | None) -> None` for the later tasks to extend through `arch/joinery.py`, `arch/dimensions.py`, `arch/mouldings.py`, `arch/textures.py` hooks, each `add(arch, collections) -> list[bpy.types.Object]`.

- [ ] **Step 1: tests.** `test_scene_objects` (synthetic architecture with door, window, partition → probe lists `Wall_M*`, `Partition_P1`, `Floor`, `Ceiling_*` in `Coque`; `Scan` hidden); `test_wall_cut` (probe ray-casts through the door centre → no hit on the host wall; 0.3 m beside → hit); `test_glb_written` (`architecture.glb` parses, node names include the wall ids); `test_decimate` (pure Python: 100 k-face plane → < 5 k faces, bbox within 3 cm).
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implement.** Read the `blender-mcp` skill's gotchas section first (depsgraph evaluation, operator context). Use `bmesh`/data API, not `bpy.ops`, except the glTF export.
- [ ] **Step 4:** tests pass; run the stage on the real capture with `--no-llm`, open `renders/persp_1.png` and check walls are cut.
- [ ] **Step 5: commit** `feat(room-model): build the architecture scene in headless Blender`.

### Task 5: Joinery catalogue and opening rig — `sonnet-high`

**Files:**
- Create: `blender/arch/joinery.py`, `blender/arch/catalogue.py`
- Test: extend `tests/test_blender.py`

**Interfaces:**
- Catalogue keys exactly as the spec: doors `hinged_single`, `hinged_double`, `sliding`, `pocket`, `french_window`; windows `casement_single`, `casement_double`, `tilt_turn`, `sliding`, `fixed`. Element fields read: `type` (null → `hinged_single` for doors, `casement_double` for windows, `null` for `opening`), `hinge` (`left`|`right`, default `left`), `swing` (`in`|`out`, default `in`), `panels` (int, default 0), `handle` (bool, default true).
- Per door/window: empty `<id>` in `Menuiseries` with custom props `Ouverture` (float 0–1, default 0), `Sens` (int 0/1), `AngleMax` (degrees, default 90); children: frame `<id>_Frame`, leaves `<id>_Leaf_<k>`, glass `<id>_Glass_<k>` for windows and `french_window`. Drivers: hinged leaf rotation Z = ±`Ouverture`·`AngleMax` (sign by hinge, `Sens` and leaf index); sliding/pocket leaf translation along the wall = `Ouverture`·leaf width; `tilt_turn` turns like a casement (tilt not rigged).
- Moulded panels: `panels` raised rectangles on each leaf face, evenly split along the height.

- [ ] **Step 1: tests.** `test_rig_hinged` (set `Ouverture = 1` on `Door_1`, evaluate depsgraph → leaf rotated 90° ± 0.5°; `Sens = 1` → −90°); `test_rig_double` (both leaves move opposite ways with one slider); `test_rig_sliding` (leaf translated by its width); `test_static_glb_closed` (GLB leaf transforms are identity relative to the frame).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** pass, plus a render with every door at `Ouverture = 0.6` (`renders/open.png`).
- [ ] **Step 5: commit** `feat(room-model): add rigged doors and windows from a short catalogue`.

### Task 6: Dimensions — MeasureIt with native fallback — `sonnet-medium`

**Files:**
- Create: `blender/arch/dimensions.py`
- Test: extend `tests/test_blender.py`

**Interfaces:**
- `dimensions.add(arch, collections)`: one dimension per `arch["dimensions"]` entry whose geometry is known (two endpoints in room frame, computed from the elements). Text: `"<value cm, 1 decimal> cm ± <sigma cm>"` in French format (`286,0 cm ± 1,8`).
- MeasureIt path: detect with `addon_utils.check("measureit")` / extension module `bl_ext.blender_org.measureit`; read its source under the user extensions folder (`%APPDATA%/Blender Foundation/Blender/<ver>/extensions/blender_org/measureit`) to find how a segment between two vertices is declared, and declare each dimension on a helper mesh with two vertices named `Dim_<id>` in `Cotes`. Native path: curve + two arrow cones + text object, grouped under empty `Dim_<id>`.
- Scene custom property `dimensions_backend`: `"measureit"` or `"native"`; also written to `blender_report.json` by the stage.

- [ ] **Step 1: tests.** `test_dimensions_native` (run with env `ROOM_MODEL_NO_MEASUREIT=1` → one `Dim_*` per dimension, text matches `^\d+,\d cm ± \d+,\d$`); `test_dimensions_fallback` (same env → `dimensions_backend == "native"`); `test_dimensions_measureit` (skipped when the extension is missing → backend `measureit`, MeasureIt data present on `Dim_*`).
- [ ] **Step 2–4:** FAIL, implement, pass.
- [ ] **Step 5: commit** `feat(room-model): add dimensions through MeasureIt with a native fallback`.

### Task 7: Niches and mouldings in Blender — `sonnet-medium`

**Files:**
- Create: `blender/arch/niches.py`, `blender/arch/mouldings.py`, `blender/arch/profiles.py`
- Test: extend `tests/test_blender.py` (tests feed `architecture.json` directly; producing these elements from the photos is Task 9)

**Interfaces:**
- Niche element: `{"id": "Niche_1", "kind": "niche", "host", "params": {"center_s", "sill", "width", "height", "depth"}, "through": bool, "dividers": int, "shelves": [heights above sill, m], "frame": {"profile", "width"}}`. Built as a host cut (through → full thickness; else `depth`, with a back face), `dividers` evenly spaced vertical boards 2 cm thick, `shelves` boards, a moulded frame on each visible face.
- Profiles (`profiles.py`, 2D polylines in metres, scaled by `height`/`projection`): `cornice`: `cove`, `ogee`, `stepped`, `flat`; `skirting`: `flat`, `bevel`, `ogee`; `architrave`: `flat`, `moulded`; `panel_frame`: `quarter_round`, `ogee`. Unknown profile → `flat` + flag `unknown_profile`.
- Cornice/skirting: one sweep along the floor polygon (skirting) or each ceiling region outline at the ceiling level (cornice), interrupted at door and opening spans (skirting only). Architraves: around each door, window, opening and niche with a `frame`. Panel frames under `À vérifier`.

- [ ] **Step 1: tests.** `test_niche_through` (ray through the niche centre passes both faces; 2 dividers → 3 compartments); `test_skirting_gaps` (skirting has no geometry inside the door span); `test_cornice_height` (cornice bbox top = ceiling level ± 1 mm, height as given); `test_unknown_profile` (flag set, object still built).
- [ ] **Step 2–4:** FAIL, implement, pass.
- [ ] **Step 5: commit** `feat(room-model): add niches, cornices, skirting and architraves`.

**Checkpoint (orchestrator):** run `all --no-llm` on the capture, write trace `09-bureau-ouvertures-rig` (blend, glb, renders, measurement sheet, index.html), hand the measurement list to the user.

### Task 8: `vision` stage — `sonnet-high`

**Files:**
- Create: `room_model/vision.py`, `room_model/vision_schema.json`, `room_model/vision_prompt.md`
- Modify: `room_model/context.py`, `room_model/__main__.py` (`--no-llm`, `--llm-model opus`)
- Test: `tests/test_vision.py`, `tests/fake_claude.py` (reads the prompt, writes a canned answer)

**Interfaces:**
- `vision.prepare(ctx) -> Path` writes `vision/pending/`: `elements.json` (compact: id, kind, host, RoomPlan dims, which crops), one sheet per opening (`<id>.jpg`: its 3 best upright views side by side, RoomPlan rectangle drawn, crop ids printed), `walls_<k>.jpg` overview sheets (4 views each) for mouldings and undeclared niches, `prompt.md` (from `vision_prompt.md`, French domain words, English JSON keys), `schema.json`. At most 12 sheets.
- `vision.run_llm(pending: Path, model: str) -> dict`: `[$ROOM_MODEL_CLAUDE or "claude", "-p", "--model", model, "--output-format", "json", "--allowedTools", "Read"]` with the prompt on stdin, cwd = `pending`; parse the `result` field as JSON, validate against `schema.json` by hand (no jsonschema dependency).
- Answer schema: `{"elements": [{"id", "type", "hinge", "swing", "panels", "handle", "frame": {"profile", "width_cm"}, "corners": [{"crop": "<crop id>", "uv": [[u, v] x4]}], "confidence": 0-1}], "new_elements": [{"kind": "niche", "host_hint": "<wall id>", "through", "dividers", "shelves_count", "corners": [...]}], "cornice": {"profile", "height_cm", "projection_cm"} | null, "skirting": {"profile", "height_cm"} | null, "panel_frames": [{"host_hint", "corners": [...], "profile"}]}`. `uv` are pixels of the upright crop.
- `vision.run(ctx)` → `vision/answers.json` (the validated answer + `model`, `duration_s`), `vision/rejected.json`. With `--no-llm`: `{"skipped": true}`.

- [ ] **Step 1: tests.** `test_prepare_sheets` (synthetic with photos → one sheet per opening, ≤ 12, images upright: the synthetic floor colour is in the bottom third); `test_run_with_fake_claude` (`ROOM_MODEL_CLAUDE=tests/fake_claude.py` → answers.json holds the canned types); `test_bad_answers_rejected` (fake returns an unknown id, an unknown type and non-JSON text in three runs → stage succeeds, rejects listed, no element changed); `test_no_llm` (`--no-llm` → `{"skipped": true}`).
- [ ] **Step 2–4:** FAIL, implement, pass.
- [ ] **Step 5: commit** `feat(room-model): add the vision stage driven by claude -p`.

### Task 9: Merge vision into the architecture — `sonnet-medium`

**Files:**
- Modify: `room_model/architecture.py`, `room_model/report.py` (section « Corrections LLM »: element, parameter, scan, LLM, difference cm)
- Test: extend `tests/test_architecture.py`

**Interfaces:**
- Corners → metres: each crop's 4 corners mapped `to_sensor`, back-projected on the host face plane; per corner, the median over crops; width = top/bottom edge mean, height = side mean, `center_s` and `sill` from the rectangle. Sigma = spread across crops (min 1 cm). Applied when ≥ 2 crops agree within 3 cm; the parameter becomes source `llm` with `scan_value`/`scan_sigma`, element flag `llm_corrected`. A single-crop answer is applied too, flag `llm_single_view`.
- New niche: host = wall whose plane is nearest the back-projected centre (≤ 0.3 m), id `Niche_<k>`; `through` from the answer; LiDAR check: through when the host has no samples behind the niche rect on face B — disagreement adds flag `through_disputed`.
- Cornice/skirting/panel frames copied with source `assumed` (heights in metres).
- Tape corrections: `architecture.run` reads `--out/measurements.json` when present (CLI flag `--tape <file>` copies a given file there). Source priority tape > llm > lidar > roomplan > assumed. A tape value replaces the parameter it measures (source `tape`, sigma 0.005, `scan_value` kept, flag `tape_corrected`): opening width/height/sill directly; ceiling height moves the ceiling level; a width/length station moves the wall of the pair with the larger offset sigma along its normal so the station equals the tape value (both walls `lidar` with equal sigma → the second one). Unknown ids are listed in the report, never applied. The survey page shows the tape value in its row once applied.

- [ ] **Step 1: tests.** `test_llm_correction_applied` (canned corners 4 cm wider than RoomPlan on 2 crops → width source `llm`, `scan_value` kept, flag set); `test_llm_disagreeing_crops` (crops 8 cm apart → no correction, flag `llm_inconsistent`); `test_new_niche_hosted`; `test_report_corrections_table`; `test_tape_overrides_llm` (tape and LLM both on `Door_1-w` → tape wins, source `tape`); `test_tape_moves_wall` (station 3 cm longer on tape → the higher-sigma wall moves 3 cm, the other stays); `test_tape_unknown_id` (listed, not applied).
- [ ] **Step 2–4:** FAIL, implement, pass.
- [ ] **Step 5: commit** `feat(room-model): apply LLM corrections and declared niches with scan values kept`.

**Checkpoint (orchestrator):** run `vision` (Opus) on the capture, rebuild, trace `10-bureau-llm-moulures`; hand the survey page (`measurements/index.html`) to the user; with their `measurements.json`, run `compare-tape`, rebuild with `--tape`, and put both reports in the trace.

### Task 10: Projected textures — `sonnet-high`

**Files:**
- Create: `room_model/textures.py`, `blender/arch/textures.py`
- Modify: `room_model/context.py`, `room_model/__main__.py`
- Test: `tests/test_textures.py`, extend `tests/test_blender.py`

**Interfaces:**
- Per surface (each wall/partition face, floor, each ceiling): a planar texel grid at 0.02 m (`origin`, `u_axis`, `v_axis`, `size` in texels). For each texel, candidate photos where it projects inside the image, passes the depth test against `raster.depth_map` of the full LiDAR mesh (|depth − texel depth| < 3 cm) and is outside every RoomPlan object box grown by 5 cm; colour = weighted median of the 3 best (weight: cos view angle / distance²), photos exposure-matched per surface by median gain on shared texels.
- Holes (no candidate): filled with a CC0 texture fetched once into `textures/fill/` via `.agents/skills/scan-to-threejs/scripts/fetch_textures.py get` (walls/ceilings `polyhaven:white_plaster_02`, floor `polyhaven:oak_veneer_01`, overridable in `architecture.json["materials"]`), tinted to the surface's median observed colour; 8-texel feathered seam.
- Output: `textures/<element>_<face>.png` + `textures/textures.json` `{element_id: {face: {"image", "origin", "u_axis", "v_axis", "texel", "coverage"}}}`. Blender side: planar UVs from the same axes, image texture material per face; surfaces cut by openings keep UVs from the uncut plane.
- Network access only for the fill textures; if the download fails, holes are filled with the flat median colour and `textures.json` notes it.

- [ ] **Step 1: tests.** `test_projection_colour` (synthetic photos, wall colour red → texture median red ± 5 levels); `test_furniture_masked` (a box in front of a wall with its own colour → none of that colour in the wall texture, holes filled); `test_coverage_reported`; `test_blender_uv` (probe: texture pixel at a known wall point matches).
- [ ] **Step 2–4:** FAIL, implement, pass; time the stage on the capture (must stay within the 30 min budget with the other stages).
- [ ] **Step 5: commit** `feat(room-model): project photos onto walls, floor and ceilings with an occlusion mask`.

**Checkpoint (orchestrator):** trace `11-bureau-textures`.

### Task 11: Integration and docs — `sonnet-low`

**Files:**
- Modify: `docs/room-model.md` (stages table, Blender section: how to open, collections, the `Ouverture` slider, MeasureIt, `--no-llm`, `compare-tape`), `room_model/__main__.py` (`all` order, `--no-llm` passthrough)
- Test: `tests/test_cli.py` (`all --no-llm` on synthetic with photos runs every stage; Blender stages skipped with a printed note when `BLENDER` is missing)

- [ ] **Step 1–4:** test, FAIL, implement, pass.
- [ ] **Step 5: commit** `docs(room-model): document the architecture stages and the Blender scene`.

**Checkpoint (orchestrator):** full run timed on the capture, trace `12-bureau-final`, whole-branch review.
