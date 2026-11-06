# Room model M1 (shell, measured) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CPU-only Python pipeline that turns a House Capture export into a measured architectural
shell (floor, walls, ceiling levels, beams, columns) with a sigma on every parameter, exported as a
named-node GLB, a dimensioned plan, a dimensions table and a report.

**Architecture:** Package `pipelines/room_model/` with four stages run by
`python -m room_model <stage>`: `registration` (photo poses, port of the spike), `shell` (LiDAR
fits seeded by RoomPlan → `shell.json`), `model` (stations and derived dimensions → `model.json`),
`export` (GLB, SVG, CSV, report from `model.json`). Every geometric stage works in the room frame
of `bundle_to_room.py`. Tests run on a synthetic capture generator with known truth.

**Tech Stack:** Python 3.12 (uv), numpy, scipy, plyfile, shapely, pycolmap, pillow, pytest.

**Spec:** `docs/specs/room-model-pipeline.md` — its "Decisions" section overrides the rest.
Spike to port from: `/mnt/d/dev/spikes/couloir-photogrammetry/` (`scripts/`, `REPORT.md`, `ENV.md`).

## Global Constraints

- Worktree `~/dev/3d-model-anything-room-model`, branch `feat/room-model-pipeline`. Never touch `~/dev/3d-model-anything` (the main checkout, on its own branch with the user's uncommitted files).
- Python 3.12 venv at `pipelines/room_model/.venv` (git-ignored): `uv venv --python 3.12 pipelines/room_model/.venv && uv pip install --python pipelines/room_model/.venv/bin/python -r pipelines/room_model/requirements.txt`.
- Pins (`pipelines/room_model/requirements.txt`): `numpy==2.5.4 scipy==1.18.1 plyfile==1.1.5 shapely==2.2.0 pycolmap==4.2.1 pillow==13.0.0 pytest==9.2.0`. No open3d, no torch, no GPU.
- CLI, run from `pipelines/room_model/`: `.venv/bin/python -m room_model <registration|shell|model|export|all> --capture <dir> --out <dir> [--room 0] [--turn 0] [--force] [--keep-work]`. A stage whose output exists is skipped unless `--force`.
- Outputs only under `--out` (convention `data/room-model/<room>/`, git-ignored). < 500 MB per room; shell + model + export < 5 min per room on CPU.
- Frame: shell, model and export use the room frame of `.agents/skills/scan-to-threejs/scripts/bundle_to_room.py` (`transform.worldToRoom`, metres, +Y up). Plan coordinates are `[x, z]`. Registration stays in the ARKit frame.
- **Private data:** never commit anything derived from a real capture (coordinates, dimensions, photos, outputs, reference polygons), in code, tests, docs or commit messages. Tests use the synthetic generator only.
- Language: code, identifiers, JSON keys, CSV headers, node names, comments, commit messages in English. User-visible text inside generated files (`report.md`, `plan.svg` labels, the `label` column and field of dimensions) in French with correct accents.
- Comments ≤ 3 lines, docstrings ≤ 10 lines; check with `~/.claude/scripts/comment-budget.sh 2eb71d3`.
- Tests are behavioural, on synthetic captures; no mocks of this package's own modules. Run: `cd pipelines/room_model && .venv/bin/python -m pytest -q`.
- Shell: `rm`/`cp`/`mv` are aliased to `-i`; use `\rm`, `command cp`, `mv -f`.
- Commit only your task's files (`git add <paths>`, never `-A`; never `docs/specs/` or `docs/plans/`). Conventional Commits, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- ARKit face classes in the PLY `class`: 0 none, 1 wall, 2 floor, 3 ceiling, 4 table, 5 seat, 6 window, 7 door.

## Review Focus

1. RoomPlan splits one wall into two collinear walls → the corner between them must come from projection, not from intersecting two near-parallel lines (test in Task 2).
2. The PLY holds faces of an earlier room (`inroom == 0`) next to a wall → they must not move that wall (test in Task 2).
3. RoomPlan has a wall that is not on the floor outline (interior partition) → listed in diagnostics, no crash, labels unchanged (test in Task 1).
4. A patch of ceiling the LiDAR never saw → no spurious ceiling level and no beam (test in Task 3).
5. A tall cabinet classified as wall, standing against a wall but not reaching the ceiling → not a column, listed as unexplained vertical surface (test in Task 4).

## Shared formats

**`shell.json`** (written by `shell`, extended by Tasks 3–4), all coordinates in the room frame:

```json
{
  "schema": "room-model/shell-1",
  "source": {"capture": "<dir name>", "room": 0},
  "frame": {"world_to_room": [16 floats, column-major], "turn": 0},
  "parameters": {"wall.M1.offset": {"value": 1.234, "sigma": 0.011, "unit": "m", "source": "lidar"}},
  "elements": [
    {"id": "Floor", "kind": "floor", "params": {"level": "floor.level"}, "polygon": [[x, z], ...]},
    {"id": "Wall_M1", "kind": "wall", "label": "M1", "roomplan_id": "...", "start": [x, z], "end": [x, z], "flags": [],
     "params": {"azimuth": "wall.M1.azimuth", "offset": "wall.M1.offset", "thickness": "wall.M1.thickness", "top": "wall.M1.top"}},
    {"id": "Ceiling_1", "kind": "ceiling", "params": {"level": "ceiling.1.level"}, "region": [{"exterior": [[x, z], ...], "holes": []}]},
    {"id": "Beam_1", "kind": "beam", "axis": [[x, z], [x, z]], "params": {"underside": "beam.1.underside", "width": "beam.1.width", "top": "beam.1.top"}},
    {"id": "Column_1", "kind": "column", "engaged_wall": null,
     "params": {"center_x": "column.1.center_x", "center_z": "column.1.center_z", "size_x": "column.1.size_x", "size_z": "column.1.size_z", "top": "column.1.top"}}
  ],
  "diagnostics": {"walls": {}, "floor": {}, "ceilings": [], "unexplained_vertical": [], "off_outline_walls": [], "flags": []}
}
```

- A wall's interior normal is `(cos azimuth, sin azimuth)` in `(x, z)`; its line is `n · (x, z) = offset`. `start`/`end` are the polygon corners in outline order. `parameters[*].source` ∈ `lidar | roomplan | assumed`; `unit` ∈ `m | rad`. Levels (`floor.level`, `ceiling.k.level`, `beam.k.underside`, `*.top`) are absolute `y` in the room frame.
- **`model.json`** = `shell.json` with `"schema": "room-model/1"`, `"version": "v1-scan"` and `"dimensions": [{"id", "kind", "label", "value", "sigma", "elements", "at"}]` (metres; `at` is `[x, z]` or `null`).

---

### Task 1: Package, inputs and synthetic capture

Tier: `sonnet-medium`.

**Files:**
- Create: `pipelines/room_model/requirements.txt`, `pipelines/room_model/pytest.ini` (`pythonpath = .`, `testpaths = tests`)
- Create: `pipelines/room_model/room_model/__init__.py`, `__main__.py`, `context.py`, `capture.py`, `frame.py`, `roomplan.py`
- Create: `pipelines/room_model/tests/synthetic.py`, `tests/conftest.py`, `tests/test_inputs.py`, `tests/test_cli.py`
- Modify: `.gitignore` (add `data/room-model/` and `pipelines/room_model/.venv/`)

**Interfaces:**
- Produces:
  - `context.StageContext(capture: Path, out: Path, room: int, turn: int, force: bool, keep_work: bool)`; `context.STAGE_OUTPUTS = {"registration": "registration/poses.json", "shell": "shell.json", "model": "model.json", "export": "model_measured.glb"}`; `context.require(ctx, stage) -> Path` raising `StageInputMissing("run <stage> first: <path>")`.
  - `__main__`: stages resolved lazily from `{"registration": "room_model.registration", "shell": "room_model.shell", "model": "room_model.model", "export": "room_model.export"}`, each module exposing `run(ctx) -> Path`; `all` runs them in that order; a missing module exits 2 with `stage <name> not implemented`. Later tasks never edit `__main__.py` except Task 8 (`compare` subcommand).
  - `capture.py` (port of spike `capture_io.py`): `colmajor`, `load_manifest`, `load_photos(capture, room) -> list[Photo]`, `mesh_path(capture, room)`, `load_mesh_ply(path) -> dict` (keys as the spike), `load_roomplan(capture, room) -> dict`.
  - `frame.world_to_room(capture: Path, room: int, turn: int) -> np.ndarray` (4×4): runs `bundle_to_room.py <capture> --room N --turn K -o <tmp>` with `sys.executable` and reads `transform.worldToRoom`. Never reimplement the heading logic.
  - `roomplan.py` (port of spike `room_model.py`): `Room(data: dict, world_to_room: np.ndarray)` with walls labelled `M1..Mn` in floor-outline order, `doors` `P1..`, `floor_y`, `outline` (room frame), `station_candidates(...)` unchanged, plus `off_outline: list[str]` (RoomPlan wall ids matched to no outline edge). An outline edge with no wall within 0.4 m gets a seed wall `ident="outline-<k>"` and is listed in `seeded_from_outline`. Replace the spike's assertion and its matplotlib dependency (use shapely).
  - `tests/synthetic.py`: `make_capture(root: Path, options: SyntheticOptions) -> SyntheticTruth`, writing a `house-capture/5` folder (`manifest.json`, `rooms/room-0.json`, `mesh/room-0.ply`, `mesh/room-0-colors.bin`, `photos/photos.json` = `[]`).

**Synthetic scene** (local frame, metres; truth values the later tests assert against):
- L-shaped floor, outline `(0,0) (6,0) (6,2.5) (2.5,2.5) (2.5,4.5) (0,4.5)`; wall names `south` (z=0), `east` (x=6), `step` (z=2.5, x 2.5–6), `wing_east` (x=2.5), `north` (z=4.5), `west` (x=0).
- Ceilings: 2.50 over the main part, 2.30 over the wing (x 0–2.5, z 2.5–4.5), with the vertical riser between them.
- Beam: x 3.85–4.15, z 0–2.5, underside 2.20, top 2.50 (two side faces + soffit; no ceiling faces above it).
- Columns: free-standing x 1.30–1.70, z 1.00–1.40, floor to 2.50; engaged on `south` x 5.0–5.5, z 0–0.25, floor to 2.50.
- Door on `east`: centre z 1.25, width 0.90, height 2.05 — hole in the wall, leaf 5 cm behind the wall face, class 7.
- Clutter: table against `west` (x 0–0.6, z 0.5–1.7, h 0.75, class 4); tall cabinet against `west` (x 0–0.6, z 3.0–4.0, h 1.90, **class 1**); earlier-room panel 5 cm behind `south` (z = −0.05, x 1–3, y 0–2, normal −z, class 1, `inroom = 0`). No ceiling faces above column footprints.
- Mesh: surfaces on a 0.05 m grid, two triangles per cell, vertex noise N(0, 0.004) along the face normal (seeded), per-face class, `inroom = 1` except the panel, `coverage` = 1.
- World placement: yaw 0.5 rad about +Y, then translation (1.2, −1.45, −0.8).
- RoomPlan JSON: same keys as a real `rooms/room-N.json` (read the spike copy `/mnt/d/dev/spikes/couloir-photogrammetry/data/couloir/rooms/room-0.json` for shape only). Walls are the true walls offset along their normal by `[+0.02, −0.015, +0.03, −0.02, +0.01, −0.025]` m and turned by ±0.8° alternately, height 2.45, transform z axis pointing into the room, rectangular `polygonCorners`. Floor 0.01 m above the true floor, `polygonCorners` = corners of the perturbed walls, floor transform with local z = world up as in real captures. One door on `east`. No beam, no column.
- `SyntheticOptions(seed=0, noise_m=0.004, cell_m=0.05, split_south_wall=False, extra_roomplan_wall=False, drop_wall_faces: str | None = None, ceiling_hole=False, other_room_panel=True)`; `split_south_wall` makes RoomPlan emit `south` as two collinear walls split at x = 3; `extra_roomplan_wall` adds a RoomPlan wall at x = 1.0, z 2.6–4.4 not on the outline; `drop_wall_faces` removes the mesh faces of one named wall; `ceiling_hole` removes the 2.50 ceiling over x 4.6–5.6, z 1.0–2.0.
- `SyntheticTruth`: `world_from_local` (4×4, so tests map local truth into the room frame with `world_to_room`), `wall_ident: dict[str, list[str]]` (true name → RoomPlan identifiers, two for a split wall), `split_point` (local `[x, z]`), `widths: {("south","step"): 2.50, ("west","east"): 6.00, ("west","wing_east"): 2.50, ("south","north"): 4.50}`, `ceilings: [2.50, 2.30]`, `beam`, `columns` (dicts with the values above), heights relative to the true floor.

- [ ] **Step 1: Write the failing tests** in `tests/test_inputs.py`:
  - `test_room_frame_aligns_walls`: on the default synthetic capture, every RoomPlan wall normal mapped by `world_to_room` has `max(|n_x|, |n_z|) > 0.999` and the RoomPlan floor maps to |y| < 0.03.
  - `test_walls_follow_outline_order`: `Room` has 6 walls `M1..M6`, each true name maps to exactly one label, one door `P1` attached to the `east` wall.
  - `test_off_outline_wall_is_reported`: with `extra_roomplan_wall=True`, still 6 outline walls with the same labels, and `room.off_outline` holds the extra wall's identifier.
  - `test_mesh_roundtrip`: `load_mesh_ply` returns faces with classes {1,2,3,4,7} present and `inroom` 0 only on the panel faces.
- [ ] **Step 2: Write the failing CLI tests** in `tests/test_cli.py`: `model` before `shell` exits non-zero with `run shell first`; an unknown stage module exits 2 with `not implemented`; a stage is skipped when its output exists and rerun with `--force` (use a temporary fake stage module registered through the same lazy mechanism).
- [ ] **Step 3: Run** `.venv/bin/python -m pytest -q` → the new tests FAIL.
- [ ] **Step 4: Implement** the files above; the generator writes the PLY with plyfile (vertex `x y z nx ny nz red green blue coverage inroom`, face `vertex_indices class`).
- [ ] **Step 5: Run** the tests → PASS. Also run `frame.world_to_room` on the spike's couloir copy and check it equals `transform.worldToRoom` of the couloir's reference `room.json` (path on the private drive, given in the brief) to 1e-9 (print only the max difference, never the matrix).
- [ ] **Step 6: Commit** `feat(room-model): add package, capture inputs and synthetic capture`.

### Task 2: Shell — walls, floor, polygon

Tier: `sonnet-medium`.

**Files:**
- Create: `room_model/lidar.py`, `room_model/planes.py`, `room_model/shell.py`, `tests/test_shell.py`

**Interfaces:**
- Consumes: Task 1 (`capture`, `frame`, `roomplan.Room`, `StageContext`).
- Produces:
  - `lidar.Samples(P, N, cls)` (room frame, float64 / int); `lidar.load_samples(ctx, world_to_room, density=4000.0, seed=0) -> Samples`: keep faces whose 3 vertices have `inroom == 1`, drop welded components smaller than the spike's `MIN_COMPONENT_AREA` (port `lidar_baselines.build_clean` + `geom_io.weld_components`), then sample points on faces proportionally to area with face normal and class.
  - `planes.PlaneFit(normal, point, rms, n_inliers, support_m2, along_std)`; `planes.fit_plane(P, seed_n, seed_p, window, min_threshold=0.012, iterations=8) -> PlaneFit | None` (port `trimmed_plane_fit` + supported-area raster of `plane_stats`; `along_std` = std of inlier coordinates along the horizontal in-plane axis).
  - `planes.offset_sigma(fit) = hypot(rms / sqrt(max(support_m2 / 0.01, 1)), SIGMA_SYS_OFFSET)` with `SIGMA_SYS_OFFSET = 0.01`; `planes.azimuth_sigma(fit) = hypot(rms / (sqrt(max(support_m2 / 0.01, 1)) * max(along_std, 0.05)), SIGMA_SYS_AZIMUTH)` with `SIGMA_SYS_AZIMUTH = radians(0.25)`.
  - `shell.run(ctx) -> Path` writing `shell.json` (format above) with `Floor` and `Wall_*` elements; Tasks 3–4 add their elements through `shell.build(ctx) -> dict`, which `run` serialises.

**Rules:**
- Wall candidates (port spike `wall_candidates`): within 0.08 m of the RoomPlan seed plane, |s| < w/2 − 0.10, height from floor + 0.2 to RoomPlan top − 0.2, |N·n_seed| > 0.75, class ∉ {4, 5, 6, 7}, outside each RoomPlan door rectangle grown by 0.08. Fewer than 200 candidates or a failed fit → keep the RoomPlan plane, `source: roomplan`, sigma 0.05 m / 2°, flag `roomplan_fallback`. Support < 0.5 m² → flag `low_support`.
- Plan parameters from the 3D fit: azimuth from the horizontal part of the normal, offset of the fit point; wall tilt (degrees from vertical) goes to `diagnostics.walls`.
- Floor: samples with N_y > 0.9, class ∉ {4, 5}, |y − RoomPlan floor| < 0.08, at least 0.12 m inside the outline; `floor.level` = fitted plane height at the polygon centroid; tilt to diagnostics.
- Corners: consecutive walls with normals more than 10° apart → line intersection; otherwise the midpoint of the RoomPlan corner projected on both lines, flag `near_parallel_corner`.
- `wall.<label>.thickness` = 0.20, sigma 0.05, `source: assumed` (no flag). `wall.<label>.top` = RoomPlan top, `source: roomplan` (Task 3 replaces it).
- `diagnostics.walls[label]`: `rms_cm`, `support_m2`, `offset_vs_roomplan_cm`, `angle_vs_roomplan_deg`, `tilt_deg`, `n_points`, `centroid` (`[x, z]` of the inliers, used by Task 5); `diagnostics.off_outline_walls` from `Room.off_outline`.
- `shell.wall_lines(doc) -> list[dict]` with `label`, `normal` (2,), `offset`, `start`, `end` for each wall, read by Tasks 3–4.

- [ ] **Step 1: Write the failing tests** in `tests/test_shell.py` (helper: distance between two fitted walls by true name, measured at the middle of their overlap):
  - `test_widths_recovered`: for each pair in `truth.widths`, |fitted − truth| < 0.01 m and |fitted − truth| < 3 × hypot(sigma_a, sigma_b).
  - `test_well_supported_wall_sigma_is_tight`: every `lidar` wall with support > 2 m² has offset sigma < 0.015 m.
  - `test_floor_level`: |`floor.level` − true floor in room frame| < 0.005 m.
  - `test_clutter_does_not_bias_west_wall`: the `west` offset error < 0.005 m despite the table and the cabinet.
  - `test_missing_wall_falls_back_to_roomplan`: `drop_wall_faces="wing_east"` → that wall has `source: roomplan`, flag `roomplan_fallback`, sigma ≥ 0.05; widths not involving `wing_east` still within 0.01 m.
  - `test_split_wall_corner_is_stable`: `split_south_wall=True` → 7 walls, the corner between the two south halves lies within 0.02 m of the true south line and within 0.05 m of `truth.split_point`, flagged `near_parallel_corner`.
  - `test_other_room_faces_ignored`: the `south` offset differs by less than 1 mm between the default capture and `other_room_panel=False` (the panel sits inside the 0.08 m fit window, so only the `inroom` filter keeps it out).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `lidar.py`, `planes.py`, `shell.py`.
- [ ] **Step 4: Run** → PASS, and `python -m room_model shell` on the synthetic capture writes a `shell.json` that validates against the format above.
- [ ] **Step 5: Commit** `feat(room-model): fit walls and floor on LiDAR seeded by RoomPlan`.

### Task 3: Ceiling levels, regions and beams

Tier: `sonnet-medium`.

**Files:**
- Create: `room_model/ceilings.py`, `tests/test_ceilings.py`
- Modify: `room_model/shell.py` (call `ceilings.detect` inside `build`, set wall tops)

**Interfaces:**
- Consumes: `Samples`, `fit_plane`, `offset_sigma`, the floor polygon and floor level from Task 2.
- Produces: `ceilings.detect(samples, polygon: shapely.Polygon, floor_level: float) -> CeilingResult(levels, beams, unobserved_fraction: float)`, where each level is `{"level", "sigma", "region": shapely.MultiPolygon, "support_m2"}` and each beam `{"axis": [[x, z], [x, z]], "underside", "underside_sigma", "width", "width_sigma", "top"}`; `shell.build` turns them into `Ceiling_<k>` / `Beam_<k>` elements and parameters, ceilings numbered by descending level.

**Algorithm:**
1. Candidates: N_y < −0.9, y > floor + 1.6, at least 0.05 m inside the polygon.
2. Levels: spike `ceiling_levels` (greedy over a 1 cm histogram, window 0.03) with `min_area = 0.3 m²`, `max_levels = 8`; refit each level with `fit_plane` for its value and sigma.
3. Per level: occupancy on a 2 cm plan grid (cells with ≥ 1 point within ±0.03 of the level), binary closing with 2 iterations, 8-connected components ≥ 0.1 m².
4. Component shape: principal axes of its cell centres; major/minor extent = 2nd–98th percentile spans. **Beam** if minor ∈ [0.08, 0.80] m, major/minor ≥ 3, and at least one long side has, within 0.15 m, cells of a higher level. Else it belongs to that level's ceiling region.
5. Beam: axis = major axis through the component centre, clipped to the polygon; `underside` = level (fit sigma); `width` = distance between the two side-face planes when each has ≥ 0.05 m² of vertical points inside the strip grown by 0.10 m (sigma from `offset_sigma`), else the minor extent with sigma 0.02; `top` = the lowest higher level adjacent to its long sides.
6. Region per level = union of its non-beam components as shapely polygons, simplified at 0.02, intersected with the floor polygon. A level whose region is empty after beams are removed is dropped. `unobserved_fraction` = 1 − area(all regions and beams) / area(polygon).
7. Wall tops: points every 0.10 m along each wall, 0.15 m inside the room; `top` = max level of the regions containing them, `source: lidar`; none → keep the RoomPlan top.

- [ ] **Step 1: Write the failing tests** in `tests/test_ceilings.py` (heights relative to `floor.level`):
  - `test_two_ceiling_levels`: exactly two `Ceiling_*` elements at 2.50 and 2.30 (± 0.01), the 2.30 region covering ≥ 90 % of the wing.
  - `test_beam_recovered`: one `Beam_*`, underside 2.20 ± 0.01, width 0.30 ± 0.02, axis parallel to the `east` wall within 1°, centreline 4.00 ± 0.02 m from the `west` wall, length ≥ 2.3 m, top 2.50 ± 0.01.
  - `test_wall_tops_follow_ceilings`: `north` and `wing_east` tops at 2.30 ± 0.01, `south` and `east` tops at 2.50 ± 0.01, all `source: lidar`.
  - `test_ceiling_hole_creates_no_beam`: `ceiling_hole=True` → still one beam and two levels, `unobserved_fraction` > 0.04.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `ceilings.py` and the `shell.build` integration.
- [ ] **Step 4: Run** the whole suite → PASS.
- [ ] **Step 5: Commit** `feat(room-model): detect ceiling levels and beams`.

### Task 4: Columns

Tier: `sonnet-medium`.

**Files:**
- Create: `room_model/columns.py`, `tests/test_columns.py`
- Modify: `room_model/shell.py` (call `columns.detect` inside `build`)

**Interfaces:**
- Consumes: `Samples`, fitted wall lines, floor level, ceiling regions and levels from Task 3.
- Produces: `columns.detect(samples, walls: list[dict] (from shell.wall_lines), polygon: shapely.Polygon, floor_level: float, ceilings: list[dict] (CeilingResult.levels)) -> ColumnResult(columns: list[dict], unexplained: list[dict])`; `shell.build` adds `Column_<k>` elements (numbered by x then z) and `diagnostics.unexplained_vertical`.

**Algorithm:**
1. Vertical samples: |N_y| < 0.2, class ∉ {4, 5, 6, 7}, inside the floor polygon, y from floor + 0.3 to the lowest ceiling level (not beam) − 0.3, farther than 0.04 m from every fitted wall line within that wall's extent ± 0.10.
2. Plan density at 1 cm cells; a cell is occupied when its count ≥ 0.5 × density × 0.01 × band height. Binary closing 1 iteration, 8-connected components with ≥ 40 cells.
3. Per component: axis-aligned bounding rectangle in the room frame. **Engaged** when one side is within 0.06 m of a wall line (that side snaps to the wall); occupied faces required on ≥ 3 sides (free-standing) or on the 3 non-wall sides (engaged); both sizes in [0.10, 1.50] m; and vertical samples (without the band limit) reaching within 0.15 m of the ceiling level above the centre. All three hold → column; else → `unexplained` entry (`bbox`, `height_max`, `reason`).
4. Parameters: centre and sizes, sigma 0.01 per face (`hypot` for sizes); `top` = ceiling level above the centre, `source: lidar`.

- [ ] **Step 1: Write the failing tests** in `tests/test_columns.py`:
  - `test_free_standing_column`: one column with sizes 0.40 ± 0.02, centre within 0.02 of truth (via distances to `west` and `south`), `engaged_wall` null, top 2.50 ± 0.01 above the floor.
  - `test_engaged_column`: one column engaged on the `south` wall's label, size along that wall 0.50 ± 0.02, depth 0.25 ± 0.02.
  - `test_tall_cabinet_is_not_a_column`: exactly 2 columns; one `unexplained` entry overlapping the cabinet footprint.
  - `test_beam_is_not_a_column`: no column or unexplained entry overlapping the beam footprint.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** the whole suite → PASS.
- [ ] **Step 5: Commit** `feat(room-model): detect free-standing and engaged columns`.

### Task 5: Model stage — stations and dimensions

Tier: `sonnet-low`.

**Files:**
- Create: `room_model/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `shell.json` (format above), `roomplan.Room.station_candidates`.
- Produces: `model.run(ctx) -> Path` writing `model.json`; `model.line_sigma(sigma_offset, sigma_azimuth, delta_s) -> float = sqrt(sigma_offset² + (sigma_azimuth · delta_s)²)`, where `delta_s` is the distance along the wall from `diagnostics.walls[label].centroid`.

**Dimensions** (`id`, `kind`, French `label`):
- Stations: port spike `eval_geometry.build_stations` so ids `S01..` and their walls match the spike exactly; value = distance between the two fitted lines along wall A's normal at the station point; sigma = hypot of both `line_sigma`; kind `width` or `length` as the spike; label e.g. `Largeur M1–M3 à 1,20 m de M1/M2`.
- `H<k>`: ceiling level − floor level, sigma hypot of both; label `Hauteur sous plafond C<k>`.
- `B<k>-u` (`beam_underside`, `Hauteur sous poutre B<k>`), `B<k>-w` (`beam_width`, `Largeur de la poutre B<k>`).
- `COL<k>-x`, `COL<k>-z` (`column_size`), and `COL<k>-dx`, `COL<k>-dz` (`column_offset`: distance from the nearest face to the nearest wall parallel to it, labels name that wall).

- [ ] **Step 1: Write the failing tests**:
  - `test_line_sigma`: `line_sigma(0.01, 0.0, 5.0) == 0.01`; `line_sigma(0.0, 0.002, 3.0) == pytest.approx(0.006)`.
  - `test_station_widths`: on the synthetic model, every `width` station between `south` and `step` is 2.50 ± 0.01 and |error| < 3σ.
  - `test_heights_and_structures`: `H1` 2.50 ± 0.01, `H2` 2.30 ± 0.01, `B1-u` 2.20 ± 0.01, `B1-w` 0.30 ± 0.02, two columns with four dimensions each.
  - `test_model_json_roundtrip`: `model.json` loads, `schema == "room-model/1"`, `version == "v1-scan"`, every element param id exists in `parameters`, every dimension has `sigma > 0`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(room-model): derive key dimensions with uncertainties`.

### Task 6: Export stage — GLB, plan, CSV, report

Tier: `sonnet-medium`.

**Files:**
- Create: `room_model/gltf.py`, `room_model/meshes.py`, `room_model/plan_svg.py`, `room_model/report.py`, `room_model/export.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: `model.json`; optional `registration/poses.json` (summary block, Task 7).
- Produces: `export.run(ctx) -> Path` writing `model_measured.glb`, `plan.svg`, `dimensions.csv`, `report.md`; `gltf.write_glb(path, nodes: list[GltfNode])` with `GltfNode(name, positions, indices, material: str | None, extras: dict, children: list[str])`.

**Rules:**
- `gltf.py`: minimal glTF 2.0 binary writer (no new dependency): one buffer, float32 positions, uint32 indices, flat normals by duplicated vertices, one material per element kind (`floor`, `wall`, `ceiling`, `beam`, `column`) with a fixed base colour and `doubleSided: true`, node `extras`. Output must be byte-identical for identical input (sorted keys, no timestamps).
- Node tree: root `Coque` → `Floor`, `Ceiling_<k>`, `Wall_<label>`, `Beam_<k>`, `Column_<k>`. Each node's `extras`: `{"kind", "params": {name: {"id", "value", "sigma", "source"}}, "flags"}`.
- `meshes.py`: floor = polygon triangulated with `shapely.constrained_delaunay_triangles` at `floor.level`; ceiling = each region polygon at its level; wall = box behind the interior face from `start` to `end`, extended by its thickness at both ends, `thickness` deep, from `floor.level` to `top`; beam and column = boxes. Room frame written unchanged (glTF is +Y up). No door openings in M1.
- `plan_svg.py`: top view, x right and z down, 1 m = 100 px, polygon with wall labels, ceiling regions shaded with their height, beams dashed with `B<k> s/p <height>`, columns filled, each station as a dimension line labelled `<value> ± <sigma>` in cm, a 1 m scale bar. Element ids as SVG `id` attributes.
- `dimensions.csv`: header `id,kind,label,value_cm,sigma_cm,source,elements`, values with one decimal, `source` = the weakest source among the parameters used (`assumed` < `roomplan` < `lidar`), `elements` joined with `;`.
- `report.md` (French): sections `Entrées`, `Repère`, `Recalage` (from `poses.json` when present, else « non exécuté »), `Murs` (table: label, source, σ cm, RMS cm, support m², écart et angle vs RoomPlan, inclinaison, alertes), `Sol`, `Plafonds` (with the unobserved fraction), `Poutres`, `Poteaux`, `Surfaces verticales non expliquées`, `Orthogonalité` (deviation from 90°/180° between consecutive walls), `Alertes` (every flag in plain French, e.g. `roomplan_fallback` → « mur non vu par le LiDAR, position RoomPlan conservée »), `Épaisseurs` (« supposées à 20 cm »).

- [ ] **Step 1: Write the failing tests** in `tests/test_export.py` (parse the GLB JSON chunk with `struct` + `json`, no extra library):
  - `test_glb_nodes_and_extras`: node names are exactly `Coque`, `Floor`, `Ceiling_1`, `Ceiling_2`, `Wall_M1..M6`, `Beam_1`, `Column_1`, `Column_2`; every element node's `extras.params` has sigmas.
  - `test_glb_geometry_matches_model`: `Beam_1`'s position accessor min y equals `beam.1.underside` within 1 mm, each `Wall_*` max y equals its `top` within 1 mm, `Floor` y equals `floor.level`.
  - `test_export_is_deterministic`: two exports → identical bytes for the GLB, SVG and CSV.
  - `test_plan_svg`: parses with `xml.etree`, contains `id` for every wall, beam and column, and a text for every station id.
  - `test_dimensions_csv`: one row per dimension, header exact, `S01` value equal to the model value in cm at 0.1.
  - `test_report_flags_fallback`: with `drop_wall_faces="wing_east"`, the report's `Alertes` section names that wall's label and says « non vu par le LiDAR ».
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** the whole suite → PASS.
- [ ] **Step 5: Commit** `feat(room-model): export measured GLB, plan, dimensions and report`.

### Task 7: Registration stage (port)

Tier: `sonnet-medium`. Independent of Tasks 2–6 (only `registration.py` and its test); can run in parallel with them.

**Files:**
- Create: `room_model/registration.py`, `tests/test_registration.py`

**Interfaces:**
- Consumes: `capture.load_photos`, `StageContext`.
- Produces: `registration.run(ctx) -> Path` writing `registration/poses.json` and the COLMAP model in `registration/sparse/`:
  ```json
  {"schema": "room-model/registration-1", "frame": "arkit",
   "summary": {"reprojection_px_arkit": 0.0, "reprojection_px_refined": 0.0, "sim3_scale": 1.0, "points3d": 0, "refined": 0, "weak": 0, "seconds": 0.0},
   "photos": [{"name": "0000.jpg", "c2w_arkit": [16 floats, column-major], "refined": true, "n_obs": 0, "correction_cm": 0.0, "correction_deg": 0.0}]}
  ```
- Also produces, for the tests: `select_pairs(photos, max_dist, max_angle, neighbors) -> list[tuple[str, str]]` (spike semantics: a pair when close and similarly oriented, or when the sequence gap ≤ `neighbors`), `umeyama(src, dst) -> (scale, R, t)`, `keep_weak_at_arkit(refined: dict, arkit: dict, n_obs: dict, min_obs=50) -> dict`.

**Rules:** port spike `to_colmap.py` + `refine_poses.py` with their parameters (SIFT `peak_threshold 0.003`, `max_features 16384`, BA on images with ≥ 50 observations, Sim3 back to ARKit, weak images kept at the ARKit pose). Read the spike `REPORT.md` and `ENV.md` for the pycolmap 4.x traps; key images by name, never by id. Images and `database.db` live in `registration/work/`, deleted at the end unless `--keep-work`.

- [ ] **Step 1: Write the failing tests**:
  - `test_select_pairs`: on 20 synthetic poses along a corridor (some facing opposite ways), the result is exactly the set of `(i < j)` pairs with (distance < `max_dist` and angle < `max_angle`) or `j − i ≤ neighbors`, no duplicates; two opposite-facing cameras far apart in the sequence are not paired.
  - `test_umeyama_recovers_similarity`: a known (scale 1.01, 30° yaw, translation) is recovered to 1e-9 on noiseless points.
  - `test_weak_photos_keep_arkit_pose`: photos with `n_obs < 50` return their ARKit pose exactly, the others their refined pose.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Run on the couloir copy** (`--capture /mnt/d/dev/spikes/couloir-photogrammetry/data/couloir --out data/room-model/couloir`): refined reprojection ≤ 0.8 px, |sim3_scale − 1| ≤ 0.003, weak count between 20 and 45, median camera-centre difference to the spike's `colmap/sparse/0` (by name) ≤ 1 cm, `registration/work/` gone, wall time noted.
- [ ] **Step 6: Commit** `feat(room-model): port ARKit pose refinement`.

### Task 8: Compare, docs and couloir validation

Tier: `sonnet-medium` (escalate a real-data discrepancy of unknown cause to `sonnet-high`).

**Files:**
- Create: `room_model/compare.py`, `tests/test_compare.py`, `docs/room-model.md`
- Modify: `room_model/__main__.py` (`compare --model <model.json> --reference <ref.json> [-o compare.md]`)

**Interfaces:**
- Reference format (generic, any source): `{"polygon": [[x, z], ...], "edge_heights": [...], "dimensions": {"S01": 2.5}}`, every key optional, metres, room frame.
- `compare.compare(model: dict, reference: dict) -> str` (Markdown): per model wall, the reference edge with |angle| < 5° and the nearest midpoint, offset Δ (cm) and angle Δ (°); wall top vs that edge's height; per shared dimension id, model, reference, Δ cm and Δ/σ.

- [ ] **Step 1: Test** `test_compare_detects_offset`: a reference polygon equal to the synthetic model polygon with one edge moved by 3 cm reports Δ = 3.0 cm on that wall only.
- [ ] **Step 2: Implement**, run the suite → PASS. Write `docs/room-model.md` (setup, CLI, outputs, frame, sigma model, limits of M1) in the style of `docs/export.md`. Commit `feat(room-model): add compare command and docs`.
- [ ] **Step 3: Validate on the couloir** (outputs only in `data/room-model/couloir/`, references built in the session scratchpad, nothing committed): run `all`; build `reference-llm.json` from `~/dev/3d-model-anything/demos/couloir/shell.ts` (`POLY`, `EDGE_HEIGHT`, read-only) and `reference-spike.json` from the spike `out/eval/geometry.json` (`lidar_clean` station widths); run `compare` against both.
- [ ] **Step 4: Check and write `data/room-model/couloir/validation.md`:** station widths within 1.5 cm of the spike's; wall RMS within 0.2 cm of the spike's planes (support ≥ 0.5 m²); edges shared with the LLM model within 3 cm and wall tops within 3 cm; the corridor beam the spike saw (height given in the brief) found as `Beam_*`; the pillar the LLM model drew and RoomPlan missed found as a column or listed as unexplained; shell + model + export time, `du -sh` of the output dir (< 500 MB). Every miss is explained or turned into a fix task.
- [ ] **Step 5 (optional, asks first):** smoke-run `shell`, `model`, `export` on atrium and trone from their Drive zips extracted into the scratchpad (~0.6 GB, deleted afterwards); no crash, reports read.

---

Budget: about 1–1.5 h of agent work, CPU only; venv ≈ 0.4 GB (git-ignored); outputs < 500 MB per room.
Out of scope (later milestones): openings and wall thickness from jambs (M3), networks (M2), squared variant, `room.json`, `rooms.mjs` stage and Blender import script (M5), measurement plan and adjustment (M4).
