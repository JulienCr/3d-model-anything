# Room model pipeline — spec (draft)

Status: draft for review. Branch `feat/room-model-pipeline`, based on `feat/gltf-blender-export`.

## Goal

Turn a House Capture export (RoomPlan + LiDAR mesh + depth frames + ARKit-posed photos) into an
**accurate, clean architectural model** that a person can import into Blender to plan work
(partitions, sets) — without an LLM doing the modelling.

Priorities, in order:
1. **Dimensions you can trust**, each with an uncertainty, confirmable by tape measurements.
2. **Networks** (pipes, ducts, cable trays, conduits, wall-mounted equipment): positions,
   diameters, underside heights — they are the main design constraint on site.
3. **Architectural shell**: floor, walls with real positions, ceiling levels, beams, columns,
   recesses, door and window openings.

Non-goals: photoreal rendering (a Gaussian splat is a separate, optional viewer), furniture,
materials beyond simple per-element finishes, multi-room house assembly (rooms keep their own
frame; assembly stays in the existing house tooling).

LLM budget: the LLM is used for **one task only** — writing the on-site measurement plan from a
compact JSON summary (target < 10k tokens per room). All geometry is algorithmic.

Compute budget: CPU only, < 5 min per room on a desktop, < 500 MB of outputs per room.

## Lessons from the photogrammetry spike (2026-11-05)

Spike workspace: `D:\dev\spikes\couloir-photogrammetry` (`REPORT.md`, `out/eval/`).

- ARKit poses refined by bundle adjustment on SIFT tracks: reprojection 1.54 → 0.70 px, scale
  preserved (Sim3 1.0014). Good enough to triangulate thin structures from photos.
- iPhone LiDAR, plane-fitted: walls 0.3–0.8 cm RMS, widths within 1–3 cm of RoomPlan. LiDAR is
  the geometry backbone. RoomPlan is a good **topology** (which walls, which doors) but misses
  beams, ceiling steps, columns and recesses, and its wall heights are not ceiling heights.
- Dense MVS / 2DGS / 3DGS from these captures are not usable as geometry (sparse views, blank
  walls). Photos are useful for **edges**: pipe silhouettes, door jambs, beam edges.
- The `out/eval` toolkit (RoomPlan-seeded plane fitting, stations, door detection, measurement
  template) is the seed of this pipeline.

## Inputs and frames

- Capture folder `house-capture/5` (`manifest.json`, `rooms/room-N.json`, `mesh/room-N.ply`,
  `photos/photos.json`, `depth/frames.json`). Format: `.agents/skills/scan-to-threejs/references/house-capture.md`.
- Work in the ARKit session frame; export in the room frame used by `bundle_to_room.py`
  (metres, +Y up, floor at 0), so outputs overlay the existing `<room>-lidar.glb` and
  `<room>-photos.glb` exports.
- Optional: `measurements.json` (tape measurements, see stage 6).

## Outputs (`data/room-model/<room>/`, git-ignored)

| File | Content |
|---|---|
| `model.json` | Parametric model: elements, parameters (value, sigma, sources), constraints |
| `model_measured.glb` | Raw fitted geometry, one named node per element |
| `model_squared.glb` | Same topology, walls snapped orthogonal/parallel within tolerance |
| `room.json` | Enriched room description, compatible with `room-runtime` (polygon, walls, openings) plus new fields (ceilings, beams, networks) |
| `plan.svg`, `dimensions.csv` | Dimensioned floor plan and the table of key dimensions |
| `measurement_plan.json` | Candidate measurements selected for the site visit |
| `report.md` | Deviations, uncertainties, squaring residuals, flagged inconsistencies |

Node naming: `Floor`, `Ceiling_<k>`, `Wall_<id>`, `Beam_<k>`, `Column_<k>`, `Door_<id>`,
`Window_<id>`, `Pipe_<k>`, `Duct_<k>`, `Tray_<k>`, `Box_<k>` (wall equipment). `userData` carries
the parameter ids and sigmas so Blender users can see what is measured vs inferred.

## Stages

Each stage reads the previous stage's files and is skipped when its output exists (same
convention as `tools/rooms.mjs`). Python package `pipelines/room_model/`, CLI
`python -m room_model <stage> --capture <dir> --out <dir>`.

1. **Registration** — port `refine_poses.py` from the spike: pose-prior pair selection, SIFT,
   triangulation with fixed ARKit poses, BA, Sim3 back to ARKit. Output refined poses + sparse
   points + per-photo observation counts. Photos with < 50 observations keep the ARKit pose and
   are marked weak.
2. **Shell** — RoomPlan gives walls, openings and adjacency. Each wall plane is refit on LiDAR
   points (robust, area-weighted, class-aware). Corners = intersections of adjacent fitted planes
   → closed floor polygon. Floor plane; ceiling levels from horizontal-surface histograms; beams
   (horizontal soffits below a ceiling level, bounded by vertical faces); columns and recesses from
   a 1 cm plan-view density map of vertical surfaces, compared with the RoomPlan polygon. Openings
   refined from LiDAR holes and, where visible, vertical jamb edges in photos.
3. **Networks** — LiDAR points not explained by the shell (beyond a few cm from shell surfaces)
   are clustered; each cluster is fit with primitives: cylinder (pipe, conduit), oriented box
   (duct, tray, wall equipment), with connectivity along axes (straight runs, elbows as
   junctions). Then **photo refinement**: project each pipe hypothesis into the photos that see
   it, detect the two silhouette edges near the projected lines, and solve axis + radius by
   multi-view least squares. Thin pipes the LiDAR misses are sought from photo line pairs that are
   consistent across ≥ 3 views. Every element records its sources and observation counts.
4. **Parametric model + uncertainty** — all geometry expressed as parameters (plane offsets,
   heights, axis points, radii, opening extents) with sigmas from fit residuals, coverage and
   number of views. `model.json` is the single source for the later stages.
5. **Measurement plan** — algorithmic part: generate candidate tape measurements between
   well-defined features (wall–wall widths at stations, floor–ceiling and floor–beam heights,
   pipe underside heights and offsets to walls, door widths/heights, column positions), each with
   the expected value and the parameters it constrains; greedily select ≤ N (default 15) by
   expected uncertainty reduction weighted by importance (networks and partition-relevant walls
   first). The plan drawing (SVG) is generated by code. LLM part: from a compact JSON of the
   selected candidates and plan, write the "what / where / how" instructions in the user's
   language and order the visit. Output: an Artifact page (like the reference "Relevé" page) with
   a **shared database** so values typed on site (phone) are stored, plus an offline HTML fallback
   with a copy button.
6. **Adjustment** — tape measurements (sigma ≈ 0.5 cm) are added as strong constraints, LiDAR and
   photo fits as weak priors; weighted least squares over the parametric model; measurements with
   normalised residual > 3 are flagged "remesurer" rather than silently absorbed.
7. **Export** — build meshes from parameters (walls with thickness, openings cut, beams, pipes as
   cylinders with elbows, boxes), write the two GLBs, `room.json`, plan, CSV and report. Squared
   variant: orthogonal/parallel snapping within 2° (configurable), keeping measured distances;
   residuals reported.

Integration: a `room_model` stage in `tools/rooms.mjs` after `reconstruct_bundle`; the existing
viewer can load the enriched `room.json` via `room-runtime` without a hand-written demo module.

## Validation

- Couloir first. Compare with the LLM-built model (`demos/couloir/shell.ts` on `environments`:
  polygon, per-edge heights; `data/exports/couloir/couloir.glb`) and with the spike's `out/eval`
  stations; then with tape measurements entered through the measurement plan.
- Behavioural tests on synthetic scans (box room with a beam, a column and two pipes, with noise):
  recovered dimensions within tolerance, pipe radius and underside height within tolerance,
  adjustment pulls the model to injected tape measurements and flags an injected outlier.
- Then atrium and trone, unchanged code.

## Milestones (each one shippable, each one budgeted)

1. **M1 Shell, measured**: stages 1, 2, 4, 7 (measured GLB, plan, CSV). Port the spike's
   `capture_io`, `refine_poses`, `room_model`/`eval_*` code.
2. **M2 Networks from LiDAR**: stage 3 without photos.
3. **M3 Photo refinement**: pipes and openings from photo edges.
4. **M4 Measurement loop**: stage 5 (candidates, plan, Artifact with database) and stage 6.
5. **M5 Squared variant + integration**: `model_squared.glb`, enriched `room.json`, `rooms.mjs` stage.
6. **M6 Pipeline UI**: a local interface to run the stages on a capture and follow their progress
   (stage state, timings, logs, links to the outputs), on top of the same Python stages; the CLI stays.

## Decisions (grilling rounds 1–3, 2026-11-06) — these override the sections above

- **Source priority**: tape > photo edges > LiDAR planes > RoomPlan (topology only).
- **Walls**: RoomPlan seeds the wall list; LiDAR may add columns/recesses (flagged in the report).
- **Wall thickness**: measured at jambs or when both faces are in the scan; otherwise assumed from
  the wall type recognised by vision — plasterboard (BA13) partition 10 cm, everything else 20 cm —
  marked `assumed` and offered as a measurement candidate.
- **Networks**: parametric primitives (cylindrical straight runs + elbows, boxes). Detect down to
  16 mm; below 4 cm a network needs ≥ 3 agreeing photos, else it goes to "À vérifier".
- **Typing**: LLM vision on one contact sheet of photo crops per room types networks (detailed list:
  évacuation EU/EV/EP, eau froide, eau chaude, chauffage, gaz, ventilation, électricité (gaine),
  chemin de câbles, incendie (sprinkler, RIA), équipement mural, non identifié) **and walls**
  (béton, parpaing, BA13, autre). Best effort — "non identifié" is a valid answer. Low confidence →
  pre-filled type to confirm on the measurement page.
- **LLM execution**: two uses only (vision typing, measurement-plan wording). The pipeline writes a
  `pending/` folder (compact JSON + contact sheet); a Claude Code session or `claude -p` processes it
  and writes the answer back. No API key.
- **Unexplained geometry**: separate "scan residual" layer (decimated, grey).
- **Accuracy target after a site visit**: ±1 cm on key dimensions (widths, heights, network
  underside heights, network–wall offsets), ±2 cm elsewhere.
- **Measurement visit**: laser distance meter + tape; 15–20 measurements per room; **one page per
  visit** (several rooms), one photo crop per measurement with endpoints drawn; page = Artifact with
  a shared database, synced to `measurements.json` by a Claude session, plus a JSON download button.
- **Versions**: one model version per measurement set (`v1-scan`, `v2-releve-<date>`, …) with a
  diff report.
- **Squared variant**: 2° tolerance.
- **Blender**: GLB + import script for **Blender 5.1+** (user on 5.1, may move to 5.2 LTS):
  collections (Coque, Réseaux, À vérifier, Reste), colours by type, key dimensions as annotations.
- **Milestones**: M1 shell = walls, floor, ceilings, **beams and columns** (openings move to M3);
  M2 networks from LiDAR; M3 photos (thin networks, openings, vision typing); M4 measurement loop;
  M5 squared variant + `rooms.mjs`/viewer integration.
- **M1 validation** (no tape measurements yet): spike `out/eval` stations, the LLM-built couloir
  model, synthetic behavioural tests.

## M1.5 Architecture → Blender (grilling rounds 1–4, 2026-11-11) — overrides the sections above

Goal: a Blender scene of the **architecture only**, usable as is: walls and interior partitions,
doors, windows, through-openings and niches cut in the walls, doors and windows rigged to open,
dimensions, mouldings, walls textured from the photos. No furniture. Comes before M2 (networks).
Development room: one private capture with 3 doors, 2 windows, an interior partition carrying a
floor-height opening, and a mid-height through-niche (pass-through with shelves) that RoomPlan
did not detect.

### Delivery

- `architecture.json` (schema `room-model/arch-1`) is the single source; a Blender script
  `blender/build_scene.py` builds the scene from it. The pipeline runs it headless and writes
  `architecture.blend`, `architecture.glb` (static, doors and windows closed) and viewport
  renders. `model_measured.glb` stays as the M1 output.
- Blender 5.1+; headless runs use `BLENDER` (default: the Steam install, 5.3.0).
- Walls are joined by mitre at corners, partitions end on the face they meet, and an opening cuts
  every wall it crosses. Cornice and skirting run along the wall faces.
- Collections: `Coque`, `Menuiseries`, `Moulures`, `Cotes`, `À vérifier`, `Scan` (LiDAR mesh,
  decimated, hidden by default).

### Elements

- **Interior partitions**: a RoomPlan wall off the floor outline but inside the floor polygon is a
  partition (two faces, both in the room); no longer dropped.
- **Openings**: RoomPlan doors, windows and openings, hosted on the fitted wall or partition.
  Generic kinds: `door`, `window`, `opening` (through, floor-based), `niche` (sill above floor;
  `through: true` when open on both faces). A niche carries `dividers` (vertical) and `shelves`
  (horizontal) and a moulded `frame`.
- **Catalogue** (short, extend only when the photos show something it lacks): doors `hinged_single`,
  `hinged_double`, `sliding`, `pocket`, `french_window`; windows `casement_single`,
  `casement_double`, `tilt_turn`, `sliding`, `fixed`. Parameters: hinge side, swing (in/out),
  moulded panel count, handle.
- **Rig**: an empty parent per door/window with custom properties `Ouverture` (0–1) and two
  checkboxes, `Inverser sens` (flips swing) and `Inverser charnière` (hinges on the other jamb),
  unchecked = as detected; drivers rotate leaves (max angle property, default 90°) or translate sliding
  leaves. A double door has one slider for both leaves.
- **Mouldings**: cornice and skirting along the perimeter, architraves around openings, from a
  small profile catalogue; heights estimated by the LLM, source `assumed`. Wall panel frames
  (boiseries) optional, placed under `À vérifier`.
- **Dimensions**: MeasureIt when the extension is available, checked by the script; otherwise
  native objects (curve + arrows + text) in `Cotes`. Content: wall lengths, ceiling heights,
  width × height of every opening, window sill heights, width stations.
- **Materials**: walls, floor and ceilings get textures projected from the photos (2 cm per texel),
  with an occlusion mask (LiDAR depth plus RoomPlan object boxes) so furniture is not printed;
  masked texels are filled with a CC0 texture (`fetch_textures.py`) tinted to the measured colour.
  Joinery and mouldings get plain named materials (paint, wood, glass).

### LLM (vision)

- Stage of the pipeline: writes `vision/pending/` (gravity-upright photo crops per element, the
  RoomPlan rectangle drawn on them, a compact JSON and the answer schema), runs `claude -p` with
  Opus, no API key; `--no-llm` skips it.
- The LLM types every element (catalogue type, hinge side, swing, panels, moulding profiles) and
  **may correct dimensions by default**. It answers in pixel corners on the crops, never in
  centimetres; the pipeline back-projects them onto the host wall plane with the photo poses.
  Every corrected value keeps the scan value beside it and carries the flag `llm_corrected`.
- It can declare elements RoomPlan missed (the through-niche) the same way.

### Validation

- Tape measurements (6–8 at the development room: door and window widths and heights, the niche)
  in `measurements.json`; the pipeline writes the list to take, one annotated photo crop per
  measurement, and a report comparing the scan value and the LLM value to the tape.
- Budget: < 30 min of compute per room excluding the LLM pass, < 500 MB per room, about ten photo
  sheets per room for the LLM.
- Traces: `D:\dev\room-model-traces\09-…` onward, one per delivery step: 09 openings and rig
  (no LLM), 10 LLM pass and mouldings plus the tape comparison, 11 projected textures, 12 static
  GLB and cleanup.

### Survey page (2026-11-11, user request)

- The pipeline writes `measurements/index.html`, a local self-contained page (never published):
  a numbered SVG plan, one form row per number (what and where to measure, photo crop with the
  endpoints drawn, current value with source and sigma, priority badge, input in cm), and a
  download of `measurements.json`.
- Rerunning with that file applies the tape values to the model (priority tape > LLM > LiDAR >
  RoomPlan > assumed) and rebuilds the Blender scene; a station moves the less certain wall.

### Photo anchoring of fixture edges (2026-11-11, research spike `~/dev/spikes/photo-anchoring/REPORT.md`) — overrides "CPU only"

- **GPU is part of the pipeline** (user decision): torch and SAM 2.1 are regular dependencies; an
  NVIDIA GPU is expected (developed on an RTX 4090). The "CPU only / no torch" lines above are void.
- New stage `anchor` (after `openings` and `registration`, before `architecture`, output
  `anchors.json`): rectify each host face from the posed photos, segment the element with SAM 2.1
  prompted by the projected model rectangle, snap each mask edge to a gradient peak, triangulate each
  edge's depth behind the face across views, median over views. Floor-based bottoms (doors, floor
  niches, openings) are the LiDAR floor plane by rule. Sigma comes from the depth ambiguity, not only
  the view spread.
- Measurement semantics per kind: door = **leaf** (the moving panel; double leaf = union of both);
  window = sash/leaf (candidates + pick when the mask fails: curtains, frosted glass); niche and
  opening = jamb to jamb, floor (or sill) to head underside.
- The LLM no longer gives coordinates: geometry proposes lettered candidate lines on rectified strips
  and the LLM picks letters (`edge_pick`); typing stays in `vision`.
- Anchored edges carry their depth; survey endpoints and Blender geometry use them at that depth.
- Joinery from the leaf: leaf = anchored or tape value; frame (jamb) visible width = a stated constant
  or the LLM; wall cut = leaf + 2 × jamb (height + 1 jamb); casing = remaining width up to the LiDAR
  outer extent, flagged `to_verify` outside a plausible range.
- Survey page: lines on the anchored edges with what to tape spelled out ("vantail", "deux vantaux
  ensemble", "passage", "niche"), a durable store (linked `measurements.json` via the File System
  Access API with autosave, pre-fill from the existing file, browser copy keyed by capture, import).
