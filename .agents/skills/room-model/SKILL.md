---
name: room-model
description: Operate the room-model pipeline (house-capture/5 iPhone export to a measured architectural model, Blender scene and tape-measure survey page) on behalf of the user. Use whenever the user mentions a new room capture or scan zip, running or rerunning room_model, the Blender scene or its renders, the measurement/survey page, tape values (measurements.json), manual corrections (overrides.json: a door that opens the other way, a through niche, removing an element), architecture.json flags, or a pipeline error. Use it even when the user never says "pipeline" or "room_model".
---

# room-model operator

The user drives this pipeline through you and does not type commands. Run them, read the outputs, report in plain French, and ask only when a choice is theirs to make. Code, file names and JSON keys stay English; what you say to the user follows their language.

Sources of truth, in order: the code (`pipelines/room_model/room_model/__main__.py`, `overrides.py`, `tape_merge.py`), `docs/room-model.md`, `docs/specs/room-model-pipeline.md`, `docs/room-model-guide.html` (French operator guide). When this skill and the code disagree, the code wins: say so.

## Rules that cost the most when missed

- **Never publish or commit capture data.** `data/captures-in/` and `data/room-model/` are git-ignored; keep it so. No upload, no artifact, no issue or PR text containing real values (dimensions, room names, photos) from a capture.
- **Code changes are generic for any capture.** Nothing in code, tests or docs may depend on one room: no wall labels, counts, dimensions or room names. Behaviour comes from capture data, the Blender catalogue and the schema; tests use the synthetic fixtures (`pipelines/room_model/tests/synthetic.py`). If a fix only works for the room at hand, it is the wrong fix.
- **Verify, do not infer from the exit code.** A skipped stage exits 0. Read the line `stage X: wrote ...` versus `exists, skipped`, and check the output's mtime.
- Work in `pipelines/room_model/`; use `\rm`/`command cp` style non-interactive commands (shell aliases prompt and hang).

## Environment (check once per session)

```bash
cd pipelines/room_model
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True required for `anchor`
echo "$BLENDER"; command -v claude
```

- venv `pipelines/room_model/.venv` (Python 3.12, several GB, git-ignored). If missing: `uv venv --python 3.12 pipelines/room_model/.venv` then `uv pip install --python pipelines/room_model/.venv/bin/python -r pipelines/room_model/requirements.txt`.
- NVIDIA GPU required (torch, SAM 2.1 in `anchor`; first run downloads the model, needs network once). Without CUDA it still runs, far slower.
- `BLENDER` = Blender 5.1+ executable (from WSL a Windows `.exe` works). Unset: the Steam path is tried. Missing: `all` skips the stage with a message, a lone `blender` fails.
- `claude` CLI on PATH (or `ROOM_MODEL_CLAUDE`) for `vision` and `edge_pick`. Missing or `--no-llm`: those stages are skipped and the architecture is built from RoomPlan types and scan values.

## Workflow 1: new capture

1. Get the zip from the user (ask for its path if not given). Pick a room slug (ASCII, from the user's wording).
2. Unzip safely into a fresh directory: `unzip -l` first and refuse entries starting with `/` or containing `..`; extract with `unzip -n capture.zip -d data/captures-in/<room>`; the target must not already exist with content. If the archive wraps everything in one folder, point `--capture` at the folder that holds `manifest.json`.
3. Run the whole chain (GPU and a few minutes; vision about 2 min, textures tens of seconds, Blender seconds). Prefer background execution and report when done:

```bash
cd pipelines/room_model
.venv/bin/python -m room_model all --capture ../../data/captures-in/<room> --out ../../data/room-model/<room>
```

   Multi-room capture: `--room N`; the room frame off by quarter turns: `--turn N`. `--llm-model` goes to `claude -p`, `--keep-work` keeps intermediates.
4. Check the result (next section) before telling the user it is done.

## Checking a result

Do all of these, then summarise in a few lines: what is solid, what is flagged, what needs the tape.

- **Renders**: look at `renders/top.png`, `persp_1.png`, `persp_2.png`, `open.png` (Read tool). Compare with the photos in the capture when something looks off: missing door, wrong swing, floating moulding.
- **`architecture.json`** (schema `room-model/arch-1`): list elements and dimensions carrying `flags`, above all `to_verify`; also `user_override`. Parameters carry `value`, `sigma`, `source` (`tape` > `llm` > `lidar` > `roomplan` > `assumed`); large sigma or `assumed` marks what to measure. Also read `diagnostics` (`tape`, `overrides`, `vision`, `anchor` entries list what was rejected or not applied).
- **`report.md`** (French): inputs, registration, per-wall table, floor, ceilings, beams (`à vérifier`), columns, unexplained surfaces, orthogonality, `Alertes`; after `architecture` also `Corrections LLM`, `Mesures au mètre`, `Corrections manuelles` (with a `Refusées` list: always relay it).
- **`blender_report.json`**: `dimensions_backend` (MeasureIt or native fallback) and `measureit_error`. `blender.log` ends with the real failure when Blender fails.
- **`textures/textures.json`**: coverage; holes mean the user may need more photos of that surface.
- **Flag meanings**: `docs/room-model.md` (Flags table). Walls: `roomplan_fallback`, `shifted_seed`, `low_support`; stations: `obstacle_or_uneven_wall`.

## Which stage to rerun after what

Order: `registration, shell, model, export, openings, vision, anchor, edge_pick, architecture, textures, blender`. A stage is skipped when its output exists, unless `--force` or an input it merges is newer (`STALE_AFTER` in `__main__.py`):

| Stage | Reruns by itself when newer |
|---|---|
| `anchor` | `openings.json`, `vision/answers.json`, `overrides.json` |
| `edge_pick` | `anchors.json` |
| `architecture` | `anchors.json`, `edge_pick/answers.json`, `vision/answers.json`, `measurements.json`, `overrides.json` |
| `textures` | `architecture.json` |
| `blender` | `architecture.json`, `textures/textures.json` |

Consequences:

- **Tape values or overrides changed**: run `architecture` then `blender` (about 1 min). Add `textures` between them if the geometry moved by several centimetres, so textures follow. Plain `all` also chains them, but `architecture` + `blender` is cheaper to state and to check.
- **Edited file has an old mtime** (copied with preserved date): nothing triggers. `touch` it or use `--force` on `architecture`.
- **Code changed**: never triggers anything. `--force` on the stage you changed; downstream stages with a rule recompute themselves, the others need `--force` too.
- **A failed LLM pass** (`vision/answers.json` holds only `error`) is redone without `--force`; `{"skipped": true}` comes from `--no-llm` and is redone once the flag is dropped.
- After any rerun, re-read the relevant `stage X: wrote` lines.

## The survey (tape measurements)

`measurements/index.html` is written by `architecture`: a local self-contained page, never published, never served from anywhere. Tell the user to open it from disk (Chrome or Edge for autosave), link `measurements.json` in `--out`, tape according to the badge (leaf, both leaves together, passage, niche), and tell you when done. It saves `measurements.json` v2: `{"version": 2, "capture": ..., "values": {"<dimension id>": {"value_m", "date", "note"}}}`; the legacy `{"<id>": metres}` is also accepted. Notes without value are kept and worth reading (a curtain hiding a window, a doubtful measure).

Apply it:

```bash
.venv/bin/python -m room_model architecture --capture <cap> --out <out> --tape <out>/measurements.json
.venv/bin/python -m room_model blender --capture <cap> --out <out>
```

`--tape` (on `architecture` and `all`) copies the file to `<out>/measurements.json`. Effects: an opening dimension replaces the parameter it measures; a ceiling height moves the ceiling; a station moves the wall of its pair with the larger offset sigma (bounded); a leaf larger than its hole is refused. Whatever cannot be applied is listed in `report.md` (`Mesures au mètre`) and `diagnostics.tape`; relay it. Compare scan, LLM and tape: `compare-tape --arch <out>/architecture.json --tape <out>/measurements.json -o tape.md`.

Reconciling human measurements against the model (conflicts, outliers, deciding which source wins) is the job of the `room-reconcile` skill (`.agents/skills/room-reconcile/`): hand over to it once the survey is filled. `architecture` writes `reconcile.json` and `reconcile.md` for it.

## Overrides: translating the user's requests

`overrides.json` (in `--out`, or `--overrides <file>` on `architecture`/`all`): `{"<element id>": {field: value}}`, applied last. Find the element id in `architecture.json` (`elements[*].id`, such as `Door_2`, `Niche_1`) by looking at the plan, renders and the element's host/position; if the user's wording is ambiguous between two elements, show the render and ask. Write the whole file (merge with the existing one, do not drop earlier entries). Field list, checks and a request-to-JSON table: [references/overrides.md](references/overrides.md).

After writing: run `architecture` then `blender`, then read `Corrections manuelles` in `report.md`. A field listed under `Refusées` was not applied (the rest of its element was); fix the value or tell the user why.

## When to ask the user

Ask when the answer is a physical fact or a taste you cannot read from the data: which of two doors they mean, whether a niche really goes through, a measurement that contradicts the scan by more than a few sigma, whether a flagged beam exists. Do not ask what the render, photos or `architecture.json` settle. With three or more open decisions, use the `decision-sheet` skill if it is available (a form to tick) rather than a wall of questions; otherwise a short numbered list, one line each, with your recommendation first. Never put dimensions from a real capture into anything published.

## Troubleshooting

Symptom to remedy table (Blender not found, WSLENV, skipped stage, LLM error, unmeasured window, door wider than hole, textures, CUDA, offline `anchor`, no photos): [references/troubleshooting.md](references/troubleshooting.md). The French original is in `docs/room-model-guide.html`, section 3.

## Other commands (check against `__main__.py` before use)

- `compare --model <model.json> --reference <reference.json> [-o compare.md]`
- `compare-tape --arch <architecture.json> --tape <measurements.json> [-o tape.md]`
- Tests: `cd pipelines/room_model && .venv/bin/python -m pytest -q`
