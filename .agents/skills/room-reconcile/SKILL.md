---
name: room-reconcile
description: Reconcile a tape-measure survey (measurements.json) with the room-model pipeline's model, using reconcile.json, and settle each conflict by data-file edits, a logged reason or a question to the user. Use it whenever the user says they measured, finished the survey or sent tape values, or asks why a measure was refused, which value to trust or whether the model now matches the room, even without saying "reconcile".
---

# room-reconcile

The user measured their room with a tape following the survey page. Your job is to turn their values into a model they can trust. Every human value either gets confirmed by the model, gets a reasoned decision from you, or goes back to them as a short question. Speak French to the user; files, ids and JSON stay English.

Read `.agents/skills/room-model/SKILL.md` first if it is not loaded: it has the environment checks, the CLI, the rerun rules and the privacy rules. The code wins over this skill when they disagree (`pipelines/room_model/room_model/reconcile.py`, `tape_merge.py`, `overrides.py`). If you find a mismatch, say so.

## Ground rules

- **Never force a value.** The pipeline already keeps a `probable_tape_error` out of the model: it rebuilds without it, flags the dimension `tape_set_aside` and lists it as « écartée, à remesurer ». Do not edit it away or "correct" it; ask the user to re-measure. A tape value is a person's observation; the model is an estimate. When they disagree and the numbers cannot settle it, the person decides.
- **Act only through data files**: `<out>/measurements.json` (values and notes) and `<out>/overrides.json`. Never edit code, architecture.json or reconcile.json; they are outputs.
- **Keep everything local.** Values, notes and logs stay in `<out>`, which is git-ignored. Do not paste real dimensions into commits, issues, PRs or anything published.
- **Log every decision** in `<out>/reconcile-log.md` with its numeric reason, so the user, or the next agent, can see why a value won. Format: [references/decisions.md](references/decisions.md).

## Workflow

1. **Apply what the user measured.** The survey page saves `measurements.json` v2: `{"version": 2, "values": {"<dim id>": {"value_m", "date", "note"}}}`; the legacy flat `{"<id>": metres}` is accepted too. If the user gives you the file, run `architecture --tape <file>`; if it is already in `<out>`, run `architecture` (it reruns when the file is newer; `--force` otherwise). The stage writes `reconcile.json` and `reconcile.md` with the model.
2. **Check the reconciliation ran**: if `diagnostics.reconcile_error` is set in `architecture.json`, or `reconcile.json` holds only an `error`, there are no checks to rely on. Do not arbitrate and do not compare residuals: report the error and fall back to the room-model skill's tape application.
3. **Read**, in this order:
   - `diagnostics.tape` in `architecture.json`: entries `not_applied` (unknown id, refused leaf, a wall that would move too far, a value set aside) and `set_aside`. An unknown id is a legacy orphan: an id the survey saved that is absent from `architecture.json` `dimensions` (the row was renamed or removed since). Map it when exactly one current dimension measures the same element and quantity; otherwise ask.
   - every note in `measurements.json`, with or without a value: a curtain, a skirting in the way, "not sure". Notes change how much a value is worth.
   - `reconcile.json`: `checks` (verdict, residual, sigma), `parameters[*]` (sources, retained, residuals, flags), `elements` (failing checks per element) and `actions`.
4. **Sort each finding** with the rules in [references/decisions.md](references/decisions.md): applied as is / arbitrated by you / asked to the user. A threshold you apply must appear in the log with the number that crossed it.
5. **Act** by translating action ids into file edits: [references/actions.md](references/actions.md) (`measure:`, `prefer:`, `override:`). Merge with the existing files; never drop entries you were not asked to touch.
6. **Ask** what is left in one go. With three or more questions, use the `decision-sheet` skill if available (a form to tick); otherwise a short numbered list, one line each, your recommendation first, the numbers in centimetres.
7. **Rerun and verify**: `architecture`, then `textures` if any wall moved by more than about 3 cm (compare wall `offset` parameters before and after), then `blender`. Read the `stage X: wrote` lines. Unless `reconcile_error` is set, compare the new `reconcile.json` with the previous one: fewer failing checks and smaller |residual|/sigma on what you touched. If something got worse, revert your edit, log it and ask.
8. **Report** in a few French lines: what was confirmed, what you decided and why, what you are asking, what remains to measure. Point to `reconcile.md` and `reconcile-log.md`.

## Dry runs and tests

To rehearse on synthetic data, copy a synthetic out dir to a scratch folder and add fake tape values with one deliberate typo. The typo must come back as `probable_tape_error` naming its id, and `architecture.json` must not carry it (`tape_set_aside`). Never use a real capture for a demo you show anyone else.
