---
name: room-reconcile
description: Reconcile the user's tape-measure survey with the room-model pipeline, after they filled the survey page (measurements.json). Reads measurements.json and reconcile.json, checks every human value against the model's consistency checks, sorts conflicts into applied / arbitrated by you with a numeric reason / asked to the user, acts only by editing measurements.json or overrides.json, reruns architecture and blender, verifies the residuals went down and logs every decision in reconcile-log.md. Use it whenever the user says they measured, finished the survey, sent tape values, or asks why a measure was refused, whether the model now matches the room, or which value to trust, even if they never say "reconcile". It complements the room-model operator skill and does not replace it.
---

# room-reconcile

The user measured their room with a tape following the survey page. Your job is to turn their values into a model they can trust. Every human value either gets confirmed by the model, gets a reasoned decision from you, or goes back to them as a short question. Speak French to the user; files, ids and JSON stay English.

Read `.agents/skills/room-model/SKILL.md` first if it is not loaded: it has the environment checks, the CLI, the rerun rules and the privacy rules. The code wins over this skill when they disagree (`pipelines/room_model/room_model/reconcile.py`, `tape_merge.py`, `overrides.py`). If you find a mismatch, say so.

## Ground rules

- **Never force a value.** A `probable_tape_error` is not applied and not "corrected" by you. Ask the user to re-measure. A tape value is a person's observation; the model is an estimate. When they disagree and the numbers cannot settle it, the person decides.
- **Act only through data files**: `<out>/measurements.json` (values and notes) and `<out>/overrides.json`. Never edit code, architecture.json or reconcile.json; they are outputs.
- **Keep everything local.** Values, notes and logs stay in `<out>`, which is git-ignored. Do not paste real dimensions into commits, issues, PRs or anything published.
- **Log every decision** in `<out>/reconcile-log.md` with its numeric reason, so the user, or the next agent, can see why a value won. Format: [references/decisions.md](references/decisions.md).

## Workflow

1. **Apply what the user measured.** The survey page saves `measurements.json` v2: `{"version": 2, "values": {"<dim id>": {"value_m", "date", "note"}}}`; the legacy flat `{"<id>": metres}` is accepted too. If the user gives you the file, run `architecture --tape <file>`; if it is already in `<out>`, run `architecture` (it reruns when the file is newer; `--force` otherwise). The stage writes `reconcile.json` and `reconcile.md` with the model.
2. **Read**, in this order:
   - `diagnostics.tape` in `architecture.json`: entries `not_applied` (unknown id, refused leaf, a wall that would move too far). An unknown id is often a legacy orphan, a survey row whose dimension was renamed (`Door_1-w` before the leaf rows existed). Map it when one current id obviously measures the same thing; otherwise ask.
   - every note in `measurements.json`, with or without a value: a curtain, a skirting in the way, "not sure". Notes change how much a value is worth.
   - `reconcile.json`: `checks` (verdict, residual, sigma), `parameters[*]` (sources, retained, residuals, flags), `elements` (failing checks per element) and `actions`.
3. **Sort each finding** with the rules in [references/decisions.md](references/decisions.md): applied as is / arbitrated by you / asked to the user. A threshold you apply must appear in the log with the number that crossed it.
4. **Act** by translating action ids into file edits: [references/actions.md](references/actions.md) (`measure:`, `prefer:`, `override:`). Merge with the existing files; never drop entries you were not asked to touch.
5. **Ask** what is left in one go. With three or more questions, use the `decision-sheet` skill if available (a form to tick); otherwise a short numbered list, one line each, your recommendation first, the numbers in centimetres.
6. **Rerun and verify**: `architecture`, then `textures` if any wall moved by more than about 3 cm (compare wall `offset` parameters before and after), then `blender`. Read the `stage X: wrote` lines. Compare the new `reconcile.json` with the previous one: fewer failing checks and smaller |residual|/sigma on what you touched. If something got worse, revert your edit, log it and ask.
7. **Report** in a few French lines: what was confirmed, what you decided and why, what you are asking, what remains to measure. Point to `reconcile.md` and `reconcile-log.md`.

## Dry runs and tests

To rehearse on synthetic data, copy a synthetic out dir to a scratch folder and add fake tape values with one deliberate typo. The typo must come back as `probable_tape_error` naming its id. Never use a real capture for a demo you show anyone else.
