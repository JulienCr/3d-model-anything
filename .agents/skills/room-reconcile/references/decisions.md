# Sorting findings, and the log

Sigmas are in `reconcile.json` (`sigma` on checks, `sigma` on every source). Tape sigma is 5 mm (`tape_merge.TAPE_SIGMA`). "Beyond 3σ" means |residual| > 3 × sigma, the same threshold the checks use (`reconcile.Z`). The other thresholds below (2 cm, 3 cm) are agent heuristics, not pipeline rules: 2 cm is where the survey page already marks a scan value as « à reprendre », 3 cm is about what a careful tape reading can still be off by (hook play, a skirting in the way). Quote them as heuristics in the log.

## Applied as is (log one line per value)

A tape value whose dimension has no failing check and no `conflict` flag: the pipeline already applied it, nothing to do. Also a tape value that disagrees with an LLM or `assumed` source only: those were never measurements.

## Arbitrated by you (log the numbers)

Decide only when the numbers settle it, and write them down.

| Finding | Decision | Reason to log |
|---|---|---|
| `tape_vs_scan` (one tape value against the scan) where the scan source is `roomplan`, or σ_scan ≥ 2 cm, or the element carries `to_verify`, or a note mentions an occluder | keep the tape | "tape X cm vs scan Y ± σ cm (source); scan unreliable here because ..." |
| a parameter retained from `llm` with a `conflict` against `lidar` or `roomplan`, and no tape value | `prefer` the scan source (`prefer:` action) | "LLM X cm vs LiDAR Y ± σ cm, Δ = n σ; clicks are not measurements" |
| `corner_square` `square_within_noise` | nothing; the corner is square within its noise | "offset X cm < 3σ (Y cm)" |
| a legacy orphan: an id saved in measurements.json but absent from `architecture.json` `dimensions`, with exactly one current dimension on the same element and quantity | move the value to the current id, keep the note | "old id → new id, same element and quantity" |

## Asked to the user

- **`probable_tape_error`:** "La cote <row> (X cm) ne s'accorde pas avec les autres ; le reste du relevé implique Y cm. Pouvez-vous la remesurer ?" The pipeline has already set it aside (`tape_set_aside`); leave the file as it is and write the new value over it when they answer.
- **`tape_consistency` `conflict` with several suspects:** ask for the first two in the ranked list, not all of them.
- **`tape_vs_scan` with a reliable scan:** LiDAR, σ < 2 cm, no flag, |Δ| > 3σ and > 3 cm. Ask which is right, with both values and your recommendation. Then edit `measurements.json`:
  - **scan chosen:** `{"value_m": null, "note": "scan retenu : <their words>"}` for that id. The tape value leaves the model, and the note keeps the trace.
  - **tape chosen:** keep the value and add `"note": "mètre confirmé : <their words>"`.
- **`corner_square` `out_of_square`:** ask whether the wall really runs off square (old buildings often do), or whether the user can tape the two walls the action names.
- **`leaf_fits`, `opening_overlap`, `opening_inside_host` in conflict:** show the render and ask; propose the action from `reconcile.json`.
- **`wall_top_vs_ceiling` in conflict:** a ceiling height to measure (`measure:H..`), or a beam or soffit the scan missed.

Group questions; use `decision-sheet` when there are three or more.

## reconcile-log.md

Append, never rewrite, in `<out>/reconcile-log.md`:

```markdown
## 2026-11-12 14:05 — reconcile pass 2
Inputs: measurements.json (mtime 14:01), reconcile.json before: 3 failing checks.

| Id | Finding | Category | Decision | Reason (numbers) | File |
|---|---|---|---|---|---|
| L-M3 | probable_tape_error | asked | value set aside, re-measure asked | tape 312.0 cm, survey implies 302.1 cm, Δ 9.9 cm > 3σ (1.1 cm) | measurements.json |
| opening.Door_3.height | conflict llm vs lidar | arbitrated | prefer lidar | LLM 202.1 vs LiDAR 210.5 ± 1.6 cm, Δ 5.3 σ | overrides.json |

After rerun: 1 failing check (was 3); sum |residual|/σ on touched items 9.1 → 0.4.
```

The log is in `<out>`, never committed and never published.
