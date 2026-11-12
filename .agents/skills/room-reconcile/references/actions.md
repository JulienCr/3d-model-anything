# Action ids → file edits

`reconcile.json` lists action ids in `actions`, and per check or parameter. `reconcile.md` gives a French reading of each. Every id below has an executable form; if you meet an id that is not listed here, read `reconcile.py` and `overrides.py` before acting.

## `measure:<dimension id>`

Ask the user to measure that dimension with the tape. The id is a row of the survey page (`measurements/index.html`), so tell them the row number or label from the page, not the raw id.

When they answer, write the value into `measurements.json` v2 and keep their note:

```json
{"version": 2, "values": {"Door_3-h": {"value_m": 2.103, "date": "2026-11-12", "note": "remesuré"}}}
```

- **Re-measure after a `probable_tape_error`:** the pipeline already keeps the value out of the model (`tape_set_aside`), so leave the entry and overwrite `value_m` when the new reading arrives. `{"value_m": null, "note": "..."}` is the form for a value the user withdraws: a note without a value is kept and never applied.
- **Stations:** `S..` rows are the distance between two parallel walls at a marked spot. Remind the user which one a row is before they measure again.
- **`measure:L-<label>`** is a wall corner to corner, read along the wall above the skirtings (they shorten a reading at the floor). Applied, it moves the end wall with the larger sigma; a value that would move it too far is refused and listed.

## `prefer:<parameter id>:<source>`

The retained value of a parameter is not the one to trust: an LLM click, or a tape value the loops reject. Send it back to its scan value with an override on the element that owns the parameter:

```json
{"Door_3": {"prefer": {"height": "lidar"}}}
```

- **Element:** the one whose `params` maps a name to that parameter id (`opening.Door_3.height` → element `Door_3`, name `height`). Look it up in `architecture.json`; do not parse the id.
- **Source:** `scan` or the scan's own source name (`lidar`, `roomplan`). Anything else is rejected and listed in `report.md` under `Corrections manuelles / Refusées`.
- **Precedence:** overrides apply last, after the tape. A `prefer` therefore also beats a tape value; use it on a tape value only after the user agreed the tape was wrong.
- **Undoing it:** remove the entry from `overrides.json`.

## `override:<element id>:<field>`

Write that field in `overrides.json`: `{"<element id>": {"<field>": <value>}}`. Fields and allowed values are in `.agents/skills/room-model/references/overrides.md`. The reconciliation proposes `override:<id>:remove` for two openings that overlap on one face: the smaller is often a duplicate (a niche declared over a door). Confirm with the render or the user before removing anything: removal is the one edit a user notices most.

## After any edit

Run `architecture`, then `blender`, from `pipelines/room_model/`. Add `textures` between them when walls moved by more than about 3 cm. Then re-read `reconcile.json`.
