# overrides.json reference

Source: `pipelines/room_model/room_model/overrides.py`. Format `{"<element id>": {field: value}}`. Values are validated against the vision answer schema (`vision_schema.json`) and the Blender catalogue (`blender/arch/catalogue.py`), so a value the LLM could have said is a value the user can say. A rejected field is listed (`Refusées`), never applied; the other fields of the element still apply. Applied elements get the flag `user_override`.

## Fields

| Field | Applies to | Meaning |
|---|---|---|
| `kind` | door, niche, opening | Convert between these three kinds only. Fields of the old kind that the new one lacks are dropped; a niche gets `through: false`, no dividers or shelves, and an assumed recess depth. Processed first. |
| `type` | door, window | Door: `hinged_single`, `hinged_double`, `sliding`, `pocket`, `french_window`; window: `casement_single`, `casement_double`, `tilt_turn`, `sliding`, `fixed` (source: `pipelines/room_model/blender/arch/catalogue.py`; a wrong one is rejected with the list). `null` = untyped. |
| `hinge` | door, window | `left` / `right`, `null`. Read from the side where the leaf is pushed, facing it (for sliding/pocket: the side where the leaf stacks). Omit for double leaves and fixed frames. |
| `swing` | door, window | `in` = opens into the room, `out` = opens toward the corridor / another room / outside; `null`. |
| `panels` | door, window | Integer 0 to 8 (moulded panels). |
| `handle` | door, window | `true` / `false`. |
| `frame` | door, window, niche, opening | Casing: `{"profile": "flat" or "moulded", "width_cm": 0.5 to 30}` or `null` for none. |
| `through` | niche only | `true` = goes all the way through its host; `false` = recess with an assumed depth. |
| `dividers` | niche only | Number of vertical dividers. |
| `shelves_count` | niche only | Number of shelves, evenly spaced. |
| `remove` | door, window, niche, opening, panel_frame, cornice, skirting | `true` deletes the element and parameters nothing else uses. Other kinds (walls, floor, ceilings) cannot be removed. |

Enums come from `pipelines/room_model/room_model/vision_schema.json` and the catalogue; re-read them if the code moved, never invent a value.

## Request to override

| User says | Override (ids are examples; look them up) |
|---|---|
| "door 2 opens toward the corridor" | Work out whether the corridor is outside the room (render, plan): then `{"Door_2": {"swing": "out"}}`, else `"in"`. `hinge` is read from the pushing side, so reversing the swing mirrors it: if the leaf must stay on the same jamb, flip `hinge` too. Rerun and check `renders/open.png` (set `Ouverture` to 1 in Blender, or look at the open render). |
| "the hinges are on the other side" | `{"Door_2": {"hinge": "right"}}` (the other value than the current one) |
| "it is a double door" | `{"Door_2": {"type": "hinged_double"}}` |
| "the niche goes all the way through" | `{"Niche_1": {"through": true}}` |
| "the niche has two shelves" | `{"Niche_1": {"shelves_count": 2}}` |
| "that is a niche, not a door" | `{"Door_3": {"kind": "niche"}}` (then `through`, `shelves_count` as needed) |
| "remove niche 2" / "there is no skirting there" | `{"Niche_2": {"remove": true}}`, `{"Skirting_3": {"remove": true}}` |
| "no casing on that window" | `{"Window_1": {"frame": null}}` |

Wording that is not in the table: map it to a field above or tell the user the pipeline cannot express it yet (the LLM and overrides only choose within the catalogue; a missing fixture is a feature request, filed as an issue with no real-capture values).

## Procedure

1. Read the existing `<out>/overrides.json` (may be absent) and merge.
2. Write the file with the Write/Edit tool, valid JSON.
3. `architecture` then `blender` from `pipelines/room_model/` (add `--overrides <out>/overrides.json` or rely on the file already being in `--out`; the mtime rule reruns the stage).
4. Read `Corrections manuelles` in `report.md`, look at the renders, report what changed.
