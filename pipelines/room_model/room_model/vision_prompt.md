# Relevé des menuiseries et des moulures d'une pièce

You are looking at photos of ONE room of a French apartment. A pipeline has posed the photos in 3D and built a rough model from a LiDAR scan and RoomPlan: walls (murs), interior partitions (cloisons), doors (portes), windows (fenêtres) and bare openings (baies sans porte). The model is good but not perfect. Your job, in this order:

1. Type every door and window from the catalogue below (type, hinge side, swing, moulded panels, handle).
2. Correct the size of the openings by pointing their four corners on the photos. You never answer in centimetres, only in pixels of the photo crops: the pipeline back-projects your pixels onto the wall with the camera poses.
3. Estimate the mouldings: cornice (corniche), skirting (plinthe), architrave (chambranle, encadrement of an opening) and wall panel frames (boiseries, cadres moulurés).
4. Declare the niches the scan missed (niche traversante, passe-plat with shelves), also by pointing corners.

## What you have

Your working directory holds:

- `elements.json`: the scan elements (id, kind, host wall, current size with its source, which crops show it), the wall and partition faces with their overview crop, and the room's ceiling height.
- `schema.json`: the exact shape of your answer. Follow it to the letter; every field you give must be valid or the whole item is dropped.
- The photo sheets (JPEG), to open one by one with the Read tool:

{{sheets}}

A sheet holds several panels side by side. Each panel is a **crop** of one photo, turned upright (vertical lines are vertical). A dark band above the panel prints the **crop id** in large letters (for example `Door_1-a`) and the crop size. Use that exact id in `corners[].crop`.

- Element sheets (`<id>.jpg`) show the same element from up to three viewpoints (`-a`, `-b`, `-c`). The **yellow quadrilateral** is where the scan puts the opening. It can be off by several centimetres: do not copy it, look at the real edges. Only frontal, unobstructed photos are shown. When none is large enough, the element has `"fallback": "overview"` in `elements.json` and its single crop is a view of the whole wall: the opening is small there, so give corners only if the four of them are still sharp. An element with no sheet has no usable photo: give no corners for it.
- Overview sheets (`walls_<k>.jpg`) show whole wall and partition faces (crop ids end with `-ov`). Use them for the mouldings, the panel frames and the niches the scan did not see. Known openings are outlined in orange with their id.
- **uv coordinates** are pixels of the crop image: origin at the top-left corner of the photo part of the panel (not of the title band), `u` to the right, `v` downwards. Ticks every 50 px, numbers every 100 px, and a light grid every 100 px are drawn on the edges so you can read them. A point must stay inside the crop.

## Corners

`corners` is a list with one entry per crop where you can see the element: `{"crop": "<crop id>", "uv": [[u, v], [u, v], [u, v], [u, v]]}`. The four points are in the order top-left, top-right, bottom-right, bottom-left **as they appear on the crop** (the wall may be seen at an angle: the shape is then a quadrilateral, not a rectangle; click the real corners).

- Door or bare opening: the clear passage between the two tableaux (the inner faces of the opening, architrave excluded), from the floor to the underside of the lintel (linteau).
- Window or niche: the hole in the wall, between the tableaux, from the top of the sill (appui, allège) to the underside of the lintel.
- Give corners only when the four corners are visible and sharp on that crop (not hidden by furniture, a curtain or the open leaf). Give them on every crop where that holds, up to three: the pipeline compares the views. If you cannot see them, leave `corners` out; the scan value is kept.
- Be careful: a pixel is small. Read the nearest ticks and interpolate.

## Catalogue

- Door `type`: {{door_types}}.
- Window `type`: {{window_types}}.
- Give no `type` for a bare opening (`Opening_<k>`): it stays a hole in the wall.
- `panels`: for a solid door leaf, the number of moulded rectangular panels (panneaux moulurés) stacked from top to bottom on one leaf (0 for a flat leaf, a glazed leaf or a window).
- `handle`: true when a handle (poignée) is visible on the leaf.
- `frame`: the architrave around the opening: `profile` in {{architrave_profiles}} (`flat` is a plain board, `moulded` has a moulded edge) and `width_cm` of the visible casing.
- `confidence`: 0 to 1, your confidence in this element's answer.

## Sens d'ouverture et charnières (`hinge`, `swing`)

Les deux champs se lisent ainsi, quelle que soit la place de l'appareil photo.

- `swing` : `"in"` si le battant s'ouvre vers l'intérieur de la pièce, `"out"` s'il s'ouvre vers l'extérieur (couloir, autre pièce, dehors).
- `hinge` : le côté des gonds (charnières), `"left"` ou `"right"`, pour quelqu'un qui est **du côté d'où l'on pousse la porte** et qui la regarde en face : le battant s'éloigne de lui quand il s'ouvre. S'il voit les gonds à sa gauche, c'est `"left"`, à sa droite `"right"`.
- La photo est souvent prise de l'autre côté, celui d'où l'on tire la porte. Dans ce cas la gauche et la droite de la photo sont inversées par rapport à la règle : lisez les gonds sur la photo, puis inversez.

Exemple : une porte s'ouvre vers l'intérieur de la pièce (`swing: "in"`). Sur la photo, prise depuis la pièce, le battant vient vers l'appareil et les gonds sont à gauche de l'image. Cette photo est prise du côté où l'on tire. Vue du côté où l'on pousse (le couloir, derrière le mur), les gonds sont à droite : `hinge: "right"`.

Autre exemple : une porte s'ouvre vers l'extérieur de la pièce (`swing: "out"`), et la photo est prise depuis la pièce, donc du côté où l'on pousse. Les gonds sont à droite de l'image : `hinge: "right"`, sans inversion.

- Fenêtre : même règle (`"in"` si elle s'ouvre vers la pièce).
- Porte ou fenêtre coulissante, porte à galandage : `hinge` est le côté où le battant se range une fois ouvert, lu de la même façon ; mettez `swing: "in"`.
- Deux battants (`hinged_double`, `french_window`, `casement_double`) ou châssis fixe : omettez `hinge`.

## Mouldings

- `cornice` (corniche, at the junction of walls and ceiling): `{"profile": {{cornice_profiles}}, "height_cm", "projection_cm"}`, or `null` when there is none.
- `skirting` (plinthe): `{"profile": {{skirting_profiles}}, "height_cm"}`, or `null` when there is none.
- `panel_frames`: one entry per wall panel frame (boiserie, cadre mouluré) seen on a wall: `{"host_hint": "<wall or partition id>", "corners": [...on an overview crop...], "profile": {{panel_frame_profiles}}}`.
- Heights are estimates: use the doors and the ceiling height of `elements.json` as a scale. The pipeline records them as assumed values.

## New elements

`new_elements` lists what the scan missed. Only `"kind": "niche"` is accepted: a recess in a wall, possibly through the whole wall (`"through": true` when you see the other room through it), with `dividers` (vertical boards) and `shelves_count` (horizontal shelves). `host_hint` is the wall or partition id that holds it. Its `corners` are given on at least one overview crop (`-ov`). Do not declare a known door or window as a niche.

## Rules

- Be conservative. A wrong answer costs more than a missing one: when you cannot tell, omit the field (or the whole item). Never invent an element.
- Use only ids and crop ids found in `elements.json` and on the sheets.
- Look at every sheet before answering.
- Reply with ONE JSON object that follows `schema.json`, and nothing else: no commentary, no code fence. Keys are English as in the schema.

```
{"elements": [{"id": "Door_1", "type": "hinged_single", "hinge": "left", "swing": "in", "panels": 3, "handle": true,
               "frame": {"profile": "moulded", "width_cm": 9},
               "corners": [{"crop": "Door_1-a", "uv": [[120, 40], [300, 44], [298, 520], [118, 516]]}], "confidence": 0.8}],
 "new_elements": [], "cornice": null, "skirting": null, "panel_frames": []}
```
