"""User overrides (overrides.json, {"<element id>": {field: value}}), applied after the LLM answers and the tape.

Values are checked with the schema of the vision answers and the Blender catalogue, so what the model could say
is what the user can say. A rejected field is listed, never applied; the rest of the element is still applied.
"""
from __future__ import annotations

import json

from room_model.llm_merge import RECT_KINDS, _elements, _flag, recess_depth, shelf_heights
from room_model.vision import PACKAGE, catalogue, check

RETYPE = ("door", "niche", "opening")
REMOVABLE = (*RECT_KINDS, "panel_frame", "cornice", "skirting")
TYPING = ("type", "hinge", "swing", "panels", "handle")
NICHE_FIELDS = ("through", "dividers", "shelves_count")
FIELDS = ("kind", *TYPING, "frame", *NICHE_FIELDS, "remove", "prefer")
# A parameter can be sent back to its scan value (the model and openings stages), named `scan` or by its own source.
SCAN_SOURCES = ("scan", "lidar", "roomplan", "assumed")
NICHE_ONLY = ("through", "dividers", "shelves")
SCHEMA_OF = {**{f: "elements" for f in (*TYPING, "frame")}, **{f: "new_elements" for f in NICHE_FIELDS}}


def _schema() -> dict:
    return json.loads((PACKAGE / "vision_schema.json").read_text(encoding="utf-8"))


def _prefer_problem(el: dict, value, scan: dict) -> str | None:
    if not isinstance(value, dict) or not value:
        return 'un objet {"<paramètre>": "scan"} est attendu'
    for name, source in value.items():
        if name not in el.get("params", {}):
            return f"{name!r} n'est pas un paramètre de cet élément ({', '.join(el.get('params', {}))})"
        kept = scan.get(el["params"][name])
        if kept is None:
            return f"{name} n'a pas de valeur du scan (élément déclaré par le LLM)"
        if source not in ("scan", kept["source"]):
            return f"aucune valeur {source!r} conservée pour {name} (possibles : scan, {kept['source']})"
    return None


def _problem(el: dict, field: str, value, scan: dict | None = None) -> str | None:
    """Why `value` cannot be given to `field` of `el` (None when it can)."""
    if field == "prefer":
        return _prefer_problem(el, value, scan or {})
    cat = catalogue()
    kind = el["kind"]
    if field == "kind":
        return None if value in RETYPE and kind in RETYPE else f"seuls door, niche et opening se convertissent entre eux ({kind} -> {value!r})"
    if field == "remove":
        if not isinstance(value, bool):
            return "true ou false attendu"
        return None if kind in REMOVABLE else f"un élément {kind} ne se supprime pas"
    if field in NICHE_FIELDS and kind != "niche":
        return f"{field} ne concerne que les niches (cet élément est un {kind})"
    if field in TYPING and kind not in cat.CATALOGUE:
        return f"un élément {kind} n'a pas de {field}"
    if field == "frame" and kind not in RECT_KINDS:
        return f"un élément {kind} n'a pas de cadre"
    schema = _schema()
    node = schema["properties"][SCHEMA_OF[field]]["items"]["properties"][field]
    errors = check(value, node, schema, field)
    if errors:
        return "; ".join(errors)
    if field == "type" and value is not None and value not in cat.CATALOGUE[kind]:
        return f"{value!r} n'existe pas pour un {kind} ({', '.join(cat.CATALOGUE[kind])})"
    return None


def _make_recess(arch: dict, el: dict) -> None:
    """A niche that does not go through needs a depth inside its host: one that spans the host is replaced by the assumed recess."""
    host = _elements(arch).get(el.get("host"))
    if host is not None:
        thickness = arch["parameters"][host["params"]["thickness"]]["value"]
        depth = arch["parameters"][el["params"]["depth"]]
        if depth["value"] >= thickness - 1e-3:
            depth.update(value=recess_depth(thickness), source="assumed")


def _retype(arch: dict, el: dict, kind: str) -> None:
    """Changes the kind; fields of the old kind that the new one does not use are dropped."""
    for key in (*TYPING, *NICHE_ONLY):
        el.pop(key, None)
    el["kind"] = kind
    if kind == "niche":
        el.update(through=False, dividers=0, shelves=[])
        _make_recess(arch, el)
    else:
        el["type"] = None


def _set(arch: dict, el: dict, field: str, value, scan: dict | None = None) -> object:
    if field == "prefer":  # back to the scan value: the source, value and sigma the scan gave
        for name in value:
            arch["parameters"][el["params"][name]] = dict(scan[el["params"][name]])
        return value
    if field == "kind":
        _retype(arch, el, value)
    elif field == "frame":
        el["frame"] = None if value is None else {"profile": value["profile"], "width": float(value["width_cm"]) / 100}
    elif field == "shelves_count":
        el["shelves"] = shelf_heights(arch["parameters"][el["params"]["height"]]["value"], value)
    else:
        el[field] = value
    if field == "through" and not value:
        _make_recess(arch, el)
    return value


def apply(arch: dict, overrides: dict, scan: dict | None = None) -> dict:
    """Applies the overrides to `arch` in place; {"applied": [{element, field, value}], "rejected": [{id, field, reason}], "removed": [ids]}.

    `scan` holds the parameters as the model and openings stages wrote them, for `prefer: {"<param>": "scan"}` (or
    the scan's own source name: `lidar`, `roomplan`).
    """
    rep = {"applied": [], "rejected": [], "removed": []}
    for ident, fields in overrides.items():
        el = _elements(arch).get(ident)
        if el is None or not isinstance(fields, dict):
            rep["rejected"].append({"id": ident, "field": None, "reason": "élément inconnu" if el is None else "un objet {champ: valeur} est attendu"})
            continue
        if fields.get("remove") is True:
            if (why := _problem(el, "remove", True)):
                rep["rejected"].append({"id": ident, "field": "remove", "reason": why})
                continue
            arch["elements"].remove(el)
            for pid in el.get("params", {}).values():
                if not any(pid in other.get("params", {}).values() for other in arch["elements"]):
                    arch["parameters"].pop(pid, None)
            rep["removed"].append(ident)
            continue
        done = False
        for field in sorted(fields, key=lambda f: f != "kind"):  # the kind first: the other fields are checked against it
            value = fields[field]
            if field == "remove":
                continue
            why = f"champ inconnu (attendus : {', '.join(FIELDS)})" if field not in FIELDS else _problem(el, field, value, scan)
            if why:
                rep["rejected"].append({"id": ident, "field": field, "reason": why})
                continue
            rep["applied"].append({"element": ident, "field": field, "value": _set(arch, el, field, value, scan)})
            done = True
        if done:
            _flag(el, "user_override")
    return rep
