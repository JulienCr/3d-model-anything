"""Reconciliation: every source of each parameter, global consistency checks and the actions they suggest.

Written by the architecture stage next to architecture.json, so that it never lags behind it: reconcile.json for an
agent (action ids `measure:<dimension id>`, `prefer:<parameter id>:<source>`, `override:<element id>:<field>`) and
reconcile.md, the same in French for a person. Nothing here changes the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from room_model.fmt import fr

SCHEMA = "room-model/reconcile-1"
Z = 3.0  # combined sigmas beyond which two values disagree: consistent measures do so 0.3 % of the time
SAME_POINT = 1e-6  # m; two wall ends this close are one corner (the model writes them identical)
SQUARE_RANGE = np.radians(30)  # a corner further than this from 90 degrees is a deliberate angle, not a squaring question
OPENING_PARAMS = {"opening_width": "width", "opening_height": "height", "opening_sill": "sill", "leaf_width": "leaf_width",
                  "leaf_height": "leaf_height"}
SOURCE_WORDS = {"roomplan": "RoomPlan", "lidar": "LiDAR", "assumed": "supposé", "llm": "LLM", "photo": "photos",
                "tape": "mètre", "override": "correction"}
VERDICT_WORDS = {"ok": "cohérent", "conflict": "incohérent", "probable_tape_error": "erreur de mètre probable",
                 "out_of_square": "hors d'équerre", "square_within_noise": "d'équerre au bruit près"}


# --- sources ---------------------------------------------------------------------------------------------------------

def _dims_of_params(arch: dict) -> dict:
    """{parameter id: [dimension ids]} for the dimensions whose value is one parameter."""
    elements = {e["id"]: e for e in arch["elements"]}
    out = {}
    for dim in arch["dimensions"]:
        name = OPENING_PARAMS.get(dim["kind"])
        el = elements.get(dim["elements"][0]) if dim["elements"] else None
        if name and el is not None and name in el["params"]:
            out.setdefault(el["params"][name], []).append(dim["id"])
    return out


def parameters(layers: list[tuple[str, dict]]) -> dict:
    """Per parameter, the value each layer set (scan first, labelled by its own source), the retained one, residuals
    against it in metres and combined sigmas, and `conflict` when a measured source disagrees beyond Z sigmas.

    `layers`: [(label, architecture)] in merge order (scan, llm, photo, tape, override); the last one is retained.
    """
    final = layers[-1][1]["parameters"]
    out = {}
    for pid, kept in final.items():
        seen, previous = [], None
        for label, arch in layers:
            p = arch["parameters"].get(pid)
            if p is None or (previous is not None and p["value"] == previous["value"] and p["sigma"] == previous["sigma"]):
                previous = p if p is not None else previous
                continue
            seen.append({"source": p["source"] if label == "scan" else label, "value": p["value"], "sigma": p["sigma"]})
            previous = p
        if not seen:
            continue
        residuals = {s["source"]: {"delta": s["value"] - kept["value"], "z": _z(s["value"] - kept["value"], s["sigma"], kept["sigma"])}
                     for s in seen if s["source"] != kept["source"]}
        flags = ["conflict"] if any(abs(r["z"]) > Z and src not in ("llm", "assumed") for src, r in residuals.items()) else []
        out[pid] = {"sources": seen, "retained": {k: kept[k] for k in ("value", "sigma", "source")}, "residuals": residuals,
                    "flags": flags}
    return out


def _z(delta: float, *sigmas: float) -> float:
    sigma = float(np.hypot(*sigmas)) if len(sigmas) > 1 else float(sigmas[0])
    return float(delta / sigma) if sigma > 0 else (0.0 if delta == 0 else float(np.sign(delta)) * np.inf)


# --- geometry helpers ------------------------------------------------------------------------------------------------

def _value(arch: dict, pid: str) -> float:
    return arch["parameters"][pid]["value"]


def _sigma(arch: dict, pid: str) -> float:
    return arch["parameters"][pid]["sigma"]


def _outline(arch: dict) -> list[dict] | None:
    """The walls in outline order (each one's end is the next one's start); None when they do not close."""
    walls = [e for e in arch["elements"] if e["kind"] == "wall"]
    if not walls:
        return None
    order, current = [walls[0]], walls[0]
    while True:
        nxt = next((w for w in walls if np.linalg.norm(np.subtract(w["start"], current["end"])) < SAME_POINT), None)
        if nxt is None:
            return None
        if nxt is walls[0]:
            return order if len(order) == len(walls) else None
        order.append(nxt)
        current = nxt


def _direction(wall: dict) -> np.ndarray:
    d = np.subtract(wall["end"], wall["start"])
    return d / np.linalg.norm(d)


def _length(el: dict) -> float:
    return float(np.linalg.norm(np.subtract(el["end"], el["start"])))


# --- tape loops ------------------------------------------------------------------------------------------------------

def _loops(arch: dict) -> list[dict]:
    """Linear relations between dimensions the geometry imposes: {name, terms [(dimension id, coefficient)]}.

    A station between two walls equals the walls between them projected on its direction, both ways round the outline;
    the outline closes along two axes; two stations between the same walls differ by what the model says.
    """
    outline = _outline(arch)
    dims = {d["id"]: d for d in arch["dimensions"]}
    length_of = {d["elements"][0]: d["id"] for d in arch["dimensions"] if d["kind"] == "wall_length"}
    loops = []
    if outline and all(w["id"] in length_of for w in outline):
        axis = _direction(outline[0])
        for name, e in (("x", axis), ("z", np.array([-axis[1], axis[0]]))):
            loops.append({"name": f"closure-{name}", "terms": [(length_of[w["id"]], float(_direction(w) @ e)) for w in outline]})
        index = {w["id"]: i for i, w in enumerate(outline)}
        for dim in (d for d in arch["dimensions"] if d["kind"] in ("width", "length")):
            a, b = dim["elements"][:2]
            if a not in index or b not in index:
                continue
            wa, wb = outline[index[a]], outline[index[b]]
            n = np.subtract(np.add(wb["start"], wb["end"]), np.add(wa["start"], wa["end"]))
            n = n / np.linalg.norm(n)
            for start, stop, sign in ((index[a], index[b], -1.0), (index[b], index[a], 1.0)):
                chain, i = [], (start + 1) % len(outline)
                while i != stop:
                    chain.append(outline[i])
                    i = (i + 1) % len(outline)
                terms = [(length_of[w["id"]], float(_direction(w) @ n)) for w in chain if abs(_direction(w) @ n) > 1e-9]
                loops.append({"name": f"{dim['id']}-{'ab' if sign < 0 else 'ba'}", "terms": terms + [(dim["id"], sign)]})
        stations = [d for d in arch["dimensions"] if d["kind"] in ("width", "length")]
        for i, s1 in enumerate(stations):
            for s2 in stations[i + 1:]:
                if set(s1["elements"]) == set(s2["elements"]):
                    loops.append({"name": f"{s1['id']}-{s2['id']}", "terms": [(s1["id"], 1.0), (s2["id"], -1.0)]})
    return [lp for lp in loops if all(t[0] in dims for t in lp["terms"])]


def _loop_residual(loop: dict, values: dict, model: dict) -> tuple[float, float]:
    """(residual, sigma): the relation evaluated on `values` minus the same on the model's own values (the model is
    consistent by construction, so only measured values can break it). values/model: {id: (value, sigma)}."""
    r = sum(c * (values[i][0] - model[i][0]) for i, c in loop["terms"])
    return float(r), float(np.sqrt(sum((c * values[i][1]) ** 2 for i, c in loop["terms"])))


def tape_checks(base: dict, tape: dict | None) -> list[dict]:
    """Closed relations involving at least one tape value, each with its verdict; when they fail, the tape entry whose
    removal (back to the model value) clears every failure is reported once as a probable tape error."""
    from room_model.tape_merge import TAPE_SIGMA, normalize

    raw = {k: v for k, v in normalize(tape or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    model = {d["id"]: (d["value"], d["sigma"]) for d in base["dimensions"]}
    loops = [lp for lp in _loops(base) if any(i in raw for i, _ in lp["terms"])]
    if not loops:
        return []

    def evaluate(keys) -> list[tuple[dict, float, float]]:
        values = {**model, **{k: (float(raw[k]), TAPE_SIGMA) for k in keys if k in model}}
        return [(lp, *_loop_residual(lp, values, model)) for lp in loops]

    results = evaluate(raw)
    failing = [(lp, r, s) for lp, r, s in results if abs(r) > Z * s]
    out = [{"id": f"loop:{lp['name']}", "kind": "tape_loop", "verdict": "conflict" if abs(r) > Z * s else "ok", "residual": r,
            "sigma": s, "dims": [i for i, _ in lp["terms"] if i in raw], "actions": []} for lp, r, s in results]
    if failing:
        suspects = sorted({i for lp, _, _ in failing for i, _ in lp["terms"] if i in raw})
        cleared = [k for k in suspects if not any(abs(r) > Z * s for _, r, s in evaluate([x for x in raw if x != k]))]
        if len(cleared) == 1:
            k = cleared[0]
            lp, r, _ = next(f for f in failing if k in dict(f[0]["terms"]))
            implied = float(raw[k]) - r / dict(lp["terms"])[k]
            out.append({"id": f"tape:{k}", "kind": "tape_consistency", "verdict": "probable_tape_error", "dims": [k],
                        "residual": float(raw[k]) - implied, "sigma": TAPE_SIGMA, "implied": implied,
                        "actions": [f"measure:{k}"]})
        else:
            out.append({"id": "tape:loops", "kind": "tape_consistency", "verdict": "conflict", "dims": suspects,
                        "residual": max(abs(r) for _, r, _ in failing), "sigma": min(s for _, _, s in failing),
                        "actions": [f"measure:{k}" for k in suspects]})
    return out


# --- element checks --------------------------------------------------------------------------------------------------

def _rect(arch: dict, el: dict) -> tuple[float, float, float, float, float, float]:
    """(s0, s1, y0, y1, sigma along, sigma up) of an opening on its host."""
    p = {k: _value(arch, v) for k, v in el["params"].items()}
    sig = {k: _sigma(arch, v) for k, v in el["params"].items()}
    return (p["center_s"] - p["width"] / 2, p["center_s"] + p["width"] / 2, p["sill"], p["sill"] + p["height"],
            float(np.hypot(sig["center_s"], sig["width"] / 2)), float(np.hypot(sig["sill"], sig["height"])))


def _verdict(residual: float, sigma: float) -> str:
    return "conflict" if residual > Z * sigma else "ok"


def element_checks(arch: dict) -> list[dict]:
    """Openings inside their host and not overlapping, leaves within their holes, corners square or not, wall tops
    against the ceilings."""
    from room_model.anchor import leaf_excess

    elements = {e["id"]: e for e in arch["elements"]}
    openings = [e for e in arch["elements"] if e["kind"] in ("door", "window", "opening", "niche") and e.get("host") in elements
                and {"center_s", "width", "sill", "height"} <= set(e["params"])]
    out = []
    for el in openings:
        host = elements[el["host"]]
        s0, s1, y0, y1, ss, sy = _rect(arch, el)
        top = _value(arch, host["params"]["top"]) - _value(arch, "floor.level") if "top" in host["params"] else np.inf
        over = max(0.0, -s0, s1 - _length(host), y1 - top)
        out.append({"id": f"inside:{el['id']}", "kind": "opening_inside_host", "verdict": _verdict(over, max(ss, sy)),
                    "residual": over, "sigma": max(ss, sy), "elements": [el["id"], host["id"]],
                    "actions": [f"measure:{el['id']}-w"] if over > Z * max(ss, sy) else []})
    for i, a in enumerate(openings):
        for b in openings[i + 1:]:
            if a["host"] != b["host"] or a.get("face") != b.get("face"):
                continue
            ra, rb = _rect(arch, a), _rect(arch, b)
            overlap = min(min(ra[1], rb[1]) - max(ra[0], rb[0]), min(ra[3], rb[3]) - max(ra[2], rb[2]))
            if overlap <= 0:
                continue
            sigma = float(np.hypot(max(ra[4], ra[5]), max(rb[4], rb[5])))
            smaller = min((a, b), key=lambda e: (_rect(arch, e)[1] - _rect(arch, e)[0]) * (_rect(arch, e)[3] - _rect(arch, e)[2]))
            out.append({"id": f"overlap:{a['id']}:{b['id']}", "kind": "opening_overlap", "verdict": _verdict(overlap, sigma),
                        "residual": overlap, "sigma": sigma, "elements": [a["id"], b["id"]],
                        "actions": [f"override:{smaller['id']}:remove", f"measure:{a['id']}-w", f"measure:{b['id']}-w"]})
    for el in openings:
        for name, suffix in (("width", "leaf-w"), ("height", "leaf-h")):
            pid = el["params"].get(f"leaf_{name}")
            if pid is None:
                continue
            excess, tol = leaf_excess(arch, el, name, arch["parameters"][pid], arch["parameters"][el["params"][name]])
            out.append({"id": f"leaf:{el['id']}:{name}", "kind": "leaf_fits", "verdict": "conflict" if excess > tol else "ok",
                        "residual": excess, "sigma": tol, "elements": [el["id"]],
                        "actions": [f"measure:{el['id']}-{suffix}", f"measure:{el['id']}-{suffix[-1]}"] if excess > tol else []})
    out += _corners(arch) + _wall_tops(arch)
    return out


def _corners(arch: dict) -> list[dict]:
    """Each outline corner's departure from 90 degrees as the offset it makes at the end of the shorter wall, against
    that offset's sigma (the two walls' azimuth sigmas over the same length)."""
    outline = _outline(arch) or []
    out = []
    for a, b in zip(outline, outline[1:] + outline[:1]):
        (ax, az), (bx, bz) = _direction(a), _direction(b)
        cross = ax * bz - az * bx
        angle = float(np.arcsin(np.clip(abs(cross), -1.0, 1.0)))
        if abs(angle - np.pi / 2) > SQUARE_RANGE:
            continue
        lever = min(_length(a), _length(b))
        offset = (np.pi / 2 - angle) * lever
        sigma = float(np.hypot(_sigma(arch, a["params"]["azimuth"]), _sigma(arch, b["params"]["azimuth"]))) * lever
        flags = sorted({f for w in (a, b) for f in w.get("flags", []) if "squar" in f or "parallel" in f})
        out.append({"id": f"corner:{a['id']}:{b['id']}", "kind": "corner_square",
                    "verdict": "out_of_square" if offset > Z * sigma else "square_within_noise", "residual": float(offset),
                    "sigma": sigma, "elements": [a["id"], b["id"]], "flags": flags,
                    "actions": [f"measure:L-{a['label']}", f"measure:L-{b['label']}"] if offset > Z * sigma else []})
    return out


def _wall_tops(arch: dict) -> list[dict]:
    """Each wall top against the nearest ceiling level (a wall meets one of the ceilings)."""
    levels = [(e["id"], _value(arch, e["params"]["level"]), _sigma(arch, e["params"]["level"]))
              for e in arch["elements"] if e["kind"] == "ceiling" and "level" in e["params"]]
    out = []
    for wall in (e for e in arch["elements"] if e["kind"] == "wall" and "top" in e["params"]):
        top, sigma_top = _value(arch, wall["params"]["top"]), _sigma(arch, wall["params"]["top"])
        if not levels:
            break
        ident, level, sigma_level = min(levels, key=lambda c: abs(c[1] - top))
        sigma = float(np.hypot(sigma_top, sigma_level))
        out.append({"id": f"top:{wall['id']}", "kind": "wall_top_vs_ceiling", "verdict": _verdict(abs(top - level), sigma),
                    "residual": float(top - level), "sigma": sigma, "elements": [wall["id"], ident], "actions": []})
    return out


# --- document --------------------------------------------------------------------------------------------------------

def build(layers: list[tuple[str, dict]], tape: dict | None) -> dict:
    """The reconcile document. `layers` as in `parameters`; the pre-tape layer gives the model values of the loops."""
    final = layers[-1][1]
    base = next((arch for label, arch in reversed(layers) if label not in ("tape", "override")), final)
    params = parameters(layers)
    dims_of = _dims_of_params(final)
    checks = element_checks(final) + tape_checks(base, tape)
    suspicious = {d for c in checks if c["verdict"] == "probable_tape_error" for d in c["dims"]}
    for pid, entry in params.items():
        actions = []
        if "conflict" in entry["flags"]:
            doubtful = entry["retained"]["source"] == "llm" or (  # clicks, or a tape value the loops reject
                entry["retained"]["source"] == "tape" and any(d in suspicious for d in dims_of.get(pid, [])))
            if doubtful:
                measured = [s for s in entry["sources"] if s["source"] not in ("tape", "llm", "assumed")]
                best = min(measured, key=lambda s: s["sigma"], default=None)
                actions += [f"prefer:{pid}:{best['source']}"] if best else []
            actions += [f"measure:{d}" for d in dims_of.get(pid, [])]
        entry["actions"] = list(dict.fromkeys(actions))
    actions = list(dict.fromkeys(a for c in checks for a in c["actions"]))
    actions += [a for e in params.values() for a in e["actions"] if a not in actions]
    summary = {v: sum(c["verdict"] == v for c in checks) for v in sorted({c["verdict"] for c in checks})}
    return {"schema": SCHEMA, "summary": summary, "checks": checks, "parameters": params, "actions": actions}


def markdown(doc: dict) -> str:
    """reconcile.md: the checks that fail, the parameters in conflict and the actions, in French."""
    cm = lambda v: fr(v * 100)  # noqa: E731
    lines = ["# Réconciliation du relevé", "", "Bilan des contrôles : " + ", ".join(
        f"{VERDICT_WORDS.get(v, v)} : {n}" for v, n in doc["summary"].items()) + ".", ""]
    bad = [c for c in doc["checks"] if c["verdict"] not in ("ok", "square_within_noise")]
    lines += ["## Contrôles à revoir", ""] if bad else ["Aucun contrôle en défaut.", ""]
    if bad:
        lines += ["| Contrôle | Verdict | Écart (cm) | Tolérance σ (cm) | Concerne |", "|---|---|---|---|---|"]
        lines += [f"| {c['id']} | {VERDICT_WORDS.get(c['verdict'], c['verdict'])} | {cm(c['residual'])} | {cm(c['sigma'])} | "
                  f"{', '.join(c.get('dims') or c.get('elements') or [])} |" for c in bad]
        lines.append("")
    tape = [c for c in bad if c["verdict"] == "probable_tape_error"]
    for c in tape:
        lines.append(f"- La mesure **{c['dims'][0]}** ne s'accorde pas avec les autres : le reste du relevé implique "
                     f"{cm(c['implied'])} cm, écart {cm(c['residual'])} cm. À remesurer avant de l'appliquer.")
    conflicts = {pid: e for pid, e in doc["parameters"].items() if "conflict" in e["flags"]}
    if conflicts:
        lines += ["", "## Paramètres en désaccord", "", "| Paramètre | Retenu | Autres sources |", "|---|---|---|"]
        for pid, e in conflicts.items():
            others = "; ".join(f"{SOURCE_WORDS.get(s['source'], s['source'])} {cm(s['value'])} ± {cm(s['sigma'])}"
                               for s in e["sources"] if s["source"] != e["retained"]["source"])
            lines.append(f"| {pid} | {cm(e['retained']['value'])} ± {cm(e['retained']['sigma'])} cm "
                         f"({SOURCE_WORDS.get(e['retained']['source'], e['retained']['source'])}) | {others} |")
    if doc["actions"]:
        lines += ["", "## Actions proposées", ""] + [f"- `{a}`" for a in doc["actions"]]
    return "\n".join(lines) + "\n"


def write(ctx, final: dict, inputs: dict) -> Path:
    """reconcile.json and reconcile.md in the output folder; `inputs` holds model_doc, openings_doc, vision, anchors,
    picks, tape and overrides as the architecture stage read them, to rebuild each merge layer."""
    from room_model.architecture import build as build_arch
    from room_model.edge_pick import usable as picks_usable
    from room_model.llm_merge import usable as vision_usable

    m, o = inputs["model_doc"], inputs["openings_doc"]
    vision = inputs["vision"] if vision_usable(inputs["vision"]) else None
    anchors, picks = inputs["anchors"], inputs["picks"] if picks_usable(inputs["picks"]) else None
    tape, overrides = inputs["tape"], inputs["overrides"]
    steps = [("scan", (None, None, None, None)), ("llm", (vision, None, None, None)), ("photo", (vision, None, anchors, picks)),
             ("tape", (vision, tape, anchors, picks))]
    present = {"llm": vision is not None, "photo": anchors is not None, "tape": bool(tape)}
    layers = [(label, build_arch(m, o, v, t, ctx, None, a, p)) for label, (v, t, a, p) in steps if label == "scan" or present[label]]
    layers.append(("override" if overrides else layers[-1][0], final))
    doc = build(layers, tape)
    (ctx.out / "reconcile.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (ctx.out / "reconcile.md").write_text(markdown(doc), encoding="utf-8")
    return ctx.out / "reconcile.json"
