"""Merge the vision answers into the architecture: corrected openings, declared niches, mouldings and typing.

Corner pixels of a crop become metres by `crops.json` (crop pixel -> upright pixel -> sensor pixel) and a ray cast on
the face plane of the host. Nothing here raises on a bad answer: what cannot be applied is listed with its reason.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import replace
from functools import cached_property

import numpy as np

from room_model.architecture import _face_point, _value
from room_model.capture import colmajor
from room_model.lidar import SAMPLE_DENSITY, load_samples
from room_model.photo_geom import backproject, load_posed_photos, to_sensor

AGREEMENT = 0.03  # m; crops whose rectangles differ by less than this on every parameter agree
MIN_SIGMA = 0.01  # m
MAX_DEVIATION = 0.30  # m between the scan and the LLM rectangle, beyond which the answer is not believed
HOST_RANGE = 0.30  # m; a declared element belongs to the nearest host plane within this distance
FLOOR_REACH = 0.02  # m; an opening whose sill is below this reaches the floor, so its sill is not measured
BOUNDS_MARGIN = 0.50  # m a back-projected corner may overhang its host before it counts as a mistake
FACE_B_BAND = 0.04  # m around the far face of a partition that still counts as its surface
THROUGH_SHARE = 0.10  # share of the expected LiDAR samples behind a niche below which the niche goes through
OVERLAP = 0.5  # share of a declared niche covered by an existing opening above which the niche is refused
DEFAULT_RECESS = 0.15  # m; depth assumed for a niche that does not go through, at most half the host thickness
ASSUMED_SIGMA = 0.02  # m
RECT_KINDS = ("door", "window", "opening", "niche")
HOST_KINDS = ("wall", "partition")
TYPING = ("type", "hinge", "swing", "panels", "handle")
NAMES = {"width": "largeur", "height": "hauteur", "sill": "allège", "center_s": "position"}


class Unusable(Exception):
    """An answer item that cannot be applied; the message is the reason shown in the report."""


def usable(answers: object) -> bool:
    """A vision document holding answers: not absent, not the `--no-llm` placeholder, not the record of a failed call."""
    return isinstance(answers, dict) and not answers.get("skipped") and "error" not in answers


class Evidence:
    """Photos and LiDAR samples of the capture, loaded when first needed; empty without a stage context."""

    def __init__(self, arch: dict, ctx):
        self.ctx, self.w2r = ctx, colmajor(arch["frame"]["world_to_room"], 4)

    @cached_property
    def photos(self) -> dict:
        return {} if self.ctx is None else {p.name: p for p in load_posed_photos(self.ctx, self.w2r)}

    @cached_property
    def samples(self):
        return None if self.ctx is None else load_samples(self.ctx, self.w2r)


class Report:
    def __init__(self):
        self.doc = {"corrections": [], "typing": [], "corrected": [], "single_view": [], "inconsistent": [], "declared": [], "not_applied": []}

    def refuse(self, ident: str, reason: str) -> None:
        self.doc["not_applied"].append({"id": ident, "reason": reason})


# --- geometry ---------------------------------------------------------------------------------------------------

def _elements(arch: dict) -> dict:
    return {e["id"]: e for e in arch["elements"]}


def _plane(arch: dict, host: dict, face: str | None) -> tuple[np.ndarray, float]:
    az = _value(arch, host["params"]["azimuth"])
    shift = _value(arch, host["params"]["thickness"]) if face == "B" else 0.0
    return np.array([np.cos(az), 0.0, np.sin(az)]), _value(arch, host["params"]["offset"]) - shift


def _length(host: dict) -> float:
    return float(np.linalg.norm(np.subtract(host["end"], host["start"])))


def _to_face(arch: dict, host: dict, pts: np.ndarray) -> np.ndarray:
    """(n, 2) of (s, y): metres along the host from its start, and above the floor."""
    start, end = np.array(host["start"], float), np.array(host["end"], float)
    s = (pts[:, [0, 2]] - start) @ ((end - start) / np.linalg.norm(end - start))
    return np.column_stack([s, pts[:, 1] - _value(arch, "floor.level")])


def _crop_points(crop_id: str, uv, crops: dict, evidence: Evidence, plane) -> np.ndarray:
    """Room-frame points (4, 3) of four crop pixels cast on `plane`."""
    info = crops.get(crop_id)
    if info is None:
        raise Unusable(f"vue {crop_id} inconnue")
    photo = evidence.photos.get(info["photo"])
    if photo is None:
        raise Unusable(f"photo {info['photo']} indisponible")
    uv = np.asarray(uv, float)
    if uv.shape != (4, 2) or not np.isfinite(uv).all():
        raise Unusable(f"{crop_id} : moins de 4 coins valides")
    box, size = np.array(info["box"], float), np.array(info["size"], float)
    upright = box[:2] + uv * (box[2:] - box[:2]) / size
    sensor = to_sensor(replace(photo, up_turns=int(info.get("up_turns", photo.up_turns))), upright)
    points = backproject(photo, sensor, plane)
    if points is None:
        raise Unusable(f"{crop_id} : rétroprojection impossible sur le plan de la face")
    return points


def _rects(arch: dict, host: dict, face: str | None, corners: list, crops: dict, evidence: Evidence):
    """([(crop id, rectangle (4, 2) of (s, y), points (4, 3))], reasons) for the crops that give four usable corners."""
    plane, length = _plane(arch, host, face), _length(host)
    top = _value(arch, host["params"]["top"]) - _value(arch, "floor.level")
    found, reasons = [], []
    for corner in corners:
        crop_id = corner.get("crop") if isinstance(corner, dict) else None
        try:
            points = _crop_points(crop_id, corner["uv"], crops, evidence, plane)
            rect = _to_face(arch, host, points)
            low, high = rect.min(0), rect.max(0)
            if low[0] < -BOUNDS_MARGIN or high[0] > length + BOUNDS_MARGIN or low[1] < -BOUNDS_MARGIN or high[1] > top + BOUNDS_MARGIN:
                raise Unusable(f"{crop_id} : coins hors de l'élément hôte")
            found.append((crop_id, rect, points))
        except Unusable as exc:
            reasons.append(str(exc))
        except (KeyError, TypeError, ValueError, IndexError):
            reasons.append(f"{crop_id} : coins mal formés")
    return found, reasons


def _rect_values(rect: np.ndarray) -> dict:
    """Corners are top-left, top-right, bottom-right, bottom-left as seen on the crop."""
    tl, tr, br, bl = rect
    return {"width": (abs(tr[0] - tl[0]) + abs(br[0] - bl[0])) / 2, "center_s": float(rect[:, 0].mean()),
            "bottom": float((bl[1] + br[1]) / 2), "top": float((tl[1] + tr[1]) / 2)}


def _consensus(rects: list[np.ndarray], floor_reach: bool, sill: float) -> tuple[dict, dict, list[int], bool]:
    """(values, sigmas, kept crops, single view) of rectangles; Unusable when no two crops agree within AGREEMENT."""
    per_crop = [_rect_values(r) for r in rects]
    for v in per_crop:
        v["height"] = v["top"] - (sill if floor_reach else v["bottom"])
    names = ["width", "height", "center_s"] + ([] if floor_reach else ["bottom"])
    table = np.array([[v[n] for n in names] for v in per_crop])
    kept = [0] if len(rects) == 1 else None
    for size in range(len(rects), 1, -1):
        fits = [(np.ptp(table[list(c)], axis=0).max(), c) for c in itertools.combinations(range(len(rects)), size)]
        fits = [f for f in fits if f[0] <= AGREEMENT]
        if fits:
            kept = list(min(fits)[1])
            break
    if kept is None:
        raise Unusable("inconsistent")
    final = _rect_values(np.median(np.stack([rects[i] for i in kept]), axis=0))
    final["height"] = final["top"] - (sill if floor_reach else final["bottom"])
    std = {n: max(float(np.std(table[kept, i])), MIN_SIGMA) for i, n in enumerate(names)}
    values = {"width": final["width"], "height": final["height"], "center_s": final["center_s"], "sill": final["bottom"]}
    sigmas = {"width": std["width"], "height": std["height"], "center_s": std["center_s"], "sill": std.get("bottom", MIN_SIGMA)}
    return values, sigmas, kept, len(rects) == 1


# --- openings ---------------------------------------------------------------------------------------------------

def _flag(el: dict, flag: str) -> None:
    flags = el.setdefault("flags", [])
    if flag not in flags:
        flags.append(flag)


def _copy_typing(el: dict, item: dict, rep: Report) -> None:
    for field in TYPING:
        if field in item:
            el[field] = item[field]
            rep.doc["typing"].append({"element": el["id"], "field": field, "value": item[field]})
    if "frame" in item:
        spec = item["frame"]
        el["frame"] = None if spec is None else {"profile": spec["profile"], "width": float(spec["width_cm"]) / 100}
        rep.doc["typing"].append({"element": el["id"], "field": "frame", "value": None if spec is None else {"profile": spec["profile"], "width_cm": spec["width_cm"]}})


def _set_llm(arch: dict, el: dict, name: str, value: float, sigma: float, rep: Report) -> None:
    p = arch["parameters"][el["params"][name]]
    rep.doc["corrections"].append({"element": el["id"], "parameter": name, "scan": p["value"], "llm": float(value)})
    p.update(scan_value=p["value"], scan_sigma=p["sigma"], llm_value=float(value), llm_sigma=float(sigma), value=float(value), sigma=float(sigma), source="llm")


def _correct_opening(arch: dict, el: dict, item: dict, crops: dict, evidence: Evidence, rep: Report) -> None:
    host = _elements(arch).get(el.get("host"))
    if host is None or host["kind"] not in HOST_KINDS:
        raise Unusable("aucun plan hôte pour projeter les coins (élément orphelin)")
    found, reasons = _rects(arch, host, el.get("face"), item["corners"], crops, evidence)
    if not found:
        raise Unusable("aucune vue exploitable : " + " ; ".join(reasons))
    scan = {n: arch["parameters"][pid]["value"] for n, pid in el["params"].items()}
    try:
        values, sigmas, kept, single = _consensus([r for _, r, _ in found], scan["sill"] <= FLOOR_REACH, scan["sill"])
    except Unusable:
        _flag(el, "llm_inconsistent")
        rep.doc["inconsistent"].append(el["id"])
        return
    names = [n for n in ("width", "height", "center_s", "sill") if not (n == "sill" and scan["sill"] <= FLOOR_REACH)]
    worst = max(names, key=lambda n: abs(values[n] - scan[n]))
    if abs(values[worst] - scan[worst]) > MAX_DEVIATION:
        raise Unusable(f"écart au scan de {abs(values[worst] - scan[worst]) * 100:.0f} cm sur la {NAMES[worst]} (au-delà de 30 cm), réponse écartée".replace(".", ","))
    for n in names:
        _set_llm(arch, el, n, values[n], sigmas[n], rep)
    _flag(el, "llm_single_view" if single else "llm_corrected")
    rep.doc["single_view" if single else "corrected"].append(el["id"])


def _apply_element(arch: dict, item: dict, crops: dict, evidence: Evidence, rep: Report) -> None:
    el = _elements(arch).get(item["id"])
    if el is None or el["kind"] not in RECT_KINDS:
        raise Unusable("élément inconnu de l'architecture")
    _copy_typing(el, item, rep)
    if item.get("corners"):
        _correct_opening(arch, el, item, crops, evidence, rep)


# --- declared elements -------------------------------------------------------------------------------------------

def _face_of(crop_id: str) -> str | None:
    found = re.search(r"-([AB])-ov$", str(crop_id))
    return found.group(1) if found else None


def _nearest(arch: dict, centre: np.ndarray, prefer: tuple) -> tuple[dict, str | None]:
    """The (host, face) whose plane is nearest the point within HOST_RANGE, and along the host; `prefer` wins ties."""
    best = None
    for host in arch["elements"]:
        if host["kind"] not in HOST_KINDS:
            continue
        s = _to_face(arch, host, centre[None, :])[0, 0]
        if not -HOST_RANGE <= s <= _length(host) + HOST_RANGE:
            continue
        for face in (("A", "B") if host["kind"] == "partition" else (None,)):
            normal, offset = _plane(arch, host, face)
            key = (round(abs(float(normal @ centre) - offset), 3), (host["id"], face) != prefer)
            if key[0] <= HOST_RANGE and (best is None or key < best[0]):
                best = (key, host, face)
    if best is None:
        raise Unusable("aucun mur ni cloison à moins de 30 cm du centre projeté")
    return best[1], best[2]


def _locate(arch: dict, hint: str, corners: list, crops: dict, evidence: Evidence):
    """(host, face, [(crop id, rectangle, points)]): the corners cast on the hinted host, then on the nearest host plane."""
    host = _elements(arch).get(hint)
    if host is None or host["kind"] not in HOST_KINDS:
        raise Unusable(f"hôte {hint!r} inconnu")
    faces = Counter(_face_of(c.get("crop")) for c in corners if isinstance(c, dict))
    if not faces:
        raise Unusable("aucun coin fourni")
    face = max(faces, key=lambda f: (faces[f], f is None))
    usable_corners = [c for c in corners if isinstance(c, dict) and _face_of(c.get("crop")) == face]
    for _ in range(2):
        face = None if host["kind"] == "wall" else face or "A"
        found, reasons = _rects(arch, host, face, usable_corners, crops, evidence)
        if not found:
            raise Unusable("aucune vue exploitable : " + " ; ".join(reasons))
        near, near_face = _nearest(arch, np.concatenate([p for _, _, p in found]).mean(0), (host["id"], face))
        if (near["id"], near_face) == (host["id"], face):
            return host, face, found
        host, face = near, near_face
    raise Unusable("hôte instable")


def _lidar_through(arch: dict, host: dict, face: str, s: tuple, y: tuple, evidence: Evidence) -> bool | None:
    """True when no LiDAR sample lies on the far face of a partition behind the rectangle; None when it cannot be told."""
    if host["kind"] != "partition" or "one_face_seen" in host.get("flags", []):
        return None
    try:
        samples = evidence.samples
    except (OSError, ValueError, KeyError):
        return None
    if samples is None:
        return None
    normal, near = _plane(arch, host, face)
    far = near + (-1 if face == "A" else 1) * _value(arch, host["params"]["thickness"])
    n2 = normal[[0, 2]]
    q = samples.P[:, [0, 2]] @ n2 - far
    along = _to_face(arch, host, samples.P)
    hit = (np.abs(q) < FACE_B_BAND) & (np.abs(samples.N[:, [0, 2]] @ n2) > 0.75) & (along[:, 0] >= s[0]) & (along[:, 0] <= s[1]) \
        & (along[:, 1] >= y[0]) & (along[:, 1] <= y[1])
    return bool(hit.sum() < THROUGH_SHARE * (s[1] - s[0]) * (y[1] - y[0]) * SAMPLE_DENSITY)


def _overlapped_by(arch: dict, host: dict, s: tuple, y: tuple) -> str | None:
    area = (s[1] - s[0]) * (y[1] - y[0])
    for other in arch["elements"]:
        if other.get("host") != host["id"] or other["kind"] not in RECT_KINDS or "center_s" not in other["params"]:
            continue
        p = {n: _value(arch, pid) for n, pid in other["params"].items()}
        s_overlap = min(s[1], p["center_s"] + p["width"] / 2) - max(s[0], p["center_s"] - p["width"] / 2)
        y_overlap = min(y[1], p["sill"] + p["height"]) - max(y[0], p["sill"])
        if s_overlap > 0 and y_overlap > 0 and s_overlap * y_overlap > OVERLAP * area:
            return other["id"]
    return None


def _new_param(arch: dict, pid: str, value: float, sigma: float, source: str) -> None:
    arch["parameters"][pid] = {"value": float(value), "sigma": float(sigma), "unit": "m", "source": source}


def _declare_niche(arch: dict, item: dict, crops: dict, evidence: Evidence, rep: Report) -> None:
    host, face, found = _locate(arch, item["host_hint"], item["corners"], crops, evidence)
    try:
        values, sigmas, _, single = _consensus([r for _, r, _ in found], False, 0.0)
    except Unusable:
        raise Unusable("vues en désaccord de plus de 3 cm, niche écartée") from None
    s, y = (values["center_s"] - values["width"] / 2, values["center_s"] + values["width"] / 2), (values["sill"], values["sill"] + values["height"])
    if values["width"] <= 0.02 or values["height"] <= 0.02:
        raise Unusable("rectangle dégénéré")
    if (other := _overlapped_by(arch, host, s, y)):
        raise Unusable(f"recouvre l'ouverture {other}")
    k = 1 + max([int(e["id"].split("_")[1]) for e in arch["elements"] if e["kind"] == "niche"], default=0)
    ident, thickness = f"Niche_{k}", _value(arch, host["params"]["thickness"])
    through = bool(item.get("through", False))
    flags = ["llm_single_view"] if single else []
    if (lidar := _lidar_through(arch, host, face, s, y, evidence)) is not None and lidar != through:
        flags.append("through_disputed")
    params = {n: f"niche.{k}.{n}" for n in ("center_s", "sill", "width", "height", "depth")}
    for n in ("center_s", "sill", "width", "height"):
        _new_param(arch, params[n], values[n], sigmas[n], "llm")
    if through:
        arch["parameters"][params["depth"]] = dict(arch["parameters"][host["params"]["thickness"]])
    else:
        _new_param(arch, params["depth"], min(DEFAULT_RECESS, thickness / 2), ASSUMED_SIGMA * 2.5, "assumed")
    count = int(item.get("shelves_count") or 0)
    spec = item.get("frame")
    x, z = _face_point(arch, host, face, values["center_s"])
    arch["elements"].append({
        "id": ident, "kind": "niche", "host": host["id"], "face": face, "params": params, "through": through,
        "dividers": int(item.get("dividers") or 0), "shelves": [values["height"] * (i + 1) / (count + 1) for i in range(count)],
        "frame": None if not spec else {"profile": spec["profile"], "width": float(spec["width_cm"]) / 100},
        "center": [float(x), float(z)], "flags": flags})
    for n in ("center_s", "sill", "width", "height"):
        rep.doc["corrections"].append({"element": ident, "parameter": n, "scan": None, "llm": float(values[n])})
    rep.doc["declared"].append(ident)


def _declare_mouldings(arch: dict, answers: dict, rep: Report) -> None:
    specs = (("Cornice", "cornice", answers.get("cornice"), ("height", "projection")), ("Skirting", "skirting", answers.get("skirting"), ("height",)))
    for ident, kind, spec, names in specs:
        if not spec:
            continue
        params = {}
        for n in names:
            params[n] = f"{kind}.{n}"
            _new_param(arch, params[n], float(spec[f"{n}_cm"]) / 100, ASSUMED_SIGMA, "assumed")
            rep.doc["corrections"].append({"element": ident, "parameter": n, "scan": None, "llm": float(spec[f"{n}_cm"]) / 100})
        arch["elements"].append({"id": ident, "kind": kind, "profile": spec["profile"], "params": params})
        rep.doc["declared"].append(ident)


def _declare_panel_frame(arch: dict, item: dict, crops: dict, evidence: Evidence, rep: Report) -> None:
    host, face, found = _locate(arch, item["host_hint"], item["corners"], crops, evidence)
    try:
        values, _, kept, single = _consensus([r for _, r, _ in found], False, 0.0)
    except Unusable:
        raise Unusable("vues en désaccord de plus de 3 cm, cadre écarté") from None
    rects = np.stack([found[i][1] for i in kept])
    low, high = np.median(rects, axis=0).min(0), np.median(rects, axis=0).max(0)
    k = 1 + sum(e["kind"] == "panel_frame" for e in arch["elements"])
    arch["elements"].append({"id": f"PanelFrame_{k}", "kind": "panel_frame", "host": host["id"], "face": face,
                             "rect": [float(low[0]), float(low[1]), float(high[0]), float(high[1])], "profile": item["profile"],
                             "flags": ["to_verify", *(["llm_single_view"] if single else [])]})
    rep.doc["declared"].append(f"PanelFrame_{k}")


def apply(arch: dict, answers: dict, ctx=None) -> dict:
    """Merges the answers into `arch` in place; returns the report stored under diagnostics.vision."""
    rep, evidence, crops = Report(), Evidence(arch, ctx), answers.get("crops") or {}
    groups = (("elements", lambda i, it: it.get("id", f"elements[{i}]"), _apply_element),
              ("new_elements", lambda i, it: f"new_elements[{i}] ({it.get('kind', 'niche')}, {it.get('host_hint')})", _declare_niche),
              ("panel_frames", lambda i, it: f"panel_frames[{i}] ({it.get('host_hint')})", _declare_panel_frame))
    for key, label, apply_item in groups:
        for i, item in enumerate(answers.get(key) or []):
            ident = label(i, item) if isinstance(item, dict) else f"{key}[{i}]"
            try:
                apply_item(arch, item, crops, evidence, rep)
            except Unusable as exc:
                rep.refuse(ident, str(exc))
            except (KeyError, TypeError, ValueError, IndexError, AttributeError, np.linalg.LinAlgError) as exc:
                rep.refuse(ident, f"réponse mal formée ({type(exc).__name__}: {exc})")
    try:
        _declare_mouldings(arch, answers, rep)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        rep.refuse("cornice/skirting", f"réponse mal formée ({type(exc).__name__}: {exc})")
    return rep.doc
