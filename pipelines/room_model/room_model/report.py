"""French Markdown report of a model document."""
from __future__ import annotations

import numpy as np

from room_model import ceilings, shell
from room_model.model import SOURCE_RANK
from room_model.fmt import fr
from room_model.plan_svg import STATION_KINDS

WALL_FLAGS = {
    "roomplan_fallback": "mur non vu par le LiDAR, position RoomPlan conservée",
    "shifted_seed": "plan RoomPlan à plus de 8 cm de la surface, ajusté sur la surface trouvée dans ±30 cm",
    "low_support": "appui LiDAR faible, ajustement peu fiable",
    "seeded_from_outline": "mur absent de la liste RoomPlan, initialisé depuis le contour du sol",
    "near_parallel_corner": "angle quasi plat avec un mur voisin, coin obtenu par projection",
    "out_of_square": "hors équerre : écart supérieur à l'incertitude de mesure, angle mesuré conservé",
}
PARTITION_FLAGS = {
    "one_face_seen": "une seule face vue par le LiDAR, épaisseur supposée",
    "roomplan_fallback": "cloison non vue par le LiDAR, position RoomPlan conservée",
    "out_of_square": WALL_FLAGS["out_of_square"],
}
CEILING_FLAGS = {"assumed": "niveau supposé : aucun point LiDAR dans la zone, repris du plus grand voisin"}
SQUARING_STATE = {"squared": "équerré", "out_of_square": "hors équerre"}
GENERAL_FLAGS = {
    "floor:roomplan_fallback": "sol non vu par le LiDAR, niveau RoomPlan conservé",
    "columns:no_ceiling": "aucun plafond détecté, poteaux non recherchés",
}
UNEXPLAINED = {
    "not_full_height": "n'atteint pas le plafond (meuble ou cloison basse)",
    "open_sides": "faces non fermées, pas un poteau",
    "size_out_of_range": "dimensions hors de la plage d'un poteau",
}
HORIZONTAL = {
    "too_narrow": "trop étroit pour une poutre",
    "too_shallow": f"descend de moins de {fr(ceilings.MIN_BEAM_DROP * 100, 0)} cm sous le plafond",
    "suspended": "suspendu sous un plafond visible (luminaire, gaine, chemin de câbles)",
}
TO_VERIFY = "poutre probable, à confirmer (photos ou relevé)"
SOURCES = {"lidar": "LiDAR", "roomplan": "RoomPlan", "assumed": "supposé", "llm": "LLM", "photo": "photos"}
OBSTRUCTED = "obstacle_or_uneven_wall"
UNOBSERVED_ALERT = 0.05  # share of the ceiling the LiDAR never saw above which an alert is raised
STATION_FLAG = "section de mur encombrée ou irrégulière, mesure à prendre avec réserve"
DASH = "—"


def _cell(value, digits=1) -> str:
    return DASH if value is None else fr(value, digits)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["Aucun."]
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


class _Report:
    def __init__(self, doc: dict, poses: dict | None):
        self.doc, self.poses = doc, poses
        self.params = doc["parameters"]
        self.diag = doc["diagnostics"]
        self.els = {k: [e for e in doc["elements"] if e["kind"] == k] for k in ("floor", "wall", "partition", "ceiling", "beam", "column")}
        self.floor_y = self.params["floor.level"]["value"]

    def p(self, pid: str) -> dict:
        return self.params[pid]

    def level(self, pid: str) -> str:
        v = self.p(pid)
        return f"{fr(v['value'] * 100)} ± {fr(v['sigma'] * 100)} cm"

    def source(self, pids: list[str]) -> str:
        return SOURCES[min((self.p(x)["source"] for x in pids), key=SOURCE_RANK.__getitem__)]

    def sections(self) -> list[tuple[str, list[str]]]:
        return [("Entrées", self.inputs()), ("Repère", self.frame()), ("Recalage", self.registration()),
                ("Murs", self.walls()), ("Cloisons", self.partitions()), ("Sol", self.floor()), ("Plafonds", self.ceilings()), ("Poutres", self.beams()),
                ("Poteaux", self.columns()), ("Surfaces verticales non expliquées", self.unexplained()), ("Éléments horizontaux non expliqués (réseaux, luminaires ?)", self.horizontal()),
                ("Équerrage", self.squaring()), ("Alertes", self.alerts()), ("Épaisseurs", self.thickness())]

    def inputs(self) -> list[str]:
        src, n = self.doc["source"], self.doc
        stations = [d for d in n["dimensions"] if d["kind"] in STATION_KINDS]
        counts = " ; ".join(f"{name} : {len(self.els[k])}" for name, k in
                            (("murs", "wall"), ("cloisons", "partition"), ("plafonds", "ceiling"), ("poutres", "beam"), ("poteaux", "column")))
        return [f"- Capture : `{src['capture']}`, pièce {src['room']}",
                f"- Modèle : `{n['schema']}`, version `{n['version']}`",
                f"- Éléments : {counts}",
                f"- Cotes : {len(n['dimensions'])} dont {len(stations)} stations de largeur ou longueur"]

    def frame(self) -> list[str]:
        return [f"- Quarts de tour supplémentaires : {self.doc['frame']['turn']}",
                "- Repère pièce : mètres, Y vers le haut, plan en coordonnées [x, z], murs numérotés dans l'ordre du contour",
                f"- Niveau du sol : y = {self.level('floor.level')}"]

    def registration(self) -> list[str]:
        if not self.poses:
            return ["non exécuté"]
        s = self.poses["summary"]
        reason = [f"- Raffinement écarté, poses ARKit conservées : {self.poses['fallback']}"] if self.poses.get("fallback") else []
        return reason + [f"- Reprojection ARKit : {fr(s['reprojection_px_arkit'], 2)} px",
                f"- Reprojection après recalage : {fr(s['reprojection_px_refined'], 2)} px",
                f"- Échelle Sim3 : {fr(s['sim3_scale'], 4)}",
                f"- Points 3D : {s['points3d']}",
                f"- Images recalées : {s['refined']}, faibles : {s['weak']}",
                f"- Durée : {fr(s['seconds'])} s"]

    def walls(self) -> list[str]:
        rows = []
        for el in self.els["wall"]:
            label, d = el["label"], self.diag["walls"][el["label"]]
            ids = [el["params"]["azimuth"], el["params"]["offset"]]
            alerts = "; ".join(WALL_FLAGS.get(f, f) for f in el["flags"] if f != "squared") or DASH
            rows.append([label, self.source(ids), fr(self.p(ids[1])["sigma"] * 100), _cell(d["rms_cm"], 2),
                         _cell(d["support_m2"]), f"{_cell(d['offset_vs_roomplan_cm'])} cm, {_cell(d['angle_vs_roomplan_deg'])}°",
                         _cell(d["tilt_deg"], 2) + ("" if d["tilt_deg"] is None else "°"), alerts])
        return _table(["Mur", "Source", "σ cm", "RMS cm", "Appui m²", "Écart / angle vs RoomPlan", "Inclinaison", "Alertes"], rows)

    def partitions(self) -> list[str]:
        support = {d["id"]: d for d in self.diag.get("partitions", [])}
        rows = []
        for el in self.els["partition"]:
            p, d = el["params"], support.get(el["id"], {})
            alerts = "; ".join(PARTITION_FLAGS.get(f, f) for f in el["flags"] if f != "squared") or DASH
            length = float(np.linalg.norm(np.subtract(el["end"], el["start"])))
            rows.append([el["label"], self.level(p["thickness"]), SOURCES[self.p(p["thickness"])["source"]],
                         fr(length, 2) + " m", f"{_cell(d.get('support_a_m2'))} / {_cell(d.get('support_b_m2'))}", alerts])
        return _table(["Cloison", "Épaisseur", "Source", "Longueur", "Appui faces A / B m²", "Alertes"], rows)

    def floor(self) -> list[str]:
        d = self.diag["floor"]
        out = [f"- Niveau : y = {self.level('floor.level')} (source : {SOURCES[self.p('floor.level')['source']]})"]
        if d.get("rms_cm") is not None:
            out += [f"- RMS : {fr(d['rms_cm'], 2)} cm, appui {fr(d['support_m2'])} m², inclinaison {fr(d['tilt_deg'], 2)}°",
                    f"- Écart au sol RoomPlan : {fr(d['offset_vs_roomplan_cm'])} cm"]
        return out

    def ceilings(self) -> list[str]:
        by_id = {c["id"]: c for c in self.diag["ceilings"]}
        rows = []
        for el in self.els["ceiling"]:
            c, pid = by_id.get(el["id"], {}), el["params"]["level"]
            rows.append([el["id"].split("_")[1], self.level(pid), fr((self.p(pid)["value"] - self.floor_y) * 100, 0) + " cm",
                         _cell(c.get("area_m2")), _cell(c.get("support_m2")), self.source([pid])])
        frac = self.diag.get("ceiling_unobserved_fraction")
        tail = [] if frac is None else ["", f"Part du plafond non observée par le LiDAR : {fr(frac * 100)} %"]
        return _table(["Plafond", "Niveau y", "Hauteur sous plafond", "Surface m²", "Appui m²", "Source"], rows) + tail

    def beams(self) -> list[str]:
        rows = []
        for el in self.els["beam"]:
            p = el["params"]
            length = float(np.linalg.norm(np.subtract(*el["axis"])))
            rows.append([el["id"].split("_")[1], fr((self.p(p["underside"])["value"] - self.floor_y) * 100) + " ± " +
                         fr(self.p(p["underside"])["sigma"] * 100) + " cm", self.level(p["width"]), fr(length, 2) + " m",
                         "à vérifier" if "to_verify" in el.get("flags", []) else "confirmée"])
        return _table(["Poutre", "Hauteur sous poutre", "Largeur", "Longueur", "État"], rows)

    def columns(self) -> list[str]:
        rows = []
        engaged = {c["id"]: c["engaged_walls"] for c in self.diag.get("columns", [])}
        for el in self.els["column"]:
            p = el["params"]
            walls = engaged.get(el["id"]) or [w for w in [el["engaged_wall"]] if w]
            rows.append([el["id"].split("_")[1], f"({fr(self.p(p['center_x'])['value'], 2)} ; {fr(self.p(p['center_z'])['value'], 2)}) m",
                         self.level(p["size_x"]), self.level(p["size_z"]), ", ".join(walls) or "isolé"])
        return _table(["Poteau", "Centre [x ; z]", "Taille x", "Taille z", "Mur engagé"], rows)

    def unexplained(self) -> list[str]:
        rows = []
        for u in self.diag["unexplained_vertical"]:
            (x0, z0), (x1, z1) = u["bbox"]
            rows.append([f"[{fr(x0, 2)} ; {fr(z0, 2)}] à [{fr(x1, 2)} ; {fr(z1, 2)}]", fr(u["height_max"], 2) + " m",
                         UNEXPLAINED.get(u["reason"], u["reason"])])
        return _table(["Emprise", "Hauteur max", "Raison"], rows)

    def horizontal(self) -> list[str]:
        rows = []
        for u in self.diag.get("unexplained_horizontal", []):
            (x0, z0), (x1, z1) = u["bbox"]
            rows.append([f"[{fr(x0, 2)} ; {fr(z0, 2)}] à [{fr(x1, 2)} ; {fr(z1, 2)}]",
                         fr((u["underside"] - self.floor_y) * 100) + " cm", fr(u["width"] * 100, 0) + " cm",
                         fr(u["length"], 2) + " m", HORIZONTAL.get(u["reason"], u["reason"])])
        return _table(["Emprise", "Hauteur dessous", "Largeur", "Longueur", "Raison"], rows)

    def squaring(self) -> list[str]:
        sq = self.diag.get("squaring")
        if sq is None:
            lines = shell.wall_lines(self.doc)
            corners = []
            for a, b in zip(lines[-1:] + lines[:-1], lines):
                angle = float(np.degrees(np.arccos(np.clip(a["normal"] @ b["normal"], -1, 1))))
                corners.append({"between": [a["label"], b["label"]], "measured_deg": angle, "retained_deg": angle})
            sq = {"corners": corners, "walls": {}, "partitions": {}}
        rows = []
        for c in sq["corners"]:
            a, b = c["between"]
            dev = min(abs(c["retained_deg"] - ref) for ref in (0.0, 90.0, 180.0))
            seen = {w: sq["walls"].get(w, {}) for w in (a, b)}
            notes = [f"{w} hors équerre" for w, e in seen.items() if e.get("state") == "out_of_square"]
            notes += [f"{w} équerre imposée" for w, e in seen.items() if e.get("forced")]
            squared = any(e.get("state") == "squared" for e in seen.values())
            rows.append([f"{a} → {b}", fr(c["measured_deg"]) + "°", fr(c["retained_deg"]) + "°", fr(dev) + "°",
                         ", ".join(notes) or (SQUARING_STATE["squared"] if squared else DASH)])
        out = _table(["Murs consécutifs", "Angle mesuré entre normales", "Angle retenu", "Écart retenu à l'angle droit ou à l'alignement", "État"], rows)
        if "frame_deg" in sq:
            out += ["", f"- Axes de la pièce : {fr(sq['frame_deg'], 2)}° par rapport au repère pièce ; un mur est équerré quand son écart à l'axe ne dépasse "
                        f"pas max({fr(sq['floor_deg'], 0)}°, {fr(sq['sigmas'], 0)} σ de son azimut)"]
        forced = [(f"Mur {k}", e) for k, e in sq["walls"].items()] + [(f"Cloison {k}", e) for k, e in sq["partitions"].items()]
        out += [f"- {name} : équerre imposée, écart de {fr(e['deviation_deg'])}° soit {fr(e['deviation_deg'] / e['sigma_deg'])} σ (σ = {fr(e['sigma_deg'], 2)}°), "
                "au-delà de l'incertitude mais sous la tolérance d'équerrage" for name, e in forced if e.get("forced")]
        out += [f"- Cloison {label} : azimut mesuré {fr(p['measured_deg'])}°, écart {fr(p['deviation_deg'])}° à l'axe, {SQUARING_STATE[p['state']]}"
                for label, p in sq["partitions"].items()]
        return out

    def flag_text(self, texts: dict, group: str, el: dict, flag: str) -> str:
        entry = self.diag.get("squaring", {}).get(group, {}).get(el["label"])
        if flag == "out_of_square" and entry:
            return f"{texts[flag]} ({fr(entry['deviation_deg'])}° pour une tolérance de {fr(entry['tolerance_deg'])}°)"
        return texts.get(flag, flag)

    def alerts(self) -> list[str]:
        out = []
        for el in self.els["wall"]:
            out += [f"- Mur {el['label']} : {self.flag_text(WALL_FLAGS, 'walls', el, f)}" for f in el["flags"] if f != "squared"]
        out += [f"- {GENERAL_FLAGS[f]}" for f in self.diag["flags"] if f in GENERAL_FLAGS]
        known = set(GENERAL_FLAGS) | {f"{e['label']}:{f}" for e in self.els["wall"] + self.els["partition"] for f in e["flags"]}
        out += [f"- {f}" for f in self.diag["flags"] if f not in known]
        out += [f"- Cloison {e['label']} : {self.flag_text(PARTITION_FLAGS, 'partitions', e, f)}" for e in self.els["partition"]
                for f in e["flags"] if f != "squared"]
        out += [f"- Plafond {e['id'].split('_')[1]} : {CEILING_FLAGS.get(f, f)}" for e in self.els["ceiling"] for f in e.get("flags", [])]
        out += [f"- Poutre {e['id'].split('_')[1]} : {TO_VERIFY}" for e in self.els["beam"] if "to_verify" in e.get("flags", [])]
        out += [f"- Mur RoomPlan hors du contour du sol (cloison ?), ignoré : {i}" for i in self.diag["off_outline_walls"]]
        out += [f"- Porte rattachée à un mur hors contour, ignorée : {i}" for i in self.diag.get("orphan_doors", [])]
        out += [f"- Plafond : un second niveau à {fr((n['level'] - self.floor_y) * 100, 0)} cm du sol ({fr(n['support_m2'])} m² d'appui) près de "
                f"[{fr(n['at'][0], 2)} ; {fr(n['at'][1], 2)}] n'a pas pu être découpé proprement, la zone garde un seul niveau"
                for n in self.diag.get("ceiling_notes", [])]
        if (frac := self.diag.get("ceiling_unobserved_fraction")) and frac > UNOBSERVED_ALERT:
            out.append(f"- {fr(frac * 100)} % du plafond non observé par le LiDAR")
        out += [f"- Station {d['id']} : {STATION_FLAG}" for d in self.doc["dimensions"]
                if d["kind"] in STATION_KINDS and OBSTRUCTED in d.get("flags", [])]
        return out or ["Aucune."]

    def thickness(self) -> list[str]:
        values = {self.p(e["params"]["thickness"])["value"] for e in self.els["wall"]}
        cm = fr(next(iter(values)) * 100, 0) if len(values) == 1 else "valeurs variables"
        return [f"Épaisseurs de murs supposées à {cm} cm, non mesurées : elles n'influencent pas les cotes entre faces intérieures."]


def render(doc: dict, poses: dict | None = None) -> str:
    r = _Report(doc, poses)
    out = [f"# Rapport de mesure — {doc['source']['capture']}", ""]
    for title, body in r.sections():
        out += [f"## {title}", "", *body, ""]
    return "\n".join(out)


PARAMETER_NAMES = {"kind": "nature", "through": "traversante", "dividers": "séparations", "shelves_count": "étagères", "width": "largeur", "height": "hauteur", "sill": "allège", "center_s": "position", "projection": "saillie",
                   "type": "type", "hinge": "charnières", "swing": "sens d'ouverture", "panels": "panneaux", "handle": "poignée", "frame": "cadre"}
CORRECTION_TITLES = ("Corrections LLM", "Mesures au mètre", "Corrections manuelles")


def _cm(value) -> str:
    return DASH if value is None else fr(value * 100)


def _gap(delta) -> str:
    from room_model.tape import signed_cm

    return DASH if delta is None else signed_cm(delta)


def _typed(value) -> str:
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, dict):
        return f"{value['profile']}, {fr(value['width_cm'])} cm"
    return DASH if value is None else str(value)


def _llm_section(arch: dict) -> list[str]:
    vision = arch["diagnostics"].get("vision")
    if vision is None:
        note = arch["diagnostics"].get("vision_unreadable")
        return [f"Aucune réponse du LLM appliquée : fichier de réponses inutilisable ({note})." if note
                else "Aucune réponse du LLM appliquée (étape vision non exécutée, ignorée ou en échec)."]
    counts = (("corrigés (au moins deux vues d'accord à 3 cm)", "corrected"), ("corrigés sur une seule vue", "single_view"),
              ("laissés tels quels (vues en désaccord)", "inconsistent"), ("déclarés", "declared"))
    out = ["- " + " ; ".join(f"{len(vision[key])} {label}" for label, key in counts) + f" ; {len(vision['not_applied'])} non appliqués", ""]
    rows = [[c["element"], PARAMETER_NAMES.get(c["parameter"], c["parameter"]), _cm(c["scan"]), _cm(c["llm"]),
             _gap(None if c["scan"] is None else c["llm"] - c["scan"])] for c in vision["corrections"]]
    rows += [[t["element"], PARAMETER_NAMES.get(t["field"], t["field"]), DASH, _typed(t["value"]), DASH] for t in vision["typing"]]
    out += _table(["Élément", "Paramètre", "Scan (cm)", "LLM (cm)", "Écart (cm)"], rows) if rows else ["Aucune valeur corrigée ni déclarée."]
    if vision["inconsistent"]:
        out += ["", "Vues en désaccord, valeurs du scan conservées : " + ", ".join(vision["inconsistent"]) + "."]
    if vision["single_view"]:
        out += ["", "Corrigés sur une seule vue, à vérifier : " + ", ".join(vision["single_view"]) + "."]
    if vision["not_applied"]:
        out += ["", "Non appliqués :", ""] + [f"- {n['id']} : {n['reason']}" for n in vision["not_applied"]]
    return out


def _tape_section(arch: dict) -> list[str]:
    tape = arch["diagnostics"].get("tape")
    if tape is None:
        return ["Aucune mesure au mètre fournie."]
    rows = [[r["id"], _cm(r["scan"]), _cm(r["llm"]), _cm(r["tape"]), _cm(r["applied"]), r["effect"] or DASH] for r in tape["applied"]]
    out = _table(["Cote", "Scan (cm)", "LLM (cm)", "Mètre (cm)", "Retenue (cm)", "Effet"], rows) if rows else ["Aucune cote appliquée."]
    if tape["derived"]:
        out += ["", "Cotes déplacées avec un mur recalé :", ""] + _table(
            ["Cote", "Scan (cm)", "Retenue (cm)"], [[d["id"], _cm(d["scan"]), _cm(d["applied"])] for d in tape["derived"]])
    if tape["not_applied"]:
        out += ["", "Non appliquées :", ""] + [f"- {n['id']} : {n['reason']}" for n in tape["not_applied"]]
    return out


def _manual_section(arch: dict) -> list[str]:
    manual = arch["diagnostics"].get("overrides")
    if manual is None:
        return ["Aucune correction manuelle fournie."]
    rows = [[a["element"], PARAMETER_NAMES.get(a["field"], a["field"]), _typed(a["value"])] for a in manual["applied"]]
    rows += [[i, "suppression", "oui"] for i in manual["removed"]]
    out = _table(["Élément", "Champ", "Valeur"], rows) if rows else ["Aucune correction appliquée."]
    if manual["rejected"]:
        out += ["", "Refusées :", ""] + [f"- {r['id']}{'' if r['field'] is None else ' / ' + r['field']} : {r['reason']}" for r in manual["rejected"]]
    return out


def render_corrections(arch: dict) -> list[tuple[str, list[str]]]:
    """The report sections that come from the architecture stage: LLM corrections, tape measurements, manual corrections."""
    return [("Corrections LLM", _llm_section(arch)), ("Mesures au mètre", _tape_section(arch)), ("Corrections manuelles", _manual_section(arch))]


def write_sections(path, arch: dict) -> None:
    """Replaces the sections of `render_corrections` in the report at `path` (created when absent), keeping the others."""
    kept, skipping = [], False
    for line in (path.read_text(encoding="utf-8").splitlines() if path.exists() else [f"# Rapport de mesure — {arch['source']['capture']}"]):
        if line.startswith("## "):
            skipping = line[3:].strip() in CORRECTION_TITLES
        if not skipping:
            kept.append(line)
    out = ["\n".join(kept).rstrip("\n"), ""]
    for title, body in render_corrections(arch):
        out += [f"## {title}", "", *body, ""]
    path.write_text("\n".join(out), encoding="utf-8")
