"""French Markdown report of a model document."""
from __future__ import annotations

import numpy as np

from room_model import shell
from room_model.model import SOURCE_RANK
from room_model.plan_svg import fr

WALL_FLAGS = {
    "roomplan_fallback": "mur non vu par le LiDAR, position RoomPlan conservée",
    "low_support": "appui LiDAR faible, ajustement peu fiable",
    "seeded_from_outline": "mur absent de la liste RoomPlan, initialisé depuis le contour du sol",
    "near_parallel_corner": "angle quasi plat avec un mur voisin, coin obtenu par projection",
}
GENERAL_FLAGS = {
    "floor:roomplan_fallback": "sol non vu par le LiDAR, niveau RoomPlan conservé",
    "columns:no_ceiling": "aucun plafond détecté, poteaux non recherchés",
}
UNEXPLAINED = {
    "not_full_height": "n'atteint pas le plafond (meuble ou cloison basse)",
    "open_sides": "faces non fermées, pas un poteau",
    "size_out_of_range": "dimensions hors de la plage d'un poteau",
}
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
        self.els = {k: [e for e in doc["elements"] if e["kind"] == k] for k in ("floor", "wall", "ceiling", "beam", "column")}
        self.floor_y = self.params["floor.level"]["value"]

    def p(self, pid: str) -> dict:
        return self.params[pid]

    def level(self, pid: str) -> str:
        v = self.p(pid)
        return f"{fr(v['value'] * 100)} ± {fr(v['sigma'] * 100)} cm"

    def source(self, pids: list[str]) -> str:
        return min((self.p(x)["source"] for x in pids), key=SOURCE_RANK.__getitem__)

    def sections(self) -> list[tuple[str, list[str]]]:
        return [("Entrées", self.inputs()), ("Repère", self.frame()), ("Recalage", self.registration()),
                ("Murs", self.walls()), ("Sol", self.floor()), ("Plafonds", self.ceilings()), ("Poutres", self.beams()),
                ("Poteaux", self.columns()), ("Surfaces verticales non expliquées", self.unexplained()),
                ("Orthogonalité", self.orthogonality()), ("Alertes", self.alerts()), ("Épaisseurs", self.thickness())]

    def inputs(self) -> list[str]:
        src, n = self.doc["source"], self.doc
        stations = [d for d in n["dimensions"] if d["id"].startswith("S")]
        counts = " ; ".join(f"{name} : {len(self.els[k])}" for name, k in
                            (("murs", "wall"), ("plafonds", "ceiling"), ("poutres", "beam"), ("poteaux", "column")))
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
        return [f"- Reprojection ARKit : {fr(s['reprojection_px_arkit'], 2)} px",
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
            alerts = "; ".join(WALL_FLAGS.get(f, f) for f in el["flags"]) or DASH
            rows.append([label, self.source(ids), fr(self.p(ids[1])["sigma"] * 100), _cell(d["rms_cm"], 2),
                         _cell(d["support_m2"]), f"{_cell(d['offset_vs_roomplan_cm'])} cm, {_cell(d['angle_vs_roomplan_deg'])}°",
                         _cell(d["tilt_deg"], 2) + ("" if d["tilt_deg"] is None else "°"), alerts])
        return _table(["Mur", "Source", "σ cm", "RMS cm", "Appui m²", "Écart / angle vs RoomPlan", "Inclinaison", "Alertes"], rows)

    def floor(self) -> list[str]:
        d = self.diag["floor"]
        out = [f"- Niveau : y = {self.level('floor.level')} (source : {self.p('floor.level')['source']})"]
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
                         _cell(c.get("area_m2")), _cell(c.get("support_m2"))])
        frac = self.diag.get("ceiling_unobserved_fraction")
        tail = [] if frac is None else ["", f"Part du plafond non observée par le LiDAR : {fr(frac * 100)} %"]
        return _table(["Plafond", "Niveau y", "Hauteur sous plafond", "Surface m²", "Appui m²"], rows) + tail

    def beams(self) -> list[str]:
        rows = []
        for el in self.els["beam"]:
            p = el["params"]
            length = float(np.linalg.norm(np.subtract(*el["axis"])))
            rows.append([el["id"].split("_")[1], fr((self.p(p["underside"])["value"] - self.floor_y) * 100) + " ± " +
                         fr(self.p(p["underside"])["sigma"] * 100) + " cm", self.level(p["width"]), fr(length, 2) + " m"])
        return _table(["Poutre", "Hauteur sous poutre", "Largeur", "Longueur"], rows)

    def columns(self) -> list[str]:
        rows = []
        for el in self.els["column"]:
            p = el["params"]
            rows.append([el["id"].split("_")[1], f"({fr(self.p(p['center_x'])['value'], 2)} ; {fr(self.p(p['center_z'])['value'], 2)}) m",
                         self.level(p["size_x"]), self.level(p["size_z"]), el["engaged_wall"] or "isolé"])
        return _table(["Poteau", "Centre [x ; z]", "Taille x", "Taille z", "Mur engagé"], rows)

    def unexplained(self) -> list[str]:
        rows = []
        for u in self.diag["unexplained_vertical"]:
            (x0, z0), (x1, z1) = u["bbox"]
            rows.append([f"[{fr(x0, 2)} ; {fr(z0, 2)}] à [{fr(x1, 2)} ; {fr(z1, 2)}]", fr(u["height_max"], 2) + " m",
                         UNEXPLAINED.get(u["reason"], u["reason"])])
        return _table(["Emprise", "Hauteur max", "Raison"], rows)

    def orthogonality(self) -> list[str]:
        lines = shell.wall_lines(self.doc)
        rows = []
        for a, b in zip(lines, lines[1:] + lines[:1]):
            angle = float(np.degrees(np.arccos(np.clip(a["normal"] @ b["normal"], -1, 1))))
            dev = min(abs(angle - ref) for ref in (0.0, 90.0, 180.0))
            rows.append([f"{a['label']} → {b['label']}", fr(angle) + "°", fr(dev) + "°"])
        return _table(["Murs consécutifs", "Angle entre normales", "Écart à 90° ou 180°"], rows)

    def alerts(self) -> list[str]:
        out = []
        for el in self.els["wall"]:
            out += [f"- Mur {el['label']} : {WALL_FLAGS.get(f, f)}" for f in el["flags"]]
        out += [f"- {GENERAL_FLAGS[f]}" for f in self.diag["flags"] if f in GENERAL_FLAGS]
        known = set(GENERAL_FLAGS) | {f"{e['label']}:{f}" for e in self.els["wall"] for f in e["flags"]}
        out += [f"- {f}" for f in self.diag["flags"] if f not in known]
        out += [f"- Mur RoomPlan hors du contour du sol (cloison ?), ignoré : {i}" for i in self.diag["off_outline_walls"]]
        if frac := self.diag.get("ceiling_unobserved_fraction"):
            out.append(f"- {fr(frac * 100)} % du plafond non observé par le LiDAR")
        out += [f"- Station {d['id']} : {STATION_FLAG}" for d in self.doc["dimensions"] if d.get("flags")]
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
