"""Dimensioned top view of the model as SVG: x to the right, z down, 1 m = 100 px."""
from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np
import shapely
from shapely import affinity

from room_model import shell

PX_PER_M = 100.0
MARGIN_M = 0.9
CHAR_PX = 6.8  # approximate glyph width of the 12 px label font
LABEL_FRACTIONS = (0.5, 0.35, 0.65, 0.25, 0.75, 0.15, 0.85)
STYLE = (
    "text{font-family:sans-serif;font-size:12px;fill:#222;paint-order:stroke;stroke:#fff;stroke-width:3px}"
    ".wall{fill:#c8c8c0;stroke:#444;stroke-width:1}.floor{fill:#f6f2ea;stroke:none}"
    ".ceiling{fill:#5b8def;fill-opacity:0.18;stroke:#5b8def;stroke-width:1}"
    ".beam{fill:none;stroke:#8a5a2b;stroke-width:2;stroke-dasharray:8 5}"
    ".column{fill:#4a5568;stroke:#222;stroke-width:1}"
    ".dim{stroke:#c0392b;stroke-width:1.5;fill:none}.dimtext{fill:#c0392b}.scale{stroke:#000;stroke-width:3}"
)


def fr(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


class _Plan:
    def __init__(self, doc: dict):
        self.doc = doc
        self.params = doc["parameters"]
        self.walls = {w["label"]: w for w in shell.wall_lines(doc)}
        self.out: list[str] = []
        self.boxes: list[shapely.Polygon] = []
        self.segments: list[shapely.LineString] = []
        floor = next(e for e in doc["elements"] if e["kind"] == "floor")
        self.floor_y = self.params[floor["params"]["level"]]["value"]
        self.bodies = {el["label"]: self._body(el) for el in doc["elements"] if el["kind"] == "wall"}
        pts = np.vstack([floor["polygon"]] + list(self.bodies.values()))
        self.x0, self.z0 = pts.min(0) - MARGIN_M
        self.w, self.h = (pts.max(0) + MARGIN_M - [self.x0, self.z0]) * PX_PER_M

    def value(self, pid: str) -> float:
        return self.params[pid]["value"]

    def _body(self, el: dict) -> np.ndarray:
        w = self.walls[el["label"]]
        thickness = self.value(el["params"]["thickness"])
        t = (w["end"] - w["start"]) / np.linalg.norm(w["end"] - w["start"])
        a, b = w["start"] - t * thickness, w["end"] + t * thickness
        return np.array([a, b, b - w["normal"] * thickness, a - w["normal"] * thickness])

    def px(self, p) -> tuple[float, float]:
        return (p[0] - self.x0) * PX_PER_M, (p[1] - self.z0) * PX_PER_M

    def points(self, pts) -> str:
        return " ".join("%.1f,%.1f" % self.px(p) for p in pts)

    def text(self, p, content: str, cls: str = "", anchor: str = "middle", dy: float = 4.0) -> None:
        x, y = self.px(p)
        w = len(content) * CHAR_PX
        x0 = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
        self.boxes.append(shapely.box(x0, y + dy - 11, x0 + w, y + dy + 3))
        self.out.append(f'<text class="{cls}" x="{x:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}">{escape(content)}</text>')

    def draw(self) -> str:
        els = self.doc["elements"]
        floor = next(e for e in els if e["kind"] == "floor")
        self.out.append(f'<polygon id="Floor" class="floor" points="{self.points(floor["polygon"])}"/>')
        for el in els:
            if el["kind"] == "ceiling":
                self._ceiling(el)
        for el in els:
            if el["kind"] == "wall":
                self._wall(el)
        for el in els:
            if el["kind"] == "beam":
                self._beam(el)
            elif el["kind"] == "column":
                self._column(el)
        stations = [d for d in self.doc["dimensions"] if d["id"].startswith("S") and d["at"] is not None]
        ends = {d["id"]: self._ends(d) for d in stations}
        self.segments = [shapely.LineString(e) for e in ends.values()]
        for dim in stations:
            self._station(dim, *ends[dim["id"]])
        self._scale_bar()
        return (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" '
                f'height="{self.h:.0f}" viewBox="0 0 {self.w:.0f} {self.h:.0f}">\n<style>{STYLE}</style>\n'
                f'<rect width="100%" height="100%" fill="#fff"/>\n' + "\n".join(self.out) + "\n</svg>\n")

    def _ceiling(self, el: dict) -> None:
        k = el["id"].split("_")[1]
        d = "".join("M" + " L".join("%.1f,%.1f" % self.px(p) for p in ring) + " Z"
                    for r in el["region"] for ring in [r["exterior"], *r["holes"]])
        self.out.append(f'<path id="{el["id"]}" class="ceiling" fill-rule="evenodd" d="{d}"/>')
        region = shapely.MultiPolygon([shapely.Polygon(r["exterior"], r["holes"]) for r in el["region"]])
        height = self.value(el["params"]["level"]) - self.floor_y
        c = region.representative_point()
        self.text((c.x, c.y), f"Plafond {k} : {fr(height * 100, 0)} cm", "ceilingtext", dy=-16.0)

    def _wall(self, el: dict) -> None:
        label = el["label"]
        self.out.append(f'<polygon id="{el["id"]}" class="wall" points="{self.points(self.bodies[label])}"/>')
        w = self.walls[label]
        mid = (w["start"] + w["end"]) / 2 - w["normal"] * (self.value(el["params"]["thickness"]) + 0.25)
        self.text(mid, label, "walllabel")

    def _beam(self, el: dict) -> None:
        k = el["id"].split("_")[1]
        a, b = np.asarray(el["axis"], float)
        t = (b - a) / np.linalg.norm(b - a)
        width = self.value(el["params"]["width"])
        side = np.array([-t[1], t[0]]) * width / 2
        self.out.append(f'<polygon id="{el["id"]}" class="beam" points="{self.points([a - side, b - side, b + side, a + side])}"/>')
        under = self.value(el["params"]["underside"]) - self.floor_y
        near_start = a + 0.2 * (b - a) + (width / 2 + 0.1) * np.array([-t[1], t[0]])
        self.text(near_start, f"B{k} s/p {fr(under * 100, 0)} cm" + (" ?" if "to_verify" in el.get("flags", []) else ""), "beamtext", anchor="end")

    def _column(self, el: dict) -> None:
        p = el["params"]
        cx, cz, sx, sz = (self.value(p[n]) for n in ("center_x", "center_z", "size_x", "size_z"))
        quad = [[cx - sx / 2, cz - sz / 2], [cx + sx / 2, cz - sz / 2], [cx + sx / 2, cz + sz / 2], [cx - sx / 2, cz + sz / 2]]
        self.out.append(f'<polygon id="{el["id"]}" class="column" points="{self.points(quad)}"/>')
        self.text((cx + sx / 2 + 0.05, cz), f"P{el['id'].split('_')[1]}", "columntext", anchor="start")

    def _ends(self, dim: dict) -> tuple[tuple[float, float], tuple[float, float]]:
        n = self.walls[dim["elements"][0].removeprefix("Wall_")]["normal"]
        at = np.asarray(dim["at"], float)
        return self.px(at - n * dim["value"] / 2), self.px(at + n * dim["value"] / 2)

    def _label_slot(self, a, b, width: float):
        """Rotated label polygon, centre and angle at the first spot along a-b that touches no other text or dimension line."""
        a, b = np.array(a), np.array(b)
        angle = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
        angle = angle - 180 if angle > 90 else angle + 180 if angle <= -90 else angle
        rad = np.radians(angle)
        perp = np.array([np.sin(rad), -np.cos(rad)])
        slot = None
        for f in LABEL_FRACTIONS:
            c = a + f * (b - a) + perp * 9
            box = affinity.rotate(shapely.box(c[0] - width / 2, c[1] - 7, c[0] + width / 2, c[1] + 7), angle, origin=tuple(c))
            slot = slot or (box, c, angle)
            if not any(box.intersects(o) for o in self.boxes + self.segments):
                return box, c, angle
        return slot

    def _station(self, dim: dict, pa, pb) -> None:
        (xa, ya), (xb, yb) = pa, pb
        n = np.array([xb - xa, yb - ya]) / np.hypot(xb - xa, yb - ya)
        tick = np.array([-n[1], n[0]]) * 6
        parts = [f"M{xa:.1f},{ya:.1f} L{xb:.1f},{yb:.1f}"]
        for x, y in ((xa, ya), (xb, yb)):
            parts.append(f"M{x - tick[0]:.1f},{y - tick[1]:.1f} L{x + tick[0]:.1f},{y + tick[1]:.1f}")
        self.out.append(f'<path id="dim-{dim["id"]}" class="dim" d="{" ".join(parts)}"/>')
        content = f"{dim['id']} {fr(dim['value'] * 100)} ± {fr(dim['sigma'] * 100)} cm"
        box, c, angle = self._label_slot(pa, pb, len(content) * CHAR_PX)
        self.boxes.append(box)
        self.out.append(f'<text id="label-{dim["id"]}" class="dimtext" x="{c[0]:.1f}" y="{c[1] + 4:.1f}" text-anchor="middle" '
                        f'transform="rotate({angle:.1f} {c[0]:.1f} {c[1]:.1f})">{escape(content)}</text>')

    def _scale_bar(self) -> None:
        x, y = 20.0, self.h - 20.0
        self.out.append(f'<path class="scale" d="M{x:.1f},{y:.1f} L{x + PX_PER_M:.1f},{y:.1f}"/>')
        self.out.append(f'<text x="{x + PX_PER_M + 8:.1f}" y="{y + 4:.1f}">1 m</text>')


def render(doc: dict) -> str:
    return _Plan(doc).draw()
