"""Wall cut, leaf, jambs and casings of an opening, from what was measured (pure Python, shared by shell, joinery, mouldings).

Doors and windows are built from their leaf: the cut is the leaf plus the visible jamb on each side, the casing is
what remains of the outer extent (`width`, `height`: the LiDAR hole, casings included). Openings and niches are
measured jamb to jamb: the cut is their size and the casing, when any, lies outside it.
"""
from dataclasses import dataclass

from . import frame

# Visible width of the fixed frame around a leaf: the sightline of a standard interior door frame (4 to 6 cm) and of a window dormant.
JAMB_VISIBLE = 0.05
# A casing under 2 cm is a moulding edge, not an architrave; over 20 cm the outer extent holds something else (a reveal, a recess).
CASING_MIN, CASING_MAX = 0.02, 0.20
MIN_LEAF = 0.10  # m: a smaller derived leaf means the casing was over-estimated
SILL_TOL = 0.02  # m: an opening whose sill is below this reaches the floor, so its casing has no bottom bar
JOINERY_KINDS = ("door", "window")


@dataclass(frozen=True)
class Layout:
    """Sizes in metres. `centre` is the cut's position along the host (`leaf_center_s` when measured, else `center_s`),
    `bottom` the height of the cut above the floor, `casings` the drawn casing widths (left, top, right, bottom; 0: none).

    `leaf_w` and `leaf_h` are None for an opening or a niche. `flags` holds `leaf_estimated` and `to_verify`.
    """
    cut_w: float
    cut_h: float
    bottom: float
    centre: float
    casings: tuple[float, float, float, float]
    jamb: float = JAMB_VISIBLE
    leaf_w: float | None = None
    leaf_h: float | None = None
    flags: tuple[str, ...] = ()

    @property
    def span(self) -> tuple[float, float]:
        """(low, high) position of the cut along the host."""
        return self.centre - self.cut_w / 2, self.centre + self.cut_w / 2

    @property
    def outer_span(self) -> tuple[float, float]:
        """The same out to the outer edge of the casings."""
        return self.span[0] - self.casings[0], self.span[1] + self.casings[2]

    @property
    def lowest(self) -> float:
        """Height of the lowest thing the opening puts on the wall: the casing's bottom bar when it has one."""
        return self.bottom - self.casings[3]


def _param(arch: dict, el: dict, name: str) -> float | None:
    pid = el["params"].get(name)
    return None if pid is None else frame.value(arch, pid)


def _clamp(raw: float, flags: list) -> float:
    if not CASING_MIN <= raw <= CASING_MAX:
        flags.append("to_verify")
    return min(max(raw, CASING_MIN), CASING_MAX)


def resolve(arch: dict, el: dict) -> Layout:
    """The layout of a door, window, opening or niche."""
    w, h, sill, outer_centre = (_param(arch, el, k) for k in ("width", "height", "sill", "center_s"))
    centre = _param(arch, el, "leaf_center_s")
    centre = outer_centre if centre is None else centre
    spec = el.get("frame") or {}
    casing = spec.get("width")
    if el["kind"] not in JOINERY_KINDS:
        c = casing or 0.0
        return Layout(w, h, sill, centre, (c, c, c, c if sill > SILL_TOL else 0.0))
    door = el["kind"] == "door"
    jamb = _param(arch, el, "jamb_width")
    jamb = JAMB_VISIBLE if jamb is None else jamb  # an explicit 0 (a frameless leaf) is a value
    flags = []
    leaf_w, leaf_h = _param(arch, el, "leaf_width"), _param(arch, el, "leaf_height")
    if leaf_w is None or leaf_h is None:
        flags.append("leaf_estimated")
    if leaf_w is None:
        leaf_w = w - 2 * ((casing or 0.0) + jamb)
        if leaf_w < MIN_LEAF:
            leaf_w = max(w - 2 * jamb, MIN_LEAF)
            flags.append("to_verify")
    ends = 1 if door else 2  # a door has no jamb or casing at the floor
    if leaf_h is None:
        leaf_h = h - ends * ((casing or 0.0) + jamb)
        if leaf_h < MIN_LEAF:
            leaf_h = max(h - ends * jamb, MIN_LEAF)
            flags.append("to_verify")
    cut_w, cut_h = leaf_w + 2 * jamb, leaf_h + ends * jamb
    low, high = max(centre - cut_w / 2, outer_centre - w / 2), min(centre + cut_w / 2, outer_centre + w / 2)
    if high - low < cut_w - 1e-9 or cut_h > h + 1e-9:  # the cut never opens the wall beyond the measured outer extent
        flags.append("to_verify")
        if high - low < MIN_LEAF + 2 * jamb:  # the leaf lies (almost) outside the extent: keep its width, centred in it
            low, high = outer_centre - min(cut_w, w) / 2, outer_centre + min(cut_w, w) / 2
        centre, cut_w, cut_h = (low + high) / 2, high - low, min(cut_h, h)
        leaf_w, leaf_h = max(cut_w - 2 * jamb, MIN_LEAF), max(cut_h - ends * jamb, MIN_LEAF)
    bottom = sill if door else sill + max(0.0, (h - cut_h) / 2)
    casings = (0.0,) * 4
    if casing is not None:  # what is left between each edge of the cut and the outer extent
        casings = (_clamp((centre - cut_w / 2) - (outer_centre - w / 2), flags), _clamp(sill + h - (bottom + cut_h), flags),
                   _clamp((outer_centre + w / 2) - (centre + cut_w / 2), flags), 0.0 if door else _clamp(bottom - sill, flags))
    return Layout(cut_w, cut_h, bottom, centre, casings, jamb, leaf_w, leaf_h, tuple(dict.fromkeys(flags)))
