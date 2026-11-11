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
    """Sizes in metres; `bottom` is the height of the cut above the floor, `casing` the drawn casing width (0: none).

    `leaf_w` and `leaf_h` are None for an opening or a niche. `flags` holds `leaf_estimated` and `to_verify`.
    """
    cut_w: float
    cut_h: float
    bottom: float
    casing: float
    bottom_casing: bool
    jamb: float = JAMB_VISIBLE
    leaf_w: float | None = None
    leaf_h: float | None = None
    flags: tuple[str, ...] = ()

    @property
    def half_outer(self) -> float:
        """Half the width the casings reach to, from the centre of the cut."""
        return self.cut_w / 2 + self.casing


def _param(arch: dict, el: dict, name: str) -> float | None:
    pid = el["params"].get(name)
    return None if pid is None else frame.value(arch, pid)


def resolve(arch: dict, el: dict) -> Layout:
    """The layout of a door, window, opening or niche."""
    w, h, sill = (_param(arch, el, k) for k in ("width", "height", "sill"))
    spec = el.get("frame") or {}
    casing = spec.get("width")
    if el["kind"] not in JOINERY_KINDS:
        return Layout(w, h, sill, casing or 0.0, sill > SILL_TOL)
    door = el["kind"] == "door"
    jamb = _param(arch, el, "jamb_width") or JAMB_VISIBLE
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
    bottom = sill if door else sill + max(0.0, (h - cut_h) / 2)
    drawn = 0.0
    if casing is not None:
        drawn = (w - cut_w) / 2
        if not CASING_MIN <= drawn <= CASING_MAX:
            drawn = min(max(drawn, CASING_MIN), CASING_MAX)
            flags.append("to_verify")
    return Layout(cut_w, cut_h, bottom, drawn, not door, jamb, leaf_w, leaf_h, tuple(dict.fromkeys(flags)))
