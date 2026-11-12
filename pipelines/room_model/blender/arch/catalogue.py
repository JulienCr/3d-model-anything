"""The joinery catalogue: what each door and window type is made of, and the typing fields an element may carry."""
from dataclasses import dataclass

HINGED, SLIDING, POCKET, FIXED = "hinged", "sliding", "pocket", "fixed"


@dataclass(frozen=True)
class Spec:
    """`leaves` moving or fixed panels; `motion` how `Ouverture` moves them; `glazed` leaves are rails around glass."""
    leaves: int
    motion: str
    glazed: bool = False


CATALOGUE = {
    "door": {
        "hinged_single": Spec(1, HINGED),
        "hinged_double": Spec(2, HINGED),
        "sliding": Spec(1, SLIDING),
        "pocket": Spec(1, POCKET),
        "french_window": Spec(2, HINGED, glazed=True),
    },
    "window": {
        "casement_single": Spec(1, HINGED, glazed=True),
        "casement_double": Spec(2, HINGED, glazed=True),
        "tilt_turn": Spec(1, HINGED, glazed=True),  # turns like a casement; the tilt is not rigged
        "sliding": Spec(2, SLIDING, glazed=True),
        "fixed": Spec(1, FIXED, glazed=True),
    },
}
DEFAULT_TYPE = {"door": "hinged_single", "window": "casement_double"}
SIDES = ("left", "right")
SWINGS = ("in", "out")


@dataclass(frozen=True)
class Joinery:
    """An element's resolved typing. `hinge` is the side of the hinges (or of the stack) for a viewer facing the swing direction."""
    kind: str
    type: str
    spec: Spec
    hinge: str
    swing: str
    panels: int
    handle: bool


def resolve(element: dict) -> Joinery | None:
    """The typing of a door or window with the brief's defaults; None for every other kind (an `opening` stays a bare cut)."""
    kind = element["kind"]
    if kind not in CATALOGUE:
        return None
    name = element.get("type") or DEFAULT_TYPE[kind]
    if name not in CATALOGUE[kind]:
        raise ValueError(f"{element['id']}: unknown {kind} type {name!r}, expected one of {', '.join(CATALOGUE[kind])}")
    hinge, swing = element.get("hinge") or "left", element.get("swing") or "in"
    if hinge not in SIDES or swing not in SWINGS:
        raise ValueError(f"{element['id']}: hinge {hinge!r} / swing {swing!r} must be one of {SIDES} / {SWINGS}")
    panels = int(element.get("panels") or 0)
    if panels < 0:
        raise ValueError(f"{element['id']}: panels must be >= 0, got {panels}")
    return Joinery(kind, name, CATALOGUE[kind][name], hinge, swing, panels, bool(element.get("handle", True)))
