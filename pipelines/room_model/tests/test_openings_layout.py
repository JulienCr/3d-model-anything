"""Sizes of the wall cut, leaf, jambs and casings of an opening (pure Python, no Blender)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender"))
from arch import openings  # noqa: E402

J = openings.JAMB_VISIBLE


def _arch(kind, width=1.04, height=2.12, sill=0.0, frame=None, **extra):
    params = {"width": width, "height": height, "sill": sill, "center_s": 1.0, **extra}
    el = {"id": "El_1", "kind": kind, "host": "Wall_M1", "face": None, "params": {k: f"p.{k}" for k in params}, "frame": frame}
    return {"parameters": {f"p.{k}": {"value": v} for k, v in params.items()}, "elements": [el]}, el


def _layout(kind, **kw):
    arch, el = _arch(kind, **kw)
    return openings.resolve(arch, el)


def test_leaf_given_sets_the_cut_and_the_casing_is_what_remains():
    lay = _layout("door", leaf_width=0.8, leaf_height=2.0, frame={"width": 0.07})
    assert lay.leaf_w == pytest.approx(0.8) and lay.cut_w == pytest.approx(0.8 + 2 * J)
    assert lay.cut_h == pytest.approx(2.0 + J) and lay.bottom == 0.0
    assert lay.casings[0] == pytest.approx((1.04 - lay.cut_w) / 2) and lay.casings[2] == lay.casings[0] and lay.flags == ()
    assert lay.casings[1] == pytest.approx(2.12 - lay.cut_h) and lay.casings[3] == 0.0


def test_leaf_is_derived_from_the_outer_extent_and_the_casing():
    lay = _layout("door", frame={"width": 0.07})
    assert lay.leaf_w == pytest.approx(1.04 - 2 * (0.07 + J)) and lay.leaf_h == pytest.approx(2.12 - (0.07 + J))
    assert lay.cut_w == pytest.approx(1.04 - 2 * 0.07) and lay.casings == pytest.approx((0.07, 0.07, 0.07, 0.0))
    assert lay.flags == ("leaf_estimated",)


def test_leaf_without_a_casing_is_the_outer_extent_minus_the_jambs():
    lay = _layout("door", width=0.9)
    assert lay.leaf_w == pytest.approx(0.9 - 2 * J) and lay.cut_w == pytest.approx(0.9) and lay.casings == (0.0,) * 4


def test_window_has_a_jamb_and_a_casing_at_the_top_and_the_bottom():
    lay = _layout("window", height=1.2, sill=0.9, frame={"width": 0.06}, width=1.0)
    assert lay.leaf_h == pytest.approx(1.2 - 2 * (0.06 + J)) and lay.cut_h == pytest.approx(1.2 - 2 * 0.06)
    assert lay.bottom == pytest.approx(0.9 + 0.06) and lay.casings[3] == pytest.approx(0.06) and lay.lowest == pytest.approx(0.9)


def test_jamb_width_overrides_the_constant():
    lay = _layout("door", width=0.9, jamb_width=0.08)
    assert lay.jamb == 0.08 and lay.leaf_w == pytest.approx(0.9 - 0.16) and lay.cut_w == pytest.approx(0.9)


@pytest.mark.parametrize("outer,expected", [(1.5, openings.CASING_MAX), (0.92, openings.CASING_MIN)])
def test_casing_out_of_range_is_clamped_and_flagged(outer, expected):
    lay = _layout("door", width=outer, leaf_width=0.8, leaf_height=2.0, frame={"width": 0.07})
    assert lay.casings[0] == lay.casings[2] == expected and "to_verify" in lay.flags


def test_a_tiny_derived_leaf_falls_back_to_the_jambs_alone():
    lay = _layout("door", width=0.4, frame={"width": 0.15})
    assert lay.leaf_w == pytest.approx(0.4 - 2 * J) and "to_verify" in lay.flags


@pytest.mark.parametrize("kind", ["opening", "niche"])
def test_openings_and_niches_are_cut_at_their_measured_size(kind):
    lay = _layout(kind, width=0.9, height=2.0, sill=0.3, frame={"width": 0.05})
    assert (lay.cut_w, lay.cut_h, lay.bottom, lay.casings, lay.flags) == (0.9, 2.0, 0.3, (0.05,) * 4, ())
    assert lay.leaf_w is None and lay.lowest == pytest.approx(0.25)


def test_an_off_centre_leaf_moves_the_cut_and_makes_the_casings_unequal():
    lay = _layout("door", leaf_width=0.8, leaf_height=2.0, leaf_center_s=1.04, frame={"width": 0.07})
    assert lay.centre == pytest.approx(1.04) and lay.span == pytest.approx((1.04 - 0.45, 1.04 + 0.45))
    assert lay.casings[0] == pytest.approx((1.04 - 0.45) - (1.0 - 0.52)) and lay.casings[2] == pytest.approx((1.0 + 0.52) - (1.04 + 0.45))
    assert lay.outer_span == pytest.approx((1.0 - 0.52, 1.0 + 0.52)) and lay.flags == ()


def test_each_casing_is_clamped_on_its_own_and_flagged():
    lay = _layout("door", leaf_width=0.8, leaf_height=2.0, leaf_center_s=1.12, frame={"width": 0.07})
    assert lay.casings[0] == pytest.approx(0.19) and lay.casings[2] == openings.CASING_MIN
    assert "to_verify" in lay.flags


def test_leaf_centre_defaults_to_the_element_centre():
    assert _layout("door", frame={"width": 0.07}).centre == 1.0


def test_a_window_without_a_casing_starts_at_its_cut_not_at_its_outer_sill():
    lay = _layout("window", sill=0.08, height=1.2, width=1.0, leaf_width=0.8, leaf_height=0.9)
    assert lay.casings == (0.0,) * 4 and lay.lowest == lay.bottom == pytest.approx(0.08 + (1.2 - 1.0) / 2)
