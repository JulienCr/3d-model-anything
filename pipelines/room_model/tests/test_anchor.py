import json

import numpy as np
import pytest
from scipy import ndimage

from room_model import anchor, model, openings, ortho, shell
from room_model.ortho import FaceFrame
from room_model.photo_geom import DepthMaps
from helpers import make_case
from synthetic import DOOR, FacePatch, render_face

FRAME = FaceFrame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
WALL, CASING, LEAF, DARK = 150, 225, 190, 40
LEAF_S, LEAF_TOP, LEAF_DELTA = (0.55, 1.25), 2.08, -0.04
BOUNDS = (-0.20, 0.06)


def _wall(hole_s, hole_y, extra=()):
    """Wall at delta 0 around a hole, the patches in `extra` and a dark space far behind."""
    (s0, s1), (y0, y1) = hole_s, hole_y
    return [FacePatch((-2, s0), (-1, 4), 0.0, WALL), FacePatch((s1, 3), (-1, 4), 0.0, WALL),
            FacePatch((s0, s1), (y1, 4), 0.0, WALL), FacePatch((s0, s1), (-1, y0), 0.0, WALL), *extra,
            FacePatch((-2, 3), (-1, 4), -0.6, DARK)]


def _door_patches(leaves=(LEAF_S,)):
    casing = [FacePatch((0.40, 0.50), (0, 2.20), 0.015, CASING), FacePatch((1.30, 1.40), (0, 2.20), 0.015, CASING),
              FacePatch((0.40, 1.40), (2.10, 2.20), 0.015, CASING)]
    leaf = [FacePatch(s, (0.01, LEAF_TOP), LEAF_DELTA, LEAF) for s in leaves]
    return _wall((0.40, 1.40), (0.0, 2.20), casing + leaf)


def _views(patches, angles=(-35, 0, 30), target_s=0.9):
    out = []
    for k, angle in enumerate(angles):
        pos = (target_s + 2.2 * np.tan(np.radians(angle)), 1.3, 2.2)
        image, photo = render_face(FRAME, patches, pos, (target_s, 1.05, 0.0), name=f"{k}.jpg")
        out.append((photo, image))
    return out


@pytest.fixture(scope="module")
def door_views():
    return _views(_door_patches())


def _at_face(s_true, delta, camera):
    """Where an edge at (s_true, delta) appears on the delta = 0 ortho image of a camera at (s, y, delta)."""
    return s_true + (camera[0] - s_true) * (0.0 - delta) / (camera[2] - delta)


def colour_stub(rgb, box, points=None):
    """Stand-in for SAM: the region of uniform colour under the box centre, or nothing when it is a sliver."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    ref = np.median(rgb[cy - 2:cy + 3, cx - 2:cx + 3].reshape(-1, 3), axis=0)
    same = np.abs(rgb.astype(float) - ref).max(axis=2) < 20
    labels, _ = ndimage.label(same)
    mask = labels == labels[cy, cx]
    return [(mask, 0.9)]


def nothing_stub(rgb, box, points=None):
    return []


# --- ortho ---------------------------------------------------------------------------------------------------------

def test_ortho_round_trip_puts_a_mark_back_on_its_face_coordinates():
    mark = FacePatch((0.70, 0.72), (1.10, 1.12), -0.05, 250)
    image, photo = render_face(FRAME, [mark, FacePatch((-2, 3), (-1, 4), -0.05, 60)], (0.2, 1.4, 2.0), (0.8, 1.1, 0.0))
    o = ortho.ortho(photo, image, FRAME, (0.5, 0.9, 0.9, 1.3), delta=-0.05)
    bright = o.rgb[..., 0] > 155
    rows, cols = np.nonzero(bright)
    assert abs(o.ss[cols].mean() - 0.71) < 0.001 and abs(o.ys[rows].mean() - 1.11) < 0.001
    shifted = ortho.ortho(photo, image, FRAME, (0.5, 0.9, 0.9, 1.3), delta=0.0)
    cols = np.nonzero(shifted.rgb[..., 0] > 155)[1]
    assert abs(shifted.ss[cols].mean() - 0.71) > 0.005  # the same mark read on the wrong plane moves


# --- edge triangulation --------------------------------------------------------------------------------------------

def _left_observations(views):
    obs, at_face = [], []
    for photo, image in views:
        cam = ortho.camera_coords(FRAME, photo)
        expected = _at_face(LEAF_S[0], LEAF_DELTA, cam)
        o = ortho.ortho(photo, image, FRAME, (0.3, 1.5, -0.05, 2.3))
        found = ortho.snap(o, "s", (0.8, 1.4), expected + 0.004, anchor.SNAP)
        at_face.append((found, expected, cam))
        obs.append((found, cam[0], cam[2]))
    return obs, at_face


def test_triangulated_edge_recovers_the_position_and_depth_of_a_recessed_leaf(door_views):
    obs, at_face = _left_observations(door_views)
    for found, expected, _ in at_face:
        assert abs(found - expected) < 0.002
    fix = ortho.triangulate(obs, BOUNDS)
    assert abs(fix.pos - LEAF_S[0]) < 0.002 and abs(fix.delta - LEAF_DELTA) < 0.01 and fix.sigma < 0.003 and fix.views == 3


def test_reading_a_recessed_edge_on_the_face_plane_is_off_by_the_parallax(door_views):
    _, at_face = _left_observations(door_views)
    oblique = at_face[2]
    assert abs(oblique[0] - LEAF_S[0]) > 0.015  # what projecting the model on the face plane does


def test_a_single_view_carries_the_depth_ambiguity_in_its_sigma(door_views):
    obs, at_face = _left_observations(door_views)
    found, _, cam = at_face[2]
    fix = ortho.triangulate([obs[2]], BOUNDS)
    span = abs(cam[0] - found) / cam[2] * (BOUNDS[1] - BOUNDS[0])  # how far the edge moves over the possible depths
    assert 0.4 * span < fix.sigma < 0.6 * span + 0.003 and fix.views == 1


# --- recipes, with a colour stub in place of SAM --------------------------------------------------------------------

def _measure(kind, rect, views, segment=colour_stub):
    return anchor.measure(kind, FRAME, rect, views, segment, BOUNDS, floor_sigma=0.01)


def test_door_is_measured_by_its_leaf_from_the_floor(door_views):
    out = _measure("door", (0.42, 1.38, 0.0, 2.15), door_views)
    m = out["measures"]
    assert out["status"] == "anchored" and len(out["views"]) == 3
    assert abs(m["width"]["value"] - (LEAF_S[1] - LEAF_S[0])) < 0.003
    assert abs(m["center_s"]["value"] - sum(LEAF_S) / 2) < 0.002
    assert abs(m["height"]["value"] - LEAF_TOP) < 0.004 and m["sill"]["value"] == 0.0
    assert abs(out["edges"]["left"]["delta"] - LEAF_DELTA) < 0.01 and abs(out["edges"]["top"]["delta"] - LEAF_DELTA) < 0.015
    assert m["height"]["sigma"] >= 0.01  # the floor's own sigma


def test_double_leaf_is_the_union_of_both_leaves():
    views = _views(_door_patches(leaves=((0.55, 0.895), (0.905, 1.25))))
    out = _measure("door", (0.42, 1.38, 0.0, 2.15), views)
    assert out["status"] == "anchored" and abs(out["measures"]["width"]["value"] - 0.70) < 0.003


def test_floor_niche_runs_jamb_to_jamb_and_down_to_the_floor_past_its_plinth():
    back = [FacePatch((0.3, 1.4), (0.12, 2.2), -0.10, 200), FacePatch((0.3, 1.4), (-1, 0.12), -0.09, 110)]
    views = _views(_wall((0.50, 1.20), (0.0, 1.90), back), angles=(-25, 0, 20), target_s=0.85)
    out = _measure("niche", (0.47, 1.30, 0.06, 1.95), views)
    m = out["measures"]
    assert out["status"] == "anchored"
    assert abs(m["width"]["value"] - 0.70) < 0.003 and abs(m["height"]["value"] - 1.90) < 0.004 and m["sill"]["value"] == 0.0


def test_window_whose_mask_fails_gives_candidate_edges_only():
    sash = [FacePatch((0.6, 1.6), (0.9, 2.1), -0.08, 235)]
    views = _views(_wall((0.60, 1.60), (0.9, 2.1), sash), angles=(-20, 0, 20), target_s=1.1)
    out = _measure("window", (0.58, 1.65, 0.88, 2.12), views, segment=nothing_stub)
    assert out["status"] == "candidates" and "measures" not in out
    for side, truth in (("left", 0.60), ("right", 1.60), ("bottom", 0.9), ("top", 2.1)):
        assert min(abs(c["pos"] - truth) for c in out["candidates"][side]) < 0.003, side


# --- the stage on a synthetic capture --------------------------------------------------------------------------------

def _stage(tmp_path, **options):
    ctx, _ = make_case(tmp_path, "cap", photos=10, photo_scale=3, **options)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    return anchor.build(ctx, colour_stub)


def test_stage_anchors_the_closed_door_of_a_synthetic_capture(tmp_path):
    doc = _stage(tmp_path)
    door = doc["elements"]["Door_1"]
    assert doc["schema"] == anchor.SCHEMA and door["status"] == "anchored" and len(door["views"]) >= 2
    assert abs(door["measures"]["width"]["value"] - DOOR["w"]) < 0.015
    assert abs(door["measures"]["height"]["value"] - DOOR["h"]) < 0.015
    assert -0.09 < door["edges"]["left"]["delta"] < -0.01  # the synthetic leaf is 5 cm behind the wall face


def test_stage_flags_an_open_door_without_measuring_it(tmp_path):
    door = _stage(tmp_path, door_open=True)["elements"]["Door_1"]
    assert door["status"] == "open" and "measures" not in door


def test_stage_writes_anchors_json(tmp_path):
    ctx, _ = make_case(tmp_path, "cap", photos=10, photo_scale=3)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    path = anchor.run(ctx, colour_stub)
    assert path == ctx.out / "anchors.json" and json.loads(path.read_text(encoding="utf-8"))["elements"]["Door_1"]["status"] == "anchored"


def _quads(rects):
    """Mesh (vertices, faces) of face rectangles (s0, s1, y0, y1, delta) in FRAME."""
    verts, faces = [], []
    for s0, s1, y0, y1, d in rects:
        k = len(verts)
        verts += list(FRAME.point(np.array([s0, s1, s1, s0]), np.array([y0, y0, y1, y1]), d))
        faces += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
    return np.array(verts), np.array(faces)


@pytest.mark.parametrize("leaf, closed", [((0.5, 1.3, 0.0, 2.1, -0.04), True),  # in the doorway
                                          (None, False),  # nothing behind: the next room is not scanned
                                          ((0.5, 1.3, 0.0, 2.1, 0.6), False)])  # something in front: the leaf swung in
def test_a_door_is_closed_only_when_the_mesh_in_its_doorway_lies_within_the_wall(leaf, closed):
    wall = [(-2, 0.5, 0, 3, 0.0), (1.3, 3, 0, 3, 0.0), (0.5, 1.3, 2.1, 3, 0.0)]
    maps = DepthMaps(*_quads(wall + ([leaf] if leaf else [])))
    _, photo = render_face(FRAME, [FacePatch((-2, 3), (-1, 4), 0.0, 100)], (0.9, 1.3, 2.2), (0.9, 1.05, 0.0))
    assert anchor.leaf_closed(maps, photo, FRAME, (0.5, 1.3, 0.0, 2.1), thickness=0.2) is closed
