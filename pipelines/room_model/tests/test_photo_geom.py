import numpy as np
import pytest
from PIL import Image

from room_model import photo_geom, raster
from room_model.frame import world_to_room
from room_model.photo_geom import DepthMaps, PosedPhoto
from helpers import make_case, truth_to_room


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_upright_roundtrip(tmp_path, turns):
    rng = np.random.default_rng(turns)
    w, h = 40, 30
    pixels = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "t.png")
    photo = PosedPhoto("t.png", tmp_path / "t.png", np.eye(3), np.eye(4), (w, h), turns)

    up = photo_geom.upright(photo)
    assert up.size == photo_geom.upright_size(photo)
    uv = rng.uniform(0, [w, h], (50, 2))
    assert np.allclose(photo_geom.to_sensor(photo, photo_geom.to_upright(photo, uv)), uv)

    cells = rng.integers(0, [w, h], (50, 2))
    for (i, j), (x, y) in zip(cells, np.floor(photo_geom.to_upright(photo, cells + 0.5)).astype(int)):
        assert up.getpixel((int(x), int(y))) == tuple(pixels[j, i])


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    ctx, truth = make_case(tmp_path_factory.mktemp("photo-geom"), "cap", photos=8)
    return ctx, truth, photo_geom.load_posed_photos(ctx)


def test_synthetic_photos_are_stored_sideways_and_turned_upright(case):
    _, _, photos = case
    assert len(photos) == 8 and {p.up_turns for p in photos} == {1, 3}
    for p in photos:
        assert p.size == (320, 240) and photo_geom.upright_size(p) == (240, 320)
        x = p.centre + 2.0 * p.c2w_room[:3, 2]
        (u0, v0), (u1, v1) = photo_geom.to_upright(p, photo_geom.project(p, np.array([x + [0, 1.0, 0], x]))[0])
        assert v0 < v1 and abs(u0 - u1) < 0.2 * abs(v0 - v1), p.name  # room +Y is vertical and points up
        arr = np.asarray(photo_geom.upright(p), float).mean(axis=2)
        assert arr[: arr.shape[0] // 3].mean() > arr[-(arr.shape[0] // 3):].mean(), p.name  # bright ceiling above, floor below


def test_backproject_hits_wall(case):
    ctx, truth, photos = case
    T = truth_to_room(ctx, truth)
    corner = (T @ [6.0, 1.4, 0.0, 1.0])[:3]  # east/south corner at mid-height
    normal = T[:3, :3] @ [1.0, 0.0, 0.0]  # east wall, local +x
    uv, depth = photo_geom.project(photos[0], corner)
    assert depth[0] > 1.0 and 0 < uv[0, 0] < 320 and 0 < uv[0, 1] < 240
    hit = photo_geom.backproject(photos[0], uv, (normal, float(normal @ corner)))
    assert np.linalg.norm(hit[0] - corner) < 0.01


def test_backproject_refuses_rays_that_never_meet_the_plane(case):
    ctx, truth, photos = case
    photo = photos[0]
    centre = photo.c2w_room[:3, 3]
    forward = photo.c2w_room[:3, 2]
    plane_ahead = (forward, float(forward @ centre) + 3.0)
    assert photo_geom.backproject(photo, np.array([[photo.K[0, 2], photo.K[1, 2]]]), plane_ahead) is not None
    parallel = (photo.c2w_room[:3, 0], float(photo.c2w_room[:3, 0] @ centre) + 1.0)  # contains the optical axis direction
    assert photo_geom.backproject(photo, np.array([[photo.K[0, 2], photo.K[1, 2]]]), parallel) is None
    behind = (forward, float(forward @ centre) - 3.0)  # the plane lies behind the camera
    assert photo_geom.backproject(photo, np.array([[photo.K[0, 2], photo.K[1, 2]]]), behind) is None
    grazing = (photo.c2w_room[:3, 0], float(photo.c2w_room[:3, 0] @ centre) + 1.0)  # one ray meets it, the other is parallel
    assert photo_geom.backproject(photo, np.array([[photo.K[0, 2] + 100.0, photo.K[1, 2]], [photo.K[0, 2], photo.K[1, 2]]]), grazing) is None
    hit = photo_geom.backproject(photo, np.array([[photo.K[0, 2], photo.K[1, 2]]]), plane_ahead)
    assert np.isfinite(hit).all() and hit[0] @ forward == pytest.approx(forward @ centre + 3.0)


def test_depth_map_agrees_with_projection_of_the_lidar_mesh(case):
    ctx, truth, photos = case
    T = truth_to_room(ctx, truth)
    maps = DepthMaps(*photo_geom.load_room_mesh(ctx, world_to_room(ctx.capture, ctx.room, ctx.turn)))
    pts = np.array([(T @ [6.0, y, z, 1.0])[:3] for y, z in ((1.0, 0.5), (1.6, 2.0), (0.6, 2.2))])  # east wall
    uv, depth = photo_geom.project(photos[0], pts)
    zbuf, scale = maps.get(photos[0])
    cols, rows = (uv[:, 0] * scale).astype(int), (uv[:, 1] * scale).astype(int)
    assert np.allclose(zbuf[rows, cols], depth, atol=0.03)
    assert maps.visible(photos[0], pts).all()


def _plane(z, half_x, half_y, n=10):
    xs, ys = np.linspace(-half_x, half_x, n), np.linspace(-half_y, half_y, n)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, float(z))])
    idx = np.arange(gx.size).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    return verts, np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])


def _pinhole(name, z, tmp_path):
    K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[2, 3] = z
    return PosedPhoto(name, tmp_path / name, K, c2w, (160, 120), 0)


def _wall_and_box():
    wall_v, wall_f = _plane(5.0, 4.0, 3.0)
    box_v, box_f = _plane(3.0, 0.5, 0.5)
    return np.vstack([wall_v, box_v]), np.vstack([wall_f, box_f + len(wall_v)])


def test_depth_map_occlusion(tmp_path):
    verts, faces = _wall_and_box()
    photo = _pinhole("a", 0.0, tmp_path)
    depth = raster.depth_map(verts, faces, photo.K, photo.w2c, photo.size)
    assert depth[60, 80] == pytest.approx(3.0, abs=1e-6) and depth[60, 140] == pytest.approx(5.0, abs=1e-6)
    assert np.isinf(depth[5, 5]) or depth[5, 5] == pytest.approx(5.0, abs=1e-6)

    visible = DepthMaps(verts, faces).visible(photo, np.array([[0, 0, 5.0], [2.0, 0, 5.0], [0, 0, 3.0], [0, 0, 2.5]]))
    assert visible.tolist() == [False, True, True, True]  # the wall point behind the box fails the 3 cm test


def test_best_views_prefer_the_closer_photo_and_skip_those_that_cannot_see(tmp_path):
    verts, faces = _wall_and_box()
    near, far = _pinhole("near", 1.0, tmp_path), _pinhole("far", -2.0, tmp_path)
    behind_wall = _pinhole("behind", 8.0, tmp_path)  # same direction, but the wall is behind it
    span = np.array([[1.5, 0, 5.0], [2.5, 0, 5.0]])
    assert [p.name for p in photo_geom.best_views([far, behind_wall, near], span, (verts, faces))] == ["near", "far"]
    hidden = np.array([[-0.2, 0, 5.0], [0.2, 0, 5.0]])  # behind the box
    assert photo_geom.best_views([near, far], hidden, (verts, faces)) == []
    thin = np.array([[0.0, 0, 3.05], [0.1, 0, 3.05]])  # 5 cm behind the box: a frame or skirting in front of an end
    assert photo_geom.best_views([near], thin, (verts, faces)) == []
    assert [p.name for p in photo_geom.best_views([near], thin, (verts, faces), tolerance=0.10)] == ["near"]


def test_best_views_rank_box_occluded_views_last(tmp_path):
    wall = _plane(5.0, 4.0, 3.0)[0:2]  # the mesh has no trace of the furniture
    near, far = _pinhole("near", 1.0, tmp_path), _pinhole("far", -2.0, tmp_path)
    span = np.array([[1.5, 0, 5.0], [2.5, 0, 5.0]])
    chair = photo_geom.RoomBox(np.array([0.7, 0, 3.0]), np.eye(3), np.array([0.2, 0.2, 0.2]))
    assert [p.name for p in photo_geom.best_views([near, far], span, wall)] == ["near", "far"]
    assert [p.name for p in photo_geom.best_views([near, far], span, wall, boxes=[chair])] == ["far", "near"]
    assert [p.name for p in photo_geom.best_views([near], span, wall, boxes=[chair])] == ["near"]  # still usable alone
    assert 0 < photo_geom.occluded_fraction(near.centre, span, [chair]) < 1 and photo_geom.occluded_fraction(far.centre, span, [chair]) == 0
    around_span = photo_geom.RoomBox(np.array([2.0, 0, 5.0]), np.eye(3), np.array([1.0, 1.0, 0.1]))
    assert photo_geom.occluded_fraction(near.centre, span, [around_span]) == 0  # a box holding the span is no obstacle
    around_camera = photo_geom.RoomBox(near.centre, np.eye(3), np.array([1.0, 1.0, 1.0]))
    assert photo_geom.occluded_fraction(near.centre, span, [around_camera]) == 0  # nor one the photographer stands in
    turned = photo_geom.room_boxes({"objects": [{"transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.7, 0, 3.0, 1],
                                                  "dimensions": [0.4, 0.4, 0.4]}]}, np.eye(4))
    assert [p.name for p in photo_geom.best_views([near, far], span, wall, boxes=turned)] == ["far", "near"]


# --- eligibility of a planar rectangle ---------------------------------------------------------------------------

def _camera(name, position, target, tmp_path, f=1000.0, size=(640, 480)):
    """Pinhole looking from `position` to `target` (room +Y up), 640x480 with a long focal so rectangles stay in frame."""
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, -np.cross(right, forward), forward, position
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    return PosedPhoto(name, tmp_path / name, K, c2w, size, 0)


RECT = np.array([[-0.5, -0.4, 5.0], [0.5, -0.4, 5.0], [0.5, 0.4, 5.0], [-0.5, 0.4, 5.0]])  # on the wall z = 5, facing -z
TOWARDS_VIEWER = np.array([0.0, 0.0, -1.0])
CENTRE = np.array([0.0, 0.0, 5.0])


def _seen_from(angle_deg, distance, tmp_path, name="cam"):
    a = np.radians(angle_deg)
    return _camera(name, CENTRE + distance * np.array([np.sin(a), 0.0, -np.cos(a)]), CENTRE, tmp_path)


def _names(views):
    return [v.photo.name for v in views]


def test_rect_views_accept_a_frontal_unoccluded_view(tmp_path):
    wall = _plane(5.0, 4.0, 3.0)
    views = photo_geom.rect_views([_seen_from(0, 3.0, tmp_path, "front"), _seen_from(50, 3.0, tmp_path, "side")], RECT, wall, TOWARDS_VIEWER)
    assert _names(views) == ["front", "side"]  # larger area x cos first
    assert views[0].cos == pytest.approx(1.0) and views[1].cos == pytest.approx(np.cos(np.radians(50)))
    assert views[0].seen == 1.0


def test_rect_views_drop_a_view_behind_an_occluding_wall(tmp_path):
    wall_v, wall_f = _plane(5.0, 4.0, 3.0)
    screen_v, screen_f = _plane(3.5, 1.5, 2.0)  # a wall between the camera and the rectangle, wider than it in the image
    mesh = (np.vstack([wall_v, screen_v]), np.vstack([wall_f, screen_f + len(wall_v)]))
    front = _seen_from(0, 3.0, tmp_path, "front")  # at z = 2: the rectangle projects inside the image, the screen hides it
    uv, depth = photo_geom.project(front, RECT)
    assert ((uv > 0).all() and (uv[:, 0] < 640).all() and (uv[:, 1] < 480).all() and (depth > 0).all())
    around = _camera("around", np.array([0.0, 0.0, 4.0]), CENTRE, tmp_path)  # past the screen, 1 m from the wall: too small
    assert photo_geom.rect_views([front], RECT, mesh, TOWARDS_VIEWER) == []
    assert _names(photo_geom.rect_views([front], RECT, (wall_v, wall_f), TOWARDS_VIEWER)) == ["front"]
    assert photo_geom.rect_views([around], RECT, mesh, TOWARDS_VIEWER) == []


@pytest.mark.parametrize("angle, eligible", [(55, True), (59, True), (62, False), (68, False)])
def test_rect_views_need_the_camera_to_face_the_rectangle(tmp_path, angle, eligible):
    camera = _seen_from(angle, 3.0, tmp_path)
    uv, _ = photo_geom.project(camera, RECT)
    assert min(np.linalg.norm(uv[1] - uv[0]), np.linalg.norm(uv[3] - uv[0])) >= 120  # wide enough: only the angle can reject it
    assert bool(photo_geom.rect_views([camera], RECT, _plane(5.0, 4.0, 3.0), TOWARDS_VIEWER)) is eligible


def test_rect_views_need_a_large_enough_projection(tmp_path):
    wall = _plane(5.0, 4.0, 3.0)
    assert photo_geom.rect_views([_seen_from(0, 6.0, tmp_path)], RECT, wall, TOWARDS_VIEWER)  # 133 px
    assert photo_geom.rect_views([_seen_from(0, 10.0, tmp_path)], RECT, wall, TOWARDS_VIEWER) == []  # 80 px on the shorter side


def _slab(x0, x1, y0, y1):
    """A box 30 cm in front of the wall hiding what the camera at z = 2 sees through x0..x1, y0..y1 on the wall (z = 5)."""
    k = 2.7 / 3.0  # the box sits at z = 4.7: rays from z = 2 cross it at 0.9 of their wall offset
    return photo_geom.RoomBox(np.array([k * (x0 + x1) / 2, k * (y0 + y1) / 2, 4.7]), np.eye(3), np.array([k * (x1 - x0) / 2 - 0.03, k * (y1 - y0) / 2 - 0.03, 0.1]))


def test_rect_views_accept_a_partly_hidden_frame_when_every_corner_is_seen(tmp_path):
    wall, camera = _plane(5.0, 4.0, 3.0), _seen_from(0, 3.0, tmp_path)
    top, bottom = _slab(-0.35, 0.35, 0.3, 1.0), _slab(-0.35, 0.35, -1.0, -0.3)  # the middle three points of the top and bottom edges
    band = photo_geom.rect_band(RECT)
    assert len(band) == 16 and photo_geom.occluded_mask(camera.centre, band, [top, bottom]).sum() == 6  # 62 % of the band seen
    kept = photo_geom.rect_views([camera], RECT, wall, TOWARDS_VIEWER, boxes=[top, bottom])
    assert _names(kept) == ["cam"] and kept[0].seen == pytest.approx(10 / 16)
    sides = [_slab(-0.6, -0.1, -1.0, 1.0), _slab(0.1, 0.6, -1.0, 1.0)]  # two thirds of the band hidden
    assert photo_geom.rect_views([camera], RECT, wall, TOWARDS_VIEWER, boxes=sides) == []


def test_rect_views_drop_a_view_with_one_corner_hidden(tmp_path):
    wall, camera = _plane(5.0, 4.0, 3.0), _seen_from(0, 3.0, tmp_path)
    corner = _slab(-0.6, -0.3, 0.2, 1.0)  # the top-left corner of the rectangle only
    assert photo_geom.occluded_mask(camera.centre, photo_geom.rect_corner_samples(RECT)[3], [corner]).sum() >= 4
    assert photo_geom.occluded_mask(camera.centre, photo_geom.rect_band(RECT), [corner]).mean() <= 0.25  # the frame band is mostly seen
    assert photo_geom.rect_views([camera], RECT, wall, TOWARDS_VIEWER, boxes=[corner]) == []
    assert photo_geom.rect_views([camera], RECT, wall, TOWARDS_VIEWER)  # without the box it is eligible


def test_rect_views_drop_a_view_with_a_corner_outside_the_image(tmp_path):
    wall = _plane(5.0, 4.0, 3.0)
    aside = _camera("aside", CENTRE + np.array([0.0, 0.0, -3.0]), CENTRE + np.array([0.6, 0.0, 0.0]), tmp_path)  # one edge leaves the image
    uv, _ = photo_geom.project(aside, RECT)
    assert (uv[:, 0] > 640).any() and (uv[:, 0] < 640).any() and (uv[:, 0] > 0).all()
    assert photo_geom.rect_views([aside], RECT, wall, TOWARDS_VIEWER) == []


def test_rect_views_tolerate_a_casing_standing_out_of_the_plane(tmp_path):
    camera = _seen_from(0, 3.0, tmp_path)
    assert photo_geom.rect_views([camera], RECT, _plane(4.95, 4.0, 3.0), TOWARDS_VIEWER)  # the surface 5 cm in front of the plane
    assert photo_geom.rect_views([camera], RECT, _plane(4.85, 4.0, 3.0), TOWARDS_VIEWER) == []  # 15 cm: something stands there


def test_rect_grid_stops_three_centimetres_inside_the_edges():
    grid = photo_geom.rect_grid(RECT)
    assert grid.shape == (25, 3) and np.allclose(grid[:, 2], 5.0)
    corners = photo_geom.rect_corner_samples(RECT)
    assert corners.shape == (4, 5, 3) and np.allclose(corners[0, 0], [-0.47, -0.37, 5.0]) and np.allclose(corners[2, 0], [0.47, 0.37, 5.0])
    assert np.abs(corners[:, 1:] - corners[:, :1]).max() == pytest.approx(photo_geom.CORNER_RADIUS)
    assert grid[:, 0].min() == pytest.approx(-0.47) and grid[:, 0].max() == pytest.approx(0.47)
    assert grid[:, 1].min() == pytest.approx(-0.37) and grid[:, 1].max() == pytest.approx(0.37)
    assert len(np.unique(grid[:, 0].round(6))) == 5 and len(np.unique(grid[:, 1].round(6))) == 5


def test_an_opening_is_eligible_from_the_side_of_either_face(tmp_path):
    wall = _plane(5.0, 4.0, 3.0)
    side_b = _camera("b", CENTRE + np.array([0.0, 0.0, 3.0]), CENTRE, tmp_path)  # the host's face A looks to -z; this camera is on face B
    assert photo_geom.rect_views([side_b], RECT, wall, TOWARDS_VIEWER, two_sided=True)  # a hole is seen through from either side
    assert photo_geom.rect_views([side_b], RECT, wall, TOWARDS_VIEWER) == []  # a recess that is not through is not


def test_rect_views_without_a_normal_accept_either_side_of_the_rectangle(tmp_path):
    wall = _plane(5.0, 4.0, 3.0)
    behind = _camera("behind", np.array([0.0, 0.0, 8.0]), CENTRE, tmp_path)
    assert photo_geom.rect_views([behind], RECT, wall, -TOWARDS_VIEWER)  # a normal pointing at the camera
    assert photo_geom.rect_views([behind], RECT, wall, TOWARDS_VIEWER) == []  # the viewing side is the other one
    assert photo_geom.rect_views([behind], RECT, wall, TOWARDS_VIEWER, two_sided=True)
    assert photo_geom.rect_views([behind], RECT, wall)  # normal taken from the rectangle itself


def test_best_views_drop_the_photos_that_do_not_face_the_span(tmp_path):
    wall = _plane(5.0, 4.0, 3.0)
    span = np.array([[-0.5, 0.0, 5.0], [0.5, 0.0, 5.0]])
    cameras = [_seen_from(a, 3.0, tmp_path, f"a{a}") for a in (0, 65, 75)]
    assert _names_of(photo_geom.best_views(cameras, span, wall, 3, TOWARDS_VIEWER)) == ["a0", "a65", "a75"]
    kept = photo_geom.best_views(cameras, span, wall, 3, TOWARDS_VIEWER, min_cos=photo_geom.SURVEY_FACING_COS)
    assert _names_of(kept) == ["a0", "a65"]  # 65 degrees stay, 75 do not


def _names_of(photos):
    return [p.name for p in photos]
