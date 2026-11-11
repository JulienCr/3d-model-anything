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
