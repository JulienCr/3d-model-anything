import json

import numpy as np
import pytest
import shapely
from PIL import Image

from room_model import architecture, model, openings, raster, shell, textures
from room_model.context import NoPhotos, output_path
from room_model.photo_geom import PosedPhoto, RoomBox
from helpers import make_case
from synthetic import _look_at

SIZE, FOCAL = (640, 480), 500.0
K = np.array([[FOCAL, 0, SIZE[0] / 2], [0, FOCAL, SIZE[1] / 2], [0, 0, 1.0]])
WALL_RED, BOX_BLUE, SKY = (200, 30, 30), (30, 30, 200), (135, 206, 235)
BOX = {"x": (1.6, 2.4), "z": (0.6, 1.4), "y": (0.0, 1.0)}
EYES = [(1.0, 1.4, 3.0), (2.0, 1.4, 3.2), (3.0, 1.4, 3.0), (2.0, 1.7, 2.4)]


def _quad(a, b, c, d):
    return np.array([a, b, c, d], float), np.array([[0, 1, 2], [0, 2, 3]])


def _wall_mesh():
    """The red wall z = 0, x in [0, 4], y in [0, 2.5], facing +z."""
    return _quad((0, 0, 0), (4, 0, 0), (4, 2.5, 0), (0, 2.5, 0))


def _box_mesh():
    (x0, x1), (z0, z1), (y0, y1) = BOX["x"], BOX["z"], BOX["y"]
    faces = [_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)), _quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)),
             _quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)), _quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))]
    return np.concatenate([v for v, _ in faces]), np.concatenate([f + 4 * k for k, (_, f) in enumerate(faces)])


def _room_box():
    centre = np.array([sum(BOX[k]) / 2 for k in ("x", "y", "z")])
    return RoomBox(centre, np.eye(3), np.array([(BOX[k][1] - BOX[k][0]) / 2 for k in ("x", "y", "z")]))


def _photos(tmp_path, wall_colour=WALL_RED, gains=(1.0,) * len(EYES), with_box=True):
    """Posed photos of the wall (and the blue box in front of it) rendered at one exposure gain each."""
    parts = [(*_wall_mesh(), wall_colour)] + ([(*_box_mesh(), BOX_BLUE)] if with_box else [])
    verts = np.concatenate([v for v, _, _ in parts])
    offsets = np.cumsum([0] + [len(v) for v, _, _ in parts[:-1]])
    faces = np.concatenate([f + o for (_, f, _), o in zip(parts, offsets)])
    colours = np.concatenate([np.tile(c, (len(f), 1)) for _, f, c in parts]).astype(float)
    out = []
    for k, (eye, gain) in enumerate(zip(EYES, gains)):
        c2w = _look_at(eye, (2.0, 1.2, 0.0), 0.0)
        _, ids = raster.rasterize(verts, faces, K, np.linalg.inv(c2w), SIZE)
        image = np.where((ids >= 0)[..., None], colours[np.maximum(ids, 0)] * gain, np.array(SKY, float))
        path = tmp_path / f"{k}.png"
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)
        out.append(PosedPhoto(path.name, path, K, c2w, SIZE, 0))
    return out


def _wall_surface():
    return textures.Surface("Wall_M1", "A", "wall", np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), (200, 125))


def _texture(tmp_path, mesh="wall+box", boxes=(), fill=None, **photo_args):
    surface = _wall_surface()
    parts = {"wall": [_wall_mesh()], "wall+box": [_wall_mesh(), _box_mesh()]}[mesh]
    verts = np.concatenate([v for v, _ in parts])
    faces = np.concatenate([f + sum(len(v) for v, _ in parts[:k]) for k, (_, f) in enumerate(parts)])
    photos = _photos(tmp_path, **photo_args)
    observed = textures.observe([surface], photos, (verts, faces), list(boxes))
    return surface, textures.paint(surface, observed[surface.key], fill or textures.Fill.flat())


def _tile(tmp_path):
    noise = np.random.default_rng(0).integers(60, 200, (64, 64, 1)).repeat(3, axis=2).astype(np.uint8)
    path = tmp_path / "tile.png"
    Image.fromarray(noise).save(path)
    return textures.Fill.from_image(path)


def _bluish(rgb):
    return (rgb[..., 2].astype(int) > rgb[..., 0].astype(int) + 60).mean()


def test_projection_colour(tmp_path):
    _, painting = _texture(tmp_path, mesh="wall", with_box=False)
    assert painting.coverage > 0.9
    median = np.median(painting.rgb.reshape(-1, 3), axis=0)
    assert np.abs(median - WALL_RED).max() <= 5


@pytest.mark.parametrize("lidar_has_box, roomplan_box", [(True, False), (False, True), (True, True)])
def test_furniture_masked(tmp_path, lidar_has_box, roomplan_box):
    _, painting = _texture(tmp_path, mesh="wall+box" if lidar_has_box else "wall", boxes=[_room_box()] if roomplan_box else [])
    assert _bluish(painting.rgb) == 0
    assert 0.5 < painting.coverage < 0.97
    shadow = painting.rgb[10:25, 90:110]  # wall behind the box, near the floor: nothing sees it
    assert np.abs(np.median(shadow.reshape(-1, 3), axis=0) - WALL_RED).max() <= 5


def test_unmasked_furniture_leaks_into_the_wall(tmp_path):
    _, painting = _texture(tmp_path, mesh="wall")
    assert _bluish(painting.rgb) > 0.02


def test_holes_take_the_tinted_fill_with_a_feathered_seam(tmp_path):
    _, painting = _texture(tmp_path, fill=_tile(tmp_path))
    hole = painting.rgb[10:25, 90:110].reshape(-1, 3).astype(float)
    assert hole.std(axis=0).max() > 3 and np.abs(hole.mean(axis=0) - WALL_RED).max() <= 20
    # grey tile, tinted red: the red channel dominates the fill as it does the wall
    assert (hole[:, 0] > hole[:, 1] + 40).all()
    assert painting.fill == "tile.png"


def test_exposure_is_matched_between_photos(tmp_path):
    _, painting = _texture(tmp_path, mesh="wall", with_box=False, wall_colour=(120, 70, 60), gains=(0.8, 1.0, 1.25, 1.0))
    seen = painting.rgb[10:110, 20:180].reshape(-1, 3).astype(float)
    assert seen.std(axis=0).max() < 4


def test_coverage_is_the_share_of_texels_with_a_photo(tmp_path):
    surface, painting = _texture(tmp_path, mesh="wall", with_box=False)
    # the wall is 4 m wide and the cameras see it all but a margin at the far ends
    assert 0.9 < painting.coverage <= 1.0
    far = _wall_surface()
    far.origin = np.array([0.0, 0.0, -30.0])  # nothing sees this plane: no photo, no coverage
    observed = textures.observe([far], _photos(tmp_path), (*_wall_mesh(),), [])
    assert textures.paint(far, observed[far.key], textures.Fill.flat()) is None


def test_no_fill_available_gives_the_flat_median_colour(tmp_path):
    _, painting = _texture(tmp_path, fill=textures.Fill.flat("download failed"))
    assert painting.fill == "flat" and painting.note == "download failed"
    hole = painting.rgb[10:25, 90:110].reshape(-1, 3)
    assert np.ptp(hole, axis=0).max() <= 1


def test_fill_is_fetched_once_and_cached(tmp_path, monkeypatch):
    calls = tmp_path / "calls.txt"
    script = tmp_path / "fetch.py"
    script.write_text(
        "import sys\nfrom PIL import Image\n"
        f"open({str(calls)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "out = sys.argv[2]\nname = sys.argv[3].split('=')[0]\n"
        "Image.new('RGB', (8, 8), (90, 90, 90)).save(f'{out}/{name}_diff.jpg')\n")
    monkeypatch.setenv("ROOM_MODEL_FETCH_TEXTURES", str(script))
    monkeypatch.delenv("ROOM_MODEL_FILL", raising=False)
    arch = {"materials": {"floor": "ambientcg:Wood001"}}
    directory = tmp_path / "textures" / "fill"
    first = textures.Fill.load(arch, directory)
    second = textures.Fill.load(arch, directory)
    lines = calls.read_text().splitlines()
    assert len(lines) == 2  # white_plaster_02 (walls and ceilings share it) and the floor override, once each
    assert any("polyhaven:white_plaster_02" in l for l in lines) and any("ambientcg:Wood001" in l for l in lines)
    assert first.sources == second.sources == {"wall": "polyhaven:white_plaster_02", "ceiling": "polyhaven:white_plaster_02",
                                               "floor": "ambientcg:Wood001"}


def test_failed_download_falls_back_to_flat_and_says_so(tmp_path, monkeypatch):
    script = tmp_path / "fetch.py"
    script.write_text("import sys\nsys.exit('no network')\n")
    monkeypatch.setenv("ROOM_MODEL_FETCH_TEXTURES", str(script))
    monkeypatch.delenv("ROOM_MODEL_FILL", raising=False)
    fill = textures.Fill.load({}, tmp_path / "fill")
    assert set(fill.sources.values()) == {"flat"} and "no network" in fill.note


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    ctx, truth = make_case(tmp_path_factory.mktemp("textures"), "tex", partition=True, window=True, photos=10)
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    mp = pytest.MonkeyPatch()
    mp.setenv("ROOM_MODEL_FILL", "none")
    try:
        path = textures.run(ctx)
    finally:
        mp.undo()
    return ctx, json.loads((ctx.out / "architecture.json").read_text(encoding="utf-8")), json.loads(path.read_text(encoding="utf-8"))


def test_stage_writes_one_texture_per_surface_face(stage):
    ctx, arch, doc = stage
    assert output_path(ctx, "textures") == ctx.out / "textures" / "textures.json"
    walls = [e["id"] for e in arch["elements"] if e["kind"] == "wall"]
    assert len(set(walls) & set(doc)) >= 4 and {"Floor", "Ceiling_1", "Partition_P1"} <= set(doc)  # ten photos do not see every wall
    assert "A" in doc["Partition_P1"] and set(doc) <= {e["id"] for e in arch["elements"]}
    for element, faces in doc.items():
        for face, entry in faces.items():
            image = Image.open(ctx.out / entry["image"])
            assert entry["image"] == f"textures/{element}_{face}.png" and list(image.size) == entry["size"]
            assert entry["texel"] == 0.02 and 0.0 < entry["coverage"] <= 1.0
            assert len(entry["origin"]) == len(entry["u_axis"]) == len(entry["v_axis"]) == 3
            assert entry["fill"] == "flat" and entry["fill_note"]


def test_floor_texture_covers_only_the_outline(stage):
    _, arch, _ = stage
    floor = next(s for s in textures.surfaces_of(arch) if s.kind == "floor")
    area = shapely.Polygon(next(e for e in arch["elements"] if e["kind"] == "floor")["polygon"]).area
    nu, nv = floor.size
    assert floor.mask.mean() == pytest.approx(area / (nu * nv * textures.TEXEL ** 2), abs=0.02) and floor.mask.mean() < 0.9, "an L-shaped room"


def test_stage_axes_are_unit_and_orthogonal_to_the_face(stage):
    _, arch, doc = stage
    for faces in doc.values():
        for entry in faces.values():
            u, v = np.array(entry["u_axis"]), np.array(entry["v_axis"])
            assert np.linalg.norm(u) == pytest.approx(1) and np.linalg.norm(v) == pytest.approx(1) and abs(u @ v) < 1e-9
    wall = arch["elements"][next(i for i, e in enumerate(arch["elements"]) if e["kind"] == "wall")]
    entry = doc[wall["id"]]["A"]
    assert entry["v_axis"] == [0.0, 1.0, 0.0] and entry["origin"][1] == pytest.approx(arch["parameters"]["floor.level"]["value"])


def test_furniture_against_the_west_wall_is_not_painted_on_it(stage):
    ctx, arch, doc = stage
    coverages = {e: f["A"]["coverage"] for e, f in doc.items() if e.startswith("Wall_")}
    assert min(coverages.values()) < 1.0  # the table and the cabinet hide part of the walls they stand against


def test_stage_without_photos_raises_no_photos(tmp_path):
    ctx, _ = make_case(tmp_path, "bare")
    shell.run(ctx)
    model.run(ctx)
    openings.run(ctx)
    architecture.run(ctx)
    with pytest.raises(NoPhotos):
        textures.run(ctx)
