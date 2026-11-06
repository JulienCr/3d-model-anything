import json

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from room_model import registration
from room_model.capture import Photo, colmajor
from room_model.context import StageContext
from room_model.registration import fallback_reason, keep_weak_at_arkit, photo_records, select_pairs, umeyama


def _photo(i, x, yaw_deg):
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    c2w[:3, 3] = [x, 1.5, 0.0]
    return Photo(f"{i:04d}.jpg", f"photos/{i:04d}.jpg", 4, 4, np.eye(3), c2w, float(i), 0)


def test_select_pairs():
    # a corridor walked forward, with every third camera looking the other way
    photos = [_photo(i, 0.7 * i, 180.0 if i % 3 == 0 else 0.0) for i in range(20)]
    max_dist, max_angle, neighbors = 3.0, 75.0, 2
    pairs = select_pairs(photos, max_dist, max_angle, neighbors)

    expected = set()
    for i in range(20):
        for j in range(i + 1, 20):
            d = abs(0.7 * (j - i))
            opposite = (i % 3 == 0) != (j % 3 == 0)
            if (d < max_dist and not opposite) or j - i <= neighbors:
                expected.add((photos[i].name, photos[j].name))
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == expected
    assert (photos[0].name, photos[15].name) not in set(pairs)  # far apart, opposite
    assert (photos[1].name, photos[6].name) not in set(pairs)  # far apart, same way


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(40, 3))
    R_true = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    t_true = np.array([0.3, -1.2, 2.0])
    dst = 1.01 * src @ R_true.T + t_true
    s, R, t = umeyama(src, dst)
    assert abs(s - 1.01) < 1e-9
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)


def test_weak_photos_keep_arkit_pose():
    arkit = {n: np.eye(4) * k for k, n in enumerate(["a", "b", "c"], start=1)}
    refined = {n: np.eye(4) * (10 + k) for k, n in enumerate(["a", "b", "c"], start=1)}
    out = keep_weak_at_arkit(refined, arkit, {"a": 49, "b": 50, "c": 900})
    assert np.array_equal(out["a"], arkit["a"])
    assert np.array_equal(out["b"], refined["b"])
    assert np.array_equal(out["c"], refined["c"])


def test_photo_records_roundtrip_column_major():
    arkit = np.eye(4)
    arkit[:3, :3] = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    arkit[:3, 3] = [1.0, 2.0, 3.0]
    recs = photo_records(["w.jpg"], {"w.jpg": arkit}, {"w.jpg": False}, {"w.jpg": 7},
                         {"w.jpg": 9.0}, {"w.jpg": 9.0})
    assert recs[0]["c2w_arkit"][12:15] == [1.0, 2.0, 3.0]
    assert np.array_equal(colmajor(recs[0]["c2w_arkit"], 4), arkit)
    assert recs[0]["correction_cm"] == 0.0 and recs[0]["n_obs"] == 7


def test_fallback_needs_three_well_observed_photos():
    ok = dict(n_registered=10, n_photos=10, scale=1.0, after_px=0.5, before_px=1.0, rms_cm=2.0)
    assert fallback_reason(2, **ok) == "too few well-observed photos"
    assert fallback_reason(0, **ok) == "too few well-observed photos"
    assert fallback_reason(3, **ok) is None


def _write_jpegs(capture, n=4):
    entries = []
    for i in range(n):
        rel = f"photos/{i:04d}.jpg"
        Image.new("RGB", (32, 24), (i * 20, 90, 120)).save(capture / rel, "JPEG")
        c2w = np.eye(4)
        c2w[0, 3] = 0.1 * i
        entries.append({"file": rel, "width": 32, "height": 24, "intrinsics": list(np.eye(3).T.flatten()),
                        "transform": list(c2w.T.flatten()), "timestamp": float(i), "room": 0})
    (capture / "photos" / "photos.json").write_text(json.dumps(entries), encoding="utf-8")


@pytest.mark.parametrize("keep_work", [False, True])
def test_run_cleans_work_dir_when_a_step_fails(build, tmp_path, monkeypatch, keep_work):
    import pycolmap

    root, _ = build()
    _write_jpegs(root)

    def boom(*args, **kwargs):
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(pycolmap, "extract_features", boom)
    ctx = StageContext(root, tmp_path / "out", keep_work=keep_work)
    with pytest.raises(RuntimeError, match="extraction failed"):
        registration.run(ctx)
    assert (tmp_path / "out" / "registration" / "work").exists() is keep_work


def test_run_rejects_a_room_without_photos(build, tmp_path):
    root, _ = build()
    with pytest.raises(ValueError, match="room 0"):
        registration.run(StageContext(root, tmp_path / "out"))
    assert not (tmp_path / "out" / "registration" / "work").exists()
