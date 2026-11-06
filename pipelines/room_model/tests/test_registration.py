import numpy as np
from scipy.spatial.transform import Rotation

from room_model.capture import Photo
from room_model.registration import keep_weak_at_arkit, select_pairs, umeyama


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
