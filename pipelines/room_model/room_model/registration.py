"""Registration stage: ARKit photo poses refined by SIFT, triangulation and bundle adjustment.

Intrinsics and ARKit poses are priors; BA runs on well-observed photos only, the result is
aligned back to the ARKit frame with a Sim3, and weak photos keep their ARKit pose.
"""
from __future__ import annotations

import json
import shutil
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from room_model.capture import ARKIT_TO_CV, Photo, invert_rigid, load_photos
from room_model.context import NoPhotos, StageContext, output_path

MAX_DIST = 5.0
MAX_ANGLE = 75.0
NEIGHBORS = 5
PEAK_THRESHOLD = 0.003
MAX_FEATURES = 16384
MIN_OBS = 50
MIN_GOOD = 3
FILTER_PX = 2.0
SCHEMA = "room-model/registration-1"


def select_pairs(photos, max_dist, max_angle, neighbors):
    """Pairs (i < j) of nearby, similarly oriented photos, plus sequence neighbours."""
    C = np.array([p.c2w_cv[:3, 3] for p in photos])
    Z = np.array([p.c2w_cv[:3, 2] for p in photos])
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    A = np.degrees(np.arccos(np.clip(Z @ Z.T, -1, 1)))
    return [
        (photos[i].name, photos[j].name)
        for i in range(len(photos))
        for j in range(i + 1, len(photos))
        if (D[i, j] < max_dist and A[i, j] < max_angle) or j - i <= neighbors
    ]


def umeyama(src, dst):
    """Least-squares similarity with dst ~ scale * R @ src + t; returns (scale, R, t)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / xs.var(0).sum())
    return scale, R, mu_d - scale * R @ mu_s


def keep_weak_at_arkit(refined, arkit, n_obs, min_obs=MIN_OBS):
    """name -> pose: the refined pose when n_obs >= min_obs, else the ARKit pose."""
    return {n: (refined[n] if n_obs[n] >= min_obs else arkit[n]) for n in refined}


def fallback_reason(n_good, n_registered, n_photos, scale, after_px, before_px, rms_cm):
    """Why the refinement must be discarded (ARKit poses kept), or None when it is accepted."""
    if n_good < MIN_GOOD:
        return "too few well-observed photos"
    if n_registered < n_photos:
        return "not all images registered"
    if not 0.95 < scale < 1.05:
        return f"Sim3 scale {scale:.4f} outside [0.95, 1.05]"
    if after_px > before_px:
        return "BA increased the mean reprojection error"
    if rms_cm > 25:
        return f"camera centre RMS after alignment {rms_cm:.1f} cm"
    return None


def photo_records(names, poses, is_refined, n_obs, dt_cm, dr_deg):
    """poses.json `photos` entries; c2w_arkit is the 4x4 pose flattened column-major."""
    return [
        {
            "name": n,
            "c2w_arkit": [float(v) for v in poses[n].T.flatten()],
            "refined": is_refined[n],
            "n_obs": int(n_obs[n]),
            "correction_cm": dt_cm[n] if is_refined[n] else 0.0,
            "correction_deg": dr_deg[n] if is_refined[n] else 0.0,
        }
        for n in names
    ]


def _strip_exif(data: bytes) -> bytes:
    """Drop every APP1 segment (EXIF/XMP, incl. Orientation) without touching the scan data."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file")
    out, i = [data[:2]], 2
    while True:
        marker = data[i + 1]
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker == 0xDA:
            out.append(data[i:])
            break
        if marker != 0xE1:
            out.append(data[i : i + 2 + length])
        i += 2 + length
    return b"".join(out)


def _copy_images(capture: Path, photos: list[Photo], images: Path) -> None:
    images.mkdir(parents=True, exist_ok=True)
    for p in photos:
        (images / p.name).write_bytes(_strip_exif((capture / p.rel_file).read_bytes()))
        with Image.open(images / p.name) as im:
            if im.size != (p.width, p.height):
                raise ValueError(f"{p.name}: image is {im.size}, the capture says {(p.width, p.height)}")


def _extract_and_write_prior(pycolmap, work: Path, photos: list[Photo]) -> Path:
    """SIFT on every image, then write the ARKit prior model using the database ids.

    Extraction is multithreaded, so database ids do not follow name order: key by name.
    """
    by_name = {p.name: p for p in photos}
    db_path = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.peak_threshold = PEAK_THRESHOLD
    ext.sift.max_num_features = MAX_FEATURES
    pycolmap.extract_features(db_path, work / "images", image_names=[p.name for p in photos],
                              camera_mode=pycolmap.CameraMode.PER_IMAGE, reader_options=reader,
                              extraction_options=ext)
    cams, imgs = [], []
    with pycolmap.Database.open(str(db_path)) as db:
        for dbim in sorted(db.read_all_images(), key=lambda im: im.image_id):
            p = by_name[dbim.name]
            K = p.K
            params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
            dbcam = db.read_camera(dbim.camera_id)
            dbcam.model, dbcam.width, dbcam.height = "PINHOLE", p.width, p.height
            dbcam.params = np.array(params)
            dbcam.has_prior_focal_length = True
            db.update_camera(dbcam)
            m = p.w2c_cv
            x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
            t = m[:3, 3]
            cams.append(f"{dbim.camera_id} PINHOLE {p.width} {p.height} " + " ".join(f"{v:.9f}" for v in params))
            # the trailing newline leaves the empty POINTS2D line that COLMAP's images.txt format requires
            imgs.append(f"{dbim.image_id} {w:.12f} {x:.12f} {y:.12f} {z:.12f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                        f"{dbim.camera_id} {dbim.name}\n")
    prior = work / "prior"
    prior.mkdir(exist_ok=True)
    (prior / "cameras.txt").write_text("\n".join(cams) + "\n", encoding="utf-8", newline="\n")
    (prior / "images.txt").write_text("\n".join(imgs) + "\n", encoding="utf-8", newline="\n")
    (prior / "points3D.txt").write_text("", encoding="utf-8", newline="\n")
    return prior


def _match(pycolmap, work: Path, photos: list[Photo]) -> int:
    pairs = select_pairs(photos, MAX_DIST, MAX_ANGLE, NEIGHBORS)
    pairs_file = work / "pairs.txt"
    pairs_file.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n", encoding="utf-8", newline="\n")
    opts = pycolmap.ImportedPairingOptions()
    opts.match_list_path = str(pairs_file)
    pycolmap.match_image_pairs(work / "database.db", pairing_options=opts)
    return len(pairs)


def _mean_reprojection(rec) -> float:
    return float(rec.compute_mean_reprojection_error())


def _obs_per_image(rec) -> dict[str, int]:
    return {im.name: sum(1 for p in im.points2D if p.has_point3D()) for im in rec.images.values()}


def _poses(rec, names):
    """name -> (camera centre, w2c rotation) in the model frame."""
    out = {}
    for n in names:
        m = rec.find_image_with_name(n).cam_from_world().matrix()
        out[n] = (-m[:3, :3].T @ m[:3, 3], m[:3, :3])
    return out


def _bundle_adjust(pycolmap, rec, good: list[str]) -> None:
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = False
    opts.refine_principal_point = False
    opts.refine_extra_params = False
    opts.ceres.solver_options.max_num_iterations = 1000
    for _ in range(2):
        config = pycolmap.BundleAdjustmentConfig()
        for n in good:
            config.add_image(rec.find_image_with_name(n).image_id)
        config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        pycolmap.create_default_bundle_adjuster(opts, config, rec).solve()
        rec.update_point_3d_errors()
        pycolmap.ObservationManager(rec).filter_all_points3D(FILTER_PX, 1.5)


def _rot_angle_deg(Ra, Rb) -> float:
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def run(ctx: StageContext) -> Path:
    import pycolmap

    t0 = time.time()
    photos = load_photos(ctx.capture, ctx.room)
    if not photos:
        raise NoPhotos(f"no photos for room {ctx.room} in {ctx.capture}")
    names = [p.name for p in photos]
    reg = ctx.out / "registration"
    work, sparse = reg / "work", reg / "sparse"
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(sparse, ignore_errors=True)
    work.mkdir(parents=True)
    sparse.mkdir(parents=True)
    try:
        _copy_images(ctx.capture, photos, work / "images")
        prior_dir = _extract_and_write_prior(pycolmap, work, photos)
        print(f"features done {time.time() - t0:.0f}s")
        n_pairs = _match(pycolmap, work, photos)
        print(f"matching done {time.time() - t0:.0f}s ({n_pairs} pairs)")

        prior = pycolmap.Reconstruction(str(prior_dir))
        (work / "tri").mkdir()
        pycolmap.triangulate_points(prior, work / "database.db", work / "images", work / "tri").write_binary(
            str(work / "tri"))
        tri = pycolmap.Reconstruction(str(work / "tri"))
        before_px = _mean_reprojection(tri)
        obs_tri = _obs_per_image(tri)
        weak = [n for n in names if obs_tri[n] < MIN_OBS]
        good = [n for n in names if obs_tri[n] >= MIN_OBS]

        ark = _poses(prior, names)
        scale, after_ba_px, rms_cm, ref = 1.0, before_px, 0.0, ark
        dt_cm = {n: 0.0 for n in names}
        dr_deg = {n: 0.0 for n in names}
        rec = tri
        if len(good) >= MIN_GOOD:
            rec = pycolmap.Reconstruction(str(work / "tri"))
            _bundle_adjust(pycolmap, rec, good)
            after_ba_px = _mean_reprojection(rec)
            print(f"BA done {time.time() - t0:.0f}s")

            good_poses = _poses(rec, good)
            c_ref = np.array([good_poses[n][0] for n in good])
            scale, R, t = umeyama(c_ref, np.array([ark[n][0] for n in good]))
            rec.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(R), t))
            for n in weak:
                im = rec.find_image_with_name(n)
                im.frame.set_cam_from_world(im.camera_id, prior.find_image_with_name(n).cam_from_world())
            rec.update_point_3d_errors()
            ref = _poses(rec, names)
            dt_cm = {n: float(np.linalg.norm(ref[n][0] - ark[n][0]) * 100) for n in names}
            dr_deg = {n: _rot_angle_deg(ref[n][1], ark[n][1]) for n in names}
            rms_cm = float(np.sqrt(np.mean([dt_cm[n] ** 2 for n in good])))

        fallback = fallback_reason(len(good), rec.num_reg_images(), len(names), scale, after_ba_px,
                                   before_px, rms_cm)
        final = tri if fallback else rec
        final.write_binary(str(sparse))
        final.write_text(str(sparse))
        obs_final = _obs_per_image(final)

        arkit_c2w = {p.name: p.c2w_arkit for p in photos}
        c2w_ref = {}
        for n in names:
            if fallback:
                c2w_ref[n] = arkit_c2w[n]
                continue
            m = np.eye(4)
            m[:3, :3] = ref[n][1]
            m[:3, 3] = -ref[n][1] @ ref[n][0]
            c2w_ref[n] = invert_rigid(m) @ ARKIT_TO_CV
        poses = keep_weak_at_arkit(c2w_ref, arkit_c2w, obs_tri, MIN_OBS)
        is_refined = {n: not fallback and obs_tri[n] >= MIN_OBS for n in names}

        result = {
            "schema": SCHEMA,
            "frame": "arkit",
            "summary": {
                "reprojection_px_arkit": before_px,
                "reprojection_px_refined": _mean_reprojection(final),
                "sim3_scale": float(scale),
                "points3d": int(final.num_points3D()),
                "refined": sum(is_refined.values()),
                "weak": len(names) - sum(is_refined.values()),
                "seconds": time.time() - t0,
            },
            "fallback": fallback,
            "photos": photo_records(names, poses, is_refined, obs_tri if fallback else obs_final, dt_cm, dr_deg),
        }
        out = output_path(ctx, "registration")
        out.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
        return out
    finally:
        if not ctx.keep_work:
            shutil.rmtree(work, ignore_errors=True)
