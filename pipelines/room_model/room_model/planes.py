"""Robust plane fitting on LiDAR samples, with honest parameter sigmas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 1.0, 0.0])
CELL = 0.02  # m, raster cell turning inlier points into a supported area
INLIER = 0.03  # m
SIGMA_SYS_OFFSET = 0.01  # m, systematic floor of any fitted offset
SIGMA_SYS_AZIMUTH = np.radians(0.25)


@dataclass
class PlaneFit:
    normal: np.ndarray
    point: np.ndarray
    rms: float  # m, over the inliers
    n_inliers: int
    support_m2: float
    along_std: float  # m, std of inlier coordinates along the horizontal in-plane axis
    centroid: np.ndarray | None = None  # mean of the inliers


def mad_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def plane_from_points(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return vt[2], c


def fit_plane(P, seed_n, seed_p, window, min_threshold=0.012, iterations=8) -> PlaneFit | None:
    """Iteratively trimmed least squares started on a seed plane; the normal keeps the seed's orientation."""
    seed_n = np.asarray(seed_n, float) / np.linalg.norm(seed_n)
    n, p, thr = seed_n, np.asarray(seed_p, float).copy(), window
    for _ in range(iterations):
        sel = np.abs((P - p) @ n) < thr
        if sel.sum() < 30:
            return None
        n_new, p_new = plane_from_points(P[sel])
        if n_new @ seed_n < 0:
            n_new = -n_new
        thr = float(np.clip(3.0 * mad_sigma((P[sel] - p_new) @ n_new), min_threshold, window))
        done = np.linalg.norm(n_new - n) < 1e-6 and np.linalg.norm(p_new - p) < 1e-5
        n, p = n_new, p_new
        if done:
            break
    res = (P - p) @ n
    inl = np.abs(res) < INLIER
    k = int(inl.sum())
    if k < 30:
        return None
    e1 = np.cross(n, UP if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    rel = P[inl] - p
    along = rel @ e1
    q = np.floor(np.stack([along, rel @ e2], 1) / CELL).astype(np.int64)
    cells = len(np.unique(q[:, 0] * 100003 + q[:, 1]))
    return PlaneFit(normal=n, point=p, rms=float(np.sqrt(np.mean(res[inl] ** 2))), n_inliers=k,
                    support_m2=float(cells * CELL**2), along_std=float(along.std()), centroid=P[inl].mean(0))


def _effective_samples(fit: PlaneFit) -> float:
    return np.sqrt(max(fit.support_m2 / 0.01, 1.0))


def offset_sigma(fit: PlaneFit) -> float:
    return float(np.hypot(fit.rms / _effective_samples(fit), SIGMA_SYS_OFFSET))


def azimuth_sigma(fit: PlaneFit) -> float:
    return float(np.hypot(fit.rms / (_effective_samples(fit) * max(fit.along_std, 0.05)), SIGMA_SYS_AZIMUTH))
