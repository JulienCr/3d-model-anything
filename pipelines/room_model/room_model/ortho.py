"""Ortho images of a host face from posed photos, gradient edges on them, and edge depth across views."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from room_model.photo_geom import MIN_DEPTH, PosedPhoto, project

RES = 0.002  # m per ortho pixel: a phone photo at 2-3 m resolves 1.5-2.5 mm, a finer grid adds no detail
SMOOTH = 1.0  # ortho px of Gaussian blur before the gradient: removes the resampling grid, keeps a 2 mm edge
LOCALISATION = 0.002  # m; a gradient peak is located to about one ortho pixel, the noise floor of an edge
STRONG = 0.5  # a peak under half the strongest one nearby is texture or shading, not a material boundary
DELTA_STEP = 0.0005  # m between the depths tried along the face normal: 1 mm of depth moves an edge < 2 mm at 60 degrees


@dataclass(frozen=True)
class FaceFrame:
    """A host face: s metres along the host from its start, y above the floor, delta along `normal` (into the room)."""

    origin: np.ndarray  # room point (s, y, delta) = (0, 0, 0): host start, floor level, on the face
    s_axis: np.ndarray
    normal: np.ndarray

    def point(self, s, y, delta=0.0) -> np.ndarray:
        s, y, delta = np.broadcast_arrays(*(np.asarray(v, float) for v in (s, y, delta)))
        return self.origin + s[..., None] * self.s_axis + y[..., None] * np.array([0.0, 1.0, 0.0]) + delta[..., None] * self.normal

    def coords(self, pts) -> np.ndarray:
        """(..., 3) of (s, y, delta) for room points."""
        d = np.asarray(pts, float) - self.origin
        return np.stack([d @ self.s_axis, d[..., 1], d @ self.normal], axis=-1)


@dataclass
class Ortho:
    rgb: np.ndarray  # (h, w, 3) uint8, row 0 at the top (highest y)
    valid: np.ndarray  # (h, w) bool: the pixel comes from inside the photo, in front of the camera
    ss: np.ndarray  # (w,) s of the pixel centres
    ys: np.ndarray  # (h,) y of the pixel centres, decreasing

    def col(self, s):
        return (np.asarray(s, float) - self.ss[0]) / (self.ss[1] - self.ss[0])

    def row(self, y):
        return (np.asarray(y, float) - self.ys[0]) / (self.ys[1] - self.ys[0])


def ortho(photo: PosedPhoto, image: np.ndarray, frame: FaceFrame, box, delta: float = 0.0, res: float = RES) -> Ortho:
    """The plane `delta` m in front of the face, window box = (s0, s1, y0, y1), seen by `photo` (`image`: sensor RGB)."""
    s0, s1, y0, y1 = box
    ss = s0 + (np.arange(max(1, int(round((s1 - s0) / res)))) + 0.5) * res
    ys = y1 - (np.arange(max(1, int(round((y1 - y0) / res)))) + 0.5) * res
    S, Y = np.meshgrid(ss, ys)
    uv, depth = project(photo, frame.point(S.ravel(), Y.ravel(), delta))
    h, w = image.shape[:2]
    u, v = uv[:, 0] - 0.5, uv[:, 1] - 0.5  # continuous coordinates: pixel i covers [i, i + 1)
    valid = (depth > MIN_DEPTH) & (u >= 0) & (v >= 0) & (u <= w - 1) & (v <= h - 1)
    coords = np.stack([np.where(valid, v, 0), np.where(valid, u, 0)])
    rgb = np.stack([ndimage.map_coordinates(image[..., c].astype(np.float32), coords, order=1) for c in range(3)], axis=-1)
    rgb[~valid] = 0
    return Ortho(np.clip(rgb, 0, 255).astype(np.uint8).reshape(S.shape + (3,)), valid.reshape(S.shape), ss, ys)


def gradient(o: Ortho, axis: str) -> np.ndarray:
    """|d grey / d s| ('s', vertical edges) or |d grey / d y| ('y', horizontal edges); 0 next to invalid pixels."""
    grey = ndimage.gaussian_filter(o.rgb.astype(np.float32).mean(axis=2), SMOOTH)
    g = np.abs(ndimage.sobel(grey, axis=1 if axis == "s" else 0))
    return np.where(ndimage.binary_erosion(o.valid, iterations=3), g, 0.0)


def profile(o: Ortho, axis: str, band) -> tuple[np.ndarray, np.ndarray]:
    """(coords, mean gradient) across the edges of `axis`, over the band (y range for 's', s range for 'y'); NaN where
    less than half of the band is valid."""
    g = gradient(o, axis)
    if axis == "s":
        sel, m = (o.ys >= band[0]) & (o.ys <= band[1]), o.valid
        num, cnt = g[sel].sum(0), m[sel].sum(0)
        x, total = o.ss, sel.sum()
    else:
        sel, m = (o.ss >= band[0]) & (o.ss <= band[1]), o.valid
        num, cnt = g[:, sel].sum(1), m[:, sel].sum(1)
        x, total = o.ys, sel.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(cnt > 0.5 * max(total, 1), num / np.maximum(cnt, 1), np.nan)
    return x, prof


def peaks(x: np.ndarray, p: np.ndarray, lo: float, hi: float, k: int = 4) -> list[tuple[float, float]]:
    """Up to k local maxima of p with lo <= x <= hi, strongest first, as (sub-pixel position, strength)."""
    out = []
    for i in range(1, len(p) - 1):
        a, b, c = p[i - 1:i + 2]
        if not (lo <= x[i] <= hi) or not np.isfinite([a, b, c]).all() or not (b >= a and b > c) or b <= 0:
            continue
        den = a - 2 * b + c
        out.append((float(x[i] + (0.5 * (a - c) / den if den < 0 else 0.0) * (x[1] - x[0])), float(b)))
    return sorted(out, key=lambda t: -t[1])[:k]


def snap(o: Ortho, axis: str, band, guess: float, win: float) -> float | None:
    """The gradient peak nearest to `guess` within `win` m among those at least STRONG of the strongest; None if none."""
    x, p = profile(o, axis, band)
    found = peaks(x, p, guess - win, guess + win, len(x))
    if not found:
        return None
    return min((f for f in found if f[1] >= STRONG * found[0][1]), key=lambda f: abs(f[0] - guess))[0]


@dataclass
class EdgeFix:
    pos: float  # s or y of the edge, metres
    delta: float  # depth of the edge along the face normal, metres (negative = behind the face)
    sigma: float  # of `pos`: the view spread and the depth ambiguity
    views: int
    depths: tuple[float, float]  # the depths the views cannot tell apart


def camera_coords(frame: FaceFrame, photo: PosedPhoto) -> np.ndarray:
    return frame.coords(photo.centre)


def triangulate(observations: list[tuple[float, float, float]], bounds: tuple[float, float]) -> EdgeFix | None:
    """Edge from its position at delta = 0 in each view, given as (pos0, camera pos, camera delta).

    The ray of view i meets depth d at pos0 + (cam - pos0) * d / cam_delta. The depths within `bounds` where the
    views' spread stays within LOCALISATION of the best make the ambiguity: half the range of positions over them,
    with the spread, is the sigma; the depth kept is their centre. One view, or views without baseline across the
    edge: the whole bounds are ambiguous.
    """
    obs = np.array([o for o in observations if o[2] > MIN_DEPTH], float).reshape(-1, 3)
    if not len(obs):
        return None
    ds = np.arange(bounds[0], bounds[1] + DELTA_STEP / 2, DELTA_STEP)
    pos = obs[:, 0][None, :] + (obs[:, 1] - obs[:, 0])[None, :] * ds[:, None] / obs[:, 2][None, :]
    rms = pos.std(axis=1)
    mean = pos.mean(axis=1)
    near = rms <= rms.min() + LOCALISATION
    best = int(np.argmin(np.abs(ds - ds[near].mean())))
    ambiguity = (mean[near].max() - mean[near].min()) / 2
    sigma = float(np.hypot(max(rms[best], LOCALISATION) / np.sqrt(len(obs)), ambiguity))
    return EdgeFix(float(mean[best]), float(ds[best]), sigma, len(obs), (float(ds[near].min()), float(ds[near].max())))
