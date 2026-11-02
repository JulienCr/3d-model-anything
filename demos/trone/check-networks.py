#!/usr/bin/env python3
"""Compare the modelled pipe/duct networks (demos/trone/networks.json) with the LiDAR depth frames.

Run from the repo root with the project venv. Writes a table to stdout and a plan overlay to
data/captures/trone-networks-plan.png. The fused cloud is cached in data/scans/trone/depth_cloud.npz.
Per run: residual of scan points to the modelled surface (lower half, the side the scanner saw), and
per 0.3 m window the measured bottom height and lateral offset of the bottom versus the model.
"""
import json
import os
import zlib
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

BUNDLE = 'data/captures-in/trone'
ROOM = 'data/demos/trone/room.json'
CACHE = 'data/scans/trone/depth_cloud.npz'
GEOMETRY = json.load(open('demos/trone/geometry.json'))  # shared with shell.ts
FLOOR_Y = GEOMETRY['floorY']
POLY = GEOMETRY['polygon']


def depth_cloud():
    if os.path.exists(CACHE):
        return np.load(CACHE)['P'].astype(np.float64)
    w2r = np.array(json.load(open(ROOM))['transform']['worldToRoom']).reshape(4, 4).T
    flip = np.diag([1, -1, -1, 1.0])
    pts = []
    for f in json.load(open(f'{BUNDLE}/depth/frames.json')):
        w, h = f['width'], f['height']
        d = np.frombuffer(zlib.decompress(open(f"{BUNDLE}/{f['file']}", 'rb').read(), -15), '<f2').astype(np.float32).reshape(h, w)
        c = np.fromfile(f"{BUNDLE}/{f['confidence']}", np.uint8).reshape(h, w)
        k = np.array(f['intrinsics']).reshape(3, 3).T
        ys, xs = np.mgrid[0:h, 0:w]
        ok = (c >= 2) & (d > 0.15) & (d < 3.0)
        z = d[ok]
        cam = np.c_[(xs[ok] - k[0, 2]) * z / k[0, 0], (ys[ok] - k[1, 2]) * z / k[1, 1], z, np.ones_like(z)]
        pts.append(((w2r @ np.array(f['transform']).reshape(4, 4).T @ flip) @ cam.T).T[:, :3])
    p = np.concatenate(pts)
    p[:, 1] -= FLOOR_Y
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).voxel_down_sample(0.01)
    q = np.asarray(pc.points)
    np.savez(CACHE, P=q.astype(np.float32))
    return q


def ceiling_mask(q):
    """Points within 2.5 cm of the local ceiling (97th height percentile per 10 cm cell)."""
    key = np.floor(q[:, [0, 2]] / 0.1).astype(int)
    k = key[:, 0] * 100000 + key[:, 1]
    order = np.argsort(k)
    ks, ys = k[order], q[order, 1]
    top = np.empty(len(q))
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1, len(ks)]
    for s, e in zip(starts[:-1], starts[1:]):
        top[order[s:e]] = np.percentile(ys[s:e], 97)
    return q[:, 1] > top - 0.025


def wall_dist(q):
    best = np.full(len(q), np.inf)
    for i in range(len(POLY)):
        a, b = np.array(POLY[i]), np.array(POLY[(i + 1) % len(POLY)])
        ab = b - a
        t = np.clip(((q[:, [0, 2]] - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.linalg.norm(q[:, [0, 2]] - (a + t[:, None] * ab), axis=1))
    return best


def surface_dist(Q, a, b, r):
    ab = b - a
    t = np.clip((Q - a) @ ab / (ab @ ab), 0, 1)
    return np.abs(np.linalg.norm(Q - (a + t[:, None] * ab), axis=1) - r)


def check_segment(P, a, b, r, others=()):
    """Scan residual of one segment, keeping only points closer to it than to any other modelled surface."""
    ab = b - a
    L = np.linalg.norm(ab)
    d = ab / L
    rel = P - a
    t = rel @ d
    radial = rel - np.outer(t, d)
    dist = np.linalg.norm(radial, axis=1)
    k = (t > 0.05) & (t < L - 0.05) & (dist > r - 0.04) & (dist < r + 0.06)
    horizontal = abs(d[1]) < 0.3
    if horizontal:
        k &= radial[:, 1] < 0.5 * r  # the lower half, which the scanner saw from below
    q = P[k]
    if len(q) < 30:
        return None
    keep = ~ceiling_mask(q) & (wall_dist(q) > 0.03)
    own = np.abs(dist[k] - r)
    for oa, ob, orr in others:
        keep &= own <= surface_dist(q, oa, ob, orr)
    if keep.sum() < 30:
        return None
    t, radial, dist = t[k][keep], radial[k][keep], dist[k][keep]
    res = np.abs(dist - r)
    out = {'n': int(len(res)), 'mean': res.mean(), 'p95': np.percentile(res, 95)}
    if horizontal:
        side = np.cross(d, [0, 1, 0])
        side /= np.linalg.norm(side)
        dys, dhs = [], []
        for t0 in np.arange(0.05, L - 0.3, 0.3):
            w = (t >= t0) & (t < t0 + 0.3) & (np.abs(radial @ side) < 0.7 * r)
            if w.sum() < 15:
                continue
            bottom = np.percentile(radial[w, 1], 5)
            low = w & (radial[:, 1] < bottom + 0.015)
            dys.append(bottom + r)
            dhs.append(np.median(radial[low] @ side))
        if dys:
            out.update({'dy_mean': np.mean(np.abs(dys)), 'dy_max': np.max(np.abs(dys)), 'dh_mean': np.mean(np.abs(dhs)), 'dh_max': np.max(np.abs(dhs))})
    return out


# Slices perpendicular to the runs: (axis, position, across range, y range, pipes counted in photos, photo refs).
SLICES = [('x', x, (1.4, 2.7), (1.85, 2.12), 5, '88-91') for x in (-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5)] + [
    ('x', 6.5, (1.4, 2.7), (1.85, 2.12), 6, '88-91 + conduit 186-192'), ('x', 8.0, (1.3, 1.9), (1.85, 2.12), 4, '173: 3 cream + 1 dark'),
    ('z', 1.4, (-1.5, -0.6), (1.85, 2.12), 5, '278-282'), ('z', 1.7, (-1.5, -0.6), (1.85, 2.12), 5, '278-282'),
    ('x', -8.0, (-2.9, -1.9), (1.9, 2.17), 1, '3, 12, 227-233'), ('x', -6.0, (-2.9, -1.9), (1.9, 2.17), 1, '225-233'),
    ('x', -4.5, (-2.9, -1.9), (1.9, 2.17), 1, '224-226'), ('x', -2.5, (-2.9, -1.9), (1.9, 2.17), 1, '222-224'),
    ('x', -1.7, (-2.9, -1.9), (1.9, 2.17), 2, '222, 223'), ('z', -1.0, (-3.0, -0.4), (1.9, 2.17), 2, '264, 270, 280'),
    ('z', -1.0, (5.3, 6.6), (1.85, 2.17), 2, '186-192'), ('x', 8.0, (0.5, 1.4), (1.9, 2.25), 2, '117, 174'),
]


def scan_count(P, axis, c, across, yr):
    """Pipe bottoms in a 10 cm slab: local minima of the lowest-surface profile, 6 cm apart, >= 8 mm deep."""
    i, j = (0, 2) if axis == 'x' else (2, 0)
    k = (np.abs(P[:, i] - c) < 0.05) & (P[:, j] > across[0]) & (P[:, j] < across[1]) & (P[:, 1] > yr[0]) & (P[:, 1] < yr[1])
    q = P[k]
    if len(q) < 20:
        return 0
    q = q[~ceiling_mask(q) & (wall_dist(q) > 0.03)]
    bins = np.arange(across[0], across[1], 0.01)
    prof = np.full(len(bins), np.nan)
    idx = np.floor((q[:, j] - across[0]) / 0.01).astype(int)
    for b in range(len(bins)):
        t = q[idx == b, 1]
        if len(t) >= 4:
            prof[b] = np.percentile(t, 10)
    mins = []
    for b in range(len(bins)):
        if not np.isfinite(prof[b]):
            continue
        w = prof[max(0, b - 5):b + 6]
        rim = np.concatenate([prof[max(0, b - 7):max(0, b - 3)], prof[b + 4:b + 8]])
        rim = rim[np.isfinite(rim)]
        if prof[b] <= np.nanmin(w) and len(rim) and rim.max() - prof[b] >= 0.008:
            if not mins or bins[b] - mins[-1] >= 0.06:
                mins.append(bins[b])
    return len(mins)


def model_count(runs, axis, c, across, yr):
    i, j = (0, 2) if axis == 'x' else (2, 0)
    n = set()
    for run in runs:
        for a, b in zip(run['points'], run['points'][1:]):
            if (a[i] - c) * (b[i] - c) > 0 or a[i] == b[i]:
                continue
            t = (c - a[i]) / (b[i] - a[i])
            m = [a[k] + t * (b[k] - a[k]) for k in range(3)]
            if across[0] < m[j] < across[1] and yr[0] - 0.1 < m[1] < yr[1] + 0.1:
                n.add(run['id'])
    return sorted(n)


def plan(P, runs, out):
    x0, x1, z0, z1, ppm = -9.3, 9.0, -3.6, 3.3, 70
    W, H = int((x1 - x0) * ppm) + 1, int((z1 - z0) * ppm) + 1
    k = (P[:, 1] > 1.8) & (P[:, 1] < 2.16) & (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 2] > z0) & (P[:, 2] < z1)
    q = P[k]
    q = q[~ceiling_mask(q)]
    img = np.zeros((H, W, 3), np.uint8)
    ix, iz = ((q[:, 0] - x0) * ppm).astype(int), ((q[:, 2] - z0) * ppm).astype(int)
    t = np.clip((q[:, 1] - 1.8) / 0.36, 0, 1)
    img[iz, ix] = np.stack([60 + 120 * t, 60 + 60 * (1 - t), 160 * (1 - t)], -1).astype(np.uint8)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    px = lambda x, z: ((x - x0) * ppm, (z - z0) * ppm)
    dr.line([px(*p) for p in POLY + [POLY[0]]], fill=(90, 90, 90))
    for run in runs:
        pts = run['points']
        dr.line([px(p[0], p[2]) for p in pts], fill=(255, 255, 255), width=1)
        for p, q2 in zip(pts, pts[1:]):
            a, b = np.array([p[0], p[2]]), np.array([q2[0], q2[2]])
            if np.linalg.norm(b - a) < 1e-6:
                continue
            n = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a) * run['d'] / 2
            for s in (-1, 1):
                dr.line([px(*(a + s * n)), px(*(b + s * n))], fill=(255, 220, 0), width=1)
        dr.text(px(pts[0][0], pts[0][2]), run['id'], fill=(0, 255, 255))
    for x in range(-9, 10):
        dr.text(px(x, z0 + 0.05), str(x), fill=(200, 200, 200))
    im.save(out)


def elevation(P, runs, out, z0, z1, ids):
    """Side view (x, y) of the scan points in a z band, with the modelled pipe bottoms and tops of `ids`."""
    x0, x1, y0, y1, ppm = -9.2, 8.9, 1.75, 2.3, 120
    W, H = int((x1 - x0) * ppm) + 1, int((y1 - y0) * ppm) + 1
    k = (P[:, 2] > z0) & (P[:, 2] < z1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 0] > x0) & (P[:, 0] < x1)
    img = np.zeros((H, W, 3), np.uint8)
    img[((y1 - P[k, 1]) * ppm).astype(int), ((P[k, 0] - x0) * ppm).astype(int)] = (150, 110, 70)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    px = lambda x, y: ((x - x0) * ppm, (y1 - y) * ppm)
    for run in runs:
        if run['id'] not in ids:
            continue
        r = run['d'] / 2
        for a, b in zip(run['points'], run['points'][1:]):
            for s_ in (-r, r):
                dr.line([px(a[0], a[1] + s_), px(b[0], b[1] + s_)], fill=(255, 220, 0))
        dr.text(px(run['points'][0][0], run['points'][0][1] + r + 0.02), run['id'], fill=(0, 255, 255))
    for x in range(-9, 9):
        dr.line([px(x, y0), px(x, y1)], fill=(60, 60, 60))
        dr.text(px(x + 0.02, y1 - 0.01), str(x), fill=(200, 200, 200))
    for y in (1.8, 1.9, 2.0, 2.1, 2.2):
        dr.line([px(x0, y), px(x1, y)], fill=(50, 50, 50))
        dr.text(px(x0 + 0.02, y + 0.012), f'{y:.1f}', fill=(255, 255, 0))
    im.save(out)


if __name__ == '__main__':
    P = depth_cloud()
    runs = json.load(open('demos/trone/networks.json'))['runs']
    print(f"{'run':5} {'d mm':>5} {'pts':>6} {'surf mean':>9} {'surf p95':>8} {'bottom dy mean/max':>18} {'lateral dh mean/max':>19}")
    segs = [(run['id'], np.array(a, float), np.array(b, float), run['d'] / 2) for run in runs for a, b in zip(run['points'], run['points'][1:])]
    for run in runs:
        rows = []
        for rid, a, b, r in segs:
            if rid != run['id']:
                continue
            near = [(oa, ob, orr) for oid, oa, ob, orr in segs if oid != rid and min(np.linalg.norm(oa - a), np.linalg.norm(ob - b),
                    np.linalg.norm((oa + ob) / 2 - (a + b) / 2)) < 0.5 * (np.linalg.norm(b - a) + np.linalg.norm(ob - oa)) + 0.3]
            rows.append(check_segment(P, a, b, r, near))
        rows = [r for r in rows if r]
        if not rows:
            print(f"{run['id']:5} {run['d'] * 1000:5.0f}  no scan support ({run['confidence']})")
            continue
        n = sum(r['n'] for r in rows)
        mean = sum(r['mean'] * r['n'] for r in rows) / n
        p95 = max(r['p95'] for r in rows)
        dy = [r for r in rows if 'dy_mean' in r]
        dys = f"{1000 * np.mean([r['dy_mean'] for r in dy]):5.0f}/{1000 * max(r['dy_max'] for r in dy):4.0f} mm" if dy else '-'
        dhs = f"{1000 * np.mean([r['dh_mean'] for r in dy]):5.0f}/{1000 * max(r['dh_max'] for r in dy):4.0f} mm" if dy else '-'
        print(f"{run['id']:5} {run['d'] * 1000:5.0f} {n:6d} {1000 * mean:6.1f} mm {1000 * p95:5.1f} mm {dys:>18} {dhs:>19}")
    print()
    print(f"{'slice':10} {'across':>13} {'photos':>6} {'model':>5} {'scan':>4}  model runs / photo refs")
    for axis, c, across, yr, photo, refs in SLICES:
        ids = model_count(runs, axis, c, across, yr)
        print(f"{axis}={c:<8} {across[0]:5.1f}..{across[1]:<5.1f} {photo:6d} {len(ids):5d} {scan_count(P, axis, c, across, yr):4d}  {','.join(ids)} / {refs}")
    plan(P, runs, 'data/captures/trone-networks-plan.png')
    elevation(P, runs, 'data/captures/trone-networks-elev-north.png', -2.55, -2.2, {'N1a'})
    elevation(P, runs, 'data/captures/trone-networks-elev-lanes.png', 1.6, 2.2, {'N3a', 'N3b', 'N2a', 'N2b'})
    print('overlays: data/captures/trone-networks-plan.png, -elev-north.png, -elev-lanes.png')
