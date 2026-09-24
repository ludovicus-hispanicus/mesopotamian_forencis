"""Synthetic tablet surfaces with known ground truth, for testing without real scans.

The surface is one face of a pillow-shaped tablet (a height field) sampled at
the scanner's lateral resolution, carrying:

* clay texture (multi-scale noise),
* rows of wedge impressions with sharp edges,
* a whorl-type fingerprint on the flat face,
* an arch-type fingerprint on the curved margin (different ridge period),
* a patch of straight stripes imitating structured-light "wave" artifacts,
  deliberately at a ridge-like period to make it a hard negative,
* white measurement noise.

Heights are in millimetres. Fingerprint ridges are impressed as grooves.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .mesh import Mesh, grid_mesh


def _smooth_disk(r, radius, edge):
    return np.clip((radius - r) / edge, 0, 1) ** 2 * (3 - 2 * np.clip((radius - r) / edge, 0, 1))


def make_tablet(size=(30.0, 22.0), spacing=0.04, seed=0):
    """Return ``(mesh, truth)``; ``truth`` lists fingerprint / artifact / wedge locations."""
    rng = np.random.default_rng(seed)
    W, H = size
    nx, ny = int(W / spacing) + 1, int(H / spacing) + 1
    y, x = np.mgrid[0:ny, 0:nx] * spacing
    xc, yc = x - W / 2, y - H / 2

    # pillow-shaped face: flat centre, rounded margins (radius of curvature a few mm)
    a, b, h0 = W / 2, H / 2, 5.0
    z = h0 * (1 - np.clip(np.abs(xc / a), 0, 1) ** 4) * (1 - np.clip(np.abs(yc / b), 0, 1) ** 4)

    def noise(sigma_mm, amp_mm):
        n = ndimage.gaussian_filter(rng.standard_normal(z.shape), sigma_mm / spacing)
        return n / (n.std() + 1e-12) * amp_mm

    z += noise(2.0, 0.030) + noise(0.5, 0.010) + noise(0.12, 0.003)

    truth = {"fingerprints": [], "artifacts": [], "wedges": [], "spacing": spacing}

    # wedges in text lines, skipping the fingerprint areas
    fp1 = dict(cx=-5.0, cy=-2.0, radius=5.0, freq=2.1, kind="whorl")
    fp2 = dict(cx=11.0, cy=6.5, radius=3.5, freq=2.7, kind="arch")
    stripes = dict(cx=5.5, cy=-6.0, half=(3.5, 2.5), period=0.5, angle=0.35)
    keep_out = [(fp1["cx"], fp1["cy"], fp1["radius"] + 1.0), (fp2["cx"], fp2["cy"], fp2["radius"] + 1.0),
                (stripes["cx"], stripes["cy"], 4.5)]
    wedge_depth = np.zeros_like(z)
    for row_y in np.arange(-8.0, 9.0, 3.4):
        for col_x in np.arange(-11.0, 11.5, 2.6):
            wx, wy = col_x + rng.normal(0, 0.3), row_y + rng.normal(0, 0.2)
            if any(np.hypot(wx - kx, wy - ky) < kr for kx, ky, kr in keep_out):
                continue
            ang = rng.choice([0.0, np.pi / 2, np.pi / 4]) + rng.normal(0, 0.1)
            s = (xc - wx) * np.cos(ang) + (yc - wy) * np.sin(ang)
            t = -(xc - wx) * np.sin(ang) + (yc - wy) * np.cos(ang)
            head = 0.45 * np.clip(1 - s / 1.0 - 2 * np.abs(t) / 1.2, 0, None) * (s >= 0)
            tail = 0.20 * np.clip(1 - 2 * np.abs(t) / 0.25, 0, None) * np.clip(1 - s / 2.8, 0, 1) * (s >= 0)
            wedge_depth = np.maximum(wedge_depth, np.maximum(head, tail))
            truth["wedges"].append([wx, wy])
    z -= wedge_depth

    # whorl fingerprint: warped concentric ridges
    u, v = xc - fp1["cx"], yc - fp1["cy"]
    r1 = np.hypot(u, v)
    g = np.sqrt(u**2 + (v / 1.35) ** 2) + 0.25 * np.sin(0.6 * u + 0.4 * v)
    phase = 2 * np.pi * fp1["freq"] * g
    groove = 0.5 * (1 + np.cos(phase))
    z -= 0.025 * groove * _smooth_disk(r1, fp1["radius"], 1.0)

    # arch fingerprint on the rounded margin: ridges bending over a hump
    u, v = xc - fp2["cx"], yc - fp2["cy"]
    r2 = np.hypot(u, v)
    g = v + 1.4 * np.exp(-(u**2) / 6.0) + 0.08 * u
    z -= 0.020 * 0.5 * (1 + np.cos(2 * np.pi * fp2["freq"] * g)) * _smooth_disk(r2, fp2["radius"], 0.8)

    # structured-light stripe artifact: straight, single period, sharp-edged segment
    u, v = xc - stripes["cx"], yc - stripes["cy"]
    inside = (np.abs(u) < stripes["half"][0]) & (np.abs(v) < stripes["half"][1])
    d = u * np.cos(stripes["angle"]) + v * np.sin(stripes["angle"])
    z += 0.008 * np.sin(2 * np.pi * d / stripes["period"]) * inside

    z += rng.normal(0, 0.002, z.shape)

    mesh = grid_mesh(z, spacing, origin=(-W / 2, -H / 2))
    mesh.name = f"synthetic_seed{seed}"

    def surface_point(px, py):
        i = int(round((py + H / 2) / spacing))
        j = int(round((px + W / 2) / spacing))
        return [float(px), float(py), float(z[i, j])]

    for fp in (fp1, fp2):
        truth["fingerprints"].append({**fp, "centre": surface_point(fp["cx"], fp["cy"]),
                                      "ridge_period_mm": 1.0 / fp["freq"]})
    truth["artifacts"].append({**stripes, "kind": "scanner_stripes",
                               "centre": surface_point(stripes["cx"], stripes["cy"])})
    return mesh, truth
