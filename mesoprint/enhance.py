"""Gabor ridge enhancement, the standard fingerprint enhancement (Hong, Wan &
Jain 1998) applied to an MSII map.

At each pixel the map is filtered with a Gabor kernel tuned to the local ridge
direction and ridge spacing: a cosine across the ridges at the local period,
under a Gaussian window elongated along the ridges. Ridges reinforce
themselves, while grain and anything not periodic in that direction is
suppressed. Direction and period come from the periodicity map
(:mod:`mesoprint.periodicity`): a gradient-based orientation field does not
work here, because the gradient is dominated by the clay grain, which is finer
than the ridges. Filtering is done per (orientation, period) bin with FFTs.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage

from .periodicity import LAGS, Periodicity


def smooth_orientation(theta: np.ndarray, weight: np.ndarray, spacing: float, sigma_mm: float = 0.8) -> np.ndarray:
    """Weighted smoothing of an orientation field (mod pi) via doubled angles."""
    s = sigma_mm / spacing
    w = np.maximum(weight, 0.0)
    c2 = ndimage.gaussian_filter(w * np.cos(2 * theta), s)
    s2 = ndimage.gaussian_filter(w * np.sin(2 * theta), s)
    return 0.5 * np.arctan2(s2, c2)


def gabor_enhance(signal: np.ndarray, valid: np.ndarray, spacing: float, per: Periodicity,
                  n_orient: int = 16, periods=LAGS, across: float = 0.5, along: float = 0.8,
                  min_contrast: float = 0.2, guide: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Gabor-filtered ``signal`` (same sign convention: positive = groove).

    ``per`` gives the local ridge direction, period and periodicity contrast.
    ``across`` and ``along`` are the Gaussian widths in units of the local period.
    ``guide`` = (mask, ridge direction in radians) from reviewer strokes: inside
    the mask the stroke direction overrides the automatic orientation.
    """
    h, w = signal.shape
    x = np.where(valid, signal, 0.0).astype(np.float32)
    weight = np.where(valid, np.clip(per.contrast, 0, 1), 0.0)
    theta = per.theta.copy()
    if guide is not None:
        gmask, gtheta = guide
        theta = np.where(gmask, gtheta, theta)
        weight = np.where(gmask, 4.0, weight)  # the reviewer's word outweighs the automatic estimate
    normal = smooth_orientation(theta + np.pi / 2, weight, spacing)
    # local frequency: smooth 1/period where the surface is periodic, else the median
    known = valid & (per.contrast > min_contrast) & (per.period > 0)
    f = np.where(known, 1.0 / np.maximum(per.period, 1e-6), 0.0)
    s = 1.0 / spacing
    cover = ndimage.gaussian_filter(known.astype(float), s)
    fs = ndimage.gaussian_filter(f, s) / np.maximum(cover, 1e-9)
    f_med = float(np.median(f[known])) if known.any() else 1.0 / 0.5
    fs = np.where(cover > 0.05, fs, f_med)
    periods = np.asarray(periods, float)
    pb = np.argmin(np.abs(1.0 / periods[None, None, :] - fs[..., None]), axis=-1)
    ob = np.round(normal / (np.pi / n_orient)).astype(int) % n_orient

    half = int(np.ceil(3 * along * periods.max() / spacing))
    shape = (sfft.next_fast_len(h + 2 * half), sfft.next_fast_len(w + 2 * half, real=True))
    X = np.zeros(shape, np.float32)
    X[:h, :w] = x
    FX = sfft.rfft2(X, workers=-1)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1] * spacing
    out = np.zeros((h, w), np.float32)
    for o in range(n_orient):
        th = o * np.pi / n_orient  # ridge-normal direction
        xr = xx * np.cos(th) + yy * np.sin(th)  # across the ridges
        yr = -xx * np.sin(th) + yy * np.cos(th)  # along the ridges
        for k, T in enumerate(periods):
            sel = valid & (ob == o) & (pb == k)
            if not sel.any():
                continue
            env = np.exp(-0.5 * (xr**2 / (across * T) ** 2 + yr**2 / (along * T) ** 2))
            g = env * np.cos(2 * np.pi * xr / T)
            g -= env * (g.sum() / env.sum())  # zero mean, so flat areas give no response
            K = np.zeros(shape, np.float32)
            K[:2 * half + 1, :2 * half + 1] = g
            K = np.roll(K, (-half, -half), axis=(0, 1))
            R = sfft.irfft2(FX * sfft.rfft2(K, workers=-1), s=shape, workers=-1)[:h, :w]
            out[sel] = R[sel]
    return out
