"""Turn a small piece of the mesh surface into a 2D height map ("relief").

A patch is flattened by orthogonal projection onto its best-fit plane. That is
accurate for the few millimetres a fingerprint patch covers, including the
rounded tablet edges where prints are common, as long as the patch is small
compared with the surface's radius of curvature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Frame:
    """Local orthonormal frame: ``u``, ``v`` span the tangent plane, ``n`` is the outward normal."""

    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray

    def to_local(self, points: np.ndarray) -> np.ndarray:
        d = points - self.origin
        return np.column_stack([d @ self.u, d @ self.v, d @ self.n])

    def to_world(self, uvw: np.ndarray) -> np.ndarray:
        uvw = np.atleast_2d(uvw)
        return self.origin + uvw[:, :1] * self.u + uvw[:, 1:2] * self.v + uvw[:, 2:3] * self.n

    def as_dict(self) -> dict:
        return {k: [float(x) for x in getattr(self, k)] for k in ("origin", "u", "v", "n")}


def fit_frame(points: np.ndarray, normal_hint: np.ndarray | None = None,
              origin: np.ndarray | None = None, right_hint: np.ndarray | None = None) -> Frame:
    """Best-fit plane frame. ``u`` follows ``right_hint`` projected onto the plane
    (e.g. the tablet face's right axis, so images are upright), else the
    points' principal axis."""
    c = points.mean(axis=0)
    cov = np.cov((points - c).T)
    _, vecs = np.linalg.eigh(cov)
    n, u = vecs[:, 0], vecs[:, 2]
    if normal_hint is not None and n @ normal_hint < 0:
        n = -n
    if right_hint is not None:
        r = np.asarray(right_hint, float)
        r = r - (r @ n) * n
        if np.linalg.norm(r) > 1e-6:
            u = r / np.linalg.norm(r)
    v = np.cross(n, u)
    return Frame(c if origin is None else np.asarray(origin, float), u, v, n)


def rasterize(uvw: np.ndarray, spacing: float, radius: float, fill_sigma_px: float = 1.0):
    """Bin points into a square grid of side ``2*radius`` (mean height per cell).

    Empty cells are filled by normalised convolution. Returns ``(height, valid)``
    where ``valid`` marks cells inside the disk of ``radius`` supported by data.
    Row index follows ``v``, column index follows ``u``.
    """
    n = int(np.ceil(2 * radius / spacing)) + 1
    ix = np.floor((uvw[:, 0] + radius) / spacing + 0.5).astype(np.int64)
    iy = np.floor((uvw[:, 1] + radius) / spacing + 0.5).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    flat = iy[ok] * n + ix[ok]
    cnt = np.bincount(flat, minlength=n * n).reshape(n, n)
    tot = np.bincount(flat, weights=uvw[ok, 2], minlength=n * n).reshape(n, n)
    has = cnt > 0
    mean = np.where(has, tot / np.maximum(cnt, 1), 0.0)

    num = ndimage.gaussian_filter(mean, fill_sigma_px)
    den = ndimage.gaussian_filter(has.astype(float), fill_sigma_px)
    filled = np.where(has, mean, num / np.maximum(den, 1e-9))

    yy, xx = (np.mgrid[0:n, 0:n] - (n - 1) / 2) * spacing
    disk = xx**2 + yy**2 <= radius**2
    valid = disk & (den > 0.2)
    return np.where(valid, filled, 0.0), valid


def interpolate_grid(uvw: np.ndarray, spacing: float, radius: float, max_gap: float | None = None,
                     values: np.ndarray | None = None):
    """Like :func:`rasterize`, but linearly interpolates the triangulated points
    instead of binning them, so a grid finer than the scan has no empty cells.

    Cells farther than ``max_gap`` (default: 3 x the points' median spacing) from
    any point are real holes in the scan and stay invalid. Returns
    ``(height, valid)``, or ``(height, valid, value_grid)`` if per-point
    ``values`` are given.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree

    n = int(np.ceil(2 * radius / spacing)) + 1
    axis = (np.arange(n) - (n - 1) / 2) * spacing
    gu, gv = np.meshgrid(axis, axis)
    tree = cKDTree(uvw[:, :2])
    if max_gap is None:
        sample = uvw[:: max(1, len(uvw) // 5000), :2]
        max_gap = 3 * float(np.median(tree.query(sample, k=2)[0][:, 1]))
    cols = uvw[:, 2:3] if values is None else np.column_stack([uvw[:, 2], values])
    res = LinearNDInterpolator(uvw[:, :2], cols)(gu, gv)
    height = res[..., 0]
    dist = tree.query(np.column_stack([gu.ravel(), gv.ravel()]))[0].reshape(n, n)
    valid = np.isfinite(height) & (dist <= max_gap) & (gu**2 + gv**2 <= radius**2)
    if values is None:
        return np.where(valid, height, 0.0), valid
    return np.where(valid, height, 0.0), valid, np.where(valid, res[..., 1], 0.0)


def fill_invalid(height: np.ndarray, valid: np.ndarray, spacing: float, sigma_mm: float = 0.5) -> np.ndarray:
    """Continue the surface smoothly into invalid cells, so that filters running
    over the edge of the data see no step."""
    s = sigma_mm / spacing
    num = ndimage.gaussian_filter(np.where(valid, height, 0.0), s)
    den = ndimage.gaussian_filter(valid.astype(float), s)
    far = den < 1e-6
    out = np.where(valid, height, num / np.maximum(den, 1e-6))
    if far.any():
        out[far] = height[valid].mean() if valid.any() else 0.0
    return out


def detrend(height: np.ndarray, valid: np.ndarray, spacing: float, sigma_mm: float = 0.8) -> np.ndarray:
    """Remove the tablet's overall shape, leaving fine relief (same units as ``height``).

    First a quadratic surface is subtracted (patch curvature), then a normalised
    Gaussian low-pass of width ``sigma_mm`` (clay undulations, large wedges).
    Fingerprint ridges (period 0.3-0.6 mm) pass essentially unattenuated.
    """
    ny, nx = height.shape
    yy, xx = np.mgrid[0:ny, 0:nx] * spacing
    xs, ys, zs = xx[valid], yy[valid], height[valid]
    if len(zs) < 10:
        return np.zeros_like(height)
    A = np.column_stack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys])
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    full = np.column_stack([np.ones(xx.size), xx.ravel(), yy.ravel(), xx.ravel() ** 2,
                            (xx * yy).ravel(), yy.ravel() ** 2]) @ coef
    r = np.where(valid, height - full.reshape(height.shape), 0.0)
    s = sigma_mm / spacing
    m = valid.astype(float)
    low = ndimage.gaussian_filter(r, s) / np.maximum(ndimage.gaussian_filter(m, s), 1e-9)
    return np.where(valid, r - low, 0.0)
