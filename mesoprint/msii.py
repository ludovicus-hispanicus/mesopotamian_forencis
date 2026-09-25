"""Integral invariants (as in GigaMesh's MSII filter) on a height map.

For a ball of radius ``r`` centred on a surface point, the invariant is the
share of the ball's volume that lies inside the clay: 0.5 on a flat surface,
above 0.5 in a groove, below 0.5 on a ridge. With ``r`` about half the ridge
period (0.25-0.35 mm) fingerprint ridges stand out clearly, while broader shapes
(tablet curvature, clay undulation, wedge bodies) are suppressed: to first order
the invariant is the height minus its average over the disk of radius ``r``,
a high-pass with its cut-off near ``0.4 / r`` cycles/mm.

GigaMesh computes the same quantity on the mesh with a voxelised sphere. Here
the columns of the ball are integrated exactly on the patch's height grid,
which is equivalent for the small, nearly flat patches used for fingerprints.
If the mesh is a GigaMesh MSII export (``*_r0.30_n4_v256.volume.ply``), its
per-vertex values are used instead (:func:`file_msii`). On SM 036475 the two
agree closely (correlation 0.81 at r = 0.3 mm, same print mask within 1%).
"""

from __future__ import annotations

import re

import numpy as np
from scipy import ndimage

from .raster import fill_invalid

#: Default ball radius, mm: about half an adult ridge period.
MSII_RADIUS = 0.3

_GIGAMESH_NAME = re.compile(r"_r(\d+(?:\.\d+)?)_n(\d+)_v(\d+)")


def file_msii(mesh, radius: float = MSII_RADIUS, r_max: float | None = None):
    """Per-vertex MSII values stored by GigaMesh, at the scale closest to ``radius``.

    GigaMesh writes a ``feature_vector`` of K volume invariants per vertex for the
    radii ``r_max * (K - k) / K``, k = 0..K-1 (largest first), scaled to [-1, 1]
    with grooves positive. ``r_max`` is read from the file name (``_r0.30_``)
    unless given. Returns ``(values, radius_used)``, or None if the mesh has no
    feature vectors or ``r_max`` is unknown.
    """
    fv = mesh.extra.get("feature_vector")
    if fv is None or np.ndim(fv) != 2:
        return None
    if r_max is None:
        m = _GIGAMESH_NAME.search(mesh.name)
        if not m:
            return None
        r_max = float(m.group(1))
    K = fv.shape[1]
    radii = r_max * (K - np.arange(K)) / K
    k = int(np.argmin(np.abs(radii - radius)))
    return np.asarray(fv[:, k], dtype=np.float64), float(radii[k])


def ridge_signal(height: np.ndarray, valid: np.ndarray, spacing: float, radius: float = MSII_RADIUS):
    """Ridge-scale MSII map minus 0.5 (positive in grooves), and the cells whose
    ball lies on data."""
    ms = volume_fraction(fill_invalid(height, valid, spacing), spacing, radius) - 0.5
    return ms, ndimage.binary_erosion(valid, iterations=max(1, int(radius / spacing)))


def volume_fraction(height: np.ndarray, spacing: float, radius: float = MSII_RADIUS) -> np.ndarray:
    """Share of the ball of ``radius`` (mm) inside the clay, per grid cell.

    ``height`` must be gap-free (see :func:`mesoprint.raster.fill_invalid`);
    heights and ``spacing`` in mm. The ball column at offset ``d`` spans
    ``h(p) +- s`` with ``s = sqrt(r^2 - |d|^2)`` and is filled up to ``h(p + d)``,
    so steep walls saturate instead of dominating.
    """
    R = int(np.floor(radius / spacing))
    if R < 1:
        raise ValueError(f"radius {radius} mm is below the grid spacing {spacing} mm")
    ny, nx = height.shape
    pad = np.pad(height, R, mode="edge")
    acc = np.zeros_like(height)
    total = 0.0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            q = (dx * dx + dy * dy) * spacing**2
            if q >= radius**2:
                continue
            s = np.sqrt(radius**2 - q)
            nb = pad[R + dy:R + dy + ny, R + dx:R + dx + nx]
            acc += np.clip(nb - height, -s, s) + s
            total += 2 * s
    return acc / total  # exactly 0.5 on a plane
