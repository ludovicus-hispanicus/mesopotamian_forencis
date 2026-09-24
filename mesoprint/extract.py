"""Extract a fingerprint region as flat, calibrated images plus ridge measurements.

Outputs for a region:

* ``relief``   - fine surface relief as seen looking at the tablet (light = high).
* ``enhanced`` - ridge-band filtered and contrast-normalised relief.
* ``print``    - ``enhanced`` mirrored left-right so it reads like an ink print of
  the finger: clay grooves are the finger's ridges and are drawn dark.

Images are written at a stated resolution (default 1000 ppi) with the DPI
stored in the PNG, so forensic tools measure them correctly. Note that the
region is flattened by projection onto a plane; strongly curved areas (tablet
corners) will be foreshortened towards the rim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from .features import RIDGE_BAND, RidgeFeatures, bandpass, ridge_features
from .mesh import Mesh
from .raster import Frame, detrend, fit_frame, rasterize

MM_PER_INCH = 25.4


@dataclass
class Extraction:
    frame: Frame
    radius: float
    spacing: float  # mm per pixel
    height: np.ndarray  # raw height above the frame plane, mm
    relief: np.ndarray  # detrended, mm
    valid: np.ndarray
    enhanced: np.ndarray  # dimensionless, ~[-1, 1]
    features: RidgeFeatures

    @property
    def ppi(self) -> float:
        return MM_PER_INCH / self.spacing

    def measurements(self) -> dict:
        f = self.features
        return {
            "ridge_frequency_per_mm": round(f.peak_freq, 4),
            "mean_ridge_breadth_mm": round(f.ridge_period_mm, 4),  # ridge + furrow, as in Kamp et al.
            "ridge_count_per_5mm": round(5 * f.peak_freq, 2),
            "ridge_amplitude_um_rms": round(f.band_amp_um, 2),
            "local_coherence": round(f.local_coherence, 3),
            "straightness": round(f.straightness, 3),
            "area_mm2": round(float(self.valid.sum()) * self.spacing**2, 1),
        }


def extract_region(mesh: Mesh, centre, radius: float = 7.0, spacing: float = MM_PER_INCH / 1000,
                   normals: np.ndarray | None = None, tree: cKDTree | None = None,
                   normal_cos: float = 0.3, highpass_sigma: float = 1.0, band=RIDGE_BAND) -> Extraction:
    centre = np.asarray(centre, float)
    V = mesh.vertices
    if normals is None and mesh.faces is not None:
        normals = mesh.vertex_normals()
    tree = tree or cKDTree(V)
    idx = np.asarray(tree.query_ball_point(centre, radius * 1.1))
    if len(idx) < 100:
        raise ValueError(f"only {len(idx)} vertices within {radius} mm of {centre.tolist()}")

    # orient the plane on the central part, where the print is flattest
    core = idx[np.linalg.norm(V[idx] - centre, axis=1) < radius / 2]
    hint = None if normals is None else normals[core].mean(axis=0)
    frame = fit_frame(V[core], hint, origin=centre)
    pts = V[idx]
    if normals is not None:
        pts = pts[normals[idx] @ frame.n > normal_cos]
    uvw = frame.to_local(pts)

    # bin at the scan's own resolution (never interpolating across holes), then
    # resample to the requested output resolution
    native = max(_median_spacing(uvw[:, :2]), spacing)
    h0, v0 = rasterize(uvw, native, radius)
    n = int(np.ceil(2 * radius / spacing)) + 1
    zoom = (n - 1) / (h0.shape[0] - 1)
    height = ndimage.zoom(h0, zoom, order=3, grid_mode=False)[:n, :n]
    vz = ndimage.zoom(v0.astype(float), zoom, order=1, grid_mode=False)[:n, :n] > 0.5
    axis = (np.arange(n) - (n - 1) / 2) * spacing
    gu, gv = np.meshgrid(axis, axis)
    valid = vz & (gu**2 + gv**2 <= radius**2)
    height = np.where(valid, height, 0.0)

    relief = detrend(height, valid, spacing, highpass_sigma)
    feats = ridge_features(relief, valid, spacing, band)
    f0 = feats.peak_freq if feats.peak_freq > 0 else float(np.mean(band))
    lo, hi = max(0.55 * f0, 0.8), min(1.7 * f0, 6.0)
    B = bandpass(relief, spacing, lo, hi)
    s = 0.6 / spacing
    m = valid.astype(float)
    local_rms = np.sqrt(ndimage.gaussian_filter(B**2 * m, s) / np.maximum(ndimage.gaussian_filter(m, s), 1e-9))
    enhanced = np.where(valid, np.clip(B / (2 * local_rms + 1e-12), -1, 1), 0.0)
    return Extraction(frame, radius, spacing, height, relief, valid, enhanced, feats)


def _median_spacing(uv: np.ndarray, sample: int = 5000) -> float:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(uv), min(sample, len(uv)), replace=False)
    d, _ = cKDTree(uv).query(uv[idx], k=2)
    return float(np.median(d[:, 1]))


def to_uint8(img: np.ndarray, valid: np.ndarray, lo_pct=1.0, hi_pct=99.0, background=255) -> np.ndarray:
    vals = img[valid]
    if vals.size == 0:
        return np.full(img.shape, background, np.uint8)
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    out = np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1) * 255
    return np.where(valid, out, background).astype(np.uint8)


def save_png(path: str | Path, img8: np.ndarray, ppi: float) -> None:
    # row 0 of our grids is -v; flip so +v points up in the image
    Image.fromarray(np.flipud(img8)).save(str(path), dpi=(ppi, ppi))


def save_extraction(ex: Extraction, out_dir: str | Path, stem: str, extra_meta: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "relief": out / f"{stem}_relief.png",
        "enhanced": out / f"{stem}_enhanced.png",
        "print": out / f"{stem}_print.png",
    }
    save_png(files["relief"], to_uint8(ex.relief, ex.valid), ex.ppi)
    enh8 = to_uint8(ex.enhanced, ex.valid, 0.5, 99.5)
    save_png(files["enhanced"], enh8, ex.ppi)
    save_png(files["print"], np.fliplr(enh8), ex.ppi)
    meta = {
        "ppi": round(ex.ppi, 1),
        "mm_per_pixel": ex.spacing,
        "radius_mm": ex.radius,
        "frame": ex.frame.as_dict(),
        "image_axes": "columns = +u, rows = +v (up); 'print' is mirrored left-right",
        "measurements": ex.measurements(),
        "files": {k: p.name for k, p in files.items()},
        **(extra_meta or {}),
    }
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta
