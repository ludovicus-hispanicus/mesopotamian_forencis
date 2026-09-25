"""Extract a fingerprint region as flat, calibrated images plus ridge measurements.

Outputs for a region:

* ``relief``   - fine surface relief as seen looking at the tablet (light = high).
* ``msii``     - ridge-scale MSII map, shown like the relief: dark = groove in the
  clay (= finger ridge), light = raised clay.
* ``enhanced`` - Gabor-enhanced MSII map (:mod:`mesoprint.enhance`), contrast-normalised, same shading.
* ``print``    - ``enhanced`` cut to the print mask (clay grooves = finger ridges,
  drawn dark). All views are as seen on the tablet; an inked print of the finger
  would be the mirror image.
* ``mask``     - the relief with the isolated print area outlined.

The print is isolated with the periodicity map (:mod:`mesoprint.periodicity`):
the connected area where the surface repeats itself across parallel ridges.
Wedges, cracks and plain clay around it fall outside, and ridge measurements are
taken inside the mask only.

Images are upright with respect to the tablet face the region lies on (same
orientation as that face in the fat-cross overview) and are written at a stated
resolution (default 1000 ppi) with the DPI stored in the PNG, so forensic tools
measure them correctly. Note that the
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

from .enhance import gabor_enhance
from .features import RIDGE_BAND, RidgeFeatures, ridge_features
from .mesh import Mesh
from .msii import MSII_RADIUS, ridge_signal
from .periodicity import ridge_periodicity
from .raster import Frame, detrend, fit_frame, interpolate_grid
from .render import VIEWS

MM_PER_INCH = 25.4
MIN_PRINT_MM2 = 4.0  # smaller periodic specks are not treated as a print


@dataclass
class Extraction:
    frame: Frame
    radius: float
    spacing: float  # mm per pixel
    height: np.ndarray  # raw height above the frame plane, mm
    relief: np.ndarray  # detrended, mm
    valid: np.ndarray
    enhanced: np.ndarray  # dimensionless, ~[-1, 1]
    features: RidgeFeatures  # measured inside ``mask`` if a print was isolated, else on ``valid``
    msii: np.ndarray  # ridge-scale volume fraction minus 0.5
    contrast: np.ndarray  # periodicity contrast, see mesoprint.periodicity
    mask: np.ndarray  # isolated print area (all False if none was found)
    ridge_raw: np.ndarray | None = None  # enhanced before fading, for skeletons inside a reviewer's mask

    @property
    def ppi(self) -> float:
        return MM_PER_INCH / self.spacing

    @property
    def isolated(self) -> bool:
        return bool(self.mask.any())

    def measurements(self) -> dict:
        f = self.features
        region = self.mask if self.isolated else self.valid
        return {
            "ridge_frequency_per_mm": round(f.peak_freq, 4),
            "mean_ridge_breadth_mm": round(f.ridge_period_mm, 4),  # ridge + furrow, as in Kamp et al.
            "ridge_count_per_5mm": round(5 * f.peak_freq, 2),
            "ridge_amplitude_um_rms": round(f.band_amp_um, 2),
            "local_coherence": round(f.local_coherence, 3),
            "straightness": round(f.straightness, 3),
            "area_mm2": round(float(region.sum()) * self.spacing**2, 1),
            "isolated": self.isolated,
        }


def extract_region(mesh: Mesh, centre, radius: float = 8.0, spacing: float = MM_PER_INCH / 1000,
                   normals: np.ndarray | None = None, tree: cKDTree | None = None,
                   normal_cos: float = 0.3, highpass_sigma: float = 1.0, band=RIDGE_BAND,
                   msii_radius: float = MSII_RADIUS, mask_level: float = 0.4,
                   msii_values: np.ndarray | None = None) -> Extraction:
    """``msii_values``: optional per-vertex MSII (e.g. from a GigaMesh file); if
    None, the MSII map is computed from the extracted heights."""
    centre = np.asarray(centre, float)
    V = mesh.vertices
    if normals is None and mesh.faces is not None:
        normals = mesh.vertex_normals()
    tree = tree or cKDTree(V)
    idx = np.asarray(tree.query_ball_point(centre, radius * 1.1))
    if len(idx) < 100:
        raise ValueError(f"only {len(idx)} vertices within {radius} mm of {centre.tolist()}")

    # orient the plane on the central part, where the print is flattest, and turn
    # the image so that "up" is the up of the tablet face it lies on
    core = idx[np.linalg.norm(V[idx] - centre, axis=1) < radius / 2]
    hint = None if normals is None else normals[core].mean(axis=0)
    right = None if hint is None else VIEWS[max(VIEWS, key=lambda k: hint @ VIEWS[k][0])][1]
    frame = fit_frame(V[core], hint, origin=centre, right_hint=right)
    if normals is not None:
        idx = idx[normals[idx] @ frame.n > normal_cos]
    uvw = frame.to_local(V[idx])

    # interpolate the surface onto the output grid; real scan holes stay invalid
    if msii_values is None:
        height, valid = interpolate_grid(uvw, spacing, radius)
        msii, ms_ok = ridge_signal(height, valid, spacing, msii_radius)
    else:
        height, valid, msii = interpolate_grid(uvw, spacing, radius, values=msii_values[idx])
        ms_ok = valid
    relief = detrend(height, valid, spacing, highpass_sigma)
    # the most periodic direction lets the mask follow ridges past a crossing crack
    per = ridge_periodicity(msii, ms_ok, spacing, select="contrast")
    mask = print_mask(per.contrast, ms_ok, spacing, mask_level)

    feats = ridge_features(relief, mask if mask.any() else valid, spacing, band)
    # Gabor enhancement along the local ridge direction and spacing (positive = groove,
    # like the MSII map). The response is scaled globally and faded by the local
    # periodicity, so areas without real ridges stay grey instead of being
    # normalised into stripes.
    G = gabor_enhance(msii, ms_ok, spacing, per)
    raw, enhanced = fade_enhancement(G, ms_ok, per.contrast, spacing)
    return Extraction(frame, radius, spacing, height, relief, valid, enhanced, feats,
                      np.where(ms_ok, msii, 0.0), np.where(ms_ok, per.contrast, 0.0), mask, raw)


def fade_enhancement(G: np.ndarray, ok: np.ndarray, contrast: np.ndarray, spacing: float,
                     keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scale the Gabor response and fade it by the local periodicity. Returns
    ``(raw, faded)``: ``raw`` is the scaled response everywhere (used for the
    skeleton inside a reviewer's mask), ``faded`` the display version. ``keep``
    marks areas shown in full regardless (guide strokes)."""
    fade = np.clip(ndimage.gaussian_filter(np.where(ok, contrast, 0.0), 0.5 / spacing) / 0.5, 0.0, 1.0)
    if keep is not None:
        fade = np.maximum(fade, keep)
    ref = np.abs(G[ok & (fade > 0.5)])
    scale = float(np.percentile(ref, 95)) if ref.size else float(np.percentile(np.abs(G[ok]), 95)) if ok.any() else 1.0
    raw = np.where(ok, np.clip(G / (scale + 1e-12), -1, 1), 0.0)
    return raw, raw * fade


def print_mask(contrast: np.ndarray, valid: np.ndarray, spacing: float, level: float = 0.4,
               close_mm: float = 1.0, min_area_mm2: float = MIN_PRINT_MM2) -> np.ndarray:
    """Area covered by periodic ridges: contrast above ``level``, specks removed,
    gaps up to ``close_mm`` (e.g. a crack or ruling crossing the print) bridged
    and holes filled. Empty if no area reaches ``min_area_mm2``."""
    hot = valid & (contrast > level)
    hot = ndimage.binary_opening(hot, iterations=max(1, int(0.15 / spacing)))
    lab, n = ndimage.label(hot)
    if n == 0:
        return np.zeros_like(valid)
    area = ndimage.sum(np.ones_like(hot, float), lab, np.arange(1, n + 1)) * spacing**2
    keep = np.isin(lab, 1 + np.flatnonzero(area >= min_area_mm2 / 4))
    r = max(1, int(close_mm / 2 / spacing))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = xx**2 + yy**2 <= r * r
    pad = r + 1
    grown = np.pad(keep, pad)
    grown = ndimage.binary_erosion(ndimage.binary_dilation(grown, disk), disk)[pad:-pad, pad:-pad]
    grown = ndimage.binary_fill_holes(grown) & valid
    lab, n = ndimage.label(grown)
    if n == 0:
        return np.zeros_like(valid)
    area = ndimage.sum(np.ones_like(grown, float), lab, np.arange(1, n + 1)) * spacing**2
    return np.isin(lab, 1 + np.flatnonzero(area >= min_area_mm2))


def to_uint8(img: np.ndarray, valid: np.ndarray, lo_pct=1.0, hi_pct=99.0, background=255) -> np.ndarray:
    vals = img[valid]
    if vals.size == 0:
        return np.full(img.shape, background, np.uint8)
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    out = np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1) * 255
    return np.where(valid, out, background).astype(np.uint8)


def groove_dark8(img: np.ndarray, valid: np.ndarray, lo_pct=1.0, hi_pct=99.0) -> np.ndarray:
    """8-bit image of a groove-positive map (MSII, enhanced) with grooves dark and
    raised clay light, like the relief and a photograph; background white."""
    return np.where(valid, 255 - to_uint8(img, valid, lo_pct, hi_pct, background=0), 255).astype(np.uint8)


def save_png(path: str | Path, img8: np.ndarray, ppi: float) -> None:
    # row 0 of our grids is -v; flip so +v points up in the image
    Image.fromarray(np.flipud(img8)).save(str(path), dpi=(ppi, ppi))


def mask_overlay(gray8: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB: the print area as is, the rest dimmed, the mask outline in orange."""
    rgb = np.repeat(gray8[..., None].astype(float), 3, axis=2)
    outside = valid & ~mask
    rgb[outside] = rgb[outside] * 0.45 + 255 * 0.55 * np.array([0.75, 0.8, 0.9])
    edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
    rgb[edge] = (240, 120, 0)
    return rgb.astype(np.uint8)


def save_extraction(ex: Extraction, out_dir: str | Path, stem: str, extra_meta: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "relief": out / f"{stem}_relief.png",
        "msii": out / f"{stem}_msii.png",
        "enhanced": out / f"{stem}_enhanced.png",
        "print": out / f"{stem}_print.png",
        "mask": out / f"{stem}_mask.png",
    }
    rel8 = to_uint8(ex.relief, ex.valid)
    save_png(files["relief"], rel8, ex.ppi)
    save_png(files["msii"], groove_dark8(ex.msii, ex.valid, 2, 98), ex.ppi)
    enh8 = groove_dark8(ex.enhanced, ex.valid, 0.5, 99.5)
    save_png(files["enhanced"], enh8, ex.ppi)
    # print: grooves (finger ridges) dark, cut to the mask, as seen on the tablet
    cut = ex.mask if ex.isolated else ex.valid
    save_png(files["print"], np.where(cut, enh8, 255), ex.ppi)
    save_png(files["mask"], mask_overlay(rel8, ex.valid, ex.mask), ex.ppi)
    meta = {
        "ppi": round(ex.ppi, 1),
        "mm_per_pixel": ex.spacing,
        "radius_mm": ex.radius,
        "frame": ex.frame.as_dict(),
        "image_axes": "columns = +u (right of the tablet face), rows = +v (up of the face); "
                      "all views as seen on the tablet; 'print' is cut to the mask",
        "measurements": ex.measurements(),
        "files": {k: p.name for k, p in files.items()},
        **(extra_meta or {}),
    }
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta
