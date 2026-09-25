"""Ridge-pattern features of a 2D relief patch.

A fingerprint impression is a quasi-periodic ridge texture with a narrow band
of spatial frequencies (adult ridge period ~0.45-0.55 mm, narrower for
children and many women) whose orientation varies smoothly and is locally very
consistent. Wedges, cracks and clay grain lack that combination. The features
below measure each ingredient separately so they can be inspected, and later
used to train a classifier on labelled tablets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

#: Ridge frequency band in cycles/mm (periods ~0.28-0.63 mm).
RIDGE_BAND = (1.6, 3.6)
#: Reference band for the energy ratio (periods 0.125-4 mm).
REF_BAND = (0.25, 8.0)


@dataclass
class RidgeFeatures:
    band_ratio: float  # share of relief energy in the ridge band, 0..1
    band_amp_um: float  # RMS amplitude of the ridge-band relief, micrometres
    peak_freq: float  # dominant ridge frequency, cycles/mm
    spectral_spread: float  # relative spread of frequency within band
    local_coherence: float  # orientation consistency at ~1 mm scale, 0..1
    global_coherence: float  # share of ridge area near the dominant orientation, 0..1
    coverage: float  # fraction of the patch where ridge texture is present, 0..1

    @property
    def ridge_period_mm(self) -> float:
        return 1.0 / self.peak_freq if self.peak_freq > 0 else float("nan")

    @property
    def straightness(self) -> float:
        """~1 when all ridges in the patch are parallel (as in scanner stripe
        artifacts), lower when they curve (as in most fingerprint areas)."""
        return self.global_coherence

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ridge_period_mm"] = self.ridge_period_mm
        d["straightness"] = self.straightness
        return d


def frequency_grid(shape, spacing: float):
    fy = np.fft.fftfreq(shape[0], d=spacing)
    fx = np.fft.fftfreq(shape[1], d=spacing)
    return np.hypot(*np.meshgrid(fy, fx, indexing="ij"))


def band_transfer(freq: np.ndarray, lo: float, hi: float, soft: float = 0.25) -> np.ndarray:
    """Smooth (raised-cosine edged) band-pass transfer function."""
    h = np.zeros_like(freq)
    w_lo, w_hi = lo * soft, hi * soft
    h[(freq >= lo) & (freq <= hi)] = 1.0
    a = (freq > lo - w_lo) & (freq < lo)
    h[a] = 0.5 - 0.5 * np.cos(np.pi * (freq[a] - (lo - w_lo)) / w_lo)
    b = (freq > hi) & (freq < hi + w_hi)
    h[b] = 0.5 + 0.5 * np.cos(np.pi * (freq[b] - hi) / w_hi)
    return h


def bandpass(img: np.ndarray, spacing: float, lo: float, hi: float) -> np.ndarray:
    F = frequency_grid(img.shape, spacing)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * band_transfer(F, lo, hi)))


def orientation_field(img: np.ndarray, spacing: float, sigma_mm: float = 0.5):
    """Structure-tensor orientation; returns (theta, coherence, energy) maps.
    ``theta`` is the ridge-normal direction in radians (x = columns)."""
    gy, gx = np.gradient(img)
    s = sigma_mm / spacing
    jxx = ndimage.gaussian_filter(gx * gx, s)
    jyy = ndimage.gaussian_filter(gy * gy, s)
    jxy = ndimage.gaussian_filter(gx * gy, s)
    energy = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / np.maximum(energy, 1e-30)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return theta, coh, energy


def ridge_features(relief: np.ndarray, valid: np.ndarray, spacing: float,
                   band=RIDGE_BAND, ref=REF_BAND) -> RidgeFeatures:
    """Compute :class:`RidgeFeatures` for a detrended relief patch (heights in mm)."""
    taper = ndimage.gaussian_filter(valid.astype(float), 0.3 / spacing) * valid
    x = np.where(valid, relief - relief[valid].mean(), 0.0) * taper

    spec = np.fft.fft2(x)
    P = np.abs(spec) ** 2
    F = frequency_grid(x.shape, spacing)
    inband = (F >= band[0]) & (F <= band[1])
    inref = (F >= ref[0]) & (F <= ref[1])
    e_ref = P[inref].sum()
    e_band = P[inband].sum()
    band_ratio = float(e_band / e_ref) if e_ref > 0 else 0.0

    if e_band > 0:
        fm = float((P[inband] * F[inband]).sum() / e_band)
        spread = float(np.sqrt((P[inband] * (F[inband] - fm) ** 2).sum() / e_band) / fm)
        peak = _radial_peak(P, F, band, spacing, x.shape)
    else:
        fm, spread, peak = 0.0, 1.0, 0.0

    B = np.real(np.fft.ifft2(spec * band_transfer(F, *band)))
    inner = ndimage.binary_erosion(valid, iterations=max(1, int(round(0.4 / spacing))))
    if inner.sum() < 20:
        inner = valid
    amp_um = float(np.sqrt(np.mean(B[inner] ** 2)) * 1000.0)

    theta, coh, energy = orientation_field(B, spacing, 0.5)
    w = energy * inner
    local_coh = float((coh * w).sum() / max(w.sum(), 1e-30))
    global_coh = _dominant_orientation_share(theta, w * coh**2)

    env = ndimage.gaussian_filter(B**2, 0.5 / spacing)[inner]
    coverage = float(np.mean(env > 0.25 * np.percentile(env, 90))) if env.size else 0.0

    return RidgeFeatures(band_ratio, amp_um, peak or fm, spread, local_coh, global_coh, coverage)


def _dominant_orientation_share(theta: np.ndarray, weight: np.ndarray, half_width_deg: float = 12.0) -> float:
    """Share of the (weighted) ridge area oriented within +-``half_width_deg`` of
    the dominant direction. Robust to a few stray features such as a wedge edge
    crossing an otherwise straight stripe pattern."""
    nb = 180
    deg = np.degrees(theta) % 180.0
    hist = np.bincount(np.minimum((deg * nb / 180).astype(int), nb - 1).ravel(),
                       weights=weight.ravel(), minlength=nb)
    total = hist.sum()
    if total <= 0:
        return 0.0
    k = int(round(half_width_deg * nb / 180))
    window = np.ones(2 * k + 1)
    circ = np.convolve(np.concatenate([hist[-k:], hist, hist[:k]]), window, mode="valid")
    return float(circ.max() / total)


def _radial_peak(P, F, band, spacing, shape) -> float:
    """Frequency where the radially averaged spectrum rises most above the
    background, within ``band``.

    Clay relief has a steep, roughly power-law spectrum, so the raw maximum
    tends to sit at the low edge of the band. The background is modelled by a
    straight line in log-log space fitted outside the band and divided out;
    the peak of what remains is refined by parabolic interpolation.
    """
    df = 1.0 / (max(shape) * spacing)
    bins = np.floor(F / df).astype(int)
    sel = F <= 8.0
    prof = np.bincount(bins[sel], weights=P[sel]) / np.maximum(np.bincount(bins[sel]), 1)
    freqs = (np.arange(len(prof)) + 0.5) * df
    lp = np.log(np.maximum(prof, 1e-300))
    outside = ((freqs >= 0.4) & (freqs <= band[0] * 0.75)) | ((freqs >= band[1] * 1.3) & (freqs <= 8.0))
    if outside.sum() >= 3:
        slope, icpt = np.polyfit(np.log(freqs[outside]), lp[outside], 1)
        lp = lp - (slope * np.log(freqs) + icpt)
    lo_i = int(np.searchsorted(freqs, band[0]))
    hi_i = int(np.searchsorted(freqs, band[1]))
    if hi_i <= lo_i:
        return 0.0
    i = lo_i + int(np.argmax(lp[lo_i:hi_i]))
    if 0 < i < len(lp) - 1:
        a, b, c = lp[i - 1], lp[i], lp[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(freqs[i] + np.clip(off, -0.5, 0.5) * df)
    return float(freqs[i])


def ridge_score(f: RidgeFeatures, noise_um: float = 2.0, straight_lo: float = 0.75,
                straight_weight: float = 0.8) -> float:
    """Heuristic fingerprint likelihood in [0, 1].

    Product of: share of energy in the ridge band, local orientation
    consistency, spatial coverage of the texture, and an amplitude factor that
    suppresses sub-noise texture. Patches whose ridges are all parallel
    (``straightness`` above ``straight_lo``) are down-weighted because that is
    the signature of structured-light stripe artifacts. This is a
    hand-tuned starting point, to be replaced by a classifier trained on the
    labelled tablets.
    """
    amp = f.band_amp_um / (f.band_amp_um + noise_um)
    penalty = straightness_penalty(f.straightness, straight_lo, straight_weight)
    return float(np.clip(f.band_ratio * f.local_coherence * f.coverage * amp * penalty, 0.0, 1.0))


def straightness_penalty(straightness: float, straight_lo: float = 0.75, straight_weight: float = 0.8) -> float:
    """1 for curved ridges, falling to ``1 - straight_weight`` for perfectly parallel ones."""
    s = np.clip((straightness - straight_lo) / (1.0 - straight_lo), 0.0, 1.0)
    return float(1.0 - straight_weight * s)


def fingerprint_score(periodic_cover: float, straightness: float) -> float:
    """Detection score in [0, 1]: the share of the patch covered by clearly
    periodic ridges (see :mod:`mesoprint.periodicity`), down-weighted when all
    ridges are parallel, the signature of structured-light scanner stripes.

    Calibrated on one real print (SM 036475): the print's patches reach 0.65,
    the rest of that tablet stays below 0.31 (99% below 0.10).
    """
    return float(np.clip(periodic_cover * straightness_penalty(straightness), 0.0, 1.0))
