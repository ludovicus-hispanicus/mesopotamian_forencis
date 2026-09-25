"""Compare two prints by ridge-map correlation.

Minutiae matching needs a dozen good minutiae; prints on tablets are often a
few square millimetres with two or three. Correlating the enhanced ridge maps
(grooves dark, 1000 ppi, mirrored the same way) over all rotations and shifts
works for such small prints: two impressions of the same finger area line up
with a high normalised correlation over their overlap, different fingers do not.

Score = max over rotation and shift of the zero-mean normalised cross-correlation
of the two ridge maps within the overlap of their masks, with the overlap at
least ``min_overlap_mm2``. The masked correlation is done with FFTs (sums of
products, and of squares, over the overlap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
from scipy import ndimage


@dataclass
class PrintMap:
    print_id: str
    ridge: np.ndarray  # float, zero mean inside mask, 0 outside
    mask: np.ndarray  # bool
    mm_per_px: float

    @classmethod
    def from_images(cls, print_id: str, print8: np.ndarray, mask: np.ndarray, mm_per_px: float, band=(1.0, 4.0),
                    downsample: int = 2):
        """``print8``: grooves dark (0..255), white outside the mask. ``downsample``
        by block averaging (1000 ppi -> 500 ppi by default: 10 px per ridge period)."""
        x = (255.0 - print8.astype(float)) / 255.0
        x = np.where(mask, x, 0.0)
        if downsample > 1:
            d = downsample
            h, w = (x.shape[0] // d) * d, (x.shape[1] // d) * d
            x = x[:h, :w].reshape(h // d, d, w // d, d).mean(axis=(1, 3))
            mask = mask[:h, :w].reshape(h // d, d, w // d, d).mean(axis=(1, 3)) > 0.5
            mm_per_px *= d
        # band-pass around ridge frequencies so that only ridge structure correlates
        from .features import bandpass

        b = bandpass(x, mm_per_px, *band)
        b = np.where(mask, b - b[mask].mean(), 0.0) if mask.any() else b
        return cls(print_id, b.astype(np.float32), mask.astype(bool), mm_per_px)

    def rotated(self, deg: float) -> tuple[np.ndarray, np.ndarray]:
        r = ndimage.rotate(self.ridge, deg, reshape=True, order=1, mode="constant", cval=0.0)
        m = ndimage.rotate(self.mask.astype(np.float32), deg, reshape=True, order=1, mode="constant", cval=0.0) > 0.5
        return np.where(m, r, 0.0).astype(np.float32), m


def _corr(a: np.ndarray, ma: np.ndarray, b: np.ndarray, mb: np.ndarray, min_overlap_px: int):
    """Normalised cross-correlation of ``a`` and ``b`` over the overlap of their
    masks, for every shift of b relative to a (FFT). Returns (score, shift)."""
    h, w = a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]
    shape = (sfft.next_fast_len(h), sfft.next_fast_len(w, real=True))

    def F(x):
        z = np.zeros(shape, np.float32)
        z[:x.shape[0], :x.shape[1]] = x
        return sfft.rfft2(z, workers=-1)

    def xc(A, B):  # sum over overlap of A(p) * B(p + shift), as a map over shifts
        return sfft.irfft2(A * np.conj(B), s=shape, workers=-1)

    ma_f, mb_f = ma.astype(np.float32), mb.astype(np.float32)
    FA, FA2, FMA = F(a), F(a * a), F(ma_f)
    FB, FB2, FMB = F(b), F(b * b), F(mb_f)
    n = xc(FMA, FMB)  # overlap pixel count
    sab, saa, sbb = xc(FA, FB), xc(FA2, FMB), xc(FMA, FB2)
    sa, sb = xc(FA, FMB), xc(FMA, FB)
    ok = n >= min_overlap_px
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab - sa * sb / np.maximum(n, 1)
        va = saa - sa * sa / np.maximum(n, 1)
        vb = sbb - sb * sb / np.maximum(n, 1)
        r = np.where(ok, cov / np.sqrt(np.maximum(va * vb, 1e-12)), -1.0)
    k = int(np.argmax(r))
    dy, dx = np.unravel_index(k, r.shape)
    return float(r.flat[k]), (int(dy), int(dx)), float(n.flat[k])


def match(a: PrintMap, b: PrintMap, step_deg: float = 3.0, min_overlap_mm2: float = 3.0) -> dict:
    """Best correlation of ``b`` against ``a`` over rotations; both must share ``mm_per_px``."""
    min_px = int(min_overlap_mm2 / a.mm_per_px**2)
    best = {"score": -1.0, "rotation_deg": None, "shift_px": None, "overlap_mm2": None}
    for deg in np.arange(0.0, 360.0, step_deg):
        rb, mb = b.rotated(deg)
        s, shift, n = _corr(a.ridge, a.mask, rb, mb, min_px)
        if s > best["score"]:
            best = {"score": round(s, 4), "rotation_deg": float(deg), "shift_px": shift,
                    "overlap_mm2": round(n * a.mm_per_px**2, 1)}
    return {"a": a.print_id, "b": b.print_id, **best}
