"""Local ridge periodicity: does the surface repeat itself across the ridges?

A fingerprint is a stack of parallel ridges at a steady spacing. At each point
the signal is smoothed along a trial ridge direction and compared with itself
shifted across the ridges (a local normalised autocorrelation, as in the
x-signature of Hong, Wan & Jain 1998). One period away it should look the same,
half a period away inverted, so

    contrast = (rho(tau) - rho(tau / 2)) / 2

is about 1 on ridges and about 0 on a single line (crack, ruling, wedge edge),
on clay grain, and on smooth surfaces, which correlate with themselves at every
lag. The direction with the most along-ridge energy is used at each pixel and
the best lag in :data:`LAGS` is kept, which also gives a rough local period.

On SM 036475 this separates the known print from wedges, cracks, rulings, breaks
and clay far better than band energy or orientation coherence, which all rate
those as highly as the print (see docs/REPORT.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft

#: Candidate ridge periods, mm (children to adults, with margin).
LAGS = (0.30, 0.36, 0.43, 0.51, 0.60, 0.70)


@dataclass
class Periodicity:
    contrast: np.ndarray  # per pixel, ~[-1, 1]; > 0.5 means clearly periodic
    period: np.ndarray  # mm, lag of the best contrast (coarse, from LAGS)
    theta: np.ndarray  # ridge direction of the dominant orientation, radians (x = columns)


def ridge_periodicity(signal: np.ndarray, valid: np.ndarray, spacing: float, lags=LAGS,
                      n_orient: int = 8, along_sigma: float = 0.5, window_sigma: float = 0.8,
                      select: str = "energy") -> Periodicity:
    """Periodicity maps for one image ``(h, w)`` or a stack ``(n, h, w)`` of equally sized images.

    ``signal`` should be high-passed (e.g. an MSII map minus 0.5); cells outside
    ``valid`` are ignored. ``along_sigma`` is the smoothing along the ridges and
    ``window_sigma`` the size of the neighbourhood the correlation is averaged over (mm).
    ``select`` picks the direction per pixel: ``"energy"`` (dominant texture
    direction) or ``"contrast"`` (most periodic direction, which follows ridges
    past a crack or ruling that dominates the energy).
    """
    single = signal.ndim == 2
    sig = signal[None] if single else signal
    val = valid[None] if single else valid
    n, h, w = sig.shape
    # zero padding of one lag keeps shifted copies from wrapping onto data
    margin = int(np.ceil(max(lags) / spacing))
    shape = (sfft.next_fast_len(h + margin), sfft.next_fast_len(w + margin, real=True))
    X = np.zeros((n, *shape), np.float32)
    for i in range(n):
        v = val[i]
        if v.any():
            X[i, :h, :w] = np.where(v, sig[i] - sig[i][v].mean(), 0.0)
    fy = sfft.fftfreq(shape[0], spacing)[:, None].astype(np.float32)
    fx = sfft.rfftfreq(shape[1], spacing)[None, :].astype(np.float32)
    FX = sfft.rfft2(X, workers=-1)
    win = np.exp(-2 * np.pi**2 * window_sigma**2 * (fx**2 + fy**2)).astype(np.float32)

    def local_mean(Y):
        return sfft.irfft2(sfft.rfft2(Y, workers=-1) * win, s=shape, workers=-1)

    all_lags = sorted(set(lags) | {t / 2 for t in lags})
    best_e = np.zeros((n, *shape), np.float32)
    best_c = np.zeros_like(best_e)
    best_p = np.zeros_like(best_e)
    best_t = np.zeros_like(best_e)
    for k in range(n_orient):
        th = np.pi * k / n_orient
        t, nrm = (np.cos(th), np.sin(th)), (-np.sin(th), np.cos(th))
        FS = FX * np.exp(-2 * np.pi**2 * along_sigma**2 * (fx * t[0] + fy * t[1]) ** 2).astype(np.float32)
        S = sfft.irfft2(FS, s=shape, workers=-1)
        E0 = np.maximum(local_mean(S * S), 1e-30)
        fn = fx * nrm[0] + fy * nrm[1]
        # the mean of the +tau and -tau shifts is a cosine phase factor
        rho = {tau: local_mean(S * sfft.irfft2(FS * np.cos(2 * np.pi * fn * tau).astype(np.float32),
                                               s=shape, workers=-1)) / E0 for tau in all_lags}
        c_k = np.full_like(best_c, -2.0)
        p_k = np.zeros_like(best_p)
        for tau in lags:
            c = 0.5 * (rho[tau] - rho[tau / 2])
            better = c > c_k
            c_k = np.where(better, c, c_k)
            p_k = np.where(better, tau, p_k)
        take = (E0 > best_e) if select == "energy" else (c_k > best_c) | (k == 0)
        best_e = np.where(take, E0, best_e)
        best_c = np.where(take, c_k, best_c)
        best_p = np.where(take, p_k, best_p)
        best_t = np.where(take, th, best_t)
    out = [a[:, :h, :w] for a in (best_c, best_p, best_t)]
    if single:
        out = [a[0] for a in out]
    return Periodicity(*out)
