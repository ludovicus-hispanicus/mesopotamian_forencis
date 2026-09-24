import numpy as np

from mesoprint.features import ridge_features, ridge_score

SPACING = 0.04
N = 151
_ax = (np.arange(N) - N // 2) * SPACING
X, Y = np.meshgrid(_ax, _ax)
DISK = X**2 + Y**2 <= (N // 2 * SPACING) ** 2


def feats(z):
    return ridge_features(np.where(DISK, z, 0.0), DISK, SPACING)


def test_parallel_ridges_frequency_and_straightness():
    f = feats(0.01 * np.cos(2 * np.pi * 2.5 * (X * 0.8 + Y * 0.6)))
    assert abs(f.peak_freq - 2.5) / 2.5 < 0.05
    assert f.band_ratio > 0.8
    assert f.straightness > 0.95


def test_curved_ridges_beat_straight_stripes():
    rings = feats(0.01 * np.cos(2 * np.pi * 2.2 * np.hypot(X + 1.5, Y + 1.0)))
    stripes = feats(0.01 * np.cos(2 * np.pi * 2.2 * X))
    assert rings.straightness < 0.85
    assert ridge_score(rings) > 2 * ridge_score(stripes)


def test_noise_scores_low():
    rng = np.random.default_rng(1)
    f = feats(rng.normal(0, 0.01, X.shape))
    assert ridge_score(f) < 0.1


def test_sub_noise_amplitude_is_suppressed():
    ridges = 0.01 * np.cos(2 * np.pi * 2.2 * np.hypot(X + 1.5, Y + 1.0))
    assert ridge_score(feats(ridges * 0.02)) < 0.5 * ridge_score(feats(ridges))
