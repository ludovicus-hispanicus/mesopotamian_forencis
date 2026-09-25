import numpy as np
from plyfile import PlyData, PlyElement

from mesoprint.mesh import load_ply
from mesoprint.msii import file_msii, volume_fraction
from mesoprint.periodicity import ridge_periodicity

SPACING = 0.04
N = 151
_ax = (np.arange(N) - N // 2) * SPACING
X, Y = np.meshgrid(_ax, _ax)
DISK = X**2 + Y**2 <= (N // 2 * SPACING) ** 2
INNER = X**2 + Y**2 <= 1.5**2


def test_plane_is_one_half():
    m = volume_fraction(0.3 * X - 0.2 * Y, SPACING, 0.3)
    assert np.allclose(m[20:-20, 20:-20], 0.5, atol=1e-6)


def test_grooves_above_half_ridges_below():
    z = 0.02 * np.cos(2 * np.pi * X / 0.5)  # crests at x = 0, grooves at x = 0.25
    m = volume_fraction(z, SPACING, 0.25)
    col = N // 2
    assert m[col, col] < 0.5 < m[col, col + round(0.25 / SPACING)]


def test_periodic_ridges_have_high_contrast():
    z = 0.01 * np.cos(2 * np.pi * (X * 0.8 + Y * 0.6) / 0.5)
    p = ridge_periodicity(z, DISK, SPACING)
    assert np.median(p.contrast[INNER]) > 0.7
    assert abs(np.median(p.period[INNER]) - 0.5) <= 0.02


def test_gigamesh_feature_vectors_are_read_and_picked(tmp_path):
    n = 5
    v = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("quality", "f4"),
                           ("feature_vector", "f4", (16,))])
    v["x"] = np.arange(n)
    v["feature_vector"] = np.arange(16)[None, :] + 100 * np.arange(n)[:, None]
    f = np.zeros(1, dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = [0, 1, 2]
    path = tmp_path / "SM_000000_GMO_r0.30_n4_v256.volume.ply"
    PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(f, "face")]).write(str(path))

    mesh = load_ply(path)
    fv = mesh.extra["feature_vector"]
    assert fv.shape == (n, 16)
    vals, r = file_msii(mesh, 0.3)  # radii 0.30, 0.28125, ... 0.01875, largest first
    assert r == 0.3 and np.array_equal(vals, fv[:, 0])
    vals, r = file_msii(mesh, 0.15)
    assert r == 0.15 and np.array_equal(vals, fv[:, 8])
    mesh.name = "no_radius_in_name"
    assert file_msii(mesh, 0.3) is None


def test_single_line_and_noise_have_low_contrast():
    line = -0.03 * np.exp(-0.5 * (X / 0.08) ** 2)  # one groove, like a crack or ruling
    noise = np.random.default_rng(0).normal(0, 0.01, X.shape)
    for z in (line, noise):
        p = ridge_periodicity(z, DISK, SPACING)
        assert np.mean(p.contrast[INNER] > 0.5) < 0.1
