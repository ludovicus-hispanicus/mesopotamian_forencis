import numpy as np

from mesoprint.mesh import Mesh, grid_mesh, load_ply, save_ply
from mesoprint.synth import make_tablet


def test_grid_mesh_roundtrip(tmp_path):
    z = np.random.default_rng(0).normal(0, 0.01, (20, 30))
    mesh = grid_mesh(z, 0.05)
    path = tmp_path / "m.ply"
    save_ply(path, mesh, {"quality": np.arange(mesh.n_vertices, dtype=float)})
    back = load_ply(path)
    assert back.n_vertices == 600 and len(back.faces) == 2 * 19 * 29
    np.testing.assert_allclose(back.vertices, mesh.vertices, atol=1e-5)
    np.testing.assert_array_equal(back.faces, mesh.faces)
    assert "quality" in back.extra


def test_flat_normals_and_spacing():
    mesh = grid_mesh(np.zeros((10, 10)), 0.04)
    n = mesh.vertex_normals()
    np.testing.assert_allclose(n, np.tile([0, 0, 1.0], (100, 1)), atol=1e-9)
    assert abs(mesh.median_edge_length() - 0.04) < 1e-9


def test_normals_point_out_of_clay_even_if_winding_is_inverted():
    mesh, _ = make_tablet(spacing=0.1)
    flipped = Mesh(mesh.vertices, mesh.faces[:, ::-1].copy())
    for m in (mesh, flipped):
        n = m.vertex_normals()
        centre = np.argmin(np.linalg.norm(m.vertices[:, :2], axis=1))
        assert n[centre, 2] > 0.9
