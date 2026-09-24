"""Triangle-mesh I/O and basic geometry.

Coordinates are assumed to be in millimetres, which is what GigaMesh exports
for the Sulaymaniyah and HeiCuBeDa scans. Use ``Mesh.scaled`` if a file uses
other units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

_FACE_LIST_NAMES = ("vertex_indices", "vertex_index")


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float64, millimetres
    faces: np.ndarray | None = None  # (M, 3) int64
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # other per-vertex properties
    name: str = ""

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def scaled(self, factor: float) -> "Mesh":
        return Mesh(self.vertices * factor, self.faces, self.extra, self.name)

    def vertex_normals(self) -> np.ndarray:
        """Area-weighted vertex normals (unit length)."""
        if self.faces is None or len(self.faces) == 0:
            raise ValueError("mesh has no faces; normals cannot be computed")
        v, f = self.vertices, self.faces
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # length = 2 * area
        n = np.zeros_like(v)
        for k in range(3):
            n[:, k] = sum(np.bincount(f[:, j], weights=fn[:, k], minlength=len(v)) for j in range(3))
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return n / norm

    def median_edge_length(self, sample: int = 200_000, seed: int = 0) -> float:
        """Median edge length, i.e. the scan's effective lateral resolution."""
        if self.faces is None or len(self.faces) == 0:
            from scipy.spatial import cKDTree

            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.vertices), min(sample, len(self.vertices)), replace=False)
            d, _ = cKDTree(self.vertices).query(self.vertices[idx], k=2)
            return float(np.median(d[:, 1]))
        rng = np.random.default_rng(seed)
        f = self.faces
        if len(f) > sample:
            f = f[rng.choice(len(f), sample, replace=False)]
        v = self.vertices
        e = np.concatenate([
            np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1),
            np.linalg.norm(v[f[:, 1]] - v[f[:, 2]], axis=1),
            np.linalg.norm(v[f[:, 2]] - v[f[:, 0]], axis=1),
        ])
        return float(np.median(e))

    def summary(self) -> dict:
        lo, hi = self.vertices.min(axis=0), self.vertices.max(axis=0)
        spacing = self.median_edge_length()
        return {
            "name": self.name,
            "vertices": int(self.n_vertices),
            "faces": int(0 if self.faces is None else len(self.faces)),
            "bbox_mm": [round(float(x), 3) for x in (hi - lo)],
            "median_edge_mm": round(spacing, 5),
            "samples_per_ridge_0.45mm": round(0.45 / spacing, 1),
            "vertex_properties": sorted(self.extra),
        }


def load_ply(path: str | Path) -> Mesh:
    path = Path(path)
    known = {}
    face_list = _face_list_property(path)
    if face_list is not None:
        known = {"face": {face_list: 3}}
    data = PlyData.read(str(path), known_list_len=known) if known else PlyData.read(str(path))

    vert = data["vertex"].data
    vertices = np.column_stack([vert["x"], vert["y"], vert["z"]]).astype(np.float64)
    extra = {n: np.asarray(vert[n]) for n in vert.dtype.names if n not in ("x", "y", "z")}

    faces = None
    if "face" in data:
        fdata = data["face"].data
        name = next((n for n in _FACE_LIST_NAMES if n in fdata.dtype.names), None)
        if name is not None:
            col = fdata[name]
            if col.dtype == object:  # variable-length lists: keep triangles only
                tri = [np.asarray(x) for x in col if len(x) == 3]
                faces = np.array(tri, dtype=np.int64).reshape(-1, 3)
            else:
                faces = np.asarray(col, dtype=np.int64).reshape(-1, 3)
    return Mesh(vertices, faces, extra, path.stem)


def _face_list_property(path: Path) -> str | None:
    """Name of the face index list if the PLY is binary and has one; lets plyfile use
    its fast fixed-length reader (``known_list_len``) for triangle meshes."""
    element, binary = None, False
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("format"):
                binary = "binary" in line
            elif line.startswith("element"):
                element = line.split()[1]
            elif line.startswith("property list") and element == "face":
                name = line.split()[-1]
                if name in _FACE_LIST_NAMES and binary:
                    return name
            elif line == "end_header":
                break
    return None


def save_ply(path: str | Path, mesh: Mesh, scalars: dict[str, np.ndarray] | None = None,
             colors: np.ndarray | None = None) -> None:
    """Write a binary PLY. ``scalars`` become float vertex properties (e.g. ``quality``),
    ``colors`` is an (N, 3) uint8 array written as red/green/blue."""
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    scalars = scalars or {}
    fields += [(k, "f4") for k in scalars]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(mesh.n_vertices, dtype=fields)
    arr["x"], arr["y"], arr["z"] = mesh.vertices.T
    for k, s in scalars.items():
        arr[k] = s
    if colors is not None:
        arr["red"], arr["green"], arr["blue"] = colors.T
    elements = [PlyElement.describe(arr, "vertex")]
    if mesh.faces is not None:
        farr = np.empty(len(mesh.faces), dtype=[("vertex_indices", "i4", (3,))])
        farr["vertex_indices"] = mesh.faces
        elements.append(PlyElement.describe(farr, "face"))
    PlyData(elements, text=False).write(str(path))


def grid_mesh(z: np.ndarray, spacing: float, origin=(0.0, 0.0)) -> Mesh:
    """Triangulate a height field sampled on a regular grid (rows = y, cols = x)."""
    ny, nx = z.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    vertices = np.column_stack([
        origin[0] + xs.ravel() * spacing,
        origin[1] + ys.ravel() * spacing,
        z.ravel(),
    ])
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    return Mesh(vertices, faces)
