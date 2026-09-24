"""Scan a whole mesh for fingerprint-like surface regions.

The surface is covered with overlapping circular patches (default 6 mm across,
every 2 mm). Each patch is flattened to a relief map and scored with
:func:`mesoprint.features.ridge_score`. Scores are interpolated back to the
vertices as a heat map, and neighbouring high-scoring patches are grouped into
candidate regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .features import RIDGE_BAND, RidgeFeatures, ridge_features, ridge_score
from .mesh import Mesh
from .raster import Frame, detrend, fit_frame, rasterize


@dataclass
class DetectParams:
    patch_radius: float = 3.0  # mm
    stride: float = 2.0  # mm between patch centres
    grid_spacing: float | None = None  # mm; default: mesh edge length, at least min_grid_spacing
    min_grid_spacing: float = 0.04  # ridges need no finer sampling; keeps dense meshes fast
    highpass_sigma: float = 0.8  # mm, see raster.detrend
    normal_cos: float = 0.3  # drop points facing away from the patch normal
    min_valid: float = 0.6  # minimum fraction of the patch disk covered by data
    band: tuple[float, float] = RIDGE_BAND
    threshold: float = 0.10  # score above which a patch counts towards a candidate


@dataclass
class PatchResult:
    centre: np.ndarray
    normal: np.ndarray
    valid_frac: float
    features: RidgeFeatures | None
    score: float

    def as_dict(self) -> dict:
        d = {"centre": [float(x) for x in self.centre], "normal": [float(x) for x in self.normal],
             "valid_frac": self.valid_frac, "score": self.score}
        if self.features is not None:
            d.update(self.features.as_dict())
        return d


@dataclass
class Candidate:
    centre: np.ndarray
    normal: np.ndarray
    score: float  # best patch score in the region
    mean_score: float
    n_patches: int
    area_mm2: float  # rough: patches * stride^2
    ridge_period_mm: float
    straightness: float
    patch_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"centre": [round(float(x), 3) for x in self.centre],
                "normal": [round(float(x), 4) for x in self.normal],
                "score": round(self.score, 4), "mean_score": round(self.mean_score, 4),
                "n_patches": self.n_patches, "area_mm2": round(self.area_mm2, 1),
                "ridge_period_mm": round(self.ridge_period_mm, 4),
                "straightness": round(self.straightness, 3)}


@dataclass
class Detection:
    params: DetectParams
    spacing: float
    patches: list[PatchResult]
    vertex_score: np.ndarray
    candidates: list[Candidate]


def sample_centres(vertices: np.ndarray, stride: float) -> np.ndarray:
    """One vertex per ``stride``-sized voxel: the one nearest the voxel's centroid."""
    keys = np.floor(vertices / stride).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    k = inv.max() + 1
    cnt = np.bincount(inv, minlength=k)
    mean = np.column_stack([np.bincount(inv, vertices[:, i], k) for i in range(3)]) / cnt[:, None]
    d = np.linalg.norm(vertices - mean[inv], axis=1)
    order = np.lexsort((d, inv))
    first = np.ones(len(order), bool)
    first[1:] = inv[order[1:]] != inv[order[:-1]]
    return order[first]


def analyse_patch(points: np.ndarray, normals: np.ndarray | None, centre: np.ndarray,
                  radius: float, spacing: float, params: DetectParams):
    """Return ``(frame, relief, valid, features)``; features is None if the patch is unusable."""
    hint = None if normals is None else normals.mean(axis=0)
    frame = fit_frame(points, hint, origin=centre)
    if normals is not None:
        keep = normals @ frame.n > params.normal_cos
        points = points[keep]
    uvw = frame.to_local(points)
    height, valid = rasterize(uvw, spacing, radius)
    disk = np.pi * radius**2 / spacing**2
    if valid.sum() < params.min_valid * disk:
        return frame, None, valid, None
    relief = detrend(height, valid, spacing, params.highpass_sigma)
    return frame, relief, valid, ridge_features(relief, valid, spacing, params.band)


def detect(mesh: Mesh, params: DetectParams | None = None, normals: np.ndarray | None = None,
           progress=None) -> Detection:
    params = params or DetectParams()
    V = mesh.vertices
    if normals is None and mesh.faces is not None:
        normals = mesh.vertex_normals()
    spacing = params.grid_spacing or max(mesh.median_edge_length(), params.min_grid_spacing)
    tree = cKDTree(V)
    centres_idx = sample_centres(V, params.stride)

    patches: list[PatchResult] = []
    for k, ci in enumerate(centres_idx):
        if progress and k % 50 == 0:
            progress(k, len(centres_idx))
        c = V[ci]
        idx = tree.query_ball_point(c, params.patch_radius)
        if len(idx) < 50:
            continue
        idx = np.asarray(idx)
        frame, _, valid, feats = analyse_patch(V[idx], None if normals is None else normals[idx], c,
                                               params.patch_radius, spacing, params)
        disk = np.pi * params.patch_radius**2 / spacing**2
        score = ridge_score(feats) if feats is not None else 0.0
        patches.append(PatchResult(c, frame.n, float(valid.sum() / disk), feats, score))
    if progress:
        progress(len(centres_idx), len(centres_idx))

    vertex_score = _vertex_scores(V, patches, params.patch_radius)
    candidates = _group_candidates(patches, params)
    return Detection(params, spacing, patches, vertex_score, candidates)


def _vertex_scores(V: np.ndarray, patches: list[PatchResult], radius: float) -> np.ndarray:
    if not patches:
        return np.zeros(len(V))
    C = np.array([p.centre for p in patches])
    S = np.array([p.score for p in patches])
    k = min(8, len(C))
    d, j = cKDTree(C).query(V, k=k, distance_upper_bound=radius)
    d, j = np.atleast_2d(d.T).T, np.atleast_2d(j.T).T
    ok = np.isfinite(d)
    w = np.where(ok, np.exp(-0.5 * (d / (radius / 2)) ** 2), 0.0)
    s = np.where(ok, S[np.minimum(j, len(S) - 1)], 0.0)
    return (w * s).sum(1) / np.maximum(w.sum(1), 1e-12)


def _group_candidates(patches: list[PatchResult], params: DetectParams) -> list[Candidate]:
    hot = [i for i, p in enumerate(patches) if p.score >= params.threshold]
    if not hot:
        return []
    C = np.array([patches[i].centre for i in hot])
    pairs = cKDTree(C).query_pairs(params.stride * 1.6, output_type="ndarray")
    n = len(hot)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        coo_matrix((n, n))
    _, labels = connected_components(adj, directed=False)
    out = []
    for lab in np.unique(labels):
        members = [hot[i] for i in np.where(labels == lab)[0]]
        s = np.array([patches[i].score for i in members])
        cs = np.array([patches[i].centre for i in members])
        ns = np.array([patches[i].normal for i in members])
        w = s / s.sum()
        best = members[int(np.argmax(s))]
        nrm = (ns * w[:, None]).sum(0)
        f = [patches[i].features for i in members]
        period = float(np.average([x.ridge_period_mm for x in f], weights=s))
        straight = float(np.average([x.straightness for x in f], weights=s))
        out.append(Candidate(centre=patches[best].centre, normal=nrm / np.linalg.norm(nrm),
                             score=float(s.max()), mean_score=float(s.mean()), n_patches=len(members),
                             area_mm2=len(members) * params.stride**2, ridge_period_mm=period,
                             straightness=straight, patch_ids=members))
    out.sort(key=lambda c: c.score * np.sqrt(c.n_patches), reverse=True)
    return out
