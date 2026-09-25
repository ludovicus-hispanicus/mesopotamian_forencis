"""Ridge skeleton and minutiae from an enhanced print, plus reviewer traces.

The enhanced (Gabor) image is binarised inside the mask (positive = groove in
the clay = finger ridge), thinned to one-pixel lines, pruned of short spurs and
vectorised into polylines. Ridge endings (one neighbour on the skeleton) and
bifurcations (three) are the minutiae; those close to the mask border are
dropped, as the standard practice is, because the border cuts ridges.

Reviewer strokes (:func:`apply_traces`) come in two kinds: *guide* strokes only
steer the enhancement (see :func:`mesoprint.enhance.gabor_enhance`), *trace*
strokes are the ridge lines themselves and replace the automatic skeleton
within a band around them, so the database knows which lines were detected and
which were drawn.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

_NB = np.ones((3, 3), int)
_NB[1, 1] = 0


def _neighbours(skel: np.ndarray) -> np.ndarray:
    return ndimage.convolve(skel.astype(int), _NB, mode="constant")


def prune(skel: np.ndarray, min_px: int) -> np.ndarray:
    """Remove end branches shorter than ``min_px`` (iteratively erodes endpoints,
    then regrows from what survived so long branches keep their full length)."""
    s = skel.copy()
    for _ in range(min_px):
        ends = s & (_neighbours(s) <= 1)
        if not ends.any():
            break
        s &= ~ends
    # regrow: add back skeleton pixels connected to the pruned core
    core = s.copy()
    for _ in range(min_px):
        grow = skel & ~core & (ndimage.convolve(core.astype(int), _NB, mode="constant") > 0)
        if not grow.any():
            break
        core |= grow
    return core


def ridge_skeleton(enhanced: np.ndarray, mask: np.ndarray, spacing: float, min_branch_mm: float = 0.3,
                   min_len_mm: float = 0.6) -> np.ndarray:
    """One-pixel-wide ridge (clay groove) lines inside ``mask``."""
    grooves = mask & (ndimage.gaussian_filter(enhanced, 0.03 / spacing) > 0)
    grooves = ndimage.binary_opening(grooves, iterations=max(1, int(0.05 / spacing)))
    skel = skeletonize(grooves)
    skel = prune(skel, int(min_branch_mm / spacing))
    # drop isolated fragments shorter than min_len_mm
    lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(np.ones_like(skel, float), lab, np.arange(1, n + 1))
        skel &= np.isin(lab, 1 + np.flatnonzero(sizes >= min_len_mm / spacing))
    return skel


def minutiae(skel: np.ndarray, mask: np.ndarray, spacing: float, border_mm: float = 0.6,
             merge_mm: float = 0.25) -> list[dict]:
    """Ridge endings and bifurcations, away from the mask border, as
    ``{"x", "y", "type", "angle"}`` in grid coordinates (row = y) and degrees."""
    nb = _neighbours(skel)
    inner = ndimage.binary_erosion(mask, iterations=max(1, int(border_mm / spacing)))
    pts = []
    for kind, cond in (("ending", nb == 1), ("bifurcation", nb >= 3)):
        for y, x in np.argwhere(skel & cond & inner):
            pts.append({"x": int(x), "y": int(y), "type": kind})
    # merge clusters of bifurcation pixels (a junction can span 2-3 pixels)
    keep, r = [], merge_mm / spacing
    for p in pts:
        if any(q["type"] == p["type"] and np.hypot(q["x"] - p["x"], q["y"] - p["y"]) < r for q in keep):
            continue
        keep.append(p)
    for p in keep:
        p["angle"] = _direction(skel, p["x"], p["y"], int(round(0.3 / spacing)))
    return keep


def _direction(skel: np.ndarray, x: int, y: int, r: int) -> float:
    """Direction (degrees, x = columns, y = rows) of the skeleton around a point."""
    y0, y1, x0, x1 = max(0, y - r), y + r + 1, max(0, x - r), x + r + 1
    yy, xx = np.nonzero(skel[y0:y1, x0:x1])
    if len(xx) < 3:
        return 0.0
    dx, dy = xx - (x - x0), yy - (y - y0)
    return float(np.degrees(0.5 * np.arctan2(2 * (dx * dy).sum(), (dx * dx - dy * dy).sum())))


def polylines(skel: np.ndarray, min_pts: int = 4) -> list[list[list[int]]]:
    """Vectorise the skeleton into chains ``[[x, y], ...]`` between junctions and ends."""
    nb = _neighbours(skel)
    node = skel & (nb != 2)  # ends and junctions
    visited = np.zeros_like(skel)
    out = []
    h, w = skel.shape

    def step(y, x, py, px):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx] \
                        and not visited[y + dy, x + dx] and (y + dy, x + dx) != (py, px):
                    yield y + dy, x + dx

    starts = [tuple(p) for p in np.argwhere(node)]
    for sy, sx in starts:
        for ny, nx in list(step(sy, sx, -1, -1)):
            if visited[ny, nx]:
                continue
            chain = [[int(sx), int(sy)], [int(nx), int(ny)]]
            visited[ny, nx] = True
            py, px, y, x = sy, sx, ny, nx
            while not node[y, x]:
                nxt = [q for q in step(y, x, py, px)]
                if not nxt:
                    break
                py, px = y, x
                y, x = nxt[0]
                visited[y, x] = True
                chain.append([int(x), int(y)])
            if len(chain) >= min_pts:
                out.append(chain)
    visited |= node
    # closed loops without any node
    for y, x in np.argwhere(skel & ~visited):
        if visited[y, x]:
            continue
        chain = [[int(x), int(y)]]
        visited[y, x] = True
        py, px = -1, -1
        while True:
            nxt = [q for q in step(y, x, py, px)]
            if not nxt:
                break
            py, px = y, x
            y, x = nxt[0]
            visited[y, x] = True
            chain.append([int(x), int(y)])
        if len(chain) >= min_pts:
            out.append(chain)
    return out


def ridge_spacing(lines: list[list[list[int]]], shape, spacing: float, lo_mm: float = 0.25,
                  hi_mm: float = 1.2) -> dict:
    """Ridge breadth from the skeleton: for every pixel of every line, the distance
    to the nearest *other* line (perpendicular to the ridge, except at ends and
    junctions, which the ``lo_mm``/``hi_mm`` limits exclude). Returns the median,
    quartiles, and the number of pixels measured."""
    if len(lines) < 2:
        return {"mean_ridge_breadth_mm": None, "n": 0}
    lab = np.zeros(shape, np.int32)
    for i, pts in enumerate(lines, 1):
        for x, y in pts:
            lab[y, x] = i
    d_all = []
    for i in range(1, len(lines) + 1):
        others = (lab > 0) & (lab != i)
        if not others.any():
            continue
        dist = ndimage.distance_transform_edt(~others) * spacing
        d = dist[lab == i]
        d_all.append(d[(d >= lo_mm) & (d <= hi_mm)])
    d = np.concatenate(d_all) if d_all else np.array([])
    if d.size < 20:
        return {"mean_ridge_breadth_mm": None, "n": int(d.size)}
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    return {"mean_ridge_breadth_mm": round(float(med), 3), "q1": round(float(q1), 3), "q3": round(float(q3), 3),
            "n": int(d.size), "length_mm": round(float(d.size) * spacing, 1)}


def rasterise_strokes(strokes: list[dict], shape, width_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Band around trace strokes and their tangent direction (radians) per pixel."""
    band = np.zeros(shape, bool)
    theta = np.zeros(shape, float)
    for s in strokes:
        pts = np.asarray(s["pts"], float)
        if len(pts) < 2:
            continue
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0)) + 1)
            xs = np.clip(np.round(np.linspace(x0, x1, n)).astype(int), 0, shape[1] - 1)
            ys = np.clip(np.round(np.linspace(y0, y1, n)).astype(int), 0, shape[0] - 1)
            line = np.zeros(shape, bool)
            line[ys, xs] = True
            seg = ndimage.binary_dilation(line, iterations=width_px) if width_px > 0 else line  # 0 would mean "until stable"
            band |= seg
            theta[seg] = np.arctan2(y1 - y0, x1 - x0)
    return band, theta


def apply_traces(skel: np.ndarray, strokes: list[dict], spacing: float, band_mm: float = 0.35):
    """Replace the automatic skeleton by the reviewer's trace strokes within a band
    around them. Returns ``(skeleton, traced polylines)``."""
    traces = [s for s in strokes if s.get("mode") == "trace" and len(s.get("pts", [])) >= 2]
    if not traces:
        return skel, []
    band, _ = rasterise_strokes(traces, skel.shape, max(1, int(band_mm / spacing)))
    out = skel & ~band
    lines = []
    for s in traces:
        pts = [[int(round(x)), int(round(y))] for x, y in s["pts"]]
        lines.append(pts)
        line, _ = rasterise_strokes([s], skel.shape, 0)
        out |= line
    return out, lines
