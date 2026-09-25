"""Quick orthographic renderings of a mesh with a score overlay (fat-cross layout).

Views assume the GigaMesh orientation convention: obverse ("front") faces +Z,
X points right and Y up. Point splatting with a z-buffer is enough at scan
resolution and avoids an OpenGL dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# (name, view direction towards the viewer, image-right axis, image-up axis)
VIEWS = {
    "front": (np.array([0, 0, 1.0]), np.array([1, 0, 0.0]), np.array([0, 1, 0.0])),
    "back": (np.array([0, 0, -1.0]), np.array([1, 0, 0.0]), np.array([0, -1, 0.0])),
    "top": (np.array([0, 1, 0.0]), np.array([1, 0, 0.0]), np.array([0, 0, -1.0])),
    "bottom": (np.array([0, -1, 0.0]), np.array([1, 0, 0.0]), np.array([0, 0, 1.0])),
    "left": (np.array([-1, 0, 0.0]), np.array([0, 0, 1.0]), np.array([0, 1, 0.0])),
    "right": (np.array([1, 0, 0.0]), np.array([0, 0, -1.0]), np.array([0, 1, 0.0])),
}

_CMAP = np.array([  # dark purple -> red -> yellow
    [0.00, 40, 11, 84], [0.35, 150, 30, 110], [0.65, 230, 90, 40], [1.00, 250, 230, 60]])


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def colormap(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return np.column_stack([np.interp(t, _CMAP[:, 0], _CMAP[:, k]) for k in (1, 2, 3)])


def _zbuffer(V, N, view, px_mm, bounds=None):
    """Nearest facing vertex per pixel: ``(vertex ids, pixel ids, (w, h), bounds)``."""
    d, right, up = VIEWS[view]
    x, y, depth = V @ right, V @ up, V @ d
    if bounds is None:
        bounds = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    x0, x1, y0, y1 = bounds
    w, h = int((x1 - x0) / px_mm) + 1, int((y1 - y0) / px_mm) + 1
    facing = N @ d > 0
    ix = ((x - x0) / px_mm).astype(np.int64)
    iy = ((y1 - y) / px_mm).astype(np.int64)
    ok = facing & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    pix = iy[ok] * w + ix[ok]
    order = np.lexsort((depth[ok], pix))
    last = np.ones(len(order), bool)
    last[:-1] = pix[order[1:]] != pix[order[:-1]]
    return np.where(ok)[0][order[last]], pix[order[last]], (w, h), bounds


def render_layers(V, N, scalar, view, px_mm=0.1, lo=0.05, hi=0.35, base=None, bounds=None):
    """One orthographic view as separate layers.

    Returns ``(image, heat, bounds)``: the base image (RGB; the shaded surface,
    or per-vertex ``base`` grey levels in [0, 1] such as an MSII map), a heat
    overlay (RGBA, None without ``scalar``) that composites on top of it, and
    the view bounds ``(x0, x1, y0, y1)`` in view coordinates, for placing markers.
    """
    d, right, up = VIEWS[view]
    sel, p, (w, h), bounds = _zbuffer(V, N, view, px_mm, bounds)
    light = np.array([-0.4, 0.5, 0.75])
    light = light[0] * right + light[1] * up + light[2] * d
    light /= np.linalg.norm(light)
    shade = np.clip(N[sel] @ light, 0, 1) * 0.8 + 0.2
    gray = shade * 235 if base is None else np.clip(base[sel], 0, 1) * 255
    drawn = np.zeros(h * w, bool)
    drawn[p] = True
    img = np.full((h * w, 3), 255, np.uint8)
    img[p] = np.repeat(gray[:, None], 3, axis=1).astype(np.uint8)
    img = _fill_gaps(img.reshape(h, w, 3), drawn.reshape(h, w))
    heat = None
    if scalar is not None:
        t = (scalar[sel] - lo) / (hi - lo)
        rgba = np.zeros((h * w, 4), np.uint8)
        rgba[p, :3] = (colormap(t) * shade[:, None]).astype(np.uint8)
        rgba[p, 3] = (np.clip(t * 1.5, 0, 0.85) * 255).astype(np.uint8)
        heat = _fill_gaps(rgba.reshape(h, w, 4), drawn.reshape(h, w))
    return img, heat, bounds


def render_view(V, N, scalar, view, px_mm=0.1, lo=0.05, hi=0.35, bounds=None, base=None):
    img, heat, _ = render_layers(V, N, scalar, view, px_mm, lo, hi, base, bounds)
    if heat is None:
        return img
    a = heat[..., 3:4] / 255.0
    return (img * (1 - a) + heat[..., :3] * a).astype(np.uint8)


def _fill_gaps(img: np.ndarray, drawn: np.ndarray, max_gap_px: int = 4) -> np.ndarray:
    """Fill splatting gaps inside the object outline with the nearest drawn pixel."""
    from scipy import ndimage

    outline = ndimage.binary_closing(drawn, iterations=max_gap_px, border_value=0)
    holes = outline & ~drawn
    if not holes.any():
        return img
    _, (iy, ix) = ndimage.distance_transform_edt(~drawn, return_indices=True)
    img[holes] = img[iy[holes], ix[holes]]
    return img


def fatcross_layout(sizes: dict, head: int = 0, gap: int = 8):
    """Pixel origin of each view on the fat-cross canvas and the canvas size;
    ``sizes`` maps view -> (height, width)."""
    fh, fw = sizes["front"]
    lw, rw = sizes["left"][1], sizes["right"][1]
    th, bh, kh = sizes["top"][0], sizes["bottom"][0], sizes["back"][0]
    cx = gap + lw + gap
    y = head + gap
    origin = {"top": (cx, y)}; y += th + gap
    origin.update(left=(gap, y), front=(cx, y), right=(cx + fw + gap, y)); y += fh + gap
    origin["bottom"] = (cx, y); y += bh + gap
    origin["back"] = (cx, y)
    return origin, (lw + fw + rw + 4 * gap, head + th + fh + bh + kh + 5 * gap)


def compose_fatcross(images: dict, head: int = 0, transparent: bool = False):
    """Paste the six view images into the fat-cross layout: ``(canvas, origins)``."""
    origin, size = fatcross_layout({k: im.shape[:2] for k, im in images.items()}, head)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0)) if transparent else Image.new("RGB", size, "white")
    for view, xy in origin.items():
        canvas.paste(Image.fromarray(images[view]), xy)
    return canvas, origin


def marker_pixel(point, view, origin, bounds, px_mm):
    """Canvas pixel of a 3D point in the given view."""
    _, right, up = VIEWS[view]
    return (origin[0] + (point @ right - bounds[0]) / px_mm, origin[1] + (bounds[3] - point @ up) / px_mm)


def render_fatcross(V, N, scalar, path, px_mm=0.1, lo=0.05, hi=0.35, markers=None, title=None):
    """Save a fat-cross overview: top / (left, front, right) / bottom / back."""
    views, bounds = {}, {}
    for k in VIEWS:
        img, heat, bounds[k] = render_layers(V, N, scalar, k, px_mm, lo, hi)
        if heat is not None:
            a = heat[..., 3:4] / 255.0
            img = (img * (1 - a) + heat[..., :3] * a).astype(np.uint8)
        views[k] = img
    canvas, origin = compose_fatcross(views, head=30 if title else 0)
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    if title:
        draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for label, c, nrm in markers or []:
        # label each candidate on the view it faces most directly
        view = max(VIEWS, key=lambda k: nrm @ VIEWS[k][0])
        px, py = marker_pixel(np.asarray(c), view, origin[view], bounds[view], px_mm)
        draw.ellipse((px - 12, py - 12, px + 12, py + 12), outline=(0, 120, 255), width=2)
        draw.text((px + 14, py - 8), label, fill=(0, 90, 220), font=font)
    canvas.save(str(path))
    return path
