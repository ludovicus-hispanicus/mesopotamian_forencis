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


def render_view(V, N, scalar, view, px_mm=0.1, lo=0.05, hi=0.35, bounds=None):
    d, right, up = VIEWS[view]
    x, y, depth = V @ right, V @ up, V @ d
    if bounds is None:
        bounds = (x.min(), x.max(), y.min(), y.max())
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
    sel = np.where(ok)[0][order[last]]
    p = pix[order[last]]

    light = np.array([-0.4, 0.5, 0.75])
    light = light[0] * right + light[1] * up + light[2] * d
    light /= np.linalg.norm(light)
    shade = np.clip(N[sel] @ light, 0, 1) * 0.8 + 0.2
    rgb = np.repeat(shade[:, None] * 235, 3, axis=1)
    if scalar is not None:
        t = (scalar[sel] - lo) / (hi - lo)
        alpha = np.clip(t * 1.5, 0, 0.85)[:, None]
        rgb = rgb * (1 - alpha) + colormap(t) * shade[:, None] * alpha
    img = np.full((h * w, 3), 255, np.uint8)
    img[p] = rgb.astype(np.uint8)
    img = img.reshape(h, w, 3)
    return _fill_gaps(img, np.isin(np.arange(h * w), p).reshape(h, w))


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
    return img


def render_fatcross(V, N, scalar, path, px_mm=0.1, lo=0.05, hi=0.35, markers=None, title=None):
    """Save a fat-cross overview: top / (left, front, right) / bottom / back."""
    views = {k: render_view(V, N, scalar, k, px_mm, lo, hi) for k in VIEWS}
    gap = 8
    fh, fw = views["front"].shape[:2]
    lw, rw = views["left"].shape[1], views["right"].shape[1]
    th, bh, kh = views["top"].shape[0], views["bottom"].shape[0], views["back"].shape[0]
    head = 30 if title else 0
    H = head + th + fh + bh + kh + 5 * gap
    W = lw + fw + rw + 4 * gap
    canvas = Image.new("RGB", (W, H), "white")
    cx = gap + lw + gap
    y = head + gap
    canvas.paste(Image.fromarray(views["top"]), (cx, y)); y += th + gap
    canvas.paste(Image.fromarray(views["left"]), (gap, y))
    canvas.paste(Image.fromarray(views["front"]), (cx, y))
    canvas.paste(Image.fromarray(views["right"]), (cx + fw + gap, y)); y += fh + gap
    canvas.paste(Image.fromarray(views["bottom"]), (cx, y)); y += bh + gap
    canvas.paste(Image.fromarray(views["back"]), (cx, y))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    if title:
        draw.text((gap, 8), title, fill=(0, 0, 0), font=font)
    if markers:
        # label candidates on the front view if they face it, else on the back
        x0 = (V @ VIEWS["front"][1]).min()
        y1 = (V @ VIEWS["front"][2]).max()
        fy = head + gap + th + gap
        by = fy + fh + gap + bh + gap
        bx0 = (V @ VIEWS["back"][1]).min()
        by1 = (V @ VIEWS["back"][2]).max()
        for label, c, nrm in markers:
            for view, ox, oy, xm, ym in (("front", cx, fy, x0, y1), ("back", cx, by, bx0, by1)):
                d, right, up = VIEWS[view]
                if nrm @ d > 0.3:
                    px = ox + (c @ right - xm) / px_mm
                    py = oy + (ym - c @ up) / px_mm
                    draw.ellipse((px - 12, py - 12, px + 12, py + 12), outline=(0, 120, 255), width=2)
                    draw.text((px + 14, py - 8), label, fill=(0, 90, 220), font=font)
                    break
    canvas.save(str(path))
    return path
