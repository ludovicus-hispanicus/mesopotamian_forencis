"""Local review app for the detection results.

``mesoprint review [--results results/batch_r100] [--assets assets/3D-Models]``
starts a small web server on this PC only (nothing leaves the machine) and
opens the browser. For each tablet it shows the fat-cross overview (shaded
surface or MSII, with the score heat map) and, per candidate, the flattened
MSII / relief / enhanced images with the print mask. The mask can be
re-extracted at a larger radius or another periodicity level, painted larger
or smaller, and measured: an automatic spectral ridge breadth, and a manual
ridge-count line as in Fowler et al. 2020. Each candidate gets a verdict.
Missed prints are added by clicking on the overview.

Verdicts, edited masks and measurements go to ``<results>/SM_<id>/review.json``
and ``<results>/SM_<id>/review/``; ``scripts/collect_reviews.py`` gathers them.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from .catalogue import LABELS, NOTES, PERIODS, info
from .detect import vertex_scores
from .enhance import gabor_enhance
from .extract import Extraction, extract_region, fade_enhancement, groove_dark8, mask_overlay, to_uint8
from .features import ridge_features
from .mesh import Mesh, load_ply
from .msii import file_msii
from .periodicity import ridge_periodicity
from .render import VIEWS, compose_fatcross, marker_pixel, render_layers
from .skeleton import apply_traces, minutiae, polylines, rasterise_strokes, ridge_skeleton, ridge_spacing

REASONS = ["crack or ruling", "wedge", "clay texture", "textile or mat", "scanner waves", "edge or break", "other"]
FACE = {"front": "obverse", "back": "reverse", "top": "top", "bottom": "bottom", "left": "left", "right": "right"}
#: Periods 0.28-0.83 mm: wider than the detection band, so broad ridges are not pinned at its edge.
MEASURE_BAND = (1.2, 3.6)
PPI = 1000.0
ADDED_FROM = 100  # ids of candidates added by the reviewer


def _unmirror_legacy(results: Path) -> float:
    """Print and clean images saved before 2026-09-25 were mirrored; flip them back
    once (marker file) and return the marker's time, so older clean images regenerate."""
    marker = results / ".views_unmirrored"
    if not marker.exists():
        for p in results.glob("SM_*/review/cand*_print.png"):
            im = Image.open(p)
            info = im.info.get("dpi", (PPI, PPI))
            im.transpose(Image.FLIP_LEFT_RIGHT).save(p, dpi=info)
        marker.write_text("print images flipped to tablet orientation")
    return marker.stat().st_mtime


@dataclass
class Tablet:
    tid: str
    folder: Path
    summary: dict
    mesh: Mesh
    normals: np.ndarray
    tree: cKDTree
    msii: np.ndarray | None
    msii_desc: str
    layout: dict

    @property
    def review_file(self) -> Path:
        return self.folder / "review.json"

    def review(self) -> dict:
        r = {"candidates": {}, "added": [], "moved": {}}
        if self.review_file.exists():
            r.update(json.loads(self.review_file.read_text()))
        return r

    def save_review(self, r: dict) -> None:
        self.review_file.write_text(json.dumps(r, indent=2))

    def candidates(self) -> list[dict]:
        r = self.review()
        out = [dict(c) for c in self.summary["candidates"]] + [dict(a) for a in r["added"]]
        for c in out:  # centres dragged by the reviewer override the detected ones
            if str(c["id"]) in r["moved"]:
                c.update(r["moved"][str(c["id"])], moved=True)
        return out

    def candidate(self, cid: int) -> dict:
        for c in self.candidates():
            if c["id"] == cid:
                return c
        raise KeyError(cid)

    def marker(self, c: dict) -> dict:
        n = np.asarray(c["normal"])
        view = max(VIEWS, key=lambda k: n @ VIEWS[k][0])
        L = self.layout["views"][view]
        px, py = marker_pixel(np.asarray(c["centre"]), view, (L["ox"], L["oy"]),
                              (L["x0"], L["x1"], L["y0"], L["y1"]), self.layout["px_mm"])
        return {"view": view, "face": FACE[view], "px": float(px), "py": float(py)}

    def pick(self, view: str, px: float, py: float):
        """3D point and local normal under an overview pixel, or None off the object."""
        d, right, up = VIEWS[view]
        L, mm = self.layout["views"][view], self.layout["px_mm"]
        xw, yw = L["x0"] + (px - L["ox"]) * mm, L["y1"] - (py - L["oy"]) * mm
        V, N = self.mesh.vertices, self.normals
        ok = (N @ d > 0.2) & (np.abs(V @ right - xw) < 0.3) & (np.abs(V @ up - yw) < 0.3)
        if not ok.any():
            return None
        idx = np.flatnonzero(ok)
        i = idx[np.argmax(V[idx] @ d)]  # nearest to the viewer
        n = N[self.tree.query_ball_point(V[i], 1.0)].mean(axis=0)
        return V[i], n / np.linalg.norm(n)


@dataclass
class Extracted:
    key: str
    tid: str
    cid: int
    ex: Extraction
    params: dict
    enhanced_guided: np.ndarray | None = None  # enhancement recomputed under reviewer guide strokes
    skeleton: dict | None = None  # last skeleton result (polylines, minutiae), image coordinates

    @property
    def enhanced(self) -> np.ndarray:
        return self.ex.enhanced if self.enhanced_guided is None else self.enhanced_guided


class Session:
    def __init__(self, results: Path, assets: Path):
        self.results, self.assets = results, assets
        self.tablet: Tablet | None = None
        self.extractions: dict[str, Extracted] = {}
        self.lock = threading.Lock()

    def tablet_dirs(self) -> list[Path]:
        return sorted(d for d in self.results.glob("SM_*") if (d / "candidates.json").exists())

    def mesh_path(self, summary: dict) -> Path:
        return self.assets / f"{summary['mesh']['name']}.ply"

    def load(self, tid: str) -> Tablet:
        if self.tablet is not None and self.tablet.tid == tid:
            return self.tablet
        folder = self.results / f"SM_{tid}"
        summary = json.loads((folder / "candidates.json").read_text())
        path = self.mesh_path(summary)
        if not path.exists():
            raise FileNotFoundError(f"mesh not found: {path}")
        self.tablet = None
        self.extractions.clear()
        mesh = load_ply(path)
        normals = mesh.vertex_normals()
        got = file_msii(mesh, summary["params"].get("msii_radius", 0.3))
        msii, desc = (got[0], f"GigaMesh file, r = {got[1]:.4f} mm") if got else (None, "computed per patch")
        t = Tablet(tid, folder, summary, mesh, normals, cKDTree(mesh.vertices), msii, desc, {})
        t.layout = _overviews(t)
        self.tablet = t
        return t

    def extract(self, t: Tablet, cid: int, radius: float, level: float) -> Extracted:
        key = f"{t.tid}_{cid}_{radius:g}_{level:g}"
        if key not in self.extractions:
            c = t.candidate(cid)
            ex = extract_region(t.mesh, c["centre"], radius, 25.4 / PPI, t.normals, t.tree,
                                mask_level=level, msii_values=t.msii)
            self.extractions[key] = Extracted(key, t.tid, cid, ex, {"radius_mm": radius, "mask_level": level})
        return self.extractions[key]


def _overviews(t: Tablet, px_mm: float = 0.05) -> dict:
    """Render (once, cached on disk) the overview layers: shaded surface, MSII
    grey levels, and the score heat map as a transparent overlay."""
    out = t.folder / "review"
    out.mkdir(exist_ok=True)
    layout_file = out / "layout.json"
    if layout_file.exists():
        layout = json.loads(layout_file.read_text())
        if layout.get("version") == 3 and all((out / f"overview_{k}.png").exists() for k in layout["layers"]):
            return layout
    V, N = t.mesh.vertices, t.normals
    thr = t.summary["params"]["threshold"]
    rows = list(csv.DictReader(open(t.folder / "patches.csv")))
    C = np.array([[float(r["cx"]), float(r["cy"]), float(r["cz"])] for r in rows])
    S = np.array([float(r["score"]) for r in rows])
    score = vertex_scores(V, C, S, t.summary["params"]["patch_radius"])
    base = None
    if t.msii is not None:
        p1, p99 = np.percentile(t.msii, [1, 99])
        base = (t.msii - p1) / (p99 - p1)  # light = groove
    surf, heat, msii, msii_inv, views = {}, {}, {}, {}, {}
    for k in VIEWS:
        surf[k], heat[k], b = render_layers(V, N, score, k, px_mm, 0.5 * thr, 2.0 * thr)
        if base is not None:
            msii[k] = render_layers(V, N, None, k, px_mm, base=base, bounds=b)[0]
            msii_inv[k] = render_layers(V, N, None, k, px_mm, base=1.0 - base, bounds=b)[0]  # grooves dark
        views[k] = {"x0": b[0], "x1": b[1], "y0": b[2], "y1": b[3], "w": surf[k].shape[1], "h": surf[k].shape[0]}
    canvas, origin = compose_fatcross(surf)
    canvas.save(out / "overview_surface.png")
    compose_fatcross(heat, transparent=True)[0].save(out / "overview_heat.png")
    layers = ["surface", "heat"]
    if base is not None:
        compose_fatcross(msii)[0].save(out / "overview_msii.png")
        compose_fatcross(msii_inv)[0].save(out / "overview_msii_inv.png")
        layers += ["msii", "msii_inv"]
    for k, (ox, oy) in origin.items():
        views[k].update(ox=ox, oy=oy)
    layout = {"version": 3, "views": views, "px_mm": px_mm, "width": canvas.width, "height": canvas.height,
              "layers": layers, "heat_range": [0.5 * thr, 2.0 * thr]}
    layout_file.write_text(json.dumps(layout))
    return layout


def measure(ex: Extraction, mask: np.ndarray) -> dict:
    """Ridge measurements inside ``mask`` (or the whole region if it is empty)."""
    region = mask if mask.any() else ex.valid
    f = ridge_features(np.where(region, ex.msii, 0.0), region, ex.spacing, MEASURE_BAND)
    g = ridge_features(ex.relief, region, ex.spacing, MEASURE_BAND)
    return {"mean_ridge_breadth_mm": round(f.ridge_period_mm, 3), "ridge_count_per_5mm": round(5 * f.peak_freq, 1),
            "ridge_amplitude_um_rms": round(g.band_amp_um, 1), "straightness": round(f.straightness, 2),
            "area_mm2": round(float(region.sum()) * ex.spacing**2, 1), "isolated": bool(mask.any())}


def _save_skeleton_png(path: Path, base8: np.ndarray, skel: dict) -> None:
    """The MSII image (upright) with detected lines green, traced magenta, minutiae marked."""
    from PIL import ImageDraw

    im = Image.fromarray(np.flipud(base8)).convert("RGB")
    d = ImageDraw.Draw(im)
    for p in skel["polylines"]:
        d.line([tuple(q) for q in p["pts"]], fill=(0, 190, 80) if p["source"] == "detected" else (220, 40, 200), width=2)
    for m in skel["minutiae"]:
        c = (230, 40, 40) if m["type"] == "ending" else (40, 110, 240)
        d.ellipse((m["x"] - 5, m["y"] - 5, m["x"] + 5, m["y"] + 5), outline=c, width=2)
    im.save(path, dpi=(PPI, PPI))


def _save_clean_png(path: Path, skel: dict, size: int, mm_per_px: float, breadth_mm: float | None) -> None:
    """A clean fingerprint: the skeleton lines drawn with a ridge width (about
    half the ridge period), black on white, as seen on the tablet."""
    from PIL import ImageDraw

    width = max(3, int(round(0.5 * (breadth_mm or 0.55) / mm_per_px)))
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    for p in skel["polylines"]:
        pts = [tuple(q) for q in p["pts"]]
        d.line(pts, fill=0, width=width, joint="curve")
        for q in (pts[0], pts[-1]):
            d.ellipse((q[0] - width / 2, q[1] - width / 2, q[0] + width / 2, q[1] + width / 2), fill=0)
    im.save(path, dpi=(PPI, PPI))


def _png(arr8: np.ndarray) -> bytes:
    """PNG of a grid array, flipped so +v points up (like the saved extractions)."""
    buf = io.BytesIO()
    Image.fromarray(np.flipud(arr8)).save(buf, "PNG", dpi=(PPI, PPI))
    return buf.getvalue()


def layer_png(e: Extracted, layer: str) -> bytes:
    ex = e.ex
    if layer == "msii":  # dark = groove, like the relief
        return _png(groove_dark8(ex.msii, ex.valid, 2, 98))
    if layer == "relief":
        return _png(to_uint8(ex.relief, ex.valid))
    if layer == "enhanced":
        return _png(groove_dark8(e.enhanced, ex.valid, 0.5, 99.5))
    if layer == "mask":
        return _png((ex.mask * 255).astype(np.uint8))
    if layer == "valid":
        return _png((ex.valid * 255).astype(np.uint8))
    raise KeyError(layer)


def decode_mask(data_url: str, shape) -> np.ndarray:
    """Mask painted in the browser (PNG data URL, opaque = inside) as a grid array."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    a = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))[..., 3] > 127
    if a.shape != tuple(shape):
        raise ValueError(f"mask is {a.shape}, extraction is {tuple(shape)}")
    return np.flipud(a)


def _flip_pts(pts, n):
    """Browser image coordinates (row 0 = top) <-> grid coordinates (row 0 = bottom)."""
    return [[float(x), float(n - 1 - y)] for x, y in pts]


def compute_skeleton(e: Extracted, mask: np.ndarray, strokes: list[dict]) -> dict:
    """Enhancement under guide strokes, skeleton with trace strokes applied, minutiae.
    Strokes and results use browser image coordinates."""
    ex = e.ex
    n, sp = ex.valid.shape[0], ex.spacing
    grid_strokes = [{**s, "pts": _flip_pts(s["pts"], n)} for s in strokes if len(s.get("pts", [])) >= 2]
    guides = [s for s in grid_strokes if s.get("mode") == "guide"]
    ms_ok = ex.valid & (ex.msii != 0)
    if guides:
        band, theta = rasterise_strokes(guides, ex.valid.shape, max(1, int(0.5 / sp)))
        per = ridge_periodicity(ex.msii, ms_ok, sp, select="contrast")
        G = gabor_enhance(ex.msii, ms_ok, sp, per, guide=(band, theta))
        raw, e.enhanced_guided = fade_enhancement(G, ms_ok, per.contrast, sp, keep=band)
    else:
        e.enhanced_guided = None
        raw = ex.ridge_raw if ex.ridge_raw is not None else ex.enhanced
    # inside the reviewer's mask the unfaded response is used: the mask asserts ridges are there
    skel = ridge_skeleton(np.where(mask, raw, e.enhanced), mask, sp)
    skel, traced = apply_traces(skel, grid_strokes, sp)
    mn = minutiae(skel, mask, sp)
    auto = polylines(skel & ~rasterise_strokes([s for s in grid_strokes if s.get("mode") == "trace"], skel.shape, 0)[0])
    res = {"polylines": [{"pts": _flip_pts(p, n), "source": "detected"} for p in auto]
                        + [{"pts": _flip_pts(p, n), "source": "traced"} for p in traced],
           "minutiae": [{**m, "y": n - 1 - m["y"], "angle": -m["angle"]} for m in mn],
           "ridge_length_mm": round(float(skel.sum()) * sp, 1), "guided": bool(guides), "n_strokes": len(grid_strokes),
           "spacing": ridge_spacing(auto + traced, skel.shape, sp)}
    e.skeleton = res
    return res


def morph(mask: np.ndarray, mm: float, spacing: float, grow: bool) -> np.ndarray:
    r = max(1, int(round(mm / spacing)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = xx**2 + yy**2 <= r * r
    return ndimage.binary_dilation(mask, disk) if grow else ndimage.binary_erosion(mask, disk)


def create_app(results: Path, assets: Path):
    from flask import Flask, abort, jsonify, request, send_file

    app = Flask(__name__)
    S = Session(Path(results).resolve(), Path(assets).resolve())  # send_file needs absolute paths
    unmirrored_at = _unmirror_legacy(S.results)

    def tablet(tid: str) -> Tablet:
        with S.lock:
            try:
                return S.load(tid)
            except FileNotFoundError as e:
                abort(404, str(e))

    def png(data: bytes):
        return send_file(io.BytesIO(data), mimetype="image/png", max_age=0)

    @app.get("/")
    def index():
        return (Path(__file__).with_name("review.html")).read_text(encoding="utf-8")

    @app.get("/gallery")
    def gallery():
        """All reviewed prints with their views and results, one row each."""
        from .db import face_of

        rows = []
        for d in S.tablet_dirs():
            rf = d / "review.json"
            if not rf.exists():
                continue
            tid = d.name[3:]
            meta = info(tid)
            summary = json.loads((d / "candidates.json").read_text())
            review = json.loads(rf.read_text())
            cands = {c["id"]: dict(c) for c in summary["candidates"] + review.get("added", [])}
            for cid, mv in review.get("moved", {}).items():
                if int(cid) in cands:
                    cands[int(cid)].update(mv)
            counter: dict[str, int] = {}
            for cid_s, r in sorted(review["candidates"].items(), key=lambda kv: int(kv[0])):
                c = cands.get(int(cid_s), {})
                face = face_of(c.get("normal", [0, 0, 1]))
                counter[face] = counter.get(face, 0) + 1
                pid = f"SM{tid}_{face}_{counter[face]}"
                stem = f"cand{int(cid_s):02d}"
                m, man, sk = r.get("measurements", {}), r.get("manual") or {}, r.get("skeleton") or {}
                skel_json = d / "review" / f"{stem}_skeleton.json"
                if skel_json.exists() and (not (d / "review" / f"{stem}_clean.png").exists()
                                           or (d / "review" / f"{stem}_clean.png").stat().st_mtime < unmirrored_at):
                    s = json.loads(skel_json.read_text())
                    size = int(round(2 * (r.get("params", {}).get("radius_mm") or 8) / (25.4 / PPI))) + 1
                    _save_clean_png(d / "review" / f"{stem}_clean.png", s, size, 25.4 / PPI, sk.get("ridge_breadth_mm"))
                imgs = [(k, f"/api/tablet/{tid}/review/{stem}_{k}.png") for k in ("msii", "mask", "print", "skeleton", "clean")
                        if (d / "review" / f"{stem}_{k}.png").exists()]
                facts = [f"verdict <b>{r['verdict']}</b>" + (f" ({r['reason']})" if r.get("reason") else ""),
                         f"{face}, candidate #{cid_s}" + (f", score {c['score']:.2f}" if c.get("score") is not None else ", added by reviewer"),
                         f"mask {m.get('area_mm2', '–')} mm²",
                         f"ridge breadth: manual {man.get('mean_ridge_breadth_mm', '–')} · skeleton {sk.get('ridge_breadth_mm', '–')} · spectral {m.get('mean_ridge_breadth_mm', '–')} mm",
                         f"ridges per 5 mm {m.get('ridge_count_per_5mm', '–')} · depth {m.get('ridge_amplitude_um_rms', '–')} µm",
                         f"minutiae {sk.get('n_minutiae', '–')} · ridge {sk.get('ridge_length_mm', '–')} mm" + (" · guided" if sk.get("guided") else ""),
                         f"{r.get('reviewer', '')} {r.get('time', '')}" + (f" · {r['note']}" if r.get("note") else "")]
                rows.append((meta, tid, pid, facts, imgs))

        # arrange: period (chronological) -> provenience -> tablet -> print
        def key(row):
            meta, tid, pid = row[0], row[1], row[2]
            per = PERIODS.index(meta["period"]) if meta["period"] in PERIODS else len(PERIODS)
            return (per, meta["provenience"] == "", meta["provenience"], tid, pid)

        rows.sort(key=key)
        html, last = [], (None, None, None)
        for meta, tid, pid, facts, imgs in rows:
            per, prov = meta["period"], meta["provenience"] or "provenience unknown"
            if per != last[0]:
                n = sum(1 for r in rows if r[0]["period"] == per)
                html.append(f"<h2 class='period'>{per} <small>{n} print{'s' if n != 1 else ''}</small></h2>")
            if (per, prov) != last[:2]:
                html.append(f"<h3 class='prov'>{prov}</h3>")
            if (per, prov, tid) != last:
                t_facts = " · ".join(x for x in (
                    f"<b>SM {tid}</b>", meta.get("t_no", ""), meta.get("genre", ""), meta.get("dates", ""),
                    " ".join(y for y in (meta.get("publication", ""), meta.get("publication_no", ""), meta.get("publication_page", "")) if y),
                    f"catalogue: fingerprints {meta['fingerprints']}" if meta["fingerprints"] else "", meta["note"]) if x)
                html.append(f"<div class='tablet'>{t_facts}</div>")
            last = (per, prov, tid)
            html.append(f"<section><h4>{pid}</h4><div class='facts'>{'<br>'.join(facts)}</div><div class='imgs'>"
                        + "".join(f"<figure><a href='{u}' target='_blank'><img src='{u}' loading='lazy'></a><figcaption>{k}</figcaption></figure>" for k, u in imgs)
                        + "</div></section>")
        toc = " · ".join(f"<a href='#'>{p}</a> ({sum(1 for r in rows if r[0]['period'] == p)})" for p in PERIODS if any(r[0]["period"] == p for r in rows))
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>mesoprint gallery</title><style>
body{{font:13px/1.4 system-ui,sans-serif;margin:0;padding:16px 22px;background:#f3f4f6;color:#15171a}}
h1{{font-size:16px}} h2.period{{font-size:15px;margin:22px 0 4px;border-bottom:2px solid #1d6fd6;padding-bottom:3px}} h2.period small{{color:#6b7280;font-weight:normal}}
h3.prov{{font-size:13px;margin:10px 0 4px;color:#374151}} .tablet{{margin:6px 0 6px;color:#374151}}
section{{background:#fff;border:1px solid #d9dbe0;border-radius:8px;padding:12px 16px;margin:0 0 12px}}
h4{{font-size:14px;margin:0 0 6px}} .facts{{color:#374151;margin-bottom:8px}}
.imgs{{display:flex;gap:10px;flex-wrap:wrap}} figure{{margin:0;text-align:center}} img{{height:220px;border:1px solid #ddd;background:#fff}}
figcaption{{font-size:11px;color:#6b7280}}</style></head><body><h1>mesoprint gallery · {len(rows)} reviewed prints, by period and provenience</h1>
<p style="color:#6b7280">{toc}<br>views, all as seen on the tablet (up = up of that face): MSII (dark = groove), mask, print (enhanced, cut to mask), skeleton (detected green, traced magenta, endings red, bifurcations blue), clean (skeleton drawn with ridge width). An inked print of the finger would be the mirror image. Images are 1000 ppi; click to open. Metadata from the "3D scans catalogue" sheet.</p>
{''.join(html)}</body></html>"""

    @app.get("/api/tablets")
    def tablets():
        out = []
        for d in S.tablet_dirs():
            j = json.loads((d / "candidates.json").read_text())
            r = json.loads((d / "review.json").read_text()) if (d / "review.json").exists() else {"candidates": {}, "added": []}
            tid = d.name[3:]
            c = info(tid)
            out.append({"id": tid, "label": c["fingerprints"], "note": c["note"], "period": c["period"],
                        "provenience": c["provenience"],
                        "n": len(j["candidates"]) + len(r["added"]), "reviewed": len(r["candidates"]),
                        "mesh": S.mesh_path(j).exists(), "mesh_name": j["mesh"]["name"]})
        return jsonify(out)

    @app.get("/api/tablet/<tid>")
    def tablet_info(tid):
        t = tablet(tid)
        r = t.review()
        cands = []
        for c in t.candidates():
            cands.append({**c, "marker": t.marker(c), "review": r["candidates"].get(str(c["id"]))})
        c = info(tid)
        return jsonify({"id": tid, "label": c["fingerprints"], "note": c["note"], "msii": t.msii_desc,
                        "period": c["period"], "provenience": c["provenience"], "genre": c.get("genre", ""),
                        "layout": t.layout, "candidates": cands, "reasons": REASONS,
                        "threshold": t.summary["params"]["threshold"]})

    @app.get("/api/tablet/<tid>/overview/<layer>.png")
    def overview(tid, layer):
        t = tablet(tid)
        f = t.folder / "review" / f"overview_{layer}.png"
        if not f.exists():
            abort(404)
        return send_file(f, mimetype="image/png", max_age=0)

    @app.get("/api/tablet/<tid>/review/<name>")
    def review_file(tid, name):
        f = S.results / f"SM_{tid}" / "review" / Path(name).name
        if not f.exists():
            abort(404)
        return send_file(f, mimetype="image/png", max_age=0)

    def painted_mask(e: Extracted, j: dict) -> np.ndarray:
        return decode_mask(j["mask"], e.ex.valid.shape) & e.ex.valid

    @app.get("/api/tablet/<tid>/cand/<int:cid>/extract")
    def extract(tid, cid):
        t = tablet(tid)
        radius = float(request.args.get("radius", 8.0))
        level = float(request.args.get("level", 0.4))
        with S.lock:
            e = S.extract(t, cid, radius, level)
        ex = e.ex
        return jsonify({"key": e.key, "size": int(ex.valid.shape[0]), "mm_per_px": ex.spacing, **e.params,
                        "measurements": measure(ex, ex.mask), "layers": ["msii", "relief", "enhanced"]})

    @app.get("/api/ex/<key>/<layer>.png")
    def ex_layer(key, layer):
        e = S.extractions.get(key) or abort(404)
        return png(layer_png(e, layer))

    @app.post("/api/ex/<key>/skeleton")
    def ex_skeleton(key):
        e = S.extractions.get(key) or abort(404)
        j = request.json
        with S.lock:
            return jsonify(compute_skeleton(e, painted_mask(e, j), j.get("strokes", [])))

    @app.post("/api/ex/<key>/measure")
    def ex_measure(key):
        e = S.extractions.get(key) or abort(404)
        return jsonify(measure(e.ex, painted_mask(e, request.json)))

    @app.post("/api/ex/<key>/morph")
    def ex_morph(key):
        e = S.extractions.get(key) or abort(404)
        j = request.json
        mask = morph(painted_mask(e, j), float(j.get("mm", 0.3)), e.ex.spacing, j.get("op", "grow") == "grow") & e.ex.valid
        return png(_png((mask * 255).astype(np.uint8)))

    @app.post("/api/tablet/<tid>/cand/<int:cid>/review")
    def review(tid, cid):
        t = tablet(tid)
        j = request.json
        e = S.extractions.get(j["key"]) or abort(404)
        ex = e.ex
        mask = painted_mask(e, j) if j.get("mask") else ex.mask
        meas = measure(ex, mask)
        out = t.folder / "review"
        out.mkdir(exist_ok=True)
        stem = f"cand{cid:02d}"
        strokes = j.get("strokes") or []
        skel = compute_skeleton(e, mask, strokes) if (j.get("skeleton") or strokes) else None  # may re-enhance
        rel8 = to_uint8(ex.relief, ex.valid)
        enh8 = groove_dark8(e.enhanced, ex.valid, 0.5, 99.5)
        if skel is not None:
            (out / f"{stem}_skeleton.json").write_text(json.dumps({"strokes": strokes, **skel}))
            _save_skeleton_png(out / f"{stem}_skeleton.png", groove_dark8(ex.msii, ex.valid, 2, 98), skel)
            _save_clean_png(out / f"{stem}_clean.png", skel, ex.valid.shape[0], ex.spacing,
                            skel["spacing"].get("mean_ridge_breadth_mm"))
        Image.fromarray(np.flipud(mask_overlay(rel8, ex.valid, mask))).save(out / f"{stem}_mask.png", dpi=(PPI, PPI))
        Image.fromarray(np.flipud(np.where(mask, enh8, 255))).save(out / f"{stem}_print.png", dpi=(PPI, PPI))
        Image.fromarray(np.flipud(groove_dark8(ex.msii, ex.valid, 2, 98))).save(out / f"{stem}_msii.png", dpi=(PPI, PPI))
        Image.fromarray(np.flipud((mask * 255).astype(np.uint8))).save(out / f"{stem}_maskonly.png", dpi=(PPI, PPI))
        entry = {"verdict": j["verdict"], "reason": j.get("reason", ""), "note": j.get("note", ""),
                 "reviewer": j.get("reviewer", ""), "time": time.strftime("%Y-%m-%d %H:%M"),
                 "centre": ex.frame.origin.tolist(), "params": e.params, "measurements": meas,
                 "manual": j.get("manual"), "strokes": strokes,
                 "skeleton": None if skel is None else {k: skel[k] for k in ("ridge_length_mm", "guided", "n_strokes")}
                 | {"n_minutiae": len(skel["minutiae"]), "file": f"{stem}_skeleton.json",
                    "ridge_breadth_mm": skel["spacing"].get("mean_ridge_breadth_mm"),
                    "ridge_breadth_q1_q3": [skel["spacing"].get("q1"), skel["spacing"].get("q3")]},
                 "files": [f"{stem}_{k}.png" for k in ("mask", "print", "msii", "maskonly")]}
        r = t.review()
        r["candidates"][str(cid)] = entry
        t.save_review(r)
        return jsonify({"review": entry})

    @app.post("/api/tablet/<tid>/cand/add")
    def add(tid):
        t = tablet(tid)
        j = request.json
        got = t.pick(j["view"], float(j["px"]), float(j["py"]))
        if got is None:
            abort(400, "not on the object")
        centre, normal = got
        r = t.review()
        cid = ADDED_FROM + len(r["added"]) + 1
        c = {"id": cid, "centre": [round(float(x), 3) for x in centre], "normal": [round(float(x), 4) for x in normal],
             "score": None, "n_patches": 0, "added": True}
        r["added"].append(c)
        t.save_review(r)
        return jsonify({**c, "marker": t.marker(c), "review": None})

    @app.post("/api/tablet/<tid>/cand/<int:cid>/move")
    def move(tid, cid):
        t = tablet(tid)
        j = request.json
        got = t.pick(j["view"], float(j["px"]), float(j["py"]))
        if got is None:
            abort(400, "not on the object")
        centre, normal = got
        r = t.review()
        r["moved"][str(cid)] = {"centre": [round(float(x), 3) for x in centre],
                                "normal": [round(float(x), 4) for x in normal]}
        t.save_review(r)
        S.extractions = {k: e for k, e in S.extractions.items() if e.cid != cid}  # stale centre
        c = t.candidate(cid)
        return jsonify({**c, "marker": t.marker(c), "review": r["candidates"].get(str(cid))})

    @app.post("/api/tablet/<tid>/cand/<int:cid>/remove")
    def remove(tid, cid):
        t = tablet(tid)
        r = t.review()
        r["added"] = [a for a in r["added"] if a["id"] != cid]
        r["candidates"].pop(str(cid), None)
        r["moved"].pop(str(cid), None)
        t.save_review(r)
        return jsonify({"ok": True})

    return app


def serve(results: str | Path, assets: str | Path, port: int = 8765, open_browser: bool = True) -> None:
    app = create_app(Path(results), Path(assets))
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    print(f"mesoprint review: http://127.0.0.1:{port}/  (results: {results}, meshes: {assets}; Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
