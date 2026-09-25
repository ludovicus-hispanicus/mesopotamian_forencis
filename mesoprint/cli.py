"""Command-line interface: ``python -m mesoprint <command> ...``"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .detect import DetectParams, detect
from .extract import extract_region, save_extraction
from .mesh import load_ply, save_ply
from .msii import MSII_RADIUS, file_msii
from .render import colormap, render_fatcross


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load(path: str, scale: float):
    t = time.time()
    mesh = load_ply(path)
    if scale != 1.0:
        mesh = mesh.scaled(scale)
    _log(f"loaded {mesh.name}: {mesh.n_vertices:,} vertices in {time.time() - t:.1f}s")
    return mesh


def _msii(mesh, args):
    """Per-vertex MSII from a GigaMesh export, unless ``--compute-msii``; returns
    ``(values or None, description for the log and JSON)``."""
    if not args.compute_msii:
        got = file_msii(mesh, args.msii_radius)
        if got is not None:
            return got[0], f"GigaMesh file, r = {got[1]:.4f} mm"
    return None, f"computed from the mesh, r = {args.msii_radius} mm"


def cmd_info(args) -> None:
    mesh = _load(args.mesh, args.scale)
    print(json.dumps(mesh.summary(), indent=2))


def cmd_detect(args) -> None:
    mesh = _load(args.mesh, args.scale)
    out = Path(args.out or f"{mesh.name}_fingerprints")
    out.mkdir(parents=True, exist_ok=True)
    normals = mesh.vertex_normals()
    params = DetectParams(patch_radius=args.radius, stride=args.stride, threshold=args.threshold,
                          msii_radius=args.msii_radius)
    msii, msii_source = _msii(mesh, args)
    _log(f"MSII: {msii_source}")

    t = time.time()
    det = detect(mesh, params, normals, progress=lambda k, n: _log(f"  patches {k}/{n}"), msii=msii)
    _log(f"scored {len(det.patches)} patches in {time.time() - t:.1f}s "
         f"(grid {det.spacing * 1000:.1f} um); {len(det.candidates)} candidate regions")

    with open(out / "patches.csv", "w", newline="") as fh:
        rows = [p.as_dict() for p in det.patches]
        keys = ["cx", "cy", "cz", "nx", "ny", "nz", "valid_frac", "score"] + \
            [k for k in rows[0] if k not in ("centre", "normal", "valid_frac", "score")] if rows else []
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["cx"], r["cy"], r["cz"] = r.pop("centre")
            r["nx"], r["ny"], r["nz"] = r.pop("normal")
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    cands = det.candidates[: args.top]
    summary = {"mesh": mesh.summary(), "params": {**vars(params), "band": list(params.band)},
               "msii_source": msii_source, "grid_spacing_mm": det.spacing, "candidates": []}
    tree = cKDTree(mesh.vertices)
    for i, c in enumerate(cands, 1):
        entry = {"id": i, **c.as_dict()}
        if not args.no_extract:
            try:
                ex = extract_region(mesh, c.centre, args.extract_radius, 25.4 / args.ppi, normals, tree,
                                    msii_radius=args.msii_radius, msii_values=msii)
                meta = save_extraction(ex, out / "candidates", f"cand{i:02d}", {"candidate": entry})
                entry["measurements"] = meta["measurements"]
                entry["files"] = meta["files"]
            except ValueError as e:
                entry["extract_error"] = str(e)
        summary["candidates"].append(entry)
        _log(f"  #{i}: score {c.score:.3f} x{c.n_patches} patches, period {c.ridge_period_mm:.3f} mm "
             f"at {np.round(c.centre, 2).tolist()}")
    (out / "candidates.json").write_text(json.dumps(summary, indent=2))

    # heat map colours: faint from half the threshold, full at twice the threshold
    lo, hi = 0.5 * params.threshold, 2.0 * params.threshold
    if not args.no_render:
        markers = [(f"#{i}", c.centre, c.normal) for i, c in enumerate(cands, 1)]
        render_fatcross(mesh.vertices, normals, det.vertex_score, out / "overview.png", px_mm=args.render_px,
                        lo=lo, hi=hi, markers=markers, title=f"{mesh.name}: fingerprint score")
    if args.heatmap_ply:
        s = det.vertex_score
        rgb = colormap((s - lo) / (hi - lo)).astype(np.uint8)
        save_ply(out / f"{mesh.name}_score.ply", mesh, {"quality": s.astype(np.float32)}, rgb)
    _log(f"results in {out}/")


def cmd_extract(args) -> None:
    mesh = _load(args.mesh, args.scale)
    msii, msii_source = _msii(mesh, args)
    _log(f"MSII: {msii_source}")
    ex = extract_region(mesh, args.centre, args.radius, 25.4 / args.ppi, msii_radius=args.msii_radius,
                        msii_values=msii)
    meta = save_extraction(ex, args.out, args.name or f"{mesh.name}_region")
    print(json.dumps(meta["measurements"], indent=2))


def cmd_synth(args) -> None:
    from .synth import make_tablet

    mesh, truth = make_tablet(seed=args.seed, spacing=args.spacing)
    save_ply(args.out, mesh)
    Path(args.out).with_suffix(".truth.json").write_text(json.dumps(truth, indent=2))
    _log(f"wrote {args.out} ({mesh.n_vertices:,} vertices) and ground truth")


def cmd_review(args) -> None:
    try:
        from .review import serve
    except ImportError as e:  # flask is optional
        raise SystemExit(f"{e}\nthe review app needs Flask: pip install flask") from e
    serve(args.results, args.assets, args.port, not args.no_browser)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="mesoprint", description="Detect and extract fingerprints on 3D scans of clay objects.")
    p.add_argument("--scale", type=float, default=1.0, help="multiply coordinates to get millimetres")
    p.add_argument("--msii-radius", type=float, default=MSII_RADIUS,
                   help="ball radius of the ridge-scale MSII map, mm (about half the ridge period)")
    p.add_argument("--compute-msii", action="store_true",
                   help="compute MSII even if the mesh is a GigaMesh MSII export (*_r0.30_n4_v256.volume.ply)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="print mesh size and resolution")
    s.add_argument("mesh")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("detect", help="score the whole surface and extract the best candidates")
    s.add_argument("mesh")
    s.add_argument("-o", "--out")
    s.add_argument("--radius", type=float, default=3.0, help="patch radius, mm")
    s.add_argument("--stride", type=float, default=2.0, help="distance between patch centres, mm")
    s.add_argument("--threshold", type=float, default=DetectParams.threshold)
    s.add_argument("--top", type=int, default=10, help="candidates to report/extract")
    s.add_argument("--extract-radius", type=float, default=8.0)
    s.add_argument("--ppi", type=float, default=1000.0)
    s.add_argument("--render-px", type=float, default=0.1, help="overview pixel size, mm")
    s.add_argument("--no-extract", action="store_true")
    s.add_argument("--no-render", action="store_true")
    s.add_argument("--heatmap-ply", action="store_true", help="also write the mesh coloured by score")
    s.set_defaults(func=cmd_detect)

    s = sub.add_parser("extract", help="extract one region given its centre")
    s.add_argument("mesh")
    s.add_argument("--centre", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    s.add_argument("--radius", type=float, default=8.0)
    s.add_argument("--ppi", type=float, default=1000.0)
    s.add_argument("-o", "--out", default=".")
    s.add_argument("--name")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("synth", help="write a synthetic test tablet with ground truth")
    s.add_argument("out")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--spacing", type=float, default=0.04)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("review", help="open the local review app for a results folder")
    s.add_argument("--results", default="results/batch_r100", help="folder with SM_<id>/candidates.json")
    s.add_argument("--assets", default="assets/3D-Models", help="folder with the meshes")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_review)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
