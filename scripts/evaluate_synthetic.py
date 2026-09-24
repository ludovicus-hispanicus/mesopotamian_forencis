"""Evaluate detection and measurement on synthetic tablets with known ground truth.

Usage:  python scripts/evaluate_synthetic.py [--seeds 0 3 5 11 42] [--out table.md]

Prints a Markdown table (one row per random tablet) used in docs/REPORT.md.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mesoprint.detect import DetectParams, detect
from mesoprint.extract import extract_region
from mesoprint.synth import make_tablet


def _d(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def evaluate(seed: int, stride: float = 1.5) -> dict:
    mesh, truth = make_tablet(seed=seed)
    normals = mesh.vertex_normals()
    det = detect(mesh, DetectParams(stride=stride), normals)
    whorl, arch = truth["fingerprints"]
    art = truth["artifacts"][0]
    marks = [whorl["centre"], arch["centre"], art["centre"]]

    def best(c, r=3.0):
        s = [p.score for p in det.patches if _d(p.centre, c) < r]
        return max(s) if s else 0.0

    def rank(c, r):
        for i, cand in enumerate(det.candidates, 1):
            if _d(cand.centre, c) < r:
                return i
        return None

    bg = [p.score for p in det.patches if all(_d(p.centre, m) > 7 for m in marks)]
    core = [p.score for p in det.patches if _d(p.centre, art["centre"]) < 2.5 and p.features
            and p.features.straightness > 0.9]
    ex_w = extract_region(mesh, whorl["centre"], 6.0, normals=normals).measurements()
    ex_a = extract_region(mesh, arch["centre"], 6.0, normals=normals).measurements()
    return {
        "seed": seed,
        "whorl": best(whorl["centre"]), "arch": best(arch["centre"]),
        "stripes_max": best(art["centre"], 3.5), "stripes_core": max(core) if core else 0.0,
        "bg_p99": float(np.percentile(bg, 99)), "bg_max": float(max(bg)),
        "rank_whorl": rank(whorl["centre"], whorl["radius"]),
        "rank_arch": rank(arch["centre"], arch["radius"] + 1),
        "rank_stripes": rank(art["centre"], 4.5),
        "period_whorl": ex_w["mean_ridge_breadth_mm"], "period_arch": ex_a["mean_ridge_breadth_mm"],
        "truth_whorl": whorl["ridge_period_mm"], "truth_arch": arch["ridge_period_mm"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 3, 5, 11, 42])
    ap.add_argument("--out")
    args = ap.parse_args()
    t = time.time()
    rows = [evaluate(s) for s in args.seeds]
    fmt = lambda r: "-" if r is None else str(r)  # noqa: E731
    lines = [
        "| seed | whorl | arch | stripes (max / core) | background (p99 / max) | rank whorl / arch / stripes "
        "| ridge breadth whorl (truth 0.476*) | ridge breadth arch (truth 0.370) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        err = 100 * (r["period_arch"] - r["truth_arch"]) / r["truth_arch"]
        lines.append(
            f"| {r['seed']} | {r['whorl']:.3f} | {r['arch']:.3f} | {r['stripes_max']:.3f} / {r['stripes_core']:.3f} "
            f"| {r['bg_p99']:.3f} / {r['bg_max']:.3f} "
            f"| {fmt(r['rank_whorl'])} / {fmt(r['rank_arch'])} / {fmt(r['rank_stripes'])} "
            f"| {r['period_whorl']:.3f} | {r['period_arch']:.3f} ({err:+.1f}%) |")
    table = "\n".join(lines)
    print(table)
    print(f"\n({len(rows)} tablets in {time.time() - t:.0f}s)")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(table + "\n")


if __name__ == "__main__":
    main()
