"""Gather the verdicts saved by the review app into one table.

Usage:  python scripts/collect_reviews.py [--results results/batch_r100] [--out results/batch_r100/reviews.csv]

One row per reviewed candidate: tablet, candidate id, detection score, face,
verdict (print / no / unsure), reason, automatic and manual ridge breadth,
mask area, extraction parameters, reviewer, time and note. These rows are the
labels for recalibrating the threshold and, later, for training a classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mesoprint.catalogue import LABELS, info
from mesoprint.render import VIEWS

import numpy as np

FACE = {"front": "obverse", "back": "reverse", "top": "top", "bottom": "bottom", "left": "left", "right": "right"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results/batch_r100")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    results = Path(args.results)
    rows = []
    for rf in sorted(results.glob("SM_*/review.json")):
        tid = rf.parent.name[3:]
        review = json.loads(rf.read_text())
        summary = json.loads((rf.parent / "candidates.json").read_text())
        cands = {c["id"]: c for c in summary["candidates"] + review.get("added", [])}
        for cid, r in review["candidates"].items():
            c = cands.get(int(cid), {})
            n = np.asarray(c.get("normal", [0, 0, 1]))
            m, man = r.get("measurements", {}), r.get("manual") or {}
            rows.append({
                "tablet": f"SM {tid}", "catalogue": LABELS.get(tid, ""), "period": info(tid)["period"],
                "provenience": info(tid)["provenience"], "candidate": cid,
                "added_by_reviewer": bool(c.get("added")), "score": c.get("score"), "n_patches": c.get("n_patches"),
                "face": FACE[max(VIEWS, key=lambda k: n @ VIEWS[k][0])],
                "centre": " ".join(f"{x:.1f}" for x in r.get("centre", c.get("centre", []))),
                "verdict": r["verdict"], "reason": r.get("reason", ""),
                "auto_ridge_breadth_mm": m.get("mean_ridge_breadth_mm"), "manual_ridge_breadth_mm": man.get("mean_ridge_breadth_mm"),
                "skeleton_ridge_breadth_mm": (r.get("skeleton") or {}).get("ridge_breadth_mm"),
                "n_minutiae": (r.get("skeleton") or {}).get("n_minutiae"),
                "manual_ridges": man.get("n_ridges"), "manual_length_mm": man.get("length_mm"),
                "mask_area_mm2": m.get("area_mm2"), "ridge_depth_um": m.get("ridge_amplitude_um_rms"),
                "radius_mm": r.get("params", {}).get("radius_mm"), "mask_level": r.get("params", {}).get("mask_level"),
                "reviewer": r.get("reviewer", ""), "time": r.get("time", ""), "note": r.get("note", ""),
            })
    out = Path(args.out or results / "reviews.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["tablet"])
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["verdict"]] = by.get(r["verdict"], 0) + 1
    print(f"{len(rows)} reviewed candidates from {len(set(r['tablet'] for r in rows))} tablets: {by} -> {out}")


if __name__ == "__main__":
    main()
