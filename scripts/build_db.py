"""Build (or rebuild) the fingerprint database from the review folders.

Usage:  python scripts/build_db.py [--results results/batch_r100] [--db results/prints.sqlite]
        python scripts/build_db.py --assign SM036475_right_1 IND_001 0.9 "matched by hand" [--by name]

The tablet / print / minutiae / ridge tables are rebuilt from what the review
app saved; individuals, their print assignments and match scores are kept.
Print IDs are <tablet>_<face>_<n>, e.g. SM036475_right_1.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from mesoprint.db import assign, rebuild


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results/batch_r100")
    ap.add_argument("--db", default="results/prints.sqlite")
    ap.add_argument("--assign", nargs=4, metavar=("PRINT", "INDIVIDUAL", "CONFIDENCE", "BASIS"))
    ap.add_argument("--by", default="")
    args = ap.parse_args(argv)
    if args.assign:
        p, ind, conf, basis = args.assign
        assign(args.db, p, ind, float(conf), basis, args.by)
        print(f"{p} -> {ind} ({conf}, {basis})")
        return
    stats = rebuild(args.results, args.db)
    print(f"{args.db}: {stats}")
    con = sqlite3.connect(args.db)
    print(f"{'print_id':22s} {'verdict':8s} {'area':>6s} {'manual':>7s} {'skel':>6s} {'spect':>6s} {'minut':>5s} reviewer")
    for row in con.execute("SELECT print_id, verdict, mask_area_mm2, breadth_manual_mm, breadth_skeleton_mm, "
                           "breadth_spectral_mm, n_minutiae, reviewer FROM prints ORDER BY print_id"):
        pid, v, area, bm, bs, bp, nm, rv = row
        f = lambda x, w: f"{x:{w}.3f}" if isinstance(x, float) else f"{'-':>{w}s}"
        print(f"{pid:22s} {v:8s} {f(area, 6)} {f(bm, 7)} {f(bs, 6)} {f(bp, 6)} {str(nm or '-'):>5s} {rv}")


if __name__ == "__main__":
    main()
