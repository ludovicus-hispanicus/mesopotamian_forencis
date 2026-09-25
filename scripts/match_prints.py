"""Correlate every confirmed print against every other and write the scores to the database.

Usage:  python scripts/match_prints.py [--results results/batch_r100] [--db results/prints.sqlite] [--step 3]

Reads each print's saved ridge image (``candNN_print.png``, grooves dark)
and mask from the review folder, scores all pairs with
:func:`mesoprint.match.match`, stores them in the ``matches`` table and prints
the matrix. Pairs from different tablets give the reference level for
"different finger"; a pair standing clearly above it is worth a closer look.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from mesoprint.match import PrintMap, match


def load_print(folder: Path, cid: int, pid: str) -> PrintMap | None:
    p, m = folder / f"cand{cid:02d}_print.png", folder / f"cand{cid:02d}_maskonly.png"
    if not (p.exists() and m.exists()):
        return None
    print8 = np.asarray(Image.open(p).convert("L"))
    mask = np.asarray(Image.open(m).convert("L")) > 127
    if mask.sum() < 100:
        return None
    return PrintMap.from_images(pid, print8, mask, 25.4 / 1000.0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results/batch_r100")
    ap.add_argument("--db", default="results/prints.sqlite")
    ap.add_argument("--step", type=float, default=3.0, help="rotation step, degrees")
    args = ap.parse_args(argv)
    con = sqlite3.connect(args.db)
    rows = con.execute("SELECT print_id, candidate, folder, mask_area_mm2 FROM prints WHERE verdict='print' ORDER BY print_id").fetchall()
    prints = []
    for pid, cid, folder, area in rows:
        pm = load_print(Path(folder), cid, pid)
        if pm is not None:
            prints.append(pm)
    print(f"{len(prints)} prints with ridge images: " + ", ".join(f"{p.print_id} ({p.mask.sum() * p.mm_per_px**2:.0f} mm2)" for p in prints))
    run = time.strftime("%Y-%m-%d %H:%M")
    con.execute("DELETE FROM matches WHERE method='ridge-correlation'")
    scores = {}
    for a, b in combinations(prints, 2):
        # the overlap must cover at least half of the smaller print (and 8 mm2): in a
        # few mm2 any two ridge patterns of similar spacing line up somewhere
        area = min(a.mask.sum(), b.mask.sum()) * a.mm_per_px**2
        r = match(a, b, step_deg=args.step, min_overlap_mm2=max(8.0, 0.5 * area))
        scores[(a.print_id, b.print_id)] = r
        con.execute("INSERT INTO matches VALUES (?,?,?,?,?)", (a.print_id, b.print_id, "ridge-correlation", r["score"], run))
        print(f"  {a.print_id:20s} vs {b.print_id:20s}  {r['score']:.3f}  rot {r['rotation_deg'] or 0:5.0f}  overlap {r['overlap_mm2'] or 0:5.1f} mm2", flush=True)
    con.commit()
    ids = [p.print_id for p in prints]
    print("\nscore matrix (ridge correlation, -1..1):")
    print(" " * 20 + " ".join(f"{i[-8:]:>8s}" for i in ids))
    for i in ids:
        row = []
        for j in ids:
            if i == j:
                row.append("    -   ")
            else:
                r = scores.get((i, j)) or scores.get((j, i))
                row.append(f"{r['score']:8.3f}")
        print(f"{i:20s}" + " ".join(row))
    (Path(args.db).parent / "matches.json").write_text(json.dumps(list(scores.values()), indent=1))


if __name__ == "__main__":
    main()
