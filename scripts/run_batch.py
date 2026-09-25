"""Run ``mesoprint detect`` on many scans and summarise the candidates per tablet.

Usage:  python scripts/run_batch.py [meshes ...] [--out results/batch_r100] [--redo]

Default input: every GigaMesh MSII export ``assets/3D-Models/*_r1.00_*.volume.ply``.
Each tablet gets ``<out>/SM_<id>/`` (as written by ``mesoprint detect``); the table
``<out>/summary.csv`` lists the catalogue label, the number of candidates and, for
the best ones, score, size, ridge period and the face they are on.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from mesoprint.catalogue import LABELS, NOTES
from mesoprint.cli import main as mesoprint
from mesoprint.render import VIEWS

FACES = {"front": "obverse", "back": "reverse", "top": "top", "bottom": "bottom", "left": "left", "right": "right"}


def face_of(normal) -> str:
    return FACES[max(VIEWS, key=lambda k: np.asarray(normal) @ VIEWS[k][0])]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("meshes", nargs="*")
    ap.add_argument("--out", default="results/batch_r100")
    ap.add_argument("--redo", action="store_true", help="rerun tablets that already have results")
    ap.add_argument("--top", type=int, default=3, help="candidates per tablet in the summary")
    args = ap.parse_args(argv)
    meshes = [Path(m) for m in args.meshes] or sorted(Path("assets/3D-Models").glob("*_r1.00_*.volume.ply"))
    out = Path(args.out)
    for mesh in meshes:
        m = re.search(r"SM_(\d{6})", mesh.name)
        dest = out / f"SM_{m.group(1) if m else mesh.stem}"
        if args.redo or not (dest / "candidates.json").exists():
            t = time.time()
            print(f"== {dest.name}: {mesh.name}", file=sys.stderr, flush=True)
            mesoprint(["detect", str(mesh), "-o", str(dest)])
            print(f"   {time.time() - t:.0f}s", file=sys.stderr, flush=True)

    # summarise every tablet in the output folder, including earlier runs
    rows = []
    for res_file in sorted(out.glob("SM_*/candidates.json")):
        tid = res_file.parent.name[3:]
        res = json.loads(res_file.read_text())
        cands = res["candidates"]
        row = {"tablet": f"SM {tid}", "catalogue": LABELS.get(tid, ""), "note": NOTES.get(tid, ""),
               "candidates": len(cands), "msii": res.get("msii_source", "")}
        for i, c in enumerate(cands[: args.top], 1):
            meas = c.get("measurements", {})
            row.update({f"c{i}_score": c["score"], f"c{i}_patches": c["n_patches"], f"c{i}_face": face_of(c["normal"]),
                        f"c{i}_centre": " ".join(f"{x:.1f}" for x in c["centre"]),
                        f"c{i}_breadth_mm": meas.get("mean_ridge_breadth_mm", ""),
                        f"c{i}_area_mm2": meas.get("area_mm2", "")})
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        best = f"{r['c1_score']:.2f} x{r['c1_patches']} on {r['c1_face']}" if r["candidates"] else "-"
        print(f"{r['tablet']}  {r['catalogue']:5s}  {r['candidates']:2d} candidates  best {best}  {r['note']}")
    print(f"summary: {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
