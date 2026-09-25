"""SQLite database of extracted fingerprints, built from the review folders.

Print IDs are ``<tablet>_<face>_<n>`` (e.g. ``SM036475_right_1``): what is
certain about a print, and nothing else. Which person made it is a later,
revisable inference, kept in ``individuals`` / ``print_individuals``.

Tables
------
tablets            tablet_id, catalogue (fingerprint flag), note, mesh_name, vertices, edge_mm, t_no,
                   period (normalised), period_raw, provenience, dates, genre, archive, publication
prints             print_id, tablet_id, candidate, face, cx, cy, cz, verdict, reason,
                   score, mask_area_mm2, breadth_manual_mm, breadth_skeleton_mm,
                   breadth_spectral_mm, ridges_per_5mm, n_minutiae, ridge_length_mm,
                   radius_mm, mm_per_px, reviewer, reviewed, note, folder
minutiae           print_id, x_mm, y_mm, angle_deg, type, source
ridges             print_id, idx, source, points (JSON, mm)
individuals        individual_id, label, note
print_individuals  print_id, individual_id, confidence, basis, decided_by, decided
matches            print_a, print_b, method, score, run

Coordinates of minutiae and ridges are millimetres in the print image (x right,
y up, origin at the image's bottom-left), with the 3D frame stored per print so
they can be put back on the tablet.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .catalogue import info
from .render import VIEWS

FACE = {"front": "obverse", "back": "reverse", "top": "top", "bottom": "bottom", "left": "left", "right": "right"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS tablets (tablet_id TEXT PRIMARY KEY, catalogue TEXT, note TEXT, mesh_name TEXT,
    vertices INTEGER, edge_mm REAL, t_no TEXT, period TEXT, period_raw TEXT, provenience TEXT, dates TEXT,
    genre TEXT, archive TEXT, publication TEXT);
CREATE TABLE IF NOT EXISTS prints (print_id TEXT PRIMARY KEY, tablet_id TEXT, candidate INTEGER, face TEXT,
    cx REAL, cy REAL, cz REAL, verdict TEXT, reason TEXT, score REAL, mask_area_mm2 REAL,
    breadth_manual_mm REAL, breadth_skeleton_mm REAL, breadth_spectral_mm REAL, ridges_per_5mm REAL,
    n_minutiae INTEGER, ridge_length_mm REAL, radius_mm REAL, mm_per_px REAL, reviewer TEXT, reviewed TEXT,
    note TEXT, folder TEXT, frame TEXT);
CREATE TABLE IF NOT EXISTS minutiae (print_id TEXT, x_mm REAL, y_mm REAL, angle_deg REAL, type TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS ridges (print_id TEXT, idx INTEGER, source TEXT, points TEXT);
CREATE TABLE IF NOT EXISTS individuals (individual_id TEXT PRIMARY KEY, label TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS print_individuals (print_id TEXT, individual_id TEXT, confidence REAL, basis TEXT,
    decided_by TEXT, decided TEXT);
CREATE TABLE IF NOT EXISTS matches (print_a TEXT, print_b TEXT, method TEXT, score REAL, run TEXT);
"""


def face_of(normal) -> str:
    n = np.asarray(normal, float)
    return FACE[max(VIEWS, key=lambda k: n @ VIEWS[k][0])]


def open_db(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    return con


def rebuild(results: str | Path, db_path: str | Path, keep_tables=("individuals", "print_individuals", "matches")) -> dict:
    """Rebuild tablets / prints / minutiae / ridges from ``<results>/SM_*/review.json``;
    the inference tables are kept."""
    results = Path(results)
    con = open_db(db_path)
    for t in ("tablets", "prints", "minutiae", "ridges"):
        con.execute(f"DROP TABLE {t}")  # recreated with the current schema
    con.executescript(SCHEMA)
    n_prints = n_min = 0
    for rf in sorted(results.glob("SM_*/review.json")):
        folder = rf.parent
        tid = folder.name[3:]
        summary = json.loads((folder / "candidates.json").read_text())
        review = json.loads(rf.read_text())
        mesh = summary["mesh"]
        c = info(tid)
        pub = " ".join(x for x in (c.get("publication", ""), c.get("publication_no", ""), c.get("publication_page", "")) if x)
        con.execute("INSERT OR REPLACE INTO tablets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"SM{tid}", c["fingerprints"], c["note"], mesh["name"], mesh["vertices"], mesh["median_edge_mm"],
                     c.get("t_no", ""), c["period"], c.get("period_raw", ""), c["provenience"], c.get("dates", ""),
                     c.get("genre", ""), c.get("archive", ""), pub))
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
            centre = r.get("centre") or c.get("centre") or [None] * 3
            m, man, sk = r.get("measurements", {}), r.get("manual") or {}, r.get("skeleton") or {}
            frame = None
            skel_file = folder / "review" / f"cand{int(cid_s):02d}_skeleton.json"
            cand_json = folder / "candidates" / f"cand{int(cid_s):02d}.json"
            if cand_json.exists():
                frame = json.dumps(json.loads(cand_json.read_text()).get("frame"))
            mm_px = 25.4 / 1000.0
            con.execute("INSERT OR REPLACE INTO prints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (pid, f"SM{tid}", int(cid_s), face, *centre, r["verdict"], r.get("reason", ""), c.get("score"),
                         m.get("area_mm2"), man.get("mean_ridge_breadth_mm"), sk.get("ridge_breadth_mm"),
                         m.get("mean_ridge_breadth_mm"), m.get("ridge_count_per_5mm"), sk.get("n_minutiae"),
                         sk.get("ridge_length_mm"), r.get("params", {}).get("radius_mm"), mm_px, r.get("reviewer", ""),
                         r.get("time", ""), r.get("note", ""), str(folder / "review"), frame))
            n_prints += 1
            if skel_file.exists():
                s = json.loads(skel_file.read_text())
                size = 2 * (r.get("params", {}).get("radius_mm") or 0) / mm_px + 1  # image height in px
                for mn in s.get("minutiae", []):
                    con.execute("INSERT INTO minutiae VALUES (?,?,?,?,?,?)",
                                (pid, mn["x"] * mm_px, (size - 1 - mn["y"]) * mm_px, mn.get("angle"), mn["type"], "detected"))
                    n_min += 1
                for i, p in enumerate(s.get("polylines", [])):
                    pts = [[round(x * mm_px, 4), round((size - 1 - y) * mm_px, 4)] for x, y in p["pts"]]
                    con.execute("INSERT INTO ridges VALUES (?,?,?,?)", (pid, i, p["source"], json.dumps(pts)))
    con.commit()
    stats = {"tablets": con.execute("SELECT COUNT(*) FROM tablets").fetchone()[0], "prints": n_prints,
             "confirmed": con.execute("SELECT COUNT(*) FROM prints WHERE verdict='print'").fetchone()[0],
             "minutiae": n_min, "individuals": con.execute("SELECT COUNT(*) FROM individuals").fetchone()[0]}
    con.close()
    return stats


def assign(db_path: str | Path, print_id: str, individual_id: str, confidence: float, basis: str,
           decided_by: str = "", label: str = "") -> None:
    """Link a print to an individual (creating the individual if new)."""
    import time

    con = open_db(db_path)
    con.execute("INSERT OR IGNORE INTO individuals VALUES (?,?,?)", (individual_id, label, ""))
    con.execute("DELETE FROM print_individuals WHERE print_id=? AND individual_id=?", (print_id, individual_id))
    con.execute("INSERT INTO print_individuals VALUES (?,?,?,?,?,?)",
                (print_id, individual_id, confidence, basis, decided_by, time.strftime("%Y-%m-%d %H:%M")))
    con.commit()
    con.close()
