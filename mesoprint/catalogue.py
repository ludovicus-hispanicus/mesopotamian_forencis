"""Catalogue metadata for the Sulaymaniyah tablets.

``results/catalogue.csv`` is exported from the project's Google Sheet
"3D scans catalogue" (sheet "Catalogue of tablets"); ``info(tid)`` returns a
tablet's record with a normalised period. The hard-coded ``LABELS`` / ``NOTES``
(docs/REPORT.md, section 3.3) are the fallback when the CSV is absent.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

LABELS = {
    **{i: "yes" for i in ("036475", "036917", "037316", "037319", "037321", "037324", "037326", "037377",
                          "037717", "039043", "039941", "043643", "043644")},
    **{i: "maybe" for i in ("036359", "036390", "036413", "036425", "036506")},
}
NOTES = {"039043": "scanner waves, seal", "037244": "scanner waves", "037318": "ink label",
         "036425": "ink label", "037323": "ink label", "037319": "remarks only"}

#: Chronological order of the normalised period names.
PERIODS = ["Pre-Sargonic", "Sargonic", "Ur III", "Old Babylonian", "Late Old Babylonian", "Middle Babylonian",
           "Neo-Assyrian", "Neo-Babylonian", "unknown"]

CATALOGUE_CSV = Path(__file__).resolve().parents[1] / "results" / "catalogue.csv"


def normalise_period(raw: str) -> str:
    p = (raw or "").strip().lower()
    if not p:
        return "unknown"
    if "pre" in p and "sargonic" in p:
        return "Pre-Sargonic"
    if "sargonic" in p:
        return "Sargonic"
    if "ur iii" in p or "ur3" in p:
        return "Ur III"
    if "late old bab" in p:
        return "Late Old Babylonian"
    if "old bab" in p:
        return "Old Babylonian"
    if "middle bab" in p:
        return "Middle Babylonian"
    if "assyrian" in p:
        return "Neo-Assyrian"
    if "babylonian" in p:
        return "Neo-Babylonian"
    return raw.strip()


@lru_cache(maxsize=1)
def catalogue(path: str | Path | None = None) -> dict[str, dict]:
    """All records keyed by the 6-digit museum number, or {} without the CSV."""
    p = Path(path) if path else CATALOGUE_CSV
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            tid = r["tablet"].replace("SM", "")
            r["period_raw"] = r.get("period", "")
            r["period"] = normalise_period(r["period_raw"])
            r["provenience"] = (r.get("provenience") or "").strip().rstrip(".") or ""
            out[tid] = r
    return out


def info(tid: str) -> dict:
    """Metadata for a tablet id like ``"036475"``; always has period, provenience, fingerprints, note."""
    r = dict(catalogue().get(tid, {}))
    r.setdefault("tablet", f"SM{tid}")
    r.setdefault("period", "unknown")
    r.setdefault("provenience", "")
    r["fingerprints"] = (r.get("fingerprints") or LABELS.get(tid, "")).strip()
    r["note"] = NOTES.get(tid, "")
    return r
