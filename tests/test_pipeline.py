"""End-to-end check on a synthetic tablet with known fingerprints and artifacts."""

import numpy as np
import pytest

from mesoprint.detect import DetectParams, detect
from mesoprint.extract import extract_region
from mesoprint.synth import make_tablet


@pytest.fixture(scope="module")
def run():
    mesh, truth = make_tablet(seed=3)
    normals = mesh.vertex_normals()
    det = detect(mesh, DetectParams(stride=1.5), normals)
    return mesh, normals, truth, det


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def test_fingerprints_are_top_candidates(run):
    _, _, truth, det = run
    whorl, arch = truth["fingerprints"]
    top = det.candidates[:2]
    assert _dist(top[0].centre, whorl["centre"]) < whorl["radius"]
    assert any(_dist(c.centre, arch["centre"]) < arch["radius"] + 1 for c in top)


def test_scanner_stripes_rank_below_fingerprints(run):
    _, _, truth, det = run
    art = truth["artifacts"][0]
    fps = truth["fingerprints"]
    rank = {}
    for i, c in enumerate(det.candidates):
        for key, ref, r in [("whorl", fps[0]["centre"], 5), ("arch", fps[1]["centre"], 4.5),
                            ("stripes", art["centre"], 4.5)]:
            if _dist(c.centre, ref) < r:
                rank.setdefault(key, i)
    assert "stripes" not in rank or rank["stripes"] > max(rank["whorl"], rank["arch"])
    # the body of the stripe patch (perfectly parallel ridges) is held down by the
    # straightness penalty to at most the candidate threshold, prints reach ~1
    core = [p for p in det.patches if _dist(p.centre, art["centre"]) < 2.5 and p.periodic_straightness > 0.95]
    assert core and max(p.score for p in core) <= det.params.threshold + 1e-6
    assert max(c.score for c in det.candidates[:2]) > 0.8


def test_wedges_and_clay_score_low(run):
    _, _, truth, det = run
    marks = [f["centre"] for f in truth["fingerprints"]] + [truth["artifacts"][0]["centre"]]
    bg = [p.score for p in det.patches if all(_dist(p.centre, m) > 7 for m in marks)]
    assert len(bg) > 50
    assert np.percentile(bg, 99) < 0.08


@pytest.mark.parametrize("radius", [3.0, 6.0])
def test_extraction_measures_ridge_period(run, radius):
    mesh, normals, truth, _ = run
    arch = truth["fingerprints"][1]
    ex = extract_region(mesh, arch["centre"], radius=radius, normals=normals)
    period = ex.measurements()["mean_ridge_breadth_mm"]
    assert abs(period - arch["ridge_period_mm"]) / arch["ridge_period_mm"] < 0.08
    assert ex.ppi == pytest.approx(1000, rel=1e-6)
