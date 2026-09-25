# Changelog

All notable changes to this project. Background and reasoning for each change are in
[docs/REPORT.md](docs/REPORT.md), section 6.

## [Unreleased] – 2026-09-24, first real scan (SM 036475)

v0.1 did not find the known print on SM 036475 (right side, bottom): best score 0.047
against a threshold of 0.10. Its ridge-band energy and coherence rated cracks, rulings,
wedge rows and clay grain as highly as the print. The print is clearly visible in
raking light and in an MSII map at r = 0.3 mm.

### Added
- `msii.py`: MSII-style integral invariant (share of a ball inside the clay) on patch
  height maps; `--msii-radius` (default 0.3 mm, about half the ridge period).
- `periodicity.py`: per-pixel ridge periodicity, `(rho(tau) - rho(tau/2)) / 2` of the
  local autocorrelation across the ridges. About 1 on ridges, about 0 on single lines,
  grain and smooth surfaces.
- Print isolation in `extract.py`: a mask of the connected periodic area, bridged
  across cracks. New outputs `*_msii.png` and `*_mask.png`; `*_print.png` is cut to the
  mask; measurements are taken inside the mask (`isolated` in the JSON).
- GigaMesh MSII exports (`*_r0.30_n4_v256.volume.ply`) are read directly: the
  per-vertex `feature_vector` (16 radii, `r * (16 - k) / 16`, largest first) is loaded
  with a fast fixed-length reader, and the component closest to `--msii-radius` is used
  instead of computing MSII (`msii.file_msii`). `--compute-msii` forces the computation.
  The run's `candidates.json` records the source (`msii_source`).
- `enhance.py`: Gabor ridge enhancement (Hong, Wan & Jain 1998) driven by the
  periodicity direction and period (a gradient orientation field fails on clay: the
  grain dominates the gradient). The response is faded by the local periodicity so
  non-ridge areas stay grey. Replaces the band-pass `enhanced` image, which turned
  real ridges into blobs.
- Extractions are upright with respect to the tablet face (image up = face up).
- `skeleton.py`: ridge skeleton from the enhanced image (binarise inside the mask,
  thin, prune spurs, vectorise), ridge endings and bifurcations away from the mask
  border. Reviewer strokes in the app: *guide* strokes override the ridge direction
  for the Gabor enhancement around them, *trace* strokes replace the automatic
  skeleton within 0.35 mm and are stored as "traced". Saved per candidate as
  `candNN_skeleton.json` (polylines with source, minutiae, strokes) and
  `candNN_skeleton.png`.
- Review app: ridge lines and rulers are kept (several per print, combined mean
  ridge breadth = total span / total periods, ends of a ridge line count as groove
  centres, dots on the grooves between are detected and can be dragged, added or
  removed); brightness/contrast; overview zoom/pan at 0.05 mm/px; MSII base with
  grooves dark; drag the selected circle to move a candidate.
- All images now share one shading: dark = groove in the clay (finger ridge).
- `mesoprint review`: a local review app (`review.py`, `review.html`; needs Flask).
  Overview with surface/MSII base and heat-map toggle, candidate editor with the
  print mask (paint, erase, grow, shrink, re-extract at 4–15 mm radius and any mask
  level), automatic and manual (ridge-line) ridge breadth, verdicts with reasons,
  adding missed prints by clicking the overview. Output: `review.json` and `review/`
  per tablet; `scripts/collect_reviews.py` gathers them into `reviews.csv`.
- `scripts/run_batch.py`: runs detection over every MSII export and writes
  `summary.csv`. `catalogue.py` holds the catalogue labels.
- `render.render_layers` / `compose_fatcross` / `marker_pixel`: overview as separate
  base and heat layers with a known layout, so markers and clicks map to 3D.
- `raster.interpolate_grid` (triangle interpolation for extractions) and
  `raster.fill_invalid`.
- Tests: `tests/test_msii.py` (MSII sign and plane value; periodicity on ridges,
  a single line and noise).

### Changed
- Detection score is now `fingerprint_score`: the share of the patch covered by
  periodic ridges × the straightness penalty, with straightness measured on the
  periodic pixels only. Threshold 0.10 → 0.20. The v0.1 score is still written to
  `patches.csv` as `band_score`.
- `enhanced` and `print` images are made from the MSII map instead of the relief.
- Default extraction radius 6 → 8 mm.
- The overview marks candidates on whichever of the six views they face (before: front
  and back only), and the heat-map colours follow the threshold.

### Fixed
- Extractions binned points finer than the scan's vertex spacing. Empty cells showed as
  white dots with ringing around them.

### Results
- SM 036475, blind: one candidate, the known print (score 0.75, 5 patches). The best
  patch outside the print scores 0.09; 99% of all patches score below 0.05. Mask 37 mm²,
  mean ridge breadth 0.59–0.62 mm.
- GigaMesh's own MSII for SM 036475 (r = 0.30 file, component 0; r = 1.00 file,
  component 11 = 0.3125 mm) matches the computed map (correlation 0.81) and gives the
  same single candidate: print score 0.67, best elsewhere 0.14 / 0.12, mask 37 / 36 mm²,
  ridge breadth 0.60 mm.
- Synthetic tablets (seeds 3, 5, 11, 42): both prints rank first and second (score 1.0);
  clay and wedges 99th percentile 0.03–0.05; the stripe body is held at ≤ 0.2.
- Detection takes ~100 s for 2.3 M vertices (was ~40 s).

## [0.1.0] – 2026-09-24 (branch `claude/great-ride-ul2pd4`)

### Removed
- `sourceafis.py` and `run_analysis.py`: the 2D sliding-window prototype. It did not
  work (see report, section 2).
- `__pycache__/` from version control.

### Added
- The `mesoprint` package, which works directly on 3D meshes:
  - `mesh.py`: PLY loading and saving (fast path for triangle meshes), vertex normals
    oriented out of the clay, resolution summary.
  - `raster.py`: flattens surface patches to height maps and removes tablet
    curvature and large-scale relief.
  - `features.py`: ridge features (ridge-band energy, depth, frequency, local
    coherence, coverage, straightness) and a heuristic fingerprint score.
  - `detect.py`: patch sampling over the whole surface, vertex heat map, candidate
    regions.
  - `extract.py`: 1000 ppi relief, enhanced and mirrored "print" images with DPI
    metadata, plus ridge measurements.
  - `render.py`: fat-cross overview with heat map and candidate markers.
  - `synth.py`: synthetic tablets with wedges, whorl and arch prints and
    scanner-stripe artifacts, with ground truth.
  - `cli.py`: `mesoprint info | detect | extract | synth`.
- Tests: `tests/` (12 tests: features, mesh I/O and normals, end-to-end pipeline).
- `scripts/evaluate_synthetic.py`: reproducible multi-seed evaluation table.
- Documentation: `README.md`, `docs/REPORT.md`, this changelog, and example outputs
  in `docs/synthetic_example/`.
- `pyproject.toml` (Python ≥ 3.8; numpy, scipy, plyfile, pillow) and `.gitignore`.
  The `.gitignore` excludes `*.ply` except cropped test regions under `samples/`.

### Changed during development (before first real-data run)
- Straightness: first defined as a coherence ratio, then as a weighted orientation
  resultant, finally as the share of ridge area within ±12° of the dominant
  direction. The final version is robust to single wedge edges.
- Straightness penalty onset 0.8 → 0.9 → 0.75 (retuned after each straightness
  change), and candidate threshold 0.25 → 0.10, so that prints with near-parallel
  ridges are not suppressed.
- Detection grid floored at 40 µm (whole run on a 2.6 M-vertex test mesh:
  140 s → 75 s).
- Extraction: Delaunay interpolation replaced by binning at native resolution plus
  cubic resampling (about 8 s → about 1 s per candidate; no interpolation across
  holes). Whole run: 75 s → 47 s.
- Overview rendering: splatting gaps filled with the nearest drawn pixel inside the
  outline.

### Fixed
- Ridge frequency was biased low on large extraction windows (0.634 mm measured vs
  0.370 mm true). The clay's power-law spectrum is now removed before peak picking;
  error is within ±5%.
- Relief sign on meshes with inward-wound triangles (normals now oriented by signed
  volume).
- PLY files with non-triangle faces now fall back to the general reader instead of
  failing.
- Python 3.8 compatibility (dict merge syntax in `cli.py`).

### Commits
- `f6000dc`: Start mesoprint: 3D fingerprint detection and extraction from scratch.
- `4b2a25f`: Harden mesoprint for real GigaMesh meshes.
- (this commit): Project report, changelog and synthetic evaluation script.
