# Changelog

All notable changes to this project. Background and reasoning for each change are in
[docs/REPORT.md](docs/REPORT.md), section 6.

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
