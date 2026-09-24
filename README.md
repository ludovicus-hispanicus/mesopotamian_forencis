# mesoprint

Detect and extract ancient fingerprint impressions on 3D scans of cuneiform tablets
and other clay objects.

The tool works directly on the mesh (PLY exported from GigaMesh or any scanner). It
does not use 2D renderings, so depth is measured in real units (µm) and ridge spacing
in millimetres.

> **Status:** v0.1 has been validated only on synthetic tablets. It has not yet been
> run on the Sulaymaniyah scans, and scores and thresholds will need recalibrating on
> them.

## Why this is feasible with the Sulaymaniyah scans

| | Value |
|---|---|
| Lateral resolution (Hexagon SmartScan, S-125) | ≈ 40 µm, about 11 samples per 0.45 mm ridge period |
| Depth resolution | 5 µm; ridge impressions are typically 10–50 µm deep |
| Equivalent image resolution | ≈ 635 ppi, above the 500 ppi forensic minimum |

## How it works

1. **Patches.** The surface is covered with overlapping 6 mm disks, one every 2 mm.
   Each disk is projected onto its best-fit plane and binned into a height map.
   Points facing away from the plane are ignored, so rounded edges work too.
2. **Relief.** A quadratic surface (the tablet's curvature) and a Gaussian low-pass
   (0.8 mm) are subtracted. This leaves the fine relief: wedges, clay grain, and
   fingerprint ridges.
3. **Ridge features** for each patch (`mesoprint/features.py`):
   - `band_ratio`: share of relief energy at ridge frequencies, 1.6–3.6 ridges/mm
     (periods 0.28–0.63 mm).
   - `band_amp_um`: RMS depth of that ridge-band relief.
   - `local_coherence`: how consistent the ridge orientation is at about 1 mm scale.
   - `coverage`: how evenly the ridge texture fills the patch. Single wedge edges
     "ring" in the ridge band but cover little of the patch.
   - `straightness`: share of ridge area within ±12° of the dominant direction.
     Near 1 means parallel stripes, the signature of structured-light scanner
     artifacts.
   - `peak_freq`: ridge frequency, measured after removing the clay's power-law
     background spectrum.
4. **Score.** `band_ratio × local_coherence × coverage × amplitude factor ×
   straightness penalty`. This is a hand-tuned starting point, meant to be replaced by
   a classifier trained on labelled tablets.
5. **Candidates.** Neighbouring patches above the threshold are grouped and ranked.
   Scores are also interpolated back onto the vertices as a heat map.
6. **Extraction.** For each candidate, a 12 mm disk is flattened and resampled to
   1000 ppi (DPI stored in the PNG). It is written as:
   - `*_relief.png`: the surface as seen on the tablet (light = high).
   - `*_enhanced.png`: ridge-band filtered, contrast-normalised.
   - `*_print.png`: the enhanced image mirrored, so it reads like an ink print of the
     finger (clay grooves = finger ridges, shown dark).
   - `*.json`: frame, pixel size and measurements: ridge frequency, mean ridge
     breadth (ridge + furrow, as used for age and sex estimation in the Tel Burna
     study), ridge count per 5 mm, and ridge depth.

## Install and use

```bash
pip install -e .            # Python 3.8+; numpy, scipy, plyfile, pillow

mesoprint info SM_039043_GMOCF.ply          # size, resolution, samples per ridge
mesoprint detect SM_039043_GMOCF.ply -o out/ --heatmap-ply
mesoprint extract SM_039043_GMOCF.ply --centre 12.3 -4.1 8.0 --radius 6 -o out/
mesoprint synth test.ply                    # synthetic tablet + ground truth JSON
```

`detect` writes:
- `overview.png`: fat-cross views with the score heat map and numbered candidates.
- `candidates.json` and `candidates/`: the extracted images and measurements.
- `patches.csv`: every patch's features, for analysis and training.
- `<name>_score.ply` (optional): the mesh coloured by score, with a `quality`
  property readable in GigaMesh and MeshLab.

To share results for calibration, write them inside the repo, e.g.
`-o results/SM_036475`, and commit that folder. It is small; the optional heat-map
PLY is git-ignored, like all `*.ply` files except cropped test regions under
`samples/`.

Coordinates are taken to be millimetres; use `--scale` otherwise. The overview
assumes GigaMesh orientation (obverse facing +Z, X right, Y up).

Runtime: about 45 s for a 2.6 M-vertex mesh on 4 cores.

## Validation so far (synthetic)

`mesoprint/synth.py` builds a tablet face with clay texture, rows of wedges, a whorl
print, an arch print on the curved margin, and a patch of straight stripes at a
ridge-like period (0.5 mm), imitating scanner artifacts. `pytest` checks the
following, on a different random tablet from the one used for tuning:

- Both prints are the top two candidates.
- Wedges and clay stay below 0.08 (99th percentile). Prints score 0.16–0.39.
- The stripe body is suppressed (< 0.1) and always ranks below both prints.
- Extracted ridge breadth is within 8% of the truth (typically within 5%).

## Known limitations and next steps

- **No real data yet.** Next: run on the tablets the catalogue marks with
  fingerprints: 036475, 036917, 037316, 037319, 037321, 037324, 037326, 037377,
  037717, 039043, 039941, 043643 and 043644. Use unmarked tablets as negatives and
  the "maybe" tablets (036359, 036390, 036413, 036425, 036506) for review.
  Then calibrate the thresholds.
- **Scanner waves** (SM 039043, SM 037244) are modelled only by a guess. A patch
  straddling the edge of a stripe area can still reach a moderate score.
- **Flattening** is a plane projection, so prints wrapping round a corner are
  foreshortened. Proper unrolling (e.g. LSCM) is future work.
- **Next steps:** an interactive viewer to accept or reject candidates (which also
  produces training labels), a learned classifier, and minutiae extraction for
  matching prints between tablets.
