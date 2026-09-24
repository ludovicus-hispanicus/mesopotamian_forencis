# Fingerprint detection on 3D scans of cuneiform tablets: project report

*Status: 24 September 2026 · branch `claude/great-ride-ul2pd4` · package `mesoprint` v0.1.0*

## 1. Summary

The project was restarted from scratch. The earlier prototype worked on 2D renderings
and did not produce usable results. The new tool, `mesoprint`, works directly on the
3D meshes (PLY):

- It scores the whole tablet surface for fingerprint-like ridge patterns.
- It groups high-scoring areas into candidate regions.
- For each candidate it exports flattened 1000 ppi images and ridge measurements
  (ridge frequency, mean ridge breadth, ridge depth).

**What is established**
- The Sulaymaniyah scans have enough resolution for the task (section 4).
- On synthetic tablets with known ground truth, the tool ranks both planted prints
  first on every test tablet. It keeps wedges and clay texture far below the prints'
  scores and measures ridge breadth to within about 5% (section 7).

**What is not yet established**
- Performance on real scans. No real mesh has been processed yet: the files could not
  be transferred into the development environment (section 10). The next step is to
  run the tool on SM 036475 and the other tablets the catalogue marks with
  fingerprints, and then calibrate.

## 2. Starting point

The repository contained `sourceafis.py` and `run_analysis.py`, a Jupyter Book
template and a bibliography. The prototype did not work, for these reasons:

- **It used 2D PNG renderings** (GigaMesh MSII images) instead of the meshes, so the
  real depth and scale information was already gone.
- **It used fixed 200 × 200 pixel windows** with no relation to physical size.
  Fingerprint ridges are defined by a physical spacing of about 0.3–0.6 mm.
- **Its ridge-direction test was wrong.** It averaged raw angles, which is invalid
  for orientations. Cuneiform wedges are also strongly oriented, so they passed the
  test as easily as ridges.
- **Its thresholds were arbitrary.** Despite its name, SourceAFIS was never called.

Decision: remove the prototype and build a mesh-based pipeline. The book template and
the bibliography were left untouched.

## 3. Data

### 3.1 Scans

The scans are the 44 published objects of the Sulaymaniyah Museum dataset (Sáenz,
Bauer, Mara, Altaweel, Gordin), plus further unpublished scans from the same campaign.

- **Scanner:** Hexagon SmartScan (structured light), S-125 field of view, September
  2023.
- **Processing:** GigaMesh (cleaning, orientation, MSII).
- **Mesh sizes:** 1.1–10.6 M vertices, average about 2.7 M. The PLY files are
  roughly 80–400 MB each.

### 3.2 Where the files are

| Location | Content | Accessible from the development environment? |
|---|---|---|
| heidICON pool `cuneiform_sulaimaniya` (DOIs 10.11588/heidicon/23958493–23958536) | 44 published GMOCF models | No, host blocked by the environment's network policy |
| Google Drive `Suli/PLY_GMO` | GMO meshes of the whole campaign | Listing only. The connector downloads files ≤ 10 MB, and direct Drive download hosts are blocked |
| Google Drive `Suli/Examples/Fingerprints`, `Seal+fingerprints` | Small preview renders (160 × 440 px) | Yes, but too small for analysis |
| Local PC: `bibliography\3D-Models\SM_036475_GMO.ply` | One mesh, 151 MB | No. It is not on GitHub, and GitHub refuses files over 100 MB |

### 3.3 Reference labels

The Google Sheet "3D scans catalogue" records fingerprints in its `fingerprints`
column and in the scanning-day remarks (`fp`, `fp?`). These labels are the reference
for evaluation.

| Status | Tablets (SM) |
|---|---|
| Fingerprint: yes | 036475, 036917, 037316, 037319 (remarks only), 037321, 037324, 037326, 037377, 037717, 039043 (with seal), 039941, 043643, 043644 |
| Fingerprint: maybe | 036359, 036390, 036413, 036425, 036506 |
| Known scan artifacts (hard negatives) | Structured-light waves: 039043, 037244. Ink-label patches: 037318, 036425, 037323 |

The other tablets carry no fingerprint mark and are treated as probable negatives. A
blank field may also mean "not checked", so these labels should be confirmed.

## 4. Feasibility

| Quantity | Value | Consequence |
|---|---|---|
| Lateral resolution | ≈ 40 µm | 11–14 samples per adult ridge period (0.45–0.55 mm). About 3 is the minimum; about 6 is comfortable |
| Depth resolution | 5 µm | Ridge impressions in clay are typically 10–50 µm deep, so they are measurable |
| Equivalent image resolution | ≈ 635 ppi | Above the 500 ppi forensic minimum |

Three levels of ambition:

1. **Detection** (where are the prints?): feasible.
2. **Extraction and measurement** (clean print image; ridge breadth and density for
   age and sex estimation, as in the Tel Burna study): feasible.
3. **Identification** (the same person across tablets): uncertain. Ancient prints are
   partial and distorted. This is to be attempted only after levels 1 and 2 work on
   real data.

## 5. Method

The source is in `mesoprint/`. The steps, in order:

| Step | What happens | Main parameters (default) and rationale |
|---|---|---|
| Load | Binary PLY via `plyfile`, with a fast path for triangle meshes. Vertex normals are oriented out of the clay (signed volume) | Units taken as mm (`--scale` otherwise) |
| Patches | One patch centre per 2 mm voxel of the surface. Each patch is a 3 mm-radius disk | 6 mm diameter covers about 12 ridges; 2 mm stride gives overlap |
| Flatten | Best-fit plane (PCA). Points facing away from it are dropped. Heights are binned on a grid and small gaps filled | Grid = mesh spacing, at least 40 µm. Normal cut-off cos = 0.3 |
| Relief | Subtract a fitted quadratic (tablet curvature), then a Gaussian low-pass | σ = 0.8 mm; passes ridges unattenuated, removes clay undulation and large wedges |
| Features | `band_ratio`: energy share in 1.6–3.6 ridges/mm. `band_amp_um`: RMS ridge depth. `local_coherence`: orientation consistency at ~1 mm. `coverage`: how evenly the texture fills the patch. `straightness`: share of ridge area within ±12° of the dominant direction. `peak_freq`: ridge frequency after dividing out the clay's power-law spectrum | Band = ridge periods 0.28–0.63 mm, which spans children to adults |
| Score | `band_ratio × local_coherence × coverage × depth factor × straightness penalty` | Depth factor = a / (a + 2 µm). The penalty starts at straightness 0.75 and removes up to 80% |
| Candidates | Patches scoring ≥ 0.10 are linked when closer than 1.6 × stride. Groups are ranked by best score × √patches | Rewards extended ridge areas over single patches |
| Extraction | 6 mm radius. Binned at the scan's own resolution (no interpolation across holes), cubic-resampled to 1000 ppi, detrended (σ = 1 mm). The enhanced image is band-passed around the measured ridge frequency and locally contrast-normalised | The print image is the enhanced image mirrored left-right (clay is a negative), with grooves shown dark |
| Output | `overview.png` (fat cross with heat map), `candidates.json`, `patches.csv` (all features), per-candidate `*_relief/enhanced/print.png` with DPI and `*.json` measurements, optional score PLY | `patches.csv` is intended as training data for a later classifier |

The score is deliberately a transparent, hand-tuned product of interpretable
features. It is a starting point to be recalibrated, and eventually replaced by a
classifier trained on the labelled tablets.

## 6. Development log: findings and decisions

Changes are listed in the order they were made. The commit-level list is in
`CHANGELOG.md`.

1. **First calibration** (synthetic tablet, seed 0). The whorl print scored
   0.24–0.39 against at most 0.04 for wedges and clay: a clear separation. But
   straightness, first defined as whole-patch ÷ local coherence, also penalised the
   arch print. Its ridges are nearly parallel and scored 0.02–0.22. **Decision:**
   start the penalty later and lower the candidate threshold from 0.25 to 0.10.
   Real prints away from the core are often nearly parallel too.
2. **Stripe edge weakness** (seeds 3, 5, 11, 42). A patch straddling the edge of the
   synthetic scanner stripes scored up to 0.195. There, a wedge tail and the stripe
   boundary diluted the straightness measure (0.43), so the penalty did not apply.
   First fix: a coherence-weighted orientation resultant, which was not sufficient.
   Second fix: **straightness = share of ridge area within ±12° of the dominant
   direction**. The body of the stripe area now reads 0.95–1.0 and is suppressed.
   Patches straddling the stripe edge can still reach about 0.1–0.2. They remain
   single-patch hits that rank below both prints. **Decision:** stop tuning against
   a guessed artifact model and recalibrate on the real waves of SM 039043 and
   SM 037244.
3. **Performance** (2.6 M-vertex mesh). The patch grid followed the mesh spacing
   (18 µm), which made detection about 5× slower than needed. **Fix:** grid floor at
   40 µm (runtime 140 s → 75 s).
4. **Extraction speed.** Delaunay interpolation took about 8 s per candidate on dense
   meshes. **Fix:** bin at native resolution, then resample (runtime 75 s → 47 s).
   This also avoids interpolating across holes.
5. **Ridge-breadth bug.** Over a 12 mm extraction window, the arch print measured
   0.634 mm instead of 0.370 mm. Clay and wedge energy rises steeply towards low
   frequencies, so the raw spectral maximum sat at the lower edge of the ridge band.
   **Fix:** fit a power law to the spectrum outside the band, divide it out, then
   pick the peak. Error is now within ±5%, verified at 3 mm and 6 mm radius.
6. **Robustness for real files.**
   - Normals are oriented outwards by signed volume. Inward-wound meshes would
     otherwise swap ridges and grooves.
   - The loader falls back to plyfile's general reader if a file has non-triangle
     faces.
   - The code runs on Python 3.8.
   - Overview gaps are filled with the nearest rendered pixel inside the outline.
7. **Closed-mesh check.** A closed synthetic tablet (3.1 M vertices, 40 × 26 × 16 mm)
   ran end to end in about 70 s. The front print was found (score 0.65) and shown on
   the obverse in the fat cross. A second print on the side was not found, because
   the test mesh itself had only 0.7 mm vertex spacing there. This is an artifact
   of the test mesh, not of the pipeline, and a real scan is uniformly about 40 µm.

## 7. Validation on synthetic data

`mesoprint/synth.py` generates one face of a pillow-shaped tablet at 40 µm resolution
(30 × 22 mm, about 414 k vertices). It carries:

- multi-scale clay texture;
- rows of sharp-edged wedges;
- a **whorl** print (25 µm deep, ridge period 0.476 mm along one axis, 0.643 mm
  along the other);
- an **arch** print on the curved margin (20 µm deep, period 0.370 mm);
- a patch of straight **scanner-like stripes** (8 µm, period 0.5 mm, deliberately
  inside the ridge band);
- 2 µm measurement noise.

Results with the current code (`python scripts/evaluate_synthetic.py`). Seed 0 was
used for tuning; the others were not.

| seed | whorl | arch | stripes (max / core) | background (p99 / max) | rank whorl / arch / stripes | ridge breadth whorl (truth 0.476*) | ridge breadth arch (truth 0.370) |
|---|---|---|---|---|---|---|---|
| 0 | 0.393 | 0.241 | 0.109 / 0.056 | 0.030 / 0.036 | 1 / 2 / 3 | 0.471 | 0.354 (-4.3%) |
| 3 | 0.320 | 0.169 | 0.164 / 0.056 | 0.034 / 0.035 | 1 / 2 / 3 | 0.473 | 0.383 (+3.5%) |
| 5 | 0.365 | 0.178 | 0.140 / 0.054 | 0.041 / 0.043 | 1 / 2 / 3 | 0.473 | 0.355 (-4.0%) |
| 11 | 0.325 | 0.158 | 0.195 / 0.053 | 0.035 / 0.042 | 1 / 2 / 3 | 0.473 | 0.357 (-3.6%) |
| 42 | 0.379 | 0.178 | 0.105 / 0.058 | 0.038 / 0.041 | 1 / 2 / 3 | 0.471 | 0.358 (-3.4%) |

\* The whorl is elongated, with a period of 0.476 mm across and 0.643 mm along its
axis. The measurement reports the dominant one.

How to read the table:
- Both prints are always the top two candidates.
- Wedges and clay never exceed 0.043, well under the lowest print score (0.158).
- The body of the stripe area is suppressed to ≤ 0.058.
- The weakest point is a patch at the stripe edge. In seed 11 it scored 0.195, higher
  than the arch's best patch (0.158). It still ranks third, because the arch covers
  many patches and the stripe hit only a few. On real data such cases may need the
  region-level check planned in section 9.
- Measured ridge breadth is within 5% of the truth.

The automated tests (`pytest`, 12 tests) cover:
- feature behaviour on analytic patterns;
- PLY round-trip and normals, including inverted winding;
- the full pipeline on an unseen random tablet (seed 3).

Example outputs are in `docs/synthetic_example/`.

## 8. Known limitations

- **Not yet run on a real scan.** All thresholds are provisional.
- **Scanner waves** are modelled from a guess (straight, single period). Their real
  appearance on SM 039043 and SM 037244 may differ.
- **Flattening is a plane projection per region.** Prints wrapping round a tablet
  corner are foreshortened towards the rim. Proper unrolling (e.g. LSCM
  parametrisation) is not implemented.
- **The score is heuristic**, not learned. Its absolute values have no probabilistic
  meaning.
- **The overview render** assumes GigaMesh orientation (obverse facing +Z) and is a
  point-splat preview, not a publication rendering.
- **No interactive viewer yet.**

## 9. Next steps

1. **Run the tool on SM 036475** (catalogue: fingerprint yes), then on the other
   "yes" tablets. Compare candidates with the known print locations.
2. **Run it on negatives and artifact tablets** (039043, 037244 for waves; 037318,
   036425, 037323 for ink labels) and recalibrate the score and the straightness
   penalty.
3. **Add a region-level check** for artifacts: a stripe area keeps one orientation
   across neighbouring patches; a fingerprint does not.
4. **Build an interactive viewer** to accept or reject candidates. The decisions
   become training labels for a learned classifier, fed by `patches.csv` features.
5. **Measure ridge breadth on confirmed prints** and compare with published age and
   sex reference data.
6. **Later:** proper unrolling of curved prints, minutiae extraction, and
   cross-tablet comparison.

## 10. Getting data into the project

The development environment cannot read the local PC, heidICON, or Drive files over
10 MB. GitHub rejects any file over 100 MB, so `SM_036475_GMO.ply` (151 MB) cannot be
pushed as a normal file. Workable routes, easiest first:

1. **Run `mesoprint` locally and push only the results.** The results are a few MB:
   ```
   mesoprint detect bibliography\3D-Models\SM_036475_GMO.ply -o results\SM_036475 --heatmap-ply
   git add results
   git commit -m "SM 036475 detection results"
   git push
   ```
   The `.gitignore` on this branch ignores `*.ply`, so the model is not included by
   accident.
2. **Push a cropped region.** In GigaMesh, cut out the area with the print (or one
   side of the tablet) and save it as a binary PLY under 100 MB in `samples/`. That
   folder is allowed by `.gitignore`, and the development environment can then
   process it directly.
3. **Allow the heidICON hosts** (`heidicon.ub.uni-heidelberg.de`, `doi.org`) in the
   cloud environment's network settings. This gives direct access to all 44
   published models.

Git LFS would also work for large files, but it needs setup on both sides and has
bandwidth quotas.

### If GitHub rejected a push because of the model

A rejected push means the model is inside a local commit. Deleting the file afterwards
is not enough, because the commit still contains it. Undo the unpushed commit(s) while
keeping the files, then commit again without the model:

```
git status
git log --oneline origin/<branch>..HEAD        # commits GitHub does not have yet
git reset --soft origin/<branch>               # undo them; all changes stay staged
git restore --staged bibliography/3D-Models/SM_036475_GMO.ply
git commit -m "<same message as before>"
git push
```

Replace `<branch>` with the branch you were on (`main` or `claude/great-ride-ul2pd4`).
In GitHub Desktop, *History → right-click the latest commit → Undo commit* does the
same for the latest commit. Afterwards, untick the `.ply` file before committing
again.

## 11. Reproducing

```
pip install -e ".[test]"
pytest                                   # 12 tests, about 25 s
python scripts/evaluate_synthetic.py     # table in section 7, about 2 min
mesoprint synth test.ply                 # synthetic tablet + ground truth
mesoprint detect test.ply -o out --stride 1.5
```
