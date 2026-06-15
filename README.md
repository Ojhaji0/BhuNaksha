# BhuMe Take-Home — Cadastral Boundary Correction

Official Indian village land-plot boundaries are georeferenced from old
hand-drawn cadastral maps. This georeferencing drifts the recorded boundaries
**5–20 m** away from where the fields actually sit on the ground. This project
corrects each plot's boundary back onto its real field using satellite imagery
and ML-detected field edges, and attaches a calibrated confidence score to every
correction.

## Approach

The core insight: within a single village sheet most of the drift is
**coherent** — the whole sheet slid in roughly one direction during
georeferencing. So correction happens in two stages, global then local.

1. **Global median shift** — From the few hand-checked `example_truths`, compute
   `truth_centroid − official_centroid` for each plot and take the **median**
   (robust to outliers). This single `(dx, dy)` vector is applied to every plot
   as a first pass.

2. **Local cross-correlation refinement** — For each plot individually, the
   shifted polygon's edge mask is cross-correlated (FFT, `scipy.signal.fftconvolve`)
   against the `boundaries.tif` raster of ML-detected field edges. The
   cross-correlation peak gives a residual per-plot offset that snaps the
   boundary onto the real field edge.

   - A **Gaussian spatial prior** centered on the global-shift position
     multiplies the correlation surface, suppressing false peaks that arise when
     a *neighbouring* plot's edge looks similar to the target — critical in dense
     villages.
   - The **search radius adapts to pixel size**: fine-resolution imagery (dense
     villages) gets a tighter window (~12 m), coarse imagery (open farmland) a
     wider one (~24 m).
   - A **quality gate** (`MIN_REFINE_QUALITY`) only trusts the local delta when
     the correlation peak is clearly prominent; otherwise it falls back to the
     global shift alone.

3. **Confidence** — A calibrated score combining three independent signals:

   | Signal | Weight | Meaning |
   |---|---:|---|
   | Edge quality × **unambiguity** | 55% | Peak sharpness, *modulated by how much it dominates its nearest rival* (Lowe-style ratio test) — a sharp but ambiguous peak (e.g. a neighbour's edge) is untrustworthy |
   | Area factor `exp(−|ln(area_ratio)|·1.5)` | 25% | `area_ratio ≈ 1.0` → fixable placement problem |
   | Shift consistency `exp(−dev/25 m)` | 20% | Large deviation from global shift is suspicious |

   The **unambiguity** factor (`1 − 2nd-peak prominence / top-peak prominence`)
   is what makes confidence calibrated in dense villages: there a plot can snap
   *confidently onto the wrong edge*, producing a sharp peak that raw prominence
   would reward. Folding unambiguity into the edge term raised calibration on
   both villages (see below) without changing any geometry.

4. **Flagging (restraint)** — A plot is **flagged** and left *unmoved* when:
   - `area_ratio` falls outside `[0.40, 2.50]` — an area problem (subdivision,
     digitising error), not a placement drift that translation can fix; or
   - confidence is below `0.28` — the edge signal is too weak to trust.

   Leaving a plot in place is more honest than moving it on a guess.

## Results (vs public `example_truths`)

| Village | Median IoU (official → ours) | Accurate @ IoU≥.5 | Median centroid err | Calibration AUC |
|---|---|---|---|---|
| Vadnerbhairav (open farmland, 2.4 m/px) | 0.612 → **0.870** (+0.233, 100% of plots improved) | **100%** | **4.5 m** | 0.73 → **0.80** |
| Malatavadi (dense, 1.2 m/px) | 0.510 → **0.566** (+0.149, 67% of plots improved) | **67%** | **5.6 m** | 0.00 → **0.33** |

Vadnerbhairav improves strongly because field edges are clear and the
correlation finds clean peaks. Malatavadi is harder — crowded adjacent
boundaries create ambiguous correlation surfaces — where the Gaussian prior,
tighter adaptive search radius, and the unambiguity-weighted confidence matter
most.

**Calibration** was improved by the unambiguity factor described above: it
moved both villages up (Vadnerbhairav 0.73→0.80; Malatavadi, whose single
example *miss* had a sharp-but-ambiguous peak that previously earned top
confidence, 0.00→0.33) **without changing any geometry**, so IoU and centroid
error are untouched. The diagnostic that drove this — measuring each candidate
signal against the actual per-plot IoU — is in [`diagnose_confidence.py`](diagnose_confidence.py).

> IoU/centroid figures are from the official in-browser self-score tool;
> calibration figures are from the local scorer in `bhume_kit.py`, which mirrors
> the same concordance metric. All are scored against the small public example
> set (6 plots in Vadnerbhairav, 3 in Malatavadi), so calibration especially is
> noisy — treat these as a directional check, not a grade. Malatavadi stalls at
> 0.33 because its best-placed example plot genuinely has the weakest edge
> signal; pushing past that on 3 points would just be overfitting. The real
> grade uses a larger hidden set.

## Usage

```bash
# install dependencies
pip install geopandas rasterio shapely numpy scipy pyproj

# run one or both villages
python solve.py Vadnerbhairav
python solve.py Malatavadi
python solve.py Vadnerbhairav Malatavadi
```

Each run writes `<village>/predictions.geojson` (EPSG:4326 FeatureCollection)
and prints a self-score against that village's `example_truths`.

## Repository layout

```
solve.py                     Main solution (algorithm + CLI)
bhume_kit.py                 Provided I/O + scoring helpers
quickstart.py                Provided starter
<village>/
  input.geojson              Official (drifted) plot boundaries
  example_truths.geojson     Hand-checked ground truth for a few plots
  predictions.geojson        Output: corrected boundaries + confidence
transcripts/
  ai_conversation.md         AI-assisted development transcript
```

> Large rasters (`imagery.tif`, `boundaries.tif`) are git-ignored to stay within
> GitHub size limits. Place them under each `<village>/` directory to reproduce.

## Output schema

Each feature in `predictions.geojson` carries:

| Property | Description |
|---|---|
| `plot_number` | Plot identifier |
| `status` | `corrected` or `flagged` |
| `confidence` | Calibrated score in `[0, 1]` (`0` for flagged area outliers) |
| `method_note` | Human-readable explanation of the decision |
| `geometry` | Corrected polygon (or original, if flagged) |

## What I'd improve with more time

- **Rotation correction** — the global stage assumes pure translation; some
  sheets were also slightly rotated during georeferencing.
- **Density-aware confidence** — the reliability of the edge signal depends on
  how crowded a village is. The unambiguity factor handles much of this, but an
  explicit local plot-density term could weight the edge vs. shift-consistency
  signals per neighbourhood rather than globally.
- **Multi-scale cross-correlation** — coarse-to-fine search for robustness to
  large drifts while keeping sub-pixel precision.

> Note: an earlier idea — confidence from *alignment improvement*
> (`alignment(corrected) − alignment(global-only)`) — was implemented and
> measured against the example truths in `diagnose_confidence.py`, but it
> correlated **negatively** with accuracy on both villages, so it was dropped in
> favour of the unambiguity (Lowe-ratio) factor. Measuring candidate signals
> before committing to them is the point of that diagnostic.
