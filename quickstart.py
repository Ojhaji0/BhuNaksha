"""
BhuMe take-home — quickstart demo.

Runs the full loop for one village:
  load → inspect one plot's imagery → apply global-median-shift baseline
  → save → self-score → save a patch image

Usage:
  python quickstart.py Vadnerbhairav
  python quickstart.py Malatavadi
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import rasterio
from PIL import Image
from shapely.affinity import translate

from bhume_kit import load, write_predictions, score, global_median_shift, patch_for_plot

village_path = sys.argv[1] if len(sys.argv) > 1 else "Vadnerbhairav"

print(f"\n── BhuMe quickstart: {village_path} ──")

# ── 1. Load ────────────────────────────────────────────────────────────────────
bundle = load(village_path)
plots  = bundle.plots
print(f"Loaded {len(plots)} plots.")

# ── 2. Inspect one plot's imagery ─────────────────────────────────────────────
example_plot = plots.iloc[0]
print(f"\nExample plot: {example_plot['plot_number']}")
print(f"  map_area_sqm:      {example_plot['map_area_sqm']:.0f}")
print(f"  recorded_area_sqm: {example_plot['recorded_area_sqm']}")
print(f"  geometry type:     {example_plot.geometry.geom_type}")

with rasterio.open(bundle.imagery_path) as src:
    data, window, win_tr = patch_for_plot(src, example_plot.geometry, padding=2.0)
    # Save RGB patch as PNG
    if data.shape[0] >= 3:
        rgb = data[:3].transpose(1, 2, 0)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save("patch_example.png")
        print(f"  Saved patch_example.png  ({rgb.shape[1]}×{rgb.shape[0]} px)")

# ── 3. Naive baseline: one global shift for every plot ────────────────────────
gdx, gdy = global_median_shift(village_path)
print(f"\nGlobal median shift: dx={gdx*111000:+.1f} m  dy={gdy*111000:+.1f} m")

rows = []
for _, row in plots.iterrows():
    shifted = translate(row.geometry, gdx, gdy)
    rows.append({
        "plot_number": str(row["plot_number"]),
        "status":      "corrected",
        "confidence":  0.5,          # flat → AUC ≈ 0.5 (baseline floor)
        "method_note": "global_median_shift",
        "geometry":    shifted,
    })

baseline_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

import os, pathlib
out_path = str(pathlib.Path(village_path) / "predictions_baseline.geojson")
write_predictions(out_path, baseline_gdf)

# ── 4. Self-score ──────────────────────────────────────────────────────────────
print("\nSelf-score (baseline):")
score(baseline_gdf, village_path)

print("\nDone. Run solve.py for the full per-plot refinement:\n"
      f"  python solve.py {village_path}\n")
