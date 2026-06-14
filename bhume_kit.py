"""
BhuMe take-home — helper functions.

Mirrors the official starter-kit API so solve.py can use familiar names.
All geometries are EPSG:4326 (lon, lat) unless otherwise noted.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
import rasterio.transform
from shapely.affinity import translate


# ─── data container ────────────────────────────────────────────────────────────

@dataclass
class VillageBundle:
    plots: gpd.GeoDataFrame          # EPSG:4326
    imagery_path: str
    boundaries_path: str
    example_truths: Optional[gpd.GeoDataFrame]
    village_path: str
    village: str = field(init=False)

    def __post_init__(self):
        self.village = self.plots["village"].iloc[0] if len(self.plots) else ""


# ─── load ───────────────────────────────────────────────────────────────────────

def load(village_path: str) -> VillageBundle:
    """Load all village data in one call."""
    p = Path(village_path)
    plots = gpd.read_file(p / "input.geojson")
    tp = p / "example_truths.geojson"
    truths = gpd.read_file(tp) if tp.exists() else None
    return VillageBundle(
        plots=plots,
        imagery_path=str(p / "imagery.tif"),
        boundaries_path=str(p / "boundaries.tif"),
        example_truths=truths,
        village_path=str(p),
    )


# ─── coordinate conversion ─────────────────────────────────────────────────────

def _get_transformer(from_epsg: int, to_epsg: int):
    from pyproj import Transformer
    return Transformer.from_crs(from_epsg, to_epsg, always_xy=True)


def lonlat_to_pixel(
    src: rasterio.DatasetReader, lon: float, lat: float
) -> Tuple[int, int]:
    """
    Convert (lon, lat) in EPSG:4326 to (col, row) pixel coords in src.
    """
    t = _get_transformer(4326, src.crs.to_epsg())
    x, y = t.transform(lon, lat)
    row, col = src.index(x, y)
    return int(col), int(row)


def pixel_to_lonlat(
    src: rasterio.DatasetReader, col: int, row: int
) -> Tuple[float, float]:
    """
    Convert (col, row) pixel coords in src to (lon, lat) in EPSG:4326.
    """
    x, y = rasterio.transform.xy(src.transform, row, col)
    t = _get_transformer(src.crs.to_epsg(), 4326)
    return t.transform(x, y)


# ─── patch extraction ──────────────────────────────────────────────────────────

def patch_for_plot(
    src: rasterio.DatasetReader,
    geom,                      # shapely geometry in EPSG:4326
    padding: float = 1.5,
) -> Tuple[np.ndarray, Window, rasterio.transform.Affine]:
    """
    Return (data, window, win_transform) for the padded bounding box of geom.
    data shape: (bands, H, W).
    """
    b = geom.bounds  # (minx, miny, maxx, maxy) in lon/lat
    dw = (b[2] - b[0]) * (padding - 1)
    dh = (b[3] - b[1]) * (padding - 1)
    padded = (b[0] - dw, b[1] - dh, b[2] + dw, b[3] + dh)

    t = _get_transformer(4326, src.crs.to_epsg())
    left, bottom = t.transform(padded[0], padded[1])
    right, top   = t.transform(padded[2], padded[3])

    window = from_bounds(left, bottom, right, top, src.transform)
    # clamp to raster extent
    full = Window(0, 0, src.width, src.height)
    window = window.intersection(full)

    data = src.read(window=window)
    win_transform = src.window_transform(window)
    return data, window, win_transform


# ─── output ────────────────────────────────────────────────────────────────────

def write_predictions(path: str, gdf: gpd.GeoDataFrame) -> None:
    """
    Write predictions.geojson in the required contract format.
    Required columns: plot_number, status, confidence, method_note, geometry.
    """
    required = ["plot_number", "status", "confidence", "method_note", "geometry"]
    for col in required[:-1]:
        if col not in gdf.columns:
            gdf = gdf.copy()
            gdf[col] = None
    out = gdf[required].copy()
    out.to_file(path, driver="GeoJSON")
    corrected = int((out["status"] == "corrected").sum())
    flagged   = int((out["status"] == "flagged").sum())
    print(f"[write_predictions] {len(out)} plots → {path}")
    print(f"  corrected={corrected}  flagged={flagged}")


# ─── scoring ──────────────────────────────────────────────────────────────────

def score(preds: gpd.GeoDataFrame, village_path: str) -> dict:
    """
    Score predictions against example truths.
    Computes: median IoU, centroid error, AUC (calibration), Spearman.
    """
    from scipy.stats import spearmanr

    p = Path(village_path)
    truths   = gpd.read_file(p / "example_truths.geojson")
    official = gpd.read_file(p / "input.geojson")

    CRS_M = "EPSG:32643"   # UTM 43N — metres, covers Maharashtra
    preds_m   = preds.to_crs(CRS_M)
    truths_m  = truths.to_crs(CRS_M)
    official_m = official.to_crs(CRS_M)

    rows = []
    for _, truth in truths_m.iterrows():
        pn = str(truth["plot_number"])
        pred_row = preds_m[preds_m["plot_number"] == pn]
        off_row  = official_m[official_m["plot_number"] == pn]
        if pred_row.empty:
            continue
        pr = pred_row.iloc[0]
        if pr["status"] == "flagged":
            continue

        def iou(a, b):
            u = a.union(b).area
            return a.intersection(b).area / u if u > 0 else 0.0

        pgeom = pr.geometry
        tgeom = truth.geometry
        ogeom = off_row.iloc[0].geometry if not off_row.empty else pgeom

        rows.append({
            "iou_pred": iou(pgeom, tgeom),
            "iou_off":  iou(ogeom, tgeom),
            "cerr_m":   pgeom.centroid.distance(tgeom.centroid),
            "conf":     float(pr.get("confidence", 0.5)),
        })

    if not rows:
        print("[score] No example plots scored (all flagged or missing).")
        return {}

    iou_p = [r["iou_pred"] for r in rows]
    iou_o = [r["iou_off"]  for r in rows]
    cerr  = [r["cerr_m"]   for r in rows]
    conf  = [r["conf"]      for r in rows]

    med_p = float(np.median(iou_p))
    med_o = float(np.median(iou_o))
    med_c = float(np.median(cerr))

    auc_val = None
    sp_val  = None
    if len(conf) >= 2 and len(set(conf)) > 1:
        sp_r, _ = spearmanr(conf, iou_p)
        sp_val  = float(sp_r)
        n = len(conf)
        conc = tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                if conf[i] != conf[j]:
                    tot += 1
                    if (conf[i] > conf[j]) == (iou_p[i] > iou_p[j]):
                        conc += 1
        auc_val = conc / tot if tot else None

    print(f"[score] {len(rows)}/{len(truths)} example plots:")
    print(f"  median IoU  pred={med_p:.3f}  official={med_o:.3f}  Δ={med_p-med_o:+.3f}")
    print(f"  median centroid error: {med_c:.1f} m")
    auc_s = f"{auc_val:.3f}" if auc_val is not None else "—"
    sp_s  = f"{sp_val:.3f}"  if sp_val  is not None else "—"
    print(f"  calibration AUC: {auc_s}   Spearman: {sp_s}")

    return {
        "n": len(rows),
        "median_iou_pred":     med_p,
        "median_iou_official": med_o,
        "improvement":         med_p - med_o,
        "median_centroid_error_m": med_c,
        "auc":      auc_val,
        "spearman": sp_val,
    }


# ─── global median shift ──────────────────────────────────────────────────────

def global_median_shift(village_path: str) -> Tuple[float, float]:
    """
    Compute (dx_lon, dy_lat): median centroid shift from official → truth.
    Apply this single offset to every plot as the naive baseline.
    """
    p = Path(village_path)
    official = gpd.read_file(p / "input.geojson")
    truths   = gpd.read_file(p / "example_truths.geojson")

    dxs, dys = [], []
    for _, t in truths.iterrows():
        pn = str(t["plot_number"])
        o = official[official["plot_number"] == pn]
        if o.empty:
            continue
        tc = t.geometry.centroid
        oc = o.iloc[0].geometry.centroid
        dxs.append(tc.x - oc.x)
        dys.append(tc.y - oc.y)

    if not dxs:
        return 0.0, 0.0
    return float(np.median(dxs)), float(np.median(dys))
