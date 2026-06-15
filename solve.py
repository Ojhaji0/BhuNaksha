"""
BhuMe take-home — main solution.

Algorithm
---------
For each plot:
  1. Apply the global median shift (one vector learned from example truths).
  2. Refine per-plot by cross-correlating the shifted plot's edge mask with
     boundaries.tif (ML-detected field edges). The cross-correlation peak gives
     an additional local offset that snaps the boundary onto real field edges.
  3. Estimate confidence from the cross-correlation peak quality — its
     sharpness *modulated by how unambiguous it is* (a sharp peak that has a
     near-equal rival is untrustworthy, e.g. a neighbour's edge in a dense
     village) — the area ratio (map_area / recorded_area ≈ 1.0 → fixable
     placement problem), and how consistent the total shift is with the global
     pattern.
  4. Flag the plot when the area ratio signals a genuine area problem, or when
     the edge signal is too weak to trust (confidence below threshold).

Usage
-----
  python solve.py Vadnerbhairav
  python solve.py Malatavadi
  python solve.py Vadnerbhairav Malatavadi   # both villages
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from scipy.signal import fftconvolve
from scipy.ndimage import binary_erosion
from shapely.affinity import translate
from shapely.ops import transform as shp_transform
from pathlib import Path
from pyproj import Transformer

from bhume_kit import load, write_predictions, score, global_median_shift

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Base search radius for local cross-correlation refinement (metres).
# The actual radius is scaled by pixel size of boundaries.tif:
#   adaptive_search = SEARCH_RADIUS_M * (px_m / 2.0)
# so fine-resolution imagery (dense villages) gets a tighter window (fewer
# false peaks from adjacent plots) and coarser imagery (open farmland) gets
# a wider window that can reach further from the global shift.
SEARCH_RADIUS_M = 20

# Plots whose map_area / recorded_area falls outside this band almost certainly
# have an area problem (subdivision, digitising error) rather than a pure
# placement drift — flagging protects calibration.
AREA_RATIO_MIN = 0.40
AREA_RATIO_MAX = 2.50

# Minimum combined confidence to emit a "corrected" record.
# Below this the fix is too uncertain; flagging is the honest answer.
CONFIDENCE_THRESHOLD = 0.28

# Minimum cross-correlation quality before we trust the local refinement.
# Below this we keep the global-shift-only position (safer for dense villages
# like Malatavadi where many nearby edges create false cross-corr peaks).
MIN_REFINE_QUALITY = 0.45


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _project(geom, t):
    """Project a shapely geometry using a pyproj Transformer."""
    return shp_transform(t.transform, geom)


def _edge_mask(geom_native, win_transform, width, height):
    """
    Rasterise the polygon edge (boundary pixels only) at the given resolution.
    Returns (mask float32, edge_pixel_count).
    """
    if geom_native is None or geom_native.is_empty:
        return np.zeros((height, width), np.float32), 0

    filled = rasterize(
        [(geom_native, 1)],
        out_shape=(height, width),
        transform=win_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    if filled.sum() == 0:
        return filled.astype(np.float32), 0

    # Thin edge: filled minus eroded interior
    eroded = binary_erosion(filled, iterations=2)
    edge = (filled.astype(np.int16) - eroded.astype(np.int16)).clip(0, 1)
    edge = edge.astype(np.float32)
    return edge, int(edge.sum())


# ──────────────────────────────────────────────────────────────────────────────
# Cross-correlation refinement
# ──────────────────────────────────────────────────────────────────────────────

def local_refine(
    shifted_geom_4326,
    bsrc: rasterio.DatasetReader,
    t_fwd,   # EPSG:4326 → boundaries CRS
    t_inv,   # boundaries CRS → EPSG:4326
    search_m: float = SEARCH_RADIUS_M,
    return_extra: bool = False,
):
    """
    Slide the shifted plot edge over boundaries.tif and find the offset that
    maximises edge-to-boundary alignment.

    Returns
    -------
    (dx_lon, dy_lat, edge_quality)                      when return_extra=False
    (dx_lon, dy_lat, edge_quality, unambiguity, improve) when return_extra=True
        dx_lon, dy_lat : refinement shift in degrees (EPSG:4326)
        edge_quality   : normalised peak prominence ∈ [0, 1]
        unambiguity    : 1 − (2nd peak prominence / top peak prominence) ∈ [0, 1];
                         low when several comparable peaks compete (dense villages,
                         neighbour edges) → the chosen peak is not trustworthy.
        improve        : raw alignment gain of the chosen offset over the
                         global-shift (centre) position, normalised ∈ [0, 1].
    """
    def _ret(dx, dy, eq, unambig=0.0, improve=0.0):
        return (dx, dy, eq, unambig, improve) if return_extra else (dx, dy, eq)

    geom_nat = _project(shifted_geom_4326, t_fwd)
    bounds = geom_nat.bounds   # (minx, miny, maxx, maxy) in boundaries CRS

    px_m       = abs(bsrc.transform.a)          # metres per pixel
    search_px  = max(4, int(search_m / px_m))
    pad_px     = search_px + 4

    # Map bounding box to pixel coordinates
    try:
        r_top, c_left  = bsrc.index(bounds[0], bounds[3])
        r_bot, c_right = bsrc.index(bounds[2], bounds[1])
    except Exception:
        return _ret(0.0, 0.0, 0.0)

    r0 = max(0, min(r_top,  r_bot)   - pad_px)
    r1 = min(bsrc.height, max(r_top,  r_bot)   + pad_px)
    c0 = max(0, min(c_left, c_right) - pad_px)
    c1 = min(bsrc.width,  max(c_left, c_right) + pad_px)

    if r1 - r0 < 6 or c1 - c0 < 6:
        return _ret(0.0, 0.0, 0.0)

    win       = Window(c0, r0, c1 - c0, r1 - r0)
    bpatch    = bsrc.read(1, window=win).astype(np.float32)
    win_tr    = bsrc.window_transform(win)
    ph, pw    = bpatch.shape

    bmax = bpatch.max()
    if bmax < 1.0:
        return _ret(0.0, 0.0, 0.0)
    bpatch_n = bpatch / bmax

    edge, edge_cnt = _edge_mask(geom_nat, win_tr, pw, ph)
    if edge_cnt < 6:
        return _ret(0.0, 0.0, 0.0)

    edge_n = edge / edge.sum()

    # FFT cross-correlation: find shift that maximises edge ∩ boundary signal
    xcorr_raw = fftconvolve(bpatch_n, edge_n[::-1, ::-1], mode="same")

    # Gaussian spatial prior: strongly penalise shifts far from the current
    # (globally-shifted) position.  This defeats false peaks that arise when a
    # nearby plot's edge looks similar to our target edge.
    # sigma = half the search radius so the prior is still permissive within ±SEARCH_RADIUS_M.
    sigma_px = max(3.0, search_px / 2.0)
    cr, cc = xcorr_raw.shape[0] // 2, xcorr_raw.shape[1] // 2
    rr, cc_g = np.ogrid[:xcorr_raw.shape[0], :xcorr_raw.shape[1]]
    gaussian_prior = np.exp(-((rr - cr)**2 + (cc_g - cc)**2) / (2 * sigma_px**2))
    xcorr = xcorr_raw * gaussian_prior

    # Restrict peak search to ±search_px from centre
    rl = max(0, cr - search_px);  rh = min(xcorr.shape[0], cr + search_px + 1)
    cl = max(0, cc - search_px);  ch = min(xcorr.shape[1], cc + search_px + 1)
    region = xcorr[rl:rh, cl:ch]

    peak_idx = np.unravel_index(np.argmax(region), region.shape)
    peak_val  = float(region[peak_idx])
    mean_val  = float(region.mean())
    std_val   = float(region.std())

    # Normalised peak prominence (z-score → [0,1])
    z = (peak_val - mean_val) / (std_val + 1e-8)
    edge_quality = float(np.clip(z / 6.0, 0.0, 1.0))

    # ── Unambiguity (Lowe-style ratio test) ─────────────────────────────────
    # Mask a small neighbourhood around the top peak and find the next-best
    # competing peak.  When a second peak is nearly as strong (dense villages,
    # neighbour edges) the chosen offset is not trustworthy → unambiguity → 0.
    pr, pc = peak_idx
    masked = region.copy()
    excl = max(2, int(round(2.0 / px_m)) + 1)   # ~2 m exclusion radius
    r_lo = max(0, pr - excl); r_hi = min(region.shape[0], pr + excl + 1)
    c_lo = max(0, pc - excl); c_hi = min(region.shape[1], pc + excl + 1)
    masked[r_lo:r_hi, c_lo:c_hi] = mean_val
    peak2_val = float(masked.max())
    prom1 = peak_val - mean_val
    prom2 = peak2_val - mean_val
    unambig = float(np.clip(1.0 - prom2 / (prom1 + 1e-8), 0.0, 1.0)) if prom1 > 0 else 0.0

    # ── Improvement over the global-shift (centre) position ─────────────────
    # Raw (prior-free) alignment gain of the chosen offset vs leaving the plot
    # at the global shift.  Uses xcorr_raw so the spatial prior doesn't bias it.
    centre_val = float(xcorr_raw[cr, cc])
    raw_peak   = float(xcorr_raw[rl + pr, cl + pc])
    improve = float(np.clip((raw_peak - centre_val) / (abs(raw_peak) + 1e-8), 0.0, 1.0))

    # Pixel shift (row increases downward → negate for y)
    dy_px = peak_idx[0] - (cr - rl)
    dx_px = peak_idx[1] - (cc - cl)

    if dx_px == 0 and dy_px == 0:
        return _ret(0.0, 0.0, edge_quality, unambig, improve)

    # Pixel shift → metres → lon/lat delta
    dx_m = dx_px * px_m
    dy_m = -dy_px * px_m  # negate: row↓ ⟹ south

    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    lon0, lat0 = t_inv.transform(cx,        cy)
    lon1, lat1 = t_inv.transform(cx + dx_m, cy + dy_m)

    return _ret(lon1 - lon0, lat1 - lat0, edge_quality, unambig, improve)


# ──────────────────────────────────────────────────────────────────────────────
# Confidence scoring
# ──────────────────────────────────────────────────────────────────────────────

def _area_ratio(map_sqm, rec_sqm):
    """Ratio of drawn area to recorded area. 1.0 = match. None rec_sqm → 1.0."""
    if not rec_sqm or rec_sqm <= 0:
        return 1.0
    return float(max(0.01, map_sqm / rec_sqm))


def _confidence(edge_quality, ar, total_dx, total_dy, gdx, gdy, unambig=1.0):
    """
    Calibrated confidence score.

    edge_quality  — cross-correlation peak prominence
    unambig       — peak unambiguity (1 − 2nd-peak ratio); how much the chosen
                    peak dominates its competitors
    ar            — map_area / recorded_area
    total_dx/dy   — actual shift applied (degrees)
    gdx/gdy       — global shift (degrees)

    Higher confidence → plot is more likely to be correctly placed after
    correction. This ordering matters for AUC more than the absolute values.

    A sharp cross-correlation peak is only trustworthy when it is also
    *unambiguous*.  In dense villages a plot can snap confidently onto a
    neighbour's edge — a sharp but ambiguous peak.  Folding `unambig` into the
    edge term (Lowe-style ratio test) down-weights exactly those cases, which
    is what fixes calibration there.  Verified on the example truths: this
    raises calibration AUC on both villages versus using raw edge_quality.
    """
    # Edge term: sharp peak counts fully only when unambiguous
    eq_eff = edge_quality * (0.35 + 0.65 * float(np.clip(unambig, 0.0, 1.0)))

    # Area factor: penalty for area ratio far from 1.0
    log_ar    = abs(np.log(max(ar, 0.01)))
    area_f    = float(np.exp(-log_ar * 1.5))

    # Shift-consistency factor: large deviation from global → less certain
    dev_m = np.sqrt((total_dx - gdx)**2 + (total_dy - gdy)**2) * 111_000
    shift_f = float(np.exp(-dev_m / 25.0))

    # Weighted combination
    conf = 0.55 * eq_eff + 0.25 * area_f + 0.20 * shift_f
    return float(np.clip(conf, 0.05, 0.95))


# ──────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────────────

def process_village(village_path: str) -> gpd.GeoDataFrame:
    print(f"\n{'='*60}")
    print(f"Village: {village_path}")
    print("="*60)

    bundle = load(village_path)
    plots  = bundle.plots

    # ── Global median shift ────────────────────────────────────────────────
    gdx, gdy = global_median_shift(village_path)
    gdx_m = gdx * 111_000
    gdy_m = gdy * 111_000
    print(f"Global shift: dx={gdx_m:+.1f} m  dy={gdy_m:+.1f} m")

    # ── Coordinate transformers ────────────────────────────────────────────
    with rasterio.open(bundle.boundaries_path) as bsrc:
        nat_epsg = bsrc.crs.to_epsg()

    t_fwd = Transformer.from_crs(4326, nat_epsg, always_xy=True)
    t_inv = Transformer.from_crs(nat_epsg, 4326,  always_xy=True)

    results = []

    with rasterio.open(bundle.boundaries_path) as bsrc:
        px_m = abs(bsrc.transform.a)                   # metres per pixel
        search_m = SEARCH_RADIUS_M * px_m / 2.0        # adaptive: ~12m fine, ~24m coarse
        print(f"Pixel size: {px_m:.2f} m  →  adaptive search radius: {search_m:.1f} m")
        n = len(plots)
        print(f"Processing {n} plots…")

        for idx, (_, row) in enumerate(plots.iterrows()):
            if idx % 500 == 0 and idx:
                pct = 100 * idx // n
                print(f"  {idx}/{n}  ({pct}%)")

            geom    = row.geometry
            pn      = str(row["plot_number"])
            map_a   = float(row.get("map_area_sqm") or 0)
            rec_a   = row.get("recorded_area_sqm")

            ar = _area_ratio(map_a, rec_a)

            # ── Step 1: apply global shift ─────────────────────────────────
            glob_shifted = translate(geom, gdx, gdy)

            # ── Step 2: local cross-correlation refinement ─────────────────
            try:
                ldx, ldy, eq, unambig, _ = local_refine(
                    glob_shifted, bsrc, t_fwd, t_inv, search_m=search_m,
                    return_extra=True,
                )
            except Exception:
                ldx, ldy, eq, unambig = 0.0, 0.0, 0.0, 0.0

            # Only apply the local delta when the cross-correlation peak is
            # clearly prominent.  Dense villages (small plots, many nearby
            # edges) produce noisy peaks that move plots to wrong positions;
            # falling back to global-only is safer there.
            if eq >= MIN_REFINE_QUALITY:
                total_dx = gdx + ldx
                total_dy = gdy + ldy
            else:
                total_dx = gdx
                total_dy = gdy
                ldx = ldy = 0.0

            final_geom = translate(geom, total_dx, total_dy)

            # ── Step 3: confidence ─────────────────────────────────────────
            conf = _confidence(eq, ar, total_dx, total_dy, gdx, gdy, unambig=unambig)

            # ── Step 4: flagging decision ──────────────────────────────────
            if ar < AREA_RATIO_MIN or ar > AREA_RATIO_MAX:
                status = "flagged"
                note   = f"area_ratio={ar:.2f} (outside [{AREA_RATIO_MIN},{AREA_RATIO_MAX}])"
                out_g  = geom
                conf   = 0.0
            elif conf < CONFIDENCE_THRESHOLD:
                status = "flagged"
                note   = f"low confidence ({conf:.2f}), eq={eq:.2f}"
                out_g  = geom
            else:
                status = "corrected"
                note   = (
                    f"global_shift+xcorr eq={eq:.2f} ar={ar:.2f} "
                    f"dx={total_dx*111000:+.1f}m dy={total_dy*111000:+.1f}m"
                )
                out_g  = final_geom

            results.append({
                "plot_number": pn,
                "status":      status,
                "confidence":  round(conf, 4),
                "method_note": note,
                "geometry":    out_g,
            })

    out_gdf = gpd.GeoDataFrame(results, crs="EPSG:4326")

    n_cor = int((out_gdf["status"] == "corrected").sum())
    n_flg = int((out_gdf["status"] == "flagged").sum())
    print(f"Done: corrected={n_cor}  flagged={n_flg}")

    # ── Write output ───────────────────────────────────────────────────────
    out_path = str(Path(village_path) / "predictions.geojson")
    write_predictions(out_path, out_gdf)

    # ── Self-score ─────────────────────────────────────────────────────────
    print("\nSelf-score vs example truths:")
    score(out_gdf, village_path)

    return out_gdf


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["Vadnerbhairav"]
    for vpath in targets:
        process_village(vpath)
