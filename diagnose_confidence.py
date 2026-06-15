"""
Diagnostic: for each example-truth plot, run the same correction pipeline and
record every confidence signal alongside the actual IoU achieved against the
truth. Lets us see which signal predicts accuracy (and rebuild confidence to
maximise calibration AUC) instead of guessing.

Usage:
  py -3.10 diagnose_confidence.py Vadnerbhairav Malatavadi
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import rasterio
from shapely.affinity import translate
from pathlib import Path
from pyproj import Transformer

from bhume_kit import load, global_median_shift
from solve import (
    local_refine, _area_ratio, _confidence,
    SEARCH_RADIUS_M, MIN_REFINE_QUALITY,
)

CRS_M = "EPSG:32643"


def iou(a, b):
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def diagnose(village_path):
    print(f"\n{'='*78}\n{village_path}\n{'='*78}")
    bundle = load(village_path)
    plots = bundle.plots
    truths = bundle.example_truths
    gdx, gdy = global_median_shift(village_path)

    with rasterio.open(bundle.boundaries_path) as bsrc:
        nat = bsrc.crs.to_epsg()
    t_fwd = Transformer.from_crs(4326, nat, always_xy=True)
    t_inv = Transformer.from_crs(nat, 4326, always_xy=True)

    truths_m = truths.to_crs(CRS_M)

    rows = []
    with rasterio.open(bundle.boundaries_path) as bsrc:
        px_m = abs(bsrc.transform.a)
        search_m = SEARCH_RADIUS_M * px_m / 2.0
        for _, tr in truths.iterrows():
            pn = str(tr["plot_number"])
            prow = plots[plots["plot_number"].astype(str) == pn]
            if prow.empty:
                continue
            row = prow.iloc[0]
            geom = row.geometry
            map_a = float(row.get("map_area_sqm") or 0)
            rec_a = row.get("recorded_area_sqm")
            ar = _area_ratio(map_a, rec_a)

            glob = translate(geom, gdx, gdy)
            try:
                ldx, ldy, eq, unambig, improve = local_refine(
                    glob, bsrc, t_fwd, t_inv, search_m=search_m, return_extra=True
                )
            except Exception:
                ldx, ldy, eq, unambig, improve = 0.0, 0.0, 0.0, 0.0, 0.0

            if eq >= MIN_REFINE_QUALITY:
                tdx, tdy = gdx + ldx, gdy + ldy
            else:
                tdx, tdy = gdx, gdy

            final = translate(geom, tdx, tdy)
            conf = _confidence(eq, ar, tdx, tdy, gdx, gdy)

            # actual IoU vs truth (in metres CRS)
            tgeom = truths_m[truths_m["plot_number"].astype(str) == pn].iloc[0].geometry
            fgeom_m = gpd.GeoSeries([final], crs="EPSG:4326").to_crs(CRS_M).iloc[0]
            iou_pred = iou(fgeom_m, tgeom)

            # IoU if we used the GLOBAL shift only (no local refinement)
            gg_m = gpd.GeoSeries([glob], crs="EPSG:4326").to_crs(CRS_M).iloc[0]
            iou_global = iou(gg_m, tgeom)

            # area factor / shift factor for inspection
            area_f = float(np.exp(-abs(np.log(max(ar, 0.01))) * 1.5))
            dev_m = np.sqrt((tdx - gdx)**2 + (tdy - gdy)**2) * 111_000
            shift_f = float(np.exp(-dev_m / 25.0))

            rows.append(dict(pn=pn, iou=iou_pred, iou_g=iou_global, conf=conf,
                             eq=eq, unambig=unambig, eq_unambig=eq * unambig,
                             improve=improve, area_f=area_f, shift_f=shift_f, ar=ar))

    print(f"{'plot':>10} {'IoU':>6} {'IoU_glob':>8} {'Δlocal':>7} {'conf':>6} "
          f"{'eq':>6} {'unamb':>6} {'improv':>6} {'shift_f':>7} {'ar':>5}")
    for r in rows:
        dl = r['iou'] - r['iou_g']
        print(f"{r['pn']:>10} {r['iou']:6.3f} {r['iou_g']:8.3f} {dl:+7.3f} "
              f"{r['conf']:6.3f} {r['eq']:6.3f} {r['unambig']:6.3f} "
              f"{r['improve']:6.3f} {r['shift_f']:7.3f} {r['ar']:5.2f}")
    med_loc = float(np.median([r['iou'] for r in rows]))
    med_glob = float(np.median([r['iou_g'] for r in rows]))
    print(f"  median IoU:  global+local={med_loc:.3f}   global-only={med_glob:.3f}   "
          f"Δ={med_loc-med_glob:+.3f}")

    # rank-correlation of each signal vs IoU
    from scipy.stats import spearmanr
    print("\nSpearman(signal, IoU) — higher = better predictor of accuracy:")
    for key in ["conf", "eq", "unambig", "eq_unambig", "improve", "area_f", "shift_f"]:
        vals = [r[key] for r in rows]
        if len(set(vals)) > 1 and not any(np.isnan(vals)):
            sr, _ = spearmanr(vals, [r["iou"] for r in rows])
            print(f"  {key:>8}: {sr:+.3f}")
        else:
            print(f"  {key:>8}:   (constant/NaN)")
    return rows


def auc_concordance(conf, iou):
    """Pairwise concordance of (conf, iou) — the same metric the tool uses."""
    n = len(conf); conc = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if conf[i] != conf[j]:
                tot += 1
                if (conf[i] > conf[j]) == (iou[i] > iou[j]):
                    conc += 1
    return conc / tot if tot else None


# Candidate confidence formulas to compare (all use already-computed signals).
# eq_eff folds unambiguity into edge quality: a sharp peak only counts when it
# is unambiguous (Lowe-style ratio test).
def _cands(r):
    eq, un, ar = r["eq"], r["unambig"], r["area_f"]
    sf = r["shift_f"]
    eq_eff = eq * (0.35 + 0.65 * un)        # unambiguity modulates edge quality
    return {
        "A_current   0.60eq+0.25area+0.15shift":
            0.60 * eq + 0.25 * ar + 0.15 * sf,
        "B_eqeff     0.55eqeff+0.25area+0.20shift":
            0.55 * eq_eff + 0.25 * ar + 0.20 * sf,
        "C_eqeff_shift 0.45eqeff+0.20area+0.35shift":
            0.45 * eq_eff + 0.20 * ar + 0.35 * sf,
        "D_unambig   0.45(eq*un)+0.25area+0.30shift":
            0.45 * (eq * un) + 0.25 * ar + 0.30 * sf,
    }


if __name__ == "__main__":
    from scipy.stats import spearmanr
    targets = sys.argv[1:] or ["Vadnerbhairav", "Malatavadi"]
    all_rows = {v: diagnose(v) for v in targets}

    print(f"\n{'='*78}\nCONFIDENCE FORMULA COMPARISON  (AUC = concordance, Spearman vs IoU)\n{'='*78}")
    names = list(_cands(all_rows[targets[0]][0]).keys())
    for name in names:
        print(f"\n{name}")
        for v in targets:
            rows = all_rows[v]
            confs = [np.clip(_cands(r)[name], 0.05, 0.95) for r in rows]
            ious = [r["iou"] for r in rows]
            auc = auc_concordance(confs, ious)
            sp = spearmanr(confs, ious)[0] if len(set(confs)) > 1 else float("nan")
            auc_s = f"{auc:.3f}" if auc is not None else "—"
            print(f"    {v:>14}: AUC={auc_s}  Spearman={sp:+.3f}")
