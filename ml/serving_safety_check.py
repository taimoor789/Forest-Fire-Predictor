"""
Mandatory, accuracy-independent serving-safety gate (docs/PREREGISTRATION.md):
every feature's live production median must fall inside the training set's
5th-95th percentile range. This is the direct, mechanical fix for the
original bug: the quarantined model's training DMC median was 500.0 (the
saturation ceiling, 54.2% of rows pinned there) against live production's
actual mean of 9.14 -- no accuracy metric would ever have caught that; only
comparing distributions does. A model that fails this does not ship,
regardless of test scores.

Live values come from fwi_predictions.json (today's served FWI codes) and
data/fwi_state.json (per-cell recent dmc/dc history, needed to reconstruct
dc_trend_7d/bui_trend_7d the same way advance_one_day() does -- never
recomputed via a different method than production's own).

Known, honest gap: hist_fire_count_prior_20y_log1p and years_since_last_fire
have NO live equivalent right now -- data/canada_fire_grid.csv only carries
the old quarantined binary historical_fire column (see docs/DATA_PROVENANCE.md),
not the leak-free count/recency features Stage 2 built. historical_fire_zone
in fwi_predictions.json is checked as an approximate proxy for
hist_fire_any_prior only, explicitly flagged as approximate. Stage 12 would
need to regenerate canada_fire_grid.csv with the real features before those
two can be checked for real -- reported as NOT YET CHECKABLE here, not
silently skipped or faked.

Run as a module from the repo root: python3 -m ml.serving_safety_check
"""

import json

import numpy as np
import pandas as pd

from fire_risk import CanadianFireWeatherIndex, TREND_HISTORY_DAYS
from ml import config, train

_fwi_calc = CanadianFireWeatherIndex()

# Checked against training's 5th-95th percentile range.
CHECKED_FEATURES = ["ffmc", "dmc", "dc", "isi", "bui", "fwi", "dc_trend_7d", "bui_trend_7d"]
APPROXIMATE_FEATURES = ["hist_fire_any_prior"]  # live proxy is a different (older, leaky) definition
NOT_CHECKABLE_FEATURES = ["hist_fire_count_prior_20y_log1p", "years_since_last_fire"]


def load_live_features():
    with open("fwi_predictions.json") as f:
        preds = json.load(f)["data"]
    with open(config.DATA_DIR / "fwi_state.json") as f:
        state = json.load(f)

    rows = []
    for entry in preds:
        lat, lon = entry["lat"], entry["lon"]
        key = f"{lat:.4f}_{lon:.4f}"
        fwi_codes = entry["fire_weather_indices"]

        dc_trend_7d, bui_trend_7d = None, None
        cell_state = state.get(key)
        if cell_state is not None:
            recent = cell_state.get("recent", [])
            if len(recent) >= TREND_HISTORY_DAYS:
                week_ago = recent[-TREND_HISTORY_DAYS]
                dc_trend_7d = fwi_codes["dc"] - week_ago["dc"]
                bui_week_ago = _fwi_calc.calculate_bui(week_ago["dmc"], week_ago["dc"])
                bui_trend_7d = fwi_codes["bui"] - bui_week_ago

        rows.append({
            "lat": lat, "lon": lon,
            "ffmc": fwi_codes["ffmc"], "dmc": fwi_codes["dmc"], "dc": fwi_codes["dc"],
            "isi": fwi_codes["isi"], "bui": fwi_codes["bui"], "fwi": fwi_codes["fwi"],
            "dc_trend_7d": dc_trend_7d, "bui_trend_7d": bui_trend_7d,
            "hist_fire_any_prior": entry.get("historical_fire_zone"),
        })
    return pd.DataFrame(rows)


def run():
    print("Loading live production data (fwi_predictions.json + data/fwi_state.json)...")
    live = load_live_features()
    print(f"{len(live)} live cells loaded")

    print("Loading training data (fit_train from Stage 10)...")
    df = pd.read_parquet(config.DATA_DIR / "dataset_full.parquet")
    df = train._add_derived_columns(df)
    all_provinces = sorted(df["province"].unique())
    df = train._add_province_dummies(df, all_provinces)

    from ml import splits  # recompute folds exactly as Stage 10 did
    block_positive_counts = splits.compute_block_positive_counts(df, train.TARGET_COL)
    folds = splits.spatial_folds(df["block_id"].unique(), block_positive_counts=block_positive_counts)
    fit_train_mask, _, _ = splits.calibration_split(df, folds)
    fit_train = df[fit_train_mask]

    print(f"\n{'Feature':<30} {'Train 5th-95th %ile':<28} {'Live median':<15} {'Status'}")
    print("-" * 90)

    results = {}
    all_pass = True

    for feat in CHECKED_FEATURES + APPROXIMATE_FEATURES:
        train_vals = fit_train[feat].dropna().values
        live_vals = live[feat].dropna().values
        if len(live_vals) == 0:
            print(f"{feat:<30} {'(no live data)':<28}")
            continue

        p5, p95 = np.percentile(train_vals, [5, 95])
        live_median = np.median(live_vals)
        in_range = p5 <= live_median <= p95
        status = "PASS" if in_range else "FAIL"
        approx_tag = " (approximate proxy)" if feat in APPROXIMATE_FEATURES else ""
        print(f"{feat:<30} [{p5:.3f}, {p95:.3f}]{'':<8} {live_median:<15.3f} {status}{approx_tag}")

        results[feat] = {
            "train_p5": float(p5), "train_p95": float(p95),
            "live_median": float(live_median), "pass": bool(in_range),
            "approximate": feat in APPROXIMATE_FEATURES,
        }
        if feat in CHECKED_FEATURES and not in_range:
            all_pass = False

    print(f"\nNOT YET CHECKABLE (no live equivalent exists until Stage 12 regenerates "
          f"canada_fire_grid.csv with leak-free historical_fire features):")
    for feat in NOT_CHECKABLE_FEATURES:
        print(f"  {feat}")

    print(f"\n{'=' * 40}")
    print(f"Serving-safety gate (checked features only): {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 40}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "serving_safety_check.json"
    with open(out_path, "w") as f:
        json.dump({
            "checked_features": results,
            "not_yet_checkable": NOT_CHECKABLE_FEATURES,
            "gate_pass": all_pass,
        }, f, indent=2)
    print(f"\nWrote {out_path}")
    return all_pass, results


if __name__ == "__main__":
    run()
