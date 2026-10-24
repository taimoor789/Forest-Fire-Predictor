"""
Daily snapshot of served FWI/ML tiers + a label-free distribution check, for
the live shadow-mode observation period. See docs/PREREGISTRATION.md's
2026-09-14 amendment -- this is the PRIMARY, highest-power half of that
design (n=7,537/day, needs no real fires, available from day one), not the
low-power hotspot-matching half (ml/cwfis_hotspots.py).

Reads fwi_predictions.json / model_info.json / model_components/tiers.json
as already checked out from `deploy` by the calling workflow (read-only --
this script never touches `deploy` itself). Compares each checked feature's
live median against the fixed training bands already committed in
results/serving_safety_check.json (tracked on `main`) -- it CANNOT
recompute those bands itself, since that requires the 516MB gitignored
data/dataset_full.parquet, unavailable in CI.

Run as a module: python3 -m ml.shadow_snapshot --log-dir <path> --deploy-sha-file <path>
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml import shadow_log

# The 6 core FWI-derived features -- the ones results/serving_safety_check.json
# actually has bands for AND that are cheaply available per-cell in
# fwi_predictions.json's fire_weather_indices block. Deliberately excludes
# dc_trend_7d/bui_trend_7d: reconstructing those here would require pulling
# in ml.train's full dependency tree (data/fwi_state.json's `recent` history
# + fire_risk.calculate_bui) for marginal value -- flagged as NOT CHECKED
# below, same pattern ml/serving_safety_check.py already uses for its own
# known gaps (hist_fire_count_prior_20y_log1p, years_since_last_fire).
T1_FEATURES = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
NOT_CHECKED_FEATURES = ["dc_trend_7d", "bui_trend_7d"]

PERCENTILES = [5, 25, 50, 75, 95]


def load_predictions_df() -> pd.DataFrame:
    with open("fwi_predictions.json") as f:
        preds = json.load(f)
    rows = []
    for r in preds["data"]:
        fwi_codes = r.get("fire_weather_indices", {})
        rows.append({
            "cell_id": f"{r['lat']:.4f}_{r['lon']:.4f}",
            "lat": r["lat"], "lon": r["lon"],
            "ffmc": fwi_codes.get("ffmc"), "dmc": fwi_codes.get("dmc"), "dc": fwi_codes.get("dc"),
            "isi": fwi_codes.get("isi"), "bui": fwi_codes.get("bui"), "fwi": r.get("fwi"),
            "danger_class": r.get("danger_class"),
            "ml_risk_probability": r.get("ml_risk_probability"),
            "ml_danger_class": r.get("ml_danger_class"),
        })
    return pd.DataFrame(rows), preds.get("timestamp")


def check_timestamp_is_today(timestamp_str: str):
    if not timestamp_str:
        raise ValueError("fwi_predictions.json has no timestamp field -- cannot verify freshness")
    ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    age_h = (datetime.now(timezone.utc) - ts.replace(tzinfo=ts.tzinfo or timezone.utc)).total_seconds() / 3600
    if age_h > 30:  # matches main.py's relaxed 26h/50h health thresholds, with headroom
        raise ValueError(
            f"fwi_predictions.json timestamp is {age_h:.1f}h old -- this looks like a stale read from "
            f"`deploy`, not a fresh pipeline run. Refusing to snapshot it under today's date."
        )
    return age_h


def check_grid_consistency(log_dir, df: pd.DataFrame):
    cached = shadow_log.load_grid_cells(log_dir)
    if cached is None:
        print("  grid_cells.csv missing on shadow-eval-log -- deriving fresh (should not normally happen; "
              "it's meant to be bootstrapped once, see the session record)")
        shadow_log.save_grid_cells(log_dir, df[["cell_id", "lat", "lon"]])
        return
    today_ids = set(df["cell_id"])
    cached_ids = set(cached["cell_id"])
    if today_ids != cached_ids:
        added = today_ids - cached_ids
        removed = cached_ids - today_ids
        print(f"  WARNING: cell set changed since grid_cells.csv was cached -- "
              f"{len(added)} added, {len(removed)} removed. This is itself a real signal "
              f"(e.g. canada_fire_grid.csv was regenerated), not necessarily an error.")


def compute_t1(df: pd.DataFrame) -> dict:
    with open("results/serving_safety_check.json") as f:
        safety = json.load(f)
    checked = safety["checked_features"]

    results = {}
    for feat in T1_FEATURES:
        if feat not in checked:
            results[feat] = {"status": "no_reference_band"}
            continue
        band = checked[feat]
        live_median = float(df[feat].median())
        in_band = band["train_p5"] <= live_median <= band["train_p95"]
        results[feat] = {
            "train_p5": band["train_p5"], "train_p95": band["train_p95"],
            "live_median": live_median, "pass": in_band,
        }
    results["_not_checked"] = NOT_CHECKED_FEATURES
    results["_all_pass"] = all(v.get("pass", True) for v in results.values() if isinstance(v, dict))
    return results


def build_meta(df: pd.DataFrame, deploy_sha: str, timestamp_age_h: float) -> dict:
    percentiles = {
        feat: {f"p{p}": float(np.percentile(df[feat].dropna(), p)) for p in PERCENTILES}
        for feat in T1_FEATURES + ["ml_risk_probability"]
    }
    danger_class_counts = df["danger_class"].value_counts().to_dict()
    ml_danger_class_counts = df["ml_danger_class"].value_counts(dropna=True).to_dict()

    valid_ml = df["ml_risk_probability"].notna()
    if valid_ml.sum() >= 10:
        rho, _ = spearmanr(df.loc[valid_ml, "fwi"], df.loc[valid_ml, "ml_risk_probability"])
        spearman_fwi_vs_ml = float(rho)
    else:
        spearman_fwi_vs_ml = None

    return {
        "snapshot_date": shadow_log.today_str(),
        "n_cells": len(df),
        "n_ml_scored": int(valid_ml.sum()),
        "predictions_timestamp_age_hours": round(timestamp_age_h, 2),
        "deploy_sha": deploy_sha,
        "percentiles": percentiles,
        "danger_class_counts": danger_class_counts,
        "ml_danger_class_counts": ml_danger_class_counts,
        "spearman_fwi_vs_ml_risk_probability": spearman_fwi_vs_ml,
        "t1_serving_safety_check": compute_t1(df),
    }


def run(log_dir: str, deploy_sha_file: str):
    print("=== Shadow-eval daily snapshot ===")
    df, timestamp_str = load_predictions_df()
    print(f"  {len(df)} cells loaded from fwi_predictions.json")

    age_h = check_timestamp_is_today(timestamp_str)
    print(f"  predictions timestamp age: {age_h:.2f}h -- OK")

    check_grid_consistency(log_dir, df)

    with open(deploy_sha_file) as f:
        deploy_sha = f.read().strip()

    meta = build_meta(df, deploy_sha, age_h)
    print(f"  T1 (serving-safety) all-pass today: {meta['t1_serving_safety_check']['_all_pass']}")
    print(f"  danger_class: {meta['danger_class_counts']}")
    print(f"  ml_danger_class: {meta['ml_danger_class_counts']}")
    print(f"  Spearman(fwi, ml_risk_probability): {meta['spearman_fwi_vs_ml_risk_probability']}")

    date_str = shadow_log.today_str()
    pred_dir = shadow_log.predictions_dir(log_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    out_df = df[["cell_id", "lat", "lon", "fwi", "danger_class", "ml_risk_probability", "ml_danger_class"]].copy()
    out_df["fwi"] = out_df["fwi"].round(4)
    out_df["ml_risk_probability"] = out_df["ml_risk_probability"].round(8)
    csv_path = pred_dir / f"{date_str}.csv"
    meta_path = pred_dir / f"{date_str}.meta.json"
    out_df.to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {csv_path}, {meta_path}")

    shadow_log.commit_and_push(log_dir, [csv_path, meta_path], f"predictions: snapshot {date_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--deploy-sha-file", required=True)
    args = parser.parse_args()
    run(args.log_dir, args.deploy_sha_file)
