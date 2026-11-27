"""
Stage 9: runs the full pre-registered ablation ladder (docs/PREREGISTRATION.md)
-- baselines B0-B4 and ablation rungs A1-A5 -- across all three split
regimes (ml/splits.py), both unweighted and area-weighted, and writes
results/ablation_{run_id}.json. This is the largest new component of the
rebuild: no evaluation code for this pipeline existed anywhere before
Stage 9 (verified by full-history grep).

A1-A5 all use the SAME fixed hyperparameters
(n_estimators=100, max_depth=10, min_samples_leaf=30, class_weight="balanced",
random_state=config.SEED) -- lifted directly from the original (quarantined)
scripts/train_final_model.py:28-31 for continuity, not tuned per rung. Real
hyperparameter selection is Stage 10's job, done via nested CV inside the
training folds only -- doing it here, rung by rung, would let a rung's
config quietly adapt to its own test set, contaminating the ablation's own
comparisons.

SHAP is computed only at A5 (the full-featured rung), per the plan --
running it at every rung would multiply an already-long job for marginal
insight, since earlier rungs are strict feature subsets of A5 anyway.

Run as a module from the repo root: python3 -m ml.build_ablation
"""

import json
import time

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from ml import baselines, config, evaluate, manifest, splits

RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_leaf=30,
                  class_weight="balanced", n_jobs=-1, random_state=config.SEED)

TARGET_COL = f"label_w{config.PRIMARY_LABEL_WINDOW}"

ABLATION_RUNGS = {
    "A1_fwi_codes": ["ffmc", "dmc", "dc", "isi", "bui", "fwi"],
    "A2_plus_trends": ["ffmc", "dmc", "dc", "isi", "bui", "fwi", "dc_trend_7d", "bui_trend_7d"],
    "A3_plus_season": ["ffmc", "dmc", "dc", "isi", "bui", "fwi", "dc_trend_7d", "bui_trend_7d",
                        "day_of_year", "month"],
    "A4_plus_history": ["ffmc", "dmc", "dc", "isi", "bui", "fwi", "dc_trend_7d", "bui_trend_7d",
                         "day_of_year", "month",
                         "hist_fire_count_prior_20y_log1p", "hist_fire_any_prior", "years_since_last_fire"],
    "A5_plus_province": None,  # filled in after province one-hot columns are known
}


def _add_derived_columns(df):
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    return df


def _add_province_dummies(df, all_provinces):
    for p in all_provinces:
        df[f"prov_{p}"] = (df["province"] == p).astype("int8")
    return df


def _fit_rf(train_df, feature_cols):
    X = train_df[feature_cols].values
    y = train_df[TARGET_COL].values
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)
    return model


def _eval_rung(name, score_test, y_test, weights_test, block_ids_test, is_probability):
    unweighted = evaluate.compute_metrics(y_test, score_test, weights=None, is_probability=is_probability)
    weighted = evaluate.compute_metrics(y_test, score_test, weights=weights_test, is_probability=is_probability)

    def _pr_auc(y, s):
        return average_precision_score(y, s)

    point, lo, hi = evaluate.block_bootstrap_ci(y_test, score_test, block_ids_test, _pr_auc)
    return {
        "unweighted": unweighted,
        "area_weighted": weighted,
        "pr_auc_ci95": {"point": point, "lo": lo, "hi": hi},
    }


def run_regime(regime_name, train_df, test_df, all_provinces, run_shap=False):
    print(f"\n{'=' * 60}\n{regime_name}: train={len(train_df)}, test={len(test_df)}\n{'=' * 60}")
    y_test = test_df[TARGET_COL].values
    weights_test = test_df["area_weight"].values
    block_ids_test = test_df["block_id"].values

    results = {}

    for name, fn in baselines.BASELINES.items():
        t0 = time.time()
        score = fn(test_df, train_df)
        results[name] = _eval_rung(name, score, y_test, weights_test, block_ids_test, is_probability=False)
        print(f"  {name}: pr_auc={results[name]['unweighted']['pr_auc']:.4f} "
              f"(CI {results[name]['pr_auc_ci95']['lo']:.4f}-{results[name]['pr_auc_ci95']['hi']:.4f}) "
              f"[{time.time() - t0:.1f}s]")

    ABLATION_RUNGS["A5_plus_province"] = ABLATION_RUNGS["A4_plus_history"] + [f"prov_{p}" for p in all_provinces]

    fitted_a5_model = None
    for rung_name, feature_cols in ABLATION_RUNGS.items():
        t0 = time.time()
        model = _fit_rf(train_df, feature_cols)
        score = model.predict_proba(test_df[feature_cols].values)[:, 1]
        results[rung_name] = _eval_rung(rung_name, score, y_test, weights_test, block_ids_test, is_probability=True)
        print(f"  {rung_name}: pr_auc={results[rung_name]['unweighted']['pr_auc']:.4f} "
              f"(CI {results[rung_name]['pr_auc_ci95']['lo']:.4f}-{results[rung_name]['pr_auc_ci95']['hi']:.4f}) "
              f"[{time.time() - t0:.1f}s]")
        if rung_name == "A5_plus_province":
            fitted_a5_model = (model, feature_cols)

    # Gate condition 1: full model (A5) vs B1 (raw FWI), paired CI on the difference.
    a5_model, a5_features = fitted_a5_model
    a5_score = a5_model.predict_proba(test_df[a5_features].values)[:, 1]
    b1_score = baselines.score_b1(test_df)
    diff_point, diff_lo, diff_hi = evaluate.paired_block_bootstrap_diff(
        y_test, a5_score, b1_score, block_ids_test, average_precision_score
    )
    results["_gate_A5_vs_B1_paired_diff"] = {"point": diff_point, "lo": diff_lo, "hi": diff_hi}
    relative_lift = diff_point / results["B1_raw_fwi"]["unweighted"]["pr_auc"]
    results["_gate_A5_vs_B1_relative_lift"] = relative_lift
    print(f"  A5 vs B1 paired PR-AUC diff: {diff_point:.4f} (CI {diff_lo:.4f}-{diff_hi:.4f}), "
          f"relative lift {relative_lift * 100:.1f}%")

    if run_shap:
        print("  Computing SHAP for A5 (sampled test rows)...")
        sample = test_df.sample(n=min(5000, len(test_df)), random_state=config.SEED)
        explainer = shap.TreeExplainer(a5_model)
        shap_values = explainer.shap_values(sample[a5_features].values)
        # binary classifier -> shap_values is (n, n_features, 2) or a list of two arrays depending on version;
        # normalize to the positive-class contributions.
        sv = shap_values[..., 1] if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 else shap_values[1]
        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_by_feature = dict(zip(a5_features, mean_abs_shap.tolist()))

        fwi_features = set(ABLATION_RUNGS["A2_plus_trends"])
        fwi_shap = sum(v for k, v in shap_by_feature.items() if k in fwi_features)
        total_shap = sum(shap_by_feature.values())
        results["_shap_a5"] = {
            "by_feature": shap_by_feature,
            "fwi_derived_share": fwi_shap / total_shap if total_shap else None,
            "other_share": 1 - (fwi_shap / total_shap) if total_shap else None,
        }
        print(f"  SHAP: FWI-derived features {results['_shap_a5']['fwi_derived_share'] * 100:.1f}% of |SHAP|, "
              f"other {results['_shap_a5']['other_share'] * 100:.1f}%")

    return results


def build():
    run_id = manifest.start_run("stage9_ablation")
    t_start = time.time()

    print("Loading dataset...")
    df = pd.read_parquet(config.DATA_DIR / "dataset_full.parquet")
    df = _add_derived_columns(df)
    all_provinces = sorted(df["province"].unique())
    df = _add_province_dummies(df, all_provinces)
    print(f"{len(df)} rows, {len(all_provinces)} provinces: {all_provinces}")

    block_positive_counts = splits.compute_block_positive_counts(df, TARGET_COL)
    folds = splits.spatial_folds(df["block_id"].unique(), block_positive_counts=block_positive_counts)

    all_results = {}

    train_mask, test_mask = splits.temporal_split(df)
    all_results["temporal"] = run_regime("TEMPORAL", df[train_mask], df[test_mask], all_provinces)

    train_mask, test_mask = splits.spatial_split(df, folds, fold_idx=0)
    all_results["spatial"] = run_regime("SPATIAL (fold 0)", df[train_mask], df[test_mask], all_provinces)

    # Deciding regime -- SHAP computed here only, per docs/PREREGISTRATION.md.
    train_mask, test_mask = splits.spatio_temporal_split(df, folds)
    all_results["spatio_temporal"] = run_regime(
        "SPATIO-TEMPORAL (deciding regime)", df[train_mask], df[test_mask], all_provinces, run_shap=True
    )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"ablation_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path} in {time.time() - t_start:.0f}s total")

    manifest.record(
        run_id, out_path,
        upstream=[config.DATA_DIR / "dataset_full.parquet"],
        extra={"regimes": list(all_results.keys()), "rf_params": RF_PARAMS},
    )
    print(f"Manifest entry recorded: run_id={run_id}")
    return all_results


if __name__ == "__main__":
    build()
