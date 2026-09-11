"""
Stage 10: final model selection, calibration, and out-of-sample tier
derivation. Structural fix for the original tier/model mismatch (verified:
ml_tier_thresholds.json was ~10 hours older than the model it was supposedly
calibrated for -- calibrated against a different fit than what shipped):
fit on train only -> calibrate on a DISJOINT calibration split -> derive
tier bounds on that SAME calibration split -> report tier fire rates on the
untouched test split -> one atomic write of model + features + tiers +
metrics, one manifest run_id, a load-time hash assertion downstream can
check before trusting the pair.

Model selection is nested: RandomForest (the original, unmodified
hyperparameters from the now-quarantined scripts/train_final_model.py --
kept as a fixed reference point, not retuned, since there's no reproducible
prior baseline to prefer over it) plus a small pre-registered LightGBM grid,
compared via inner cross-validation using whichever spatial folds remain
after folds 0 and 1 (Stage 8's test and calibration sets, never touched
here). The
selection rule (highest mean inner-CV PR-AUC) and the candidate grid are
both fixed before running, not chosen after seeing results.

Uses the exact same A5 feature set Stage 9 validated (docs/PREREGISTRATION.md)
-- feature-derivation logic is intentionally duplicated from
ml/build_ablation.py rather than imported, so touching this file can't
retroactively change what Stage 9's already-committed ablation numbers mean.

Run as a module from the repo root: python3 -m ml.train
"""

import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from ml import config, manifest, splits

TARGET_COL = f"label_w{config.PRIMARY_LABEL_WINDOW}"

RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_leaf=30,
                  class_weight="balanced", n_jobs=-1, random_state=config.SEED)

# Small, pre-registered grid -- fixed before running, not chosen after
# seeing results. LightGBM is included because, per docs/PREREGISTRATION.md
# and the rebuild plan, "no reproducible XGBoost baseline exists to prefer
# over RF" was true only because nothing was ever properly evaluated;
# now that a real harness exists (Stage 9), it's worth a genuine comparison.
LGBM_CANDIDATES = {
    "lgbm_a": dict(num_leaves=31, learning_rate=0.05, n_estimators=200, is_unbalance=True,
                    random_state=config.SEED, n_jobs=-1, verbosity=-1),
    "lgbm_b": dict(num_leaves=63, learning_rate=0.05, n_estimators=300, is_unbalance=True,
                    random_state=config.SEED, n_jobs=-1, verbosity=-1),
    "lgbm_c": dict(num_leaves=31, learning_rate=0.1, n_estimators=150, is_unbalance=True,
                    random_state=config.SEED, n_jobs=-1, verbosity=-1),
}

N_TIERS = 6  # matches get_danger_class's 6 named classes (Very Low..Extreme)
TIER_NAMES = ["Very Low", "Low", "Moderate", "High", "Very High", "Extreme"]


def _add_derived_columns(df):
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    return df


def _add_province_dummies(df, all_provinces):
    for p in all_provinces:
        df[f"prov_{p}"] = (df["province"] == p).astype("int8")
    return df


def _full_feature_cols(all_provinces):
    return (["ffmc", "dmc", "dc", "isi", "bui", "fwi", "dc_trend_7d", "bui_trend_7d",
              "day_of_year", "month",
              "hist_fire_count_prior_20y_log1p", "hist_fire_any_prior", "years_since_last_fire"]
             + [f"prov_{p}" for p in all_provinces])


def _fit_candidate(name, params, X_train, y_train):
    if name.startswith("lgbm"):
        model = lgb.LGBMClassifier(**params)
    else:
        model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def inner_cv_select(df, fit_train_mask, folds, feature_cols):
    """Inner spatial CV over whatever folds remain after folds 0 (test) and
    1 (calibration) -- entirely within fit_train, never touching those two.
    Not hardcoded to 3: ml.splits.spatial_folds can return fewer than
    N_SPATIAL_FOLDS if a degenerate (near-zero-positive) fold got merged
    into a healthy one during repair -- see splits.py's docstring for the
    Arctic-tundra fold this guards against."""
    candidates = {"rf_original": RF_PARAMS, **LGBM_CANDIDATES}
    inner_pool_folds = folds[2:]
    n_inner = len(inner_pool_folds)
    assert n_inner >= 2, f"only {n_inner} fold(s) left for inner CV after reserving test+calibration -- too few"

    fit_train_df = df[fit_train_mask]
    scores = {name: [] for name in candidates}

    for val_i in range(n_inner):
        val_blocks = inner_pool_folds[val_i]
        train_blocks = set().union(*(inner_pool_folds[j] for j in range(n_inner) if j != val_i))

        inner_train = fit_train_df[fit_train_df["block_id"].isin(train_blocks)]
        inner_val = fit_train_df[fit_train_df["block_id"].isin(val_blocks)]
        if inner_val[TARGET_COL].sum() == 0 or inner_train[TARGET_COL].sum() == 0:
            print(f"  inner fold {val_i}: skipped, one side has zero positives")
            continue

        X_train = inner_train[feature_cols].values
        y_train = inner_train[TARGET_COL].values
        X_val = inner_val[feature_cols].values
        y_val = inner_val[TARGET_COL].values

        for name, params in candidates.items():
            t0 = time.time()
            model = _fit_candidate(name, params, X_train, y_train)
            score = model.predict_proba(X_val)[:, 1]
            pr_auc = average_precision_score(y_val, score)
            scores[name].append(pr_auc)
            print(f"  inner fold {val_i}, {name}: pr_auc={pr_auc:.4f} [{time.time() - t0:.1f}s]")

    mean_scores = {name: float(np.mean(v)) for name, v in scores.items() if v}
    print("\nInner-CV mean PR-AUC by candidate:")
    for name, s in sorted(mean_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {s:.4f}")

    best_name = max(mean_scores, key=mean_scores.get)
    return best_name, candidates[best_name], mean_scores


def derive_tiers(calib_probs, calib_labels, n_tiers=N_TIERS):
    """Equal-frequency (quantile) tier boundaries on the calibration
    split's calibrated probabilities -- matches production's 6-named-class
    UX. Returns (bounds, fire_rates_on_calibration_split)."""
    quantiles = np.linspace(0, 1, n_tiers + 1)
    bounds = np.quantile(calib_probs, quantiles)
    bounds[0], bounds[-1] = 0.0, 1.0 + 1e-9  # cover the full [0,1] range, half-open bins
    bounds = np.unique(bounds)  # duplicate quantiles (e.g. many rows at prob=0) collapse bins, not fail

    fire_rates = []
    for i in range(len(bounds) - 1):
        mask = (calib_probs >= bounds[i]) & (calib_probs < bounds[i + 1])
        fire_rates.append(float(calib_labels[mask].mean()) if mask.sum() else float("nan"))
    return bounds.tolist(), fire_rates


def tier_fire_rates_on_test(test_probs, test_labels, bounds):
    fire_rates = []
    for i in range(len(bounds) - 1):
        mask = (test_probs >= bounds[i]) & (test_probs < bounds[i + 1])
        fire_rates.append({"n": int(mask.sum()), "fire_rate": float(test_labels[mask].mean()) if mask.sum() else None})
    return fire_rates


def build():
    run_id = manifest.start_run("stage10_train")
    t_start = time.time()

    print("Loading dataset...")
    df = pd.read_parquet(config.DATA_DIR / "dataset_full.parquet")
    df = _add_derived_columns(df)
    all_provinces = sorted(df["province"].unique())
    df = _add_province_dummies(df, all_provinces)
    feature_cols = _full_feature_cols(all_provinces)
    print(f"{len(df)} rows, {len(feature_cols)} features")

    block_positive_counts = splits.compute_block_positive_counts(df, TARGET_COL)
    folds = splits.spatial_folds(df["block_id"].unique(), block_positive_counts=block_positive_counts)
    fit_train_mask, calib_mask, test_mask = splits.calibration_split(df, folds)
    print(f"fit_train={fit_train_mask.sum()}, calib={calib_mask.sum()}, test={test_mask.sum()}")

    print("\n=== Inner CV model selection (folds 2/3/4 only) ===")
    best_name, best_params, inner_scores = inner_cv_select(df, fit_train_mask, folds, feature_cols)
    print(f"\nSelected: {best_name}")

    print(f"\n=== Fitting {best_name} on full fit_train ===")
    fit_train_df = df[fit_train_mask]
    t0 = time.time()
    final_model = _fit_candidate(best_name, best_params, fit_train_df[feature_cols].values, fit_train_df[TARGET_COL].values)
    print(f"  fit time: {time.time() - t0:.1f}s")

    print("\n=== Calibrating on the disjoint calibration split ===")
    calib_df = df[calib_mask]
    calib_raw_scores = final_model.predict_proba(calib_df[feature_cols].values)[:, 1]
    calib_labels = calib_df[TARGET_COL].values

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(calib_raw_scores, calib_labels)
    calib_calibrated = calibrator.predict(calib_raw_scores)

    from ml.evaluate import expected_calibration_error
    ece_raw = expected_calibration_error(calib_labels, calib_raw_scores)
    ece_calibrated = expected_calibration_error(calib_labels, calib_calibrated)
    print(f"  ECE on calibration split: raw={ece_raw:.4f} -> calibrated={ece_calibrated:.4f}")

    print("\n=== Deriving tier bounds on the calibration split ===")
    tier_bounds, calib_tier_rates = derive_tiers(calib_calibrated, calib_labels)
    n_tiers_actual = len(tier_bounds) - 1
    tier_names = TIER_NAMES[:n_tiers_actual] if n_tiers_actual <= N_TIERS else \
        [f"Tier{i}" for i in range(n_tiers_actual)]
    print(f"  {n_tiers_actual} tiers, calibration-split fire rates: "
          f"{[f'{r:.4f}' if r == r else 'nan' for r in calib_tier_rates]}")

    print("\n=== Evaluating on the untouched test split ===")
    test_df = df[test_mask]
    test_raw_scores = final_model.predict_proba(test_df[feature_cols].values)[:, 1]
    test_calibrated = calibrator.predict(test_raw_scores)
    test_labels = test_df[TARGET_COL].values

    test_ece = expected_calibration_error(test_labels, test_calibrated)
    test_pr_auc = average_precision_score(test_labels, test_calibrated)
    print(f"  test PR-AUC (calibrated score, same ranking as raw): {test_pr_auc:.4f}")
    print(f"  test ECE (calibrated): {test_ece:.4f} (gate: <= {config.GATE_MAX_CALIBRATION_ECE})")

    test_tier_rates = tier_fire_rates_on_test(test_calibrated, test_labels, tier_bounds)
    rates_only = [r["fire_rate"] for r in test_tier_rates if r["fire_rate"] is not None]
    is_monotone = all(rates_only[i] <= rates_only[i + 1] for i in range(len(rates_only) - 1))
    rate_strs = [f"{r['fire_rate']:.4f}" if r["fire_rate"] is not None else "nan (empty tier)" for r in test_tier_rates]
    print(f"  test-split tier fire rates: {rate_strs}")
    print(f"  strictly monotone out-of-sample: {is_monotone}")

    config.MODEL_COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_COMPONENTS_DIR / "model.pkl"
    calibrator_path = config.MODEL_COMPONENTS_DIR / "calibrator.pkl"
    features_path = config.MODEL_COMPONENTS_DIR / "features.json"
    tiers_path = config.MODEL_COMPONENTS_DIR / "tiers.json"

    joblib.dump(final_model, model_path)
    joblib.dump(calibrator, calibrator_path)
    with open(features_path, "w") as f:
        json.dump({"feature_cols": feature_cols, "provinces": all_provinces, "model_type": best_name}, f, indent=2)
    with open(tiers_path, "w") as f:
        json.dump({
            "tier_names": tier_names,
            "tier_bounds": tier_bounds,
            "calibration_split_fire_rates": calib_tier_rates,
            "test_split_fire_rates": test_tier_rates,
            "monotone_out_of_sample": is_monotone,
        }, f, indent=2)

    for p, upstream in [(model_path, []), (calibrator_path, [model_path]),
                          (features_path, []), (tiers_path, [calibrator_path])]:
        manifest.record(run_id, p, upstream=upstream + [config.DATA_DIR / "dataset_full.parquet"],
                          extra={"model_type": best_name})

    summary = {
        "run_id": run_id,
        "selected_model": best_name,
        "inner_cv_scores": inner_scores,
        "test_pr_auc_calibrated": test_pr_auc,
        "test_ece_calibrated": test_ece,
        "gate_ece_pass": test_ece <= config.GATE_MAX_CALIBRATION_ECE,
        "monotone_out_of_sample": is_monotone,
        "tier_names": tier_names,
        "tier_bounds": tier_bounds,
        "test_tier_rates": test_tier_rates,
    }
    summary_path = config.RESULTS_DIR / f"train_summary_{run_id}.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {model_path}, {calibrator_path}, {features_path}, {tiers_path}, {summary_path}")
    print(f"Total time: {time.time() - t_start:.0f}s")
    return summary


if __name__ == "__main__":
    build()
