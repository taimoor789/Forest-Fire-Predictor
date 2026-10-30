"""
Stage 11: the actual ship/no-ship decision against docs/PREREGISTRATION.md's
gate, evaluated on the EXACT artifact that would ship (model_components/
model.pkl + calibrator.pkl from Stage 10), not transferred from Stage 9's
ablation rung (which was trained on a slightly larger set -- the ablation's
A5 included what's now the calibration fold, since Stage 10 carves that out
before fitting). Every gate number here is computed fresh against the
shipped artifact's own test-split predictions.

Run as a module from the repo root: python3 -m ml.stage11_gate
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml import baselines, config, evaluate, splits, train

TARGET_COL = train.TARGET_COL


def run():
    print("Loading dataset and reconstructing Stage 10's exact test split...")
    df = pd.read_parquet(config.DATA_DIR / "dataset_full.parquet")
    df = train._add_derived_columns(df)
    all_provinces = sorted(df["province"].unique())
    df = train._add_province_dummies(df, all_provinces)

    block_positive_counts = splits.compute_block_positive_counts(df, TARGET_COL)
    folds = splits.spatial_folds(df["block_id"].unique(), block_positive_counts=block_positive_counts)
    fit_train_mask, calib_mask, test_mask = splits.calibration_split(df, folds)

    with open(config.MODEL_COMPONENTS_DIR / "features.json") as f:
        feat_schema = json.load(f)
    feature_cols = feat_schema["feature_cols"]
    assert feat_schema["provinces"] == all_provinces, "province set drifted since Stage 10 -- do not proceed"

    model = joblib.load(config.MODEL_COMPONENTS_DIR / "model.pkl")
    calibrator = joblib.load(config.MODEL_COMPONENTS_DIR / "calibrator.pkl")

    test_df = df[test_mask]
    y_test = test_df[TARGET_COL].values
    block_ids_test = test_df["block_id"].values

    raw_score = model.predict_proba(test_df[feature_cols].values)[:, 1]
    full_model_score = calibrator.predict(raw_score)  # isotonic -- monotonic, ranking unaffected by calibration

    b1_score = baselines.score_b1(test_df)
    b2_score = baselines.score_b2(test_df)

    full_metrics = evaluate.compute_metrics(y_test, full_model_score, is_probability=True)
    b1_metrics = evaluate.compute_metrics(y_test, b1_score)
    b2_metrics = evaluate.compute_metrics(y_test, b2_score)

    print(f"\nFull model:  PR-AUC={full_metrics['pr_auc']:.4f}  precision@top5%={full_metrics['precision_top5pct']:.4f}  ECE={full_metrics['ece']:.4f}")
    print(f"B1 raw FWI:  PR-AUC={b1_metrics['pr_auc']:.4f}  precision@top5%={b1_metrics['precision_top5pct']:.4f}")
    print(f"B2 tiers:    PR-AUC={b2_metrics['pr_auc']:.4f}  precision@top5%={b2_metrics['precision_top5pct']:.4f}")

    # ---- Condition 1: beats B1 by >=25% relative PR-AUC, paired CI lower bound > 0 ----
    diff_point, diff_lo, diff_hi = evaluate.paired_block_bootstrap_diff(
        y_test, full_model_score, b1_score, block_ids_test, average_precision_score
    )
    relative_lift_vs_b1 = diff_point / b1_metrics["pr_auc"]
    cond1_pass = (relative_lift_vs_b1 >= config.GATE_MIN_RELATIVE_PR_AUC_LIFT_VS_RAW_FWI) and (diff_lo > 0)
    print(f"\nCondition 1 (beats B1 by >=25%, CI lower bound > 0): "
          f"diff={diff_point:.4f} (CI {diff_lo:.4f}-{diff_hi:.4f}), relative_lift={relative_lift_vs_b1 * 100:.1f}% "
          f"-> {'PASS' if cond1_pass else 'FAIL'}")

    # ---- Condition 2: beats B2 on PR-AUC AND precision@top5% ----
    cond2_pass = (full_metrics["pr_auc"] > b2_metrics["pr_auc"]) and \
                 (full_metrics["precision_top5pct"] > b2_metrics["precision_top5pct"])
    print(f"Condition 2 (beats B2 on both PR-AUC and precision@top5%): -> {'PASS' if cond2_pass else 'FAIL'}")

    # ---- Condition 3: ECE <= 0.02, monotone tiers (already established in Stage 10) ----
    with open(config.RESULTS_DIR / "train_summary_stage10_train_20260911T073316Z_9454c374.json") as f:
        train_summary = json.load(f)
    cond3_pass = train_summary["gate_ece_pass"] and train_summary["monotone_out_of_sample"]
    print(f"Condition 3 (ECE <= {config.GATE_MAX_CALIBRATION_ECE}, monotone tiers): "
          f"ECE={train_summary['test_ece_calibrated']:.5f}, monotone={train_summary['monotone_out_of_sample']} "
          f"-> {'PASS' if cond3_pass else 'FAIL'}")

    # ---- Condition 4: lift not attributable solely to A1-A2 (from Stage 9's ablation) ----
    with open("results/ablation_stage9_ablation_20260911T025429Z_e6826fd5.json") as f:
        ablation = json.load(f)["spatio_temporal"]
    a1_pr_auc = ablation["A1_fwi_codes"]["unweighted"]["pr_auc"]
    a2_pr_auc = ablation["A2_plus_trends"]["unweighted"]["pr_auc"]
    a5_pr_auc = ablation["A5_plus_province"]["unweighted"]["pr_auc"]
    b1_pr_auc_stage9 = ablation["B1_raw_fwi"]["unweighted"]["pr_auc"]
    a1_a2_share_of_lift = (a2_pr_auc - b1_pr_auc_stage9) / (a5_pr_auc - b1_pr_auc_stage9)
    cond4_pass = a1_a2_share_of_lift < 0.5  # A1-A2 must not be the majority of the total lift
    print(f"Condition 4 (lift not primarily A1-A2 nonlinear-FWI-reshaping, from Stage 9's ablation): "
          f"A1-A2 share of total A5-vs-B1 lift = {a1_a2_share_of_lift * 100:.1f}% -> {'PASS' if cond4_pass else 'FAIL'}")

    # ---- Serving safety gate (Stage 11 prerequisite, already run) ----
    with open(config.RESULTS_DIR / "serving_safety_check.json") as f:
        safety = json.load(f)
    safety_pass = safety["gate_pass"]
    print(f"Serving-safety distribution gate: -> {'PASS' if safety_pass else 'FAIL'}")

    all_pass = cond1_pass and cond2_pass and cond3_pass and cond4_pass and safety_pass
    print(f"\n{'=' * 50}")
    print(f"STAGE 11 DECISION: {'SHIP' if all_pass else 'DO NOT SHIP'}")
    print(f"{'=' * 50}")

    decision = {
        "condition_1_beats_b1": {
            "pass": cond1_pass, "diff_point": diff_point, "diff_ci_lo": diff_lo, "diff_ci_hi": diff_hi,
            "relative_lift_pct": relative_lift_vs_b1 * 100,
        },
        "condition_2_beats_b2": {
            "pass": cond2_pass,
            "full_pr_auc": full_metrics["pr_auc"], "b2_pr_auc": b2_metrics["pr_auc"],
            "full_precision_top5pct": full_metrics["precision_top5pct"], "b2_precision_top5pct": b2_metrics["precision_top5pct"],
        },
        "condition_3_calibrated_monotone": {
            "pass": cond3_pass, "ece": train_summary["test_ece_calibrated"],
            "monotone": train_summary["monotone_out_of_sample"],
        },
        "condition_4_not_just_fwi_reshaping": {
            "pass": cond4_pass, "a1_a2_share_of_lift_pct": a1_a2_share_of_lift * 100,
        },
        "serving_safety_gate": {"pass": safety_pass},
        "final_decision": "SHIP" if all_pass else "DO_NOT_SHIP",
    }
    out_path = config.RESULTS_DIR / "stage11_gate_decision.json"
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"\nWrote {out_path}")
    return decision


if __name__ == "__main__":
    run()
