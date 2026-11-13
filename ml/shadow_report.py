"""
Manually-run report for the live shadow-mode observation period. See
docs/PREREGISTRATION.md's 2026-09-14 amendment -- this evaluates the
pre-registered tripwire conditions (T1-T5), not a rigorous accuracy
comparison. At the expected sample size (~35 Canadian hotspots/day
clustering into an estimated 20-40 distinct fire complexes/week), this can
detect gross breakage and cannot detect a difference like "55% vs 45%" --
margins are roughly +/-18 percentage points. Framed and reported as a
tripwire throughout, per the pre-registration text.

Deliberately does NOT use ml.evaluate.block_bootstrap_ci /
paired_block_bootstrap_diff (would silently drop zero-positive resamples
at this sample size, producing a confidently-wrong tight CI) or
expected_calibration_error/Brier (wrong modality -- hotspot detection is
not the label the model was calibrated against). Both exclusions are
pre-registered, not incidental.

Run as a module: python3 -m ml.shadow_report --log-dir <path>
"""

import argparse
import json
import math
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

from ml import config, shadow_log
from ml.evaluate import precision_recall_at_k

N_MIN = 50
FORWARD_WINDOWS = {"W0": 0, "W1": 1, "W2": 2}  # union window [0, W] days forward; W1 is primary
PRIMARY_WINDOW = "W1"
TOP_K_FRACTIONS = sorted(set(config.TOP_K_FRACTIONS) | {0.10})


def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - half) / denom, (centre + half) / denom)


def load_evaluable_data(log_dir):
    predictions = shadow_log.read_all_predictions(log_dir)
    hotspots = shadow_log.read_all_hotspots(log_dir)
    hotspot_status = shadow_log.read_all_hotspot_statuses(log_dir)
    meta_records = shadow_log.read_all_prediction_meta(log_dir)

    if predictions.empty:
        raise SystemExit("No predictions snapshots found yet -- nothing to report.")

    predictions["snapshot_date"] = pd.to_datetime(predictions["snapshot_date"]).dt.date
    all_dates = sorted(predictions["snapshot_date"].unique())

    # A calendar day is "covered" if >=1 hotspot pull that day had status=ok.
    # Failed/suspect days are excluded from the denominator entirely, never
    # scored as zero positives (docs/PREREGISTRATION.md's exclusion rule).
    covered_days = set()
    if not hotspot_status.empty:
        hotspot_status["fetched_at"] = pd.to_datetime(hotspot_status["fetched_at"])
        hotspot_status["fetch_date"] = hotspot_status["fetched_at"].dt.date
        covered_days = set(hotspot_status.loc[hotspot_status["status"] == "ok", "fetch_date"])

    if not hotspots.empty:
        hotspots["detection_date"] = pd.to_datetime(hotspots["rep_date"]).dt.date

    # Trailing PRIMARY window's worth of prediction-days are censored --
    # not enough time has passed to know if a fire followed them yet.
    max_w = max(FORWARD_WINDOWS.values())
    last_evaluable_date = max(all_dates) - timedelta(days=max_w) if all_dates else None
    evaluable_dates = [d for d in all_dates if d <= last_evaluable_date and d in covered_days]
    excluded_uncovered = [d for d in all_dates if d <= last_evaluable_date and d not in covered_days]
    excluded_censored = [d for d in all_dates if d > last_evaluable_date]

    print(f"Prediction snapshot dates: {all_dates}")
    print(f"Hotspot-covered dates: {sorted(covered_days)}")
    print(f"Evaluable dates (covered, not censored): {evaluable_dates}")
    if excluded_uncovered:
        print(f"Excluded (no covered hotspot pull that day): {excluded_uncovered}")
    if excluded_censored:
        print(f"Excluded (censored -- too recent, forward window incomplete): {excluded_censored}")

    return predictions, hotspots, evaluable_dates, meta_records


def build_labels(predictions, hotspots, evaluable_dates, window_days):
    """For each evaluable (cell_id, date), 1 if a hotspot was attributed to
    that cell with detection_date in [date, date+window_days]."""
    if hotspots.empty:
        labels = predictions[predictions["snapshot_date"].isin(evaluable_dates)].copy()
        labels["positive"] = 0
        return labels

    hotspot_dates_by_cell = hotspots.groupby("cell_id")["detection_date"].apply(set).to_dict()

    rows = predictions[predictions["snapshot_date"].isin(evaluable_dates)].copy()

    def _is_positive(row):
        dates = hotspot_dates_by_cell.get(row["cell_id"])
        if not dates:
            return 0
        window = {row["snapshot_date"] + timedelta(days=o) for o in range(0, window_days + 1)}
        return int(bool(dates & window))

    rows["positive"] = rows.apply(_is_positive, axis=1)
    return rows


def per_day_topk(labels: pd.DataFrame, score_col: str, k_fraction: float):
    """Ranks WITHIN each day before pooling -- never a global cross-week
    ranking, which would let one hot day monopolize the top-k."""
    total_captured, total_flagged, total_positive = 0, 0, 0
    per_day = []
    for date, day_df in labels.groupby("snapshot_date"):
        n = len(day_df)
        k = max(1, int(np.ceil(n * k_fraction)))
        top_idx = np.argsort(-day_df[score_col].values, kind="stable")[:k]
        captured = int(day_df["positive"].values[top_idx].sum())
        day_positive = int(day_df["positive"].sum())
        total_captured += captured
        total_flagged += k
        total_positive += day_positive
        per_day.append({"date": str(date), "n": n, "k": k, "day_positives": day_positive, "captured": captured})
    return total_captured, total_flagged, total_positive, per_day


def tier_table(labels: pd.DataFrame, tier_col: str):
    rows = []
    for tier, tier_df in labels.groupby(tier_col, dropna=True):
        n = len(tier_df)
        positives = int(tier_df["positive"].sum())
        rows.append({
            "tier": tier, "n_cells_flagged": n,
            "pct_of_grid": round(n / len(labels) * 100, 2) if len(labels) else None,
            "positives_captured": positives,
            "realized_rate": positives / n if n else None,
        })
    return rows


def check_monotonicity(rows, tier_order):
    rates = []
    for t in tier_order:
        match = [r["realized_rate"] for r in rows if r["tier"] == t]
        if match and match[0] is not None:
            rates.append(match[0])
    return all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)) if len(rates) >= 2 else None


def run(log_dir: str):
    predictions, hotspots, evaluable_dates, meta_records = load_evaluable_data(log_dir)

    if not evaluable_dates:
        print("\nNo evaluable dates yet (either no covered hotspot pulls, or everything is still censored).")
        print("VERDICT: INCONCLUSIVE -- too early to evaluate.")
        return

    print(f"\n=== T1: serving-safety distribution check (from daily snapshots) ===")
    t1_days = [m for m in meta_records if m["snapshot_date"] in [str(d) for d in evaluable_dates]]
    t1_pass_days = sum(1 for m in t1_days if m["t1_serving_safety_check"]["_all_pass"])
    # Scaled proportionally to however many days have accumulated so far --
    # the pre-registered rule is ">=6 of 7", i.e. a >=85.7% pass rate; at the
    # full 7-day mark this reduces to exactly "6 of 7". Using a flat "6"
    # regardless of how many days exist would make T1 mathematically
    # unpassable during any partial/interim check with fewer than 6 days
    # accumulated -- a real bug caught by testing this against a 4-day
    # synthetic run before relying on it.
    t1_required = math.ceil(len(t1_days) * 6 / 7) if t1_days else 0
    print(f"  {t1_pass_days}/{len(t1_days)} evaluated days pass T1 "
          f"(need >={t1_required}/{len(t1_days)}, scaled from the pre-registered 6/7 rule)")
    for m in t1_days:
        print(f"  {m['snapshot_date']}: all_pass={m['t1_serving_safety_check']['_all_pass']}")
    t1_verdict = len(t1_days) > 0 and t1_pass_days >= t1_required

    print(f"\n=== Fire-occurrence tripwire (window={PRIMARY_WINDOW}, {FORWARD_WINDOWS[PRIMARY_WINDOW]}-day union forward) ===")
    labels_by_window = {name: build_labels(predictions, hotspots, evaluable_dates, w) for name, w in FORWARD_WINDOWS.items()}
    primary_labels = labels_by_window[PRIMARY_WINDOW]

    n_positive_cell_days = int(primary_labels["positive"].sum())
    n_distinct_fire_days = primary_labels.loc[primary_labels["positive"] == 1, ["cell_id", "snapshot_date"]].drop_duplicates().shape[0]
    print(f"  Evaluable (cell, date) rows: {len(primary_labels)}")
    print(f"  Positive (cell, date) pairs at {PRIMARY_WINDOW}: {n_positive_cell_days} "
          f"({n_distinct_fire_days} distinct (cell, day) positives)")

    inconclusive = n_distinct_fire_days < N_MIN
    if inconclusive:
        print(f"\n  {n_distinct_fire_days} < N_MIN={N_MIN} -- fire-occurrence tripwire is INCONCLUSIVE, extend the window.")

    for name in FORWARD_WINDOWS:
        lab = labels_by_window[name]
        pos = int(lab["positive"].sum())
        print(f"  [{name}] positive rows: {pos} ({'concurrent, not forward' if name == 'W0' else 'forward'})")

    print(f"\n=== Recall@k / lift (matched-area, ranked within-day, primary window {PRIMARY_WINDOW}) ===")
    results_by_system = {}
    for label, score_col in [("FWI (fwi)", "fwi"), ("ML (ml_risk_probability)", "ml_risk_probability")]:
        print(f"\n  -- {label} --")
        scored = primary_labels.dropna(subset=[score_col])
        system_results = {}
        for k_frac in TOP_K_FRACTIONS:
            captured, flagged, total_pos, per_day = per_day_topk(scored, score_col, k_frac)
            precision = captured / flagged if flagged else float("nan")
            recall = captured / total_pos if total_pos else float("nan")
            lift = recall / k_frac if k_frac else float("nan")
            lo, hi = _wilson_ci(captured, total_pos) if total_pos else (float("nan"), float("nan"))
            print(f"    top-{k_frac*100:.0f}%: captured={captured}/{total_pos}, recall={recall:.3f} "
                  f"(95% CI {lo:.3f}-{hi:.3f}), precision={precision:.4f}, lift={lift:.2f}x")
            system_results[k_frac] = {"captured": captured, "total_pos": total_pos, "recall": recall,
                                        "recall_ci": (lo, hi), "precision": precision, "lift": lift, "per_day": per_day}
        results_by_system[label] = system_results

    print("\n  -- Per-day breakdown (top-5%, primary window) --")
    for label in results_by_system:
        print(f"  {label}:")
        for row in results_by_system[label][0.05]["per_day"]:
            print(f"    {row}")

    fwi_wins = ml_wins = 0
    fwi_days = {r["date"]: r["captured"] for r in results_by_system["FWI (fwi)"][0.05]["per_day"]}
    ml_days = {r["date"]: r["captured"] for r in results_by_system["ML (ml_risk_probability)"][0.05]["per_day"]}
    for d in fwi_days:
        if d in ml_days:
            if ml_days[d] > fwi_days[d]:
                ml_wins += 1
            elif fwi_days[d] > ml_days[d]:
                fwi_wins += 1
    n_compared_days = fwi_wins + ml_wins
    sign_test_p = stats.binomtest(ml_wins, n_compared_days, 0.5).pvalue if n_compared_days > 0 else float("nan")
    print(f"\n  Day-level sign test (top-5% captures): ML won {ml_wins}, FWI won {fwi_wins}, "
          f"ties excluded, n={n_compared_days}, two-sided p={sign_test_p:.3f} "
          f"(note: {n_compared_days} days cannot reach significance -- reported for transparency only)")

    print("\n=== Tier tables (realized rate + alarm area, primary window) ===")
    print("\n  -- FWI danger_class (6 tiers) --")
    fwi_tiers = tier_table(primary_labels, "danger_class")
    for r in fwi_tiers:
        print(f"    {r}")
    fwi_order = ["Very Low", "Low", "Moderate", "High", "Very High", "Extreme"]
    fwi_monotone = check_monotonicity(fwi_tiers, fwi_order)

    print("\n  -- ML ml_danger_class (4 tiers) --")
    ml_tiers = tier_table(primary_labels.dropna(subset=["ml_danger_class"]), "ml_danger_class")
    for r in ml_tiers:
        print(f"    {r}")
    ml_order = ["Very Low", "Low", "Moderate", "High"]
    ml_monotone = check_monotonicity(ml_tiers, ml_order)
    print(f"\n  ML tier fire-rate shape reference (Stage 10, NFDB-labelled, Apr-Oct test split -- "
          f"different modality/base rate, compare ORDERING only, never absolute levels): "
          f"[0.00014, 0.00064, 0.0072, 0.0872]")

    print("\n=== Tripwire verdicts ===")
    t2 = ml_monotone if ml_monotone is not None else None
    t2_str = "PASS" if t2 else ("FAIL" if t2 is False else "N/A (insufficient data)")
    print(f"  T1 (serving-safety, >=6/7 days in-band):        {'PASS' if t1_verdict else 'FAIL'}")
    print(f"  T2 (ML tier monotonicity):                       {t2_str}")
    print(f"  T3 (ML tier shares non-degenerate):              see meta.json danger_class_counts per day, not auto-scored here")

    ml_recall5 = results_by_system["ML (ml_risk_probability)"][0.05]["recall"]
    fwi_recall5 = results_by_system["FWI (fwi)"][0.05]["recall"]
    t4 = (not np.isnan(ml_recall5) and not np.isnan(fwi_recall5) and fwi_recall5 > 0
          and ml_recall5 >= 0.7 * fwi_recall5)
    print(f"  T4 (ML recall@top5% >= 0.7x FWI's):              {'PASS' if t4 else 'FAIL'} "
          f"(ML={ml_recall5:.3f}, FWI={fwi_recall5:.3f})")

    ml_lift5 = results_by_system["ML (ml_risk_probability)"][0.05]["lift"]
    t5 = not np.isnan(ml_lift5) and ml_lift5 > 1.0
    print(f"  T5 (ML lift@top5% > 1.0):                        {'PASS' if t5 else 'FAIL'} (lift={ml_lift5:.2f}x)")

    print("\n=== Overall ===")
    if inconclusive:
        print("VERDICT: INCONCLUSIVE (fewer than N_MIN distinct fire-day positives) -- extend the window.")
    elif t1_verdict and t2 and t4 and t5:
        print("VERDICT: All tripwires pass. Consider promotion -- this is still a separate, subsequent decision.")
    elif not t1_verdict:
        print("VERDICT: T1 (primary) fails -- investigate provider drift before doing anything else. Do not promote.")
    elif t2 is False:
        print("VERDICT: T2/T3 fail -- do not promote.")
    else:
        print("VERDICT: T4/T5 fail -- extend the window, insufficient evidence to decide either way.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()
    run(args.log_dir)
