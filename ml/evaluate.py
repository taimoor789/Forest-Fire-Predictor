"""
Stage 9: evaluation harness. No evaluation code exists anywhere else in the
repo or its git history for either RandomForest or XGBoost -- this is
written from scratch, not adapted from something that already existed.

PR-AUC is the headline metric (not ROC-AUC): the positive rate is low
(~1.27% overall, see docs/DATA_PROVENANCE.md's Stage 7 row), where ROC-AUC
can look deceptively good while PR-AUC exposes how much of the ranking is
actually useful. All metrics are reported both unweighted and area-weighted
(`area_weight` from Stage 1, to counter the Arctic's over-representation by
cell COUNT relative to true land area).

CIs use block bootstrap -- resampling unique block_id values with
replacement, never individual rows -- because rows within a block are
spatially and temporally autocorrelated; a row-level bootstrap would
understate the true uncertainty. See docs/PREREGISTRATION.md for how these
functions feed the Stage 11 gate.
"""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml import config


def precision_recall_at_k(y_true, y_score, k_fraction, weights=None):
    """Precision/recall among the top k_fraction of rows by score. Ties at
    the cutoff are all included (np.argsort is stable but ties can still
    straddle the cutoff for a discrete score like B2's tier ordinal) --
    acceptable here since this exists to compare a discretized incumbent
    against continuous scores, not to be tie-free itself.

    With `weights` (area_weight): the top-k ROWS selected are unchanged
    (still ranked by raw score) -- only the precision/recall counts within
    that slice are area-weighted, so a covered-but-sparse Arctic cell
    doesn't count the same as a densely-representative boreal one."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_fraction)))
    top_idx = np.argsort(-y_score, kind="stable")[:k]

    if weights is None:
        n_pos_total = y_true.sum()
        n_pos_in_top = y_true[top_idx].sum()
        denom_top = k
    else:
        weights = np.asarray(weights)
        n_pos_total = (y_true * weights).sum()
        n_pos_in_top = (y_true[top_idx] * weights[top_idx]).sum()
        denom_top = weights[top_idx].sum()

    precision = n_pos_in_top / denom_top
    recall = n_pos_in_top / n_pos_total if n_pos_total > 0 else np.nan
    return float(precision), float(recall)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Mean absolute gap between predicted probability and observed
    frequency, bin-weighted by bin size. Only meaningful for genuinely
    calibrated probabilities (Stage 10's output) -- computing this on a raw
    ranking score (a baseline's FWI value, say) is meaningless and callers
    should not report it for those."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_weight = mask.sum() / n
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += bin_weight * abs(bin_conf - bin_acc)
    return float(ece)


def _weighted_average_precision(y_true, y_score, sample_weight):
    # sklearn's average_precision_score supports sample_weight directly.
    return average_precision_score(y_true, y_score, sample_weight=sample_weight)


def compute_metrics(y_true, y_score, weights=None, is_probability=False):
    """y_score: a ranking score for PR-AUC/ROC-AUC/precision@k (any
    monotonic transform is fine for these). If is_probability, y_score is
    ALSO treated as a calibrated probability and Brier/ECE are computed on
    it too -- otherwise those two keys are omitted (reporting Brier/ECE on
    an uncalibrated raw score, like B1's raw FWI value, would be
    meaningless and misleading)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        raise ValueError("compute_metrics: y_true has only one class in this split -- cannot compute rank metrics")

    metrics = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "base_rate": float(np.average(y_true, weights=weights)),
        "pr_auc": float(average_precision_score(y_true, y_score, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(y_true, y_score, sample_weight=weights)),
    }
    for k_frac, label in [(0.01, "top1pct"), (0.05, "top5pct")]:
        precision, recall = precision_recall_at_k(y_true, y_score, k_frac, weights=weights)
        metrics[f"precision_{label}"] = precision
        metrics[f"recall_{label}"] = recall

    if is_probability:
        metrics["brier"] = float(brier_score_loss(y_true, y_score, sample_weight=weights))
        metrics["ece"] = expected_calibration_error(y_true, y_score)

    return metrics


def block_bootstrap_ci(y_true, y_score, block_ids, metric_fn, n_bootstrap=None, seed=None, ci=0.95):
    """metric_fn(y_true_subset, y_score_subset) -> float. Resamples unique
    block_ids with replacement (a block can appear more than once per
    resample -- that's the point, it's what makes this a block bootstrap
    rather than a stratified split), reconstructs the corresponding rows,
    and recomputes metric_fn each time. Returns (point_estimate, lo, hi)."""
    n_bootstrap = n_bootstrap or config.N_BOOTSTRAP
    seed = config.SEED if seed is None else seed
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    block_ids = np.asarray(block_ids)

    unique_blocks = np.unique(block_ids)
    rows_by_block = {b: np.where(block_ids == b)[0] for b in unique_blocks}

    point = metric_fn(y_true, y_score)

    rng = np.random.RandomState(seed)
    boot_values = []
    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([rows_by_block[b] for b in sampled_blocks])
        try:
            boot_values.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue  # a resample with only one class present -- skip, don't crash the whole CI
    boot_values = np.array(boot_values)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_values, [alpha, 1 - alpha])
    return float(point), float(lo), float(hi)


def paired_block_bootstrap_diff(y_true, score_a, score_b, block_ids, metric_fn, n_bootstrap=None, seed=None, ci=0.95):
    """CI on metric_fn(score_a) - metric_fn(score_b), using the SAME
    bootstrap resamples for both (paired), which is what Stage 11's gate
    condition 1 needs: a comparison, not two independent CIs whose overlap
    is eyeballed."""
    n_bootstrap = n_bootstrap or config.N_BOOTSTRAP
    seed = config.SEED if seed is None else seed
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    block_ids = np.asarray(block_ids)

    unique_blocks = np.unique(block_ids)
    rows_by_block = {b: np.where(block_ids == b)[0] for b in unique_blocks}

    point = metric_fn(y_true, score_a) - metric_fn(y_true, score_b)

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([rows_by_block[b] for b in sampled_blocks])
        try:
            diffs.append(metric_fn(y_true[idx], score_a[idx]) - metric_fn(y_true[idx], score_b[idx]))
        except ValueError:
            continue
    diffs = np.array(diffs)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    return float(point), float(lo), float(hi)
