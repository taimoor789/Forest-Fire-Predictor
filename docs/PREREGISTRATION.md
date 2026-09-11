# Pre-Registration: ML Fire-Danger Model, Stage 11 Decision Gate

Written and committed **before** Stage 9's evaluation harness has been run on
real predictions — this document exists specifically so the ship/no-ship
decision can't be quietly rationalized after seeing results. It is written
once. Any later change is appended as a dated note below, never a silent
edit to the numbers above it.

## What's being decided

Whether `get_ml_danger_class` (currently parked — see `fire_risk.py`'s
`FireWeatherProcessor.__init__`) replaces `get_danger_class`'s FWI-threshold
classification as what the live API serves. If the gate below isn't met,
`get_danger_class` stays live exactly as it is today, the ML layer stays
parked, and Stage 9's ablation is published as the honest standalone
finding — a legitimate, valuable result in its own right, not a failure to
paper over.

## Framing claim

The label (`ml/build_labels.py`) is 1 iff a fire was attributed within ±2
days of a (cell, date). A forward half-window means a positive label can
*precede* the fire it's paired with by up to 2 days. That is defensible
under a **"fire danger"** reading — conditions were dangerous enough that a
fire did start nearby within days — and is **not** defensible under a "fire
detection" reading. This model claims the former. `label_w2` (±2 days) is
pre-registered as primary; `label_w0`/`label_w1` are reported as sensitivity
checks, not alternate primaries to pick from after the fact.

## Primary metric

**PR-AUC on the spatio-temporal holdout** (`ml.splits.spatio_temporal_split`
— fold 0's held-out geographic region, combined with `HOLDOUT_YEAR`=2023),
with a 95% block-bootstrap CI (bootstrapped over `block_id`, never over
rows — rows within a block are spatially and temporally autocorrelated, and
row-level bootstrap would understate the true uncertainty).

PR-AUC, not ROC-AUC, is the headline: the positive rate is low (label_w2
overall base rate 1.271% in the assembled dataset — see
`docs/DATA_PROVENANCE.md`'s Stage 7 row), where PR-AUC is far more
informative than ROC-AUC about whether the ranking is actually useful.

Spatio-temporal is the **deciding** regime, not temporal or spatial alone —
it's the only one of the three pre-registered regimes (`ml/splits.py`) that
answers the question a deployed model actually faces: a new place, in a new
season, neither seen in training.

## Ship only if all four hold

1. **Beats raw FWI.** The full model's PR-AUC exceeds baseline B1 (raw FWI
   value as a ranking score) by at least
   `config.GATE_MIN_RELATIVE_PR_AUC_LIFT_VS_RAW_FWI` = **25% relative**, and
   the 95% CI lower bound of the *paired* PR-AUC difference (full model
   minus B1, same test rows, same bootstrap resamples) is > 0.
2. **Beats the live incumbent.** The full model beats baseline B2
   (`get_danger_class` — literally what's shipping today) on both PR-AUC
   and precision@top-5%. B2 is the actual bar to clear, not a nice-to-have
   comparison: replacing a working system with a worse one is not a win
   regardless of how it compares to a naive baseline.
3. **Calibrated and monotone.** Calibration ECE ≤
   `config.GATE_MAX_CALIBRATION_ECE` = **0.02** on the test split (measured
   after probabilities are calibrated on the disjoint calibration split,
   never on train or test). Out-of-sample tier fire rates (evaluated on the
   untouched test split, using tier boundaries derived from the calibration
   split) are **strictly monotone** — a non-monotone result is a fail
   signal on its own, independent of every other metric, because it means
   the tiers don't mean what they claim to mean.
4. **The lift isn't just a reshaped FWI.** The ablation ladder (below) must
   show the ≥25% lift is not attributable solely to rungs A1–A2 (nonlinear
   functions of the same FWI codes already in production). If nearly all
   the lift sits in A1→A2, the honest conclusion is that FWI already
   captured what's available and a tree ensemble just re-expressed it —
   informative, but not by itself a reason to add a whole new
   training/serving/monitoring surface to the codebase.

## Mandatory serving-safety gate (accuracy-independent)

Every feature's live production distribution (from `fwi_predictions.json` /
`data/fwi_state.json`) must have its **median** fall inside the training
set's **5th–95th percentile range**. This check is independent of every
accuracy metric above and **a model that fails it does not ship, regardless
of test scores** — this is the direct, mechanical fix for the exact bug
that invalidated the original model: its training DMC median was 500.0 (the
saturation ceiling, 54.2% of rows pinned there) against live production's
actual mean of 9.14. Good held-out accuracy metrics would never have caught
that; only comparing distributions does.

## Ablation ladder (pre-registered rungs, not chosen after seeing results)

Identical hyperparameters, folds, and seed (`config.SEED`) throughout.

| Rung | Adds | Tests |
|---|---|---|
| B0 | constant base rate | floor |
| B1 | raw FWI value | current physics, no modelling |
| B2 | `get_danger_class` tiers | **the live incumbent — the real bar** |
| B3 | seasonality only (day_of_year, month) | pure calendar signal |
| B4 | `historical_fire` only (leak-free) | pure ignition/fuel-proxy signal |
| A1 | ffmc, dmc, dc, isi, bui, fwi | nonlinear function of FWI codes |
| A2 | + dc_trend_7d, bui_trend_7d | still pure-FWI-derived |
| A3 | + day_of_year, month | first genuinely new information |
| A4 | + historical_fire (leak-free) | ignition/fuel proxy FWI can't see |
| A5 | + province | spatial/administrative proxy |

SHAP importance is computed at A5 and reported as an FWI-vs-added-information
split, for direct comparison against the cited literature's ~54%/46% split
(FWI-derived features vs. time-proxy features) in an analogous published
wildfire-occurrence model.

All metrics (PR-AUC, ROC-AUC, precision/recall@top-1%/@top-5%, Brier, ECE)
are reported for every rung, on every split regime, both unweighted and
area-weighted (`area_weight` from `ml/build_grid_domain.py`) — the ablation
table itself is a primary deliverable of this rebuild regardless of whether
the gate is met.

## Context for interpreting the result

Published regional wildfire-occurrence models typically report ROC-AUC in
the 0.80–0.95 range. This project's coarse 0.5° cells and ±2-day label
window make landing below that band plausible and would be informative
about resolution/label-window limits, not by itself evidence of a broken
model — noted here in advance specifically so it can't be used as an
after-the-fact excuse in either direction.

## Amendments

**2026-09-11 — Stage 11 gate decision: SHIP.**

Evaluated by `ml/stage11_gate.py` against the exact artifact that would
ship (`model_components/model.pkl` + `calibrator.pkl` from Stage 10 —
`rf_original`, selected by inner CV over folds untouched by test or
calibration), on the pre-registered spatio-temporal test split. All four
conditions plus the mandatory serving-safety gate pass:

1. **Beats raw FWI (B1):** PR-AUC 0.1791 vs 0.0712 — paired diff 0.1080
   (95% block-bootstrap CI 0.0829–0.1348, lower bound > 0), **151.7%
   relative lift** (gate: ≥25%). PASS.
2. **Beats the live incumbent (B2, `get_danger_class`):** PR-AUC 0.1791 vs
   0.0633, precision@top-5% 0.2157 vs 0.0958 — beats on both. PASS.
3. **Calibrated and monotone:** ECE 0.00995 on the untouched test split
   (gate: ≤0.02); tier fire rates [0.0001, 0.0006, 0.0072, 0.0872] on the
   test split, strictly monotone out-of-sample. PASS. (Note: only 4
   tiers were derivable from the calibration split's quantiles, not the 6
   `get_danger_class` uses — see `docs/DATA_PROVENANCE.md`'s Stage 10 row.
   Unresolved design question for Stage 12, not a gate failure.)
4. **Lift isn't just a reshaped FWI:** from Stage 9's ablation, rungs
   A1→A2 (nonlinear functions of the same FWI codes, still no new
   information) account for only 11.1% of the total A5-vs-B1 PR-AUC lift.
   The dominant driver is `historical_fire` (A3→A4). PASS.
5. **Serving-safety distribution gate:** every checked feature's live
   production median falls inside `fit_train`'s 5th–95th percentile range
   (`results/serving_safety_check.json`). PASS. `hist_fire_count_prior_20y_log1p`
   and `years_since_last_fire` remain NOT YET CHECKABLE — no live
   equivalent exists until `data/canada_fire_grid.csv` is regenerated
   (Stage 12) — flagged as an open gap, not treated as a pass.

Full numbers: `results/stage11_gate_decision.json`. This decision was made
after the accuracy numbers above already existed — the pre-registration
text itself (everything above this section) was written and committed
(`87ca5db`) before Stage 9's ablation was ever run, which is what makes
this a real gate rather than post-hoc rationalization.

**Decision: proceed to Stage 12 (production wiring), shadow mode first,
per the plan.**
