# Data Provenance

This tracks what every data/model artifact in this project is, what produced
it, and — most importantly — which artifacts are **quarantined**: known
invalid, not to be reused, kept around only as an audit trail.

`data/` and `model_components/` are both gitignored (see `.gitignore`), so
this file — not git history — is the durable record of where each artifact
came from. Pair it with `artifacts/manifest.json` (`ml/manifest.py`) for the
machine-readable version: every artifact the `ml/` rebuild pipeline writes is
recorded there with a run ID, a config hash, and its upstream dependencies.

## ⚠️ Quarantined — do not reuse

These were produced by the pre-rebuild ML pipeline and are known invalid.
Superseded versions are moved to `archive/` rather than deleted (Stage 12),
so nothing can load them by accident once the rebuild's artifacts exist.

| Artifact | Why it's quarantined |
|---|---|
| `data/canada_fire_grid.csv:historical_fire` | Built from NFDB's all-time record with no date filter — includes each target year's own fires. Measured: `P(fire in 2023 \| historical_fire=0) = 0.0` exactly, across 200,020 rows. The other two columns (`lat`, `lon`) are fine; only this column is quarantined. |
| `data/fwi_backtest_2023.csv` | Produced by `forest_fire_data_preparation.ipynb` cell 6's FWI engine, which still has the pre-fix DMC/DC bugs (wrong day-length table, missing moisture→DMC conversion, DC missing its 0.5× multiplier) and flat (non-seasonal) starting codes. |
| `data/ml_features_2023.csv` | Downstream of `fwi_backtest_2023.csv` — inherits all of its formula bugs, plus the wrong danger-class thresholds and the 1.15× historical-fire multiplier baked into a `risk_prob` column. |
| `data/ml_train_featured.csv`, `data/ml_test_featured.csv` | A pure date-cut of `ml_features_2023.csv` with `actual_fire` appended. No spatial holdout (`P(historical_fire=1)` is identical to 13 decimal places between the two files — every test cell was seen in training). The `risk_prob` column is a near-perfect proxy for `historical_fire` (12 discrete values from the 1.15× multiplier) — a leak trap for anyone who reuses "all numeric columns." |
| `data/locations_to_pull.csv` | The weather-sampling frame: 2,033 cells that burned in 2023 + 800 hand-picked controls (2,166 of 2,833 already `historical_fire=1`). Case-control biased, no offset/weighting applied. |
| `data/nasa_power_weather_filtered.csv` | Wind is m/s, fed into the FWI formulas unconverted (production expects km/h) — measured mean 3.49 m/s vs. production's 19.60 km/h. Daily means used where FWI requires local-noon readings — measured RH mean 87.2% (53.5% of rows >90%) vs. production's local-noon 78.7% (34.3% >90%). Single season only (2022-11-15 to 2023-12-31). |
| `model_components/fire_risk_ml_model.pkl` | Trained on `ml_train_featured.csv` + `ml_test_featured.csv` **concatenated** (`scripts/train_final_model.py:11-15`, comment: "since the split's validation job is done") — no held-out data, no evaluation computed anywhere. Top feature importance is the leaked `historical_fire` (0.137). |
| `model_components/fire_risk_ml_features.json` | Schema for the quarantined model above — same run, same invalidity. |
| `model_components/ml_tier_thresholds.json` | File timestamp (Sep 6, 02:43) is ~10 hours older than the model it's supposedly calibrated for (Sep 6, 12:33) — verified directly. It was calibrated against a *different* model fit than the one actually shipped. |

**Not quarantined — genuinely reusable, with a caveat:**

| Artifact | Status |
|---|---|
| `data/nfdb_fires_2023.csv` | Sound: 6,847 rows, zero nulls, the ~0.8% positive-longitude rows are genuine source data-entry errors correctly filtered by the notebook. Reusable as-is for a 2023-only extract, but Stage 2 needs the *full* multi-year NFDB archive, which this file is not. |
| `data/ground_truth_2023.csv` | Spatial-join methodology (EPSG:3347, NFDB filter) is sound and leak-free. The 25km buffer attribution radius is not — see the rebuild plan's Stage 3 (17.43% of fires silently dropped, buffer doesn't tile the 0.5° grid). Reusable as a *reference* for validating the Stage 3 rebuild's output, not as a direct input. |

## Active / live production artifacts (not part of the ML rebuild)

| Artifact | Produced by | Notes |
|---|---|---|
| `data/canada_fire_grid.csv` (`lat`, `lon` columns) | `forest_fire_data_preparation.ipynb` cell 2 | The live 14,952-cell grid. Read by `collect_weather_grid_eccc.py` and `fire_risk.py`. Not mutated by the ML rebuild until Stage 12, and even then only the `historical_fire` column + a new `in_canada` column are touched. |
| `data/fwi_state.json` | `fire_risk.py`'s `save_fwi_state` | Live persisted per-cell FFMC/DMC/DC + 7-day trend history. Reference distribution for Stage 5's weather-source sanity gate and Stage 11's serving-safety gate. |
| `weather_data/*.csv` | `collect_weather_grid_eccc.py` | Live daily weather pulls (HRDPS/GDPS/HRDPA). Reference for `test_weather_adapter_units.py` and `test_replay_parity.py`. |
| `fwi_predictions.json`, `model_info.json` | `fire_risk.py`'s `main()` | Live API output. Reference distribution for the Stage 11 serving-safety gate. |

## ML rebuild artifacts (produced by `ml/`, tracked in `artifacts/manifest.json`)

Populated as each stage runs. See `artifacts/manifest.json` for the
authoritative, timestamped, hash-verified record — this table is a
human-readable index, not the source of truth.

| Artifact | Stage | Status |
|---|---|---|
| `data/grid_domain_v1.parquet` | 1 | **built** — 7,537/14,952 cells (50.4%) in Canada after a real land-mask (vs. the original audit's rough box-estimate of ~71.5% in-domain); Arctic (>=70.5N) is 29.76% of in-domain cells by count but 14.98% by area |
| `data/historical_fire_by_year.parquet` | 2 | **built** — 45,222 rows (7,537 in-domain cells × 6 target years). Full NFDB archive (448,284 valid fires, 1930–2025) attributed to nearest in-domain cell (99.98% attributed, vs. the old buffer join's 82.6%). `hist_fire_any_prior` rate ~50–52% across target years. Leak-freedom verified by `test_historical_fire_leakage.py`, including a negative control confirming it catches a simulated leak. |
| `data/ground_truth_by_year.parquet` | 3 | **built** — 22,589 (cell, date) rows across 2019–2023. Nearest-cell attribution: 99.99% of in-scope fires attributed (vs. the original 25km buffer's 82.6%), median distance 17.3km. `data/ground_truth_buffer_sensitivity.parquet` is the 30km-buffer sensitivity variant (97.7% of fires matched, 14,822 multi-attributed pairs — buffers allow one fire to match several cells, unlike Voronoi). |
| `data/labels_by_year.parquet` | 4 | **built** — 8,064,590 rows (full cross product: 7,537 in-domain cells × 214 fire-season days × 5 years, 2019–2023), zero implicit negatives. Base rates: label_w0 0.273%, label_w1 0.751%, **label_w2 1.185% (primary)**. Verified: label_w0 ⊆ label_w1 ⊆ label_w2 with zero violations, zero duplicate (cell, date, year) rows. |
| `data/reference/era5/{hourly,precip}_{year}.grib` | 5 | **acquired** — ERA5-Land via Copernicus CDS API, 2019–2023, ~1.7GB total (raw source files, not a manifest-tracked derived artifact). CaSPAr (the original plan, same model family as production) was unreachable for 10+ hours with no outage info, so ERA5 is the fallback — a different provider than live production, residual bias expected and measurable. `hourly_{year}.grib`: temp/dewpoint/wind-u/wind-v/pressure at the 5 UTC hours corresponding to local noon somewhere in Canada. `precip_{year}.grib`: total_precipitation at all 24 hourly forecast-steps of each day's accumulation run (needed to reconstruct a rolling 24h sum — see `ml/weather/adapter.py`, not a single reading). The `precip_2023.grib` file's `time` dimension spans Feb 28–Oct 31 (246 entries) rather than the intended Mar 31–Oct 31 (215 days): a CDS request-builder bug (fixed in this commit, not re-downloaded) flattened per-month day lists into a single day-of-month union before sending them, so CDS's month×day cross product silently expanded the request to ~all of March–October. Harmless for the already-downloaded files — it only added extra lead-in days, never wrong or missing calendar dates — verified directly against the file's `valid_time` coordinate. |
| `data/fwi_replay_{year}.parquet` | 6 | not yet built |
| `data/dataset_{years}.parquet` | 7 | not yet built |
| `results/ablation_{run_id}.json` | 9 | not yet built |
| `model_components/model.pkl` + `features.json` + `tiers.json` | 10 | not yet built |
