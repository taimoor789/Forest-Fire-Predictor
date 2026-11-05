"""
Stage 2: leak-free historical_fire, replacing the original
data/canada_fire_grid.csv:historical_fire column, which was built from NFDB's
all-time record with no date filter -- it included each target year's own
fires (measured: P(fire in 2023 | historical_fire=0) = 0.0 exactly across
200,020 rows) and was the #1 feature by importance in the quarantined model.

For target year Y, only fires with YEAR <= Y-1 may contribute -- enforced
structurally here, not just by convention, and checked directly by
test_historical_fire_leakage.py.

Every NFDB fire is attributed to its nearest in-domain grid cell ONCE (via
ml/attribution.py, shared with Stage 3 so the two don't silently diverge on
what "near this cell" means), independent of year. Per-year variants are
then just a matter of filtering that attribution by date -- attribution
doesn't need to be recomputed per year.

Emits several variants (see ml/config.py) rather than committing to one
blind: hist_fire_count_prior_20y (+log1p) is pre-registered as primary --
a 40+ year binary saturates across the whole boreal forest and carries
almost no signal, while a bounded recent count is robust to NFDB's changing
reporting completeness across eras and still proxies human ignition
pressure / fuel-vegetation type, information FWI structurally can't see.

Run as a module from the repo root: python3 -m ml.build_historical_fire
"""

from datetime import datetime

import numpy as np
import pandas as pd

from ml import config, manifest, nfdb

# Kept as module-level aliases so callers (and test_historical_fire_leakage.py)
# don't need to know these moved to ml/nfdb.py when Stage 3 needed to share them.
load_nfdb = nfdb.load_nfdb
attribute_all_fires = nfdb.attribute_all_fires


def features_for_year(attributed_fires: pd.DataFrame, grid_domain: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Only fires strictly before target_year contribute -- the leak-free
    contract. Returns one row per in-domain cell."""
    prior = attributed_fires[
        (attributed_fires["year"] < target_year) & (attributed_fires["year"] >= config.HISTORICAL_FIRE_MIN_YEAR)
    ]
    window_20y = prior[prior["year"] >= target_year - config.HISTORICAL_FIRE_LOOKBACK_YEARS]
    window_10y = prior[prior["year"] >= target_year - 10]

    any_prior = prior.groupby("cell_id").size().rename("hist_fire_count_any_prior")
    count_20y = window_20y.groupby("cell_id").size().rename("hist_fire_count_prior_20y")
    count_10y = window_10y.groupby("cell_id").size().rename("hist_fire_count_prior_10y")
    area_20y = window_20y.groupby("cell_id")["size_ha"].sum().rename("hist_area_burned_prior_20y")
    last_fire_year = prior.groupby("cell_id")["year"].max().rename("last_fire_year")

    out = grid_domain[["cell_id"]].copy()
    out["target_year"] = target_year
    out = out.merge(any_prior, on="cell_id", how="left")
    out = out.merge(count_20y, on="cell_id", how="left")
    out = out.merge(count_10y, on="cell_id", how="left")
    out = out.merge(area_20y, on="cell_id", how="left")
    out = out.merge(last_fire_year, on="cell_id", how="left")

    out["hist_fire_count_any_prior"] = out["hist_fire_count_any_prior"].fillna(0).astype(int)
    out["hist_fire_count_prior_20y"] = out["hist_fire_count_prior_20y"].fillna(0).astype(int)
    out["hist_fire_count_prior_10y"] = out["hist_fire_count_prior_10y"].fillna(0).astype(int)
    out["hist_area_burned_prior_20y"] = out["hist_area_burned_prior_20y"].fillna(0.0)

    out["hist_fire_any_prior"] = (out["hist_fire_count_any_prior"] > 0).astype(int)
    out["hist_fire_count_prior_20y_log1p"] = np.log1p(out["hist_fire_count_prior_20y"])
    out["hist_area_burned_prior_20y_log1p"] = np.log1p(out["hist_area_burned_prior_20y"])
    out["years_since_last_fire"] = np.where(
        out["last_fire_year"].notna(),
        (target_year - out["last_fire_year"]).clip(upper=40),
        40,
    ).astype(int)

    return out.drop(columns=["last_fire_year"])


def build():
    run_id = manifest.start_run("stage2_historical_fire")

    grid_domain = pd.read_parquet(config.GRID_DOMAIN_PATH)
    grid_domain = grid_domain[grid_domain["in_canada"]].reset_index(drop=True)
    print(f"{len(grid_domain)} in-domain cells")

    fires = load_nfdb()
    attributed = attribute_all_fires(fires, grid_domain)

    current_year = datetime.now().year
    target_years = sorted(set(config.TRAINING_YEARS + [config.HOLDOUT_YEAR, current_year]))
    print(f"\nBuilding historical_fire features for target years: {target_years}")

    frames = [features_for_year(attributed, grid_domain, y) for y in target_years]
    out = pd.concat(frames, ignore_index=True)

    print(f"\n{len(out)} (cell, target_year) rows")
    print("hist_fire_any_prior rate by target_year:")
    print(out.groupby("target_year")["hist_fire_any_prior"].mean())

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_DIR / "historical_fire_by_year.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    entry = manifest.record(
        run_id, out_path,
        upstream=[config.GRID_DOMAIN_PATH],
        extra={
            "nfdb_source": "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_shp.zip",
            "nfdb_rows_raw": int(len(fires)) + 0,  # post-bad-row-filter count already in fires
            "n_fires_valid": len(fires), "n_fires_attributed": len(attributed),
            "target_years": target_years,
        },
    )
    print(f"Manifest entry recorded: run_id={run_id}")
    return out


if __name__ == "__main__":
    build()
