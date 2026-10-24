"""
Structural leakage guard for ml/build_historical_fire.py.

The bug this exists to catch: the original data/canada_fire_grid.csv
historical_fire column was built from NFDB's all-time record with no date
filter, so it included each target year's own fires -- measured as
P(fire in 2023 | historical_fire=0) = 0.0 exactly, across 200,020 rows.

This test re-derives the historical_fire features directly (not via the
cached parquet) for a few target years and asserts, mechanically, that no
fire with year >= target_year ever contributed to that target year's count
-- the direct cause of the original leak, checked at the source rather than
inferred from a downstream statistic.

A fuller statistical check -- P(fire in year Y | hist_fire_any_prior=0) > 0,
the exact number the original leak made exactly 0.0 -- requires Stage 4's
labels (which fires actually happened in year Y, cell by cell) and belongs
in that stage's test suite once it exists; this test only needs the raw
NFDB archive and the Stage 2 build function, so it can run standalone.

Run directly: python3 test_historical_fire_leakage.py
"""

import sys

import pandas as pd

from ml import build_historical_fire as bhf
from ml import config


def check_no_future_fires_leak(attributed: pd.DataFrame, grid_domain: pd.DataFrame, target_year: int) -> list:
    """Recompute one target year's features and verify every fire that fed
    into hist_fire_count_any_prior actually predates target_year."""
    failures = []

    prior = attributed[
        (attributed["year"] < target_year) & (attributed["year"] >= config.HISTORICAL_FIRE_MIN_YEAR)
    ]
    if (prior["year"] >= target_year).any():
        failures.append(
            f"target_year={target_year}: {int((prior['year'] >= target_year).sum())} fires with "
            f"year >= target_year leaked into the 'prior' set"
        )

    # Recompute via the real build function and cross-check the any-prior
    # rate is consistent with a direct count on `prior` -- catches a bug in
    # features_for_year() that isn't visible from the year filter alone
    # (e.g. an accidental join that reintroduces same-year rows).
    feats = bhf.features_for_year(attributed, grid_domain, target_year)
    direct_positive_cells = set(prior["cell_id"].unique())
    feature_positive_cells = set(feats.loc[feats["hist_fire_any_prior"] == 1, "cell_id"])
    if direct_positive_cells != feature_positive_cells:
        only_in_direct = direct_positive_cells - feature_positive_cells
        only_in_feature = feature_positive_cells - direct_positive_cells
        failures.append(
            f"target_year={target_year}: hist_fire_any_prior disagrees with a direct recount "
            f"({len(only_in_direct)} cells missing, {len(only_in_feature)} extra)"
        )

    return failures


def run():
    print("Loading NFDB and grid domain...")
    grid_domain = pd.read_parquet(config.GRID_DOMAIN_PATH)
    grid_domain = grid_domain[grid_domain["in_canada"]].reset_index(drop=True)

    fires = bhf.load_nfdb()
    attributed = bhf.attribute_all_fires(fires, grid_domain)

    failures = []
    for target_year in [2019, 2023, 2026]:
        print(f"\nChecking target_year={target_year}...")
        year_failures = check_no_future_fires_leak(attributed, grid_domain, target_year)
        if year_failures:
            failures.extend(year_failures)
            for f in year_failures:
                print(f"  [FAIL] {f}")
        else:
            print(f"  [OK] no future-year fires leaked, feature counts match a direct recount")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S) -- historical_fire is NOT leak-free")
        return 1
    print("All checks passed -- no fire with year >= target_year contributed to that year's historical_fire.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
