"""
Stage 4: label construction. This code does not exist anywhere in the repo
or its git history -- not in the notebook, not in any commit -- so it's
written fresh here, not "fixed."

label(cell, date) = 1 iff a ground-truth fire was attributed to that cell
within +/-W days of `date`. +/-2 days is confirmed (empirically, since the
original code is gone) as the window that reproduces 100% of the known 2023
positives from the raw ground-truth rows; +/-1 covers only 63.7%. All three
windows (w0/w1/w2) are emitted as separate columns -- a reported
sensitivity, not a buried constant -- with w2 pre-registered as primary.

Framing: a forward half-window means a positive label can precede its fire
by up to W days. That's defensible under a "fire danger" reading (conditions
were dangerous enough that a fire did start nearby within days) and not
under a "fire detection" reading. This rebuild claims the former -- see
docs/PREREGISTRATION.md.

Row universe is the full cross product of in-domain cells x fire-season
dates x target years, zero everywhere uncovered -- never positives-only
joined with implicit negatives, which is how a case-control frame sneaks
back in (see ml/build_dataset.py / Stage 7 for why that matters here).

Run as a module from the repo root: python3 -m ml.build_labels
"""

from datetime import date

import numpy as np
import pandas as pd

from ml import config, manifest

MAX_WINDOW = max(config.LABEL_WINDOWS_DAYS)


def season_dates(year: int) -> np.ndarray:
    start = date(year, *config.FIRE_SEASON_START_MD)
    end = date(year, *config.FIRE_SEASON_END_MD)
    n_days = (end - start).days + 1
    return np.array([np.datetime64(start) + np.timedelta64(i, "D") for i in range(n_days)])


def labels_for_cell_year(cell_dates: np.ndarray, fire_dates: np.ndarray) -> np.ndarray:
    """Days from each date in cell_dates to the nearest fire_date (int16,
    large sentinel if fire_dates is empty). Vectorized via broadcasting --
    cheap since fire_dates is tiny per cell (usually 0-3 events)."""
    if len(fire_dates) == 0:
        return np.full(len(cell_dates), 9999, dtype=np.int16)
    diffs = np.abs((cell_dates[:, None] - fire_dates[None, :]).astype("timedelta64[D]").astype(np.int32))
    return diffs.min(axis=1).astype(np.int16)


def build():
    run_id = manifest.start_run("stage4_labels")

    grid_domain = pd.read_parquet(config.GRID_DOMAIN_PATH)
    grid_domain = grid_domain[grid_domain["in_canada"]].reset_index(drop=True)
    cell_ids = grid_domain["cell_id"].values
    print(f"{len(cell_ids)} in-domain cells")

    ground_truth = pd.read_parquet(config.DATA_DIR / "ground_truth_by_year.parquet")
    ground_truth["rep_date"] = pd.to_datetime(ground_truth["rep_date"]).values.astype("datetime64[D]")

    target_years = sorted(set(config.TRAINING_YEARS + [config.HOLDOUT_YEAR]))
    print(f"Target years: {target_years}")

    # cell -> {year -> array of fire dates}, built once, reused per cell/year lookup
    fire_dates_by_cell_year = {}
    for (cell_id, year), grp in ground_truth.groupby(["cell_id", ground_truth["rep_date"].dt.year]):
        fire_dates_by_cell_year[(cell_id, int(year))] = grp["rep_date"].values.astype("datetime64[D]")

    frames = []
    for year in target_years:
        dates = season_dates(year).astype("datetime64[D]")
        n_dates = len(dates)
        print(f"\nYear {year}: {n_dates} fire-season dates ({config.FIRE_SEASON_START_MD} to {config.FIRE_SEASON_END_MD})")

        cells_with_fire = [c for c in cell_ids if (c, year) in fire_dates_by_cell_year]
        print(f"  {len(cells_with_fire)}/{len(cell_ids)} cells have >=1 ground-truth fire this year")

        # Cells with zero fires this year: all labels are 0, no distance math needed.
        n_empty = len(cell_ids) - len(cells_with_fire)
        empty_block = pd.DataFrame({
            "cell_id": np.repeat([c for c in cell_ids if (c, year) not in fire_dates_by_cell_year], n_dates),
            "date": np.tile(dates, n_empty),
        })
        empty_block["days_to_nearest_fire"] = np.int16(9999)

        # Cells with >=1 fire: real per-cell distance computation.
        active_blocks = []
        for cell_id in cells_with_fire:
            fire_dates = fire_dates_by_cell_year[(cell_id, year)]
            days = labels_for_cell_year(dates, fire_dates)
            active_blocks.append(pd.DataFrame({"cell_id": cell_id, "date": dates, "days_to_nearest_fire": days}))
        active_block = pd.concat(active_blocks, ignore_index=True) if active_blocks else pd.DataFrame(
            columns=["cell_id", "date", "days_to_nearest_fire"]
        )

        year_df = pd.concat([empty_block, active_block], ignore_index=True)
        year_df["year"] = year
        frames.append(year_df)

    out = pd.concat(frames, ignore_index=True)
    for w in config.LABEL_WINDOWS_DAYS:
        out[f"label_w{w}"] = (out["days_to_nearest_fire"] <= w).astype(np.int8)
    out["cell_id"] = out["cell_id"].astype("category")

    print(f"\n{len(out)} total (cell, date) rows across {len(target_years)} years")
    for w in config.LABEL_WINDOWS_DAYS:
        rate = out[f"label_w{w}"].mean()
        print(f"  label_w{w}: {out[f'label_w{w}'].sum()} positive ({rate * 100:.3f}% base rate)")

    primary_col = f"label_w{config.PRIMARY_LABEL_WINDOW}"
    print(f"\nPrimary label: {primary_col}")
    print(out.groupby("year")[primary_col].agg(["sum", "mean"]))

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_DIR / "labels_by_year.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    manifest.record(
        run_id, out_path,
        upstream=[config.GRID_DOMAIN_PATH, config.DATA_DIR / "ground_truth_by_year.parquet"],
        extra={
            "target_years": target_years,
            "label_windows": config.LABEL_WINDOWS_DAYS,
            "primary_window": config.PRIMARY_LABEL_WINDOW,
            "n_rows": len(out),
            "base_rates": {f"w{w}": float(out[f"label_w{w}"].mean()) for w in config.LABEL_WINDOWS_DAYS},
        },
    )
    print(f"Manifest entry recorded: run_id={run_id}")
    return out


if __name__ == "__main__":
    build()
