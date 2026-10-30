"""
Stage 6: FWI replay via fire_risk.py's actual production code -- never
reimplemented. This is the structural fix for the original bug where the
backtest notebook hand-transcribed the FWI formulas once and silently kept
using the pre-fix versions years after production moved on.

Per in-domain cell, per target year: config.REPLAY_WARMUP_DAYS days of real
ERA5-Land weather before the fire season starts are replayed once through
FireWeatherProcessor.bootstrap_from_window() -- exactly the call production
makes on a cold start, seeded from get_seasonal_initial_codes() for the
first available day's month, never a flat spring default -- to reach a
state that's past its bootstrap transient before the modelling window
begins. Only the resulting (ffmc, dmc, dc, recent) state is kept from the
warmup; only season-window days are written to the output parquet (the plan
scopes it that way -- warmup exists purely to seed accumulation state, not
to be modelling rows).

From there the season is stepped forward day by day with advance_one_day(),
honoring GAP_REINIT_DAYS=3 exactly as process_all_locations() does
(fire_risk.py:662-667): a gap this long since the last processed day forces
a fresh seasonal reinit instead of continuing from a stale code. ERA5-Land
has no missing days inside a requested range, so this branch is expected to
never fire in practice -- it's here for fidelity to the production
sequencing logic, not because real gaps are anticipated.

dc_trend_7d / bui_trend_7d come straight out of advance_one_day()'s own
8-entry `recent` ring buffer -- never recomputed via a groupby().shift(7),
which is what silently breaks across gap/reinit boundaries.

Run as a module from the repo root: python3 -m ml.replay_fwi
"""

import multiprocessing as mp
from datetime import date

import pandas as pd

from ml import config, manifest
from ml.weather import adapter
from fire_risk import FireWeatherProcessor, GAP_REINIT_DAYS

N_WORKERS = min(4, mp.cpu_count())  # conservative -- this session's sandbox
                                      # has repeatedly hit memory-pressure kills
                                      # on wide parallelism


def replay_one_cell(processor: FireWeatherProcessor, cell_id: str, cell_weather: pd.DataFrame,
                     season_start: pd.Timestamp) -> pd.DataFrame:
    """cell_weather: one cell's full replay-year weather (warmup + season),
    with the adapter's 5 columns plus `date`. Returns one row per
    season-window date."""
    cell_weather = cell_weather.sort_values("date").reset_index(drop=True)
    cell_weather["file_date"] = cell_weather["date"].dt.strftime("%Y-%m-%d")

    warmup = cell_weather[cell_weather["date"] < season_start]
    season = cell_weather[cell_weather["date"] >= season_start].reset_index(drop=True)
    if season.empty:
        return pd.DataFrame()

    ffmc, dmc, dc, recent = processor.bootstrap_from_window(warmup, season["date"].iloc[0])
    last_date = warmup["date"].max() if len(warmup) else None

    rows = []
    for _, row in season.iterrows():
        current_date = row["date"]
        if last_date is not None:
            gap_days = (current_date.normalize() - last_date.normalize()).days
            if gap_days > GAP_REINIT_DAYS:
                seasonal = processor.fwi_calculator.get_seasonal_initial_codes(current_date.month)
                ffmc, dmc, dc, recent = seasonal["ffmc"], seasonal["dmc"], seasonal["dc"], []

        ffmc, dmc, dc, recent, result = processor.advance_one_day(ffmc, dmc, dc, recent, row, current_date)
        rows.append({"cell_id": cell_id, "date": current_date, **result})
        last_date = current_date

    return pd.DataFrame(rows)


def _process_chunk(args):
    weather_chunk, season_start = args
    processor = FireWeatherProcessor()  # one per worker, never shared across processes
    frames = [
        replay_one_cell(processor, cell_id, group, season_start)
        for cell_id, group in weather_chunk.groupby("cell_id", observed=True)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def replay_year(year: int, grid_domain: pd.DataFrame) -> pd.DataFrame:
    print(f"\n=== Replaying {year} ===")
    weather = adapter.extract_year(year, grid_domain)
    season_start = pd.Timestamp(date(year, *config.FIRE_SEASON_START_MD))
    print(f"  {weather['cell_id'].nunique()} cells, {len(weather)} weather rows, "
          f"season starts {season_start.date()}")

    cell_ids = weather["cell_id"].unique()
    chunks = [cell_ids[i::N_WORKERS] for i in range(N_WORKERS)]
    chunk_args = [(weather[weather["cell_id"].isin(c)], season_start) for c in chunks]

    if N_WORKERS > 1:
        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(_process_chunk, chunk_args)
    else:
        results = [_process_chunk(a) for a in chunk_args]

    out = pd.concat(results, ignore_index=True)
    print(f"  {len(out)} replayed (cell, date) rows")
    return out


def replay_all():
    run_id = manifest.start_run("stage6_replay")

    grid_domain = pd.read_parquet(config.GRID_DOMAIN_PATH)
    grid_domain = grid_domain[grid_domain["in_canada"]].reset_index(drop=True)
    print(f"{len(grid_domain)} in-domain cells")

    target_years = sorted(set(config.TRAINING_YEARS + [config.HOLDOUT_YEAR]))
    for year in target_years:
        out_path = config.DATA_DIR / f"fwi_replay_{year}.parquet"
        if out_path.exists():
            print(f"\n{out_path} already exists, skipping {year}")
            continue

        out = replay_year(year, grid_domain)
        out["cell_id"] = out["cell_id"].astype("category")

        for col in ("ffmc", "dmc", "dc"):
            print(f"  {col}: mean={out[col].mean():.2f} median={out[col].median():.2f} "
                  f"min={out[col].min():.2f} max={out[col].max():.2f}")

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        print(f"  Wrote {out_path}")

        manifest.record(
            run_id, out_path,
            # the raw ERA5 .grib files aren't manifest-tracked (see
            # docs/DATA_PROVENANCE.md -- they're external source downloads,
            # not a derived artifact record() can chain to), so only the
            # grid domain is recorded as an upstream dependency here.
            upstream=[config.GRID_DOMAIN_PATH],
            extra={
                "year": year,
                "n_rows": len(out),
                "n_cells": out["cell_id"].nunique(),
                "warmup_days": config.REPLAY_WARMUP_DAYS,
                "gap_reinit_days": GAP_REINIT_DAYS,
                "ffmc_mean": float(out["ffmc"].mean()),
                "dmc_mean": float(out["dmc"].mean()),
                "dc_mean": float(out["dc"].mean()),
            },
        )
        print(f"  Manifest entry recorded: run_id={run_id}")


if __name__ == "__main__":
    replay_all()
