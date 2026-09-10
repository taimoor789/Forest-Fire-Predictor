"""
Stage 6 anti-drift guard for ml/replay_fwi.py.

What this does and doesn't test: it does NOT compare ERA5-Land-driven replay
output against live production's persisted fwi_state.json codes -- those are
driven by different weather providers (ERA5-Land vs. ECCC HRDPS/GDPS/HRDPA,
see ml/config.py's Stage 5 comment) and are expected to diverge. There is
also no historical fwi_state.json snapshot in the repo to replay against --
the file holds only the latest state, overwritten each run.

What it DOES test, using real rows from a live weather_data/*.csv: that
ml.replay_fwi.replay_one_cell()'s orchestration (bootstrap_from_window() +
advance_one_day() + the GAP_REINIT_DAYS check) reduces to EXACTLY the same
result as calling those same fire_risk.py methods directly -- catching any
transcription bug (wrong column, wrong date type, an off-by-one in the gap
check) introduced into replay_fwi.py without touching fire_risk.py itself.
This is the permanent guard against the exact failure mode that invalidated
the original backtest notebook: FWI logic re-derived by hand, once, that
silently drifted from production.
"""

import glob

import pandas as pd
import pytest

from fire_risk import FireWeatherProcessor
from ml.replay_fwi import replay_one_cell


def _latest_weather_rows(n=5):
    f = sorted(glob.glob("weather_data/*.csv"))[-1]
    df = pd.read_csv(f)
    return df.sample(n=n, random_state=42).reset_index(drop=True)


def test_single_day_bootstrap_matches_direct_calls():
    """No warmup history (bootstrap from an empty window) is bootstrap_from_
    window's own documented fallback -- exactly what a first-ever run for a
    cell looks like in production too."""
    rows = _latest_weather_rows(5)

    for _, row in rows.iterrows():
        current_date = pd.to_datetime(row.get("date", "2023-07-15"))
        cell_id = f"{row['lat']:.4f}_{row['lon']:.4f}"

        weather = pd.DataFrame([{
            "cell_id": cell_id,
            "date": current_date,
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "wind_speed": row["wind_speed"],
            "pressure": row["pressure"],
            "precip_24h_mm": row["precip_24h_mm"],
        }])

        replay_result = replay_one_cell(FireWeatherProcessor(), cell_id, weather, current_date)
        assert len(replay_result) == 1
        replay_result = replay_result.iloc[0]

        direct_processor = FireWeatherProcessor()
        ffmc0, dmc0, dc0, recent0 = direct_processor.bootstrap_from_window(
            pd.DataFrame(columns=["file_date"]), current_date
        )
        _, _, _, _, expected = direct_processor.advance_one_day(ffmc0, dmc0, dc0, recent0, row, current_date)

        for key in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr", "dc_trend_7d", "bui_trend_7d"):
            assert replay_result[key] == pytest.approx(expected[key], abs=1e-9), (
                f"{key} mismatch: replay={replay_result[key]} vs direct={expected[key]}"
            )


def test_gap_reinit_triggers_seasonal_reset():
    """A gap longer than GAP_REINIT_DAYS between two weather rows for the
    same cell must force a fresh seasonal reinit, exactly as
    process_all_locations() does (fire_risk.py:662-667) -- not a silent
    continuation from stale codes."""
    rows = _latest_weather_rows(1)
    row = rows.iloc[0]
    cell_id = f"{row['lat']:.4f}_{row['lon']:.4f}"

    day1 = pd.Timestamp("2023-05-01")
    day2 = pd.Timestamp("2023-05-10")  # 9-day gap, > GAP_REINIT_DAYS=3

    weather = pd.DataFrame([
        {"cell_id": cell_id, "date": day1, "temperature": row["temperature"],
         "humidity": row["humidity"], "wind_speed": row["wind_speed"],
         "pressure": row["pressure"], "precip_24h_mm": row["precip_24h_mm"]},
        {"cell_id": cell_id, "date": day2, "temperature": row["temperature"],
         "humidity": row["humidity"], "wind_speed": row["wind_speed"],
         "pressure": row["pressure"], "precip_24h_mm": row["precip_24h_mm"]},
    ])

    replay_result = replay_one_cell(FireWeatherProcessor(), cell_id, weather, day1)
    assert len(replay_result) == 2
    day2_replayed = replay_result.iloc[1]

    # Expected: day2 is a fresh seasonal reinit (same weather row as day1,
    # applied to a fresh seasonal seed for day2's month) -- NOT a
    # continuation from day1's advanced codes.
    direct_processor = FireWeatherProcessor()
    seasonal = direct_processor.fwi_calculator.get_seasonal_initial_codes(day2.month)
    _, _, _, _, expected = direct_processor.advance_one_day(
        seasonal["ffmc"], seasonal["dmc"], seasonal["dc"], [], row, day2
    )

    for key in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"):
        assert day2_replayed[key] == pytest.approx(expected[key], abs=1e-9), (
            f"{key} mismatch after gap reinit: replay={day2_replayed[key]} vs expected={expected[key]}"
        )
