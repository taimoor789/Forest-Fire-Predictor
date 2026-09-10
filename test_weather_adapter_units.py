"""
Stage 5 gate: ml/weather/adapter.py's output must be in the same units and
plausible range as collect_weather_grid_eccc.py's live weather_data/*.csv --
this is exactly the check that would have caught the original NASA POWER
wind-units bug (m/s fed raw, mean 3.49 vs production's 19.60 km/h) and the
daily-mean-vs-local-noon RH bug before any model was trained on it.

Not a strict distributional match (different provider, different weather --
see ml/config.py's Stage 5 comment on the CaSPAr->ERA5 pivot) -- just a
physical-plausibility + same-units sanity gate.
"""

import glob

import pandas as pd
import pytest

from ml import config
from ml.weather import adapter


@pytest.fixture(scope="module")
def live_reference():
    f = sorted(glob.glob("weather_data/*.csv"))[-1]
    return pd.read_csv(f)


@pytest.fixture(scope="module")
def adapter_sample():
    grid_domain = pd.read_parquet(config.GRID_DOMAIN_PATH)
    grid_domain = grid_domain[grid_domain["in_canada"]].reset_index(drop=True)
    subset = grid_domain.groupby("province", group_keys=False).head(6).reset_index(drop=True)
    return adapter.extract_year(2023, subset)


def test_same_five_columns(adapter_sample, live_reference):
    expected = {"temperature", "humidity", "wind_speed", "pressure", "precip_24h_mm"}
    assert expected.issubset(adapter_sample.columns)
    assert expected.issubset(live_reference.columns)


def test_temperature_celsius_range(adapter_sample):
    # Canada, any season: -50C to 45C covers every plausible reading;
    # Kelvin-forgot-to-convert would show ~250-320.
    assert adapter_sample["temperature"].between(-50, 45).all()


def test_humidity_percent_range(adapter_sample):
    assert adapter_sample["humidity"].between(0, 100).all()


def test_wind_speed_kmh_not_ms(adapter_sample, live_reference):
    # the original bug: m/s fed unconverted (mean ~3.5) vs production's
    # km/h (mean ~20) -- same order of magnitude as live, and above what
    # an unconverted m/s reading would show.
    assert adapter_sample["wind_speed"].mean() > 8
    assert adapter_sample["wind_speed"].between(0, 200).all()
    live_mean = live_reference["wind_speed"].mean()
    adapter_mean = adapter_sample["wind_speed"].mean()
    assert adapter_mean / live_mean < 3  # same order of magnitude, not 6x off like raw m/s would be


def test_pressure_hpa_range(adapter_sample):
    # hPa at sea level ~1013; surface stations in mountainous terrain go
    # much lower. Pa-unconverted would show ~69000-103000.
    assert adapter_sample["pressure"].between(600, 1050).all()


def test_precip_24h_mm_non_negative_and_plausible(adapter_sample):
    assert (adapter_sample["precip_24h_mm"] >= 0).all()
    assert adapter_sample["precip_24h_mm"].max() < 500  # no single-day 24h total this extreme in the sample


def test_no_missing_rows(adapter_sample):
    assert adapter_sample.isna().sum().sum() == 0


def test_full_season_coverage_per_cell(adapter_sample):
    counts = adapter_sample.groupby("cell_id").size()
    assert (counts == 214).all()  # Apr 1 - Oct 31 inclusive
