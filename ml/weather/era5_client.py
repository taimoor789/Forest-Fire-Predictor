"""
Stage 5: historical weather acquisition via ERA5-Land (Copernicus CDS API).

Used instead of the originally-planned CaSPAr HRDPS/GDPS archive -- CaSPAr
(caspar-data.ca) was unreachable for 10+ hours from two independent
networks with no outage information available, so the fallback was used
instead (see the session record for the decision). ERA5-Land is a different
model family than the live ECCC collector (collect_weather_grid_eccc.py),
so there's a residual, measurable train/serve provider bias this ERA5 path
doesn't erase the way CaSPAr would have -- worth remembering when
interpreting Stage 9's results.

Two request groups per year, each comfortably under the CDS API's 12,000
fields/request limit (verified: raising it in one request even at ~5,300
fields worked fine):
  - "hourly" group: temperature, dewpoint (for RH), wind u/v, surface
    pressure, at only the ~5 distinct UTC hours that correspond to local
    noon somewhere in Canada (computed via
    collect_weather_grid_eccc.target_utc_hour_for_offset -- the same
    function the live collector uses, not a re-derived list, so the two
    can't quietly disagree on what "local noon" means).
  - "precip" group: total_precipitation at all 24 hours -- needed as a
    rolling 24h sum (see ml/weather/adapter.py), not a single reading.

Regridded server-side to the project's exact 0.5-degree grid (grid=[0.5,
0.5]) -- verified this lines up exactly with the grid canada_fire_grid.csv
already uses (41-83N, -141 to -52W), so no further spatial interpolation
is needed on the Python side.

Run as a module from the repo root: python3 -m ml.weather.era5_client
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import cdsapi

from ml import config
from collect_weather_grid_eccc import PROVINCE_UTC_OFFSET, target_utc_hour_for_offset

RAW_DIR = config.REFERENCE_DIR / "era5"

HOURLY_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]
PRECIP_VARIABLE = "total_precipitation"

TARGET_UTC_HOURS = sorted(set(target_utc_hour_for_offset(off) for off in PROVINCE_UTC_OFFSET.values()))

AREA = [config.GRID_MAX_LAT, config.GRID_MIN_LON, config.GRID_MIN_LAT, config.GRID_MAX_LON]  # N, W, S, E
GRID = [config.GRID_STEP_DEG, config.GRID_STEP_DEG]


def season_day_range(year: int, lead_in_days: int = 1):
    """Fire season dates for `year`, plus lead_in_days before the start --
    the precip pull needs this so the FIRST season day already has a full
    24h trailing window to sum."""
    start = date(year, *config.FIRE_SEASON_START_MD) - timedelta(days=lead_in_days)
    end = date(year, *config.FIRE_SEASON_END_MD)
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def _months_days(days):
    """CDS wants separate month/day lists, not a date range -- group by month."""
    by_month = {}
    for d in days:
        by_month.setdefault(f"{d.month:02d}", set()).add(f"{d.day:02d}")
    return by_month


def fetch_hourly(client: cdsapi.Client, year: int):
    """temp/dewpoint/wind/pressure at the ~5 local-noon UTC hours, one
    request for the whole season (no precip lead-in needed here)."""
    out_path = RAW_DIR / f"hourly_{year}.grib"
    if out_path.exists():
        print(f"  {out_path} already exists, skipping")
        return out_path

    days = season_day_range(year, lead_in_days=0)
    by_month = _months_days(days)
    print(f"  Requesting hourly vars for {year}: {sum(len(v) for v in by_month.values())} days x "
          f"{len(TARGET_UTC_HOURS)} hours x {len(HOURLY_VARIABLES)} vars")

    r = client.retrieve("reanalysis-era5-land", {
        "variable": HOURLY_VARIABLES,
        "year": str(year),
        "month": sorted(by_month.keys()),
        "day": sorted({d for days_set in by_month.values() for d in days_set}),
        "time": [f"{h:02d}:00" for h in TARGET_UTC_HOURS],
        "area": AREA,
        "grid": GRID,
        "data_format": "grib",
    })
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    r.download(str(out_path))
    print(f"  Wrote {out_path}")
    return out_path


def fetch_precip(client: cdsapi.Client, year: int):
    """total_precipitation at all 24 hours, with a 1-day lead-in, one
    request for the whole season."""
    out_path = RAW_DIR / f"precip_{year}.grib"
    if out_path.exists():
        print(f"  {out_path} already exists, skipping")
        return out_path

    days = season_day_range(year, lead_in_days=1)
    by_month = _months_days(days)
    n_days = sum(len(v) for v in by_month.values())
    print(f"  Requesting precip for {year}: {n_days} days x 24 hours = {n_days * 24} fields")

    r = client.retrieve("reanalysis-era5-land", {
        "variable": [PRECIP_VARIABLE],
        "year": str(year),
        "month": sorted(by_month.keys()),
        "day": sorted({d for days_set in by_month.values() for d in days_set}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "grid": GRID,
        "data_format": "grib",
    })
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    r.download(str(out_path))
    print(f"  Wrote {out_path}")
    return out_path


def fetch_all():
    client = cdsapi.Client()
    target_years = sorted(set(config.TRAINING_YEARS + [config.HOLDOUT_YEAR]))
    print(f"Fetching ERA5-Land for years: {target_years}")
    print(f"Target UTC hours (local noon somewhere in Canada): {TARGET_UTC_HOURS}")

    for year in target_years:
        print(f"\n=== {year} ===")
        fetch_hourly(client, year)
        fetch_precip(client, year)

    print("\nAll requests complete.")


if __name__ == "__main__":
    sys.exit(fetch_all())
