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

Two request groups per year, each split into date-range chunks of at most
CHUNK_MAX_DAYS days. The nominal CDS API limit is 12,000 fields/request, but
CDS also enforces an undocumented "cost" limit that rejected a 289-day/
7,225-field hourly request outright ("cost limits exceeded") while a
214-day/5,350-field request the same shape had worked fine earlier in this
project -- so chunks are sized well under that working precedent rather than
the nominal limit:
  - "hourly" group: temperature, dewpoint (for RH), wind u/v, surface
    pressure, at only the ~5 distinct UTC hours that correspond to local
    noon somewhere in Canada (computed via
    collect_weather_grid_eccc.target_utc_hour_for_offset -- the same
    function the live collector uses, not a re-derived list, so the two
    can't quietly disagree on what "local noon" means). Covers
    config.REPLAY_WARMUP_DAYS days before the fire season starts, not just
    the season itself, so Stage 6's replay has real weather to burn DC/BUI
    in on rather than starting the modelling window at a flat seasonal seed.
  - "precip" group: total_precipitation at all 24 hourly forecast-steps of
    each day's accumulation run -- needed to reconstruct a rolling 24h sum
    (see ml/weather/adapter.py), not a single reading. Same warmup window
    plus one extra lead-in day so even the first warmup day gets a full
    24h trailing total.

Uses the "date" request field (explicit YYYY-MM-DD strings), not separate
year/month/day lists -- those get cross-producted by CDS independently
(verified against real data: an earlier version of this file requesting
month=[03..10] and a day-of-month union silently expanded the request to
~all of March-October instead of the intended range).

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

CHUNK_MAX_DAYS = 150  # comfortably under the 214-day request size already verified to work


def season_day_range(year: int, lead_in_days: int = 1):
    """Fire season dates for `year`, plus lead_in_days before the start.
    Used with two different lead-ins: config.REPLAY_WARMUP_DAYS for the
    hourly pull (so Stage 6's replay has real weather to burn in DC/BUI
    before the modelling window starts, rather than starting the season
    right at a flat seasonal seed) and REPLAY_WARMUP_DAYS+1 for precip
    (one extra day so even the first warmup day has a full 24h trailing
    window to sum)."""
    start = date(year, *config.FIRE_SEASON_START_MD) - timedelta(days=lead_in_days)
    end = date(year, *config.FIRE_SEASON_END_MD)
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def _date_strings(days):
    """CDS's year/month/day list fields are independent and get cross-
    producted (verified: requesting month=[03,04] and the day-of-month
    union across those months pulls in every valid March-1..April-30 date,
    not just the intended days -- harmless for the already-downloaded
    2019-2023 files, since it only adds extra lead-in, but avoided going
    forward with the "date" field, which takes explicit YYYY-MM-DD strings."""
    return [d.isoformat() for d in days]


def _chunk_days(days, max_days=CHUNK_MAX_DAYS):
    return [days[i:i + max_days] for i in range(0, len(days), max_days)]


def chunk_paths(kind: str, year: int, n_chunks: int):
    """kind: 'hourly' or 'precip'. Filenames adapter.py's loader globs for."""
    return [RAW_DIR / f"{kind}_{year}_part{i}.grib" for i in range(n_chunks)]


def fetch_hourly(client: cdsapi.Client, year: int):
    """temp/dewpoint/wind/pressure at the ~5 local-noon UTC hours, chunked
    into <=CHUNK_MAX_DAYS-day requests (no precip lead-in needed here)."""
    days = season_day_range(year, lead_in_days=config.REPLAY_WARMUP_DAYS)
    chunks = _chunk_days(days)
    paths = chunk_paths("hourly", year, len(chunks))

    for chunk_days, out_path in zip(chunks, paths):
        if out_path.exists():
            print(f"  {out_path} already exists, skipping")
            continue
        print(f"  Requesting hourly vars for {year} [{chunk_days[0]}..{chunk_days[-1]}]: "
              f"{len(chunk_days)} days x {len(TARGET_UTC_HOURS)} hours x {len(HOURLY_VARIABLES)} vars")
        r = client.retrieve("reanalysis-era5-land", {
            "variable": HOURLY_VARIABLES,
            "date": _date_strings(chunk_days),
            "time": [f"{h:02d}:00" for h in TARGET_UTC_HOURS],
            "area": AREA,
            "grid": GRID,
            "data_format": "grib",
        })
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        r.download(str(out_path))
        print(f"  Wrote {out_path}")
    return paths


def fetch_precip(client: cdsapi.Client, year: int):
    """total_precipitation at all 24 hours, chunked into <=CHUNK_MAX_DAYS-day
    requests, with a 1-day lead-in before the warmup window starts."""
    days = season_day_range(year, lead_in_days=config.REPLAY_WARMUP_DAYS + 1)
    chunks = _chunk_days(days)
    paths = chunk_paths("precip", year, len(chunks))

    for chunk_days, out_path in zip(chunks, paths):
        if out_path.exists():
            print(f"  {out_path} already exists, skipping")
            continue
        print(f"  Requesting precip for {year} [{chunk_days[0]}..{chunk_days[-1]}]: "
              f"{len(chunk_days)} days x 24 hours = {len(chunk_days) * 24} fields")
        r = client.retrieve("reanalysis-era5-land", {
            "variable": [PRECIP_VARIABLE],
            "date": _date_strings(chunk_days),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "grid": GRID,
            "data_format": "grib",
        })
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        r.download(str(out_path))
        print(f"  Wrote {out_path}")
    return paths


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
