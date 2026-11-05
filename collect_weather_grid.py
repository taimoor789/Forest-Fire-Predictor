"""
Pulls per-cell weather from Open-Meteo for every grid cell in
data/canada_fire_grid.csv. Kept as a fallback alongside
collect_weather_grid_eccc.py (the primary collector).

Notes that matter for correctness, not just what the code does:
- Pulled at 12:00 local standard time (the hour the FWI system is defined
  on) and precip is `daily.precipitation_sum`, a true 24h total -- not an
  hourly snapshot, which understates the FWI wetting branches.
- Wind requested in km/h directly (`wind_speed_unit=kmh`); FWI expects km/h.
- Output file is written once per day and left alone on later runs the same
  day (`--force` to refetch) -- FWI needs one value per day, not whichever
  hour's snapshot a repeated run happened to catch.
- Optional `openmeteo_api_key` in config.json switches to the paid customer
  endpoint. Without one, the free tier's documented 600 calls/min limit can
  be throttled far lower from a shared egress IP with no clear recovery
  window -- if that recurs, a key is the real fix, not more retry tuning.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

FREE_API_URL = "https://api.open-meteo.com/v1/forecast"
CUSTOMER_API_URL = "https://customer-api.open-meteo.com/v1/forecast"
GRID_FILE = "data/canada_fire_grid.csv"
OUTPUT_DIR = "weather_data"

try:
    with open("config.json") as f:
        API_KEY = (json.load(f).get("openmeteo_api_key") or "").strip()
except FileNotFoundError:
    API_KEY = ""

if API_KEY:
    API_URL = CUSTOMER_API_URL
    BATCH_SIZE = 100
    BATCH_PAUSE = 0.5  # paid plans have no rate limit; still pace requests
                        # modestly rather than firing them with zero gap
    RATE_LIMIT_DELAY = 10
    logger.info("Using Open-Meteo customer API (key configured)")
else:
    API_URL = FREE_API_URL
    # Conservative batch size for Open-Meteo's multi-location comma-separated
    # request format. Empirically the API (or a proxy in front of it) starts
    # rejecting requests with HTTP 414 (URI too long) somewhere between 500
    # and 600 locations in one call; 100 stays well clear of that.
    BATCH_SIZE = 100
    # Open-Meteo's documented free-tier limit is 600 calls/minute, and
    # empirically each location in a multi-location request counts as one
    # call against that budget. 15s between 100-location batches keeps this
    # at 400 locations/min, a real margin under 600 -- though in practice
    # from a shared/sandboxed egress IP even this can get throttled well
    # before the documented ceiling, with no reliable recovery window. See
    # the module docstring: an API key (config.json "openmeteo_api_key")
    # is the real fix if that happens consistently.
    BATCH_PAUSE = 15.0
    RATE_LIMIT_DELAY = 65  # back off past the full window rather than
                            # retrying inside it, which only compounds the 429s

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds, multiplied by attempt number (5xx / timeout)
REQUEST_TIMEOUT = 30

HOURLY_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
DAILY_VARS = "precipitation_sum"
NOON = "T12:00"


def fetch_batch(session, lats, lons, retry=0):
    """Fetch one batch of locations. Returns the parsed JSON list, or None on failure."""
    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "hourly": HOURLY_VARS,
        "daily": DAILY_VARS,
        "wind_speed_unit": "kmh",
        "timezone": "auto",
        "forecast_days": 1,
    }
    if API_KEY:
        params["apikey"] = API_KEY
    try:
        resp = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 200:
            data = resp.json()
            # A single-location request returns one object, not a list
            return data if isinstance(data, list) else [data]

        if resp.status_code == 429:
            if retry < MAX_RETRIES:
                logger.warning(
                    f"Rate limited (per-minute cap). Waiting {RATE_LIMIT_DELAY}s "
                    f"(attempt {retry + 1}/{MAX_RETRIES})"
                )
                time.sleep(RATE_LIMIT_DELAY)
                return fetch_batch(session, lats, lons, retry + 1)
            logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries for a batch of {len(lats)} cells")
            return None

        if resp.status_code >= 500:
            if retry < MAX_RETRIES:
                logger.warning(f"Server error {resp.status_code}. Retrying...")
                time.sleep(RETRY_DELAY)
                return fetch_batch(session, lats, lons, retry + 1)
            logger.error(f"Server error after {MAX_RETRIES} retries for a batch of {len(lats)} cells")
            return None

        logger.error(f"Unexpected status {resp.status_code} for a batch of {len(lats)} cells: {resp.text[:300]}")
        return None

    except requests.Timeout:
        if retry < MAX_RETRIES:
            logger.warning(f"Timeout. Retrying (attempt {retry + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            return fetch_batch(session, lats, lons, retry + 1)
        logger.error(f"Timeout after {MAX_RETRIES} retries for a batch of {len(lats)} cells")
        return None

    except requests.RequestException as e:
        logger.error(f"Request error for a batch of {len(lats)} cells: {e}")
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return fetch_batch(session, lats, lons, retry + 1)
        return None


def extract_noon_values(location):
    """Pull the 12:00-local-standard-time hourly values plus the 24h precip total."""
    hourly = location.get("hourly", {}) or {}
    times = hourly.get("time", []) or []

    idx = None
    for i, t in enumerate(times):
        if t.endswith(NOON):
            idx = i
            break
    if idx is None and times:
        # Shouldn't happen with forecast_days=1, but fall back to the
        # midpoint of the day rather than silently dropping the cell.
        idx = len(times) // 2

    def at(var):
        vals = hourly.get(var)
        if idx is None or not vals or idx >= len(vals):
            return None
        return vals[idx]

    daily = location.get("daily", {}) or {}
    precip_vals = daily.get("precipitation_sum") or []
    precip_24h = precip_vals[0] if precip_vals else None

    return {
        "temperature": at("temperature_2m"),
        "humidity": at("relative_humidity_2m"),
        "wind_speed": at("wind_speed_10m"),
        "pressure": at("surface_pressure"),
        "precip_24h_mm": precip_24h,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Refetch and overwrite today's file if it already exists")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    calgary_tz = ZoneInfo("America/Edmonton")
    now_local = datetime.now(calgary_tz)
    today_str = now_local.strftime("%Y-%m-%d")
    output_path = Path(OUTPUT_DIR) / f"{today_str}.csv"

    if output_path.exists() and not args.force:
        logger.info(f"{output_path} already exists for today; skipping fetch (use --force to refetch)")
        return 0

    grid = pd.read_csv(GRID_FILE, usecols=["lat", "lon", "in_canada"])
    before = len(grid)
    grid = grid[grid["in_canada"]].drop(columns=["in_canada"]).reset_index(drop=True)
    logger.info(f"Filtered grid to in_canada cells: {before} -> {len(grid)}")
    logger.info(f"Fetching weather for {len(grid)} grid cells from Open-Meteo...")

    session = requests.Session()
    rows = []
    n_batches = math.ceil(len(grid) / BATCH_SIZE)
    failed_cells = 0

    for b in range(n_batches):
        batch = grid.iloc[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        results = fetch_batch(session, batch["lat"].tolist(), batch["lon"].tolist())

        if results is None or len(results) != len(batch):
            if results is not None:
                logger.warning(
                    f"Batch {b + 1}/{n_batches}: expected {len(batch)} results, got {len(results)}; "
                    f"skipping this batch"
                )
            failed_cells += len(batch)
            continue

        for (_, cell), location in zip(batch.iterrows(), results):
            values = extract_noon_values(location)
            rows.append({
                "lat": cell["lat"],
                "lon": cell["lon"],
                "date": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                **values,
            })

        if (b + 1) % 20 == 0 or b == n_batches - 1:
            logger.info(f"Fetched batch {b + 1}/{n_batches} ({len(rows)} cells so far)")

        time.sleep(BATCH_PAUSE)

    if not rows:
        logger.error("CRITICAL: No weather data retrieved! Exiting.")
        raise RuntimeError("Failed to retrieve any weather data")

    df_output = pd.DataFrame(rows)
    df_output.to_csv(output_path, index=False)

    logger.info(f"Saved weather for {len(df_output)}/{len(grid)} grid cells to {output_path}")
    if failed_cells:
        logger.warning(f"{failed_cells} cells could not be fetched and are missing from this file")

    return 0


if __name__ == "__main__":
    sys.exit(main())
