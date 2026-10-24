"""
Pulls weather from ECCC's MSC Datamart (dd.weather.gc.ca) instead of
Open-Meteo -- a free, no-key file server with no observed rate limit, unlike
collect_weather_grid.py's Open-Meteo path, which stalled around ~2,000 of
14,952 cells on this network.

Architecture is deliberately bulk, not per-cell: a handful of whole-continent
GRIB2 downloads sampled locally, O(1) requests instead of Open-Meteo's
O(150) batched calls -- that's what avoids the same rate-limit wall.

Two model sources by latitude: HRDPS (2.5km, temp/humidity/wind) + HRDPA
(the analysis counterpart, used for precip since FWI's rain formulas need
actual accumulated rain, not a forecast) below ~70.6N; GDPS (15km, global)
above that, since HRDPS's domain doesn't cover the high Arctic at all.
GDPS is deliberately used rather than falling back to the nearest in-domain
HRDPS cell -- that would reintroduce the station-smearing problem this
migration exists to fix. GDPS has no matching analysis product, so its
cells use GDPS's own forecast precip instead of an HRDPA equivalent.

Each cell picks the forecast lead-hour nearest its local noon (standard
time) from the latest published run -- this buckets 14,952 cells into a
handful of distinct lead-hours (one per timezone present), not one file
per cell.

Output: weather_data/YYYY-MM-DD.csv, same schema as collect_weather_grid.py.
"""

import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import eccodes
from scipy.spatial import cKDTree

from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

BASE_URL = "https://dd.weather.gc.ca"
GRID_FILE = "data/canada_fire_grid.csv"
OUTPUT_DIR = "weather_data"
REQUEST_TIMEOUT = 90  # these are multi-MB whole-continent files

# Stage 12 (ML rebuild): canada_fire_grid.csv now carries an `in_canada`
# column from a real land mask (ml/build_grid_domain.py), vs. the coarse
# province bounding boxes this grid was originally built from -- filtering
# to it cuts the processed grid from 14,952 to 7,537 cells (the other 7,415
# are ocean/US territory the old boxes let through). Filtered here, at read
# time, rather than by physically shrinking the CSV, specifically so this
# is reversible with one flag if the filter ever needs to be backed out.
FILTER_TO_IN_CANADA = True

# HRDPS's continental domain tops out at ~70.61N in practice (verified
# against a real file). Cells at or below this use HRDPS/HRDPA; cells
# above it fall back to GDPS. The grid's own 0.5-degree rows land exactly
# on this boundary (70.5 vs 71.0), so there's no per-cell ambiguity.
HRDPS_LAT_CUTOFF = 70.5

HRDPS_RUN_HOURS = [18, 12, 6, 0]
GDPS_RUN_HOURS = [12, 0]  # GDPS is issued twice daily

# Standard-time (not daylight) UTC offsets by province, used only to pick
# which forecast lead-time hour approximates each cell's local noon. This
# is intentionally coarse: it selects an integer hour from a smoothly
# varying field, so provincial-boundary imprecision doesn't matter the way
# it would for an actual timezone-correctness use case.
PROVINCE_UTC_OFFSET = {
    'BC': -8, 'YT': -8,
    'AB': -7, 'NT': -7,
    'SK': -6, 'MB': -6,
    'ON': -5, 'QC': -5, 'NU': -5,
    'NB': -4, 'NS': -4, 'PE': -4,
    'NL': -3.5,
    'Unknown': -6,
}

# Mirrors FireWeatherProcessor.get_province() in fire_risk.py.
PROVINCE_BOUNDS = {
    'BC': (48.3, -139.1, 60.0, -114.1),
    'AB': (49.0, -120.0, 60.0, -110.0),
    'SK': (49.0, -110.0, 60.0, -101.4),
    'MB': (49.0, -102.0, 60.0, -88.9),
    'ON': (41.0, -95.2, 56.9, -74.3),
    'QC': (45.0, -79.8, 62.6, -57.1),
    'NB': (44.6, -69.1, 48.1, -63.7),
    'NS': (43.4, -66.4, 47.1, -59.7),
    'PE': (45.9, -64.4, 47.1, -62.0),
    'NL': (46.6, -67.8, 60.4, -52.6),
    'YT': (60.0, -141.0, 69.6, -124.0),
    'NT': (60.0, -136.0, 78.8, -102.0),
    'NU': (60.0, -110.0, 83.1, -61.0),
}
PRIORITY_ORDER = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']


def get_province(lat, lon):
    for prov in PRIORITY_ORDER:
        min_lat, min_lon, max_lat, max_lon = PROVINCE_BOUNDS[prov]
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return prov
    return "Unknown"


def list_dir(url):
    """Names of files/subdirectories in an Apache-style HTTP directory listing."""
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return []
    return re.findall(r'href="([^"?][^"]*)"', resp.text)


def find_file(names, must_contain):
    """First name containing every substring in must_contain (order-independent)."""
    for name in names:
        if all(s in name for s in must_contain):
            return name
    return None


def find_latest_run(model_dir_url_fn, run_hours, now_utc, max_lookback=8, is_ready=None):
    """
    model_dir_url_fn(date_str, run_hour) -> directory URL to test.
    Walks backward through run_hours (today, then earlier days) until one
    passes `is_ready(names)` (default: just non-empty). Returns
    (date_str, run_hour, listing) for the most recent run that qualifies.
    Used for HRDPA, where there's no per-cell lead-hour selection (the file
    is always "the latest complete 24h analysis"), unlike the forecast
    sources below.

    A merely non-empty directory isn't always enough: HRDPA's 24h
    accumulation needs the full period's observations ingested first, so a
    just-issued run's directory can exist and list other files while
    APCP-Accum24h specifically isn't published yet -- `is_ready` lets the
    caller check for the exact file it needs, not just "something's there".
    """
    if is_ready is None:
        is_ready = bool

    candidate = now_utc
    tried = 0
    while tried < max_lookback:
        date_str = candidate.strftime("%Y%m%d")
        for run_hour in run_hours:
            run_dt = candidate.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=run_hour)
            if run_dt > now_utc:
                continue
            url = model_dir_url_fn(date_str, run_hour)
            names = list_dir(url)
            if is_ready(names):
                return date_str, run_hour, names
            tried += 1
            if tried >= max_lookback:
                break
        candidate -= timedelta(days=1)
    raise RuntimeError(f"No published run found after {max_lookback} attempts")


def select_run_and_lead(lead_dir_url_fn, run_hours, target_utc_hour, now_utc, max_candidates=12):
    """
    Pick the most recent model run that can reach `target_utc_hour` (today,
    UTC) with a small forward-looking lead, and confirm that specific
    lead-hour directory is actually published.

    Naively computing lead = (target - run_hour) % 24 against only the
    single latest run is wrong whenever a timezone's local-noon UTC hour
    falls *before* that run's start hour: the modulo wraps forward to
    almost a full day later (tomorrow's near-noon) instead of falling back
    to an already-published earlier run that covers today's target hour
    directly with a short lead. This walks candidate runs newest-first and
    only accepts one whose start time is at or before the target time.

    lead_dir_url_fn(date_str, run_hour, lead_hour) -> directory URL to test.
    Returns (date_str, run_hour, lead_hours).
    """
    today0 = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    target_dt = today0 + timedelta(hours=target_utc_hour)

    candidates = []
    for back_days in range(3):
        day0 = today0 - timedelta(days=back_days)
        for rh in sorted(run_hours, reverse=True):
            candidates.append(day0 + timedelta(hours=rh))
    candidates.sort(reverse=True)

    checked = 0
    for run_dt in candidates:
        if run_dt > target_dt or run_dt > now_utc:
            continue  # can't reach the target with a forward-only forecast
        lead = int(round((target_dt - run_dt).total_seconds() / 3600))
        date_str, run_hour = run_dt.strftime("%Y%m%d"), run_dt.hour
        if list_dir(lead_dir_url_fn(date_str, run_hour, lead)):
            return date_str, run_hour, lead
        checked += 1
        if checked >= max_candidates:
            break
    raise RuntimeError(f"No published run found reaching target UTC hour {target_utc_hour}")


def download_and_parse(url, retries=3, delay=5):
    """Download a GRIB2 file to a temp path and parse its first message.
    eccodes needs a real file descriptor (codes_grib_new_from_file rejects
    an in-memory BytesIO), so this downloads to disk rather than to memory.
    Returns (lats, lons, values, units); the temp file is always cleaned up.
    """
    import tempfile
    tmp_path = None
    try:
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
                        tmp.write(resp.content)
                        tmp_path = tmp.name
                    break
                logger.warning(f"HTTP {resp.status_code} for {url}")
            except requests.RequestException as e:
                logger.warning(f"Request error for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        else:
            raise RuntimeError(f"Failed to download {url} after {retries} attempts")

        with open(tmp_path, "rb") as f:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                raise ValueError(f"No GRIB message found in {url}")
            lats = eccodes.codes_get_array(gid, "latitudes")
            lons = eccodes.codes_get_array(gid, "longitudes")
            values = eccodes.codes_get_array(gid, "values")
            try:
                units = eccodes.codes_get(gid, "units")
            except Exception:
                units = "unknown"
            eccodes.codes_release(gid)
        return lats, lons, values, units
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def sample(lats, lons, values, target_lat, target_lon):
    """Nearest-neighbor sample at each target point via a KD-tree."""
    tree = cKDTree(np.column_stack([lats, lons]))
    _, idx = tree.query(np.column_stack([target_lat, target_lon]))
    return values[idx]


def hrdps_dir_url(date_str, run_hour, lead_hour):
    return f"{BASE_URL}/{date_str}/WXO-DD/model_hrdps/continental/2.5km/{run_hour:02d}/{lead_hour:03d}/"


def hrdpa_dir_url(date_str, run_hour):
    return f"{BASE_URL}/{date_str}/WXO-DD/model_hrdpa/2.5km/{run_hour:02d}/"


def gdps_dir_url(date_str, run_hour, lead_hour):
    return f"{BASE_URL}/{date_str}/WXO-DD/model_gdps/15km/{run_hour:02d}/{lead_hour:03d}/"


def target_utc_hour_for_offset(utc_offset):
    """The UTC hour corresponding to local noon (standard time) at this offset."""
    return round(12 - utc_offset) % 24


def fetch_hrdps_fields(date_str, run_hour, lead_hour):
    """Download+parse TMP/RH/WSPD/PRES for one lead hour. Returns dict of (lats,lons,values)."""
    url = hrdps_dir_url(date_str, run_hour, lead_hour)
    names = list_dir(url)
    if not names:
        raise RuntimeError(f"No files listed at {url}")

    # TMP_Sfc/WSPD_Sfc are the post-processed "WEonG" surface product, which
    # (verified) isn't computed at lead 000 (T+0, the initial-conditions
    # step) -- only at lead >= 1. TMP_AGL-2m/WIND_AGL-10m are the
    # underlying raw model-level fields instead: present at every lead
    # hour including 000, and the same quantity (WIND_AGL-10m's GRIB
    # shortName is "10si", identical to WSPD_Sfc's -- confirmed by
    # comparing both at a lead hour where both exist). RH_AGL-2m and
    # PRES_Sfc don't have this gap.
    wanted = {
        "temperature": ["_TMP_AGL-2m_"],
        "humidity": ["_RH_AGL-2m_"],
        "wind_speed": ["_WIND_AGL-10m_"],
        "pressure": ["_PRES_Sfc_"],
    }
    fields = {}
    for key, must_contain in wanted.items():
        fname = find_file(names, must_contain)
        if fname is None:
            raise RuntimeError(f"Could not find a file matching {must_contain} in {url}")
        logger.info(f"  downloading {fname}")
        lats, lons, values, units = download_and_parse(url + fname)
        fields[key] = (lats, lons, values, units)
    return fields


def fetch_gdps_fields(date_str, run_hour, lead_hour):
    """Temp/RH/wind/pressure only -- see fetch_gdps_precip for why
    precipitation is handled separately."""
    url = gdps_dir_url(date_str, run_hour, lead_hour)
    names = list_dir(url)
    if not names:
        raise RuntimeError(f"No files listed at {url}")

    wanted = {
        "temperature": ["_AirTemp_AGL-2m_"],
        "humidity": ["_RelativeHumidity_AGL-2m_"],
        "wind_speed": ["_WindSpeed_AGL-10m_"],
        "pressure": ["_Pressure_Sfc_"],
    }
    fields = {}
    for key, must_contain in wanted.items():
        fname = find_file(names, must_contain)
        if fname is None:
            raise RuntimeError(f"Could not find a file matching {must_contain} in {url}")
        logger.info(f"  downloading {fname}")
        lats, lons, values, units = download_and_parse(url + fname)
        fields[key] = (lats, lons, values, units)
    return fields


def fetch_gdps_precip(date_str, run_hour, lead_hour=24):
    """GDPS only publishes Precip-Accum24h at 24h-multiple lead times
    (verified: absent at 000/003/006/012/018, present at 024) -- unlike
    temp/RH/wind, which publish every hour. So this always uses lead=24
    regardless of which lead hour temp/RH/wind used for local noon,
    mirroring how HRDPA precip is decoupled from HRDPS's lead-hour choice.
    """
    url = gdps_dir_url(date_str, run_hour, lead_hour)
    names = list_dir(url)
    fname = find_file(names, ["_Precip-Accum24h_Sfc_"])
    if fname is None:
        raise RuntimeError(f"No Precip-Accum24h file found in {url}")
    logger.info(f"  downloading {fname}")
    return download_and_parse(url + fname)


def fetch_hrdpa_precip(date_str, run_hour):
    url = hrdpa_dir_url(date_str, run_hour)
    names = list_dir(url)
    if not names:
        raise RuntimeError(f"No files listed at {url}")

    # Prefer the final (non-Prelim) 24h accumulation; fall back to Prelim
    # if the final hasn't been published yet for this run.
    prelim = [n for n in names if "APCP-Accum24h" in n and "Prelim" in n]
    final = [n for n in names if "APCP-Accum24h" in n and "Prelim" not in n]
    fname = final[0] if final else (prelim[0] if prelim else None)
    if fname is None:
        raise RuntimeError(f"No APCP-Accum24h file found in {url}")
    logger.info(f"  downloading {fname}")
    return download_and_parse(url + fname)


def convert_units(key, values, units):
    """Normalize each field to the schema fire_risk.py expects."""
    if key == "temperature":
        return values - 273.15 if "K" in units else values
    if key == "wind_speed":
        return values * 3.6 if "s" in units and "m" in units else values  # m/s -> km/h
    if key == "pressure":
        return values / 100.0 if units == "Pa" else values  # Pa -> hPa
    return values  # humidity (%) and precip (mm) are already in the right units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Refetch and overwrite today's file if it already exists")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    output_path = Path(OUTPUT_DIR) / f"{today_str}.csv"

    if output_path.exists() and not args.force:
        logger.info(f"{output_path} already exists for today; skipping fetch (use --force to refetch)")
        return 0

    grid = pd.read_csv(GRID_FILE, usecols=["lat", "lon", "in_canada"])
    if FILTER_TO_IN_CANADA:
        before = len(grid)
        grid = grid[grid["in_canada"]].drop(columns=["in_canada"]).reset_index(drop=True)
        logger.info(f"Filtered grid to in_canada cells: {before} -> {len(grid)}")
    grid["province"] = [get_province(r.lat, r.lon) for r in grid.itertuples()]
    grid["utc_offset"] = grid["province"].map(PROVINCE_UTC_OFFSET).fillna(PROVINCE_UTC_OFFSET["Unknown"])
    grid["in_hrdps_domain"] = grid["lat"] <= HRDPS_LAT_CUTOFF

    logger.info(f"Grid: {len(grid)} cells, {grid['in_hrdps_domain'].sum()} in HRDPS domain, "
                f"{(~grid['in_hrdps_domain']).sum()} needing GDPS (high Arctic)")

    results = pd.DataFrame(index=grid.index, columns=[
        "temperature", "humidity", "wind_speed", "pressure", "precip_24h_mm"
    ], dtype=float)

    # ---- HRDPS: temperature / humidity / wind / pressure, bucketed by lead hour ----
    # Each UTC-offset group picks its own (run, lead) pair -- not one shared
    # run for the whole domain -- since a zone whose local-noon UTC hour has
    # already passed relative to the very latest run needs to fall back to
    # an earlier run instead (see select_run_and_lead's docstring).
    hrdps_cells = grid[grid["in_hrdps_domain"]]
    if len(hrdps_cells):
        for offset, group in hrdps_cells.groupby("utc_offset"):
            target_hour = target_utc_hour_for_offset(offset)
            date_str, run_hour, lead = select_run_and_lead(
                hrdps_dir_url, HRDPS_RUN_HOURS, target_hour, now_utc
            )
            logger.info(f"HRDPS {date_str} {run_hour:02d}Z lead {lead:03d} for UTC offset {offset} ({len(group)} cells)")
            fields = fetch_hrdps_fields(date_str, run_hour, lead)
            for key, (lats, lons, values, units) in fields.items():
                sampled = sample(lats, lons, values, group["lat"].values, group["lon"].values)
                results.loc[group.index, key] = convert_units(key, sampled, units)

        # ---- HRDPA: precipitation (analysis, not forecast) ----
        # Freshest published analysis, independent of which HRDPS run(s)
        # were used above for temp/RH/wind -- HRDPA's own publication
        # schedule can lag HRDPS's forecast runs.
        hrdpa_date, hrdpa_run, _ = find_latest_run(
            lambda d, r: hrdpa_dir_url(d, r), HRDPS_RUN_HOURS, now_utc,
            is_ready=lambda names: any("APCP-Accum24h" in n for n in names)
        )
        logger.info(f"Using HRDPA {hrdpa_date} {hrdpa_run:02d}Z for precipitation")
        lats, lons, values, units = fetch_hrdpa_precip(hrdpa_date, hrdpa_run)
        sampled = sample(lats, lons, values, hrdps_cells["lat"].values, hrdps_cells["lon"].values)
        results.loc[hrdps_cells.index, "precip_24h_mm"] = convert_units("precip_24h_mm", sampled, units)

    # ---- GDPS: temp/RH/wind/pressure, for cells above HRDPS's domain (high Arctic) ----
    gdps_cells = grid[~grid["in_hrdps_domain"]]
    if len(gdps_cells):
        for offset, group in gdps_cells.groupby("utc_offset"):
            target_hour = target_utc_hour_for_offset(offset)
            date_str, run_hour, lead = select_run_and_lead(
                gdps_dir_url, GDPS_RUN_HOURS, target_hour, now_utc
            )
            logger.info(f"GDPS {date_str} {run_hour:02d}Z lead {lead:03d} for UTC offset {offset} ({len(group)} cells)")
            fields = fetch_gdps_fields(date_str, run_hour, lead)
            for key, (lats, lons, values, units) in fields.items():
                sampled = sample(lats, lons, values, group["lat"].values, group["lon"].values)
                results.loc[group.index, key] = convert_units(key, sampled, units)

        # ---- GDPS precip: only published at 24h-multiple lead times, so
        # picked independently of the per-offset noon lead above (same
        # reasoning as HRDPA vs HRDPS). No GDPS analysis-product
        # equivalent to HRDPA is available, so this is a 24h-accumulated
        # forecast standing in for an observation, for these ~4,300
        # high-Arctic cells only.
        gdps_precip_date, gdps_precip_run, _ = find_latest_run(
            lambda d, r: gdps_dir_url(d, r, 24), GDPS_RUN_HOURS, now_utc,
            is_ready=lambda names: any("_Precip-Accum24h_Sfc_" in n for n in names)
        )
        logger.info(f"Using GDPS {gdps_precip_date} {gdps_precip_run:02d}Z lead 024 for high-Arctic precipitation")
        lats, lons, values, units = fetch_gdps_precip(gdps_precip_date, gdps_precip_run)
        sampled = sample(lats, lons, values, gdps_cells["lat"].values, gdps_cells["lon"].values)
        results.loc[gdps_cells.index, "precip_24h_mm"] = convert_units("precip_24h_mm", sampled, units)

    missing = results.isna().any(axis=1).sum()
    if missing:
        logger.warning(f"{missing} cells missing one or more fields after sampling")

    df_output = pd.DataFrame({
        "lat": grid["lat"],
        "lon": grid["lon"],
        "date": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": results["temperature"],
        "humidity": results["humidity"],
        "wind_speed": results["wind_speed"],
        "pressure": results["pressure"],
        "precip_24h_mm": results["precip_24h_mm"],
    })
    df_output.to_csv(output_path, index=False)
    logger.info(f"Saved weather for {len(df_output)}/{len(grid)} grid cells to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
