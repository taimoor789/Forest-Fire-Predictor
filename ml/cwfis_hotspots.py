"""
Fetches CWFIS satellite fire-detection hotspots (the only near-real-time
Canadian fire-occurrence source available -- NFDB, used for training, has
real reporting lag) and attributes them to grid cells for the live
shadow-mode observation period. See docs/PREREGISTRATION.md's 2026-09-14
amendment for the full design.

Source: https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows -- a public,
no-auth WFS endpoint, verified live before this was written. Returns
`hotspots_last24hrs`: a ROLLING 24H WINDOW WITH NO BACKFILL. Any day not
captured is permanently lost -- this is why the workflow runs this 3x/day
rather than once, and why a failed pull must go red rather than being
silently skipped (see the workflow file's comments).

Domain filtering: bbox-prefiltered to the grid bounds for speed only. The
REAL filter is ml.attribution.assign_nearest_cell at
config.GROUND_TRUTH_MAX_ATTRIBUTION_KM -- the same nearest-cell join
Stage 2/3 of the ML rebuild use, not a second, independently-written
"is this in Canada" check via the feed's own agency codes (which would be
exactly the kind of divergent-definition bug this project's shared
attribution module exists to prevent). `agency` is kept as a recorded
column for auditing only, never as a filter.

Run as a module: python3 -m ml.cwfis_hotspots --log-dir <path>
"""

import argparse
import io
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from ml import config, shadow_log
from ml.attribution import assign_nearest_cell, coverage_report

WFS_URL = "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows"
WFS_PARAMS = {
    "service": "WFS",
    "version": "1.0.0",
    "request": "GetFeature",
    "typeName": "public:hotspots_last24hrs",
    "outputFormat": "csv",
}
USER_AGENT = "forest-fire-predictor-shadow-eval/1.0 (github.com/taimoor789/Forest-Fire-Predictor)"

REQUIRED_COLUMNS = {"lat", "lon", "rep_date", "agency"}
# A continent-wide total near zero during fire season is implausible --
# more likely a service exception or empty error page than a real quiet
# day. Verified directly: a real pull returned 849 rows continent-wide.
MIN_PLAUSIBLE_TOTAL_ROWS = 5

RETRY_DELAYS_S = [10, 30, 60]


def _fetch_once() -> requests.Response:
    return requests.get(WFS_URL, params=WFS_PARAMS, headers={"User-Agent": USER_AGENT}, timeout=60)


def fetch_with_retries():
    """Returns (raw_bytes, error_message). error_message is None on success.
    Retries transient failures; does NOT retry a well-formed-but-implausible
    response (that's a `suspect` status, not a network failure)."""
    last_error = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS_S):
        if delay:
            print(f"  retrying in {delay}s (attempt {attempt + 1}/{len(RETRY_DELAYS_S) + 1})...")
            time.sleep(delay)
        try:
            resp = _fetch_once()
        except requests.RequestException as e:
            last_error = f"request exception: {e}"
            print(f"  attempt {attempt + 1} failed: {last_error}")
            continue

        if resp.status_code != 200:
            last_error = f"HTTP {resp.status_code}"
            print(f"  attempt {attempt + 1} failed: {last_error}")
            continue

        body = resp.content
        # GeoServer's classic trap: a service exception comes back as
        # HTTP 200 with an XML body, not a CSV. outputFormat=csv is a
        # request, not a promise -- verified this is a real failure mode
        # worth checking explicitly, not a theoretical one.
        head = body[:500].lstrip()
        if head.startswith(b"<") or b"ServiceException" in body[:2000]:
            last_error = f"HTTP 200 but body looks like an XML error, not CSV: {head[:200]!r}"
            print(f"  attempt {attempt + 1} failed: {last_error}")
            continue

        return body, None

    return None, last_error or "unknown failure"


def validate_and_parse(raw_bytes: bytes):
    """Returns (df, error_message). error_message is None on success."""
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        return None, f"CSV parse failed: {e}"

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return None, f"missing required columns: {missing} (got {list(df.columns)})"

    if len(df) < MIN_PLAUSIBLE_TOTAL_ROWS:
        return None, f"suspect: only {len(df)} continent-wide rows (expected several hundred in fire season)"

    return df, None


def attribute_to_grid(df: pd.DataFrame, grid_cells: pd.DataFrame):
    """Bbox-prefilter for speed, then nearest-cell attribution is the real
    domain filter. Returns (attributed_df, coverage_report_dict)."""
    in_bbox = (
        df["lat"].between(config.GRID_MIN_LAT, config.GRID_MAX_LAT)
        & df["lon"].between(config.GRID_MIN_LON, config.GRID_MAX_LON)
    )
    bbox_df = df[in_bbox].reset_index(drop=True)
    print(f"  bbox prefilter: {len(df)} -> {len(bbox_df)} rows")

    if len(bbox_df) == 0:
        return bbox_df.assign(cell_id=[], dist_km=[]), {
            "n_fires": 0, "n_attributed": 0, "n_dropped": 0, "pct_dropped": 0.0,
        }

    cell_ids, dist_km = assign_nearest_cell(
        bbox_df["lat"].values, bbox_df["lon"].values, grid_cells, max_km=config.GROUND_TRUTH_MAX_ATTRIBUTION_KM
    )
    report = coverage_report(len(bbox_df), cell_ids, dist_km, config.GROUND_TRUTH_MAX_ATTRIBUTION_KM)

    attributed = bbox_df.copy()
    attributed["cell_id"] = cell_ids
    attributed["dist_km"] = dist_km
    attributed = attributed[attributed["cell_id"].notna()].reset_index(drop=True)
    return attributed[["cell_id", "lat", "lon", "rep_date", "agency", "dist_km"]], report


def run(log_dir: str):
    pid = shadow_log.pull_id()
    print(f"=== CWFIS hotspot pull {pid} ===")

    grid_cells = shadow_log.load_grid_cells(log_dir)
    if grid_cells is None:
        print("FATAL: grid_cells.csv not found on shadow-eval-log -- it must be bootstrapped before this runs.")
        sys.exit(1)
    print(f"  {len(grid_cells)} in-domain cells loaded")

    print("Fetching...")
    raw_bytes, fetch_error = fetch_with_retries()

    status = {
        "pull_id": pid,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    if fetch_error:
        status["status"] = "failed"
        status["error"] = fetch_error
        shadow_log.write_hotspot_pull(log_dir, pid, raw_bytes, None, status)
        print(f"FATAL: {fetch_error}")
        _push(log_dir, pid)
        sys.exit(1)  # go red -- no continue-on-error, per docs/PREREGISTRATION.md's exclusion rule

    df, validate_error = validate_and_parse(raw_bytes)
    if validate_error:
        is_suspect = validate_error.startswith("suspect")
        status["status"] = "suspect" if is_suspect else "failed"
        status["error"] = validate_error
        shadow_log.write_hotspot_pull(log_dir, pid, raw_bytes, None, status)
        print(f"FATAL: {validate_error}")
        _push(log_dir, pid)
        sys.exit(1)

    attributed, coverage = attribute_to_grid(df, grid_cells)
    status["status"] = "ok"
    status["n_rows_total_continent"] = len(df)
    status["coverage"] = coverage

    shadow_log.write_hotspot_pull(log_dir, pid, raw_bytes, attributed, status)
    print(f"  wrote {len(attributed)} attributed hotspots, status=ok")
    _push(log_dir, pid)


def _push(log_dir, pid):
    d = shadow_log.hotspots_dir(log_dir)
    paths = [d / f"{pid}.raw.csv.gz", d / f"{pid}.csv", d / f"{pid}.status.json"]
    paths = [p for p in paths if p.exists()]
    shadow_log.commit_and_push(log_dir, paths, f"hotspots: CWFIS pull {pid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()
    run(args.log_dir)
