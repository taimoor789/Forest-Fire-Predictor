"""
Shared schema/IO for the shadow-eval-log branch -- one definition of paths
and the status-record format, imported by the fetch/snapshot/report scripts
so they can't silently diverge on schema. Same reasoning ml/attribution.py
already established as this project's pattern for shared primitives.

`log_dir` throughout is the path to a SEPARATE git checkout of the
shadow-eval-log branch (in CI: a second `actions/checkout@v4` step with its
own `path:`, not a subdirectory of the main repo's own git tree) -- every
git operation here is explicitly scoped to it via `git -C <log_dir>`, never
touching the calling script's own repo/branch context.

Layout on shadow-eval-log:
  grid_cells.csv                  -- {cell_id, lat, lon}, written once, the
                                      CI-safe substitute for
                                      data/grid_domain_v1.parquet (not
                                      tracked in git, unavailable in CI)
  hotspots/<pull_id>.raw.csv.gz    -- unbackfillable raw CWFIS pull
  hotspots/<pull_id>.csv           -- attributed to cell_id, deduped
  hotspots/<pull_id>.status.json   -- fetch status (ok/suspect/failed)
  predictions/<date>.csv           -- per-cell FWI/ML tiers that day
  predictions/<date>.meta.json     -- distribution summary + T1 check
"""

from __future__ import annotations  # local dev venv is Python 3.9; `X | None` hints need this there

import gzip
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GIT_USER_NAME = "github-actions[bot]"
GIT_USER_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"


def pull_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def hotspots_dir(log_dir) -> Path:
    return Path(log_dir) / "hotspots"


def predictions_dir(log_dir) -> Path:
    return Path(log_dir) / "predictions"


def grid_cells_path(log_dir) -> Path:
    return Path(log_dir) / "grid_cells.csv"


def load_grid_cells(log_dir) -> pd.DataFrame | None:
    p = grid_cells_path(log_dir)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"cell_id": str})


def save_grid_cells(log_dir, df: pd.DataFrame):
    grid_cells_path(log_dir).parent.mkdir(parents=True, exist_ok=True)
    df[["cell_id", "lat", "lon"]].to_csv(grid_cells_path(log_dir), index=False)


def derive_grid_cells_from_predictions(fwi_predictions_path) -> pd.DataFrame:
    """data/grid_domain_v1.parquet is not tracked in git and unavailable in
    CI -- this is the CI-safe substitute. fwi_predictions.json doesn't carry
    cell_id directly, so it's derived with the exact same format
    FireWeatherProcessor._cell_key (fire_risk.py) and ml.config.cell_id()
    both use: f"{lat:.4f}_{lon:.4f}". Bootstrapped once (see the session
    record for why it's done this way rather than by either scheduled
    workflow), asserted-consistent by shadow_snapshot.py on every later run."""
    with open(fwi_predictions_path) as f:
        preds = json.load(f)
    rows = [{"lat": r["lat"], "lon": r["lon"]} for r in preds["data"]]
    df = pd.DataFrame(rows).drop_duplicates()
    df["cell_id"] = df.apply(lambda r: f"{r['lat']:.4f}_{r['lon']:.4f}", axis=1)
    return df[["cell_id", "lat", "lon"]].sort_values("cell_id").reset_index(drop=True)


def write_hotspot_pull(log_dir, pid: str, raw_bytes: bytes, attributed_df: pd.DataFrame | None, status: dict):
    """attributed_df is None when the pull failed outright (no CSV to write,
    only raw bytes if any were received, plus the status record)."""
    d = hotspots_dir(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    if raw_bytes:
        with gzip.open(d / f"{pid}.raw.csv.gz", "wb") as f:
            f.write(raw_bytes)
    if attributed_df is not None:
        attributed_df.to_csv(d / f"{pid}.csv", index=False)
    with open(d / f"{pid}.status.json", "w") as f:
        json.dump(status, f, indent=2)


def read_all_hotspots(log_dir) -> pd.DataFrame:
    """Concatenates every attributed hotspot CSV. Does not dedupe across
    pulls -- callers needing unique (cell_id, date) positives should do
    that themselves, since what counts as a duplicate depends on the
    analysis (e.g. ml/shadow_report.py dedupes on (cell_id, detection_date)
    when counting distinct fire-days, but a status audit might want every
    raw pull visible)."""
    frames = []
    d = hotspots_dir(log_dir)
    if d.exists():
        for p in sorted(d.glob("*.csv")):
            frames.append(pd.read_csv(p, dtype={"cell_id": str}))
    if not frames:
        return pd.DataFrame(columns=["cell_id", "lat", "lon", "rep_date", "agency", "dist_km", "pull_id"])
    return pd.concat(frames, ignore_index=True)


def read_all_hotspot_statuses(log_dir) -> pd.DataFrame:
    d = hotspots_dir(log_dir)
    rows = []
    if d.exists():
        for p in sorted(d.glob("*.status.json")):
            with open(p) as f:
                rows.append(json.load(f))
    return pd.DataFrame(rows)


def read_all_predictions(log_dir) -> pd.DataFrame:
    frames = []
    d = predictions_dir(log_dir)
    if d.exists():
        for p in sorted(d.glob("*.csv")):
            df = pd.read_csv(p, dtype={"cell_id": str})
            df["snapshot_date"] = p.stem
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["cell_id", "lat", "lon", "fwi", "danger_class",
                                       "ml_risk_probability", "ml_danger_class", "snapshot_date"])
    return pd.concat(frames, ignore_index=True)


def read_all_prediction_meta(log_dir) -> list:
    d = predictions_dir(log_dir)
    rows = []
    if d.exists():
        for p in sorted(d.glob("*.meta.json")):
            with open(p) as f:
                rows.append(json.load(f))
    return rows


def _run(args, log_dir, check=True):
    return subprocess.run(args, cwd=str(log_dir), capture_output=True, text=True, check=check)


def commit_and_push(log_dir, paths: list, message: str, max_retries: int = 5):
    """Fetch-rebase-retry push loop. Safe against the two workflows racing
    each other because they write disjoint subpaths (hotspots/ vs.
    predictions/) -- a rebase here can only ever replay a commit that
    touches different files, so it always succeeds; this loop exists to
    handle the race, not to resolve real conflicts (there shouldn't be
    any)."""
    _run(["git", "config", "user.name", GIT_USER_NAME], log_dir)
    _run(["git", "config", "user.email", GIT_USER_EMAIL], log_dir)

    for attempt in range(1, max_retries + 1):
        _run(["git", "add"] + [str(p) for p in paths], log_dir)
        diff = _run(["git", "diff", "--cached", "--quiet"], log_dir, check=False)
        if diff.returncode == 0:
            print("shadow_log.commit_and_push: nothing to commit")
            return
        _run(["git", "commit", "-m", message], log_dir)

        push = _run(["git", "push", "origin", "HEAD:shadow-eval-log"], log_dir, check=False)
        if push.returncode == 0:
            print(f"shadow_log.commit_and_push: pushed on attempt {attempt}")
            return

        print(f"shadow_log.commit_and_push: push rejected on attempt {attempt}, "
              f"fetching + rebasing and retrying\n{push.stderr}")
        _run(["git", "fetch", "origin", "shadow-eval-log"], log_dir)
        rebase = _run(["git", "rebase", "origin/shadow-eval-log"], log_dir, check=False)
        if rebase.returncode != 0:
            _run(["git", "rebase", "--abort"], log_dir, check=False)
            raise RuntimeError(
                f"shadow_log.commit_and_push: rebase failed on attempt {attempt} -- this should be "
                f"impossible given disjoint write paths, investigate rather than retry blindly.\n"
                f"{rebase.stdout}\n{rebase.stderr}"
            )
        time.sleep(2 * attempt)  # brief backoff before the next push attempt

    raise RuntimeError(f"shadow_log.commit_and_push: failed to push after {max_retries} attempts")
