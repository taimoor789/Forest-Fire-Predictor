"""
Provenance ledger for every artifact the ML rebuild produces.

Exists because the original ml_tier_thresholds.json was calibrated against a
DIFFERENT model fit than the one actually shipped -- their timestamps are
~10 hours apart, verified directly, and nothing enforced they came from the
same run. Fix: every artifact write is tagged with a run_id (one per
start_run() call); anything loading two related artifacts calls
verify_same_run() first and fails loudly on a mismatch instead of silently
serving a mismatched pair.

Entries also record a hash of ml/config.py and the git SHA -- data/ and
model_components/ are both gitignored, so this file is the only durable
record of what produced a given artifact.
"""

import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

from ml.config import MANIFEST_PATH, REPO_ROOT, SEED


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _row_count(path: Path):
    """Best-effort row count for csv/parquet; None for anything else (e.g. a .pkl)."""
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).metadata.num_rows
        if path.suffix == ".csv":
            with open(path, "rb") as f:
                return sum(1 for _ in f) - 1  # minus header
    except Exception:
        pass
    return None


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def config_hash() -> str:
    """Hash of ml/config.py's own source -- any change to the config values
    that produced an artifact shows up here, so a stale artifact built under
    an old config is distinguishable from a fresh one even if both exist on
    disk with the same filename."""
    return _sha256(Path(__file__).parent / "config.py")


def _load() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"entries": []}


def _save(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(MANIFEST_PATH)  # atomic


def start_run(stage: str, notes: str = "") -> str:
    """Call once at the top of a stage script. Every record() call in that
    script should pass the returned run_id, so all artifacts it produces are
    tagged as having come from the same invocation."""
    run_id = f"{stage}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    return run_id


def record(run_id: str, path, upstream: list = None, extra: dict = None) -> dict:
    """Record one artifact write. `upstream` is a list of paths this artifact
    was built from (their most recent manifest entries are looked up and their
    hashes embedded, so a full dependency chain is reconstructable from any
    single entry)."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    upstream_entries = []
    for up in (upstream or []):
        up_entry = latest(up)
        if up_entry is None:
            raise ValueError(
                f"record(): upstream artifact {up} has no manifest entry -- "
                f"record it before using it as an upstream dependency"
            )
        upstream_entries.append({"path": up_entry["path"], "sha256": up_entry["sha256"], "run_id": up_entry["run_id"]})

    entry = {
        "run_id": run_id,
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256(path),
        "rows": _row_count(path),
        "git_sha": _git_sha(),
        "config_hash": config_hash(),
        "seed": SEED,
        "upstream": upstream_entries,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        entry["extra"] = extra

    manifest = _load()
    manifest["entries"].append(entry)
    _save(manifest)
    return entry


def latest(path) -> dict:
    """Most recent manifest entry for a path, or None if it's never been recorded."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    rel = str(path.relative_to(REPO_ROOT))

    manifest = _load()
    matches = [e for e in manifest["entries"] if e["path"] == rel]
    return matches[-1] if matches else None


def verify_same_run(*paths):
    """Assert every given artifact's most recent manifest entry shares the
    same run_id -- i.e. they were produced together, not by two separate runs
    that happen to share a directory. Raises AssertionError with a clear
    message (including each artifact's actual run_id and timestamp) if not.

    This is the direct, mechanical fix for the tier-thresholds/model mismatch:
    calling verify_same_run("model_components/fire_risk_ml_model.pkl",
    "model_components/ml_tier_thresholds.json") before loading either would
    have failed loudly instead of silently serving mismatched artifacts.
    """
    entries = {}
    for p in paths:
        e = latest(p)
        if e is None:
            raise AssertionError(f"verify_same_run(): {p} has no manifest entry at all")
        entries[str(p)] = e

    run_ids = {e["run_id"] for e in entries.values()}
    if len(run_ids) > 1:
        detail = "\n".join(
            f"  {p}: run_id={e['run_id']} written {e['timestamp_utc']}"
            for p, e in entries.items()
        )
        raise AssertionError(
            f"verify_same_run(): artifacts came from DIFFERENT runs, not one "
            f"coordinated write:\n{detail}"
        )
