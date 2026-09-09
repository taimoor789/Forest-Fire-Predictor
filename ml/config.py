"""
Single source of truth for the ML rebuild: grid bounds, spatial attribution
params, label windows, training years, seed, and paths. Every ml/ script
imports from here rather than hardcoding a value, so ml/manifest.py's
config_hash can catch a script that silently drifted from the rest.

Constants shared with the live serving path (GAP_REINIT_DAYS,
HRDPS_LAT_CUTOFF) are imported from their source modules, never redefined,
so a production change flows through instead of silently diverging.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `import fire_risk` / `collect_weather_grid_eccc` work from ml/ scripts

from fire_risk import GAP_REINIT_DAYS, TREND_HISTORY_DAYS          # noqa: E402
from collect_weather_grid_eccc import HRDPS_LAT_CUTOFF              # noqa: E402

# ---- Reproducibility ----
SEED = 42  # matches scripts/train_final_model.py's original random_state,
           # kept for continuity -- not otherwise load-bearing since that
           # model is quarantined.

# ---- Grid domain (matches forest_fire_data_preparation.ipynb cell 2 /
# data/canada_fire_grid.csv's existing 0.5-degree grid -- Stage 1 masks this
# down with a real land boundary, it does not change the raw grid spacing) ----
GRID_MIN_LAT, GRID_MAX_LAT = 41.0, 83.0
GRID_MIN_LON, GRID_MAX_LON = -141.0, -52.0
GRID_STEP_DEG = 0.5

# ---- Stage 1: land mask ----
# Natural Earth 1:10m Admin 0 Countries, vendored + checksummed into
# data/reference/ (see ml/build_grid_domain.py). Buffered outward so genuine
# coastal cells aren't dropped on a centroid technicality -- a plain rectangle
# mask (the existing PROVINCE_BOUNDS) is exactly what let ~1,930 US cells and
# ~2,325 ocean cells into the original training data.
LAND_MASK_BUFFER_KM = 10
CELL_ID_FMT = "{lat:.4f}_{lon:.4f}"  # matches FireWeatherProcessor._cell_key in fire_risk.py,
                                      # so grid_domain joins directly to data/fwi_state.json

# Spatial CV block size for Stage 8's group-k-fold -- degrees, roughly
# 400-500km at these latitudes.
SPATIAL_BLOCK_SIZE_DEG = 4.0
SPATIAL_BLOCK_BUFFER_RING = 1  # exclude cells in blocks adjacent to a held-out
                                 # test block from training, to stop spatial
                                 # autocorrelation bleeding across the boundary

# ---- Stage 2: historical_fire ----
# NRCan NFDB point archive (full history, not just the 2023 extract already
# in data/nfdb_fires_2023.csv) -- vendored into data/reference/nfdb/.
NFDB_FULL_ARCHIVE_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_point/current_version/NFDB_point.zip"
HISTORICAL_FIRE_LOOKBACK_YEARS = 20  # log1p(count of fires in the prior 20 years) is primary;
                                       # see ml/build_historical_fire.py for the sensitivity variants
HISTORICAL_FIRE_MIN_YEAR = 1980       # NFDB reporting is considered reasonably complete from here

# ---- Stage 3: ground truth attribution ----
# Nearest-cell (Voronoi) assignment replaces the original 25km buffer join,
# which neither tiles the grid's 55.6km latitude spacing nor its 19-42km
# longitude spacing (measured: 17.43% of 2023 fires >25km from any node were
# silently dropped). Buffer kept only as a documented sensitivity check.
GROUND_TRUTH_MAX_ATTRIBUTION_KM = 60.0   # cap beyond which a fire is unattributed, not forced to a distant cell
GROUND_TRUTH_SENSITIVITY_BUFFER_KM = 30.0  # secondary buffer-join variant, reported not primary

# ---- Stage 4: label construction ----
# +/-2 days confirmed (empirically, since the original code no longer exists
# anywhere) as the window that reproduces 100% of the known 2023 positives
# from the raw ground-truth rows; +/-1 covers only 63.7%.
LABEL_WINDOWS_DAYS = [0, 1, 2]  # all three emitted as separate columns; label_w2 is primary
PRIMARY_LABEL_WINDOW = 2

# ---- Stage 5/6/7: weather + replay + season window ----
# CaSPAr HRDPS/GDPS archive, same model families as the live collector
# (collect_weather_grid_eccc.py). 5 seasons: enough for the spatio-temporal
# holdout (Stage 8) to mean something without an XL multi-decade pull.
TRAINING_YEARS = [2019, 2020, 2021, 2022, 2023]
HOLDOUT_YEAR = 2023  # most recent -- Stage 8's temporal regime holds this out entirely

REPLAY_WARMUP_DAYS = 75  # days of weather to replay before the modelling window starts,
                           # so accumulated codes are past their bootstrap transient
                           # (production's own bootstrap_from_window uses up to 45 days
                           # of window when it has to fall back; this is deliberately longer
                           # since a full offline replay can afford it)

FIRE_SEASON_START_MD = (4, 1)   # Apr 1 -- modelling ROWS are restricted to this window;
FIRE_SEASON_END_MD = (10, 31)   # Oct 31 -- the full year is still replayed for accumulation state

# ---- Stage 9: evaluation ----
# Published regional wildfire-occurrence models typically land ROC-AUC
# 0.80-0.95; this project's positive rate (~2.6% in the original, leaked
# dataset -- expect this to shift once Stage 7's case-control sampling is
# fixed to population sampling) makes PR-AUC the more informative headline.
TOP_K_FRACTIONS = [0.01, 0.05]  # precision/recall@top-1%, @top-5%
N_BOOTSTRAP = 1000               # block bootstrap (over block_id, never rows) for CIs

# ---- Stage 11: pre-registered decision gate ----
# See docs/PREREGISTRATION.md for the authoritative, committed version of
# these numbers -- this module must not be the source of truth for the gate
# itself (that file is written once and amended only with dated notes,
# specifically so it can't be quietly loosened after seeing results). These
# constants exist only so ml/evaluate.py can print a live comparison; treat
# docs/PREREGISTRATION.md as authoritative if the two ever disagree.
GATE_MIN_RELATIVE_PR_AUC_LIFT_VS_RAW_FWI = 0.25
GATE_MAX_CALIBRATION_ECE = 0.02

# ---- Paths ----
DATA_DIR = REPO_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
DOCS_DIR = REPO_ROOT / "docs"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARCHIVE_DIR = REPO_ROOT / "archive"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

GRID_DOMAIN_PATH = DATA_DIR / "grid_domain_v1.parquet"


def cell_id(lat: float, lon: float) -> str:
    """Matches FireWeatherProcessor._cell_key in fire_risk.py exactly."""
    return CELL_ID_FMT.format(lat=lat, lon=lon)
