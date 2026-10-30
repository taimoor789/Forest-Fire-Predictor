"""
Stage 8: pre-registered train/test split regimes. Written and locked before
Stage 9's evaluation harness runs -- see docs/PREREGISTRATION.md (Stage 11)
for why that ordering matters: a split chosen after seeing results is the
exact post-hoc-rationalization failure mode this rebuild exists to avoid.

Three regimes, none of which existed anywhere in the original pipeline
(verified: the original split was a pure date cut over the SAME cells --
P(historical_fire=1) identical to 13 decimal places between train and test):

  - Temporal: all cells, train on years < HOLDOUT_YEAR, test on
    HOLDOUT_YEAR. Tests "next season, same places."
  - Spatial: group-k-fold on block_id (~4-degree blocks from Stage 1), all
    years in both train and test, with a SPATIAL_BLOCK_BUFFER_RING of
    adjacent blocks excluded from training around each held-out fold so
    spatial autocorrelation can't bleed across the fold boundary. Tests
    "new place, same season range."
  - Spatio-temporal: both at once, using fold 0 (fixed by SEED, never
    re-rolled after seeing results) combined with the temporal holdout.
    Pre-registered as the deciding regime for Stage 11's gate -- the only
    one of the three that answers the operationally relevant question ("new
    place, new season") a deployed model actually faces.

Plus a calibration split (Stage 10): fold 1's blocks, restricted to the
training years -- disjoint from both the spatio-temporal fit-train and the
test, reserved purely for probability calibration.

Run as a module from the repo root: python3 -m ml.splits
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ml import config


def _block_coords(block_id: str):
    lat_bin, lon_bin = block_id.split("_")
    return float(lat_bin), float(lon_bin)


def _buffer_ring(test_blocks: set, all_blocks: set, ring: int = config.SPATIAL_BLOCK_BUFFER_RING) -> set:
    """Blocks within `ring` block-widths (SPATIAL_BLOCK_SIZE_DEG each) of any
    test block -- excluded from training so spatial autocorrelation can't
    bleed across the fold boundary."""
    test_coords = {_block_coords(b) for b in test_blocks}
    step = config.SPATIAL_BLOCK_SIZE_DEG
    buffer_blocks = set()
    for b in all_blocks:
        if b in test_blocks:
            continue
        lat, lon = _block_coords(b)
        for tlat, tlon in test_coords:
            if abs(lat - tlat) <= ring * step + 1e-6 and abs(lon - tlon) <= ring * step + 1e-6:
                buffer_blocks.add(b)
                break
    return buffer_blocks


def _raw_spatial_folds(block_ids, k: int, seed: int):
    unique_blocks = np.array(sorted(set(block_ids)))
    coords = np.array([_block_coords(b) for b in unique_blocks])
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(coords)
    return [set(unique_blocks[labels == i]) for i in range(k)], coords, unique_blocks


def spatial_folds(block_ids, k: int = config.N_SPATIAL_FOLDS, seed: int = config.SEED,
                    block_positive_counts: dict = None, min_fold_positives: int = 1000):
    """Assigns each unique block_id to one of k folds via KMeans on block
    centroid coordinates -- geographically CONTIGUOUS folds, not a random
    scatter. A random per-block assignment was tried first and rejected:
    on this project's dense 4-degree block grid (172 blocks covering all of
    Canada), a random 20%-per-fold test set has a same-or-adjacent-fold
    neighbor almost everywhere, so the buffer ring (SPATIAL_BLOCK_BUFFER_RING)
    ended up excluding 63% of all rows from training -- verified directly.
    Contiguous regions keep the buffer to just each region's boundary.

    `block_positive_counts`: pass {block_id: total positive-row count
    across all years} to guard against a real failure mode found in this
    project -- a purely geographic KMeans partition landed an entire fold
    (33 blocks, 907K downstream rows) in the high Arctic tundra, which
    structurally never burns, giving Stage 10's calibration split ZERO
    positives and silently collapsing IsotonicRegression to a constant.
    Any fold with fewer than min_fold_positives total positive rows has
    its blocks reassigned to their nearest positive-containing fold by
    centroid distance -- deterministic, based only on label counts (never
    on any model's performance), so this can't become a way to p-hack
    results after the fact. Omit `block_positive_counts` to get the raw,
    unrepaired partition."""
    folds, coords, unique_blocks = _raw_spatial_folds(block_ids, k, seed)
    if block_positive_counts is None:
        return folds

    coord_by_block = dict(zip(unique_blocks, coords))
    fold_positive_counts = [sum(block_positive_counts.get(b, 0) for b in f) for f in folds]

    degenerate = [i for i, c in enumerate(fold_positive_counts) if c < min_fold_positives]
    if not degenerate:
        return folds

    healthy = [i for i in range(k) if i not in degenerate]
    assert healthy, "every spatial fold is degenerate -- min_fold_positives is too strict or data is broken"

    for i in degenerate:
        for block in list(folds[i]):
            block_coord = coord_by_block[block]
            # nearest HEALTHY fold by min distance to any of its member blocks
            best_fold, best_dist = None, np.inf
            for h in healthy:
                fold_coords = np.array([coord_by_block[b] for b in folds[h]])
                dist = np.linalg.norm(fold_coords - block_coord, axis=1).min()
                if dist < best_dist:
                    best_dist, best_fold = dist, h
            folds[i].discard(block)
            folds[best_fold].add(block)
        print(f"  spatial_folds: fold {i} was degenerate ({fold_positive_counts[i]} positive rows total) "
              f"-- its blocks were reassigned to nearest healthy folds")

    return [f for f in folds if f]  # degenerate folds are now empty -- drop them


def compute_block_positive_counts(df: pd.DataFrame, target_col: str = f"label_w{config.PRIMARY_LABEL_WINDOW}") -> dict:
    """{block_id: total positive-row count across ALL years in df} -- feed
    straight into spatial_folds' block_positive_counts to repair any
    degenerate (zero-fire) fold before it's used anywhere."""
    return df.groupby("block_id")[target_col].sum().to_dict()


def temporal_split(df: pd.DataFrame):
    train_mask = df["year"] < config.HOLDOUT_YEAR
    test_mask = df["year"] == config.HOLDOUT_YEAR
    return train_mask, test_mask


def spatial_split(df: pd.DataFrame, folds, fold_idx: int):
    all_blocks = set(df["block_id"].unique())
    test_blocks = folds[fold_idx]
    buffer_blocks = _buffer_ring(test_blocks, all_blocks)
    train_blocks = all_blocks - test_blocks - buffer_blocks

    train_mask = df["block_id"].isin(train_blocks)
    test_mask = df["block_id"].isin(test_blocks)
    return train_mask, test_mask


def spatio_temporal_split(df: pd.DataFrame, folds):
    """The pre-registered deciding regime for Stage 11: fold 0's held-out
    blocks combined with the temporal holdout year."""
    test_blocks = folds[0]
    all_blocks = set(df["block_id"].unique())
    buffer_blocks = _buffer_ring(test_blocks, all_blocks)
    train_blocks = all_blocks - test_blocks - buffer_blocks

    train_mask = df["block_id"].isin(train_blocks) & (df["year"] < config.HOLDOUT_YEAR)
    test_mask = df["block_id"].isin(test_blocks) & (df["year"] == config.HOLDOUT_YEAR)
    return train_mask, test_mask


def calibration_split(df: pd.DataFrame, folds):
    """Disjoint from the spatio-temporal train/test above: fold 1's blocks,
    restricted to the training years, reserved purely for Stage 10's
    probability calibration -- never used to fit the model, never touched
    for evaluation."""
    train_mask, test_mask = spatio_temporal_split(df, folds)
    calib_blocks = folds[1]
    calib_mask = df["block_id"].isin(calib_blocks) & (df["year"] < config.HOLDOUT_YEAR)
    fit_train_mask = train_mask & ~calib_mask
    return fit_train_mask, calib_mask, test_mask


if __name__ == "__main__":
    dataset = pd.read_parquet(config.DATA_DIR / "dataset_full.parquet",
                                columns=["block_id", "year", f"label_w{config.PRIMARY_LABEL_WINDOW}"])
    block_positive_counts = compute_block_positive_counts(dataset)
    folds = spatial_folds(dataset["block_id"].unique(), block_positive_counts=block_positive_counts)
    print(f"{len(folds)} spatial folds, sizes: {[len(f) for f in folds]}")

    train_mask, test_mask = temporal_split(dataset)
    print(f"\nTemporal: train={train_mask.sum()}, test={test_mask.sum()}")

    train_mask, test_mask = spatial_split(dataset, folds, fold_idx=0)
    print(f"Spatial (fold 0 held out): train={train_mask.sum()}, test={test_mask.sum()}, "
          f"buffer-excluded={len(dataset) - train_mask.sum() - test_mask.sum()}")

    train_mask, test_mask = spatio_temporal_split(dataset, folds)
    print(f"Spatio-temporal (deciding regime): train={train_mask.sum()}, test={test_mask.sum()}")

    fit_train_mask, calib_mask, test_mask = calibration_split(dataset, folds)
    print(f"With calibration carved out: fit_train={fit_train_mask.sum()}, "
          f"calib={calib_mask.sum()}, test={test_mask.sum()}")
