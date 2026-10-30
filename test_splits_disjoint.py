"""
Stage 8 guard: none of the pre-registered split regimes may leak a cell or
date across a train/test (or train/calibration/test) boundary. Runs against
the real data/dataset_full.parquet, not a synthetic fixture -- the failure
mode this guards against (the original pipeline's train/test split sharing
identical P(historical_fire=1) to 13 decimal places) only shows up against
real data's actual block/year structure.
"""

import pandas as pd
import pytest

from ml import config, splits


TARGET_COL = f"label_w{config.PRIMARY_LABEL_WINDOW}"


@pytest.fixture(scope="module")
def dataset():
    return pd.read_parquet(config.DATA_DIR / "dataset_full.parquet",
                             columns=["cell_id", "date", "block_id", "year", TARGET_COL])


@pytest.fixture(scope="module")
def folds(dataset):
    block_positive_counts = splits.compute_block_positive_counts(dataset, TARGET_COL)
    return splits.spatial_folds(dataset["block_id"].unique(), block_positive_counts=block_positive_counts)


def test_no_degenerate_fold(dataset, folds):
    """Regression test: a purely geographic KMeans partition once landed an
    entire fold (33 blocks) in the high Arctic tundra, which structurally
    never burns -- 907,360 downstream rows with zero fire labels, which
    silently collapsed Stage 10's IsotonicRegression calibrator to a
    constant (verified: test PR-AUC dropped to exactly the base rate).
    Every fold must have a real, non-trivial positive count."""
    for i, f in enumerate(folds):
        sub = dataset[dataset["block_id"].isin(f)]
        assert sub[TARGET_COL].sum() >= 1000, f"fold {i} has too few positives ({sub[TARGET_COL].sum()})"


def test_temporal_split_no_year_overlap(dataset):
    train_mask, test_mask = splits.temporal_split(dataset)
    assert not (train_mask & test_mask).any()
    assert dataset.loc[train_mask, "year"].max() < config.HOLDOUT_YEAR
    assert (dataset.loc[test_mask, "year"] == config.HOLDOUT_YEAR).all()


def test_spatial_folds_partition_all_blocks(dataset, folds):
    all_blocks = set(dataset["block_id"].unique())
    union = set().union(*folds)
    assert union == all_blocks, "spatial_folds must partition every block, none left out"
    for i, fold_a in enumerate(folds):
        for fold_b in folds[i + 1:]:
            assert not (fold_a & fold_b), "folds must be pairwise disjoint"


def test_spatial_split_no_block_in_both_train_and_test(dataset, folds):
    for fold_idx in range(len(folds)):
        train_mask, test_mask = splits.spatial_split(dataset, folds, fold_idx)
        train_blocks = set(dataset.loc[train_mask, "block_id"].unique())
        test_blocks = set(dataset.loc[test_mask, "block_id"].unique())
        assert not (train_blocks & test_blocks), f"fold {fold_idx}: a block appears in both train and test"
        assert not (train_mask & test_mask).any()


def test_spatial_split_buffer_excluded_from_train(dataset, folds):
    """Every block within the buffer ring of the test blocks must be in
    NEITHER train nor test -- excluded entirely, not silently in train."""
    train_mask, test_mask = splits.spatial_split(dataset, folds, fold_idx=0)
    all_blocks = set(dataset["block_id"].unique())
    test_blocks = folds[0]
    buffer_blocks = splits._buffer_ring(test_blocks, all_blocks)
    assert buffer_blocks, "sanity check: fold 0 should have a nonempty buffer ring on this domain"
    train_blocks = set(dataset.loc[train_mask, "block_id"].unique())
    assert not (buffer_blocks & train_blocks), "a buffer-ring block leaked into train"
    assert not (buffer_blocks & test_blocks)


def test_spatio_temporal_split_disjoint_on_both_axes(dataset, folds):
    train_mask, test_mask = splits.spatio_temporal_split(dataset, folds)
    assert not (train_mask & test_mask).any()

    train_blocks = set(dataset.loc[train_mask, "block_id"].unique())
    test_blocks = set(dataset.loc[test_mask, "block_id"].unique())
    assert not (train_blocks & test_blocks)

    assert dataset.loc[train_mask, "year"].max() < config.HOLDOUT_YEAR
    assert (dataset.loc[test_mask, "year"] == config.HOLDOUT_YEAR).all()
    assert test_blocks == folds[0]


def test_calibration_split_three_way_disjoint(dataset, folds):
    fit_train_mask, calib_mask, test_mask = splits.calibration_split(dataset, folds)

    assert not (fit_train_mask & calib_mask).any()
    assert not (fit_train_mask & test_mask).any()
    assert not (calib_mask & test_mask).any()

    fit_train_blocks = set(dataset.loc[fit_train_mask, "block_id"].unique())
    calib_blocks = set(dataset.loc[calib_mask, "block_id"].unique())
    test_blocks = set(dataset.loc[test_mask, "block_id"].unique())
    assert not (fit_train_blocks & calib_blocks)
    assert not (fit_train_blocks & test_blocks)
    assert not (calib_blocks & test_blocks)
    assert calib_blocks == folds[1]

    assert (dataset.loc[calib_mask, "year"] < config.HOLDOUT_YEAR).all()


def test_no_cell_date_row_in_two_regime_masks_at_once(dataset, folds):
    """Belt-and-suspenders row-level check across all three regimes."""
    for train_mask, test_mask in [
        splits.temporal_split(dataset),
        splits.spatial_split(dataset, folds, fold_idx=0),
        splits.spatio_temporal_split(dataset, folds),
    ]:
        overlap = dataset.index[train_mask & test_mask]
        assert len(overlap) == 0
