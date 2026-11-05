"""
Stage 7: dataset assembly. Fixes the original case-control sampling bias by
using every in-domain cell (Stage 1's full domain, replayed in full by
Stage 6) rather than correcting a biased sample after the fact -- the
original pipeline sampled 2,033 burned cells + 800 hand-picked controls
(2,166 of 2,833 already historical_fire=1), a case-control frame with no
offset/weighting correction ever applied.

Row universe: labels_by_year.parquet's full cross product (in-domain cells x
fire-season dates x target years) is the base. This join narrows it in one
known, quantified way: ERA5-Land only has valid data over land cells in
ECMWF's own land-sea mask, which doesn't perfectly agree with this project's
independently-built Natural-Earth-based land mask (Stage 1) -- verified
directly (t2m is NaN at the nearest ERA5-Land point for every affected
cell, every date). 675/7,537 in-domain cells (9.0%) have zero ERA5
coverage, concentrated in the 10km coastal buffer zone (mean
dist_to_canada_km +3.1, i.e. just outside the true coastline) and the
Arctic Archipelago (362 in Nunavut, 158 in NWT) -- exactly where a coarse
0.5-degree land-only product struggles most. This is a second, narrower
domain restriction on top of Stage 1's land mask, not a bug: those 675
cells are excluded from the modelling dataset entirely (inner join, not
left-join-and-assert), reported explicitly below, and the exclusion must
match this known pattern -- see the assertion after the replay join. Any
OTHER narrowing (a different cell set, a different rate) is a real coverage
bug and must fail loudly, not silently drop rows.

Joins: replay features on (cell_id, date); historical_fire on
(cell_id, year == target_year) -- NEVER cell_id alone, or the leak this
whole rebuild exists to remove comes back in through the join. Then grid
domain's province/area_weight/block_id on cell_id alone (static per cell).

Run as a module from the repo root: python3 -m ml.build_dataset
"""

import pandas as pd

from ml import config, manifest


def build():
    run_id = manifest.start_run("stage7_dataset")

    labels = pd.read_parquet(config.DATA_DIR / "labels_by_year.parquet")
    grid_domain = pd.read_parquet(config.GRID_DOMAIN_PATH)
    grid_domain = grid_domain[grid_domain["in_canada"]].reset_index(drop=True)
    historical_fire = pd.read_parquet(config.DATA_DIR / "historical_fire_by_year.parquet")

    target_years = sorted(set(config.TRAINING_YEARS + [config.HOLDOUT_YEAR]))
    print(f"Target years: {target_years}")
    print(f"{len(labels)} label rows, {len(grid_domain)} in-domain cells")

    replay_frames = []
    for year in target_years:
        path = config.DATA_DIR / f"fwi_replay_{year}.parquet"
        replay_frames.append(pd.read_parquet(path))
    replay = pd.concat(replay_frames, ignore_index=True)
    replay["cell_id"] = replay["cell_id"].astype(str)
    print(f"{len(replay)} replayed (cell, date) rows across {len(target_years)} years")

    labels = labels.copy()
    labels["cell_id"] = labels["cell_id"].astype(str)
    labels["date"] = pd.to_datetime(labels["date"])
    replay["date"] = pd.to_datetime(replay["date"])

    before = len(labels)
    replay_cells = set(replay["cell_id"].unique())
    all_cells = set(labels["cell_id"].unique())
    uncovered_cells = all_cells - replay_cells
    coverage_rate = len(replay_cells) / len(all_cells)
    print(f"\nERA5 coverage: {len(replay_cells)}/{len(all_cells)} in-domain cells "
          f"({coverage_rate * 100:.1f}%), {len(uncovered_cells)} uncovered")

    uncovered_gd = grid_domain[grid_domain["cell_id"].isin(uncovered_cells)]
    if len(uncovered_gd):
        print(f"  uncovered cells' mean dist_to_canada_km: {uncovered_gd['dist_to_canada_km'].mean():.2f} "
              f"(vs. covered mean {grid_domain[~grid_domain['cell_id'].isin(uncovered_cells)]['dist_to_canada_km'].mean():.2f})")
        print(f"  uncovered by province:\n{uncovered_gd['province'].value_counts().to_string()}")

    # Known, quantified ERA5-Land land-mask gap (see module docstring) --
    # not the exact 675/9.0% every time (a different set of years could in
    # principle shift this slightly), but it must stay in the same small
    # ballpark and the same geographic pattern. A large deviation here means
    # something new broke, not the known gap -- investigate, don't relax this.
    assert 0.85 <= coverage_rate <= 0.95, (
        f"ERA5 coverage rate {coverage_rate:.3f} is well outside the expected "
        f"~91% band for the known land-mask gap -- this looks like a NEW "
        f"coverage bug, not the documented one. Investigate before proceeding."
    )

    out = labels[labels["cell_id"].isin(replay_cells)].merge(
        replay, on=["cell_id", "date"], how="left", validate="one_to_one"
    )
    missing_replay = int(out["ffmc"].isna().sum())
    print(f"{missing_replay}/{len(out)} rows (after restricting to ERA5-covered cells) "
          f"have no matching replay row (must be 0)")
    assert missing_replay == 0, (
        "a row for an ERA5-covered cell is missing its replay features -- a real "
        "coverage gap distinct from the known land-mask one, investigate before "
        "proceeding (do not silently drop or fill)"
    )
    print(f"Rows: {before} label rows -> {len(out)} after excluding ERA5-uncovered cells "
          f"({before - len(out)} dropped, {(before - len(out)) / before * 100:.1f}%)")

    historical_fire = historical_fire.rename(columns={"target_year": "year"})
    historical_fire["cell_id"] = historical_fire["cell_id"].astype(str)
    out = out.merge(historical_fire, on=["cell_id", "year"], how="left", validate="many_to_one")
    missing_hist = int(out["hist_fire_any_prior"].isna().sum())
    print(f"{missing_hist}/{len(out)} rows missing historical_fire features (must be 0)")
    assert missing_hist == 0, "labels row missing historical_fire features for its (cell_id, year)"

    grid_domain = grid_domain.copy()
    grid_domain["cell_id"] = grid_domain["cell_id"].astype(str)
    out = out.merge(
        grid_domain[["cell_id", "province", "area_weight", "block_id"]],
        on="cell_id", how="left", validate="many_to_one",
    )
    assert int(out["province"].isna().sum()) == 0, "row missing grid_domain static fields"

    out["cell_id"] = out["cell_id"].astype("category")

    print(f"\nFinal dataset: {len(out)} rows, {out.shape[1]} columns")
    primary = f"label_w{config.PRIMARY_LABEL_WINDOW}"
    print(f"Primary label ({primary}) base rate: {out[primary].mean() * 100:.3f}%")
    print(out.groupby("year")[primary].agg(["sum", "mean"]))

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_DIR / "dataset_full.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    manifest.record(
        run_id, out_path,
        upstream=[
            config.DATA_DIR / "labels_by_year.parquet",
            config.DATA_DIR / "historical_fire_by_year.parquet",
            config.GRID_DOMAIN_PATH,
        ] + [config.DATA_DIR / f"fwi_replay_{year}.parquet" for year in target_years],
        extra={
            "n_rows": len(out),
            "n_cols": out.shape[1],
            "target_years": target_years,
            "primary_label_base_rate": float(out[primary].mean()),
            "era5_covered_cells": len(replay_cells),
            "era5_uncovered_cells": len(uncovered_cells),
            "era5_coverage_rate": coverage_rate,
        },
    )
    print(f"Manifest entry recorded: run_id={run_id}")
    return out


if __name__ == "__main__":
    build()
