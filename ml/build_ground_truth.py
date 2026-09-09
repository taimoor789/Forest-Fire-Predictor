"""
Stage 3: ground truth (cell_id, date) fire records, replacing the original
25km-buffer join, which neither tiles the grid's 55.6km latitude spacing nor
its 19-42km longitude spacing -- 17.43% of 2023 fires were >25km from any
grid node and silently dropped from ground truth.

Primary output uses the same nearest-cell (Voronoi) attribution as Stage 2
(ml/nfdb.py, shared so the two can't silently diverge on what "near this
cell" means), generalized across all target years rather than just 2023.
A 30km-buffer join is also emitted as a documented sensitivity check --
unlike Voronoi, a fire can match more than one cell under a buffer, so the
two aren't measuring quite the same thing ("nearest cell" vs. "within Rkm").

Run as a module from the repo root: python3 -m ml.build_ground_truth
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ml import config, manifest, nfdb

METRIC_CRS = "EPSG:3347"


def build_buffer_sensitivity(fires: pd.DataFrame, grid_domain: pd.DataFrame, buffer_km: float) -> pd.DataFrame:
    """A fire may attribute to multiple cells here (whichever cells' buffer
    circles it falls inside), unlike the primary Voronoi assignment, which
    is exactly one nearest cell per fire."""
    fire_points = gpd.GeoDataFrame(
        fires[["rep_date", "year"]].reset_index(drop=True),
        geometry=[Point(lon, lat) for lat, lon in zip(fires["lat"], fires["lon"])],
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    cells = gpd.GeoDataFrame(
        grid_domain[["cell_id"]].reset_index(drop=True),
        geometry=[Point(lon, lat) for lat, lon in zip(grid_domain["lat"], grid_domain["lon"])],
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)
    cells["geometry"] = cells.geometry.buffer(buffer_km * 1000)

    joined = gpd.sjoin(fire_points, cells, predicate="within")[["cell_id", "rep_date", "year"]]
    n_fires = len(fires)
    n_matched = joined.index.nunique()
    print(f"Buffer sensitivity ({buffer_km}km): {n_matched}/{n_fires} fires matched at least one cell "
          f"({(n_fires - n_matched) / n_fires * 100:.2f}% dropped), "
          f"{len(joined)} total (fire, cell) pairs ({len(joined) - n_matched} multi-attributed)")
    return joined.drop_duplicates(subset=["cell_id", "rep_date"]).reset_index(drop=True)


def build():
    run_id = manifest.start_run("stage3_ground_truth")

    grid_domain = pd.read_parquet(config.GRID_DOMAIN_PATH)
    grid_domain = grid_domain[grid_domain["in_canada"]].reset_index(drop=True)
    print(f"{len(grid_domain)} in-domain cells")

    fires = nfdb.load_nfdb()
    n_before_date_filter = len(fires)
    fires = fires[fires["rep_date"].notna()].copy()
    print(f"{len(fires)}/{n_before_date_filter} fires have a parseable REP_DATE "
          f"(ground truth needs a date; historical_fire in Stage 2 only needed the year)")

    target_years = sorted(set(config.TRAINING_YEARS + [config.HOLDOUT_YEAR]))
    fires_in_scope = fires[fires["year"].isin(target_years)].copy()
    print(f"{len(fires_in_scope)} fires in target years {target_years}")

    # ---- Primary: nearest-cell (Voronoi) attribution ----
    attributed = nfdb.attribute_all_fires(fires_in_scope, grid_domain)
    primary = (
        attributed[["cell_id", "rep_date", "year", "dist_km"]]
        .drop_duplicates(subset=["cell_id", "rep_date"])
        .reset_index(drop=True)
    )
    print(f"\n{len(primary)} unique (cell, date) ground-truth rows (Voronoi, primary)")
    print(primary.groupby("year").size())

    # ---- Sensitivity: 30km buffer join ----
    print()
    buffer_variant = build_buffer_sensitivity(fires_in_scope, grid_domain, config.GROUND_TRUTH_SENSITIVITY_BUFFER_KM)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    primary_path = config.DATA_DIR / "ground_truth_by_year.parquet"
    primary.to_parquet(primary_path, index=False)
    print(f"\nWrote {primary_path}")

    buffer_path = config.DATA_DIR / "ground_truth_buffer_sensitivity.parquet"
    buffer_variant.to_parquet(buffer_path, index=False)
    print(f"Wrote {buffer_path}")

    manifest.record(
        run_id, primary_path,
        upstream=[config.GRID_DOMAIN_PATH],
        extra={"target_years": target_years, "n_fires_in_scope": len(fires_in_scope), "n_rows": len(primary)},
    )
    manifest.record(
        run_id, buffer_path,
        upstream=[config.GRID_DOMAIN_PATH],
        extra={"buffer_km": config.GROUND_TRUTH_SENSITIVITY_BUFFER_KM, "n_rows": len(buffer_variant)},
    )
    print(f"Manifest entries recorded: run_id={run_id}")
    return primary


if __name__ == "__main__":
    build()
