"""
Loads and cleans the full NFDB point archive -- shared by Stage 2
(historical_fire) and Stage 3 (ground truth) so both use one definition of
"a valid historical fire," not two that could silently diverge.
"""

import geopandas as gpd
import pandas as pd

from ml import attribution, config

NFDB_SHP = config.REFERENCE_DIR / "nfdb" / "NFDB_point_20260811.shp"

# The raw archive has 243 rows (0.05%) with corrupted LATITUDE/LONGITUDE
# attribute values (e.g. LATITUDE=-115.667, LONGITUDE=116188.0), independent
# of the geometry column. Generous enough to keep every real Canadian fire,
# tight enough to drop the garbage.
LAT_BOUNDS = (40.0, 85.0)
LON_BOUNDS = (-145.0, -50.0)


def load_nfdb() -> pd.DataFrame:
    """year, rep_date (parsed, NaT if unparseable), lat, lon, size_ha for
    every structurally valid NFDB record."""
    print(f"Loading full NFDB archive from {NFDB_SHP}...")
    gdf = gpd.read_file(NFDB_SHP)
    print(f"  {len(gdf)} raw rows")

    before = len(gdf)
    valid = (
        (gdf["YEAR"] != -999)
        & gdf["LATITUDE"].between(*LAT_BOUNDS)
        & gdf["LONGITUDE"].between(*LON_BOUNDS)
    )
    gdf = gdf[valid].copy()
    print(f"  {len(gdf)} valid rows ({before - len(gdf)} dropped: bad YEAR or out-of-bounds coordinates)")
    print(f"  year range: {gdf['YEAR'].min()}-{gdf['YEAR'].max()}")

    out = gdf[["YEAR", "REP_DATE", "LATITUDE", "LONGITUDE", "SIZE_HA"]].rename(
        columns={"YEAR": "year", "REP_DATE": "rep_date", "LATITUDE": "lat", "LONGITUDE": "lon", "SIZE_HA": "size_ha"}
    )
    out["rep_date"] = pd.to_datetime(out["rep_date"], errors="coerce")
    return out.reset_index(drop=True)


def attribute_all_fires(fires: pd.DataFrame, grid_domain: pd.DataFrame) -> pd.DataFrame:
    """Attach cell_id (nearest in-domain cell) + dist_km to every fire;
    drops fires beyond config.GROUND_TRUTH_MAX_ATTRIBUTION_KM."""
    print(f"Attributing {len(fires)} fires to nearest in-domain cell "
          f"(cap {config.GROUND_TRUTH_MAX_ATTRIBUTION_KM}km)...")
    cell_ids, dist_km = attribution.assign_nearest_cell(
        fires["lat"].values, fires["lon"].values, grid_domain, config.GROUND_TRUTH_MAX_ATTRIBUTION_KM
    )
    attribution.coverage_report(len(fires), cell_ids, dist_km, config.GROUND_TRUTH_MAX_ATTRIBUTION_KM)
    fires = fires.copy()
    fires["cell_id"] = cell_ids
    fires["dist_km"] = dist_km
    return fires[fires["cell_id"].notna()].copy()
