"""
Stage 1: annotate the 14,952-cell grid with a real land mask, area weights,
province (via production's get_province, not a copy), and spatial-CV blocks.

Reads lat/lon from data/canada_fire_grid.csv but never writes to it --
output is data/grid_domain_v1.parquet. historical_fire isn't carried over;
it's rebuilt leak-free in Stage 2.

Uses a real country polygon (Natural Earth 1:10m Admin 0), buffered +10km,
rather than a rectangle -- a rectangle can't tell ocean or US territory from
Canada, which is exactly what let bad cells into the original training data.

Run as a module from the repo root: python3 -m ml.build_grid_domain
"""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ml import config, manifest
from fire_risk import FireWeatherProcessor

NATURAL_EARTH_SHP = config.REFERENCE_DIR / "natural_earth" / "ne_10m_admin_0_countries.shp"
METRIC_CRS = "EPSG:3347"  # Statistics Canada Lambert -- same projection used
                           # throughout the rest of this project's spatial joins


def cell_area_km2(lat: float, step_deg: float = config.GRID_STEP_DEG) -> float:
    """Area of a step_deg x step_deg lat/lon cell centered at `lat`, spherical
    approximation (R=6371km). Matches the ~2325km^2 at 41N / ~375km^2 at 83N
    figures from the audit to within the expected spherical-vs-ellipsoidal
    rounding."""
    R = 6371.0
    step_rad = math.radians(step_deg)
    return (R ** 2) * step_rad * step_rad * math.cos(math.radians(lat))


def build():
    run_id = manifest.start_run("stage1_grid_domain")

    print(f"Loading Natural Earth Canada boundary from {NATURAL_EARTH_SHP}...")
    countries = gpd.read_file(NATURAL_EARTH_SHP)
    canada = countries[countries["ADMIN"] == "Canada"]
    assert len(canada) == 1, f"expected exactly one Canada row, got {len(canada)}"

    canada_metric = canada.to_crs(METRIC_CRS)
    canada_poly_metric = canada_metric.geometry.iloc[0]
    canada_buffered_metric = canada_poly_metric.buffer(config.LAND_MASK_BUFFER_KM * 1000)

    print("Loading grid cells from data/canada_fire_grid.csv...")
    grid = pd.read_csv(config.DATA_DIR / "canada_fire_grid.csv", usecols=["lat", "lon"])
    print(f"  {len(grid)} cells")

    points_geo = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in zip(grid["lat"], grid["lon"])],
        crs="EPSG:4326",
    )
    points_metric = points_geo.to_crs(METRIC_CRS)

    print("Computing land mask (point-in-polygon against buffered Canada boundary)...")
    in_canada = points_metric.within(canada_buffered_metric).values

    print("Computing diagnostic distance to the (unbuffered) Canada boundary...")
    # All in METRIC_CRS (meters) -- distance() to a polygon's boundary is
    # accurate regardless of whether the point is inside or outside it;
    # negate for points inside the (unbuffered) polygon so the sign reads as
    # "how far outside (+) / inside (-) the coastline."
    dist_km = points_metric.distance(canada_poly_metric.boundary).values / 1000.0
    inside_unbuffered = points_metric.within(canada_poly_metric).values
    dist_km_signed = np.where(inside_unbuffered, -dist_km, dist_km)

    print("Computing province via fire_risk.FireWeatherProcessor.get_province (production code, not a local copy)...")
    processor = FireWeatherProcessor()
    province = [processor.get_province(lat, lon) for lat, lon in zip(grid["lat"], grid["lon"])]

    print("Computing area weights (cos-latitude)...")
    area_km2 = np.array([cell_area_km2(lat) for lat in grid["lat"]])
    area_weight = area_km2 / area_km2.sum()

    print("Assigning spatial-CV blocks...")
    block_lat = (np.floor(grid["lat"] / config.SPATIAL_BLOCK_SIZE_DEG) * config.SPATIAL_BLOCK_SIZE_DEG)
    block_lon = (np.floor(grid["lon"] / config.SPATIAL_BLOCK_SIZE_DEG) * config.SPATIAL_BLOCK_SIZE_DEG)
    block_id = [f"{bl:.1f}_{bo:.1f}" for bl, bo in zip(block_lat, block_lon)]

    out = pd.DataFrame({
        "cell_id": [config.cell_id(lat, lon) for lat, lon in zip(grid["lat"], grid["lon"])],
        "lat": grid["lat"].values,
        "lon": grid["lon"].values,
        "in_canada": in_canada,
        "dist_to_canada_km": dist_km_signed,
        "province": province,
        "cell_area_km2": area_km2,
        "area_weight": area_weight,
        "block_id": block_id,
    })

    n_in = out["in_canada"].sum()
    n_arctic = (out["lat"] >= config.HRDPS_LAT_CUTOFF).sum()
    arctic_frac_count = n_arctic / len(out)
    arctic_frac_area = out.loc[out["lat"] >= config.HRDPS_LAT_CUTOFF, "area_weight"].sum()
    print(f"\nResults: {n_in}/{len(out)} cells in Canada ({n_in/len(out)*100:.1f}%), "
          f"{len(out)-n_in} masked out")
    print(f"  Arctic (>= {config.HRDPS_LAT_CUTOFF}N): {arctic_frac_count*100:.2f}% by count, "
          f"{arctic_frac_area*100:.2f}% by area")
    print("\nProvince distribution (in-Canada cells):")
    print(out[out["in_canada"]]["province"].value_counts())
    print(f"\nUnknown province among in-Canada cells: {((out['in_canada']) & (out['province']=='Unknown')).sum()}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.GRID_DOMAIN_PATH, index=False)
    print(f"\nWrote {config.GRID_DOMAIN_PATH}")

    entry = manifest.record(
        run_id, config.GRID_DOMAIN_PATH,
        upstream=[],
        extra={
            "natural_earth_source": "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip",
            "n_cells_total": len(out), "n_cells_in_canada": int(n_in),
            "arctic_frac_count": float(arctic_frac_count), "arctic_frac_area": float(arctic_frac_area),
        },
    )
    print(f"Manifest entry recorded: run_id={run_id}")
    return out


if __name__ == "__main__":
    build()
