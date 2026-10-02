"""
Shared nearest-in-domain-cell attribution, used by both Stage 2
(historical_fire) and Stage 3 (ground truth) so the two don't end up with
two independently-written, silently-different definitions of "which cell
does this fire belong to" -- the exact kind of divergence the rest of this
rebuild exists to eliminate.

Replaces the original 25km buffer join (which doesn't tile the grid's
55.6km latitude spacing / 19-42km longitude spacing -- 17.43% of 2023 fires
were >25km from any node and silently dropped) with nearest-cell (Voronoi)
assignment via a KD-tree in projected metres, capped at a generous distance
so only fires far offshore or deep outside the domain go unattributed.
"""

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point

METRIC_CRS = "EPSG:3347"


def assign_nearest_cell(fire_lats, fire_lons, grid_domain, max_km):
    """
    fire_lats/fire_lons: array-like of fire point coordinates (EPSG:4326).
    grid_domain: DataFrame with cell_id, lat, lon (already land-masked to the
    in-domain cells only -- pass grid_domain[grid_domain.in_canada] in).
    max_km: cap beyond which a fire is left unattributed rather than forced
    onto a distant cell.

    Returns (cell_ids, dist_km): cell_ids[i] is None if fire i's nearest
    in-domain cell is farther than max_km away.
    """
    fire_points = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in zip(fire_lats, fire_lons)], crs="EPSG:4326"
    ).to_crs(METRIC_CRS)
    cell_points = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in zip(grid_domain["lat"], grid_domain["lon"])], crs="EPSG:4326"
    ).to_crs(METRIC_CRS)

    tree = cKDTree(np.column_stack([cell_points.x, cell_points.y]))
    dist_m, idx = tree.query(np.column_stack([fire_points.x, fire_points.y]))
    dist_km = dist_m / 1000.0

    cell_id_array = grid_domain["cell_id"].values
    cell_ids = np.where(dist_km <= max_km, cell_id_array[idx], None)
    return cell_ids, dist_km


def coverage_report(n_fires, cell_ids, dist_km, max_km):
    """Prints and returns the attribution coverage summary every attribution
    run should surface -- the 17.43%-of-fires-silently-dropped number from
    the original 25km buffer join must never again be invisible until an
    audit finds it."""
    attributed = sum(c is not None for c in cell_ids)
    dropped = n_fires - attributed
    report = {
        "n_fires": n_fires,
        "n_attributed": attributed,
        "n_dropped": dropped,
        "pct_dropped": dropped / n_fires * 100 if n_fires else 0.0,
        "dist_km_median": float(np.median(dist_km)) if len(dist_km) else None,
        "dist_km_p95": float(np.percentile(dist_km, 95)) if len(dist_km) else None,
        "dist_km_max": float(np.max(dist_km)) if len(dist_km) else None,
        "max_km_cap": max_km,
    }
    print(f"Attribution coverage: {attributed}/{n_fires} fires attributed "
          f"({report['pct_dropped']:.2f}% dropped, cap={max_km}km)")
    print(f"  distance to nearest cell: median={report['dist_km_median']:.2f}km "
          f"p95={report['dist_km_p95']:.2f}km max={report['dist_km_max']:.2f}km")
    return report
