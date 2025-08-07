import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

grid_gdf = pd.read_csv("canada_fire_grid.csv") 

# Define some representative stations/cities
stations = [
    ("Vancouver", 49.2827, -123.1207),
    ("Kelowna", 49.8880, -119.4960),
    ("Kamloops", 50.6745, -120.3273),
    ("Calgary", 51.0447, -114.0719),
    ("Edmonton", 53.5461, -113.4938),
    ("Fort McMurray", 56.7266, -111.3790),
    ("Saskatoon", 52.1579, -106.6702),
    ("Regina", 50.4452, -104.6189),
    ("Winnipeg", 49.8951, -97.1384),
    ("Thunder Bay", 48.3809, -89.2477),
    ("Ottawa", 45.4215, -75.6972),
    ("Toronto", 43.6510, -79.3470),
    ("Sudbury", 46.4917, -80.9930),
    ("Montreal", 45.5019, -73.5674),
    ("Quebec City", 46.8139, -71.2080),
    ("Halifax", 44.6488, -63.5752),
    ("Whitehorse", 60.7212, -135.0568),
    ("Yellowknife", 62.4540, -114.3718),
    ("Prince George", 53.9171, -122.7497),
    ("Victoria", 48.4284, -123.3656),
    ("Smithers", 54.7800, -127.1743),
    ("Dease Lake", 58.4356, -130.0089),
    ("Fort St. John", 56.2524, -120.8466),
    ("High Level", 58.5169, -117.1360),
    ("Peace River", 56.2333, -117.2833),
    ("La Ronge", 55.1000, -105.3000),
    ("Flin Flon", 54.7682, -101.8779),
    ("Churchill", 58.7684, -94.1650),
    ("Moosonee", 51.2794, -80.6463),
    ("Timmins", 48.4758, -81.3305),
    ("Val-d'Or", 48.1086, -77.7972),
    ("Chibougamau", 49.9167, -74.3667),
    ("Schefferville", 54.8000, -66.8167),
    ("Goose Bay", 53.3019, -60.3267),
    ("St. John's", 47.5615, -52.7126),
    ("Iqaluit", 63.7467, -68.5170),
    ("Rankin Inlet", 62.8090, -92.0853),
    ("Cambridge Bay", 69.1167, -105.0667)
]

#Convert stations to GeoDataFrame for spatial operations
stations_gdf = gpd.GeoDataFrame(
    stations,
    columns=["name", "lat", "lon"],
    geometry=[Point(lon, lat) for _, lat, lon in stations],
    crs="EPSG:4326"
)

# Convert grid_gdf to GeoDataFrame
grid_gdf = gpd.GeoDataFrame(
    grid_gdf,
    geometry=[Point(lon, lat) for lat, lon in zip(grid_gdf["lat"], grid_gdf["lon"])],
    crs="EPSG:4326"
)

#Project both to meters for accurate distances
stations_gdf = stations_gdf.to_crs(epsg=3347)
grid_gdf = grid_gdf.to_crs(epsg=3347)

#Find nearest station for each grid cell
nearest_station_index = []
for point in grid_gdf.geometry:
  distances = stations_gdf.geometry.distance(point)
  nearest_station_index.append(stations_gdf.loc[distances.idxmin(), "name"])

#Save mapping
grid_gdf["nearest_station_name"] = nearest_station_index

grid_gdf[["lat", "lon", "nearest_station_name"]].to_csv("stations.csv", index=False)

print("Nearest station mapping saved")
