import pandas as pd

WEATHER_FILE = "nasa_power_weather_2023.csv"
LOCATIONS_FILE = "locations_to_pull.csv"
OUTPUT_FILE = "nasa_power_weather_filtered.csv"


def snap_to_grid(lat, lon, step=0.5):
    return round(lat / step) * step, round(lon / step) * step


def main():
    print("Loading weather data (this may take a minute for 7M+ rows)...")
    weather = pd.read_csv(WEATHER_FILE)
    print(f"Loaded {len(weather)} rows")

    before = len(weather)
    weather = weather.drop_duplicates()
    print(f"Dropped {before - len(weather)} exact duplicate rows "
          f"(tile boundary overlap)")

    print("Snapping coordinates to the 0.5-degree target grid...")
    snapped = weather["lat"].combine(weather["lon"], snap_to_grid)
    weather["grid_lat"] = snapped.apply(lambda x: x[0])
    weather["grid_lon"] = snapped.apply(lambda x: x[1])

    locations = pd.read_csv(LOCATIONS_FILE)
    locations["lat"] = locations["lat"].round(4)
    locations["lon"] = locations["lon"].round(4)
    target_cells = set(zip(locations["lat"], locations["lon"]))
    print(f"{len(target_cells)} target locations to keep")

    weather["_key"] = list(zip(weather["grid_lat"], weather["grid_lon"]))
    filtered = weather[weather["_key"].isin(target_cells)].copy()

    # if multiple native points snap to the same target cell, average them
    filtered = (
        filtered.groupby(["grid_lat", "grid_lon", "date"])
        [["T2M", "RH2M", "WS10M", "PRECTOTCORR"]]
        .mean()
        .reset_index()
        .rename(columns={"grid_lat": "lat", "grid_lon": "lon"})
    )

    filtered.to_csv(OUTPUT_FILE, index=False)

    covered = filtered[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"\nFiltered from {len(weather)} down to {len(filtered)} rows")
    print(f"Covers {covered}/{len(target_cells)} target locations")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()