import os
import time
import json
import requests
import pandas as pd

LOCATIONS_FILE = "locations_to_pull.csv"
TILES_FILE = "needed_tiles.csv"
OUTPUT_FILE = "nasa_power_weather_2023.csv"
PROGRESS_FILE = "nasa_power_progress.json"

TILE_SIZE = 10
PARAMETERS = ["T2M", "RH2M", "WS10M", "PRECTOTCORR"]

# Split the ~412-day range into 2 chunks to stay under the 366-day cap
DATE_CHUNKS = [
    ("20221115", "20231114"),  # 365 days
    ("20231115", "20231231"),  # 47 days
]

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
REQUEST_TIMEOUT = 180
REQUEST_DELAY_SEC = 2
MAX_RETRIES = 3


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_progress(done):
    with open(PROGRESS_FILE, "w") as f:
        json.dump([list(x) for x in done], f)


def append_to_output(df):
    write_header = not os.path.exists(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)


def fetch_one(lat_min, lat_max, lon_min, lon_max, start, end, param):
    params = {
        "parameters": param,
        "community": "AG",
        "format": "JSON",
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "start": start,
        "end": end,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}: {resp.text[:300]}")
                time.sleep(10 * (attempt + 1))
                continue
            data = resp.json()
            rows = []
            for feature in data["features"]:
                lon, lat = feature["geometry"]["coordinates"][:2]
                values = feature["properties"]["parameter"][param]
                for date, val in values.items():
                    rows.append({"lat": lat, "lon": lon, "date": date, param: val})
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"    error: {e}")
            time.sleep(10 * (attempt + 1))
    return None


def main():
    tiles = pd.read_csv(TILES_FILE)
    done = load_progress()
    print(f"{len(tiles)} tiles x {len(DATE_CHUNKS)} date chunks x "
          f"{len(PARAMETERS)} parameters = "
          f"{len(tiles) * len(DATE_CHUNKS) * len(PARAMETERS)} total calls")
    print(f"{len(done)} already completed, resuming...\n")

    total_calls = len(tiles) * len(DATE_CHUNKS) * len(PARAMETERS)
    call_num = 0

    for _, tile in tiles.iterrows():
        lat_min, lon_min = tile["lat_tile"], tile["lon_tile"]
        lat_max, lon_max = lat_min + TILE_SIZE, lon_min + TILE_SIZE

        for chunk_idx, (start, end) in enumerate(DATE_CHUNKS):
            # merge all 4 parameters for this tile+chunk before saving,
            # so each saved row already has all 4 variables together
            chunk_key_base = (lat_min, lon_min, chunk_idx)
            merged = None
            all_params_done = all(
                (lat_min, lon_min, chunk_idx, p) in done for p in PARAMETERS
            )
            if all_params_done:
                continue

            for param in PARAMETERS:
                call_num += 1
                key = (lat_min, lon_min, chunk_idx, param)
                if key in done:
                    continue

                print(f"[{call_num}/{total_calls}] tile ({lat_min},{lon_min}) "
                      f"chunk {chunk_idx} param {param}")
                df = fetch_one(lat_min, lat_max, lon_min, lon_max, start, end, param)

                if df is None:
                    print(f"    FAILED permanently, will retry on next run")
                    continue

                if merged is None:
                    merged = df
                else:
                    merged = merged.merge(df, on=["lat", "lon", "date"], how="outer")

                done.add(key)
                save_progress(done)
                time.sleep(REQUEST_DELAY_SEC)

            if merged is not None:
                append_to_output(merged)

    final_done_tiles = len({(k[0], k[1]) for k in done})
    print(f"\nDone. {len(done)}/{total_calls} calls completed across "
          f"{final_done_tiles} tiles.")
    print(f"Output: {OUTPUT_FILE}")
    print("If any calls failed, just rerun this script -- it resumes automatically.")


if __name__ == "__main__":
    main()