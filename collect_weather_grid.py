#Need to assign station weather to all grid cells mapped to it and fetch weather for each station.

import pandas as pd
import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import time 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5

#load api key 
API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable not set")

#Read nearest station mapping 
mapping_df = pd.read_csv("stations.csv")

#List of unique station names 
unique_stations = mapping_df["nearest_station_name"].unique()

#Keep this in sync with the one in station_mapping.py 
station_coords = {
    "Vancouver": (49.2827, -123.1207),
    "Kelowna": (49.8880, -119.4960),
    "Kamloops": (50.6745, -120.3273),
    "Calgary": (51.0447, -114.0719),
    "Edmonton": (53.5461, -113.4938),
    "Fort McMurray": (56.7266, -111.3790),
    "Saskatoon": (52.1579, -106.6702),
    "Regina": (50.4452, -104.6189),
    "Winnipeg": (49.8951, -97.1384),
    "Thunder Bay": (48.3809, -89.2477),
    "Ottawa": (45.4215, -75.6972),
    "Toronto": (43.6510, -79.3470),
    "Sudbury": (46.4917, -80.9930),
    "Montreal": (45.5019, -73.5674),
    "Quebec City": (46.8139, -71.2080),
    "Halifax": (44.6488, -63.5752),
    "Whitehorse": (60.7212, -135.0568),
    "Yellowknife": (62.4540, -114.3718),
    "Prince George": (53.9171, -122.7497),
    "Victoria": (48.4284, -123.3656),
    "Smithers": (54.7800, -127.1743),
    "Dease Lake": (58.4356, -130.0089),
    "Fort St. John": (56.2524, -120.8466),
    "High Level": (58.5169, -117.1360),
    "Peace River": (56.2333, -117.2833),
    "La Ronge": (55.1000, -105.3000),
    "Flin Flon": (54.7682, -101.8779),
    "Churchill": (58.7684, -94.1650),
    "Moosonee": (51.2794, -80.6463),
    "Timmins": (48.4758, -81.3305),
    "Val-d'Or": (48.1086, -77.7972),
    "Chibougamau": (49.9167, -74.3667),
    "Schefferville": (54.8000, -66.8167),
    "Goose Bay": (53.3019, -60.3267),
    "St. John's": (47.5615, -52.7126),
    "Iqaluit": (63.7467, -68.5170),
    "Rankin Inlet": (62.8090, -92.0853),
    "Cambridge Bay": (69.1167, -105.0667)
}

#function to get weather for a station
def get_weather(lat, lon, retry=0):
    #Returns a dict of values 
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=6)
        if response.status_code == 200:
            data = response.json()

            # main: temp, feels_like, pressure, humidity, temp_min/max
            main = data.get("main", {})
            # wind: speed, deg, gust 
            wind = data.get("wind", {}) or {}
            # clouds: percentage
            clouds = data.get("clouds", {}) or {}
            # precipitation: "rain" and "snow" fields may or may not exist
            rain = data.get("rain", {}) or {}
            snow = data.get("snow", {}) or {}
            # "weather" is a list of condition objects; take first for main/description
            weather_list = data.get("weather", [])
            weather_main = weather_list[0].get("main") if weather_list else None
            weather_desc = weather_list[0].get("description") if weather_list else None

            # Build the returned dictionary
            return {
                "temperature": main.get("temp"),                    # °C
                "feels_like": main.get("feels_like"),               # °C
                "temp_min": main.get("temp_min"),                   # °C
                "temp_max": main.get("temp_max"),                   # °C
                "pressure": main.get("pressure"),                   # hPa
                "humidity": main.get("humidity"),                   # %
                "wind_speed": wind.get("speed"),                    # m/s
                "wind_deg": wind.get("deg"),                        # degrees
                "wind_gust": wind.get("gust"),                      # m/s 
                "clouds_pct": clouds.get("all"),                    # %
                "visibility": data.get("visibility"),               # meters 
                "rain_1h": rain.get("1h", 0.0),                     # mm in last 1h (0 if missing)
                "rain_3h": rain.get("3h", 0.0),                     # mm in last 3h (0 if missing)
                "snow_1h": snow.get("1h", 0.0),                     # mm in last 1h (0 if missing)
                "snow_3h": snow.get("3h", 0.0),                     # mm in last 3h (0 if missing)
                "weather_main": weather_main,                       
                "weather_description": weather_desc,              
                "timestamp_utc": data.get("dt")                     
            }
        elif response.status_code == 429:  # Rate limited
            if retry < MAX_RETRIES:
                logger.warning(f"Rate limited, retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                return get_weather(lat, lon, retry + 1)
            else:
                logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries")
                return None
        else:
            logger.error(f"Status code {response.status_code}")
            return None
    except requests.Timeout:
        if retry < MAX_RETRIES:
            logger.warning(f"Timeout, retrying...")
            time.sleep(RETRY_DELAY)
            return get_weather(lat, lon, retry + 1)
        return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

# fetch weather for each station (once per unique station)
station_weather = {}
for station in unique_stations:  # for every station referenced by mapping.csv
    if station in station_coords:  # if the station is in the station_coords dict above
        lat, lon = station_coords[station]  # get lat/lon from the dict
        weather = get_weather(lat, lon)    
        if weather:                        
            station_weather[station] = weather
    else:
        print(f"No coordinates found for station: {station}")

calgary_tz = ZoneInfo("America/Edmonton")
calgary_time = datetime.now(calgary_tz).strftime("%Y-%m-%d %H:%M:%S")

#Create output rows
grid_weather = []
for _, row in mapping_df.iterrows():
    station = row["nearest_station_name"]
    if station in station_weather:
        w = station_weather[station]  #the dict returned by get_weather

        # Append a row with the expanded set of fields
        grid_weather.append({
            "lat": row["lat"],
            "lon": row["lon"],
            "date": calgary_time,                      
            "nearest_station": station,                 
            "temperature": w.get("temperature"),
            "humidity": w.get("humidity"),
            "wind_speed": w.get("wind_speed"),
            "feels_like": w.get("feels_like"),
            "temp_min": w.get("temp_min"),
            "temp_max": w.get("temp_max"),
            "pressure": w.get("pressure"),
            "wind_deg": w.get("wind_deg"),
            "wind_gust": w.get("wind_gust"),
            "clouds_pct": w.get("clouds_pct"),
            "visibility_m": w.get("visibility"),
            "rain_1h_mm": w.get("rain_1h"),
            "rain_3h_mm": w.get("rain_3h"),
            "snow_1h_mm": w.get("snow_1h"),
            "snow_3h_mm": w.get("snow_3h"),
            "weather_main": w.get("weather_main"),
            "weather_description": w.get("weather_description"),
            "timestamp_utc": w.get("timestamp_utc")
        })
    else:
        # If there's no station weather (API failed), append row with None values
        grid_weather.append({
            "lat": row["lat"],
            "lon": row["lon"],
            "date": calgary_time,
            "nearest_station": station,
            "temperature": None,
            "humidity": None,
            "wind_speed": None,
            "feels_like": None,
            "temp_min": None,
            "temp_max": None,
            "pressure": None,
            "wind_deg": None,
            "wind_gust": None,
            "clouds_pct": None,
            "visibility_m": None,
            "rain_1h_mm": None,
            "rain_3h_mm": None,
            "snow_1h_mm": None,
            "snow_3h_mm": None,
            "weather_main": None,
            "weather_description": None,
            "timestamp_utc": None
        })

#Make sure the folder exists 
os.makedirs("weather_data", exist_ok=True)

#Save today's data as a separate CSV file 
today_str = datetime.now().strftime("%Y-%m-%d")
output_path = f"weather_data/{today_str}.csv"

#Convert to DataFrame and write CSV
pd.DataFrame(grid_weather).to_csv(output_path, index=False)

print(f"Saved weather for {len(grid_weather)} grid cells at {calgary_time}")