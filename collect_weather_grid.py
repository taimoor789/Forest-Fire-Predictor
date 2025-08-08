#Need to assign station weather to all grid cells mapped to it and fetch weather for each station.

import pandas as pd
import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import os

#load api key
with open("config.json") as f:
    config = json.load(f)

API_KEY = config["openweather_api_key"]

 #Read nearest station mapping
mapping_df = pd.read_csv("stations.csv") 

#List of unique station names
unique_stations = mapping_df["nearest_station_name"].unique()


# Keep this in sync with the one in station_mapping.py
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
def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }
        else:
            print(f"Failed for {lat},{lon} — Status code {response.status_code}")
            return None
    except Exception as e:
        print(f" Error for {lat},{lon}: {e}")
        return None

#fetch weather for each station
station_weather = {}
for station in unique_stations: #for every station
    if station in station_coords: #if the station is in the station coords dict above
        lat, lon = station_coords[station] #use the dict's key to get the lat/lon from the tuple
        weather = get_weather(lat, lon) #use that lat/lon in the get_weather function
        if weather: #if there's weather data
            station_weather[station] = weather #store it in station_weather using dict key
    else:
        print(f" No coordinates found for {station}")

#Assign weather to all grid cells
calgary_tz = ZoneInfo("America/Edmonton")
calgary_time = datetime.now(calgary_tz).strftime("%Y-%m-%d %H:%M:%S")

#creating dict using all cells with all weather 
#info required
grid_weather = []
for _, row in mapping_df.iterrows():
    station = row["nearest_station_name"]
    if station in station_weather:
        grid_weather.append({
            "lat": row["lat"],
            "lon": row["lon"],
            "date": calgary_time,
            "temperature": station_weather[station]["temperature"],
            "humidity": station_weather[station]["humidity"],
            "wind_speed": station_weather[station]["wind_speed"]
        })

#save to csv
output_file = "grid_weather_history.csv"

# If file doesn't exist, create it with header
if not os.path.exists(output_file):
    pd.DataFrame(columns=["lat", "lon", "date", "temperature", "humidity", "wind_speed"]).to_csv(output_file, index=False)

#Append today's data
#pd.DataFrame(grid_weather).to_csv(output_file, mode="a", header=False, index=False)

#Make sure the folder exists
os.makedirs("weather_data", exist_ok=True)

#Save today's data as a separate CSV file
today_str = datetime.now().strftime("%Y-%m-%d")
output_path = f"weather_data/{today_str}.csv"

pd.DataFrame(grid_weather).to_csv(output_path, index=False)

print(f"Saved weather for {len(grid_weather)} grid cells at {calgary_time}")