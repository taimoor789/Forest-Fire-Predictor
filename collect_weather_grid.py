import pandas as pd
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import time 
from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

MAX_RETRIES = 3 #Number of times to retry failed requests
RETRY_DELAY = 5  #Seconds to wait between retries
REQUEST_TIMEOUT = 10 #Seconds before request times out

# Load API key with validation
API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable not set")

# Read nearest station mapping
try:
    mapping_df = pd.read_csv("stations.csv")
    logger.info(f"Loaded {len(mapping_df)} grid cells from stations.csv")
except FileNotFoundError:
    logger.error("stations.csv not found!")
    raise
except Exception as e:
    logger.error(f"Error reading stations.csv: {e}")
    raise

# Station coordinates
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

def get_weather(lat, lon, retry=0):
    """
    Fetch weather data with proper error handling and retries.
    Returns dict of weather values or None on failure.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        #Make HTTP GET request with timeout
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        
        #Success, parse and return weather data
        if response.status_code == 200: 
            data = response.json() #convert json data to dict
            
            #Extract data with safe defaults
            #If key doesn't exist, returns {} instead of throwing KeyError
            main = data.get("main", {}) or {}
            wind = data.get("wind", {}) or {}
            clouds = data.get("clouds", {}) or {}
            rain = data.get("rain", {}) or {}
            snow = data.get("snow", {}) or {}
            weather_list = data.get("weather", [])
            weather_main = weather_list[0].get("main") if weather_list else "Unknown"
            weather_desc = weather_list[0].get("description") if weather_list else "No description"

            #Return structured weather data dictionary
            return {
                "temperature": main.get("temp", 15.0),
                "feels_like": main.get("feels_like", 15.0),
                "temp_min": main.get("temp_min", 15.0),
                "temp_max": main.get("temp_max", 15.0),
                "pressure": main.get("pressure", 1013.0),
                "humidity": main.get("humidity", 50.0),
                "wind_speed": wind.get("speed", 0.0),
                "wind_deg": wind.get("deg", 0.0),
                "wind_gust": wind.get("gust", 0.0),
                "clouds_pct": clouds.get("all", 0.0),
                "visibility": data.get("visibility", 10000),
                "rain_1h": rain.get("1h", 0.0),
                "rain_3h": rain.get("3h", 0.0),
                "snow_1h": snow.get("1h", 0.0),
                "snow_3h": snow.get("3h", 0.0),
                "weather_main": weather_main,
                "weather_description": weather_desc,
                "timestamp_utc": data.get("dt", int(time.time()))
            }
            
        elif response.status_code == 429:  # Rate limited
            if retry < MAX_RETRIES:
                wait_time = RETRY_DELAY * (retry + 1)  
                logger.warning(f"Rate limited for {lat},{lon}. Retrying in {wait_time}s... (attempt {retry + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                return get_weather(lat, lon, retry + 1) # Recursive retry
            else:
                logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries for {lat},{lon}")
                return None
     
        elif response.status_code == 401:
            logger.error("Invalid API key! Check OPENWEATHER_API_KEY")
            return None
            
        elif response.status_code >= 500:
            if retry < MAX_RETRIES:
                logger.warning(f"Server error {response.status_code} for {lat},{lon}. Retrying...")
                time.sleep(RETRY_DELAY)
                return get_weather(lat, lon, retry + 1)
            else:
                logger.error(f"Server error after {MAX_RETRIES} retries for {lat},{lon}")
                return None
        else:
            logger.error(f"Unexpected status code {response.status_code} for {lat},{lon}")
            return None
    
    #Retry on network timeouts
    except requests.Timeout:
        if retry < MAX_RETRIES:
            logger.warning(f"Timeout for {lat},{lon}. Retrying... (attempt {retry + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            return get_weather(lat, lon, retry + 1)
        else:
            logger.error(f"Timeout after {MAX_RETRIES} retries for {lat},{lon}")
            return None

    #Network errors: Connection issues, DNS failures  
    except requests.RequestException as e:
        logger.error(f"Request error for {lat},{lon}: {e}")
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return get_weather(lat, lon, retry + 1)
        return None
    
    #Catch all for any other issues
    except Exception as e:
        logger.error(f"Unexpected error getting weather for {lat},{lon}: {e}")
        return None

#Get unique stations (avoid duplicate API calls)
unique_stations = mapping_df["nearest_station_name"].unique()
logger.info(f"Fetching weather for {len(unique_stations)} unique stations")

#Initialize tracking dictionaries
station_weather = {}
successful_fetches = 0
failed_fetches = 0

#Loop through each station and fetch weather
for i, station in enumerate(unique_stations, 1):
    if station in station_coords:
        lat, lon = station_coords[station]
        logger.info(f"Fetching weather for {station} ({i}/{len(unique_stations)})...")
        
        weather = get_weather(lat, lon)
        
        if weather:
            station_weather[station] = weather
            successful_fetches += 1
        else:
            logger.warning(f"Failed to get weather for {station}")
            failed_fetches += 1
            
         #Small delay to avoid hammering the API
        time.sleep(0.1)
    else:
        logger.warning(f"No coordinates found for station: {station}")
        failed_fetches += 1

logger.info(f"Weather fetch complete: {successful_fetches} successful, {failed_fetches} failed")

if successful_fetches == 0:
    logger.error("CRITICAL: No weather data retrieved! Exiting.")
    raise RuntimeError("Failed to retrieve any weather data")

#Get Calgary time
calgary_tz = ZoneInfo("America/Edmonton")
calgary_time = datetime.now(calgary_tz).strftime("%Y-%m-%d %H:%M:%S")

#Map weather to grid cells
grid_weather = []

#Iterate through each grid cell from mapping file
for _, row in mapping_df.iterrows():
    station = row["nearest_station_name"]

    #If we have weather data for this station, use it
    if station in station_weather:
        w = station_weather[station]
        
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
        #Station failed - use None values
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

#Ensure output directory exists
os.makedirs("weather_data", exist_ok=True)

#Create filename with today's date
today_str = datetime.now().strftime("%Y-%m-%d")
output_path = f"weather_data/{today_str}.csv"

#Write to CSV
try:
    df_output = pd.DataFrame(grid_weather)
    df_output.to_csv(output_path, index=False)
    logger.info(f"✓ Saved weather for {len(grid_weather)} grid cells to {output_path}")
    logger.info(f"✓ Data timestamp: {calgary_time}")
except Exception as e:
    logger.error(f"Failed to save CSV: {e}")
    raise
