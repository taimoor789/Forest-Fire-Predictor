import pandas as pd
import json
import os
import glob
from datetime import datetime
import subprocess
import sys
from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def update_weather_conditions():
    """
    Update current weather conditions without recalculating FWI.
    """
    try:
        # Fetch fresh weather data
        logger.info("Fetching current weather data...")
        result = subprocess.run(
            [sys.executable, "collect_weather_grid.py"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        logger.info(" Weather data collected")
        
        # Load existing FWI predictions
        if not os.path.exists("fwi_predictions.json"):
            logger.error("No existing FWI predictions found. Run daily_update.py first.")
            return False
        
        with open("fwi_predictions.json", "r") as f:
            fwi_data = json.load(f)
        
        logger.info(f"Loaded {len(fwi_data['data'])} existing FWI predictions")
        
        # Load fresh weather data
        weather_files = glob.glob("weather_data/*.csv")
        if not weather_files:
            logger.error("No weather data files found")
            return False
        
        latest_weather_file = max(weather_files, key=os.path.getctime)
        current_weather = pd.read_csv(latest_weather_file)
        logger.info(f"Loaded current weather from {latest_weather_file}")
        
        # Update weather fields in predictions
        updated_count = 0
        for prediction in fwi_data['data']:
            lat = prediction['lat']
            lon = prediction['lon']
            
            # Find matching weather data
            weather_match = current_weather[
                (current_weather['lat'] == lat) & 
                (current_weather['lon'] == lon)
            ]
            
            if len(weather_match) > 0:
                weather_row = weather_match.iloc[0]
                
                # Update only weather fields, preserve FWI values
                prediction['weather_features']['temperature'] = float(weather_row.get('temperature', 15))
                prediction['weather_features']['humidity'] = float(weather_row.get('humidity', 50))
                prediction['weather_features']['wind_speed'] = float(weather_row.get('wind_speed', 10))
                prediction['weather_features']['pressure'] = float(weather_row.get('pressure', 1013))
                prediction['weather_features']['rain_1h_mm'] = float(weather_row.get('rain_1h_mm', 0))
                prediction['weather_features']['rain_3h_mm'] = float(weather_row.get('rain_3h_mm', 0))
                prediction['weather_features']['snow_1h_mm'] = float(weather_row.get('snow_1h_mm', 0))
                prediction['weather_features']['snow_3h_mm'] = float(weather_row.get('snow_3h_mm', 0))
                
                # Update derived weather features
                temp = prediction['weather_features']['temperature']
                humidity = prediction['weather_features']['humidity']
                wind = prediction['weather_features']['wind_speed']
                total_precip = (prediction['weather_features']['rain_1h_mm'] + 
                               prediction['weather_features']['rain_3h_mm'] + 
                               prediction['weather_features']['snow_1h_mm'] + 
                               prediction['weather_features']['snow_3h_mm'])
                
                prediction['weather_features']['is_hot'] = 1 if temp > 25 else 0
                prediction['weather_features']['is_dry'] = 1 if humidity < 30 else 0
                prediction['weather_features']['humidity_temp_ratio'] = humidity / (temp + 1)
                prediction['weather_features']['is_windy'] = 1 if wind > 15 else 0
                prediction['weather_features']['total_precip'] = total_precip
                prediction['weather_features']['has_recent_precip'] = 1 if total_precip > 0 else 0
                
                updated_count += 1
        
        logger.info(f" Updated weather for {updated_count} locations")
        
        # Update metadata timestamps
        weather_update_time = datetime.now().isoformat()
        fwi_data['weather_last_updated'] = weather_update_time
        fwi_data['last_updated'] = weather_update_time  
        fwi_data['timestamp'] = weather_update_time      

        # Add update type to distinguish from full FWI recalculation
        fwi_data['last_update_type'] = 'weather_only'
        
        # Preserve original FWI calculation timestamp
        if 'fwi_calculated_at' not in fwi_data:
            fwi_data['fwi_calculated_at'] = fwi_data.get('last_updated', weather_update_time)
        
        # Save updated data
        with open("fwi_predictions.json", "w") as f:
            json.dump(fwi_data, f, indent=2)
        
        logger.info("=" * 70)
        logger.info("Weather Update Complete!")
        logger.info(f"Updated {updated_count} locations")
        logger.info(f"Weather timestamp: {weather_update_time}")
        logger.info(f"FWI values preserved from: {fwi_data.get('fwi_calculated_at', 'unknown')}")
        logger.info("=" * 70)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Weather collection failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Weather update failed: {str(e)}")
        return False

def main():
    """Main entry point for weather-only updates"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--weather-only':
        success = update_weather_conditions()
        sys.exit(0 if success else 1)
    else:
        print("Usage: python update_weather.py --weather-only")
        sys.exit(1)

if __name__ == "__main__":
    main()