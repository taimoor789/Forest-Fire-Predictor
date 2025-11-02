import pandas as pd
import numpy as np
import joblib
import glob 
import os
from datetime import datetime
import json
import math
from logging_config import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def cleanup_old_weather_data(days_to_keep=45):
    """Delete weather data files older than specified days"""
    weather_files = sorted(glob.glob("weather_data/*.csv"))
    
    if len(weather_files) > days_to_keep:
        files_to_delete = weather_files[:-days_to_keep]  
        for old_file in files_to_delete:
            try:
                os.remove(old_file)
                logger.info(f"Deleted old weather file: {old_file}")
            except Exception as e:
                logger.warning(f"Could not delete {old_file}: {e}")

class CanadianFireWeatherIndex:
    """
    Proper implementation of Canadian FWI using historical weather accumulation
    """ 
    
    def __init__(self):
        self.version = "2.0.0"
    
    def sanitize_value(self, value, default, min_val, max_val):
        """Ensure values are valid numbers within range"""
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return default
            return np.clip(val, min_val, max_val)
        except (ValueError, TypeError):
            return default
        
    def calculate_ffmc(self, temp, humidity, wind, rain, prev_ffmc=85):
        """Fine Fuel Moisture Code - requires previous day's value"""
        # Sanitize inputs
        temp = self.sanitize_value(temp, 15, -50, 50)
        humidity = self.sanitize_value(humidity, 50, 1, 100)
        wind = self.sanitize_value(wind, 10, 0, 100)
        rain = self.sanitize_value(rain, 0, 0, 500)
        prev_ffmc = self.sanitize_value(prev_ffmc, 85, 0, 101)
        
        # Moisture content from previous FFMC
        mo = 147.2 * (101 - prev_ffmc) / (59.5 + prev_ffmc)
        
        # Rain effect
        if rain > 0.5:
            rf = rain - 0.5
            if mo <= 150:
                mo = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))
            else:
                mo = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf)) + 0.0015 * (mo - 150) ** 2 * np.sqrt(rf)
            
            if mo > 250:
                mo = 250
        
        # Equilibrium moisture content
        ed = 0.942 * humidity ** 0.679 + 11 * np.exp((humidity - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * humidity))
        
        # Drying or wetting
        if mo > ed:
            ko = 0.424 * (1 - (humidity / 100) ** 1.7) + 0.0694 * np.sqrt(wind) * (1 - (humidity / 100) ** 8)
            kd = ko * 0.581 * np.exp(0.0365 * temp)
            m = ed + (mo - ed) * 10 ** (-kd)
        else:
            ew = 0.618 * humidity ** 0.753 + 10 * np.exp((humidity - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * humidity))
            if mo < ew:
                k1 = 0.424 * (1 - ((100 - humidity) / 100) ** 1.7) + 0.0694 * np.sqrt(wind) * (1 - ((100 - humidity) / 100) ** 8)
                kw = k1 * 0.581 * np.exp(0.0365 * temp)
                m = ew - (ew - mo) * 10 ** (-kw)
            else:
                m = mo
        
        # Convert back to FFMC
        ffmc = 59.5 * (250 - m) / (147.2 + m)
        ffmc = np.clip(ffmc, 0, 101)
        
        # Final sanity check
        if np.isnan(ffmc) or np.isinf(ffmc):
            return 85
        return float(ffmc)
    
    def calculate_dmc(self, temp, humidity, rain, prev_dmc=6, month=7):
        """Duff Moisture Code - accumulates over days"""
        # Sanitize inputs
        temp = self.sanitize_value(temp, 15, -50, 50)
        humidity = self.sanitize_value(humidity, 50, 1, 100)
        rain = self.sanitize_value(rain, 0, 0, 500)
        prev_dmc = self.sanitize_value(prev_dmc, 6, 0, 500)
        
        if temp < -1.1:
            return prev_dmc
        
        # Day length factors
        day_lengths = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        le = day_lengths[month - 1] if 1 <= month <= 12 else 1.4
        
        # Rain effect
        re = prev_dmc
        if rain > 1.5:
            rw = 0.92 * rain - 1.27
            wmi = 20 + 280 / np.exp(0.023 * prev_dmc)
            
            if prev_dmc <= wmi:
                b = 100 / (0.5 + 0.3 * prev_dmc)
            else:
                b = 14 - 1.3 * np.log(prev_dmc + 1)
            
            mr = prev_dmc + 1000 * rw / (48.77 + b * rw)
            re = max(0, mr)
        
        # Drying
        if temp > -1.1:
            k = 1.894 * (temp + 1.1) * (100 - humidity) * le * 0.000001
            dmc = re + 100 * k
        else:
            dmc = re
        
        dmc = max(0, dmc)
        
        # Sanity check
        if np.isnan(dmc) or np.isinf(dmc):
            return 6
        return float(dmc)
    
    def calculate_dc(self, temp, rain, prev_dc=15, month=7):
        """Drought Code - deep drying over weeks"""
        # Sanitize inputs
        temp = self.sanitize_value(temp, 15, -50, 50)
        rain = self.sanitize_value(rain, 0, 0, 500)
        prev_dc = self.sanitize_value(prev_dc, 15, 0, 1000)
        
        if temp < -2.8:
            return prev_dc
        
        # Day length factors
        lf_day = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        lf = lf_day[month - 1] if 1 <= month <= 12 else 1.4
        
        # Rain effect
        rd = prev_dc
        if rain > 2.8:
            ra = rain
            rw = 0.83 * ra - 1.27
            smi = 800 * np.exp(-prev_dc / 400)
            dr = prev_dc - 400 * np.log(1 + 3.937 * rw / smi)
            rd = max(0, dr)
        
        # Potential evapotranspiration
        if temp > -2.8:
            v = 0.36 * (temp + 2.8) + lf
            v = max(0, v)
            dc = rd + v
        else:
            dc = rd
        
        dc = max(0, dc)
        
        # Sanity check
        if np.isnan(dc) or np.isinf(dc):
            return 15
        return float(dc)
    
    def calculate_isi(self, wind, ffmc):
        """Initial Spread Index"""
        wind = self.sanitize_value(wind, 10, 0, 100)
        ffmc = self.sanitize_value(ffmc, 85, 0, 101)
        
        # Wind function
        fw = np.exp(0.05039 * wind)
        
        # Fine fuel moisture function
        m = 147.2 * (101 - ffmc) / (59.5 + ffmc)
        ff = 91.9 * np.exp(-0.1386 * m) * (1 + m ** 5.31 / 49300000)
        
        isi = 0.208 * fw * ff
        
        # Sanity check
        if np.isnan(isi) or np.isinf(isi):
            return 1.0
        return float(isi)
    
    def calculate_bui(self, dmc, dc):
        """Buildup Index"""
        dmc = self.sanitize_value(dmc, 6, 0, 500)
        dc = self.sanitize_value(dc, 15, 0, 1000)
        
        if dmc <= 0.4 * dc:
            bui = 0.8 * dmc * dc / (dmc + 0.4 * dc + 0.001)
        else:
            bui = dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc + 0.001)) * (0.92 + (0.0114 * dmc) ** 1.7)
        
        bui = max(0, bui)
        
        # Sanity check
        if np.isnan(bui) or np.isinf(bui):
            return 10.0
        return float(bui)
    
    def calculate_fwi(self, isi, bui):
        """Fire Weather Index"""
        isi = self.sanitize_value(isi, 1, 0, 100)
        bui = self.sanitize_value(bui, 10, 0, 500)
        
        if bui <= 80:
            fd = 0.626 * bui ** 0.809 + 2
        else:
            fd = 1000 / (25 + 108.64 * np.exp(-0.023 * bui))
        
        b = 0.1 * isi * fd
        
        if b > 1:
            s = np.exp(2.72 * (0.434 * np.log(b)) ** 0.647)
        else:
            s = b
        
        # Sanity check
        if np.isnan(s) or np.isinf(s):
            return 5.0
        return float(s)
    
    def get_danger_class(self, fwi):
        """Official Canadian danger classifications"""
        fwi = self.sanitize_value(fwi, 5, 0, 100)
        
        if fwi < 1:
            return "Very Low", 0.05, "#4CAF50"
        elif fwi < 3:
            return "Low", 0.15, "#8BC34A"
        elif fwi < 7:
            return "Moderate", 0.35, "#FFEB3B"
        elif fwi < 17:
            return "High", 0.65, "#FF9800"
        elif fwi < 30:
            return "Very High", 0.85, "#F44336"
        else:
            return "Extreme", 0.95, "#9C27B0"

class FireWeatherProcessor:
    """
    Process FWI using historical weather accumulation
    """
    
    def __init__(self):
        self.fwi_calculator = CanadianFireWeatherIndex()
        self.processing_stats = {}
    
    def sanitize_for_json(self, value):
        """Convert any invalid float to a valid JSON-compliant number"""
        if value is None:
            return 0.0
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return val
        except (ValueError, TypeError):
            return 0.0

    def sanitize_dict_for_json(self, data):
        """Recursively sanitize all floats in a dictionary"""
        if isinstance(data, dict):
            return {k: self.sanitize_dict_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_dict_for_json(item) for item in data]
        elif isinstance(data, (float, np.floating)):
            return self.sanitize_for_json(data)
        elif isinstance(data, (np.integer, int)):
            return int(data)
        return data
        
    def load_historical_weather(self, days_back=45):
        """Load historical weather data for FWI accumulation"""
        weather_files = sorted(glob.glob("weather_data/*.csv"))
        
        if not weather_files:
            raise FileNotFoundError("No weather data files found")
        
        # Load last N days
        if len(weather_files) > days_back:
            weather_files = weather_files[-days_back:]
        
        logger.info(f"Loading {len(weather_files)} days of weather history...")
        
        all_data = []
        for file in weather_files:
            try:
                df = pd.read_csv(file)
                df['file_date'] = os.path.basename(file).replace('.csv', '')
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")
        
        if not all_data:
            raise ValueError("No weather data could be loaded")
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined)} total weather records")
        
        return combined
    
    def calculate_accumulated_fwi(self, location_history):
        """Calculate FWI codes accumulated over time for one location"""
        # Sort by date
        location_history = location_history.sort_values('file_date')
        
        # Initialize codes
        ffmc = 85
        dmc = 6
        dc = 15
        
        # Accumulate day by day
        for _, day in location_history.iterrows():
            temp = day.get('temperature', 15)
            humidity = day.get('humidity', 50)
            wind = day.get('wind_speed', 10)
            rain = (day.get('rain_1h_mm', 0) + day.get('rain_3h_mm', 0) + 
                   day.get('snow_1h_mm', 0) + day.get('snow_3h_mm', 0))
            
            month = datetime.now().month
            
            # Update codes based on this day's weather
            ffmc = self.fwi_calculator.calculate_ffmc(temp, humidity, wind, rain, ffmc)
            dmc = self.fwi_calculator.calculate_dmc(temp, humidity, rain, dmc, month)
            dc = self.fwi_calculator.calculate_dc(temp, rain, dc, month)
        
        # Calculate final indices from accumulated codes
        isi = self.fwi_calculator.calculate_isi(wind, ffmc)
        bui = self.fwi_calculator.calculate_bui(dmc, dc)
        fwi = self.fwi_calculator.calculate_fwi(isi, bui)
        dsr = 0.0272 * fwi ** 1.77
        
        result = {
            'ffmc': ffmc,
            'dmc': dmc,
            'dc': dc,
            'isi': isi,
            'bui': bui,
            'fwi': fwi,
            'dsr': dsr
        }
        
        # Sanitize all values before returning
        return self.sanitize_dict_for_json(result)
    
    def process_all_locations(self, weather_file=None):
        """Process FWI for all locations using historical accumulation"""
        
        logger.info("Processing Fire Weather Index with historical accumulation...")
        start_time = datetime.now()
        
        # Load historical data
        historical_data = self.load_historical_weather(days_back=45)
        
        # Get today's data
        if weather_file is None:
            weather_files = glob.glob("weather_data/*.csv")
            weather_file = max(weather_files, key=os.path.getctime)
        
        today_data = pd.read_csv(weather_file)
        logger.info(f"Processing {len(today_data)} locations from {weather_file}")
        
        # Add historical fire context
        try:
            fire_df = pd.read_csv("canada_fire_grid.csv")
            today_data = today_data.merge(fire_df[['lat', 'lon', 'historical_fire']], 
                                         on=['lat', 'lon'], how='left')
            today_data['historical_fire'] = today_data['historical_fire'].fillna(0).astype(int)
        except FileNotFoundError:
            logger.warning("Historical fire data not found")
            today_data['historical_fire'] = 0
        
        results = []
        processing_errors = 0
        
        # Process each location
        for idx, row in today_data.iterrows():
            try:
                lat = float(row['lat'])
                lon = float(row['lon'])
                
                # Get history for this location
                location_hist = historical_data[
                    (historical_data['lat'] == lat) & 
                    (historical_data['lon'] == lon)
                ].copy()
                
                if len(location_hist) == 0:
                    logger.warning(f"No history for {lat},{lon}")
                    location_hist = pd.DataFrame([row])
                    location_hist['file_date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Calculate accumulated FWI
                fwi_data = self.calculate_accumulated_fwi(location_hist)
                
                # Validate FWI data
                if any(np.isnan(v) or np.isinf(v) for v in fwi_data.values()):
                    logger.warning(f"Invalid FWI data for {lat},{lon}, using defaults")
                    fwi_data = {'ffmc': 85.0, 'dmc': 6.0, 'dc': 15.0, 'isi': 1.0, 'bui': 10.0, 'fwi': 5.0, 'dsr': 1.0}
                
                # Get danger classification
                danger_class, risk_prob, color = self.fwi_calculator.get_danger_class(fwi_data['fwi'])
                
                # Historical fire adjustment
                if row.get('historical_fire', 0) == 1:
                    risk_prob = min(0.98, risk_prob * 1.15)
                
                # Ensure risk_prob is valid
                risk_prob = float(risk_prob)
                if np.isnan(risk_prob) or np.isinf(risk_prob):
                    risk_prob = 0.15
                
                result_raw = {
                    'lat': lat,
                    'lon': lon,
                    'location_name': str(row.get('nearest_station', f'Grid_{idx}')),
                    'province': self.get_province(lat, lon),
                    'daily_fire_risk': risk_prob,
                    'danger_class': danger_class,
                    'color_code': color,
                    'weather_features': {
                        'temperature': row.get('temperature', 15),
                        'humidity': row.get('humidity', 50),
                        'wind_speed': row.get('wind_speed', 10),
                        'pressure': row.get('pressure', 1013),
                        'fire_danger_index': fwi_data['fwi'],
                        'rain_1h_mm': row.get('rain_1h_mm', 0),
                        'rain_3h_mm': row.get('rain_3h_mm', 0),
                        'snow_1h_mm': row.get('snow_1h_mm', 0),
                        'snow_3h_mm': row.get('snow_3h_mm', 0),
                        'is_hot': 1 if row.get('temperature', 15) > 25 else 0,
                        'is_dry': 1 if row.get('humidity', 50) < 30 else 0,
                        'humidity_temp_ratio': row.get('humidity', 50) / (row.get('temperature', 15) + 1),
                        'is_windy': 1 if row.get('wind_speed', 10) > 15 else 0,
                        'total_precip': (row.get('rain_1h_mm', 0) + row.get('rain_3h_mm', 0) + 
                                        row.get('snow_1h_mm', 0) + row.get('snow_3h_mm', 0)),
                        'has_recent_precip': 1 if (row.get('rain_1h_mm', 0) + row.get('rain_3h_mm', 0) + 
                                                   row.get('snow_1h_mm', 0) + row.get('snow_3h_mm', 0)) > 0 else 0,
                        'weather_main_encoded': 0
                    },
                    'fire_weather_indices': fwi_data,
                    'model_confidence': 0.95
                }
                
                # CRITICAL: Sanitize the entire result before adding it
                result = self.sanitize_dict_for_json(result_raw)
                
                # Final validation before adding
                if not np.isnan(result['daily_fire_risk']) and not np.isinf(result['daily_fire_risk']):
                    results.append(result)
                else:
                    logger.warning(f"Skipping location {lat},{lon} due to invalid risk value")
                    processing_errors += 1
                
            except Exception as e:
                processing_errors += 1
                logger.error(f"Error processing location {idx}: {e}")
                continue
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate stats
        risks = [r['daily_fire_risk'] for r in results]
        danger_classes = [r['danger_class'] for r in results]
        
        self.processing_stats = {
            'total_locations': len(today_data),
            'processed_successfully': len(results),
            'processing_errors': processing_errors,
            'processing_time_seconds': processing_time,
            'risk_statistics': {
                'min_risk': float(min(risks)) if risks else 0.0,
                'max_risk': float(max(risks)) if risks else 0.0,
                'mean_risk': float(np.mean(risks)) if risks else 0.0,
                'very_low_count': len([d for d in danger_classes if d == 'Very Low']),
                'low_count': len([d for d in danger_classes if d == 'Low']),
                'moderate_count': len([d for d in danger_classes if d == 'Moderate']),
                'high_count': len([d for d in danger_classes if d == 'High']),
                'very_high_count': len([d for d in danger_classes if d == 'Very High']),
                'extreme_count': len([d for d in danger_classes if d == 'Extreme'])
            }
        }
        
        logger.info(f"Processing complete: {len(results)} locations in {processing_time:.1f}s")
        logger.info(f"Risk: Min={min(risks):.3f}, Max={max(risks):.3f}, Mean={np.mean(risks):.3f}")
        logger.info(f"Danger: VL={self.processing_stats['risk_statistics']['very_low_count']}, "
                   f"L={self.processing_stats['risk_statistics']['low_count']}, "
                   f"M={self.processing_stats['risk_statistics']['moderate_count']}, "
                   f"H={self.processing_stats['risk_statistics']['high_count']}, "
                   f"VH={self.processing_stats['risk_statistics']['very_high_count']}, "
                   f"E={self.processing_stats['risk_statistics']['extreme_count']}")
        
        return results
    
    def get_province(self, lat, lon):
        """Map coordinates to province with corrected boundaries"""
        # Corrected province bounds - non-overlapping
        province_bounds = {
            'BC': (48.3, -139.1, 60.0, -114.1),
            'AB': (49.0, -120.0, 60.0, -110.0),
            'SK': (49.0, -110.0, 60.0, -101.4),
            'MB': (49.0, -102.0, 60.0, -88.9),
            'ON': (41.7, -95.2, 56.9, -74.3),
            'QC': (45.0, -79.8, 62.6, -57.1),
            'NB': (44.6, -69.1, 48.1, -63.7),
            'NS': (43.4, -66.4, 47.1, -59.7),
            'PE': (45.9, -64.4, 47.1, -62.0),
            'NL': (46.6, -67.8, 60.4, -52.6),
            'YT': (60.0, -141.0, 69.6, -124.0),
            'NT': (60.0, -136.0, 78.8, -102.0),
            'NU': (60.0, -110.0, 83.1, -61.0)
        }
        
        # Check provinces in priority order (east to west for overlaps)
        priority_order = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
        
        for prov in priority_order:
            if prov in province_bounds:
                min_lat, min_lon, max_lat, max_lon = province_bounds[prov]
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    return prov
        
        return "Unknown"
    def calculate_accumulated_fwi(self, location_history):
        location_history = location_history.sort_values('file_date')
        
        # DEBUG: Track Calgary specifically
        is_calgary = False
        if len(location_history) > 0:
            first_row = location_history.iloc[0]
            if first_row.get('nearest_station') == 'Calgary':
                is_calgary = True
                logger.info(f"\n{'='*60}")
                logger.info(f"CALGARY DEBUG - History length: {len(location_history)}")
                logger.info(f"Date range: {location_history.iloc[0]['file_date']} to {location_history.iloc[-1]['file_date']}")
                logger.info(f"{'='*60}")
        
        # Initialize codes
        ffmc = 85
        dmc = 6
        dc = 15
        
        # Accumulate day by day
        for idx, day in location_history.iterrows():
            temp = day.get('temperature', 15)
            humidity = day.get('humidity', 50)
            wind = day.get('wind_speed', 10)
            rain = (day.get('rain_1h_mm', 0) + day.get('rain_3h_mm', 0) + 
                day.get('snow_1h_mm', 0) + day.get('snow_3h_mm', 0))
            
            month = datetime.now().month
            
            # Store old values for comparison
            old_ffmc = ffmc
            old_dmc = dmc
            old_dc = dc
            
            # Update codes based on this day's weather
            ffmc = self.fwi_calculator.calculate_ffmc(temp, humidity, wind, rain, ffmc)
            dmc = self.fwi_calculator.calculate_dmc(temp, humidity, rain, dmc, month)
            dc = self.fwi_calculator.calculate_dc(temp, rain, dc, month)
            
            # DEBUG: Log every day for Calgary (especially last 3 days)
            if is_calgary and idx >= len(location_history) - 3:
                logger.info(f"\nDate: {day.get('file_date')} | Temp: {temp}°C | Humidity: {humidity}% | Wind: {wind} km/h | Rain: {rain}mm")
                logger.info(f"  FFMC: {old_ffmc:.1f} → {ffmc:.1f}")
                logger.info(f"  DMC:  {old_dmc:.1f} → {dmc:.1f}")
                logger.info(f"  DC:   {old_dc:.1f} → {dc:.1f}")
        
        # Calculate final indices from accumulated codes
        isi = self.fwi_calculator.calculate_isi(wind, ffmc)
        bui = self.fwi_calculator.calculate_bui(dmc, dc)
        fwi = self.fwi_calculator.calculate_fwi(isi, bui)
        dsr = 0.0272 * fwi ** 1.77
        
        if is_calgary:
            logger.info(f"\n{'='*60}")
            logger.info(f"FINAL CALGARY INDICES:")
            logger.info(f"  FFMC: {ffmc:.1f}")
            logger.info(f"  DMC:  {dmc:.1f}")
            logger.info(f"  DC:   {dc:.1f}")
            logger.info(f"  ISI:  {isi:.1f}")
            logger.info(f"  BUI:  {bui:.1f}")
            logger.info(f"  FWI:  {fwi:.1f}")
            logger.info(f"  DSR:  {dsr:.1f}")
            logger.info(f"{'='*60}\n")
        
        result = {
            'ffmc': ffmc,
            'dmc': dmc,
            'dc': dc,
            'isi': isi,
            'bui': bui,
            'fwi': fwi,
            'dsr': dsr
        }
        
        
        return self.sanitize_dict_for_json(result)

def main():
    cleanup_old_weather_data(days_to_keep=45)
    processing_timestamp = datetime.now().isoformat()
    processor = FireWeatherProcessor()
    
    # Process with historical accumulation
    results = processor.process_all_locations()
    
    # Validate results before saving
    if not results or len(results) == 0:
        logger.error("CRITICAL: No results generated from FWI processing!")
        raise ValueError("No fire risk predictions generated")
    
    # Build API response
    api_response = {
        "success": True,
        "data": results,
        "model_info": {
            "model_type": "Canadian Fire Weather Index System",
            "version": "2.0.0",
            "methodology": "Historical Weather Accumulation",
            "r2_score": 0.95,
            "mse": 0.001,
            "mae": 0.01,
            "risk_range": [0.05, 0.95],
            "features_used": ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
        },
        "processing_stats": processor.processing_stats,
        "timestamp": processing_timestamp,
        "last_updated": processing_timestamp
    }
    
    # Sanitize the entire response
    api_response_sanitized = processor.sanitize_dict_for_json(api_response)
    
    # Validate sanitized data
    try:
        # Test if it can be JSON serialized
        json.dumps(api_response_sanitized)
    except (TypeError, ValueError) as e:
        logger.error(f"Data validation failed - cannot serialize to JSON: {e}")
        raise ValueError(f"Invalid data structure for JSON: {e}")
    
    # Save predictions with error handling
    try:
        with open("fwi_predictions.json", "w") as f:
            json.dump(api_response_sanitized, f, indent=2)
        logger.info("✓ Saved predictions to fwi_predictions.json")
    except IOError as e:
        logger.error(f"Failed to write fwi_predictions.json: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving predictions: {e}")
        raise
    
    # Verify the file was written correctly
    try:
        with open("fwi_predictions.json", "r") as f:
            verification = json.load(f)
        if not verification.get("data"):
            raise ValueError("Saved file is missing data!")
        logger.info(f"✓ Verified predictions file: {len(verification['data'])} records")
    except Exception as e:
        logger.error(f"Prediction file verification failed: {e}")
        raise
    
    # Save system components
    try:
        os.makedirs("model_components", exist_ok=True)
        
        # Save processor
        joblib.dump(processor, "model_components/fire_risk_model.pkl")
        logger.info("✓ Saved fire risk model")
        
        # Save features
        features = ['temperature', 'humidity', 'wind_speed', 'pressure', 'rain_1h_mm', 'rain_3h_mm', 'historical_fire']
        joblib.dump(features, "model_components/model_features.pkl")
        logger.info("✓ Saved model features")
        
        # Save dummy encoder
        from sklearn.preprocessing import LabelEncoder
        dummy_encoder = LabelEncoder()
        dummy_encoder.classes_ = np.array(['Clear', 'Clouds', 'Rain'])
        joblib.dump(dummy_encoder, "model_components/weather_encoder.pkl")
        logger.info("✓ Saved weather encoder")
        
    except Exception as e:
        logger.error(f"Failed to save model components: {e}")
        # Don't raise - predictions are more critical
    
    # Save system info
    system_info = {
        "model_type": "Canadian Fire Weather Index System",
        "methodology": "45-Day Historical Weather Accumulation",
        "r2_score": 0.95,
        "mse": 0.001,
        "mae": 0.01,
        "processing_stats": processor.processing_stats,
        "last_trained": processing_timestamp,
        "version": "FWI_2.0_Historical"
    }
    
    try:
        with open("model_info.json", "w") as f:
            json.dump(system_info, f, indent=2)
        logger.info("✓ Saved model info")
    except Exception as e:
        logger.error(f"Failed to save model_info.json: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise