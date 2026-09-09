import pandas as pd
import numpy as np
import joblib
import glob 
import os
from datetime import datetime
import json
import logging
import math
import gc  # Garbage collection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DANGER_CLASS_COLORS = {
    "Very Low": "#4CAF50",
    "Low": "#8BC34A",
    "Moderate": "#FFEB3B",
    "High": "#FF9800",
    "Very High": "#F44336",
    "Extreme": "#9C27B0",
}

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
    Pure implementation of Canadian FWI System
    Uses official algorithms without modifications
    """ 
    
    def __init__(self):
        self.version = "2.1.0"
    
    def sanitize_value(self, value, default, min_val, max_val):
        """Ensure values are valid numbers within range"""
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return default
            return np.clip(val, min_val, max_val)
        except (ValueError, TypeError):
            return default
    
    def get_seasonal_initial_codes(self, month):
        """
        Get realistic starting FWI codes based on seasonal weather patterns.
        
        These values represent typical fuel moisture conditions at the start 
        of each season after the preceding months' weather patterns.
        """
        
        if month in [12, 1, 2]:  # Winter
            # After fall rains, fuels are VERY wet
            # Snow cover, high humidity, cold temps
            return {
                'ffmc': 72.0,  # Wet fine fuels
                'dmc': 2.0,    # Very low duff moisture loss
                'dc': 8.0,     # Deep moisture retention
                'season_name': 'Winter'
            }
            
        elif month in [3, 4]:  # Early Spring
            # Thawing, but still plenty of moisture
            # Ground saturated from snowmelt
            return {
                'ffmc': 76.0,  # Fuels starting to dry
                'dmc': 4.0,    # Some drying beginning
                'dc': 12.0,    # Still good deep moisture
                'season_name': 'Early Spring'
            }
            
        elif month in [5, 6]:  # Late Spring
            # Warmer, drying accelerating
            # But still have spring moisture
            return {
                'ffmc': 81.0,  # Moderate drying
                'dmc': 10.0,   # Duff layer drying
                'dc': 25.0,    # Deep moisture declining
                'season_name': 'Late Spring'
            }
            
        elif month in [7, 8, 9]:  # Summer (Peak Fire Season)
            # Peak drying, highest fire danger period
            # Weeks of hot, dry weather
            return {
                'ffmc': 86.0,  # Dry fine fuels
                'dmc': 18.0,   # Significant duff drying
                'dc': 50.0,    # Deep drought accumulation
                'season_name': 'Summer'
            }
            
        else:  # October-November (Fall)
            # Summer dryness still present
            # But rains starting to return
            return {
                'ffmc': 80.0,  # Still relatively dry
                'dmc': 12.0,   # Moderate duff moisture
                'dc': 35.0,    # Drought code still elevated
                'season_name': 'Fall'
            }
    
    def calculate_ffmc(self, temp, humidity, wind, rain, prev_ffmc=85):
        """
        Fine Fuel Moisture Code (FFMC)
        
        Represents moisture content of litter and fine fuels (1-2 hour timelag).
        This is the top layer of the forest floor that dries/wets quickly.
        """
        # Sanitize inputs
        temp = self.sanitize_value(temp, 15, -50, 50)
        humidity = self.sanitize_value(humidity, 50, 1, 100)
        wind = self.sanitize_value(wind, 10, 0, 100)
        rain = self.sanitize_value(rain, 0, 0, 500)
        prev_ffmc = self.sanitize_value(prev_ffmc, 85, 0, 101)
        
        # Moisture content from previous FFMC
        mo = 147.2 * (101 - prev_ffmc) / (59.5 + prev_ffmc)
        
        # Rain effect - wetting of fine fuels
        if rain > 0.5:
            rf = rain - 0.5
            if mo <= 150:
                mo = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))
            else:
                mo = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf)) + 0.0015 * (mo - 150) ** 2 * np.sqrt(rf)
            
            if mo > 250:
                mo = 250
        
        # Equilibrium moisture content from drying
        ed = 0.942 * humidity ** 0.679 + 11 * np.exp((humidity - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * humidity))
        
        # Drying or wetting
        if mo > ed:
            # Drying conditions
            ko = 0.424 * (1 - (humidity / 100) ** 1.7) + 0.0694 * np.sqrt(wind) * (1 - (humidity / 100) ** 8)
            kd = ko * 0.581 * np.exp(0.0365 * temp)
            m = ed + (mo - ed) * 10 ** (-kd)
        else:
            # Wetting conditions
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
        temp = self.sanitize_value(temp, 15, -50, 50)
        humidity = self.sanitize_value(humidity, 50, 1, 100)
        rain = self.sanitize_value(rain, 0, 0, 500)
        prev_dmc = self.sanitize_value(prev_dmc, 6, 0, 500)


        DMC_LE = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        le = DMC_LE[month - 1] if 1 <= month <= 12 else 9.0

        dmc_after_rain = prev_dmc  # base for drying if there's no rain today

        if rain > 1.5:
            pe = 0.92 * rain - 1.27  # effective rainfall

            # moisture equivalent of yesterday's DMC
            wmi = 20 + np.exp(5.6348 - prev_dmc / 43.43)

            if prev_dmc <= 33:
                b = 100 / (0.5 + 0.3 * prev_dmc)
            elif prev_dmc <= 65:
                b = 14 - 1.3 * np.log(prev_dmc)
            else:
                b = 6.2 * np.log(prev_dmc) - 17.2

            # base is wmi (moisture equivalent)
            mr = wmi + 1000 * pe / (48.77 + b * pe)

            # convert back from moisture units to DMC units
            dmc_after_rain = 244.72 - 43.43 * np.log(mr - 20)
            dmc_after_rain = max(0, dmc_after_rain)

        temp_for_k = max(temp, -1.1)
        k = 1.894 * (temp_for_k + 1.1) * (100 - humidity) * le * 1e-6

        dmc = dmc_after_rain + 100 * k
        dmc = max(0, dmc)

        if np.isnan(dmc) or np.isinf(dmc):
            return 6
        return float(dmc)
    
    def calculate_dc(self, temp, rain, prev_dc=15, month=7):
        temp = self.sanitize_value(temp, 15, -50, 50)
        rain = self.sanitize_value(rain, 0, 0, 500)
        prev_dc = self.sanitize_value(prev_dc, 15, 0, 1000)

        DC_LF = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        lf = DC_LF[month - 1] if 1 <= month <= 12 else 1.4

        dc_after_rain = prev_dc

        if rain > 2.8:
            pd = 0.83 * rain - 1.27
            smi = 800 * np.exp(-prev_dc / 400)
            dr = prev_dc - 400 * np.log(1 + 3.937 * pd / smi)
            dc_after_rain = max(0, dr)

        temp_for_v = max(temp, -2.8)
        v = 0.36 * (temp_for_v + 2.8) + lf
        v = max(0, v)

        dc = dc_after_rain + 0.5 * v
        dc = max(0, dc)

        if np.isnan(dc) or np.isinf(dc):
            return 15
        return float(dc)
    
    def calculate_isi(self, wind, ffmc):
        """
        Initial Spread Index (ISI)
        
        Combines FFMC and wind speed to estimate fire spread rate.
        Represents the rate of fire spread without fuel considerations.
        
        """
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
        """
        Buildup Index (BUI)
        
        Combines DMC and DC to represent total fuel available for combustion.
        Indicates the amount of fuel available for fire.

        """
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
        """
        Fire Weather Index (FWI)
        
        Combines ISI and BUI to produce a general index of fire intensity.
        This is the primary output of the FWI System.
        
        """
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
        """
        Canadian FWI1987 Fire Danger Classification

        These are the FWI1987 (Van Wagner, 1987) danger class boundaries.
        They represent fire behavior potential, not ignition probability.
        Exact class boundaries vary by provincial/territorial fire agency;
        these are not a single official ECCC standard.

        """
        fwi = self.sanitize_value(fwi, 5, 0, 100)

        if fwi < 2:
            return "Very Low", fwi, "#4CAF50"
        elif fwi < 4:
            return "Low", fwi, "#8BC34A"
        elif fwi < 8:
            return "Moderate", fwi, "#FFEB3B"
        elif fwi < 18:
            return "High", fwi, "#FF9800"
        elif fwi < 30:
            return "Very High", fwi, "#F44336"
        else:
            return "Extreme", fwi, "#9C27B0"

FWI_STATE_FILE = "data/fwi_state.json"
GAP_REINIT_DAYS = 3     # a gap this long or longer triggers a seasonal reinit
                        # rather than continuing from stale persisted codes
TREND_HISTORY_DAYS = 8  # need 7 days back plus today for dc_trend_7d/bui_trend_7d

class FireWeatherProcessor:

    def __init__(self):
        self.fwi_calculator = CanadianFireWeatherIndex()
        self.processing_stats = {}

        # The ML danger-classification layer (get_ml_danger_class below) is
        # parked, not deleted: it was trained on leaked labels, saturated
        # features, and buggy formulas (see the Phase 1 audit), so
        # process_all_locations classifies with the FWI thresholds directly
        # instead. Its model/schema/thresholds are intentionally not loaded
        # here -- Phase 2 will need to retrain and reload them properly
        # before this can be revived.

    def _cell_key(self, lat, lon):
        return f"{lat:.4f}_{lon:.4f}"

    def load_fwi_state(self):
        """Load each cell's persisted FFMC/DMC/DC + trend history from the last run."""
        if os.path.exists(FWI_STATE_FILE):
            try:
                with open(FWI_STATE_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {FWI_STATE_FILE}: {e}; starting fresh for all cells")
        return {}

    def save_fwi_state(self, state):
        """Persist state atomically so a crash mid-write can't leave a truncated file."""
        os.makedirs(os.path.dirname(FWI_STATE_FILE) or ".", exist_ok=True)
        tmp_path = FWI_STATE_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, FWI_STATE_FILE)

    def _num_or(self, row, col, default=0.0):
        """Get row[col] as a plain number, treating a missing column or NaN as default."""
        val = row.get(col, default)
        return default if pd.isna(val) else val

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

    def get_ml_danger_class(self, lat, lon, ffmc, dmc, dc, isi, bui, fwi, date,     historical_fire, dc_trend_7d=0.0, bui_trend_7d=0.0):

        province = self.get_province(lat, lon) 
        day_of_year = date.timetuple().tm_yday
        month = date.month

        row = {
            "ffmc": ffmc, "dmc": dmc, "dc": dc, "isi": isi, "bui": bui, "fwi": fwi,
            "day_of_year": day_of_year, "month": month,
            "dc_trend_7d": dc_trend_7d, "bui_trend_7d": bui_trend_7d,
            "historical_fire": historical_fire,
        }
        for prov_col in self.ml_feature_schema["province_dummy_columns"]:
            row[prov_col] = 1 if prov_col == f"prov_{province}" else 0

        X = pd.DataFrame([row])[self.ml_feature_schema["full_column_order"]]
        raw_score = self.ml_model.predict_proba(X)[0, 1]

        bounds = self.ml_tier_thresholds["tier_bounds"]
        names = self.ml_tier_thresholds["tier_names"]
        fire_rates = self.ml_tier_thresholds["tier_actual_fire_rates"]

        # Half-open bins: [bounds[i], bounds[i+1]) so a score sitting exactly
        # on a boundary lands in the tier above it, not the one below.
        for i in range(len(bounds) - 2):
            if bounds[i] <= raw_score < bounds[i + 1]:
                return names[i], fire_rates[i]
        return names[-1], fire_rates[-1]
        
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
                # Use dtype to reduce memory usage
                df = pd.read_csv(file, dtype={
                    'lat': 'float32',
                    'lon': 'float32',
                    'temperature': 'float32',
                    'humidity': 'float32',
                    'wind_speed': 'float32',
                    'pressure': 'float32',
                    'precip_24h_mm': 'float32',
                    # legacy columns from the retired OpenWeather/station
                    # collector -- still present in older files inside the
                    # 45-day window; calculate_accumulated_fwi() falls back
                    # to summing these when precip_24h_mm isn't available.
                    'rain_1h_mm': 'float32',
                    'rain_3h_mm': 'float32',
                    'snow_1h_mm': 'float32',
                    'snow_3h_mm': 'float32'
                })

                # Reject implausible/sentinel rows instead of letting
                # sanitize_value() silently clip them into valid-looking
                # numbers (e.g. a -999 sentinel becomes 1% humidity).
                # A row failing any check is dropped from this day's history.
                before = len(df)
                valid = pd.Series(True, index=df.index)
                if 'temperature' in df:
                    valid &= df['temperature'].between(-60, 55)
                if 'humidity' in df:
                    valid &= df['humidity'].between(0, 100)
                if 'wind_speed' in df:
                    valid &= df['wind_speed'].between(0, 200)
                for precip_col in ('precip_24h_mm', 'rain_1h_mm', 'rain_3h_mm', 'snow_1h_mm', 'snow_3h_mm'):
                    if precip_col in df:
                        valid &= df[precip_col].between(0, 500) | df[precip_col].isna()
                df = df[valid]
                dropped = before - len(df)
                if dropped:
                    logger.warning(
                        f"{file}: dropped {dropped}/{before} rows with implausible "
                        f"weather values (e.g. sentinel/out-of-range data)"
                    )
                if df.empty:
                    logger.warning(f"{file}: no valid rows after filtering, skipping day entirely")
                    continue

                df['file_date'] = os.path.basename(file).replace('.csv', '')
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")

        if not all_data:
            raise ValueError("No weather data could be loaded")
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined)} total weather records")
        
        # Force garbage collection
        gc.collect()
        
        return combined
    
    def _get_precip_24h(self, day):
        """precip_24h_mm (Open-Meteo, a true 24h accumulation) is preferred;
        fall back to summing the legacy hourly/3h fields for older files
        still inside the window that predate the switch away from the
        OpenWeather station collector."""
        precip_24h = day.get('precip_24h_mm', None)
        if precip_24h is not None and not pd.isna(precip_24h):
            return precip_24h
        return (self._num_or(day, 'rain_1h_mm') + self._num_or(day, 'rain_3h_mm') +
            self._num_or(day, 'snow_1h_mm') + self._num_or(day, 'snow_3h_mm'))

    def advance_one_day(self, prev_ffmc, prev_dmc, prev_dc, recent, day, current_date):
        """Apply one day's weather to persisted FFMC/DMC/DC codes and update
        the trailing dc/bui history used for the 7-day trend features.

        `recent` is a list of up to TREND_HISTORY_DAYS {date, dmc, dc}
        dicts, oldest first; the entry 7 days back gives dc_trend_7d /
        bui_trend_7d. Returns (ffmc, dmc, dc, recent, result_dict).
        """
        temp = day.get('temperature', 15)
        humidity = day.get('humidity', 50)
        wind = day.get('wind_speed', 10)
        rain = self._get_precip_24h(day)
        month = current_date.month

        ffmc = self.fwi_calculator.calculate_ffmc(temp, humidity, wind, rain, prev_ffmc)
        dmc = self.fwi_calculator.calculate_dmc(temp, humidity, rain, prev_dmc, month)
        dc = self.fwi_calculator.calculate_dc(temp, rain, prev_dc, month)

        isi = self.fwi_calculator.calculate_isi(wind, ffmc)
        bui = self.fwi_calculator.calculate_bui(dmc, dc)
        fwi = self.fwi_calculator.calculate_fwi(isi, bui)
        dsr = 0.0272 * fwi ** 1.77

        recent = list(recent) + [{'date': current_date.strftime('%Y-%m-%d'), 'dmc': dmc, 'dc': dc}]
        recent = recent[-TREND_HISTORY_DAYS:]

        if len(recent) >= TREND_HISTORY_DAYS:
            week_ago = recent[-TREND_HISTORY_DAYS]  # 7 days before the entry we just added
            dc_trend_7d = dc - week_ago['dc']
            bui_7d_ago = self.fwi_calculator.calculate_bui(week_ago['dmc'], week_ago['dc'])
            bui_trend_7d = bui - bui_7d_ago
        else:
            dc_trend_7d = 0.0
            bui_trend_7d = 0.0

        result = {
            'ffmc': ffmc, 'dmc': dmc, 'dc': dc, 'isi': isi, 'bui': bui, 'fwi': fwi, 'dsr': dsr,
            'dc_trend_7d': dc_trend_7d, 'bui_trend_7d': bui_trend_7d,
        }
        return ffmc, dmc, dc, recent, self.sanitize_dict_for_json(result)

    def bootstrap_from_window(self, location_history, current_date):
        """First-run / gap-recovery path: no persisted state to continue
        from, so replay whatever weather history is on disk (up to 45
        days), starting from get_seasonal_initial_codes() for the first
        available day rather than a flat spring-startup default. Returns
        (ffmc, dmc, dc, recent) ready to hand to advance_one_day() for
        today.
        """
        location_history = location_history.sort_values('file_date')

        if len(location_history) == 0:
            seasonal = self.fwi_calculator.get_seasonal_initial_codes(current_date.month)
            return seasonal['ffmc'], seasonal['dmc'], seasonal['dc'], []

        first_date = pd.to_datetime(location_history.iloc[0]['file_date'])
        seasonal = self.fwi_calculator.get_seasonal_initial_codes(first_date.month)
        ffmc, dmc, dc = seasonal['ffmc'], seasonal['dmc'], seasonal['dc']
        recent = []

        for _, day in location_history.iterrows():
            day_date = pd.to_datetime(day['file_date'])
            ffmc, dmc, dc, recent, _ = self.advance_one_day(ffmc, dmc, dc, recent, day, day_date)

        return ffmc, dmc, dc, recent
    
    def process_all_locations(self, weather_file=None):
        
        logger.info("Processing Pure Canadian Fire Weather Index System...")
        start_time = datetime.now()

        # Load each cell's persisted FFMC/DMC/DC + trend history from the
        # last run. Cells with no entry (first run) or a gap of more than
        # GAP_REINIT_DAYS since their last entry fall back to replaying the
        # 45-day weather window from seasonal initial codes -- see
        # bootstrap_from_window(). Everything else continues directly from
        # yesterday's persisted codes, which is what lets DC reach real
        # seasonal drought values instead of resetting every run.
        fwi_state = self.load_fwi_state()
        historical_data = None  # lazily loaded only if a cell needs bootstrapping

        # Get today's data
        if weather_file is None:
            weather_files = glob.glob("weather_data/*.csv")
            weather_file = max(weather_files, key=os.path.getctime)
        
        today_data = pd.read_csv(weather_file, dtype={
            'lat': 'float32',
            'lon': 'float32',
            'temperature': 'float32',
            'humidity': 'float32',
            'wind_speed': 'float32',
            'pressure': 'float32',
            'precip_24h_mm': 'float32'
        })
        logger.info(f"Processing {len(today_data)} locations from {weather_file}")

        # Same date format load_historical_weather() stamps onto each row
        # (file_date), used below to exclude today's own file from the
        # bootstrap replay window -- it's read separately as today_data and
        # applied once via advance_one_day(), so replaying it too would
        # double-apply today's weather.
        weather_file_date_str = os.path.basename(weather_file).replace('.csv', '')

        # Add historical fire context
        try:
            fire_df = pd.read_csv("data/canada_fire_grid.csv", usecols=['lat', 'lon', 'historical_fire'])
            today_data = today_data.merge(fire_df, on=['lat', 'lon'], how='left')
            today_data['historical_fire'] = today_data['historical_fire'].fillna(0).astype('int8')
        except FileNotFoundError:
            logger.warning("Historical fire data not found")
            today_data['historical_fire'] = 0
        
        results = []
        processing_errors = 0
        
        # Process in batches to reduce memory pressure
        BATCH_SIZE = 500
        total_batches = (len(today_data) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(0, len(today_data), BATCH_SIZE):
            batch_end = min(batch_idx + BATCH_SIZE, len(today_data))
            batch = today_data.iloc[batch_idx:batch_end]
            
            logger.info(f"Processing batch {batch_idx//BATCH_SIZE + 1}/{total_batches}")
            
            for idx, row in batch.iterrows():
                try:
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    today_date = pd.to_datetime(row.get('date', datetime.now().isoformat()))

                    key = self._cell_key(lat, lon)
                    cell_state = fwi_state.get(key)

                    needs_bootstrap = cell_state is None
                    if cell_state is not None:
                        gap_days = (today_date.normalize() - pd.to_datetime(cell_state['date']).normalize()).days
                        if gap_days > GAP_REINIT_DAYS:
                            needs_bootstrap = True
                            logger.info(f"{lat},{lon}: {gap_days}-day gap since last run, reinitializing from seasonal codes")

                    if needs_bootstrap:
                        if historical_data is None:
                            historical_data = self.load_historical_weather(days_back=45)
                            # Excluded once, up front -- see weather_file_date_str above.
                            historical_data = historical_data[historical_data['file_date'] != weather_file_date_str]
                        location_hist = historical_data[
                            (historical_data['lat'] == lat) &
                            (historical_data['lon'] == lon)
                        ]
                        ffmc, dmc, dc, recent = self.bootstrap_from_window(location_hist, today_date)
                    else:
                        ffmc, dmc, dc = cell_state['ffmc'], cell_state['dmc'], cell_state['dc']
                        recent = cell_state.get('recent', [])

                    ffmc, dmc, dc, recent, fwi_data = self.advance_one_day(ffmc, dmc, dc, recent, row, today_date)

                    # Validate FWI data
                    if any(np.isnan(v) or np.isinf(v) for v in [fwi_data.get('ffmc', 0), fwi_data.get('dmc', 0), fwi_data.get('dc', 0)]):
                        logger.warning(f"Invalid FWI data for {lat},{lon}, using defaults")
                        fwi_data = {
                            'ffmc': 85.0, 'dmc': 6.0, 'dc': 15.0, 'isi': 1.0,
                            'bui': 10.0, 'fwi': 5.0, 'dsr': 1.0
                        }
                        ffmc, dmc, dc = fwi_data['ffmc'], fwi_data['dmc'], fwi_data['dc']

                    fwi_state[key] = {
                        'ffmc': ffmc, 'dmc': dmc, 'dc': dc,
                        'date': today_date.strftime('%Y-%m-%d'),
                        'recent': recent,
                    }

                    # FWI-threshold danger classification (the ML layer is
                    # parked -- see FireWeatherProcessor.__init__).
                    danger_class, _, color = self.fwi_calculator.get_danger_class(fwi_data['fwi'])
                    adjusted_fwi = fwi_data['fwi']

                    # precip_24h_mm (Open-Meteo, a true 24h total) is
                    # authoritative; fall back to summing the legacy
                    # hourly/3h fields for files predating the collector
                    # switch. Never mislabel a 24h total as an hourly one.
                    precip_24h = self._num_or(row, 'precip_24h_mm', None)
                    if precip_24h is None:
                        precip_24h = (self._num_or(row, 'rain_1h_mm') + self._num_or(row, 'rain_3h_mm') +
                            self._num_or(row, 'snow_1h_mm') + self._num_or(row, 'snow_3h_mm'))
                    
                    # Ensure FWI is valid
                    if np.isnan(adjusted_fwi) or np.isinf(adjusted_fwi):
                        adjusted_fwi = 5.0
                        danger_class = "Moderate"
                        color = "#FFEB3B"
                    
                    result_raw = {
                        'lat': lat,
                        'lon': lon,
                        'location_name': str(row.get('nearest_station', f'Grid_{idx}')),
                        'province': self.get_province(lat, lon),
                        'fwi': adjusted_fwi,  # Fire Weather Index value (not percentage!)
                        'danger_class': danger_class,
                        'color_code': color,
                        'weather_features': {
                            'temperature': row.get('temperature', 15),
                            'humidity': row.get('humidity', 50),
                            'wind_speed': row.get('wind_speed', 10),
                            'pressure': row.get('pressure', 1013),
                            'precip_24h_mm': precip_24h,  # true 24h accumulation, not an hourly snapshot
                            'is_hot': 1 if row.get('temperature', 15) > 25 else 0,
                            'is_dry': 1 if row.get('humidity', 50) < 30 else 0,
                            'humidity_temp_ratio': row.get('humidity', 50) / (row.get('temperature', 15) + 1),
                            'is_windy': 1 if row.get('wind_speed', 10) > 15 else 0,
                            'total_precip': precip_24h,
                            'has_recent_precip': 1 if precip_24h > 0 else 0,
                            'weather_main_encoded': 0
                        },
                        'fire_weather_indices': {
                            'ffmc': fwi_data['ffmc'],
                            'dmc': fwi_data['dmc'],
                            'dc': fwi_data['dc'],
                            'isi': fwi_data['isi'],
                            'bui': fwi_data['bui'],
                            'fwi': fwi_data['fwi'],
                            'dsr': fwi_data['dsr']
                        },
                        'historical_fire_zone': bool(row.get('historical_fire', 0)),
                    }
                    
                    result = self.sanitize_dict_for_json(result_raw)
                    
                    # Final validation before adding
                    if not np.isnan(result['fwi']) and not np.isinf(result['fwi']):
                        results.append(result)
                    else:
                        logger.warning(f"Skipping location {lat},{lon} due to invalid FWI value")
                        processing_errors += 1
                    
                except Exception as e:
                    processing_errors += 1
                    logger.error(f"Error processing location {idx}: {e}")
                    continue
            
            # Force garbage collection after each batch
            gc.collect()

        self.save_fwi_state(fwi_state)
        logger.info(f"Persisted FWI state for {len(fwi_state)} cells to {FWI_STATE_FILE}")

        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate stats
        fwi_values = [r['fwi'] for r in results]
        danger_classes = [r['danger_class'] for r in results]
        
        self.processing_stats = {
            'total_locations': len(today_data),
            'processed_successfully': len(results),
            'processing_errors': processing_errors,
            'processing_time_seconds': processing_time,
            'fwi_statistics': {
                'min_fwi': float(min(fwi_values)) if fwi_values else 0.0,
                'max_fwi': float(max(fwi_values)) if fwi_values else 0.0,
                'mean_fwi': float(np.mean(fwi_values)) if fwi_values else 0.0,
                'very_low_count': len([d for d in danger_classes if d == 'Very Low']),
                'low_count': len([d for d in danger_classes if d == 'Low']),
                'moderate_count': len([d for d in danger_classes if d == 'Moderate']),
                'high_count': len([d for d in danger_classes if d == 'High']),
                'very_high_count': len([d for d in danger_classes if d == 'Very High']),
                'extreme_count': len([d for d in danger_classes if d == 'Extreme'])
            }
        }
        
        logger.info(f"Processing complete: {len(results)} locations in {processing_time:.1f}s")
        logger.info(f"FWI: Min={min(fwi_values):.1f}, Max={max(fwi_values):.1f}, Mean={np.mean(fwi_values):.1f}")
        
        return results
    
    def get_province(self, lat, lon):
        """Map coordinates to province with corrected boundaries"""
        province_bounds = {
            'BC': (48.3, -139.1, 60.0, -114.1),
            'AB': (49.0, -120.0, 60.0, -110.0),
            'SK': (49.0, -110.0, 60.0, -101.4),
            'MB': (49.0, -102.0, 60.0, -88.9),
            'ON': (41.0, -95.2, 56.9, -74.3),  # extended from 41.7 -- the grid itself starts at 41.0
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

def main():
    cleanup_old_weather_data(days_to_keep=45)  
    processing_timestamp = datetime.now().isoformat()
    processor = FireWeatherProcessor()
    
    # Process with pure FWI algorithm + seasonal initial codes
    results = processor.process_all_locations()
    
    # Save results
    api_response = {
        "success": True,
        "data": results,
        "model_info": {
            "model_type": "Canadian Fire Weather Index System",
            "version": "2.1.0",
            "fwi_standard": "FWI1987 (Van Wagner, 1987)",
            "methodology": "Persisted daily FFMC/DMC/DC accumulation per cell, with seasonal "
                            "reinitialization on first run or after a gap of more than "
                            f"{GAP_REINIT_DAYS} days; falls back to replaying the 45-day weather "
                            "window when no prior state exists for a cell.",
            "algorithm": "Pure CFWIS (Van Wagner, 1987) - No modifications to FWI output",
            "seasonal_approach": "Realistic initial fuel moisture codes based on seasonal weather patterns",
            "components": ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "DSR"]
        },
        "processing_stats": processor.processing_stats,
        "timestamp": processing_timestamp,
        "last_updated": processing_timestamp,
        "fwi_calculated_at": processing_timestamp,  # When FWI was calculated
        "weather_last_updated": processing_timestamp,  # Same on initial calculation
        "last_update_type": "full_fwi_calculation",
        "notes": {
            "fwi_interpretation": "FWI represents fire behavior potential (spread rate, intensity) if ignition occurs",
            "not_a_probability": "FWI does NOT predict the probability of a fire starting",
            "danger_classes": "FWI1987 danger class thresholds; exact class boundaries vary by "
                               "provincial/territorial fire agency",
            "update_schedule": "FWI: Daily at noon | Weather: Hourly"
        }
    }

    #Sanitize the entire response
    api_response_sanitized = processor.sanitize_dict_for_json(api_response)

    with open("fwi_predictions.json", "w") as f:
        json.dump(api_response_sanitized, f, indent=2)

    # Save system components
    os.makedirs("model_components", exist_ok=True)
    joblib.dump(processor, "model_components/fire_risk_model.pkl")

    system_info = {
        "model_type": "Canadian Fire Weather Index System",
        "fwi_standard": "FWI1987 (Van Wagner, 1987)",
        "methodology": "Persisted daily FFMC/DMC/DC accumulation per cell with seasonal reinitialization",
        "algorithm": "Pure CFWIS (Van Wagner, 1987)",
        "processing_stats": processor.processing_stats,
        "last_trained": processing_timestamp,
        "version": "FWI_2.1_Pure"
    }

    with open("model_info.json", "w") as f:
        json.dump(system_info, f, indent=2)
    
    print("=" * 70)
    print("Pure Canadian Fire Weather Index System Ready!")
    print(f"✓ Processing completed at: {processing_timestamp}")
    print(f"✓ Persisted daily FFMC/DMC/DC accumulation (seasonal reinit on gaps > {GAP_REINIT_DAYS} days)")
    print(f"✓ Processed {len(results)} locations successfully")
    print("=" * 70)

if __name__ == "__main__":
    main()