from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
import json
import subprocess
import sys
from pathlib import Path
import logging
from contextlib import asynccontextmanager

#Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Global variables for model components
model = None
label_encoder = None
features = None
model_info_data = None

#Load model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    #Startup
    global model, label_encoder, features, model_info_data
    logger.info("Loading model components...")
    
    try: 
        model = joblib.load("model_components/fire_risk_model.pkl")
        label_encoder = joblib.load("model_components/weather_encoder.pkl")
        features = joblib.load("model_components/model_features.pkl")

        #Load model info from JSON file
        try: 
            with open("model_info.json", "r") as f:
                model_info_data = json.load(f)
                logger.info(f"Model info loaded: accuracy={model_info_data.get('accuracy', 'N/A')}")
        except FileNotFoundError:
            logger.warning("model_info.json not found, using defaults")
            model_info_data = {
                "accuracy": 0.804, 
                "roc_auc": 0.933,
                "last_trained": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"Model loading error: {e}")
        model_info_data = {
            "accuracy": 0.0,
            "roc_auc": 0.0,
            "last_trained": "never"
        }
    
    yield  # This separates startup from shutdown
    # Shutdown code would go here (if needed)

#Initialize FastAPI app with lifespan
app = FastAPI(
    title="Forest Fire Predictor API", 
    version="1.0.0",
    lifespan=lifespan
)

#Enable Cross-Origin Resource Sharing so Next.js frontend can call backend
app.add_middleware(
  CORSMiddleware,
  #the next.js app, 3001 as backup
  allow_origins=['http://localhost:3000', 'http://localhost:3001'], 
  allow_credentials=True,  #Allow cookies/credentials
  allow_methods=["*"],  #Allow all HTTP methods (GET, POST, etc.)
  allow_headers=["*"] #Allow all headers
)

def get_latest_weather():
  weather_files = glob.glob("weather_data/*.csv")   
  if not weather_files:
    raise HTTPException(status_code=500, detail="No weather data found")
  latest_file = max(weather_files, key=os.path.getctime)
  try:
      df = pd.read_csv(latest_file)
      logger.info(f"Loaded {len(df)} weather records")
      return df
  except Exception as e:
      logger.error(f"Error reading weather data: {e}")
      raise HTTPException(status_code=500, detail=f"Error reading weather data: {str(e)}")

def apply_feature_engineering(df):
    
    #Temperature features
    df['temp_range'] = df['temp_max'] - df['temp_min'] 
    df['is_hot'] = (df['temperature'] > 25).astype(int)
    
    #Humidity features
    df['is_dry'] = (df['humidity'] < 30).astype(int)
    df['humidity_temp_ratio'] = df['humidity'] / (df['temperature'] + 1)
    
    #Wind features  
    df['is_windy'] = (df['wind_speed'] > 5).astype(int)
    df['wind_gust_filled'] = df['wind_gust'].fillna(df['wind_speed'])
    
    #Precipitation features
    df['total_precip'] = df['rain_1h_mm'] + df['snow_1h_mm']
    df['has_recent_precip'] = (df['total_precip'] > 0).astype(int)
    
    #Pressure features
    df['is_high_pressure'] = (df['pressure'] > 1020).astype(int)
    
    #Fire danger index 
    df['fire_danger_index'] = (
        (df['temperature'] / 40) * 0.3 +
        ((100 - df['humidity']) / 100) * 0.3 +
        (df['wind_speed'] / 20) * 0.2 +
        ((1040 - df['pressure']) / 100) * 0.1 +
        (1 - df['has_recent_precip']) * 0.1
    )
    
    #Encode weather_main using trained encoder
    weather_main_filled = df['weather_main'].fillna('Unknown')
    weather_encoded = []
    for weather in weather_main_filled:
        if weather in label_encoder.classes_:
            weather_encoded.append(label_encoder.transform([weather])[0])
        else:
            weather_encoded.append(0)  # fallback
    df['weather_main_encoded'] = weather_encoded
    
    #Fill missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def get_province_from_station(station_name):
    #Map station names to provinces
    province_mapping = {
        "Vancouver": "BC", "Kelowna": "BC", "Kamloops": "BC", "Victoria": "BC", "Prince George": "BC", "Smithers": "BC", "Dease Lake": "BC", "Calgary": "AB", "Edmonton": "AB", "Fort McMurray": "AB", "Fort St. John": "AB", "High Level": "AB", "Peace River": "AB", "Saskatoon": "SK", "Regina": "SK", "La Ronge": "SK", "Winnipeg": "MB", "Flin Flon": "MB", "Churchill": "MB", "Thunder Bay": "ON", "Ottawa": "ON", "Toronto": "ON", "Sudbury": "ON",
        "Moosonee": "ON", "Timmins": "ON", "Montreal": "QC", "Quebec City": "QC", "Val-d'Or": "QC",  "Chibougamau": "QC", "Schefferville": "QC", "Halifax": "NS", "Goose Bay": "NL", "St. John's": "NL", "Whitehorse": "YT", "Yellowknife": "NT", "Iqaluit": "NU", "Rankin Inlet": "NU", "Cambridge Bay": "NU"
    }
    return province_mapping.get(station_name, "Unknown")

@app.get("/")
async def root():
    return {"message": "Forest Fire Predictor API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
   return {
      "status": "healthy" if model else "model_not_loaded",
      "timestamp": datetime.now().isoformat(),
      "model_loaded": model is not None,
      "features_count": len(features) if features else 0,
      "weather_data_available": len(glob.glob("weather_data/*.csv")) > 0
   }

@app.get("/api/model/info")
async def get_model_info():
   if not model:
      raise HTTPException(status_code=500, detail="Model not loaded")
   return {
      "accuracy": model_info_data.get("accuracy", 0.0),
      "roc_auc": model_info_data.get("roc_auc", 0.0),    
      "features": features,
      "version": "1.0.0",
      "last_trained": model_info_data.get("last_trained", "unknown"),
      "training_records": model_info_data.get("training_records", 0)
   }

@app.get("/api/predict/fire-risk")
async def get_fire_risk_predictions():
    #Get predictions for all locations
    if not model:
      raise HTTPException(status_code=500, detail="Model not loaded")
    
    try: 
       #Load latest weather data
       weather_df = get_latest_weather()
      
       #Apply feature engineering
       weather_df = apply_feature_engineering(weather_df)
 
       #Get predictions for all locations
       X = weather_df[features]
       probabilities = model.predict_proba(X)[:, 1]  #Fire probability
       confidence = np.max(model.predict_proba(X), axis=1) #Model confidence

       #Build response data 
       predictions = []
       for i, row in weather_df.iterrows():
          station_name = row.get('nearest_station', f'Location {i}')
          province = get_province_from_station(station_name)

          predictions.append({
             "lat": float(row['lat']),
             "lng": float(row['lon']),
             "location_name": station_name,
             "province": province,
             "fire_risk_probability": float(probabilities[i]),
             "weather_features": {
                  "temperature": float(row['temperature']),
                  "humidity": float(row['humidity']),
                  "wind_speed": float(row['wind_speed']),
                  "pressure": float(row['pressure']),
                  "fire_danger_index": float(row['fire_danger_index']),
                  "rain_1h_mm": float(row['rain_1h_mm']),
                  "rain_3h_mm": float(row['rain_3h_mm']),
                  "snow_1h_mm": float(row['snow_1h_mm']),
                  "snow_3h_mm": float(row['snow_3h_mm']),
                  "is_hot": int(row['is_hot']),
                  "is_dry": int(row['is_dry']),
                  "humidity_temp_ratio": float(row['humidity_temp_ratio']),
                  "is_windy": int(row['is_windy']),
                  "total_precip": float(row['total_precip']),
                  "has_recent_precip": int(row['has_recent_precip']),
                  "weather_main_encoded": int(row['weather_main_encoded'])
               },
               "model_confidence": float(confidence[i]),
               "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
           })
          
       return {
           "success": True,
           "data": predictions,
            "model_info": {
               "version": "1.0.0",
               "accuracy": model_info_data.get("accuracy", 0.0),
               "roc_auc": model_info_data.get("roc_auc", 0.0),
               "features_used": features
            },
           "timestamp": datetime.now().isoformat()
         }
    except Exception as e:
       raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/model/retrain")
async def retrain_model():
   #trigger daily_update pipeline
   try:
      result = subprocess.run(
         [sys.executable, "daily_update.py", "--pipeline-only"],
         capture_output=True,
         text=True,
         check=True,
         timeout=300
      )

      #Reload model after retraining
      await lifespan()

      return {"success": True, "message": "Model retrained successfully"}
   except Exception as e:
      logger.error(f"Retraining failed: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.post("/api/model/reload")
async def reload_model():
   #Reload the model components without retraining
   try: 
      await lifespan()
      return {"success": True, "message": "Model reloaded successfully"}
   except Exception as e:
       logger.error(f"Model reload failed: {e}")
       raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
   import uvicorn 
   uvicorn.run(app, host="0.0.0.0", port=8000)

      
   
