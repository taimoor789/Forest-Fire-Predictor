from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import joblib
import os
from datetime import datetime
import glob
import json
import subprocess
import sys
from pathlib import Path
from contextlib import asynccontextmanager

STARTUP_TIME = datetime.now()

# Use centralized logging
from logging_config import setup_logging, get_logger

setup_logging(console_output=True, file_output=True)
logger = get_logger(__name__)

# Pydantic models for request/response validation
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    system_type: str
    system_loaded: bool
    weather_data_available: bool = False
    last_updated: Optional[str] = None

class ModelInfoResponse(BaseModel):
    model_type: str
    methodology: str
    r2_score: float
    mse: float
    mae: float
    risk_range: List[float]
    features: List[str]
    version: str
    last_trained: str
    training_records: int = 0
    confidence: str

class SystemReloadResponse(BaseModel):
    success: bool
    message: str
    timestamp: str

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: str

class DataFreshnessStatus(BaseModel):
    is_fresh: bool
    age_hours: float
    last_updated: str
    status: str  # "fresh", "stale", "missing"

class ComponentHealth(BaseModel):
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str

class DetailedHealthResponse(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    system_type: str
    system_loaded: bool
    components: List[ComponentHealth]
    data_freshness: DataFreshnessStatus
    uptime_seconds: Optional[float] = None

# Request validation models
class RetrainRequest(BaseModel):
    force: bool = Field(default=False, description="Force retrain even if recent data exists")
    
    @validator('force')
    def validate_force(cls, v):
        if not isinstance(v, bool):
            raise ValueError('force must be a boolean')
        return v

# Global variables for system components
fire_weather_processor = None
system_info_data = None

# Load system on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global fire_weather_processor, system_info_data
    logger.info("=" * 70)
    logger.info("Loading Fire Weather Index System...")
    
    try: 
        fire_weather_processor = True  # Flag that system is ready
        
        # Load system info from JSON file
        try: 
            with open("model_info.json", "r") as f:
                system_info_data = json.load(f)
                logger.info("Fire Weather Index System loaded successfully")
        except FileNotFoundError:
            logger.warning("model_info.json not found - using defaults")
            system_info_data = {
                "model_type": "Canadian Fire Weather Index System",
                "methodology": "45-Day Historical Weather Accumulation",
                "r2_score": 0.95,
                "last_trained": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Fire Weather System loading error: {e}")
        fire_weather_processor = None
    
    logger.info("=" * 70)
    yield
    
    # Shutdown
    logger.info("Shutting down Fire Weather Index System")

#CORS Settings - Allow both local and production
ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://localhost:3001',
    'https://fire-risk-predictor.vercel.app',
    'https://d1aexr3nj3xzld.cloudfront.net',  
]

VERCEL_DOMAIN = os.environ.get('VERCEL_DOMAIN')
CUSTOM_DOMAIN = os.environ.get('CUSTOM_DOMAIN')

if VERCEL_DOMAIN:
    ALLOWED_ORIGINS.append(f'https://{VERCEL_DOMAIN}')
    ALLOWED_ORIGINS.append(f'https://{VERCEL_DOMAIN.replace(".vercel.app", "")}.vercel.app')

if CUSTOM_DOMAIN:
    ALLOWED_ORIGINS.append(f'https://{CUSTOM_DOMAIN}')
    ALLOWED_ORIGINS.append(f'http://{CUSTOM_DOMAIN}')

app = FastAPI(
    title="Forest Fire Risk Prediction API", 
    description="Production fire risk assessment using Canadian Fire Weather Index System",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Add OPTIONS for preflight
    allow_headers=["*"],
    max_age=3600
)

print(f"CORS Allowed Origins: {ALLOWED_ORIGINS}")  # Debug log

# Helper function for error responses
def create_error_response(detail: str, error_code: Optional[str] = None) -> dict:
    return {
        "detail": detail,
        "error_code": error_code,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Forest Fire Risk Prediction API", 
        "version": "2.0.0", 
        "system": "Canadian Fire Weather Index",
        "status": "running",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns detailed status of all system components.
    """
    components = []
    overall_status = "healthy"
    
    # 1. Check if system is loaded
    if fire_weather_processor:
        components.append(ComponentHealth(
            name="Fire Weather Processor",
            status="healthy",
            message="System loaded and ready"
        ))
    else:
        components.append(ComponentHealth(
            name="Fire Weather Processor",
            status="unhealthy",
            message="System not loaded"
        ))
        overall_status = "unhealthy"
    
    # 2. Check weather data availability
    try:
        weather_files = glob.glob("weather_data/*.csv")
        if len(weather_files) > 0:
            latest_file = max(weather_files, key=os.path.getmtime)
            file_age = (datetime.now().timestamp() - os.path.getmtime(latest_file)) / 3600
            
            if file_age < 2:  # Less than 2 hours old
                components.append(ComponentHealth(
                    name="Weather Data",
                    status="healthy",
                    message=f"{len(weather_files)} files available, latest is {file_age:.1f}h old"
                ))
            elif file_age < 24:  # Less than 24 hours old
                components.append(ComponentHealth(
                    name="Weather Data",
                    status="degraded",
                    message=f"Latest data is {file_age:.1f}h old (expected <2h)"
                ))
                if overall_status == "healthy":
                    overall_status = "degraded"
            else:
                components.append(ComponentHealth(
                    name="Weather Data",
                    status="unhealthy",
                    message=f"Latest data is {file_age:.1f}h old (stale!)"
                ))
                overall_status = "unhealthy"
        else:
            components.append(ComponentHealth(
                name="Weather Data",
                status="unhealthy",
                message="No weather data files found"
            ))
            overall_status = "unhealthy"
    except Exception as e:
        components.append(ComponentHealth(
            name="Weather Data",
            status="unhealthy",
            message=f"Error checking weather data: {str(e)}"
        ))
        overall_status = "unhealthy"
    
    # 3. Check predictions file
    try:
        if os.path.exists("fwi_predictions.json"):
            file_size = os.path.getsize("fwi_predictions.json")
            file_age = (datetime.now().timestamp() - os.path.getmtime("fwi_predictions.json")) / 3600
            
            if file_size < 1024:  # Less than 1KB is suspicious
                components.append(ComponentHealth(
                    name="Predictions Cache",
                    status="unhealthy",
                    message=f"Predictions file suspiciously small ({file_size} bytes)"
                ))
                overall_status = "unhealthy"
            elif file_age < 2:
                components.append(ComponentHealth(
                    name="Predictions Cache",
                    status="healthy",
                    message=f"Predictions are {file_age:.1f}h old"
                ))
            elif file_age < 24:
                components.append(ComponentHealth(
                    name="Predictions Cache",
                    status="degraded",
                    message=f"Predictions are {file_age:.1f}h old (expected <2h)"
                ))
                if overall_status == "healthy":
                    overall_status = "degraded"
            else:
                components.append(ComponentHealth(
                    name="Predictions Cache",
                    status="unhealthy",
                    message=f"Predictions are {file_age:.1f}h old (stale!)"
                ))
                overall_status = "unhealthy"
            
            # Validate JSON structure
            try:
                with open("fwi_predictions.json", "r") as f:
                    pred_data = json.load(f)
                if not pred_data.get("data"):
                    components.append(ComponentHealth(
                        name="Predictions Validation",
                        status="unhealthy",
                        message="Predictions file has no data"
                    ))
                    overall_status = "unhealthy"
                else:
                    components.append(ComponentHealth(
                        name="Predictions Validation",
                        status="healthy",
                        message=f"{len(pred_data['data'])} predictions available"
                    ))
            except json.JSONDecodeError:
                components.append(ComponentHealth(
                    name="Predictions Validation",
                    status="unhealthy",
                    message="Predictions file is corrupted (invalid JSON)"
                ))
                overall_status = "unhealthy"
        else:
            components.append(ComponentHealth(
                name="Predictions Cache",
                status="unhealthy",
                message="Predictions file not found"
            ))
            overall_status = "unhealthy"
    except Exception as e:
        components.append(ComponentHealth(
            name="Predictions Cache",
            status="unhealthy",
            message=f"Error checking predictions: {str(e)}"
        ))
        overall_status = "unhealthy"
    
    # 4. Check model components
    required_files = [
        "model_components/fire_risk_model.pkl",
        "model_info.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        components.append(ComponentHealth(
            name="Model Components",
            status="unhealthy",
            message=f"Missing files: {', '.join(missing_files)}"
        ))
        overall_status = "unhealthy"
    else:
        components.append(ComponentHealth(
            name="Model Components",
            status="healthy",
            message="All model files present"
        ))
    
    # 5. Data freshness summary
    data_freshness_status = DataFreshnessStatus(
        is_fresh=False,
        age_hours=0.0,
        last_updated="unknown",
        status="missing"
    )
    
    if system_info_data and "last_trained" in system_info_data:
        try:
            last_updated_dt = datetime.fromisoformat(system_info_data["last_trained"].replace('Z', '+00:00'))
            age_hours = (datetime.now(last_updated_dt.tzinfo or None) - last_updated_dt).total_seconds() / 3600
            
            data_freshness_status = DataFreshnessStatus(
                is_fresh=age_hours < 2,
                age_hours=round(age_hours, 2),
                last_updated=system_info_data["last_trained"],
                status="fresh" if age_hours < 2 else ("stale" if age_hours < 24 else "very_stale")
            )
        except Exception:
            data_freshness_status.last_updated = system_info_data.get("last_trained", "unknown")
            
    uptime = (datetime.now() - STARTUP_TIME).total_seconds()

    from fastapi.responses import JSONResponse
    
    response_data = DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        system_type="Canadian Fire Weather Index System",
        system_loaded=fire_weather_processor is not None,
        components=components,
        data_freshness=data_freshness_status,
        uptime_seconds=round(uptime, 2)
    )
    
    return JSONResponse(
        content=response_data.dict(),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    ) 

@app.get("/api/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information and statistics"""
    if not fire_weather_processor:
        raise HTTPException(
            status_code=503,
            detail="Fire Weather System not loaded"
        )
    
    try:
        training_records = 0
        if system_info_data and "processing_stats" in system_info_data:
            training_records = system_info_data["processing_stats"].get("processed_successfully", 0)
        
        return ModelInfoResponse(
            model_type=system_info_data.get("model_type", "Canadian Fire Weather Index System"),
            methodology=system_info_data.get("methodology", "Environment and Climate Change Canada Official Algorithm"),
            r2_score=system_info_data.get("r2_score", 0.95),
            mse=system_info_data.get("mse", 0.001),
            mae=system_info_data.get("mae", 0.01),
            risk_range=system_info_data.get("risk_range", [0.05, 0.95]),
            features=["FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "Temperature", "Humidity", "Wind", "Precipitation"],
            version="2.0.0",
            last_trained=system_info_data.get("last_trained", "unknown"),
            training_records=training_records,
            confidence="High - Based on established fire weather science"
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

@app.get("/api/predict/fire-risk")
async def get_fire_risk_predictions():
    """Get fire risk predictions with validation"""
    if not fire_weather_processor:
        raise HTTPException(
            status_code=503,
            detail="Fire Weather System not loaded"
        )
    
    try: 
        # Load cached predictions
        try:
            with open("fwi_predictions.json", "r") as f:
                cached_predictions = json.load(f)
            
            # Validate response structure
            if not cached_predictions.get("success"):
                raise ValueError("Invalid prediction data structure")
            
            if not cached_predictions.get("data"):
                raise ValueError("No prediction data available")
            
            logger.info(f"Returning {len(cached_predictions['data'])} Fire Weather Index predictions")
            
            # Create response with no-cache headers
            from fastapi.responses import JSONResponse
            
            return JSONResponse(
                content=cached_predictions,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
            
        except FileNotFoundError:
            logger.error("No predictions available - fwi_predictions.json not found")
            raise HTTPException(
                status_code=404,
                detail="No predictions available - run fire_risk.py first"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted predictions file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Prediction data is corrupted"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fire risk prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Fire risk prediction failed: {str(e)}"
        )

@app.post("/api/system/retrain", response_model=SystemReloadResponse)
async def retrain_system(request: Optional[RetrainRequest] = None):
    """Trigger the data pipeline to refresh weather data and recalculate fire weather indices"""
    
    # Validate request
    if request is None:
        request = RetrainRequest()
    
    try:
        logger.info("Triggering fire weather system refresh...")
        
        # Check if recent data exists
        if not request.force:
            prediction_file = "fwi_predictions.json"
            if os.path.exists(prediction_file):
                file_age = datetime.now().timestamp() - os.path.getmtime(prediction_file)
                if file_age < 3600:  # Less than 1 hour old
                    logger.info("Recent predictions exist, skipping retrain")
                    return SystemReloadResponse(
                        success=True,
                        message="Recent predictions already exist (use force=true to override)",
                        timestamp=datetime.now().isoformat()
                    )
        
        result = subprocess.run(
            [sys.executable, "daily_update.py", "--pipeline-only"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )

        # Reload system components
        global fire_weather_processor, system_info_data
        
        try:
            fire_weather_processor = joblib.load("model_components/fire_risk_model.pkl")
            with open("model_info.json", "r") as f:
                system_info_data = json.load(f)
            logger.info("Fire Weather System reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload system after refresh: {e}")
            raise HTTPException(status_code=500, detail="System refresh succeeded but reload failed")

        return SystemReloadResponse(
            success=True,
            message="Fire Weather System refreshed successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except subprocess.TimeoutExpired:
        logger.error("System refresh timed out")
        raise HTTPException(status_code=504, detail="System refresh timed out (>5 minutes)")
    except subprocess.CalledProcessError as e:
        logger.error(f"System refresh failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"System refresh failed: {e.stderr}")
    except Exception as e:
        logger.error(f"System refresh error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System refresh error: {str(e)}")

@app.post("/api/system/reload", response_model=SystemReloadResponse)
async def reload_system():
    """Reload the Fire Weather System components without refreshing weather data"""
    try: 
        global fire_weather_processor, system_info_data
        
        fire_weather_processor = joblib.load("model_components/fire_risk_model.pkl")
        
        with open("model_info.json", "r") as f:
            system_info_data = json.load(f)
            
        logger.info("Fire Weather System reloaded successfully")
        return SystemReloadResponse(
            success=True,
            message="Fire Weather System reloaded successfully",
            timestamp=datetime.now().isoformat()
        )
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise HTTPException(status_code=404, detail=f"Required system file not found: {str(e)}")
    except Exception as e:
        logger.error(f"System reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"System reload failed: {str(e)}")

@app.get("/api/stats")
async def get_system_stats():
    """Get detailed system processing statistics"""
    if not system_info_data:
        raise HTTPException(status_code=503, detail="System info not available")
    
    try:
        # Get cache info if available
        cache_status = {"predictions_cached": False, "cache_age_hours": None}
        if os.path.exists("fwi_predictions.json"):
            cache_status["predictions_cached"] = True
            try:
                cache_time = os.path.getmtime("fwi_predictions.json")
                cache_age = (datetime.now().timestamp() - cache_time) / 3600
                cache_status["cache_age_hours"] = round(cache_age, 1)
            except Exception:
                cache_status["cache_age_hours"] = "unknown"
        
        return {
            "system_info": system_info_data,
            "weather_files_available": len(glob.glob("weather_data/*.csv")),
            "current_time": datetime.now().isoformat(),
            "cache_status": cache_status,
            "system_status": {
                "processor_loaded": fire_weather_processor is not None,
                "last_calculation": system_info_data.get("last_trained", "unknown") if system_info_data else "unknown",
                "system_type": "Canadian Fire Weather Index"
            }
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

@app.get("/api/danger-classes")
async def get_danger_classes():
    """Get fire danger class definitions and color codes"""
    return {
        "danger_classes": [
            {"name": "Very Low", "range": "0-1 FWI", "color": "#4CAF50", "description": "Fires start easily but spread slowly"},
            {"name": "Low", "range": "1-3 FWI", "color": "#8BC34A", "description": "Fires start easily and spread at low to moderate rates"},
            {"name": "Moderate", "range": "3-7 FWI", "color": "#FFEB3B", "description": "Fires start easily and spread at moderate rates"},
            {"name": "High", "range": "7-17 FWI", "color": "#FF9800", "description": "Fires start easily and spread at high rates"},
            {"name": "Very High", "range": "17-30 FWI", "color": "#F44336", "description": "Fires start very easily and spread at very high rates"},
            {"name": "Extreme", "range": "30+ FWI", "color": "#9C27B0", "description": "Fires start very easily and spread at extreme rates"}
        ],
        "system": "Canadian Fire Weather Index",
        "authority": "Environment and Climate Change Canada"
    }

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)