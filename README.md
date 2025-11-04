Forest Fire Risk Predictor - Backend

Real-time fire risk assessment API using the Canadian Fire Weather Index (FWI) System.

Overview
Python-based backend that:

Fetches weather data from OpenWeather API for 15,000+ grid cells across Canada
Calculates fire risk using 45-day historical weather accumulation
Implements official Canadian FWI algorithm (FFMC, DMC, DC, ISI, BUI, FWI)
Provides REST API for fire risk predictions

Tech Stack

FastAPI - REST API framework
Pandas/NumPy - Data processing
Canadian FWI System - Official fire weather calculations
AWS Elastic Beanstalk - Deployment platform

How It Works

Weather Collection (collect_weather_grid.py)

Fetches current weather for 38 weather stations
Maps data to 15,000+ grid cells (50km × 50km)
Stores daily CSV files


FWI Calculation (fire_risk.py)

Loads 45 days of historical weather
Calculates FWI codes (FFMC, DMC, DC) with daily accumulation
Computes fire danger indices (ISI, BUI, FWI)
Generates risk probabilities (0-100%)


API Server (main.py)

Serves cached predictions
Handles system reloads
Provides health monitoring


Data Sources

Weather: OpenWeather API (hourly updates)
Historical Fires: Natural Resources Canada (NFDB)
FWI Algorithm: Environment and Climate Change Canada

Deployment
Deployed on AWS Elastic Beanstalk with:

Hourly cron job for weather updates
Auto-scaling (1-2 instances)
CloudWatch logging