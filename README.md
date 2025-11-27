# Forest Fire Risk Predictor - Backend

> **Production-grade fire risk assessment API powered by Canada's official Fire Weather Index System**

A Python-based backend that processes real-time weather data and calculates fire danger levels for 15,000+ locations across Canada, updated hourly with 30-day historical weather accumulation.

---

## Overview

The backend implements Environment Canada's **Canadian Fire Weather Index (FWI) System** - the official algorithm used by Canadian wildfire agencies for fire danger rating.

### **Key Capabilities**
- 🌡️ **Real-time weather** from 38 stations via OpenWeather API
- 📍 **15,000+ grid cells** covering all of Canada (50km × 50km resolution)
- 📈 **30-day accumulation** for accurate moisture codes
- ⚡ **Hourly updates** via automated cron jobs
- 🎯 **Official FWI algorithm** (FFMC, DMC, DC, ISI, BUI, FWI, DSR)
- 🔒 **Production deployment** on AWS Elastic Beanstalk

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | High-performance REST API with automatic docs |
| **Data Processing** | Pandas + NumPy | Efficient manipulation of weather/fire data |
| **Weather API** | OpenWeather | Real-time weather for 38 Canadian stations |
| **FWI Algorithm** | Custom Implementation | Official Canadian Fire Weather Index formulas |
| **Deployment** | AWS Elastic Beanstalk | Auto-scaling, monitoring, zero-downtime updates |
| **Task Scheduling** | Linux Cron | Hourly automated weather collection |
| **Storage** | Local CSV + JSON | 30-day weather history + cached predictions |

---

## Fire Weather Index System

### **What is FWI?**

The **Canadian Fire Weather Index (FWI) System** is the official method used by Canadian wildfire agencies to assess fire danger. It tracks moisture in different fuel layers and calculates fire behavior potential.

### **Components**

#### **Fuel Moisture Codes** (track drying over time)
- **FFMC** (Fine Fuel Moisture Code) - Surface litter (1-2 day lag)
- **DMC** (Duff Moisture Code) - Decomposed organic matter (15+ day lag)
- **DC** (Drought Code) - Deep soil moisture (50+ day lag)

#### **Fire Behavior Indices** (predict fire intensity)
- **ISI** (Initial Spread Index) - Rate of fire spread
- **BUI** (Buildup Index) - Fuel available for combustion
- **FWI** (Fire Weather Index) - Overall fire intensity potential
- **DSR** (Daily Severity Rating) - Fire difficulty rating

### **Danger Classes**

| FWI Range | Class | Color | Description |
|-----------|-------|-------|-------------|
| 0-1 | Very Low | 🟢 Green | Fires start with difficulty |
| 1-3 | Low | 🟡 Yellow-Green | Fires spread slowly |
| 3-7 | Moderate | 🟡 Yellow | Moderate fire behavior |
| 7-17 | High | 🟠 Orange | High fire intensity |
| 17-30 | Very High | 🔴 Red | Extreme fire behavior |
| 30+ | Extreme | 🟣 Purple | Explosive fire growth |

---

## Data Sources

### **Weather Data**
- **Provider:** OpenWeather API
- **Frequency:** Hourly
- **Stations:** 38 major Canadian locations
- **Coverage:** All provinces and territories

### **Historical Fire Data**
- **Source:** Natural Resources Canada - National Fire Database (NFDB)
- **Purpose:** Historical fire occurrence for risk adjustment
- **Format:** Shapefile → Grid mapping

### **FWI Algorithm**
- **Authority:** Environment and Climate Change Canada
- **Standard:** Official Canadian Forest Service formulas
- **Reference:** [CWFIS Fire Weather Index](https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi)

---

## Acknowledgments

- **Environment and Climate Change Canada** - FWI System development
- **Canadian Forest Service** - Fire weather research
- **Natural Resources Canada** - National Fire Database
- **OpenWeather** - Weather API services

---

<div align="center">

**Protecting Canadian communities through data-driven fire risk assessment 🇨🇦**

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=for-the-badge&logo=fastapi)
![AWS](https://img.shields.io/badge/AWS-Deployed-orange?style=for-the-badge&logo=amazon-aws)

</div>