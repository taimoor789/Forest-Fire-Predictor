# Forest Fire Risk Predictor - Backend

> **Fire risk assessment API built on the Canadian Fire Weather Index (FWI1987) System**

A Python-based backend that processes weather data and calculates fire danger levels for ~15,000 grid cells across Canada, using persisted daily FFMC/DMC/DC accumulation with seasonal reinitialization.

---

## Overview

The backend implements the **Canadian Fire Weather Index (FWI1987) System** (Van Wagner, 1987) — the standard fire weather index formulas used across Canadian wildfire agencies, though exact danger-class boundaries vary by provincial/territorial agency.

### **Key Capabilities**
- 🌡️ **Gridded weather** for every cell individually, from ECCC's HRDPS/GDPS/HRDPA (Open-Meteo kept as a fallback)
- 📍 **~7,500 grid cells** actually processed, filtered from a ~15,000-cell 0.5° grid down to real Canadian land (see `in_canada` in `data/canada_fire_grid.csv`)
- 📈 **Persisted daily accumulation** per cell, with seasonal reinitialization on first run or after a data gap
- 🎯 **FWI1987 algorithm** (FFMC, DMC, DC, ISI, BUI, FWI, DSR)
- 🧪 **ML danger-tier model** (see `ml/` and `docs/PREREGISTRATION.md`) running in shadow mode — computed and logged alongside every prediction, not yet served
- ⚠️ **Not currently scheduled** — see Deployment below

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | High-performance REST API with automatic docs |
| **Data Processing** | Pandas + NumPy | Efficient manipulation of weather/fire data |
| **Weather API** | ECCC (HRDPS/GDPS/HRDPA), Open-Meteo fallback | Gridded forecast weather, one pull per grid cell |
| **FWI Algorithm** | Custom Implementation | FWI1987 (Van Wagner, 1987) formulas |
| **Task Scheduling** | *(none currently — see Deployment)* | |
| **Storage** | Local CSV + JSON | Weather history, persisted FWI state, cached predictions |

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

These are this system's own FWI1987 threshold boundaries (`get_danger_class()` in `fire_risk.py`) — exact boundaries vary by provincial/territorial fire agency, so treat this as one reasonable set, not a single official ECCC standard.

| FWI Range | Class | Color | Description |
|-----------|-------|-------|-------------|
| 0-2 | Very Low | 🟢 Green | Fires start with difficulty |
| 2-4 | Low | 🟡 Yellow-Green | Fires spread slowly |
| 4-8 | Moderate | 🟡 Yellow | Moderate fire behavior |
| 8-18 | High | 🟠 Orange | High fire intensity |
| 18-30 | Very High | 🔴 Red | Extreme fire behavior |
| 30+ | Extreme | 🟣 Purple | Explosive fire growth |

---

## Data Sources

### **Weather Data**
- **Provider:** ECCC HRDPS/GDPS/HRDPA (`collect_weather_grid_eccc.py`), with Open-Meteo (`collect_weather_grid.py`) kept as a fallback
- **Frequency:** Once per day (see Deployment)
- **Coverage:** ~7,500 grid cells actually in Canada (real land mask, see `in_canada` in `data/canada_fire_grid.csv`) — no station interpolation

### **Historical Fire Data**
- **Source:** Natural Resources Canada - National Fire Database (NFDB)
- **Purpose:** Historical fire occurrence context
- **Format:** Shapefile → Grid mapping

### **FWI Algorithm**
- **Standard:** FWI1987 (Van Wagner, 1987)
- **Reference:** [CWFIS Fire Weather Index](https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi)

---

## Deployment

The previous AWS Elastic Beanstalk deployment expired, so no scheduled cron currently runs this pipeline in production. `daily_update.py --pipeline-only` runs the full pipeline (weather collection + FWI calculation) locally or from any scheduler; a migration to a managed host + scheduled job is planned but not yet implemented.

---

## Acknowledgments

- **Van Wagner, C.E.** - FWI System development (Van Wagner, 1987; Van Wagner & Pickett, 1985)
- **Canadian Forest Service** - Fire weather research
- **Natural Resources Canada** - National Fire Database
- **Environment and Climate Change Canada (ECCC)** - HRDPS/GDPS/HRDPA weather data
- **Open-Meteo** - Weather API fallback

---

<div align="center">

</div>
