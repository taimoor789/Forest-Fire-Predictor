# Forest Fire Risk Predictor - Backend

> **Fire risk assessment API built on the Canadian Fire Weather Index (FWI1987) System**

A Python-based backend that runs a fully automated daily pipeline — collecting fresh weather data and calculating fire danger levels for 7,537 grid cells across Canada — using persisted daily FFMC/DMC/DC accumulation with seasonal reinitialization.

**Live:** [forestfirepredictor.com](https://forestfirepredictor.com) · **API:** [forest-fire-predictor-api.onrender.com](https://forest-fire-predictor-api.onrender.com)

---

## Overview

The backend implements the **Canadian Fire Weather Index (FWI1987) System** (Van Wagner, 1987) — the standard fire weather index formulas used across Canadian wildfire agencies, though exact danger-class boundaries vary by provincial/territorial agency.

### **Key Capabilities**
- 🌡️ **Gridded weather** for every cell individually, from ECCC's HRDPS/GDPS/HRDPA (Open-Meteo kept as a fallback)
- 📍 **7,537 grid cells** actually processed, filtered from a ~15,000-cell 0.5° grid down to real Canadian land (see `in_canada` in `data/canada_fire_grid.csv`)
- 📈 **Persisted daily accumulation** per cell, with seasonal reinitialization on first run or after a data gap
- 🎯 **FWI1987 algorithm** (FFMC, DMC, DC, ISI, BUI, FWI, DSR)
- 🧪 **ML danger-tier model** (Random Forest, scikit-learn — see `ml/` and `docs/PREREGISTRATION.md`), trained with spatio-temporal cross-validation and probability calibration, a 151.7% relative PR-AUC lift over raw FWI. Runs in shadow mode today — computed and logged alongside every prediction, currently under live validation against real satellite fire detections before promotion to primary
- 🤖 **Fully automated daily pipeline**, scheduled via GitHub Actions — see Deployment below

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | High-performance REST API with automatic docs |
| **Data Processing** | Pandas + NumPy | Efficient manipulation of weather/fire data |
| **Weather API** | ECCC (HRDPS/GDPS/HRDPA), Open-Meteo fallback | Gridded forecast weather, one pull per grid cell |
| **FWI Algorithm** | Custom Implementation | FWI1987 (Van Wagner, 1987) formulas |
| **ML Model** | scikit-learn (Random Forest) | Shadow-mode danger-tier prediction, see Overview |
| **Task Scheduling** | GitHub Actions | Daily pipeline run, auto-deploy trigger, live ML shadow-eval capture |
| **Hosting** | Render (API) + Vercel (frontend) | Free-tier deployment, auto-deploy on push |
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
- **Frequency:** Once per day, automated (see Deployment)
- **Coverage:** 7,537 grid cells actually in Canada (real land mask, see `in_canada` in `data/canada_fire_grid.csv`) — no station interpolation

### **Historical Fire Data**
- **Source:** Natural Resources Canada - National Fire Database (NFDB)
- **Purpose:** Historical fire occurrence context
- **Format:** Shapefile → Grid mapping

### **FWI Algorithm**
- **Standard:** FWI1987 (Van Wagner, 1987)
- **Reference:** [CWFIS Fire Weather Index](https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi)

---

## Deployment

Fully automated, no manual steps required day to day:

1. **`.github/workflows/daily-pipeline.yml`** runs once a day on a GitHub Actions schedule: collects fresh ECCC weather grids, recomputes the FWI1987 indices for all 7,537 cells, scores the ML shadow model, and validates the output before publishing.
2. On success, it force-pushes a single size-bounded orphan commit to the `deploy` branch, containing only what the API needs to serve (predictions, model artifacts, weather history).
3. **Render** (free tier) auto-deploys the FastAPI backend from `deploy` on every new commit.
4. The frontend (Next.js, hosted on Vercel) fetches from that API with an hourly refresh and a 60-second check for new data.

A second, isolated pair of workflows (`.github/workflows/shadow-eval-*.yml`, branch `shadow-eval-log`) runs a live validation check for the ML model — comparing its shadow-mode predictions against real satellite fire detections from CWFIS under pre-registered statistical criteria, before it's promoted from shadow mode to primary. See `docs/PREREGISTRATION.md` for the full design and decision criteria.

`daily_update.py --pipeline-only` can still run the same pipeline manually (locally or from any other scheduler) for testing.

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
