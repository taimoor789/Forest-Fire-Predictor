"""
Normalizes the raw ERA5-Land GRIB files (ml/weather/era5_client.py) into
exactly the five keys fire_risk.py's advance_one_day() reads -- temperature
(C), humidity (%), wind_speed (km/h), pressure (hPa), precip_24h_mm -- in
production's exact units, so Stage 6's replay can call advance_one_day()
directly with no further conversion.

Derivations, verified against real downloaded files before being relied on:
  - Only 2m_temperature and 2m_dewpoint_temperature are available (no direct
    RH variable) -- relative humidity is derived via the standard
    August-Roche-Magnus approximation, accurate to within ~0.1-0.4% for
    typical atmospheric conditions.
  - Only 10m u/v wind components are available -- speed = sqrt(u^2+v^2).
  - total_precipitation ("tp") is CUMULATIVE from 00:00 UTC that GRIB
    "time" through each hourly "step" (confirmed empirically: values rise
    monotonically from step=1 to step=24 within one day, matching ECMWF's
    documented forecast-style accumulation convention) -- NOT a per-hour
    increment. A true 24h total ending at UTC hour H on date D is:
        daily_total(D-1) - partial(D-1, H) + partial(D, H)
    where daily_total(D) = tp(time=D, step=24) and partial(D, H) =
    tp(time=D, step=H). Summing 24 raw hourly values (treating them as
    increments) would silently overcount by roughly a factor proportional
    to the accumulation curve -- this was checked against real data, not
    assumed from documentation alone.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ml import config
from collect_weather_grid_eccc import PROVINCE_UTC_OFFSET, target_utc_hour_for_offset

RAW_DIR = config.REFERENCE_DIR / "era5"


def _unzipped_grib_path(zip_path: Path) -> Path:
    """era5_client.py's downloads are zip-wrapped regardless of the
    requested format (verified for both netcdf and grib) -- extract once,
    cache the extracted path alongside the zip."""
    extract_dir = zip_path.with_suffix("")  # e.g. hourly_2019.grib -> hourly_2019/
    extract_dir.mkdir(exist_ok=True)
    grib_path = extract_dir / "data.grib"
    if not grib_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert len(names) == 1, f"expected exactly one file in {zip_path}, got {names}"
            with zf.open(names[0]) as src, open(grib_path, "wb") as dst:
                dst.write(src.read())
    return grib_path


def load_hourly(year: int) -> xr.Dataset:
    path = _unzipped_grib_path(RAW_DIR / f"hourly_{year}.grib")
    return xr.open_dataset(path, engine="cfgrib")


def load_precip(year: int) -> xr.Dataset:
    path = _unzipped_grib_path(RAW_DIR / f"precip_{year}.grib")
    return xr.open_dataset(path, engine="cfgrib")


def relative_humidity_pct(t2m_k, d2m_k):
    """August-Roche-Magnus approximation. Inputs in Kelvin."""
    t_c = t2m_k - 273.15
    td_c = d2m_k - 273.15
    numerator = np.exp((17.625 * td_c) / (243.04 + td_c))
    denominator = np.exp((17.625 * t_c) / (243.04 + t_c))
    return np.clip(100.0 * numerator / denominator, 0.0, 100.0)


def wind_speed_kmh(u10, v10):
    return np.sqrt(u10 ** 2 + v10 ** 2) * 3.6  # m/s -> km/h


def extract_year(year: int, grid_domain: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell_id, date) with temperature/humidity/wind_speed/
    pressure/precip_24h_mm in fire_risk.py's exact units. `grid_domain`
    should already be filtered to in-domain cells."""
    print(f"Loading ERA5-Land GRIB files for {year}...")
    hourly = load_hourly(year)
    precip = load_precip(year)

    grid_domain = grid_domain.copy()
    grid_domain["target_utc_hour"] = grid_domain["province"].map(
        lambda p: target_utc_hour_for_offset(PROVINCE_UTC_OFFSET.get(p, PROVINCE_UTC_OFFSET["Unknown"]))
    )

    all_frames = []
    for target_hour, group in grid_domain.groupby("target_utc_hour"):
        print(f"  UTC hour {target_hour}: {len(group)} cells")
        lat_da = xr.DataArray(group["lat"].values, dims="cell")
        lon_da = xr.DataArray(group["lon"].values, dims="cell")

        step = pd.Timedelta(hours=int(target_hour))
        h = hourly.sel(step=step, method="nearest").sel(latitude=lat_da, longitude=lon_da, method="nearest")

        temperature_c = (h["t2m"] - 273.15).values          # (time, cell)
        humidity_pct = relative_humidity_pct(h["t2m"].values, h["d2m"].values)
        wind_kmh = wind_speed_kmh(h["u10"].values, h["v10"].values)
        pressure_hpa = (h["sp"] / 100.0).values

        # ---- precip: true 24h total ending at target_hour, per the
        # cumulative-accumulation formula derived from real observed data
        # (see module docstring) ----
        p_sel = precip.sel(latitude=lat_da, longitude=lon_da, method="nearest")
        daily_total = p_sel["tp"].sel(step=pd.Timedelta(hours=24), method="nearest")   # (time, cell)
        partial = p_sel["tp"].sel(step=step, method="nearest")                         # (time, cell)

        dates = pd.to_datetime(h["time"].values)
        daily_total_by_date = dict(zip(pd.to_datetime(daily_total["time"].values), daily_total.values))
        partial_by_date = dict(zip(pd.to_datetime(partial["time"].values), partial.values))

        precip_24h_mm = np.full((len(dates), len(group)), np.nan)
        for i, d in enumerate(dates):
            prev = d - pd.Timedelta(days=1)
            if prev not in daily_total_by_date or d not in partial_by_date or prev not in partial_by_date:
                continue  # first date in range with no lead-in data -- dropped, not fabricated
            precip_24h_mm[i, :] = (
                daily_total_by_date[prev] - partial_by_date[prev] + partial_by_date[d]
            ) * 1000.0  # metres -> mm

        for i, cell_id in enumerate(group["cell_id"].values):
            frame = pd.DataFrame({
                "cell_id": cell_id,
                "date": dates,
                "temperature": temperature_c[:, i],
                "humidity": humidity_pct[:, i],
                "wind_speed": wind_kmh[:, i],
                "pressure": pressure_hpa[:, i],
                "precip_24h_mm": precip_24h_mm[:, i],
            })
            all_frames.append(frame)

    out = pd.concat(all_frames, ignore_index=True)
    out = out.dropna(subset=["precip_24h_mm"]).reset_index(drop=True)  # drops the lead-in day itself
    out["precip_24h_mm"] = out["precip_24h_mm"].clip(lower=0.0)  # sub-1e-4mm float noise, not real negative precip
    return out
