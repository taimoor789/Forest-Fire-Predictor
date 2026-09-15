"""
Regression test for the Canadian FWI System formulas in fire_risk.py.

The oracle here is an independent, direct transcription of the official
cffdrs R package source (github.com/cran/cffdrs/tree/master/R --
fine_fuel_moisture_code.r, duff_moisture_code.r, drought_code.r,
initial_spread_index.r, buildup_index.r, fire_weather_index.r), which cites
Van Wagner & Pickett (1985) and Van Wagner (1987) directly. It is written
from scratch here rather than copied from fire_risk.py, specifically so a
bug shared between the two wouldn't hide from this test.

The reference uses the R package's more precise constant
250*59.5/101 = 147.27722... in the FFMC/ISI moisture-content conversion;
fire_risk.py uses the standard textbook rounding of 147.2 (used throughout
the Van Wagner 1987 publication and most implementations). That's a real,
understood, ~0.05% difference -- not a bug -- so comparisons use a loose
enough tolerance to pass on that rounding while still failing hard on an
actual formula error (e.g. the pre-fix DC bug was a ~30% error, and the
pre-fix DMC bug produced single-day jumps of 80-100+ points -- both wildly
outside this tolerance).

Run directly: python test_fwi_formulas.py
"""

import sys
import numpy as np

from fire_risk import CanadianFireWeatherIndex

fwi_calc = CanadianFireWeatherIndex()


# ---- Independent oracle: official cffdrs formulas, transcribed from source ----

def r_ffmc(ffmc_yda, temp, rh, ws, prec):
    C = 250.0 * 59.5 / 101.0
    wmo = C * (101 - ffmc_yda) / (59.5 + ffmc_yda)
    if prec > 0.5:
        ra = prec - 0.5
        if wmo > 150:
            wmo = (wmo + 0.0015 * (wmo - 150) ** 2 * np.sqrt(ra)
                   + 42.5 * ra * np.exp(-100 / (251 - wmo)) * (1 - np.exp(-6.93 / ra)))
        else:
            wmo = wmo + 42.5 * ra * np.exp(-100 / (251 - wmo)) * (1 - np.exp(-6.93 / ra))
        wmo = min(wmo, 250)
    ed = 0.942 * rh ** 0.679 + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh))
    ew = 0.618 * rh ** 0.753 + 10 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh))
    if wmo > ed:
        z = 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - (rh / 100) ** 8)
        x = z * 0.581 * np.exp(0.0365 * temp)
        wm = ed + (wmo - ed) / 10 ** x
    elif wmo < ew:
        z = 0.424 * (1 - ((100 - rh) / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - ((100 - rh) / 100) ** 8)
        x = z * 0.581 * np.exp(0.0365 * temp)
        wm = ew - (ew - wmo) / 10 ** x
    else:
        wm = wmo
    ffmc1 = 59.5 * (250 - wm) / (C + wm)
    return min(max(ffmc1, 0), 101)


def r_dmc(dmc_yda, temp, rh, prec, mon):
    ell01 = [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6]
    temp = max(temp, -1.1)
    rk = 1.894 * (temp + 1.1) * (100 - rh) * ell01[mon - 1] * 1e-4
    if prec <= 1.5:
        pr = dmc_yda
    else:
        rw = 0.92 * prec - 1.27
        wmi = 20 + 280 / np.exp(0.023 * dmc_yda)
        if dmc_yda <= 33:
            b = 100 / (0.5 + 0.3 * dmc_yda)
        elif dmc_yda <= 65:
            b = 14 - 1.3 * np.log(dmc_yda)
        else:
            b = 6.2 * np.log(dmc_yda) - 17.2
        wmr = wmi + 1000 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - np.log(wmr - 20))
    pr = max(pr, 0)
    return max(pr + rk, 0)


def r_dc(dc_yda, temp, prec, mon):
    fl01 = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5, 2.4, 0.4, -1.6, -1.6]
    temp = max(temp, -2.8)
    pe = (0.36 * (temp + 2.8) + fl01[mon - 1]) / 2
    pe = max(pe, 0)
    if prec <= 2.8:
        dr = dc_yda
    else:
        rw = 0.83 * prec - 1.27
        smi = 800 * np.exp(-dc_yda / 400)
        dr0 = dc_yda - 400 * np.log(1 + 3.937 * rw / smi)
        dr = max(dr0, 0)
    return max(dr + pe, 0)


def r_isi(ffmc, ws):
    C = 250.0 * 59.5 / 101.0
    fm = C * (101 - ffmc) / (59.5 + ffmc)
    fw = np.exp(0.05039 * ws)
    ff = 91.9 * np.exp(-0.1386 * fm) * (1 + fm ** 5.31 / 49300000)
    return 0.208 * fw * ff


def r_bui(dmc, dc):
    bui1 = 0 if (dmc == 0 and dc == 0) else 0.8 * dc * dmc / (dmc + 0.4 * dc)
    if bui1 < dmc:
        p = 0 if dmc == 0 else (dmc - bui1) / dmc
        cc = 0.92 + (0.0114 * dmc) ** 1.7
        return max(dmc - cc * p, 0)
    return bui1


def r_fwi(isi, bui):
    if bui > 80:
        bb = 0.1 * isi * (1000 / (25 + 108.64 / np.exp(0.023 * bui)))
    else:
        bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2)
    if bb <= 1:
        return bb
    return np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647)


# ---- Scenarios: cover the no-rain path, both DMC/DC rain branches (above
# and below their respective thresholds), a cold-clamp day that still gets
# rain-wetted, and each DMC b-coefficient branch (<=33, <=65, >65). ----

SCENARIOS = [
    # name, ffmc_yda, dmc_yda, dc_yda, temp, rh, ws, rain, month
    ("no_rain_summer",              85, 6,   15,  17,  42, 25, 0.0,  7),
    ("heavy_rain_both_thresholds",  60, 40,  300, 22,  55, 10, 25.0, 8),
    ("light_rain_below_threshold",  90, 20,  100, 18,  60, 15, 1.0,  6),
    ("cold_clamp_with_rain",        70, 3,   10,  -5,  80, 20, 4.0,  1),
    ("dmc_b_branch_le33",           80, 10,  50,  20,  50, 15, 5.0,  5),
    ("dmc_b_branch_le65",           80, 50,  150, 20,  50, 15, 5.0,  6),
    ("dmc_b_branch_gt65",           80, 200, 500, 25,  30, 30, 5.0,  9),
]

REL_TOL = 0.02   # 2%: comfortably covers the known 147.2-vs-147.2772
ABS_TOL = 0.05   # constant rounding, well under any real formula error


def close_enough(actual, expected):
    return abs(actual - expected) <= max(ABS_TOL, REL_TOL * abs(expected))


def run():
    failures = []
    for name, ffmc_y, dmc_y, dc_y, temp, rh, ws, rain, mon in SCENARIOS:
        expected_ffmc = r_ffmc(ffmc_y, temp, rh, ws, rain)
        expected_dmc = r_dmc(dmc_y, temp, rh, rain, mon)
        expected_dc = r_dc(dc_y, temp, rain, mon)
        expected_isi = r_isi(expected_ffmc, ws)
        expected_bui = r_bui(expected_dmc, expected_dc)
        expected_fwi = r_fwi(expected_isi, expected_bui)

        actual_ffmc = fwi_calc.calculate_ffmc(temp, rh, ws, rain, ffmc_y)
        actual_dmc = fwi_calc.calculate_dmc(temp, rh, rain, dmc_y, mon)
        actual_dc = fwi_calc.calculate_dc(temp, rain, dc_y, mon)
        actual_isi = fwi_calc.calculate_isi(ws, actual_ffmc)
        actual_bui = fwi_calc.calculate_bui(actual_dmc, actual_dc)
        actual_fwi = fwi_calc.calculate_fwi(actual_isi, actual_bui)

        for label, actual, expected in [
            ("FFMC", actual_ffmc, expected_ffmc),
            ("DMC", actual_dmc, expected_dmc),
            ("DC", actual_dc, expected_dc),
            ("ISI", actual_isi, expected_isi),
            ("BUI", actual_bui, expected_bui),
            ("FWI", actual_fwi, expected_fwi),
        ]:
            ok = close_enough(actual, expected)
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] {name:32s} {label:4s} actual={actual:10.4f}  expected={expected:10.4f}")
            if not ok:
                failures.append(f"{name}/{label}: actual={actual} expected={expected}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(SCENARIOS) * 6} checks passed across {len(SCENARIOS)} scenarios.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
