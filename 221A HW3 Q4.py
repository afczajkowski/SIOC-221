#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:25:12 2025

@author: auroraczajkowski
"""

import pandas as pd
import numpy as np

# file paths
files = {
    2015: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 3 data/NOAA46047year2015.txt",
    2016: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 3 data/NOAA46047year2016.txt",
}

cols = ["YY","MM","DD","hh","mm","WDIR","WSPD","GST","WVHT","DPD","APD","MWD",
        "PRES","ATMP","WTMP","DEWP","VIS","TIDE"]
vars_target = ["WSPD", "WVHT", "WTMP", "ATMP"]

def load_noaa(path):
    """Read and clean NOAA/NDBC txt file."""
    phys_cols = ["WSPD","GST","WVHT","DPD","APD","MWD","PRES","ATMP","WTMP","DEWP","VIS","TIDE"]
    na_map = {c: [99, 99.0, 999, 999.0, 9999, 9999.0] for c in phys_cols}
    df = pd.read_csv(path, delim_whitespace=True, comment="#",
                     names=cols, header=None, na_values=na_map)
    df["datetime"] = pd.to_datetime(
        dict(year=df["YY"], month=df["MM"], day=df["DD"], hour=df["hh"], minute=df["mm"]),
        errors="coerce"
    )
    df = df.set_index("datetime").sort_index()
    df[phys_cols] = df[phys_cols].apply(pd.to_numeric, errors="coerce")
    return df

# load and daily-average
df_year = {yr: load_noaa(path) for yr, path in files.items()}
df_day = {yr: df.resample("1D").mean() for yr, df in df_year.items()}

# augmented fit (annual + semiannual)
def fit_annual_semiannual(df, var):
    """Fit y = a0 + a1*sin(ωt)+a2*cos(ωt) + b1*sin(2ωt)+b2*cos(2ωt)."""
    s = df[var].dropna()
    if s.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    t = (s.index - s.index[0]).days.values.astype(float)
    omega = 2 * np.pi / 365.0
    # design matrix with annual + semi-annual
    X = np.column_stack([
        np.ones_like(t),
        np.sin(omega * t), np.cos(omega * t),
        np.sin(2 * omega * t), np.cos(2 * omega * t)
    ])
    y = s.values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1, a2, b1, b2 = coeffs
    annual_amp = np.sqrt(a1**2 + a2**2)
    semi_amp   = np.sqrt(b1**2 + b2**2)
    return a0, a1, a2, b1, b2, annual_amp, semi_amp

# compute fits
results = []
for var in vars_target:
    for yr, df in df_day.items():
        a0, a1, a2, b1, b2, amp_ann, amp_semi = fit_annual_semiannual(df, var)
        results.append({
            "variable": var,
            "year": yr,
            "mean (a0)": a0,
            "annual sine (a1)": a1,
            "annual cos (a2)": a2,
            "semi sine (b1)": b1,
            "semi cos (b2)": b2,
            "annual amplitude": amp_ann,
            "semi-annual amplitude": amp_semi
        })

res_df = pd.DataFrame(results)
print("\n=== Annual + Semi-Annual Least Squares Fit Results ===")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
