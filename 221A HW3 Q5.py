#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 13:17:12 2025

@author: auroraczajkowski
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# file paths
files = {
    2015: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 3 data/NOAA46047year2015.txt",
    2016: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 3 data/NOAA46047year2016.txt",
}

# columns & variables
cols = ["YY","MM","DD","hh","mm","WDIR","WSPD","GST","WVHT","DPD","APD","MWD",
        "PRES","ATMP","WTMP","DEWP","VIS","TIDE"]
phys_cols = ["WSPD","GST","WVHT","DPD","APD","MWD","PRES","ATMP","WTMP","DEWP","VIS","TIDE"]
vars_target = ["WSPD", "WVHT", "WTMP", "ATMP"]

# colors
color_2015 = "#FF69B4"  # pink
color_2016 = "#800080"  # purple

def load_noaa(path):
    """Read NDBC whitespace text, skip # lines, set datetime, convert missing sentinels to NaN."""
    na_map = {c: [99, 99.0, 99.00, 999, 999.0, 999.00, 9999, 9999.0] for c in phys_cols}
    df = pd.read_csv(path, delim_whitespace=True, comment="#",
                     names=cols, header=None, na_values=na_map)
    df["datetime"] = pd.to_datetime(
        dict(year=df["YY"], month=df["MM"], day=df["DD"], hour=df["hh"], minute=df["mm"]),
        errors="coerce"
    )
    df = df.set_index("datetime").sort_index()
    df[phys_cols] = df[phys_cols].apply(pd.to_numeric, errors="coerce")
    return df

def monthly_mean_sem_from_daily(daily_series):
    """Monthly mean & SEM using daily means and n_eff=floor(days/7), min 1."""
    out = []
    for mo, g in daily_series.groupby(daily_series.index.month):
        g = g.dropna()
        if g.empty:
            out.append((mo, np.nan, np.nan))
            continue
        mean_val = g.mean()
        std_val  = g.std(ddof=1)
        n_eff    = max(int(np.floor(g.size / 7.0)), 1)
        sem_val  = (std_val / np.sqrt(n_eff)) if np.isfinite(std_val) else np.nan
        out.append((mo, mean_val, sem_val))
    res = pd.DataFrame(out, columns=["month","mean","sem"]).set_index("month").reindex(range(1,13))
    return res

def fit_annual_cycle_from_daily(daily_series):
    """Fit y = a0 + a1*sin(ωt) + a2*cos(ωt) to daily means (ω=2π/365)."""
    s = daily_series.dropna()
    if s.empty:
        return np.nan, np.nan, np.nan
    t = (s.index - s.index[0]).days.values.astype(float)
    omega = 2 * np.pi / 365.0
    X = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    y = s.values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1, a2 = coeffs
    return a0, a1, a2

def fit_annual_semiannual_from_daily(daily_series):
    """Fit y = a0 + a1*sin(ωt)+a2*cos(ωt)+b1*sin(2ωt)+b2*cos(2ωt)."""
    s = daily_series.dropna()
    if s.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    t = (s.index - s.index[0]).days.values.astype(float)
    omega = 2 * np.pi / 365.0
    X = np.column_stack([
        np.ones_like(t),
        np.sin(omega * t), np.cos(omega * t),
        np.sin(2 * omega * t), np.cos(2 * omega * t)
    ])
    y = s.values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs  # a0, a1, a2, b1, b2

# load & daily-average 
df_year = {yr: load_noaa(path) for yr, path in files.items()}
df_daily = {yr: df.resample("1D").mean() for yr, df in df_year.items()}

# compute monthly stats + fits for each var & year 
stats = {}   # {var: {yr: DataFrame(month-> mean, sem)}}
fitsA  = {}  # annual-only
fitsB  = {}  # annual+semi
for var in vars_target:
    stats[var] = {}
    fitsA[var] = {}
    fitsB[var] = {}
    for yr in files:
        daily = df_daily[yr][var]
        stats[var][yr] = monthly_mean_sem_from_daily(daily)
        fitsA[var][yr] = fit_annual_cycle_from_daily(daily)
        fitsB[var][yr] = fit_annual_semiannual_from_daily(daily)

# plotting (2x2): monthly means±SEM + fitted curves 
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
axes = axes.ravel()
month_ticks  = np.arange(1, 13)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
titles = {"WSPD": "Wind Speed (m/s)",
          "WVHT": "Wave Height (m)",
          "WTMP": "Water Temperature (°C)",
          "ATMP": "Air Temperature (°C)"}

omega = 2 * np.pi / 365.0
t_grid = np.arange(0, 365.0)
x_fit = 1 + 11.0 * (t_grid / 364.0)

for ax, var in zip(axes, vars_target):
    m2015 = stats[var][2015]
    m2016 = stats[var][2016]

    ax.errorbar(m2015.index, m2015["mean"], yerr=m2015["sem"],
                fmt="o", linewidth=1.2, capsize=3, color=color_2015, label="2015 mean±SEM")
    ax.errorbar(m2016.index, m2016["mean"], yerr=m2016["sem"],
                fmt="s", linewidth=1.2, capsize=3, color=color_2016, label="2016 mean±SEM")

    # Fitted curves (annual-only solid, annual+semi dashed)
    a0_15, a1_15, a2_15 = fitsA[var][2015]
    a0_16, a1_16, a2_16 = fitsA[var][2016]
    a0_15b, a1_15b, a2_15b, b1_15, b2_15 = fitsB[var][2015]
    a0_16b, a1_16b, a2_16b, b1_16, b2_16 = fitsB[var][2016]

    yA15 = a0_15 + a1_15*np.sin(omega*t_grid) + a2_15*np.cos(omega*t_grid)
    yA16 = a0_16 + a1_16*np.sin(omega*t_grid) + a2_16*np.cos(omega*t_grid)
    yB15 = a0_15b + a1_15b*np.sin(omega*t_grid) + a2_15b*np.cos(omega*t_grid) + b1_15*np.sin(2*omega*t_grid) + b2_15*np.cos(2*omega*t_grid)
    yB16 = a0_16b + a1_16b*np.sin(omega*t_grid) + a2_16b*np.cos(omega*t_grid) + b1_16*np.sin(2*omega*t_grid) + b2_16*np.cos(2*omega*t_grid)

    ax.plot(x_fit, yA15, color=color_2015, linestyle=":", linewidth=1.0)
    ax.plot(x_fit, yA16, color=color_2016, linestyle=":", linewidth=1.0)
    ax.plot(x_fit, yB15, color=color_2015, linestyle="-", linewidth=1.8, alpha=0.85)
    ax.plot(x_fit, yB16, color=color_2016, linestyle="--", linewidth=1.8, alpha=0.85)

    ax.set_title(titles[var], fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)

# legend inside WTMP panel (index 2)
axes[2].legend(frameon=False, loc="upper left", fontsize=9)

for i in [2, 3]:
    axes[i].set_xlabel("Month", fontsize=12)

fig.text(0.06, 0.5, "Value (monthly means ± SEM)  with annual and semi-annual fits",
         va="center", rotation="vertical", fontsize=12)
plt.tight_layout(rect=[0.06, 0.06, 1, 1])
plt.show()

# ---- χ² Misfit Comparison ----
def model_misfit(y_obs, y_fit, sigma2=1.0, M=3):
    resid = y_obs - y_fit
    N = len(resid)
    chi2_val = np.sum((resid**2) / sigma2)
    red_chi2 = chi2_val / (N - M)
    mse = np.mean(resid**2)
    return chi2_val, red_chi2, mse

misfits = []
for var in vars_target:
    for yr, df in df_daily.items():
        y_obs = df[var].dropna()
        if y_obs.empty:
            continue
        t = (y_obs.index - y_obs.index[0]).days.values.astype(float)
        N = len(t)
        omega = 2 * np.pi / 365.0

        # annual-only model
        a0, a1, a2 = fitsA[var][yr]
        yA = a0 + a1*np.sin(omega*t) + a2*np.cos(omega*t)
        chiA, redA, mseA = model_misfit(y_obs.values, yA, sigma2=np.var(y_obs), M=3)

        # annual+semi model
        a0b, a1b, a2b, b1, b2 = fitsB[var][yr]
        yB = a0b + a1b*np.sin(omega*t) + a2b*np.cos(omega*t) + b1*np.sin(2*omega*t) + b2*np.cos(2*omega*t)
        chiB, redB, mseB = model_misfit(y_obs.values, yB, sigma2=np.var(y_obs), M=5)

        nuA = N - 3
        nuB = N - 5
        delta = chiA - chiB
        df_diff = nuA - nuB
        p_val = 1 - chi2.cdf(delta, df_diff)

        misfits.append({
            "variable": var,
            "year": yr,
            "N": N,
            "χ²_annual": chiA,
            "χ²_red_annual": redA,
            "χ²_annual+semi": chiB,
            "χ²_red_annual+semi": redB,
            "p(improvement)": p_val
        })

misfit_df = pd.DataFrame(misfits)
print("\n=== χ² Misfit Comparison: Annual vs Annual+Semi-Annual ===")
print(misfit_df.to_string(index=False, float_format=lambda x: f"{x:9.3f}"))
