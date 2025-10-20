#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:15:07 2025

@author: auroraczajkowski
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    t = (s.index - s.index[0]).days.values.astype(float)  # days since start
    omega = 2 * np.pi / 365.0
    X = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    y = s.values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1, a2 = coeffs
    return a0, a1, a2  # mean, sine, cosine coeffs

# load & daily-average 
df_year = {yr: load_noaa(path) for yr, path in files.items()}
df_daily = {yr: df.resample("1D").mean() for yr, df in df_year.items()}

# compute monthly stats + fits for each var & year 
stats = {}   # {var: {yr: DataFrame(month-> mean, sem)}}
fits  = {}   # {var: {yr: (a0,a1,a2)}}
for var in vars_target:
    stats[var] = {}
    fits[var]  = {}
    for yr in files:
        daily = df_daily[yr][var]
        stats[var][yr] = monthly_mean_sem_from_daily(daily)
        fits[var][yr]  = fit_annual_cycle_from_daily(daily)

# plotting (2x2): monthly means±SEM + fitted curves 
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
axes = axes.ravel()
month_ticks  = np.arange(1, 13)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
titles = {"WSPD": "Wind Speed (m/s)",
          "WVHT": "Wave Height (m)",
          "WTMP": "Water Temperature (°C)",
          "ATMP": "Air Temperature (°C)"}

for ax, var in zip(axes, vars_target):
    m2015 = stats[var][2015]
    m2016 = stats[var][2016]

    # Monthly means ± SEM
    ax.errorbar(m2015.index, m2015["mean"], yerr=m2015["sem"],
                fmt="o", linewidth=1.2, capsize=3, color=color_2015, label="2015 mean±SEM")
    ax.errorbar(m2016.index, m2016["mean"], yerr=m2016["sem"],
                fmt="s", linewidth=1.2, capsize=3, color=color_2016, label="2016 mean±SEM")

    # Fitted annual cycles
    t_grid = np.arange(0, 365.0)
    omega = 2 * np.pi / 365.0
    a0_15, a1_15, a2_15 = fits[var][2015]
    a0_16, a1_16, a2_16 = fits[var][2016]
    y_fit_2015 = a0_15 + a1_15 * np.sin(omega * t_grid) + a2_15 * np.cos(omega * t_grid)
    y_fit_2016 = a0_16 + a1_16 * np.sin(omega * t_grid) + a2_16 * np.cos(omega * t_grid)
    x_fit = 1 + 11.0 * (t_grid / 364.0)
    ax.plot(x_fit, y_fit_2015, color=color_2015, linewidth=1.5, alpha=0.8)
    ax.plot(x_fit, y_fit_2016, color=color_2016, linewidth=1.5, alpha=0.8, linestyle="--")

    ax.set_title(titles[var], fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)

# Legend placed inside upper-left panel (for example, WSPD)
axes[0].legend(frameon=False, loc="upper left", fontsize=9)

# Month labels appear under both left and right columns
for i in [2, 3]:  # bottom row of subplots
    axes[i].set_xlabel("Month", fontsize=12)

fig.text(0.06, 0.5, "Value (monthly means ± SEM)  with annual-cycle fit",
         va="center", rotation="vertical", fontsize=12)

plt.tight_layout(rect=[0.06, 0.06, 1, 1])
plt.show()

# print mean & amplitude summary (optional helpful table)
def amp(a1, a2): return np.sqrt(a1*a1 + a2*a2)
summary_rows = []
for var in vars_target:
    for yr in [2015, 2016]:
        a0, a1, a2 = fits[var][yr]
        summary_rows.append({
            "variable": var,
            "year": yr,
            "mean (a0)": a0,
            "annual amplitude": amp(a1, a2)
        })
summary = pd.DataFrame(summary_rows).sort_values(["variable","year"])
print("\n=== Mean and annual amplitude (fit to daily means) ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
