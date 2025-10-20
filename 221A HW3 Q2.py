#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:52:02 2025

@author: auroraczajkowski
"""

# Monthly means + SEM plots (no raw data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file paths
files = {
    2015: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 3 data/NOAA46047year2015.txt",
    2016: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 3 data/NOAA46047year2016.txt",
}

# known column layout for NDBC text
cols = ["YY","MM","DD","hh","mm","WDIR","WSPD","GST","WVHT","DPD","APD","MWD",
        "PRES","ATMP","WTMP","DEWP","VIS","TIDE"]

vars_target = ["WSPD", "WVHT", "WTMP", "ATMP"]  # wind speed, wave height, water temp, air temp

def load_noaa(path):
    """Read NDBC whitespace text, skip comment lines, set datetime index,
    and mark NOAA sentinel missing values as NaN for physical variables."""
    phys_cols = ["WSPD","GST","WVHT","DPD","APD","MWD","PRES","ATMP","WTMP","DEWP","VIS","TIDE"]
    na_map = {c: [99, 99.0, 99.00, 999, 999.0, 999.00, 9999, 9999.0] for c in phys_cols}

    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment="#",
        names=cols,
        header=None,
        na_values=na_map
    )
    df["datetime"] = pd.to_datetime(
        dict(year=df["YY"], month=df["MM"], day=df["DD"], hour=df["hh"], minute=df["mm"]),
        errors="coerce"
    )
    df = df.set_index("datetime").sort_index()
    df[phys_cols] = df[phys_cols].apply(pd.to_numeric, errors="coerce")
    return df

def monthly_mean_sem(df, var):
    """
    Monthly mean and SEM for one variable using:
      - daily means first (reduce autocorrelation/bursty sampling)
      - n_eff = floor(#days_with_data / 7), min 1
      - SEM = std(daily_means, ddof=1) / sqrt(n_eff)
    Returns DataFrame indexed by month (1..12) with columns ['mean','sem'].
    """
    daily = df[var].resample("1D").mean()  # daily means
    out = []
    for mo, g in daily.groupby(daily.index.month):
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

# load data
df_year = {yr: load_noaa(path) for yr, path in files.items()}

# compute monthly stats for each variable and year
stats = {var: {yr: monthly_mean_sem(df_year[yr], var) for yr in files.keys()}
         for var in vars_target}

# plotting (2x2 panel, monthly means with SEM)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.ravel()
month_ticks = range(1, 13)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

titles = {
    "WSPD": "Wind Speed (m/s)",
    "WVHT": "Wave Height (m)",
    "WTMP": "Water Temperature (°C)",
    "ATMP": "Air Temperature (°C)"
}

for ax, var in zip(axes, vars_target):
    # 2015
    m2015 = stats[var][2015]
    ax.errorbar(m2015.index, m2015["mean"], yerr=m2015["sem"],
                fmt="-o", color="#FF69B4", linewidth=1.2, capsize=3, label="2015")
    # 2016
    m2016 = stats[var][2016]
    ax.errorbar(m2016.index, m2016["mean"], yerr=m2016["sem"],
                fmt="-s", color="#800080", linewidth=1.2, capsize=3, label="2016")

    ax.set_title(titles[var], fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, rotation=0)

# only one legend for the whole figure
axes[0].legend(frameon=False, loc="best")

# shared labels
fig.text(0.5, 0.04, "Month", ha="center", fontsize=12)
fig.text(0.06, 0.5, "Monthly Mean ± SEM", va="center", rotation="vertical", fontsize=12)

plt.tight_layout(rect=[0.06, 0.06, 1, 1])
plt.show()
