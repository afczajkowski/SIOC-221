#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:19:51 2025

@author: auroraczajkowski
"""

# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

# define file paths for both years of data
files = {
    2015: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 3 data/NOAA46047year2015.txt",
    2016: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 3 data/NOAA46047year2016.txt",
}

# define column names
cols = ["YY","MM","DD","hh","mm","WDIR","WSPD","GST","WVHT","DPD","APD","MWD",
        "PRES","ATMP","WTMP","DEWP","VIS","TIDE"]

def load_noaa(path):
    """Read NOAA/NDBC whitespace-delimited text, build datetime index,
    and clean sentinel values (99/999/etc) for physical variables."""
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

    # datetime index
    df["datetime"] = pd.to_datetime(
        dict(year=df["YY"], month=df["MM"], day=df["DD"], hour=df["hh"], minute=df["mm"]),
        errors="coerce"
    )
    df = df.set_index("datetime").sort_index()

    # ensure phys columns are numeric (NaNs preserved)
    df[phys_cols] = df[phys_cols].apply(pd.to_numeric, errors="coerce")
    return df

# load (after defining load_noaa)
df15 = load_noaa(files[2015])
df16 = load_noaa(files[2016])

# variables (rows) and labels
vars_rows = [
    ("WSPD", "Wind Speed (m/s)"),
    ("WVHT", "Wave Height (m)"),
    ("WTMP", "Water Temperature (°C)"),
    ("ATMP", "Air Temperature (°C)"),
]

# define colors for each year
year_colors = {2015: "#FF69B4",  # pink
               2016: "#800080"}  # purple

# build 4×2 figure
fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex="col")
years = [(2015, df15), (2016, df16)]

# titles for columns
axes[0, 0].set_title("2015", fontsize=13)
axes[0, 1].set_title("2016", fontsize=13)

# loop over variables and years
for r, (var, label) in enumerate(vars_rows):
    for c, (yr, df) in enumerate(years):
        ax = axes[r, c]
        if var in df.columns:
            ax.plot(df.index, df[var], color=year_colors[yr], linewidth=0.8)
        else:
            ax.text(0.5, 0.5, f"{var} not found", ha="center", va="center", transform=ax.transAxes)

        # custom shared y-limits
        if var == "WSPD":
            ax.set_ylim(0, 20)
        elif var == "WVHT":
            ax.set_ylim(0, 6)
        elif var == "WTMP":
            ax.set_ylim(10, 25)
        elif var == "ATMP":
            ax.set_ylim(5, 30)

        ax.set_ylabel(label, fontsize=11)
        ax.grid(alpha=0.3)
        if r == len(vars_rows) - 1:
            ax.set_xlabel("Date", fontsize=11)

plt.tight_layout()
plt.show()
