#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:03:06 2025

@author: auroraczajkowski
"""

#import necessary packages 
import os
import xarray as xr
import pandas as pd
    #importing but technically unused because built into xarray 
import numpy as np
import matplotlib.pyplot as plt

#load all files and define what variable we are looking for 
files = {
    2005: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2005.nc",
    2006: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2006.nc",
    2007: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2007.nc",
    2008: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2008.nc",
    2009: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2009.nc",
    2010: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2010.nc",
    2011: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2011.nc",
    2012: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2012.nc",
    2013: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2013.nc",
    2014: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2014.nc",
    2015: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2015.nc",
    2016: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2016.nc",
    2017: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2017.nc",
    2018: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2018.nc",
    2019: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2019.nc",
    2020: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2020.nc",
    2021: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2021.nc",
}
var_name = "temperature"

#setting limits and fontsizes for all figures 
TIMESERIES_YLIM = (10, 30)   # set to None for auto
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 11
SUPTITLE_FONTSIZE = 16

#loading years into pandas and changing time to gregorian from julian 
year_series = {}
for y, path in sorted(files.items()):
    if not os.path.exists(path):
        print(f"WARNING: missing {y}: {path}")
        continue
    ds = xr.open_dataset(path, decode_times=True)
    s = ds[var_name].convert_calendar("proleptic_gregorian").to_pandas().dropna()
    year_series[y] = s

#sanity check to make sure code has loaded all the files properly. i like to have this when working with more than one file
years = list(year_series.keys())
n = len(years)
if n == 0:
    raise RuntimeError("No data loaded. Check your file paths.")

#start making the 17 panel figure. i'm defining it as 20 panels and hiding what i don't use 
def make_axes_grid(title, rows=4, cols=5, sharex=False, sharey=False):
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12), sharex=sharex, sharey=sharey)
    axes = axes.ravel()
    # basic styling & per-panel titles now done in the plotting loops
    fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes

#turning off exrtra axes because i defined a 4x5 (20 panel) fiure 
def finalize_grid(axes, used):
    for j in range(used, len(axes)):
        axes[j].axis("off")

#time series figure 
fig, axes = make_axes_grid("Scripps Pier SST — Time Series",
                           sharex=False, sharey=False)

for i, y in enumerate(years):
    ax = axes[i]
    s = year_series[y]
    s.plot(ax=ax, lw=0.8)
    ax.set_title(str(y), fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Temp (°C)", fontsize=LABEL_FONTSIZE)
    if TIMESERIES_YLIM is not None:
        ax.set_ylim(*TIMESERIES_YLIM)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    # rotate x tick labels for readability
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.5, w_pad=1.0, h_pad=1.0)
finalize_grid(axes, len(years))
plt.show()

#define binning for pdf 
#robust pooled range (1–99 percentile) so shapes are comparable
all_vals = np.concatenate([s.values for s in year_series.values()])
all_vals = all_vals[np.isfinite(all_vals)]
p1, p99 = np.percentile(all_vals, [1, 99])
dbin = 0.1
edges = np.arange(np.floor(p1), np.ceil(p99) + dbin, dbin)
centers = (edges[:-1] + edges[1:]) / 2

#pdf figure 
fig, axes = make_axes_grid("Scripps Pier SST — PDFs (density=True, common bins)",
                           sharex=False, sharey=False)

for i, y in enumerate(years):
    ax = axes[i]
    vals = year_series[y].values
    pdf_vals, _ = np.histogram(vals, bins=edges, density=True)
    ax.plot(centers, pdf_vals)
    ax.set_title(str(y), fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Temp (°C)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("PDF", fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.5, w_pad=1.0, h_pad=1.0)
finalize_grid(axes, len(years))
plt.show()

# loop through each year and calculate mean + stdev
stats = []
for y, s in year_series.items():
    mean_temp = s.mean()
    std_temp = s.std()
    n = s.size  # number of samples

    print(f"{y}: mean = {mean_temp:.2f} °C, std = {std_temp:.2f} °C, n = {n}")
    stats.append({"year": y, "mean": mean_temp, "std": std_temp, "n": n})