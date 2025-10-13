#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 09:24:34 2025

@author: auroraczajkowski
"""
# --- Imports ---
import os
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# --- File paths ---
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

years_eval = [2014, 2020]
var_names = ["temperature", "pressure"]   # two variables to plot

# --- Load selected years into a DataFrame per year with both variables ---
year_series = {}  # {year: DataFrame(columns=var_names)}
for y in years_eval:
    path = files.get(y, None)
    if path is None or not os.path.exists(path):
        print(f"WARNING: missing {y}: {path}")
        continue
    ds = xr.open_dataset(path, decode_times=True)
    have = [vn for vn in var_names if vn in ds.variables]
    if not have:
        print(f"WARNING: {y} has none of {var_names}")
        ds.close()
        continue
    s = ds[have].convert_calendar("proleptic_gregorian").to_pandas().dropna()
    ds.close()
    s = s[[vn for vn in var_names if vn in s.columns]]  # keep order
    if s.empty:
        print(f"WARNING: {y} produced empty DataFrame after dropna.")
        continue
    year_series[y] = s

if not year_series:
    raise RuntimeError("No data loaded for requested years. Check file paths/variable names.")

# --- Pooled, robust bins per variable (1–99th pct over both years) ---
def make_bins_for_variable(year_series, var_name, dbin=0.1):
    vals = []
    for y, df in year_series.items():
        if var_name in df.columns:
            v = df[var_name].to_numpy()
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(v)
    if not vals:
        return None, None
    allv = np.concatenate(vals)
    p1, p99 = np.percentile(allv, [1, 99])
    lo = np.floor(p1)
    hi = np.ceil(p99)
    edges = np.arange(lo, hi + dbin, dbin)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers

# --- Gaussian and uniform PDFs matching mean & variance ---
def gaussian_pdf(x, mu, sigma):
    if sigma <= 0 or not np.isfinite(sigma):
        return np.full_like(x, np.nan)
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def uniform_params_from_mean_var(mu, sigma):
    # For Uniform(a,b): mean=(a+b)/2=mu; var=(b-a)^2/12 = sigma^2
    if sigma <= 0 or not np.isfinite(sigma):
        return None, None
    width = math.sqrt(12.0) * sigma
    a = mu - 0.5 * width
    b = mu + 0.5 * width
    return a, b

def uniform_pdf(x, a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return np.full_like(x, np.nan)
    height = 1.0 / (b - a)
    out = np.zeros_like(x, dtype=float)
    mask = (x >= a) & (x <= b)
    out[mask] = height
    return out

# --- Build the 2x2 figure: rows = variables, cols = years (2014 | 2020) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
axes = np.asarray(axes)

# Titles and axis labels
var_labels = {
    "temperature": "Temperature (°C)",
    "pressure": "Pressure (hPa)"  # adjust if your units differ
}

for r, vn in enumerate(var_names):
    # bins per variable (common across both years)
    dbin = 0.1 if vn == "temperature" else 0.5
    edges, centers = make_bins_for_variable(year_series, vn, dbin=dbin)
    if edges is None:
        # no data for this variable at all
        for c, y in enumerate(years_eval):
            ax = axes[r, c]
            ax.text(0.5, 0.5, f"No finite {vn} data\nfor {years_eval}", ha='center', va='center')
            ax.axis('off')
        continue

    for c, y in enumerate(years_eval):
        ax = axes[r, c]
        df = year_series.get(y, None)
        if df is None or vn not in df.columns:
            ax.text(0.5, 0.5, f"{vn} missing in {y}", ha='center', va='center')
            ax.set_axis_off()
            continue

        vals = df[vn].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, f"No finite {vn} in {y}", ha='center', va='center')
            ax.set_axis_off()
            continue

        # histogram-based PDF (density=True)
        pdf_vals, _ = np.histogram(vals, bins=edges, density=True)

        # observed mean & std
        mu = np.mean(vals)
        sigma = np.std(vals, ddof=0)  # population sigma for consistency

        # theoretical PDFs with observed μ, σ^2
        gpdf = gaussian_pdf(centers, mu, sigma)
        a, b = uniform_params_from_mean_var(mu, sigma)
        updf = uniform_pdf(centers, a, b)

        # plot
        ax.plot(centers, pdf_vals, label="Empirical (hist)", linewidth=1.5)
        if np.isfinite(gpdf).all():
            ax.plot(centers, gpdf, linestyle="--", label="Gaussian(μ, σ²)", linewidth=1.2)
        else:
            ax.text(0.02, 0.95, "σ≈0 → Gaussian skipped", transform=ax.transAxes, va='top', fontsize=8)

        if np.isfinite(updf).any():
            ax.plot(centers, updf, linestyle=":", label="Uniform(μ, σ²)", linewidth=1.2)
        else:
            ax.text(0.02, 0.88, "σ≈0 → Uniform skipped", transform=ax.transAxes, va='top', fontsize=8)

        # Titles/labels/formatting
        ax.set_title(f"{vn.capitalize()} — {y}")
        ax.set_xlabel(var_labels.get(vn, vn))
        ax.set_ylabel("PDF")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, ncol=2, loc="best")

        # --- Set y-axis limits by variable ---
        if vn == "temperature":
            ax.set_ylim(0, 0.4)
        elif vn == "pressure":
            ax.set_ylim(0, 0.8)

fig.suptitle("Scripps Pier — PDFs for 2014 & 2020 (Empirical vs Gaussian & Uniform)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
