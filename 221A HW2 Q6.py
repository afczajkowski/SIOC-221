#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:14:37 2025

@author: auroraczajkowski
"""

#import packages
import os, math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

#import files 
files = {
    2014: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2014.nc",
    2020: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2020.nc",
}
years_eval = [2014, 2020]
var_names = ["temperature", "pressure"]
Ns = [1, 2, 4, 8, 16]  # averaging block sizes; feel free to tweak

#load years and 
year_series = {}  # {year: DataFrame with requested columns}
for y in years_eval:
    path = files.get(y)
    if not path or not os.path.exists(path):
        print(f"WARNING: missing {y}: {path}")
        continue
    ds = xr.open_dataset(path, decode_times=True)
    have = [vn for vn in var_names if vn in ds.variables]
    if not have:
        print(f"WARNING: {y} has none of {var_names}")
        continue
    df = ds[have].convert_calendar("proleptic_gregorian").to_pandas().dropna()
    year_series[y] = df

if not year_series:
    raise RuntimeError("No data loaded for requested years.")

# ---- HELPERS ----
def common_bins(series_list, dbin=0.1):
    """Common robust bins from pooled 1–99th percentiles across a list of 1-D arrays."""
    allv = np.concatenate([v[np.isfinite(v)] for v in series_list if v.size])
    p1, p99 = np.percentile(allv, [1, 99])
    lo = np.floor(p1)
    hi = np.ceil(p99)
    edges = np.arange(lo, hi + dbin, dbin)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

def block_means(x, N):
    """Non-overlapping N-sample block averages; drop tail if not divisible."""
    x = np.asarray(x)
    n = (len(x) // N) * N
    if n == 0:  # too short
        return np.array([])
    xb = x[:n].reshape(-1, N).mean(axis=1)
    return xb

def gaussian_pdf(x, mu, sigma):
    if sigma <= 0 or not np.isfinite(sigma):  # avoid degenerate
        return np.full_like(x, np.nan)
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

#figure 1: 2x2 PDFs for (temperature, pressure) × (2014, 2020) under averaging ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
axes = np.asarray(axes)
row_labels = {"temperature": "Temperature (°C)", "pressure": "Pressure (hPa)"}

for r, vn in enumerate(var_names):
    #choose bin width by variable scale:
    dbin = 0.1 if vn == "temperature" else 0.5
    #build common bins for this variable using raw (N=1) data of both years:
    pooled_vals = []
    for y in years_eval:
        df = year_series.get(y)
        if (df is not None) and (vn in df.columns):
            v = df[vn].to_numpy()
            v = v[np.isfinite(v)]
            if v.size:
                pooled_vals.append(v)
    if not pooled_vals:
        #no data for this variable at all
        for c in range(2):
            ax = axes[r, c]
            ax.text(0.5, 0.5, f"No finite {vn} data", ha='center', va='center')
            ax.axis('off')
        continue
    edges, centers = common_bins(pooled_vals, dbin=dbin)

    for c, y in enumerate(years_eval):
        ax = axes[r, c]
        df = year_series.get(y)
        if df is None or vn not in df.columns:
            ax.text(0.5, 0.5, f"{vn} missing in {y}", ha='center', va='center')
            ax.axis('off')
            continue

        raw = df[vn].to_numpy()
        raw = raw[np.isfinite(raw)]

        #plot empirical PDFs for several averaging sizes N
        for N in Ns:
            xN = block_means(raw, N)
            if xN.size == 0:
                continue
            pdf_vals, _ = np.histogram(xN, bins=edges, density=True)
            ax.plot(centers, pdf_vals, linewidth=1.2, label=f"N={N}")

        #also overplot Gaussian with observed μ, σ/√N for the *largest* N as a reference
        if raw.size:
            mu = np.mean(raw)
            sigma = np.std(raw, ddof=0)
            Nref = max(Ns)
            gpdf = gaussian_pdf(centers, mu, sigma/np.sqrt(Nref))
            if np.isfinite(gpdf).all():
                ax.plot(centers, gpdf, linestyle="--", linewidth=1.3, label=f"Gaussian μ, σ/√{Nref}")

        ax.set_title(f"{vn.capitalize()} — {y}")
        ax.set_xlabel(row_labels.get(vn, vn))
        ax.set_ylabel("PDF")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, ncol=2, loc="best")

fig.suptitle("Variance Reduction by Averaging — PDFs for 2014 & 2020", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

