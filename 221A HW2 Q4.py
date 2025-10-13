#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:32:12 2025

@author: auroraczajkowski
"""

#import necessary packages 
import os
import xarray as xr
import pandas as pd
    #importing but technically unused because built into xarray 
import numpy as np
import matplotlib.pyplot as plt
import math

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
#define the variables that we are looking for 
var_names = ["temperature","pressure"]

files = {year: path for year, path in files.items() if year in [2014, 2020]}
    #defining the years we are looking for 

year_series = {}
    #this creates an empty dictionary 
for y, path in sorted(files.items()):
    #beginning of for loop to make sure we have each path 
    if not os.path.exists(path):
        print(f"WARNING: missing {y}: {path}")
        continue
    ds = xr.open_dataset(path, decode_times=True)
        #load file with xarray 
    s = ds[var_names].convert_calendar("proleptic_gregorian").to_pandas().dropna()
        #this line is pretty complicated:
            #var_names is a list of variables which returns as a dataset
            #convert_calendar converts to gregorian time for normal datetime plotting 
            #to_pandas is useful to convert xarray object to into a pandas dataframe
                #each column is a variable in a pandas dataframe
            #dropna cleans by removing rows with missing variables 
    year_series[y] = s
    
data = {}
for vn in var_names:
    if vn in ds.variables:
        da = ds[vn].convert_calendar("proleptic_gregorian")
        data[vn] = da.to_series().dropna()
    else:
        print(f"WARNING: {vn} not found in {path}")
year_series[y] = data

#sanity check to make sure code has loaded all the files properly. i like to have this when working with more than one file
years = list(year_series.keys())
n = len(years)
if n == 0:
    raise RuntimeError("No data loaded. Check your file paths.")
 
#collects variables and defines them as NumPy array 
def collect_values(year_series, var_name):
    """Return a 1-D numpy array of finite values for var_name pooled across years."""
    chunks = []
    for y, obj in year_series.items():
        # obj is either a dict of Series OR a DataFrame
        if isinstance(obj, dict):
            s = obj.get(var_name, None)
        else:  # assume DataFrame-like
            s = obj[var_name] if (hasattr(obj, "columns") and var_name in obj.columns) else None

        if s is None:
            continue

        a = np.asarray(pd.Series(s).to_numpy()).ravel()   # ensure 1-D
        a = a[np.isfinite(a)]
        if a.size:
            chunks.append(a)

    if not chunks:
        raise ValueError(f"No finite values found for variable '{var_name}'.")
    return np.concatenate(chunks, axis=0)

#pooled values for temperature
all_vals = collect_values(year_series, "temperature")

#robust pooled range (1–99 percentile) so shapes are comparable
p1, p99 = np.percentile(all_vals, [1, 99])
dbin = 0.1
edges = np.arange(np.floor(p1), np.ceil(p99) + dbin, dbin)
centers = (edges[:-1] + edges[1:]) / 2
#extreme-value likelihoods: observed vs Gaussian


years_eval = [2014, 2020]
vn = "temperature"

def gaussian_tail_z(z):
    # one-sided tail P(Z >= z) for Z~N(0,1)
    return 0.5 * math.erfc(z / math.sqrt(2.0))

results = []

for y in years_eval:
    #pull temperature series for the year
    if y not in year_series or vn not in year_series[y]:
        print(f"Skipping {y}: {vn} not available.")
        continue
    vals = year_series[y][vn].values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print(f"Skipping {y}: no finite {vn} values.")
        continue

    mu  = np.mean(vals)
    sig = np.std(vals, ddof=0)
        #population-style to match Gaussian σ

    thr = mu + 3.0 * sig

    #observed/empirical tail probability
    #direct (unbinned): fraction of samples ≥ threshold
    p_empirical_direct = np.mean(vals >= thr)

    #histogram-based: integrate PDF from threshold to max
    counts, _ = np.histogram(vals, bins=edges, density=True)
    widths = np.diff(edges)
    pdf = counts  
        #density per bin
    #ind contribution of bins whose interval is fully >= thr
    mask_full = edges[:-1] >= thr
    tail_full = np.sum(pdf[mask_full] * widths[mask_full])
    #partial contribution from the first bin that contains thr (if any)
    i_part = np.searchsorted(edges, thr) - 1
    tail_partial = 0.0
    if 0 <= i_part < len(pdf) and not mask_full[i_part]:
        #assume density is constant within bin (histogram rectangle)
        frac = (edges[i_part+1] - thr) / widths[i_part]
        tail_partial = pdf[i_part] * widths[i_part] * max(0.0, min(1.0, frac))
    p_empirical_hist = tail_partial + tail_full

    #gaussian tail probability at 3σ above μ
    #standard normal tail for z = 3 (independent of μ,σ once threshold is μ+3σ)
    p_gauss = gaussian_tail_z(3.0)

    results.append({
        "year": y,
        "mu": mu,
        "sigma": sig,
        "threshold_mu_plus_3sigma": thr,
        "p_empirical_direct": p_empirical_direct,
        "p_empirical_hist": p_empirical_hist,
        "p_gaussian_mu_sigma": p_gauss
    })

#print results
print("Extreme (>= μ + 3σ) likelihoods for temperature")
for r in results:
    y = r["year"]
    print(f"\nYear {y}")
    print(f"  mean μ = {r['mu']:.3f}, σ = {r['sigma']:.3f}, threshold = {r['threshold_mu_plus_3sigma']:.3f}")
    print(f"  Observed (direct fraction) : {r['p_empirical_direct']:.6f}")
    print(f"  Observed (hist integral)   : {r['p_empirical_hist']:.6f}")
    print(f"  Gaussian (μ,σ; z=3 tail)   : {r['p_gaussian_mu_sigma']:.6f}")
    