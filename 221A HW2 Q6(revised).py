#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:16:06 2025

@author: auroraczajkowski
"""
#import necessary packages
import os
import numpy as np
import pandas as pd
import xarray as xr

#import files
FILES = {
    2014: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2014.nc",
    2020: "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2020.nc",
}
VAR_NAMES = ["temperature", "pressure"]
YEARS = [2014, 2020]


def _find_time_dim(da_or_ds):
    for cand in ("time", "Time", "datetime", "date"):
        if (hasattr(da_or_ds, "dims") and cand in da_or_ds.dims) or \
           (hasattr(da_or_ds, "coords") and cand in da_or_ds.coords):
            return cand
    raise ValueError("No time dimension/coordinate found.")

def _to_proleptic(da, time_dim=None):
    if time_dim is None:
        time_dim = _find_time_dim(da)
    try:
        return da.convert_calendar("proleptic_gregorian", dim=time_dim, use_cftime=False)
    except Exception:
        return da

def da_to_series(da):
    tdim = _find_time_dim(da)
    da = _to_proleptic(da, time_dim=tdim)
    s = da.to_series()
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        s.index = pd.to_datetime(da[tdim].values)
    return s.sort_index()

def nonoverlapping_block_means(series, N):
    """Compute non-overlapping block means of length N (truncates tail)."""
    a = pd.Series(series).dropna().to_numpy()
    m = len(a) // N
    if m == 0:
        return np.array([])
    return a[:m*N].reshape(m, N).mean(axis=1)

#load data
year_series = {}
for y, path in FILES.items():
    if not os.path.exists(path):
        print(f"WARNING: missing {y}: {path}")
        continue
    ds = xr.open_dataset(path, decode_times=True)
    per_var = {}
    for vn in VAR_NAMES:
        if vn in ds.variables:
            per_var[vn] = da_to_series(ds[vn]).dropna()
        else:
            print(f"WARNING: {vn} not found in {os.path.basename(path)} (year {y})")
    year_series[y] = per_var

#compute and print variances with increasing N
N_list = [1, 2, 4, 8, 16, 32, 64, 128]

print("\n=== Variance of non-overlapping N-sample means ===")
for y in YEARS:
    if y not in year_series:
        continue
    for vn in VAR_NAMES:
        s = year_series[y].get(vn)
        if s is None or s.empty:
            continue

        base_var = s.var()  # variance of full-resolution data
        print(f"\n{vn.capitalize()} — {y}")
        print("   N    var(block means)      base_var/N (iid ref)     ratio[varN / (base/N)]")

        for N in N_list:
            bmeans = nonoverlapping_block_means(s.values, N)
            if bmeans.size < 2:
                continue
            varN = bmeans.var(ddof=1)
            ref  = base_var / N
            ratio = varN / ref if ref > 0 else np.nan
            print(f"{N:4d}   {varN:16.6g}   {ref:18.6g}        {ratio:10.3f}")
