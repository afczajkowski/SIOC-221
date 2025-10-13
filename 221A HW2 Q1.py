#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:37:34 2025

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
#define the variables that we are looking for 
var_names = ["temperature","pressure"]

files = {year: path for year, path in files.items() if year in [2014, 2020]}


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
    
#define years for plotting 
years_to_plot = [2014, 2020]  
    #choose which years to include
years = [y for y in years_to_plot if y in year_series]

#make a 2x2 figure
n_rows = len(var_names)  # temperature, pressure
n_cols = len(years)

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(5*n_cols, 3*n_rows), sharex=False, sharey=False
)
if n_rows == 1:
    axes = np.array([axes])

for r, vn in enumerate(var_names):
    for c, y in enumerate(years):
        ax = axes[r, c]
        s = year_series[y].get(vn, None)
        if s is not None:
            ax.plot(s.index, s.values, linewidth=0.8)
        ax.set_title(f"{vn.capitalize()} — {y}", fontsize=11)
        ax.set_xlabel("Time")
        if "pressure" in vn.lower():
            ax.set_ylim(1, 7)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle("Scripps Pier — Temperature and Pressure (2014 & 2020)", fontsize=14)
plt.show()