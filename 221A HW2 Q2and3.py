#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:12:54 2025

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
    
#compute means for each year 
print("\n=== Mean values by year ===")

for y, data in year_series.items():
    print(f"\nYear: {y}")
    for vn in var_names:
        s = data[vn]

        #raw mean (using all data points)
        raw_mean = s.mean()

        #subsampled (daily) mean
        daily_mean = s.resample("1D").mean().mean()

        print(f"  {vn.capitalize()}: raw mean = {raw_mean:.3f}, daily mean = {daily_mean:.3f}")
        
#compute variance for each year
print("\n=== Variance comparison by year ===")

for y, data in year_series.items():
    print(f"\nYear: {y}")
    for vn in var_names:
        s = data[vn]

        #raw variance (using all data points)
        raw_var = s.var()

        #subsampled (daily) variance
        daily_var = s.resample("1D").mean().var()

        print(f"  {vn.capitalize()}: raw variance = {raw_var:.3f}, daily variance = {daily_var:.3f}")

