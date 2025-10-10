#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:39:24 2025

@author: auroraczajkowski
"""

#import the packages we need
import xarray as xr
import pandas as pd
    #pandas is not explicitly used but later i transform into pandas object for datetime
    #xarray uses pandas so i don't need to import but for ease of understanding
import matplotlib.pyplot as plt
import numpy as np

#define file path 
nc_path = "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2021.nc"

#open dataset and peek at values. specifically looking at time to see what format it is in
ds = xr.open_dataset(nc_path, decode_times=True)
print(ds)  
    #this tells us attributes etc of data set                         
print(ds["time"].values[:5])    
    #this prints the first few times, i didn't originally had this but the x labels were julian time and ugly so i wanted to fix    

#choose what variable we want to work with, we want to make a temperature timeseries so we are defining temperature
var_name = "temperature"           
da = ds[var_name]

#convert from julian calendar, super unneccesary but makes plot cleaner
da_greg = da.convert_calendar("proleptic_gregorian")

#converting into pandas because i prefer to plot with pandas, cleaner for labeling purposes  
df = da_greg.to_pandas()           

ax = df.plot(figsize=(10, 10), color='#f04ef2')
ax.set_title("Scripps Pier SST 2021")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (C)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#compute mean and stdev for year data 
mean_temp = df.mean()
std_temp = df.std()

print(f"Mean SST (2021): {mean_temp:.2f} °C")
print(f"Std Dev SST (2021): {std_temp:.2f} °C")

#figure for PDF 
fig, [ax1, ax2] = plt.subplots(1, 2, figsize = (16, 8))
hist = ax1.hist(df, bins=30, density=False)
ax1.set_ylabel('Counts', fontsize=14)
ax1.set_xlabel('Temperature (C)', fontsize=14)
ax1.set_title('Histogram', fontsize=18)
# Now, with density=True
pdf = ax2.hist(df, bins=30, density=True)
ax2.set_ylabel('Probability density', fontsize=14)
ax2.set_xlabel('Temperature (C)', fontsize=14)
ax2.set_title('PDF', fontsize=18)


dbin = 0.1
bin_min = 10
bin_max = 30

x = np.asarray(df).ravel().astype(float)
x = x[np.isfinite(x)]

bins = np.arange(bin_min, bin_max, dbin)  # these are bin centers
counts = []
half = dbin / 2

for c in bins:
    ind = (x > c - half) & (x <= c + half)
    counts.append(ind.sum())

counts = np.array(counts)

plt.figure(figsize=(8, 8))
plt.plot(bins, counts)
plt.xlabel('Temperature (C)', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.title('SST Count Distribution 2021')
plt.show()

edges = np.arange(bin_min, bin_max + dbin, dbin)
pdf, edges = np.histogram(x, bins=edges, density=True)
centers = (edges[:-1] + edges[1:]) / 2

plt.figure(figsize=(8, 6))
plt.plot(centers, pdf)
plt.xlabel('Temperature (°C)', fontsize=14)
plt.ylabel('PDF', fontsize=14)
plt.title('SST 2021 PDF')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("PDF integral ~", np.trapz(pdf, centers))