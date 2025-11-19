#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:36:32 2025

@author: auroraczajkowski
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

# load files
FILES = [
    r"/Users/auroraczajkowski/Desktop/SIOC 221A/HW 5 data/OS_T8S110W_DM134A-20150425_D_WIND_10min.nc",
    r"/Users/auroraczajkowski/Desktop/SIOC 221A/HW 5 data/OS_T8S110W_DM183A-20160321_D_WIND_10min.nc",
    r"/Users/auroraczajkowski/Desktop/SIOC 221A/HW 5 data/OS_T8S110W_DM231A-20170606_D_WIND_10min.nc",
]

FIG_TITLE = "T8S110W Wind (10-min)"
PANEL_TITLES = {
    "WSPD": "Wind Speed (WSPD, m/s)",
    "UWND": "Zonal Wind (UWND, m/s)",
    "VWND": "Meridional Wind (VWND, m/s)",
}
YLABELS = {  
    "WSPD": "WSPD (m/s)",
    "UWND": "UWND (m/s)",
    "VWND": "VWND (m/s)",
}
SHOW_LEGEND = True
LEGEND_TITLE = "File"
LINEWIDTH = 0.9
FIGSIZE = (11, 7)
GRID_ALPHA = 0.3


# QUESTION 1: MAKE A PRELIMINARY ASSESSMENT OF THE DATA
# Gap filling toggle
FILL_GAPS = True  # set to False to skip interpolation but we want it :)

if not FILES:
    raise SystemExit("Added local .nc file paths to the FILES list.")

def load_one_to_df(path):
    """Open one NetCDF and return a tidy DataFrame at the single HEIGHT."""
    ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True)
    hval = float(ds["HEIGHT"].values[0])  # single level (e.g., 4 m)
    wspd = ds["WSPD"].sel(HEIGHT=hval).to_series()
    uwnd = ds["UWND"].sel(HEIGHT=hval).to_series()
    vwnd = ds["VWND"].sel(HEIGHT=hval).to_series()
    df = pd.DataFrame({"WSPD": wspd, "UWND": uwnd, "VWND": vwnd})
    df.index.name = "TIME"
    df = df.sort_index()
    df["__source__"] = os.path.basename(path)
    return df

def median_cadence(df):
    diffs = df.index.to_series().diff().dropna()
    return diffs.median() if len(diffs) else pd.NaT

# Load
dfs = []
for p in FILES:
    df = load_one_to_df(p)
    dfs.append(df)
    c = median_cadence(df)
    mins = float(c / pd.Timedelta(minutes=1)) if pd.notna(c) else np.nan
    print(f"Loaded {os.path.basename(p)} | {df.index.min()} → {df.index.max()} | N={len(df)} | dt≈{mins:.1f} min")

# sort by start time
dfs = sorted(dfs, key=lambda d: d.index.min())

# Inter-file gaps
print("\nInter-file gaps:")
for i in range(len(dfs) - 1):
    end_i = dfs[i].index.max()
    start_j = dfs[i + 1].index.min()
    print(f"  {dfs[i]['__source__'].iloc[0]} → {dfs[i+1]['__source__'].iloc[0]} : {start_j - end_i}")

# Fill intra-file gaps to a regular grid
if FILL_GAPS:
    print("\nInterpolating intra-file gaps (time-linear)…")
    filled_dfs = []
    for df in dfs:
        c = median_cadence(df)
        if pd.isna(c):
            print(f"  {df['__source__'].iloc[0]}: skip (no cadence)")
            filled_dfs.append(df)
            continue
        reg_time = pd.date_range(df.index.min(), df.index.max(), freq=c)
        reg = df.reindex(reg_time)
        n_before = reg[["WSPD", "UWND", "VWND"]].isna().sum().sum()
        for v in ["WSPD", "UWND", "VWND"]:
            reg[v] = reg[v].interpolate(method="time", limit_direction="both")
        n_after = reg[["WSPD", "UWND", "VWND"]].isna().sum().sum()
        reg["__source__"] = df["__source__"].iloc[0]
        filled_dfs.append(reg)
        print(f"  {df['__source__'].iloc[0]}: {len(reg)} stamps | NaNs {n_before} → {n_after}")
    dfs = filled_dfs

# Plot raw time series 
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

# choose cute colors for WSPD, UWND, VWND
colors = {
    "WSPD": "tab:blue",   # blue
    "UWND": "#FF69B4",    # pink
    "VWND": "purple"      # purple
}

titles = {
    "WSPD": "Wind Speed (WSPD, m/s)",
    "UWND": "Zonal Wind (UWND, m/s)",
    "VWND": "Meridional Wind (VWND, m/s)"
}

for ax, var in zip(axes, ["WSPD", "UWND", "VWND"]):
    for df in dfs:
        ax.plot(df.index, df[var], lw=0.9, color=colors[var],
                label=df["__source__"].iloc[0])
    ax.set_ylabel(var)
    ax.set_title(titles[var], fontsize=12)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time")
axes[0].legend(loc="upper right", fontsize=8, ncols=2)
plt.tight_layout()
plt.show()

# QUESTION 2: SEGMENT THE DATA
# segment the data into 60 day segments with 50% overlap
SEG_DAYS = 60
OVERLAP_DAYS = 30
SEG = pd.Timedelta(days=SEG_DAYS)
STEP = pd.Timedelta(days=OVERLAP_DAYS)

print("60-day segments with 50% overlap")

for df in dfs:
    src = df["__source__"].iloc[0]
    cadence = median_cadence(df)
    dt_days = cadence / pd.Timedelta(days=1)
    npts_per_seg = int(round(SEG_DAYS / dt_days))

    # starting times every 30 days until 60-day chunk would exceed record
    t0 = df.index.min()
    t_end = df.index.max()
    seg_starts = []
    while t0 + SEG <= t_end:
        seg_starts.append(t0)
        t0 = t0 + STEP

    n_segments = len(seg_starts)
    print(f"\nFile: {src}")
    print(f"  cadence: {cadence}  ({dt_days*24*60:.1f} min)")
    print(f"  segment length: {SEG_DAYS} days")
    print(f"  overlap: {OVERLAP_DAYS} days")
    print(f"  → segments: {n_segments}")
    print(f"  → points per segment: {npts_per_seg}")

    #building actual segment example 
    segments = []
    for start in seg_starts:
        stop = start + SEG
        seg = df.loc[start:stop]
        segments.append(seg)

    # Frequency resolution and Nyquist (in cycles/day)
    T_days = SEG_DAYS
    Δt_days = dt_days
    f_res = 1.0 / T_days
    f_nyquist = 0.5 / Δt_days
    print(f"  frequency resolution: {f_res:.5f} cycles/day")
    print(f"  Nyquist frequency:    {f_nyquist:.2f} cycles/day")

# QUESTION 3: COMPUTE AND PLOT SPECTRA USING THREE DIFFERENT APPROACHES, AND FOR 2015 DATA ONLY
# configurations for plots 
TARGET_YEAR   = 2015
SEG_DAYS      = 60
STEP_DAYS     = 30   
    #50% overlap
PLOT_MAX_CPD  = 10   
    #x-axis limit (cycles/day)
FIG_TITLE     = "T8S110W 2015 — Wind Speed Spectra"
XLABEL        = "Frequency (cycles/day)"
YLABEL        = "Variance density ((m/s)^2 per cpd)"

PANEL_STYLES = {
    "raw":          {"color": "tab:blue",  "title": "Raw"},
    "detrend":      {"color": "#FF69B4",   "title": "Detrended"},
    "detrend_hann": {"color": "purple",    "title": "Detrended + Hanning"},
}
LINEWIDTH = 1.3
GRID_ALPHA = 0.4

def _median_cadence(df):
    diffs = df.index.to_series().diff().dropna()
    return diffs.median() if len(diffs) else pd.NaT

def _linear_detrend(x):
    """Remove best-fit line from x using index-based linear trend."""
    n = len(x)
    t = np.arange(n, dtype=float)
    m = np.isfinite(x)
    if m.sum() < 2:
        return x.copy()
    p = np.polyfit(t[m], x[m], 1)
    return x - (p[0]*t + p[1])

def _periodogram_one_chunk(x, dt_days, mode="raw"):
    """
    Periodogram for one chunk using rFFT; returns (freq_cpd, Pxx).
    mode in {"raw","detrend","detrend_hann"}.
    Normalization yields units ~ (m/s)^2 per cycles/day.
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    # detrend
    if mode in ("detrend", "detrend_hann"):
        x = _linear_detrend(x)

    # window
    if mode == "detrend_hann":
        w = np.hanning(n)
    else:
        w = np.ones(n)

    U = (w**2).mean()
        #window power normalization
    xw = x * w

    Fs_cpd = 1.0 / dt_days          
        #samples per day
    X = np.fft.rfft(xw, n=n)
    f = np.fft.rfftfreq(n, d=dt_days)  
        #cycles/day

    P = (np.abs(X)**2) / (Fs_cpd * n * U)  
        #one-sided power spectral density
    if n > 2:
        P[1:-1] *= 2.0                       
            #double interior bins

    return f, P

def _build_segments(df, seg_days=60, step_days=30, col="WSPD"):
    """Return list of full 60-day segments (no NaNs), dt_days, and N per segment."""
    dt = _median_cadence(df)
    if pd.isna(dt):
        return [], np.nan, 0
    dt_days = dt / pd.Timedelta(days=1)

    seg = pd.Timedelta(days=seg_days)
    step = pd.Timedelta(days=step_days)

    starts = []
    t0 = df.index.min()
    tmax = df.index.max()
    while t0 + seg <= tmax:
        starts.append(t0)
        t0 = t0 + step

    # expected points for inclusive [start, start+seg] slice
    expected = int(round(seg_days / dt_days)) + 1

    chunks = []
    for s in starts:
        e = s + seg
        v = df.loc[s:e, col].to_numpy()
        if v.size >= expected:
            v = v[:expected]
            if np.isfinite(v).all():
                chunks.append(v)

    if not chunks:
        return [], dt_days, 0

    N = min(len(c) for c in chunks)  # enforce equal length
    chunks = [c[:N] for c in chunks]
    return chunks, dt_days, N

# pick the 2015 file from dfs (by date coverage)
df2015 = None
for df in dfs:
    if df.index.min().year <= TARGET_YEAR <= df.index.max().year:
        df2015 = df
        break
if df2015 is None:
    raise SystemExit("Could not find a 2015 record in loaded files.")

# build segments from WSPD
segments, dt_days, N = _build_segments(df2015, SEG_DAYS, STEP_DAYS, col="WSPD")
if len(segments) == 0:
    raise SystemExit("No valid 60-day segments found for 2015 (check gaps or cadence).")

print(f"\nQ3: 2015 spectra using {len(segments)} segments | N per segment = {N} | dt ≈ {dt_days*24*60:.1f} minutes")
print(f"    Frequency resolution ≈ {1/SEG_DAYS:.5f} cpd | Nyquist ≈ {0.5/dt_days:.2f} cpd")

# compute average spectra for the three methods
modes = ["raw", "detrend", "detrend_hann"]
avg_psd = {}
freqs_ref = None

for mode in modes:
    Ps = []
    for seg in segments:
        f, P = _periodogram_one_chunk(seg, dt_days, mode=mode)
        if freqs_ref is None:
            freqs_ref = f
        else:
            m = min(len(freqs_ref), len(f))
            f = f[:m]; P = P[:m]
        Ps.append(P)
    Ps = np.vstack(Ps)
    avg_psd[mode] = Ps.mean(axis=0)

# plot three panel figure so we can read it nicely 
fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)

for ax, mode in zip(axes, ["raw", "detrend", "detrend_hann"]):
    style = PANEL_STYLES[mode]
    ax.semilogy(freqs_ref, avg_psd[mode], color=style["color"], lw=LINEWIDTH)
    ax.set_xlim(0, 10^3)
    ax.set_ylabel(YLABEL)
    ax.set_title(style["title"], fontsize=12)
    ax.grid(True, which="both", alpha=GRID_ALPHA)

axes[-1].set_xlabel(XLABEL)
plt.suptitle(FIG_TITLE, fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# QUESTION 4: ADD UNCERTAINTY ESTIMATES TO THE 2015 WIND SPECTRA
# being brave and trying scipy stats for chi squared. i think it should spit an error if it doesnt work :/
try:
    from scipy.stats import chi2
except Exception as e:
    raise SystemExit(
        "This block needs SciPy for chi-square confidence limits.\n"
        "Install with: pip install scipy\n"
        f"Original import error: {e}"
    )

# configuration time! 
alpha = 0.05  # 95% CI
panel_styles = {
    "raw":          {"color": "tab:blue",  "title": "Raw"},
    "detrend":      {"color": "#FF69B4",   "title": "Detrended"},
    "detrend_hann": {"color": "purple",    "title": "Detrended + Hanning"},
}
LINEWIDTH = 1.3
GRID_ALPHA = 0.4

# sensible defaults 
PLOT_MAX_CPD = globals().get("PLOT_MAX_CPD", 10)
XLABEL       = globals().get("XLABEL", "Frequency (cycles/day)")
YLABEL       = globals().get("YLABEL", "Variance density ((m/s)^2 per cpd)")
FIG_TITLE    = globals().get("FIG_TITLE", "T8S110W 2015 — Wind Speed Spectra")

# Degrees of freedom from number of segments averaged
M = len(segments)           
    # number of 60-day chunks used in the average
DOF = 2 * M                
    # periodogram average: ~chi^2 with 2M DOF

# Multiplicative confidence factors for PSD
chi2_lo = chi2.ppf(1 - alpha/2, DOF)
chi2_hi = chi2.ppf(alpha/2, DOF)
factor_lo = DOF / chi2_lo   
    # lower multiplier
factor_hi = DOF / chi2_hi   
    # upper multiplier

print("\nQ4: Uncertainty for 2015 spectra")
print(f"  Segments averaged: {M}")
print(f"  DOF: {DOF}")
print(f"  95% CI factors: ×[{factor_lo:.2f}, {factor_hi:.2f}]")

# Build CI envelopes for each spectrum and plot in three panels
fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)

for ax, mode in zip(axes, ["raw", "detrend", "detrend_hann"]):
    P = avg_psd[mode]
    lo = P * factor_lo
    hi = P * factor_hi
    style = panel_styles[mode]

    ax.fill_between(freqs_ref, lo, hi, color=style["color"], alpha=0.18, linewidth=0)
    ax.semilogy(freqs_ref, P, color=style["color"], lw=LINEWIDTH)

    ax.set_xlim(0, PLOT_MAX_CPD)
    ax.set_ylabel(YLABEL)
    ax.set_title(style["title"], fontsize=12)
    ax.grid(True, which="both", alpha=GRID_ALPHA)

axes[-1].set_xlabel(XLABEL)
plt.suptitle(FIG_TITLE + " — 95% Confidence Bands", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# QUESTION 5: COMPUTE THE WIND SPEED SPECTRUM USING DATA FROM ALL THREE DATA FILES
# define some defaults 
PLOT_MAX_CPD = globals().get("PLOT_MAX_CPD", 10)
XLABEL       = globals().get("XLABEL", "Frequency (cycles/day)")
YLABEL       = globals().get("YLABEL", "Variance density ((m/s)^2 per cpd)")
FIG_TITLE    = globals().get("FIG_TITLE", "T8S110W — Wind Speed Spectra")
LINEWIDTH    = 1.5
GRID_ALPHA   = 0.4
alpha        = 0.05  # 95% CI

# Colors
COLOR_2015   = "purple"
COLOR_MULTI  = "black"

# Build ALL 60-day 50% overlap segments from all files (WSPD)
all_segments = []
dt_days_list = []

for df in dfs:
    segs, dt_d, N = _build_segments(df, seg_days=60, step_days=30, col="WSPD")
    if segs:
        all_segments.extend(segs)
        dt_days_list.append(float(dt_d))

if len(all_segments) == 0:
    raise SystemExit("Q5: No valid segments found across files (check gaps/cadence).")

# Check cadence consistency
dt_days_arr = np.array(dt_days_list)
if np.nanmax(np.abs(dt_days_arr - np.nanmedian(dt_days_arr))) > 1e-6:
    raise SystemExit("Q5: Mixed cadences detected across files. "
                     "For simplicity this block expects the same sampling for all years.")

dt_days_all = float(np.nanmedian(dt_days_arr))

# Compute periodograms for each segment (detrended + Hanning) and average
Ps_multi = []
freqs_multi = None
for seg in all_segments:
    f, P = _periodogram_one_chunk(seg, dt_days_all, mode="detrend_hann")
    if freqs_multi is None:
        freqs_multi = f
    else:
        m = min(len(freqs_multi), len(f))
        f = f[:m]; P = P[:m]
    Ps_multi.append(P)

Ps_multi = np.vstack(Ps_multi)
P_multi_mean = Ps_multi.mean(axis=0)
freqs_multi = freqs_multi[:P_multi_mean.size]

# Build 95% CI for multi-year and rebuild for 2015
M_multi = Ps_multi.shape[0]         
     # number of segments averaged (all years)
DOF_multi = 2 * M_multi
lo_multi = DOF_multi / chi2.ppf(1 - alpha/2, DOF_multi)
hi_multi = DOF_multi / chi2.ppf(alpha/2, DOF_multi)

# From Q3 and Q4 I already have: segments (2015 only), avg_psd["detrend_hann"], freqs_ref
M_2015 = len(segments)
DOF_2015 = 2 * M_2015
lo_2015 = DOF_2015 / chi2.ppf(1 - alpha/2, DOF_2015)
hi_2015 = DOF_2015 / chi2.ppf(alpha/2, DOF_2015)

# Align frequency grids if needed (they should match but let's guard anyway)
m = min(len(freqs_ref), len(freqs_multi))
freqs_plot = freqs_ref[:m]
P_2015 = avg_psd["detrend_hann"][:m]
P_multi_mean = P_multi_mean[:m]

#Plot overlay with 95% CI bands
plt.figure(figsize=(10.5, 6.5))

# 2015 band + line
plt.fill_between(freqs_plot, P_2015 * lo_2015, P_2015 * hi_2015,
                 color=COLOR_2015, alpha=0.18, linewidth=0, label="2015 (95% CI)")
plt.semilogy(freqs_plot, P_2015, color=COLOR_2015, lw=LINEWIDTH, label="2015 (Detrended + Hanning)")

# Multi-year band + line
plt.fill_between(freqs_plot, P_multi_mean * lo_multi, P_multi_mean * hi_multi,
                 color=COLOR_MULTI, alpha=0.15, linewidth=0, label="All years (95% CI)")
plt.semilogy(freqs_plot, P_multi_mean, color=COLOR_MULTI, lw=LINEWIDTH, label="All years (Detrended + Hanning)")

plt.xlim(0, PLOT_MAX_CPD)
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)
plt.title(FIG_TITLE + " — Detrended + Hanning: 2015 vs All Years")
plt.grid(True, which="both", alpha=GRID_ALPHA)
plt.legend()
plt.tight_layout()
plt.show()

# Print quick summary
print("\nQ5 summary:")
print(f"  2015:   segments averaged = {M_2015}, DOF = {DOF_2015}, 95% factors ×[{lo_2015:.2f}, {hi_2015:.2f}]")
print(f"  All yrs: segments averaged = {M_multi}, DOF = {DOF_multi}, 95% factors ×[{lo_multi:.2f}, {hi_multi:.2f}]")

# QUESTION 6: COMPARE THE SPECTRA FOR WIND SPEED, ZONAL WIND, AND MERIDIONAL WIND 
# configurations! 
SEG_DAYS     = 60
STEP_DAYS    = 30
PLOT_MAX_CPD = 10
XLABEL       = "Frequency (cycles/day)"
YLABEL       = "Variance density ((m/s)^2 per cpd)"
FIG_TITLE    = "T8S110W — Spectra for WSPD, UWND, VWND (detrended + Hanning)"
LINEWIDTH    = 1.4
GRID_ALPHA   = 0.4
SHOW_CI      = True     
    # show 95 % CI
ALPHA_CI     = 0.18     
    # shading transparency
ALPHA        = 0.05     
    # for 95 % CI level

COLORS = {"WSPD": "tab:blue", "UWND": "#FF69B4", "VWND": "purple"}

def _gather_spectrum_all_files(dfs, varname):
    """Compute detrend+Hanning periodograms for all 60-day segments."""
    all_segments, dt_days_list = [], []
    for df in dfs:
        segs, dt_days, _ = _build_segments(df, seg_days=SEG_DAYS,
                                           step_days=STEP_DAYS, col=varname)
        if segs:
            all_segments.extend(segs)
            dt_days_list.append(float(dt_days))
    if not all_segments:
        raise SystemExit(f"Q6: No valid segments for {varname}")

    dt_days_arr = np.array(dt_days_list)
    dt_med = float(np.nanmedian(dt_days_arr))
    if np.nanmax(np.abs(dt_days_arr - dt_med)) > 1e-6:
        raise SystemExit(f"Mixed cadences detected for {varname}")

    Ps, f_ref = [], None
    for seg in all_segments:
        f, P = _periodogram_one_chunk(seg, dt_med, mode="detrend_hann")
        if f_ref is None:
            f_ref = f
        else:
            m = min(len(f_ref), len(f))
            f, P = f[:m], P[:m]
        Ps.append(P)

    Ps = np.vstack(Ps)
    P_mean = Ps.mean(axis=0)
    M = Ps.shape[0]
    return f_ref[:P_mean.size], P_mean, M, dt_med


# Compute spectra for each variable
freqs_W, P_W, M_W, dtW = _gather_spectrum_all_files(dfs, "WSPD")
freqs_U, P_U, M_U, dtU = _gather_spectrum_all_files(dfs, "UWND")
freqs_V, P_V, M_V, dtV = _gather_spectrum_all_files(dfs, "VWND")

# Align frequency grids (guard against length differences)
m = min(len(freqs_W), len(freqs_U), len(freqs_V))
freqs = freqs_W[:m]
P_W, P_U, P_V = P_W[:m], P_U[:m], P_V[:m]

# Confidence intervals (95 %)
if SHOW_CI:
    def ci_factors(M):
        DOF = 2 * M
        lo = DOF / chi2.ppf(1 - ALPHA/2, DOF)
        hi = DOF / chi2.ppf(ALPHA/2, DOF)
        return lo, hi, DOF
    loW, hiW, DOF_W = ci_factors(M_W)
    loU, hiU, DOF_U = ci_factors(M_U)
    loV, hiV, DOF_V = ci_factors(M_V)
    print(f"Q6 DOF  WSPD={DOF_W}  UWND={DOF_U}  VWND={DOF_V}")

# Plot three-panel figure
fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
VAR_ORDER = [("WSPD", P_W, loW, hiW, M_W),
             ("UWND", P_U, loU, hiU, M_U),
             ("VWND", P_V, loV, hiV, M_V)]

for ax, (var, P, lo, hi, M) in zip(axes, VAR_ORDER):
    color = COLORS[var]
    if SHOW_CI:
        ax.fill_between(freqs, P*lo, P*hi, color=color, alpha=ALPHA_CI,
                        linewidth=0, label=f"{var} (95% CI)")
    ax.semilogy(freqs, P, color=color, lw=LINEWIDTH, label=f"{var} (m/s)")
    ax.set_xlim(0, PLOT_MAX_CPD)
    ax.set_ylabel(YLABEL)
    ax.set_title(var, fontsize=12)
    ax.grid(True, which="both", alpha=GRID_ALPHA)
    ax.legend()

axes[-1].set_xlabel(XLABEL)
plt.suptitle(FIG_TITLE, fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nQ6 summary:")
print(f"Segments averaged  |  WSPD: {M_W:3d}   UWND: {M_U:3d}   VWND: {M_V:3d}")
print("All spectra computed using detrended + Hanning, 60-day windows, 50 % overlap.")
