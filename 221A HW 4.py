#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 09:27:31 2025

@author: auroraczajkowski
"""

# Import packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

# Load file paths
NC_PATH = "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2021.nc"

def load_pier_local(nc_path, pressure_var_candidates=("pressure","sea_water_pressure","PRESSURE","pres")):
    """
    Load a single local NetCDF, return (time_datetime64, pressure_array).
    """
    ds = xr.open_dataset(nc_path, decode_times=True, mask_and_scale=True)

    # Normalize time coordinates 
    time_name = "time" if "time" in ds.coords else next(
        (t for t in ["TIME", "Time", "t"] if t in ds.coords), None
    )
    if time_name is None:
        raise ValueError("No time coordinate found in file.")
    if time_name != "time":
        ds = ds.rename({time_name: "time"})

    # Find pressure variable
    pvar = next((v for v in pressure_var_candidates if v in ds.data_vars), None)
    if pvar is None:
        raise ValueError(f"No pressure variable found. Available: {list(ds.data_vars)}")

    time = ds["time"].values          # datetime64[ns]
    pressure = ds[pvar].values.astype(float)

    # Basic sanity
    if time.size != pressure.size:
        raise ValueError("Time and pressure lengths differ.")

    return time, pressure, pvar

time_dt, pressure, pvar_name = load_pier_local(NC_PATH)

# Select window we want to look at 
start_1b, end_1b = 70521, 88190
start_0b, end_excl = start_1b - 1, end_1b

idx = np.arange(start_0b, end_excl, dtype=int)
orig_times_dt = time_dt[idx]      # datetime64
orig_pres      = pressure[idx]    # float

# For uniform time math, convert to epoch seconds (int64)
orig_times_s = orig_times_dt.astype("datetime64[s]").astype("int64")

# Build time vector (240 seconds)
def make_uniform_time(t_seconds, dt_sec=240):
    total_seconds = int(t_seconds[-1] - t_seconds[0])
    N = total_seconds // dt_sec + 1
    t_uniform = t_seconds[0] + np.arange(N, dtype=int) * dt_sec
    return t_uniform

sub_time_uniform = make_uniform_time(orig_times_s, dt_sec=240)

# Loop-based gap fill by inserting neighbor average 
# Gaps are 1-based "positions before the gap" within the subset
# I based this code off of the MATLAB filling strategy and tried my best to convert to Python
gaps_1based = [2420, 4952, 7496, 15042]

def fill_gaps_with_averages(series, gaps_1b):
    """
    MATLAB-equivalent strategy:
      - Copy data up to position g (1-based count of elements),
      - Insert one sample = 0.5*(series[g-1] + series[g]),
      - Continue. Output length = len(series) + len(gaps).
    """
    gaps = list(gaps_1b)
    L = len(series)
    out = np.empty(L + len(gaps), dtype=float)

    src_start = 0   # 0-based in original
    dst = 0        # 0-based in output

    for g in gaps:
        # copy chunk up to (not including) python index g
        out[dst:dst + (g - src_start)] = series[src_start:g]
        dst += (g - src_start)
        # insert average between elements g-1 and g (1-based -> 0-based)
        out[dst] = 0.5 * (series[g - 1] + series[g])
        dst += 1
        src_start = g

    # tail
    out[dst:] = series[src_start:]
    return out

sub_pres_filled = fill_gaps_with_averages(orig_pres, gaps_1based)

# Make a matching uniform time for the filled series
sub_time_filled_s = orig_times_s[0] + np.arange(len(sub_pres_filled), dtype=int) * 240


# Visual Evaluation (Question 1): Are data uniformly spaced? What's the increment? How long is the record?
dt = np.diff(orig_times_s)  # seconds between adjacent original samples (subset window)

mean_dt = float(np.mean(dt))
median_dt = float(np.median(dt))
unique_dt = np.unique(dt)

total_days = (orig_times_s[-1] - orig_times_s[0]) / (24 * 3600)

print("Visual Evaluation (Pressure)")
print(f"Variable used: {pvar_name}")
print(f"Samples in subset (original): {orig_pres.size}")
print(f"Record length (subset): {total_days:.2f} days")
print(f"Mean Δt:   {mean_dt:.2f} s")
print(f"Median Δt: {median_dt:.2f} s")
print(f"Unique Δt values (s) [first 10]: {unique_dt[:10]}{' ...' if unique_dt.size>10 else ''}")
print(f"Uniform spacing across whole subset? {'Yes' if np.all(dt == dt[0]) else 'No'}")

# Identify ~uniform regions (+/- 5% of median)
tol = 0.05 * median_dt
uniform_mask = np.abs(dt - median_dt) < tol
uniform_fraction = 100.0 * np.mean(uniform_mask)
print(f"Fraction with ~uniform spacing (+/- 5%): {uniform_fraction:.1f}%")

# Find longest contiguous run near the median Δt
runs = []
start = None
for i, ok in enumerate(uniform_mask):
    if ok and start is None:
        start = i
    if (not ok or i == len(uniform_mask) - 1) and start is not None:
        end = i if not ok else i
        runs.append((start, end))
        start = None

if runs:
    longest = max(runs, key=lambda ab: ab[1] - ab[0] + 1)
    i0, i1 = longest
    days_run = (orig_times_s[i1 + 1] - orig_times_s[i0]) / (24 * 3600)
    print(f"Longest ~uniform segment: indices [{i0}..{i1+1}] "
          f"({i1 - i0 + 1} intervals, ~{days_run:.2f} days)")
else:
    print("No clearly uniform segment found by +/- 5% criterion.")

# Plots 
# Helper x-axis in days since start for readability
days_from_start = (orig_times_s - orig_times_s[0]) / (24 * 3600)
days_from_start_filled = (sub_time_filled_s - sub_time_filled_s[0]) / (24 * 3600)

# Pressure time series (original subset)
plt.figure(figsize=(11, 4))
plt.plot(days_from_start, orig_pres, lw=0.8)
plt.title("Scripps Pier Pressure — 2021 subset (original)")
plt.xlabel("Days since start of subset")
plt.ylabel("Pressure (dbar)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# delta T between adjacent measurements (original subset)
plt.figure(figsize=(11, 3))
plt.plot(days_from_start[1:], dt, lw=0.7)
plt.title("Increment Between Adjacent Measurements (delta t)")
plt.xlabel("Days since start of subset")
plt.ylabel("delta t (seconds)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

#e Filled series for reference (after gap insertion)
plt.figure(figsize=(11, 4))
plt.plot(days_from_start_filled, sub_pres_filled, lw=0.8)
plt.title("Pressure — gap-filled (neighbor average inserted at four gaps)")
plt.xlabel("Days since start of subset")
plt.ylabel("Pressure (dbar)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


# Least-squares fit (Question 2): mean + O1, K1, M2 
import numpy as np
import matplotlib.pyplot as plt

# Inputs
t_sec = sub_time_filled_s.astype(np.int64)
y = np.asarray(sub_pres_filled, dtype=float)

# Clean NaNs
mask = np.isfinite(t_sec) & np.isfinite(y)
t_sec = t_sec[mask]
y = y[mask]

# Time origin for numerical stability
t0 = t_sec[0]
t = t_sec - t0
t_days = t / 86400.0

# Tidal constituents (hours)
period_h = {
    "O1": 25.82,   # Principal lunar diurnal
    "K1": 23.93,   # Luni-solar diurnal
    "M2": 12.42,   # Principal lunar semidiurnal
}
# Frequencies and angular frequencies
freq_cpd = {name: 24.0 / P for name, P in period_h.items()}          # cycles/day (for reporting)
omega = {name: 2.0 * np.pi / (P * 3600.0) for name, P in period_h.items()}  # rad/sec

# Design matrix: [1, sin(ω_O1 t), cos(ω_O1 t), sin(ω_K1 t), cos(ω_K1 t), sin(ω_M2 t), cos(ω_M2 t)]
cols = [np.ones_like(t, dtype=float)]
names = ["mean"]
for name in ["O1", "K1", "M2"]:
    w = omega[name]
    cols += [np.sin(w * t), np.cos(w * t)]
    names += [f"{name}_sin", f"{name}_cos"]
X = np.column_stack(cols)

# Solve least squares
beta, *_ = np.linalg.lstsq(X, y, rcond=None)

# Extract results
mean_val = beta[0]
results = []
i = 1
for name in ["O1", "K1", "M2"]:
    a_sin, b_cos = float(beta[i]), float(beta[i+1])
    amp = np.hypot(a_sin, b_cos)
        # total amplitude
    phase = np.arctan2(a_sin, b_cos)
        # radians
    results.append((name, amp, phase, a_sin, b_cos))
    i += 2

# Model reconstruction
y_model = X @ beta

# Print summary
print("\n Least-Squares Fit: mean + O1, K1, M2")
print(f"Mean pressure: {mean_val:.4f} (dbar)")
for (name, amp, phase, a_sin, b_cos) in results:
    print(f"{name}:  f={freq_cpd[name]:.4f} cpd,  amplitude={amp:.4f},  "
          f"phase={phase:.4f} rad ({np.degrees(phase):.1f}°),  "
          f"a_sin={a_sin:.4f}, b_cos={b_cos:.4f}")

# Fit skill
ss_res = np.sum((y - y_model)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
print(f"R^2 (model vs data): {r2:.4f}")

# Plot raw vs model
plt.figure(figsize=(12, 4))
plt.plot(t_days, y, lw=0.8, label="raw")
plt.plot(t_days, y_model, lw=1.1, label="model (mean + O1,K1,M2)")
plt.xlabel("Days since start of subset")
plt.ylabel("Pressure (dbar)")
plt.title("Least-squares tidal fit (mean + O1, K1, M2)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# plot individual constituent contributions
#plt.figure(figsize=(12, 4))
#for (name, _, _, a_sin, b_cos) in results:
    #w = omega[name]
   # comp = a_sin*np.sin(w*t) + b_cos*np.cos(w*t)
   # plt.plot(t_days, comp, lw=0.9, label=name)
#plt.xlabel("Days since start of subset")
#plt.ylabel("Pressure contribution (dbar)")
#plt.title("Individual constituent contributions (O1, K1, M2)")
#plt.grid(True, alpha=0.4)
#plt.legend()
#plt.tight_layout()
#plt.show()


# Fourier Transform Analysis (Question 3): O1, K1, M2 
import numpy as np
import matplotlib.pyplot as plt

# Inputs from earlier
t_sec = sub_time_filled_s.astype(np.int64)
y = np.asarray(sub_pres_filled, dtype=float)

# Sampling parameters
dt = np.median(np.diff(t_sec))
    # seconds per sample (≈240 s)
fs = 1.0 / dt
    # Hz
N = y.size

# FFT and frequency axis (cycles per day)
Y = np.fft.fft(y)
freq_hz = np.fft.fftfreq(N, d=dt)
freq_cpd = freq_hz * 86400.0
    # convert to cycles/day
pos = freq_hz > 0
    # positive frequencies
f_pos = freq_cpd[pos]
Y_pos = Y[pos]
mag_pos = np.abs(Y_pos)

# Define tidal constituents and expected frequencies
tidal_period_h = {
    "O1": 25.82,   # Principal lunar diurnal
    "K1": 23.93,   # Luni-solar diurnal
    "M2": 12.42,   # Principal lunar semidiurnal
}
tidal_freq_cpd = {name: 24.0 / period for name, period in tidal_period_h.items()}  # cpd

# (a) Plot real and imaginary parts in the tidal band
plt.figure(figsize=(11, 4))
plt.plot(f_pos, Y_pos.real, lw=0.8, label="Real")
plt.plot(f_pos, Y_pos.imag, lw=0.8, label="Imag")

# Mark the tidal frequencies
for name, f_tide in tidal_freq_cpd.items():
    plt.axvline(f_tide, color="k", ls="--", lw=0.8)
    plt.text(f_tide + 0.02, np.max(Y_pos.real)*0.6, name,
             rotation=90, va='bottom', fontsize=9)

plt.xlim(0, 3)
plt.xticks(np.arange(0, 3.5, 0.5))
plt.xlabel("Frequency (cycles per day)")
plt.ylabel("FFT Coefficient")
plt.title("Fourier Transform — Real and Imaginary Parts (0–3 cpd)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# (b) Amplitudes from FFT
print("\n Major Tidal Constituents from FFT")
amp_fft = {}
phase_fft = {}
for name, f_target in tidal_freq_cpd.items():
    j = np.argmin(np.abs(f_pos - f_target))
    A = 2.0 * np.abs(Y_pos[j]) / N
        # two-sided normalization
    phi = np.angle(Y_pos[j])
    amp_fft[name] = A
    phase_fft[name] = phi
    print(f"{name}:  f≈{f_pos[j]:.4f} cpd  (target {f_target:.4f})  "
          f"amplitude={A:.4f}  phase(rad)={phi:.3f}")

# Mean (DC term)
mean_fft = Y[0].real / N
print(f"\nMean pressure from FFT (DC/N): {mean_fft:.4f}")

# (c) Compare to least-squares fit 
try:
    print("\n Comparison to Least-Squares Fit")
    print(f"Mean:  LS={mean_val:.4f},  FFT={mean_fft:.4f}")
    for (name, amp_ls, phase_ls, *_ ) in results:
        if name in amp_fft:
            print(f"{name}:  LS amplitude={amp_ls:.4f}  |  FFT amplitude={amp_fft[name]:.4f}")
except NameError:
    print("\n(Least-squares results not found in this session; skipping comparison.)")

# (d) Spectral Energy Plot
S_pos = np.abs(Y_pos)**2

plt.figure(figsize=(11, 4))
plt.plot(f_pos, S_pos, lw=0.8)
for name, f_tide in tidal_freq_cpd.items():
    plt.axvline(f_tide, color="k", ls="--", lw=0.8)
    plt.text(f_tide + 0.02, np.max(S_pos)*0.6, name,
             rotation=90, va='bottom', fontsize=9)

plt.xlim(0, 3)
plt.xticks(np.arange(0, 3.5, 0.5))
plt.xlabel("Frequency (cycles per day)")
plt.ylabel("|Y(f)|² (arbitrary units)")
plt.title("Spectral Energy — Tidal Frequency Band (0–3 cpd)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# (e) Spectrum color classification
valid = (f_pos > 0)
f_for_fit = f_pos[valid]
S_for_fit = S_pos[valid]
q = np.quantile(S_for_fit, 0.98)
mask_broad = S_for_fit < q
f_fit = f_for_fit[mask_broad]
S_fit = S_for_fit[mask_broad]

if f_fit.size > 10:
    x = np.log10(f_fit)
    ylog = np.log10(S_fit)
    slope, intercept = np.polyfit(x, ylog, 1)
    trend = "red (low-frequency dominant)" if slope < -0.2 else (
             "blue (high-frequency dominant)" if slope > 0.2 else "white-ish (flat)")
    print(f"\nBroadband spectral slope (log10 S vs log10 f): {slope:.2f} → spectrum appears {trend}.")
else:
    print("\nNot enough data to estimate broadband spectral color.")
