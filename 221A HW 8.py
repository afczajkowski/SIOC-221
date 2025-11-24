# QUESTION 1: REVISITING NORMALIZATION AND PARSEVAL
import numpy as np
import xarray as xr

# Load dataset
ds = xr.open_dataset("/Users/auroraczajkowski/Desktop/SIOC 221A/HW 1 data/scripps_pier-2021.nc", decode_times=False)
x_full = ds["temperature"].values.astype(float)

t = ds["time"].values
dt = float(t[1] - t[0])
fs = 1.0 / dt

print(f"Loaded data: {len(x_full)} samples, dt={dt}s")

# Parseval's theorm verification    
N2 = 4096
x2 = x_full[:N2]
x2 = x2 - np.mean(x2)

# FFT
X = np.fft.rfft(x2)
f = np.fft.rfftfreq(N2, dt)

# PSD normalization (rectangular)
P = (1/(fs*N2)) * np.abs(X)**2
if N2 % 2 == 0:
    P[1:-1] *= 2
else:
    P[1:] *= 2

df = f[1] - f[0]

var_time = np.var(x2)
var_freq = np.sum(P) * df

print("\n EXACT PARSEVAL (2 DOF)")
print("Time-domain variance:   ", var_time)
print("Freq-domain integral:   ", var_freq)
print("Relative error:         ", (var_freq - var_time)/var_time)

# Parseval's theorm verification using Welch's method

Nblock = 4096
xw = x_full[:Nblock]
xw = xw - np.mean(xw)

nperseg = 512
noverlap = nperseg // 2
step = nperseg - noverlap

w = np.hanning(nperseg)
U = np.sum(w**2) / nperseg   
    # window power normalization
scale = 1/(fs * U * nperseg)

segments = []
for start in range(0, Nblock - nperseg + 1, step):
    seg = xw[start:start+nperseg] * w
    Xs = np.fft.rfft(seg)
    Pseg = scale * np.abs(Xs)**2
    if nperseg % 2 == 0:
        Pseg[1:-1] *= 2
    else:
        Pseg[1:] *= 2
    segments.append(Pseg)

segments = np.array(segments)
Pavg = segments.mean(axis=0)
dfw = fs / nperseg

var_time_w = np.var(xw)
var_freq_w = np.sum(Pavg) * dfw

print("\n WELCH PARSEVAL (Hanning windows)")
print("Segments used:          ", len(segments))
print("Time-domain variance:   ", var_time_w)
print("Freq-domain integral:   ", var_freq_w)
print("Relative error:         ", (var_freq_w - var_time_w)/var_time_w)

#cQUESTION 2: ALIASING
# Tidal constituents (periods in hours)
tides = [
    ("S1",  "Solar diurnal",                    24.0000),
    ("2N2", "Second-order elliptical lunar",    12.9054),
    ("N2",  "Larger elliptical lunar",          12.6583),
    ("M2",  "Principal lunar",                  12.4206),
    ("S2",  "Principal solar semidiurnal",      12.0000),
    ("K2",  "Declinational solar",              11.9672),
]

# Frequency in cycles per day: f = 24 / period_hours
def tidal_frequency_cpd(period_hours: float) -> float:
    return 24.0 / period_hours

# SWOT orbit parameters
orbits = [
    ("Fast-sampling", 0.99349),   
        # repeat period in days
    ("Science",       20.86455),
]

# Aliasing function
def alias_frequency(f_true: float, fs: float) -> float:
    """
    Return aliased frequency (cycles/day) of a signal with true frequency f_true
    when sampled at frequency fs (cycles/day).

    Fold the frequency into the Nyquist band [0, fs/2] using:
        f_alias = |f_true - n * fs|
    with n = round(f_true / fs).
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    n = int(round(f_true / fs))
    f_alias = abs(f_true - n * fs)
    return f_alias

# Main calculations
print(" TIDAL ALIASING FOR SWOT")

# Precompute true frequencies
tidal_freqs = []
print("True tidal frequencies:")
for symbol, name, period_h in tides:
    f_true = tidal_frequency_cpd(period_h)  
        # cycles per day
    tidal_freqs.append(f_true)
    print(f"  {symbol:3s} ({name:35s}): "
          f"period = {period_h:8.4f} h,  f = {f_true:7.4f} c/day")
print()

# For each orbit: compute aliasing and required mission length
for orbit_name, T_repeat_days in orbits:
    fs = 1.0 / T_repeat_days  
        # cycles per day (one sample per repeat)
    f_nyquist = fs / 2.0

    print(f"Orbit: {orbit_name}")
    print(f"  Repeat period T_s = {T_repeat_days:.5f} days")
    print(f"  Sampling freq  f_s = {fs:.5f} cycles/day")
    print(f"  Nyquist freq   f_N = {f_nyquist:.5f} cycles/day\n")

    # Compute aliased frequencies for each tide
    f_alias_list = []
    print("  Aliased frequencies and periods:")
    print("  (Aliased periods are in days; 0 means exactly constant in time)")
    for (symbol, name, period_h), f_true in zip(tides, tidal_freqs):
        f_alias = alias_frequency(f_true, fs)
        f_alias_list.append(f_alias)

        if f_alias == 0:
            alias_period_str = "∞ (DC)"
        else:
            T_alias_days = 1.0 / f_alias
            alias_period_str = f"{T_alias_days:8.2f} d"

        print(f"    {symbol:3s}: f_true = {f_true:7.4f} c/day  "
              f"->  f_alias = {f_alias:8.4f} c/day,  T_alias = {alias_period_str}")
    print()

    # How long to operate to separate these frequencies?
    # Spectral resolution ~ 1/T_total; need T_total >= 1 / |f_i - f_j|
    f_alias_array = np.array(f_alias_list)
    n = len(f_alias_array)

    df_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            df = abs(f_alias_array[i] - f_alias_array[j])
            if df > 0:
                df_pairs.append((df, i, j))

    if not df_pairs:
        print("  All aliased frequencies collapsed; cannot distinguish them.\n")
        continue

    # Smallest non-zero separation in frequency
    df_min, i_min, j_min = min(df_pairs, key=lambda x: x[0])
    T_required_days = 1.0 / df_min
    T_required_months = T_required_days / 30.0

    tide_i = tides[i_min][0]
    tide_j = tides[j_min][0]

    print("  Frequency separation analysis:")
    print(f"    Smallest non-zero |deltf| between aliased tides:")
    print(f"      |f_alias({tide_i}) - f_alias({tide_j})| = {df_min:.6f} cycles/day")
    print(f"    To resolve these two in a spectrum, need record length:")
    print(f"      T_total >= 1 / |deltf| ≈ {T_required_days:8.1f} days "
          f"(≈ {T_required_months:5.2f} months)")
    print("    That mission duration is sufficient to separate all the listed")
    print("    tidal constituents in this orbit (it resolves the closest pair).\n")

# QUESTION 3: SPECTRA OF ALIASED SIGNALS
import xarray as xr
import matplotlib.pyplot as plt

# Define name of file 
FILENAME = "/Users/auroraczajkowski/Desktop/SIOC 221A/HW 5 data/OS_T8S110W_DM134A-20150425_D_WIND_10min.nc"

SEGMENT_DAYS = 60            
    # length of segment
SUBSAMPLE_FACTOR = 40        
    # every 40th point -> 6.667 hours for 10-min data

# Load datafile
ds = xr.open_dataset(FILENAME)

time = ds["TIME"].values                     
    # datetime
wspd = ds["WSPD"][:, 0].values.astype(float) 
    # 1D wind speed (m/s)

# Gap fill linearly in time using xarray
wspd_da = ds["WSPD"][:, 0]
wspd_filled = wspd_da.interpolate_na(dim="TIME", method="linear").values

time0 = time[0]
end_time = time0 + np.timedelta64(SEGMENT_DAYS, "D")

mask = (time >= time0) & (time < end_time)
t_seg = time[mask]
x_full = wspd_filled[mask]

# Remove mean
x_full = x_full - np.mean(x_full)

# Sampling interval (seconds)
dt_sec = float(np.median(np.diff(t_seg).astype("timedelta64[s]").astype(float)))
fs_Hz = 1.0 / dt_sec
fs_cpd = fs_Hz * 86400.0    
    # cycles per day

print("DATA INFO")
print(f"File: {FILENAME}")
print(f"Segment length: {SEGMENT_DAYS} days")
print(f"N_full = {len(x_full)} points")
print(f"dt = {dt_sec/60:.2f} minutes, fs = {fs_cpd:.3f} cycles/day\n")

# one sided peridogram with rectangular window 
def periodogram_rect(x, dt_sec):
    """
    One-sided PSD of real signal x with rectangular window.
    PSD units: (x^2) / Hz. Parseval: var(x) ~ sum(PSD)*df.
    """
    N = len(x)
    fs = 1.0 / dt_sec
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt_sec)  
        # Hz

    Pxx = (1.0 / (fs * N)) * np.abs(X)**2  
        # two-sided base

    # one-sided adjustment
    if N % 2 == 0:
        Pxx[1:-1] *= 2.0
    else:
        Pxx[1:] *= 2.0

    return freqs, Pxx

# a) full reolution spectrum 

f_full_Hz, P_full = periodogram_rect(x_full, dt_sec)
f_full_cpd = f_full_Hz * 86400.0
df_full_Hz = f_full_Hz[1] - f_full_Hz[0]
df_full_cpd = df_full_Hz * 86400.0

var_time_full = np.var(x_full)
var_freq_full = np.sum(P_full) * df_full_Hz

print("FULL-RESOLUTION SPECTRUM")
print(f"Frequency resolution (full): df = {df_full_cpd:.4f} cycles/day")
print(f"Nyquist (full): f_N = {f_full_cpd[-1]:.3f} cycles/day")
print(f"Variance (time): {var_time_full:.4f}, from PSD: {var_freq_full:.4f}\n")

# b) subsampled spectrum 

# Subsample
x_sub = x_full[::SUBSAMPLE_FACTOR]
t_sub = t_seg[::SUBSAMPLE_FACTOR]

N_sub = len(x_sub)
dt_sub_sec = dt_sec * SUBSAMPLE_FACTOR
fs_sub_Hz = 1.0 / dt_sub_sec
fs_sub_cpd = fs_sub_Hz * 86400.0

x_sub = x_sub - np.mean(x_sub)

f_sub_Hz, P_sub = periodogram_rect(x_sub, dt_sub_sec)
f_sub_cpd = f_sub_Hz * 86400.0
df_sub_Hz = f_sub_Hz[1] - f_sub_Hz[0]
df_sub_cpd = df_sub_Hz * 86400.0

var_time_sub = np.var(x_sub)
var_freq_sub = np.sum(P_sub) * df_sub_Hz

print("SUBSAMPLED SPECTRUM")
print(f"N_sub = {N_sub}")
print(f"dt_sub = {dt_sub_sec/3600:.3f} hours")
print(f"fs_sub = {fs_sub_cpd:.3f} cycles/day")

# c) Resolution & Nyquist for subsampled data
nyq_sub_cpd = f_sub_cpd[-1]
print(f"Resolution (df_sub) = {df_sub_cpd:.4f} cycles/day")
print(f"Nyquist (subsampled) = {nyq_sub_cpd:.3f} cycles/day\n")

print(f"Variance_sub (time): {var_time_sub:.4f}, from PSD: {var_freq_sub:.4f}")
print(f"Ratio var_sub / var_full = {var_time_sub/var_time_full:.3f}\n")

# overlay spectra 

plt.figure(figsize=(7,5))
plt.loglog(f_full_cpd[1:], P_full[1:], label="Full (10-min)")
plt.loglog(f_sub_cpd[1:], P_sub[1:], label=f"Subsampled x{SUBSAMPLE_FACTOR}")
plt.xlabel("Frequency [cpd]")
plt.ylabel("PSD [ (m/s)^2 / Hz ]")
plt.title("Wind Speed Spectrum (2021)")
plt.legend()
plt.tight_layout()
plt.show()

# d) frequency identification 

# Simple peak detection: find max 
idx_full_peak = np.argmax(P_full[1:]) + 1
idx_sub_peak  = np.argmax(P_sub[1:]) + 1
peak_full_cpd = f_full_cpd[idx_full_peak]
peak_sub_cpd  = f_sub_cpd[idx_sub_peak]

print("PEAK FREQUENCIES")
print(f"Dominant peak (full spectrum): {peak_full_cpd:.3f} cycles/day")
print(f"Dominant peak (subsampled):    {peak_sub_cpd:.3f} cycles/day\n")

# Theoretical alias of a semi-diurnal tide (2 cpd) for the subsampled data
f_semi_cpd = 2.0  
    # semi-diurnal (12h) frequency
fs_sub = fs_sub_cpd

# alias formula: f_alias = |f_true - n*fs|, n = round(f_true/fs)
n_alias = int(round(f_semi_cpd / fs_sub))
f_alias_cpd = abs(f_semi_cpd - n_alias * fs_sub)
T_alias_days = np.inf if f_alias_cpd == 0 else 1.0 / f_alias_cpd

print("=== SEMI-DIURNAL ALIAS (for subsampled data) ===")
print(f"Sampling freq (subsampled): fs = {fs_sub:.3f} cycles/day")
print(f"Semi-diurnal true freq:     f = {f_semi_cpd:.3f} cycles/day")
print(f"n = round(f/fs) = {n_alias}")
print(f"Aliased freq:               f_alias = {f_alias_cpd:.3f} cycles/day")
print(f"Aliased period:             T_alias ≈ {T_alias_days*24:.1f} hours\n")

# e) ENERGY COMPARISON

print("ENERGY COMPARISON")
print("Energy ~ variance in time domain, or integral of PSD.")
print(f"Full record var (time):      {var_time_full:.4f}")
print(f"Full record ∑PSD·df (freq):  {var_freq_full:.4f}")
print(f"Subsampled var (time):       {var_time_sub:.4f}")
print(f"Subsampled ∑PSD·df (freq):   {var_freq_sub:.4f}")
print("\nNote:")
print(" - With correct normalization, each spectrum's integral matches its own variance.")
print(" - The subsampled series has less high-frequency energy, so its variance is smaller.")
print(" - If you used a raw |FFT|^2 with no fs/N scaling, the subsampled spectrum")
print("   would differ from the full one by about the subsample factor (~40).")
