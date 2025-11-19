#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:13:20 2025

@author: auroraczajkowski
"""

import numpy as np
import matplotlib.pyplot as plt

# QUESTION 1: COMPUTE TWO SPECTRA 
# generate 10,000 elemet data set with Gaussian white noise 
N = 10000          
# number of points
dt = 1.0           
    # time step
white_noise = np.random.normal(0, 1, N)

# integrate white noise 
integrated = np.cumsum(white_noise) * dt

# compute spectra 
# FFT
freqs = np.fft.rfftfreq(N, dt)
wn_spectrum = np.abs(np.fft.rfft(white_noise))**2
int_spectrum = np.abs(np.fft.rfft(integrated))**2

# plot data set 
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(white_noise)
plt.title("White Noise Time Series")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.subplot(1,2,2)
plt.plot(integrated, color="pink")
plt.title("Integrated Noise")
plt.xlabel("Sample")
plt.ylabel("Value")

plt.tight_layout()
plt.show()

# plot spectra 
plt.figure(figsize=(7,5))
plt.loglog(freqs[1:], wn_spectrum[1:], label="White noise", color="purple")
plt.loglog(freqs[1:], int_spectrum[1:], label="Integrated (1/f^2)", color="pink")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Power Spectra")
plt.legend()
plt.show()

# segmented spectra with no windowing 

L = 500              
    # segment length
fs = 1.0 / dt       
    # sampling frequency
df = fs / L          
    # frequency spacing

# building segments for white noise data set 
# 20 non-overlapping segments (shape: 20 x 500)
wn_seg1 = white_noise.reshape(20, L)

# 19 overlapped segments (start at index 250, then every 500 points)
wn_seg2 = white_noise[L//2 : L//2 + L*19].reshape(19, L)

# stack them into total 39 segments
wn_segments = np.vstack((wn_seg1, wn_seg2))

# build segments for integrated noise 
int_seg1 = integrated.reshape(20, L)
int_seg2 = integrated[L//2 : L//2 + L*19].reshape(19, L)
int_segments = np.vstack((int_seg1, int_seg2))


def compute_psd_from_segments(segments, dt):
   # compute one sided array 
    nseg, L = segments.shape
    fs = 1.0 / dt
    df = fs / L

    # FFT of each segment along axis=1
    X = np.fft.rfft(segments, axis=1)

    # One-sided PSD with correct units (≈ variance per Hz)
    # S(f_k) = (2*dt/L) * |X_k|^2, except DC and Nyquist
    S = (2.0 * dt / L) * (np.abs(X) ** 2)
    S[:, 0] /= 2.0               
        # fix DC
    if L % 2 == 0:
        S[:, -1] /= 2.0          
            # fix Nyquist if present

    # average across segments
    S_mean = S.mean(axis=0)
    freqs_seg = np.fft.rfftfreq(L, dt)

    return freqs_seg, S_mean, df


# PSDs for both signals 
freqs_seg, psd_wn, df = compute_psd_from_segments(wn_segments, dt)
_,          psd_int, _ = compute_psd_from_segments(int_segments, dt)

# Parseval's theorm check
# variance in time domain
var_wn_time  = np.mean(white_noise**2)
var_int_time = np.mean(integrated**2)

# variance from frequency domain: sum S(f) * df
var_wn_freq  = np.sum(psd_wn)  * df
var_int_freq = np.sum(psd_int) * df

print("\nParseval check (white noise):")
print("  variance (time domain)   =", var_wn_time)
print("  variance (from spectrum) =", var_wn_freq)
print("  ratio freq/time          =", var_wn_freq / var_wn_time)

print("\nParseval check (integrated noise):")
print("  variance (time domain)   =", var_int_time)
print("  variance (from spectrum) =", var_int_freq)
print("  ratio freq/time          =", var_int_freq / var_int_time)

# plot segmented spectra 
plt.figure(figsize=(7,5))
plt.loglog(freqs_seg[1:], psd_wn[1:],  label="White noise (segmented)",   color="purple")
plt.loglog(freqs_seg[1:], psd_int[1:], label="Integrated (segmented)",    color="pink")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power spectral density")
plt.title("Segmented Spectra (500-pt, 50% overlap)")
plt.legend()
plt.tight_layout()
plt.show()

# QUESTION 2: ANALYTICAL SPECTRA 

# variance of the white noise spectra 
var_wn = np.mean(white_noise**2)

# Sampling frequency
fs = 1.0 / dt

# analytical spectrum for white noise 
# Parseval: variance = integral_0^{fs/2} S_wn(f) df
# For flat white noise S_wn(f) = S0:
#    var_wn = S0 * (fs/2)  -->  S0 = 2 * var_wn / fs
S0 = 2.0 * var_wn / fs   # constant (one-sided) PSD level
S_wn_analytic = S0 * np.ones_like(freqs_seg)

# analytical spectrum for integrated noise
# If y(t) = integral x(t) dt, then in Fourier space:
#    Y(f) = X(f) / (i 2pi f)
# So the PSDs obey:
#    S_int(f) = S_wn(f) / (2pi f)^2
# Use the analytical flat S_wn(f) = S0:
S_int_analytic = np.zeros_like(freqs_seg)
# avoid f = 0 (singularity)
nonzero = freqs_seg > 0
S_int_analytic[nonzero] = S0 / (2.0 * np.pi * freqs_seg[nonzero])**2

# plot both spectra 

plt.figure(figsize=(7,5))
# numerical (from segments)
plt.loglog(freqs_seg[1:], psd_wn[1:],  label="White noise (estimated)",  color="purple")
# analytical
plt.loglog(freqs_seg[1:], S_wn_analytic[1:], "--", label="White noise (analytic)", color="purple")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Power spectral density")
plt.title("White Noise: Estimated vs Analytical Spectrum")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
# numerical (from segments)
plt.loglog(freqs_seg[1:], psd_int[1:],  label="Integrated (estimated)",  color="pink")
# analytical
plt.loglog(freqs_seg[1:], S_int_analytic[1:], "--", label="Integrated (analytic)", color="pink")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Power spectral density")
plt.title("Integrated Noise: Estimated vs Analytical Spectrum")
plt.legend()
plt.tight_layout()
plt.show()

# QUESTION 3: ERROR BARS 

# Number of segments used in the Welch average
M_wn  = wn_segments.shape[0]   
    # should be 39
M_int = int_segments.shape[0]  
    # same here

# 1-sigma uncertainties for each frequency bin
err_wn  = psd_wn  / np.sqrt(M_wn)
err_int = psd_int / np.sqrt(M_int)

# To avoid a super-cluttered plot, only show error bars at every 4th frequency
# Confirmed this in class, could have just plotted one because error bars are consistent 
idx = np.arange(1, len(freqs_seg), 4)   # skip f=0

# white noise with error bars 
plt.figure(figsize=(7,5))
plt.loglog(freqs_seg[1:], psd_wn[1:], label="White noise PSD (estimated)", color="purple")

# add error bars (1-sigma)
plt.errorbar(freqs_seg[idx], psd_wn[idx],
             yerr=err_wn[idx],
             fmt='o', markersize=3, capsize=2,
             linestyle='none', color="purple",
             label="1σ error bars")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Power spectral density")
plt.title("White Noise PSD with Error Bars")
plt.legend()
plt.tight_layout()
plt.show()

# integrated noise with error bars 
plt.figure(figsize=(7,5))
plt.loglog(freqs_seg[1:], psd_int[1:], label="Integrated PSD (estimated)", color="pink")

plt.errorbar(freqs_seg[idx], psd_int[idx],
             yerr=err_int[idx],
             fmt='o', markersize=3, capsize=2,
             linestyle='none', color="pink",
             label="1σ error bars")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Power spectral density")
plt.title("Integrated Noise PSD with Error Bars")
plt.legend()
plt.tight_layout()
plt.show()

# QUESTION 4: MONTE CARLO SIMULATION TO VERIFY CHI SQUARED ERROR BARS 

import numpy as np

# Monte Carlo parameters
Nens = 200        
    # number of realizations
L = 500           
    # segment length
Nseg = 20         
    # segments per realization (no overlap)
dt = 1.0          
    # sampling interval
fs = 1/dt
df = fs / L

# Container for 200 PSDs
psd_ensemble = np.zeros((Nens, L//2 + 1))

for i in range(Nens):
    # Generate Nseg independent white-noise segments (500x20)
    data = np.random.normal(0, 1, (Nseg, L))

    # FFT each segment
    X = np.fft.rfft(data, axis=1)

    # One-sided PSD for each segment
    S = (2*dt/L) * (np.abs(X)**2)
    S[:, 0] /= 2.0
    if L % 2 == 0:
        S[:, -1] /= 2.0

    # Average over the 20 segments → 1 PSD per ensemble member
    psd_ensemble[i, :] = S.mean(axis=0)


# Build a giant PDF by merging all frequencies
    # (allowed because white noise is identical at all f)
all_values = psd_ensemble[:, 1:].flatten()   # drop DC

# Examine PDF histogram to test chi squared distribution
plt.figure(figsize=(7,5))
plt.hist(all_values / np.mean(all_values), bins=50, density=True, color="purple", alpha=0.7)
plt.xlabel("PSD / mean(PSD)")
plt.ylabel("Probability density")
plt.title("Monte Carlo PSD PDF (all frequencies merged)")
plt.show()

# Compute 95% confidence limits at each frequency
lower_limit = np.zeros(psd_ensemble.shape[1])
upper_limit = np.zeros(psd_ensemble.shape[1])

for k in range(psd_ensemble.shape[1]):
    sorted_vals = np.sort(psd_ensemble[:, k])
    lower_limit[k] = sorted_vals[int(0.025 * Nens)]   
        # 5th value
    upper_limit[k] = sorted_vals[int(0.975 * Nens)]   
        # 195th value

ratio_upper_lower = upper_limit / lower_limit

# Compare Monte Carlo 95% spread to theory
#    For M = 20 segments, the PSD has 40 DOF --> chi squared, 40
#    Expected 95% range is:
#    S_low = S0 * chi squared (0.025(40)) / 40
#    S_high = S0 * chi squared (0.975(40)) / 40
#    Ratio = S_high / S_low  (should be ~2.5)

plt.figure(figsize=(7,5))
plt.plot(ratio_upper_lower[1:], color="pink")
plt.xlabel("Frequency bin")
plt.ylabel("Upper / Lower ratio")
plt.title("Monte Carlo 95% Confidence Ratio")
plt.tight_layout()
plt.show()

print("Mean ratio of 95% limits over all frequencies:")
print(np.mean(ratio_upper_lower[1:]))

# QUESTION 5: MONTE CARLO PROCESS FOR A HANNING WINDOW

import numpy as np
import matplotlib.pyplot as plt

Nens = 200       
    # number of realizations
L = 500          
    # segment length
N = 10000        
    # total samples per realization (same as your main data)
dt = 1.0
fs = 1.0 / dt
df = fs / L

# Hanning window and its normalization factor
window = np.hanning(L)
U = np.mean(window**2)   
    # average of w^2, used for correct PSD normalization

# container for 200 PSDs
psd_ensemble_hann = np.zeros((Nens, L//2 + 1))

for i in range(Nens):
    # 1D white-noise realization
    x = np.random.normal(0, 1, N)

    # build 500-pt segments with 50% overlap
    seg1 = x.reshape(20, L)   
        # 20 non-overlapping segments
    seg2 = x[L//2 : L//2 + L*19].reshape(19, L)  
        # 19 overlapped segments
    segments = np.vstack((seg1, seg2))           
        # 39 segments total, shape (39, 500)

    # apply Hanning window to each segment
    segments_w = segments * window  
        # broadcasts window over rows

    # FFT of each windowed segment
    X = np.fft.rfft(segments_w, axis=1)

    # one-sided PSD with window normalization:
    # S(f) = (2 * dt / (L * U)) * |X|^2
    S = (2.0 * dt / (L * U)) * (np.abs(X)**2)
    S[:, 0] /= 2.0
    if L % 2 == 0:
        S[:, -1] /= 2.0

    # average over segments --> one PSD for this realization
    psd_ensemble_hann[i, :] = S.mean(axis=0)


# Build PDF by merging all frequencies (white noise)
all_values_hann = psd_ensemble_hann[:, 1:].flatten()   
    # drop DC

plt.figure(figsize=(7,5))
plt.hist(all_values_hann / np.mean(all_values_hann),
         bins=50, density=True)
plt.xlabel("PSD / mean(PSD)")
plt.ylabel("Probability density")
plt.title("Monte Carlo PSD PDF (Hanning, 50% overlap)")
plt.tight_layout()
plt.show()

# 95% confidence limits at each frequency
lower_limit_hann = np.zeros(psd_ensemble_hann.shape[1])
upper_limit_hann = np.zeros(psd_ensemble_hann.shape[1])

for k in range(psd_ensemble_hann.shape[1]):
    sorted_vals = np.sort(psd_ensemble_hann[:, k])
    lower_limit_hann[k] = sorted_vals[int(0.025 * Nens)]   # 5th value
    upper_limit_hann[k] = sorted_vals[int(0.975 * Nens)]   # 195th value

ratio_upper_lower_hann = upper_limit_hann / lower_limit_hann

plt.figure(figsize=(7,5))
plt.plot(ratio_upper_lower_hann[1:])
plt.xlabel("Frequency bin")
plt.ylabel("Upper / Lower ratio")
plt.title("95% Confidence Ratio (Hanning + 50% overlap)")
plt.tight_layout()
plt.show()

print("Mean 95% ratio (Hanning + 50% overlap, ignoring DC):")
print(np.mean(ratio_upper_lower_hann[1:]))

# QUESTION 6: MONTE CARLO FOR COSINE WINDOW 
# cos(pi t / T),  -T/2 < t < T/2  (discrete approximation)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2   # for chi-squared quantiles

Nens = 200       
    # number of realizations
L    = 500      
    # segment length
N    = 10000     
    # total samples in each realization
dt   = 1.0
fs   = 1.0 / dt
df   = fs / L

# build cosine window: w(n) ≈ cos(pi t/T), -T/2 < t < T/2
n = np.arange(L)
# t goes from -(L-1)/2 to +(L-1)/2 (in sample units), T ≈ L
t = n - (L - 1) / 2.0
window_cos = np.cos(np.pi * t / L)
U_cos = np.mean(window_cos**2)   
    # average of w^2, for PSD normalization

# container for PSDs from each realization
psd_ensemble_cos = np.zeros((Nens, L//2 + 1))

for i in range(Nens):
    # one realization of white noise
    x = np.random.normal(0, 1, N)

    # 500-pt segments with 50% overlap, 20 non-overlap + 19 overlap = 39 segments
    seg1 = x.reshape(20, L)                               
        # non-overlapping
    seg2 = x[L//2 : L//2 + L*19].reshape(19, L)           
        # overlapped
    segments = np.vstack((seg1, seg2))                    
        # shape (39, 500)

    # apply cosine window
    segments_w = segments * window_cos

    # FFT each windowed segment
    X = np.fft.rfft(segments_w, axis=1)

    # one-sided PSD with window normalization
    S = (2.0 * dt / (L * U_cos)) * (np.abs(X)**2)
    S[:, 0] /= 2.0
    if L % 2 == 0:
        S[:, -1] /= 2.0

    # average over segments, one PSD for this realization
    psd_ensemble_cos[i, :] = S.mean(axis=0)


# Look at the PDF of normalized PSD values (all freqs merged)
all_vals_cos = psd_ensemble_cos[:, 1:-1].flatten()   
    # drop DC & Nyquist
norm_vals_cos = all_vals_cos / np.mean(all_vals_cos)

plt.figure(figsize=(7,5))
plt.hist(norm_vals_cos, bins=50, density=True)
plt.xlabel("PSD / mean(PSD)")
plt.ylabel("Probability density")
plt.title("Monte Carlo PSD PDF (cosine window, 50% overlap)")
plt.tight_layout()
plt.show()

# Estimate effective DOF from the variance of normalized PSD
#    For chi^2_nu:  S_hat / S_true = chi2_nu / nu,  Var = 2/nu
var_norm = np.var(norm_vals_cos, ddof=1)
nu_eff_var = 2.0 / var_norm
print(f"Effective DOF from variance method: nu_eff ≈ {nu_eff_var:.1f}")

# Also estimate DOF from the 95% confidence ratio at each frequency
lower_cos = np.zeros(psd_ensemble_cos.shape[1])
upper_cos = np.zeros(psd_ensemble_cos.shape[1])

for k in range(psd_ensemble_cos.shape[1]):
    vals = np.sort(psd_ensemble_cos[:, k])
    lower_cos[k] = vals[int(0.025 * Nens)]
    upper_cos[k] = vals[int(0.975 * Nens)]

ratio_cos = upper_cos / lower_cos

plt.figure(figsize=(7,5))
plt.plot(ratio_cos[1:-1])
plt.xlabel("Frequency bin")
plt.ylabel("Upper / Lower ratio")
plt.title("95% Confidence Ratio (cosine window, 50% overlap)")
plt.tight_layout()
plt.show()

mean_ratio_cos = np.mean(ratio_cos[1:-1])
print(f"Mean 95% ratio (ignoring DC & Nyquist): R ≈ {mean_ratio_cos:.3f}")

# now find nu such that chi2_0.975(nu)/chi2_0.025(nu) ~ mean_ratio_cos
nus = np.arange(4, 200)   # search range for DOF
theory_ratios = chi2.ppf(0.975, nus) / chi2.ppf(0.025, nus)
idx_best = np.argmin((theory_ratios - mean_ratio_cos)**2)
nu_eff_ratio = nus[idx_best]

print(f"Effective DOF from 95% ratio method: nu_eff ≈ {nu_eff_ratio} "
      f"(theoretical ratio = {theory_ratios[idx_best]:.3f})")

