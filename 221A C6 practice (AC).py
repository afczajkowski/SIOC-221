#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 12:02:03 2025

@author: auroraczajkowski
"""

import numpy as np
import matplotlib.pyplot as plt

def wave_analysis(n, m):
    out = {}
    out['time'] = np.arange(0, 2 * np.pi, 0.01)
    out['wave1'] = np.cos(2 * np.pi * m * out['time'])
    out['wave2'] = np.cos(2 * np.pi * n * out['time'])
    out['product'] = out['wave1'] * out['wave2']
    out['integral'] = np.trapz(out['product'], out['time'])
    return out

# Run the function
result = wave_analysis(n=2, m=3)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(result['time'], result['wave1'])
axs[0].set_ylabel('Wave 1')

axs[1].plot(result['time'], result['wave2'])
axs[1].set_ylabel('Wave 2')

axs[2].plot(result['time'], result['product'])
axs[2].set_ylabel('Product')
axs[2].set_xlabel('Time')

plt.tight_layout()
plt.show()
