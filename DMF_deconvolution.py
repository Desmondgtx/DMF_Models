# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel
Modified by: Diego Garrido (2025) - Added HRF deconvolution analysis

Run a simulation using the Dynamic Mean Field (DMF) model with HRF deconvolution.
The output includes BOLD-like signals, their deconvolution, HRF estimation and impulse response.
"""

#%% Libraries

import numpy as np
from scipy import signal, stats
import DMF as DMF
import matplotlib.pyplot as plt
import deconvolution as bd
import time

#%% DMF Simulation Setup

# Simulation parameters
DMF.tmax = 1020
DMF.dt = 0.001
DMF_ratio = 1 / DMF.dt
decimate = 10
DMF.downsampling = int(DMF_ratio / decimate)
DMF.downsampling_rates = 1
BOLD_dt = DMF.dt * DMF.downsampling

# Network parameters
struct = np.loadtxt("SC_opti_25julio.txt")
FCe = np.load("average_90x90FC_HCPchina_symm.npy")
DMF.SC = struct / np.mean(np.sum(struct, 0))
DMF.nnodes = len(DMF.SC)

# Model parameters
DMF.G = 1.07
DMF.sigma = 0.4

DMF.update()

#%% Run DMF Simulation

start = time.perf_counter()
BOLD_signals, rates, t = DMF.Sim(verbose=True, return_rates=True)
simulation_time = time.perf_counter() - start

print(f"\nSimulation completed in {simulation_time:.2f} seconds")

#%% Standard BOLD Processing

BOLD_signals = BOLD_signals[int(120 / BOLD_dt):, :]

a0, b0 = signal.bessel(3, [2 * BOLD_dt * 0.01, 2 * BOLD_dt * 0.1], btype='bandpass')
BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis=0)
BOLD_filt = BOLD_filt[int(60 / BOLD_dt):-int(60 / BOLD_dt)]

FC = np.corrcoef(BOLD_filt.T)

# f, psd = signal.welch(BOLD_signals, fs=1 / BOLD_dt, axis=0, nperseg=2000, noverlap=1500)
# f_f, psd_f = signal.welch(BOLD_filt, fs=1 / BOLD_dt, axis=0, nperseg=2000, noverlap=1500)

#%% HRF Deconvolution Analysis

TR = BOLD_dt
para = {
    'TR': TR,
    'len': 32,
    'localK': 2,
    'thr': 1.0,
    'dt': TR,
    'T': 16,
    'passband': [0.01, 0.1],
    'passband_deconvolve': [0.01, 0.1]
}

region_variance = np.var(BOLD_filt, axis=0)
representative_region = np.argmax(region_variance)

bold_signal = BOLD_filt[:, representative_region]

deconv_start = time.perf_counter()
results = bd.deconvolve_with_canonical_hrf(bold_signal, para)
deconv_time = time.perf_counter() - deconv_start

print(f"\nDeconvolution completed in {deconv_time:.2f} seconds")
print(f"\nHRF Parameters (region {representative_region}):")
print(f"  Height: {results['hrf_params']['height']:.4f}")
print(f"  Time to Peak: {results['hrf_params']['time_to_peak']:.2f} s")
print(f"  FWHM: {results['hrf_params']['fwhm']:.2f} s")
print(f"  Number of events detected: {len(results['events'])}")

deconvolved_signals = np.zeros_like(BOLD_filt)
hrf_params_all = {
    'height': np.zeros(DMF.nnodes),
    'time_to_peak': np.zeros(DMF.nnodes),
    'fwhm': np.zeros(DMF.nnodes)
}

for i in range(DMF.nnodes):
    if i % 10 == 0:
        print(f"  Processing region {i}/{DMF.nnodes}")
    
    result_i = bd.deconvolve_with_canonical_hrf(BOLD_filt[:, i], para)
    deconvolved_signals[:, i] = result_i['deconvolved']
    hrf_params_all['height'][i] = result_i['hrf_params']['height']
    hrf_params_all['time_to_peak'][i] = result_i['hrf_params']['time_to_peak']
    hrf_params_all['fwhm'][i] = result_i['hrf_params']['fwhm']

FC_deconv = np.corrcoef(deconvolved_signals.T)

#%% Recover HRF from DMF Model

# Use mean BOLD signal to estimate average HRF shape
bold_mean = np.mean(BOLD_filt, axis=1)

para_dmf = {
    'TR': BOLD_dt,
    'len': 32,
    'localK': 2,
    'thr': 0.8,
    'dt': BOLD_dt,
    'T': 16
}

results_dmf_hrf = bd.deconvolve_with_canonical_hrf(bold_mean, para_dmf)

#%% Plotting Results

ii = (1, 2, 3, 4, 11, 12, 13, 14)

# Figure 1: BOLD signals and time series
plt.figure(1, figsize=(14, 10))
plt.clf()

plt.subplot2grid((3, 6), (0, 0), colspan=4)
plt.plot(BOLD_signals[:, ii])
plt.title('Raw BOLD Signals (DMF)', fontweight='bold')
plt.ylabel('Amplitude')

t_BOLDf = t[1800:-600]
ax2 = plt.subplot2grid((3, 6), (1, 0), colspan=4)
ax2.plot(t_BOLDf, BOLD_filt[:, ii])
ax2.set_title('Filtered BOLD Signals (0.01-0.1 Hz)', fontweight='bold')
ax2.set_ylabel('Amplitude')
ax2.spines[['right', 'top']].set_visible(False)

ax3 = plt.subplot2grid((3, 6), (2, 0), colspan=4)
ax3.plot(t_BOLDf, deconvolved_signals[:, ii])
ax3.set_title('Deconvolved Signals (rsHRF)', fontweight='bold')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude')
ax3.spines[['right', 'top']].set_visible(False)

# plt.subplot2grid((3, 6), (0, 4), colspan=2)
# plt.plot(f, psd[:, ::10])
# plt.xlim((0, 1))
# plt.title('PSD - Raw BOLD')
# plt.xlabel('Frequency (Hz)')

# plt.subplot2grid((3, 6), (1, 4), colspan=2)
# plt.plot(f_f, psd_f[:, ::10])
# plt.xlim((0, 1))
# plt.title('PSD - Filtered BOLD')
# plt.xlabel('Frequency (Hz)')

# ax_psd_deconv = plt.subplot2grid((3, 6), (2, 4), colspan=2)
# f_deconv, psd_deconv = signal.welch(deconvolved_signals, fs=1/BOLD_dt, axis=0, nperseg=2000, noverlap=1500)
# ax_psd_deconv.plot(f_deconv, psd_deconv[:, ::10])
# ax_psd_deconv.set_xlim((0, 1))
# ax_psd_deconv.set_title('PSD - Deconvolved')
# ax_psd_deconv.set_xlabel('Frequency (Hz)')

plt.tight_layout()

# Figure 2: Connectivity matrices
plt.figure(2, figsize=(14, 8))
plt.clf()

plt.subplot(221)
plt.imshow(struct, cmap='jet')
plt.title('Structural Connectivity', fontweight='bold')
plt.colorbar()

plt.subplot(223)
plt.imshow(FCe, cmap='jet', vmin=-1, vmax=1)
plt.title('Empirical FC', fontweight='bold')
plt.colorbar()

plt.subplot(222)
plt.imshow(FC, cmap='jet', vmin=-1, vmax=1)
plt.title('FC - Filtered BOLD (DMF)', fontweight='bold')
plt.colorbar()

plt.subplot(224)
plt.imshow(FC_deconv, cmap='jet', vmin=-1, vmax=1)
plt.title('FC - Deconvolved Signals', fontweight='bold')
plt.colorbar()

plt.tight_layout()

FCvecE = FCe[np.tril_indices_from(FCe, -1)]
FCvec_BOLD = FC[np.tril_indices_from(FC, -1)]
FCvec_deconv = FC_deconv[np.tril_indices_from(FC_deconv, -1)]

corr_bold = stats.pearsonr(FCvecE, FCvec_BOLD)[0]
corr_deconv = stats.pearsonr(FCvecE, FCvec_deconv)[0]
eucl_bold = np.sqrt(np.sum((FCvecE - FCvec_BOLD)**2))
eucl_deconv = np.sqrt(np.sum((FCvecE - FCvec_deconv)**2))

# Figure 3: HRF from resting-state BOLD
fig_hrf, fig_bold = bd.plot_hrf_and_deconvolution(results)

# Figure 4: HRF recovered from DMF model
plt.figure(4, figsize=(12, 8))
plt.clf()

plt.subplot(221)
plt.plot(t_BOLDf, bold_mean, 'b-', linewidth=2, label='Mean BOLD')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Mean BOLD Signal (DMF)', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(222)
plt.plot(results_dmf_hrf['hrf_time'], results_dmf_hrf['hrf'], 'k-', linewidth=2.5)
plt.fill_between(results_dmf_hrf['hrf_time'], results_dmf_hrf['hrf'], alpha=0.3)
peak_idx = np.argmax(results_dmf_hrf['hrf'])
plt.axvline(results_dmf_hrf['hrf_time'][peak_idx], color='r', linestyle='--', 
            label=f'Peak: {results_dmf_hrf["hrf_time"][peak_idx]:.1f}s')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('HRF from Mean BOLD (DMF)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(223)
plt.plot(results['hrf_time'], results['hrf'], 'b-', linewidth=2.5, label='Single Region', alpha=0.7)
plt.plot(results_dmf_hrf['hrf_time'], results_dmf_hrf['hrf'], 'r-', linewidth=2.5, label='Mean BOLD', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('HRF Comparison', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(224)
dmf_params_text = f"Mean BOLD HRF Parameters:\n" \
                  f"Height: {results_dmf_hrf['hrf_params']['height']:.4f}\n" \
                  f"Time to Peak: {results_dmf_hrf['hrf_params']['time_to_peak']:.2f} s\n" \
                  f"FWHM: {results_dmf_hrf['hrf_params']['fwhm']:.2f} s\n\n" \
                  f"Single Region HRF Parameters:\n" \
                  f"Height: {results['hrf_params']['height']:.4f}\n" \
                  f"Time to Peak: {results['hrf_params']['time_to_peak']:.2f} s\n" \
                  f"FWHM: {results['hrf_params']['fwhm']:.2f} s"
plt.text(0.1, 0.5, dmf_params_text, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')

plt.tight_layout()

#%% Summary Statistics

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nGlobal Coupling G: {DMF.G}")

print("\nHRF Parameters (Mean ± Std across all regions):")
print(f"  Height:        {np.mean(hrf_params_all['height']):.4f} ± {np.std(hrf_params_all['height']):.4f}")
print(f"  Time to Peak:  {np.mean(hrf_params_all['time_to_peak']):.2f} ± {np.std(hrf_params_all['time_to_peak']):.2f} s")
print(f"  FWHM:          {np.mean(hrf_params_all['fwhm']):.2f} ± {np.std(hrf_params_all['fwhm']):.2f} s")

print("\nHRF from Mean BOLD Signal:")
print(f"  Height:        {results_dmf_hrf['hrf_params']['height']:.4f}")
print(f"  Time to Peak:  {results_dmf_hrf['hrf_params']['time_to_peak']:.2f} s")
print(f"  FWHM:          {results_dmf_hrf['hrf_params']['fwhm']:.2f} s")

print("\nFunctional Connectivity:")
print(f"  Mean FC (BOLD):        {np.mean(FC[np.tril_indices_from(FC, -1)]):.4f}")
print(f"  Mean FC (Deconvolved): {np.mean(FC_deconv[np.tril_indices_from(FC_deconv, -1)]):.4f}")

print("\nCorrelation with Empirical FC:")
print(f"  BOLD:        r = {corr_bold:.4f}, Euclidean = {eucl_bold:.4f}")
print(f"  Deconvolved: r = {corr_deconv:.4f}, Euclidean = {eucl_deconv:.4f}")

print("\n" + "=" * 60)
print("Analysis completed successfully!")
print("=" * 60)

plt.show()