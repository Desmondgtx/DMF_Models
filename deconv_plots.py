# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 22:43:17 2025

@author: yangy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hemodynamic Model and BOLD Deconvolution using rsHRF
Integrates original functions with rsHRF library
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.sparse import lil_matrix


#%% Original BOLD Model Parameters

taus = 0.65
tauf = 0.41
tauo = 0.98
epsilon = 0.5

nu = 40.3
r0 = 25
alpha = 0.32
E0 = 0.4
TE = 0.04
V0 = 0.04

itaus = 1 / taus
itauf = 1 / tauf
itauo = 1 / tauo
ialpha = 1 / alpha

k1 = 4.3 * nu * E0 * TE
k2 = epsilon * r0 * E0 * TE
k3 = 1 - epsilon


#%% Original BOLD Model Functions

def BOLD_response(y, rE, t):
    s, f, v, q = y
    
    s_dot = 1 * rE + 0 - itaus * s - itauf * (f - 1) 
    f_dot = s  
    v_dot = (f - v ** ialpha) * itauo
    q_dot = (f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** ialpha / v) * itauo
    
    return np.vstack((s_dot, f_dot, v_dot, q_dot))


def BOLD_signal(q, v):
    b_dot = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
    return b_dot


def Sim(rE, nnodes, dt):
    Ntotal = rE.shape[0]
    
    ic_BOLD = np.ones((1, nnodes)) * np.array([0.1, 1, 1, 1])[:, None]
    BOLD_vars = np.zeros((Ntotal, 4, nnodes))
    BOLD_vars[0, :, :] = ic_BOLD
    
    for i in range(1, Ntotal):
        BOLD_vars[i, :, :] = BOLD_vars[i - 1, :, :] + dt * BOLD_response(
            BOLD_vars[i - 1, :, :], rE[i - 1, :], i - 1)
    
    y = BOLD_signal(BOLD_vars[:, 3, :], BOLD_vars[:, 2, :])
    
    return y


def ParamsBOLD():
    pardict = {}
    for var in ('taus', 'tauf', 'tauo', 'nu', 'r0', 'alpha', 'epsilon', 'E0', 'V0', 'TE', 'k1', 'k2', 'k3'):
        pardict[var] = eval(var)
    return pardict


#%% rsHRF Integration - Event Detection

def detect_bold_events(bold_signal, thr, localK, temporal_mask=[]):
    N = len(bold_signal)
    data = lil_matrix((1, N))
    matrix = bold_signal[:, np.newaxis]
    matrix = np.nan_to_num(matrix)
    
    if len(temporal_mask) == 0:
        matrix = stats.zscore(matrix, ddof=1)
        for t in range(1 + localK, N - localK + 1):
            if (matrix[t - 1, 0] > thr and 
                np.all(matrix[t - localK - 1:t - 1, 0] < matrix[t - 1, 0]) and 
                np.all(matrix[t - 1, 0] > matrix[t:t + localK, 0])):
                data[0, t - 1] = 1
    else:
        tmp = temporal_mask.copy()
        for i in range(len(temporal_mask)):
            if tmp[i] == 1:
                temporal_mask[i] = i
        datm = np.mean(matrix[temporal_mask])
        datstd = np.std(matrix[temporal_mask])
        if datstd == 0:
            datstd = 1
        matrix = (matrix - datm) / datstd
        for t in range(1 + localK, N - localK + 1):
            if tmp[t - 1]:
                if (matrix[t - 1, 0] > thr and 
                    np.all(matrix[t - localK - 1:t - 1, 0] < matrix[t - 1, 0]) and 
                    np.all(matrix[t - 1, 0] > matrix[t:t + localK, 0])):
                    data[0, t - 1] = 1.
    
    return data


def get_hrf_parameters(hrf, dt):
    param = np.zeros(3)
    
    if np.any(hrf):
        n = int(np.fix(len(hrf) * 0.8))
        p = np.argmax(np.abs(hrf[:n]))
        h = hrf[p]
        
        if h > 0:
            v = (hrf >= (h / 2))
        else:
            v = (hrf <= (h / 2))
        v = v.astype(int)
        b = np.argmin(np.diff(v))
        v[b + 1:] = 0
        w = np.sum(v)
        
        cnt = p - 1
        g = hrf[1:] - hrf[:-1]
        
        while cnt > 0 and np.abs(g[cnt]) < 0.001:
            h = hrf[cnt - 1]
            p = cnt
            cnt = cnt - 1
        
        param[0] = h
        param[1] = (p + 1) * dt
        param[2] = w * dt
    
    return param


#%% rsHRF Deconvolution

def wiener_deconvolution(bold_signal, hrf, regularization=0.1):
    N = len(bold_signal)
    nh = len(hrf)
    
    hrf_padded = np.append(hrf, np.zeros(N - nh))
    
    H = np.fft.fft(hrf_padded)
    Y = np.fft.fft(bold_signal)
    
    Phh = np.abs(H) ** 2
    noise_power = regularization * np.mean(Phh)
    
    WienerFilter = np.conj(H) / (Phh + noise_power)
    
    deconvolved = np.real(np.fft.ifft(WienerFilter * Y))
    
    return deconvolved


def canonical_hrf(t, peak=6, undershoot=16):
    from scipy.stats import gamma as gamma_dist
    
    a1, a2 = 6, 12
    b1, b2 = 0.9, 0.9
    c = 0.35
    
    d1 = a1 * b1
    d2 = a2 * b2
    
    hrf = ((t / d1) ** a1 * np.exp(-(t - d1) / b1) - 
           c * (t / d2) ** a2 * np.exp(-(t - d2) / b2))
    
    hrf = hrf / np.max(hrf)
    
    return hrf


#%% Main Deconvolution Pipeline

def deconvolve_bold_rsHRF(bold_signal, TR=1.0, hrf_length=32, localK=2, threshold=1.0):
    nobs = len(bold_signal)
    
    bold_normalized = stats.zscore(bold_signal, ddof=1)
    bold_normalized = np.nan_to_num(bold_normalized)
    
    events = detect_bold_events(bold_normalized, threshold, localK)
    event_indices = events.toarray().flatten().nonzero()[0]
    
    if len(event_indices) == 0:
        t_hrf = np.arange(0, hrf_length, TR)
        hrf_canonical = canonical_hrf(t_hrf)
    else:
        t_hrf = np.arange(0, hrf_length, TR)
        hrf_canonical = canonical_hrf(t_hrf)
    
    deconvolved_signal = wiener_deconvolution(bold_signal, hrf_canonical)
    
    hrf_params = get_hrf_parameters(hrf_canonical, TR)
    
    results = {
        'bold_original': bold_signal,
        'bold_normalized': bold_normalized,
        'deconvolved': deconvolved_signal,
        'hrf_canonical': hrf_canonical,
        'hrf_time': t_hrf,
        'events': event_indices,
        'hrf_params': {
            'height': hrf_params[0],
            'time_to_peak': hrf_params[1],
            'fwhm': hrf_params[2]
        },
        'TR': TR
    }
    
    return results


#%% Visualization

def plot_deconvolution_results(results, title="BOLD Deconvolution"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    TR = results['TR']
    nobs = len(results['bold_original'])
    time = np.arange(nobs) * TR
    
    # Plot 1: HRF Canonical
    axes[0].plot(results['hrf_time'], results['hrf_canonical'], 'g-', linewidth=2.5)
    axes[0].fill_between(results['hrf_time'], results['hrf_canonical'], alpha=0.3, color='green')
    peak_idx = np.argmax(results['hrf_canonical'])
    axes[0].axvline(results['hrf_time'][peak_idx], color='red', linestyle='--',
                    label=f'Peak: {results["hrf_time"][peak_idx]:.1f}s')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Hemodynamic Response Function (Canonical)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: BOLD signals comparison
    axes[1].plot(time, results['bold_normalized'], 'b-', linewidth=1, label='BOLD (normalized)', alpha=0.7)
    axes[1].plot(time, results['deconvolved'], 'r-', linewidth=1, label='Deconvolved', alpha=0.7)
    
    if len(results['events']) > 0:
        event_plot = np.zeros(nobs)
        event_plot[results['events']] = 1
        markerline, stemlines, baseline = axes[1].stem(time, event_plot, linefmt='k-', 
                                                        markerfmt='kd', basefmt='k-')
        plt.setp(baseline, linewidth=1)
        plt.setp(stemlines, linewidth=1)
        plt.setp(markerline, markersize=3)
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (normalized)')
    axes[1].set_title('BOLD Signal vs Deconvolved Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Original BOLD
    axes[2].plot(time, results['bold_original'], 'b-', linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('BOLD Signal')
    axes[2].set_title('Original BOLD Signal')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


#%% Example Usage

if __name__ == "__main__":
    
    # Simulation parameters
    dt = 1E-3
    tmax = 600
    N = int(tmax / dt)
    t = np.linspace(0, tmax, N)
    delta = 1
    
    # Generate synthetic neural signals
    y1 = np.sin(np.pi * 8 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t)**2)
    y2 = np.sin(np.pi * 16 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t)**2)
    y3 = np.sin(np.pi * 8 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t + np.pi/2)**2)
    
    y = np.vstack((y1, y2, y3)).T
    
    # Simulate BOLD signals
    BOLD_signals = Sim(y, 3, dt)
    
    # Filter BOLD signals
    a0, b0 = signal.bessel(2, [2 * dt * 0.01, 2 * dt * 0.1], btype='bandpass')
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals[60000:, :], axis=0)
    
    # Downsample for deconvolution (simulate TR=1s)
    TR = 1.0
    downsample_factor = int(TR / dt)
    BOLD_downsampled = BOLD_filt[::downsample_factor, 0]
    
    # Deconvolve BOLD signal
    results = deconvolve_bold_rsHRF(BOLD_downsampled, TR=TR, hrf_length=32, localK=2, threshold=1.0)
    
    # Plot results
    fig = plot_deconvolution_results(results, title="rsHRF BOLD Deconvolution")
    plt.show()
    
    # Print HRF parameters
    print("\nHRF Parameters:")
    print(f"  Height: {results['hrf_params']['height']:.4f}")
    print(f"  Time to Peak: {results['hrf_params']['time_to_peak']:.2f} s")
    print(f"  FWHM: {results['hrf_params']['fwhm']:.2f} s")
    print(f"  Number of events detected: {len(results['events'])}")