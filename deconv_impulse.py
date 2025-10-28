# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 22:43:17 2025

@author: yangy

Hemodynamic Model and BOLD Deconvolution using rsHRF
Integrates original functions with rsHRF library

References:
[1] Stephan, K. E., Weiskopf, N., Drysdale, P. M., Robinson, P. A., & Friston, K. J. 
(2007). Comparing hemodynamic models with DCM. Neuroimage, 38(3), 387-401.

[2] Wu, G. R., Liao, W., Stramaglia, S., Ding, J. R., Chen, H., & Marinazzo, D. 
(2013). A blind deconvolution approach to recover effective connectivity brain 
networks from resting state fMRI data. Medical image analysis, 17(3), 365-374.
"""

#%% Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.sparse import lil_matrix
from scipy.stats import gamma as gamma_dist


#%% Original BOLD Model Parameters

### FREE PARAMETERS
## Balloon-Windkessel Hemodynamic Model of BOLD Activity 
## Paper: Stephan et al. (2007) - Neural Mass Modeling (p. 1603)
taus = 0.65     # time constant for signal decay (s) [Kappa]
tauf = 0.41     # time constant for feedback regulation (s) [Gamma]
tauo = 0.98     # time constant for volume and deoxyhemoglobin content change (s)
                # (blood volume is v, deoxyhemoglobin content is q)
epsilon = 0.5   # ratio of intra and extravascular signal (dimensionless)

### FIXED PARAMETERS
## Stephan et al. (2007) p. 391 - Kinetics constants
nu = 40.3       # frequency offset at the outer surface of the magnetized
                # vessel for fully deoxygenated blood at 1.5 Tesla (s^-1)
r0 = 25         # slope of the relation between the intravascular relaxation 
                # rate and oxygen saturation (s^-1)         
alpha = 0.32    # resistance of the veins; stiffness constant (dimensionless)
E0 = 0.4        # resting oxygen extraction fraction (dimensionless)
TE = 0.04       # echo time (s) - determined by the experiment
V0 = 0.04       # resting venous blood volume fraction (dimensionless)

# Inverse variables
itaus = 1 / taus    # inverse of Kappa (1/s)
itauf = 1 / tauf    # inverse of Gamma (1/s)
itauo = 1 / tauo    # inverse of tauo (1/s)
ialpha = 1 / alpha  # inverse of alpha (dimensionless)

# Kinetics constants - Stephan et al. (2007) p. 391
k1 = 4.3 * nu * E0 * TE     # signal constant (dimensionless)
k2 = epsilon * r0 * E0 * TE # signal constant (dimensionless)
k3 = 1 - epsilon            # signal constant (dimensionless)


#%% Original BOLD Model Functions

def BOLD_response(y, rE, t):
    """
    Generates BOLD response using firing rates through the Balloon-Windkessel model.
    This implements the hemodynamic state equations from Stephan et al. (2007).
    ----------
    Parameters:
    y : numpy array (4 x n_nodes)
        Contains the hemodynamic state variables:
        s: vasodilatory signal (dimensionless)
        f: blood inflow (dimensionless, relative to resting state)
        v: blood volume (dimensionless, relative to resting state)
        q: deoxyhemoglobin content (dimensionless, relative to resting state)
    rE : numpy array (n_nodes,)
        Firing rates of neural populations/neurons
    t : float
        Current simulation time point (not used in computation but kept for compatibility)
    ----------
    Returns:
    numpy array (4 x n_nodes)
        Time derivatives of [s, f, v, q] at time t
    
    Reference: Stephan et al. (2007) Eq. (2-5), p. 388
    """
    s, f, v, q = y
    
    # Stephan et al. (2007) Eq. (2-5), p. 388
    s_dot = 1 * rE + 0 - itaus * s - itauf * (f - 1) # Signal compartment - Eq. (2)
    f_dot = s                                        # Blood inflow - Eq. (3)
    v_dot = (f - v ** ialpha) * itauo                # Blood volume - Eq. (4)
    q_dot = (f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** ialpha / v) * itauo # Deoxyhemoglobin content - Eq. (5)

    # Corrected according to the paper
    # s_dot = 0.5 * rE + 3 - itaus * s - itauf * (f - 1)
    # f_dot = s  
    # v_dot = f - v ** (1 / ialpha) 
    # q_dot = (f * ((1 - E0) ** (1 / f)) / E0 - q * v ** (1 / ialpha) / v) 
    
    return np.vstack((s_dot, f_dot, v_dot, q_dot))


def BOLD_signal(q, v):
    """
    Returns the BOLD signal using deoxyhemoglobin content and blood volume.
    This implements the BOLD signal equation from Stephan et al. (2007).
    ----------
    Parameters:
    q : numpy array (time_points x n_nodes)
        Deoxyhemoglobin content over time (dimensionless)
    v : numpy array (time_points x n_nodes)
        Blood volume over time (dimensionless)
    ----------
    Returns:
    numpy array (time_points x n_nodes)
        BOLD signal change (dimensionless)
    
    Reference: Stephan et al. (2007) Eq. (6), p. 388 - BOLD signal change equation
    """
    # Stephan et al. (2007) Eq. (6), p. 388 - BOLD signal change equation
    b_dot = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
    
    return b_dot


def Sim(rE, nnodes, dt):
    """
    Simulates BOLD-like signals (raw non-filtered) using Euler integration.
    The time unit in this model is seconds.
    ----------
    Parameters:
    rE : numpy array (time_points x n_nodes)
        Matrix containing firing rates of each node over time
    nnodes : int
        Number of nodes/brain regions
    dt : float
        Integration time step in seconds (the inverse of sampling rate)
    ----------
    Returns:
    numpy array (time_points x n_nodes)
        Raw BOLD-like signals for each node
    
    Reference: Stephan et al. (2007) - Numerical integration of hemodynamic model
    """
    Ntotal = rE.shape[0]
    
    # Initial conditions: s=0.1, f=1, v=1, q=1 (resting state)
    # Stephan et al. (2007) - typical initial conditions
    ic_BOLD = np.ones((1, nnodes)) * np.array([0.1, 1, 1, 1])[:, None]
    BOLD_vars = np.zeros((Ntotal, 4, nnodes))
    BOLD_vars[0, :, :] = ic_BOLD
    
    # Euler method integration
    for i in range(1, Ntotal):
        BOLD_vars[i, :, :] = BOLD_vars[i - 1, :, :] + dt * BOLD_response(BOLD_vars[i - 1, :, :], rE[i - 1, :], i - 1)
    
    # Extract BOLD signal from deoxyhemoglobin content (q) and blood volume (v)
    y = BOLD_signal(BOLD_vars[:, 3, :], BOLD_vars[:, 2, :])
    
    return y


def ParamsBOLD():
    """
    Returns a dictionary with all BOLD model parameters.
    Useful for exporting parameter configurations.
    ----------
    Parameters:
    None
    ----------
    Returns:
    dict
        Dictionary containing all model parameters
    
    Reference: Stephan et al. (2007) - Table 1, p. 391
    """
    pardict = {}
    for var in ('taus', 'tauf', 'tauo', 'nu', 'r0', 'alpha', 'epsilon', 'E0', 'V0', 'TE', 'k1', 'k2', 'k3'):
        pardict[var] = eval(var)
    return pardict


#%% rsHRF Integration - Event Detection

def detect_bold_events(bold_signal, thr, localK, temporal_mask=[]):
    """
    Detects spontaneous BOLD events (peaks) in resting-state fMRI data.
    Based on Wu et al. (2013) blind deconvolution approach.
    ----------
    Parameters:
    bold_signal : numpy array (time_points,)
        BOLD signal time series
    thr : float
        Threshold for event detection (in standard deviations)
    localK : int
        Local neighborhood size for peak detection (in time points)
    temporal_mask : list or numpy array, optional
        Binary mask indicating valid time points (default: [])
    ----------
    Returns:
    scipy.sparse.lil_matrix (1 x time_points)
        Sparse matrix with 1s at detected event locations
    
    Reference: Wu et al. (2013) Section 2.1, Fig. 1
    """
    N = len(bold_signal)
    data = lil_matrix((1, N))
    matrix = bold_signal[:, np.newaxis]
    matrix = np.nan_to_num(matrix)
    
    if len(temporal_mask) == 0:
        # Z-score normalization
        matrix = stats.zscore(matrix, ddof=1)
        
        # Detect local maxima above threshold
        # Wu et al. (2013) - Event detection criterion
        for t in range(1 + localK, N - localK + 1):
            if (matrix[t - 1, 0] > thr and 
                np.all(matrix[t - localK - 1:t - 1, 0] < matrix[t - 1, 0]) and 
                np.all(matrix[t - 1, 0] > matrix[t:t + localK, 0])):
                data[0, t - 1] = 1
    else:
        # With temporal mask (for censoring motion artifacts, etc.)
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
    """
    Extracts HRF shape parameters: height, time-to-peak, and FWHM.
    Based on parameter extraction from rsHRF toolbox.
    ----------
    Parameters:
    hrf : numpy array (time_points,)
        Hemodynamic response function
    dt : float
        Time resolution (sampling interval) in seconds
    ----------
    Returns:
    numpy array (3,)
        [height, time_to_peak, FWHM] of the HRF
        height: peak amplitude
        time_to_peak: time to reach peak (seconds)
        FWHM: full-width at half-maximum (seconds)
    
    Reference: Similar to parameters.py in rsHRF toolbox
    """
    param = np.zeros(3)
    
    if np.any(hrf):
        # Consider only first 80% to avoid tail effects
        n = int(np.fix(len(hrf) * 0.8))
        p = np.argmax(np.abs(hrf[:n]))
        h = hrf[p]
        
        # Calculate FWHM
        if h > 0:
            v = (hrf >= (h / 2))
        else:
            v = (hrf <= (h / 2))
        v = v.astype(int)
        b = np.argmin(np.diff(v))
        v[b + 1:] = 0
        w = np.sum(v)
        
        # Refine peak detection
        cnt = p - 1
        g = hrf[1:] - hrf[:-1]
        
        while cnt > 0 and np.abs(g[cnt]) < 0.001:
            h = hrf[cnt - 1]
            p = cnt
            cnt = cnt - 1
        
        param[0] = h                # height
        param[1] = (p + 1) * dt     # time to peak (seconds)
        param[2] = w * dt           # FWHM (seconds)
    
    return param


#%% rsHRF Deconvolution

def wiener_deconvolution(bold_signal, hrf, regularization=0.1):
    """
    Performs Wiener deconvolution to recover neural signal from BOLD.
    Implements regularized inverse filtering in frequency domain.
    ----------
    Parameters:
    bold_signal : numpy array (time_points,)
        Observed BOLD signal
    hrf : numpy array (hrf_length,)
        Hemodynamic response function
    regularization : float, optional
        Regularization parameter to avoid noise amplification (default: 0.1)
    ----------
    Returns:
    numpy array (time_points,)
        Deconvolved (estimated neural) signal
    
    Reference: Wu et al. (2013) Eq. (4-5), Section 2.1
    """
    N = len(bold_signal)
    nh = len(hrf)
    
    # Zero-pad HRF to match signal length
    hrf_padded = np.append(hrf, np.zeros(N - nh))
    
    # Transform to frequency domain
    H = np.fft.fft(hrf_padded)
    Y = np.fft.fft(bold_signal)
    
    # Wiener filter: Wu et al. (2013) Eq. (4)
    Phh = np.abs(H) ** 2
    noise_power = regularization * np.mean(Phh)
    
    WienerFilter = np.conj(H) / (Phh + noise_power)
    
    # Inverse transform to get deconvolved signal
    deconvolved = np.real(np.fft.ifft(WienerFilter * Y))
    
    return deconvolved


def canonical_hrf(t, peak=6, undershoot=16):
    """
    Generates canonical (double-gamma) hemodynamic response function.
    Based on standard SPM HRF model.
    ----------
    Parameters:
    t : numpy array (time_points,)
        Time vector in seconds
    peak : float, optional
        Time to peak of positive response (default: 6 seconds)
    undershoot : float, optional
        Time to peak of negative undershoot (default: 16 seconds)
    ----------
    Returns:
    numpy array (time_points,)
        Canonical HRF normalized to peak=1
    
    Reference: Similar to SPM canonical HRF
    """
    # Double-gamma function parameters
    a1, a2 = 6, 12      # shape parameters
    b1, b2 = 0.9, 0.9   # scale parameters
    c = 0.35            # ratio of undershoot to main response
    
    d1 = a1 * b1
    d2 = a2 * b2
    
    # Difference of two gamma functions
    hrf = ((t / d1) ** a1 * np.exp(-(t - d1) / b1) - 
           c * (t / d2) ** a2 * np.exp(-(t - d2) / b2))
    
    # Normalize to peak amplitude = 1
    hrf = hrf / np.max(hrf)
    
    return hrf


#%% Main Deconvolution Pipeline

def deconvolve_bold_rsHRF(bold_signal, TR=1.0, hrf_length=32, localK=2, threshold=1.0):
    """
    Main pipeline for BOLD signal deconvolution using rsHRF approach.
    Detects events, estimates HRF, and performs Wiener deconvolution.
    ----------
    Parameters:
    bold_signal : numpy array (time_points,)
        Observed BOLD signal
    TR : float, optional
        Repetition time in seconds (default: 1.0)
    hrf_length : float, optional
        Length of HRF in seconds (default: 32)
    localK : int, optional
        Local neighborhood for event detection (default: 2)
    threshold : float, optional
        Threshold for event detection in standard deviations (default: 1.0)
    ----------
    Returns:
    dict
        Dictionary containing:
        - 'bold_original': original BOLD signal
        - 'bold_normalized': z-scored BOLD signal
        - 'deconvolved': deconvolved signal
        - 'hrf_canonical': estimated HRF
        - 'hrf_time': time vector for HRF
        - 'events': indices of detected events
        - 'hrf_params': dictionary with height, time_to_peak, fwhm
        - 'TR': repetition time
    
    Reference: Wu et al. (2013) - Complete deconvolution pipeline
    """
    nobs = len(bold_signal)
    
    # Normalize BOLD signal
    bold_normalized = stats.zscore(bold_signal, ddof=1)
    bold_normalized = np.nan_to_num(bold_normalized)
    
    # Detect spontaneous BOLD events
    events = detect_bold_events(bold_normalized, threshold, localK)
    event_indices = events.toarray().flatten().nonzero()[0]
    
    # Generate canonical HRF
    t_hrf = np.arange(0, hrf_length, TR)
    hrf_canonical = canonical_hrf(t_hrf)
    
    # Perform Wiener deconvolution
    deconvolved_signal = wiener_deconvolution(bold_signal, hrf_canonical)
    
    # Extract HRF parameters
    hrf_params = get_hrf_parameters(hrf_canonical, TR)
    
    # Package results
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
    """
    Visualizes deconvolution results in separate figures.
    Creates two separate plots: HRF and BOLD comparison.
    ----------
    Parameters:
    results : dict
        Dictionary returned by deconvolve_bold_rsHRF()
    title : str, optional
        Title prefix for plots (default: "BOLD Deconvolution")
    ----------
    Returns:
    tuple of matplotlib.figure.Figure
        (fig_hrf, fig_bold) - two separate figure objects
    """
    TR = results['TR']
    nobs = len(results['bold_original'])
    time = np.arange(nobs) * TR
    
    # Figure 1: HRF Canonical
    fig_hrf = plt.figure(figsize=(10, 5))
    plt.plot(results['hrf_time'], results['hrf_canonical'], 'g-', linewidth=2.5)
    plt.fill_between(results['hrf_time'], results['hrf_canonical'], alpha=0.3, color='green')
    peak_idx = np.argmax(results['hrf_canonical'])
    plt.axvline(results['hrf_time'][peak_idx], color='red', linestyle='--',
                label=f'Peak: {results["hrf_time"][peak_idx]:.1f}s')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Hemodynamic Response Function (Canonical)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Figure 2: BOLD signals comparison
    fig_bold = plt.figure(figsize=(12, 6))
    plt.plot(time, results['bold_normalized'], 'b-', linewidth=1, label='BOLD (normalized)', alpha=0.7)
    plt.plot(time, results['deconvolved'], 'r-', linewidth=1, label='Deconvolved', alpha=0.7)
    
    # Add detected events as stem plot
    if len(results['events']) > 0:
        event_plot = np.zeros(nobs)
        event_plot[results['events']] = 1
        markerline, stemlines, baseline = plt.stem(time, event_plot, linefmt='k-', 
                                                    markerfmt='kd', basefmt='k-')
        plt.setp(baseline, linewidth=1)
        plt.setp(stemlines, linewidth=1)
        plt.setp(markerline, markersize=3)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude (normalized)', fontsize=12)
    plt.title('BOLD Signal vs Deconvolved Signal', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig_hrf, fig_bold


#%% Impulse Response Visualization

def plot_impulse_response(impulse_time=35, tstop=50, dt=0.01, impulse_amplitude=20, 
                         TR=1.0, hrf_length=32):
    """
    Plots BOLD impulse response and its deconvolution for a single stimulus event.
    Useful for visualizing the HRF and validating deconvolution on controlled input.
    ----------
    Parameters:
    impulse_time : float, optional
        Time at which impulse occurs in seconds (default: 35)
    tstop : float, optional
        Total simulation time in seconds (default: 50)
    dt : float, optional
        Integration time step in seconds (default: 0.01)
    impulse_amplitude : float, optional
        Amplitude of the impulse (default: 20)
    TR : float, optional
        Repetition time for deconvolution in seconds (default: 1.0)
    hrf_length : float, optional
        Length of HRF in seconds (default: 32)
    ----------
    Returns:
    matplotlib.figure.Figure
        Figure object with impulse response plots
    
    Reference: Similar to boldImpulse.py - impulse response analysis
    """
    # Generate time vector and impulse stimulus
    time = np.arange(0, tstop, dt)
    x_t = np.zeros_like(time)
    x_t[time == impulse_time] = impulse_amplitude
    
    # Simulate BOLD response to impulse
    BOLD = Sim(x_t[:, None], 1, dt).flatten()
    
    # Downsample for deconvolution
    downsample_factor = int(TR / dt)
    BOLD_downsampled = BOLD[::downsample_factor]
    time_downsampled = time[::downsample_factor]
    
    # Deconvolve BOLD signal
    results = deconvolve_bold_rsHRF(BOLD_downsampled, TR=TR, hrf_length=hrf_length, 
                                    localK=1, threshold=0.5)
    
    # Create figure with two y-axes
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(111)
    
    # Plot BOLD signals
    ax1.plot(time, BOLD, 'b-', linewidth=2, label='BOLD Response', alpha=0.8)
    ax1.plot(time_downsampled, results['deconvolved'], 'r-', linewidth=2, 
             label='Deconvolved Signal', alpha=0.8)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("BOLD Signal", color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot stimulus on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(time, x_t, 'g-', linewidth=2, label='Stimulus')
    ax2.set_ylabel("Stimulus Amplitude", color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.title('BOLD Impulse Response and Deconvolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


#%% Example Usage

if __name__ == "__main__":
    
    # Simulation parameters
    dt = 1E-3       # integration time step (1 ms)
    tmax = 600      # total simulation time (600 s)
    N = int(tmax / dt)
    t = np.linspace(0, tmax, N)
    delta = 1
    
    # Generate synthetic neural signals
    # Three different oscillatory patterns
    y1 = np.sin(np.pi * 8 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t)**2)
    y2 = np.sin(np.pi * 16 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t)**2)
    y3 = np.sin(np.pi * 8 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t + np.pi/2)**2)
    
    y = np.vstack((y1, y2, y3)).T
    
    # Simulate BOLD signals using hemodynamic model
    print("Simulating BOLD signals...")
    BOLD_signals = Sim(y, 3, dt)
    
    # Filter BOLD signals (bandpass 0.01-0.1 Hz)
    print("Filtering BOLD signals...")
    a0, b0 = signal.bessel(2, [2 * dt * 0.01, 2 * dt * 0.1], btype='bandpass')
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals[60000:, :], axis=0)
    
    # Downsample for deconvolution (simulate TR=1s)
    TR = 1.0
    downsample_factor = int(TR / dt)
    BOLD_downsampled = BOLD_filt[::downsample_factor, 0]
    
    # Deconvolve BOLD signal
    print("Performing deconvolution...")
    results = deconvolve_bold_rsHRF(BOLD_downsampled, TR=TR, hrf_length=32, localK=2, threshold=1.0)
    
    # Plot results in separate figures
    print("Generating plots...")
    fig_hrf, fig_bold = plot_deconvolution_results(results, title="rsHRF BOLD Deconvolution")
    plt.show()
    
    # Print HRF parameters
    print("\n" + "="*50)
    print("HRF Parameters:")
    print("="*50)
    print(f"  Height: {results['hrf_params']['height']:.4f}")
    print(f"  Time to Peak: {results['hrf_params']['time_to_peak']:.2f} s")
    print(f"  FWHM: {results['hrf_params']['fwhm']:.2f} s")
    print(f"  Number of events detected: {len(results['events'])}")
    print("="*50)
    
    # Example: Plot impulse response
    print("\nGenerating impulse response plot...")
    fig_impulse = plot_impulse_response(impulse_time=35, tstop=50, dt=0.01, 
                                        impulse_amplitude=20, TR=1.0, hrf_length=32)
    plt.show()
    
    print("\nAll plots generated successfully!")