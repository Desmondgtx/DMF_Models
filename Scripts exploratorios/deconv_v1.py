# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 2025

@author: Diego Garrido

Generalized Hemodynamic Model to reproduce fMRI BOLD-like signals.

[1] Stephan, K. E., Weiskopf, N., Drysdale, P. M., Robinson, P. A., & Friston, K. J. 
(2007). Comparing hemodynamic models with DCM. Neuroimage, 38(3), 387-401.

[2] Deco, Gustavo, et al. "Whole-brain multimodal neuroimaging model using serotonin 
receptor maps explains non-linear functional effects of LSD." Current Biology 
28.19 (2018): 3065-3074.

[3] Wu, G. R., Liao, W., Stramaglia, S., Ding, J. R., Chen, H., & Marinazzo, D. 
(2013). A blind deconvolution approach to recover effective connectivity brain 
networks from resting state fMRI data. Medical image analysis, 17(3), 365-374.
"""

#%% Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.sparse import lil_matrix
from scipy.special import gammaln


#%% Parameters

### FREE PARAMETERS
## BK Hemodynamic Model of BOLD Activit 
## Paper: Neural Mass Modeling (p. 1603)
taus = 0.65     # time constant for signal decay             # Kappa
tauf = 0.41     # time constant for feedback regulation      # Gamma
tauo = 0.98     # time constant for volume and deoxyhemoglobin content change 
                # (blood volume changes is V_n, deoxyhemoglobin content is q_n)
epsilon = 0.5   # ratio of intra and extravascular signal

### FIXED PARAMETERS
nu = 40.3       # frequency offset at the outer surface of the magnetized
                # vessel for fully deoxygenated blood at 1.5 Tesla (s^-1)
r0 = 25         # slope of the relation between the intravascular relaxation 
                # rate and oxygen saturation (s^-1)         
alpha = 0.32    # resistance of the veins; stiffness constant
E0 = 0.4        # resting oxygen extraction fraction
TE = 0.04       # echo time (!!determined by the experiment)
V0 = 0.04       # resting venous blood volume fraction

# Inverse variables
# itauX = inverse of tauX
itaus = 1 / taus # Inverse of Kappa
itauf = 1 / tauf # Inverse of Gamma
itauo = 1 / tauo
ialpha = 1 / alpha

# Kinetics constants (Stephan et al. (2007) p. 391)
k1 = 4.3 * nu * E0 * TE
k2 = epsilon * r0 * E0 * TE
k3 = 1 - epsilon


#%% Original BOLD Model Functions

def update():
    BOLD_response.recompile()
    BOLD_signal.recompile()

def BOLD_response(y, rE, t):
    """
    This function generates a BOLD response using the firing rates rE.
    ----------
    Parameters:
    y : Contains the following variables: (numpy array)
        s: vasodilatory signal. The blood vessel vasodilatation.
        f: blood inflow. Increases with the vasodilatation.
        v: blood volumen. Increases with blood inflow.
        q: deoxyhemoglobin content.
    rE: Firing rates of neural populations/neurons. (numpy array)
    t : Current simulation time point. (float)
    
    E0: resting oxygen extraction fraction (in the papers is the "p" parameter)
    (maybe replace E0 with "t" parameter idk)
    -------
    Returns:
    Numpy array with s, f, v, q derivatives at time t.
    """
    
    s, f, v, q = y
    
    # Deco et al. (2018) p. e5
    s_dot = 1 * rE + 0 - itaus * s - itauf * (f - 1) 
    f_dot = s  
    v_dot = (f - v ** ialpha) * itauo
    q_dot = (f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** ialpha / v) * itauo
    
    # I think this is the exact translation
    # s_dot = 0.5 * rE + 3 - itaus * s - itauf * (f - 1)
    # f_dot = s  
    # v_dot = f - v ** (1 / ialpha) 
    # q_dot = (f * ((1 - E0) ** (1 / f)) / E0 - q * v ** (1 / ialpha) / v) 
    
    
    return(np.vstack((s_dot, f_dot, v_dot, q_dot)))


def BOLD_signal(q, v):
    """
    This function returns the BOLD signal using deoxyhemoglobin content and
    blood volumen as inputs.
    ----------
    Parameters:
    q: numpy array. deoxyhemoglobin content over time.
    v: numpy array. blood volumen over time.
    """
    
    # Stephan et al., 2007 (p. 388) - BOLD signal change equation
    b_dot = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
    
    return b_dot


def Sim(rE, nnodes, dt):
    """
    Simulate the BOLD-like signals (raw non-filtered) with the current parameter values.
    Note that the time unit in this model is seconds.
    ----------
    Parameters:
    rE : time x nodes matrix which values contains the firing rates of each node (numpy array)
    nnodes : number of nodes (integer)
    dt : actual integration step (the inverse of the sampling rate) (float)
    ------
    ValueError:
        An error raises if the number of nodes of rE did not match with the given 
        number by nnodes
    -------
    Returns:
    y : Raw BOLD-like signals for each node (numpy array)
    """
    
    # Don't know where the next equations come from 
    
    Ntotal = rE.shape[0]
    
    ic_BOLD = np.ones((1, nnodes)) * np.array([0.1, 1, 1, 1])[:, None] #initial conditions
    BOLD_vars = np.zeros((Ntotal,4,nnodes)) #matrix for storing the values
    BOLD_vars[0,:,:] = ic_BOLD

    
    #Solve the ODEs with Euler
    for i in range(1,Ntotal):
        BOLD_vars[i,:,:] = BOLD_vars[i - 1,:,:] + dt * BOLD_response(BOLD_vars[i - 1,:,:], rE[i - 1,:], i - 1)
    
    y = BOLD_signal(BOLD_vars[:,3,:], BOLD_vars[:,2,:])
    
    return(y)


def ParamsBOLD():
    """
    Stephan et al. (2007) - Table 1, p. 391
    """
    pardict = {}
    for var in ('taus', 'tauf', 'tauo', 'nu', 'r0', 'alpha', 'epsilon', 'E0', 'V0', 'TE', 'k1', 'k2', 'k3'):
        pardict[var] = eval(var)
    return pardict


#%% rsHRF Functions 

def wgr_BOLD_event_vector(N, matrix, thr, k, temporal_mask=[]):
    """
    Detect BOLD event.
    Reference: hrf_estimation.py lines 110-140
    
    event > thr & event < 3.1
    ----------
    Parameters:
    N : Length of the BOLD signal time series (int)
    matrix : BOLD signal time series (numpy array)
    thr : Threshold for event detection (numpy array)
    k : Local neighborhood size for peak detection (int)
    temporal_mask : Binary mask indicating valid time points (numpy array, optional*)
    ----------
    Returns:
    Sparse matrix with 1s at detected event locations
    
    Wu et al. (2013) Section 2.1, Fig. 1
    """
    data = lil_matrix((1, N))
    matrix = matrix[:, np.newaxis]
    matrix = np.nan_to_num(matrix)
    
    if 0 in np.array(temporal_mask).shape:
        matrix = stats.zscore(matrix, ddof=1)
        for t in range(1 + k, N - k + 1):
            if matrix[t - 1, 0] > thr[0] and \
                    np.all(matrix[t - k - 1:t - 1, 0] < matrix[t - 1, 0]) and \
                    np.all(matrix[t - 1, 0] > matrix[t:t + k, 0]):
                data[0, t - 1] = 1
    else:
        tmp = temporal_mask
        for i in range(len(temporal_mask)):
            if tmp[i] == 1:
                temporal_mask[i] = i
        datm = np.mean(matrix[temporal_mask])
        datstd = np.std(matrix[temporal_mask])
        if datstd == 0: 
            datstd = 1
        matrix = (matrix - datm) / datstd
        
        for t in range(1 + k, N - k + 1):
            if tmp[t - 1]:
                if matrix[t - 1, 0] > thr[0] and \
                        np.all(matrix[t - k - 1:t - 1, 0] < matrix[t - 1, 0]) and \
                        np.all(matrix[t - 1, 0] > matrix[t:t + k, 0]):
                    data[0, t - 1] = 1.
    
    return data



def wgr_get_parameters(hdrf, dt):
    """
    Find Model Parameters
    Reference: parameters.py lines 6-44
    
    h - Height
    p - Time to peak (in units of dt where dt = TR/para.T)
    w - Width at half-peak
    ----------
    Parameters:
    hdrf : Hemodynamic response function (numpy array)
    dt : Time resolution (sampling interval) in seconds (float)
    ----------
    Returns:
    time_to_peak: time to reach peak (seconds)
    """
    param = np.zeros((3, 1))

    if(np.any(hdrf)):
        n = np.fix(np.amax(hdrf.shape) * 0.8)

        p = np.argmax(np.absolute(hdrf[np.arange(0, n, dtype='int')]))
        h = hdrf[p]

        # Calculate FWHM
        if h > 0:
            v = (hdrf >= (h / 2))
        else:
            v = (hdrf <= (h / 2))
        v = v.astype(int)
        b = np.argmin(np.diff(v))
        v[b + 1:] = 0
        w = np.sum(v)

        cnt = p - 1
        g = hdrf[1:] - hdrf[0:-1]

        while cnt > 0 and np.abs(g[cnt]) < 0.001:
            h = hdrf[cnt - 1]
            p = cnt
            cnt = cnt - 1

        param[0] = h
        param[1] = (p + 1) * dt
        param[2] = w * dt

    else:
        print('.')
    
    return param.ravel()


def spm_hrf(RT, P=None, fMRI_T=16):
    """
    Hemodynamic response function (HRF) using SPM canonical form
    Reference: spm.py lines 65-100
    ----------
    Parameters:
    RT : Scan repeat time (TR) in seconds (float)
    P : Parameters of the response function (two gamma functions) (numpy array)
        defaults (seconds):
        P[0] - Delay of Response (relative to onset):   6
        P[1] - Delay of Undershoot (relative to onset): 16
        P[2] - Dispersion of Response:                  1
        P[3] - Dispersion of Undershoot:                1
        P[4] - Ratio of Response to Undershoot:         6
        P[5] - Onset (seconds):                         0
        P[6] - Length of Kernel (seconds):              32
    ----------
    Returns:
    Hemodynamic response function
    """
    p = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
    if P is not None:
        p[0:len(P)] = P
    _spm_Gpdf = lambda x, h, l: \
        np.exp(h * np.log(l) + (h - 1) * np.log(x) - (l * x) - gammaln(h))
    # modelled hemodynamic response function - {mixture of Gammas}
    dt = RT / float(fMRI_T)
    u = np.arange(0, int(p[6] / dt + 1)) - p[5] / dt
    with np.errstate(divide='ignore'):  # Known division-by-zero
        hrf = _spm_Gpdf(
            u, p[0] / p[2], dt / p[2]
        ) - _spm_Gpdf(
            u, p[1] / p[3], dt / p[3]
        ) / p[4]
    idx = np.arange(0, int((p[6] / RT) + 1)) * fMRI_T
    hrf = hrf[idx]
    hrf = np.nan_to_num(hrf)
    hrf = hrf / np.sum(hrf)
    
    return hrf


def wgr_spm_get_canonhrf(xBF):
    """
    Get canonical HRF with optional time and dispersion derivatives.
    Reference: canon_hrf2dd.py lines 7-29
    ----------
    Parameters:
    xBF : Dictionary with keys:
            'dt' - time resolution
            'T' - microtime resolution (fMRI_T)
            'len' - length of kernel in seconds
            'TD_DD' - 0: no derivatives, 1: time derivative, 2: time + dispersion derivatives
    ----------
    Returns:
    Basis functions (canonical HRF and optional derivatives)
    """
    dt     = xBF['dt']
    fMRI_T = xBF['T']
    p      = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
    p[len(p) - 1] = xBF['len']
    bf = spm_hrf(dt, p, fMRI_T)
    bf = bf[:, np.newaxis]
    # time-derivative
    if xBF.get('TD_DD', 0) >= 1:
        dp   = 1
        p[5] = p[5] + dp
        D    = (bf[:, 0] - spm_hrf(dt, p, fMRI_T)) / dp
        D    = D[:, np.newaxis]
        bf   = np.append(bf, D, axis=1)
        p[5] = p[5] - dp
        # dispersion-derivative
        if xBF['TD_DD'] == 2:
            dp   = 0.01
            p[2] = p[2] + dp
            D    = (bf[:, 0] - spm_hrf(dt, p, fMRI_T)) / dp
            D    = D[:, np.newaxis]
            bf   = np.append(bf, D, axis=1)
    
    return bf


def wgr_deconvolution(bold_signal, hrf, regularization=0.1):
    """
    Performs Wiener deconvolution to recover neural signal from BOLD.
    Reference: fourD_rsHRF.py lines 111-119 (non-iterative version)
    ----------
    Parameters:
    bold_signal : Observed BOLD signal (numpy array)
    hrf : Hemodynamic response function (numpy array)
    regularization : Regularization parameter to avoid noise amplification (float)
    ----------
    Returns:
    Deconvolved (estimated neural) signal
    
    Wu et al. (2013) Eq. (4-5), Section 2.1
    """
    N = len(bold_signal)
    nh = max(hrf.shape) if hasattr(hrf, 'shape') and len(hrf.shape) > 1 else len(hrf)

    hrf_padded = np.append(hrf, np.zeros(N - nh))
    
    # Transform to frequency domain
    H = np.fft.fft(hrf_padded, axis=0)
    M = np.fft.fft(bold_signal, axis=0)
    
    # Wiener filter: Wu et al. (2013) Eq. (4)
    # fourD_rsHRF.py: H.conj() * M / (H * H.conj() + .1*np.mean((H * H.conj())))
    data_deconv = np.fft.ifft(H.conj() * M / (H * H.conj() + regularization * np.mean((H * H.conj()))))
    
    return np.real(data_deconv)



#%% Integrated Pipeline Functions

def deconvolve_with_canonical_hrf(bold_signal, para, temporal_mask=[]):
    """
    Main pipeline for BOLD signal deconvolution using rsHRF approach.
    Integrates functions from rsHRF library with original naming.
    ----------
    Parameters:
    bold_signal : Observed BOLD signal (numpy array)
    para : dict
        Parameters dictionary with keys:
        'TR' - repetition time in seconds
        'len' - HRF length in seconds
        'localK' - local neighborhood for event detection
        'thr' - threshold for event detection
        'dt' - time resolution for HRF (typically TR)
        'T' - microtime resolution (fMRI_T)
    temporal_mask : Binary mask for valid time points (default: []) (numpy array)
    ----------
    Returns:
    dict
        Dictionary containing:
        - 'bold_original': original BOLD signal
        - 'bold_normalized': z-scored BOLD signal
        - 'deconvolved': deconvolved signal
        - 'hrf': estimated HRF
        - 'hrf_time': time vector for HRF
        - 'events': indices of detected events
        - 'hrf_params': dictionary with height, time_to_peak, fwhm
        - 'para': parameters used
    
    Reference: Integrates multiple rsHRF functions
    """
    
    # Normalize BOLD signal (from fourD_rsHRF.py line 68)
    bold_normalized = stats.zscore(bold_signal, ddof=1)
    bold_normalized = np.nan_to_num(bold_normalized)
    
    # Detect spontaneous BOLD events (wgr_BOLD_event_vector)
    N = len(bold_normalized)
    thr = [para['thr']] if isinstance(para['thr'], (int, float)) else para['thr']
    events = wgr_BOLD_event_vector(N, bold_normalized, thr, para['localK'], temporal_mask)
    event_indices = events.toarray().flatten('C').ravel().nonzero()[0]
    
    # Generate canonical HRF using rsHRF naming
    xBF = {
        'dt': para['dt'],
        'T': para.get('T', 16),
        'len': para['len'],
        'TD_DD': 0  # No derivatives
    }
    hrf_canonical = wgr_spm_get_canonhrf(xBF)
    hrf_canonical = hrf_canonical[:, 0]  # Extract first column (canonical only)
    
    # Time vector for HRF - must match length of hrf_canonical
    # spm_hrf generates int(len/TR) + 1 points
    n_points = len(hrf_canonical)
    t_hrf = np.linspace(0, para['len'], n_points)
    
    # Perform Wiener deconvolution
    deconvolved_signal = wgr_deconvolution(bold_signal, hrf_canonical)
    
    # Extract HRF parameters
    hrf_params = wgr_get_parameters(hrf_canonical, para['dt'])
    
    # Package results
    results = {
        'bold_original': bold_signal,
        'bold_normalized': bold_normalized,
        'deconvolved': deconvolved_signal,
        'hrf': hrf_canonical,
        'hrf_time': t_hrf,
        'events': event_indices,
        'hrf_params': {
            'height': hrf_params[0],
            'time_to_peak': hrf_params[1],
            'fwhm': hrf_params[2]
        },
        'para': para
    }
    
    return results


#%% Visualization Functions

def plot_hrf_and_deconvolution(results):
    """
    Visualizes HRF and deconvolution results in separate figures.
    Inspired by plotting in fourD_rsHRF.py lines 119-132
    ----------
    Parameters:
    results : Dictionary returned by deconvolve_with_canonical_hrf() (dict)
    ----------
    Returns:
    fig_hrf, fig_bold
    
    Reference: fourD_rsHRF.py lines 180 - 200
    """
    para = results['para']
    TR = para['TR']
    nobs = len(results['bold_original'])
    time = np.arange(nobs) * TR
    
    
    # Figure 1: HRF (similar to fourD_rsHRF.py line 120-124)
    fig_hrf = plt.figure(figsize=(10, 5))
    plt.plot(results['hrf_time'], results['hrf'], 'g-', linewidth=2.5)
    plt.fill_between(results['hrf_time'], results['hrf'], alpha=0.3, color='green')
    peak_idx = np.argmax(results['hrf'])
    plt.axvline(results['hrf_time'][peak_idx], color='red', linestyle='--',label=f'Peak: {results["hrf_time"][peak_idx]:.1f}s')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Hemodynamic Response Function (Canonical)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Figure 2: BOLD signals comparison (similar to fourD_rsHRF.py line 125-132)
    fig_bold = plt.figure(figsize=(12, 6))
    
    plt.plot(time, results['bold_normalized'], 'b-', linewidth=1, label='BOLD (normalized)', alpha=0.7) # HRF of BOLD signal
    plt.plot(time, stats.zscore(results['deconvolved'], ddof=1), 'r-', linewidth=1, label='Deconvolved', alpha=0.7) # HRF of deconvolved signal
    
    # Add detected events as stem plot (from fourD_rsHRF.py line 129-132)
    if len(results['events']) > 0:
        event_plot = np.zeros(nobs)
        event_plot[results['events']] = 1
        markerline, stemlines, baseline = plt.stem(time, event_plot, linefmt='k-', 
                                                    markerfmt='kd', basefmt='k-')
        plt.setp(baseline, linewidth=1)
        plt.setp(stemlines, linewidth=1)
        plt.setp(markerline, markersize=3)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (normalized)')
    plt.title('BOLD Signal vs Deconvolved Signal', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig_hrf, fig_bold


def plot_impulse_response_rsHRF(impulse_time=35, tstop=50, dt=0.01, impulse_amplitude=20, 
                                para=None):
    """
    Plots BOLD impulse response and its deconvolution for a single stimulus event.
    Uses rsHRF naming conventions.
    ----------
    Parameters:
    impulse_time : Time at which impulse occurs in seconds (float)
    tstop : Total simulation time in seconds (float)
    dt : Integration time step in seconds (float)
    impulse_amplitude : Amplitude of the impulse (float)
    para : Parameters for HRF estimation (dict)
    ----------
    Returns:
    Figure object with impulse response plots
    
    """
    
    # Generate time vector and impulse stimulus
    time = np.arange(0, tstop, dt)
    x_t = np.zeros_like(time)
    x_t[time == impulse_time] = impulse_amplitude
    
    # Simulate BOLD response to impulse
    BOLD = Sim(x_t[:, None], 1, dt).flatten()
    
    # Default parameters if not provided
    if para is None:
        TR = 1.0
        para = {
            'TR': TR,
            'len': 32,
            'localK': 1,
            'thr': 0.5,
            'dt': TR,
            'T': 16
        }
    
    # Downsample for deconvolution
    TR = para['TR']
    downsample_factor = int(TR / dt)
    BOLD_downsampled = BOLD[::downsample_factor]
    time_downsampled = time[::downsample_factor]
    
    # Deconvolve BOLD signal using rsHRF functions
    results = deconvolve_with_canonical_hrf(BOLD_downsampled, para)
    
    # Create figure with two y-axes
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(111)
    
    # Plot BOLD signals
    ax1.plot(time, BOLD, 'b-', linewidth=2, label='BOLD Response', alpha=0.8)
    ax1.plot(time_downsampled, results['deconvolved'], 'r-', linewidth=2, label='Deconvolved Signal', alpha=0.8)
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
    
    plt.title('BOLD Impulse Response and Deconvolution (rsHRF)', fontsize=14, fontweight='bold')
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
    BOLD_signals = Sim(y, 3, dt)
    
    # Filter BOLD signals (bandpass 0.01-0.1 Hz)
    a0, b0 = signal.bessel(2, [2 * dt * 0.01, 2 * dt * 0.1], btype='bandpass')
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals[60000:, :], axis=0)
    
    # Downsample for deconvolution (simulate TR=1s)
    TR = 1.0
    downsample_factor = int(TR / dt)
    BOLD_downsampled = BOLD_filt[::downsample_factor, 0]
    
    # Setup parameters using rsHRF convention
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
    
    # Deconvolve BOLD signal using rsHRF naming
    results = deconvolve_with_canonical_hrf(BOLD_downsampled, para)
    
    # Recover HRF from original BOLD signal (without deconvolution)
    results_original = deconvolve_with_canonical_hrf(results['deconvolved'], para)

    
    # Plot HRF from original signal
    fig_hrf_original = plt.figure(figsize=(10, 5))
    plt.plot(results_original['hrf_time'], results_original['hrf'], 'b-', linewidth=2.5)
    plt.fill_between(results_original['hrf_time'], results_original['hrf'], alpha=0.3, color='blue')
    peak_idx_orig = np.argmax(results_original['hrf'])
    plt.axvline(results_original['hrf_time'][peak_idx_orig], color='red', linestyle='--', label=f'Peak: {results_original["hrf_time"][peak_idx_orig]:.1f}s')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('HRF from Original BOLD Signal', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot comparison of both HRFs
    fig_comparison = plt.figure(figsize=(12, 6))
    plt.plot(results_original['hrf_time'], results_original['hrf'], 'b-', linewidth=2.5, label='HRF from Original BOLD', alpha=0.8)
    plt.plot(results['hrf_time'], results['hrf'], 'g-', linewidth=2.5, label='HRF from Deconvolved', alpha=0.8)
    
    # Mark peaks for both HRFs
    peak_idx_orig = np.argmax(results_original['hrf'])
    peak_idx_deconv = np.argmax(results['hrf'])
    plt.axvline(results_original['hrf_time'][peak_idx_orig], color='blue', linestyle='--', 
                alpha=0.7, label=f'Original Peak: {results_original["hrf_time"][peak_idx_orig]:.1f}s')
    plt.axvline(results['hrf_time'][peak_idx_deconv], color='green', linestyle='--', 
                alpha=0.7, label=f'Deconvolved Peak: {results["hrf_time"][peak_idx_deconv]:.1f}s')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('HRF Comparison: Original vs Deconvolved BOLD Signal', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot results in separate figures
    fig_hrf, fig_bold = plot_hrf_and_deconvolution(results)
    plt.show()
    
    # Print HRF parameters
    print("HRF Parameters (wgr_get_parameters):")
    print(f"\n  Height: {results['hrf_params']['height']:.4f}")
    print(f"  Time to Peak: {results['hrf_params']['time_to_peak']:.2f} s")
    print(f"  FWHM: {results['hrf_params']['fwhm']:.2f} s")
    print(f"  Number of events detected: {len(results['events'])}")
    
    # Example: Plot impulse response using rsHRF functions
    para_impulse = {
        'TR': 1.0,
        'len': 32,
        'localK': 1,
        'thr': 0.5,
        'dt': 1.0,
        'T': 16
    }
    fig_impulse = plot_impulse_response_rsHRF(impulse_time=35, tstop=50, dt=0.01, impulse_amplitude=20, para=para_impulse)
    plt.show()
    