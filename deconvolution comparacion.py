# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 18:25:36 2025

@author: Diego Garrido

Generalized Hemodynamic Model to reproduce fMRI BOLD-like signals.

Referencias:
[1] Stephan et al. (2007). Comparing hemodynamic models with DCM. Neuroimage.
[2] Deco et al. (2018). Whole-brain multimodal neuroimaging model. Current Biology.
[3] Wu et al. (2013). A blind deconvolution approach. Medical Image Analysis.
"""

#%% Libraries

import numpy as np
from scipy import signal, stats
from scipy.sparse import lil_matrix

# rsHRF
# from rsHRF import spm_dep          # Funciones SPM (spm_hrf para HRF canónica)
from rsHRF import processing       # rest_filter.rest_IdealFilter (filtrado bandpass)
from rsHRF import parameters       # wgr_get_parameters (extrae height, T2P, FWHM)
from rsHRF import basis_functions  # get_basis_function (genera funciones base)
from rsHRF import utils            # hrf_estimation.compute_hrf (estima HRF)
from rsHRF import iterative_wiener_deconv  # Deconvolución Wiener iterativa

# Import MATLAB and set backend
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

#%% Parameters

### FREE PARAMETERS
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
itaus = 1 / taus # Inverse of Kappa
itauf = 1 / tauf # Inverse of Gamma
itauo = 1 / tauo
ialpha = 1 / alpha


# Kinetics constants (Stephan et al. (2007) p. 391)
k1 = 4.3 * nu * E0 * TE
k2 = epsilon * r0 * E0 * TE
k3 = 1 - epsilon


#%% BOLD model functions

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


#%% rsHRF Pipeline Function

def rsHRF_estimate_HRF(bold_sig, para, temporal_mask=[], n_jobs=-1, wiener=False):
    """
    Pipeline completo (replica fourD_rsHRF.demo_rsHRF):
    1. Preprocesamiento: z-score + filtrado bandpass
    2. Detección de eventos: picos locales > threshold
    3. Generación de funciones base (canon2dd, gamma, etc.)
    4. Estimación lag óptimo
    5. Ajuste GLM 
    6. Reconstrucción HRF = funciones_base × beta
    7. Extracción parámetros (height, T2P, FWHM)
    8. Deconvolución Wiener
    
    ----------
    Parameters
    bold_sig : Señal BOLD de entrada (array)
    para : Parámetros de estimación (dict)
    temporal_mask : Máscara temporal para excluir timepoints (array, opcional)
    n_jobs : Número de cores para paralelización (-1 = todos)
    wiener : True = Wiener iterativo, False = Wiener simple
    
    -------
    Returns
    dict con: hrfa, hrfa_TR, event_bold, PARA, bold_sig, data_deconv, para
    """
    
    # Ensure 2D: (nobs, nvoxels)
    if bold_sig.ndim == 1:
        bold_sig = bold_sig[:, np.newaxis]
    nobs, nvar = bold_sig.shape
    
    
    # for time-series input(fourD_rsHRF.py línea 68)
    # bold_sig = np.nan_to_num(stats.zscore(bold_sig, ddof=1, axis=0))
    
    # Filtro para deconvolución (fourD_rsHRF.py línea 80)
    bold_sig_deconv = processing.rest_filter.rest_IdealFilter(
        bold_sig, para['TR'], para['passband_deconvolve'])
    
    # Filtro para estimación HRF (fourD_rsHRF.py línea 83)
    bold_sig = processing.rest_filter.rest_IdealFilter(
        bold_sig, para['TR'], para['passband'])
    
    # Estimación HRF (fourD_rsHRF.py líneas 89-98)
    # Internamente utiliza: 
    # (algunas funciones no se invocan directamente si no através de get_basis_function)
    #   - wgr_BOLD_event_vector(): Detecta eventos (picos locales)
    #   - get_basis_function(): Genera funciones base
    #   - wgr_hrf_fit(): Estima lag óptimo + ajusta GLM
    #   - knee.knee_pt(): Selecciona lag óptimo
    if para['estimation'] not in ['sFIR', 'FIR']:
        bf = basis_functions.basis_functions.get_basis_function(bold_sig.shape, para)
        beta_hrf, event_bold = utils.hrf_estimation.compute_hrf(
            bold_sig, para, temporal_mask, n_jobs, bf=bf)
        hrfa = np.dot(bf, beta_hrf[np.arange(0, bf.shape[1]), :])
    else:
        para['T'] = 1
        beta_hrf, event_bold = utils.hrf_estimation.compute_hrf(
            bold_sig, para, temporal_mask, n_jobs)
        hrfa = beta_hrf[:-1, :]
    
    
    # Extracción de parámetros HRF (fourD_rsHRF.py líneas 111)
    PARA = np.zeros((3, nvar))
    for i in range(nvar):
        
        # wgr_get_parameters retorna: [height, time_to_peak, fwhm]
        PARA[:, i] = parameters.wgr_get_parameters(hrfa[:, i], para['TR'] / para['T'])

    hrfa_TR = signal.resample_poly(hrfa, 1, para['T']) if para['T'] > 1 else hrfa
    
    # Deconvolución (fourD_rsHRF.py líneas 121)
    data_deconv = np.zeros(bold_sig.shape)
    for i in range(nvar):
        hrf = hrfa_TR[:, i]
        if wiener:
            data_deconv[:, i] = iterative_wiener_deconv.rsHRF_iterative_wiener_deconv(
                bold_sig_deconv[:, i], hrf
            )
        else:
            # Deconvolución Wiener simple: s = F^-1{ H* × M / (|H|² + λ) }
            H = np.fft.fft(np.append(hrf, np.zeros(nobs - len(hrf))))
            M = np.fft.fft(bold_sig_deconv[:, i])
            data_deconv[:, i] = np.real(np.fft.ifft(
                H.conj() * M / (H * H.conj() + 0.1 * np.mean(H * H.conj()))
            ))
    
    return {
        'hrfa': hrfa,              # HRF a resolución microtime (T bins por TR)
        'hrfa_TR': hrfa_TR,        # HRF resampleada a TR
        'event_bold': event_bold,  # Índices de eventos detectados
        'PARA': PARA,              # [height, time_to_peak, fwhm] × nvoxels
        'bold_sig': bold_sig,      # Señal BOLD filtrada
        'data_deconv': data_deconv,# Señal deconvolucionada
        'para': para
    }


def get_default_para(TR, estimation='canon2dd'):
    """
    Get default parameters for rsHRF estimation.
    
    -------
    Parameters
    TR : Repetition time in seconds (float)
    estimation :
        Estimation method:
        - 'canon2dd': Canonical HRF + temporal + dispersion derivatives (DEFAULT)
        - 'sFIR': Smoothed Finite Impulse Response
        - 'FIR': Finite Impulse Response
        - 'gamma': Gamma basis functions
        - 'fourier': Fourier basis set
        - 'hanning': Fourier with Hanning window
        
    -------
    Returns
    dict con parametros
    """
    para = {
        'TR': TR,
        'T': 1,                    # Microtime resolution (bins per TR)
        'T0': 1,                   # Microtime onset (reference bin)
        'AR_lag': 0,               # AR order for autocorrelation correction
        'thr': 1.0,                # Threshold for event detection (in std)
        'len': 32,                 # HRF length in seconds
        'min_onset_search': 4,     # Minimum lag to search (seconds)
        'max_onset_search': 8,     # Maximum lag to search (seconds)
        'estimation': estimation,
        'passband': [0.01, 0.1],           # Bandpass for HRF estimation
        'passband_deconvolve': [0.0, np.inf],  # Bandpass for deconvolution
        'TD_DD': 2,                # 0=canon only, 1=+temporal, 2=+dispersion
        'order': 3,                # Number of basis vectors (for gamma/fourier)
        'localK': 2  # Window for local maxima detection
    }
    para['dt'] = TR / para['T']
    para['lag'] = np.arange(
        np.fix(para['min_onset_search'] / para['dt']),
        np.fix(para['max_onset_search'] / para['dt']) + 1,
        dtype='int'
    )
    return para


#%% Visualization Functions (fourD_rsHRF.py líneas 180-200)

def plot_hrf(results, pos=0):
    """Plot estimated HRF."""
    para, hrfa_TR = results['para'], results['hrfa_TR']
    while pos < hrfa_TR.shape[1] and not np.any(hrfa_TR[:, pos]):
        pos += 1
        
    plt.figure(1)
    plt.plot(para['TR'] * np.arange(1, hrfa_TR.shape[0] + 1), hrfa_TR[:, pos], linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Estimated HRF')
    plt.savefig('estimación_HRF.png', dpi=500)


def plot_deconvolution(results, pos=0):
    """Plot BOLD vs deconvolved signal with detected events."""
    para, hrfa_TR = results['para'], results['hrfa_TR']
    bold_sig, data_deconv, event_bold = results['bold_sig'], results['data_deconv'], results['event_bold']
    nobs = bold_sig.shape[0]
    while pos < hrfa_TR.shape[1] and not np.any(hrfa_TR[:, pos]):
        pos += 1
    event_plot = lil_matrix((1, nobs))
    if event_bold.size and len(event_bold[pos]) > 0:
        event_plot[:, event_bold[pos]] = 1
    event_plot = np.ravel(event_plot.toarray())
    time = para['TR'] * np.arange(1, nobs + 1)
    
    plt.figure(2)
    plt.plot(time, np.nan_to_num(stats.zscore(bold_sig[:, pos], ddof=1)), linewidth=1) # señal original
    plt.plot(time, np.nan_to_num(stats.zscore(data_deconv[:, pos], ddof=1)), color='r', linewidth=1) # deconv
    markerline, stemlines, baseline = plt.stem(time, event_plot)
    plt.setp(baseline, 'color', 'k', 'markersize', 1)
    plt.setp(stemlines, 'color', 'k')
    plt.setp(markerline, 'color', 'k', 'markersize', 3, 'marker', 'd')
    plt.legend(['BOLD', 'Deconvolved BOLD', 'Events'], loc='best')
    plt.xlabel('time (s)')
    plt.savefig('deconvolución.png', dpi=500)


#%% 

if __name__ == "__main__":
    
    # Señal simulada
    # Parámetros
    dt = 1E-3
    tmax = 800
    TR = 1   
    
    
    # #%% Comparar con MATLAB
    # import scipy.io as sio
    
    # # Cargar datos y ejecutar Python
    # bold_data = np.loadtxt('sub-01_bold.txt', delimiter='\t')
    # para = get_default_para(TR=1, estimation='canon2dd')
    # results = rsHRF_estimate_HRF(bold_data, para, n_jobs=1)
    
    # # Cargar MATLAB
    # mat = sio.loadmat('matlab_hrf_extracted_canon2dd.mat')    
    # hrfa_matlab = mat['hrfa']
    # PARA_matlab = mat['PARA']
    
    # # Comparar
    # hrfa_python = results['hrfa_TR']
    # PARA_python = results['PARA']
    # n_rois = hrfa_matlab.shape[1]
    
    # # Igualar longitudes
    # n_bins = min(hrfa_matlab.shape[0], hrfa_python.shape[0])
    # hrf_corrs = [np.corrcoef(hrfa_matlab[:n_bins, i], hrfa_python[:n_bins, i])[0,1] 
    #              for i in range(n_rois)]
    
    # # Plot
    # plt.plot(hrfa_python[:, 0], 'b-', label='Python')
    # plt.plot(hrfa_matlab[:, 0], 'r-', label='MATLAB')
    # plt.tight_layout()
    # plt.show()
    
    
    
    #%% Comparación MATLAB vs Python
    
    import scipy.io as sio
    
    # HRF estimada desde MATLAB
    subject_01 = sio.loadmat('Results/hrf_subject_01.mat')
    subject_02 = sio.loadmat('Results/hrf_subject_02.mat')
    subject_03 = sio.loadmat('Results/hrf_subject_03.mat')
    subject_04 = sio.loadmat('Results/hrf_subject_04.mat')
    subject_05 = sio.loadmat('Results/hrf_subject_05.mat')
    
    hrfa_s01 = subject_01['hrfa']
    hrfa_s02 = subject_02['hrfa']
    hrfa_s03 = subject_03['hrfa']
    hrfa_s04 = subject_04['hrfa']
    hrfa_s05 = subject_05['hrfa']
    
    # Realizar la deconvolución desde Python
    s_01 = np.loadtxt('Subjects/sub-01_bold.txt', delimiter='\t')
    s_02 = np.loadtxt('Subjects/sub-02_bold.txt', delimiter='\t')
    s_03 = np.loadtxt('Subjects/sub-03_bold.txt', delimiter='\t')
    s_04 = np.loadtxt('Subjects/sub-04_bold.txt', delimiter='\t')
    s_05 = np.loadtxt('Subjects/sub-05_bold.txt', delimiter='\t')
    
    para = get_default_para(TR=1, estimation='canon2dd')
    
    results_01 = rsHRF_estimate_HRF(s_01, para, n_jobs=1)
    results_02 = rsHRF_estimate_HRF(s_02, para, n_jobs=1)
    results_03 = rsHRF_estimate_HRF(s_03, para, n_jobs=1)
    results_04 = rsHRF_estimate_HRF(s_04, para, n_jobs=1)
    results_05 = rsHRF_estimate_HRF(s_05, para, n_jobs=1)
    
    hrfa_python_01 = results_01['hrfa_TR']
    hrfa_python_02 = results_02['hrfa_TR']
    hrfa_python_03 = results_03['hrfa_TR']
    hrfa_python_04 = results_04['hrfa_TR']
    hrfa_python_05 = results_05['hrfa_TR']
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MATLAB
    axes[0].plot(hrfa_s01[:, 0], 'r-', label='Subject 01')
    axes[0].plot(hrfa_s02[:, 0], 'b-', label='Subject 02')
    axes[0].plot(hrfa_s03[:, 0], 'g-', label='Subject 03')
    axes[0].plot(hrfa_s04[:, 0], 'c-', label='Subject 04')
    axes[0].plot(hrfa_s05[:, 0], 'm-', label='Subject 05')
    axes[0].set_xlabel('Time (bins)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('MATLAB')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Python
    axes[1].plot(hrfa_python_01[:, 0], 'r-', label='Subject 01')
    axes[1].plot(hrfa_python_02[:, 0], 'b-', label='Subject 02')
    axes[1].plot(hrfa_python_03[:, 0], 'g-', label='Subject 03')
    axes[1].plot(hrfa_python_04[:, 0], 'c-', label='Subject 04')
    axes[1].plot(hrfa_python_05[:, 0], 'm-', label='Subject 05')
    axes[1].set_xlabel('Time (bins)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Python')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('HRF Estimation Comparison: MATLAB vs Python', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparacion_matlab_python.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
