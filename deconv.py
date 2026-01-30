# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 22:38:02 2026

@author: yangy
"""

# Libraries

import numpy as np

import scipy.stats as stats
from scipy import signal
from scipy.sparse import lil_matrix

# rsHRF
from rsHRF import processing       # rest_filter.rest_IdealFilter (filtrado bandpass)
from rsHRF import parameters       # wgr_get_parameters (extract height, T2P, FWHM)
from rsHRF import basis_functions  # get_basis_function
from rsHRF import utils            # hrf_estimation.compute_hrf (estimates HRF)
from rsHRF import iterative_wiener_deconv  # Wiener deconvolution iterative

# Import MATLAB and set backend
import matplotlib.pyplot as plt
plt.switch_backend('QtAgg')


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
        'T': 1,                                # Microtime resolution (bins per TR)
        'T0': 1,                               # Microtime onset (reference bin)
        'AR_lag': 0,                           # AR order for autocorrelation correction
        'thr': 1.0,                            # Threshold for event detection (in std)
        'len': 32,                             # HRF length in seconds
        'min_onset_search': 4,                 # Minimum lag to search (seconds)
        'max_onset_search': 8,                 # Maximum lag to search (seconds)
        'estimation': estimation,              # Estimation method
        'passband': [0.01, 0.1],               # Bandpass for HRF estimation
        'passband_deconvolve': [0.0, np.inf],  # Bandpass for deconvolution
        'TD_DD': 2,                            # 0 = canon only, 1 = +temporal, 2 = +dispersion
        'order': 3,                            # Number of basis vectors (for gamma/fourier)
        'localK': 2                            # Window for local maxima detection
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
    # plt.savefig('estimación_HRF.png', dpi=500)


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
    # plt.savefig('deconvolución.png', dpi=500)

