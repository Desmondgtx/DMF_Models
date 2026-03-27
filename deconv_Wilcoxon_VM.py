# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 22:38:02 2026

@author: yangy
"""

#%% Libraries

import numpy as np
import pandas as pd

import scipy.stats as stats
from scipy import signal
from scipy.sparse import lil_matrix

from nilearn import image, datasets
from nilearn.input_data import NiftiLabelsMasker

# rsHRF
from rsHRF import processing       # rest_filter.rest_IdealFilter (filtrado bandpass)
from rsHRF import parameters       # wgr_get_parameters (extract height, T2P, FWHM)
from rsHRF import basis_functions  # get_basis_function
from rsHRF import utils            # hrf_estimation.compute_hrf (estimates HRF)
from rsHRF import iterative_wiener_deconv  # Wiener deconvolution iterative

# Import MATLAB and set backend
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')


#%% rsHRF Pipeline Function

def rsHRF_estimate_HRF(bold_sig, para, temporal_mask=[], n_jobs=-1, wiener=False):
    """
    Pipeline (replicates fourD_rsHRF.demo_rsHRF):
    1. Preprocessing: z-score + bandpass filter
    2. Event detection: local peaks > threshold
    3. Base functions (canon2dd, gamma, etc.)
    4. Lag estimtation
    5. GLM 
    6. HRF reconstruction = base functions × beta
    7. Parameter Extraction (height, T2P, FWHM)
    8. Wiener deconvolution
    
    ----------
    Parameters
    bold_sig : BOLD Signal (input) (array)
    para : Parameter estimation (dict)
    temporal_mask : Temporal mask for excluding timepoints (opcional) (array)
    n_jobs : Number of cores for paralelization Número de cores para paralelización (-1 = all)
    wiener : True = Iterative Wiener, False = Simple Wiener
    
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
    
    # Deconvolution Filter (fourD_rsHRF.py línea 80)
    bold_sig_deconv = processing.rest_filter.rest_IdealFilter(
        bold_sig, para['TR'], para['passband_deconvolve'])
    
    # HRF estimation filter (fourD_rsHRF.py línea 83)
    bold_sig = processing.rest_filter.rest_IdealFilter(
        bold_sig, para['TR'], para['passband'])
    
    # HRF estimation (fourD_rsHRF.py líneas 89-98)
    # internally use: 
    # (internal functions of get_basis_function)
    #   - wgr_BOLD_event_vector(): Event detection (local peaks)
    #   - get_basis_function(): Generates base functions
    #   - wgr_hrf_fit(): Estimates optimus lag and adjust GLM
    #   - knee.knee_pt(): Select optimus lag
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
    
    # HRF parameter extraction (fourD_rsHRF.py líneas 111)
    PARA = np.zeros((3, nvar))
    for i in range(nvar):
        
        # wgr_get_parameters returns: [height, time_to_peak, fwhm]
        PARA[:, i] = parameters.wgr_get_parameters(hrfa[:, i], para['TR'] / para['T'])

    hrfa_TR = signal.resample_poly(hrfa, 1, para['T']) if para['T'] > 1 else hrfa
    
    # Deconvolution (fourD_rsHRF.py líneas 121)
    data_deconv = np.zeros(bold_sig.shape)
    for i in range(nvar):
        hrf = hrfa_TR[:, i]
        if wiener:
            data_deconv[:, i] = iterative_wiener_deconv.rsHRF_iterative_wiener_deconv(
                bold_sig_deconv[:, i], hrf
            )
        else:
            # Wiener simple deconvolution: s = F^-1{ H* × M / (|H|² + λ) }
            H = np.fft.fft(np.append(hrf, np.zeros(nobs - len(hrf))))
            M = np.fft.fft(bold_sig_deconv[:, i])
            data_deconv[:, i] = np.real(np.fft.ifft(
                H.conj() * M / (H * H.conj() + 0.1 * np.mean(H * H.conj()))
            ))
    
    return {
        'hrfa': hrfa,              # HRF microtime resolution (T bins por TR)
        'hrfa_TR': hrfa_TR,        # HRF resampled to TR
        'event_bold': event_bold,  # Event detection index
        'PARA': PARA,              # [height, time_to_peak, fwhm] × nvoxels
        'bold_sig': bold_sig,      # Filtered BOLD signal
        'data_deconv': data_deconv,# Deconvolution signal
        'para': para               # Parameters
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
        'TR': TR,                              # Repetition Time
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



#%%

# Import data

## Subject 01
subject_01 = image.load_img("C:/Users/yangy/Desktop/DMF_Models/Subjects Medel/sub-02CB_task-restawake_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# Confounds
confounds_tsv_01 = ("C:/Users/yangy/Desktop/DMF_Models/Subjects Medel/sub-02CB_task-restawake_run-1_desc-confounds_timeseries.tsv")
confounds_01 = pd.read_csv(confounds_tsv_01, sep="\t")
confound_cols = ["csf", 
                 "white_matter",
                 "global_signal"]
confounds_selected = confounds_01[confound_cols]

# Atlas
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=100,
    yeo_networks=7,
    resolution_mm=2)

atlas_img = atlas.maps
labels = atlas.labels

# Masker
masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0,
    verbose=1
)

# Final Variable
time_series_01 = masker.fit_transform(subject_01, confounds=confounds_selected)


#%%

## Subject 02
subject_02 = image.load_img("C:/Users/yangy/Desktop/DMF_Models/Subjects Medel/sub-02CB_task-restdeep_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# Confounds
confounds_tsv_02 = ("C:/Users/yangy/Desktop/DMF_Models/Subjects Medel/sub-02CB_task-restdeep_run-1_desc-confounds_timeseries.tsv")
confounds_02 = pd.read_csv(confounds_tsv_02, sep="\t")

# Final Variable
time_series_02 = masker.fit_transform(subject_02, confounds=confounds_selected)


#%% rsHRF between conditions

para = get_default_para(TR=2)


results_1 = rsHRF_estimate_HRF(time_series_01, para) # [height, time_to_peak, fwhm] × areas
results_2 = rsHRF_estimate_HRF(time_series_02, para) # [height, time_to_peak, fwhm] × areas

# Plot difference
plot_hrf(results_1)
plot_hrf(results_2)

plot_deconvolution(results_1)

# Peak (altura)
peak_1 = np.max(results_1['hrfa_TR'][:, 0])
peak_2 = np.max(results_2['hrfa_TR'][:, 0])
print(f"Peak = {peak_1: .4f}")
print(f"Peak = {peak_2: .4f}")

# Time to peak 
t2p_1 = np.argmax(results_1['hrfa_TR'][:, 0]) * para['TR']
t2p_2 = np.argmax(results_2['hrfa_TR'][:, 0]) * para['TR']
print(f"Peak at t={t2p_1: .2f} s")
print(f"Peak at t={t2p_2: .2f} s")



### PRUEBA HACER UN WILCOXON ENTRE PEAK HEIGHT DE CADA AREA CON RESPECTO A SI MISMA
### (TEST ESTADÍSTICO DE 100 (areas en awake) x 100(areas en propofol))

## Wilcoxon Rank

# Save data
# data = pd.DataFrame(results_1['hrfa'])
# data.to_csv('results_1.csv')

# data_2 = pd.DataFrame(results_2['hrfa'])
# data_2.to_csv('results_2.csv')


# Arrays: (timepoints, áreas) = (17, 100)
hrfa_1 = results_1['hrfa_TR']  # Awake
hrfa_2 = results_2['hrfa_TR']  # Propofol

n_areas = hrfa_1.shape[1]
p_values = np.zeros(n_areas)
statistics = np.zeros(n_areas)

for i in range(n_areas):
    stat, p = stats.wilcoxon(hrfa_1[:, i], hrfa_2[:, i])
    statistics[i] = stat
    p_values[i] = p

# Áreas significativas (sin corrección)
sig_areas = np.where(p_values < 0.05)[0]
print(f"Áreas significativas (p < 0.05): {len(sig_areas)}/{n_areas}")
print(f"Índices: {sig_areas}")

# Con corrección Bonferroni
alpha_corrected = 0.05 / n_areas
sig_bonferroni = np.where(p_values < alpha_corrected)[0]
print(f"\nCon Bonferroni (α = {alpha_corrected:.5f}): {len(sig_bonferroni)}")



#%% HRF between different confounds

# Select different confounds
confound_cols_2 = ["csf", 
                 "white_matter",
                 "global_signal",
                 "csf_wm"]
confounds_selected_2 = confounds_01[confound_cols_2]

# Final Variables
time_series_03 = masker.fit_transform(subject_01, confounds=confounds_selected_2)
time_series_04 = masker.fit_transform(subject_01)

# Parameters for estimation
para = get_default_para(TR = 2, estimation = 'canon2dd')

# Get HRF
results_3 = rsHRF_estimate_HRF(time_series_03, para, n_jobs=1) # [height, time_to_peak, fwhm] × areas
results_4 = rsHRF_estimate_HRF(time_series_04, para, n_jobs=1) # [height, time_to_peak, fwhm] × areas

# Plot HRF
plot_hrf(results_3)
plot_hrf(results_4)

# Peak (altura)
peak_1 = np.max(results_3['hrfa_TR'][:, 0]) 
peak_2 = np.max(results_4['hrfa_TR'][:, 0])
print(f"Peak = {peak_1: .4f}")
print(f"Peak = {peak_2: .4f}")


# Time to peak 
t2p_1 = np.argmax(results_3['hrfa_TR'][:, 0]) * para['TR']
t2p_2 = np.argmax(results_4['hrfa_TR'][:, 0]) * para['TR']
print(f"Peak at t = {t2p_1: .2f} s")
print(f"Peak at t = {t2p_2: .2f} s")



#%% Wilcoxon between different confounds


# Arrays: (timepoints, áreas) = (17, 100)
hrfa_3 = results_3['hrfa_TR']  # Awake
hrfa_4 = results_4['hrfa_TR']  # Propofol

n_areas = hrfa_1.shape[1]
p_values = np.zeros(n_areas)
statistics = np.zeros(n_areas)

for i in range(n_areas):
    stat, p = stats.wilcoxon(hrfa_3[:, i], hrfa_4[:, i])
    statistics[i] = stat
    p_values[i] = p

# Áreas significativas (sin corrección)
sig_areas = np.where(p_values < 0.05)[0]
print(f"Áreas significativas (p < 0.05): {len(sig_areas)}/{n_areas}")
print(f"Índices: {sig_areas}")

# Con corrección Bonferroni
alpha_corrected = 0.05 / n_areas
sig_bonferroni = np.where(p_values < alpha_corrected)[0]
print(f"\nCon Bonferroni (α = {alpha_corrected:.5f}): {len(sig_bonferroni)}")



#%% Plot all HRF from every area

# plt.figure()
# plt.plot(results_1['hrfa_TR'])
# plt.xlabel('Time (bins)')
# plt.show()
