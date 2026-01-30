# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel

Run a simulation using the Dynamic Mean Field (DMF) model. The output of the model are BOLD-like signals.
"""


#%% Libraries

import numpy as np
from scipy import signal,stats
import matplotlib.pyplot as plt

from anarpy.utils.FCDutil import fcd

import BOLDModel as BD
import DMF as DMF
import deconv as DC

import time


#%%

# Simulation parameters
DMF.tmax = 1020                     # Total simulation time in seconds
DMF.dt = 0.001                      # Integration step in seconds. Suggestion: don't move
DMF_ratio = 1/DMF.dt                # TR
decimate = 10
DMF.downsampling = int(DMF_ratio/decimate)
DMF.downsampling_rates = 1
BOLD_dt =  DMF.dt*DMF.downsampling  # Inverse of the BOLD sampling rate


# Network parameters
# struct = np.loadtxt('structural_Deco_AAL.txt')
struct = np.loadtxt("SC_opti_25julio.txt")
FCe = np.load("average_90x90FC_HCPchina_symm.npy")
DMF.SC = struct / np.mean(np.sum(struct,0))
DMF.nnodes = len(DMF.SC)


#%% BOLD 1

# Model parameters
DMF.G = 1.07        #Global coupling
DMF.sigma = 0.4     # 0.25 #Noise scaling factor

#Update new parameters
DMF.update()

# Counter
start = time.perf_counter()

# Simulating
BOLD_signals, rates, t = DMF.Sim(verbose = True, return_rates = True)
print(time.perf_counter() - start)

BOLD_signals = BOLD_signals[int(120/BOLD_dt):,:]

# Filtering
a0,b0 = signal.bessel(3,[2 * BOLD_dt * 0.01, 2 * BOLD_dt * 0.1], btype = 'bandpass')
BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis = 0)
BOLD_filt = BOLD_filt[int(60/BOLD_dt):-int(60/BOLD_dt)]


# Functional connectivity (FC) matrix - Pearson Matrix Correlation
FC = np.corrcoef(BOLD_filt.T)

## Formula of Pearson's correlation = r(x,y) = cov(x,y) / (σ_x * σ_y)
## where x and y are signals and σ is standard eviation


FC_rates = np.corrcoef(rates[9000:-6000,:].T)
f,psd = signal.welch(BOLD_signals,fs=1/BOLD_dt,axis=0,nperseg=2000, noverlap=1500)
f_f,psd_f = signal.welch(BOLD_filt,fs=1/BOLD_dt,axis=0,nperseg=2000, noverlap=1500)
f_r,psd_rates = signal.welch(rates[90000:-60000],fs=1000,axis=0,nperseg=5000)


FCDmat,FCs,shift = fcd.extract_FCD(BOLD_filt,wwidth=200,maxNwindows=500,
                                   olap=0.9,coldata=True,mode='corr',modeFCD='euclidean')



#%% BOLD 2

# Change Parameters
BD.itauf = 1/0.3
BD.itaus = 1/0.3
BD.itauo = 1/0.3

DMF.G = 2           #Global coupling
DMF.sigma = 0.2     # 0.25 #Noise scaling factor

#Update new parameters
DMF.update()
BD.update()

# Simulating
BOLD2 = BD.Sim(rates, DMF.nnodes, 0.001)
BOLD2 = BOLD2[int(120/0.001)::100]

# Filtering
BOLD2_filt = signal.filtfilt(a0, b0, BOLD2, axis = 0)
BOLD2_filt = BOLD2_filt[int(60/BOLD_dt):-int(60/BOLD_dt)]

# Functional connectivity (FC) matrix
f2,psd2 = signal.welch(BOLD2,fs=1/BOLD_dt,axis=0,nperseg=1500)
f2_f,psd2_f = signal.welch(BOLD2_filt,fs=1/BOLD_dt,axis=0,nperseg=1500, noverlap=1000)
FC2 = np.corrcoef(BOLD2_filt, rowvar=False)


#%% Alternative simulation and processing

# # BOLD 1
# DMF.G = 1.07
# DMF.sigma = 0.4
# DMF.update()
# BOLD_1, rates_1, t = DMF.Sim(verbose=True, return_rates=True)

# # BOLD 2
# DMF.G = 2.0
# DMF.sigma = 0.8
# DMF.update()
# BOLD_2, rates_2, t = DMF.Sim(verbose=True, return_rates=True)

# # Processing
# BOLD_1 = BOLD_1[int(120/BOLD_dt):, :]
# BOLD_2 = BOLD_2[int(120/BOLD_dt):, :]

# BOLD_filt_1 = signal.filtfilt(a0, b0, BOLD_1, axis=0)
# BOLD_filt_2 = signal.filtfilt(a0, b0, BOLD_2, axis=0)

# BOLD_filt_1 = BOLD_filt_1[int(60/BOLD_dt):-int(60/BOLD_dt):10, :]  # TR=1
# BOLD_filt_2 = BOLD_filt_2[int(60/BOLD_dt):-int(60/BOLD_dt):10, :]


#%% Plots

ii=(1,2,3,4, 11,12,13,14)

plt.figure(3)
plt.clf()
plt.subplot2grid((2,6),(0,0),colspan=4)
plt.plot(BOLD_signals[:,ii])

t_BOLDf = t[1800:-600]
ax2 = plt.subplot2grid((2,6),(1,0),colspan=4)
ax2.plot(t_BOLDf,BOLD_filt[:,ii])
ax2.spines[['right', 'top']].set_visible(False)

plt.subplot2grid((2,6),(0,4),colspan=2)
plt.plot(f,psd[:,::10])
plt.xlim((0,1))

plt.subplot2grid((2,6),(1,4),colspan=2)
plt.plot(f_f,psd_f[:,::10])
plt.xlim((0,1))

plt.tight_layout()


#%%

t_rates = np.linspace(0,t[-1],rates.shape[0])

plt.figure(4)
plt.clf()

plt.subplot(221)
# plt.imshow(FCDmat, cmap='jet' )
plt.imshow(struct, cmap='jet' )

plt.subplot(223)
plt.imshow(FCe, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()

plt.subplot(222)
plt.imshow(FC2, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()

plt.subplot(224)
plt.imshow(FC, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()

FCvecE = FCe[np.tril_indices_from(FCe,-1)]
FCvecS = FC[np.tril_indices_from(FC,-1)]

print(f"G {DMF.G}")
print(f"correlacion {stats.pearsonr(FCvecE,FCvecS)[0]}")
Eucl = np.sqrt(np.sum((FCvecE-FCvecS)**2))
print(f"eucilidiana {Eucl}")

# fcd.plotFC(FCs, minmax=[-1,1], cmap='jet', deltaT=0.1*shift)


#%%

# Plot simulated signals
plt.figure(5)
plt.clf()
plt.plot(BOLD_filt[:,::20])
plt.plot(BOLD2_filt[:,::20],'--')

# Subsample
BOLD_filt_TR1 = BOLD_filt[::10, :]  # subsample (TR from 0.1 to 1)
BOLD_filt_TR2 = BOLD2_filt[::10, :]  # subsample (TR from 0.1 to 1)

# Parameters
para = DC.get_default_para(TR = 1, estimation='canon2dd')

# HRF
results_1 = DC.rsHRF_estimate_HRF(BOLD_filt_TR1, para)
results_2 = DC.rsHRF_estimate_HRF(BOLD_filt_TR2, para)

# Plot HRF
DC.plot_hrf(results_1)
DC.plot_hrf(results_2)

# Plot deconvolved signal
DC.plot_deconvolution(results_1)
DC.plot_deconvolution(results_2)

# Peak (height)
peak_1 = np.max(results_1['hrfa_TR'][:, 0])
peak_2 = np.max(results_2['hrfa_TR'][:, 0])
print(f"Peak = {peak_1: .4f}")
print(f"Peak = {peak_2: .4f}")

# Time to peak 
t2p_1 = np.argmax(results_1['hrfa_TR'][:, 0])
t2p_2 = np.argmax(results_2['hrfa_TR'][:, 0])
print(f"Peak at t={t2p_1: .2f} s")
print(f"Peak at t={t2p_2: .2f} s")


