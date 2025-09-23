# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel

Run a simulation using the Dynamic Mean Field (DMF) model. The output of the model are BOLD-like signals.

"""

#%% Libraries

import numpy as np
from scipy import signal,stats
from scipy.io import savemat
import DMF as DMF
import matplotlib.pyplot as plt
from anarpy.utils.FCDutil import fcd
import BOLDModel as bd
import time


#%%

# import Regularity as Reg

# Simulation parameters
DMF.tmax = 1020 # 1050 #Total simulation time in seconds
DMF.dt = 0.001 #integration step in seconds. Suggestion: don't move
DMF_ratio = 1/DMF.dt   # Tr deseado
decimate = 10
DMF.downsampling = int(DMF_ratio/decimate)
DMF.downsampling_rates = 1
BOLD_dt =  DMF.dt*DMF.downsampling #Inverse of the BOLD sampling rate


# Network parameters
struct = np.loadtxt("SC_opti_25julio.txt")
FCe = np.load("average_90x90FC_HCPchina_symm.npy")
# struct = np.loadtxt('structural_Deco_AAL.txt')
DMF.SC = struct / np.mean(np.sum(struct,0))
DMF.nnodes = len(DMF.SC)


# Model parameters
DMF.G = 1.07        #Global coupling
DMF.sigma = 0.4     # 0.25 #Noise scaling factor


#Update new parameters
DMF.update()

start = time.perf_counter()

# Simulating
BOLD_signals, rates, t = DMF.Sim(verbose = True, return_rates = True)

print(time.perf_counter() - start)


#%%

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


# FCDmat,FCs,shift = fcd.extract_FCD(BOLD_filt,wwidth=200,maxNwindows=500,
#                                    olap=0.9,coldata=True,mode='corr',modeFCD='euclidean')


#%% Plots

bd.itauf = 1 / 0.6
bd.BOLD_response.recompile()

BOLD2 = bd.Sim(rates, DMF.nnodes, 0.001)
BOLD2 = BOLD2[int(120/0.001)::100]
BOLD2_filt = signal.filtfilt(a0, b0, BOLD2, axis = 0)
BOLD2_filt = BOLD2_filt[int(60/BOLD_dt):-int(60/BOLD_dt)]

f2,psd2 = signal.welch(BOLD2,fs=1/BOLD_dt,axis=0,nperseg=1500)

f2_f,psd2_f = signal.welch(BOLD2_filt,fs=1/BOLD_dt,axis=0,nperseg=1500, noverlap=1000)

FC2 = np.corrcoef(BOLD2_filt, rowvar=False)


#%%

ii=(1,2,3,4, 11,12,13,14)

plt.figure(1)
plt.clf()
plt.subplot2grid((2,6),(0,0),colspan=4)
# plt.plot(BOLD_signals[:,::10])
plt.plot(BOLD_signals[:,ii])
# plt.plot(BOLD_signals)

# plt.subplot2grid((3,6),(1,0),colspan=4)
# # plt.plot(BOLD2[9000:-6000,::10])
# plt.plot(BOLD2[:,ii])
t_BOLDf = t[1800:-600]
ax2 = plt.subplot2grid((2,6),(1,0),colspan=4)
# plt.plot(BOLD_filt[:,::10])
ax2.plot(t_BOLDf,BOLD_filt[:,ii])
# ax2.set_xlim((300,350))
# plt.subplot2grid((2,6),(3,0),colspan=4)
# # plt.plot(BOLD2_filt[:,::10])
# plt.plot(BOLD2_filt[:,ii])
ax2.spines[['right', 'top']].set_visible(False)

plt.subplot2grid((2,6),(0,4),colspan=2)
plt.plot(f,psd[:,::10])
plt.xlim((0,1))

# plt.subplot2grid((3,6),(1,4),colspan=2)
# plt.plot(f2,psd2[:,::10])
# plt.xlim((0,1))

plt.subplot2grid((2,6),(1,4),colspan=2)
plt.plot(f_f,psd_f[:,::10])
plt.xlim((0,1))

# plt.subplot2grid((3,6),(3,4),colspan=2)
# plt.plot(f2_f,psd2_f[:,::10])
# plt.xlim((0,1))


plt.tight_layout()



#%%
t_rates = np.linspace(0,t[-1],rates.shape[0])

plt.figure(2)
plt.clf()

# ax1 = plt.subplot(211)
# ax1.plot(t_rates[18000:-6000],rates[18000:-6000,ii])
# ax1.set_xlim((300,350))

# ax1.spines[['right', 'top']].set_visible(False)
# # plt.subplot(212)
# plt.plot(J_t[:,::10])

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

plt.figure(4)
plt.clf()
plt.plot(BOLD_filt[:,::20])
plt.plot(BOLD2_filt[:,::20],'--')


