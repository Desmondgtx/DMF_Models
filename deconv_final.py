#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:18:02 2019

@author: Carlos Coronel

Generalized Hemodynamic Model to reproduce fMRI BOLD-like signals.

[1] Stephan, K. E., et al. "Comparing hemodynamic models with DCM." 
Neuroimage, 38(3), (2007): 387-401.

[2] Deco, Gustavo, et al. "Whole-brain multimodal neuroimaging model using serotonin 
receptor maps explains non-linear functional effects of LSD." Current Biology 
28.19, (2018): 3065-3074.

^ Wu, Guo-Rong, et al. "A blind deconvolution apporach to recover effective connectivity
brain networks from resting state fMRI data" Medical Image Analysis, 17(3), (2018): 365–374.
"""

#%% Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


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


#%% Functions

def update():
    BOLD_response.recompile()
    BOLD_signal.recompile()


# @jit(float64[:,:](float64[:,:],float64[:],float64), nopython = True)
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
    Numpy array with s, f, v and q derivatives at time t.
    """
    
    s, f, v, q = y
    
    # Deco et al. (2018) p. e5
    s_dot = 1 * rE + 0 - itaus * s - itauf * (f - 1) 
    f_dot = s  
    v_dot = (f - v ** ialpha) * itauo
    q_dot = (f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** ialpha / v) * itauo
    
    # s_dot = 0.5 * rE + 3 - itaus * s - itauf * (f - 1)
    # f_dot = s  
    # v_dot = f - v ** (1 / ialpha) 
    # q_dot = (f * ((1 - E0) ** (1 / f)) / E0 - q * v ** (1 / ialpha) / v) 
    
    
    return(np.vstack((s_dot, f_dot, v_dot, q_dot)))


# @jit(float64[:,:](float64[:,:],float64[:,:]), nopython = True)    
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
    
    return(b_dot)


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
    
    # Don't know where this equations came from 
    
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
    pardict={}
    for var in ('taus','tauf','tauo','nu','r0','alpha','epsilon','E0','V0','TE','k1','k2','k3'):
        pardict[var] = eval(var)
        
    return pardict    


#%% Deconvolution Functions

def deconvolve_bold_rsHRF(bold_signal, TR=1.0, T=32, T0=1):
    """
    Deconvolución BOLD usando rsHRF
    
    Parameters:
    -----------
    bold_signal : array (time x regions)
        Señal BOLD a deconvolucionar
    TR : float
        Tiempo de repetición (sampling rate)
    T : int
        Longitud temporal de la HRF en segundos
    T0 : int
        Onset temporal de la HRF
        
    Returns:
    --------
    results : dict
        Diccionario con:
        - 'bold_preprocessed': señal BOLD original
        - 'data_deconv': señal deconvolucionada
        - 'hrfs': funciones de respuesta hemodinámica estimadas
        - 'events': índices de eventos detectados
        - 'TR': tiempo de repetición
    """
    
    nobs, nregions = bold_signal.shape
    
    # Parámetros rsHRF
    para = {}
    para['TR'] = TR
    para['T'] = T  # Longitud HRF en segundos
    para['T0'] = T0  # Onset
    para['dt'] = TR
    
    # Inicializar variables
    data_deconv = np.zeros_like(bold_signal)
    hrfs = []
    events = []
    
    # Procesar cada región
    for region in range(nregions):
        signal_region = bold_signal[:, region]
 
        # Fallback: usar señal original
        data_deconv[:, region] = signal_region
        hrfs.append(None)
        events.append([])
        
        
    return {
        'bold_preprocessed': bold_signal,
        'data_deconv': data_deconv,
        'hrfs': hrfs,
        'events': events,
        'TR': TR
    }

def canon_hrf(t, peak=4.0, undershoot=16.0):
    """
    Generate canonical HRF using difference of gamma functions.
    
    Parameters:
    -----------
    t : array
        Time vector in seconds
    peak : float
        Time to peak (not used directly, implicit in parameters)
    undershoot : float
        Time to undershoot (not used directly, implicit in parameters)
    
    Returns:
    --------
    hrf : array
    """
    # Parameters for double-gamma HRF
    a1, a2 = 6, 12
    b1, b2 = 0.9, 0.9
    c = 0.35
    
    d1 = a1 * b1
    d2 = a2 * b2
    
    hrf = (t / d1) ** a1 * np.exp(-(t - d1) / b1) - c * (t / d2) ** a2 * np.exp(-(t - d2) / b2)
    hrf = hrf / np.max(hrf)  # Normalize
    
    return hrf


#%% Main Execution
    
if __name__=="__main__":
    
    dt = 1E-3
    tmax = 600
    N = int(tmax/dt)
    t = np.linspace(0,tmax,N)
    delta = 1
    
    y1 = np.sin(np.pi * 8 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t)**2)
    y2 = np.sin(np.pi * 16 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t)**2)
    y3 = np.sin(np.pi * 8 * t)**2 * np.exp(-delta * np.cos(np.pi * 0.05 * t + np.pi/2)**2)
    
    ## Everything on the same plot
    # plt.figure(1)
    # plt.clf()
    # plt.plot(y1)
    # plt.plot(y2)
    # plt.plot(y3)
    
    ## Subplots
    plt.subplot(3, 1, 1) # 3 rows, 1 column, first plot
    plt.plot(y1, color='green')
    plt.title("y1")

    plt.subplot(3, 1, 2) # 3 rows, 1 column, second plot
    plt.plot(y2, color='orange')
    plt.title("y2")
    
    plt.subplot(3, 1, 3) # 3 rows, 1 column, third plot
    plt.plot(y3, color='purple')
    plt.title("y3")

    plt.tight_layout()
    plt.show()
    
    
    y = np.vstack((y1,y2,y3)).T
    
    BOLD_signals = Sim(y, 3, dt)
    a0, b0 = signal.bessel(2, [2 * dt * 0.01, 2 * dt * 0.1], btype = 'bandpass')
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals[60000:,:], axis = 0)
    
    plt.figure(2)
    plt.clf()
    plt.plot(BOLD_filt)
    plt.show()
    
    # Plot canonical HRF
    plt.figure(3)
    plt.clf()
    hrf_time = np.linspace(0, 32, 1000)
    hrf = canon_hrf(hrf_time)
    peak_idx = np.argmax(hrf)
    
    plt.plot(hrf_time, hrf, 'g-', linewidth=2.5)
    plt.fill_between(hrf_time, hrf, alpha=0.3, color='green')
    plt.axvline(hrf_time[peak_idx], color='red', label=f'Peak: {hrf_time[peak_idx]:.1f}s')
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('Hemodynamic Response Function (canonical)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    # Deconvolution
    deconvolved = deconvolve_bold_rsHRF(BOLD_signals, TR=1.0, T=32, T0=1)
    data_deconv = deconvolved["data_deconv"]


#%% Bold Impulse

nnodes = 1
dt = 0.01
tstop = 50
time=np.arange(0,tstop,dt)

x_t = np.zeros_like(time)
x_t[time==30]=20

BOLD = Sim(x_t[:,None], nnodes, dt) # Sim(re, nnodes, dt)

deconvolved = deconvolve_bold_rsHRF(BOLD, TR=1.0, T=32, T0=1)
deconvolved_signal = deconvolved["data_deconv"]


plt.clf()
ax1 = plt.subplot(111)
ax1.plot(time,deconvolved_signal)

ax2 = ax1.twinx()
ax2.plot(time,x_t, 'g')
ax2.set_ylim((-10,250))
ax1.set_ylim((-0.0005,0.0025))
plt.xlabel("Tiempo (s)")
ax1.set_ylabel("Señal BOLD", color='C0')

plt.tight_layout()