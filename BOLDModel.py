#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:18:02 2019

@author: Carlos Coronel

Generalized Hemodynamic Model to reproduce fMRI BOLD-like signals.

[1] Stephan, K. E., Weiskopf, N., Drysdale, P. M., Robinson, P. A., & Friston, K. J. 
(2007). Comparing hemodynamic models with DCM. Neuroimage, 38(3), 387-401.

[2] Deco, Gustavo, et al. "Whole-brain multimodal neuroimaging model using serotonin 
receptor maps explains non-linear functional effects of LSD." Current Biology 
28.19 (2018): 3065-3074.
"""

#%% Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numba import jit,float64
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


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


@jit(float64[:,:](float64[:,:],float64[:],float64), nopython = True)
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


@jit(float64[:,:](float64[:,:],float64[:,:]), nopython = True)    
def BOLD_signal(q, v):
    """
    This function returns the BOLD signal using deoxyhemoglobin content and
    blood volumen as inputs.
    ----------
    Parameters:
    q: numpy array. deoxyhemoglobin content over time.
    v: numpy array. blood volumen over time.
    """
    
    # Herzog et al. (2024) p. 1603 Formula 3.D
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
    pardict={}
    for var in ('taus','tauf','tauo','nu','r0','alpha','epsilon','E0','V0','TE','k1','k2','k3'):
        pardict[var] = eval(var)
        
    return pardict    
    

#%% Plots

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
    plt.subplot(3, 1, 1) # 2 rows, 1 column, first plot
    plt.plot(y1, color='green')
    plt.title("y1")

    plt.subplot(3, 1, 2) # 2 rows, 1 column, second plot
    plt.plot(y2, color='orange')
    plt.title("y2")
    
    plt.subplot(3, 1, 3) # 2 rows, 1 column, second plot
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
    