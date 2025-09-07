# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:54:54 2020

@author: Carlos Coronel
"""

import numpy as np
import BOLDModel as BD
from scipy import signal
from numba import jit,float64, vectorize,njit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#Network parameters
# SC = np.loadtxt('structural_Deco_AAL.txt')
SC = np.random.uniform(size=(45,45))
SC /= np.mean(np.sum(SC,0))
nnodes = len(SC)

#Simulation parameters
tmax = 780 #time in seconds
dt = 0.001 #integration step in seconds
downsampling = 1000 #BOLD downsampling
downsampling_rates = 10 #Firing rates downsampling


#Model parameters
gE, gI = 310, 615 #slope (gain factors) of excitatory and inhibitory, respectively, input-to-output functions
IthrE, IthrI = 0.403, 0.287 #thresholds current above which the firing rates increase linearly with the input currents
tauNMDA, tauGABA = 0.1, 0.01 #Gating decay time constants
gamma = 0.641 #control NMDA receptors gating time decay
dE, dI = 0.16, 0.087 #onstants determining the shape of the curvature of H around Ith
I0 = 0.382 #The overall effective external input
WE, WI = 1, 0.7 #scales the effective external input
W_plus = 1.4 #weight of recurrent excitation
sigma = 3 #Noise scaling factor
JNMDA = 0.15 #weights all excitatory synaptic couplings
G = 0 #Global coupling


#Synaptic plasticity parameters
target = 3 #target mean firing rate in Hz


@njit
#This function is just for setting the random seed
def set_seed(seed):
    np.random.seed(seed)
    return(seed)

#Input-to-output function (excitatory)
@vectorize([float64(float64,float64,float64,float64)],nopython=True)
def rE(IE,gE,IthrE,dE):
    return(gE * (IE - IthrE) / (1 - np.exp(-dE * gE * (IE - IthrE))))


#Input-to-output function (inhibitory)
@vectorize([float64(float64,float64,float64,float64)],nopython=True)
def rI(II,gI,IthrI,dI):
    return(gI * (II - IthrI) / (1 - np.exp(-dI * gI * (II - IthrI))))

    
#Mean Field Model
# @jit(float64[:,:],float64(float64[:,:],float64[:,:],float64[:]),nopython=True)  
@njit
def mean_field(y,SC,params):
    
    SE, SI = y
    G, WE, WI, W_plus, I0, JNMDA, tauNMDA, tauGABA, gamma, gE, IthrE, dE, gI, IthrI, dI = params
    
    IE_t = WE * I0 + W_plus * JNMDA * SE + G * JNMDA * SC @ SE - JGABA * SI
    II_t = WI * I0 + JNMDA * SE - SI * 1
    
    rE_t = rE(IE_t,gE, IthrE, dE)
    rI_t = rI(II_t,gI, IthrI, dI)
    
    SE_dot = -SE / tauNMDA + (1 - SE) * gamma * rE_t 
    SI_dot = -SI / tauGABA + rI_t
       
    return np.vstack((SE_dot,SI_dot)), rE_t


#Mean Field Model
# @jit(float64[:,:](float64),nopython=True)  
@njit
def Noise(sigma):
    SE_dot = sigma * np.random.normal(0,1,nnodes)
    SI_dot = sigma * np.random.normal(0,1,nnodes)
    
    return(np.vstack((SE_dot,SI_dot)))  


#This recompiles the model functions
def update():
    mean_field.recompile()
    Noise.recompile()


def Sim(verbose = False, return_rates = False, seed = None):
    """
    Run a network simulation with the current parameter values.
    
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.
    
    return_rates : Boolean, optional
        If True, firing rates of excitatory populations were returned, at a sampling rate of (1 / dt / downsampling_rates)
        The default is False.    
        
    seed : Int or None(default)
        If not None, sets the random seed for the simulation noise

    Raises
    ------
    ValueError
        An error raises if the dimensions of SC and the number of nodes
        do not match.

    Returns
    -------
    Y_t : ndarray
        BOLD-like signals for each node.
    Y_t_rates: ndarray
        Firing rates for each node (only if return_rates = True)
    t : TYPE
        Values of time.

    """
    global SC, JGABA
    JGABA = 0.75 * G * SC.sum(axis=0) + 1
    
    #All parameters of the DMF model
    params = np.array([G, WE, WI, W_plus, I0, JNMDA, tauNMDA, tauGABA, gamma, gE, IthrE, dE, gI, IthrI, dI])

    #Setting the random seed
    #It controls the noise and initial conditions
    if seed is not None:
        set_seed(seed)
    
    if SC.shape[0] != SC.shape[1] or SC.shape[0] != nnodes:
        raise ValueError("check SC dimensions (",SC.shape,") and number of nodes (",nnodes,")")
    
    if SC.dtype is not np.dtype('float64'):
        try:
            SC = SC.astype(np.float64)
        except:
            raise TypeError("SC must be of numeric type, preferred float")
        
    #Simulation time
    Nsim = int(tmax / dt)
    timeSim = np.linspace(0,tmax,Nsim)
    #Time after downsampling
    NReal = int(np.ceil(tmax / dt / downsampling))
    timeReal = np.linspace(0,tmax,NReal)
    
    #Initial conditions    
    neural_ic = np.ones((1,nnodes)) * np.array([1,1])[:,None] 
    neural_ic *= np.random.uniform(-0.04,0.04,((2,nnodes)))
    rE = np.zeros(nnodes)
    
    neural_Var = neural_ic
    BOLD_ic = np.ones((1, nnodes)) * np.array([0.1, 1, 1, 1])[:, None]
    BOLD_Var = BOLD_ic #BD.BOLD_response(BOLD_ic, neural_Var[-1,:], 0)
    
    Y_t = np.zeros((len(timeReal),nnodes))  #Matrix to save values (BOLD)
    
    if return_rates == True:
        Y_t_rates = np.zeros((len(timeSim)//downsampling_rates,nnodes)) #Matrix to save values (firing rates)
        # Y_t_rates[-1,:] = neural_Var[-1,:]    
        
        for i in range(0,Nsim):
            if i%downsampling == 0:
                Y_t[i//downsampling] = BD.BOLD_signal(BOLD_Var[[3],:], BOLD_Var[[2],:])
            if i%downsampling_rates == 0:
                Y_t_rates[i//downsampling_rates,:] = rE             
            if i%int(10/dt)==0 and verbose:
                print("%g of %g s"%(int(timeSim[i]),tmax))
            derivs, rE = mean_field(neural_Var, SC, params)
            neural_Var += derivs * dt + Noise(sigma) * np.sqrt(dt)
            BOLD_Var += BD.BOLD_response(BOLD_Var, rE, i) * dt
                    
        return Y_t, Y_t_rates, timeReal
    
    
    else:
        for i in range(1,Nsim):
            if i%downsampling == 0:
                Y_t[i//downsampling,:] = BD.BOLD_signal(BOLD_Var[[3],:], BOLD_Var[[2],:])            
            if i%int(10/dt)==0 and verbose:
                print("%g of %g s"%(int(timeSim[i]),tmax))
            derivs, rE = mean_field(neural_Var, SC, params)
            neural_Var += derivs * dt + Noise(sigma) * np.sqrt(dt)
            BOLD_Var += BD.BOLD_response(BOLD_Var, rE, i) * dt
        return Y_t, timeReal    
    
    
    
        
def ParamsNode():
    pardict={}
    for var in ('gE','gI','IthrE','IthrI','tauNMDA','tauGABA','gamma',
                'dE','dI','I0','WE','WI','W_plus','sigma','JNMDA',
                'target','tau_p'):
        pardict[var]=eval(var)
        
    
 #%%
if __name__=="__main__":
    
    import matplotlib.pyplot as plt    

    G = 1.1
    sigma=0.1
    update()
    BOLD_signals,t = Sim(verbose = True)
    BOLD_signals = BOLD_signals[60:,:]
    BOLD_dt = 1

    
    #Filter the BOLD-like signal between 0.01 and 0.1 Hz
    Fmin, Fmax = 0.01, 0.1
    a0, b0 = signal.bessel(2, [2 * BOLD_dt * Fmin, 2 * BOLD_dt * Fmax], btype = 'bandpass')
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis = 0)        
    BOLD_filt = BOLD_filt[60:660,:]
  
    FC = np.corrcoef(BOLD_filt.T) #Functional Connectivity (FC) matrix
    mean_corr = np.mean(FC)
    print(mean_corr) 

    plt.figure(3)
    plt.clf()
    plt.imshow(FC,vmin=-1,vmax=1,cmap='RdBu') 
    
    plt.figure(4)
    plt.clf()
    plt.plot(BOLD_filt); 

