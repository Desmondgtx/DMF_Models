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
from scipy import signal, stats
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


#%% ============================================================================
# DECONVOLUTION FUNCTIONS
# ============================================================================

def canonical_hrf(t, peak_time=5.4, undershoot_time=15.0, peak_disp=1.0, 
                  undershoot_disp=1.0, peak_undershoot_ratio=6.0):
    """
    Genera HRF canónica doble-gamma (modelo SPM)
    """
    # Primera gamma (peak)
    peak_gamma = stats.gamma.pdf(t, peak_time/peak_disp, scale=peak_disp)
    
    # Segunda gamma (undershoot)
    undershoot_gamma = stats.gamma.pdf(t, undershoot_time/undershoot_disp, 
                                      scale=undershoot_disp)
    
    # Combinar con ratio
    hrf = peak_gamma - undershoot_gamma / peak_undershoot_ratio
    
    # Normalizar
    if np.max(hrf) > 0:
        hrf = hrf / np.max(hrf)
    
    return hrf


def estimate_hrf_parameters(hrf, dt):
    """
    Estima parámetros de la HRF: altura, tiempo al pico, FWHM
    """
    if not np.any(hrf):
        return np.array([0, 0, 0])
    
    # Height
    height = np.max(hrf)
    
    # Time to peak
    peak_idx = np.argmax(hrf)
    time_to_peak = peak_idx * dt
    
    # FWHM (Full Width at Half Maximum)
    half_height = height / 2
    indices_above_half = np.where(hrf >= half_height)[0]
    if len(indices_above_half) > 0:
        fwhm = (indices_above_half[-1] - indices_above_half[0]) * dt
    else:
        fwhm = 0
    
    return np.array([height, time_to_peak, fwhm])


def wiener_deconvolution(signal_data, hrf, noise_level=0.1):
    """
    Deconvolución de Wiener
    """
    N = len(signal_data)
    
    # Pad HRF to signal length
    hrf_padded = np.zeros(N)
    hrf_padded[:len(hrf)] = hrf
    
    # FFT
    H = np.fft.fft(hrf_padded)
    Y = np.fft.fft(signal_data)
    
    # Wiener filter
    Phh = np.abs(H)**2
    noise_power = noise_level * np.mean(Phh)
    
    # Deconvolución
    G = np.conj(H) / (Phh + noise_power)
    deconv_signal = np.real(np.fft.ifft(G * Y))
    
    return deconv_signal


def detect_events(deconv_signal, threshold=2.0, distance=5):
    """
    Detecta eventos en la señal deconvolucionada
    """
    # Z-score
    z_signal = stats.zscore(deconv_signal)
    
    # Encontrar picos
    peaks, properties = signal.find_peaks(z_signal, 
                                         height=threshold,
                                         distance=distance)
    
    return peaks, properties


def deconvolve_bold_signal(bold_signal, TR=1.0, hrf_length=32, noise_level=0.1):
    """
    Deconvolución completa de señal BOLD
    """
    if bold_signal.ndim == 1:
        bold_signal = bold_signal.reshape(-1, 1)
    
    nobs, nregions = bold_signal.shape
    
    print(f"\n{'='*50}")
    print(f"DECONVOLUCIÓN BOLD")
    print(f"{'='*50}")
    print(f"Regiones: {nregions}")
    print(f"Puntos temporales: {nobs}")
    print(f"TR: {TR}s")
    print(f"Nivel de ruido: {noise_level}")
    
    # Crear HRF canónica
    dt_hrf = TR / 3  # Upsampling para mejor resolución
    t_hrf = np.arange(0, hrf_length, dt_hrf)
    hrf_template = canonical_hrf(t_hrf)
    
    # Downsample HRF a TR
    hrf_TR = signal.resample(hrf_template, int(hrf_length/TR))
    
    # Inicializar resultados
    data_deconv = np.zeros_like(bold_signal)
    hrfs = np.zeros((len(hrf_TR), nregions))
    parameters = np.zeros((3, nregions))
    all_events = []
    
    # Procesar cada región
    print("\nDeconvolucionando regiones...")
    for region in range(nregions):
        # Usar HRF template
        hrfs[:, region] = hrf_TR
        
        # Estimar parámetros HRF
        parameters[:, region] = estimate_hrf_parameters(hrf_TR, TR)
        
        # Deconvolución
        data_deconv[:, region] = wiener_deconvolution(
            bold_signal[:, region], 
            hrf_TR, 
            noise_level
        )
        
        # Detectar eventos
        events, _ = detect_events(data_deconv[:, region])
        all_events.append(events)
        
        if (region + 1) % 10 == 0 or region == 0:
            print(f"  Procesadas {region+1}/{nregions} regiones")
    
    print("✓ Deconvolución completada")
    
    # Estadísticas
    print(f"\n📊 ESTADÍSTICAS HRF:")
    print(f"  Height: {np.mean(parameters[0, :]):.3f} ± {np.std(parameters[0, :]):.3f}")
    print(f"  Time to peak: {np.mean(parameters[1, :]):.3f} ± {np.std(parameters[1, :]):.3f}s")
    print(f"  FWHM: {np.mean(parameters[2, :]):.3f} ± {np.std(parameters[2, :]):.3f}s")
    
    total_events = sum(len(e) for e in all_events)
    print(f"\n📊 EVENTOS DETECTADOS:")
    print(f"  Total: {total_events}")
    print(f"  Por región: {total_events/nregions:.1f}")
    
    return {
        'bold_preprocessed': bold_signal,
        'data_deconv': data_deconv,
        'hrfs': hrfs,
        'hrf_parameters': parameters,
        'events': all_events,
        'TR': TR
    }


def plot_hrf_analysis(hrf, TR, save_path=None):
    """
    Plot de la HRF canónica
    """
    time_hrf = np.arange(len(hrf)) * TR
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(time_hrf, hrf, 'g-', linewidth=2.5)
    ax.fill_between(time_hrf, 0, hrf, alpha=0.3, color='green')
    
    # Marcar pico
    peak_idx = np.argmax(hrf)
    ax.axvline(time_hrf[peak_idx], color='red', linestyle='--', 
               alpha=0.7, linewidth=2, label=f'Peak: {time_hrf[peak_idx]:.1f}s')
    
    # Estadísticas
    params = estimate_hrf_parameters(hrf, TR)
    textstr = f'Height: {params[0]:.3f}\nT2P: {params[1]:.2f}s\nFWHM: {params[2]:.2f}s'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Tiempo (s)', fontsize=12)
    ax.set_ylabel('Amplitud', fontsize=12)
    ax.set_title('Función de Respuesta Hemodinámica (canonical)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(32, time_hrf[-1])])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ HRF plot saved: {save_path}")
    
    return fig


def plot_complete_analysis(results, region_idx=None, save_path=None):
    """
    Plot completo con 3 paneles: BOLD original, HRF y Deconvolución
    Similar al segundo gráfico de referencia
    """
    # Seleccionar región con más eventos si no se especifica
    if region_idx is None:
        events_per_region = [len(e) for e in results['events']]
        region_idx = np.argmax(events_per_region)
    
    TR = results['TR']
    nobs = results['bold_preprocessed'].shape[0]
    time = np.arange(nobs) * TR
    
    # Obtener señales
    bold_signal = results['bold_preprocessed'][:, region_idx]
    deconv_signal = results['data_deconv'][:, region_idx]
    hrf = results['hrfs'][:, region_idx]
    events = results['events'][region_idx]
    
    # Normalizar
    bold_norm = stats.zscore(bold_signal)
    deconv_norm = stats.zscore(deconv_signal)
    
    # Crear figura con 3 subplots
    fig = plt.figure(figsize=(14, 10))
    
    # === Panel 1: Señal BOLD Original ===
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time, bold_norm, 'b-', linewidth=1.5, alpha=0.8)
    ax1.fill_between(time, bold_norm, alpha=0.2, color='blue')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Estadísticas
    textstr = f'Media: {np.mean(bold_signal):.3f}\nStd: {np.std(bold_signal):.3f}'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.7))
    
    ax1.set_ylabel('Amplitud (Z-score)', fontsize=11)
    ax1.set_title(f'Análisis de Deconvolución BOLD\nSeñal BOLD Original - Región {region_idx}', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time[0], time[-1]])
    
    # === Panel 2: HRF ===
    ax2 = plt.subplot(3, 1, 2)
    time_hrf = np.arange(len(hrf)) * TR
    ax2.plot(time_hrf, hrf, 'g-', linewidth=2.5)
    ax2.fill_between(time_hrf, 0, hrf, alpha=0.3, color='green')
    
    # Marcar pico
    peak_idx = np.argmax(hrf)
    ax2.axvline(time_hrf[peak_idx], color='red', linestyle='--', 
               alpha=0.7, linewidth=1.5, label=f'Peak: {time_hrf[peak_idx]:.1f}s')
    
    # Parámetros
    params = results['hrf_parameters'][:, region_idx]
    textstr = f'Height: {params[0]:.3f}\nT2P: {params[1]:.2f}s\nFWHM: {params[2]:.2f}s'
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.7))
    
    ax2.set_ylabel('Amplitud', fontsize=11)
    ax2.set_title('Función de Respuesta Hemodinámica (canonical)', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, min(32, time_hrf[-1])])
    
    # === Panel 3: Señal Deconvolucionada ===
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(time, bold_norm, 'b-', linewidth=1.5, alpha=0.5, 
            label='BOLD Original')
    ax3.plot(time, deconv_norm, 'r-', linewidth=1.5, alpha=0.8, 
            label='Deconvolucionada')
    
    # Marcar eventos
    if len(events) > 0:
        ax3.scatter(time[events], deconv_norm[events], 
                   color='orange', s=100, zorder=5, marker='v',
                   edgecolors='black', linewidths=1.5,
                   label=f'Eventos ({len(events)})')
    
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Correlación
    corr = np.corrcoef(bold_norm, deconv_norm)[0, 1]
    textstr = f'Correlación: {corr:.3f}\nEventos: {len(events)}'
    ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.7))
    
    ax3.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Z-score', fontsize=11)
    ax3.set_title('Señal Deconvolucionada', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([time[0], time[-1]])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Complete analysis plot saved: {save_path}")
    
    return fig


def plot_detailed_deconvolution(results, region_idx=None, save_path=None):
    """
    Plot detallado estilo primer gráfico de referencia
    """
    # Seleccionar región con más eventos si no se especifica
    if region_idx is None:
        events_per_region = [len(e) for e in results['events']]
        region_idx = np.argmax(events_per_region)
    
    TR = results['TR']
    nobs = results['bold_preprocessed'].shape[0]
    time = np.arange(nobs) * TR
    
    # Obtener señales
    bold_signal = results['bold_preprocessed'][:, region_idx]
    deconv_signal = results['data_deconv'][:, region_idx]
    events = results['events'][region_idx]
    
    # Normalizar señales (Z-score)
    bold_norm = stats.zscore(bold_signal)
    deconv_norm = stats.zscore(deconv_signal)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot BOLD original (azul)
    ax.plot(time, bold_norm, color='blue', linewidth=1.8, 
            alpha=0.7, label='BOLD')
    
    # Plot señal deconvolucionada (rojo)
    ax.plot(time, deconv_norm, color='red', linewidth=1.8, 
            alpha=0.85, label='Deconvolved BOLD')
    
    # Marcar eventos con triángulos negros
    if len(events) > 0:
        ax.scatter(time[events], deconv_norm[events], 
                   color='black', s=100, zorder=5, 
                   marker='^', label='Events')
    
    # Línea horizontal en cero
    ax.axhline(0, color='gray', linestyle='-', 
               alpha=0.4, linewidth=1)
    
    # Configuración de ejes
    ax.set_xlabel('time (s)', fontsize=13)
    ax.set_ylabel('amplitude (Z-score)', fontsize=13)
    ax.set_title(f'BOLD Deconvolution - Region {region_idx}', 
                 fontsize=15, fontweight='bold')
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([np.min([bold_norm.min(), deconv_norm.min()]) - 0.5, 
                 np.max([bold_norm.max(), deconv_norm.max()]) + 0.5])
    
    # Leyenda
    ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    
    # Quitar bordes superiores y derecho
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Detailed deconvolution plot saved: {save_path}")
    
    return fig


def generate_event_based_firing_rates(dt, tmax, nnodes, event_rate=0.05):
    """
    Genera firing rates con eventos discretos más pronunciados
    para obtener mejor diferencia en la deconvolución
    
    Parameters:
    -----------
    dt : float
        Paso de tiempo
    tmax : float
        Tiempo total de simulación
    nnodes : int
        Número de regiones
    event_rate : float
        Tasa de eventos por segundo (default: 0.05 = 1 evento cada 20s)
    
    Returns:
    --------
    firing_rates : array
        Matriz de firing rates con eventos
    """
    N = int(tmax/dt)
    t = np.linspace(0, tmax, N)
    
    firing_rates = np.zeros((N, nnodes))
    
    for region in range(nnodes):
        # Línea base de actividad baja
        baseline = 0.5 + 0.2 * np.random.randn(N)
        
        # Generar eventos discretos (picos breves)
        n_events = int(tmax * event_rate)
        event_times = np.random.choice(N, size=n_events, replace=False)
        event_times = np.sort(event_times)
        
        # Crear señal de eventos
        events = np.zeros(N)
        for event_time in event_times:
            # Evento con forma gaussiana breve
            event_width = int(0.5 / dt)  # 500ms de duración
            start = max(0, event_time - event_width//2)
            end = min(N, event_time + event_width//2)
            
            # Amplitud variable del evento
            amplitude = np.random.uniform(3, 8)
            gaussian = amplitude * np.exp(-0.5 * ((np.arange(start, end) - event_time) / (event_width/4))**2)
            events[start:end] += gaussian
        
        # Componente oscilatoria lenta
        slow_freq = np.random.uniform(0.01, 0.03)
        oscillation = 0.5 * np.sin(2 * np.pi * slow_freq * t)
        
        # Combinar componentes
        firing_rates[:, region] = np.maximum(0, baseline + events + oscillation)
        
        # Añadir ruido
        firing_rates[:, region] += 0.1 * np.random.randn(N)
        firing_rates[:, region] = np.maximum(0, firing_rates[:, region])
    
    return firing_rates

    
#%% Plots

if __name__=="__main__":
    
    print("\n" + "="*60)
    print(" BOLD MODEL + DECONVOLUTION ANALYSIS")
    print("="*60)
    
    dt = 1E-3
    tmax = 400  # Reducido para mejor visualización
    N = int(tmax/dt)
    t = np.linspace(0, tmax, N)
    nnodes = 3
    
    # Generar firing rates con eventos discretos
    print("\n1. Generating event-based firing rates...")
    y = generate_event_based_firing_rates(dt, tmax, nnodes, event_rate=0.08)
    
    # Simular señal BOLD
    print("2. Simulating BOLD signals...")
    BOLD_signals = Sim(y, nnodes, dt)
    
    # Filtrar señal BOLD
    print("3. Filtering BOLD signals (0.01-0.1 Hz)...")
    a0, b0 = signal.bessel(2, [2 * dt * 0.01, 2 * dt * 0.1], btype='bandpass')
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals[60000:,:], axis=0)
    
    # Normalizar (Z-score)
    BOLD_normalized = stats.zscore(BOLD_filt, axis=0)
    
    # TR efectivo
    TR = 1.0
    
    # Deconvolucionar señal BOLD
    print("\n4. Performing BOLD deconvolution...")
    deconv_results = deconvolve_bold_signal(
        BOLD_normalized,
        TR=TR,
        hrf_length=32,
        noise_level=0.05  # Reducido para mejor deconvolución
    )
    
    # Visualización
    print("\n5. Generating plots...")
    
    # Plot completo (3 paneles)
    fig_complete = plot_complete_analysis(
        deconv_results,
        region_idx=0,
        save_path='complete_analysis.png'
    )
    
    # Plot detallado (estilo primer gráfico)
    fig_detailed = plot_detailed_deconvolution(
        deconv_results,
        region_idx=0,
        save_path='deconvolution_detailed.png'
    )
    
    # Plot HRF individual
    fig_hrf = plot_hrf_analysis(
        deconv_results['hrfs'][:, 0],
        TR,
        save_path='hrf_canonical.png'
    )
    
    # Continuación del bloque if __name__=="__main__":
    
    # Plot original de señales BOLD
    print("\n6. Generating original BOLD signals plot...")
    plt.figure(4, figsize=(12, 6))
    plt.clf()
    plt.plot(BOLD_filt)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('BOLD amplitude', fontsize=12)
    plt.title('Filtered BOLD Signals', fontsize=14, fontweight='bold')
    plt.legend([f'Region {i+1}' for i in range(nnodes)], loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bold_signals.png', dpi=150, bbox_inches='tight')
    
    print("\n" + "="*60)
    print(" ANALYSIS COMPLETED")
    print("="*60)
    print("\n📁 Generated files:")
    print("  - complete_analysis.png (3 paneles)")
    print("  - deconvolution_detailed.png (estilo referencia)")
    print("  - hrf_canonical.png")
    print("  - bold_signals.png")
    
    # Resumen de eventos detectados
    print("\n📊 RESUMEN DE EVENTOS POR REGIÓN:")
    for i, events in enumerate(deconv_results['events']):
        print(f"  Región {i}: {len(events)} eventos detectados")
    
    plt.show()
    
    
    