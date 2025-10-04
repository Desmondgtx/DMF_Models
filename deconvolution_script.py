# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 22:14:19 2025

@author: Diego Garrido
"""

#%% Librerías
import numpy as np
from scipy import signal as sp_signal

def deconvolve_bold(bold_data, TR=1.0):
    """
    Deconvolución simple de señales BOLD
    """
    # 1. Crear HRF canónica
    t = np.arange(0, 30, TR)
    hrf = np.exp(-t/1.5) * (t**2)  # HRF simplificada
    hrf = hrf / np.sum(hrf)
    
    # 2. Deconvolucionar cada voxel/nodo
    n_timepoints, n_nodes = bold_data.shape
    deconv_data = np.zeros_like(bold_data)
    
    for node in range(n_nodes):
        # Método 1: Deconvolución directa
        deconv, remainder = sp_signal.deconvolve(
            bold_data[:, node], 
            hrf[:min(len(hrf), 20)]
        )
        deconv_data[:len(deconv), node] = deconv
        
    return deconv_data, hrf

# Usar con tus datos
deconv_signals, estimated_hrf = deconvolve_bold(BOLD_filt, BOLD_dt)