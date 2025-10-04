#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rsHRF Core Functions Implementation
Funciones esenciales para estimación y deconvolución HRF
Compatible con la arquitectura de clases existente
"""

import numpy as np
from scipy import signal, linalg, stats
from scipy.special import gammaln
from scipy.optimize import minimize
import warnings

# ============================================================================
# ESTIMACIÓN HRF
# ============================================================================

def canonical_hrf(dt=0.5, length=32):
    """
    Genera HRF canónica (modelo SPM)
    
    Parameters:
    -----------
    dt : float
        Resolución temporal (TR/T)
    length : int
        Longitud de la HRF en segundos
    
    Returns:
    --------
    hrf : array
        HRF canónica normalizada
    """
    t = np.arange(0, length, dt)
    
    # Parámetros del modelo de doble gamma
    a1, a2 = 6.0, 16.0  # delays
    b1, b2 = 1.0, 1.0   # dispersions
    c = 1/6             # ratio of response to undershoot
    
    # Función gamma doble
    hrf = ((t/a1)**a1) * np.exp(-(t-a1)/b1) / (a1*np.exp(-a1/b1)) - \
          c * ((t/a2)**a2) * np.exp(-(t-a2)/b2) / (a2*np.exp(-a2/b2))
    
    # Normalizar
    hrf = hrf / np.max(hrf)
    
    return hrf


def canonical_hrf_with_derivatives(dt=0.5, length=32):
    """
    HRF canónica con derivadas temporal y de dispersión
    
    Returns:
    --------
    bf : array (n_timepoints, 3)
        Base functions: [HRF, temporal_derivative, dispersion_derivative]
    """
    # HRF principal
    hrf = canonical_hrf(dt, length)
    n_points = len(hrf)
    
    # Derivada temporal (diferencia)
    hrf_td = np.gradient(hrf)
    
    # Derivada de dispersión (aproximación)
    t = np.arange(0, length, dt)
    hrf_dd = (t - t.mean()) * hrf
    
    # Crear matriz de funciones base
    bf = np.column_stack([hrf, hrf_td, hrf_dd])
    
    # Ortogonalizar
    bf = linalg.orth(bf)
    
    return bf


def gamma_basis_functions(dt=0.5, length=32, order=3):
    """
    Funciones base gamma
    
    Parameters:
    -----------
    order : int
        Número de funciones gamma
    """
    t = np.arange(0, length, dt)
    bf = np.zeros((len(t), order))
    
    for i in range(order):
        # Parámetros variables para cada función gamma
        a = 2 + i * 2  # delay
        b = 0.9        # dispersion
        bf[:, i] = stats.gamma.pdf(t, a/b, scale=b)
    
    # Normalizar cada columna
    bf = bf / np.max(bf, axis=0)
    
    # Ortogonalizar
    bf = linalg.orth(bf)
    
    return bf


def fourier_basis_functions(dt=0.5, length=32, order=3):
    """
    Funciones base de Fourier
    """
    t = np.arange(0, length, dt)
    n_points = len(t)
    bf = np.zeros((n_points, order*2))
    
    for i in range(order):
        freq = (i + 1) * 2 * np.pi / length
        bf[:, i*2] = np.sin(freq * t)
        bf[:, i*2+1] = np.cos(freq * t)
    
    # Aplicar ventana Hanning
    window = np.hanning(n_points)
    bf = bf * window[:, np.newaxis]
    
    return bf


def rsHRF_estimation_temporal_basis(data, para, temporal_mask=None, use_parallel=False):
    """
    Estimación HRF usando bases temporales
    
    Parameters:
    -----------
    data : array (n_timepoints, n_voxels)
        Datos BOLD
    para : dict
        Parámetros con campos: TR, T, dt, len, thr, lag, AR_lag, name
    temporal_mask : array bool
        Máscara temporal
    
    Returns:
    --------
    beta_hrf : array
        Coeficientes HRF
    bf : array
        Funciones base
    event_bold : list
        Eventos detectados por voxel
    """
    n_scans, n_voxels = data.shape
    
    if temporal_mask is None:
        temporal_mask = np.ones(n_scans, dtype=bool)
    
    # Generar funciones base según el tipo
    if 'Canonical' in para.get('name', 'Canonical'):
        if 'derivatives' in para.get('name', ''):
            bf = canonical_hrf_with_derivatives(para['dt'], para['len'])
        else:
            bf = canonical_hrf(para['dt'], para['len']).reshape(-1, 1)
    elif 'Gamma' in para.get('name', ''):
        bf = gamma_basis_functions(para['dt'], para['len'], para.get('order', 3))
    elif 'Fourier' in para.get('name', ''):
        bf = fourier_basis_functions(para['dt'], para['len'], para.get('order', 3))
    else:
        # Por defecto usar canónica con derivadas
        bf = canonical_hrf_with_derivatives(para['dt'], para['len'])
    
    n_basis = bf.shape[1]
    
    # Detectar eventos y estimar HRF para cada voxel
    beta_hrf = np.zeros((n_basis + 1, n_voxels))  # +1 para baseline
    event_bold = []
    
    for v in range(n_voxels):
        voxel_data = data[:, v]
        
        # Detectar eventos (picos sobre threshold)
        if 'thr' in para:
            threshold = np.mean(voxel_data) + para['thr'] * np.std(voxel_data)
            events = detect_events(voxel_data, threshold, para.get('lag', [0]))
        else:
            events = detect_events_auto(voxel_data)
        
        event_bold.append(events)
        
        # Crear matriz de diseño
        design_matrix = create_design_matrix(events, bf, n_scans, para['TR']/para['T'])
        
        # Añadir término constante (baseline)
        design_matrix = np.column_stack([design_matrix, np.ones(n_scans)])
        
        # Estimación por mínimos cuadrados
        if design_matrix.shape[1] > 0:
            # Regularización si es necesario
            beta = linalg.lstsq(design_matrix[temporal_mask], 
                               voxel_data[temporal_mask], 
                               rcond=None)[0]
            beta_hrf[:, v] = beta
    
    return beta_hrf, bf, event_bold


def rsHRF_estimation_FIR(data, para, temporal_mask=None, use_parallel=False):
    """
    Estimación HRF usando FIR (Finite Impulse Response)
    
    Returns:
    --------
    beta_hrf : array
        Coeficientes FIR + baseline + constante
    event_bold : list
        Eventos detectados
    """
    n_scans, n_voxels = data.shape
    
    if temporal_mask is None:
        temporal_mask = np.ones(n_scans, dtype=bool)
    
    # Número de bins FIR
    n_bins = int(para['len'] / (para['TR'] / para.get('T', 1)))
    
    beta_hrf = np.zeros((n_bins + 2, n_voxels))  # +2 para baseline y constante
    event_bold = []
    
    for v in range(n_voxels):
        voxel_data = data[:, v]
        
        # Detectar eventos
        threshold = np.mean(voxel_data) + para.get('thr', 1) * np.std(voxel_data)
        events = detect_events(voxel_data, threshold, para.get('lag', [0]))
        event_bold.append(events)
        
        # Crear matriz FIR
        design_matrix = create_fir_design_matrix(events, n_bins, n_scans)
        
        # Añadir baseline y constante
        design_matrix = np.column_stack([design_matrix, 
                                        np.arange(n_scans) / n_scans,  # drift lineal
                                        np.ones(n_scans)])              # constante
        
        # Estimación
        if design_matrix.shape[1] > 0 and len(events) > 0:
            beta = linalg.lstsq(design_matrix[temporal_mask], 
                               voxel_data[temporal_mask], 
                               rcond=None)[0]
            beta_hrf[:, v] = beta
    
    return beta_hrf, event_bold


# ============================================================================
# DETECCIÓN DE EVENTOS
# ============================================================================

def detect_events(signal, threshold, lag_range):
    """
    Detecta eventos en la señal BOLD
    
    Parameters:
    -----------
    signal : array
        Señal BOLD
    threshold : float
        Umbral de detección
    lag_range : array
        Rango de retrasos permitidos
    
    Returns:
    --------
    events : array
        Índices de eventos detectados
    """
    # Encontrar picos
    peaks, properties = signal.find_peaks(signal, height=threshold, 
                                         distance=int(4/0.5))  # mínimo 4 segundos entre eventos
    
    # Aplicar restricciones de lag si es necesario
    if len(lag_range) > 1:
        # Filtrar eventos según lag
        valid_peaks = []
        for peak in peaks:
            if any((peak + l >= 0 and peak + l < len(signal)) for l in lag_range):
                valid_peaks.append(peak)
        peaks = np.array(valid_peaks)
    
    return peaks


def detect_events_auto(signal, percentile=95):
    """
    Detección automática de eventos usando percentil
    """
    threshold = np.percentile(signal, percentile)
    peaks, _ = signal.find_peaks(signal, height=threshold, distance=8)
    return peaks


# ============================================================================
# MATRICES DE DISEÑO
# ============================================================================

def create_design_matrix(events, basis_functions, n_scans, dt):
    """
    Crea matriz de diseño convolucionando eventos con funciones base
    """
    n_basis = basis_functions.shape[1]
    design = np.zeros((n_scans, n_basis))
    
    # Crear tren de impulsos
    impulses = np.zeros(n_scans)
    impulses[events.astype(int)] = 1
    
    # Convolucionar con cada función base
    for i in range(n_basis):
        design[:, i] = np.convolve(impulses, basis_functions[:, i], mode='same')
    
    return design


def create_fir_design_matrix(events, n_bins, n_scans):
    """
    Crea matriz de diseño FIR
    """
    design = np.zeros((n_scans, n_bins))
    
    for event in events:
        for bin in range(n_bins):
            idx = int(event) + bin
            if 0 <= idx < n_scans:
                design[idx, bin] = 1
    
    return design


# ============================================================================
# DECONVOLUCIÓN
# ============================================================================

def rsHRF_iterative_wiener_deconv(signal, hrf, num_iterations=100):
    """
    Deconvolución iterativa de Wiener
    
    Parameters:
    -----------
    signal : array
        Señal BOLD (debe estar normalizada con zscore)
    hrf : array
        HRF estimada (normalizada por sigma)
    num_iterations : int
        Número de iteraciones
    
    Returns:
    --------
    deconv : array
        Señal deconvolucionada
    """
    # Inicialización
    deconv = signal.copy()
    n = len(signal)
    
    # Normalizar HRF
    hrf_norm = hrf / np.sum(np.abs(hrf))
    
    for iteration in range(num_iterations):
        # Convolucionar estimación actual con HRF
        conv = np.convolve(deconv, hrf_norm, mode='same')
        
        # Calcular residual
        residual = signal - conv
        
        # Actualizar con factor de aprendizaje adaptativo
        learning_rate = 1.0 / (1 + iteration * 0.1)
        deconv = deconv + learning_rate * residual
        
        # Regularización opcional (evitar valores extremos)
        deconv = np.clip(deconv, -5, 5)
    
    return deconv


def wiener_deconvolution(signal, hrf, noise_level=0.05):
    """
    Deconvolución de Wiener en dominio de Fourier
    
    Parameters:
    -----------
    signal : array
        Señal BOLD
    hrf : array
        HRF estimada
    noise_level : float
        Nivel de ruido para regularización
    
    Returns:
    --------
    deconv : array
        Señal deconvolucionada
    """
    n = len(signal)
    
    # Pad HRF to signal length
    hrf_padded = np.zeros(n)
    hrf_padded[:len(hrf)] = hrf
    
    # FFT
    Signal_fft = np.fft.fft(signal)
    HRF_fft = np.fft.fft(hrf_padded)
    
    # Wiener filter
    HRF_conj = np.conj(HRF_fft)
    denominator = np.abs(HRF_fft)**2 + noise_level
    
    # Evitar división por cero
    denominator[denominator < 1e-10] = 1e-10
    
    # Deconvolución
    Deconv_fft = (Signal_fft * HRF_conj) / denominator
    
    # IFFT
    deconv = np.real(np.fft.ifft(Deconv_fft))
    
    return deconv


# ============================================================================
# EXTRACCIÓN DE PARÁMETROS HRF
# ============================================================================

def rsHRF_get_HRF_parameters(hrf, dt):
    """
    Extrae parámetros de la HRF
    
    Parameters:
    -----------
    hrf : array
        HRF estimada
    dt : float
        Resolución temporal
    
    Returns:
    --------
    params : array (3,)
        [height, time_to_peak, FWHM]
    """
    # Height (amplitud máxima)
    height = np.max(hrf)
    
    # Time to peak
    peak_idx = np.argmax(hrf)
    time_to_peak = peak_idx * dt
    
    # FWHM (Full Width at Half Maximum)
    half_max = height / 2
    indices_above_half = np.where(hrf >= half_max)[0]
    
    if len(indices_above_half) > 0:
        fwhm = (indices_above_half[-1] - indices_above_half[0]) * dt
    else:
        fwhm = 0
    
    return np.array([height, time_to_peak, fwhm])


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def zscore(data, axis=0):
    """
    Z-score normalization
    """
    return (data - np.mean(data, axis=axis, keepdims=True)) / \
           (np.std(data, axis=axis, keepdims=True) + 1e-10)


def bandpass_filter(data, low_freq, high_freq, fs):
    """
    Filtro pasa-banda usando Butterworth
    
    Parameters:
    -----------
    data : array
        Señales a filtrar
    low_freq : float
        Frecuencia de corte inferior (Hz)
    high_freq : float
        Frecuencia de corte superior (Hz)
    fs : float
        Frecuencia de muestreo (Hz)
    """
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Diseñar filtro
    b, a = signal.butter(3, [low, high], btype='band')
    
    # Aplicar filtro
    if data.ndim == 1:
        filtered = signal.filtfilt(b, a, data)
    else:
        filtered = signal.filtfilt(b, a, data, axis=0)
    
    return filtered


# ============================================================================
# FUNCIÓN PRINCIPAL DE PIPELINE
# ============================================================================

def process_bold_to_deconv(bold_data, TR=2.0, estimation_type='canon2dd', 
                           deconv_type='wiener', passband=[0.01, 0.08]):
    """
    Pipeline completo: BOLD → HRF → Deconvolución
    
    Parameters:
    -----------
    bold_data : array (n_timepoints, n_voxels)
        Datos BOLD
    TR : float
        Tiempo de repetición
    estimation_type : str
        Tipo de estimación HRF
    deconv_type : str
        Tipo de deconvolución ('wiener' o 'iterative')
    passband : list
        Frecuencias de paso [low, high] Hz
    
    Returns:
    --------
    dict con:
        - deconv_signals: señales deconvolucionadas
        - hrf: HRF estimadas
        - hrf_params: parámetros HRF [height, time2peak, fwhm]
        - events: eventos detectados
    """
    
    n_scans, n_voxels = bold_data.shape
    
    # 1. Preprocesamiento
    print("Preprocesando datos BOLD...")
    bold_filtered = bandpass_filter(bold_data, passband[0], passband[1], 1/TR)
    bold_normalized = zscore(bold_filtered, axis=0)
    
    # 2. Configurar parámetros
    para = {
        'TR': TR,
        'T': 1,  # Sin upsampling para simplicidad
        'dt': TR,
        'len': 32,  # 32 segundos de HRF
        'thr': 1,  # 1 std sobre la media
        'lag': np.arange(2, 4),  # lag de 4-8 segundos
        'AR_lag': 1,
        'order': 3,
        'name': 'Canonical HRF (with time and dispersion derivatives)'
    }
    
    # 3. Estimar HRF
    print("Estimando HRF...")
    if 'canon' in estimation_type.lower():
        beta_hrf, bf, event_bold = rsHRF_estimation_temporal_basis(
            bold_normalized, para
        )
        hrf = bf @ beta_hrf[:-1, :]  # Excluir baseline
    else:
        beta_hrf, event_bold = rsHRF_estimation_FIR(
            bold_normalized, para
        )
        hrf = beta_hrf[:-2, :]  # Excluir baseline y constante
    
    # 4. Extraer parámetros HRF
    print("Extrayendo parámetros HRF...")
    hrf_params = np.zeros((3, n_voxels))
    for v in range(n_voxels):
        hrf_params[:, v] = rsHRF_get_HRF_parameters(hrf[:, v], para['dt'])
    
    # 5. Deconvolucionar
    print("Deconvolucionando señales...")
    deconv_signals = np.zeros_like(bold_data)
    
    for v in range(n_voxels):
        if deconv_type == 'iterative':
            deconv_signals[:, v] = rsHRF_iterative_wiener_deconv(
                bold_normalized[:, v],
                hrf[:min(len(hrf), n_scans), v],
                num_iterations=100
            )
        else:  # wiener
            deconv_signals[:, v] = wiener_deconvolution(
                bold_normalized[:, v],
                hrf[:min(len(hrf), n_scans), v],
                noise_level=0.05
            )
    
    print("¡Procesamiento completado!")
    
    return {
        'deconv_signals': deconv_signals,
        'hrf': hrf,
        'hrf_params': hrf_params,
        'events': event_bold,
        'bold_filtered': bold_filtered
    }


# ============================================================================
# EJEMPLO DE USO CON LAS CLASES EXISTENTES
# ============================================================================

def integrate_with_classes(bold_data, TR=2.0):
    """
    Ejemplo de integración con las clases HRF y Bold_Deconv
    """
    # Importar las clases (ajustar path según tu estructura)
    # from hrf import HRF
    # from bold_deconv import Bold_Deconv
    # from parameters import Parameters
    
    # Procesar datos
    results = process_bold_to_deconv(bold_data, TR)
    
    # Crear objetos con las clases
    # params = Parameters()
    # params.set_TR(TR)
    # params.set_estimation('canon2dd')
    
    # hrf_obj = HRF(
    #     label="Estimated HRF",
    #     ts=results['hrf'],
    #     subject_index="subj_001",
    #     para=params
    # )
    # hrf_obj.set_para(results['hrf_params'])
    # hrf_obj.set_event_bold(results['events'])
    
    # deconv_obj = Bold_Deconv(
    #     label="Deconvolved BOLD",
    #     ts=results['deconv_signals'],
    #     subject_index="subj_001",
    #     para=params
    # )
    # deconv_obj.set_HRF(hrf_obj)
    
    return results


if __name__ == "__main__":
    # Test con datos sintéticos
    print("Generando datos de prueba...")
    n_scans = 300
    n_voxels = 10
    TR = 2.0
    
    # Crear datos BOLD sintéticos
    t = np.arange(n_scans) * TR
    bold_test = np.zeros((n_scans, n_voxels))
    
    for v in range(n_voxels):
        # Señal con eventos periódicos
        events = np.zeros(n_scans)
        events[::30] = 1  # Evento cada 60 segundos
        
        # Convolucionar con HRF canónica
        hrf_true = canonical_hrf(TR, 32)
        bold_test[:, v] = np.convolve(events, hrf_true, mode='same')
        
        # Añadir ruido
        bold_test[:, v] += np.random.randn(n_scans) * 0.1
    
    # Procesar
    print("\nProbando pipeline completo...")
    results = process_bold_to_deconv(bold_test, TR)
    
    print("\nResultados:")
    print(f"Forma deconv_signals: {results['deconv_signals'].shape}")
    print(f"Forma HRF: {results['hrf'].shape}")
    print(f"Parámetros HRF (voxel 0):")
    print(f"  Height: {results['hrf_params'][0, 0]:.3f}")
    print(f"  Time to peak: {results['hrf_params'][1, 0]:.3f} s")
    print(f"  FWHM: {results['hrf_params'][2, 0]:.3f} s")
    print(f"Eventos detectados (voxel 0): {len(results['events'][0])} eventos")