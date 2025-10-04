#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Independiente de Estimación HRF y Deconvolución
Diseñado específicamente para integrarse con simulaciones DMF
Sin dependencias de rsHRF u otras librerías externas
Solo usa: numpy, scipy, matplotlib
"""

import numpy as np
from scipy import signal, linalg, stats, optimize
from scipy.special import gamma as gamma_func
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Importar tus módulos existentes
import DMF
import BOLDModel as BD


# ============================================================================
# PARTE 1: GENERACIÓN DE HRF CANÓNICAS
# ============================================================================

def create_canonical_hrf(TR=1.0, hrf_length=32.0):
    """
    Crea HRF canónica usando el modelo de doble gamma (SPM-like)
    
    Parameters:
    -----------
    TR : float
        Tiempo de repetición en segundos
    hrf_length : float
        Longitud de la HRF en segundos
    
    Returns:
    --------
    hrf : array
        HRF canónica normalizada
    t : array
        Vector de tiempo
    """
    t = np.arange(0, hrf_length, TR)
    
    # Parámetros del modelo doble gamma (valores de SPM)
    peak_delay = 6.0      # Retardo del pico principal
    undershoot_delay = 16.0  # Retardo del undershoot
    peak_disp = 1.0       # Dispersión del pico
    undershoot_disp = 1.0    # Dispersión del undershoot
    peak_undershoot_ratio = 0.167  # Ratio entre pico y undershoot
    
    # Primera función gamma (pico principal)
    peak = (t / peak_delay) ** (peak_delay / peak_disp) * \
           np.exp(-(t - peak_delay) / peak_disp)
    peak = peak / np.max(peak)
    
    # Segunda función gamma (undershoot)
    undershoot = (t / undershoot_delay) ** (undershoot_delay / undershoot_disp) * \
                 np.exp(-(t - undershoot_delay) / undershoot_disp)
    undershoot = undershoot / np.max(undershoot)
    
    # Combinar
    hrf = peak - peak_undershoot_ratio * undershoot
    
    # Normalizar
    hrf = hrf / np.max(np.abs(hrf))
    
    return hrf, t


def create_hrf_with_derivatives(TR=1.0, hrf_length=32.0):
    """
    Crea HRF con derivadas temporal y de dispersión
    
    Returns:
    --------
    basis : array (n_time_points, 3)
        [HRF, derivada_temporal, derivada_dispersión]
    """
    hrf, t = create_canonical_hrf(TR, hrf_length)
    
    # Derivada temporal (aproximación por diferencias)
    hrf_td = np.gradient(hrf) / TR
    hrf_td = hrf_td / np.max(np.abs(hrf_td))
    
    # Derivada de dispersión (modulación por tiempo)
    hrf_dd = (t - t.mean()) * hrf / t.std()
    hrf_dd = hrf_dd / np.max(np.abs(hrf_dd))
    
    # Construir matriz de funciones base
    basis = np.column_stack([hrf, hrf_td, hrf_dd])
    
    # Ortogonalizar usando QR
    Q, _ = linalg.qr(basis, mode='economic')
    
    return Q


# ============================================================================
# PARTE 2: DETECCIÓN DE EVENTOS EN SEÑAL BOLD
# ============================================================================

def detect_bold_events(signal_bold, threshold_factor=1.0, min_distance_seconds=4.0, TR=1.0):
    """
    Detecta eventos en señal BOLD usando umbral estadístico
    
    Parameters:
    -----------
    signal_bold : array
        Señal BOLD (debe estar normalizada)
    threshold_factor : float
        Factor multiplicativo para el umbral (media + factor*std)
    min_distance_seconds : float
        Distancia mínima entre eventos en segundos
    TR : float
        Tiempo de repetición
    
    Returns:
    --------
    events : array
        Índices de los eventos detectados
    """
    # Calcular umbral
    threshold = np.mean(signal_bold) + threshold_factor * np.std(signal_bold)
    
    # Convertir distancia mínima a muestras
    min_distance_samples = int(min_distance_seconds / TR)
    
    # Encontrar picos
    peaks, properties = signal.find_peaks(
        signal_bold,
        height=threshold,
        distance=min_distance_samples
    )
    
    return peaks


def create_event_design_matrix(events, hrf_basis, n_scans):
    """
    Crea matriz de diseño convolucionando eventos con HRF
    
    Parameters:
    -----------
    events : array
        Índices de eventos
    hrf_basis : array
        Funciones base HRF (puede ser 1D o 2D)
    n_scans : int
        Número total de scans
    
    Returns:
    --------
    design : array
        Matriz de diseño
    """
    if hrf_basis.ndim == 1:
        hrf_basis = hrf_basis.reshape(-1, 1)
    
    n_basis = hrf_basis.shape[1]
    design = np.zeros((n_scans, n_basis))
    
    # Crear tren de impulsos
    impulses = np.zeros(n_scans)
    if len(events) > 0:
        impulses[events] = 1
    
    # Convolucionar con cada función base
    for i in range(n_basis):
        # Usar modo 'same' para mantener la longitud
        design[:, i] = np.convolve(impulses, hrf_basis[:, i], mode='same')
    
    return design


# ============================================================================
# PARTE 3: ESTIMACIÓN HRF DESDE DATOS
# ============================================================================

def estimate_hrf_from_bold(bold_signal, TR=1.0, hrf_length=32.0, 
                          use_derivatives=True, threshold_factor=1.0):
    """
    Estima HRF desde señal BOLD usando regresión con funciones base
    
    Parameters:
    -----------
    bold_signal : array (n_timepoints,) o (n_timepoints, n_voxels)
        Señal BOLD
    TR : float
        Tiempo de repetición
    hrf_length : float
        Longitud asumida de la HRF
    use_derivatives : bool
        Si usar derivadas en la estimación
    threshold_factor : float
        Factor para detección de eventos
    
    Returns:
    --------
    dict con:
        - hrf_estimated: HRF estimada
        - events: eventos detectados
        - beta: coeficientes de regresión
        - basis: funciones base usadas
    """
    # Asegurar que es 2D
    if bold_signal.ndim == 1:
        bold_signal = bold_signal.reshape(-1, 1)
    
    n_scans, n_voxels = bold_signal.shape
    
    # Crear funciones base
    if use_derivatives:
        basis = create_hrf_with_derivatives(TR, hrf_length)
    else:
        hrf_canonical, _ = create_canonical_hrf(TR, hrf_length)
        basis = hrf_canonical.reshape(-1, 1)
    
    n_basis = basis.shape[1]
    
    # Almacenar resultados
    hrf_estimated = np.zeros((len(basis), n_voxels))
    all_events = []
    all_betas = np.zeros((n_basis + 1, n_voxels))  # +1 para intercepto
    
    for v in range(n_voxels):
        # Normalizar señal
        voxel_signal = stats.zscore(bold_signal[:, v])
        
        # Detectar eventos
        events = detect_bold_events(voxel_signal, threshold_factor, TR=TR)
        all_events.append(events)
        
        if len(events) > 0:
            # Crear matriz de diseño
            design = create_event_design_matrix(events, basis, n_scans)
            
            # Añadir intercepto
            design = np.column_stack([design, np.ones(n_scans)])
            
            # Regresión lineal con regularización
            try:
                # Usar pseudo-inversa para estabilidad
                beta = linalg.pinv(design) @ voxel_signal
                all_betas[:, v] = beta
                
                # Reconstruir HRF
                hrf_estimated[:, v] = basis @ beta[:-1]
            except:
                # Si falla, usar HRF canónica
                hrf_estimated[:, v] = basis[:, 0]
        else:
            # Sin eventos, usar HRF canónica
            hrf_estimated[:, v] = basis[:, 0]
    
    return {
        'hrf_estimated': hrf_estimated,
        'events': all_events,
        'beta': all_betas,
        'basis': basis
    }


# ============================================================================
# PARTE 4: DECONVOLUCIÓN ITERATIVA DE WIENER
# ============================================================================

def wiener_deconv_iterative(signal_bold, hrf, n_iterations=100, noise_level=None):
    """
    Deconvolución iterativa de Wiener (versión simplificada y robusta)
    
    Parameters:
    -----------
    signal_bold : array
        Señal BOLD a deconvolucionar
    hrf : array
        HRF estimada
    n_iterations : int
        Número de iteraciones
    noise_level : float o None
        Nivel de ruido. Si None, se estima automáticamente
    
    Returns:
    --------
    deconv : array
        Señal deconvolucionada
    """
    n = len(signal_bold)
    
    # Preparar HRF
    hrf_padded = np.zeros(n)
    hrf_padded[:min(len(hrf), n)] = hrf[:min(len(hrf), n)]
    
    # FFT
    H = np.fft.fft(hrf_padded)
    Y = np.fft.fft(signal_bold)
    
    # Estimar ruido si no se proporciona
    if noise_level is None:
        # Usar MAD (Median Absolute Deviation) para estimación robusta
        residual = signal_bold - np.mean(signal_bold)
        noise_level = 1.4826 * np.median(np.abs(residual))
    
    # Parámetros iniciales
    Phh = np.abs(H)**2
    noise_power = noise_level**2 * n
    
    # Inicialización
    # Estimación inicial usando filtro de Wiener simple
    wiener_filter = np.conj(H) / (Phh + noise_power/np.var(signal_bold))
    X = Y * wiener_filter
    Pxx = np.abs(X)**2
    
    # Iteraciones para refinar
    for iteration in range(n_iterations):
        # Actualizar filtro de Wiener con estimación actual de potencia
        wiener_filter = (np.conj(H) * Pxx) / (Phh * Pxx + noise_power)
        
        # Aplicar filtro
        X = Y * wiener_filter
        
        # Actualizar estimación de potencia con suavizado
        alpha = 0.9  # Factor de suavizado
        Pxx = alpha * Pxx + (1 - alpha) * np.abs(X)**2
        
        # Regularización para evitar inestabilidad
        Pxx = np.maximum(Pxx, 1e-10)
    
    # Transformada inversa
    deconv = np.real(np.fft.ifft(X))
    
    return deconv


def wiener_deconv_direct(signal_bold, hrf, regularization=0.1):
    """
    Deconvolución de Wiener directa (no iterativa)
    Más rápida pero menos precisa que la versión iterativa
    
    Parameters:
    -----------
    signal_bold : array
        Señal BOLD
    hrf : array
        HRF estimada
    regularization : float
        Parámetro de regularización (entre 0 y 1)
    
    Returns:
    --------
    deconv : array
        Señal deconvolucionada
    """
    n = len(signal_bold)
    
    # Preparar HRF
    hrf_padded = np.zeros(n)
    hrf_padded[:min(len(hrf), n)] = hrf[:min(len(hrf), n)]
    
    # FFT
    H = np.fft.fft(hrf_padded)
    Y = np.fft.fft(signal_bold)
    
    # Filtro de Wiener con regularización
    H_conj = np.conj(H)
    H_abs2 = np.abs(H)**2
    
    # Evitar división por cero
    denominator = H_abs2 + regularization * np.mean(H_abs2)
    
    # Aplicar filtro
    X = Y * H_conj / denominator
    
    # Transformada inversa
    deconv = np.real(np.fft.ifft(X))
    
    return deconv


# ============================================================================
# PARTE 5: EXTRACCIÓN DE PARÁMETROS HRF
# ============================================================================

def extract_hrf_parameters(hrf, TR=1.0):
    """
    Extrae parámetros característicos de la HRF
    
    Parameters:
    -----------
    hrf : array
        HRF estimada
    TR : float
        Tiempo de repetición
    
    Returns:
    --------
    dict con:
        - height: amplitud máxima
        - time_to_peak: tiempo al pico (segundos)
        - fwhm: ancho a media altura (segundos)
        - time_to_undershoot: tiempo al undershoot (segundos)
        - undershoot_amplitude: amplitud del undershoot
    """
    # Encontrar pico principal
    peak_idx = np.argmax(hrf)
    height = hrf[peak_idx]
    time_to_peak = peak_idx * TR
    
    # FWHM (Full Width at Half Maximum)
    half_max = height / 2
    indices_above_half = np.where(hrf >= half_max)[0]
    if len(indices_above_half) > 1:
        fwhm = (indices_above_half[-1] - indices_above_half[0]) * TR
    else:
        fwhm = TR  # Valor mínimo
    
    # Encontrar undershoot (mínimo después del pico)
    if peak_idx < len(hrf) - 1:
        undershoot_region = hrf[peak_idx:]
        undershoot_idx = np.argmin(undershoot_region) + peak_idx
        undershoot_amplitude = hrf[undershoot_idx]
        time_to_undershoot = undershoot_idx * TR
    else:
        undershoot_amplitude = 0
        time_to_undershoot = 0
    
    return {
        'height': height,
        'time_to_peak': time_to_peak,
        'fwhm': fwhm,
        'time_to_undershoot': time_to_undershoot,
        'undershoot_amplitude': undershoot_amplitude
    }


# ============================================================================
# PARTE 6: PIPELINE COMPLETO DMF → HRF → DECONVOLUCIÓN
# ============================================================================

class DMF_HRF_Pipeline:
    """
    Clase para encapsular el pipeline completo
    """
    
    def __init__(self, SC_matrix=None, G=1.1, sigma=0.4):
        """
        Inicializa el pipeline
        
        Parameters:
        -----------
        SC_matrix : array o str
            Matriz de conectividad estructural o path al archivo
        G : float
            Acoplamiento global para DMF
        sigma : float
            Nivel de ruido para DMF
        """
        # Configurar DMF
        if SC_matrix is not None:
            if isinstance(SC_matrix, str):
                self.SC = np.loadtxt(SC_matrix)
            else:
                self.SC = SC_matrix
            
            DMF.SC = self.SC / np.mean(np.sum(self.SC, 0))
            DMF.nnodes = len(DMF.SC)
        
        self.G = G
        self.sigma = sigma
        
        # Parámetros por defecto
        self.TR = 1.0
        self.hrf_length = 32.0
        self.deconv_method = 'iterative'
        
        # Resultados
        self.results = {}
    
    def simulate_dmf(self, tmax=600, dt=0.001, downsampling=1000, verbose=True):
        """
        Ejecuta simulación DMF
        """
        # Configurar parámetros
        DMF.tmax = tmax
        DMF.dt = dt
        DMF.downsampling = downsampling
        DMF.downsampling_rates = 10
        DMF.G = self.G
        DMF.sigma = self.sigma
        
        # Actualizar y simular
        DMF.update()
        
        if verbose:
            print(f"Simulando DMF: G={self.G}, sigma={self.sigma}, tmax={tmax}s")
        
        BOLD_signals, rates, time = DMF.Sim(verbose=verbose, return_rates=True)
        
        self.TR = DMF.dt * DMF.downsampling
        
        # Guardar resultados
        self.results['bold_raw'] = BOLD_signals
        self.results['rates'] = rates
        self.results['time'] = time
        
        return BOLD_signals
    
    def preprocess_bold(self, remove_initial=120, remove_final=60, 
                       bandpass=[0.01, 0.1]):
        """
        Preprocesa señales BOLD
        """
        if 'bold_raw' not in self.results:
            raise ValueError("Primero debes ejecutar simulate_dmf()")
        
        bold = self.results['bold_raw']
        
        # Remover transitorio
        start_idx = int(remove_initial / self.TR)
        end_idx = -int(remove_final / self.TR) if remove_final > 0 else None
        bold = bold[start_idx:end_idx, :]
        
        # Filtrado pasa-banda
        nyq = 0.5 / self.TR
        low = bandpass[0] / nyq
        high = bandpass[1] / nyq
        b, a = signal.butter(3, [low, high], btype='band')
        bold_filtered = signal.filtfilt(b, a, bold, axis=0)
        
        # Z-score
        bold_normalized = stats.zscore(bold_filtered, axis=0, ddof=1)
        
        self.results['bold_preprocessed'] = bold_normalized
        
        return bold_normalized
    
    def estimate_hrf(self, use_derivatives=True, threshold_factor=1.0):
        """
        Estima HRF desde BOLD preprocesado
        """
        if 'bold_preprocessed' not in self.results:
            raise ValueError("Primero debes ejecutar preprocess_bold()")
        
        bold = self.results['bold_preprocessed']
        
        # Estimar HRF
        hrf_results = estimate_hrf_from_bold(
            bold, 
            TR=self.TR,
            hrf_length=self.hrf_length,
            use_derivatives=use_derivatives,
            threshold_factor=threshold_factor
        )
        
        # Extraer parámetros para cada voxel
        n_voxels = bold.shape[1]
        hrf_params = []
        
        for v in range(n_voxels):
            params = extract_hrf_parameters(
                hrf_results['hrf_estimated'][:, v], 
                self.TR
            )
            hrf_params.append(params)
        
        # Guardar resultados
        self.results['hrf'] = hrf_results['hrf_estimated']
        self.results['hrf_events'] = hrf_results['events']
        self.results['hrf_params'] = hrf_params
        self.results['hrf_basis'] = hrf_results['basis']
        
        return hrf_results['hrf_estimated']
    
    def deconvolve_bold(self, method='iterative', n_iterations=100):
        """
        Deconvoluciona señales BOLD usando HRF estimada
        """
        if 'hrf' not in self.results:
            raise ValueError("Primero debes ejecutar estimate_hrf()")
        
        bold = self.results['bold_preprocessed']
        hrf = self.results['hrf']
        
        n_scans, n_voxels = bold.shape
        deconv = np.zeros_like(bold)
        
        for v in range(n_voxels):
            if method == 'iterative':
                deconv[:, v] = wiener_deconv_iterative(
                    bold[:, v], 
                    hrf[:, v],
                    n_iterations=n_iterations
                )
            else:  # direct
                deconv[:, v] = wiener_deconv_direct(
                    bold[:, v], 
                    hrf[:, v],
                    regularization=0.1
                )
        
        self.results['bold_deconvolved'] = deconv
        
        return deconv
    
    def compute_connectivity(self):
        """
        Calcula matrices de conectividad funcional
        """
        fc_results = {}
        
        if 'bold_preprocessed' in self.results:
            fc_results['FC_bold'] = np.corrcoef(
                self.results['bold_preprocessed'].T
            )
        
        if 'bold_deconvolved' in self.results:
            fc_results['FC_deconv'] = np.corrcoef(
                self.results['bold_deconvolved'].T
            )
        
        self.results.update(fc_results)
        
        return fc_results
    
    def run_complete_pipeline(self, tmax=600, verbose=True):
        """
        Ejecuta el pipeline completo
        """
        if verbose:
            print("="*60)
            print("EJECUTANDO PIPELINE COMPLETO DMF → HRF → DECONVOLUCIÓN")
            print("="*60)
        
        # 1. Simular DMF
        if verbose:
            print("\n1. Simulando DMF...")
        self.simulate_dmf(tmax=tmax, verbose=verbose)
        
        # 2. Preprocesar
        if verbose:
            print("\n2. Preprocesando BOLD...")
        self.preprocess_bold()
        
        # 3. Estimar HRF
        if verbose:
            print("\n3. Estimando HRF...")
        self.estimate_hrf()
        
        # 4. Deconvolucionar
        if verbose:
            print("\n4. Deconvolucionando...")
        self.deconvolve_bold()
        
        # 5. Calcular conectividad
        if verbose:
            print("\n5. Calculando conectividad funcional...")
        self.compute_connectivity()
        
        if verbose:
            print("\n" + "="*60)
            print("PIPELINE COMPLETADO")
            print("="*60)
            self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """
        Imprime resumen de resultados
        """
        print("\nRESUMEN DE RESULTADOS:")
        print("-"*40)
        
        # Dimensiones
        if 'bold_preprocessed' in self.results:
            shape = self.results['bold_preprocessed'].shape
            print(f"Señal BOLD: {shape[0]} timepoints, {shape[1]} nodos")
            print(f"TR efectivo: {self.TR:.3f} segundos")
        
        # HRF
        if 'hrf_params' in self.results:
            params = self.results['hrf_params']
            heights = [p['height'] for p in params]
            peaks = [p['time_to_peak'] for p in params]
            fwhms = [p['fwhm'] for p in params]
            
            print(f"\nParámetros HRF promedio:")
            print(f"  Altura: {np.mean(heights):.3f} ± {np.std(heights):.3f}")
            print(f"  Tiempo al pico: {np.mean(peaks):.2f} ± {np.std(peaks):.2f} s")
            print(f"  FWHM: {np.mean(fwhms):.2f} ± {np.std(fwhms):.2f} s")
        
        # Eventos
        if 'hrf_events' in self.results:
            n_events = [len(e) for e in self.results['hrf_events']]
            print(f"\nEventos detectados: {np.mean(n_events):.1f} ± {np.std(n_events):.1f} por nodo")
        
        # Conectividad
        if 'FC_bold' in self.results and 'FC_deconv' in self.results:
            fc_bold = self.results['FC_bold']
            fc_deconv = self.results['FC_deconv']
            
            # Extraer valores triangulares inferiores (sin diagonal)
            tril_idx = np.tril_indices_from(fc_bold, -1)
            fc_bold_vals = fc_bold[tril_idx]
            fc_deconv_vals = fc_deconv[tril_idx]
            
            print(f"\nConectividad Funcional:")
            print(f"  FC BOLD - Media: {np.mean(fc_bold_vals):.3f}, Std: {np.std(fc_bold_vals):.3f}")
            print(f"  FC Deconv - Media: {np.mean(fc_deconv_vals):.3f}, Std: {np.std(fc_deconv_vals):.3f}")
    
    def plot_results(self, node_indices=[0, 5, 10], save_fig=None):
        """
        Visualiza resultados del pipeline
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Configurar índices de nodos a mostrar
        if node_indices is None:
            node_indices = [0]
        
        # 1. Señales BOLD
        ax1 = plt.subplot(3, 3, 1)
        if 'bold_preprocessed' in self.results:
            bold = self.results['bold_preprocessed']
            time_bold = np.arange(len(bold)) * self.TR
            for idx in node_indices:
                ax1.plot(time_bold, bold[:, idx], alpha=0.7, label=f'Nodo {idx}')
        ax1.set_title('Señales BOLD Preprocesadas')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Amplitud (z-score)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Señales Deconvolucionadas
        ax2 = plt.subplot(3, 3, 2)
        if 'bold_deconvolved' in self.results:
            deconv = self.results['bold_deconvolved']
            for idx in node_indices:
                ax2.plot(time_bold, deconv[:, idx], alpha=0.7)
        ax2.set_title('Señales Deconvolucionadas')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Amplitud')
        ax2.grid(True, alpha=0.3)
        
        # 3. HRFs estimadas
        ax3 = plt.subplot(3, 3, 3)
        if 'hrf' in self.results:
            hrf = self.results['hrf']
            time_hrf = np.arange(len(hrf)) * self.TR
            for idx in node_indices:
                ax3.plot(time_hrf, hrf[:, idx], label=f'Nodo {idx}')
        ax3.set_title('HRFs Estimadas')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Amplitud')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. FC BOLD
        ax4 = plt.subplot(3, 3, 4)
        if 'FC_bold' in self.results:
            im = ax4.imshow(self.results['FC_bold'], cmap='jet', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax4, fraction=0.046)
        ax4.set_title('FC BOLD')
        ax4.set_xlabel('Nodo')
        ax4.set_ylabel('Nodo')
        
        # 5. FC Deconvolucionada
        ax5 = plt.subplot(3, 3, 5)
        if 'FC_deconv' in self.results:
            im = ax5.imshow(self.results['FC_deconv'], cmap='jet', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax5, fraction=0.046)
        ax5.set_title('FC Deconvolucionada')
        ax5.set_xlabel('Nodo')
        ax5.set_ylabel('Nodo')
        
        # 6. Diferencia FC
        ax6 = plt.subplot(3, 3, 6)
        if 'FC_bold' in self.results and 'FC_deconv' in self.results:
            fc_diff = self.results['FC_deconv'] - self.results['FC_bold']
            im = ax6.imshow(fc_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            plt.colorbar(im, ax=ax6, fraction=0.046)
        ax6.set_title('Diferencia (Deconv - BOLD)')
        ax6.set_xlabel('Nodo')
        ax6.set_ylabel('Nodo')
        
        # 7. Distribución de parámetros HRF
        ax7 = plt.subplot(3, 3, 7)
        if 'hrf_params' in self.results:
            heights = [p['height'] for p in self.results['hrf_params']]
            ax7.hist(heights, bins=20, edgecolor='black', alpha=0.7)
            ax7.axvline(np.mean(heights), color='r', linestyle='--', label='Media')
        ax7.set_title('Distribución de Amplitudes HRF')
        ax7.set_xlabel('Amplitud')
        ax7.set_ylabel('Frecuencia')
        ax7.legend()
        
        # 8. Eventos detectados con señal
        ax8 = plt.subplot(3, 3, 8)
        if 'bold_preprocessed' in self.results and 'hrf_events' in self.results:
            node_idx = node_indices[0]
            ax8.plot(time_bold, self.results['bold_preprocessed'][:, node_idx], 
                    'b-', alpha=0.7, label='BOLD')
            
            # Marcar eventos
            events = self.results['hrf_events'][node_idx]
            if len(events) > 0:
                ax8.plot(time_bold[events], 
                        self.results['bold_preprocessed'][events, node_idx], 
                        'ro', markersize=8, label='Eventos')
        ax8.set_title(f'Detección de Eventos (Nodo {node_indices[0]})')
        ax8.set_xlabel('Tiempo (s)')
        ax8.set_ylabel('Amplitud')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Comparación de espectros
        ax9 = plt.subplot(3, 3, 9)
        if 'bold_preprocessed' in self.results and 'bold_deconvolved' in self.results:
            node_idx = node_indices[0]
            
            # Calcular PSD
            f_bold, psd_bold = signal.welch(
                self.results['bold_preprocessed'][:, node_idx],
                fs=1/self.TR, nperseg=min(256, len(bold)//4)
            )
            f_deconv, psd_deconv = signal.welch(
                self.results['bold_deconvolved'][:, node_idx],
                fs=1/self.TR, nperseg=min(256, len(deconv)//4)
            )
            
            ax9.semilogy(f_bold, psd_bold, label='BOLD', alpha=0.7)
            ax9.semilogy(f_deconv, psd_deconv, label='Deconv', alpha=0.7)
            ax9.set_xlim([0, 0.5])
        ax9.set_title(f'Espectro de Potencia (Nodo {node_indices[0]})')
        ax9.set_xlabel('Frecuencia (Hz)')
        ax9.set_ylabel('PSD')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(save_fig, dpi=150, bbox_inches='tight')
            print(f"Figura guardada en: {save_fig}")
        
        plt.show()
        
        return fig
    
    def save_results(self, filename='dmf_hrf_results'):
        """
        Guarda resultados en formato .npz y .mat
        """
        # Guardar en .npz (NumPy)
        np.savez(f'{filename}.npz', **self.results)
        print(f"Resultados guardados en: {filename}.npz")
        
        # Guardar en .mat (MATLAB)
        try:
            from scipy.io import savemat
            # Filtrar solo arrays para .mat
            mat_dict = {}
            for key, value in self.results.items():
                if isinstance(value, np.ndarray):
                    mat_dict[key] = value
                elif isinstance(value, list) and len(value) > 0:
                    # Convertir listas de eventos a formato cell de MATLAB
                    if key == 'hrf_events':
                        # Crear matriz donde cada columna es un nodo
                        max_events = max(len(e) for e in value)
                        events_mat = np.full((max_events, len(value)), np.nan)
                        for i, events in enumerate(value):
                            events_mat[:len(events), i] = events
                        mat_dict[key] = events_mat
            
            savemat(f'{filename}.mat', mat_dict)
            print(f"Resultados guardados en: {filename}.mat")
        except ImportError:
            print("scipy.io no disponible, solo se guardó .npz")
        
        return filename


# ============================================================================
# FUNCIONES DE UTILIDAD ADICIONALES
# ============================================================================

def compare_with_empirical_fc(fc_simulated, fc_empirical):
    """
    Compara FC simulada con empírica
    
    Parameters:
    -----------
    fc_simulated : array
        Matriz FC simulada
    fc_empirical : array
        Matriz FC empírica
    
    Returns:
    --------
    dict con métricas de comparación
    """
    # Extraer valores triangulares inferiores
    tril_idx = np.tril_indices_from(fc_simulated, -1)
    fc_sim_vals = fc_simulated[tril_idx]
    fc_emp_vals = fc_empirical[tril_idx]
    
    # Correlación de Pearson
    correlation, p_value = stats.pearsonr(fc_sim_vals, fc_emp_vals)
    
    # Distancia euclidiana
    euclidean = np.linalg.norm(fc_sim_vals - fc_emp_vals)
    
    # Error cuadrático medio
    mse = np.mean((fc_sim_vals - fc_emp_vals)**2)
    
    # Coeficiente de determinación R²
    ss_res = np.sum((fc_emp_vals - fc_sim_vals)**2)
    ss_tot = np.sum((fc_emp_vals - np.mean(fc_emp_vals))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'euclidean_distance': euclidean,
        'mse': mse,
        'r_squared': r_squared
    }


def optimize_dmf_parameters(SC_matrix, FC_empirical, G_range=(0.5, 2.0), 
                           sigma_range=(0.1, 1.0), n_iterations=10):
    """
    Optimiza parámetros G y sigma de DMF para ajustar a FC empírica
    
    Parameters:
    -----------
    SC_matrix : array
        Matriz de conectividad estructural
    FC_empirical : array
        FC empírica objetivo
    G_range : tuple
        Rango de valores G a probar
    sigma_range : tuple
        Rango de valores sigma a probar
    n_iterations : int
        Número de combinaciones a probar
    
    Returns:
    --------
    dict con parámetros óptimos y resultados
    """
    from scipy.optimize import differential_evolution
    
    def objective(params):
        G, sigma = params
        
        # Crear pipeline
        pipeline = DMF_HRF_Pipeline(SC_matrix=SC_matrix, G=G, sigma=sigma)
        
        # Ejecutar simulación corta
        pipeline.simulate_dmf(tmax=300, verbose=False)
        pipeline.preprocess_bold()
        
        # Calcular FC
        fc_bold = np.corrcoef(pipeline.results['bold_preprocessed'].T)
        
        # Comparar con empírica
        comparison = compare_with_empirical_fc(fc_bold, FC_empirical)
        
        # Minimizar distancia negativa de correlación
        return -comparison['correlation']
    
    # Optimización
    bounds = [G_range, sigma_range]
    result = differential_evolution(objective, bounds, maxiter=n_iterations, 
                                  disp=True, seed=42)
    
    return {
        'optimal_G': result.x[0],
        'optimal_sigma': result.x[1],
        'optimal_correlation': -result.fun,
        'optimization_result': result
    }


# ============================================================================
# EJEMPLO DE USO COMPLETO
# ============================================================================

def example_usage():
    """
    Ejemplo completo de uso del pipeline
    """
    print("="*60)
    print("EJEMPLO DE USO: PIPELINE DMF → HRF → DECONVOLUCIÓN")
    print("="*60)
    
    # Crear datos sintéticos si no existen archivos reales
    try:
        # Intentar cargar datos reales
        SC = np.loadtxt("SC_opti_25julio.txt")
        FC_empirical = np.load("average_90x90FC_HCPchina_symm.npy")
        print("Datos reales cargados exitosamente")
    except:
        print("Creando datos sintéticos de prueba...")
        # Crear SC sintética
        n_nodes = 45
        SC = np.random.rand(n_nodes, n_nodes)
        SC = (SC + SC.T) / 2  # Hacer simétrica
        np.fill_diagonal(SC, 0)  # Sin autoconexiones
        
        # Crear FC empírica sintética
        FC_empirical = np.random.rand(n_nodes, n_nodes)
        FC_empirical = (FC_empirical + FC_empirical.T) / 2
        np.fill_diagonal(FC_empirical, 1)
    
    # 1. Crear pipeline
    print("\n1. Inicializando pipeline...")
    pipeline = DMF_HRF_Pipeline(SC_matrix=SC, G=1.1, sigma=0.4)
    
    # 2. Ejecutar pipeline completo
    print("\n2. Ejecutando pipeline completo...")
    results = pipeline.run_complete_pipeline(tmax=600, verbose=True)
    
    # 3. Comparar con FC empírica
    if 'FC_bold' in results:
        print("\n3. Comparando con FC empírica...")
        comparison_bold = compare_with_empirical_fc(
            results['FC_bold'], FC_empirical
        )
        print(f"   Correlación FC BOLD vs Empírica: {comparison_bold['correlation']:.3f}")
        
        if 'FC_deconv' in results:
            comparison_deconv = compare_with_empirical_fc(
                results['FC_deconv'], FC_empirical
            )
            print(f"   Correlación FC Deconv vs Empírica: {comparison_deconv['correlation']:.3f}")
            
            mejora = (comparison_deconv['correlation'] - comparison_bold['correlation']) * 100
            print(f"   Mejora con deconvolución: {mejora:.1f}%")
    
    # 4. Visualizar resultados
    print("\n4. Generando visualizaciones...")
    fig = pipeline.plot_results(node_indices=[0, 10, 20], save_fig='pipeline_results.png')
    
    # 5. Guardar resultados
    print("\n5. Guardando resultados...")
    pipeline.save_results('dmf_hrf_analysis')
    
    print("\n" + "="*60)
    print("EJEMPLO COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return pipeline


# ============================================================================
# VALIDACIÓN Y TESTING
# ============================================================================

def validate_deconvolution():
    """
    Valida que la deconvolución funciona correctamente
    """
    print("Validando deconvolución...")
    
    # Crear señal de prueba
    n_points = 500
    TR = 1.0
    
    # Señal neuronal (eventos)
    events = np.zeros(n_points)
    events[::50] = 1  # Evento cada 50 puntos
    
    # HRF conocida
    hrf_true, _ = create_canonical_hrf(TR, 32)
    
    # Convolucionar para crear BOLD
    bold_synthetic = np.convolve(events, hrf_true, mode='same')
    
    # Añadir ruido
    bold_noisy = bold_synthetic + 0.1 * np.random.randn(n_points)
    
    # Deconvolucionar
    deconv_iter = wiener_deconv_iterative(bold_noisy, hrf_true, n_iterations=100)
    deconv_direct = wiener_deconv_direct(bold_noisy, hrf_true, regularization=0.1)
    
    # Calcular correlación con eventos originales
    corr_iter = np.corrcoef(events, deconv_iter)[0, 1]
    corr_direct = np.corrcoef(events, deconv_direct)[0, 1]
    
    print(f"Correlación deconv iterativa vs eventos: {corr_iter:.3f}")
    print(f"Correlación deconv directa vs eventos: {corr_direct:.3f}")
    
    # Visualizar
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    
    axes[0].plot(events, 'k')
    axes[0].set_title('Eventos Neuronales Originales')
    axes[0].set_ylabel('Amplitud')
    
    axes[1].plot(bold_noisy, 'b')
    axes[1].set_title('Señal BOLD (con ruido)')
    axes[1].set_ylabel('Amplitud')
    
    axes[2].plot(deconv_iter, 'g')
    axes[2].set_title(f'Deconvolución Iterativa (corr={corr_iter:.3f})')
    axes[2].set_ylabel('Amplitud')
    
    axes[3].plot(deconv_direct, 'r')
    axes[3].set_title(f'Deconvolución Directa (corr={corr_direct:.3f})')
    axes[3].set_xlabel('Tiempo (TR)')
    axes[3].set_ylabel('Amplitud')
    
    plt.tight_layout()
    plt.show()
    
    return corr_iter > 0.7 and corr_direct > 0.5  # Criterio de validación


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Validar deconvolución
    print("PASO 1: Validación de algoritmos")
    print("-"*40)
    if validate_deconvolution():
        print("✓ Validación exitosa\n")
    else:
        print("✗ Validación fallida\n")
    
    # Ejecutar ejemplo completo
    print("\nPASO 2: Ejecutar pipeline completo")
    print("-"*40)
    pipeline = example_usage()
    
    print("\n¡Script ejecutado exitosamente!")
    print("Archivos generados:")
    print("  - pipeline_results.png")
    print("  - dmf_hrf_analysis.npz")
    print("  - dmf_hrf_analysis.mat")