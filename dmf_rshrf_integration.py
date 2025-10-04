# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 22:45:32 2025

@author: yangy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integración de DMF con rsHRF
Pipeline completo: Simulación DMF → Estimación HRF → Deconvolución
"""

import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt

# Importar tus módulos DMF
import DMF
import BOLDModel as BD

# Importar las funciones rsHRF implementadas
from rsHRF_core_functions import (
    process_bold_to_deconv,
    canonical_hrf,
    rsHRF_estimation_temporal_basis,
    rsHRF_iterative_wiener_deconv,
    rsHRF_get_HRF_parameters,
    bandpass_filter,
    zscore
)

# Si tienes las clases rsHRF disponibles
try:
    from Scripts.hrf import HRF
    from Scripts.bold_deconv import Bold_Deconv
    from Scripts.parameters import Parameters
    CLASSES_AVAILABLE = True
except ImportError:
    CLASSES_AVAILABLE = False
    print("Clases rsHRF no disponibles, usando funciones standalone")


def run_dmf_to_deconv_pipeline(SC_path="SC_opti_25julio.txt", 
                               FC_empirical_path="average_90x90FC_HCPchina_symm.npy",
                               G=1.1, sigma=0.4):
    """
    Pipeline completo DMF → rsHRF
    
    Parameters:
    -----------
    SC_path : str
        Path a la matriz de conectividad estructural
    FC_empirical_path : str
        Path a la FC empírica para comparación
    G : float
        Acoplamiento global
    sigma : float
        Nivel de ruido
    
    Returns:
    --------
    results : dict
        Resultados completos del pipeline
    """
    
    # =========================================================================
    # 1. CONFIGURACIÓN Y SIMULACIÓN DMF
    # =========================================================================
    print("="*60)
    print("1. CONFIGURANDO Y EJECUTANDO SIMULACIÓN DMF")
    print("="*60)
    
    # Cargar conectividad estructural
    struct = np.loadtxt(SC_path)
    DMF.SC = struct / np.mean(np.sum(struct, 0))
    DMF.nnodes = len(DMF.SC)
    
    # Cargar FC empírica si está disponible
    try:
        FCe = np.load(FC_empirical_path)
    except:
        FCe = None
        print("FC empírica no disponible")
    
    # Parámetros de simulación
    DMF.tmax = 660  # Tiempo total (segundos)
    DMF.dt = 0.001  # Step de integración
    DMF.downsampling = 1000  # Para obtener TR efectivo de 1s
    DMF.downsampling_rates = 10
    
    # Parámetros del modelo
    DMF.G = G
    DMF.sigma = sigma
    
    # Actualizar modelo
    DMF.update()
    
    # Simular
    print(f"Simulando con G={G}, sigma={sigma}")
    BOLD_signals, rates, t = DMF.Sim(verbose=True, return_rates=True)
    
    # Calcular TR efectivo
    BOLD_dt = DMF.dt * DMF.downsampling
    print(f"TR efectivo: {BOLD_dt} segundos")
    
    # =========================================================================
    # 2. PREPROCESAMIENTO BOLD
    # =========================================================================
    print("\n" + "="*60)
    print("2. PREPROCESANDO SEÑALES BOLD")
    print("="*60)
    
    # Remover transitorio inicial (120s) y final (60s)
    remove_initial = int(120 / BOLD_dt)
    remove_final = int(60 / BOLD_dt)
    BOLD_signals = BOLD_signals[remove_initial:-remove_final, :]
    
    # Filtrado pasa-banda (0.01 - 0.1 Hz)
    print("Aplicando filtro pasa-banda (0.01-0.1 Hz)...")
    BOLD_filtered = bandpass_filter(BOLD_signals, 0.01, 0.1, 1/BOLD_dt)
    
    # Normalización z-score
    BOLD_normalized = zscore(BOLD_filtered, axis=0)
    
    print(f"Forma de datos BOLD procesados: {BOLD_normalized.shape}")
    
    # =========================================================================
    # 3. ESTIMACIÓN HRF Y DECONVOLUCIÓN
    # =========================================================================
    print("\n" + "="*60)
    print("3. ESTIMACIÓN HRF Y DECONVOLUCIÓN")
    print("="*60)
    
    # Configurar parámetros para rsHRF
    para_rsHRF = {
        'TR': BOLD_dt,
        'T': 1,  # Sin upsampling
        'dt': BOLD_dt,
        'len': 32,  # Longitud HRF en segundos
        'thr': 1,  # Threshold: mean + 1*std
        'lag': np.arange(int(4/BOLD_dt), int(8/BOLD_dt)),  # 4-8 segundos de lag
        'AR_lag': 1,
        'order': 3,
        'name': 'Canonical HRF (with time and dispersion derivatives)'
    }
    
    # Opción A: Usar pipeline completo
    print("Ejecutando pipeline rsHRF completo...")
    results_rsHRF = process_bold_to_deconv(
        BOLD_normalized, 
        TR=BOLD_dt,
        estimation_type='canon2dd',
        deconv_type='iterative',
        passband=[0.01, 0.1]
    )
    
    # =========================================================================
    # 4. ANÁLISIS DE CONECTIVIDAD
    # =========================================================================
    print("\n" + "="*60)
    print("4. ANÁLISIS DE CONECTIVIDAD")
    print("="*60)
    
    # FC de señales BOLD originales filtradas
    FC_bold = np.corrcoef(BOLD_filtered.T)
    
    # FC de señales deconvolucionadas
    FC_deconv = np.corrcoef(results_rsHRF['deconv_signals'].T)
    
    # Comparación con FC empírica si está disponible
    if FCe is not None:
        # Vectorizar matrices para correlación
        FCvecE = FCe[np.tril_indices_from(FCe, -1)]
        FCvecB = FC_bold[np.tril_indices_from(FC_bold, -1)]
        FCvecD = FC_deconv[np.tril_indices_from(FC_deconv, -1)]
        
        corr_bold_empirical = stats.pearsonr(FCvecE, FCvecB)[0]
        corr_deconv_empirical = stats.pearsonr(FCvecE, FCvecD)[0]
        
        print(f"Correlación FC BOLD vs Empírica: {corr_bold_empirical:.3f}")
        print(f"Correlación FC Deconv vs Empírica: {corr_deconv_empirical:.3f}")
    
    # =========================================================================
    # 5. INTEGRACIÓN CON CLASES rsHRF (SI ESTÁN DISPONIBLES)
    # =========================================================================
    if CLASSES_AVAILABLE:
        print("\n" + "="*60)
        print("5. INTEGRANDO CON CLASES rsHRF")
        print("="*60)
        
        # Crear objeto Parameters
        params = Parameters()
        params.set_TR(BOLD_dt)
        params.set_estimation('canon2dd')
        params.set_passband("0.01,0.1")
        params.set_len(32)
        params.set_thr(1)
        
        # Crear objeto HRF
        hrf_obj = HRF(
            label="DMF_Estimated_HRF",
            ts=results_rsHRF['hrf'],
            subject_index="DMF_simulation",
            para=params
        )
        hrf_obj.set_para(results_rsHRF['hrf_params'])
        hrf_obj.set_event_bold(results_rsHRF['events'])
        
        # Crear objeto Bold_Deconv
        deconv_obj = Bold_Deconv(
            label="DMF_Deconvolved_BOLD",
            ts=results_rsHRF['deconv_signals'],
            subject_index="DMF_simulation",
            para=params
        )
        deconv_obj.set_HRF(hrf_obj)
        deconv_obj.set_event_num([len(e) for e in results_rsHRF['events']])
        
        # Guardar en formato .mat
        print("Guardando resultados en formato .mat...")
        hrf_obj.save_info("DMF_HRF_results.mat")
        deconv_obj.save_info("DMF_deconv_results.mat")
    
    # =========================================================================
    # 6. COMPILAR RESULTADOS
    # =========================================================================
    results = {
        # Datos DMF originales
        'BOLD_signals': BOLD_signals,
        'BOLD_filtered': BOLD_filtered,
        'rates': rates,
        'time': t,
        
        # Resultados rsHRF
        'deconv_signals': results_rsHRF['deconv_signals'],
        'hrf': results_rsHRF['hrf'],
        'hrf_params': results_rsHRF['hrf_params'],
        'events': results_rsHRF['events'],
        
        # Matrices FC
        'FC_bold': FC_bold,
        'FC_deconv': FC_deconv,
        'FC_empirical': FCe,
        
        # Parámetros
        'parameters': {
            'DMF': {'G': G, 'sigma': sigma, 'dt': DMF.dt},
            'rsHRF': para_rsHRF
        }
    }
    
    if CLASSES_AVAILABLE:
        results['hrf_object'] = hrf_obj
        results['deconv_object'] = deconv_obj
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return results


def visualize_results(results, node_indices=[0, 5, 10]):
    """
    Visualización completa de resultados
    
    Parameters:
    -----------
    results : dict
        Diccionario de resultados del pipeline
    node_indices : list
        Índices de nodos a visualizar
    """
    
    fig = plt.figure(figsize=(16, 12))
    
    # =========================================================================
    # Panel 1: Señales BOLD originales vs deconvolucionadas
    # =========================================================================
    ax1 = plt.subplot(4, 3, 1)
    for idx in node_indices:
        ax1.plot(results['BOLD_filtered'][:, idx], alpha=0.7, 
                label=f'Nodo {idx}')
    ax1.set_title('Señales BOLD Filtradas')
    ax1.set_xlabel('Tiempo (TR)')
    ax1.set_ylabel('Amplitud')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(4, 3, 2)
    for idx in node_indices:
        ax2.plot(results['deconv_signals'][:, idx], alpha=0.7)
    ax2.set_title('Señales Deconvolucionadas')
    ax2.set_xlabel('Tiempo (TR)')
    ax2.set_ylabel('Amplitud')
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 2: HRFs estimadas
    # =========================================================================
    ax3 = plt.subplot(4, 3, 3)
    time_hrf = np.arange(results['hrf'].shape[0]) * results['parameters']['rsHRF']['dt']
    for idx in node_indices:
        ax3.plot(time_hrf, results['hrf'][:, idx], 
                label=f'Nodo {idx}')
    ax3.set_title('HRFs Estimadas')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Amplitud')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 3: Matrices de Conectividad Funcional
    # =========================================================================
    # FC BOLD
    ax4 = plt.subplot(4, 3, 4)
    im4 = ax4.imshow(results['FC_bold'], cmap='jet', vmin=-1, vmax=1)
    ax4.set_title('FC BOLD Filtrada')
    ax4.set_xlabel('Región')
    ax4.set_ylabel('Región')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # FC Deconvolucionada
    ax5 = plt.subplot(4, 3, 5)
    im5 = ax5.imshow(results['FC_deconv'], cmap='jet', vmin=-1, vmax=1)
    ax5.set_title('FC Deconvolucionada')
    ax5.set_xlabel('Región')
    ax5.set_ylabel('Región')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # FC Empírica (si está disponible)
    if results['FC_empirical'] is not None:
        ax6 = plt.subplot(4, 3, 6)
        im6 = ax6.imshow(results['FC_empirical'], cmap='jet', vmin=-1, vmax=1)
        ax6.set_title('FC Empírica')
        ax6.set_xlabel('Región')
        ax6.set_ylabel('Región')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # =========================================================================
    # Panel 4: Parámetros HRF
    # =========================================================================
    # Distribución de Heights
    ax7 = plt.subplot(4, 3, 7)
    ax7.hist(results['hrf_params'][0, :], bins=20, edgecolor='black', alpha=0.7)
    ax7.set_title('Distribución de Amplitudes HRF')
    ax7.set_xlabel('Amplitud')
    ax7.set_ylabel('Frecuencia')
    ax7.axvline(np.mean(results['hrf_params'][0, :]), 
               color='red', linestyle='--', label='Media')
    ax7.legend()
    
    # Distribución de Time to Peak
    ax8 = plt.subplot(4, 3, 8)
    ax8.hist(results['hrf_params'][1, :], bins=20, edgecolor='black', alpha=0.7)
    ax8.set_title('Distribución Time-to-Peak')
    ax8.set_xlabel('Tiempo (s)')
    ax8.set_ylabel('Frecuencia')
    ax8.axvline(np.mean(results['hrf_params'][1, :]), 
               color='red', linestyle='--', label='Media')
    ax8.legend()
    
    # Distribución de FWHM
    ax9 = plt.subplot(4, 3, 9)
    ax9.hist(results['hrf_params'][2, :], bins=20, edgecolor='black', alpha=0.7)
    ax9.set_title('Distribución FWHM')
    ax9.set_xlabel('Ancho (s)')
    ax9.set_ylabel('Frecuencia')
    ax9.axvline(np.mean(results['hrf_params'][2, :]), 
               color='red', linestyle='--', label='Media')
    ax9.legend()
    
    # =========================================================================
    # Panel 5: Comparación de espectros de potencia
    # =========================================================================
    ax10 = plt.subplot(4, 3, 10)
    # Calcular PSD para señales BOLD
    f_bold, psd_bold = signal.welch(results['BOLD_filtered'][:, node_indices[0]], 
                                    fs=1/results['parameters']['rsHRF']['TR'], 
                                    nperseg=min(256, len(results['BOLD_filtered'])//4))
    f_deconv, psd_deconv = signal.welch(results['deconv_signals'][:, node_indices[0]], 
                                        fs=1/results['parameters']['rsHRF']['TR'],
                                        nperseg=min(256, len(results['deconv_signals'])//4))
    
    ax10.semilogy(f_bold, psd_bold, label='BOLD', alpha=0.7)
    ax10.semilogy(f_deconv, psd_deconv, label='Deconv', alpha=0.7)
    ax10.set_title(f'Espectro de Potencia (Nodo {node_indices[0]})')
    ax10.set_xlabel('Frecuencia (Hz)')
    ax10.set_ylabel('PSD')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.set_xlim([0, 0.5])
    
    # =========================================================================
    # Panel 6: Eventos detectados
    # =========================================================================
    ax11 = plt.subplot(4, 3, 11)
    event_counts = [len(e) for e in results['events']]
    ax11.bar(range(len(event_counts)), event_counts)
    ax11.set_title('Número de Eventos por Nodo')
    ax11.set_xlabel('Nodo')
    ax11.set_ylabel('# Eventos')
    ax11.set_xlim([-1, len(event_counts)])
    
    # =========================================================================
    # Panel 7: Diferencia FC
    # =========================================================================
    ax12 = plt.subplot(4, 3, 12)
    fc_diff = results['FC_deconv'] - results['FC_bold']
    im12 = ax12.imshow(fc_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax12.set_title('Diferencia FC (Deconv - BOLD)')
    ax12.set_xlabel('Región')
    ax12.set_ylabel('Región')
    plt.colorbar(im12, ax=ax12, fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def print_summary_statistics(results):
    """
    Imprime estadísticas resumidas
    """
    print("\n" + "="*60)
    print("ESTADÍSTICAS RESUMIDAS")
    print("="*60)
    
    print("\nParámetros HRF:")
    print(f"  Amplitud media: {np.mean(results['hrf_params'][0, :]):.3f} ± "
          f"{np.std(results['hrf_params'][0, :]):.3f}")
    print(f"  Time-to-peak medio: {np.mean(results['hrf_params'][1, :]):.3f} ± "
          f"{np.std(results['hrf_params'][1, :]):.3f} s")
    print(f"  FWHM medio: {np.mean(results['hrf_params'][2, :]):.3f} ± "
          f"{np.std(results['hrf_params'][2, :]):.3f} s")
    
    print("\nEventos detectados:")
    event_counts = [len(e) for e in results['events']]
    print(f"  Total de eventos: {sum(event_counts)}")
    print(f"  Eventos por nodo: {np.mean(event_counts):.1f} ± "
          f"{np.std(event_counts):.1f}")
    
    print("\nConectividad Funcional:")
    print(f"  Media FC BOLD: {np.mean(results['FC_bold']):.3f}")
    print(f"  Media FC Deconv: {np.mean(results['FC_deconv']):.3f}")
    print(f"  Std FC BOLD: {np.std(results['FC_bold']):.3f}")
    print(f"  Std FC Deconv: {np.std(results['FC_deconv']):.3f}")
    
    if results['FC_empirical'] is not None:
        FCvecE = results['FC_empirical'][np.tril_indices_from(results['FC_empirical'], -1)]
        FCvecB = results['FC_bold'][np.tril_indices_from(results['FC_bold'], -1)]
        FCvecD = results['FC_deconv'][np.tril_indices_from(results['FC_deconv'], -1)]
        
        corr_bold = stats.pearsonr(FCvecE, FCvecB)[0]
        corr_deconv = stats.pearsonr(FCvecE, FCvecD)[0]
        
        print(f"\nComparación con FC empírica:")
        print(f"  Correlación BOLD-Empírica: {corr_bold:.3f}")
        print(f"  Correlación Deconv-Empírica: {corr_deconv:.3f}")
        print(f"  Mejora: {(corr_deconv - corr_bold)*100:.1f}%")


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("PIPELINE DMF → rsHRF")
    print("="*60)
    
    # Ejecutar pipeline completo
    try:
        results = run_dmf_to_deconv_pipeline(
            SC_path="SC_opti_25julio.txt",
            FC_empirical_path="average_90x90FC_HCPchina_symm.npy",
            G=1.1,
            sigma=0.4
        )
        
        # Mostrar estadísticas
        print_summary_statistics(results)
        
        # Visualizar resultados
        print("\nGenerando visualizaciones...")
        fig = visualize_results(results, node_indices=[0, 10, 20])
        
        # Guardar resultados
        print("\nGuardando resultados...")
        np.savez('DMF_rsHRF_results.npz', **results)
        print("Resultados guardados en 'DMF_rsHRF_results.npz'")
        
        # Guardar figura
        fig.savefig('DMF_rsHRF_analysis.png', dpi=150, bbox_inches='tight')
        print("Figura guardada en 'DMF_rsHRF_analysis.png'")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Ejecutando con datos sintéticos de prueba...")
        
        # Crear datos sintéticos
        n_nodes = 45
        n_timepoints = 600
        
        # Crear SC sintética
        SC_synthetic = np.random.rand(n_nodes, n_nodes)
        SC_synthetic = (SC_synthetic + SC_synthetic.T) / 2
        np.fill_diagonal(SC_synthetic, 0)
        np.savetxt("SC_synthetic.txt", SC_synthetic)
        
        # Crear FC empírica sintética
        FC_synthetic = np.random.rand(n_nodes, n_nodes)
        FC_synthetic = (FC_synthetic + FC_synthetic.T) / 2
        np.fill_diagonal(FC_synthetic, 1)
        np.save("FC_synthetic.npy", FC_synthetic)
        
        # Ejecutar con datos sintéticos
        results = run_dmf_to_deconv_pipeline(
            SC_path="SC_synthetic.txt",
            FC_empirical_path="FC_synthetic.npy",
            G=0.9,
            sigma=0.3
        )
        
        print_summary_statistics(results)
        fig = visualize_results(results)
    
    print("\n¡Pipeline completado exitosamente!")
    print("="*60)