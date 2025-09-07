"""
Optimización del parámetro G (Global Coupling) para el modelo DMF
Script independiente para encontrar el valor óptimo de G

@author: Diego Garrido
"""

import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
import DMF
import BOLDModel as bd
import time

def prepare_dmf_model():
    """Prepara el modelo DMF con parámetros iniciales"""
    
    # Parámetros de simulación
    DMF.tmax = 780  # tiempo en segundos
    DMF.dt = 0.001  # paso de integración
    DMF.downsampling = 1000  # downsampling BOLD
    DMF.downsampling_rates = 10  # downsampling firing rates
    
    # Cargar matriz de conectividad estructural
    try:
        struct = np.loadtxt("SC_opti_25julio.txt")
    except:
        print("No se pudo cargar SC_opti_25julio.txt, usando matriz aleatoria")
        struct = np.random.uniform(size=(90, 90))
    
    DMF.SC = struct / np.mean(np.sum(struct, 0))
    DMF.nnodes = len(DMF.SC)
    
    # Cargar conectividad funcional empírica
    try:
        FCe = np.load("average_90x90FC_HCPchina_symm.npy")
    except:
        print("No se pudo cargar FC empírica, usando matriz aleatoria")
        FCe = np.random.uniform(-0.5, 0.5, size=(90, 90))
        FCe = (FCe + FCe.T) / 2  # Hacer simétrica
        np.fill_diagonal(FCe, 1)
    
    # Parámetros del modelo
    DMF.sigma = 0.4  # Factor de ruido
    
    return FCe

def process_bold_signals(BOLD_signals, BOLD_dt=0.1):
    """Procesa las señales BOLD: filtra y calcula FC"""
    
    # Remover transitorio inicial
    BOLD_signals = BOLD_signals[int(120/BOLD_dt):, :]
    
    # Filtrado pasa-banda
    a0, b0 = signal.bessel(3, [2 * BOLD_dt * 0.01, 2 * BOLD_dt * 0.1], 
                           btype='bandpass')
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis=0)
    
    # Remover más transitorio
    BOLD_filt = BOLD_filt[int(60/BOLD_dt):-int(60/BOLD_dt)]
    
    # Calcular conectividad funcional
    FC = np.corrcoef(BOLD_filt.T)
    
    return BOLD_filt, FC

def run_simulation_with_G(G_value, FCe, verbose=False):
    """
    Ejecuta simulación con un valor específico de G
    
    Parameters:
    -----------
    G_value : float
        Valor del parámetro de acoplamiento global
    FCe : array
        Matriz de conectividad funcional empírica
    verbose : bool
        Si imprimir información durante la simulación
    
    Returns:
    --------
    correlation : float
        Correlación entre FC simulada y empírica
    euclidean : float
        Distancia euclidiana entre FC simulada y empírica
    FC : array
        Matriz de conectividad funcional simulada
    """
    
    # Configurar G y actualizar modelo
    DMF.G = G_value
    DMF.update()
    
    # Ejecutar simulación
    try:
        BOLD_signals, rates, t = DMF.Sim(verbose=verbose, return_rates=True)
    except Exception as e:
        print(f"Error en simulación con G={G_value}: {e}")
        return -1, np.inf, None
    
    # Procesar señales BOLD
    BOLD_dt = DMF.dt * DMF.downsampling
    BOLD_filt, FC = process_bold_signals(BOLD_signals, BOLD_dt)
    
    # Calcular métricas
    FCvecE = FCe[np.tril_indices_from(FCe, -1)]
    FCvecS = FC[np.tril_indices_from(FC, -1)]
    
    correlation = stats.pearsonr(FCvecE, FCvecS)[0]
    euclidean = np.sqrt(np.sum((FCvecE - FCvecS)**2))
    
    return correlation, euclidean, FC

def optimize_G(G_range=(0.7, 1.2), step=0.1, plot_results=True):
    """
    Optimiza el parámetro G buscando la mejor correlación con datos empíricos
    
    Parameters:
    -----------
    G_range : tuple
        Rango de valores de G a explorar (min, max)
    step : float
        Paso entre valores de G
    plot_results : bool
        Si graficar los resultados
    
    Returns:
    --------
    best_G : float
        Valor óptimo de G
    best_correlation : float
        Mejor correlación obtenida
    best_FC : array
        Matriz FC con el mejor G
    results : dict
        Diccionario con todos los resultados
    """
    
    print("Preparando modelo DMF...")
    FCe = prepare_dmf_model()
    
    # Vectores para almacenar resultados
    G_values = np.arange(G_range[0], G_range[1], step)
    correlations = []
    euclideans = []
    
    best_G = None
    best_correlation = -1
    best_FC = None
    
    print(f"\nOptimizando G en rango [{G_range[0]}, {G_range[1]}] con paso {step}")
    print("-" * 50)
    
    for G in G_values:
        print(f"Probando G = {G:.2f}...", end=" ")
        corr, eucl, FC = run_simulation_with_G(G, FCe, verbose=False)
        
        correlations.append(corr)
        euclideans.append(eucl)
        
        print(f"Correlación = {corr:.3f}, Distancia = {eucl:.3f}")
        
        if corr > best_correlation:
            best_correlation = corr
            best_G = G
            best_FC = FC
            print(f"  → Nuevo mejor!")
    
    print("-" * 50)
    print(f"\nRESULTADO ÓPTIMO:")
    print(f"G óptimo = {best_G:.2f}")
    print(f"Correlación máxima = {best_correlation:.3f}")
    
    # Graficar resultados si se solicita
    if plot_results:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Gráfico de optimización
        ax1 = axes[0, 0]
        ax1.plot(G_values, correlations, 'b-o', label='Correlación')
        ax1.axvline(best_G, color='r', linestyle='--', label=f'Óptimo G={best_G:.2f}')
        ax1.set_xlabel('G (Global Coupling)')
        ax1.set_ylabel('Correlación de Pearson')
        ax1.set_title('Optimización del parámetro G')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico de distancia euclidiana
        ax2 = axes[0, 1]
        ax2.plot(G_values, euclideans, 'r-o', label='Distancia Euclidiana')
        ax2.axvline(best_G, color='b', linestyle='--', label=f'Óptimo G={best_G:.2f}')
        ax2.set_xlabel('G (Global Coupling)')
        ax2.set_ylabel('Distancia Euclidiana')
        ax2.set_title('Distancia vs G')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Scatter plot correlación vs distancia
        ax3 = axes[0, 2]
        sc = ax3.scatter(correlations, euclideans, c=G_values, cmap='viridis')
        ax3.scatter(best_correlation, euclideans[correlations.index(best_correlation)], 
                   color='red', s=100, marker='*', label='Óptimo')
        ax3.set_xlabel('Correlación')
        ax3.set_ylabel('Distancia Euclidiana')
        ax3.set_title('Trade-off Correlación-Distancia')
        plt.colorbar(sc, ax=ax3, label='G')
        ax3.legend()
        
        # FC Empírica
        ax4 = axes[1, 0]
        im1 = ax4.imshow(FCe, cmap='jet', vmin=-1, vmax=1)
        ax4.set_title('FC Empírica')
        ax4.set_xlabel('Región')
        ax4.set_ylabel('Región')
        plt.colorbar(im1, ax=ax4)
        
        # FC Simulada óptima
        if best_FC is not None:
            ax5 = axes[1, 1]
            im2 = ax5.imshow(best_FC, cmap='jet', vmin=-1, vmax=1)
            ax5.set_title(f'FC Simulada (G={best_G:.2f})')
            ax5.set_xlabel('Región')
            ax5.set_ylabel('Región')
            plt.colorbar(im2, ax=ax5)
            
            # Diferencia
            ax6 = axes[1, 2]
            diff = best_FC - FCe
            im3 = ax6.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            ax6.set_title('Diferencia (Simulada - Empírica)')
            ax6.set_xlabel('Región')
            ax6.set_ylabel('Región')
            plt.colorbar(im3, ax=ax6)
        
        plt.tight_layout()
        plt.show()
    
    # Preparar diccionario de resultados
    results = {
        'G_values': G_values,
        'correlations': correlations,
        'euclideans': euclideans,
        'best_G': best_G,
        'best_correlation': best_correlation,
        'best_FC': best_FC,
        'FCe': FCe
    }
    
    return best_G, best_correlation, best_FC, results

# MAIN: Ejecutar optimización
if __name__ == "__main__":
    
    # Configurar parámetros de búsqueda
    G_range = (0.7, 1.2)  # Rango de G a explorar
    step = 0.01  # Paso entre valores
    
    # Ejecutar optimización
    best_G, best_corr, best_FC, results = optimize_G(
        G_range=G_range,
        step=step,
        plot_results=True
    )
    
    # Guardar resultados
    np.save('optimization_results.npy', results)
    print(f"\nResultados guardados en 'optimization_results.npy'")
    
    # Análisis adicional opcional
    print("\n" + "="*50)
    print("ANÁLISIS ADICIONAL:")
    print("="*50)
    
    if best_FC is not None:
        # Calcular algunas métricas adicionales
        FCe = results['FCe']
        
        # Correlación por región (strength)
        strength_emp = np.mean(np.abs(FCe), axis=0)
        strength_sim = np.mean(np.abs(best_FC), axis=0)
        strength_corr = np.corrcoef(strength_emp, strength_sim)[0, 1]
        
        print(f"Correlación de strength nodal: {strength_corr:.3f}")
        
        # Error cuadrático medio
        mse = np.mean((best_FC - FCe)**2)
        print(f"Error cuadrático medio: {mse:.4f}")
        
        # Rango de valores FC
        print(f"Rango FC empírica: [{FCe.min():.3f}, {FCe.max():.3f}]")
        print(f"Rango FC simulada: [{best_FC.min():.3f}, {best_FC.max():.3f}]")


