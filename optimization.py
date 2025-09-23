"""
Función para optimizar el parámetro G (Global Coupling) del modelo DMF
"""
#%% Libraries
import numpy as np
from scipy import signal, stats
import DMF
import matplotlib.pyplot as plt


#%% Functions

def optimize_G(G_range=(0.8, 2.0), step=0.1, verbose=True, plot_results=True):
    """
    Optimiza el parámetro G del modelo DMF
    
    Parameters:
    -----------
    G_range : tuple
        Rango de valores de G (min, max)
    step : float
        Paso entre valores de G
    verbose : bool
        Si mostrar información durante la optimización
    plot_results : bool
        Si generar los gráficos de resultados
    
    Returns:
    --------
    best_G : float
        Valor óptimo de G
    best_correlation : float
        Mejor correlación obtenida
    best_FC : array
        Matriz FC con el mejor G
    """
    
    # Configurar parámetros del modelo DMF
    DMF.tmax = 780
    DMF.dt = 0.001
    DMF.downsampling = 1000
    DMF.downsampling_rates = 10
    
    # Cargar matriz de conectividad estructural
    struct = np.loadtxt("SC_opti_25julio.txt")
    DMF.SC = struct / np.mean(np.sum(struct, 0))
    DMF.nnodes = len(DMF.SC)
    
    # Cargar FC empírica
    FCe = np.load("average_90x90FC_HCPchina_symm.npy")
    
    # Configurar parámetros del modelo
    DMF.sigma = 0.4
    
    # Variables para el filtrado
    BOLD_dt = DMF.dt * DMF.downsampling
    a0, b0 = signal.bessel(3, [2 * BOLD_dt * 0.01, 2 * BOLD_dt * 0.1], btype='bandpass')
    
    # Variables para almacenar resultados
    best_G = None
    best_correlation = -1
    best_FC = None
    G_values = []
    correlations = []
    
    # Búsqueda del óptimo
    for G in np.arange(G_range[0], G_range[1], step):
        if verbose:
            print(f"Probando G = {G:.2f}")
        
        # Actualizar G y modelo
        DMF.G = G
        DMF.update()
        
        # Ejecutar simulación
        BOLD_signals, rates, t = DMF.Sim(verbose=False, return_rates=True)
        
        # Procesar señales BOLD
        BOLD_signals = BOLD_signals[int(120/BOLD_dt):, :]
        BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis=0)
        BOLD_filt = BOLD_filt[int(60/BOLD_dt):-int(60/BOLD_dt)]
        
        # Calcular FC
        FC = np.corrcoef(BOLD_filt.T)
        
        # Calcular correlación con FC empírica
        FCvecE = FCe[np.tril_indices_from(FCe, -1)]
        FCvecS = FC[np.tril_indices_from(FC, -1)]
        correlation = stats.pearsonr(FCvecE, FCvecS)[0]
        
        # Almacenar resultados
        G_values.append(G)
        correlations.append(correlation)
        
        if verbose:
            euclidean = np.sqrt(np.sum((FCvecE - FCvecS)**2))
            print(f"  Correlación = {correlation:.3f}, Distancia = {euclidean:.3f}")
        
        # Actualizar mejor resultado
        if correlation > best_correlation:
            best_correlation = correlation
            best_G = G
            best_FC = FC.copy()
            if verbose:
                print("Nuevo mejor")
    
    if verbose:
        print(f"\nÓptimo: G = {best_G}, correlación = {best_correlation:.3f}")
    
    # Generar plots
    if plot_results and best_FC is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Correlación vs G
        ax1 = axes[0]
        ax1.plot(G_values, correlations, 'b-o', linewidth=2, markersize=6)
        ax1.axvline(best_G, color='r', linestyle='--', linewidth=2, 
                   label=f'G óptimo = {best_G:.2f}')
        ax1.axhline(best_correlation, color='r', linestyle=':', alpha=0.5)
        ax1.set_xlabel('G (Global Coupling)', fontsize=12)
        ax1.set_ylabel('Correlación de Pearson', fontsize=12)
        ax1.set_title(f'Optimización de G\n(Máx. correlación = {best_correlation:.3f})', 
                     fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: FC Empírica
        ax2 = axes[1]
        im2 = ax2.imshow(FCe, cmap='jet', vmin=-1, vmax=1, aspect='auto')
        ax2.set_title('FC Empírica', fontsize=13)
        ax2.set_xlabel('Región', fontsize=12)
        ax2.set_ylabel('Región', fontsize=12)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Plot 3: FC Simulada con G óptimo
        ax3 = axes[2]
        im3 = ax3.imshow(best_FC, cmap='jet', vmin=-1, vmax=1, aspect='auto')
        ax3.set_title(f'FC Simulada (G = {best_G:.2f})', fontsize=13)
        ax3.set_xlabel('Región', fontsize=12)
        ax3.set_ylabel('Región', fontsize=12)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    return best_G, best_correlation, best_FC

#%% Run Function

if __name__ == "__main__":
    G_optimo, correlacion_max, FC_optimo = optimize_G(
        G_range=(1.0, 1.2), 
        step=0.01,
        verbose=True,
        plot_results=True
    )

# G Optimo: 1.07