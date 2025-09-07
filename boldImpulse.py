# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:20:56 2025

@author: Patricio Orio
"""
#%% Libraries

import numpy as np
import BOLDModel as bd
import matplotlib.pyplot as plt


#%% Parameters

nnodes = 1
dt = 0.01
tstop = 50
time=np.arange(0,tstop,dt)

x_t = np.zeros_like(time)
x_t[time==35]=20

# BOLD1
BOLD = bd.Sim(x_t[:,None], nnodes, dt) # Sim(re, nnodes, dt)

# Change Variables
bd.itauf = 5        # Gamma
bd.itaus = 1/0.8    # Kappa

# Recompile with new values
bd.update()

# BOLD2
BOLD2 = bd.Sim(x_t[:,None], nnodes, dt) # Sim(re, nnodes, dt)


#%% Plots

plt.figure(1)
plt.clf()
ax1 = plt.subplot(111)
ax1.plot(time,BOLD)
ax1.plot(time,BOLD2)

ax2 = ax1.twinx()
ax2.plot(time,x_t, 'g')
ax2.set_ylim((-10,250))
ax1.set_ylim((-0.0005,0.0025))
plt.xlabel("Tiempo (s)")
ax1.set_ylabel("Señal BOLD", color='C0')
ax2.set_ylabel("Estímulo", color='g')

plt.tight_layout()