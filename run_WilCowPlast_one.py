# -*- coding: utf-8 -*-.
"""
The Huber_braun neuronal model function.

@author: porio
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from anarpy.models import netWilsonCowanPlastic as WC
from anarpy.utils.FCDutil import fcd
from anarpy.utils.FCDutil import FCcluster
from enigmatoolbox import datasets



filename = 'sw-18/05-smallworld-18-p0.10'
# filename = 'mod8-18/06-mod8-18-0.0070'
# filename = 'hier-18/04-hier12-18-r0.005'

# net = np.loadtxt(f"../newNets/{filename}.txt")
# net = datasets.load_sc_as_one(parcellation='aparc')[0]
net = datasets.load_sc(parcellation='aparc')[0]
net = net/np.max(net)
net[net<0]=0

WC.tTrans=50  #transient removal with accelerated plasticity
WC.tstop=102   # actual simulation

# WC.G=0.16 #Connectivity strength
# WC.D=0.002       #noise factor
# WC.rhoE=0.11   #target value for mean excitatory activation
WC.dt=0.002   # Sampling interval

### DETERMINISTIC SIM ###
WC.G=0.08  #Connectivity strength
WC.D=0.002      #noise factor
WC.rhoE=0.14   #target value for mean excitatory activation
WC.tau_ip=2

np.random.seed(1)
nnodes=len(net)
WC.N=nnodes
WC.CM = net
# WC.P=np.random.uniform(0.3,0.4,nnodes)
WC.P=.45



#%%

Vtrace,time=WC.Sim(verbose=True)    

# These lines can be adapted to write the parameters to a text file
# print(str(WC.ParamsNode()).replace(", '","\n '"))
# print(str(WC.ParamsSim()).replace(", '","\n '"))
# print(str(WC.ParamsNet()).replace(", '","\n '"))
       
E_t=Vtrace[:,0,:]

#%%
# spec=np.abs(np.fft.fft(E_t-np.mean(E_t,0),axis=0))
# freqs=np.fft.fftfreq(len(E_t),WC.dt)
freqs,spec=signal.welch(E_t,fs=1/WC.dt, nperseg=1000, nfft=4000,noverlap=100,axis=0)

b,a=signal.bessel(4,[5*2*WC.dt, 15*2*WC.dt],btype='bandpass')
E_filt=signal.filtfilt(b,a,E_t,axis=0)

analytic=signal.hilbert(E_filt,axis=0)

remove_time = 1  # seconds at beggining and end
remove_samples = int(remove_time/WC.dt)

Trun = WC.tstop - 2*remove_time

envelope=np.abs(analytic[remove_samples:-remove_samples,:])
phase = np.angle(analytic[remove_samples:-remove_samples,:])

FC=np.corrcoef(envelope,rowvar=False)

# FCphase=fcd.phaseFC(phase)


#%%
        
phasesynch=np.abs(np.mean(np.exp(1j*phase),1))
MPsync=np.mean(phasesynch)  #Media de la fase en el tiempo
VarPsync=np.var(phasesynch)  #Varianza de la fase en el tiempo


plt.figure(104,figsize=(10,10))
plt.clf()
    
plt.subplot2grid((5,5),(0,0),rowspan=1,colspan=5)
plt.plot(time[:-remove_samples*2], phasesynch)
plt.title('mean P sync')
  
plt.subplot2grid((5,5),(2,4))
plt.imshow(FC,cmap='jet',vmax=1,vmin=-1,interpolation='none')
plt.gca().set_xticks(())
plt.gca().set_yticks(())
plt.title('Static FC - envel')

# plt.subplot2grid((5,5),(3,4))
# plt.imshow(FCphase,cmap='jet',vmax=1,vmin=0,interpolation='none')
# plt.gca().set_xticks(())
# plt.gca().set_yticks(())
# plt.title('Static FC - phase')
    
plt.subplot2grid((5,5),(1,4))
plt.imshow(WC.CM,cmap='gray_r')
plt.title('SC')

# FCempiric = datasets.load_fc_as_one(parcellation='aparc')[0]
# FCempiric = datasets.load_fc(parcellation='aparc')[0]

plt.subplot2grid((5,5),(3,4))
# plt.imshow(FCempiric,cmap='jet',vmax=1,vmin=-1,interpolation='none')
plt.gca().set_xticks(())
plt.gca().set_yticks(())
plt.title('Static FC - empiric')



#%%
# FCD,Pcorr,shift=fcd.extract_FCD(envelope.T,maxNwindows=2000,wwidth=WW,olap=0.75,
#                                 mode='corr',modeFCD='euclidean')
WW=5000  # Window size in samples

FCD,Pcorr,shift=fcd.extract_FCD(envelope.T,maxNwindows=2000,wwidth=WW,olap=0.75,
                                mode='corr',modeFCD='corr')

varFCD = np.var(FCD[np.triu_indices(len(FCD),k=4)])
viscosity = np.mean(np.diagonal(FCD,4))
    
plt.subplot2grid((5,5),(1,0),rowspan=3,colspan=3)
plt.imshow(1-FCD,vmin=0,extent=(0,Trun,Trun,0),interpolation='none',cmap='jet')
plt.title(f'FCD envel W{WW}')
plt.colorbar()


windows=[int(len(Pcorr)*f) for f in (0.18, 0.36, 0.54, 0.72, 0.9)]
axes2=[plt.subplot2grid((5,5),(4,pos)) for pos in range(5)]
for axi,ind in zip(axes2,windows):
    corrMat=np.zeros((nnodes,nnodes))
    corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[ind]
    corrMat+=corrMat.T
    corrMat+=np.eye(nnodes)
        
    axi.imshow(corrMat,vmin=-1,vmax=1,interpolation='none',cmap='jet')
        
    axi.set_xticks(())
    axi.set_yticks(())
    
    axi.set_title('t=%.4g'%(ind*Trun/len(Pcorr)))
    axi.grid()


print(np.var(FCD[np.triu_indices(len(FCD),k=4)]))
print(np.mean(np.diagonal(FCD,1)))


#%%

plt.figure(1,figsize=(10,8))
plt.clf()
plt.subplot(321)
plt.plot(time,Vtrace[:,0,::4])
plt.ylabel('E')

plt.subplot(323)
plt.plot(time,Vtrace[:,2,:])
plt.xlabel('time')
plt.ylabel('a_ie')

plt.subplot(325)
plt.loglog(freqs[:1200],spec[:1200,::4])
plt.xlabel('frequency (Hz)')
plt.ylabel('abs')


plt.subplot(222)
plt.imshow(WC.CM,cmap='gray_r')
plt.title("structural connectivity")
# plt.yticks((0,5,10,15))

plt.subplot(224)
plt.imshow(FC,cmap='jet',vmin=-1,vmax=1)
plt.colorbar()
plt.title("Envelope correlation (FC)")
# plt.yticks((0,5,10,15))

plt.subplots_adjust(hspace=0.3)


#%%
plt.figure(3)
plt.clf()

freq2,spec2=signal.welch(envelope, fs=1/WC.dt, nperseg=5000, nfft=20000,noverlap=100,axis=0)

plt.subplot(311)
plt.plot(time[:50000],envelope[:,::10])


plt.subplot(312)
plt.plot(freq2,spec2[:,::5])
plt.xlim((0,30))


plt.subplot(313)
plt.loglog(freq2,spec2[:,::5])
# plt.xlim((0,30))


#%%

FCcluster.FCcluster(Pcorr,'PCA',6,varexp=0.5,minmax=[-1,1],Trun=100)



