
import os
import matplotlib.pyplot as plt
import numpy as np
from suite2p.extraction import dcnv
import scipy.stats as st

direc = 'V:\\Rawdata\\PILOTS\\20221108_NSH07429_Spontaneous4ROI\\Spontaneous\\suite2p\\plane0'
direc = 'W:\\Users\\Matthijs\\Rawdata\\LPE09829\\2023_03_29\\suite2p\\plane1'

os.chdir(direc)
# Load the data of this plane:
F       = np.load('F.npy')
Fneu    = np.load('Fneu.npy')
spks    = np.load('spks.npy')
ops     = np.load('ops.npy', allow_pickle=True).item()
iscell  = np.load('iscell.npy')
stat    = np.load('stat.npy', allow_pickle=True)
redcell = np.load('redcell.npy')

#################### Show image of cell footprints #####################
im = np.zeros((ops['Ly'], ops['Lx']))

for n, j in enumerate(iscell[:,0]):
    if j:
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        im[ypix,xpix] = n+1

plt.imshow(im)
plt.show()

####### Neuropil correction:
Fc          = F.copy() - 0.7*Fneu

## Sliding baseline correction:
Fb = dcnv.preprocess(Fc, ops['baseline'], ops['win_baseline'], 
                                   ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])

# Helper function for delta F/F0:
def calculate_dff(F, Fneu, prc=20):
    # correct trace for neuropil contamination:
    Fc = F - 0.7 * Fneu + np.median(Fneu,axis=1,keepdims=True)
    # Establish baseline as percentile of corrected trace (50 is median)
    F0 = np.percentile(Fc,prc,axis=1,keepdims=True)
    #Compute dF / F0: 
    dFF = (Fc - F0) / F0
    return dFF

dFF     = calculate_dff(F, Fneu,prc=20)
zF      = st.zscore(dFF.copy(),axis=1)

######## plot traces with and without neuropil correction #########

#pick an exemplary neuron: 
neuronidx   = 36 #15
timeidx     = np.arange(5000,6000)

plt.figure(figsize=(15,4))
plt.plot(F[neuronidx,timeidx],'r',linewidth=0.5)
plt.plot(Fneu[neuronidx,timeidx],'k',linewidth=0.5)
plt.plot(Fc[neuronidx,timeidx],'b',linewidth=0.5)
plt.plot(Fb[neuronidx,timeidx],'g',linewidth=0.5)

######## plot deltaF/F0 traces versus z-scored traces: #########

plt.figure(figsize=(15,4))
plt.plot(dFF[neuronidx,timeidx],'r',linewidth=0.5)
plt.plot(zF[neuronidx,timeidx],'b',linewidth=0.5)

######## plot deltaF/F0 traces versus inferred spike rate through deconvolution #########
spks_norm = spks / spks.max(axis=1, keepdims=1)

plt.figure(figsize=(15,4))
plt.plot(dFF[neuronidx,timeidx],'r',linewidth=0.5)
# plt.plot(spks[neuronidx,timeidx] / np.max(spks[neuronidx,timeidx]),'k',linewidth=1)
plt.plot(spks_norm[neuronidx,timeidx],'k',linewidth=1)

selec = np.array([15,36,26,500,512,415,416,136,139,522])
temp = np.arange(len(selec))
spksselec = spks_norm[selec[:,np.newaxis],timeidx] + temp[:,np.newaxis]

plt.figure()
plt.plot(spksselec.T,linewidth=0.5)


###################### Calculate the noise level of the cells ####
# Rupprecht et al. 2021 Nat Neurosci.

noise_level = np.median(np.abs(np.diff(dFF,axis=1)),axis=1)/np.sqrt(ops['fs'])
peak_dFF = np.max(dFF,axis=1)

plt.figure(figsize=(5,4))
plt.scatter(peak_dFF[iscell[:,0]==1],noise_level[iscell[:,0]==1],s=8,c='g')
plt.scatter(peak_dFF[iscell[:,0]==0],noise_level[iscell[:,0]==0],s=8,c='r')
plt.xlabel('peak dF//F0')
plt.ylabel('noise level ')

###################### Noise level for labeled vs unlabeled cells:

plt.figure(figsize=(5,4))
plt.scatter(redcell[iscell[:,0]==1,1],noise_level[iscell[:,0]==1],s=8,c='k')
plt.xlabel('red cell probability')
plt.ylabel('noise level ')

#################### 
skew = [stat[k]['skew'] for k in range(len(stat))]

plt.figure(figsize=(5,4))
plt.scatter(redcell[iscell[:,0]==1,1],noise_level[iscell[:,0]==1],s=8,c='k')
plt.xlabel('red cell probability')
plt.ylabel('noise level ')

#Count the number of events by taking stretches with z-scored activity above 2:
nEvents         = np.sum(np.diff(np.ndarray.astype(zF > 2,dtype='uint8'))==1,axis=1)
event_rate     = nEvents / (ops['nframes'] / ops['fs'])

plt.figure(figsize=(5,4))
plt.scatter(event_rate[iscell[:,0]==1],noise_level[iscell[:,0]==1],s=8,c='k')
plt.xlabel('event rate')
plt.ylabel('noise level ')
