
import os
import matplotlib.pyplot as plt
import numpy as np
from suite2p.extraction import dcnv
import scipy.stats as st

from preprocessing.preprocesslib import calculate_dff


direc = 'V:\\Rawdata\\PILOTS\\20221108_NSH07429_Spontaneous4ROI\\Spontaneous\\suite2p\\plane0'
direc = 'W:\\Users\\Matthijs\\Rawdata\\LPE09829\\2023_03_29\\suite2p\\plane1'

direc = 'W:\\Users\\Matthijs\\Rawdata\\LPE09830\\2023_04_10\\suite2p\\plane1'

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

################################################################

baseline_perc = 15

coeff_Fneu = 0.7

####### Neuropil correction:
Fc = F - coeff_Fneu * Fneu + np.median(Fneu,axis=1,keepdims=True)

## Sliding baseline:

from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter

def calc_F0_sliding(F: np.ndarray, baseline: str, win_baseline: float,
               sig_baseline: float, fs: float, prctile_baseline: float = baseline_perc) -> np.ndarray:
    """ baseline with window 'win_baseline'
    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence
    baseline : str
        setting that describes how to compute the baseline of each trace
    win_baseline : float
        window (in seconds) for max filter
    sig_baseline : float
        width of Gaussian filter in seconds
    fs : float
        sampling rate per plane
    prctile_baseline : float
        percentile of trace to use as baseline if using `constant_prctile` for baseline
    Returns
    F : float, 2D array
        size [neurons x time], baseline fluorescence
    """

    win = int(win_baseline*fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(F,    [0., sig_baseline])
        Flow = minimum_filter1d(Flow,    win)
        Flow = maximum_filter1d(Flow,    win)
    elif baseline == 'constant':
        Flow = gaussian_filter(F,    [0., sig_baseline])
        Flow = np.amin(Flow)
    elif baseline == 'constant_prctile':
        Flow = np.percentile(F, prctile_baseline, axis=1)
        Flow = np.expand_dims(Flow, axis = 1)
    else:
        Flow = 0.
    return Flow

F0_suite2p = calc_F0_sliding(Fc, ops['baseline'], ops['win_baseline'], 
                                   ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])

# F0_suite2p = calc_F0_sliding(Fc, 'constant_prctile', ops['win_baseline'], 
#                                    ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])

#Compute dF / F0: 
dF_suite2p = (Fc - F0_suite2p) / F0_suite2p

# Establish baseline as percentile of corrected trace (50 is median)
F0_rupprecht = np.percentile(Fc,baseline_perc,axis=1,keepdims=True)

#Compute dF / F0: 
dF_rupprecht = (Fc - F0_rupprecht) / F0_rupprecht

zF_rupprecht  =st.zscore(dF_rupprecht,axis=1)

######## plot traces with and without neuropil correction #########

#pick an exemplary neuron: 
neuronidx   = 30 #15
neuronidx   = 464 #15
timeidx     = np.arange(3000,6000)
timeidx     = np.arange(27000,28500)
timeidx     = np.arange(0,3000)
timeidx     = np.arange(np.shape(F)[1])

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15,4))
ax1.plot(F[neuronidx,timeidx],'g',linewidth=0.5)
ax1.plot(Fneu[neuronidx,timeidx],'k',linewidth=0.5)
ax1.plot(F0_suite2p[neuronidx,timeidx],'b',linewidth=0.5)
# ax1.plot(F0_suite2p[neuronidx],'b',linewidth=0.5)

ax2.plot(F[neuronidx,timeidx],'r',linewidth=0.5)
ax2.plot(Fneu[neuronidx,timeidx],'k',linewidth=0.5)
ax2.axhline(F0_rupprecht[neuronidx],color='b',linewidth=0.5)

ax3.plot(dF_suite2p[neuronidx,timeidx],'g',linewidth=0.5)
ax3.plot(dF_rupprecht[neuronidx,timeidx],'r',linewidth=0.5)

######## plot deltaF/F0 traces versus z-scored traces: #########

plt.figure(figsize=(15,4))
plt.plot(dF_rupprecht[neuronidx,timeidx],'r',linewidth=0.5)
plt.plot(zF_rupprecht[neuronidx,timeidx],'b',linewidth=0.5)

######## plot deltaF/F0 traces versus inferred spike rate through deconvolution #########
spks_norm = spks / spks.max(axis=1, keepdims=1) * 5

plt.figure(figsize=(15,4))
plt.plot(dF_rupprecht[neuronidx,timeidx],'r',linewidth=0.25)
plt.plot(zF_rupprecht[neuronidx,timeidx],'b',linewidth=0.25)
# plt.plot(spks[neuronidx,timeidx] / np.max(spks[neuronidx,timeidx]),'k',linewidth=1)
plt.plot(spks_norm[neuronidx,timeidx],'k',linewidth=0.5)

selec = np.array([15,36,26,500,30,415,416,136,139,250])
temp = np.arange(len(selec))
spksselec = spks_norm[selec[:,np.newaxis],timeidx] + temp[:,np.newaxis]

plt.figure()
plt.plot(spksselec.T,linewidth=0.5)

###################### Calculate the noise level of the cells ####
# Rupprecht et al. 2021 Nat Neurosci.

noise_level = np.median(np.abs(np.diff(dF_rupprecht,axis=1)),axis=1)/np.sqrt(ops['fs'])
peak_dFF = np.max(dF_rupprecht,axis=1)
# bottom_dFF = np.min(dF_rupprecht,axis=1)
# peak_dFF = np.argmax(peak_dFF)

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
nEvents         = np.sum(np.diff(np.ndarray.astype(zF_rupprecht > 2,dtype='uint8'))==1,axis=1)
event_rate     = nEvents / (ops['nframes'] / ops['fs'])

plt.figure(figsize=(5,4))
plt.scatter(event_rate[iscell[:,0]==1],noise_level[iscell[:,0]==1],s=8,c='k')
plt.xlabel('event rate')
plt.ylabel('noise level ')
