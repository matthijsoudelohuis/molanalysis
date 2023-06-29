# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR task. 
"""

import os
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.stats as st
from scipy.stats import binned_statistic

from sklearn.decomposition import PCA

procdatadir     = "V:\\Procdata\\"

animal_ids          = ['NSH07422'] #If empty than all animals in folder will be processed
sessiondates        = ['2022_12_9']

## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = os.listdir(procdatadir)

for animal_id in animal_ids: #for each animal
    
    if len(sessiondates) == 0:
        sessiondates = os.listdir(os.path.join(procdatadir,animal_id)) 
    
    for sessiondate in sessiondates: #for each of the sessions for this animal
        savefilename    = animal_id + "_" + sessiondate + "_VR.nwb" #define save file name
        savefilename    = os.path.join(procdatadir,animal_id,savefilename) #construct output save directory string

        # Open the file in read mode "r"
        io = NWBHDF5IO(savefilename, mode="r")
        nwbfile = io.read()
        io.close()


###### Construct
F           = nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].data[:]
# F           = nwbfile.processing['ophys']['dF_F']['dF_F'].data[:]

#Filter only selected cells (ROIs) from suite2p
iscell      = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'].data[:,0]
F           = F[:,iscell==1]

ts_F        = nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].timestamps[:] #timestamps for imaging data

## Get interpolated values for behavioral variables at imaging frame rate:
ts_harp     = nwbfile.acquisition['CorridorPosition'].timestamps[:]
zpos_F      = np.interp(x=ts_F,xp=nwbfile.acquisition['CorridorPosition'].timestamps[:],
                        fp=nwbfile.acquisition['CorridorPosition'].data[:])
runspeed_F  = np.interp(x=ts_F,xp=nwbfile.acquisition['RunningSpeed'].timestamps[:],
                        fp=nwbfile.acquisition['RunningSpeed'].data[:])
trialnum_F  = np.interp(x=ts_F,xp=nwbfile.acquisition['TrialNumber'].timestamps[:],
                        fp=nwbfile.acquisition['TrialNumber'].data[:])


F_Z           = st.zscore(F,axis=0)


## Construct tensor: 3D 'matrix' of N neurons by T trials by S spatial bins

trd = nwbfile.trials.to_dataframe() #convert trials from dynamic table with vector to pandas dataframe

## Parameters for spatial binning
s_pre       = -100  #pre cm
s_post      = 100   #post cm
binsize     = 5     #spatial binning in cm

binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
bincenters  = np.arange(s_pre,s_post+binsize,binsize)

N           = np.shape(F)[1]
T           = len(trd)
S           = len(bincenters)

tensor      = np.empty([N,T,S])
tensor_z    = np.empty([N,T,S])

for iT in range(T):
    idx = trialnum_F==iT+1
    for iN in range(N):
        tensor[iN,iT,:] = binned_statistic(zpos_F[idx]-trd.loc[iT, 'stimstart'],F[idx,iN], statistic='mean', bins=binedges)[0]
        tensor_z[iN,iT,:] = binned_statistic(zpos_F[idx]-trd.loc[iT, 'stimstart'],F_Z[idx,iN], statistic='mean', bins=binedges)[0]


##### Construct heatmaps per stim:

stimtypes = ['stimA','stimB','stimC','stimD']
snakeplots = np.empty([N,S,len(stimtypes)])

for iTT in range(len(stimtypes)):
    # snakeplots[:,:,iTT] = np.nanmean(tensor[:,trd['stimright'] == stimtypes[iTT],:],axis=1)
    snakeplots[:,:,iTT] = np.nanmean(tensor_z[:,trd['stimright'] == stimtypes[iTT],:],axis=1)

fig, axes = plt.subplots(nrows=2,ncols=2)

# make data with uneven sampling in x
X, Y = np.meshgrid(bincenters, range(N))

# snakeplots
# snakeplots2 = np.sort(snakeplots,axis=1)

for iTT in range(len(stimtypes)):
    plt.subplot(2,2,iTT+1)
    # c = plt.pcolormesh(X,Y,snakeplots2[:,:,iTT], cmap = 'PuRd',vmin=-50.0,vmax=700)
    # c = plt.pcolormesh(X,Y,snakeplots[:,:,iTT], cmap = 'gnuplot',vmin=-50.0,vmax=600)
    c = plt.pcolormesh(X,Y,snakeplots[:,:,iTT], cmap = 'gnuplot',vmin=-0.5,vmax=1.5)
    # plt.imshow(snakeplots[:,:,iTT], cmap = 'autumn' , interpolation = 'nearest')
    plt.title(stimtypes[iTT],fontsize=10)
    plt.ylabel('nNeurons',fontsize=9)
    plt.xlabel('Pos. relative to stim (cm)',fontsize=9)
    plt.xlim([-90,70])
    plt.ylim([0,N])
    plt.colorbar(c)
plt.show()

##

import seaborn as sns
from sklearn.preprocessing import StandardScaler

idx_rsp     = (bincenters>-5) & (bincenters<25)
idx_bsl     = (bincenters>-100) & (bincenters<-10)
# stimactiv   = np.nanmean(tensor[:,:,idx],axis=2)

# stimactiv   = np.nanmean(tensor[:,:,idx_rsp],axis=2) - np.nanmean(tensor[:,:,idx_bsl],axis=2)
stimactiv   = np.nanmean(tensor_z[:,:,idx_rsp],axis=2) - np.nanmean(tensor_z[:,:,idx_bsl],axis=2)

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    return Xz


Xr      = stimactiv
# Xr_sc   = z_score(Xr)
Xr_sc   = Xr

pca     = PCA(n_components=15)
Xp      = pca.fit_transform(Xr_sc.T).T
# Xp      = pca.fit_transform(Xr_sc.T).T

trial_type      = trd['stimright']
trial_types     = stimtypes
t_type_ind      = [np.argwhere(np.array(trial_type) == t_type)[:, 0] for t_type in trial_types]

shade_alpha      = 0.2
lines_alpha      = 0.8
pal = sns.color_palette('husl', 4)

projections = [(0, 1), (1, 2), (0, 2)]
fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
for ax, proj in zip(axes, projections):
    for t, t_type in enumerate(trial_types):
        x = Xp[proj[0], t_type_ind[t]]
        y = Xp[proj[1], t_type_ind[t]]
        ax.scatter(x, y, c=pal[t], s=25, alpha=0.8)
        ax.set_xlabel('PC {}'.format(proj[0]+1))
        ax.set_ylabel('PC {}'.format(proj[1]+1))
        
sns.despine(fig=fig, top=True, right=True)
ax.legend(stimtypes)


snakeselec = np.array(snakeplots[:,(bincenters>-60) & (bincenters<30),:])
nBins = np.shape(snakeselec)[1]
Xa = np.reshape(snakeselec, [N,nBins*4])

n_components = 15
# Xa = z_score(Xa) #Xav_sc = center(Xav)
pca = PCA(n_components=n_components)
Xa_p = pca.fit_transform(Xa.T).T

trial_size = nBins
space = bincenters[(bincenters>-60) & (bincenters<30)]

fig, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey='row')
for comp in range(3):
    ax = axes[comp]
    for kk, type in enumerate(trial_types):
        x = Xa_p[comp, kk * trial_size :(kk+1) * trial_size]
        # x = gaussian_filter1d(x, sigma=3)
        ax.plot(space, x, c=pal[kk])
    # add_stim_to_plot(ax)
    ax.set_ylabel('PC {}'.format(comp+1))
# add_orientation_legend(axes[2])
axes[1].set_xlabel('Time (s)')
sns.despine(fig=fig, right=True, top=True)
plt.tight_layout(rect=[0, 0, 0.9, 1])

plt.imshow(Xa)




## Trial outcomes:
idx_gonogo          = np.empty(shape=(T, 4),dtype=bool)
idx_gonogo[:,0]     = (trd['rewardtrial'] == 1) & (trd.lickresponse == True)
idx_gonogo[:,1]     = (trd['rewardtrial'] == 1) & (trd.lickresponse == False)
idx_gonogo[:,2]     = (trd['rewardtrial'] == 0) & (trd.lickresponse == True)
idx_gonogo[:,3]     = (trd['rewardtrial'] == 0) & (trd.lickresponse == False)

labels_gonogo       = ['HIT', 'MISS', 'FA', 'CR']

fig, ax = plt.subplots()

plt.plot(trd['trialnum'],smooth_hitrate,color="green")
plt.plot(trd['trialnum'],smooth_farate,color="brown")
plt.xlabel('trial number')
plt.ylabel('HITrate / FArate')
# plt.ylim(0,50)
plt.xlim(window_size,)
plt.legend(['HIT','FA'])
colors = ["cyan","pink"]
for iblock in np.arange(0,ntrials,100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,1.0, 
                        fill = True, alpha=0.2,
                        color = colors[0], linewidth = 0))
for iblock in np.arange(50,ntrials,100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,1.0, 
                        fill = True, alpha=0.2,
                        color = colors[1], linewidth = 0))
    
fig, ax = plt.subplots()
plt.plot(trd['trialnum'],smooth_d_prime,color="blue")
plt.xlabel('trial number')
plt.ylabel('Dprime')
plt.ylim(0,5)
plt.xlim(window_size,)
plt.legend(['Dprime'])
colors = ["cyan","pink"]
for iblock in np.arange(0,ntrials,100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,5.0, 
                        fill = True, alpha=0.2,
                        color = colors[0], linewidth = 0))
for iblock in np.arange(50,ntrials,100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,5.0, 
                        fill = True, alpha=0.2,
                        color = colors[1], linewidth = 0))