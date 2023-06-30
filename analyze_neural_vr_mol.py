# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR task. 
"""

#TODO:
# pass arguments to load data from start
# pass arguments about type of calciumdata to load
# set ts of rewards in trialdata
# set ts of entering reward zone and stim zone in preprocessing

import os
# os.chdir('T:\\Python\\molanalysis\\')
os.chdir('E:\\Python\\molanalysis\\')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from scipy.stats import zscore
from sklearn.decomposition import PCA
from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space

protocol            = 'VR'


session_list = np.array([['LPE09829', '2023_03_29']])
session_list = np.array([['LPE09829', '2023_03_29'],
                        ['LPE09829', '2023_03_30'],
                        ['LPE09829', '2023_03_31']])

sessions = load_sessions(protocol,session_list,load_behaviordata=True,load_calciumdata=True) #Load specified list of sessions:
# sessions = filter_sessions(protocol) #load sessions that meet criteria:

sessions[0].load_data(load_behaviordata=True,load_calciumdata=True)

#Keep only first 100 cells to remain workable:
sessions[0].calciumdata = sessions[0].calciumdata.drop(sessions[0].calciumdata.columns[100:],axis=1)

F               = sessions[0].calciumdata
F_Z             = zscore(F.copy(),axis=0)

ts_F = sessions[0].ts_F.to_numpy()

ts_harp = sessions[0].behaviordata['ts'].to_numpy()

## Get interpolated values for behavioral variables at imaging frame rate:
zpos_F      = np.interp(x=ts_F,xp=ts_harp,
                        fp=sessions[0].behaviordata['zpos'])
runspeed_F  = np.interp(x=ts_F,xp=ts_harp,
                        fp=sessions[0].behaviordata['runspeed'])
trialnum_F  = np.interp(x=ts_F,xp=ts_harp,
                        fp=sessions[0].behaviordata['trialnum'])

trialdata = sessions[0].trialdata #convert trials from dynamic table with vector to pandas dataframe


# Define when animal gets reward

##############################################################################
## Construct tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

[tensor,tbins] = compute_tensor(F_Z, ts_F, trialdata['tStimStart'], t_pre, t_post, binsize,method='interp_lin')
respmat         = tensor[:,:,np.logical_and(tbins > 0,tbins < 1)].mean(axis=2)
[K,N,T]         = np.shape(tensor) #get dimensions of tensor

#Alternative method, much faster:
respmat         = compute_respmat(F_Z, ts_F, trialdata['tStimStart'],t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=True)

[K,N]           = np.shape(respmat) #get dimensions of response matrix

# #hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
# temp = pd.DataFrame(np.reshape(np.array(behaviordata['runspeed']),(len(behaviordata['runspeed']),1)))
# respmat_runspeed = compute_respmat(temp, behaviordata['ts'], trialdata['tOnset'],t_resp_start=0,t_resp_stop=1,method='mean')

################################ Spatial Tensor #################################
## Parameters for spatial binning
s_pre       = -100  #pre cm
s_post      = 100   #post cm
binsize     = 5     #spatial binning in cm

# tensor,bincenters = compute_tensor_space(F_Z,ts_F,trialdata['StimStart'],zpos_F,trialnum_F,s_pre=-100,s_post=100,binsize=5,method='interp_lin')
tensor,sbins    = compute_tensor_space(F_Z.to_numpy(),ts_F,trialdata['StimStart'],zpos_F,trialnum_F,s_pre=-100,s_post=100,binsize=5,method='binmean')
[K,N,S]         = np.shape(tensor) #get dimensions of tensor

##### Construct heatmaps per stim:

stimtypes = ['A','B','C','D']
snakeplots = np.empty([N,S,len(stimtypes)])

for iTT in range(len(stimtypes)):
    snakeplots[:,:,iTT] = np.nanmean(tensor[trialdata['stimRight'] == stimtypes[iTT],:,:],axis=0)
    # snakeplots[:,:,iTT] = np.nanmean(tensor[:,trialdata['stimRight'] == stimtypes[iTT],:],axis=1)
    # snakeplots[:,:,iTT] = np.nanmean(tensor_z[:,trialdata['stimright'] == stimtypes[iTT],:],axis=1)

fig, axes = plt.subplots(nrows=2,ncols=2)

# make data with uneven sampling in x
X, Y = np.meshgrid(sbins, range(N))

# snakeplots
# snakeplots2 = np.sort(snakeplots,axis=1)

for iTT in range(len(stimtypes)):
    plt.subplot(2,2,iTT+1)
    # c = plt.pcolormesh(X,Y,snakeplots2[:,:,iTT], cmap = 'PuRd',vmin=-50.0,vmax=700)
    c = plt.pcolormesh(X,Y,snakeplots[:,:,iTT], cmap = 'gnuplot',vmin=-0.5,vmax=3)
    # c = plt.pcolormesh(X,Y,snakeplots[:,:,iTT], cmap = 'gnuplot',vmin=-0.5,vmax=1.5)
    # c = plt.pcolormesh(X,Y,snakeplots[:,:,iTT], cmap = 'gnuplot')
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

idx_rsp     = (sbins>-10) & (sbins<30)
idx_bsl     = (sbins>-100) & (sbins<-20)
# stimactiv   = np.nanmean(tensor[:,:,idx],axis=2)

# stimactiv   = np.nanmean(tensor[:,:,idx_rsp],axis=2) - np.nanmean(tensor[:,:,idx_bsl],axis=2)
stimactiv   = np.nanmean(tensor[:,:,idx_rsp],axis=2) - np.nanmean(tensor[:,:,idx_bsl],axis=2)

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    return Xz


Xr      = stimactiv.T
# Xr_sc   = z_score(Xr)
Xr_sc   = Xr

pca     = PCA(n_components=15)
Xp      = pca.fit_transform(Xr_sc.T).T
# Xp      = pca.fit_transform(Xr_sc.T).T

trial_type      = trialdata['stimRight']
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


snakeselec = np.array(snakeplots[:,(sbins>-60) & (sbins<30),:])
nBins = np.shape(snakeselec)[1]
Xa = np.reshape(snakeselec, [N,nBins*4])

n_components = 15
# Xa = z_score(Xa) #Xav_sc = center(Xav)
pca = PCA(n_components=n_components)
Xa_p = pca.fit_transform(Xa.T).T

trial_size = nBins
space = sbins[(sbins>-60) & (sbins<30)]

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
idx_gonogo[:,0]     = (trialdata['rewardtrial'] == 1) & (trialdata.lickresponse == True)
idx_gonogo[:,1]     = (trialdata['rewardtrial'] == 1) & (trialdata.lickresponse == False)
idx_gonogo[:,2]     = (trialdata['rewardtrial'] == 0) & (trialdata.lickresponse == True)
idx_gonogo[:,3]     = (trialdata['rewardtrial'] == 0) & (trialdata.lickresponse == False)

labels_gonogo       = ['HIT', 'MISS', 'FA', 'CR']

fig, ax = plt.subplots()

plt.plot(trialdata['trialnum'],smooth_hitrate,color="green")
plt.plot(trialdata['trialnum'],smooth_farate,color="brown")
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
plt.plot(trialdata['trialnum'],smooth_d_prime,color="blue")
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
    

# from rastermap import Rastermap

# model = Rastermap(n_components=1, n_X=30, nPC=200, init='pca')

# # fit does not return anything, it adds attributes to model
# # attributes: embedding, u, s, v, isort1

# model.fit(sp)
# plt.imshow(sp[model.isort1, :])

# # fit_transform returns embedding (upsampled cluster identities)
# embedding = model.fit_transform(sp)

# # transform can be used on new samples with the same number of features as sp
# embed2 = model.transform(sp2)