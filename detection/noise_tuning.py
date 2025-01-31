# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
os.chdir('e:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
from tqdm import tqdm

import sklearn
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn import svm as SVM
# from sklearn.metrics import accuracy_score, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score
from scipy.signal import medfilt
from scipy.stats import zscore

from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive
import seaborn as sns
import matplotlib.pyplot as plt
from utils.psth import *
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *
from detection.example_cells import get_example_cells
from utils.regress_lib import *

plt.rcParams['svg.fonttype'] = 'none'

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Encoding\\')


#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

# session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11622', '2024_02_21']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11997', '2024_04_16'],
                         ['LPE11622', '2024_02_21'],
                         ['LPE11998', '2024_04_30'],
                         ['LPE12013','2024_04_25']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list        = np.array([['LPE12013','2024_04_25']])
# session_list        = np.array([['LPE12013','2024_04_26']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

sessions,nSessions = filter_sessions(protocols=protocol,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion,min_cells=100) #Load specified list of sessions

#%% Z-score the calciumdata: 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 60   #post cm
binsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')
    # Compute average response in stimulus response zone:
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
                                    sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)


#%% #################### Compute spatial runspeed ####################################
for ises,ses in enumerate(sessions): # running across the trial:
    sessions[ises].behaviordata['runspeed'] = medfilt(sessions[ises].behaviordata['runspeed'], kernel_size=51)
    [sessions[ises].runPSTH,_]     = calc_runPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)
    [sessions[ises].lickPSTH,_]    = calc_lickPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)

#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

#%% Get signal as relative to psychometric curve for all sessions:
sessions = noise_to_psy(sessions,filter_engaged=True)



#%%
ises = 3
example_cell_ids = get_example_cells(sessions[ises].sessiondata['session_id'][0])

fig = plot_mean_activity_example_neurons(sessions[ises].stensor,sbins,sessions[ises],example_cell_ids)
# fig.savefig(os.path.join(savedir,'ExampleNeuronActivity_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')

#%%
sesid = 'LPE11997_2024_04_12'
sesid = 'LPE11997_2024_04_16'
# sesid = 'LPE12385_2024_06_16'
# sesid = 'LPE12013_2024_04_25'

sessiondata = pd.concat([ses.sessiondata for ses in sessions])
ises = np.where(sessiondata['session_id']==sesid)[0][0]

ises = 30
example_cell_ids = get_example_cells(sessions[ises].sessiondata['session_id'][0])

# get some responsive cells: 
# idx                 = np.nanmean(sessions[ises].respmat,axis=1)>0.5
# idx                 = sessions[ises].celldata['sig_N']==1
# example_cell_ids = np.random.choice(sessions[ises].celldata['cell_id'][idx], size=9, replace=False)
# example_cell_ids    = (sessions[ises].celldata['cell_id'][idx]).to_numpy()

fig = plot_noise_activity_example_neurons(sessions[ises],example_cell_ids)
fig.savefig(os.path.join(savedir,'HitMiss_ExampleNeuronActivity_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')



#%% #################### Compute mean activity for saliency trial bins for all sessions ##################
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

N = len(celldata)

lickresp    = [0,1]
D           = len(lickresp)

sigtype     = 'signal_psy'
zmin        = -1
zmax        = 1
nbins_noise = 5
Z           = nbins_noise + 2

sigtype     = 'signal'
zmin        = 7
zmax        = 17
nbins_noise = 5
Z           = nbins_noise + 2

edges       = np.linspace(zmin,zmax,nbins_noise+1)
centers     = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
plotcenters = np.hstack((centers[0]-2*np.mean(np.diff(centers)),centers,centers[-1]+2*np.mean(np.diff(centers))))

S           = len(sbins)

data_sig_spatial = np.full((N,Z,S),np.nan)
data_sig_mean    = np.full((N,Z),np.nan)

data_sig_hit_spatial = np.full((N,Z,D,S),np.nan)
data_sig_hit_mean    = np.full((N,Z,D),np.nan)

min_ntrials = 5
for ises,ses in enumerate(sessions):
    print(f"\rComputing mean activity for noise trial bins for session {ises+1} / {len(sessions)}",end='\r')
    idx_N_ses = celldata['session_id']==ses.sessiondata['session_id'][0]

    #Catch trials
    idx_T           = sessions[ises].trialdata['signal']==0
    data_sig_spatial[idx_N_ses,0,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
    data_sig_mean[idx_N_ses,0]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)
    #Max trials
    idx_T           = sessions[ises].trialdata['signal']==100
    data_sig_spatial[idx_N_ses,-1,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
    data_sig_mean[idx_N_ses,-1]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

    for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
        idx_T           = np.all((sessions[ises].trialdata[sigtype]>=low,
                                sessions[ises].trialdata[sigtype]<=high), axis=0)
        if np.sum(idx_T)>=min_ntrials:
            data_sig_spatial[idx_N_ses,ibin+1,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
            data_sig_mean[idx_N_ses,ibin+1]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

    for ilr,lr in enumerate(lickresp):
        #Catch trials
        idx_T           = np.all((sessions[ises].trialdata['signal']==0, 
                                    sessions[ises].trialdata['lickResponse']==lr,
                                    sessions[ises].trialdata['engaged']==1), axis=0)
        data_sig_hit_spatial[idx_N_ses,0,ilr,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
        data_sig_hit_mean[idx_N_ses,0,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)
        #Max trials
        idx_T           = np.all((sessions[ises].trialdata['signal']==100,
                                    sessions[ises].trialdata['lickResponse']==lr,
                                    sessions[ises].trialdata['engaged']==1), axis=0)
        data_sig_hit_spatial[idx_N_ses,-1,ilr,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
        data_sig_hit_mean[idx_N_ses,-1,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

        for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
            idx_T           = np.all((sessions[ises].trialdata[sigtype]>=low,
                                    sessions[ises].trialdata[sigtype]<=high,
                                    sessions[ises].trialdata['lickResponse']==lr,
                                    sessions[ises].trialdata['engaged']==1), axis=0)
            if np.sum(idx_T)>=min_ntrials:
                data_sig_hit_spatial[idx_N_ses,ibin+1,ilr,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
                data_sig_hit_mean[idx_N_ses,ibin+1,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

#%% 
plt.plot(np.sum(np.isnan(data_sig_hit_mean[:,:,0]),axis=1))

plt.imshow(np.isnan(data_sig_hit_mean[:,:,0]),aspect='auto')

#%% 
data = copy.deepcopy(data_sig_mean.T)

print(np.shape(data))

# data = data - np.nanmin(data,axis=0,keepdims=True)
data = data - data[:,0][:,np.newaxis]

data = data / np.nanmax(data,axis=0,keepdims=True)


# plt.imshow(data,aspect='auto',cmap='Reds')
# plt.imshow(data,aspect='auto',cmap='bwr',vmin=-0.25,vmax=1.25)

#%% 
# plt.plot(plotcenters,np.nanmean(data_sig_mean,axis=0),color='k')

 # plotcolors = [sns. sns.color_palette("inferno",C)
plotcolors = ['blue']  # Start with black
plotcolors.append('red')  # Add orange at the end
plotlabels = ['miss','hit']
markerstyles = ['o','o']

fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
idx_N = celldata['roi_name']=='V1'
idx_N = celldata['roi_name']=='PM'
idx_N = celldata['sig_N']==1
idx_N = celldata['sig_N']!=1

idx_N = celldata['sig_MN']==1
idx_N = celldata['sig_MN']!=1

# idx_N = np.all((celldata['roi_name']=='V1',
                # celldata['sig_N']==1), axis=0)

for ilr,lr in enumerate(lickresp):
    # plt.plot(plotcenters,np.nanmean(data_sig_hit_mean[idx_N,:,ilr],axis=0),color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
    plt.plot(plotcenters[1:-1],np.nanmean(data_sig_hit_mean[idx_N,1:-1,ilr],axis=0),
             marker='.',markersize=15,color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
    # plt.plot(plotcenters,np.nanmean(data_sig_hit_mean[idx_N,:,ilr],axis=0),color=plotcolors[ilr], 
            #  marker='.',markersize=15,label=plotlabels[ilr],linewidth=2)
ax.legend(plotlabels,loc='upper left',fontsize=11,frameon=False)
    # plt.plot(plotcenters,np.nanmean(data_sig_hit_mean[:,0],axis=0),color='k')
# plt.plot(plotcenters,np.nanmean(data_sig_hit_mean,axis=0))
# fig.savefig(os.path.join(savedir, 'HitMiss_Mean_NonResponsiveNeurons_%dsessions.png') % (nSessions), format='png')
# fig.savefig(os.path.join(savedir, 'HitMiss_Mean_NoiseResponsiveNeurons_%dsessions.png') % (nSessions), format='png')
fig.savefig(os.path.join(savedir, 'HitMiss_Noise_Mean_NoiseResponsiveNeurons_%dsessions.png') % (nSessions), format='png')

#%% 
labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

data = copy.deepcopy(data_sig_hit_mean)
#normalize data:
# data = data - np.nanmin(data,axis=1,keepdims=True)
# data = data / np.nanmax(data,axis=1,keepdims=True)

fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2),sharey=True,sharex=True)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax = axes[ilab,iarea]
        handles = []
        # idx_N = np.all((ses.celldata['roi_name']==area, ses.celldata['labeled']==label), axis=0)
        # idx_N = np.all((celldata['roi_name']==area, celldata['labeled']==label), axis=0)

        idx_N = np.all((celldata['roi_name']==area, 
                        celldata['sig_MN']==1,
                        # celldata['sig_N']==1,
                        # celldata['sig_N']!=1,
                        # celldata['sig_MN']!=1,
                        celldata['layer']=='L2/3',
                        # celldata['layer']=='L4',
                        # celldata['layer']=='L5',
                        # ~np.any(np.isnan(data),axis=(1,2)),
                        celldata['labeled']==label), axis=0)

        if np.sum(idx_N) > 5:
            for ilr,lr in enumerate(lickresp):
                # plt.plot(plotcenters,np.nanmean(data_sig_hit_mean[idx_N,:,ilr],axis=0),color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
                # ax.plot(plotcenters[1:-1],np.nanmean(data[idx_N,1:-1,ilr],axis=0),color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
                # h =shaded_error(plotcenters,data[idx_N,:,ilr],color=plotcolors[ilr],
                            #  label=plotlabels[ilr],ax=ax,error='sem')
                h =shaded_error(plotcenters[1:-1],data[idx_N,1:-1,ilr],color=plotcolors[ilr],
                             label=plotlabels[ilr],ax=ax,error='sem')
                handles.append(h)
        
        if ilab==0 and iarea==0:
            ax.legend(handles,plotlabels,loc='upper left',fontsize=11,frameon=False)

        # ax.axhline(0,color='k',linestyle='--',linewidth=1)
        if ilab==1:
            ax.set_xlabel('Signal Strength')
        ax.set_xticks(np.round(plotcenters[1:-1],1))
        if iarea==0:
            ax.set_ylabel('Mean Activity (z)')
        # ax.set_ylim([-0.1,0.6])
        #     ax.legend(frameon=False,fontsize=6)
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'HitMiss_Noise_Mean_NoiseResponsiveNeurons_RawSignal_%dsessions_Arealabels.png') % (nSessions), format='png')
# plt.savefig(os.path.join(savedir, 'HitMiss_Noise_Mean_NonNoiseResponsiveNeurons_RawSignal_%dsessions_Arealabels.png') % (nSessions), format='png')
# plt.savefig(os.path.join(savedir, 'EncodingModel_%s_cvR2_Areas_Labels_%dsessions.png') % (version,nSessions), format='png')


#%% 
idx_N = celldata['roi_name']=='V1'
idx_N = np.ones(len(celldata),dtype=bool)
# idx_N = celldata['sig_MN']==1
# idx_N = celldata['roi_name']=='PM'
data = copy.deepcopy(data_sig_mean[idx_N,:])
print(np.shape(data))

data = data[~np.any(np.isnan(data),axis=1),:]

print(np.shape(data))

# data = data - np.nanmin(data,axis=0,keepdims=True)
# data = data - data[:,0][:,np.newaxis]
# data = data / np.nanmax(data,axis=0,keepdims=True)

data = zscore(data,axis=0)

ncomponents = 7
pca = PCA(n_components=ncomponents)
pca.fit(data)

data_pca = pca.transform(data)

fig,axes = plt.subplots(1,4,figsize=(12,3))
ax = axes[0]
ax.plot(pca.explained_variance_ratio_,linewidth=2,marker='o')
ax.set_xlabel('PC index')

ax = axes[1]
for icomp in range(ncomponents):
    ax.plot(plotcenters,pca.components_[icomp,:].T,linewidth=5/(icomp+1))
# ax.plot(plotcenters,pca.components_[:2,:].T)

# ax = axes[2]
# for icomp in range(ncomponents):
#     np.mean(np.dot(pca.components_[:,icomp],data_pca.T),axis=1)
#     ax.plot(plotcenters,pca.components_[icomp,:].T,linewidth=5/(icomp+1))

ax = axes[3]
ax.scatter(data_pca[:,0],data_pca[:,1],cmap='tab10', marker='o',s=10,alpha=0.5)

#%% 



#%% 

# for ilab,label in enumerate(labeled):
#     for iarea, area in enumerate(areas):
#         # idx_N_resp = np.logical_or(sessions[ises].celldata['sig_N'],sessions[ises].celldata['sig_M'])
#         # idx_N     = np.all((sessions[ises].celldata['roi_name']==area,
#         #                     sessions[ises].celldata['labeled']==label,
#         #                     idx_N_resp), axis=0)
#         idx_N     = np.all((sessions[ises].celldata['roi_name']==area,
#                 sessions[ises].celldata['labeled']==label), axis=0)
            




#%% 





#%% #################### Compute mean activity for saliency trial bins for all sessions ##################
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

N = len(celldata)

labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

lickresp    = [0,1]
D           = len(lickresp)

sigtype     = 'signal_psy'
zmin        = -2
zmax        = 2
nbins_noise = 5
Z           = nbins_noise + 2

sigtype     = 'signal'
zmin        = 5
zmax        = 20
nbins_noise = 5
Z           = nbins_noise + 2

edges       = np.linspace(zmin,zmax,nbins_noise+1)
centers     = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
plotcenters = np.hstack((centers[0]-2*np.mean(np.diff(centers)),centers,centers[-1]+2*np.mean(np.diff(centers))))

S           = len(sbins)

data_sig_spatial = np.full((N,Z,S),np.nan)
data_sig_mean    = np.full((N,Z),np.nan)

data_sig_hit_spatial = np.full((N,Z,D,S),np.nan)
data_sig_hit_mean    = np.full((N,Z,D),np.nan)

min_ntrials = 5
for ises,ses in enumerate(sessions):
    print(f"\rComputing mean activity for noise trial bins for session {ises+1} / {len(sessions)}",end='\r')
    idx_N_ses = celldata['session_id']==ses.sessiondata['session_id'][0]

    #Catch trials
    idx_T           = sessions[ises].trialdata['signal']==0
    data_sig_spatial[idx_N_ses,0,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
    data_sig_mean[idx_N_ses,0]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)
    #Max trials
    idx_T           = sessions[ises].trialdata['signal']==100
    data_sig_spatial[idx_N_ses,-1,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
    data_sig_mean[idx_N_ses,-1]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

    for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
        idx_T           = np.all((sessions[ises].trialdata[sigtype]>=low,
                                sessions[ises].trialdata[sigtype]<=high), axis=0)
        if np.sum(idx_T)>=min_ntrials:
            data_sig_spatial[idx_N_ses,ibin+1,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
            data_sig_mean[idx_N_ses,ibin+1]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

    for ilr,lr in enumerate(lickresp):
        #Catch trials
        idx_T           = np.all((sessions[ises].trialdata['signal']==0, 
                                    sessions[ises].trialdata['lickResponse']==lr,
                                    sessions[ises].trialdata['engaged']==1), axis=0)
        data_sig_hit_spatial[idx_N_ses,0,ilr,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
        data_sig_hit_mean[idx_N_ses,0,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)
        #Max trials
        idx_T           = np.all((sessions[ises].trialdata['signal']==100,
                                    sessions[ises].trialdata['lickResponse']==lr,
                                    sessions[ises].trialdata['engaged']==1), axis=0)
        data_sig_hit_spatial[idx_N_ses,-1,ilr,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
        data_sig_hit_mean[idx_N_ses,-1,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

        for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
            idx_T           = np.all((sessions[ises].trialdata[sigtype]>=low,
                                    sessions[ises].trialdata[sigtype]<=high,
                                    sessions[ises].trialdata['lickResponse']==lr,
                                    sessions[ises].trialdata['engaged']==1), axis=0)
            if np.sum(idx_T)>=min_ntrials:
                data_sig_hit_spatial[idx_N_ses,ibin+1,ilr,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
                data_sig_hit_mean[idx_N_ses,ibin+1,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

