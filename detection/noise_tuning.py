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

# sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                        #  load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

# sessions,nSessions = filter_sessions(protocols=protocol,load_behaviordata=True,load_videodata=False,
                        #  load_calciumdata=True,calciumversion=calciumversion,min_cells=100) #Load specified list of sessions

#%% Get signal as relative to psychometric curve for all sessions:
sessions,nSessions = filter_sessions(protocols=protocol,min_cells=100) #Load specified list of sessions
sessions = noise_to_psy(sessions,filter_engaged=True)
plot_psycurve(sessions,filter_engaged=True)

#%% Include sessions based on performance: psychometric curve for the noise #############
sessiondata = pd.concat([ses.sessiondata for ses in sessions])
zmin_thr = -0.5
zmax_thr = 0.5
guess_thr = 0.4
idx_ses = np.all((sessiondata['noise_zmin']<=zmin_thr,
                  sessiondata['noise_zmax']>=zmax_thr,
                  sessiondata['guess_rate']<=guess_thr),axis=0)
print('Filtered %d/%d sessions based on performance' % (np.sum(idx_ses),len(idx_ses)))

#%%
sessions = [sessions[i] for i in np.where(idx_ses)[0]]
nSessions = len(sessions)

#%% Load the data:           
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)

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

#%%
ises = 6
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

ises = 2
example_cell_ids = get_example_cells(sessions[ises].sessiondata['session_id'][0])

# get some responsive cells: 
idx                 = np.nanmean(sessions[ises].respmat,axis=1)>0.5
idx                 = sessions[ises].celldata['sig_N']==1
example_cell_ids = np.random.choice(sessions[ises].celldata['cell_id'][idx], size=9, replace=False)

fig = plot_noise_activity_example_neurons(sessions[ises],example_cell_ids)
# fig.savefig(os.path.join(savedir,'HitMiss_ExampleNeuronActivity_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')



#%% #################### Compute mean activity for saliency trial bins for all sessions ##################
def get_idx_noisebins(trialdata,sigtype,edges):
    """
    Bins signal values of noise into bins defined by edges, and puts trial index of 
    trials with signal 0 and 100 in first and last column
    Given a session and a set of edges (bin edges) returns a 2D boolean array with the same number of rows as trials 
    in the session and the same number of columns as bins + 2.
    """
    idx_T_noise = np.array([(trialdata[sigtype]>=low) & 
                    (trialdata[sigtype]<=high) for low,high in zip(edges[:-1],edges[1:])])
    idx_T_all = np.column_stack((trialdata[sigtype]==0,
                            idx_T_noise.T,
                            trialdata[sigtype]==100))
    return idx_T_all

def get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True):

    celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
    N           = len(celldata)

    lickresp    = [0,1]
    D           = len(lickresp)

    Z           = nbins_noise + 2

    edges       = np.linspace(zmin,zmax,nbins_noise+1)
    centers     = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
    if sigtype == 'signal_psy':
        plotcenters = np.hstack((centers[0]-2*np.mean(np.diff(centers)),centers,centers[-1]+2*np.mean(np.diff(centers))))
    elif sigtype=='signal': 
        plotcenters = np.hstack((0,centers,100))

    if splithitmiss: 
        data_mean    = np.full((N,Z,D),np.nan)
    else: 
        data_mean    = np.full((N,Z),np.nan)

    for ises,ses in enumerate(sessions):
        print(f"\rComputing mean activity for noise trial bins for session {ises+1} / {len(sessions)}",end='\r')
        idx_N_ses = celldata['session_id']==ses.sessiondata['session_id'][0]

        idx_T_all = get_idx_noisebins(sessions[ises].trialdata,sigtype,edges)

        if splithitmiss:
            for iZ in range(Z):
                for ilr,lr in enumerate(lickresp):
                    idx_T = np.all((idx_T_all[:,iZ],
                                sessions[ises].trialdata['lickResponse']==lr,
                                sessions[ises].trialdata['engaged']==1), axis=0)
                    data_mean[idx_N_ses,iZ,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)
        else: 
            for iZ in range(Z):
                data_mean[idx_N_ses,iZ]        = np.nanmean(sessions[ises].respmat[:,idx_T_all[:,iZ]],axis=1)

    return data_mean,plotcenters

def get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True):

    celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
    N           = len(celldata)

    lickresp    = [0,1]
    D           = len(lickresp)

    Z           = nbins_noise + 2

    S           = len(sbins)

    edges       = np.linspace(zmin,zmax,nbins_noise+1)
    centers     = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
    if sigtype == 'signal_psy':
        plotcenters = np.hstack((centers[0]-2*np.mean(np.diff(centers)),centers,centers[-1]+2*np.mean(np.diff(centers))))
    elif sigtype=='signal': 
        plotcenters = np.hstack((0,centers,100))

    if splithitmiss: 
        data_mean    = np.full((N,Z,S,D),np.nan)
    else: 
        data_mean    = np.full((N,Z,S),np.nan)

    for ises,ses in enumerate(sessions):
        print(f"\rComputing mean activity for noise trial bins for session {ises+1} / {len(sessions)}",end='\r')
        idx_N_ses = celldata['session_id']==ses.sessiondata['session_id'][0]

        idx_T_all = get_idx_noisebins(sessions[ises].trialdata,sigtype,edges)

        if splithitmiss:
            for iZ in range(Z):
                for ilr,lr in enumerate(lickresp):
                    idx_T = np.all((idx_T_all[:,iZ],
                                sessions[ises].trialdata['lickResponse']==lr,
                                sessions[ises].trialdata['engaged']==1), axis=0)
                    data_mean[idx_N_ses,iZ,ilr,:]        = np.nanmean(sessions[ises].tensor[:,idx_T,:],axis=1)
        else: 
            for iZ in range(Z):
                data_mean[idx_N_ses,iZ,:]      = np.nanmean(sessions[ises].tensor[:,idx_T_all[:,iZ],:],axis=1)

    return data_mean,plotcenters

#%% 


# celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
# N           = len(celldata)

# lickresp    = [0,1]
# D           = len(lickresp)

# sigtype     = 'signal_psy'
# zmin        = -1
# zmax        = 1
# nbins_noise = 5
# Z           = nbins_noise + 2

# sigtype     = 'signal'
# zmin        = 7
# zmax        = 17
# nbins_noise = 5
# Z           = nbins_noise + 2

# edges       = np.linspace(zmin,zmax,nbins_noise+1)
# centers     = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
# plotcenters = np.hstack((centers[0]-2*np.mean(np.diff(centers)),centers,centers[-1]+2*np.mean(np.diff(centers))))

# S           = len(sbins)

# data_sig_spatial = np.full((N,Z,S),np.nan)
# data_sig_mean    = np.full((N,Z),np.nan)

# data_sig_hit_spatial = np.full((N,Z,D,S),np.nan)
# data_mean_hitmiss    = np.full((N,Z,D),np.nan)

# data_frac_spatial = np.full((N,Z,S),np.nan)
# data_frac_mean    = np.full((N,Z),np.nan)

# min_ntrials     = 5

sigtype = 'signal_psy'
zmin        = -1
zmax        = 1
nbins_noise = 5

sigtype     = 'signal'
zmin        = 7
zmax        = 17
nbins_noise = 5

data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True)

# #Do statistical test between hit and miss
# for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
#     idx_T           = np.all((sessions[ises].trialdata[sigtype]>=low,
#                             sessions[ises].trialdata[sigtype]<=high,
#                             sessions[ises].trialdata['lickResponse']==lr,
#                             sessions[ises].trialdata['engaged']==1), axis=0)
#         if np.sum(idx_T)>=min_ntrials:
#             data_mean_hitmiss_diff[idx_N_ses,ibin+1,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)


#%% 


#%% 
plt.plot(np.sum(np.isnan(data_mean_hitmiss[:,:,0]),axis=1))

plt.imshow(np.isnan(data_mean_hitmiss[:,:,0]),aspect='auto')

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

from scipy.stats import ranksums
from scipy.stats import ttest_rel

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

lickresp        = [0,1]
D               = len(lickresp)

plotcolors  	= ['blue','red']  # Start with black
plotlabels      = ['miss','hit']
markerstyles    = ['o','o']
plotlocs        = np.arange(np.shape(data_mean_hitmiss)[1])

siglabels = ['nonresponsive','responsive']
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)

for sig in [0,1]:
    ax = axes[sig]
    # idx_N = celldata['sig_N']==1
    # idx_N = celldata['sig_N']!=1

    idx_N = celldata['sig_MN']==sig
    for ilr,lr in enumerate(lickresp):
        # ax.plot(plotcenters[1:-1],np.nanmean(data_mean_hitmiss[idx_N,1:-1,ilr],axis=0),
                # marker='.',markersize=15,color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
        # ax.plot(plotcenters,np.nanmean(data_mean_hitmiss[idx_N,:,ilr],axis=0),color=plotcolors[ilr], 
                #  marker='.',markersize=15,label=plotlabels[ilr],linewidth=2)
        ax.plot(plotlocs,np.nanmean(data_mean_hitmiss[idx_N,:,ilr],axis=0),color=plotcolors[ilr], 
                 marker='.',markersize=15,label=plotlabels[ilr],linewidth=2)
    
    pvals = ttest_rel(data_mean_hitmiss[idx_N,:,0],data_mean_hitmiss[idx_N,:,1],nan_policy='omit')[1] * 5
    # pvals = ranksums(data_mean_hitmiss[idx_N,:,0],data_mean_hitmiss[idx_N,:,1],nan_policy='omit')[1]
    for ipval,pval in enumerate(pvals):
        ax.text(ipval,np.nanmean(data_mean_hitmiss[idx_N,ipval,:],axis=None)+0.05,get_sig_asterisks(pval,return_ns=True),
                color='k',fontsize=10,ha='center')
    
    ax.set_xticks(plotlocs[np.array([0,1,3,5,6])],plotcenters[np.array([0,1,3,5,6])],rotation=45)
    ax.legend(plotlabels,loc='upper left',fontsize=11,frameon=False,reverse=True)
        # plt.plot(plotcenters,np.nanmean(data_mean_hitmiss[:,0],axis=0),color='k')
    ax.set_xlabel('Signal Strength (%)')
    ax.set_ylabel('Mean Activity (z)')
    ax.set_ylim([-0.05,0.55])
    ax.set_title(siglabels[sig])

fig.savefig(os.path.join(savedir, 'HitMiss_Mean_Resp_vs_NonRespNeurons_%dsessions.png') % (nSessions), format='png')
# fig.savefig(os.path.join(savedir, 'HitMiss_Mean_NoiseResponsiveNeurons_%dsessions.png') % (nSessions), format='png')
# fig.savefig(os.path.join(savedir, 'HitMiss_Noise_Mean_NoiseResponsiveNeurons_%dsessions.png') % (nSessions), format='png',bbox_inches='tight')

#%% 
labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)
clrs_areas  = get_clr_areas(areas)

data = copy.deepcopy(data_mean_hitmiss)
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
                        # celldata['layer']=='L2/3',
                        # celldata['layer']=='L4',
                        # celldata['layer']=='L5',
                        # ~np.any(np.isnan(data),axis=(1,2)),
                        celldata['labeled']==label), axis=0)

        if np.sum(idx_N) > 5:
            for ilr,lr in enumerate(lickresp):
                # plt.plot(plotcenters,np.nanmean(data_mean_hitmiss[idx_N,:,ilr],axis=0),color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
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

from scipy.stats import wilcoxon

diffdata = np.diff(data_mean_hitmiss,axis=2).squeeze()
diffdata = np.nanmean(np.diff(data_mean_hitmiss,axis=2)[:,1:-1,:],axis=1).squeeze()

fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2),sharey=False,sharex=True)
for iarea, area in enumerate(areas):
    for ilab,label in enumerate(labeled):
        ax = axes[ilab,iarea]
        handles = []
        idx_N = np.all((celldata['roi_name']==area, 
                        # celldata['sig_MN']==1,
                        # celldata['sig_N']==1,
                        # celldata['sig_N']!=1,
                        celldata['sig_MN']!=1,
                        celldata['labeled']==label), axis=0)

        df = pd.DataFrame(data = {'depth':celldata['depth'][idx_N], 'diff': diffdata[idx_N]})
        
        # sns.scatterplot(data=df,y='depth',x='diff',ax=ax,color=clrs_areas[iarea],alpha=0.5,s=5)
        sns.scatterplot(data=df,y='depth',x='diff',ax=ax,color='k',alpha=0.5,s=5)
        depth_edges = np.arange(0,500,50)
        centers     = np.stack((depth_edges[:-1],depth_edges[1:]),axis=1).mean(axis=1)

        depth_diff = np.full(len(centers),np.nan)
        depth_stats = np.full((2,len(centers)),np.nan)

        for ibin,(bd1,bd2) in enumerate(zip(depth_edges[:-1],depth_edges[1:])): 
            idx = idx_N & (celldata['depth']>=bd1) & (celldata['depth']<bd2)
            if sum(idx) >= 5:
                depth_diff[ibin] = np.nanmean(diffdata[idx])
                depth_stats[:,ibin] = wilcoxon(diffdata[idx])

            # depth_diff = np.array([np.nanmean(diffdata[idx_N & (celldata['depth']>=bd1) & (celldata['depth']<bd2)]) if sum(idx_N & (celldata['depth']>=bd1) & (celldata['depth']<bd2))>=10 else np.nan for bd1,bd2 in zip(depth_edges[:-1],depth_edges[1:])])
        
        ax.plot(depth_diff,centers,color=clrs_areas[iarea],linewidth=4)
        for ibin in range(len(depth_diff)):
            if depth_stats[1,ibin] < 0.05:
                sign = int(depth_diff[ibin]>0) * 2 - 1
                ax.text(0.35*sign,centers[ibin],get_sig_asterisks(depth_stats[1,ibin]),
                        ha='center',va='center',color=clrs_areas[iarea],fontsize=14,fontweight='bold')
        # ax.plot(depth_diff,centers,color='k',linewidth=3)
        ax.axvline(x=0,color='k',linestyle='--',linewidth=0.5)
        ax.grid(True)
        ax.set_xlim([-0.5,0.5])
        ax.set_ylim([0,500])

        ax.invert_yaxis()
        ax.set_yticks(np.arange(0,600,100))
        # ax.set_xticks(np.arange(-0.5,0.5,0.25))
        ax.set_title('%s-%s' % (area,label),fontsize=12,color=clrs_areas[iarea])
        if ilab==0:
            ax.set_ylabel('Depth (um)')
        if ilab==1:
            ax.set_xlabel(r'hit-miss ($\Delta$z)')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'DeltaHitMiss_Depth_ResponsiveNeurons_Arealabels_%dsessions.png') % (nSessions), format='png')
plt.savefig(os.path.join(savedir,'DeltaHitMiss_Depth_NonResponsiveNeurons_Arealabels_%dsessions.png') % (nSessions), format='png')




#%% 

sigtype     = 'signal'
zmin        = 7
zmax        = 17
nbins_noise = 5

data_mean,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=False)


data = copy.deepcopy(data_mean[idx_N,:])
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

fig,axes = plt.subplots(1,3,figsize=(10,3))
ax = axes[0]
ax.plot(pca.explained_variance_ratio_,linewidth=2,marker='o',color='k')
ax.set_xlabel('PC index')
ax.set_ylabel('Explained Variance')
ax.set_title('Explained Variance')

plotlocs        = np.arange(np.shape(data_mean_hitmiss)[1])

clrs = ['blue','red']
ax = axes[1]
for icomp in range(2):
    ax.plot(plotlocs,pca.components_[icomp,:].T,linewidth=3,color=clrs[icomp])
    # ax.plot(plotcenters,pca.components_[icomp,:].T,linewidth=5/(icomp+1))
# ax.plot(plotcenters,pca.components_[:2,:].T)
ax.set_title('PC components')

ax = axes[2]
ax.scatter(data_pca[:,0],data_pca[:,1],c='k', marker='o',s=10,alpha=0.5)
ax.set_title('Neuron locations in PC space')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

fig.savefig(os.path.join(savedir,'PCA_TuningCurve_%dsessions.png') % (nSessions), format='png')

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
data_mean_hitmiss    = np.full((N,Z,D),np.nan)

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
        data_mean_hitmiss[idx_N_ses,0,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)
        #Max trials
        idx_T           = np.all((sessions[ises].trialdata['signal']==100,
                                    sessions[ises].trialdata['lickResponse']==lr,
                                    sessions[ises].trialdata['engaged']==1), axis=0)
        data_sig_hit_spatial[idx_N_ses,-1,ilr,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
        data_mean_hitmiss[idx_N_ses,-1,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

        for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
            idx_T           = np.all((sessions[ises].trialdata[sigtype]>=low,
                                    sessions[ises].trialdata[sigtype]<=high,
                                    sessions[ises].trialdata['lickResponse']==lr,
                                    sessions[ises].trialdata['engaged']==1), axis=0)
            if np.sum(idx_T)>=min_ntrials:
                data_sig_hit_spatial[idx_N_ses,ibin+1,ilr,:]   = np.nanmean(sessions[ises].stensor[:,idx_T,:],axis=1)
                data_mean_hitmiss[idx_N_ses,ibin+1,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

