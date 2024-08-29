# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import binned_statistic,binned_statistic_2d

from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
# from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes
# from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
from utils.plot_lib import shaded_error
# from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *
from utils.rf_lib import smooth_rf

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
# session_list        = np.array([['LPE10919','2023_11_06']])

# load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM']) 

#%%  Load data properly:                      
# for ises in range(nSessions):    # iterate over sessions
# for ises in tqdm(range(nSessions),desc= 'Loading data and trace correlations: '):
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% ########################### Compute noise correlations: ###################################
sessions = compute_trace_correlation(sessions,binwidth=0.25,uppertriangular=False)

#%% ########################## Compute signal correlations: ###################################
sessions = compute_signal_correlation(sessions,uppertriangular=False)

# #%% ############### Plot control figure of signal and noise corrs ############################
# sesidx = 0
# fig = plt.figure(figsize=(8,5))
# plt.imshow(sessions[sesidx].trace_corr, cmap='coolwarm',
#            vmin=np.nanpercentile(sessions[sesidx].trace_corr,5),
#            vmax=np.nanpercentile(sessions[sesidx].trace_corr,95))
# plt.savefig(os.path.join(savedir,'NoiseCorrelations','TraceCorr_SP_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ##################### Compute pairwise neuronal distances: ##############################
# sessions = compute_pairwise_metrics(sessions)
sessions = compute_pairwise_delta_rf(sessions,rf_type='F')
sessions = compute_pairwise_anatomical_distance(sessions)

# smooth_rf(sessions,sig_thr=0.01,show_result=True,radius=100)

#%% Define function to filter neuronpairs based on area combination
def filter_2d_areapair(ses,areapair):
    area1,area2 = areapair.split('-')
    areafilter1 = np.meshgrid(ses.celldata['roi_name']==area1,ses.celldata['roi_name']==area2)
    areafilter1 = np.logical_and(areafilter1[0],areafilter1[1])
    areafilter2 = np.meshgrid(ses.celldata['roi_name']==area1,ses.celldata['roi_name']==area2)
    areafilter2 = np.logical_and(areafilter2[0],areafilter2[1])

    return np.logical_or(areafilter1,areafilter2)

#%% ##########################################################################################################
# DELTA ANATOMICAL DISTANCE :
# ##########################################################################################################

def bin_corr_distance(sessions,areapairs,corr_type='trace_corr',normalize=False):
    binedges = np.arange(0,1000,10) 
    nbins= len(binedges)-1
    binmean = np.full((len(sessions),len(areapairs),nbins),np.nan)
    for ises in tqdm(range(len(sessions)),desc= 'Computing trace correlations per pairwise distance: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            corrdata[corrdata>0] = np.nan
            for iap,areapair in enumerate(areapairs):
                areafilter      = filter_2d_areapair(sessions[ises],areapair)
                nanfilter       = ~np.isnan(corrdata)
                cellfilter      = np.all((areafilter,nanfilter),axis=0)
                binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_xyz[cellfilter].flatten(),
                                                    values=corrdata[cellfilter].flatten(),
                                                    statistic='mean', bins=binedges)[0]
            
    if normalize: # subtract mean NC from every session:
        binmean = binmean - np.nanmean(binmean[:,:,binedges[:-1]<600],axis=2,keepdims=True)

    return binmean,binedges

def plot_bin_corr_distance(sessions,binmean,binedges,corr_type):
    sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    protocols = np.unique(sessiondata['protocol'])

    fig,axes = plt.subplots(1,len(protocols),figsize=(4*len(protocols),4))
    handles = []
    for iprot,protocol in enumerate(protocols):
        sesidx = np.where(sessiondata['protocol']== protocol)[0]
        ax = axes[iprot]

        for iap,areapair in enumerate(areapairs):
            for ises in sesidx:
                ax.plot(binedges[:-1],binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
            handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[sesidx,iap,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
            # plt.savefig(os.path.join(savedir,'NoiseCorr_distRF_RegressOut_' + areapair + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

        ax.legend(handles,areapairs,loc='upper right',frameon=False)	
        ax.set_xlabel('Anatomical distance ($\mu$m)')
        ax.set_ylabel('Correlation')
        ax.set_xlim([10,600])
        ax.set_title('%s (%s)' % (corr_type,protocol))
        # ax.set_ylim([-0.015,0.04])
        # ax.set_ylim([0,0.08])
        ax.set_aspect('auto')
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    return fig

#%% Define the areapairs:
areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

#%% Compute pairwise trace correlations as a function of pairwise anatomical distance ###################################################################

[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='trace_corr')

#%% Make the figure per protocol:

fig = plot_bin_corr_distance(sessions,binmean,binedges,corr_type='trace_corr')
plt.savefig(os.path.join(savedir,'TraceCorr_XYZdist_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')

#%% Compute pairwise signal correlations as a function of pairwise anatomical distance ###################################################################
[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='sig_corr')

#%% Make the figure per protocol:
fig = plot_bin_corr_distance(sessions,binmean,binedges,corr_type='sig_corr')
plt.savefig(os.path.join(savedir,'SignalCorr_XYZdist_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')


# ##########################################################################################################
# DELTA RECEPTIVE FIELD:
# ##########################################################################################################


#%% ################ Pairwise trace correlations as a function of pairwise delta RF: #####################
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

binedges = np.arange(0,120,2.5) 
nbins= len(binedges)-1
binmean = np.full((nSessions,len(areapairs),nbins),np.nan)

handles = []
for ises in tqdm(range(nSessions),desc= 'Computing trace correlations per delta receptive field: '):
    if 'rf_p_F' in sessions[ises].celldata:
        for iap,areapair in enumerate(areapairs):
            signalfilter    = np.meshgrid(sessions[ises].celldata['rf_p_F']<0.001,sessions[ises].celldata['rf_p_F']<0.001)
            signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])
            
            areafilter      = filter_2d_areapair(sessions[ises],areapair)

            proxfilter      = (sessions[ises].distmat_xy>20) | (np.isnan(sessions[ises].distmat_xy))
            nanfilter       = ~np.isnan(sessions[ises].trace_corr)
            cellfilter      = np.all((signalfilter,areafilter,proxfilter,nanfilter),axis=0)

            binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_rf[cellfilter].flatten(),
                                                    values=sessions[ises].trace_corr[cellfilter].flatten(),
                                                    statistic='mean', bins=binedges)[0]

#%% Normalize the result: 
# subtract mean NC from every session:
binmean = binmean - np.nanmean(binmean[:,:,binedges[:-1]<50],axis=2,keepdims=True)

#%% Make the figure per protocol:
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
protocols = np.unique(sessiondata['protocol'])

fig,axes = plt.subplots(1,len(protocols),figsize=(4*len(protocols),4))
for iprot,protocol in enumerate(protocols):
    sesidx = np.where(sessiondata['protocol']== protocol)[0]
    ax = axes[iprot]

    for iap,areapair in enumerate(areapairs):
        for ises in sesidx:
            ax.plot(binedges[:-1],binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
        handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[sesidx,iap,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
        # plt.savefig(os.path.join(savedir,'NoiseCorr_distRF_RegressOut_' + areapair + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    ax.legend(handles,areapairs)
    ax.set_xlabel('Delta RF')
    ax.set_ylabel('Delta NoiseCorrelation')
    ax.set_xlim([-2,50])
    ax.set_title('Trace correlation (%s)' %protocol)
    # ax.set_ylim([-0.015,0.04])
    ax.set_ylim([0,0.08])
    ax.set_aspect('auto')
    ax.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'TraceCorr_distRF_Protocols_%dsessions_' %nSessions + areapair + '.png'), format = 'png')


# ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################




#%% 
rotate_prefori  = False
min_counts      = 500 # minimum pairwise observation to include bin

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap(sessions,corr_type='trace_corr',binresolution=5,
                                                         rotate_prefori=rotate_prefori,
                                                         rotate_deltaprefori=False)
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

## Show the counts of pairs:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(countsRFmat,vmin=np.percentile(countsRFmat,5),vmax=np.percentile(countsRFmat,99),interpolation='none',extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='counts')
if not rotate_prefori:
    plt.xlabel('delta Azimuth')
    plt.ylabel('delta Elevation')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Counts_%dsessions' %nSessions  + '.png'), format = 'png')
else:
    plt.xlabel('Collinear')
    plt.ylabel('Orthogonal')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Counts_Rotated_%dsessions' %nSessions  + '.png'), format = 'png')

## Show the noise correlation map:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(noiseRFmat_mean,vmin=np.nanpercentile(noiseRFmat_mean,5),vmax=np.nanpercentile(noiseRFmat_mean,95),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='noise correlation')
if not rotate_prefori:
    plt.xlabel('delta Azimuth')
    plt.ylabel('delta Elevation')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_%dsessions' %nSessions  + '.png'), format = 'png')
else:
    plt.xlabel('Collinear')
    plt.ylabel('Orthogonal')


#%% #########################################################################################
# Contrast: across areas
areas   = ['V1','PM']

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap_areas(sessions,corr_type='trace_corr',binresolution=7.5,
                                                                 rotate_prefori=False,thr_tuned=0.0,
                                                                 thr_rf_p=0.001)

min_counts = 500
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(2,2,figsize=(10,7))
for i in range(2):
    for j in range(2):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(areas[i] + '-' + areas[j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Interarea_AllProt_%dsessions' %nSessions  + '.png'), format = 'png')

#%% #########################################################################################
# Contrasts: across areas and projection identity      

[noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions,corr_type='trace_corr',binresolution=10,
                                                                 rotate_prefori=False,thr_tuned=0.0,
                                                                 thr_rf_p=0.01)

min_counts = 50
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_AllProt_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_Map_smooth_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')

