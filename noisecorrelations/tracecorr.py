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
from utils.plot_lib import shaded_error
# from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])

# load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['SP','GN','RF'],filter_areas=['V1','PM']) 

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.75,uppertriangular=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% ########################### Compute noise correlations: ###################################
# sessions = compute_trace_correlation(sessions,binwidth=0.25,uppertriangular=False)

#%% ########################## Compute signal correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% ##################### Compute pairwise neuronal distances: ##############################
# sessions = compute_pairwise_metrics(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ##################### Compute pairwise receptive field distances: ##############################
sessions = exclude_outlier_rf(sessions,radius=50,rf_thr=50) 
sessions = smooth_rf(sessions,radius=50)
sessions = compute_pairwise_delta_rf(sessions,rf_type='F')
# sessions = compute_pairwise_delta_rf(sessions,rf_type='Fsmooth')

np.save(os.path.join('e:\\Procdata\\','AllProtocols_corrdata_n84sessions.npy'),sessions,allow_pickle = True)

#%% ##########################################################################################################
# DELTA ANATOMICAL DISTANCE :
# ##########################################################################################################

#%% Define the areapairs:
areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

#%% Compute pairwise trace correlations as a function of pairwise anatomical distance ###################################################################

[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='trace_corr')

#%% Make the figure per protocol:

fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='trace_corr')
fig.savefig(os.path.join(savedir,'TraceCorr_XYZdist_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')

#%% Compute pairwise signal correlations as a function of pairwise anatomical distance ###################################################################
[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='sig_corr')

#%% Make the figure per protocol:
fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='sig_corr')
fig.savefig(os.path.join(savedir,'SignalCorr_XYZdist_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')

#%% Compute pairwise noise correlations as a function of pairwise anatomical distance ###################################################################
[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='noise_corr')

#%% Make the figure per protocol:
fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='noise_corr')
fig.savefig(os.path.join(savedir,'NoiseCorr_XYZdist_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')

# %% #######################################################################################################
# DELTA RECEPTIVE FIELD:
# ##########################################################################################################

session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                ['LPE11998','2024_05_02'], #GN
                                ['LPE12013','2024_05_02'], #GN
                                ['LPE12013','2024_05_07'], #GN
                                ['LPE11086','2023_12_15'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['session_id'].isin([x[0] + '_' + x[1] for x in session_list]))[0]
sessions_subset = [sessions[i] for i in sessions_in_list]

#%% ################ Pairwise trace correlations as a function of pairwise delta RF: #####################
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

[binmean,binedges] =  bin_corr_deltarf(sessions,areapairs,corr_type='trace_corr',normalize=False,rf_type = 'F')

#%% Make the figure:
fig = plot_bin_corr_deltarf(sessions,binmean,binedges,areapairs,corr_type='trace_corr')

fig.savefig(os.path.join(savedir,'TraceCorr_distRF_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_RF_%dsessions_' %nSessions + '.png'), format = 'png')


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
# plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Interarea_AllProt_%dsessions' %nSessions  + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Interarea_RF_%dsessions' %nSessions  + '.png'), format = 'png')

#%% #########################################################################################
# Contrasts: across areas and projection identity      

[noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions,corr_type='trace_corr',binresolution=10,
                                                                 rotate_prefori=False,thr_tuned=0.0,
                                                                 thr_rf_p=0.001)

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
# plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_RF_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_Map_smooth_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')

