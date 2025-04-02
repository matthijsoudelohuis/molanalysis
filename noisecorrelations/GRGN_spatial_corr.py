# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2022-2026, Champalimaud Center, Lisbon
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
from scipy.signal import detrend
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from preprocessing.preprocesslib import assign_layer
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error,my_ceil,my_floor
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import compute_tuning_wrapper
from utils.explorefigs import plot_excerpt
from utils.shuffle_lib import my_shuffle, corr_shuffle
from utils.gain_lib import * 
from utils.arrayop_lib import nanweightedaverage
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])

session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=['V1','PM'],session_rf=True)  
# sessions,nSessions   = filter_sessions(protocols = ['RF'],filter_areas=['V1','PM'],session_rf=True)  

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
# sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE10885_2023_10_19','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
# calciumversion = 'dF'
calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=True)
                                # calciumversion=calciumversion,keepraw=True,filter_hp=0.01)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    # sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.2,uppertriangular=False)[0]
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ##################### Smooth receptive field from Fneu gauss fit again #################
# This is overriding any Fsmooth values from preprocessing
# sessions = smooth_rf(sessions,radius=50,rf_type='Fneu',mincellsFneu=5)
# sessions = exclude_outlier_rf(sessions) 
# sessions = replace_smooth_with_Fsig(sessions) 

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% ########################## Compute signal and noise correlations: ############################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=True)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False,remove_method='RRR',remove_rank=2)

# plt.imshow(sessions[0].noise_corr,vmin=-0.03,vmax=0.05)

# #%% 
# sessions = corr_shuffle(sessions,method='random')

#%%
colors = [(0, 0, 0), (1, 0, 0), (1, 1, 1)] # first color is black, last is red
cm_red = LinearSegmentedColormap.from_list("Custom", colors, N=20)
colors = [(0, 0, 0), (0, 0, 1), (1, 1, 1)] # first color is black, last is red
cm_blue = LinearSegmentedColormap.from_list("Custom", colors, N=20)

centerthr           = [15,15,15,15]

# 2024_11_03:
# df or deconv? dF
# Fsmooth of Fneu overriding
# taking Fsmooth RF as rf_type
# noise correlations without any gain subtraction or PCA removal
# Results: within area NC drop with distance, across areas, NC increase with distance
# V1-PM lab and unl show different pattern: V1lab-PMunl show decrease that increases with delta RF
# V1unl-PMlab show peak at intermediate delta RF (30 deg)
# V1lab-PMlab too little data?
# for across area comparison (without labeling), no filternear to get more neuron pairs


#%% #####################################################################################################
# DELTA ANATOMICAL DISTANCE :
# #######################################################################################################

#%% Define the areapairs:
areapairs       = ['V1-V1','PM-PM']
clrs_areapairs  = get_clr_area_pairs(areapairs)

#%% Compute pairwise correlations as a function of pairwise anatomical distance ###################################################################
for corr_type in ['trace_corr','sig_corr','noise_corr']:
# for corr_type in ['noise_corr']:
    [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type=corr_type)

    #Make the figure per protocol:
    fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type=corr_type)
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.pdf'), format = 'pdf')

#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM']
layerpairs          = ' '
projpairs           = ' '
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'
# corr_type           = 'sig_corr'
corr_thr            = 0.025 #thr in percentile of total corr for significant pos or neg

[bins2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,binsdRF,
bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
bin_angle_surr_count,binsangle] = bin_corr_deltaxy(sessions,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,binresolution=5,noise_thr=100)

[_,bin_2d_posf,_,bin_dist_posf,_,_,bin_angle_cent_posf,_,bin_angle_surr_posf,_,_] = bin_corr_deltaxy(sessions,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='frac',filtersign='pos',corr_type=corr_type,binresolution=5,noise_thr=100,corr_thr=corr_thr)

[_,bin_2d_negf,_,bin_dist_negf,_,_,bin_angle_cent_negf,_,bin_angle_surr_negf,_,_] = bin_corr_deltaxy(sessions,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='frac',filtersign='neg',corr_type=corr_type,binresolution=5,noise_thr=100,corr_thr=corr_thr)

fig = plot_mean_frac_corr_areas(bins2d,bin_2d_count,bin_2d_mean,bin_2d_posf,bin_2d_negf,
                        binsdRF,bin_dist_count,bin_dist_mean,bin_dist_posf,bin_dist_negf,areapairs,layerpairs,projpairs)
# fig.savefig(os.path.join(savedir,'deltaXYZ','Corr_deltaXY_%s_%s_GRGN' % (corr_type,calciumversion) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'deltaXYZ','Corr_deltaXY_%s_GMremoved_%s_GRGN' % (corr_type,calciumversion) + '.png'), format = 'png')

# %% #######################################################################################################
# DELTA RECEPTIVE FIELD:
# ##########################################################################################################

# Sessions with good RF:
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

#%% Reassign layers:
for ses in sessions:
    ses.celldata = assign_layer(ses.celldata)


#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)


#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM','V1-PM']
# layerpairs          = ['L2/3-L2/3']
# layerpairs          = ['L5-L5']
layerpairs          = ' '
projpairs           = ' '

[bin_dist_mean_ses,bin_dist_count_ses,binsdRF] = bin_corr_deltarf_ses_vkeep(sessions,rf_type=rf_type,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,noise_thr=20,
                        binresolution=binresolution,normalize=False)

binres              = 0.005
binedges            = np.arange(-1,1,binres)
bincenters          = binedges[:-1] + binres/2
nbins               = len(bincenters)

hist_mean_ses = np.full(list(np.shape(bin_dist_mean_ses))[:-1] + [nbins],np.nan)

for idx in np.ndindex(*bin_dist_mean_ses.shape[:-1]):
    hist_mean_ses[idx],_ = np.histogram(bin_dist_mean_ses[idx],bins=binedges)

hist_mean_ses /= np.sum(hist_mean_ses,axis=-1,keepdims=True)

# plt.plot(bincenters,np.nanmean(hist_mean_ses,axis=(0,1,2,3,4)))
plt.plot(bincenters,np.nanmean(hist_mean_ses[:,:2,:,:,:,:],axis=(0,1,2,3)).T)
plt.plot(bincenters,np.nanmean(hist_mean_ses[:,:3,0,:,:,:],axis=(0,1,2)).T)
plt.xlim([-0.1,0.1])


#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM','V1-PM']
# areapairs           = ['V1-PM']
# areapairs           = ['PM-PM']
layerpairs          = ['L2/3-L2/3']
# layerpairs          = ['L2/3-L5']
# layerpairs          = ['L5-L5']
# layerpairs          = ' '
projpairs           = ' '

# corr_thr            = 0.025 #thr in percentile of total corr for significant pos or neg
corr_thr            = 0.01 #thr in percentile of total corr for significant pos or neg
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'
# corr_type           = 'sig_corr'
min_counts          = 50
gaussian_sigma      = 1
binresolution       = 10

[bins2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,binsdRF,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,binsangle] = bin_corr_deltarf_ses(sessions,rf_type=rf_type,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,noise_thr=20,
                        binresolution=binresolution,normalize=False)

[_,bin_2d_posf_ses,_,bin_dist_posf_ses,_,_,
 bin_angle_cent_posf_ses,_,bin_angle_surr_posf_ses,_,_] = bin_corr_deltarf_ses(sessions,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,
                        method='frac',filtersign='pos',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr,
                        binresolution=binresolution)

[_,bin_2d_negf_ses,_,bin_dist_negf_ses,_,_,bin_angle_cent_negf_ses,_,bin_angle_surr_negf_ses,_,_] = bin_corr_deltarf_ses(sessions,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,
                        method='frac',filtersign='neg',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr,
                        binresolution=binresolution)

# [bins2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,binsdRF,
# bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
# bin_angle_surr_count,binsangle] = bin_corr_deltarf(sessions,rf_type=rf_type,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
#                         method='mean',filtersign=None,corr_type=corr_type,noise_thr=20,normalize=False)

# [_,bin_2d_posf,_,bin_dist_posf,_,_,bin_angle_cent_posf,_,bin_angle_surr_posf,_,_] = bin_corr_deltarf(sessions,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,method='frac',
#                         filtersign='pos',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr)

# [_,bin_2d_negf,_,bin_dist_negf,_,_,bin_angle_cent_negf,_,bin_angle_surr_negf,_,_] = bin_corr_deltarf(sessions,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,method='frac',
#                         filtersign='neg',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr)

#%% 
plt.hist(bin_2d_count_ses.flatten(),bins=np.arange(0,1e5,10000))
plt.ylim([0,100])

#%% Plot radial tuning:
filestring = 'RadialTuning_areas_%s_%s_' % (corr_type,layerpairs[0].replace('/',''))
# filestring = 'RadialTuning_areas_%sGM_%s_' % (corr_type,layerpairs[0].replace('/',''))

fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean_sessions' + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean_sessions' + '.pdf'), format = 'pdf')

fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses,bin_dist_posf_ses,areapairs,layerpairs,projpairs,datatype='Fraction Pos.')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'posf_sessions' + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'posf_sessions' + '.pdf'), format = 'pdf')

fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses,bin_dist_negf_ses,areapairs,layerpairs,projpairs,datatype='Fraction Neg.')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'negf_sessions' + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'negf_sessions' + '.pdf'), format = 'pdf')

fig = plot_corr_radial_tuning_areas(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs)
# fig = plot_corr_radial_tuning_areas(binsdRF,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean' + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean' + '.pdf'), format = 'pdf')

fig = plot_corr_radial_tuning_areas(binsdRF,bin_dist_count_ses,bin_dist_posf_ses,areapairs,layerpairs,projpairs,datatype='Fraction Pos.')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'posf' + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'posf' + '.pdf'), format = 'pdf')

fig = plot_corr_radial_tuning_areas(binsdRF,bin_dist_count_ses,bin_dist_negf_ses,areapairs,layerpairs,projpairs,datatype='Fraction Neg.')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'negf' + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'negf' + '.pdf'), format = 'pdf')


#%% 1D delta RF weighted mean across sessions:

bin_dist_mean = nanweightedaverage(bin_dist_mean_ses,weights=bin_dist_count_ses,axis=0)
bin_dist_posf = nanweightedaverage(bin_dist_posf_ses,weights=bin_dist_count_ses,axis=0)
bin_dist_negf = nanweightedaverage(bin_dist_negf_ses,weights=bin_dist_count_ses,axis=0)
bin_dist_count = np.nansum(bin_dist_count_ses,axis=0)

fig = plot_corr_radial_tuning_areas_mean(binsdRF,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs)

#%%





#%% #########################################################################################
# Contrast: across projections:
areapairs           = ['V1-V1','PM-PM','V1-PM']
# areapairs           = ['V1-PM']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
layerpairs          = ' '
# layerpairs          = ['L2/3-L2/3']
# layerpairs          = ['L2/3-L5']

corr_thr            = 0.01 #thr in percentile of total corr for significant pos or neg
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'
# corr_type           = 'sig_corr'
binresolution       = 10
shufflefield        = None
# shufflefield        = 'corrdata'
# shufflefield      = 'labeled'
shufflefield      = 'RF'

# [bins2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,binsdRF,
# bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
# bin_angle_surr_count,binsangle] = bin_corr_deltarf(sessions,rf_type=rf_type,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
#                         method='mean',filtersign=None,corr_type=corr_type,noise_thr=20,
#                         binresolution=7.5,normalize=False)

# [_,bin_2d_posf,_,bin_dist_posf,_,_,bin_angle_cent_posf,_,bin_angle_surr_posf,_,_] = bin_corr_deltarf(sessions,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,
#                         method='frac',filtersign='pos',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr,
#                         binresolution=7.5)

# [_,bin_2d_negf,_,bin_dist_negf,_,_,bin_angle_cent_negf,_,bin_angle_surr_negf,_,_] = bin_corr_deltarf(sessions,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,
#                         method='frac',filtersign='neg',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr,
#                         binresolution=7.5)

[bins2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,binsdRF,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,binsangle] = bin_corr_deltarf_ses(sessions,rf_type=rf_type,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,noise_thr=20,
                        binresolution=binresolution,normalize=False,shufflefield=shufflefield)

# [_,bin_2d_posf_ses,_,bin_dist_posf_ses,_,_,
#  bin_angle_cent_posf_ses,_,bin_angle_surr_posf_ses,_,_] = bin_corr_deltarf_ses(sessions,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,
#                         method='frac',filtersign='pos',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr,
#                         binresolution=binresolution,shufflefield=shufflefield)

# [_,bin_2d_negf_ses,_,bin_dist_negf_ses,_,_,bin_angle_cent_negf_ses,_,bin_angle_surr_negf_ses,_,_] = bin_corr_deltarf_ses(sessions,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,
#                         method='frac',filtersign='neg',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr,
#                         binresolution=binresolution,shufflefield=shufflefield)

# bin_2d_mean = nanweightedaverage(bin_2d_mean_ses,weights=bin_2d_count_ses,axis=0)
# bin_2d_posf = nanweightedaverage(bin_2d_posf_ses,weights=bin_2d_count_ses,axis=0)
# bin_2d_negf = nanweightedaverage(bin_2d_negf_ses,weights=bin_2d_count_ses,axis=0)
# bin_2d_count = np.nansum(bin_2d_count_ses,axis=0)

# bin_dist_mean = nanweightedaverage(bin_dist_mean_ses,weights=bin_dist_count_ses,axis=0)
# bin_dist_posf = nanweightedaverage(bin_dist_posf_ses,weights=bin_dist_count_ses,axis=0)
# bin_dist_negf = nanweightedaverage(bin_dist_negf_ses,weights=bin_dist_count_ses,axis=0)
# bin_dist_count = np.nansum(bin_dist_count_ses,axis=0)

#%% Plot radial tuning:
fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs,datatype='Correlation')
# fig,fig2 = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs,datatype='Correlation')
# fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs,datatype='Correlation')
# axes = fig.get_axes()
# axes[0].set_ylim([-0.05,0.05])

fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN_shuf' % (corr_type) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN_shuf' % (corr_type) + '.pdf'), format = 'pdf')
# fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN' % (corr_type) + '.pdf'), format = 'pdf')
# fig2.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN_stats' % (corr_type) + '.png'), format = 'png')
# fig2.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN_stats' % (corr_type) + '.pdf'), format = 'pdf')

fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses,bin_dist_posf_ses,areapairs,layerpairs,projpairs,datatype='Fraction')
fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_posf_GRGN' % (corr_type) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_posf_GRGN' % (corr_type) + '.pdf'), format = 'pdf')
fig2.savefig(os.path.join(savedir,'RadialTuning_projs_%s_posf_GRGN_stats' % (corr_type) + '.png'), format = 'png')
fig2.savefig(os.path.join(savedir,'RadialTuning_projs_%s_posf_GRGN_stats' % (corr_type) + '.pdf'), format = 'pdf')

fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses,bin_dist_negf_ses,areapairs,layerpairs,projpairs,datatype='Fraction')
fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_negf_GRGN' % (corr_type) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_negf_GRGN' % (corr_type) + '.pdf'), format = 'pdf')
fig2.savefig(os.path.join(savedir,'RadialTuning_projs_%s_negf_GRGN_stats' % (corr_type) + '.png'), format = 'png')
fig2.savefig(os.path.join(savedir,'RadialTuning_projs_%s_negf_GRGN_stats' % (corr_type) + '.pdf'), format = 'pdf')


#%% 
# fig = plot_mean_corr_projs(binsdRF,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs)
# # fig.savefig(os.path.join(savedir,'deltaRF','Corr_deltaRF_V1PM_%s_%s_%s_GMsub_projs_GRGN' % (rf_type,corr_type,calciumversion) + '.pdf'), format = 'pdf')
# # fig.savefig(os.path.join(savedir,'deltaRF','Corr_deltaRF_V1PM_%s_%s_%s_GMsub_projs_GRGN' % (rf_type,corr_type,calciumversion) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'deltaRF','1DRF','Corr_deltaRF_V1PM_%s_%s_%s_projs_GRGN' % (rf_type,corr_type,calciumversion) + '.pdf'), format = 'pdf')
# fig.savefig(os.path.join(savedir,'deltaRF','1DRF','Corr_deltaRF_V1PM_%s_%s_%s_projs_GRGN' % (rf_type,corr_type,calciumversion) + '.png'), format = 'png')

# fig = plot_mean_frac_corr_projs(bins2d,bin_2d_count,bin_2d_mean,bin_2d_posf,bin_2d_negf,
#                         binsdRF,bin_dist_count,bin_dist_mean,bin_dist_posf,bin_dist_negf,areapairs,layerpairs,projpairs)
# fig.savefig(os.path.join(savedir,'deltaRF','1DRF','CorrFrac_deltaRF_V1PM_%s_%s_%s_projs_GRGN' % (rf_type,corr_type,calciumversion) + '.png'), format = 'png')

#%% Reassign layers:
for ses in sessions:
    ses.celldata = assign_layer(ses.celldata)

#%% #########################################################################################
# Contrast: across layers:
areapairs           = ['V1-PM']
projpairs           = ' '
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']

corr_thr            = 0.025 #thr in percentile of total corr for significant pos or neg
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

[bins2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,binsdRF,
bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
bin_angle_surr_count,binsangle] = bin_corr_deltarf(sessions,rf_type=rf_type,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,method='mean',
                        filtersign=None,corr_type=corr_type,noise_thr=20)

[_,bin_2d_posf,_,bin_dist_posf,_,_,bin_angle_cent_posf,_,bin_angle_surr_posf,_,_] = bin_corr_deltarf(sessions,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,method='frac',
                        filtersign='pos',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr)

[_,bin_2d_negf,_,bin_dist_negf,_,_,bin_angle_cent_negf,_,bin_angle_surr_negf,_,_] = bin_corr_deltarf(sessions,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,method='frac',
                        filtersign='neg',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr)

fig = plot_mean_corr_layers(binsdRF,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'deltaRF','Corr_deltaRF_V1PM_%s_%s_layers_GRGN' % (rf_type,corr_type) + '.png'), format = 'png')

fig = plot_mean_frac_corr_projs(bins2d,bin_2d_count,bin_2d_mean,bin_2d_posf,bin_2d_negf,
                        binsdRF,bin_dist_count,bin_dist_mean,bin_dist_posf,bin_dist_negf,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'deltaRF','Corr_deltaRF_V1PM_%s_%s_projs_GRGN' % (rf_type,corr_type) + '.png'), format = 'png')





# #%% #########################################################################################
# # Contrast: across areas, layers and projection pairs:
# areapairs           = ['V1-V1','PM-PM','V1-PM']
# layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
# projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
# #If you override any of these with input to the deltarf bin function as ' ', then these pairs will be ignored

# #%%
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
# # sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)

# #%% Make the 2D, 1D and center surround averages for each protocol and areapair and projpair (not layerpair)
# binres              = 5
# # rf_type             = 'Fneugauss'
# rf_type             = 'Fsmooth'
# filtersign          = None
# filternear          = True
# corr_type           = 'sig_corr'
# # corr_type           = 'noise_corr'
# # corr_type           = 'trace_corr'

# # for prot in ['GN','GR','IM','SP','RF']:
# for prot in [['GN','GR']]:
# # for prot in ['RF']:
# # for prot in ['IM']:
# # for prot in ['GR']:
#     sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
#     # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
#     sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
#     sessions_subset     = [sessions[i] for i in sessions_in_list]

#     [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
#                                 corr_type=corr_type,binresolution=binres,rf_type=rf_type,normalize=False,
#                                 sig_thr = 0.001,filternear=filternear)
    
#     # filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
#     filestring = '%s_%s_%s_proj' % (corr_type,rf_type,prot)

#     if np.any(bincounts):
#         fig = plot_2D_corr_map(binmean,bincounts,bincenters,min_counts = 1,gaussian_sigma=1,
#                                 areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
#         fig.suptitle(filestring)
#         plt.tight_layout()
#         # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.png'), format = 'png')
#         # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')
    
#         fig = plot_1D_corr_areas_projs(binmean,bincounts,bincenters,min_counts = 1,
#                                 areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
#         fig.suptitle(filestring)
#         plt.tight_layout()
#         # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.png'), format = 'png')
#         # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.pdf'), format = 'pdf')

#         fig = plot_center_surround_corr_areas_projs(binmean,bincenters,centerthr=15,layerpairs=' ',areapairs=areapairs,projpairs=projpairs)
#         fig.suptitle(filestring)
#         plt.tight_layout()
#         # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.png'), format = 'png')
#         # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.pdf'), format = 'pdf')

#%% Make a 2D histogram with the distribution of correlation values for each delta RF bin
binres_rf           = 5
binres_corr         = 0.1
rf_type             = 'Fsmooth'
filternear          = False
# corr_type           = 'noise_corr'
corr_type           = 'trace_corr'
# corr_type           = 'trace_corr'
# corr_type           = 'sig_corr'

#%% #########################################################################################
# Contrast: across areas, layers and projection pairs:
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
#If you override any of these with input to the deltarf bin function as ' ', then these pairs will be ignored

for prot in [['GN','GR']]:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    # [bincounts,bincenters_rf,bincenters_corr] = bin_2d_rangecorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=['L2/3-L2/3'],projpairs=projpairs,
    #                             corr_type=corr_type,binres_rf=binres_rf,binres_corr=binres_corr,rf_type=rf_type,
    #                             r2_thr=0.2,filternear=filternear,noise_thr=20)

    [bincounts,bincenters_rf,bincenters_corr] = bin_2d_rangecorr_deltarf(sessions_subset,
                    areapairs=areapairs,layerpairs=' ',projpairs=projpairs,corr_type=corr_type,
                    binres_rf=binres_rf,binres_corr=binres_corr,rf_type=rf_type,r2_thr=0.2,filternear=filternear,noise_thr=20)
    
    # filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
    filestring = '%s_%s_%s_proj' % (corr_type,rf_type,calciumversion)
    # filestring = '%s_%s_%s_GMsub_proj' % (corr_type,rf_type,calciumversion)

    if np.any(bincounts):
        fig = plot_2D_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,gaussian_sigma=1.5,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_rangecorr_norm0_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_rangecorr_norm0_%s' % filestring + '.pdf'), format = 'pdf')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaXY_2D_%s' % filestring + '.png'), format = 'png')

# %% 
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% Make a 2D histogram with the distribution of correlation values for each delta RF bin
binres_rf           = 5
rf_type             = 'Fsmooth'
filternear          = False
# corr_type           = 'noise_corr'
corr_type           = 'trace_corr'
# corr_type           = 'sig_corr'

# RF based on interpolation of Fneu of good cells, so Fsmooth
# NOT yet noise corr without GM substraction
# trace_corr!! 
# noise thr = 20
# corr thr = 0.025 or 0.05
# r2_thr = 0.2
# Filter near is false
# min_counts = 100
# filestring = '%s_%s_%s_GMsub_proj' % (corr_type,rf_type,calciumversion)
filestring = '%s_%s_%s_proj' % (corr_type,rf_type,calciumversion)

for prot in [['GN','GR']]:
    filestring          = '%s_%s_%s_proj' % (corr_type,rf_type,calciumversion)
    [bincounts,binpos,binneg,bincenters_rf] = bin_1d_fraccorr_deltarf(sessions,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,corr_type=corr_type,binres_rf=binres_rf,rf_type=rf_type,
                                r2_thr = 0.2,filternear=filternear,noise_thr=20,corr_thr=0.025)

    fig = plot_1D_fraccorr(bincounts,binpos,binneg,bincenters_rf,areapairs=areapairs,layerpairs=' ',
                           projpairs=projpairs,mincounts=100)
    # fig = plot_1D_fraccorr(bincounts,binpos,binneg,bincenters_rf,areapairs=areapairs,
                        #    layerpairs=' ',projpairs=projpairs,mincounts=100,normalize_rf=False)
    fig.savefig(os.path.join(savedir,'deltaRF','1DRF','Frac_PosNeg_DeltaRF_GNGR_%s' % filestring + '.pdf'), format = 'pdf')
    fig.savefig(os.path.join(savedir,'deltaRF','1DRF','Frac_PosNeg_DeltaRF_GNGR_%s' % filestring + '.png'), format = 'png')

#%% Make a 2D histogram with the distribution of correlation values for each delta XY bin
binres_rf           = 5
binres_corr         = 0.1
rf_type             = 'Fsmooth'
filternear          = False
corr_type           = 'noise_corr'
# corr_type           = 'sig_corr'
# corr_type           = 'trace_corr'

# for prot in ['GN','GR','IM','SP','RF']:
for prot in [['GN','GR']]:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [bincounts,bincenters_rf,bincenters_corr] = bin_2d_rangecorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
                                corr_type=corr_type,binres_rf=binres_rf,binres_corr=binres_corr,rf_type=rf_type,
                                r2_thr=0.2,filternear=filternear,noise_thr=20)
    
    # [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
    #                             corr_type=corr_type,binresolution=binres,rf_type=rf_type,normalize=False,
    #                             sig_thr = 0.001,filternear=filternear,filtersign=filtersign)
    
    filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
    # filestring = '%s_%s_%s_proj' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        fig = plot_2D_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,gaussian_sigma=1.5,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaXY_2D_%s' % filestring + '.png'), format = 'png')


#%% 

#%% Control figure of counts per bin:

# Make the figure of the counts per bin:
fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax                                              = axes[ipp,iap]
            data                                            = np.log10(bincounts[:,:,iap,ilp,ipp])
            ax.pcolor(delta_az,delta_el,data,vmin=1.5,
                                vmax=np.nanpercentile(np.log10(bincounts),99.9),cmap="hot")
            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([-50,50])
            ax.set_ylim([-50,50])
            ax.set_xlabel(u'Δ Azimuth')
            ax.set_ylabel(u'Δ Elevation')
plt.tight_layout()


# binmean = np.nanmean(binmean_ses,axis=5)
# bincounts = np.nansum(bincounts_ses,axis=5)

# binmean_ses[bincounts_ses<10]     = np.nan
