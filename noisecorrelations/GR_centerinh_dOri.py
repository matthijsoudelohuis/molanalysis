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

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import binned_statistic,binned_statistic_2d
from scipy.signal import detrend
from statannotations.Annotator import Annotator
from scipy.ndimage import gaussian_filter

from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import compute_tuning_wrapper

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\Collinear\\')


colors = [(0, 0, 0), (1, 0, 0), (1, 1, 1)] # first color is black, last is red
cm_red = LinearSegmentedColormap.from_list("Custom", colors, N=20)
colors = [(0, 0, 0), (0, 0, 1), (1, 1, 1)] # first color is black, last is red
cm_blue = LinearSegmentedColormap.from_list("Custom", colors, N=20)

centerthr           = [15,15,15,15]
centerthr           = [20,20,20,20]
gaussian_sigma = 1

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM'],session_rf=True) 

#%% Remove two sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:      
# calciumversion = 'deconv'
calciumversion = 'dF'

for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=True)
                                # calciumversion=calciumversion,keepraw=True,filter_hp=0.01)
                                # calciumversion=calciumversion,keepraw=True)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    # delattr(sessions[ises],'calciumdata')


#%% 
def ori_remapping(sessions):
    for ises in range(nSessions):
        if sessions[ises].sessiondata['protocol'][0] == 'GR':
            sessions[ises].trialdata['Orientation_orig']    = sessions[ises].trialdata['Orientation']
            sessions[ises].trialdata['Orientation']         = np.mod(270 - sessions[ises].trialdata['Orientation'],360)
    return sessions

#%%
sessions = ori_remapping(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=True,remove_method='PCA',remove_rank=1)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ##################### Compute pairwise receptive field distances: ##############################
# sessions = smooth_rf(sessions,radius=50,rf_type='Fneu',mincellsFneu=5)
# sessions = exclude_outlier_rf(sessions) 
# sessions = replace_smooth_with_Fsig(sessions) 








     #    ####### ######  ###    ######  ######  #######       #  #####  
     #    #     # #     #  #     #     # #     # #     #       # #     # 
     #    #     # #     #  #     #     # #     # #     #       # #       
######    #     # ######   #     ######  ######  #     #       #  #####  
#    #    #     # #   #    #     #       #   #   #     # #     #       # 
#    #    #     # #    #   #     #       #    #  #     # #     # #     # 
######    ####### #     # ###    #       #     # #######  #####   #####  

#%% Loop over all delta preferred orientations and store mean correlations as well as distribution of pos and neg correlations:
areapairs           = ['V1-PM']
layerpairs          = ' '
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    # sessions[ises].delta_pref = np.abs(np.mod(dpref,180))
    # sessions[ises].delta_pref[dpref == 180] = 180
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)

deltaoris           = np.unique(sessions[0].delta_pref[~np.isnan(sessions[0].delta_pref)])
ndeltaoris          = len(deltaoris)
rotate_prefori      = True
rf_type             = 'Fsmooth'
# corr_type           = 'noise_corr'
corr_type           = 'trace_corr'
tuned_thr           = 0.025
noise_thr           = 20
min_counts          = 100
# corr_thr            = 0.01
corr_thr            = 0.025 #for prctile
binresolution       =  10
#Do for one session to get the dimensions: (data is discarded)
[bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
    bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
    bin_angle_surr_count,bincenters_angle] = bin_corr_deltarf([sessions[0]],method='mean',areapairs=areapairs,
                                                              layerpairs=layerpairs,projpairs=projpairs,binresolution=binresolution)

#Init output arrays:
bin_2d_mean_oris        = np.empty((ndeltaoris,*np.shape(bin_2d_mean)))
bin_2d_posf_oris        = np.empty((ndeltaoris,*np.shape(bin_2d_mean)))
bin_2d_negf_oris        = np.empty((ndeltaoris,*np.shape(bin_2d_mean)))
bin_2d_count_oris       = np.empty((ndeltaoris,*np.shape(bin_2d_count)))

bin_dist_mean_oris      = np.empty((ndeltaoris,*np.shape(bin_dist_mean)))
bin_dist_posf_oris      = np.empty((ndeltaoris,*np.shape(bin_dist_mean)))
bin_dist_negf_oris      = np.empty((ndeltaoris,*np.shape(bin_dist_mean)))
bin_dist_count_oris     = np.empty((ndeltaoris,*np.shape(bin_dist_count)))

bin_angle_cent_mean_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_cent_mean)))
bin_angle_cent_posf_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_cent_mean)))
bin_angle_cent_negf_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_cent_mean)))
bin_angle_cent_count_oris     = np.empty((ndeltaoris,*np.shape(bin_angle_cent_count)))

bin_angle_surr_mean_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_surr_mean)))
bin_angle_surr_posf_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_surr_mean)))
bin_angle_surr_negf_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_surr_mean)))
bin_angle_surr_count_oris     = np.empty((ndeltaoris,*np.shape(bin_angle_surr_count)))

for idOri,deltaori in enumerate(deltaoris):
    [_,bin_2d_mean_oris[idOri,:,:,:,:,:],bin_2d_count_oris[idOri,:,:,:,:,:],
     bin_dist_mean_oris[idOri,:,:,:,:],bin_dist_count_oris[idOri,:,:,:],_,
     bin_angle_cent_mean_oris[idOri,:,:,:,:],bin_angle_cent_count_oris[idOri,:,:,:,:],
     bin_angle_surr_mean_oris[idOri,:,:,:,:],bin_angle_surr_count_oris[idOri,:,:,:,:],_] = bin_corr_deltarf(sessions,
                                                    method='mean',filtersign=None,areapairs=areapairs,layerpairs=layerpairs,
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
    [_,bin_2d_posf_oris[idOri,:,:,:,:,:],_,
     bin_dist_posf_oris[idOri,:,:,:,:],_,_,
     bin_angle_cent_posf_oris[idOri,:,:,:,:],_,
     bin_angle_surr_posf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='pos',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

    [_,bin_2d_negf_oris[idOri,:,:,:,:,:],_,
     bin_dist_negf_oris[idOri,:,:,:,:],_,_,
     bin_angle_cent_negf_oris[idOri,:,:,:,:],_,
     bin_angle_surr_negf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='neg',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#%% 
# corr_type           = 'noise_corr_GM'

#%% 
gaussian_sigma = 1

#%% Show spatial maps per delta ori for the mean correlation
min_counts=10
iap = 0
# iap = 1

fig = plot_2D_mean_corr_projs_dori(bin_2d_mean_oris,bin_2d_count_oris,bincenters_2d,deltaoris,
                                   areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,cmap='magma',
                                   centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma)
fig.savefig(os.path.join(savedir,'Projs','Collinear_DeltaRF_2D_projs_%s_%s_mean' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_projs_dori(bincenters_dist,bin_dist_count_oris,bin_dist_mean_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Projs','Collinear_Radial_Tuning_projs_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps per delta ori for the fraction positive 
fig = plot_2D_mean_corr_projs_dori(bin_2d_posf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_red)
fig.savefig(os.path.join(savedir,'Projs','Collinear_DeltaRF_2D_projs_%s_%s_posf' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_projs_dori(bincenters_dist,bin_dist_count_oris,bin_dist_posf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Projs','Collinear_Radial_Tuning_projs_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps per delta ori for the fraction negative 
fig = plot_2D_mean_corr_projs_dori(bin_2d_negf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_blue)
fig.savefig(os.path.join(savedir,'Projs','Collinear_DeltaRF_2D_projs_%s_%s_negf' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_projs_dori(bincenters_dist,bin_dist_count_oris,bin_dist_negf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Projs','Collinear_Radial_Tuning_projs_%s_negf' % (corr_type) + '.png'), format = 'png')

#%%
fig = plot_corr_center_tuning_projs_dori(bincenters_dist,bin_dist_count_oris,bin_dist_mean_oris,
                                         bin_dist_posf_oris,bin_dist_negf_oris,deltaoris,	
                                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Projs','Center_Tuning_projs_%s' % (corr_type) + '.png'), format = 'png')


#%% 






