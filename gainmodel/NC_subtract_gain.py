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

from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
# session_list        = np.array([['LPE10919','2023_11_06'],
#                                 ['LPE10919','2023_11_06'],
#                                 ['LPE10919','2023_11_06']]) #GR

# session_list        = np.array([['LPE09830','2023_04_10'],
#                                 ['LPE09830','2023_04_10'],
                                # ['LPE09830','2023_04_10']]) #GR
# session_list        = np.array([['LPE10919','2023_11_06']])

# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

# #%%  Load data properly
# ises = 0 # without filtering
# sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
#                             calciumversion='dF',keepraw=False)
# ises = 1 # with filtering
# filter_hp = 0.01
# sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
#                             calciumversion='dF',keepraw=True,filter_hp=filter_hp)

# ises = 2 # with filtering
# filter_hp = 0.001
# sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
#                             calciumversion='dF',keepraw=True,filter_hp=filter_hp)

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% ##################### Compute pairwise neuronal distances: ##############################
# sessions = compute_pairwise_metrics(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

#%% 
for ses in sessions:
    if 'rf_r2_Fgauss' in ses.celldata:
        ses.celldata['rf_p_Fgauss'] = ses.celldata['rf_r2_Fgauss']<0.2
        ses.celldata['rf_p_Fneugauss'] = ses.celldata['rf_r2_Fneugauss']<0.2

#%% Copy Fgauss to F
for ses in sessions:
    if 'rf_az_Fgauss' in ses.celldata:
        ses.celldata['rf_az_F'] = ses.celldata['rf_az_Fgauss']
        ses.celldata['rf_el_F'] = ses.celldata['rf_el_Fgauss']
        ses.celldata['rf_p_F'] = ses.celldata['rf_p_Fgauss']

#%% ##################### Compute pairwise receptive field distances: ##############################
sessions = smooth_rf(sessions,radius=50,rf_type='Fneugauss',mincellsFneu=5)
sessions = exclude_outlier_rf(sessions) 
sessions = replace_smooth_with_Fsig(sessions) 

#%% 
areapairs = ['V1-V1','PM-PM']
corr_type = 'noise_corr'

#%% 

sessions_orig       = copy.deepcopy(sessions)
sessions_nogain     = copy.deepcopy(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
sessions_orig       = compute_signal_noise_correlation(sessions_orig,filter_stationary=False)
# sessions_nogain     = compute_signal_noise_correlation(sessions_nogain,filter_stationary=False,remove_method='PCA',remove_rank=2)
sessions_nogain     = compute_signal_noise_correlation(sessions,filter_stationary=False,remove_method='GM')

[binmean_orig,binedges]     = bin_corr_distance(sessions_orig,areapairs=areapairs,corr_type=corr_type)
[binmean_nogain,binedges]   = bin_corr_distance(sessions_nogain,areapairs=areapairs,corr_type=corr_type)

#%% 
clrs_areapairs = get_clr_area_pairs(areapairs)
clrs_conditions = ['k','b']

fig,axes = plt.subplots(1,len(areapairs),figsize=(4*len(areapairs),4))
handles = []
for iap,areapair in enumerate(areapairs):
    ax = axes[iap] 
    for ises in range(np.shape(binmean_orig)[0]):
        ax.plot(binedges[:-1],binmean_orig[ises,iap,:].squeeze(),linewidth=0.25,color=clrs_conditions[0])
        ax.plot(binedges[:-1],binmean_nogain[ises,iap,:].squeeze(),linewidth=0.25,color=clrs_conditions[1])
    handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean_orig[:,iap,:].squeeze(),error='sem',color=clrs_conditions[0]))
    handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean_nogain[:,iap,:].squeeze(),error='sem',color=clrs_conditions[1]))
    # plt.savefig(os.path.join(savedir,'NoiseCorr_distRF_RegressOut_' + areapair + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    ax.legend(handles,['Original','Gain Removed'],loc='upper right',frameon=False)	
    ax.set_xlabel('Anatomical distance ($\mu$m)')
    ax.set_ylabel('Correlation')
    ax.set_xlim([15,600])
    ax.set_title('%s' % areapair,color=clrs_areapairs[iap])
    # ax.set_ylim([-0.01,0.04])
    ax.set_ylim([-0.01,0.09])
    ax.set_aspect('auto')
    ax.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()



