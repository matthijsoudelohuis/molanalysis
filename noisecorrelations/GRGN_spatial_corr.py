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
from scipy.optimize import curve_fit

from loaddata.session_info import filter_sessions,load_sessions
from preprocessing.preprocesslib import assign_layer
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error,my_ceil,my_floor
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import compute_tuning, compute_prefori
from utils.explorefigs import plot_excerpt
from utils.shuffle_lib import my_shuffle, corr_shuffle
from utils.gain_lib import * 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])

session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['IM'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['RF'],filter_areas=['V1','PM'],session_rf=True)  

#%% Remove two sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)
                                # calciumversion='deconv',keepraw=True)
                                # calciumversion='dF',keepraw=True,filter_hp=0.01)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False,filtersig=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

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
# sessions = compute_pairwise_delta_rf(sessions,rf_type='Fsmooth')

#%% print number of pairs:
npairs = np.zeros(nSessions)
for ises,ses in enumerate(sessions):
    npairs[ises] = np.sum(~np.isnan(ses.trace_corr))/2
print('Number of pairs: %d (mean: %d, std : %d across n=%d sessions)' % 
            (np.sum(npairs),np.mean(npairs),np.std(npairs),nSessions))

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    if sessions[ises].sessiondata['protocol'].isin(['GR','GN'])[0]:
        sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='OSI')
        sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='gOSI')
        sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='tuning_var')
        sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'])


#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False,remove_method='RRR',remove_rank=2)

plt.imshow(sessions[0].noise_corr,vmin=-0.03,vmax=0.05)

#%% 
sessions = corr_shuffle(sessions,method='random')

#%% ##########################################################################################################
# DELTA ANATOMICAL DISTANCE :
# ##########################################################################################################

#%% Define the areapairs:
areapairs       = ['V1-V1','PM-PM']
clrs_areapairs  = get_clr_area_pairs(areapairs)

#%% Compute pairwise correlations as a function of pairwise anatomical distance ###################################################################
# for corr_type in ['trace_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type=corr_type)

    #Make the figure per protocol:
    fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type=corr_type)
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.pdf'), format = 'pdf')

# #%% Compute pairwise trace correlations as a function of pairwise anatomical distance ###################################################################
# [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='trace_corr')


# %% #######################################################################################################
# DELTA RECEPTIVE FIELD:
# ##########################################################################################################

#%%
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

#%% 
protocols = ['IM']
protocols = ['SP']
protocols           = ['GR']
# protocols           = ['GN','GR','IM']	

# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list    = np.where(np.logical_and(sessiondata['protocol'].isin(protocols),
#                                 ~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20'])))[0]
# sessions_subset     = [sessions[i] for i in sessions_in_list]

#%% ################ Pairwise correlations as a function of pairwise delta RF: #####################
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

#If you override any of these then these pairs will be ignored:
layerpairs          = ' '
# areapairs           = ' '
# projpairs           = ' '
absolute            = False
corr_type           = 'trace_corr'
rf_type             = 'Fsmooth'
binres              = 5
filtersign          = 'pos'
filternear          = True

[binmean,binedges]  =  bin_corr_deltarf(sessions_subset,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs,
                                        corr_type=corr_type,binres=binres,filternear=filternear,
                                        filtersign=filtersign,normalize=False,
                                        sig_thr = 0.001,rf_type=rf_type,mincount=10,absolute=absolute)

fig = plot_bin_corr_deltarf_flex(sessions_subset,binmean,binedges,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs,corr_type='trace_corr',normalize=False)

# fig.savefig(os.path.join(savedir,'Corr_1d_arealabel_%s_%s_abs%s' % (protocols[0],corr_type,absolute) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_GN_SP_RF_0.75dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_Areas_Layers_IM_0.5dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')

fig = plot_center_surround_corr(binmean,binedges,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs)


#%% ##########################################################################################################
#   CENTER VERSUS SURROUND 
# ##########################################################################################################



#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

#%% #########################################################################################
# Contrast: across areas, layers and projection pairs:
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
#If you override any of these with input to the deltarf bin function as ' ', then these pairs will be ignored

#%% Make the 2D, 1D and center surround averages for each protocol and areapair (not layerpair or projpair)
binres              = 5
rf_type             = 'Fsmooth'
filtersign          = None
filternear          = False
# corr_type           = 'trace_corr'
corr_type           = 'noise_corr'

# for ses in sessions:
#     if 'rf_az_F' in ses.celldata and 'rf_az_Fsmooth' in ses.celldata:
#         ses.celldata['rf_az_F'] = ses.celldata['rf_az_Fsmooth']
#         ses.celldata['rf_el_F'] = ses.celldata['rf_el_Fsmooth']
        
# for prot in ['GN','GR','IM','SP','RF']:
for prot in ['GN','GR']:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=' ',
                                corr_type=corr_type,binresolution=binres,rf_type=rf_type,
                                sig_thr = 0.001,filternear=filternear,filtersign=filtersign)
    
    filestring = '%s_%s_%s_azelFsmooth' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        fig = plot_2D_corr_map(binmean,bincounts,bincenters,min_counts = 50,
                                areapairs=areapairs,layerpairs=' ',projpairs=' ')
        fig.suptitle(filestring)
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')
    
        fig = plot_1D_corr_areas(binmean,bincounts,bincenters,min_counts = 50,
                                areapairs=areapairs,layerpairs=' ',projpairs=' ')
        fig.suptitle(filestring)
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.pdf'), format = 'pdf')

        fig = plot_center_surround_corr_areas(binmean,bincenters,centerthr=15,layerpairs=' ',areapairs=areapairs,projpairs=' ')
        fig.suptitle(filestring)
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.pdf'), format = 'pdf')


    # if np.any(bincounts):
    #     figs = []
    #     axs = []

    #     fig = plot_2D_corr_map(binmean,bincounts,bincenters,min_counts = 50,
    #                             areapairs=areapairs,layerpairs=' ',projpairs=' ')
    #     fig.suptitle(filestring)
    #     figs.append(fig)
    #     axs.extend(fig.axes)
        
    #     fig = plot_1D_corr_areas(binmean,bincounts,bincenters,min_counts = 50,
    #                             areapairs=areapairs,layerpairs=' ',projpairs=' ')
    #     fig.suptitle(filestring)
    #     figs.append(fig)
    #     axs.extend(fig.axes)
        
    #     fig = plot_center_surround_corr_areas(binmean,bincenters,centerthr=15,layerpairs=' ',areapairs=areapairs,projpairs=' ')
    #     fig.suptitle(filestring)
    #     figs.append(fig)
    #     axs.extend(fig.axes)
        
    #     fig = plt.figure(figsize=(15,15))
    #     for i,ax in enumerate(axs):
    #         ax.set_position([0.1 + i*0.3,0.1,0.3,0.8])
    #         fig.add_axes(ax)
    #     fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_combined_%s' % filestring + '.png'), format = 'png')
    #     fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_combined_%s' % filestring + '.pdf'), format = 'pdf')

#%%
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False,remove_method='PCA',remove_rank=1)

#%% Make the 2D, 1D and center surround averages for each protocol and areapair and projpair (not layerpair)
binres              = 5
# rf_type             = 'Fneugauss'
rf_type             = 'Fsmooth'
filtersign          = None
filternear          = True
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

# for prot in ['GN','GR','IM','SP','RF']:
for prot in [['GN','GR']]:
# for prot in ['RF']:
# for prot in ['IM']:
# for prot in ['GR']:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
                                corr_type=corr_type,binresolution=binres,rf_type=rf_type,normalize=False,
                                sig_thr = 0.001,filternear=filternear)
    
    # filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
    filestring = '%s_%s_%s_proj' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        fig = plot_2D_corr_map(binmean,bincounts,bincenters,min_counts = 1,gaussian_sigma=1,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')
    
        fig = plot_1D_corr_areas_projs(binmean,bincounts,bincenters,min_counts = 1,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.pdf'), format = 'pdf')

        fig = plot_center_surround_corr_areas_projs(binmean,bincenters,centerthr=15,layerpairs=' ',areapairs=areapairs,projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.pdf'), format = 'pdf')

#%% Make a 2D histogram with the distribution of correlation values for each delta RF bin
binres_rf           = 2
binres_corr         = 0.1
rf_type             = 'Fsmooth'
filternear          = True
# corr_type           = 'noise_corr'
corr_type           = 'trace_corr'

for prot in [['GN','GR']]:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [bincounts,bincenters_rf,bincenters_corr] = bin_2d_rangecorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=['L2/3-L2/3'],projpairs=projpairs,
                                corr_type=corr_type,binres_rf=binres_rf,binres_corr=binres_corr,rf_type=rf_type,
                                sig_thr = 0.001,filternear=filternear,noise_thr=0.2)

    # [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
    #                             corr_type=corr_type,binresolution=binres,rf_type=rf_type,normalize=False,
    #                             sig_thr = 0.001,filternear=filternear,filtersign=filtersign)
    
    # filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
    filestring = '%s_%s_%s_proj' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        
        # fig = plot_1D_fraccorr(bincounts,bincenters_rf,bincenters_corr,gaussian_sigma=2,
                                # areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        # fig.suptitle(filestring)
        # plt.tight_layout()
        
        fig = plot_2D_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,gaussian_sigma=2,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_rangecorr_norm034_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaXY_2D_%s' % filestring + '.png'), format = 'png')

        fig = plot_perc_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_Perc_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')

#%% 
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% Make a 2D histogram with the distribution of correlation values for each delta RF bin
binres_rf           = 5
rf_type             = 'Fsmooth'
filternear          = False
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

for prot in [['GN','GR']]:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]
    filestring          = '%s_%s_proj' % (corr_type,rf_type)

    [bincounts,binpos,binneg,bincenters_rf] = bin_1d_fraccorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
    # [bincounts,binpos,binneg,bincenters_rf] = bin_1d_fraccorr_deltarf(sessions_subset,areapairs=areapairs1,layerpairs=['L2/3-L2/3'],projpairs=projpairs,
                                corr_type=corr_type,binres_rf=binres_rf,rf_type=rf_type,
                                sig_thr = 0.001,filternear=filternear,noise_thr=0.2,corr_thr=0.05)

    fig = plot_1D_fraccorr(bincounts,binpos,binneg,bincenters_rf,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,mincounts=50)
    # fig.savefig(os.path.join(savedir,'deltaRF','1DRF','Frac_PosNeg_DeltaRF_GNGR_%s' % filestring + '.png'), format = 'png')

#%% Make a 2D histogram with the distribution of correlation values for each delta XY bin
binres_rf           = 25
binres_corr         = 0.1
rf_type             = 'Fsmooth'
filternear          = False
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

# for prot in ['GN','GR','IM','SP','RF']:
for prot in [['GN','GR']]:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [bincounts,bincenters_rf,bincenters_corr] = bin_2d_rangecorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
                                corr_type=corr_type,binres_rf=binres_rf,binres_corr=binres_corr,rf_type=rf_type,
                                sig_thr = 1,filternear=filternear,noise_thr=0.2)
    
    # [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
    #                             corr_type=corr_type,binresolution=binres,rf_type=rf_type,normalize=False,
    #                             sig_thr = 0.001,filternear=filternear,filtersign=filtersign)
    
    filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
    # filestring = '%s_%s_%s_proj' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        fig = plot_2D_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,gaussian_sigma=1,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaXY_2D_%s' % filestring + '.png'), format = 'png')

        fig = plot_perc_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_Perc_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')

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
