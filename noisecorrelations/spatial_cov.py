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
from utils.plot_lib import * #get all the fixed color schemes
from utils.plot_lib import shaded_error,my_ceil,my_floor
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import compute_tuning_wrapper
from utils.explorefigs import plot_excerpt
from utils.shuffle_lib import my_shuffle, corr_shuffle
from utils.gain_lib import * 
from utils.arrayop_lib import nanweightedaverage
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')


#%% 

# Perhaps follow this approach: 

# Huang et al. 2019:
# computing the covariance matrix, subtract the first mode perhaps, 
# then bin the covariance as a function of the pairwise distance between neurons.


#%% #############################################################################
session_list        = np.array([['LPE10919_2023_11_06']])

session_list        = np.array([['LPE09665_2023_03_21'], #GR
                                ['LPE10919_2023_11_06']]) #GR


# Sessions with good RF:
session_list        = np.array([['LPE09665_2023_03_21'], #GR
                                ['LPE10884_2023_10_20'], #GR
                                ['LPE11998_2024_05_02'], #GN
                                ['LPE12013_2024_05_02'], #GN
                                ['LPE12013_2024_05_07'], #GN
                                ['LPE11086_2023_12_15'], #GR
                                ['LPE10919_2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],only_session_id=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%% Load all sessions from certain protocols: 
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=['V1','PM'],session_rf=True)  

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
# sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE10885_2023_10_19','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
calciumversion = 'dF'
# calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)
                                # calciumversion=calciumversion,keepraw=True,filter_hp=0.01)

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

# plt.imshow(sessions[0].noise_corr,vmin=-0.03,vmax=0.05)


#%% 

from sklearn.decomposition import FactorAnalysis as FA

#%% 
areas = ['V1','PM']
n_components = 20
fa = FA(n_components=n_components)

# comps = np.array([0,1,2,3,4,5,6,7,8,9])
# comps = np.array([1,2,3,4,5,6,7,8])
comps = np.arange(1,n_components)
# comps = np.array([0])
# comps = np.arange(2,n_components)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Computing noise correlations'):
    # ses. 
    [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
    if sessions[ises].sessiondata['protocol'][0]=='GR':
        resp_meanori,respmat_res        = mean_resp_gr(sessions[ises])
    elif sessions[ises].sessiondata['protocol'][0]=='GN':
        resp_meanori,respmat_res        = mean_resp_gn(sessions[ises])

    # # Compute noise correlations from residuals:
    # data = zscore(respmat_res,axis=1)
    # sessions[ises].noise_corr       = np.corrcoef(data)
    # fa.fit(data.T)
    # data_T              = fa.transform(data.T)
    # data_hat            = np.dot(data_T[:,comps], fa.components_[comps,:]).T        # Reconstruct data
    # sessions[ises].noise_cov    = np.cov(data_hat)

    stims        = np.sort(sessions[ises].trialdata['stimCond'].unique())
    trial_stim   = sessions[ises].trialdata['stimCond']
    noise_corr  = np.empty((N,N,len(stims)))  
    noise_cov   = np.empty((N,N,len(stims)))  
    for i,stim in enumerate(stims):
        data                = zscore(respmat_res[:,trial_stim==stim],axis=1)

        noise_corr[:,:,i]   = np.corrcoef(data)

        for iarea,area in enumerate(areas):
            idx_N               = ses.celldata['roi_name']==area

            fa.fit(data[idx_N,:].T)
            data_T              = fa.transform(data[idx_N,:].T)
            data[idx_N,:]       = np.dot(data_T[:,comps], fa.components_[comps,:]).T        # Reconstruct data
        
        noise_cov[:,:,i]  = np.cov(data)

    sessions[ises].noise_corr       = np.mean(noise_corr,axis=2)
    sessions[ises].noise_cov        = np.mean(noise_cov,axis=2)


#%% #####################################################################################################
# DELTA ANATOMICAL DISTANCE :
# #######################################################################################################

#%% Define the areapairs:
areapairs       = ['V1-V1','PM-PM']
clrs_areapairs  = get_clr_area_pairs(areapairs)

#%% Compute pairwise correlations as a function of pairwise anatomical distance ###################################################################
# for corr_type in ['sig_corr','noise_corr','noise_cov']:
for corr_type in ['noise_corr','noise_cov']:
# for corr_type in ['noise_corr']:
    # [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type=corr_type)
    [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type=corr_type,
                                           absolute=False)

    #Make the figure per protocol:
    fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type=corr_type)
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.pdf'), format = 'pdf')

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

#%% 

#%% Reassign layers:
for ses in sessions:
    ses.celldata = assign_layer2(ses.celldata)



#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)


#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM','V1-PM']
# areapairs           = ['V1-PM']
# areapairs           = ['PM-PM']
# layerpairs          = ['L2/3-L2/3']
# layerpairs          = ['L2/3-L5']
# layerpairs          = ['L5-L5']
layerpairs          = ' '
projpairs           = ' '
projpairs           = ['unl-unl']

# corr_thr            = 0.025 #thr in percentile of total corr for significant pos or neg
# corr_thr            = 0.01 #thr in percentile of total corr for significant pos or neg
# rf_type             = 'Fneu'
rf_type             = 'F'
# rf_type             = 'Fsmooth'
# corr_type           = 'noise_corr'
corr_type           = 'noise_cov'
# corr_type           = 'trace_corr'
# corr_type           = 'sig_corr'
binresolution       = 10
normalize           = False
# normalize = True
absolute            = False
# absolute            = True
shufflefield        = None

[bins2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,binsdRF,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,binsangle] = bin_corr_deltarf_ses(sessions,rf_type=rf_type,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,noise_thr=100,
                        binresolution=binresolution,normalize=normalize,absolute=absolute,
                        shufflefield=shufflefield,r2_thr=0.1)

# [_,bin_2d_posf_ses,_,bin_dist_posf_ses,_,_,
#  bin_angle_cent_posf_ses,_,bin_angle_surr_posf_ses,_,_] = bin_corr_deltarf_ses(sessions,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,
#                         method='frac',filtersign='pos',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr,
#                         binresolution=binresolution)

# [_,bin_2d_negf_ses,_,bin_dist_negf_ses,_,_,bin_angle_cent_negf_ses,_,bin_angle_surr_negf_ses,_,_] = bin_corr_deltarf_ses(sessions,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,rf_type=rf_type,
#                         method='frac',filtersign='neg',corr_type=corr_type,noise_thr=20,corr_thr=corr_thr,
#                         binresolution=binresolution)

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
rf_type = 'F'
nrfcellsPM = np.empty((len(sessions),1))
for ises,ses in enumerate(sessions):
    nrfcellsPM = np.sum(np.all((ses.celldata['rf_r2_' + rf_type]>0.2,
                        ses.celldata['noise_level']<20,
                        ses.celldata['roi_name'] == 'PM'),axis=0))
    print('%s: %d' % (ses.session_id,nrfcellsPM))
    
#%%
session_list        = np.array([
                                ['LPE11086_2024_01_05'], #GR
                                ['LPE09665_2023_03_21'], #GR
                                ['LPE09830_2023_04_10'], #GR
                                ['LPE10884_2023_10_20'], #GR
                                ['LPE11998_2024_05_02'], #GN
                                ['LPE12013_2024_05_02'], #GN
                                ['LPE12013_2024_05_07'], #GN
                                ['LPE12223_2024_06_08'], #GN
                                ['LPE11086_2023_12_15'], #GR
                                ['LPE10885_2023_10_12'], #GR
                                ['LPE10885_2023_10_19'], #GR
                                ['LPE10919_2023_11_06']]) #GR

                                # ['LPE11086_2024_01_10'], #GR weird inverse
                                # ['LPE11086_2024_01_08'], #GR weird inverse

sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

idx_ses             = np.isin(sessiondata['session_id'],session_list[:,0])


#%% Plot radial tuning:
filestring = 'RadialTuning_areas_%s_%s_' % (corr_type,layerpairs[0].replace('/',''))
# filestring = 'RadialTuning_areas_%sGM_%s_' % (corr_type,layerpairs[0].replace('/',''))

# fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs)

fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses[idx_ses],bin_dist_mean_ses[idx_ses],areapairs,layerpairs,projpairs)

# fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean_sessions' + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean_sessions' + '.pdf'), format = 'pdf')

#%% 
fig = plot_corr_radial_tuning_areas(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs)
# fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean' + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean' + '.pdf'), format = 'pdf')

#%% 1D delta RF weighted mean across sessions:
bin_dist_mean = nanweightedaverage(bin_dist_mean_ses,weights=bin_dist_count_ses,axis=0)
# bin_dist_posf = nanweightedaverage(bin_dist_posf_ses,weights=bin_dist_count_ses,axis=0)
# bin_dist_negf = nanweightedaverage(bin_dist_negf_ses,weights=bin_dist_count_ses,axis=0)
bin_dist_count = np.nansum(bin_dist_count_ses,axis=0)

fig = plot_corr_radial_tuning_areas_mean(binsdRF,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs)

#%%


#%% 

#Colors:
clrs_areapairs      = get_clr_area_pairs(areapairs) 
if len(areapairs)==1:
    clrs_areapairs =[clrs_areapairs]


bin_dist_data_ses = bin_dist_mean_ses 
if np.max(binsdRF)>100:
    xylim               = 250
    dim12label = 'XY (um)'
else:
    # xylim               = 65
    xylim               = 70
    dim12label = 'RF (\N{DEGREE SIGN})'

min_counts      = 50

#Compute data mean and error:
bin_dist_data_ses[bin_dist_count_ses<min_counts] = np.nan
data_mean   = np.nanmean(bin_dist_data_ses,axis=0)
data_error  = np.nanstd(bin_dist_data_ses,axis=0) / np.sqrt(np.shape(bin_dist_data_ses)[0])

fig,axes    = plt.subplots(nSessions,len(areapairs),figsize=(2*len(areapairs),nSessions*2),
                           sharex=True,sharey=True)
for ises,ses in enumerate(sessions):
    ilp = 0
    ipp = 0
    handles = []
    for iap,areapair in enumerate(areapairs):
        ax = axes[ises,iap]
        # bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
        ax.set_title(ses.session_id,fontsize=10)

        ax.plot(binsdRF,bin_dist_data_ses[ises,:,iap,ilp,ipp],color=clrs_areapairs[iap],
                alpha=0.5,linewidth=2)
     
    ax.set_xlim([0,65])
    ax.set_ylim([0,0.04])
    ax.set_xlabel(u'Δ %s' % 'RF (\N{DEGREE SIGN})') 
sns.despine(fig,top=True,right=True,offset=3)
plt.tight_layout()



#%% #########################################################################################
# Contrast: across projections:
areapairs           = ['V1-V1','PM-PM','V1-PM']
# areapairs           = ['V1-PM']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
# projpairs           = ['unl-unl']
layerpairs          = ' '
# layerpairs          = ['L2/3-L2/3']
# layerpairs          = ['L2/3-L5']

rf_type             = 'F'
# rf_type             = 'Fsmooth'
corr_type           = 'noise_cov'
# corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

binresolution       = 10
# normalize           = True
normalize           = False
# absolute            = True
absolute            = False
shufflefield        = None

# shufflefield        = 'corrdata'
# shufflefield      = 'labeled'
# shufflefield      = 'RF'

[bins2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,binsdRF,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,binsangle] = bin_corr_deltarf_ses(sessions,rf_type=rf_type,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,noise_thr=100,
                        binresolution=binresolution,normalize=normalize,absolute=absolute,
                        shufflefield=shufflefield,r2_thr=0.1)

#%% Plot radial tuning:
min_counts          = 25
fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses[idx_ses],bin_dist_mean_ses[idx_ses],areapairs,layerpairs,projpairs,datatype='Correlation')

# fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses[idx_ses],bin_dist_mean_ses[idx_ses],areapairs,layerpairs,projpairs)

# fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs,datatype='Correlation')
# fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs,datatype='Correlation')
# axes = fig.get_axes()
# axes[0].set_ylim([-0.05,0.05])

# fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN_shuf' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN_shuf' % (corr_type) + '.pdf'), format = 'pdf')


#%% 



for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):
    celldata = copy.deepcopy(sessions[ises].celldata)
    if hasattr(sessions[ises],corr_type):
        corrdata = getattr(sessions[ises],corr_type).copy()

        plt.imshow(corrdata,vmin=-0.1,vmax=0.1)
        if shufflefield == 'RF':
            celldata['rf_el_' + rf_type],celldata['rf_az_' + rf_type] = my_shuffle_celldata_joint(celldata['rf_el_' + rf_type],
                                                            celldata['rf_az_' + rf_type],celldata['roi_name'])
        elif shufflefield == 'XY':
            celldata['xloc'],celldata['yloc'] = my_shuffle_celldata_joint(celldata['xloc'],celldata['yloc'],
                                                            celldata['roi_name'])
        elif shufflefield == 'corrdata':
            corrdata = my_shuffle(corrdata,method='random',axis=None)
        elif shufflefield is not None:
            celldata = my_shuffle_celldata(celldata,shufflefield,keep_roi_name=True)
        # plt.imshow(corrdata)
        plt.imshow(corrdata,vmin=-0.1,vmax=0.1)

fig,axes = plt.subplots(1,2)
corrdata = getattr(sessions[ises],corr_type).copy()
axes[0].imshow(corrdata,vmin=-0.1,vmax=0.1)
corrdata = my_shuffle(corrdata,method='random',axis=None)
axes[1].imshow(corrdata,vmin=-0.1,vmax=0.1)

# #%% #########################################################################################
# # Contrast: across areas
# areapairs           = ['V1-V1','PM-PM','V1-PM']
# # layerpairs          = ['L2/3-L2/3']
# # layerpairs          = ['L5-L5']
# layerpairs          = ' '
# projpairs           = ' '
# binresolution = 5
# rf_type             = 'Fsmooth'
# corr_type           = 'noise_corr'

# [bin_dist_mean_ses,bin_dist_count_ses,binsdRF] = bin_corr_deltarf_ses_vkeep(sessions,rf_type=rf_type,
#                         areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
#                         method='mean',filtersign=None,corr_type=corr_type,noise_thr=20,
#                         binresolution=binresolution,normalize=False)

# binres              = 0.005
# binedges            = np.arange(-1,1,binres)
# bincenters          = binedges[:-1] + binres/2
# nbins               = len(bincenters)

# hist_mean_ses = np.full(list(np.shape(bin_dist_mean_ses))[:-1] + [nbins],np.nan)

# for idx in np.ndindex(*bin_dist_mean_ses.shape[:-1]):
#     hist_mean_ses[idx],_ = np.histogram(bin_dist_mean_ses[idx],bins=binedges)

# hist_mean_ses /= np.sum(hist_mean_ses,axis=-1,keepdims=True)

# # plt.plot(bincenters,np.nanmean(hist_mean_ses,axis=(0,1,2,3,4)))
# plt.plot(bincenters,np.nanmean(hist_mean_ses[:,:2,:,:,:,:],axis=(0,1,2,3)).T)
# plt.plot(bincenters,np.nanmean(hist_mean_ses[:,:3,0,:,:,:],axis=(0,1,2)).T)
# plt.xlim([-0.1,0.1])
