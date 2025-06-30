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
from sklearn.decomposition import FactorAnalysis as FA

from loaddata.session_info import filter_sessions,load_sessions
from preprocessing.preprocesslib import assign_layer
from utils.plot_lib import * #get all the fixed color schemes
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

#%% 
areas = ['V1','PM']
n_components = 20
fa = FA(n_components=n_components)

# comps = np.array([0,1,2,3,4,5,6,7,8,9])
# comps = np.array([1,2,3,4,5,6,7,8])
comps = np.arange(1,n_components)
# comps = np.array(0,)
# comps = np.arange(2,n_components)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Computing noise correlations'):
    
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




#%% 
#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM']
# areapairs           = ['V1-PM']
# areapairs           = ['PM-PM']
# layerpairs          = ['L2/3-L2/3']
# layerpairs          = ['L2/3-L5']
# layerpairs          = ['L5-L5']
layerpairs          = ' '
projpairs           = ' '
projpairs           = ['unl-unl', 'unl-lab', 'lab-lab']

[bincenters_2d,bin_2d,bin_2d_count,bin_dist,bin_dist_count,binsdRF,bin_angle_cent,bin_angle_cent_count,bin_angle_surr,bin_angle_surr_count,bincenters_angle] = \
    bin_corr_deltaxy(sessions,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                 corr_type='noise_cov',
                 noise_thr=20,onlysameplane=False,binresolution=20,
                 normalize=False,filtersign=None,corr_thr=0.05,shufflefield=None)

fig = plot_bin_corr_distance_projs(binsdRF,bin_dist,areapairs,layerpairs,projpairs)


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
filestring = 'RadialTuning_areas_%s_' % (corr_type)
# filestring = 'RadialTuning_areas_%sGM_%s_' % (corr_type,layerpairs[0].replace('/',''))

fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,
                                             areapairs,layerpairs,projpairs,min_counts=100)

# fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses[idx_ses],bin_dist_mean_ses[idx_ses],areapairs,layerpairs,projpairs)

fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean_sessions' + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean_sessions' + '.pdf'), format = 'pdf')

#%% 
filestring = 'RadialTuning_ExponentialFits_areas_%s_' % (corr_type)
fig = plot_corr_radial_tuning_areas(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean' + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'RadialTuning',filestring + 'mean' + '.pdf'), format = 'pdf')

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

shufflefield        = 'corrdata'
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
min_counts          = 50
# fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses[idx_ses],bin_dist_mean_ses[idx_ses],areapairs,layerpairs,projpairs,datatype='Correlation')

# fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses[idx_ses],bin_dist_mean_ses[idx_ses],areapairs,layerpairs,projpairs)

fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,areapairs,
                                    layerpairs,projpairs,datatype='Correlation',min_counts=min_counts)
# fig = plot_corr_radial_tuning_projs(binsdRF,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs,datatype='Correlation')

# fig.savefig(os.path.join(savedir,'RadialTuning','RadialTuning_projs_%s_mean' % (corr_type) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RadialTuning','RadialTuning_projs_%s_mean_shuf' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'RadialTuning_projs_%s_mean_GRGN_shuf' % (corr_type) + '.pdf'), format = 'pdf')


#%% Regress pairwise metrics onto different dimensions of covariance matrix:
# The idea is to see which dimensions of the covariance are 
# The prediction is that the first dimension is reflected shared population rate that is not retinotopically specific
# However, higher dimensions do contain retinotopically specific covariability because it is reflected in the total
# covariance.

areapairs           = ['V1-V1']

areapairs           = ['V1-V1','PM-PM','V1-PM']
# areapairs           = ['V1-PM']
projpairs           = ' '
# projpairs           = ['unl-unl']
layerpairs          = ' '
n_components        = 20

bins_RF,spatial_cov_rf,bins_XYZ,spatial_cov_xyz = regress_cov_dim(sessions,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                    # corr_type='noise_cov',rf_type='F',
                    corr_type='noise_corr',rf_type='F',
                    r2_thr=0.1,noise_thr=20,n_components=n_components)

#%% 
nareapairs = len(areapairs)
clrs_areapairs  = get_clr_area_pairs(areapairs)
bindims = np.array([[0],[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19]],dtype=object)
bindimlabels = np.array(['Dim1','Dim2-5','Dim6-10','Dim11-15','Dim16-20'],dtype=object)

#%% 
ilp = 0
ipp = 0
clrs = sns.color_palette('Spectral',len(bindims))
fig,axes = plt.subplots(1,nareapairs,figsize=(9,3),sharex=True,sharey=True)
for iap in range(nareapairs):
    ax = axes[iap]
    # ax.plot(range(n_components),np.nanmean(R2_RF_COV,axis=(0,2,3,4)))
    for ibdim,bdim in enumerate(bindims):
        data = spatial_cov_rf[np.ix_(range(nSessions),bdim,range(len(bins_RF)),[iap],[ilp],[ipp])]
        
        ax.plot(bins_RF,np.nanmean(data,axis=(0,1,3,4,5)),
                color=clrs[ibdim],label=bindimlabels[ibdim])
    
        # for ises in range(nSessions):
        # ax.plot(range(n_components),spatial_cov_rf[ises,:,iap,ilp,ipp],axis=(1,2,3)),alpha=0.5)
    ax.axhline(0,ls=':',color='k',alpha=0.5)
    ax.set_title('%s' % (areapairs[iap]),c=clrs_areapairs[iap])
    ax.set_xlabel('Delta RF (deg)')
    if iap == 0:
        ax.set_ylabel('Covariance')
    if iap==nareapairs-1:
        ax.legend(bindimlabels,frameon=False,title='Components',fontsize=7,bbox_to_anchor=(1,0.9))
ax.set_xlim([0,45])
ax.set_ylim([-0.1,0.15])
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()


# #%% 
# ilp = 0
# ipp = 0
# n_components = 15
# clrs = sns.color_palette('Spectral',n_components)
# fig,axes = plt.subplots(1,nareapairs,figsize=(9,3),sharex=True,sharey=True)
# for iap in range(nareapairs):
#     ax = axes[iap]
#     # ax.plot(range(n_components),np.nanmean(R2_RF_COV,axis=(0,2,3,4)))
#     for icomp in range(n_components):
#         ax.plot(bins_RF,np.nanmean(spatial_cov_rf[:,icomp,:,iap,ilp,ipp],axis=(0)),
#                 color=clrs[icomp])
    
#         # for ises in range(nSessions):
#         # ax.plot(range(n_components),spatial_cov_rf[ises,:,iap,ilp,ipp],axis=(1,2,3)),alpha=0.5)
#     ax.axhline(0,ls=':',color='k',alpha=0.5)

#     ax.set_xlabel('Delta RF (deg)')
#     if iap == 0:
#         ax.set_ylabel('Covariance')
#     if iap==nareapairs-1:
#         ax.legend(np.arange(n_components)+1,frameon=False,title='Components',fontsize=7,bbox_to_anchor=(1,0.9))
# sns.despine(top=True,right=True,offset=3)
# plt.tight_layout()

#%% 
ilp = 0
ipp = 0
n_components = 20
clrs = sns.color_palette('viridis',n_components)
fig,axes = plt.subplots(nSessions,nareapairs,figsize=(nareapairs*2,nSessions*2),sharex=True,sharey=True)
for iap in range(nareapairs):
    for ises in range(nSessions):
        ax = axes[ises,iap]
    # ax.plot(range(n_components),np.nanmean(R2_RF_COV,axis=(0,2,3,4)))
        for icomp in range(n_components):
            ax.plot(bins_RF,spatial_cov_rf[ises,icomp,:,iap,ilp,ipp],
                    color=clrs[icomp])
    
        # for ises in range(nSessions):
        # ax.plot(range(n_components),spatial_cov_rf[ises,:,iap,ilp,ipp],axis=(1,2,3)),alpha=0.5)

        if iap == 1 and ises==nSessions-1:
            ax.set_xlabel('Delta RF (deg)')
        if iap == 0 and ises==0:
            ax.set_ylabel('Covariance')
        if iap==nareapairs-1 and ises==nSessions//2:
            ax.legend(np.arange(n_components)+1,frameon=False,title='Components',fontsize=7,bbox_to_anchor=(1,0.9))
        ax.axhline(0,ls=':',color='k',alpha=0.5)
ax.set_xlim([0,45])
ax.set_ylim([-0.1,0.15])
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()


#%% Spatial XYZ:


#%% 
ilp = 0
ipp = 0
clrs = sns.color_palette('Spectral',len(bindims))
fig,axes = plt.subplots(1,nareapairs,figsize=(9,3),sharex=True,sharey=True)
for iap in range(nareapairs):
    ax = axes[iap]
    # ax.plot(range(n_components),np.nanmean(R2_RF_COV,axis=(0,2,3,4)))
    for ibdim,bdim in enumerate(bindims):
        data = spatial_cov_xyz[np.ix_(range(nSessions),bdim,range(len(bins_XYZ)),[iap],[ilp],[ipp])]
        
        ax.plot(bins_XYZ,np.nanmean(data,axis=(0,1,3,4,5)),
                color=clrs[ibdim],label=bindimlabels[ibdim])
    
        # for ises in range(nSessions):
        # ax.plot(range(n_components),spatial_cov_rf[ises,:,iap,ilp,ipp],axis=(1,2,3)),alpha=0.5)
    ax.axhline(0,ls=':',color='k',alpha=0.5)
    ax.set_title('%s' % (areapairs[iap]),c=clrs_areapairs[iap])
    ax.set_xlabel('Delta RF (deg)')
    if iap == 0:
        ax.set_ylabel('Covariance')
    if iap==nareapairs-1:
        ax.legend(bindimlabels,frameon=False,title='Components',fontsize=7,bbox_to_anchor=(1,0.9))
ax.set_xlim([0,750])
ax.set_ylim([-0.1,0.15])
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()

#%% 
ilp = 0
ipp = 0
n_components = 20
clrs = sns.color_palette('Spectral',n_components)
fig,axes = plt.subplots(1,nareapairs,figsize=(9,3),sharex=True,sharey=True)
for iap in range(nareapairs):
    ax = axes[iap]
    # ax.plot(range(n_components),np.nanmean(R2_RF_COV,axis=(0,2,3,4)))
    for icomp in range(n_components):
        ax.plot(bins_XYZ,np.nanmean(spatial_cov_xyz[:,icomp,:,iap,ilp,ipp],axis=(0)),
                color=clrs[icomp])
    
        # for ises in range(nSessions):
        # ax.plot(range(n_components),spatial_cov_rf[ises,:,iap,ilp,ipp],axis=(1,2,3)),alpha=0.5)

    ax.set_xlabel('Delta XYZ (um)')
    if iap == 0:
        ax.set_ylabel('Covariance')
    if iap==nareapairs-1:
        ax.legend(np.arange(n_components)+1,frameon=False,title='Components',
                  fontsize=7,bbox_to_anchor=(1,0.9),ncol=2)
    ax.axhline(0,ls=':',color='k',alpha=0.5)
ax.set_xlim([0,750])
ax.set_ylim([-0.1,0.15])

sns.despine(top=True,right=True,offset=3)
plt.tight_layout()

