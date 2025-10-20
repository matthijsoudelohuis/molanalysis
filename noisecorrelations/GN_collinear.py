# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""


# TODO: 
# Make scripts load either GR or SP
# Separate one for GN with speed extent
# Make within and across areas separate code blocks
# Change all to 15 degrees radius for the center
# do not take (weighted) mean across sessions outside plotting script, but within
# Run on behaviorally predicted or unrelated activity
# Run with and without locomotion

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
from utils.plot_lib import shaded_error
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import compute_tuning_wrapper,ori_remapping
from preprocessing.preprocesslib import assign_layer,assign_layer2

protocol = 'GR'
protocol = 'SP'
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\Collinear\\%s\\' % protocol)

colors = [(0, 0, 0), (1, 0, 0), (1, 1, 1)] # first color is black, last is red
cm_red = LinearSegmentedColormap.from_list("Custom", colors, N=20)
colors = [(0, 0, 0), (0, 0, 1), (1, 1, 1)] # first color is black, last is red
cm_blue = LinearSegmentedColormap.from_list("Custom", colors, N=20)

gaussian_sigma      = 12 #in degrees
binresolution       = 5
gaussian_sigma      = gaussian_sigma / binresolution
# protocol = 'GN'

#%% #############################################################################
# session_list        = np.array([['LPE12223_2024_06_08']])
# sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],only_session_id=session_list)

#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = [protocol],filter_areas=['V1','PM'],session_rf=True) 

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
                                calciumversion=calciumversion,keepraw=False)

#%%
sessions = ori_remapping(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=True)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    # sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=250)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 
    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 


#%%
from sklearn.decomposition import FactorAnalysis as FA

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
    # nc_behavout = np.empty((N,N,len(stims)))  

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
    # sessions[ises].nc_behavout      = np.mean(nc_behavout,axis=2)

#%% ##################### Compute pairwise receptive field distances: ##############################
# sessions = smooth_rf(sessions,radius=50,rf_type='Fneu',mincellsFneu=5)
# sessions = exclude_outlier_rf(sessions) 
# sessions = replace_smooth_with_Fsig(sessions) 

#%% Show preferred orientation across all cells in GN protocol:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

areas = ['V1','PM']
labeled = ['unl','lab']

fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharex=True,sharey=True)
for iarea,area in enumerate(areas):
    ax = axes[iarea]
    for ilab,lab in enumerate(labeled):
        df = pd.DataFrame({'Ori': celldata.loc[(celldata['roi_name']==area)&(celldata['labeled']==lab),'pref_ori']})
        ax.hist(df['Ori'],bins=np.arange(0,360+22.5,22.5)-22.5/2,
            histtype='step',color=get_clr_area_labeled([area+lab]),label=lab,alpha=1,density=True)
    leg = ax.legend(labeled,frameon=False,ncol=1,loc='upper right')
    for lh in leg.legend_handles:
        lh.set_visible(False)
    for text, color in zip(leg.texts, get_clr_area_labeled([area+labeled[0],area+labeled[1]])):
        text.set_color(color)
    ax.set_xlabel('Pref. Ori (deg)')
    if iarea == 0: 
        ax.set_ylabel('Density (a.u.)')
    ax.set_xticks(np.arange(0,360,45))
    ax.set_xlim([-10, 360]) #ax.set_xticks(np.arange(0,360,45))
    ax.set_title('%s' % area)
plt.tight_layout()
# fig.savefig(os.path.join('E:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\Tuning\\','PreferredOri_%s_GR' % (calciumversion) + '.png'), format = 'png')
# fig.savefig(os.path.join('E:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\Tuning\\','PreferredOri_%s_GR' % (corr_type) + '.png'), format = 'png')


#%% 




#%% 
# rf_type             = 'Fsmooth'
rf_type             = 'F'
areapairs            = ['V1-V1','PM-PM','V1-PM']
histedges           = np.arange(-100,100,10)
data_out            = np.empty((len(sessions),len(histedges)-1,2,len(areapairs)))

for ises in range(len(sessions)):
    for iap,areapair in enumerate(areapairs):
        source_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
        target_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
        delta_el        = source_el[:,None] - target_el[None,:]

        source_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
        target_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
        delta_az        = source_az[:,None] - target_az[None,:]

        areafilter      = filter_2d_areapair(sessions[ises],areapair)

        delta_rf        = np.sqrt(delta_az**2 + delta_el**2)
        # angle_rf        = np.mod(np.arctan2(delta_el,delta_az)-np.pi,np.pi*2)
        # angle_rf        = np.mod(angle_rf+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
        data_out[ises,:,0,iap] = np.histogram(delta_el[areafilter].flatten(),bins=histedges)[0]
        data_out[ises,:,1,iap] = np.histogram(delta_az[areafilter].flatten(),bins=histedges)[0]

#%%
fig,axes = plt.subplots(len(areapairs),2,figsize=(6,len(areapairs)*3))
for iap,areapair in enumerate(areapairs):
    ax = axes[iap,0]
    for ises in range(len(sessions)):
        ax.plot(histedges[:-1],data_out[ises,:,0,iap])
    ax.set_title('Δ El -  %s' % areapair)
    ax.set_xticks([-100,-50,0,50,100])
    ax.set_xlim([-100,100])

    ax = axes[iap,1]
    for ises in range(len(sessions)):
        ax.plot(histedges[:-1],data_out[ises,:,1,iap])
    ax.set_xlim([-100,100])
    ax.set_xticks([-100,-50,0,50,100])
    ax.set_title('Δ Az -  %s' % areapair)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Hist_Delta_Az_El_areapairs' + '.png'), format = 'png')

#%% Center surround Layers:


#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM','V1-PM','PM-V1']
# areapairs           = ['V1-PM','PM-V1']
# layerpairs          = ' '
layerpairs          = ['L2/3-L2/3','L2/3-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl']
# projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

deltaori            = None
rotate_prefori      = False
rf_type             = 'Fsmooth'
# rf_type             = 'F'
corr_type           = 'noise_corr'
# corr_type           = 'noise_cov'
# corr_type           = 'trace_corr'
tuned_thr           = 0.0
noise_thr           = 20
corr_thr            = 0.01
r2_thr              = 0.1
binresolution       = 10

[bincenters_2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

[bincenters_2d,bin_2d_posf_ses,bin_2d_count_ses,bin_dist_posf_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_posf_ses,bin_angle_cent_count_ses,bin_angle_surr_posf_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='pos',
                            areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

[bincenters_2d,bin_2d_negf_ses,bin_2d_count_ses,bin_dist_negf_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_negf_ses,bin_angle_cent_count_ses,bin_angle_surr_negf_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='neg',
                            areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#%% Compute mean over sessions
bin_2d_count = np.nansum(bin_2d_count_ses,0)
bin_2d_mean = nanweightedaverage(bin_2d_mean_ses,weights=bin_2d_count_ses,axis=0)
bin_2d_posf = nanweightedaverage(bin_2d_posf_ses,weights=bin_2d_count_ses,axis=0)
bin_2d_negf = nanweightedaverage(bin_2d_negf_ses,weights=bin_2d_count_ses,axis=0)

bin_dist_count = np.nansum(bin_dist_count_ses,0)
bin_dist_mean = nanweightedaverage(bin_dist_mean_ses,weights=bin_dist_count_ses,axis=0)
bin_dist_posf = nanweightedaverage(bin_dist_posf_ses,weights=bin_dist_count_ses,axis=0)
bin_dist_negf = nanweightedaverage(bin_dist_negf_ses,weights=bin_dist_count_ses,axis=0)

bin_angle_cent_count = np.nansum(bin_angle_cent_count_ses,0)
bin_angle_cent_mean = nanweightedaverage(bin_angle_cent_mean_ses,weights=bin_angle_cent_count_ses,axis=0)
bin_angle_cent_posf = nanweightedaverage(bin_angle_cent_posf_ses,weights=bin_angle_cent_count_ses,axis=0)    
bin_angle_cent_negf = nanweightedaverage(bin_angle_cent_negf_ses,weights=bin_angle_cent_count_ses,axis=0)

bin_angle_surr_count = np.nansum(bin_angle_surr_count_ses,0)
bin_angle_surr_mean = nanweightedaverage(bin_angle_surr_mean_ses,weights=bin_angle_surr_count_ses,axis=0)
bin_angle_surr_posf = nanweightedaverage(bin_angle_surr_posf_ses,weights=bin_angle_surr_count_ses,axis=0)    
bin_angle_surr_negf = nanweightedaverage(bin_angle_surr_negf_ses,weights=bin_angle_surr_count_ses,axis=0)

#%% Plot radial tuning:
ilp = 0
# ilp = 1
min_counts = 500
fig = plot_corr_radial_tuning_projs(bincenters_dist,bin_dist_count_ses[:,:,:,[ilp],:],
                                    bin_dist_mean_ses[:,:,:,[ilp],:],areapairs,layerpairs,projpairs,min_counts=min_counts)
fig.suptitle(layerpairs[ilp])
# fig.savefig(os.path.join(savedir,'RadialTuning_areas_projs_GN_%s_mean' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_radial_tuning_projs(bincenters_dist,bin_dist_count_ses[:,:,:,[ilp],:],
                                    bin_dist_posf_ses[:,:,:,[ilp],:],areapairs,layerpairs,projpairs,min_counts=min_counts)
fig.suptitle(layerpairs[ilp])
# fig.savefig(os.path.join(savedir,'RadialTuning_areas_projs_GN_%s_posf' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_radial_tuning_projs(bincenters_dist,bin_dist_count_ses[:,:,:,[ilp],:],
                                    bin_dist_negf_ses[:,:,:,[ilp],:],areapairs,layerpairs,projpairs,min_counts=min_counts)
fig.suptitle(layerpairs[ilp])
# fig.savefig(os.path.join(savedir,'RadialTuning_areas_projs_GN_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps for the mean correlation
fig = plot_2D_mean_corr(bin_2d_mean,bin_2d_count,bincenters_2d,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
# fig.savefig(os.path.join(savedir,'Rotated_DeltaRF_2D_projs_%s_mean' % (corr_type) + '.png'), format = 'png')


#%%
# dpref = np.arange(-360,360,1)
# # dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
# dpref2 = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)
# dpref2 = np.min(np.array([np.abs(dpref+360),np.abs(dpref-0),np.abs(dpref-360)]),axis=0)

# plt.scatter(dpref.flatten(),dpref2.flatten())



#%% 
######  ####### #       #######    #       ####### ######  ### 
#     # #       #          #      # #      #     # #     #  #  
#     # #       #          #     #   #     #     # #     #  #  
#     # #####   #          #    #     #    #     # ######   #  
#     # #       #          #    #######    #     # #   #    #  
#     # #       #          #    #     #    #     # #    #   #  
######  ####### #######    #    #     #    ####### #     # ### 

#%% Loop over all delta preferred orientations and store mean correlations as well as distribution of pos and neg correlations:
areapairs           = ['V1-V1','PM-PM','V1-PM','PM-V1']
# areapairs           = ['V1-V1']
layerpairs          = ' '
projpairs           = ' '

# deltaoris           = np.array([0,22.5,45,67.5,90])
for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)
    # sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref-0),np.abs(dpref-360)]),axis=0)
deltaoris           = np.unique(sessions[0].delta_pref[~np.isnan(sessions[0].delta_pref)]).astype(float)

for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)
deltaoris           = np.array([[0,30],[60,90]])

ndeltaoris          = len(deltaoris)

rotate_prefori      = True
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'
corr_thr            = 0.01
# 
# rf_type             = 'F'
rf_type             = 'Fsmooth'
# corr_type           = 'noise_cov'
# corr_type           = 'noise_corr'
noise_thr           = 100
r2_thr              = 0.1
binresolution       = 5
tuned_thr           = 75

# Run function once to get dimensions of output data:
[bincenters_2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                            r2_thr=1,deltaori=None,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#Init output arrays:
bin_2d_mean_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_posf_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_negf_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_count_oris_ses       = np.full((ndeltaoris,*np.shape(bin_2d_count_ses)),np.nan)

bin_dist_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_dist_count_ses)),np.nan)

bin_angle_cent_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_angle_cent_count_ses)),np.nan)

bin_angle_surr_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_angle_surr_count_ses)),np.nan)

for idOri,deltaori in enumerate(deltaoris):
# for idOri,centerori in enumerate(centeroris):
    [_,bin_2d_mean_oris_ses[idOri,:,:,:,:,:],bin_2d_count_oris_ses[idOri,:,:,:,:,:],
     bin_dist_mean_oris_ses[idOri,:,:,:,:],bin_dist_count_oris_ses[idOri,:,:,:],_,
     bin_angle_cent_mean_oris_ses[idOri,:,:,:,:],bin_angle_cent_count_oris_ses[idOri,:,:,:,:],
     bin_angle_surr_mean_oris_ses[idOri,:,:,:,:],bin_angle_surr_count_oris_ses[idOri,:,:,:,:],_] = bin_corr_deltarf_ses(sessions,
                                                    method='mean',filtersign=None,areapairs=areapairs,layerpairs=layerpairs,
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
    [_,bin_2d_posf_oris_ses[idOri,:,:,:,:,:],_,
     bin_dist_posf_oris_ses[idOri,:,:,:,:],_,_,
     bin_angle_cent_posf_oris_ses[idOri,:,:,:,:],_,
     bin_angle_surr_posf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='pos',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,corr_type=corr_type,
                                                    binresolution=binresolution,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
    [_,bin_2d_negf_oris_ses[idOri,:,:,:,:,:],_,
     bin_dist_negf_oris_ses[idOri,:,:,:,:],_,_,
     bin_angle_cent_negf_oris_ses[idOri,:,:,:,:],_,
     bin_angle_surr_negf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='neg',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,corr_type=corr_type,
                                                    binresolution=binresolution,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
#%%
deltaoris           = np.array([0,90])

#%%
bin_2d_posf_oris_ses[np.isinf(bin_2d_posf_oris_ses)] = np.nan
bin_2d_negf_oris_ses[np.isinf(bin_2d_negf_oris_ses)] = np.nan

#%% Compute mean over sessions for each orientation
bin_2d_count_oris = np.nansum(bin_2d_count_oris_ses,1)
bin_2d_mean_oris = np.nanmean(bin_2d_mean_oris_ses,1)
bin_2d_posf_oris = np.nanmean(bin_2d_posf_oris_ses,1)
bin_2d_negf_oris = np.nanmean(bin_2d_negf_oris_ses,1)

bin_dist_count_oris = np.nansum(bin_dist_count_oris_ses,1)
bin_dist_mean_oris = np.nanmean(bin_dist_mean_oris_ses,1)
bin_dist_posf_oris = np.nanmean(bin_dist_posf_oris_ses,1)
bin_dist_negf_oris = np.nanmean(bin_dist_negf_oris_ses,1)

bin_angle_cent_count_oris = np.nansum(bin_angle_cent_count_oris_ses,1)
bin_angle_cent_mean_oris = np.nanmean(bin_angle_cent_mean_oris_ses,1)
bin_angle_cent_posf_oris = np.nanmean(bin_angle_cent_posf_oris_ses,1)    
bin_angle_cent_negf_oris = np.nanmean(bin_angle_cent_negf_oris_ses,1)

bin_angle_surr_count_oris = np.nansum(bin_angle_surr_count_oris_ses,1)
bin_angle_surr_mean_oris = np.nanmean(bin_angle_surr_mean_oris_ses,1)
bin_angle_surr_posf_oris = np.nanmean(bin_angle_surr_posf_oris_ses,1)    
bin_angle_surr_negf_oris = np.nanmean(bin_angle_surr_negf_oris_ses,1)

#%% Compute mean over sessions for each orientation
bin_2d_count_oris = np.nansum(bin_2d_count_oris_ses,1)
bin_2d_mean_oris = nanweightedaverage(bin_2d_mean_oris_ses,weights=bin_2d_count_oris_ses,axis=1)
bin_2d_posf_oris = nanweightedaverage(bin_2d_posf_oris_ses,weights=bin_2d_count_oris_ses,axis=1)
bin_2d_negf_oris = nanweightedaverage(bin_2d_negf_oris_ses,weights=bin_2d_count_oris_ses,axis=1)

bin_dist_count_oris = np.nansum(bin_dist_count_oris_ses,1)
bin_dist_mean_oris = nanweightedaverage(bin_dist_mean_oris_ses,weights=bin_dist_count_oris_ses,axis=1)
bin_dist_posf_oris = nanweightedaverage(bin_dist_posf_oris_ses,weights=bin_dist_count_oris_ses,axis=1)
bin_dist_negf_oris = nanweightedaverage(bin_dist_negf_oris_ses,weights=bin_dist_count_oris_ses,axis=1)

bin_angle_cent_count_oris = np.nansum(bin_angle_cent_count_oris_ses,1)
bin_angle_cent_mean_oris = nanweightedaverage(bin_angle_cent_mean_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)
bin_angle_cent_posf_oris = nanweightedaverage(bin_angle_cent_posf_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)    
bin_angle_cent_negf_oris = nanweightedaverage(bin_angle_cent_negf_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)

bin_angle_surr_count_oris = np.nansum(bin_angle_surr_count_oris_ses,1)
bin_angle_surr_mean_oris = nanweightedaverage(bin_angle_surr_mean_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)
bin_angle_surr_posf_oris = nanweightedaverage(bin_angle_surr_posf_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)    
bin_angle_surr_negf_oris = nanweightedaverage(bin_angle_surr_negf_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)

#%% 
min_counts = 250

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_mean_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        centerthr=15,projpairs=projpairs,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_%dsessions_%s_mean' % (protocol,nSessions,corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_mean_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_%s_%dsessions_mean' % (protocol,corr_type,nSessions) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_mean_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Radial_Tuning_areas_%s_%s_%dsessions_mean' % (protocol,corr_type,nSessions) + '.png'), format = 'png')

#%% Show spatial maps per delta ori for the fraction positive 
fig = plot_2D_mean_corr_dori(bin_2d_posf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_red)
fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_%dsessions_%s_posf' % (protocol,nSessions,corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_posf_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_%s_%dsessions_posf' % (protocol,corr_type,nSessions) + '.png'), format = 'png')
# 
#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_posf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Radial_Tuning_areas_%s_%s_%dsessions_posf' % (protocol,corr_type,nSessions) + '.png'), format = 'png')

#%% Show spatial maps per delta ori for the fraction negative 
fig = plot_2D_mean_corr_dori(bin_2d_negf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_blue)
fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_%dsessions_%s_negf' % (protocol,nSessions,corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_negf_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_%s_%dsessions_negf' % (protocol,corr_type,nSessions) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_negf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Radial_Tuning_areas_%s_%s_%dsessions_negf' % (protocol,corr_type,nSessions) + '.png'), format = 'png')

#%%
min_counts = 250
centerthr = 15
rai_mean_oris_ses  = retinotopic_alignment_index(bin_dist_mean_oris_ses,bincenters_dist,bin_dist_count_oris_ses,min_counts=min_counts,centerthr=centerthr)
rai_posf_oris_ses  = retinotopic_alignment_index(bin_dist_posf_oris_ses,bincenters_dist,bin_dist_count_oris_ses,min_counts=min_counts,centerthr=centerthr)
rai_negf_oris_ses  = retinotopic_alignment_index(bin_dist_negf_oris_ses,bincenters_dist,bin_dist_count_oris_ses,min_counts=min_counts,centerthr=centerthr)
fig = plot_csi_deltaori_areas_ses(rai_mean_oris_ses,rai_posf_oris_ses,rai_negf_oris_ses,deltaoris,areapairs)
fig.savefig(os.path.join(savedir,'RAI_areas_%s_%s_%dsessions' % (protocol,corr_type,nSessions) + '.png'), format = 'png')

#%% Compute collinear selectivity index:
min_counts = 250
csi_surr_mean_oris =  collinear_selectivity_index(bin_angle_surr_mean_oris_ses,bincenters_angle,bin_angle_surr_count_oris_ses,min_counts=min_counts)    
csi_surr_posf_oris =  collinear_selectivity_index(bin_angle_surr_posf_oris_ses,bincenters_angle,bin_angle_surr_count_oris_ses,min_counts=min_counts)  
csi_surr_negf_oris =  collinear_selectivity_index(bin_angle_surr_negf_oris_ses,bincenters_angle,bin_angle_surr_count_oris_ses,min_counts=min_counts)  

# Plot the CSI values as function of delta ori for the three different areapairs
fig = plot_csi_deltaori_areas_ses(csi_surr_mean_oris,csi_surr_posf_oris,csi_surr_negf_oris,deltaoris,areapairs)
# fig = plot_csi_deltaori_areas_ses(csi_surr_mean_oris,csi_surr_posf_oris,csi_surr_negf_oris,deltaoris,areapairs)
fig.savefig(os.path.join(savedir,'CSI_areas_%s_%s_%dsessions' % (protocol,corr_type,nSessions) + '.png'), format = 'png')


     #    ####### ######  ###    ######  ######  #######       #  #####  
     #    #     # #     #  #     #     # #     # #     #       # #     # 
     #    #     # #     #  #     #     # #     # #     #       # #       
######    #     # ######   #     ######  ######  #     #       #  #####  
#    #    #     # #   #    #     #       #   #   #     # #     #       # 
#    #    #     # #    #   #     #       #    #  #     # #     # #     # 
######    ####### #     # ###    #       #     # #######  #####   #####  


#%% Loop over all delta preferred orientations and store mean correlations as well as distribution of pos and neg correlations:
areapairs           = ['V1-PM','PM-V1']
layerpairs          = ' '
# layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
layerpairs          = ['L2/3-L2/3', 'L2/3-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    # sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref-0),np.abs(dpref-360)]),axis=0)
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)

deltaoris           = np.unique(sessions[0].delta_pref[~np.isnan(sessions[0].delta_pref)]).astype(float)

for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)
deltaoris           = np.array([[0,30],[60,90]])

ndeltaoris          = len(deltaoris)
rotate_prefori      = True
corr_thr            = 0.01

# rf_type             = 'F'
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
noise_thr           = 100
r2_thr              = 0.1
binresolution       = 10
tuned_thr           = 75

#Run function once to get dimensions of output data:
[bincenters_2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                            r2_thr=1,deltaori=0,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#Init output arrays:
bin_2d_mean_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_posf_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_negf_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_count_oris_ses       = np.full((ndeltaoris,*np.shape(bin_2d_count_ses)),np.nan)

bin_dist_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_dist_count_ses)),np.nan)

bin_angle_cent_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_angle_cent_count_ses)),np.nan)

bin_angle_surr_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_angle_surr_count_ses)),np.nan)

for idOri,deltaori in enumerate(deltaoris):
# for idOri,centerori in enumerate(centeroris):
    [_,bin_2d_mean_oris_ses[idOri,:,:,:,:,:],bin_2d_count_oris_ses[idOri,:,:,:,:,:],
     bin_dist_mean_oris_ses[idOri,:,:,:,:],bin_dist_count_oris_ses[idOri,:,:,:],_,
     bin_angle_cent_mean_oris_ses[idOri,:,:,:,:],bin_angle_cent_count_oris_ses[idOri,:,:,:,:],
     bin_angle_surr_mean_oris_ses[idOri,:,:,:,:],bin_angle_surr_count_oris_ses[idOri,:,:,:,:],_] = bin_corr_deltarf_ses(sessions,
                                                    method='mean',filtersign=None,areapairs=areapairs,layerpairs=layerpairs,
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
    [_,bin_2d_posf_oris_ses[idOri,:,:,:,:,:],_,
     bin_dist_posf_oris_ses[idOri,:,:,:,:],_,_,
     bin_angle_cent_posf_oris_ses[idOri,:,:,:,:],_,
     bin_angle_surr_posf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='pos',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,corr_type=corr_type,
                                                    binresolution=binresolution,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
    [_,bin_2d_negf_oris_ses[idOri,:,:,:,:,:],_,
     bin_dist_negf_oris_ses[idOri,:,:,:,:],_,_,
     bin_angle_cent_negf_oris_ses[idOri,:,:,:,:],_,
     bin_angle_surr_negf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='neg',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,corr_type=corr_type,
                                                    binresolution=binresolution,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#%%
deltaoris           = np.array([0,90])


#%% Compute mean over sessions for each orientation
bin_2d_count_oris = np.nansum(bin_2d_count_oris_ses,1)
bin_2d_mean_oris = nanweightedaverage(bin_2d_mean_oris_ses,weights=bin_2d_count_oris_ses,axis=1)
bin_2d_posf_oris = nanweightedaverage(bin_2d_posf_oris_ses,weights=bin_2d_count_oris_ses,axis=1)
bin_2d_negf_oris = nanweightedaverage(bin_2d_negf_oris_ses,weights=bin_2d_count_oris_ses,axis=1)

bin_dist_count_oris = np.nansum(bin_dist_count_oris_ses,1)
bin_dist_mean_oris = nanweightedaverage(bin_dist_mean_oris_ses,weights=bin_dist_count_oris_ses,axis=1)
bin_dist_posf_oris = nanweightedaverage(bin_dist_posf_oris_ses,weights=bin_dist_count_oris_ses,axis=1)
bin_dist_negf_oris = nanweightedaverage(bin_dist_negf_oris_ses,weights=bin_dist_count_oris_ses,axis=1)

bin_angle_cent_count_oris = np.nansum(bin_angle_cent_count_oris_ses,1)
bin_angle_cent_mean_oris = nanweightedaverage(bin_angle_cent_mean_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)
bin_angle_cent_posf_oris = nanweightedaverage(bin_angle_cent_posf_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)    
bin_angle_cent_negf_oris = nanweightedaverage(bin_angle_cent_negf_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)

bin_angle_surr_count_oris = np.nansum(bin_angle_surr_count_oris_ses,1)
bin_angle_surr_mean_oris = nanweightedaverage(bin_angle_surr_mean_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)
bin_angle_surr_posf_oris = nanweightedaverage(bin_angle_surr_posf_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)    
bin_angle_surr_negf_oris = nanweightedaverage(bin_angle_surr_negf_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)

#%% 

#%% Show spatial maps per delta ori for the mean correlation
min_counts = 10
for iap in range(len(areapairs)):
    for ilp in range(len(layerpairs)):
        bindata     = bin_2d_mean_oris[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        bincounts   = bin_2d_count_oris[:,:,:,[iap],:,:]
        bincounts   = bincounts[:,:,:,:,[ilp],:]
        fig = plot_2D_mean_corr_projs_dori(bindata,bincounts,bincenters_2d,deltaoris,
                                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,cmap='magma',
                                           min_counts=min_counts,gaussian_sigma=gaussian_sigma)
        fig.suptitle(areapairs[iap] + ' ' + layerpairs[ilp])
        filename = 'Collinear_DeltaRF_2D_projs_%s_%s_%s_%s_mean' % (protocol,corr_type,areapairs[iap],layerpairs[ilp].replace('/',''))
        my_savefig(fig,os.path.join(savedir,'Projs'),filename,formats=['png'])

#%% Angular tuning of surround:
for iap in range(len(areapairs)):
    for ilp in range(len(layerpairs)):
        bindata = bin_angle_surr_mean_oris[:,:,[iap],:,:]
        bindata = bindata[:,:,:,[ilp],:]
        bincounts = bin_angle_surr_count_oris[:,:,[iap],:,:]
        bincounts = bincounts[:,:,:,[ilp],:]
        fig = plot_corr_angular_tuning_projs_dori(bindata,bincounts,bincenters_angle,
                    deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
        fig.suptitle(areapairs[iap] + ' ' + layerpairs[ilp])
        filename = 'Angular_Tuning_Surround_projs_%s_%s_%s_%s_mean' % (protocol,corr_type,areapairs[iap],layerpairs[ilp].replace('/',''))
        my_savefig(fig,os.path.join(savedir,'Projs'),filename,formats=['png'])

#%% Radial tuning:
for iap in range(len(areapairs)):
    for ilp in range(len(layerpairs)):
        bindata = bin_dist_mean_oris[:,:,[iap],:,:]
        bindata = bindata[:,:,:,[ilp],:]
        bincounts = bin_dist_count_oris[:,:,[iap],:,:]
        bincounts = bincounts[:,:,:,[ilp],:]
        fig = plot_corr_radial_tuning_projs_dori(bincenters_dist,bincounts,bindata,deltaoris,
                    min_counts=min_counts,areapairs=[areapairs[iap]],layerpairs=layerpairs,projpairs=projpairs)
        fig.suptitle(areapairs[iap] + ' ' + layerpairs[ilp])
        filename = 'Radial_Tuning_projs_%s_%s_%s_%s_mean' % (protocol,corr_type,areapairs[iap],layerpairs[ilp].replace('/',''))
        my_savefig(fig,os.path.join(savedir,'Projs'),filename,formats=['png'])


#%% Show spatial maps per delta ori for the positive correlation
min_counts = 50
for iap in range(len(areapairs)):
    for ilp in range(len(layerpairs)):
        bindata     = bin_2d_posf_oris[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        bincounts   = bin_2d_count_oris[:,:,:,[iap],:,:]
        bincounts   = bincounts[:,:,:,:,[ilp],:]
        fig = plot_2D_mean_corr_projs_dori(bindata,bincounts,bincenters_2d,deltaoris,
                                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,cmap=cm_red,
                                           min_counts=min_counts,gaussian_sigma=gaussian_sigma)
        fig.suptitle(areapairs[iap] + ' ' + layerpairs[ilp])
        filename = 'Collinear_DeltaRF_2D_projs_%s_%s_%s_%s_posf' % (protocol,corr_type,areapairs[iap],layerpairs[ilp].replace('/',''))
        my_savefig(fig,os.path.join(savedir,'Projs'),filename,formats=['png'])

#%% Show spatial maps per delta ori for the negative correlation
min_counts = 50
for iap in range(len(areapairs)):
    for ilp in range(len(layerpairs)):
        bindata     = bin_2d_negf_oris[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        bincounts   = bin_2d_count_oris[:,:,:,[iap],:,:]
        bincounts   = bincounts[:,:,:,:,[ilp],:]
        fig = plot_2D_mean_corr_projs_dori(bindata,bincounts,bincenters_2d,deltaoris,
                                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,cmap=cm_blue,
                                           min_counts=min_counts,gaussian_sigma=gaussian_sigma)
        fig.suptitle(areapairs[iap] + ' ' + layerpairs[ilp])
        filename = 'Collinear_DeltaRF_2D_projs_%s_%s_%s_%s_negf' % (protocol,corr_type,areapairs[iap],layerpairs[ilp].replace('/',''))
        my_savefig(fig,os.path.join(savedir,'Projs'),filename,formats=['png'])

#%% Compute collinear selectivity index:
min_counts = 250
for iap in range(len(areapairs)):
    for ilp in range(len(layerpairs)):
        bindata     = bin_angle_surr_mean_oris_ses[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        bincounts   = bin_angle_surr_count_oris_ses[:,:,:,[iap],:,:]
        bincounts   = bincounts[:,:,:,:,[ilp],:]

        csi_surr_mean_oris =  collinear_selectivity_index(bindata,bincenters_angle,bincounts,min_counts=min_counts)    
        
        bindata     = bin_angle_surr_posf_oris_ses[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        csi_surr_posf_oris =  collinear_selectivity_index(bindata,bincenters_angle,bincounts,min_counts=min_counts)
        
        bindata     = bin_angle_surr_negf_oris_ses[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        csi_surr_negf_oris =  collinear_selectivity_index(bindata,bincenters_angle,bincounts,min_counts=min_counts)
        # Plot the CSI values as function of delta ori for the three different areapairs
        fig = plot_csi_deltaori_projs_ses(csi_surr_mean_oris,csi_surr_posf_oris,csi_surr_negf_oris,deltaoris,projpairs)
        fig.suptitle(areapairs[iap] + ' ' + layerpairs[ilp])
        filename = 'CSI_projs_%s_%s_%s_%s' % (protocol,corr_type,areapairs[iap],layerpairs[ilp].replace('/',''))
        my_savefig(fig,os.path.join(savedir,'Projs'),filename,formats=['png'])

#%% Compute collinear selectivity index:
min_counts = 250
centerthr = 15
for iap in range(len(areapairs)):
    for ilp in range(len(layerpairs)):
        bindata     = bin_dist_mean_oris_ses[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        bincounts   = bin_dist_count_oris_ses[:,:,:,[iap],:,:]
        bincounts   = bincounts[:,:,:,:,[ilp],:]

        rai_surr_mean_oris =  retinotopic_alignment_index(bindata,bincenters_dist,bincounts,min_counts=min_counts,centerthr=centerthr)    
        
        bindata     = bin_dist_posf_oris_ses[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        rai_surr_posf_oris =  retinotopic_alignment_index(bindata,bincenters_dist,bincounts,min_counts=min_counts,centerthr=centerthr)    
        
        bindata     = bin_dist_negf_oris_ses[:,:,:,[iap],:,:]
        bindata     = bindata[:,:,:,:,[ilp],:]
        rai_surr_negf_oris =  retinotopic_alignment_index(bindata,bincenters_dist,bincounts,min_counts=min_counts,centerthr=centerthr)    

        # Plot the CSI values as function of delta ori for the three different areapairs
        fig = plot_rai_deltaori_projs_ses(rai_surr_mean_oris,rai_surr_posf_oris,rai_surr_negf_oris,deltaoris,projpairs)
        fig.suptitle(areapairs[iap] + ' ' + layerpairs[ilp])
        filename = 'RAI_projs_%s_%s_%s_%s' % (protocol,corr_type,areapairs[iap],layerpairs[ilp].replace('/',''))
        my_savefig(fig,os.path.join(savedir,'Projs'),filename,formats=['png'])

#%%
# print(np.sum(bin_2d_count_oris_ses,axis=(0,2,3,4,5,6)))

# counts = np.sum(bin_2d_count_oris_ses[:,:,:,[iap],:,:],axis=(0,2,3,4,5,6))
# print(counts)
# idx_ses = np.argmax(counts)

# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessiondata['session_id'][idx_ses]

# #%%
# bindata     = bin_angle_surr_mean_oris_ses[:,:,:,[iap],:,:]
# bindata     = bindata[:,:,:,:,[ilp],:]
# bincounts   = bin_angle_surr_count_oris_ses[:,:,:,[iap],:,:]
# bincounts   = bincounts[:,:,:,:,[ilp],:]

# csi_surr_mean_oris =  collinear_selectivity_index(bin_angle_surr_mean_oris_ses,bincenters_angle,bin_angle_surr_count_oris_ses,min_counts=min_counts)    
        

# idx_ses = np.argmax(csi_surr_mean_oris[1,:,0,1,0])

# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessiondata['session_id'][idx_ses]
# 'LPE09665_2023_03_21'
# 'LPE10919_2023_11_06'