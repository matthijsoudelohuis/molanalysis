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
gaussian_sigma      = 1

#%% #############################################################################

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
    delattr(sessions[ises],'calciumdata')


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

#%% ##################### Compute pairwise receptive field distances: ##############################
# sessions = smooth_rf(sessions,radius=50,rf_type='Fneu',mincellsFneu=5)
# sessions = exclude_outlier_rf(sessions) 
# sessions = replace_smooth_with_Fsig(sessions) 

#%% Show preferred orientation across all cells in GR protocol:
sessions_GR = [ses for ses in sessions if ses.protocol == 'GR']
celldata = pd.concat([ses.celldata for ses in sessions_GR]).reset_index(drop=True)

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
    ax.set_xlim([-10, 340]) #ax.set_xticks(np.arange(0,360,45))
    ax.set_title('%s' % area)
plt.tight_layout()
fig.savefig(os.path.join('E:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\Tuning\\','PreferredOri_%s_GR' % (calciumversion) + '.png'), format = 'png')
# fig.savefig(os.path.join('E:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\Tuning\\','PreferredOri_%s_GR' % (corr_type) + '.png'), format = 'png')


#%% Compute collinear selectivity index:
def collinear_selectivity_index(data,bincenters_angle):
    if np.ndim(data) == 4:
        resp_surr_col    = np.mean(data[np.isin(bincenters_angle,[0,np.pi]),:,:,:],axis=0)
        resp_surr_perp   = np.mean(data[np.isin(bincenters_angle,[np.pi/2,1.5*np.pi]),:,:,:],axis=0)
        CSI             = (resp_surr_col - resp_surr_perp) / (resp_surr_col + resp_surr_perp)
    elif np.ndim(data) == 5:
        resp_surr_col    = np.mean(data[:,np.isin(bincenters_angle,[0,np.pi]),:,:,:],axis=1)
        resp_surr_perp   = np.mean(data[:,np.isin(bincenters_angle,[np.pi/2,1.5*np.pi]),:,:,:],axis=1)
        # CSI             = (resp_surr_col - resp_surr_perp) / resp_surr_col

        CSI             = (resp_surr_col - resp_surr_perp) / (resp_surr_col + resp_surr_perp)

    else:
        raise ValueError('data must have 4 or 5 dimensions')

    return CSI

#%% 




#%% Show delta RF and angle RF in simple schematic plots:

#Binning parameters 2D:
binresolution   = 5
binlim          = 75
binedges_2d     = np.arange(-binlim,binlim,binresolution)+binresolution/2 
bincenters_2d   = binedges_2d[:-1]+binresolution/2 
nBins           = len(bincenters_2d)

delta_az,delta_el = np.meshgrid(bincenters_2d,-bincenters_2d)
delta_rf         = np.sqrt(delta_az**2 + delta_el**2)
# angle_rf         = np.mod(np.arctan2(delta_el,delta_az)-np.pi,np.pi*2)
angle_rf         = np.mod(np.arctan2(delta_az,delta_el)+np.pi/2,np.pi*2)
# angle_rf         = np.mod(np.arctan2(delta_el,delta_az)+np.pi,np.pi*2)

cmap = 'cool'
cmap = 'bone_r'
cmap = 'copper_r'

fig,axes = plt.subplots(1,2,figsize=(5,2.5))
im = axes[0].imshow(delta_rf,extent=[-binlim,binlim,-binlim,binlim],cmap=cmap)
axes[0].set_title('Δ RF')
axes[0].set_xlabel(u'Δ Azimuth (\N{DEGREE SIGN})')
axes[0].set_ylabel(u'Δ Elevation (\N{DEGREE SIGN})')
axes[0].set_xticks([-50,0,50])
axes[0].set_yticks([-50,0,50])
cbar = fig.colorbar(im,ax=axes[0],shrink=0.35,aspect=5,pad=0.2)
cbar.ax.set_title(u'ΔRF (\N{DEGREE SIGN})',fontsize=10)
cbar.set_ticks([0,50,100])

cmap = 'twilight'
# cmap = 'cool'
im = axes[1].imshow(np.rad2deg(angle_rf),extent=[-binlim,binlim,-binlim,binlim],cmap=cmap)
axes[1].set_title('RF Angle')
axes[1].set_xlabel(u'Δ Azimuth (\N{DEGREE SIGN})')
axes[1].set_xticks([-50,0,50])
axes[1].set_yticks([-50,0,50])
cbar = fig.colorbar(im,ax=axes[1],shrink=0.35,aspect=5,pad=0.2)
cbar.set_ticks([0,180,360])
cbar.ax.set_title(u'Angle (\N{DEGREE SIGN})',fontsize=10)

plt.tight_layout()
# fig.savefig(os.path.join(savedir,'DeltaRF_AngleRF_2D' + '.png'), format = 'png')

#%% Show definition of delta azimuth and elevation:

cmap = 'twilight_r'
cmap = 'coolwarm'

fig,axes = plt.subplots(1,2,figsize=(5,2.5))
im = axes[0].imshow(delta_az,extent=[-binlim,binlim,-binlim,binlim],cmap=cmap)
axes[0].set_title('Azimuth')
axes[0].set_xlabel(u'Δ Azimuth (\N{DEGREE SIGN})')
axes[0].set_ylabel(u'Δ Elevation (\N{DEGREE SIGN})')
axes[0].set_xticks([-50,0,50])
axes[0].set_yticks([-50,0,50])
cbar = fig.colorbar(im,ax=axes[0],shrink=0.35,aspect=5,pad=0.2)
cbar.set_ticks([-70,0,70])
cbar.set_ticklabels([u'-70\N{DEGREE SIGN}',u'0\N{DEGREE SIGN}',u'+70\N{DEGREE SIGN}'])

# cmap = 'cool'
im = axes[1].imshow(delta_el,extent=[-binlim,binlim,-binlim,binlim],cmap=cmap)
axes[1].set_title('Elevation')
axes[1].set_xlabel(u'Δ Azimuth (\N{DEGREE SIGN})')
axes[1].set_xticks([-50,0,50])
axes[1].set_yticks([-50,0,50])
cbar = fig.colorbar(im,ax=axes[1],shrink=0.35,aspect=5,pad=0.2)
cbar.set_ticks([-70,0,70])
cbar.set_ticklabels([u'-70\N{DEGREE SIGN}',u'0\N{DEGREE SIGN}',u'+70\N{DEGREE SIGN}'])

plt.tight_layout()
# fig.savefig(os.path.join(savedir,'DeltaAz_DeltaEl_2D' + '.png'), format = 'png')

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



#%% 
######  ####### #######    #    ####### ####### ######        #    ######  #######    #     #####  
#     # #     #    #      # #      #    #       #     #      # #   #     # #         # #   #     # 
#     # #     #    #     #   #     #    #       #     #     #   #  #     # #        #   #  #       
######  #     #    #    #     #    #    #####   #     #    #     # ######  #####   #     #  #####  
#   #   #     #    #    #######    #    #       #     #    ####### #   #   #       #######       # 
#    #  #     #    #    #     #    #    #       #     #    #     # #    #  #       #     # #     # 
#     # #######    #    #     #    #    ####### ######     #     # #     # ####### #     #  #####  

#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM','V1-PM','PM-V1']
layerpairs          = ' '
# layerpairs          = ['L2/3-L2/3']
projpairs           = ' '

deltaori            = None
rotate_prefori      = True
# rf_type             = 'Fsmooth'
rf_type             = 'F'
# corr_type           = 'noise_corr'
corr_type           = 'noise_cov'
# corr_type           = 'trace_corr'
tuned_thr           = 0.0
noise_thr           = 100
min_counts          = 100
corr_thr            = 0.01
r2_thr              = 0.1

[bincenters_2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

[bincenters_2d,bin_2d_posf_ses,bin_2d_count_ses,bin_dist_posf_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_posf_ses,bin_angle_cent_count_ses,bin_angle_surr_posf_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='pos',
                            areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

[bincenters_2d,bin_2d_negf_ses,bin_2d_count_ses,bin_dist_negf_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_negf_ses,bin_angle_cent_count_ses,bin_angle_surr_negf_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='neg',
                            areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#%% Take the (weighted) mean across sessions:
bin_2d_mean = nanweightedaverage(bin_2d_mean_ses,weights=bin_2d_count_ses,axis=0)
bin_2d_count = np.nansum(bin_2d_count_ses,axis=0)

#Huge difference! 
bin_dist_count = np.nansum(bin_dist_count_ses,axis=0)
bin_dist_mean = np.nanmean(bin_dist_mean_ses,axis=0)
bin_dist_posf = np.nanmean(bin_dist_posf_ses,axis=0)
bin_dist_negf = np.nanmean(bin_dist_negf_ses,axis=0)

bin_2d_count = np.nansum(bin_2d_count_ses,axis=0)
bin_2d_mean = np.nanmean(bin_2d_mean_ses,axis=0)
bin_2d_posf = np.nanmean(bin_2d_posf_ses,axis=0)
bin_2d_negf = np.nanmean(bin_2d_negf_ses,axis=0)

bin_angle_cent_count = np.nansum(bin_angle_cent_count_ses,axis=0)
bin_angle_cent_mean = np.nanmean(bin_angle_cent_mean_ses,axis=0)
bin_angle_cent_posf = np.nanmean(bin_angle_cent_posf_ses,axis=0)
bin_angle_cent_negf = np.nanmean(bin_angle_cent_negf_ses,axis=0)

bin_angle_surr_count = np.nansum(bin_angle_surr_count_ses,axis=0)
bin_angle_surr_mean = np.nanmean(bin_angle_surr_mean_ses,axis=0)
bin_angle_surr_posf = np.nanmean(bin_angle_surr_posf_ses,axis=0)
bin_angle_surr_negf = np.nanmean(bin_angle_surr_negf_ses,axis=0)

#%% Compute collinear selectivity index:
csi_cent_mean =  collinear_selectivity_index(bin_angle_cent_mean,bincenters_angle)
csi_surr_mean =  collinear_selectivity_index(bin_angle_surr_mean,bincenters_angle)    

csi_cent_posf =  collinear_selectivity_index(bin_angle_cent_posf,bincenters_angle)
csi_surr_posf =  collinear_selectivity_index(bin_angle_surr_posf,bincenters_angle)

csi_cent_negf =  collinear_selectivity_index(bin_angle_cent_negf,bincenters_angle)
csi_surr_negf =  collinear_selectivity_index(bin_angle_surr_negf,bincenters_angle)

#%% Show spatial maps for the mean correlation
fig = plot_2D_mean_corr(bin_2d_mean,bin_2d_count,bincenters_2d,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
# fig.savefig(os.path.join(savedir,'Rotated_DeltaRF_2D_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps for the fraction positive
fig = plot_2D_mean_corr(bin_2d_posf,bin_2d_count,bincenters_2d,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_red)
# fig.savefig(os.path.join(savedir,'Rotated_DeltaRF_2D_areas_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps for the fraction negative
fig = plot_2D_mean_corr(bin_2d_negf,bin_2d_count,bincenters_2d,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_blue)
# fig.savefig(os.path.join(savedir,'Rotated_DeltaRF_2D_areas_%s_negf' % (corr_type) + '.png'), format = 'png')


#%% Plot radial tuning:
fig = plot_corr_radial_tuning_areas_mean(bincenters_dist,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'Rotated_RadialTuning_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_radial_tuning_areas_mean(bincenters_dist,bin_dist_count,bin_dist_posf,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'Rotated_RadialTuning_areas_%s_posf' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_radial_tuning_areas_mean(bincenters_dist,bin_dist_count,bin_dist_negf,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'Rotated_RadialTuning_areas_%s_negf' % (corr_type) + '.png'), format = 'png')


#%% Plot angular tuning of the surround:
fig = plot_corr_angular_tuning(sessions,bin_angle_surr_mean,bin_angle_surr_count,
                               bincenters_angle,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Rotated_AngularTuning_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_angular_tuning(sessions,bin_angle_surr_posf,bin_angle_surr_count,
                               bincenters_angle,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Rotated_AngularTuning_areas_%s_posf' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_angular_tuning(sessions,bin_angle_surr_negf,bin_angle_surr_count,
                               bincenters_angle,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Rotated_AngularTuning_areas_%s_negf' % (corr_type) + '.png'), format = 'png')






#%% 
######  ####### #######    #    ####### ####### ######     ######  ######  #######       #  #####  
#     # #     #    #      # #      #    #       #     #    #     # #     # #     #       # #     # 
#     # #     #    #     #   #     #    #       #     #    #     # #     # #     #       # #       
######  #     #    #    #     #    #    #####   #     #    ######  ######  #     #       #  #####  
#   #   #     #    #    #######    #    #       #     #    #       #   #   #     # #     #       # 
#    #  #     #    #    #     #    #    #       #     #    #       #    #  #     # #     # #     # 
#     # #######    #    #     #    #    ####### ######     #       #     # #######  #####   #####  

#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-PM','PM-V1']
layerpairs          = ' '
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

deltaori            = None
rotate_prefori      = True
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
noise_thr           = 20
min_counts          = 100
corr_thr            = 0.01

rf_type             = 'F'
corr_type           = 'noise_cov'
tuned_thr           = 0.025
noise_thr           = 100
min_counts          = 100
corr_thr            = 0.01
r2_thr              = 0.1

[bincenters_2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

[bincenters_2d,bin_2d_posf_ses,bin_2d_count_ses,bin_dist_posf_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_posf_ses,bin_angle_cent_count_ses,bin_angle_surr_posf_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='pos',
                            areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

[bincenters_2d,bin_2d_negf_ses,bin_2d_count_ses,bin_dist_negf_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_negf_ses,bin_angle_cent_count_ses,bin_angle_surr_negf_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='neg',
                            areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                            r2_thr=r2_thr,
                            deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#%% Take the (weighted) mean across sessions:
# bin_2d_mean = nanweightedaverage(bin_2d_mean_ses,weights=bin_2d_count_ses,axis=0)
# bin_2d_count = np.nansum(bin_2d_count_ses,axis=0)

#Huge difference! 
bin_dist_count = np.nansum(bin_dist_count_ses,axis=0)
bin_dist_mean = np.nanmean(bin_dist_mean_ses,axis=0)
bin_dist_posf = np.nanmean(bin_dist_posf_ses,axis=0)
bin_dist_negf = np.nanmean(bin_dist_negf_ses,axis=0)

bin_2d_count = np.nansum(bin_2d_count_ses,axis=0)
bin_2d_mean = np.nanmean(bin_2d_mean_ses,axis=0)
bin_2d_posf = np.nanmean(bin_2d_posf_ses,axis=0)
bin_2d_negf = np.nanmean(bin_2d_negf_ses,axis=0)

bin_angle_cent_count = np.nansum(bin_angle_cent_count_ses,axis=0)
bin_angle_cent_mean = np.nanmean(bin_angle_cent_mean_ses,axis=0)
bin_angle_cent_posf = np.nanmean(bin_angle_cent_posf_ses,axis=0)
bin_angle_cent_negf = np.nanmean(bin_angle_cent_negf_ses,axis=0)

bin_angle_surr_count = np.nansum(bin_angle_surr_count_ses,axis=0)
bin_angle_surr_mean = np.nanmean(bin_angle_surr_mean_ses,axis=0)
bin_angle_surr_posf = np.nanmean(bin_angle_surr_posf_ses,axis=0)
bin_angle_surr_negf = np.nanmean(bin_angle_surr_negf_ses,axis=0)

#%% Compute collinear selectivity index:
csi_cent_mean =  collinear_selectivity_index(bin_angle_cent_mean,bincenters_angle)
csi_surr_mean =  collinear_selectivity_index(bin_angle_surr_mean,bincenters_angle)    

csi_cent_posf =  collinear_selectivity_index(bin_angle_cent_posf,bincenters_angle)
csi_surr_posf =  collinear_selectivity_index(bin_angle_surr_posf,bincenters_angle)

csi_cent_negf =  collinear_selectivity_index(bin_angle_cent_negf,bincenters_angle)
csi_surr_negf =  collinear_selectivity_index(bin_angle_surr_negf,bincenters_angle)


#%% 
# corr_type           = 'noise_corr_GM'

gaussian_sigma = 1

#%% Show spatial maps for the mean correlation
fig = plot_2D_mean_corr(bin_2d_mean,bin_2d_count,bincenters_2d,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
fig.savefig(os.path.join(savedir,'Rotated_DeltaRF_2D_projs_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps for the fraction positive
fig = plot_2D_mean_corr(bin_2d_posf,bin_2d_count,bincenters_2d,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_red)
fig.savefig(os.path.join(savedir,'Rotated_DeltaRF_2D_projs_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps for the fraction negative
fig = plot_2D_mean_corr(bin_2d_negf,bin_2d_count,bincenters_2d,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_blue)
fig.savefig(os.path.join(savedir,'Rotated_DeltaRF_2D_projs_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Plot radial tuning:
fig = plot_corr_radial_tuning_projs(bincenters_dist,bin_dist_count_ses,bin_dist_mean_ses,areapairs,layerpairs,projpairs)
# fig = plot_corr_radial_tuning_projs(bincenters_dist,bin_dist_count,bin_dist_mean,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'Rotated_RadialTuning_projs_%s_mean' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_radial_tuning_projs(bincenters_dist,bin_dist_count_ses,bin_dist_posf_ses,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'Rotated_RadialTuning_projs_%s_posf' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_radial_tuning_projs(bincenters_dist,bin_dist_count_ses,bin_dist_negf_ses,areapairs,layerpairs,projpairs)
fig.savefig(os.path.join(savedir,'Rotated_RadialTuning_projs_%s_negf' % (corr_type) + '.png'), format = 'png')


#%% Plot angular tuning:
fig = plot_corr_angular_tuning(sessions,bin_angle_surr_mean,bin_angle_surr_count,
                               bincenters_angle,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Rotated_AngularTuning_2D_projs_%s_mean' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_angular_tuning(sessions,bin_angle_surr_posf,bin_angle_surr_count,
                               bincenters_angle,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Rotated_AngularTuning_2D_projs_%s_posf' % (corr_type) + '.png'), format = 'png')

fig = plot_corr_angular_tuning(sessions,bin_angle_surr_negf,bin_angle_surr_count,
                               bincenters_angle,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Rotated_AngularTuning_2D_projs_%s_negf' % (corr_type) + '.png'), format = 'png')






#%%
 #####  ####### #     # ####### ####### ######     ####### ######  ### 
#     # #       ##    #    #    #       #     #    #     # #     #  #  
#       #       # #   #    #    #       #     #    #     # #     #  #  
#       #####   #  #  #    #    #####   ######     #     # ######   #  
#       #       #   # #    #    #       #   #      #     # #   #    #  
#     # #       #    ##    #    #       #    #     #     # #    #   #  
 #####  ####### #     #    #    ####### #     #    ####### #     # ### 

#%% Loop over all center preferred orientations and store mean correlations as well as distribution of pos and neg correlations:
areapairs           = ['V1-V1','PM-PM','V1-PM','PM-V1']
layerpairs          = ' '
projpairs           = ' '

centeroris          = np.unique(sessions[0].celldata['pref_ori'])
ncenteroris         = len(centeroris)
rotate_prefori      = False
rf_type             = 'Fsmooth'
# corr_type           = 'noise_corr'
corr_type           = 'trace_corr'
tuned_thr           = 0.025
noise_thr           = 20
min_counts          = 50

rf_type             = 'F'
corr_type           = 'noise_cov'
noise_thr           = 100
r2_thr              = 0.1
binresolution       = 10

[bincenters_2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                            r2_thr=1,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#Init output arrays:
bin_2d_mean_oris_ses        = np.full((ncenteroris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_posf_oris_ses        = np.full((ncenteroris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_negf_oris_ses        = np.full((ncenteroris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_count_oris_ses       = np.full((ncenteroris,*np.shape(bin_2d_count_ses)),np.nan)

bin_dist_mean_oris_ses      = np.full((ncenteroris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_posf_oris_ses      = np.full((ncenteroris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_negf_oris_ses      = np.full((ncenteroris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_count_oris_ses     = np.full((ncenteroris,*np.shape(bin_dist_count_ses)),np.nan)

bin_angle_cent_mean_oris_ses      = np.full((ncenteroris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_posf_oris_ses      = np.full((ncenteroris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_negf_oris_ses      = np.full((ncenteroris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_count_oris_ses     = np.full((ncenteroris,*np.shape(bin_angle_cent_count_ses)),np.nan)

bin_angle_surr_mean_oris_ses      = np.full((ncenteroris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_posf_oris_ses      = np.full((ncenteroris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_negf_oris_ses      = np.full((ncenteroris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_count_oris_ses     = np.full((ncenteroris,*np.shape(bin_angle_surr_count_ses)),np.nan)

for idOri,centerori in enumerate(centeroris):
    [_,bin_2d_mean_oris_ses[idOri,:,:,:,:,:],bin_2d_count_oris_ses[idOri,:,:,:,:,:],
     bin_dist_mean_oris_ses[idOri,:,:,:,:],bin_dist_count_oris_ses[idOri,:,:,:],_,
     bin_angle_cent_mean_oris_ses[idOri,:,:,:,:],bin_angle_cent_count_oris_ses[idOri,:,:,:,:],
     bin_angle_surr_mean_oris_ses[idOri,:,:,:,:],bin_angle_surr_count_oris_ses[idOri,:,:,:,:],_] = bin_corr_deltarf_ses(sessions,
                                                    method='mean',filtersign=None,areapairs=areapairs,layerpairs=layerpairs,
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                                                    r2_thr=r2_thr,centerori=centerori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
    # [_,bin_2d_posf_oris_ses[idOri,:,:,:,:,:],_,
    #  bin_dist_posf_oris_ses[idOri,:,:,:,:],_,_,
    #  bin_angle_cent_posf_oris_ses[idOri,:,:,:,:],_,
    #  bin_angle_surr_posf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='pos',
    #                                                 areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
    #                                                 corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,centerori=centerori,
    #                                                 r2_thr=r2_thr,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

    # [_,bin_2d_negf_oris_ses[idOri,:,:,:,:,:],_,
    #  bin_dist_negf_oris_ses[idOri,:,:,:,:],_,_,
    #  bin_angle_cent_negf_oris_ses[idOri,:,:,:,:],_,
    #  bin_angle_surr_negf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='neg',
    #                                                 areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
    #                                                 corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,centerori=centerori,
    #                                                 r2_thr=r2_thr,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)


#%% Compute mean over sessions for each orientation
bin_2d_count_oris = np.nanmean(bin_2d_count_oris_ses,1)
bin_2d_mean_oris = np.nanmean(bin_2d_mean_oris_ses,1)
bin_2d_posf_oris = np.nanmean(bin_2d_posf_oris_ses,1)
bin_2d_negf_oris = np.nanmean(bin_2d_negf_oris_ses,1)

bin_dist_count_oris = np.nanmean(bin_dist_count_oris_ses,1)
bin_dist_mean_oris = np.nanmean(bin_dist_mean_oris_ses,1)
bin_dist_posf_oris = np.nanmean(bin_dist_posf_oris_ses,1)
bin_dist_negf_oris = np.nanmean(bin_dist_negf_oris_ses,1)

bin_angle_cent_count_oris = np.nanmean(bin_angle_cent_count_oris_ses,1)
bin_angle_cent_mean_oris = np.nanmean(bin_angle_cent_mean_oris_ses,1)
bin_angle_cent_posf_oris = np.nanmean(bin_angle_cent_posf_oris_ses,1)    
bin_angle_cent_negf_oris = np.nanmean(bin_angle_cent_negf_oris_ses,1)

bin_angle_surr_count_oris = np.nanmean(bin_angle_surr_count_oris_ses,1)
bin_angle_surr_mean_oris = np.nanmean(bin_angle_surr_mean_oris_ses,1)
bin_angle_surr_posf_oris = np.nanmean(bin_angle_surr_posf_oris_ses,1)    
bin_angle_surr_negf_oris = np.nanmean(bin_angle_surr_negf_oris_ses,1)

#%% Compute collinear selectivity index:
csi_cent_mean_oris =  collinear_selectivity_index(bin_angle_cent_mean_oris,bincenters_angle)
csi_surr_mean_oris =  collinear_selectivity_index(bin_angle_surr_mean_oris,bincenters_angle)    

csi_cent_posf_oris =  collinear_selectivity_index(bin_angle_cent_posf_oris,bincenters_angle)
csi_surr_posf_oris =  collinear_selectivity_index(bin_angle_surr_posf_oris,bincenters_angle)

csi_cent_negf_oris =  collinear_selectivity_index(bin_angle_cent_negf_oris,bincenters_angle)
csi_surr_negf_oris =  collinear_selectivity_index(bin_angle_surr_negf_oris,bincenters_angle)

#%% 
gaussian_sigma = 1

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_mean_oris,bin_2d_count_oris,bincenters_2d,centeroris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=0,gaussian_sigma=gaussian_sigma)
fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_perOri_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_mean_oris,bin_angle_cent_count_oris,bincenters_angle,
            centeroris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_perOri_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_mean_oris,bin_angle_surr_count_oris,bincenters_angle,
            centeroris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_perOri_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps per delta ori for the fraction positive
fig = plot_2D_mean_corr_dori(bin_2d_posf_oris,bin_2d_count_oris,bincenters_2d,centeroris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma)
# fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_perOri_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_posf_oris,bin_angle_cent_count_oris,bincenters_angle,
            centeroris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_perOri_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_posf_oris,bin_angle_surr_count_oris,bincenters_angle,
            centeroris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_perOri_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps per delta ori for the fraction negative:
fig = plot_2D_mean_corr_dori(bin_2d_negf_oris,bin_2d_count_oris,bincenters_2d,centeroris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma)
# fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_perOri_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_negf_oris,bin_angle_cent_count_oris,bincenters_angle,
            centeroris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_perOri_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_negf_oris,bin_angle_surr_count_oris,bincenters_angle,
            centeroris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_perOri_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Plot the CSI values as function of delta ori for the three different areapairs
fig = plot_csi_deltaori_areas(csi_cent_mean_oris,csi_cent_posf_oris,csi_cent_negf_oris,centeroris,areapairs)
# fig.savefig(os.path.join(savedir,'Collinear_CSI_Cent_perOri_%s' % (corr_type) + '.png'), format = 'png')

fig = plot_csi_deltaori_areas(csi_surr_mean_oris,csi_surr_posf_oris,csi_surr_negf_oris,centeroris,areapairs)
# fig.savefig(os.path.join(savedir,'Collinear_CSI_Surr_areas_%s' % (corr_type) + '.png'), format = 'png')



















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
    # sessions[ises].delta_pref = np.abs(np.mod(dpref,180))
    # sessions[ises].delta_pref[dpref == 180] = 180
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)

deltaoris           = np.unique(sessions[0].delta_pref[~np.isnan(sessions[0].delta_pref)])
# deltaoris           = np.array([0,22.5])
ndeltaoris          = len(deltaoris)

rotate_prefori      = True
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'
tuned_thr           = 0.05
noise_thr           = 20
corr_thr            = 0.01

rf_type             = 'F'
corr_type           = 'noise_cov'
# corr_type           = 'noise_corr'
noise_thr           = 100
r2_thr              = 0.1
binresolution       = 10

[bincenters_2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                            r2_thr=1,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

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
    
    # [_,bin_2d_posf_oris_ses[idOri,:,:,:,:,:],_,
    #  bin_dist_posf_oris_ses[idOri,:,:,:,:],_,_,
    #  bin_angle_cent_posf_oris_ses[idOri,:,:,:,:],_,
    #  bin_angle_surr_posf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='pos',
    #                                                 areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
    #                                                 corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,centerori=centerori,
    #                                                 r2_thr=r2_thr,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

    # [_,bin_2d_negf_oris_ses[idOri,:,:,:,:,:],_,
    #  bin_dist_negf_oris_ses[idOri,:,:,:,:],_,_,
    #  bin_angle_cent_negf_oris_ses[idOri,:,:,:,:],_,
    #  bin_angle_surr_negf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='neg',
    #                                                 areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
    #                                                 corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,centerori=centerori,
    #                                                 r2_thr=r2_thr,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)


#%% Compute mean over sessions for each orientation
bin_2d_count_oris = np.nansum(bin_2d_count_oris_ses,1)
bin_2d_mean_oris = np.nanmean(bin_2d_mean_oris_ses,1)
bin_2d_posf_oris = np.nanmean(bin_2d_posf_oris_ses,1)
bin_2d_negf_oris = np.nanmean(bin_2d_negf_oris_ses,1)

bin_dist_count_oris = np.nansum(bin_dist_count_oris_ses,1)
bin_dist_mean_oris = np.nanmean(bin_dist_mean_oris_ses,1)
bin_dist_posf_oris = np.nanmean(bin_dist_posf_oris_ses,1)
bin_dist_negf_oris = np.nanmean(bin_dist_negf_oris_ses,1)

bin_angle_cent_mean_oris = np.nanmean(bin_angle_cent_mean_oris_ses,1)
bin_angle_cent_posf_oris = np.nanmean(bin_angle_cent_posf_oris_ses,1)    
bin_angle_cent_negf_oris = np.nanmean(bin_angle_cent_negf_oris_ses,1)

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

bin_angle_cent_mean_oris = nanweightedaverage(bin_angle_cent_mean_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)
bin_angle_cent_posf_oris = nanweightedaverage(bin_angle_cent_posf_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)    
bin_angle_cent_negf_oris = nanweightedaverage(bin_angle_cent_negf_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)

bin_angle_surr_mean_oris = nanweightedaverage(bin_angle_surr_mean_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)
bin_angle_surr_posf_oris = nanweightedaverage(bin_angle_surr_posf_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)    
bin_angle_surr_negf_oris = nanweightedaverage(bin_angle_surr_negf_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)


# #Do for one session to get the dimensions: (data is discarded)
# [bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
#     bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
#     bin_angle_surr_count,bincenters_angle] = bin_corr_deltarf([sessions[0]],method='mean',areapairs=areapairs,
#                                                               layerpairs=layerpairs,projpairs=projpairs,binresolution=5)

# #Init output arrays:
# bin_2d_mean_oris        = np.empty((ndeltaoris,*np.shape(bin_2d_mean)))
# bin_2d_posf_oris        = np.empty((ndeltaoris,*np.shape(bin_2d_mean)))
# bin_2d_negf_oris        = np.empty((ndeltaoris,*np.shape(bin_2d_mean)))
# bin_2d_count_oris       = np.empty((ndeltaoris,*np.shape(bin_2d_count)))

# bin_dist_mean_oris      = np.empty((ndeltaoris,*np.shape(bin_dist_mean)))
# bin_dist_posf_oris      = np.empty((ndeltaoris,*np.shape(bin_dist_mean)))
# bin_dist_negf_oris      = np.empty((ndeltaoris,*np.shape(bin_dist_mean)))
# bin_dist_count_oris     = np.empty((ndeltaoris,*np.shape(bin_dist_count)))

# bin_angle_cent_mean_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_cent_mean)))
# bin_angle_cent_posf_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_cent_mean)))
# bin_angle_cent_negf_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_cent_mean)))
# bin_angle_cent_count_oris     = np.empty((ndeltaoris,*np.shape(bin_angle_cent_count)))

# bin_angle_surr_mean_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_surr_mean)))
# bin_angle_surr_posf_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_surr_mean)))
# bin_angle_surr_negf_oris      = np.empty((ndeltaoris,*np.shape(bin_angle_surr_mean)))
# bin_angle_surr_count_oris     = np.empty((ndeltaoris,*np.shape(bin_angle_surr_count)))

# for idOri,deltaori in enumerate(deltaoris):
#     [_,bin_2d_mean_oris[idOri,:,:,:,:,:],bin_2d_count_oris[idOri,:,:,:,:,:],
#      bin_dist_mean_oris[idOri,:,:,:,:],bin_dist_count_oris[idOri,:,:,:],_,
#      bin_angle_cent_mean_oris[idOri,:,:,:,:],bin_angle_cent_count_oris[idOri,:,:,:,:],
#      bin_angle_surr_mean_oris[idOri,:,:,:,:],bin_angle_surr_count_oris[idOri,:,:,:,:],_] = bin_corr_deltarf(sessions,
#                                                     method='mean',filtersign=None,areapairs=areapairs,layerpairs=layerpairs,
#                                                     projpairs=projpairs,corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,
#                                                     deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
#     [_,bin_2d_posf_oris[idOri,:,:,:,:,:],_,
#      bin_dist_posf_oris[idOri,:,:,:,:],_,_,
#      bin_angle_cent_posf_oris[idOri,:,:,:,:],_,
#      bin_angle_surr_posf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='pos',
#                                                     areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
#                                                     corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
#                                                     deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#     [_,bin_2d_negf_oris[idOri,:,:,:,:,:],_,
#      bin_dist_negf_oris[idOri,:,:,:,:],_,_,
#      bin_angle_cent_negf_oris[idOri,:,:,:,:],_,
#      bin_angle_surr_negf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='neg',
#                                                     areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
#                                                     corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
#                                                     deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#%% Compute collinear selectivity index:
csi_cent_mean_oris =  collinear_selectivity_index(bin_angle_cent_mean_oris,bincenters_angle)
csi_surr_mean_oris =  collinear_selectivity_index(bin_angle_surr_mean_oris,bincenters_angle)    

csi_cent_posf_oris =  collinear_selectivity_index(bin_angle_cent_posf_oris,bincenters_angle)
csi_surr_posf_oris =  collinear_selectivity_index(bin_angle_surr_posf_oris,bincenters_angle)

csi_cent_negf_oris =  collinear_selectivity_index(bin_angle_cent_negf_oris,bincenters_angle)
csi_surr_negf_oris =  collinear_selectivity_index(bin_angle_surr_negf_oris,bincenters_angle)

# plt.plot(np.rad2deg(bincenters_angle),bin_angle_surr_mean_oris[1,:,0,0,0])
# plt.imshow(bin_2d_mean_oris[1,:,:,0,0,0])

#%% 
gaussian_sigma = 1
min_counts = 250

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_mean_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_mean_oris,bin_angle_cent_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_mean_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_mean_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Radial_Tuning_areas_%s_mean' % (corr_type) + '.png'), format = 'png')




#%% Show spatial maps per delta ori for the fraction positive 
fig = plot_2D_mean_corr_dori(bin_2d_posf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_red)
fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_posf_oris,bin_angle_cent_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_areas_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_posf_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_posf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Radial_Tuning_areas_%s_posf' % (corr_type) + '.png'), format = 'png')


#%% Show spatial maps per delta ori for the fraction negative 
fig = plot_2D_mean_corr_dori(bin_2d_negf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_blue)
fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_negf_oris,bin_angle_cent_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_areas_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_negf_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_negf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Collinear_Radial_Tuning_areas_%s_negf' % (corr_type) + '.png'), format = 'png')


#%% Plot the CSI values as function of delta ori for the three different areapairs
fig = plot_csi_deltaori_areas(csi_cent_mean_oris,csi_cent_posf_oris,csi_cent_negf_oris,deltaoris,areapairs)
fig.savefig(os.path.join(savedir,'Collinear_CSI_Cent_areas_%s' % (corr_type) + '.png'), format = 'png')

fig = plot_csi_deltaori_areas(csi_surr_mean_oris,csi_surr_posf_oris,csi_surr_negf_oris,deltaoris,areapairs)
fig.savefig(os.path.join(savedir,'Collinear_CSI_Surr_areas_%s' % (corr_type) + '.png'), format = 'png')

#%% 


















#%% 
       ####### ######  ###    ######  ### ######  #######  #####  ####### ### ####### #     # 
#####  #     # #     #  #     #     #  #  #     # #       #     #    #     #  #     # ##    # 
#    # #     # #     #  #     #     #  #  #     # #       #          #     #  #     # # #   # 
#    # #     # ######   #     #     #  #  ######  #####   #          #     #  #     # #  #  # 
#    # #     # #   #    #     #     #  #  #   #   #       #          #     #  #     # #   # # 
#    # #     # #    #   #     #     #  #  #    #  #       #     #    #     #  #     # #    ## 
#####  ####### #     # ###    ######  ### #     # #######  #####     #    ### ####### #     # 

#%% Loop over all delta preferred orientations and store mean correlations as well as distribution of pos and neg correlations:
areapairs           = ['V1-PM','PM-V1']
layerpairs          = ' '
projpairs           = ' '

# deltaoris           = np.array([0,22.5,45,67.5,90])
for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref-0),np.abs(dpref-360)]),axis=0)

deltaoris           = np.unique(sessions[0].delta_pref[~np.isnan(sessions[0].delta_pref)])
deltaoris           = np.array([0.,90.,180.])
ndeltaoris          = len(deltaoris)
rotate_prefori      = True
rf_type             = 'Fsmooth'
# corr_type           = 'noise_corr'
corr_type           = 'trace_corr'
tuned_thr           = 0.0
noise_thr           = 20
min_counts          = 50
corr_thr            = 0.01

#Do for one session to get the dimensions: (data is discarded)
[bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
    bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
    bin_angle_surr_count,bincenters_angle] = bin_corr_deltarf([sessions[0]],method='mean',areapairs=areapairs,
                                                              layerpairs=layerpairs,projpairs=projpairs,binresolution=5)

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
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
    [_,bin_2d_posf_oris[idOri,:,:,:,:,:],_,
     bin_dist_posf_oris[idOri,:,:,:,:],_,_,
     bin_angle_cent_posf_oris[idOri,:,:,:,:],_,
     bin_angle_surr_posf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='pos',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

    [_,bin_2d_negf_oris[idOri,:,:,:,:,:],_,
     bin_dist_negf_oris[idOri,:,:,:,:],_,_,
     bin_angle_cent_negf_oris[idOri,:,:,:,:],_,
     bin_angle_surr_negf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='neg',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)


#%% 
gaussian_sigma = 1

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_mean_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
# fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_mean_oris,bin_angle_cent_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_mean_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_mean' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps per delta ori for the fraction positive 
fig = plot_2D_mean_corr_dori(bin_2d_posf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_red)
# fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_posf_oris,bin_angle_cent_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_areas_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_posf_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_posf' % (corr_type) + '.png'), format = 'png')



#%% Show spatial maps per delta ori for the fraction negative 
fig = plot_2D_mean_corr_dori(bin_2d_negf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_blue)
# fig.savefig(os.path.join(savedir,'Collinear_DeltaRF_2D_areas_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_cent_negf_oris,bin_angle_cent_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Cent_areas_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_dori(bin_angle_surr_negf_oris,bin_angle_surr_count_oris,bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
# fig.savefig(os.path.join(savedir,'Collinear_Tuning_Surr_areas_%s_negf' % (corr_type) + '.png'), format = 'png')



#%% Compute collinear selectivity index:
print('Compute direction selectivity index')
csi_cent_mean_oris =  collinear_selectivity_index(bin_angle_cent_mean_oris,bincenters_angle)
csi_surr_mean_oris =  collinear_selectivity_index(bin_angle_surr_mean_oris,bincenters_angle)    

csi_cent_posf_oris =  collinear_selectivity_index(bin_angle_cent_posf_oris,bincenters_angle)
csi_surr_posf_oris =  collinear_selectivity_index(bin_angle_surr_posf_oris,bincenters_angle)

csi_cent_negf_oris =  collinear_selectivity_index(bin_angle_cent_negf_oris,bincenters_angle)
csi_surr_negf_oris =  collinear_selectivity_index(bin_angle_surr_negf_oris,bincenters_angle)


#%% Plot the CSI values as function of delta ori for the three different areapairs
fig = plot_csi_deltaori_areas(csi_cent_mean_oris,csi_cent_posf_oris,csi_cent_negf_oris,deltaoris,areapairs)
fig.savefig(os.path.join(savedir,'Collinear_CSI_Cent_areas_%s' % (corr_type) + '.png'), format = 'png')

fig = plot_csi_deltaori_areas(csi_surr_mean_oris,csi_surr_posf_oris,csi_surr_negf_oris,deltaoris,areapairs)
fig.savefig(os.path.join(savedir,'Collinear_CSI_Surr_areas_%s' % (corr_type) + '.png'), format = 'png')

#%% 











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
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'
tuned_thr           = 0.025
noise_thr           = 20
min_counts          = 100
# corr_thr            = 0.01
corr_thr            = 0.05 #for prctile

#Do for one session to get the dimensions: (data is discarded)
[bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
    bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
    bin_angle_surr_count,bincenters_angle] = bin_corr_deltarf([sessions[0]],method='mean',areapairs=areapairs,
                                                              layerpairs=layerpairs,projpairs=projpairs,binresolution=7.5)

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
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=7.5,rotate_prefori=rotate_prefori,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)
    
    [_,bin_2d_posf_oris[idOri,:,:,:,:,:],_,
     bin_dist_posf_oris[idOri,:,:,:,:],_,_,
     bin_angle_cent_posf_oris[idOri,:,:,:,:],_,
     bin_angle_surr_posf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='pos',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=7.5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

    [_,bin_2d_negf_oris[idOri,:,:,:,:,:],_,
     bin_dist_negf_oris[idOri,:,:,:,:],_,_,
     bin_angle_cent_negf_oris[idOri,:,:,:,:],_,
     bin_angle_surr_negf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='neg',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=7.5,rotate_prefori=rotate_prefori,corr_thr=corr_thr,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#%% 
# corr_type           = 'noise_corr_GM'

#%% Compute collinear selectivity index:
csi_cent_mean_oris =  collinear_selectivity_index(bin_angle_cent_mean_oris,bincenters_angle)
csi_surr_mean_oris =  collinear_selectivity_index(bin_angle_surr_mean_oris,bincenters_angle)    

csi_cent_posf_oris =  collinear_selectivity_index(bin_angle_cent_posf_oris,bincenters_angle)
csi_surr_posf_oris =  collinear_selectivity_index(bin_angle_surr_posf_oris,bincenters_angle)

csi_cent_negf_oris =  collinear_selectivity_index(bin_angle_cent_negf_oris,bincenters_angle)
csi_surr_negf_oris =  collinear_selectivity_index(bin_angle_surr_negf_oris,bincenters_angle)

#%% 
gaussian_sigma = 1

#%% Show spatial maps per delta ori for the mean correlation
min_counts=10
iap = 0
# iap = 1

fig = plot_2D_mean_corr_projs_dori(bin_2d_mean_oris[:,:,:,[iap],:,:],bin_2d_count_oris[:,:,:,[iap],:,:],bincenters_2d,deltaoris,
                                   areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,cmap='magma',
                                   centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_DeltaRF_2D_projs_%s_%s_mean' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_projs_dori(bin_angle_cent_mean_oris[:,:,[iap],:,:],bin_angle_cent_count_oris[:,:,[iap],:,:],bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_Tuning_Cent_projs_%s_%s_mean' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_projs_dori(bin_angle_surr_mean_oris[:,:,[iap],:,:],bin_angle_surr_count_oris[:,:,[iap],:,:],bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_Tuning_Surr_projs_%s_%s_mean' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_projs_dori(bincenters_dist,bin_dist_count_oris,bin_dist_mean_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Projs','Collinear_Radial_Tuning_projs_%s_mean' % (corr_type) + '.png'), format = 'png')


#%% Show spatial maps per delta ori for the fraction positive 
fig = plot_2D_mean_corr_projs_dori(bin_2d_posf_oris[:,:,:,[iap],:,:],bin_2d_count_oris[:,:,:,[iap],:,:],bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_red)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_DeltaRF_2D_projs_%s_%s_posf' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_projs_dori(bin_angle_cent_posf_oris[:,:,[iap],:,:],bin_angle_cent_count_oris[:,:,[iap],:,:],bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_Tuning_Cent_projs_%s_%s_posf' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_projs_dori(bin_angle_surr_posf_oris[:,:,[iap],:,:],bin_angle_surr_count_oris[:,:,[iap],:,:],bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_Tuning_Surr_projs_%s_%s_posf' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_projs_dori(bincenters_dist,bin_dist_count_oris,bin_dist_posf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Projs','Collinear_Radial_Tuning_projs_%s_posf' % (corr_type) + '.png'), format = 'png')

#%% Show spatial maps per delta ori for the fraction negative 
fig = plot_2D_mean_corr_projs_dori(bin_2d_negf_oris[:,:,:,[iap],:,:],bin_2d_count_oris[:,:,:,[iap],:,:],bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap=cm_blue)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_DeltaRF_2D_projs_%s_%s_negf' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show angular tuning of center area (matched RF) for each delta ori:
fig = plot_corr_angular_tuning_projs_dori(bin_angle_cent_negf_oris[:,:,[iap],:,:],bin_angle_cent_count_oris[:,:,[iap],:,:],bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_Tuning_Cent_projs_%s_%s_negf' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show angular tuning of surround area (mismatched RF) for each delta ori:
fig = plot_corr_angular_tuning_projs_dori(bin_angle_surr_negf_oris[:,:,[iap],:,:],bin_angle_surr_count_oris[:,:,[iap],:,:],bincenters_angle,
            deltaoris,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.suptitle(areapairs[iap])
fig.savefig(os.path.join(savedir,'Projs','Collinear_Tuning_Surr_projs_%s_%s_negf' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_projs_dori(bincenters_dist,bin_dist_count_oris,bin_dist_negf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)
fig.savefig(os.path.join(savedir,'Projs','Collinear_Radial_Tuning_projs_%s_negf' % (corr_type) + '.png'), format = 'png')

#%% Plot the CSI values as function of delta ori for the three different areapairs
fig = plot_csi_deltaori_projs(csi_cent_mean_oris[:,[iap],:,:],csi_cent_posf_oris[:,[iap],:,:],csi_cent_negf_oris[:,[iap],:,:],deltaoris,projpairs)
fig.savefig(os.path.join(savedir,'Projs','Collinear_CSI_Cent_projs_%s_%s' % (corr_type,areapairs[iap]) + '.png'), format = 'png')

fig = plot_csi_deltaori_projs(csi_surr_mean_oris[:,[iap],:,:],csi_surr_posf_oris[:,[iap],:,:],csi_surr_negf_oris[:,[iap],:,:],deltaoris,projpairs)
fig.savefig(os.path.join(savedir,'Projs','Collinear_CSI_Surr_projs_%s_%s' % (corr_type,areapairs[iap]) + '.png'), format = 'png')


#%% 
fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(10,5))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(areapairs),len(projpairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax = axes[iap,ipp]
            ax.imshow(np.log10(bin_2d_count[:,:,iap,ilp,ipp]),vmax=np.nanpercentile(np.log10(bin_2d_count),99.9),
                cmap="hot",interpolation="none",extent=np.flipud(bincenters_2d).flatten())
            # ax.imshow(binmean[:,:,iap,ilp,ipp],vmin=np.nanpercentile(binmean[:,:,iap,ilp,ipp],5),
            #                     vmax=np.nanpercentile(binmean[:,:,iap,ilp,ipp],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
            ax.set_title('%s\n%s' % (areapair, layerpair))
            # ax.set_xlim([-75,75])
            # ax.set_ylim([-75,75])
plt.tight_layout()

#%% 

celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

fig,ax = plt.subplots(figsize=(4,4))
ax.hist(celldata['DSI'],bins=50, histtype='step', color='k', density=True)
for perc in [10,25,50,75,90]:
    ax.axvline(np.nanpercentile(celldata['DSI'],perc),color='k',linestyle='--',linewidth=0.5)
    ax.text(np.nanpercentile(celldata['DSI'],perc),ax.get_ylim()[1]*0.95,u'%dth' % perc,ha='center',va='top',fontsize=7)
ax.set_xlabel('DSI')
ax.set_ylabel('% of data')
ax.set_xlim([0,1])

# #%% #########################################################################################
# # Contrasts: across areas and projection identity      

# [noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions_subset,corr_type='trace_corr',
#                                                                 binresolution=10,rotate_prefori=False,thr_tuned=0.0,
#                                                                 thr_rf_p=0.001,rf_type='F')

# min_counts = 50
# noiseRFmat_mean[countsRFmat<min_counts] = np.nan

# fig,axes = plt.subplots(4,4,figsize=(10,7))
# for i in range(4):
#     for j in range(4):
#         axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
#                          vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
#         axes[i,j].set_title(legendlabels[i,j])
# plt.tight_layout()
# plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_AllProt_%dsessions' %nSessions  + '.png'), format = 'png')
# # plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_GN_F_%dsessions' %nSessions  + '.png'), format = 'png')

# fig,axes = plt.subplots(4,4,figsize=(10,7))
# for i in range(4):
#     for j in range(4):
#         axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
#         axes[i,j].set_title(legendlabels[i,j])
# plt.tight_layout()
# plt.savefig(os.path.join(savedir,'2D_NC_Map_smooth_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')


5.35525

#%% 
####### #     # #     # ### #     #  #####     ####### #     # ######  #######  #####  #     # 
   #    #     # ##    #  #  ##    # #     #       #    #     # #     # #       #     # #     # 
   #    #     # # #   #  #  # #   # #             #    #     # #     # #       #       #     # 
   #    #     # #  #  #  #  #  #  # #  ####       #    ####### ######  #####    #####  ####### 
   #    #     # #   # #  #  #   # # #     #       #    #     # #   #   #             # #     # 
   #    #     # #    ##  #  #    ## #     #       #    #     # #    #  #       #     # #     # 
   #     #####  #     # ### #     #  #####        #    #     # #     # #######  #####  #     # 

#%% What is the optimal tuning threshold:
# Loop over a few tuning thresholds and show spatial selectivity to see how effect depends on level of tuning:
# areapairs           = ['V1-V1','PM-PM','V1-PM']
areapairs           = ['V1-PM','PM-V1']
layerpairs          = ' '
projpairs           = ' '

for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    sessions[ises].delta_pref = np.abs(np.mod(dpref,180))
    sessions[ises].delta_pref[dpref == 180] = 180

deltaori            = 0
rotate_prefori      = True
rf_type             = 'Fsmooth'
corr_type           = 'trace_corr'
noise_thr           = 20
min_counts          = 50

#Do for one session to get the dimensions: (data is discarded)
[bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
    bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
    bin_angle_surr_count,bincenters_angle] = bin_corr_deltarf([sessions[0]],method='mean',areapairs=areapairs,
                                                              layerpairs=layerpairs,projpairs=projpairs,binresolution=5)

tuned_thrs = [0,0.01,0.02,0.03,0.05,0.1]
ntuned_thr  = len(tuned_thrs)

#Init output arrays:
bin_2d_mean_thrs        = np.empty((ntuned_thr,*np.shape(bin_2d_mean)))
bin_2d_count_thrs        = np.empty((ntuned_thr,*np.shape(bin_2d_mean)))

bin_dist_mean_thrs      = np.empty((ntuned_thr,*np.shape(bin_dist_mean)))
bin_dist_count_thrs     = np.empty((ntuned_thr,*np.shape(bin_dist_count)))

bin_angle_cent_mean_thrs      = np.empty((ntuned_thr,*np.shape(bin_angle_cent_mean)))
bin_angle_cent_count_thrs     = np.empty((ntuned_thr,*np.shape(bin_angle_cent_count)))

bin_angle_surr_mean_thrs      = np.empty((ntuned_thr,*np.shape(bin_angle_surr_mean)))
bin_angle_surr_count_thrs     = np.empty((ntuned_thr,*np.shape(bin_angle_surr_count)))

for ithr,tuned_thr in enumerate(tuned_thrs):
    [_,bin_2d_mean_thrs[ithr,:,:,:,:,:],bin_2d_count_thrs[ithr,:,:,:,:,:],
     bin_dist_mean_thrs[ithr,:,:,:,:],bin_dist_count_thrs[ithr,:,:,:],_,
     bin_angle_cent_mean_thrs[ithr,:,:,:,:],bin_angle_cent_count_thrs[ithr,:,:,:,:],
     bin_angle_surr_mean_thrs[ithr,:,:,:,:],bin_angle_surr_count_thrs[ithr,:,:,:,:],_] = bin_corr_deltarf(sessions,
                                                    method='mean',filtersign=None,areapairs=areapairs,layerpairs=layerpairs,
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)


#%% Compute collinear selectivity index:
csi_cent_mean_thrs =  collinear_selectivity_index(bin_angle_cent_mean_thrs,bincenters_angle)
csi_surr_mean_thrs =  collinear_selectivity_index(bin_angle_surr_mean_thrs,bincenters_angle)    

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_mean_thrs,bin_2d_count_thrs,bincenters_2d,tuned_thrs,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=1)
fig.savefig(os.path.join(savedir,'Tuning_Threshold_%ddeg_%s_mean' % (deltaori,corr_type) + '.png'), format = 'png')

#%% 
plt.figure(figsize=(4,3))
for ithr,tuned_thr in enumerate(tuned_thrs):
    plt.plot(bincenters_angle,bin_angle_surr_mean_thrs[ithr,:,1,0,0],label=tuned_thr)

plt.legend(frameon=False)
plt.tight_layout()
plt.xticks(bincenters_angle[::2],np.rad2deg(bincenters_angle[::2]))
plt.xlabel('Angle (deg)')
plt.ylabel('Mean Correlation, surround')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Tuning_Threshold_%ddeg_%s_mean' % (deltaori,corr_type) + '.png'), format = 'png')

#%% 
plt.figure(figsize=(4,3))
plt.plot(tuned_thrs,csi_surr_mean_thrs[:,0,0,0])
plt.xticks(tuned_thrs)
plt.xlabel('Tuning Threshold (tuning variance)')
plt.ylabel('CSI surround (45 deg delta ori)')
plt.ylim([-1,1])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'CSI_Tuning_Threshold_%ddeg_%s' % (deltaori,corr_type) + '.png'), format = 'png')

#%% 
####### ### #     # #######    ######  ### #     # #     # ### #     #  #####  
   #     #  ##   ## #          #     #  #  ##    # ##    #  #  ##    # #     # 
   #     #  # # # # #          #     #  #  # #   # # #   #  #  # #   # #       
   #     #  #  #  # #####      ######   #  #  #  # #  #  #  #  #  #  # #  #### 
   #     #  #     # #          #     #  #  #   # # #   # #  #  #   # # #     # 
   #     #  #     # #          #     #  #  #    ## #    ##  #  #    ## #     # 
   #    ### #     # #######    ######  ### #     # #     # ### #     #  #####  

# #%%  Load data fully:      
# for ises in range(nSessions):
#     sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion=calciumversion,filter_hp=0.01)

#%% What is the effect of temporal binning on the orthogonal correlational structure?
# Loop over a few time bin widths + show spatial selectivity to see how effect depends
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ' '
projpairs           = ' '

for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    sessions[ises].delta_pref = np.abs(np.mod(dpref,180))
    sessions[ises].delta_pref[dpref == 180] = 180

deltaori            = 0
rotate_prefori      = True
rf_type             = 'Fsmooth'
corr_type           = 'trace_corr'
noise_thr           = 20
centerthr           = [15,15,15,15]
min_counts          = 50
tuned_thr           = 0

#Do for one session to get the dimensions: (data is discarded)
[bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
    bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
    bin_angle_surr_count,bincenters_angle] = bin_corr_deltarf([sessions[0]],method='mean',areapairs=areapairs,
                                                              layerpairs=layerpairs,projpairs=projpairs,binresolution=5)

binwidths = np.array([1,2,3,5,10])*(1/sessions[0].sessiondata['fs'][0]).round(2)
ntimebins  = len(binwidths)

#Init output arrays:
bin_2d_mean_thrs        = np.empty((ntimebins,*np.shape(bin_2d_mean)))
bin_2d_count_thrs        = np.empty((ntimebins,*np.shape(bin_2d_mean)))

bin_dist_mean_thrs      = np.empty((ntimebins,*np.shape(bin_dist_mean)))
bin_dist_count_thrs     = np.empty((ntimebins,*np.shape(bin_dist_count)))

bin_angle_cent_mean_thrs      = np.empty((ntimebins,*np.shape(bin_angle_cent_mean)))
bin_angle_cent_count_thrs     = np.empty((ntimebins,*np.shape(bin_angle_cent_count)))

bin_angle_surr_mean_thrs      = np.empty((ntimebins,*np.shape(bin_angle_surr_mean)))
bin_angle_surr_count_thrs     = np.empty((ntimebins,*np.shape(bin_angle_surr_count)))

for ibin,binwidth in enumerate(binwidths):

    sessions = compute_trace_correlation(sessions,binwidth=binwidth,uppertriangular=False)

    [_,bin_2d_mean_thrs[ibin,:,:,:,:,:],bin_2d_count_thrs[ibin,:,:,:,:,:],
     bin_dist_mean_thrs[ibin,:,:,:,:],bin_dist_count_thrs[ibin,:,:,:],_,
     bin_angle_cent_mean_thrs[ibin,:,:,:,:],bin_angle_cent_count_thrs[ibin,:,:,:,:],
     bin_angle_surr_mean_thrs[ibin,:,:,:,:],bin_angle_surr_count_thrs[ibin,:,:,:,:],_] = bin_corr_deltarf(sessions,
                                                    method='mean',filtersign=None,areapairs=areapairs,layerpairs=layerpairs,
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#%% Compute collinear selectivity index:
csi_cent_mean_thrs =  collinear_selectivity_index(bin_angle_cent_mean_thrs,bincenters_angle)
csi_surr_mean_thrs =  collinear_selectivity_index(bin_angle_surr_mean_thrs,bincenters_angle)   

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_mean_thrs,bin_2d_count_thrs,bincenters_2d,binwidths,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=1)
fig.savefig(os.path.join(savedir,'Time_Bins_%ddeg_%s_mean' % (deltaori,corr_type) + '.png'), format = 'png')

#%% 
plt.figure(figsize=(4,3))
for ithr,tuned_thr in enumerate(tuned_thrs):
    plt.plot(bincenters_angle,bin_angle_surr_mean_thrs[ithr,:,1,0,0],label=tuned_thr)

plt.legend(frameon=False)
plt.tight_layout()
plt.xticks(bincenters_angle[::2],np.rad2deg(bincenters_angle[::2]))
plt.xlabel('Angle (deg)')
plt.ylabel('Mean Correlation, surround')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Tuning_Threshold_%ddeg_%s_mean' % (deltaori,corr_type) + '.png'), format = 'png')

#%% 
plt.figure(figsize=(4,3))
plt.plot(tuned_thrs,csi_surr_mean_thrs[:,0,0,0])
plt.xticks(tuned_thrs)
plt.xlabel('Tuning Threshold (tuning variance)')
plt.ylabel('CSI surround (45 deg delta ori)')
plt.ylim([-1,1])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'CSI_Tuning_Threshold_%ddeg_%s' % (deltaori,corr_type) + '.png'), format = 'png')



