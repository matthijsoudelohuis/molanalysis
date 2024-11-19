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

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

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
calciumversion = 'deconv'
                
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=True,filter_hp=0.01)
                                # calciumversion=calciumversion,keepraw=True)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=True,remove_method='PCA',remove_rank=1)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ##################### Compute pairwise receptive field distances: ##############################
sessions = smooth_rf(sessions,radius=50,rf_type='Fneu',mincellsFneu=5)
sessions = exclude_outlier_rf(sessions) 
sessions = replace_smooth_with_Fsig(sessions) 

#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

# areapairs           = ['V1-V1']
# layerpairs          = ['L2/3-L2/3']
# layerpairs          = ['L2/3-L5']
projpairs           = ['unl-unl']

#If you override any of these then these pairs will be ignored:
layerpairs          = ' '
# areapairs           = ' '
# projpairs           = ' '

deltaori            = 0
# deltaori            = [-35,360]
# deltaori            = [80,100]
# deltaori            = None
rotate_prefori      = True
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

# [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
#                             corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,deltaori=deltaori,rf_type=rf_type,
#                             sig_thr = 0.001,noise_thr=0.2,tuned_thr=0.02,dsi_thr=0,normalize=False,filtersign='pos',absolute=False)

[bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
bin_angle_surr_count,bincenters_angle] = bin_corr_deltarf(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,deltaori=deltaori,rf_type=rf_type,
                            r2_thr=0.2,noise_thr=20,tuned_thr=0.05,dsi_thr=0,filtersign=None,absolute=True)

#%% Make the figure:
centerthr           = [15,15,15]
min_counts          = 50

#%% 

fig = plot_2D_mean_corr(bin_2d_mean,bin_2d_count,bincenters_2d,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=0.8)


#%% 

plt.plot(bincenters_dist,bin_dist_mean[:,0,0,0])
fig = plot_1D_mean_corr(bin_dist_mean,bin_dist_count,bincenters_dist,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)


#%% 
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Collinear' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Collinear_labeled' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Orthogonal' % (corr_type) + '.png'), format = 'png')

#%% 
plt.plot(bincenters_angle,bin_angle_surr_mean[:,0,0,0])

fig = plot_corr_surr_tuning(sessions,bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
            bin_angle_surr_count,bincenters_angle,areapairs=areapairs,layerpairs=layerpairs,
            projpairs=projpairs,corr_type=corr_type)

# fig.savefig(os.path.join(savedir,'DeltaRF_1D_Polar_%s_GR_Collinear_labeled' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_1D_Polar_%s_GR_Collinear' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Orthogonal' % (corr_type) + '.png'), format = 'png')

#%% Compute collinear selectivity index:
surr_CSI = np.empty((bin_angle_surr_count.shape[1:]))
cent_CSI = np.empty((bin_angle_cent_count.shape[1:]))

surr_PSI = np.empty((bin_angle_surr_count.shape[1:]))
cent_PSI = np.empty((bin_angle_cent_count.shape[1:]))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            resp_surr_col    = np.mean(bin_angle_surr_mean[np.isin(bincenters_angle,[0,np.pi]),iap,ilp,ipp])
            resp_surr_perp   = np.mean(bin_angle_surr_mean[np.isin(bincenters_angle,[np.pi/2,1.5*np.pi]),iap,ilp,ipp])
            surr_CSI[iap,ilp,ipp] = (resp_surr_col - resp_surr_perp) / resp_surr_col

            resp_cent_col    = np.mean(bin_angle_cent_mean[np.isin(bincenters_angle,[0,np.pi]),iap,ilp,ipp])
            resp_cent_perp   = np.mean(bin_angle_cent_mean[np.isin(bincenters_angle,[np.pi/2,1.5*np.pi]),iap,ilp,ipp])
            cent_CSI[iap,ilp,ipp] = (resp_cent_col - resp_cent_perp) / resp_cent_col

            resp_surr_perp_pref = np.mean(bin_angle_surr_mean[np.isin(bincenters_angle,[np.pi/2]),iap,ilp,ipp])
            resp_surr_perp_orth = np.mean(bin_angle_surr_mean[np.isin(bincenters_angle,[1.5*np.pi]),iap,ilp,ipp])
            surr_PSI[iap,ilp,ipp] = (resp_surr_perp_pref - resp_surr_perp_orth) / resp_surr_perp_pref

            resp_cent_perp_pref  = np.mean(bin_angle_cent_mean[np.isin(bincenters_angle,[np.pi/2]),iap,ilp,ipp])
            resp_cent_perp_orth  = np.mean(bin_angle_cent_mean[np.isin(bincenters_angle,[1.5*np.pi]),iap,ilp,ipp])
            cent_PSI[iap,ilp,ipp] = (resp_cent_perp_pref - resp_cent_perp_orth) / resp_cent_perp_pref


# #%% ########################### Compute tuning metrics on angular bin data: ##################################
# surround_OSI = np.empty(np.shape(bin_angle_surr_mean)[1:])
# surround_DSI = np.empty(np.shape(bin_angle_surr_mean)[1:])
# center_OSI = np.empty(np.shape(bin_angle_cent_mean)[1:])
# center_DSI = np.empty(np.shape(bin_angle_cent_mean)[1:])

# for x in range(np.shape(bin_angle_surr_mean)[1]):
#     for y in range(np.shape(bin_angle_surr_mean)[2]):
#         for z in range(np.shape(bin_angle_surr_mean)[3]):
#             surround_OSI[x,y,z] = compute_tuning(bin_angle_surr_mean[:,x,y,z][np.newaxis,:],bincenters_angle,tuning_metric='OSI')[0]
#             surround_DSI[x,y,z] = compute_tuning(bin_angle_surr_mean[:,x,y,z][np.newaxis,:],bincenters_angle,tuning_metric='DSI')[0]
#             center_OSI[x,y,z] = compute_tuning(bin_angle_cent_mean[:,x,y,z][np.newaxis,:],bincenters_angle,tuning_metric='OSI')[0]
#             center_DSI[x,y,z] = compute_tuning(bin_angle_cent_mean[:,x,y,z][np.newaxis,:],bincenters_angle,tuning_metric='DSI')[0]

#%% Loop over all delta preferred orientations and store mean correlations as well as distribution of pos and neg correlations:
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ' '
projpairs           = ' '

deltaoris           = np.unique(sessions[0].trialdata['Orientation'])
ndeltaoris          = len(deltaoris)
rotate_prefori      = True
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

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
                                                    deltaori=deltaori,rf_type=rf_type,noise_thr=20,tuned_thr=0.02)
    
    [_,bin_2d_posf_oris[idOri,:,:,:,:,:],_,
     bin_dist_posf_oris[idOri,:,:,:,:],_,_,
     bin_angle_cent_posf_oris[idOri,:,:,:,:],_,
     bin_angle_surr_posf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='pos',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,deltaori=deltaori,
                                                    rf_type=rf_type,noise_thr=20,tuned_thr=0.02)

    [_,bin_2d_posf_oris[idOri,:,:,:,:,:],_,
     bin_dist_posf_oris[idOri,:,:,:,:],_,_,
     bin_angle_cent_posf_oris[idOri,:,:,:,:],_,
     bin_angle_surr_posf_oris[idOri,:,:,:,:],_,_] = bin_corr_deltarf(sessions,method='frac',filtersign='neg',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,deltaori=deltaori,
                                                    rf_type=rf_type,noise_thr=20,tuned_thr=0.02)
#%% Compute CSI and PSI for each orientation:
#make a plot of the mean, pos and neg for each delta ori and for each areapair
# make a tuning plot of the surround for the mean, pos and neg for each delta ori and for each areapair
# make a collinear tuning index tuning for each delta ori for pos, neg and mean
# Now make the plot for labeled and unlabeled cells for V1-PM pair. 



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

#%% #########################################################################################
# Contrasts: across areas and projection identity      

[noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions_subset,corr_type='trace_corr',
                                                                binresolution=10,rotate_prefori=False,thr_tuned=0.0,
                                                                thr_rf_p=0.001,rf_type='F')

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
# plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_GN_F_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_Map_smooth_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')




