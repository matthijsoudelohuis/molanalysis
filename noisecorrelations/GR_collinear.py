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
from utils.tuning import compute_tuning, compute_prefori

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM'],session_rf=True) 

# #%% Remove two sessions with too much drift in them:
# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
# sessions            = [sessions[i] for i in sessions_in_list]
# nSessions           = len(sessions)

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='dF',keepraw=True)
                                calciumversion='dF',keepraw=True,filter_hp=0.01)

                                # calciumversion='deconv',keepraw=True)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='OSI')
    sessions[ises].celldata['DSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='DSI')
    sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='gOSI')
    sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='tuning_var')
    sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'])

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=False)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=True,remove_method='PCA',remove_rank=1)

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

#%% Give redcells a string label
redcelllabels = np.array(['unl','lab'])
for ses in sessions:
    ses.celldata['labeled'] = ses.celldata['redcell']
    ses.celldata['labeled'] = ses.celldata['labeled'].astype(int).apply(lambda x: redcelllabels[x])

#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

# sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list = np.where(sessiondata['protocol'].isin(['GR','GN','IM']))[0]
# sessions_subset = [sessions[i] for i in sessions_in_list]

# # #%% Remove two sessions with too much drift in them:
# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
# sessions_subset            = [sessions[i] for i in sessions_in_list]

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

deltaori            = [-15,15]
# deltaori            = [-35,360]
# deltaori            = [80,100]
# deltaori            = None
rotate_prefori      = True
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

[binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=5,rotate_prefori=rotate_prefori,deltaori=deltaori,rf_type=rf_type,
                            sig_thr = 0.001,noise_thr=0.2,tuned_thr=0.02,dsi_thr=0,normalize=False,filtersign='pos',absolute=False)

#%% Definitions of azimuth, elevation and delta RF 2D space:
delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
deltarf             = np.sqrt(delta_az**2 + delta_el**2)
anglerf             = np.mod(np.arctan2(delta_az,delta_el)+np.pi/2,np.pi*2)

#%% Make the figure:
centerthr           = [15,15,15]
min_counts          = 50

#%% 
deglim              = 60
gaussian_sigma      = 0.8
clrs_areapairs      = get_clr_area_pairs(areapairs)

fig,axes    = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax                                              = axes[ipp,iap]
            data                                            = copy.deepcopy(binmean[:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bincounts[:,:,iap,ilp,ipp]<min_counts]     = np.nan

            # ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,10),vmax=np.nanpercentile(data,95),cmap="crest")
            # ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,10),vmax=np.nanpercentile(data,95),cmap="Reds_r")
            ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,10),vmax=np.nanpercentile(data,95),cmap="hot")
            ax.set_facecolor('grey')
            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([-deglim,deglim])
            ax.set_ylim([-deglim,deglim])
            ax.set_xlabel(u'Δ deg Collinear')
            ax.set_ylabel(u'Δ deg Orthogonal')
            circle=plt.Circle((0,0),centerthr[iap], color='g', fill=False,linestyle='--',linewidth=1)
            ax.add_patch(circle)

plt.tight_layout()
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Collinear' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Collinear_labeled' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Orthogonal' % (corr_type) + '.png'), format = 'png')

#%% Average correlation values based on circular tuning:
polarbinres         = 45
# polarbinedges       = np.deg2rad(np.arange(0,360,step=polarbinres))
polarbinedges       = np.deg2rad(np.arange(0-polarbinres/2,360,step=polarbinres))
polarbincenters     = polarbinedges[:-1]+np.deg2rad(polarbinres/2)

anglerf             = np.mod(anglerf+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)

polardata           = np.empty((len(polarbincenters),2,*np.shape(binmean)[2:]))
polardata_counts    = np.zeros((len(polarbincenters),2,*np.shape(binmean)[2:]))
for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            data = binmean[:,:,iap,ilp,ipp].copy()
            data[deltarf>centerthr[iap]] = np.nan
            data[bincounts[:,:,iap,ilp,ipp]<min_counts]     = np.nan
            if np.any(~np.isnan(data)):
                polardata[:,0,iap,ilp,ipp] = binned_statistic(x=anglerf[~np.isnan(data)],
                                    values=data[~np.isnan(data)],
                                    statistic='mean',bins=polarbinedges)[0]
            
            data = bincounts[:,:,iap,ilp,ipp].copy()
            data[deltarf>centerthr[iap]] = 0
            polardata_counts[:,0,iap,ilp,ipp] = binned_statistic(x=anglerf[~np.isnan(data)],
                                    values=data[~np.isnan(data)],
                                    statistic='sum',bins=polarbinedges)[0]
            
            data = binmean[:,:,iap,ilp,ipp].copy()
            data[deltarf<=centerthr[iap]] = np.nan
            data[bincounts[:,:,iap,ilp,ipp]<min_counts]     = np.nan
            if np.any(~np.isnan(data)):
                polardata[:,1,iap,ilp,ipp]  = binned_statistic(x=anglerf[~np.isnan(data)],
                                        values=data[~np.isnan(data)],
                                        statistic='mean',bins=polarbinedges)[0]

            data = bincounts[:,:,iap,ilp,ipp].copy()
            data[deltarf<=centerthr[iap]] = 0
            polardata_counts[:,1,iap,ilp,ipp] = binned_statistic(x=anglerf[~np.isnan(data)],
                                    values=data[~np.isnan(data)],
                                    statistic='sum',bins=polarbinedges)[0]

# polardata_err = np.full(polardata.shape,np.nanstd(getattr(sessions[0],corr_type))) / np.sqrt(polardata_counts)
polardata_err = np.full(polardata.shape,np.nanstd(getattr(sessions[0],corr_type))) / polardata_counts**0.3

# Make the figure:
deglim      = 2*np.pi
fig,axes    = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            handles = []
            ax                                          = axes[ipp,iap]
            # ax.plot(polarbincenters,polardata[:,0,iap,ilp,ipp],color='k',label='center')
            # ax.plot(polarbincenters,polardata[:,1,iap,ilp,ipp],color='g',label='surround')
            handles.append(shaded_error(ax,x=polarbincenters,y=polardata[:,0,iap,ilp,ipp],yerror=polardata_err[:,0,iap,ilp,ipp],color='k'))
            handles.append(shaded_error(ax,x=polarbincenters,y=polardata[:,1,iap,ilp,ipp],yerror=polardata_err[:,1,iap,ilp,ipp],color='g'))

            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([0,deglim])
            # ax.set_ylim([0.04,0.1])
            ax.set_xticks(np.arange(0,2*np.pi,step = np.deg2rad(45)),labels=np.arange(0,360,step = 45))
            ax.set_xlabel(u'Angle (deg)')
            ax.set_ylabel(u'Correlation')
            ax.legend(handles=handles,labels=['Center','Surround'],frameon=False,fontsize=8,loc='upper right')

plt.tight_layout()
# fig.savefig(os.path.join(savedir,'DeltaRF_1D_Polar_%s_GR_Collinear_labeled' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_1D_Polar_%s_GR_Collinear' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Orthogonal' % (corr_type) + '.png'), format = 'png')

#%% 
fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(10,5))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(areapairs),len(projpairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax = axes[iap,ipp]
            ax.imshow(np.log10(bincounts[:,:,iap,ilp,ipp]),vmax=np.nanpercentile(np.log10(bincounts),99.9),
                cmap="hot",interpolation="none",extent=np.flipud(bincenters).flatten())
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




