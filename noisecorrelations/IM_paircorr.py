# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import preprocessing
from loaddata.session_info import filter_sessions,load_sessions
from scipy.signal import medfilt
from utils.plotting_style import * #get all the fixed color schemes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.stats import binned_statistic

from utils.imagelib import load_natural_images #
from utils.explorefigs import *
from utils.psth import compute_tensor,compute_respmat,construct_behav_matrix_ts_F
from loaddata.get_data_folder import get_local_drive
from utils.corr_lib import mean_resp_image,compute_signal_correlation, compute_pairwise_metrics
from utils.plot_lib import shaded_error
from utils.RRRlib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\')


#################################################
# session_list        = np.array([['LPE09665','2023_03_15']])
session_list        = np.array([['LPE11086','2023_12_16']])
sessions,nSessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')

#%% Load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'IM',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['IM']) 

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

# #Compute average response per trial:
# for ises in range(nSessions):
#     sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
#                                   t_resp_start=0,t_resp_stop=0.5,method='mean',subtr_baseline=False)

#     sessions[ises].respmat_runspeed = compute_respmat(sessions[ises].behaviordata['runspeed'],
#                                                         sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
#                                                         t_resp_start=0,t_resp_stop=0.5,method='mean')

#     sessions[ises].respmat_videome = compute_respmat(sessions[ises].videodata['motionenergy'],
#                                                     sessions[ises].videodata['timestamps'], sessions[ises].trialdata['tOnset'],
#                                                     t_resp_start=0,t_resp_stop=0.5,method='mean')

    # delattr(sessions[ises],'calciumdata')
    # delattr(sessions[ises],'videodata')
    # delattr(sessions[ises],'behaviordata')

#%% ### Load the natural images:
natimgdata = load_natural_images(onlyright=True)

#%% ##### Compute pairwise metrics:

sessions = compute_signal_correlation(sessions)

sessions = compute_pairwise_metrics(sessions)

#%%  construct dataframe with all pairwise measurements:
df_allpairs  = pd.DataFrame()

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    tempdf  = pd.DataFrame({'SignalCorrelation': sessions[ises].sig_corr.flatten(),
                    # 'DeltaPrefOri': sessions[ises].delta_pref.flatten(),
                    'AreaPair': sessions[ises].areamat.flatten(),
                    'DistXYPair': sessions[ises].distmat_xy.flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz.flatten(),
                    'DistRfPair': sessions[ises].distmat_rf.flatten(),
                    'LabelPair': sessions[ises].labelmat.flatten()}).dropna(how='all') 
                    #drop all rows that have all nan (diagonal + repeat below daig)
    df_allpairs  = pd.concat([df_allpairs, tempdf], ignore_index=True).reset_index(drop=True)

#%% ################## Noise correlations between labeled and unlabeled cells:  #########################
# labelpairs = df_allpairs['LabelPair'].unique()
labelpairs = ['unl-unl','unl-lab','lab-lab']
clrs_labelpairs = get_clr_labelpairs(labelpairs)

areapairs = ['V1-V1','V1-PM','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

plt.figure(figsize=(9,4))
for iap,areapair in enumerate(areapairs):
    ax = plt.subplot(1,3,iap+1)
    areafilter      = df_allpairs['AreaPair']==areapair
    # signalfilter    = np.meshgrid(sessions[ises].celldata['skew']>3,sessions[ises].celldata['skew']>3)
    # signalfilter = np.meshgrid(sessions[ises].celldata['redcell']==0,sessions[ises].celldata['redcell']==0)
    filter          = areafilter
    center          = df_allpairs[filter].groupby('LabelPair', as_index=False)['SignalCorrelation'].mean()['SignalCorrelation']
    err             = df_allpairs[filter].groupby('LabelPair', as_index=False)['SignalCorrelation'].sem()['SignalCorrelation']
    # sns.barplot(data=center,x='LabelPair',y='NoiseCorrelation')
    ax.bar(x=labelpairs,height=center,yerr=err,label=labelpairs,color=clrs_labelpairs)
    ax.set_yticks(np.arange(0, 0.1, step=0.01))
    ax.set_xticklabels(labelpairs)
    ax.set_ylim([0,0.075])
    ax.set_title(areapair)
    ax.set_ylabel('Signal Correlation')

plt.tight_layout()

#%% ############################################################################################
################### Signal correlations as a function of pairwise distance: ####################
############################# Labeled vs unlabeled neurons #######################################

areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

labelpairs = ['unl-unl','unl-lab','lab-lab']
clrs_labelpairs = get_clr_labelpairs(labelpairs)

binedges = np.arange(0,1000,50) 
nbins= len(binedges)-1      
binmean = np.empty((nSessions,len(areapairs),len(labelpairs),nbins))
handles = []
for iap,areapair in enumerate(areapairs):
    for ilp,labelpair in enumerate(labelpairs):
        for ises in range(nSessions):
            areafilter = sessions[ises].areamat==areapair
            labelfilter = sessions[ises].labelmat==labelpair
            # filter = sessions[ises].celldata['tuning_var']>0
            filter = np.logical_and(areafilter,labelfilter)
            # filter = np.logical_and(signalfilter,areafilter,labelfilter)
            if filter.any():
                binmean[ises,iap,ilp,:] = binned_statistic(x=sessions[ises].distmat_xy[filter].flatten(),
                                                values=sessions[ises].sig_corr[filter].flatten(),
                            statistic='mean', bins=binedges)[0]

plt.figure(figsize=(6,3))
for iap,areapair in enumerate(areapairs):
    ax = plt.subplot(1,len(areapairs),iap+1)
    for ilp,labelpair in enumerate(labelpairs):
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
        handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_labelpairs[ilp]))
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),
                                    # yerror=binmean[:,iap,ilp,:].squeeze()/5,color=clrs_labelpairs[ilp]))
    ax.set(xlabel=r'Anatomical distance ($\mu$m)',ylabel='Signal Correlation',
           yticks=np.arange(0, 1, step=0.01),xticks=np.arange(0, 600, step=100))
    ax.set(xlim=[10,500],ylim=[0,0.05])
    ax.legend(handles,labelpairs,frameon=False,loc='upper right')
    plt.tight_layout()
    ax.set_title(areapair)
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'SignalCorrelations','SignalCorr_anatomdistance_perArea_Labeled_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% 
areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

labelpairs = ['unl-unl','unl-lab','lab-lab']
clrs_labelpairs = get_clr_labelpairs(labelpairs)

areas               = ['V1','PM']
redcells            = [0,1]
redcelllabels       = ['unl','lab']
legendlabelsx        = np.empty((4),dtype='object')
legendlabelsy        = np.empty((4),dtype='object')

NCdata              = np.zeros((4,4,nSessions))

for ises in range(nSessions):
    for ixArea,xArea in enumerate(areas):
        for iyArea,yArea in enumerate(areas):
            for ixRed,xRed in enumerate(redcells):
                for iyRed,yRed in enumerate(redcells):
                    idx_source = np.logical_and(sessions[ises].celldata['roi_name']==xArea,sessions[ises].celldata['redcell']==xRed)
                    idx_target = np.logical_and(sessions[ises].celldata['roi_name']==yArea,sessions[ises].celldata['redcell']==yRed)

                    NCdata[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(sessions[ises].sig_corr[np.ix_(idx_source, idx_target)])
                    
                    legendlabelsx[ixArea*2 + ixRed]  = areas[ixArea] + redcelllabels[ixRed] 
                    legendlabelsy[iyArea*2 + iyRed]  = areas[iyArea] + redcelllabels[iyRed]

fig,axes = plt.subplots(1,1,figsize=(7,7))

axes.pcolormesh(np.arange(4),np.arange(4),np.nanmean(NCdata,axis=2))
axes.set_xticks(np.arange(4),legendlabelsx)
axes.set_yticks(np.arange(4),legendlabelsy)
plt.gca().invert_yaxis()



#%% ##################### Plot control figure of signal corrs ##############################
sesidx = 0
fig = plt.subplots(figsize=(8,5))
plt.imshow(sessions[sesidx].sig_corr, cmap='coolwarm',vmin=-0.02,vmax=0.04)
plt.savefig(os.path.join(savedir,'SignalCorrelations','Signal_Correlation_Images_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

####

