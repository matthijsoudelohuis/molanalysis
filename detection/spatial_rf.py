# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
os.chdir('e:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore

# #import personal modules
from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive
from utils.psth import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.plot_lib import * # get support functions for plotting

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\RF\\')

#%% ###############################################################
protocol            = 'DN'
calciumversion      = 'deconv'
# calciumversion      = 'dF'

sessions,nSessions = filter_sessions(protocol,session_rf=True,min_cells=100,
                           load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

#%% Z-score calcium data:
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 60   #post cm
binsize     = 2.5     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')

    ## Compute average response in stimulus response zone:
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

#%% Compute significant responsive neurons
sessions        = calc_stimresponsive_neurons(sessions,sbins)
celldata        = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% #################### Compute activity for each stimulus type for all session ##################
N           = len(celldata) #get number of cells total
S           = len(sbins) #get number of spatial bins

stimtypes   = ['C','N','M']
stimlabels  = ['catch','noise','max']
tt_mean     = np.empty([N,S,len(stimtypes)])

for ises,ses in enumerate(sessions):
    idx = celldata['session_id']==ses.sessiondata['session_id'][0]
    for iTT in range(len(stimtypes)):
        trialidx = ses.trialdata['stimcat'] == stimtypes[iTT]
        tt_mean[idx,:,iTT] = np.nanmean(sessions[ises].stensor[:,trialidx,:],axis=1)

#%% ################## Number of responsive neurons per stimulus #################################
idx             = np.all((celldata['rf_r2_Fneu']>=0.15,
                          celldata['sig_MN']==1),axis=0)
# idx             = celldata['rf_r2_Fneu']>=0.15

# rf_az_binedges     = np.linspace(celldata['rf_az_Fneu'][idx].min(),celldata['rf_az_Fneu'][idx].max(),num=6)
rf_az_binedges     = np.linspace(np.percentile(celldata['rf_az_Fneu'][idx],2),
                                 np.percentile(celldata['rf_az_Fneu'][idx],98),
                                 num=6)
rf_az_bincenters = np.round(np.stack((rf_az_binedges[:-1],rf_az_binedges[1:]),axis=1).mean(axis=1))

plotdata = np.empty([len(rf_az_binedges)-1,len(sbins),len(stimtypes)])

for irf_bin,rf_bin in enumerate(rf_az_binedges[:-1]):
    idx_resp = np.all((idx,
                       celldata['rf_az_Fneu']>=rf_bin,
                       celldata['rf_az_Fneu']<rf_az_binedges[irf_bin+1]),axis=0)
   
    plotdata[irf_bin,:,:] = np.nanmean(tt_mean[idx_resp,:,:],axis=0)

#%% Construct color panel for saliency trial bins
plotlabels = ['catch','noise','max']
plotcolors = sns.color_palette("magma", n_colors=len(rf_az_binedges))  # Add 5 colors from the magma palette

#%% ############################### Plot neuron-average activity per stim #################################
fig,axes = plt.subplots(1,3,figsize=(3.5*3,2.5),sharex=True,sharey=True)
for istim,stim in enumerate(stimlabels):
    ax = axes[istim]
    for irf_bin,rf_bin in enumerate(rf_az_binedges[:-1]):
        ax.plot(sbins, plotdata[irf_bin,:,istim], color=plotcolors[irf_bin], label=rf_az_bincenters[irf_bin],linewidth=2)
    ax.set_ylim([-0.15,0.8])
    ax.legend(frameon=False,fontsize=8,title='Azimuth (deg)')
    ax.set_xlim([-60,60])
    ax.set_title(stim)
    ax.set_xlabel('Position relative to stim (cm)')
    ax.set_ylabel('Activity (z)')
    # ax.set_yticks([0,0.1,0.2])
    add_stim_resp_win(ax)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Spatial_Mean_per_RF_Az_bin_allAreas_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% 
thr = 0.1
firstbin = np.empty([len(rf_az_binedges)-1])
for irf_bin,rf_bin in enumerate(rf_az_binedges[:-1]):
    firstbin[irf_bin] = sbins[np.where(np.logical_and(plotdata[irf_bin,:,2]>thr,sbins>-20))[0][0]]

fig,ax = plt.subplots(1,1,figsize=[3,3])
ax.scatter(rf_az_bincenters,firstbin,color='k',linewidth=2)
ax.set_xlabel('Azimuth (deg)')
ax.set_ylabel('First significant bin (cm)')
ax.set_xlim([0,150])
ax.set_ylim([-15,15])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'ResponseLocation_vs_RF_Az_allAreas_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')



#Now for elevation, shouldn't be a relationship:

#%% ################## Number of responsive neurons per stimulus #################################
idx             = np.all((celldata['rf_r2_Fneu']>=0.2,
                          celldata['sig_MN']==1),axis=0)
# idx             = celldata['rf_r2_Fneu']>=0.15

rf_el_binedges     = np.linspace(np.percentile(celldata['rf_el_Fneu'][idx],2),
                                 np.percentile(celldata['rf_el_Fneu'][idx],98),
                                 num=6)
# rf_el_binedges     = np.linspace(celldata['rf_el_Fneu'][idx].min(),celldata['rf_el_Fneu'][idx].max(),num=6)
rf_el_bincenters  = np.round(np.stack((rf_el_binedges[:-1],rf_el_binedges[1:]),axis=1).mean(axis=1))

plotdata = np.empty([len(rf_el_binedges)-1,len(sbins),len(stimtypes)])

for irf_bin,rf_bin in enumerate(rf_el_binedges[:-1]):
    idx_resp = np.all((idx,
                       celldata['rf_el_Fneu']>=rf_bin,
                       celldata['rf_el_Fneu']<=rf_el_binedges[irf_bin+1]),axis=0)
   
    plotdata[irf_bin,:,:] = np.nanmean(tt_mean[idx_resp,:,:],axis=0)

#%% Construct color panel for saliency trial bins
plotlabels = ['catch','noise','max']
plotcolors = sns.color_palette("magma", n_colors=len(rf_el_binedges))  # Add 5 colors from the magma palette

#%% ############################### Plot neuron-average activity per stim #################################
fig,axes = plt.subplots(1,3,figsize=(3.5*3,2.5),sharex=True,sharey=True)
for istim,stim in enumerate(stimlabels):
    ax = axes[istim]
    for irf_bin,rf_bin in enumerate(rf_el_binedges[:-1]):
        ax.plot(sbins, plotdata[irf_bin,:,istim], color=plotcolors[irf_bin], label=rf_el_bincenters[irf_bin],linewidth=2)
    ax.set_ylim([-0.15,0.8])
    ax.legend(frameon=False,fontsize=8,title='Elevation (deg)')
    ax.set_xlim([-60,60])
    ax.set_title(stim)
    ax.set_xlabel('Position relative to stim (cm)')
    ax.set_ylabel('Activity (z)')
    # ax.set_yticks([0,0.1,0.2])
    add_stim_resp_win(ax)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Spatial_Mean_per_RF_El_bin_allAreas_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% 
thr = 0.1
firstbin = np.empty([len(rf_el_binedges)-1])
for irf_bin,rf_bin in enumerate(rf_el_binedges[:-1]):
    firstbin[irf_bin] = sbins[np.where(np.logical_and(plotdata[irf_bin,:,2]>thr,sbins>-20))[0][0]]

fig,ax = plt.subplots(1,1,figsize=[3,3])
ax.scatter(rf_el_bincenters,firstbin,color='k',linewidth=2)
ax.set_xlabel('Elevation (deg)')
ax.set_ylabel('First significant bin (cm)')
ax.set_xlim([-15,50])
ax.set_ylim([-15,15])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'ResponseLocation_vs_RF_El_allAreas_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')


