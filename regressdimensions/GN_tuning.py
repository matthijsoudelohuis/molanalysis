# -*- coding: utf-8 -*-
"""
This script analyzes the preferred stimuli for each neuron in a multi-area
calcium imaging dataset with labeled projection neurons. The visual stimuli
are oriented gratings with jittered spatiotemporal frequency and orientation.
Spatiotemporal frequency conditions were optimized for PM and AL areas.
This script verifies whether the preferred SF and TF matches the areas and whether
labeled projection neurons are matching in preference to their target area.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
# from utils.psth import compute_respmat
from utils.tuning import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.plot_lib import shaded_error

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\NoiseRegression\\GN_Stimuli\\')

#%% #############################################################################
#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE11998','2024_05_02'], #GN
                                ['LPE12013','2024_05_02']]) #GN
sessions,nSessions   = load_sessions(protocol = 'GN',session_list=session_list)

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['GN'])

#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv')

#%% ######### Compute average response, tuning metrics, and responsive fraction per session ##################
oris, speeds    = [np.unique(sessions[ises].trialdata[col]).astype('int') for col in ('centerOrientation', 'centerSpeed')]
areas           = np.array(['AL', 'PM', 'RSP', 'V1'], dtype=object)
redcells        = np.array([0, 1])
redcelllabels   = np.array(['unl', 'lab'])

tuning_thr      = 0.05

meanmat             = np.full([oris.shape[0],speeds.shape[0],nSessions,len(areas)],np.nan)
meanmat_labeled     = np.full([oris.shape[0],speeds.shape[0],nSessions,len(areas),len(redcells)],np.nan)
fracmat             = np.full([oris.shape[0],speeds.shape[0],nSessions,len(areas)],np.nan)
fracmat_labeled     = np.full([oris.shape[0],speeds.shape[0],nSessions,len(areas),len(redcells)],np.nan)

for ises in range(nSessions):
    resp_mean,resp_res      = mean_resp_gn(sessions[ises])
    sessions[ises].celldata['tuning_var'] = compute_tuning_var(resp_mat=sessions[ises].respmat,resp_res=resp_res)
    sessions[ises].celldata['pref_ori'],sessions[ises].celldata['pref_speed'] = get_pref_orispeed(resp_mean,oris,speeds)

    for iarea,area in enumerate(areas):
        meanmat[:, :, ises, iarea] = np.nanmean(resp_mean[sessions[ises].celldata['roi_name'] == area], axis=0)
        for iredcell in redcells:
                    meanmat_labeled[:, :, ises, iarea, iredcell] = np.nanmean(resp_mean[np.logical_and(sessions[ises].celldata['roi_name'] == area,
                                                                                                        sessions[ises].celldata['redcell'] == iredcell)], axis=0)
                    
        for iori,ori in enumerate(oris):
            for ispeed,speed in enumerate(speeds):
                fracmat[iori,ispeed,ises,iarea] = np.sum(np.all((sessions[ises].celldata['roi_name'] == area,
                                                                         sessions[ises].celldata['pref_ori'] == iori,
                                                                         sessions[ises].celldata['pref_speed'] == ispeed,
                                                                         sessions[ises].celldata['tuning_var'] > tuning_thr),axis=0)) / np.sum(sessions[ises].celldata['roi_name'] == area)
                                                                                                                                            
                for iredcell in redcells:
                    fracmat_labeled[iori,ispeed,ises,iarea,iredcell] = np.sum(np.all(
                         (sessions[ises].celldata['roi_name'] == area,
                        sessions[ises].celldata['pref_ori'] == iori,
                        sessions[ises].celldata['pref_speed'] == ispeed,
                        sessions[ises].celldata['tuning_var'] > tuning_thr,
                        sessions[ises].celldata['redcell'] == iredcell),axis=0)) / np.sum(np.logical_and(
                             sessions[ises].celldata['roi_name'] == area, sessions[ises].celldata['redcell'] == iredcell))

#%% Show heatmap of the session-averaged deconvolved response per area
fig,axes = plt.subplots(1,4,figsize=(8,2))
for i,ax in enumerate(axes.flatten()):
    oris_m, speeds_m = np.meshgrid(range(oris.shape[0]), range(speeds.shape[0]), indexing='ij')
    ax.pcolor(oris_m, speeds_m, np.nanmean(meanmat[:,:,:,i],axis=2),
              cmap='hot')
    ax.set_xticks(range(len(oris)),labels=oris)
    ax.set_yticks(range(len(speeds)),labels=speeds)
    ax.set_xlabel('Orientation (deg)')
    ax.set_ylabel('Speed (deg/s)')
    ax.set_title(areas[i])
fig.tight_layout()
fig.savefig(os.path.join(savedir,'Heatmap_OriSpeed_Tuning_MeanResponsePerArea.png'), format = 'png')

#%% 

#%% Plot the fraction of neurons responsive to any orientation averaged across orientations
clrs_areas = get_clr_areas(areas)
lines_redcells = ['-','--']

handles = []
linelabels =  []
fig,ax = plt.subplots(1,1,figsize=(4,3))
for iarea,area in enumerate(areas):
    for iredcell in redcells:
        # datatoplot = np.nanmean(fracmat_labeled[:,:,:,iarea,iredcell],axis=0).squeeze()
        datatoplot = np.nansum(fracmat_labeled[:,:,:,iarea,iredcell],axis=0).squeeze()
        # linehandle = shaded_error(ax,range(len(speeds)),datatoplot.T,center='mean',error='sem',color=clrs_areas[iarea])
        handles.append(shaded_error(ax,range(len(speeds)),datatoplot.T,center='mean',error='sem',
                                       linestyle=lines_redcells[iredcell],color=clrs_areas[iarea]))
        
        # datatoplot = np.nanmean(np.nanmean(fracmat_labeled[:,:,:,iarea,iredcell],axis=2),axis=0).squeeze()
        # ax.plot(range(len(speeds)),datatoplot,color=clrs_areas[iarea],linestyle=lines_redcells[iredcell],
                # label=area+redcelllabels[iredcell])
        linelabels.append(area+redcelllabels[iredcell])
ax.set_ylim([0.02,0.45])
ax.set_xticks(range(len(speeds)),labels=speeds)
ax.set_xlabel('Speed (deg/s)')
ax.set_ylabel('Fraction of neurons')
# ax.set_title('Fraction of neurons responsive to any orientation')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1, frameon=False)
ax.legend(handles=handles,labels=linelabels,bbox_to_anchor=(1.05, 1), loc='upper left', 
          borderaxespad=0.1, frameon=False,fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'FractionResponsiveSpeed_Sum_PerArea_Labeling.png'), format = 'png')
plt.savefig(os.path.join(savedir,'FractionResponsiveSpeed_Sum_PerArea_Labeling.pdf'), format = 'pdf')


#%% ###################################################
## Compute tuning measure: how much of the variance is due to mean response per stimulus category:
fig,ax = plt.subplots(1,1,figsize=(3,3))
for ises in range(nSessions):    
    sns.histplot(sessions[ises].celldata['tuning_var'],ax=ax,element='step',fill=False,alpha=0.5,
                 stat='percent',bins=np.arange(0,1,0.025))
plt.xlim([0,1])
fig.savefig(os.path.join(savedir,'Tuning_distribution_%sSessions' + nSessions + '.png'), format = 'png')

