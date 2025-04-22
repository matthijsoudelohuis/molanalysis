# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
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

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning, compute_prefori
from utils.plot_lib import * #get all the fixed color schemes
from utils.psth import compute_tensor,compute_respmat
from utils.corr_lib import *
from utils.explorefigs import plot_excerpt,plot_PCA_gratings,plot_tuned_response

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\DescriptiveStatisticsSessions\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE09830','2023_04_10']])
# session_list        = np.array([['LPE09830','2023_04_10'],
#                                 ['LPE09830','2023_04_12']])

#%% Load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
sessions2,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

sessions.append(sessions2[0])
nSessions = len(sessions)

#%% Load proper data and compute average trial responses:
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)                      
sessions[1].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
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
    
#%% Scatter of 
plt.figure(figsize=(4,4))
# sns.histplot(x=sessions[1].celldata['pref_ori'],y=sessions[0].celldata['pref_ori'],bins=np.unique(sessions[ises].trialdata['Orientation']))
# sns.histplot(x=sessions[1].celldata['pref_ori'],y=sessions[0].celldata['pref_ori'],bins=np.arange(0,360,17),common_bins=True)
sns.histplot(x=sessions[1].celldata['pref_ori'],y=sessions[0].celldata['pref_ori'],bins=16,common_bins=True)
# plt.scatter(sessions[0].celldata['pref_ori'],sessions[1].celldata['pref_ori'],s=10,alpha=0.8)
plt.ylabel('deconv')
plt.xlabel('dF/F')
plt.title('Pref Ori')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Deconvolution','Prefori_dF_deconv_optimwindows' + '.png'), format = 'png')

#%% Scatter of 
plt.figure(figsize=(4,4))
plt.scatter(sessions[1].celldata['OSI'],sessions[0].celldata['OSI'],s=10,alpha=0.5)
plt.plot([0, 1], [0, 1], ls="--", c=".3")
plt.ylabel('deconv')
plt.xlabel('dF/F')
plt.title('OSI')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Deconvolution','OSI_dF_deconv_optimwindows' + '.png'), format = 'png')

#%% Scatter of 
plt.figure(figsize=(4,4))
plt.scatter(sessions[1].celldata['tuning_var'],sessions[0].celldata['tuning_var'],s=10,alpha=0.5)
plt.plot([0, 1], [0, 1], ls="--", c=".3")
plt.ylabel('deconv')
plt.xlabel('dF/F')
plt.title('Tuning Variance Explained')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Deconvolution','Tuning_Var_dF_deconv_optimwindows' + '.png'), format = 'png')

#%% Noise correlations:
sessions = compute_noise_correlation(sessions,uppertriangular=False)

#%% Plotting Noise Correlation distribution across all pairs:
fig,ax = plt.subplots(figsize=(5,4))
for ses in tqdm(sessions,total=len(sessions),desc= 'Kernel Density Estimation for each session: '):
    sns.kdeplot(data=ses.noise_corr.flatten(),ax=ax)
plt.xlim([-0.15,0.4])
plt.legend(labels=['deconv','dF'],frameon=False)
plt.xlabel('Noise Correlation')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Deconvolution','NC_dF_deconv' + '.png'), format = 'png')

#%% #############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

for ises in range(nSessions):
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='nearby')




#%% Show some tuned responses with calcium and deconvolved traces across orientations:
example_cells = [3,100,58,62,70]
fig = plot_tuned_response(sessions[0].tensor,sessions[0].trialdata,t_axis,example_cells)
fig.suptitle('%s - Deconvolved' % sessions[0].sessiondata['session_id'][0],fontsize=12)
# save the figure
fig.savefig(os.path.join(savedir,'TunedResponse_deconv_%s.png' % sessions[0].sessiondata['session_id']))

fig = plot_tuned_response(sessions[1].tensor,sessions[1].trialdata,t_axis,example_cells)
fig.suptitle('%s - dF/F' % sessions[1].sessiondata['session_id'][0],fontsize=12)
# save the figure
fig.savefig(os.path.join(savedir,'TunedResponse_dF_%s.png' % sessions[0].sessiondata['session_id']))

#%% Figure of complete average response for dF/F and deconv: 
fig,ax = plt.subplots(figsize=(5,4))

for ises in range(nSessions):
    data = sessions[ises].tensor.mean(axis=0).mean(axis=0)
    normalized_data = (data - data.min()) / (data.max() - data.min())
    plt.plot(t_axis,normalized_data,alpha=1)
plt.legend(labels=['deconv','dF'],frameon=False)
plt.ylabel('Normalized Activity')
plt.ylabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Deconvolution','Resp_dF_deconv' + '.png'), format = 'png')

#%% How does the tuning vary with the time window used to calciulate the response (for dF/F):
ises = 1
t_resp_start    = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7] 
t_resp_stop     = [0.5,1,1.5,2]
plotdata = np.empty((len(t_resp_start),len(t_resp_stop)))
for i, t_start in enumerate(t_resp_start): 
    for j, t_stop in enumerate(t_resp_stop): 
        sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                            t_resp_start=t_start,t_resp_stop=t_stop,subtr_baseline=False)
        sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='tuning_var')
        sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='OSI')
        plotdata[i,j] = np.mean(sessions[ises].celldata['tuning_var'])
    

fig,ax = plt.subplots(figsize=(5,4))
c = ax.pcolor(t_resp_stop,t_resp_start,plotdata,edgecolors='k', linewidths=1)
fig.colorbar(c, ax=ax,label='tuning variance explained')
idx = np.unravel_index(np.nanargmax(plotdata, axis=None), plotdata.shape)
ax.text(t_resp_stop[idx[1]],t_resp_start[idx[0]],'max')
ax.set_xlabel('t_resp_stop')
ax.set_ylabel('t_resp_start')
ax.set_title('Tuning variance - %s - dF/F' % sessions[1].sessiondata['session_id'][0],fontsize=12)
ax.set_yticks(t_resp_start)
ax.set_xticks(t_resp_stop)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'ResponseWindow_TuningVar_dF_%s.png' % sessions[1].sessiondata['session_id'][0]))

#%% How does the tuning vary with the time window used to calciulate the response (for deconvolved):
ises = 0
t_resp_start    = [0,0.1,0.2,0.3,0.4,0.5] 
t_resp_stop     = [0.5,0.75,1,1.5,2]
plotdata = np.empty((len(t_resp_start),len(t_resp_stop)))
for i, t_start in enumerate(t_resp_start): 
    for j, t_stop in enumerate(t_resp_stop): 
        sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                            t_resp_start=t_start,t_resp_stop=t_stop,subtr_baseline=False)
        sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='tuning_var')
        sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='OSI')
        plotdata[i,j] = np.mean(sessions[ises].celldata['tuning_var'])

fig,ax = plt.subplots(figsize=(5,4))
c = ax.pcolor(t_resp_stop,t_resp_start,plotdata,edgecolors='k', linewidths=1)
fig.colorbar(c, ax=ax,label='tuning variance explained')
idx = np.unravel_index(np.nanargmax(plotdata, axis=None), plotdata.shape)
ax.text(t_resp_stop[idx[1]],t_resp_start[idx[0]],'max')
ax.set_xlabel('t_resp_stop')
ax.set_ylabel('t_resp_start')
ax.set_title('Tuning variance - %s - deconv' % sessions[0].sessiondata['session_id'][0],fontsize=12)
ax.set_yticks(t_resp_start)
ax.set_xticks(t_resp_stop)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'ResponseWindow_TuningVar_deconv_%s.png' % sessions[0].sessiondata['session_id'][0]))

#%% PCA projections: 
fig = plot_PCA_gratings(sessions[0],apply_zscore=True)
fig.savefig(os.path.join(savedir,'Deconvolution','PCA_deconv_%s.png' % sessions[0].sessiondata['session_id'][0]))

fig = plot_PCA_gratings(sessions[1],apply_zscore=True)
fig.savefig(os.path.join(savedir,'Deconvolution','PCA_dF_%s.png' % sessions[0].sessiondata['session_id'][0]))


#%% Figure of complete average response for dF/F and deconv: 
fig,ax = plt.subplots(figsize=(5,4))
clr_labeled = get_clr_labeled()
labels_labeled = ['non', 'labeled']
for ilabel in [0,1]:
    data = sessions[ises].tensor[sessions[ises].celldata['redcell']==ilabel,:,:].mean(axis=0).mean(axis=0)
    plt.plot(t_axis,data,alpha=1,color=clr_labeled[ilabel])
plt.legend(labels=labels_labeled,frameon=False)
plt.ylabel('Normalized Activity')
plt.ylabel('Time (s)')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Deconvolution','Resp_dF_deconv' + '.png'), format = 'png')
