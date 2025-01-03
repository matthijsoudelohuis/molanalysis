# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Import packages
import math
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import medfilt
from scipy.stats import zscore

os.chdir('e:\\Python\\molanalysis\\')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.psth import compute_tensor_space,compute_respmat_space
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.plot_lib import *
from utils.decode_lib import *
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\')





#%% ########################## Load data #######################
protocol            = ['DN']
calciumversion      = 'deconv'

sessions,nSessions  = filter_sessions(protocol,load_calciumdata=True,load_behaviordata=True,
                                      load_videodata=True,calciumversion=calciumversion,min_cells=100)

#%% Remove sessions LPE10884 that are too bad:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE10884_2023_12_14','LPE10884_2023_12_15','LPE10884_2024_01_11',
                                                                'LPE10884_2024_01_16','LPE11622_2024_02_22']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# Only sessions that have rewardZoneOffset == 25
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(sessiondata['rewardZoneOffset'] == 25)[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 60   #post cm
sbinsize     = 5     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=sbinsize,method='binmean')

#%% #################### Compute spatial behavioral variables ####################################
for ises,ses in enumerate(sessions): # running across the trial:
    sessions[ises].behaviordata['runspeed'] = medfilt(sessions[ises].behaviordata['runspeed'], kernel_size=51)
    [sessions[ises].runPSTH,sbins]     = calc_runPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=sbinsize)
    [sessions[ises].pupilPSTH,sbins]   = calc_pupilPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=sbinsize)
    [sessions[ises].videomePSTH,sbins] = calc_videomePSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=sbinsize)
    [sessions[ises].lickPSTH,sbins]    = calc_lickPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=sbinsize)

#%% Define the number of folds for cross-validation
kfold = 5

#%% Decoding stimulus from V1 activity across space:
dec_perf_stim = np.full((nSessions,len(sbins)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
    idx_T = np.isin(ses.trialdata['stimcat'],['C','M'])
    # idx_T = np.isin(ses.trialdata['stimcat'],['C','N'])
    idx_N = ses.celldata['roi_name']=='V1'

    if np.sum(idx_T) > 50:
        # Get the maximum signal vs catch for this session
        y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()

        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        X = np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,((sbins>-5) & (sbins<20)).astype(bool))],axis=2)
        X = X.T # Transpose to K x N (samples x features)

        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        optimal_lambda = find_optimal_lambda(X,y,model_name='LogisticRegression',kfold=kfold)
        # Loop through each spatial bin
        for ibin, bincenter in enumerate(sbins):
            y   = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
            X   = ses.stensor[np.ix_(idx_N,idx_T,sbins==bincenter)].squeeze()
            X   = X.T

            X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            # Calculate the average decoding performance across folds
            dec_perf_stim[ises,ibin] = my_classifier_wrapper(X,y,model_name='LogisticRegression',kfold=kfold,lam=optimal_lambda,norm_out=True)


#%% Decoding of choice from behavioral variables:

# Define the variables to use for decoding
variables = ['runspeed', 'pupil_area', 'videoME', 'lick_rate']

# Initialize an array to store the decoding performance
dec_perf_choice = np.full((nSessions,len(sbins)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),desc='Decoding response across sessions'):
    #Correct setting: stimulus trials during engaged part of the session:
    idx = np.all((ses.trialdata['engaged']==1,np.isin(ses.trialdata['stimcat'],['M','N'])), axis=0)
    
    if np.sum(idx) > 50:
        # Get the lickresponse data for this session
        y = ses.trialdata['lickResponse'][idx].to_numpy()

        X = np.stack((ses.runPSTH[idx,:], ses.pupilPSTH[idx,:], ses.videomePSTH[idx,:], ses.lickPSTH[idx,:]), axis=2)
                #take the mean during the response window to determine optimal lambda
        with np.errstate(invalid='ignore'):
            X = np.nanmean(X[:, (sbins>=25) & (sbins<=45), :], axis=1)
        # X is now K x N, samples (trials) x features (behavioral measures)
        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        optimal_lambda = find_optimal_lambda(X,y,model_name='LOGR',kfold=kfold)

        # Loop through each spatial bin
        for ibin, bincenter in enumerate(sbins):
            
            # Define the X and y variables
            X = np.stack((ses.runPSTH[idx,ibin], ses.pupilPSTH[idx,ibin], ses.videomePSTH[idx,ibin], ses.lickPSTH[idx,ibin]), axis=1)
            # X = np.stack((ses.runPSTH[idx,ibin], ses.lickPSTH[idx,ibin]), axis=1)
            # X = np.stack((ses.pupilPSTH[idx,ibin], ses.videomePSTH[idx,ibin]), axis=1)
            
            X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            
            # Calculate the average decoding performance across folds
            dec_perf_choice[ises,ibin] = my_classifier_wrapper(X,y,model_name='LOGR',kfold=kfold,lam=optimal_lambda,norm_out=True)

#%% Show the decoding performance
# fig,axes = plt.subplots(1,2,figsize=(5,3))
fig,axes = plt.subplots(2,1,figsize=(4,6),sharex=True,sharey=True)
ax = axes[0]
for i,ses in enumerate(sessions):
# for i,ses in enumerate([sessions[3]]):
        # ax.plot(sbins,sperf[i,:],color='grey',alpha=0.5,linewidth=1)
    ax.plot(sbins,dec_perf_stim[i,:],alpha=0.5,linewidth=1)
# shaded_error(sbins,sperf,error='sem',ax=ax,color='grey')
ax.plot(sbins,np.nanmean(dec_perf_stim,axis=0),alpha=1,linewidth=2,color='k')
add_stim_resp_win(ax)
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_title('Stim Decoding')

ax = axes[1]
for i,ses in enumerate(sessions):
    ax.plot(sbins,dec_perf_choice[i,:],alpha=0.5,linewidth=1)
ax.plot(sbins,np.nanmean(dec_perf_choice,axis=0),alpha=1,linewidth=2,color='k')
# shaded_error(sbins,dec_perf_choice,error='sem',ax=ax,color='grey')
add_stim_resp_win(ax)
ax.set_xlabel('Position relative to stim (cm)')
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_title('Choice Decoding')
ax.set_xticks([-50,-25,0,25,50])
ax.set_ylim([-0.1,1])
ax.set_xlim([-60,60])
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Spatial', 'DecPerformance_Stim_Resp_averageSes.png'), format='png')


#%% Show the decoding performance per session 
# Identifies the difference in neural stimulus coding and behavioral readout of choice
# fig,axes = plt.subplots(1,2,figsize=(5,3))
nRows = int(np.ceil(nSessions/4))
fig,axes = plt.subplots(nRows,4,figsize=(12,nRows*2.5),sharex=True,sharey=True)
for i,ses in enumerate(sessions):
    ax = axes[i//4,i%4]
    ax.plot(sbins,dec_perf_stim[i,:],alpha=0.5,linewidth=1.5,color='b')
    ax.plot(sbins,dec_perf_choice[i,:],alpha=0.5,linewidth=1.5,color='g')
    # shaded_error(sbins,sperf,error='sem',ax=ax,color='grey')
    add_stim_resp_win(ax)
    ax.set_title(ses.sessiondata['session_id'][0])

    ax.set_xticks([-50,-25,0,25,50])
    ax.set_ylim([-0.1,1])
    ax.set_xlim([-60,60])
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=1)
    if i == 0:
        ax.set_ylabel('Performance \n (accuracy - shuffle)')
    if i==int(nSessions/2):
        ax.set_xlabel('Position relative to stim (cm)')
    if i == 0:
        ax.legend(['Stim','Choice'],frameon=False,fontsize=6,title='Decoding')
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Spatial', 'DecPerformance_Stim_Resp_indSes.png'), format='png')


#%% 









