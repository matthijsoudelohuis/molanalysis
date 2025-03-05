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

os.chdir('c:\\Python\\molanalysis\\')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.psth import *
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.plot_lib import *
from utils.regress_lib import *
from detection.plot_neural_activity_lib import *
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\')





#%% ########################## Load data #######################
protocol            = ['DN']
calciumversion      = 'deconv'

sessions,nSessions  = filter_sessions(protocol,load_calciumdata=True,load_behaviordata=True,
                                      load_videodata=True,calciumversion=calciumversion,min_cells=100)

# #%% Remove sessions LPE10884 that are too bad:
# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE10884_2023_12_14','LPE10884_2023_12_15','LPE10884_2024_01_11',
#                                                                 'LPE10884_2024_01_16','LPE11622_2024_02_22']))[0]
# sessions            = [sessions[i] for i in sessions_in_list]
# nSessions           = len(sessions)
# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# # Only sessions that have rewardZoneOffset == 25
# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list    = np.where(sessiondata['rewardZoneOffset'] == 25)[0]
# sessions            = [sessions[i] for i in sessions_in_list]
# nSessions           = len(sessions)
# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% 
sessions,nSessions = load_neural_performing_sessions()


#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 80   #post cm
sbinsize     = 10     #spatial binning in cm

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

#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

#%%

#%% Define the number of folds for cross-validation
kfold           = 5
lam             = 0.08
lam             = 0.8
nmintrialscond  = 10
nmodelfits      = 10

#%%

def balance_trial(X,y,sample_min_trials=20):
    N0,N1 = np.sum(y==0),np.sum(y==1)
    mintrials =  np.min([N0,N1])
    if mintrials < sample_min_trials:
        idx0 = np.random.choice(np.where(y==0)[0],size=sample_min_trials,replace=True)
        idx1 = np.random.choice(np.where(y==1)[0],size=sample_min_trials,replace=True)
        yb = np.concatenate((y[idx0],y[idx1]))
        Xb = np.concatenate((X[idx0,:],X[idx1,:]))
    else: 
        idx0 = np.random.choice(np.where(y==0)[0],size=mintrials,replace=False)
        idx1 = np.random.choice(np.where(y==1)[0],size=mintrials,replace=False)
        yb  = np.concatenate((y[idx0],y[idx1]))
        Xb  = np.concatenate((X[idx0,:],X[idx1,:]))
    return Xb,yb    

#%% Decoding stimulus from V1 activity across space:
dec_perf_stim = np.full((nSessions,len(sbins)), np.nan)
# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
    # idx_T = np.isin(ses.trialdata['stimcat'],['C','M'])
    idx_T = np.isin(ses.trialdata['stimcat'],['C','N'])
    # idx_N = ses.celldata['roi_name']=='V1'
    idx_N = ses.celldata['sig_MN']==1

    y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
    
    if np.sum(y==0) >= nmintrialscond and np.sum(y==1) >= nmintrialscond:
        # Get the maximum signal vs catch for this session
        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        idx_B = ((sbins>=-5) & (sbins<=20)).astype(bool)
        X = np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,idx_B)],axis=2)
        X = X.T # Transpose to K x N (samples x features)

        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        # lam = find_optimal_lambda(X,y,model_name='LogisticRegression',kfold=kfold)
        # Loop through each spatial bin
        for ibin, bincenter in enumerate(sbins):
            y   = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
            X   = ses.stensor[np.ix_(idx_N,idx_T,sbins==bincenter)].squeeze()
            X   = X.T

            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            # Calculate the average decoding performance across folds
            # dec_perf_stim[ises,ibin],_,_,_ = my_decoder_wrapper(X,y,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
            temp = np.empty(nmodelfits)
            for i in range(nmodelfits):
                Xb,yb           = balance_trial(X,y,sample_min_trials=10)
                temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
                # temp[i],_,_,_ = my_decoder_wrapper(X,y,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
            dec_perf_stim[ises,ibin] = np.mean(temp)

#%% Decoding of choice from behavioral variables:

# Define the variables to use for decoding
variables = ['runspeed', 'pupil_area', 'videoME', 'lick_rate']

# Initialize an array to store the decoding performance
dec_choice_beh = np.full((nSessions,len(sbins)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
    #Correct setting: stimulus trials during engaged part of the session:
    idx_T = np.all((ses.trialdata['engaged']==1,np.isin(ses.trialdata['stimcat'],['N'])), axis=0)
    # Get the lickresponse data for this session
    y = ses.trialdata['lickResponse'][idx_T].to_numpy()

    if np.sum(y==0) >= nmintrialscond and np.sum(y==1) >= nmintrialscond:
        X = np.stack((ses.runPSTH[idx_T,:], ses.pupilPSTH[idx_T,:], ses.videomePSTH[idx_T,:], ses.lickPSTH[idx_T,:]), axis=2)
                #take the mean during the response window to determine optimal lambda
        with np.errstate(invalid='ignore'):
            X = np.nanmean(X[:, (sbins>=25) & (sbins<=45), :], axis=1)
        # X is now K x N, samples (trials) x features (behavioral measures)
        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        # lam = find_optimal_lambda(X,y,model_name='LOGR',kfold=kfold)

        # Loop through each spatial bin
        for ibin, bincenter in enumerate(sbins):
            y = ses.trialdata['lickResponse'][idx_T].to_numpy()
            # Define the X and y variables
            X = np.stack((ses.runPSTH[idx_T,ibin], ses.pupilPSTH[idx_T,ibin], ses.videomePSTH[idx_T,ibin], ses.lickPSTH[idx_T,ibin]), axis=1)
            # X = np.stack((ses.runPSTH[idx_T,ibin], ses.lickPSTH[idx_T,ibin]), axis=1)
            # X = np.stack((ses.pupilPSTH[idx_T,ibin], ses.videomePSTH[idx_T,ibin]), axis=1)
            
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            
            # Calculate the average decoding performance across folds
            # dec_choice_beh[ises,ibin],_,_,_ = my_decoder_wrapper(X,y,model_name='LOGR',kfold=kfold,lam=optimal_lambda,norm_out=True)
            temp = np.empty(nmodelfits)
            for i in range(nmodelfits):
                Xb,yb = balance_trial(X,y,sample_min_trials=10)
                temp[i],_,_,_ = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
            dec_choice_beh[ises,ibin] = np.mean(temp)

#%% Decoding of choice from neural variables:
lam = 0.8
# Initialize an array to store the decoding performance
dec_choice_neu = np.full((nSessions,len(sbins)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
    tempdata = copy.deepcopy(ses.stensor)
    for isig,sig in enumerate(np.unique(ses.trialdata['signal'])):
        # print(sig)
        idx_T = ses.trialdata['signal']==sig
        tempdata[:,idx_T,:] -= np.nanmean(tempdata[:,idx_T,:],axis=1,keepdims=True)

    #Correct setting: stimulus trials during engaged part of the session:
    idx_T = np.all((ses.trialdata['engaged']==1,np.isin(ses.trialdata['stimcat'],['N'])), axis=0)

    # idx_T = np.isin(ses.trialdata['stimcat'],['C','N'])
    idx_N = np.ones(len(ses.celldata), dtype=bool)
    # idx_N = ses.celldata['sig_MN']==1

    y = ses.trialdata['lickResponse'][idx_T].to_numpy()

    if np.sum(y==0) >= nmintrialscond and np.sum(y==1) >= nmintrialscond:
        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        X = np.nanmean(tempdata[np.ix_(idx_N,idx_T,((sbins>-5) & (sbins<20)).astype(bool))],axis=2)
        X = X.T # Transpose to K x N (samples x features)

        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        # lam = find_optimal_lambda(X,y,model_name='LogisticRegression',kfold=kfold)
        # print(lam)
        # Loop through each spatial bin
        for ibin, bincenter in enumerate(sbins):
            y   = ses.trialdata['lickResponse'][idx_T].to_numpy()
            X   = tempdata[np.ix_(idx_N,idx_T,sbins==bincenter)].squeeze()
            X   = X.T

            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            # Calculate the average decoding performance across folds
            # dec_choice_neu[ises,ibin],_,_,_ = my_decoder_wrapper(X,y,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
            temp = np.empty(nmodelfits)
            for i in range(nmodelfits):
                Xb,yb = balance_trial(X,y,sample_min_trials=10)
                temp[i],_,_,_ = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
            dec_choice_neu[ises,ibin] = np.mean(temp)

#%% Show the decoding performance
# fig,axes = plt.subplots(1,2,figsize=(5,3))
colorset = sns.color_palette('husl',n_colors=nSessions)

fig,axes = plt.subplots(2,1,figsize=(4,6),sharex=True,sharey=True)
ax = axes[0]
for i,ses in enumerate(sessions):
# for i,ses in enumerate([sessions[3]]):
        # ax.plot(sbins,sperf[i,:],color='grey',alpha=0.5,linewidth=1)
    ax.plot(sbins,dec_perf_stim[i,:],alpha=0.5,linewidth=1,color=colorset[i])
# shaded_error(sbins,sperf,error='sem',ax=ax,color='grey')
ax.plot(sbins,np.nanmean(dec_perf_stim,axis=0),alpha=1,linewidth=2,color='k')
add_stim_resp_win(ax)
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_title('Stim Decoding')

ax = axes[1]
for i,ses in enumerate(sessions):
    ax.plot(sbins,dec_choice_beh[i,:],alpha=0.5,linewidth=1,color=colorset[i])
ax.plot(sbins,np.nanmean(dec_choice_beh,axis=0),alpha=1,linewidth=2,color='k')
# shaded_error(sbins,dec_choice_beh,error='sem',ax=ax,color='grey')
add_stim_resp_win(ax)
ax.set_xlabel('Position relative to stim (cm)')
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_title('Choice Decoding')
ax.set_xticks([-50,-25,0,25,50])
ax.set_ylim([-0.1,1])
ax.set_xlim([-60,60])
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'Spatial', 'DecPerformance_Stim_Resp_averageSes.png'), format='png')

#%%

#%% Show the decoding performance
colorset = sns.color_palette('husl',n_colors=nSessions)
# fig,axes = plt.subplots(1,2,figsize=(5,3))
# fig,axes = plt.subplots(3,1,figsize=(4,9),sharex=True,sharey=True)
fig,axes = plt.subplots(1,3,figsize=(9,3),sharex=True,sharey=True)

ax = axes[0]
for i,ses in enumerate(sessions):
# for i,ses in enumerate([sessions[3]]):
        # ax.plot(sbins,sperf[i,:],color='grey',alpha=0.5,linewidth=1)
    ax.plot(sbins,dec_perf_stim[i,:],alpha=0.5,linewidth=1,color=colorset[i])
# shaded_error(sbins,sperf,error='sem',ax=ax,color='grey')
ax.plot(sbins,np.nanmean(dec_perf_stim,axis=0),alpha=1,linewidth=3,color='k')
add_stim_resp_win(ax)
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_title('Stim Decoding')

ax = axes[1]
for i,ses in enumerate(sessions):
    ax.plot(sbins,dec_choice_beh[i,:],alpha=0.5,linewidth=1,color=colorset[i])
ax.plot(sbins,np.nanmean(dec_choice_beh,axis=0),alpha=1,linewidth=3,color='k')
# shaded_error(sbins,dec_choice_beh,error='sem',ax=ax,color='grey')
add_stim_resp_win(ax)
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_title('Choice Decoding (Behavioral)')

ax = axes[2]
for i,ses in enumerate(sessions):
    ax.plot(sbins,dec_choice_neu[i,:],alpha=0.5,linewidth=1,color=colorset[i])
ax.plot(sbins,np.nanmean(dec_choice_neu,axis=0),alpha=1,linewidth=3,color='k')
# shaded_error(sbins,dec_choice_neu,error='sem',ax=ax,color='grey')
add_stim_resp_win(ax)
ax.set_xlabel('Position relative to stim (cm)')
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_title('Choice Decoding (Neural)')
ax.set_xticks([-50,-25,0,25,50])
ax.set_ylim([-0.1,1])
ax.set_xlim([-60,60])
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Spatial', 'Dec_Stim_Resp_Neural_Beh_bin10cm.png'), format='png')
# plt.savefig(os.path.join(savedir, 'Spatial', 'Dec_Stim_Resp_Neural_Beh.png'), format='png')

#%% Show the decoding performance
fig,axes = plt.subplots(1,1,figsize=(4,3),sharex=True,sharey=True)
ax = axes
handles = []
handles.append(shaded_error(sbins,dec_perf_stim,color='b',alpha=0.3,linewidth=1.5,error='sem'))
handles.append(shaded_error(sbins,dec_choice_beh,color='g',alpha=0.3,linewidth=1.5,error='sem'))
handles.append(shaded_error(sbins,dec_choice_neu,color='r',alpha=0.3,linewidth=1.5,error='sem'))
add_stim_resp_win(ax)
ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles,labels=['Stim - Neural','Choice - Behav.','Choice - Neural'])
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_xlabel('Position relative to stim (cm)')
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_title('Decoding')
ax.set_xticks([-50,-25,0,25,50])
ax.set_ylim([-0.1,1])
ax.set_xlim([-60,60])
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Spatial', 'Dec_Stim_Resp_Neural_Beh_averageSes_bin10cm.png'), format='png')
# plt.savefig(os.path.join(savedir, 'Spatial', 'Dec_Stim_Resp_Neural_Beh_averageSes.png'), format='png')


#%% Show the decoding performance per session 
# Identifies the difference in neural stimulus coding and behavioral readout of choice
# fig,axes = plt.subplots(1,2,figsize=(5,3))
nperrow  = 6
nRows = int(np.ceil(nSessions/nperrow))
fig,axes = plt.subplots(nRows,nperrow,figsize=(15,nRows*2.5),sharex=True,sharey=True)
for i,ses in enumerate(sessions):
    ax = axes[i//nperrow,i%nperrow]
    ax.plot(sbins,dec_perf_stim[i,:],alpha=0.5,linewidth=1.5,color='b')
    ax.plot(sbins,dec_choice_beh[i,:],alpha=0.5,linewidth=1.5,color='g')
    ax.plot(sbins,dec_choice_neu[i,:],alpha=0.5,linewidth=1.5,color='r')
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
        ax.legend(['Stim - Neural','Choice - Behav.','Choice - Neural'],frameon=False,fontsize=6,title='Decoding')
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'Spatial', 'DecPerformance_Stim_Resp_indSes.png'), format='png')
plt.savefig(os.path.join(savedir, 'Spatial', 'DecPerformance_Stim_Choice_NeuralBehavioral_indSes.png'), format='png')

#%%


#%% Show the decoding performance per session 
# Identifies the difference in neural stimulus coding and behavioral readout of choice
fig,axes = plt.subplots(1,3,figsize=(9,3),sharex=True,sharey=True)
ax = axes[0]
for i,ses in enumerate(sessions):
    ax.plot(dec_perf_stim[i,:],dec_choice_beh[i,:],alpha=0.75,linewidth=1.5,color=colorset[i])
ax.plot(np.nanmean(dec_perf_stim,axis=0),np.nanmean(dec_choice_beh,axis=0),alpha=0.75,linewidth=3,color='k')
# ax.scatter(dec_perf_stim,dec_choice_beh,alpha=0.5,color='k')
# ax.plot(dec_perf_stim,dec_choice_beh,alpha=0.5,color='k')
ax.plot([0,1],[0,1],color='k',linewidth=1,linestyle='--')
ax.set_xlim([-0.2,1])
ax.set_ylim([-0.2,1])
ax.set_xlabel('Stim - Neural')
ax.set_ylabel('Choice - Behav.')

ax = axes[1]
for i,ses in enumerate(sessions):
    ax.plot(dec_perf_stim[i,:],dec_choice_neu[i,:],alpha=0.75,linewidth=1.5,color=colorset[i])
ax.plot(np.nanmean(dec_perf_stim,axis=0),np.nanmean(dec_choice_neu,axis=0),alpha=0.75,linewidth=3,color='k')
ax.plot([0,1],[0,1],color='k',linewidth=1,linestyle='--')
ax.set_xlabel('Stim - Neural')
ax.set_ylabel('Choice - Neural')

ax = axes[2]
for i,ses in enumerate(sessions):
    ax.plot(dec_choice_beh[i,:],dec_choice_neu[i,:],alpha=0.75,linewidth=1.5,color=colorset[i])
ax.plot(np.nanmean(dec_choice_beh,axis=0),np.nanmean(dec_choice_neu,axis=0),alpha=0.75,linewidth=3,color='k')
ax.plot([0,1],[0,1],color='k',linewidth=1,linestyle='--')
ax.set_xlabel('Choice - Behav.')
ax.set_ylabel('Choice - Neural')
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Spatial', '2D_DecPerformance_Stim_Choice_NeuralBehavioral_indSes.png'), format='png')


#%% 









