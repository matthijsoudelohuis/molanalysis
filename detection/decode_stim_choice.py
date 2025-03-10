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
from scipy.stats import zscore, wilcoxon, ranksums, ttest_rel
from matplotlib.lines import Line2D

os.chdir('c:\\Python\\molanalysis\\')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.psth import *
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.plot_lib import *
from utils.regress_lib import *
from utils.rf_lib import filter_nearlabeled

from detection.plot_neural_activity_lib import *
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Decoding\\')


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
ax.set_xticks([-50,-25,0,25,50,75])
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
ax.set_xticks([-50,-25,0,25,50,75])
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
ax.set_xticks([-50,-25,0,25,50,75])
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

    ax.set_xticks([-50,-25,0,25,50,75])
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


######  #######  #####  ####### ######  #######    #     #    #    ######   #####     #          #    ######  ####### #       ####### ######  
#     # #       #     # #     # #     # #          #     #   # #   #     # #     #    #         # #   #     # #       #       #       #     # 
#     # #       #       #     # #     # #          #     #  #   #  #     # #          #        #   #  #     # #       #       #       #     # 
#     # #####   #       #     # #     # #####      #     # #     # ######   #####     #       #     # ######  #####   #       #####   #     # 
#     # #       #       #     # #     # #           #   #  ####### #   #         #    #       ####### #     # #       #       #       #     # 
#     # #       #     # #     # #     # #            # #   #     # #    #  #     #    #       #     # #     # #       #       #       #     # 
######  #######  #####  ####### ######  #######       #    #     # #     #  #####     ####### #     # ######  ####### ####### ####### ######  

#%% Decode from V1 and PM labeled and unlabeled neurons separately
def decvar_from_arealabel_wrapper(sessions,sbins,arealabels,var='noise',nmodelfits=20,kfold=5,lam=0.08,nmintrialscond=10,
                                   nminneurons=10,nsampleneurons=20):
    np.random.seed(49)

    nSessions       = len(sessions)
    narealabels     = len(arealabels)

    dec_perf   = np.full((len(sbins),narealabels,nSessions), np.nan)
    # Loop through each session
    for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
        neuraldata = copy.deepcopy(ses.stensor)

        for ial, arealabel in enumerate(arealabels):
            if var=='noise':
                idx_T       = np.all((np.isin(ses.trialdata['stimcat'],['C','N']),
                                        # np.isin(ses.trialdata['trialOutcome'],['HIT','CR']),
                                        ses.trialdata['engaged']==1),axis=0)
                y_idx_T           = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
            elif var=='max':
                idx_T       = np.all((np.isin(ses.trialdata['stimcat'],['C','M']),
                                        # np.isin(ses.trialdata['trialOutcome'],['HIT','CR']),
                                        ses.trialdata['engaged']==1),axis=0)
                y_idx_T           = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
            elif var=='choice':
                for isig,sig in enumerate(np.unique(ses.trialdata['signal'])):
                    idx_T = ses.trialdata['signal']==sig
                    neuraldata[:,idx_T,:] -= np.nanmean(neuraldata[:,idx_T,:],axis=1,keepdims=True)

                #Correct setting: stimulus trials during engaged part of the session:
                idx_T = np.all((ses.trialdata['engaged']==1,np.isin(ses.trialdata['stimcat'],['N'])), axis=0)
                y_idx_T = ses.trialdata['lickResponse'][idx_T].to_numpy()

            idx_nearby  = filter_nearlabeled(ses,radius=50)
            idx_N       = np.all((ses.celldata['arealabel']==arealabel,
                                    ses.celldata['noise_level']<20,	
                                    idx_nearby),axis=0)

            if np.sum(y_idx_T==0) >= nmintrialscond and np.sum(y_idx_T==1) >= nmintrialscond and np.sum(idx_N) >= nminneurons:
                for ibin, bincenter in enumerate(sbins):            # Loop through each spatial bin
                    temp = np.empty(nmodelfits)
                    for i in range(nmodelfits):
                        y   = copy.deepcopy(y_idx_T)

                        if np.sum(idx_N) >= nsampleneurons:
                            # idx_Ns = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                            idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                        else:
                            idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=True)

                        X       = neuraldata[np.ix_(idx_Ns,idx_T,sbins==bincenter)].squeeze()
                        X       = X.T

                        X,y,_   = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                        Xb,yb           = balance_trial(X,y,sample_min_trials=nmintrialscond)
                        # temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
                        temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,
                                                                subtract_shuffle=False,norm_out=False)
                    dec_perf[ibin,ial,ises] = np.nanmean(temp)

    return dec_perf

#%% Decode from V1 and PM labeled and unlabeled neurons separately, split by hit/miss
def decvar_hitmiss_from_arealabel_wrapper(sessions,sbins,arealabels,var='noise',nmodelfits=20,kfold=5,lam=0.08,nmintrialscond=10,
                                   nminneurons=10,nsampleneurons=20):
    np.random.seed(49)

    nSessions       = len(sessions)
    narealabels     = len(arealabels)
    nlickresponse   = 2

    dec_perf   = np.full((len(sbins),narealabels,nlickresponse,nSessions), np.nan)
    # Loop through each session
    for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
        neuraldata = copy.deepcopy(ses.stensor)

        for ial, arealabel in enumerate(arealabels):
            for ilr in range(nlickresponse):

                if var=='noise':
                    idx_cat     = np.logical_or(np.logical_and(ses.trialdata['stimcat']=='N',
                                                               ses.trialdata['lickResponse']==ilr),
                                        ses.trialdata['stimcat']=='C')
                    idx_T       = np.all((idx_cat,
                                  ses.trialdata['engaged']==1),axis=0)

                elif var=='max':
                    idx_cat     = np.logical_or(np.logical_and(ses.trialdata['stimcat']=='M',
                                                               ses.trialdata['lickResponse']==ilr),
                                        ses.trialdata['stimcat']=='C')
                    idx_T       = np.all((idx_cat,
                                  ses.trialdata['engaged']==1),axis=0)
                elif var=='choice':
                    for isig,sig in enumerate(np.unique(ses.trialdata['signal'])):
                        idx_T = ses.trialdata['signal']==sig
                        neuraldata[:,idx_T,:] -= np.nanmean(neuraldata[:,idx_T,:],axis=1,keepdims=True)

                    #Correct setting: stimulus trials during engaged part of the session:
                    idx_T = np.all((ses.trialdata['engaged']==1,np.isin(ses.trialdata['stimcat'],['N'])), axis=0)
                    y_idx_T = ses.trialdata['lickResponse'][idx_T].to_numpy()

                idx_nearby  = filter_nearlabeled(ses,radius=50)
                idx_N       = np.all((ses.celldata['arealabel']==arealabel,
                                        ses.celldata['noise_level']<20,	
                                        idx_nearby),axis=0)

                if np.sum(y_idx_T==0) >= nmintrialscond and np.sum(y_idx_T==1) >= nmintrialscond and np.sum(idx_N) >= nminneurons:
                    for ibin, bincenter in enumerate(sbins):            # Loop through each spatial bin
                        temp = np.empty(nmodelfits)
                        for i in range(nmodelfits):
                            y   = copy.deepcopy(y_idx_T)

                            if np.sum(idx_N) >= nsampleneurons:
                                # idx_Ns = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                                idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
                            else:
                                idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=True)

                            X       = neuraldata[np.ix_(idx_Ns,idx_T,sbins==bincenter)].squeeze()
                            X       = X.T

                            X,y,_   = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                            Xb,yb           = balance_trial(X,y,sample_min_trials=nmintrialscond)
                            # temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
                            temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,
                                                                    subtract_shuffle=False,norm_out=False)
                        dec_perf[ibin,ial,ilr,ises] = np.nanmean(temp)

    return dec_perf

#%% Show the decoding performance across space for the different populations:
def plot_dec_perf_arealabel(dec_perf,sbins,arealabels,clrs_arealabels,testwin=[0,25]):

    fig,axes    = plt.subplots(1,4,figsize=(10,3),sharex=False,sharey=True,gridspec_kw={'width_ratios': [3,1,3,1]})
    markersize  = 30

    statdata    = np.nanmean(dec_perf[(sbins>=testwin[0]) & (sbins<=testwin[1]),:,:],axis=0)

    nSessions   = dec_perf.shape[2]
    # statdata    = np.nanmean(dec_perf[(sbins>=11) & (sbins<=20),:,:],axis=0)
    ax = axes[0]
    handles = []
    for ial, arealabel in enumerate(arealabels[:2]):
        handles.append(shaded_error(sbins,dec_perf[:,ial,:].T,color=clrs_arealabels[ial],alpha=0.5,linewidth=1.5,error='sem',ax=ax))
        for ises in range(nSessions):
            ax.plot(sbins,dec_perf[:,ial,ises],color=clrs_arealabels[ial],linewidth=0.2)

    ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles,labels=arealabels[:2])
    ax.set_ylabel('Decoding performance')
    ax.set_xlabel('Position relative to stim (cm)')
    add_stim_resp_win(ax)
    ax.set_xticks([-50,-25,0,25,50,75])
    ax.set_ylim([-0.1,1])
    ax.set_ylim([0.45,1])
    ax.set_xlim([-50,75])

    for ibin, bincenter in enumerate(sbins):
        t,pval = ttest_rel(dec_perf[ibin,0,:],dec_perf[ibin,1,:],nan_policy='omit')
        ax.text(bincenter, 0.9, '%s' % get_sig_asterisks(pval,return_ns=False), color='k', ha='center', fontsize=12)

    ax = axes[1]
    ax.plot([0,1],statdata[:2,:],color='k',linewidth=0.25)
    ax.scatter(np.zeros(nSessions),statdata[0,:],color='k',marker='.',s=markersize)
    ax.scatter(np.ones(nSessions),statdata[1,:],color='k',marker='.',s=markersize)
    ax.scatter([0,1],np.nanmean(statdata[:2,:],axis=1),color=clrs_arealabels[:2],marker='o',s=markersize*2,zorder=10)
    ax.errorbar([0,1],np.nanmean(statdata[:2,:],axis=1),np.nanstd(statdata[:2,:],axis=1)/np.sqrt(nSessions),
                color='k',capsize=0,elinewidth=2,zorder=0)

    t,pval = ttest_rel(statdata[0,:],statdata[1,:],nan_policy='omit')
    ax.text(0.5, 0.9, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', transform=ax.transAxes,fontsize=12)
    ax.set_xticks([0,1],arealabels[:2])
    ax.axhline(0.5,color='k',linestyle='--',linewidth=1)
    ax.text(0.5, 0.46, 'Chance', color='gray', ha='center', fontsize=8)
    ax.set_xlim([-0.3,1.3])

    ax = axes[2]
    handles = []
    for ial, arealabel in enumerate(arealabels[2:]):
        idx = ial + 2
        handles.append(shaded_error(sbins,dec_perf[:,idx,:].T,color=clrs_arealabels[idx],alpha=0.5,linewidth=1.5,error='sem',ax=ax))
        for ises in range(nSessions):
            ax.plot(sbins,dec_perf[:,idx,ises],color=clrs_arealabels[idx],linewidth=0.2)

    for ibin, bincenter in enumerate(sbins):
        t,pval = ttest_rel(dec_perf[ibin,2,:],dec_perf[ibin,3,:],nan_policy='omit')
        ax.text(bincenter, 0.9, '%s' % get_sig_asterisks(pval,return_ns=False), color='k', ha='center', fontsize=12)
    ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles,labels=arealabels[2:])
    ax.set_xlabel('Position relative to stim (cm)')
    add_stim_resp_win(ax)
    ax.set_xticks([-50,-25,0,25,50,75])
    ax.set_xlim([-50,75])

    ax = axes[3]
    ax.plot([0,1],statdata[2:,:],color='k',linewidth=0.25)
    ax.scatter(np.zeros(nSessions),statdata[2,:],color='k',marker='.',s=markersize)
    ax.scatter(np.ones(nSessions),statdata[3,:],color='k',marker='.',s=markersize)
    ax.scatter([0,1],np.nanmean(statdata[2:,:],axis=1),color=clrs_arealabels[2:],marker='o',s=markersize*2,zorder=10)
    ax.errorbar([0,1],np.nanmean(statdata[2:,:],axis=1),np.nanstd(statdata[2:,:],axis=1)/np.sqrt(nSessions),
                color='k',capsize=0,elinewidth=2,zorder=0)

    t,pval = ttest_rel(statdata[2,:],statdata[3,:],nan_policy='omit')
    ax.text(0.5, 0.9, '%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', transform=ax.transAxes,fontsize=12)
    ax.set_xticks([0,1],arealabels[:2])
    ax.axhline(0.5,color='k',linestyle='--',linewidth=1)
    ax.text(0.5, 0.46, 'Chance', color='gray', ha='center', fontsize=8)
    ax.set_xlim([-0.3,1.3])

    sns.despine(offset=3,trim=True)

    plt.tight_layout()
    return fig

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons

kfold           = 5
# lam             = 0.08
lam             = 1
nmintrialscond  = 10
nmodelfits      = 20 
nminneurons     = 10 #how many neurons in a population to include the session
nsampleneurons  = 20

#%% Decoding threshold stimulus presence:
decode_var      = 'noise'
decode_var      = 'max'
testwin         = [10,30]
# decode_var      = 'choice'
# testwin           = [20,50] 

arealabels      = ['V1unl','V1lab','PMunl','PMlab']
clrs_arealabels = get_clr_area_labeled(arealabels)

dec_perf = decvar_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons)

#%% Set to nan all unlabeled population results if not a matched labeled population:
dec_perf[:,arealabels.index('V1unl'),np.all(np.isnan(dec_perf[:,arealabels.index('V1lab'),:]),axis=0)] = np.nan
dec_perf[:,arealabels.index('PMunl'),np.all(np.isnan(dec_perf[:,arealabels.index('PMlab'),:]),axis=0)] = np.nan

#%% 
fig = plot_dec_perf_arealabel(dec_perf,sbins,arealabels,clrs_arealabels,testwin=testwin)
plt.suptitle('Decoding %s' % decode_var,fontsize=14)
plt.savefig(os.path.join(savedir, 'Dec_%s_V1PM_Labeled_%dsessions.png' % (decode_var,nSessions)), format='png')

#%% Decoding threshold stimulus presence in hits and misses:
decode_var      = 'noise'
# decode_var      = 'max'
testwin         = [10,30]
# decode_var      = 'choice'
# testwin           = [20,50] 

arealabels      = ['V1unl','V1lab','PMunl','PMlab']
clrs_arealabels = get_clr_area_labeled(arealabels)

dec_perf = decvar_hitmiss_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons)



#%% Decode from V1 and PM labeled and unlabeled neurons separately for hits and misses
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
narealabels     = len(arealabels)
clrs_arealabels = get_clr_area_labeled(arealabels)

nlickresponse   = 2



# dec_perf_stim   = np.full((len(sbins),narealabels,nlickresponse,nSessions), np.nan)

# # Loop through each session
# for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
#     for ial, arealabel in enumerate(arealabels):
#         for ilr in range(nlickresponse):
#             # idx_T       = np.isin(ses.trialdata['stimcat'],['C','N'])
#             # idx_T       = np.all((np.isin(ses.trialdata['stimcat'],['C','N']),
#                                 #   ses.trialdata['lickResponse']==ilr,
#                                 #   ses.trialdata['engaged']==1),axis=0)
            
#             idx_cat     = np.logical_or(np.logical_and(ses.trialdata['stimcat']=='N',ses.trialdata['lickResponse']==ilr),
#                                         ses.trialdata['stimcat']=='C')
#             idx_T       = np.all((idx_cat,
#                                   ses.trialdata['engaged']==1),axis=0)

#             idx_nearby  = filter_nearlabeled(ses,radius=50)
#             idx_N       = np.all((ses.celldata['arealabel']==arealabel,
#                             ses.celldata['noise_level']<20,	
#                                     idx_nearby),axis=0)

#             y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
            
#             if np.sum(y==0) >= nmintrialscond and np.sum(y==1) >= nmintrialscond and np.sum(idx_N) >= nminneurons:
#                 for ibin, bincenter in enumerate(sbins):            # Loop through each spatial bin
#                     temp = np.empty(nmodelfits)
#                     for i in range(nmodelfits):
#                         y   = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()

#                         if np.sum(idx_N) >= nsampleneurons:
#                             # idx_Ns = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
#                             idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=False)
#                         else:
#                             idx_Ns  = np.random.choice(np.where(idx_N)[0],size=nsampleneurons,replace=True)

#                         X       = ses.stensor[np.ix_(idx_Ns,idx_T,sbins==bincenter)].squeeze()
#                         X       = X.T

#                         X,y,_   = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

#                         Xb,yb           = balance_trial(X,y,sample_min_trials=10)
#                         # temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,norm_out=True)
#                         temp[i],_,_,_   = my_decoder_wrapper(Xb,yb,model_name='LogisticRegression',kfold=kfold,lam=lam,
#                                                              subtract_shuffle=False,norm_out=False)
#                     dec_perf_stim[ibin,ial,ilr,ises] = np.nanmean(temp)


#%%

dec_perf = decvar_hitmiss_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons)


#%% Show the threshold stimulus decoding performance
fig,axes = plt.subplots(1,4,figsize=(10,3),sharex=False,sharey=True)

statdata    = np.nanmean(dec_perf_stim[(sbins>0) & (sbins<=20),:,:,:],axis=0)
statdata    = np.diff(statdata,axis=1).squeeze() #get the difference between hits and misses
statdata    = np.diff(statdata,axis=1).squeeze() #get the difference between hits and misses
# statdata    = np.diff(statdata,axis=1).squeeze() #get the difference between hits and misses
statdata    = np.nanmean(dec_perf_stim[(sbins>5) & (sbins<=25),:,0,:],axis=0)
statdata    = np.nanmean(dec_perf_stim[(sbins>0) & (sbins<=25),:,1,:],axis=0)
# statdata    = np.nanmean(dec_perf_stim[(sbins>0) & (sbins<=25),:,1,:],axis=0)
# statdata    = np.nanmean(dec_perf_stim[(sbins>0) & (sbins<=20),:,:,:],axis=(0,2))

ax = axes[0]
handles = []
for ial, arealabel in enumerate(arealabels[:2]):
    # for ilr in [0]:
    # for ilr in [1]:
    for ilr in range(2):
        handles.append(shaded_error(sbins,dec_perf_stim[:,ial,ilr,:].T,color=clrs_arealabels[ial],
                                    linestyle=['--','-'][ilr],alpha=0.25,linewidth=1.5,error='sem',ax=ax))
firstlegend = ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles[1::2],labels=arealabels[:2])
leg_lines = [Line2D([0], [0], color='black', linewidth=2, linestyle='-'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='--')]
leg_labels = ['Hits','Misses']
ax.legend(leg_lines, leg_labels, loc='center left', frameon=False, fontsize=9)
ax.add_artist(firstlegend)
ax.set_ylabel('Performance \n (accuracy - shuffle)')
ax.set_xlabel('Position relative to stim (cm)')
ax.set_title('Decoding')
add_stim_resp_win(ax)
ax.set_xticks([-50,-25,0,25,50,75])
ax.set_ylim([-0.1,1])
ax.set_ylim([0.45,1])
ax.set_xlim([-60,80])

ax = axes[1]
ax.plot([0,1],statdata[:2,:],color='k',linewidth=0.5)
ax.scatter(np.zeros(nSessions),statdata[0,:],color='k',marker='.',s=20)
ax.scatter(np.ones(nSessions),statdata[1,:],color='k',marker='.',s=20)

# idx = ncells[1,:] > 20
# t,pval = ttest_rel(statdata[0,idx],statdata[1,idx],nan_policy='omit')

t,pval = ttest_rel(statdata[0,:],statdata[1,:],nan_policy='omit')
ax.text(0.5, 0.9, 'p=%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', transform=ax.transAxes,fontsize=10)
ax.set_xticks([0,1],arealabels[:2])

ax = axes[2]
handles = []
for ial, arealabel in enumerate(arealabels[2:]):
    idx = ial + 2
    for ilr in range(2):
        handles.append(shaded_error(sbins,dec_perf_stim[:,idx,ilr,:].T,color=clrs_arealabels[idx],
                                    linestyle=['--','-'][ilr],alpha=0.25,linewidth=1.5,error='sem',ax=ax))
firstlegend = ax.legend(loc='upper left',fontsize=9,frameon=False,handles=handles[1::2],labels=arealabels[2:])
leg_lines = [Line2D([0], [0], color='black', linewidth=2, linestyle='-'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='--')]
leg_labels = ['Hits','Misses']
ax.legend(leg_lines, leg_labels, loc='center left', frameon=False, fontsize=9)
ax.add_artist(firstlegend)
ax.set_xlabel('Position relative to stim (cm)')
ax.set_title('Decoding')
add_stim_resp_win(ax)
ax.set_xticks([-50,-25,0,25,50,75])
ax.set_xlim([-60,80])

ax = axes[3]
ax.plot([0,1],statdata[2:,:],color='k',linewidth=0.5)
ax.scatter(np.zeros(nSessions),statdata[2,:],color='k',marker='.',s=20)
ax.scatter(np.ones(nSessions),statdata[3,:],color='k',marker='.',s=20)
t,pval = ttest_rel(statdata[2,:],statdata[3,:],nan_policy='omit')
ax.text(0.5, 0.9, 'p=%s' % get_sig_asterisks(pval,return_ns=True), color='k', ha='center', transform=ax.transAxes,fontsize=10)
ax.set_xticks([0,1],arealabels[2:])

plt.tight_layout()


#%%
######  #######  #####  ####### ######  #######     #####  #     # ####### ###  #####  #######    #          #    ######  ####### #       ####### ######  
#     # #       #     # #     # #     # #          #     # #     # #     #  #  #     # #          #         # #   #     # #       #       #       #     # 
#     # #       #       #     # #     # #          #       #     # #     #  #  #       #          #        #   #  #     # #       #       #       #     # 
#     # #####   #       #     # #     # #####      #       ####### #     #  #  #       #####      #       #     # ######  #####   #       #####   #     # 
#     # #       #       #     # #     # #          #       #     # #     #  #  #       #          #       ####### #     # #       #       #       #     # 
#     # #       #     # #     # #     # #          #     # #     # #     #  #  #     # #          #       #     # #     # #       #       #       #     # 
######  #######  #####  ####### ######  #######     #####  #     # ####### ###  #####  #######    ####### #     # ######  ####### ####### ####### ######  

#%% 












