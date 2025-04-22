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

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.psth import *
from utils.plot_lib import * #get all the fixed color schemes
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

#%% 
sessions,nSessions,sbins = load_neural_performing_sessions()

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
# lam             = 0.08
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


# %% 
######  #######  #####  ####### ######  #######    #     #    #    ######   #####     ######  ####### ######   #####  
#     # #       #     # #     # #     # #          #     #   # #   #     # #     #    #     # #     # #     # #     # 
#     # #       #       #     # #     # #          #     #  #   #  #     # #          #     # #     # #     # #       
#     # #####   #       #     # #     # #####      #     # #     # ######   #####     ######  #     # ######   #####  
#     # #       #       #     # #     # #           #   #  ####### #   #         #    #       #     # #             # 
#     # #       #     # #     # #     # #            # #   #     # #    #  #     #    #       #     # #       #     # 
######  #######  #####  ####### ######  #######       #    #     # #     #  #####     #       ####### #        #####  


#%% Parameters for decoding from size-matched populations of the different areas
arealabels      = ['V1unl','PMunl','ALunl','RSPunl']
clrs_arealabels = get_clr_area_labeled(arealabels)
narealabels     = len(arealabels)
kfold           = 5
# lam             = 0.08
lam             = 1
nmintrialscond  = 10
nmodelfits      = 10
nminneurons     = 100 #how many neurons in a population to include the session
nsampleneurons  = 100

#%%
ncells = np.empty((nSessions,narealabels))
for i,ses in enumerate(sessions):
    for ial, arealabel in enumerate(arealabels):
        ncells[i,ial] = np.sum(ses.celldata['arealabel']==arealabel)
plt.hist(ncells.flatten(),np.arange(0,1100,25))

#%% Decoding threshold stimulus presence:
decode_var      = 'noise'
decode_var      = 'max'
# decode_var      = 'choice'
# testwin           = [20,50] 

dec_perf = decvar_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons,filter_nearby=False)

#%% 
testwin         = [0,25]
fig = plot_dec_perf_area(dec_perf,sbins,arealabels,clrs_arealabels,testwin=testwin)
plt.suptitle('Decoding %s' % decode_var,fontsize=14)
plt.savefig(os.path.join(savedir, 'Dec_%s_Area_Unlabeled_%dsessions.png' % (decode_var,nSessions)), format='png')

#%% Decoding threshold stimulus strength from V1 and PM labeled and unlabeled neurons separately:
decode_var      = 'signal'
testwin         = [0,25]
nmodelfits      = 10
nmintrialscond  = 10
nminneurons     = 100

lam             = 100

dec_perf = decvar_cont_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons)

#%% 
testwin         = [0,25]
fig = plot_dec_perf_area(dec_perf,sbins,arealabels,clrs_arealabels,testwin=testwin)
plt.suptitle('Decoding %s' % decode_var,fontsize=14)
plt.savefig(os.path.join(savedir, 'Dec_%s_Area_Unlabeled_%dsessions.png' % (decode_var,nSessions)), format='png')

#%% Start decoding from differently sized populations: 





#%%

######  #######  #####  ####### ######  #######    #     #    #    ######   #####     #          #    ######  ####### #       ####### ######  
#     # #       #     # #     # #     # #          #     #   # #   #     # #     #    #         # #   #     # #       #       #       #     # 
#     # #       #       #     # #     # #          #     #  #   #  #     # #          #        #   #  #     # #       #       #       #     # 
#     # #####   #       #     # #     # #####      #     # #     # ######   #####     #       #     # ######  #####   #       #####   #     # 
#     # #       #       #     # #     # #           #   #  ####### #   #         #    #       ####### #     # #       #       #       #     # 
#     # #       #     # #     # #     # #            # #   #     # #    #  #     #    #       #     # #     # #       #       #       #     # 
######  #######  #####  ####### ######  #######       #    #     # #     #  #####     ####### #     # ######  ####### ####### ####### ######  



#%% ########################## Load data #######################
protocol            = ['DN']
calciumversion      = 'deconv'

sessions,nSessions  = filter_sessions(protocol,load_calciumdata=True,load_behaviordata=True,
                                      load_videodata=True,calciumversion=calciumversion,min_lab_cells_PM=10,
                                      min_lab_cells_V1=10)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -60  #pre cm
s_post      = 80   #post cm
sbinsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=sbinsize,method='binmean')


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
clrs_arealabels = get_clr_area_labeled(arealabels)

kfold           = 5
# lam             = 0.08
lam             = 1
nmintrialscond  = 10
nmodelfits      = 10
nminneurons     = 20 #how many neurons in a population to include the session
nsampleneurons  = 20

#%% Decoding threshold stimulus presence:
decode_var      = 'noise'
# decode_var      = 'max'

# decode_var      = 'choice'
# testwin           = [20,50] 

dec_perf = decvar_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons,filter_nearby=True)

#%% Set to nan all unlabeled population results if not a matched labeled population:
dec_perf[:,arealabels.index('V1unl'),np.all(np.isnan(dec_perf[:,arealabels.index('V1lab'),:]),axis=0)] = np.nan
dec_perf[:,arealabels.index('PMunl'),np.all(np.isnan(dec_perf[:,arealabels.index('PMlab'),:]),axis=0)] = np.nan

#%% 
testwin         = [10,20]
# testwin         = [0,25]
# testwin         = [20,40]
fig = plot_dec_perf_arealabel(dec_perf,sbins,arealabels,clrs_arealabels,testwin=testwin)
plt.suptitle('Decoding %s' % decode_var,fontsize=14)
plt.savefig(os.path.join(savedir, 'Dec_%s_V1PM_Labeled_%dsessions.png' % (decode_var,nSessions)), format='png')

#%% 








#%% Decoding threshold stimulus presence in hits and misses:
decode_var      = 'noise'
# decode_var      = 'max'
testwin         = [5,25]
# testwin         = [10,50]

arealabels      = ['V1unl','V1lab','PMunl','PMlab']
clrs_arealabels = get_clr_area_labeled(arealabels)

dec_perf = decvar_hitmiss_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons)

#%% Set to nan all unlabeled population results if not a matched labeled population:
dec_perf[:,arealabels.index('V1unl'),:,np.all(np.isnan(dec_perf[:,arealabels.index('V1lab'),:,:]),axis=(0,1))] = np.nan
dec_perf[:,arealabels.index('PMunl'),:,np.all(np.isnan(dec_perf[:,arealabels.index('PMlab'),:,:]),axis=(0,1))] = np.nan

#%% Show figure of decoding performance for hits and misses:
testwin         = [0,30]
fig = plot_dec_perf_hitmiss_arealabel(dec_perf,sbins,arealabels,clrs_arealabels,testwin=testwin)
plt.suptitle('Decoding %s' % decode_var,fontsize=14)
plt.savefig(os.path.join(savedir, 'Dec_%s_hitmiss_V1PM_Labeled_%dsessions.png' % (decode_var,nSessions)), format='png')

#%% 



#%% Decoding threshold stimulus strength from V1 and PM labeled and unlabeled neurons separately:
decode_var      = 'signal'
testwin         = [0,25]
nmodelfits      = 50
nmintrialscond  = 20
nminneurons     = 20

lam             = 100

dec_perf = decvar_cont_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons)

#%% Set to nan all unlabeled population results if not a matched labeled population:
dec_perf[:,arealabels.index('V1unl'),np.all(np.isnan(dec_perf[:,arealabels.index('V1lab'),:]),axis=0)] = np.nan
dec_perf[:,arealabels.index('PMunl'),np.all(np.isnan(dec_perf[:,arealabels.index('PMlab'),:]),axis=0)] = np.nan

#%% Make figure and save:
fig = plot_dec_perf_arealabel(dec_perf,sbins,arealabels,clrs_arealabels,testwin=testwin)
plt.suptitle('Decoding %s' % decode_var,fontsize=14)
plt.savefig(os.path.join(savedir, 'Dec_%s_V1PM_Labeled_%dsessions.png' % (decode_var,nSessions)), format='png')


#%% Decoding threshold stimulus strength from V1 and PM labeled and unlabeled neurons separately:
decode_var      = 'signal'
testwin         = [0,25]
nmodelfits      = 10
nmintrialscond  = 20
nminneurons     = 20

lam             = 100

dec_perf = decvar_cont_hitmiss_from_arealabel_wrapper(sessions,sbins,arealabels,var=decode_var,nmodelfits=nmodelfits,
                                         kfold=kfold,lam=lam,nmintrialscond=nmintrialscond,
                                         nminneurons=nminneurons,nsampleneurons=nsampleneurons,filter_nearby=True)

#%% Set to nan all unlabeled population results if not a matched labeled population:
dec_perf[:,arealabels.index('V1unl'),:,np.all(np.isnan(dec_perf[:,arealabels.index('V1lab'),:,:]),axis=(0,1))] = np.nan
dec_perf[:,arealabels.index('PMunl'),:,np.all(np.isnan(dec_perf[:,arealabels.index('PMlab'),:,:]),axis=(0,1))] = np.nan

#%% Show figure of decoding performance for hits and misses:
testwin         = [0,25]
fig = plot_dec_perf_hitmiss_arealabel(dec_perf,sbins,arealabels,clrs_arealabels,testwin=testwin)
plt.suptitle('Decoding %s' % decode_var,fontsize=14)
plt.savefig(os.path.join(savedir, 'Dec_%s_hitmiss_V1PM_Labeled_%dsessions.png' % (decode_var,nSessions)), format='png')

