# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script analyzes whether spatial or temporal alignment of the neural 
activity captures the relevant feature encoding better.
"""

#%% Import packages
import os
os.chdir('c:\\Python\\molanalysis\\')
import numpy as np
from tqdm import tqdm
import copy

from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

#Decoding libs:
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LOGR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
from loaddata.get_data_folder import get_local_drive
from utils.plot_lib import *
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.regress_lib import * # get support functions for decoding

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - VR\\Stim\\'
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Spatial\\')
# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - DN regression\\'

#%% 
session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
# session_list = np.array([['LPE12013', '2024_04_26']])
# session_list = np.array([['LPE11997', '2024_04_16']])
# session_list = np.array([['LPE12013', '2024_04_25']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions


#%% 
sessions,nSessions  = filter_sessions(protocol,load_behaviordata=True,load_calciumdata=True,
                                      calciumversion=calciumversion,min_cells=100)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Remove sessions LPE10884 that are too bad:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE10884_2023_12_14','LPE10884_2023_12_15','LPE10884_2024_01_11',
                                                                'LPE10884_2024_01_16','LPE11622_2024_02_22']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% 
# for i in range(nSessions):
#     sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -60  #pre cm
s_post      = 60   #post cm
sbinsize     = 5     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=sbinsize,method='binmean')

#%% Decoding performance across space or across time:

kfold = 5

crossperf_stim = np.full((nSessions,len(sbins),len(sbins)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
    idx_T = np.isin(ses.trialdata['stimcat'],['C','M'])
    # idx_T = np.isin(ses.trialdata['stimcat'],['C','N'])
    idx_N = ses.celldata['roi_name']=='V1'
    # idx_N = ses.celldata['roi_name']=='PM'

    if np.sum(idx_T) > 50:
        # Get the maximum signal vs catch for this session
        y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()

        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        X = np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,((sbins>-5) & (sbins<20)).astype(bool))],axis=2)
        X = X.T

        #PREP X,y
        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        lam = find_optimal_lambda(X,y,model_name='LogisticRegression',kfold=5)

        model = LOGR(penalty='l1', solver='liblinear', C=lam)
  
        # Define the number of folds for cross-validation
        kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

        # Loop through each spatial bin
        for ibin, ibin_center in enumerate(sbins):
            Xi_orig = ses.stensor[np.ix_(idx_N,idx_T,sbins==ibin_center)].squeeze()
            Xi_orig = Xi_orig.T

            for jbin, jbin_center in enumerate(sbins):
                Xj = ses.stensor[np.ix_(idx_N,idx_T,sbins==jbin_center)].squeeze()
                Xj = Xj.T
                
                Xi = copy.deepcopy(Xi_orig)

                #PREP Xi and Xj
                idx_N_all_nan = np.logical_or(np.all(np.isnan(Xi),axis=0),np.all(np.isnan(Xj),axis=0))
                Xi = Xi[:,~idx_N_all_nan]
                Xj = Xj[:,~idx_N_all_nan]
                
                Xi = zscore(Xi, axis=0,nan_policy='omit')
                Xj = zscore(Xj, axis=0,nan_policy='omit')
                
                idx_T_any_nan = np.logical_or(np.any(np.isnan(Xi),axis=1),np.any(np.isnan(Xj),axis=1))
                Xi = Xi[~idx_T_any_nan,:]
                Xj = Xj[~idx_T_any_nan,:]
                
                y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
                y = y[~idx_T_any_nan]

                # Initialize an array to store the decoding performance
                performance = np.full((kfold,), np.nan)
                performance_shuffle = np.full((kfold,), np.nan)

                # Loop through each fold
                for ifold, (train_index, test_index) in enumerate(kf.split(Xi)):
                    # Split the data into training and testing sets
                    X_train, X_test = Xi[train_index], Xj[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # Train a classification model on the training data with regularization
                    model.fit(X_train, y_train)

                    # Make predictions on the test data
                    y_pred = model.predict(X_test)

                    # Calculate the decoding performance for this fold
                    performance[ifold] = accuracy_score(y_test, y_pred)

                    # Shuffle the labels and calculate the decoding performance for this fold
                    np.random.shuffle(y_train)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    performance_shuffle[ifold] = accuracy_score(y_test, y_pred)
                
                # subtract the shuffling performance from the average perf
                performance_avg = np.mean(performance - performance_shuffle)
                # normalize to maximal range of performance (between shuffle and 1)
                performance_avg = performance_avg / (1-np.mean(performance_shuffle))

                # Calculate the average decoding performance across folds
                crossperf_stim[ises,ibin,jbin] = performance_avg


#%% Show the decoding performance
fig,axes = plt.subplots(1,2,figsize=(8,3))

sesidx = 1

ax = axes[0]
im = ax.pcolor(sbins,sbins,crossperf_stim[sesidx,:,:],vmin=-1,vmax=1,cmap='bwr')
ax.set_xticks([-50,-25,0,25,50])
ax.set_yticks([-50,-25,0,25,50])

ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
ax.axvline(x=45, color='b', linestyle='--', linewidth=1)

ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax.axhline(y=20, color='k', linestyle='--', linewidth=1)
ax.axhline(y=25, color='b', linestyle='--', linewidth=1)
ax.axhline(y=45, color='b', linestyle='--', linewidth=1)

ax.set_xlabel('Train Position (cm)')
ax.set_ylabel('Test Position (cm)')
ax.set_title('Example Session \n%s' % sessions[sesidx].sessiondata['session_id'][0],fontsize=11)
fig.colorbar(im, ax=ax,shrink=0.5,label='Dec. Perf.')

ax = axes[1]
im = ax.pcolor(sbins,sbins,np.nanmean(crossperf_stim,axis=0),vmin=-1,vmax=1,cmap='bwr')
ax.set_xticks([-50,-25,0,25,50])
ax.set_yticks([-50,-25,0,25,50])
ax.set_yticklabels([])

ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
ax.axvline(x=45, color='b', linestyle='--', linewidth=1)

ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax.axhline(y=20, color='k', linestyle='--', linewidth=1)
ax.axhline(y=25, color='b', linestyle='--', linewidth=1)
ax.axhline(y=45, color='b', linestyle='--', linewidth=1)

ax.set_xlabel('Train Position (cm)')
ax.set_title('Session Average\n %d sessions' % nSessions,fontsize=11)

fig.colorbar(im, ax=ax,shrink=0.5,label='Dec. Perf.')

plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'CrossSpatial_DecodingPerformance_NoiseVsCatch.png'), format='png')
plt.savefig(os.path.join(savedir, 'CrossSpatial_DecodingPerformance_MaxVsCatch_V1_%dsessions.png' % nSessions), format='png')

#%% Decoding performance across space or across time:

kfold = 5

crossperf_choice = np.full((nSessions,len(sbins),len(sbins)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
    # idx_T = np.isin(ses.trialdata['stimcat'],['C','M'])
    # idx_T = np.isin(ses.trialdata['stimcat'],['N','M'])
    idx_T = np.all((np.isin(ses.trialdata['stimcat'],['N','M']),
                    ses.trialdata['engaged']==1),axis=0)
    # idx_T = np.isin(ses.trialdata['stimcat'],['C','N'])
    idx_N = ses.celldata['roi_name']=='V1'

    if np.sum(idx_T) > 50:
        # Get the maximum signal vs catch for this session
        y = (ses.trialdata['lickResponse'][idx_T] == 1).to_numpy()

        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        X = np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,((sbins>20) & (sbins<45)).astype(bool))],axis=2)
        X = X.T

        #PREP X,y
        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        lam = find_optimal_lambda(X,y,model_name='LogisticRegression',kfold=5)

        model = LOGR(penalty='l1', solver='liblinear', C=lam)
  
        # Define the number of folds for cross-validation
        kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

        # Loop through each spatial bin
        for ibin, ibin_center in enumerate(sbins):
            Xi_orig = ses.stensor[np.ix_(idx_N,idx_T,sbins==ibin_center)].squeeze()
            Xi_orig = Xi_orig.T

            for jbin, jbin_center in enumerate(sbins):
                Xj = ses.stensor[np.ix_(idx_N,idx_T,sbins==jbin_center)].squeeze()
                Xj = Xj.T
                
                Xi = copy.deepcopy(Xi_orig)

                #PREP Xi and Xj
                idx_N_all_nan = np.logical_or(np.all(np.isnan(Xi),axis=0),np.all(np.isnan(Xj),axis=0))
                Xi = Xi[:,~idx_N_all_nan]
                Xj = Xj[:,~idx_N_all_nan]
                
                Xi = zscore(Xi, axis=0,nan_policy='omit')
                Xj = zscore(Xj, axis=0,nan_policy='omit')
                
                idx_T_any_nan = np.logical_or(np.any(np.isnan(Xi),axis=1),np.any(np.isnan(Xj),axis=1))
                Xi = Xi[~idx_T_any_nan,:]
                Xj = Xj[~idx_T_any_nan,:]
                
                y = (ses.trialdata['lickResponse'][idx_T] == 1).to_numpy()
                y = y[~idx_T_any_nan]

                # Initialize an array to store the decoding performance
                performance = np.full((kfold,), np.nan)
                performance_shuffle = np.full((kfold,), np.nan)

                # Loop through each fold
                for ifold, (train_index, test_index) in enumerate(kf.split(Xi)):
                    # Split the data into training and testing sets
                    X_train, X_test = Xi[train_index], Xj[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # Train a classification model on the training data with regularization
                    model.fit(X_train, y_train)

                    # Make predictions on the test data
                    y_pred = model.predict(X_test)

                    # Calculate the decoding performance for this fold
                    performance[ifold] = accuracy_score(y_test, y_pred)

                    # Shuffle the labels and calculate the decoding performance for this fold
                    np.random.shuffle(y_train)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    performance_shuffle[ifold] = accuracy_score(y_test, y_pred)
                
                # subtract the shuffling performance from the average perf
                performance_avg = np.mean(performance - performance_shuffle)
                # normalize to maximal range of performance (between shuffle and 1)
                performance_avg = performance_avg / (1-np.mean(performance_shuffle))

                # Calculate the average decoding performance across folds
                crossperf_choice[ises,ibin,jbin] = performance_avg


#%% Show the decoding performance
fig,axes = plt.subplots(1,2,figsize=(8,3.3))

sesidx = 1

ax = axes[0]
im = ax.pcolor(sbins,sbins,crossperf_choice[sesidx,:,:],vmin=-1,vmax=1,cmap='bwr')
ax.set_xticks([-50,-25,0,25,50])
ax.set_yticks([-50,-25,0,25,50])

ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
ax.axvline(x=45, color='b', linestyle='--', linewidth=1)

ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax.axhline(y=20, color='k', linestyle='--', linewidth=1)
ax.axhline(y=25, color='b', linestyle='--', linewidth=1)
ax.axhline(y=45, color='b', linestyle='--', linewidth=1)

ax.set_xlabel('Train Position (cm)')
ax.set_ylabel('Test Position (cm)')
ax.set_title('Example Session \n%s' % sessions[sesidx].sessiondata['session_id'][0],fontsize=11)
fig.colorbar(im, ax=ax,shrink=0.5,label='Dec. Perf.')

ax = axes[1]
im = ax.pcolor(sbins,sbins,np.nanmean(crossperf_choice,axis=0),vmin=-1,vmax=1,cmap='bwr')
ax.set_xticks([-50,-25,0,25,50])
ax.set_yticks([-50,-25,0,25,50])
ax.set_yticklabels([])

ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
ax.axvline(x=45, color='b', linestyle='--', linewidth=1)

ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax.axhline(y=20, color='k', linestyle='--', linewidth=1)
ax.axhline(y=25, color='b', linestyle='--', linewidth=1)
ax.axhline(y=45, color='b', linestyle='--', linewidth=1)

ax.set_xlabel('Train Position (cm)')
ax.set_title('Session Average\n %d sessions' % nSessions,fontsize=11)

fig.colorbar(im, ax=ax,shrink=0.5,label='Dec. Perf.')

plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'CrossSpatial_DecodingPerformance_NoiseVsCatch.png'), format='png')
plt.savefig(os.path.join(savedir, 'CrossSpatial_DecodingPerformance_Choice_MaxNoise.png'), format='png')
