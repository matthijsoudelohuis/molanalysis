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
import pandas as pd
from tqdm import tqdm

from scipy.stats import zscore, ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
from loaddata.get_data_folder import get_local_drive
from utils.plot_lib import * #get all the fixed color schemes
from utils.plot_lib import *
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.regress_lib import * # get support functions for decoding

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - VR\\Stim\\'
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Alignment\\')
# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - DN regression\\'

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

#%% ############################### Spatial Resp Mat #################################
for i in range(nSessions):
    sessions[i].respmat = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                                sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)


#%% Decoding performance across space or across time:

model_names = ['LOGR','SVM','LDA','GBC']
# model_names = ['LogisticRegression','LDA']
# model_names = ['SVM','LDA']
model_names = ['GBC']
    
nModels     = len(model_names)

perfmat     = np.full((nSessions,nModels), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
    idx_T = np.isin(ses.trialdata['stimcat'],['C','M'])
    idx_T = np.isin(ses.trialdata['stimcat'],['C','N'])
    idx_N = ses.celldata['roi_name']=='V1'

    if np.sum(idx_T) > 50:
        # Get the maximum signal vs catch for this session
        y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()

        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        X = ses.respmat[np.ix_(idx_N,idx_T)]
        X = X.T
        
        X = X[:,~np.all(np.isnan(X),axis=0)] #
        idx_nan = ~np.all(np.isnan(X),axis=1)
        X = X[idx_nan,:]
        y = y[idx_nan]
        X[np.isnan(X)] = np.nanmean(X, axis=None)
        X = zscore(X, axis=1)
        X[np.isnan(X)] = np.nanmean(X, axis=None)

        # Loop through each spatial bin
        for imodel, model_name in enumerate(model_names):
            
            # Calculate the average decoding performance across folds
            perfmat[ises,imodel] = my_classifier_wrapper(X,y,model_name=model_name,kfold=5,lam=None,norm_out=True)

#%% Show the decoding performance for different models - max vs catch
fig,ax = plt.subplots(1,1,figsize=(3,3))
ax.plot(perfmat.T,marker='o',linestyle='-',markersize=5)
ax.plot(np.mean(perfmat.T,axis=1),marker='o',linestyle='-',markersize=7,linewidth=2,color='k')
ax.set_xticks(np.arange(nModels))
ax.set_xticklabels(model_names)
ax.set_yticks(np.arange(0,1.1,0.05))
ax.set_ylim([0.9,1.02])
ax.set_title('Classifier Comparison')
ax.set_ylabel('Decoding Performance \n (accuracy - shuffle)')
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'DecodingPerformance_ModelComparison_MaxVsCatch.png'), format='png')

#%% Show the decoding performance for different models - noise vs catch
fig,ax = plt.subplots(1,1,figsize=(3,3))
ax.plot(perfmat.T,marker='o',linestyle='-',markersize=5)
ax.plot(np.mean(perfmat.T,axis=1),marker='o',linestyle='-',markersize=7,linewidth=2,color='k')
ax.set_xticks(np.arange(nModels))
ax.set_xticklabels(model_names)
ax.set_yticks(np.arange(0,1.1,0.1))
ax.set_ylim([0,1.02])
ax.set_title('Classifier Comparison')
ax.set_ylabel('Decoding Performance \n (accuracy - shuffle)')
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'DecodingPerformance_ModelComparison_NoiseVsCatch.png'), format='png')

