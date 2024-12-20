# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
os.chdir('c:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
from tqdm import tqdm

# from loaddata import * #get all the loading data functions (filter_sessions,load_sessions)
from loaddata.session_info import filter_sessions,load_sessions

from scipy import stats
from scipy.stats import zscore
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score as AUC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler

from loaddata.get_data_folder import get_local_drive
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches
from utils.plotting_style import * #get all the fixed color schemes
from matplotlib.lines import Line2D
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
# session_list = np.array([['LPE11998', '2024_04_23']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list        = np.array([['LPE12013','2024_04_25']])
# session_list = np.array([['LPE09829', '2023_03_29'],
#                         ['LPE09829', '2023_03_30'],
#                         ['LPE09829', '2023_03_31']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - VR\\Stim\\'
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Encoding\\')
# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - DN regression\\'

#%% 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 60   #post cm
binsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')

## Compute average response in stimulus response zone:
for i in range(nSessions):
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

for i in range(nSessions):
    temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
                                    sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)


#%% #################### Compute spatial runspeed ####################################
for ises,ses in enumerate(sessions): # running across the trial:
    sessions[ises].behaviordata['runspeed'] = medfilt(sessions[ises].behaviordata['runspeed'], kernel_size=51)
    [sessions[ises].runPSTH,_]     = calc_runPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)
    [sessions[ises].lickPSTH,_]    = calc_lickPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)

#%% #################### Compute encoding of variables in single neurons  ####################################

ises = 0
example_cell_ids = ['LPE12385_2024_06_15_0_0075',
'LPE12385_2024_06_15_0_0126',
'LPE12385_2024_06_15_0_0105', # reduction upon stimulus zone
'LPE12385_2024_06_15_0_0114', # noise trial specific response
'LPE12385_2024_06_15_0_0031', # noise trial specific response
'LPE12385_2024_06_15_0_0158',
'LPE12385_2024_06_15_1_0001',
'LPE12385_2024_06_15_0_0046',
'LPE12385_2024_06_15_2_0248', #non stationary activity (drift?)
'LPE12385_2024_06_15_2_0499',
'LPE12385_2024_06_15_6_0089',#non stationary activity (drift?)
'LPE12385_2024_06_15_2_0353'] #variable responsiveness

#%% Show example neurons that are correlated either to the stimulus signal, lickresponse or to running speed:
ises = 0
# for iN in range(900,1100):
for iN in np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]:
    plot_snake_neuron_sortnoise(sessions[ises].stensor[iN,:,:],sbins,sessions[ises])
    plt.suptitle(sessions[ises].celldata['cell_id'][iN],fontsize=16,y=0.96)

# ises = 0
# example_cell_ids = ['LPE12385_2024_06_15_0_0098', #hit specific activity?
#             'LPE12385_2024_06_15_1_0075'] #some structure


#%%
example_cell_ids = ['LPE12385_2024_06_15_0_0075',
'LPE12385_2024_06_15_0_0126',
'LPE12385_2024_06_15_0_0105', # reduction upon stimulus zone
'LPE12385_2024_06_15_0_0114', # noise trial specific response, very nice one
'LPE12385_2024_06_15_0_0183',
'LPE12385_2024_06_15_3_0016',
'LPE12385_2024_06_15_0_0031', # noise trial specific response
'LPE12385_2024_06_15_1_0075', # hit specific activity?
'LPE12385_2024_06_15_6_0027', # hit specific activity?
'LPE12385_2024_06_15_7_0212', # hit specific activity?
'LPE12385_2024_06_15_1_0475', # very clean response
'LPE12385_2024_06_15_2_0099', # nice responses
'LPE12385_2024_06_15_2_0499'] #variable responsiveness

# 'LPE12385_2024_06_15_0_0129'

# example_cell_ids = np.randsom.choice(sessions[0].celldata['cell_id'],size=8,replace=False)

fig = plot_mean_activity_example_neurons(sessions[ises].stensor,sbins,sessions[ises],example_cell_ids)



#%% Run encoding model: 

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import zscore

# for ises, ses in tqdm(enumerate(sessions),desc='Decoding response across sessions'):

nspatbins   = len(sbins)
variables   = ['signal','lickresponse', 'runspeed']
nvars       = len(variables)

N           = len(ses.celldata)
# N = 10
weights     = np.empty((N,nspatbins,nvars))
r2_cv       = np.empty((N,nspatbins))

# Define the variables to use for decoding

# Define the number of folds for cross-validation
kfold = 5

# Loop across neurons:
for iN in tqdm(range(N),desc='Fitting encoding model across neurons'):

    # X = np.stack((ses.trialdata['signal'][idx].to_numpy(),
    #               ses.trialdata['lickResponse'][idx].to_numpy(),
    #               ses.runPSTH[idx,ibin]), axis=1)
    # X = np.nanmean(X, axis=1)
    # X = zscore(X, axis=0)
    # X = X[:,np.all(~np.isnan(X),axis=0)]
    # X = X[:,np.all(~np.isinf(X),axis=0)]

    # # Find the optimal regularization strength (lambda)
    # lambdas = np.logspace(-4, 4, 10)
    # cv_scores = np.zeros((len(lambdas),))
    # for ilambda, lambda_ in enumerate(lambdas):
    #     model = LogisticRegression(penalty='l1', solver='liblinear', C=lambda_)
    #     scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    #     cv_scores[ilambda] = np.mean(scores)
    # optimal_lambda = lambdas[np.argmax(cv_scores)]
    # # print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))
    # optimal_lambda = np.clip(optimal_lambda, 0.03, 166)
    optimal_lambda = 1

    # Loop through each spatial bin
    for ibin, bincenter in enumerate(sbins):
        idx = np.all((ses.trialdata['engaged']==1,
                ses.trialdata['stimcat']=='N',
                ~np.isnan(ses.stensor[iN,:,ibin])), axis=0)

        if np.sum(idx) > 50:
            # Get the neural response data for this bin
            y = ses.stensor[iN,idx,ibin]

            # Define the X and y variables
            X = np.stack((ses.trialdata['signal'][idx].to_numpy(),
                ses.trialdata['lickResponse'][idx].to_numpy(),
                ses.runPSTH[idx,ibin]), axis=1)
            
            # X = np.stack((ses.runPSTH[idx,ibin], ses.pupilPSTH[idx,ibin], ses.videomePSTH[idx,ibin], ses.lickPSTH[idx,ibin]), axis=1)
            X = zscore(X, axis=0)

            # Define the k-fold cross-validation object
            kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
            
            # Initialize an array to store the decoding performance for each fold
            fold_performance = np.zeros((kfold,))
            fold_performance_shuffle = np.zeros((kfold,))
            fold_weights = np.zeros((kfold,nvars))

            # Loop through each fold
            for ifold, (train_index, test_index) in enumerate(kf.split(X)):
                # Split the data into training and testing sets
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Train a linear regression model on the training data with regularization
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Compute R2 on the test set
                r2_test = model.score(X_test, y_test)
                fold_performance[ifold] = r2_test
                fold_weights[ifold,:] = model.coef_
                # r2 = model.score(x, y)
                # model = LogisticRegression(penalty='l1',solver='liblinear',C=optimal_lambda)
                # model.fit(X_train, y_train)
                
                # # Make predictions on the test data
                # y_pred = model.predict(X_test)
                
                # # Calculate the decoding performance for this fold
                # fold_performance[ifold] = accuracy_score(y_test, y_pred)
                
                # # Shuffle the labels and calculate the decoding performance for this fold
                # np.random.shuffle(y_train)
                # model.fit(X_train, y_train)
                # y_pred = model.predict(X_test)
                # fold_performance_shuffle[ifold] = accuracy_score(y_test, y_pred)
            
            # Calculate the average decoding performance across folds
            r2_cv[ises,ibin] = np.mean(fold_performance - fold_performance_shuffle)
            weights[iN,ibin,:] = np.mean(fold_weights, axis=0)

#%% Show the decoding performance

labeled     = ['unl','lab']
nlabels     = 2
areas       = np.unique(sessions[0].celldata['roi_name'])
nareas      = len(areas)

clrs_vars = sns.color_palette('inferno', 3)

fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2))

# plt.plot(sbins,r2_cv.mean(axis=0),color='k',linewidth=2)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax = axes[ilab,iarea]
        idx = np.all((sessions[0].celldata['roi_name']==area, sessions[0].celldata['labeled']==label), axis=0)
        if np.sum(idx) > 0:
            for ivar,var in enumerate(variables):
                # plt.plot(sbins,r2_cv[idx,iarea],color=clrs[ilab],linewidth=1)
                ax.plot(sbins,np.nanmean(np.abs(weights[idx,:,ivar]),axis=0),color=clrs_vars[ivar],linewidth=2,label=var)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
        ax.axvline(x=45, color='b', linestyle='--', linewidth=1)
        ax.set_xlim([-80,60])
        if ilab == 0:
            ax.set_title(area)
        if ilab == 1:
            ax.set_xlabel('Position relative to stim (cm)')
            ax.set_ylabel('Encoding weights')
        ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(savedir, 'EncodingWeights_areas_%s.png') % sessions[ises].sessiondata['session_id'][0], format='png')


