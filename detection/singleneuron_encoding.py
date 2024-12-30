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
# session_list = np.array([['LPE12385', '2024_06_16']])
# session_list = np.array([['LPE11998', '2024_04_23']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list        = np.array([['LPE12013','2024_04_25']])
# session_list        = np.array([['LPE12013','2024_04_26']])
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

from detection.example_cells import get_example_cells

example_cell_ids = get_example_cells(sessions[ises].sessiondata['session_id'][0])


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
# example_cell_ids = np.random.choice(sessions[0].celldata['cell_id'],size=8,replace=False)
# example_cell_ids = ['LPE12385_2024_06_15_0_0126',
# 'LPE12385_2024_06_15_0_0075']

fig = plot_mean_activity_example_neurons(sessions[ises].stensor,sbins,sessions[ises],example_cell_ids)
fig.savefig(os.path.join(savedir,'ExampleNeuronActivity_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')


#%% Run encoding model: 

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score
from scipy.stats import zscore

# for ises, ses in tqdm(enumerate(sessions),desc='Decoding response across sessions'):

#%% 
nspatbins   = len(sbins)
variables   = ['signal','lickresponse','runspeed']
variables   = ['signal','lickresponse','runspeed','trialnumber']
nvars       = len(variables)

#just one example cell:
idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])[0])

#a few example cells:
idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0]))

# #V1 cells:
# idx_N = np.all((np.isin(ses.celldata['roi_name'],'V1'),ses.celldata['noise_level']<10),axis=0)

# All cells:
idx_N = np.ones(len(ses.celldata),dtype=bool)
# idx_N = ses.celldata['noise_level']<10

K           = len(ses.trialdata)
N           = len(ses.celldata)
N_idx       = np.sum(idx_N)
weights     = np.full((N,nspatbins,nvars),np.nan)
r2_cv       = np.full((N,nspatbins),np.nan)
y_hat       = np.full((N,nspatbins,K),np.nan)

# Define the number of folds for cross-validation
kfold = 5

#%% 

idx_T = np.all((ses.trialdata['engaged']==1,
        ses.trialdata['stimcat']=='N',
        ~np.isnan(ses.respmat[0,:])), axis=0)

idx_T = np.all((ses.trialdata['engaged']==1,
        ~np.isnan(ses.respmat[0,:])), axis=0)

# idx_T = np.all((ses.trialdata['engaged']==1,
#                 np.isin(ses.trialdata['stimcat'],['C','M']),
#                 ~np.isnan(ses.respmat[0,:])), axis=0)

y = ses.respmat[np.ix_(idx_N,idx_T)].T
# y = np.log((1+y)/1)
y = zscore(y, axis=0)

# X = np.stack((ses.trialdata['signal'][idx_T].to_numpy(),
#               ses.trialdata['lickResponse'][idx_T].to_numpy(),
#               ses.respmat_runspeed[0,idx_T]), axis=1)

X = np.stack((ses.trialdata['signal'][idx_T].to_numpy(),
              ses.trialdata['lickResponse'][idx_T].to_numpy(),
              ses.respmat_runspeed[0,idx_T],
              ses.trialdata['trialNumber'][idx_T]), axis=1)

# X -= np.nanmean(X, axis=0, keepdims=1)
X = zscore(X, axis=0)

# Find the optimal regularization strength (lambda)
lambdas = np.logspace(-4, 4, 20)
cv_scores = np.zeros((len(lambdas),))
for ilambda, lambda_ in enumerate(lambdas):
    # model = LinearRegression(solver='liblinear', C=lambda_)
    model = Lasso(alpha=lambda_)
    # model = ElasticNet(alpha=lambda_,l1_ratio=0.9)
    # model = Ridge(alpha=lambda_)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    cv_scores[ilambda] = np.median(scores)
optimal_lambda = lambdas[np.argmax(cv_scores)]
print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))

# model = Ridge(alpha=optimal_lambda)
model = Lasso(alpha=optimal_lambda)
# model = ElasticNet(alpha=optimal_lambda,l1_ratio=0.9)

scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')

model.fit(X, y)
r2 = model.score(X, y)
print(r2)


# fig,ax = plt.subplots(1,1,figsize=(3,3))
# plt.plot(lambdas,cv_scores)
# ax.set_xscale('log')

#%% 
y_pred  = model.predict(X)
r2      = r2_score(y, y_pred, multioutput='raw_values')
idx_N   = r2 > 0.1
N_idx   = np.sum(idx_N)

#%% 

# optimal_lambda = np.clip(optimal_lambda, 0.03, 166)
# optimal_lambda = 10
# optimal_lambda = 600

crossval = False

# Loop through each spatial bin
for ibin, bincenter in enumerate(sbins):
    # idx = np.all((ses.trialdata['engaged']==1,
    #         ses.trialdata['stimcat']=='N',
    #         ~np.isnan(ses.stensor[iN,:,ibin])), axis=0)

    if np.sum(idx_T) > 50:
        # Get the neural response data for this bin
        # y = ses.stensor[:,idx,ibin]
        y = ses.stensor[np.ix_(idx_N,idx_T,sbins==bincenter)].squeeze().T

        # Define the X predictors
        X = np.stack((ses.trialdata['signal'][idx_T].to_numpy(),
            ses.trialdata['lickResponse'][idx_T].to_numpy(),
            ses.runPSTH[idx_T,ibin],
            ses.trialdata['trialNumber'][idx_T]), axis=1)

        idx_notnan = ~np.any((np.any(np.isnan(X), axis=1),np.any(np.isinf(X), axis=1),
                           np.any(np.isnan(y), axis=1),np.any(np.isinf(y), axis=1)), axis=0)

        X = X[idx_notnan,:]
        y = y[idx_notnan,:]

        # X -= np.nanmean(X, axis=0, keepdims=1)
        X = zscore(X, axis=0)

        y = zscore(y, axis=0)

        # Train a linear regression model on the training data with regularization
        # model = LinearRegression()
        model = Lasso(alpha=optimal_lambda)
        # model = ElasticNet(alpha=optimal_lambda,l1_ratio=0.9)

        if crossval:
            # Define the k-fold cross-validation object
            kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
            
            # Initialize an array to store the decoding performance for each fold
            fold_r2             = np.zeros((kfold,N_idx))
            # fold_r2_shuffle     = np.zeros((kfold,N))
            fold_weights        = np.zeros((kfold,N_idx,nvars))

            # Loop through each fold
            for ifold, (train_index, test_index) in enumerate(kf.split(X)):
                # Split the data into training and testing sets
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # model = Ridge(alpha=optimal_lambda)
                model.fit(X_train, y_train)

                # Compute R2 on the test set
                y_pred              = model.predict(X_test)
                fold_r2[ifold,:]    = r2_score(y_test, y_pred, multioutput='raw_values')
                
                fold_weights[ifold,:,:]     = model.coef_
                y_hat[np.ix_(idx_N,sbins==bincenter,test_index)] = y_pred.T[:,np.newaxis,:]
                
                # # Shuffle the labels and calculate the decoding performance for this fold
                # np.random.shuffle(y_train)
                # model.fit(X_train, y_train)
                # y_pred = model.predict(X_test)
                # fold_r2_shuffle[ifold] = accuracy_score(y_test, y_pred)
        
            # Calculate the average decoding performance across folds
            r2_cv[idx_N,ibin] = np.nanmean(fold_r2, axis=0)
            weights[idx_N,ibin,:] = np.nanmean(fold_weights, axis=0)

        else:   
            # Without cross-validation
            model.fit(X, y)
            y_pred = model.predict(X)
            r2_cv[idx_N,ibin] = r2_score(y, y_pred, multioutput='raw_values')
            y_hat[np.ix_(idx_N,sbins==bincenter,idx_T)] = y_pred.T[:,np.newaxis,:]
            weights[idx_N,ibin,:] = model.coef_


#%% Show the encoding weights
labeled     = ['unl','lab']
nlabels     = 2
areas       = np.unique(sessions[0].celldata['roi_name'])
areas       = ['V1','PM','AL','RSP']	
nareas      = len(areas)

clrs_vars   = sns.color_palette('inferno', nvars)

fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2))

# plt.plot(sbins,r2_cv.mean(axis=0),color='k',linewidth=2)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax = axes[ilab,iarea]
        idx = np.all((sessions[0].celldata['roi_name']==area, sessions[0].celldata['labeled']==label), axis=0)
        if np.sum(idx) > 0:
            # ax.plot(sbins,np.nanmean(r2_cv[idx,:],axis=0),color='k',linewidth=2)
            # plt.plot(sbins,r2_cv[idx,iarea],color=clrs[ilab],linewidth=1)
            for ivar,var in enumerate(variables):
                ax.plot(sbins,np.nanmean(np.abs(weights[idx,:,ivar]),axis=0),color=clrs_vars[ivar],linewidth=2,label=var)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
        ax.axvline(x=45, color='b', linestyle='--', linewidth=1)
        ax.set_xlim([-80,60])
        if ilab == 0:
            ax.set_title(area)
        if ilab == 1 and iarea == 1:
            ax.set_xlabel('Position relative to stim (cm)')
        if iarea==0:
            ax.set_ylabel('Encoding weights')
        if iarea==0 and ilab == 0:
            ax.legend(frameon=False,fontsize=6)

plt.tight_layout()
plt.savefig(os.path.join(savedir, 'EncodingWeights_Areas_Labels_%s.png') % sessions[ises].sessiondata['session_id'][0], format='png')


#%% 

idx_respwin = (sbins>=-5) & (sbins<=20)

for iarea, area in enumerate(areas):
    # fig,ax = plt.subplots(1,1,figsize=(8,8))
    data = weights[np.ix_(ses.celldata['roi_name']==area,idx_respwin,np.ones(nvars).astype(bool))].mean(axis=1).T
    # fig = sns.pairplot(pd.DataFrame(data.T,columns=variables),diag_kind="kde")
    fig = sns.pairplot(pd.DataFrame(data.T,columns=variables),diag_kind="hist",height=1.5)
    # ax.set_title(area)
    plt.suptitle(area)
    fig.tight_layout()
    # plt.savefig(os.path.join(savedir, 'EncodingWeights_pairplot_%s_%s.png') % (area,sessions[ises].sessiondata['session_id'][0]), format='png')


#%% 
idx_respwin = (sbins>=-5) & (sbins<=20)
# fig,ax = plt.subplots(1,1,figsize=(8,8))
df = pd.DataFrame(weights[:,idx_respwin,:].mean(axis=1),columns=variables)
df['roi_name'] = ses.celldata['roi_name']
fig = sns.pairplot(df,hue='roi_name',diag_kind="kde",height=1.5,corner=False,plot_kws={'s': 8})
fig.tight_layout()
fig.savefig(os.path.join(savedir, 'EncodingWeights_pairplot_%s.png') % (sessions[ises].sessiondata['session_id'][0]), format='png')

#%% 
fig,ax = plt.subplots(1,1,figsize=(4,3))
sns.heatmap(df.loc[:,~df.columns.isin(['roi_name'])].corr(),vmin=-1,vmax=1,cmap="vlag",ax=ax)
plt.savefig(os.path.join(savedir, 'EncodingWeights_corrheatmap_%s.png') % (sessions[ises].sessiondata['session_id'][0]), format='png')

#%% CHATGPT: 
# I am doing a lasso regression. My X predictor matrix is 122 samples by 4 features. 
# My y data to predict is 122 samples by 11 features. I am doing 5-fold cross-validation.
# If I am computing the R2 score without cross-validation this makes sense, but with cross-validation 
# each fold the r2 score takes on negative values. What is happening?


# When performing lasso regression (or any regression) with cross-validation, it’s not uncommon to see negative \( R^2 \) scores for some folds, especially under certain conditions. Here’s what might be happening in your case:

# ### 1. **Definition of \( R^2 \)**
# The \( R^2 \) score is a measure of how well the model explains the variance of the target data. It is computed as:
# \[
# R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
# \]
# where:
# - \( \text{SS}_{\text{res}} \) is the residual sum of squares (the sum of squared errors between the predictions and the actual values),
# - \( \text{SS}_{\text{tot}} \) is the total sum of squares (the variance of the actual values from their mean).

# An \( R^2 \) score can be negative when the model performs worse than simply predicting the mean of the target variable.

# ### 2. **Why Negative \( R^2 \) in Cross-Validation?**
# In cross-validation, each fold involves splitting the data into training and test sets. The model is trained on one subset and tested on another. The negative \( R^2 \) score indicates that, for certain test sets:
# - The model is overfitting to the training data and generalizes poorly to the test data.
# - The regularization applied by the lasso regression might be too strong, leading to underfitting.

# ### 3. **Factors Contributing to Negative \( R^2 \) in Your Case**
# - **Few Features Relative to Target Dimensions**: You have 4 features but are predicting 11 targets. This is a high-dimensional problem for regression and may require more features or carefully tuned regularization.
# - **Lasso Regularization**: Lasso regression applies \( L_1 \) regularization, which can shrink some coefficients to zero. If the regularization parameter (\( \alpha \)) is too high, the model can become too simplistic and fail to capture the underlying patterns.
# - **Small Dataset**: With only 122 samples, splitting into 5 folds means each test set has about 24 samples. This can lead to variability in test performance.
# - **Multitarget Prediction**: Scoring multiple targets together can exacerbate the issue because poor performance on one or more targets can drag the overall \( R^2 \) down.

# ### 4. **What to Do**
# Here are steps to diagnose and potentially improve the situation:
# - **Tune the Regularization Parameter (\( \alpha \))**: Use a grid search or similar approach to find the optimal \( \alpha \) that balances underfitting and overfitting.
# - **Check Individual Targets**: Compute \( R^2 \) scores for each of the 11 targets individually to see if some targets are particularly problematic.
# - **Standardize/Normalize Features and Targets**: Ensure your predictors and targets are scaled, as lasso regression can be sensitive to magnitudes.
# - **Increase Data if Possible**: If feasible, adding more samples can help improve generalization and stability across folds.
# - **Alternative Models**: Consider elastic net regression, which combines \( L_1 \) and \( L_2 \) penalties, as it may strike a better balance for your data.

# Negative \( R^2 \) scores in cross-validation aren't necessarily a cause for alarm but indicate that the model's generalization needs improvement. By adjusting the regularization or understanding target-wise performance, you can refine your approach.