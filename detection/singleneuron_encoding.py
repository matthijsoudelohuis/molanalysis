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

import sklearn
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn import svm as SVM
# from sklearn.metrics import accuracy_score, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score
from scipy.signal import medfilt
from scipy.stats import zscore

from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive
import seaborn as sns
import matplotlib.pyplot as plt
from utils.psth import *
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *
from detection.example_cells import get_example_cells
from utils.regress_lib import *

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

# session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11622', '2024_02_21']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11997', '2024_04_16'],
                         ['LPE11622', '2024_02_21'],
                         ['LPE11998', '2024_04_30'],
                         ['LPE12013','2024_04_25']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list        = np.array([['LPE12013','2024_04_25']])
# session_list        = np.array([['LPE12013','2024_04_26']])

# sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
#                          load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

sessions,nSessions = filter_sessions(protocols=protocol,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion,min_cells=100) #Load specified list of sessions

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Encoding\\')

#%% Z-score the calciumdata: 
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
    # Compute average response in stimulus response zone:
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
                                    sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)


#%% #################### Compute spatial runspeed ####################################
for ises,ses in enumerate(sessions): # running across the trial:
    sessions[ises].behaviordata['runspeed'] = medfilt(sessions[ises].behaviordata['runspeed'], kernel_size=51)
    [sessions[ises].runPSTH,_]     = calc_runPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)
    [sessions[ises].lickPSTH,_]    = calc_lickPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)

#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

#%% Get signal as relative to psychometric curve for all sessions:
sessions = noise_to_psy(sessions,filter_engaged=True)

#%% #################### Compute encoding of variables in single neurons  ####################################

ises = 0
example_cell_ids = get_example_cells(sessions[ises].sessiondata['session_id'][0])

# get some responsive cells: 
idx = np.nanmean(sessions[ises].respmat,axis=1)>1
example_cell_ids = (sessions[ises].celldata['cell_id'][idx]).to_numpy()

#%% Show example neurons that are correlated either to the stimulus signal, lickresponse or to running speed:
ises = 0
# for iN in range(0,100):
for iN in np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]:
    plot_snake_neuron_sortnoise(sessions[ises].stensor[iN,:,:],sbins,sessions[ises])
    plt.suptitle(sessions[ises].celldata['cell_id'][iN],fontsize=16,y=0.96)


#%%
# example_cell_ids = np.random.choice(sessions[0].celldata['cell_id'],size=8,replace=False)
# example_cell_ids = ['LPE12385_2024_06_15_0_0126',
# 'LPE12385_2024_06_15_0_0075']

fig = plot_mean_activity_example_neurons(sessions[ises].stensor,sbins,sessions[ises],example_cell_ids)
fig.savefig(os.path.join(savedir,'ExampleNeuronActivity_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')




# for ises, ses in tqdm(enumerate(sessions),desc='Decoding response across sessions'):

#%% 
nspatbins   = len(sbins)
variables   = ['signal','lickresponse','runspeed','trialnumber']
nvars       = len(variables)
kfold       = 5 # Define the number of folds for cross-validation
modelname   = 'Lasso' # Linear regression with Lasso (L1) regularization


#%% Show cross-validation results (as a function of lambda)
ises = 0
ses = sessions[ises]

#Neuron selection
# idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])[0]) #just one example cell
# idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])) #a few example cells
# idx_N = np.isin(ses.celldata['roi_name'],'V1') # #V1 cells:
# idx_N = np.logical_or(ses.celldata['sig_N'],ses.celldata['sig_M']) # Responsive cells
idx_N = np.ones(len(ses.celldata),dtype=bool) # All cells

#Trial selection
# idx_T = ses.trialdata['engaged']==1
idx_T = np.ones(len(ses.trialdata),dtype=bool)

y = ses.respmat[np.ix_(idx_N,idx_T)].T

X = np.stack((
              ses.trialdata['signal'][idx_T].to_numpy(),
              ses.trialdata['lickResponse'][idx_T].to_numpy(),
              ses.respmat_runspeed[0,idx_T],
              ses.trialdata['trialNumber'][idx_T]
              ), axis=1)

X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

# Find the optimal regularization strength (lambda)
lambdas = np.logspace(-6, 0, 20)
cv_scores = np.zeros((len(lambdas),))
for ilambda, lambda_ in enumerate(lambdas):
    model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)
    # model = ElasticNet(alpha=lambda_,l1_ratio=0.9)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    cv_scores[ilambda] = np.median(scores)
optimal_lambda = lambdas[np.argmax(cv_scores)]
print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))

model = getattr(sklearn.linear_model,modelname)(alpha=optimal_lambda)

print(np.nanmean(cross_val_score(model, X, y, cv=kfold, scoring='r2')))

fig,ax = plt.subplots(1,1,figsize=(3,3))
plt.plot(lambdas,cv_scores)
ax.set_xscale('log')


#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

modelname       = 'Lasso' # Linear regression with Lasso (L1) regularization
modelname       = 'Ridge'
scoring_type    = 'r2_score'
lam             = 0.05
version          ='v20'
N               = len(celldata)
modelvars       = get_predictors_from_modelversion(version)
nvars           = len(modelvars)

weights         = np.full((N,nspatbins,nvars),np.nan)
r2_cv           = np.full((N,nspatbins),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Fitting encoding model across sessions',total=nSessions):
    idx_N_ses = np.isin(celldata['session_id'],ses.sessiondata['session_id'][0])
    
    #Neuron selection
    # idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])[0]) #just one example cell
    # idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])) #a few example cells
    # idx_N = np.isin(ses.celldata['roi_name'],'V1') # #V1 cells:
    # idx_N = np.logical_or(ses.celldata['sig_N'],ses.celldata['sig_M']) # Responsive cells
    idx_N = np.ones(len(ses.celldata),dtype=bool) # All cells

    #Trial selection
    # idx_T = ses.trialdata['engaged']==1
    idx_T = np.ones(len(ses.trialdata),dtype=bool)

    r2_cv[idx_N_ses,:], weights[idx_N_ses,:,:], _ = enc_model_spatial_wrapper(ses,sbins,idx_N,idx_T,version=version,modelname=modelname,optimal_lambda=lam,kfold=5,scoring_type=scoring_type,crossval=True)
    # g, h, _ = enc_model_spatial_wrapper(ses,sbins,idx_N,idx_T,modelname=modelname,optimal_lambda=None,kfold=5,scoring_type = 'r2',crossval=True)


#%% Some variables that determine coloring and labeling for plots
labeled     = ['unl','lab']
nlabels     = 2
areas       = ['V1','PM','AL','RSP']	
nareas      = len(areas)
clrs_vars   = sns.color_palette('inferno', nvars)

#%% Show the crossvalidated performance across areas:
fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2),sharey=True,sharex=True)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax = axes[ilab,iarea]
        idx = np.all((ses.celldata['roi_name']==area, ses.celldata['labeled']==label), axis=0)
        idx = np.all((celldata['roi_name']==area, celldata['labeled']==label), axis=0)
        if np.sum(idx) > 0:
            ax.plot(sbins,np.nanmean(r2_cv[idx,:],axis=0),color='k',linewidth=2)
            # plt.plot(sbins,r2_cv[idx,iarea],color=clrs_vars[ivar],linewidth=1)
            # for ivar,var in enumerate(variables):
                # ax.plot(sbins,np.nanmean(np.abs(weights[idx,:,ivar]),axis=0),color=clrs_vars[ivar],linewidth=2,label=var)
        add_stim_resp_win(ax)
        ax.axhline(0,color='k',linestyle='--',linewidth=1)
        ax.set_xlim([-80,60])
        if ilab == 0:
            ax.set_title(area)
        if ilab == 1 and iarea == 1:
            ax.set_xlabel('Position relative to stim (cm)')
        if iarea==0:
            ax.set_ylabel('cvR2')
        # if iarea==0 and ilab == 0:
        #     ax.legend(frameon=False,fontsize=6)
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'EncodingModel_cvR2_Areas_Labels_%s.png') % ses.sessiondata['session_id'][0], format='png')
plt.savefig(os.path.join(savedir, 'EncodingModel_cvR2_Areas_Labels_%dsessions.png') % nSessions, format='png')

#%% Show the encoding weights across areas:
fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2),sharey=True,sharex=True)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax = axes[ilab,iarea]
        # idx = np.all((ses.celldata['roi_name']==area, ses.celldata['labeled']==label), axis=0)
        idx = np.all((celldata['roi_name']==area, celldata['labeled']==label), axis=0)
        if np.sum(idx) > 0:
            # ax.plot(sbins,np.nanmean(r2_cv[idx,:],axis=0),color='k',linewidth=2)
            # plt.plot(sbins,r2_cv[idx,iarea],color=clrs_vars[ivar],linewidth=1)
            for ivar,var in enumerate(modelvars):
                ax.plot(sbins,np.nanmean(np.abs(weights[idx,:,ivar]),axis=0),color=clrs_vars[ivar],linewidth=2,label=var)
                # ax.plot(sbins,np.nanmean(weights[idx,:,ivar],axis=0),color=clrs_vars[ivar],linewidth=2,label=var)
        add_stim_resp_win(ax)
        ax.axhline(0,color='k',linestyle='--',linewidth=1)
        ax.set_xlim([-80,60])
        if ilab == 0:
            ax.set_title(area)
        if ilab == 1 and iarea == 1:
            ax.set_xlabel('Position relative to stim (cm)')
        if iarea==0:
            ax.set_ylabel('|weights|')
        if iarea==0 and ilab == 0:
            ax.legend(frameon=False,fontsize=6)
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'EncodingWeights_Areas_Labels_%s.png') % ses.sessiondata['session_id'][0], format='png')
# plt.savefig(os.path.join(savedir, 'EncodingWeights_Areas_Labels_%dsessions.png') % nSessions, format='png')

#%% Show correlation between encoding weights per area: 
idx_respwin = (sbins>=-5) & (sbins<=20)
for iarea, area in enumerate(areas):
    # data = weights[np.ix_(ses.celldata['roi_name']==area,idx_respwin,np.ones(nvars).astype(bool))].mean(axis=1).T
    data = weights[np.ix_(celldata['roi_name']==area,idx_respwin,np.ones(nvars).astype(bool))].mean(axis=1).T
    fig = sns.pairplot(pd.DataFrame(data.T,columns=variables),diag_kind="hist",height=1.5,plot_kws={"s": 3, "alpha": 0.5, "color": "k"})
    plt.suptitle(area)
    fig.tight_layout()
    # plt.savefig(os.path.join(savedir, 'EncodingWeights_pairplot_%s_%s.png') % (area,sessions[ises].sessiondata['session_id'][0]), format='png')
    plt.savefig(os.path.join(savedir, 'EncodingWeights_pairplot_%s_%dsessions.png') % (area,nSessions), format='png')

#%% Show correlation between encoding weights per area: 
idx_respwin = (sbins>=-5) & (sbins<=20)
fig,axes = plt.subplots(2,2,figsize=(7,6))
for iarea, area in enumerate(areas):
    ax = axes[iarea//2,iarea%2]
    # data = weights[np.ix_(ses.celldata['roi_name']==area,idx_respwin,np.ones(nvars).astype(bool))].mean(axis=1).T
    data = weights[np.ix_(celldata['roi_name']==area,idx_respwin,np.ones(nvars).astype(bool))].mean(axis=1).T
    df = pd.DataFrame(data.T,columns=variables)
    # sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap="vlag",xticklabels=variables,yticklabels=variables,ax=ax)
    sns.heatmap(df.corr(),vmin=-0.5,vmax=0.5,cmap="vlag",xticklabels=variables,yticklabels=variables,ax=ax)
    ax.set_title(area)
fig.tight_layout()
# fig.savefig(os.path.join(savedir, 'EncodingWeights_corrheatmap_%s.png') % (sessions[ises].sessiondata['session_id'][0]), format='png')
fig.savefig(os.path.join(savedir, 'EncodingWeights_corrheatmap_%dsessions.png') % nSessions, format='png')

#%%%%                






#%%%%%%%%%%%%%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

modelname       = 'Lasso' # Linear regression with Lasso (L1) regularization
scoring_type    = 'r2_score'
lam             = 0.05
version          ='v14'
N               = len(celldata)
modelvars       = get_predictors_from_modelversion(version)
nvars           = len(modelvars)

weights         = np.full((N,nvars),np.nan)
r2_cv           = np.full(N,np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Fitting encoding model across sessions',total=nSessions):
    idx_N_ses = np.isin(celldata['session_id'],ses.sessiondata['session_id'][0])
    
    #Neuron selection
    # idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])[0]) #just one example cell
    idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])) #a few example cells
    # idx_N = np.isin(ses.celldata['roi_name'],'V1') # #V1 cells:
    idx_N = np.logical_or(ses.celldata['sig_N'],ses.celldata['sig_M']) # Responsive cells
    idx_N = np.ones(len(ses.celldata),dtype=bool) # All cells

    #Trial selection
    # idx_T = ses.trialdata['engaged']==1
    idx_T = np.ones(len(ses.trialdata),dtype=bool)

    r2_cv[idx_N_ses], weights[idx_N_ses,:], _, _ =  enc_model_stimwin_wrapper(ses,idx_N,idx_T,version=version,modelname=modelname,optimal_lambda=lam,kfold=5,
                                                scoring_type =scoring_type,crossval=True,subtr_shuffle=False)
    
    # r2_cv[idx_N_ses], weights[idx_N_ses,:], _ = enc_model_spatial_wrapper(ses,sbins,idx_N,idx_T,modelname=modelname,optimal_lambda=None,kfold=5,scoring_type = 'r2',crossval=True)
    # g, h, _ = enc_model_spatial_wrapper(ses,sbins,idx_N,idx_T,modelname=modelname,optimal_lambda=None,kfold=5,scoring_type = 'r2',crossval=True)

#%% Show correlation between encoding weights per area: 
for iarea, area in enumerate(areas):
    # data = weights[np.ix_(ses.celldata['roi_name']==area,idx_respwin,np.ones(nvars).astype(bool))].mean(axis=1).T
    data = weights[celldata['roi_name']==area,:].T
    fig = sns.pairplot(pd.DataFrame(data.T,columns=modelvars),diag_kind="hist",height=1.5,plot_kws={"s": 3, "alpha": 0.5, "color": "k"})
    plt.suptitle(area)
    fig.tight_layout()
    # plt.savefig(os.path.join(savedir, 'EncodingWeights_pairplot_%s_%s.png') % (area,sessions[ises].sessiondata['session_id'][0]), format='png')
    plt.savefig(os.path.join(savedir, 'EncodingWeights_StimWin_pairplot_%s_model%s_%dsessions.png') % (area,version,nSessions), format='png')

#%% Show correlation between encoding weights per area: 
fig,axes = plt.subplots(2,2,figsize=(7,6))
for iarea, area in enumerate(areas):
    ax = axes[iarea//2,iarea%2]
    data = weights[celldata['roi_name']==area,:].T
    df = pd.DataFrame(data.T,columns=modelvars)
    sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap="vlag",xticklabels=modelvars,yticklabels=modelvars,ax=ax)
    ax.set_title(area)
fig.tight_layout()
# fig.savefig(os.path.join(savedir, 'EncodingWeights_corrheatmap_%s.png') % (sessions[ises].sessiondata['session_id'][0]), format='png')
fig.savefig(os.path.join(savedir, 'EncodingWeights_StimWin_corrheatmap_model%s_%dsessions.png') % (version,nSessions), format='png')


# #%% 
# fig,ax = plt.subplots(1,1,figsize=(4,3))
# sns.heatmap(df.loc[:,~df.columns.isin(['roi_name'])].corr(),vmin=-1,vmax=1,cmap="vlag",ax=ax)
# plt.savefig(os.path.join(savedir, 'EncodingWeights_corrheatmap_%s.png') % (sessions[ises].sessiondata['session_id'][0]), format='png')


#%% 









#%% COMPARE LINEAR TO POISSON REGRESSION

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error

# LASSO AND POISSON PERFORM EQUALLY ON MSE, RIDGE WINS
# LASSO WINS ON R2
ses = sessions[28]
# nexcell = 50
# idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])[nexcell])

# idx_N = np.zeros(len(ses.celldata),dtype=bool) # Responsive cell
# idx_N[np.where(np.logical_or(ses.celldata['sig_N'],ses.celldata['sig_M']))[0][nexcell]] = True

idx_T = np.ones(len(ses.trialdata),dtype=bool)

modelvars   = get_predictors_from_modelversion(version='v8')

y           = ses.respmat[np.ix_(idx_N,idx_T)].T

X,allvars   = get_predictors(ses)               # get all predictors
X           = X[:,np.isin(allvars,modelvars)] #get only predictors of interest
X           = X[idx_T,:]                     #get only trials of interest

X,y         = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

modelnames = ['Lasso','Ridge','PoissonRegressor']
scoring_type = 'neg_mean_squared_error'

# excells = np.where(np.logical_or(ses.celldata['sig_N'],ses.celldata['sig_M']))[0]
excells = np.random.choice(np.where(np.logical_or(ses.celldata['sig_N'],ses.celldata['sig_M']))[0],size=100,replace=False)
errordata = np.zeros((len(excells),len(modelnames)))
print('Average cross-validated error for:\n')

for iN, excell in enumerate(excells):

    idx_N = np.zeros(len(ses.celldata),dtype=bool) # Responsive cell
    idx_N[excell] = True

    y       = ses.respmat[np.ix_(idx_N,idx_T)].T
    y       = (y - np.min(y)) / (np.max(y) - np.min(y))

    for imodel, modelname in enumerate(modelnames):
        # Find the optimal regularization strength (lambda) for Lasso
        lambdas = np.logspace(-6, 1, 20)
        cv_scores_lasso = np.zeros((len(lambdas),))
        for ilambda, lambda_ in enumerate(lambdas):
            model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)
            scores = cross_val_score(model, X, y.ravel(), cv=kfold, scoring=scoring_type)
            cv_scores_lasso[ilambda] = np.median(scores)
        optimal_lambda = lambdas[np.argmax(cv_scores_lasso)]
        model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)

        # Evaluate Lasso with optimal regularization value
        # lasso_model = Lasso(alpha=optimal_lambda)
        cv_scores = cross_val_score(model, X, y.ravel(), cv=kfold, scoring=scoring_type)
        average_error = -np.mean(cv_scores)
        errordata[iN,imodel] = average_error
        # print('%s Regression: %0.4f' % (modelname,average_error))

df = pd.DataFrame(errordata,columns=modelnames)
sns.barplot(data=df,errorbar='ci')

# Find which model wins
winning_model = modelnames[np.argmin(errordata.mean(axis=0))]
print('Winning model: %s' % winning_model)


#%% 
# Find the optimal regularization strength (lambda)
modelname = 'Lasso'
# modelname = 'Ridge'

lambdas = np.logspace(-6, 0, 20)
cv_scores = np.zeros((len(lambdas),))
for ilambda, lambda_ in enumerate(lambdas):
    model = getattr(sklearn.linear_model,modelname)(alpha=lambda_)

    # model = LinearRegression(solver='liblinear', C=lambda_)
    # model = Lasso(alpha=lambda_)
    # model = ElasticNet(alpha=lambda_,l1_ratio=0.9)
    # model = Ridge(alpha=lambda_)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    cv_scores[ilambda] = np.median(scores)
optimal_lambda = lambdas[np.argmax(cv_scores)]
print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))

# model = Ridge(alpha=optimal_lambda)
# model = Lasso(alpha=optimal_lambda)
model = getattr(sklearn.linear_model,modelname)(alpha=optimal_lambda)

# model = ElasticNet(alpha=optimal_lambda,l1_ratio=0.9)

print(np.nanmean(cross_val_score(model, X, y, cv=kfold, scoring='r2')))

# model.fit(X, y)
# r2 = model.score(X, y)
# print(r2)

fig,ax = plt.subplots(1,1,figsize=(3,3))
plt.plot(lambdas,cv_scores)
ax.set_xscale('log')

#%% 




#%% COMPARE MODEL VERSIONS
versions    = np.array(['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15']) 
# versions    = np.array(['v15','v16']) 
# versions    = np.array(['v1','v7','v8','v9']) 


#%% 
nspatbins   = len(sbins)
kfold       = 5 # Define the number of folds for cross-validation
modelname   = 'Lasso' # Linear regression with Lasso (L1) regularization
modelname   = 'Ridge' # Linear regression with Ridge (L2) regularization
scoring_type = 'r2_score'
# scoring_type = 'mean_squared_error'
lam         = 0.01 #for responsive neurons
# lam         = 0.05 #for all cells
# lam         = None #optimization of optimal lambda

#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

N           = len(celldata)
r2_cv       = np.full((N,len(versions)),np.nan)
for ises,ses in tqdm(enumerate(sessions),desc='Fitting encoding model across sessions',total=nSessions):
    idx_N_ses = np.isin(celldata['session_id'],ses.sessiondata['session_id'][0])
    #Neuron selection
    # idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])[0]) #just one example cell
    # idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])) #a few example cells
    idx_N = np.logical_or(ses.celldata['sig_N'],ses.celldata['sig_M']) # Responsive cells
    # idx_N = np.ones(len(ses.celldata),dtype=bool) # All cells
    
    #Trial selection
    # idx_T = ses.trialdata['engaged']==1
    idx_T = np.ones(len(ses.trialdata),dtype=bool)
    
    for iver,version in enumerate(versions):
        r2_cv[idx_N_ses,iver], _, _, modelvars =  enc_model_stimwin_wrapper(ses,idx_N,idx_T,version=version,modelname=modelname,optimal_lambda=lam,kfold=5,
                                                scoring_type = scoring_type, crossval=True,subtr_shuffle=False)
                               
                # r2_cv[idx_N_ses,iver], _, _, modelvars =  enc_model_stimwin_wrapper(ses,idx_N,idx_T,version=version,modelname='Lasso',optimal_lambda=None,kfold=5,scoring_type = 'r2',
                                # crossval=True,subtr_shuffle=False)


#%% Plot the decoding performance for the different model versions
clr_palette = sns.color_palette('husl',n_colors=len(versions))
fig,ax = plt.subplots(1,1,figsize=(len(versions)*0.5,3))
# sns.lineplot(data=r2_cv,palette=sns.color_palette('husl',n_colors=len(versions)))
sns.barplot(data=r2_cv,palette=clr_palette,ax=ax)
ax.set_xlabel('Model version')
ax.set_xticks(np.arange(len(versions)),versions.tolist())
for i,patch in enumerate(ax.patches):
    ax.text(patch.get_x() + patch.get_width()/2,0.005,
            '{:.3f}'.format(np.nanmean(r2_cv[:,i])),ha='center',va='center',fontsize=8)
ax.set_ylabel('Encoding performance \n(%s)' % scoring_type)
ax.set_title('Encoding performance for different model versions')
# ax.set_position([ax.get_position().x0,ax.get_position().y0,ax.get_position().width,ax.get_position().height])
# fig.tight_layout()
# fig.savefig(os.path.join(savedir, 'EncodingPerformance_ModelVersions_%dsessions.png') % (nSessions), format='png')
# fig.savefig(os.path.join(savedir, 'EncodingPerformance_ResponsiveCells_ModelVersions_%dsessions.png') % (nSessions), format='png')

# from utils.regress_lib import get_predictors_from_modelversion
fig,ax = plt.subplots(1,1,figsize=(len(versions)*0.7,3))
# Make a legend where for each version is displayed which model variables are included in the model
legend_str = []
for iver,version in enumerate(versions):
    # for ivar,var in enumerate(variables):
    plt.plot(0,0,color=clr_palette[iver],label=version) #,sns.barplot(data=r2_cv,palette=sns.color_palette('husl',n_colors=len(versions)),ax=ax)

    modelvars   = get_predictors_from_modelversion(version)
    modelvars_str = ',\n'.join(modelvars)
    legend_str.append('%s: %s' % (version,modelvars_str))
ax.axis('off')
ax.legend(legend_str,loc='upper right',bbox_to_anchor=(0, 0.5),ncol=np.ceil(len(versions)/3),fontsize=9,frameon=False)
plt.tight_layout()
# fig.savefig(os.path.join(savedir, 'Legend_ModelVersions.png'), format='png')


#%% OLD versION:


# crossval = True
# # Loop through each spatial bin
# for ibin, bincenter in enumerate(sbins):
#     if np.sum(idx_T) > 50:
#         y = ses.stensor[np.ix_(idx_N,idx_T,[ibin])].squeeze().T # Get the neural response data for this bin

#         # Define the X predictors
#         X = np.stack((ses.trialdata['signal'][idx_T].to_numpy(),
#             ses.trialdata['lickResponse'][idx_T].to_numpy(),
#             ses.runPSTH[idx_T,ibin],
#             ses.trialdata['trialNumber'][idx_T]), axis=1)

#         X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

#         # Train a linear regression model on the training data with regularization
#         model = Lasso(alpha=optimal_lambda)
#         # model = ElasticNet(alpha=optimal_lambda,l1_ratio=0.9)

#         if crossval:
#             # Define the k-fold cross-validation object
#             kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
            
#             # Initialize an array to store the decoding performance for each fold
#             fold_r2             = np.zeros((kfold,N_idx))
#             # fold_r2_shuffle     = np.zeros((kfold,N))
#             fold_weights        = np.zeros((kfold,N_idx,nvars))

#             # Loop through each fold
#             for ifold, (train_index, test_index) in enumerate(kf.split(X)):
#                 # Split the data into training and testing sets
#                 X_train, X_test = X[train_index], X[test_index]
#                 y_train, y_test = y[train_index], y[test_index]
                
#                 # model = Ridge(alpha=optimal_lambda)
#                 model.fit(X_train, y_train)

#                 # Compute R2 on the test set
#                 y_pred              = model.predict(X_test)
#                 fold_r2[ifold,:]    = r2_score(y_test, y_pred, multioutput='raw_values')
#                 # print(np.nanmean(r2_score(y_test, y_pred, multioutput='raw_values')))
#                 # print(np.nanmean(r2_score(y_test, y_pred)))
                
#                 fold_weights[ifold,:,:]     = model.coef_
#                 # y_hat[np.ix_(idx_N,[ibin],test_index)] = y_pred.T[:,np.newaxis,:]
#                 y_hat[np.ix_(idx_N,sbins==bincenter,test_index)] = y_pred.T[:,np.newaxis,:]
                
#                 # # Shuffle the labels and calculate the decoding performance for this fold
#                 # np.random.shuffle(y_train)
#                 # model.fit(X_train, y_train)
#                 # y_pred = model.predict(X_test)
#                 # fold_r2_shuffle[ifold] = accuracy_score(y_test, y_pred)
        
#             # Calculate the average decoding performance across folds
#             r2_cv[idx_N,ibin] = np.nanmean(fold_r2, axis=0)
#             weights[idx_N,ibin,:] = np.nanmean(fold_weights, axis=0)

#         else:   
#             # Without cross-validation
#             model.fit(X, y)
#             y_pred = model.predict(X)
#             r2_cv[idx_N,ibin] = r2_score(y, y_pred, multioutput='raw_values')
#             y_hat[np.ix_(idx_N,sbins==bincenter,idx_T)] = y_pred.T[:,np.newaxis,:]
#             weights[idx_N,ibin,:] = model.coef_

# #%% 
# y_pred  = model.predict(X)
# r2      = r2_score(y, y_pred, multioutput='raw_values')
# idx_N   = r2 > 0.1
# N_idx   = np.sum(idx_N)

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