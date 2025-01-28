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
from sklearn.linear_model import LogisticRegression as LOGR
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

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\PredHitMiss\\')

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

# session_list        = np.array([['LPE12013','2024_04_25']])
# session_list        = np.array([['LPE12013','2024_04_26']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

# sessions,nSessions = filter_sessions(protocols=protocol,load_behaviordata=True,load_videodata=False,
                        #  load_calciumdata=True,calciumversion=calciumversion,min_cells=100) #Load specified list of sessions

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

#%% 




#%% 
nspatbins   = len(sbins)
# variables   = ['signal','lickresponse','runspeed','trialnumber']
# nvars       = len(variables)
version          ='v6'
modelvars       = get_dec_predictors_from_modelversion(version)
nvars           = len(modelvars)

kfold       = 5 # Define the number of folds for cross-validation
# modelname   = 'Lasso' # Linear regression with Lasso (L1) regularization
modelname   = 'LogisticRegression' # Logistic regression with Lasso (L1) regularization


#%% Show cross-validation results (as a function of lambda)
ises = 1
ses = sessions[ises]

#Trial selection
idx_T = ses.trialdata['engaged']==1
# idx_T = np.ones(len(ses.trialdata),dtype=bool)


y        = ses.trialdata['lickResponse'].to_numpy()
# y        = ses.trialdata['rewardGiven'].to_numpy()

# y[ses.trialdata['stimcat'] != 'N'] = 0

X,allvars       = get_dec_predictors(ses)               # get all predictors
X               = X[:,np.isin(allvars,modelvars)] #get only predictors of interest
X               = X[idx_T,:]                     #get only trials of interest
y               = y[idx_T]
X,y,idx_nan     = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

lam = find_optimal_lambda(X,y,model_name='LOGR',kfold=5,clip=False)

# Find the optimal regularization strength (lambda)
lambdas = np.logspace(-4, 4, 20)
cv_scores = np.zeros((len(lambdas),))
for ilambda, lambda_ in enumerate(lambdas):
    model = LOGR(penalty='l1', solver='liblinear', C=lambda_) # LOGR(C=optimal_lambda)

    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    # cv_scores[ilambda] = np.mean(scores)
    cv_scores[ilambda] = np.median(scores)
optimal_lambda = lambdas[np.argmax(cv_scores)]
print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))

model = LOGR(penalty='l1', solver='liblinear', C=optimal_lambda) 

# print(np.nanmean(cross_val_score(model, X, y, cv=kfold, scoring='r2')))

fig,ax = plt.subplots(1,1,figsize=(3,3))
plt.plot(lambdas,cv_scores)
ax.set_xscale('log')
ax.set_xlabel('Lambda')
ax.set_ylabel('CV R2')
# fig.savefig(os.path.join(savedir,'Lambda_vs_CrossValR2_%s.png' % (sessions[ises].sessiondata['session_id'][0])), format = 'png',bbox_inches='tight')

#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

modelname       = 'LOGR' # Linear regression with Lasso (L1) regularization
scoring_type    = 'accuracy_score'
scoring_type    = 'balanced_accuracy_score'
scoring_type    = 'r2_score'
# kfold       = 5 # Define the number of folds for cross-validation
lam             = 0.05
lam             = None

#%% COMPARE MODEL VERSIONS
versions    = np.array(['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15']) 
versions    = np.array(['v1','v2','v3','v4','v5','v6','v7','v8','v9']) 
versions    = np.array(['v%d' % i for i in range(1,22)])
versions    = np.array(['v%d' % i for i in [6,28,29,30,31]])
versions    = np.array(['v%d' % i for i in range(23,32)])
versions    = np.array(['v%d' % i for i in range(1,32)])

#%% Run cross-validation for the different model versions
error_cv       = np.full((nSessions,len(versions)),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Fitting encoding model across sessions',total=nSessions):
    idx_T = ses.trialdata['engaged']==1
    y_all        = ses.trialdata['rewardGiven'].to_numpy()
    X_all,allvars       = get_dec_predictors(ses)               # get all predictors
    for iver,version in enumerate(versions):
        modelvars       = get_dec_predictors_from_modelversion(version,nneuraldims=10)
        nvars           = len(modelvars)
        X               = X_all[:,np.isin(allvars,modelvars)] #get only predictors of interest
        X               = X[idx_T,:]                     #get only trials of interest
        y               = y_all[idx_T]
        X,y,idx_nan     = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
        error_cv[ises,iver],_,_,_ = my_decoder_wrapper(X,y,model_name=modelname,kfold=kfold,lam=lam,subtract_shuffle=True,
                          scoring_type=scoring_type,norm_out=True)

#%% Plot the decoding performance for the different model versions
clr_palette = sns.color_palette('husl',n_colors=len(versions))
fig,ax = plt.subplots(1,1,figsize=(len(versions)*0.7,3))
# sns.lineplot(data=r2_cv,palette=sns.color_palette('husl',n_colors=len(versions)))
sns.barplot(data=error_cv,palette=clr_palette,ax=ax)
ax.set_xlabel('Model version')
ax.set_xticks(np.arange(len(versions)),versions.tolist())
# ax.set_xticks(np.arange(len(versions)),[get_dec_modelname(version) for version in versions],fontsize=8,rotation=45)
for i,patch in enumerate(ax.patches):
    ax.text(patch.get_x() + patch.get_width()/2,0.02,
            '{:.3f}'.format(np.nanmean(error_cv[:,i])),ha='center',va='center',fontsize=8)
ax.set_ylabel('Decoding performance \n(%s)\n(shuffle-subtracted)' % scoring_type)
ax.set_title('Decoding performance for different model versions')
plt.tight_layout()
# ax.set_position([ax.get_position().x0,ax.get_position().y0,ax.get_position().width,ax.get_position().height])
# fig.tight_layout()
# fig.savefig(os.path.join(savedir, 'DecodingPerformance_ModelVersions_%dsessions.png') % (nSessions), format='png')

#%% Make a legend where for each version is displayed which model variables are included in the model
fig,ax = plt.subplots(1,1,figsize=(len(versions)*0.1,2))
legend_str = []
for iver,version in enumerate(versions):
    plt.plot(0,0,color=clr_palette[iver],label=version) #,sns.barplot(data=r2_cv,palette=sns.color_palette('husl',n_colors=len(versions)),ax=ax)
    legend_str.append('%s: %s' % (version,get_dec_modelname(version)))
ax.axis('off')
ax.legend(legend_str,loc='upper right',bbox_to_anchor=(0, 0.9),ncol=np.ceil(len(versions)/8),fontsize=12,frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(savedir, 'Legend_ModelName_Versions.png'), format='png',bbox_inches='tight')

#%% 
fig,ax = plt.subplots(1,1,figsize=(len(versions)*0.7,3))
# Make a legend where for each version is displayed which model variables are included in the model
legend_str = []
for iver,version in enumerate(versions):
    # for ivar,var in enumerate(variables):
    plt.plot(0,0,color=clr_palette[iver],label=version) #,sns.barplot(data=r2_cv,palette=sns.color_palette('husl',n_colors=len(versions)),ax=ax)

    modelvars   = get_dec_predictors_from_modelversion(version)
    modelvars_str = ',\n'.join(modelvars)
    legend_str.append('%s: %s' % (version,modelvars_str))
ax.axis('off')
ax.legend(legend_str,loc='upper right',bbox_to_anchor=(0, 0.5),ncol=np.ceil(len(versions)/3),fontsize=9,frameon=False)
plt.tight_layout()
# fig.savefig(os.path.join(savedir, 'Legend_ModelVersions.png'), format='png',bbox_inches='tight')


#%% COMPARE MODEL VERSIONS
versions    = np.array(['v6','v31']) 
scoring_type    = 'r2_score'
scoring_type    = 'accuracy_score'

#%% Run cross-validation for the different model versions
error_cv       = np.full((nSessions,len(versions)),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Fitting encoding model across sessions',total=nSessions):
    idx_T = ses.trialdata['engaged']==1
    y_all        = ses.trialdata['rewardGiven'].to_numpy()
    X_all,allvars       = get_dec_predictors(ses)               # get all predictors
    for iver,version in enumerate(versions):
        modelvars       = get_dec_predictors_from_modelversion(version,nneuraldims=10)
        nvars           = len(modelvars)
        X               = X_all[:,np.isin(allvars,modelvars)] #get only predictors of interest
        X               = X[idx_T,:]                     #get only trials of interest
        y               = y_all[idx_T]
        X,y,idx_nan     = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
        error_cv[ises,iver],_,_,_ = my_decoder_wrapper(X,y,model_name=modelname,kfold=kfold,lam=lam,subtract_shuffle=False,
                          scoring_type=scoring_type,norm_out=False)

#%% Plot

fig,ax = plt.subplots(1,1,figsize=(4,4))
sns.scatterplot(x=error_cv[:,0],y=error_cv[:,1],ax=ax)
ax.plot([0,1],[0,1],color='k',lw=0.5)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('External signal (%s)' % scoring_type)
ax.set_ylabel('Internal signal (%s)\n(V1, PM, AL, RSP)' % scoring_type)
ax.set_title('Comparison of model versions')
plt.tight_layout()
# fig.savefig(os.path.join(savedir, 'ErrorRateComparison_v6v31_%dsessions.png') % (nSessions), format='png',bbox_inches='tight')







#%% 


#%% Plot the encoding performance for the model for the different areas: 
df = pd.DataFrame(data={'r2':r2_cv,'area':celldata['roi_name']})

fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
# sns.boxplot(x='area',y='r2',data=df,ax=axes,palette=clrs_areas,order=areas,showfliers=False)
sns.violinplot(x='area',y='r2',data=df,ax=ax,palette=clrs_areas,order=areas,
               bw=0.4,width=1.3)
# sns.violinplot(x='area',y='r2',data=df,ax=ax,palette=clrs_areas,order=areas,showfliers=False,scale='width')
ax.set_ylim([-0.2,1])
fig.savefig(os.path.join(savedir,'EncodingModel_StimWin_OverallR2_model%s_%dsessions.png') % (version,nSessions), format='png',bbox_inches='tight')

#%% Plot the encoding performance for the model for the different areas: 
df = pd.DataFrame(data=r2_cv_var,columns=modelvars)
df['area']      = celldata['roi_name']
df['labeled']   = celldata['labeled']

fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2.2),sharey=True,sharex=True)

for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax = axes[ilab,iarea]
        # sns.violinplot(df[df['area']==area],ax=ax,palette=clrs_vars,order=modelvars,showfliers=False)
        sns.barplot(data=df[(df['area']==area) & (df['labeled']==label)], ax=ax, palette=clrs_vars, order=modelvars, estimator='mean', errorbar='ci')
        ax.set_xticks([])
        ax.set_title(area + ' - ' + label,fontsize=12,color=clrs_areas[iarea])
        # if iarea == 3:
            # ax.legend(frameon=False,fontsize=6,loc='upper right')
        if iarea == 3 and ilab == 0:
            # Manually create legend handles
            handles = [plt.Line2D([0], [0], color=clrs_vars[i], lw=4) for i in range(len(modelvars))]
            ax.legend(handles, modelvars, frameon=False, fontsize=8, loc='upper right')
        if iarea == 0:
            ax.set_ylabel('R2')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'EncodingModel_StimWin_R2_perVar_model%s_%dsessions.png') % (version,nSessions), format='png')


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
    # sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap="vlag",xticklabels=modelvars,yticklabels=modelvars,ax=ax)
    sns.heatmap(df.corr(),vmin=-0.5,vmax=0.5,cmap="vlag",xticklabels=modelvars,yticklabels=modelvars,ax=ax)
    ax.set_title(area)
fig.tight_layout()
# fig.savefig(os.path.join(savedir, 'EncodingWeights_corrheatmap_%s.png') % (sessions[ises].sessiondata['session_id'][0]), format='png')
fig.savefig(os.path.join(savedir, 'EncodingWeights_StimWin_corrheatmap_model%s_%dsessions.png') % (version,nSessions), format='png')


#%% 
modelvars       = get_predictors_from_modelversion(version)
nvars           = len(modelvars)

corrmat         = np.full((nSessions,nvars,nvars),np.nan)

for ises,ses in enumerate(sessions):

    idx_T = np.ones(len(ses.trialdata),dtype=bool)
    y = ses.respmat[:,idx_T].T
    X,allvars   = get_enc_predictors(ses)               # get all predictors
    X           = X[:,np.isin(allvars,modelvars)] #get only predictors of interest
    X           = X[idx_T,:]                     #get only trials of interest

    X,y         = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

    df = pd.DataFrame(data=X,columns=modelvars)
    corrmat[ises,:,:] = df.corr().values


fig,ax = plt.subplots(1,1,figsize=(5,4))
sns.heatmap(np.nanmean(corrmat,axis=0),vmin=-1,vmax=1,cmap="vlag",ax=ax,annot=True,xticklabels=modelvars,yticklabels=modelvars)
plt.title('Predictor correlations - Model %s' % (version))
plt.savefig(os.path.join(savedir, 'EncodingMOdel_Predictors_corrheatmap_%dsessions.png') % (nSessions), format='png',bbox_inches='tight')


#%% 









#%% COMPARE LINEAR TO POISSON REGRESSION

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error

ses = sessions[28]
# nexcell = 50
# idx_N = np.isin(ses.celldata['cell_id'],get_example_cells(ses.sessiondata['session_id'][0])[nexcell])

# idx_N = np.zeros(len(ses.celldata),dtype=bool) # Responsive cell
# idx_N[np.where(np.logical_or(ses.celldata['sig_N'],ses.celldata['sig_M']))[0][nexcell]] = True

idx_T       = np.ones(len(ses.trialdata),dtype=bool)

modelvars   = get_predictors_from_modelversion(version='v8')

y           = ses.respmat[np.ix_(idx_N,idx_T)].T

X,allvars   = get_enc_predictors(ses)               # get all predictors
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
kfold       = 5 # Define the number of folds for cross-validation
modelname   = 'Lasso' # Linear regression with Lasso (L1) regularization
# modelname   = 'Ridge' # Linear regression with Ridge (L2) regularization
scoring_type = 'r2_score'
# scoring_type = 'mean_squared_error'
# lam         = 0.01 #for responsive neurons
lam         = 0.05 #Lasso, for all cells
# lam         = 200 #Ridge for all cells
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
        r2_cv[idx_N_ses,iver], _, _, _ =  enc_model_stimwin_wrapper(ses,idx_N,idx_T,version=version,modelname=modelname,optimal_lambda=lam,kfold=kfold,
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
plt.tight_layout()
# ax.set_position([ax.get_position().x0,ax.get_position().y0,ax.get_position().width,ax.get_position().height])
# fig.tight_layout()
fig.savefig(os.path.join(savedir, 'EncodingPerformance_ModelVersions_%dsessions.png') % (nSessions), format='png')
# fig.savefig(os.path.join(savedir, 'EncodingPerformance_ResponsiveCells_ModelVersions_%dsessions.png') % (nSessions), format='png')

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
fig.savefig(os.path.join(savedir, 'Legend_ModelVersions.png'), format='png',bbox_inches='tight')


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