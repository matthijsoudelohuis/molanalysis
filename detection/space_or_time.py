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

# from loaddata import * #get all the loading data functions (filter_sessions,load_sessions)
from loaddata.session_info import filter_sessions,load_sessions

# from scipy import stats
from scipy.stats import zscore
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import roc_auc_score as AUC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# from sklearn import preprocessing
# from sklearn import linear_model
# from sklearn.preprocessing import minmax_scale
# from scipy.signal import medfilt
# from sklearn.preprocessing import StandardScaler

from loaddata.get_data_folder import get_local_drive
# import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.patches
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import *
from utils.behaviorlib import * # get support functions for beh analysis 

# from matplotlib.lines import Line2D
# from utils.behaviorlib import * # get support functions for beh analysis 
# from detection.plot_neural_activity_lib import *

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
# session_list = np.array([['LPE12013', '2024_04_26']])
# session_list = np.array([['LPE11997', '2024_04_16']])
# session_list = np.array([['LPE12013', '2024_04_25']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - VR\\Stim\\'
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Alignment\\')
# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - DN regression\\'

#%% 
# for i in range(nSessions):
#     sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -60  #pre cm
s_post      = 60   #post cm
sbinsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=sbinsize,method='binmean')

#%% #################### Spatial runspeed  ####################################
for ises,ses in enumerate(sessions):
    [sessions[ises].runPSTH,bincenters] = calc_runPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=sbinsize)
   
mu_runspeed = np.nanmean(sessions[ises].runPSTH[:,(bincenters>-10) & (bincenters<50)],axis=None)
t_pre       = s_pre/mu_runspeed  #pre sec
t_post      = s_post/mu_runspeed   #post sec
tbinsize     = sbinsize/mu_runspeed  #spatial binning in cm

print('Mean running speed: %1.2f cm/s' % mu_runspeed)
print('%2.1f cm bins would correspond to %1.2f sec bins' % (sbinsize,tbinsize))

#%% ############################### Time Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for spatial binning

# t_pre       = -5  #pre sec
# t_post      = 5   #post sec
# tbinsize     = 0.6  #spatial binning in cm

for i in range(nSessions):
    sessions[i].tensor,tbins    = compute_tensor(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['tStimStart'],
                                       t_pre=t_pre,t_post=t_post,binsize=tbinsize,method='binmean')



#%% 
def plot_neuron_spacetime_alignment(ses,cell_id,sbins,tbins):
    ### Plot activity over space or over time, side by side
    # for the same trials, same neuron
    iN = np.where(ses.celldata['cell_id']==cell_id)[0][0]
    K = np.shape(ses.stensor)[1]

    fig, axes = plt.subplots(1,2,figsize=(5,2.5))
    
    ax = axes[0]

    ttypes = pd.unique(ses.trialdata['trialOutcome'])
    colors = get_clr_outcome(ttypes)

    for k in range(K):
        ax.plot(sbins,ses.stensor[iN,k,:],color='grey',alpha=0.5,linewidth=0.5)

    for i,ttype in enumerate(ttypes):
        idx = ses.trialdata['trialOutcome']==ttype
        data_mean = np.nanmean(ses.stensor[iN,idx,:],axis=0)
        data_error = np.nanstd(ses.stensor[iN,idx,:],axis=0) #/ math.sqrt(sum(idx))
        ax.plot(sbins,data_mean,label=ttype,color=colors[i],linewidth=2)
        ax.fill_between(sbins, data_mean+data_error,  data_mean-data_error, alpha=.2, linewidth=0,color=colors[i])

    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax.set_xlim(sbins[0],sbins[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Activity')

    ax = axes[1]

    for k in range(K):
        ax.plot(tbins,ses.tensor[iN,k,:],color='grey',alpha=0.5,linewidth=0.5)

    for i,ttype in enumerate(ttypes):
        idx = ses.trialdata['trialOutcome']==ttype
        data_mean = np.nanmean(ses.tensor[iN,idx,:],axis=0)
        data_error = np.nanstd(ses.tensor[iN,idx,:],axis=0) #/ math.sqrt(sum(idx))
        ax.plot(tbins,data_mean,label=ttype,color=colors[i],linewidth=2)
        ax.fill_between(tbins, data_mean+data_error,  data_mean-data_error, alpha=.2, linewidth=0,color=colors[i])

    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax.set_xlim(tbins[0],tbins[-1])
    ax.set_xlabel('Time rel. to stimulus onset (s)')
    # ax.set_ylabel('Activity')

    for ax in axes:
        # ylim = my_ceil(np.nanmax(ses.tensor[iN,:,:]),-1)
        smax = np.nanmax(ses.stensor[iN,:,:])
        tmax = np.nanmax(ses.tensor[iN,:,:])
        ylim = my_ceil(np.nanmax([smax,tmax]),-1)
        ax.set_ylim(0,ylim)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
        ax.axvline(x=45, color='b', linestyle='--', linewidth=1)
    # plt.text(3, ylim-3, 'Stim',fontsize=10)
    # plt.text(rewzonestart+3, ylim-3, 'Rew',fontsize=10)
    # plt.tight_layout()
    # plt.title(trialdata['session_id'][0],fontsize=10)
    plt.suptitle(cell_id,fontsize=11,y=0.96)
    plt.tight_layout()
    return fig


#%%
ises = 0

example_cell_ids = ['LPE12385_2024_06_15_0_0075',
'LPE12385_2024_06_15_0_0126',
'LPE12385_2024_06_15_0_0105', # reduction upon stimulus zone
'LPE12385_2024_06_15_0_0114', # noise trial specific response, very nice one
'LPE12385_2024_06_15_0_0183',
'LPE12385_2024_06_15_3_0016',
'LPE12385_2024_06_15_0_0031', # noise trial specific response
'LPE12385_2024_06_15_1_0075', # hit specific activity?
'LPE12385_2024_06_15_1_0475', # very clean response
'LPE12385_2024_06_15_2_0099', # nice responses
'LPE12385_2024_06_15_2_0499'] #variable responsiveness

example_cell_ids = ['LPE12013_2024_04_25_4_0187',
                    'LPE12013_2024_04_25_0_0007',
                    'LPE12013_2024_04_25_1_0046',
                    'LPE12013_2024_04_25_7_0161'] #

example_cell_ids = ['LPE12013_2024_04_26_2_0259',
                    'LPE12013_2024_04_26_2_0250',
                    'LPE12013_2024_04_26_0_0016',
                    'LPE12013_2024_04_26_7_0310',
                    'LPE12013_2024_04_26_0_0017'] #

example_cell_ids = ['LPE11997_2024_04_16_0_0013',
                    'LPE11997_2024_04_16_1_0001',
                    'LPE11997_2024_04_16_2_0047',
                    'LPE11997_2024_04_16_0_0134',
                    'LPE11997_2024_04_16_4_0115'] #


#%% 
example_cell_ids = np.random.choice(sessions[ises].celldata['cell_id'],size=8,replace=False)

#%% Show example neurons that are correlated either to the stimulus signal, lickresponse or to running speed:
# for iN,cell_id in np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]:
for icell,cell_id in enumerate(example_cell_ids):
    if np.isin(cell_id,sessions[ises].celldata['cell_id']):
        plot_neuron_spacetime_alignment(sessions[ises],cell_id,sbins,tbins)

#%% 
idx_N = np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]
# idx_N = sessions[ises].celldata['noise_level']<20

idx_T = np.isin(sessions[ises].trialdata['stimcat'],['C'])
idx_T = np.isin(sessions[ises].trialdata['stimcat'],['M'])
# idx_T = np.isin(sessions[ises].trialdata['stimcat'],['N'])
# idx_T = np.isin(sessions[ises].trialdata['lickResponse'],[1])

data = sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
std_space = np.nanmean(np.nanstd(data,axis=1),axis=0)
data = sessions[ises].tensor[np.ix_(idx_N,idx_T,np.ones(len(tbins)).astype(bool))]
std_time = np.nanmean(np.nanstd(data,axis=1),axis=0)
# std_time = np.nanmean(np.nanstd(sessions[ises].tensor[idx,:,:],axis=1),axis=0)


# std_space = np.nanmean(np.nanstd(sessions[ises].stensor[idx,:,:],axis=1),axis=0)
# std_time = np.nanmean(np.nanstd(sessions[ises].tensor[idx,:,:],axis=1),axis=0)

#%% Compare the variability in responses across spatial or temporal bins:
fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharey=True)
ax = axes[0]
ax.plot(sbins,std_space,color='r')
ax.set_xlim(sbins[0],sbins[-1])
ax.set_xlabel('Stimulus start (cm)')
ax.set_ylabel('Std. trials')
ax.set_title('Space')
ax.grid(True)

ax = axes[1]
ax.plot(tbins,std_time,color='b')
# ax.legend(frameon=False,fontsize=8,loc='upper left')
ax.set_xlim(tbins[0],tbins[-1])
ax.set_xlabel('Stimulus passing (sec)')
ax.set_yticklabels(axes[1].get_yticks())
ax.set_title('Time')
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Var_Comparison_%s.png') % sessions[ises].sessiondata['session_id'][0], format='png')

#%% Decoding performance across space or across time: 


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import zscore
# import sklearn


def find_optimal_lambda(X,y,model_name='LogisticRegression',kfold=5,clip=False):
    assert len(X.shape)==2, 'X must be a matrix of samples by features'
    assert len(y.shape)==1, 'y must be a vector'
    assert X.shape[0]==y.shape[0], 'X and y must have the same number of samples'
    assert model_name in ['LogisticRegression','LinearRegression','Ridge','Lasso','ElasticNet','SVR','SVC']

    # Define the k-fold cross-validation object
    kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

    # Initialize an array to store the decoding performance for each fold
    fold_performance = np.zeros((kfold,))

    # Find the optimal regularization strength (lambda)
    lambdas = np.logspace(-4, 4, 10)
    cv_scores = np.zeros((len(lambdas),))
    for ilambda, lambda_ in enumerate(lambdas):
        
        if model_name == 'LogisticRegression':
            model = LogisticRegression(penalty='l1', solver='liblinear', C=lambda_)
        elif model_name == 'LinearRegression':
            model = LinearRegression(penalty='l1', solver='liblinear', C=lambda_)
        elif model_name == 'Ridge':
            model = Ridge(solver='liblinear', C=lambda_)
        elif model_name == 'Lasso':
            model = Lasso(solver='liblinear', C=lambda_)
        
        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        cv_scores[ilambda] = np.mean(scores)
    optimal_lambda = lambdas[np.argmax(cv_scores)]
    # print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))
    if clip:
        optimal_lambda = np.clip(optimal_lambda, 0.03, 166)
    # optimal_lambda = 1
    return optimal_lambda

def my_classifier_wrapper(Xfull,Yfull,model_name='LogisticRegression',kfold=5,lam=None,subtract_shuffle=True,norm_out=False): 
    assert len(Xfull.shape)==2, 'Xfull must be a matrix of samples by features'
    assert len(Yfull.shape)==1, 'Yfull must be a vector'
    assert Xfull.shape[0]==Yfull.shape[0], 'Xfull and Yfull must have the same number of samples'
    assert model_name in ['LogisticRegression','LinearRegression','Ridge','Lasso','ElasticNet']
    assert lam is None or lam > 0
    
    
    if lam is None:
        lam = find_optimal_lambda(Xfull,Yfull,model_name=model_name,kfold=kfold)

    if model_name == 'LogisticRegression':
        model = LogisticRegression(penalty='l1', solver='liblinear', C=lam)
    elif model_name == 'LinearRegression':
        model = LinearRegression(penalty='l1', solver='liblinear', C=lam)
    elif model_name == 'Ridge':
        model = Ridge(solver='liblinear', C=lam)
    elif model_name == 'Lasso':
        model = Lasso(solver='liblinear', C=lam)

    # Define the number of folds for cross-validation
    kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

    # Initialize an array to store the decoding performance
    performance = np.full((kfold,), np.nan)
    performance_shuffle = np.full((kfold,), np.nan)

    # Loop through each fold
    for ifold, (train_index, test_index) in enumerate(kf.split(Xfull)):
        # Split the data into training and testing sets
        X_train, X_test = Xfull[train_index], Xfull[test_index]
        y_train, y_test = Yfull[train_index], Yfull[test_index]

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

    # Calculate the average decoding performance across folds
    performance_avg = np.mean(performance)
    if subtract_shuffle: # subtract the shuffling performance from the average perf
        performance_avg = np.mean(performance_avg - performance_shuffle)
    if norm_out: # normalize to maximal range of performance (between shuffle and 1)
        performance_avg = performance_avg / (1-np.mean(performance_shuffle))
    
    return performance_avg

#%% Decoding performance across space or across time:

sperf = np.full((nSessions,len(sbins)), np.nan)
tperf = np.full((nSessions,len(tbins)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),desc='Decoding response across sessions'):
    # idx = np.all((ses.trialdata['engaged']==1,ses.trialdata['stimcat']=='N'), axis=0)
    idx_T = np.isin(ses.trialdata['stimcat'],['C','M'])
    idx_N = ses.celldata['roi_name']=='V1'

    if np.sum(idx_T) > 50:
        # Get the maximum signal vs catch for this session
        y = (ses.trialdata['stimcat'][idx_T] == 'M').to_numpy()

        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        X = np.mean(ses.stensor[np.ix_(idx_N,idx_T,((sbins>-5) & (sbins<20)).astype(bool))],axis=2)
        X = X.T

        # X = np.stack((ses.runPSTH[idx,:], ses.pupilPSTH[idx,:], ses.videomePSTH[idx,:], ses.lickPSTH[idx,:]), axis=2)
        # X = np.nanmean(X, axis=1)
        X = zscore(X, axis=0)
        # X = X[:,np.all(~np.isnan(X),axis=0)]
        # X = X[:,np.all(~np.isinf(X),axis=0)]

        optimal_lambda = find_optimal_lambda(X,y,model_name='LogisticRegression',kfold=5)
        # Loop through each spatial bin
        for ibin, bincenter in enumerate(sbins):
            X = ses.stensor[np.ix_(idx_N,idx_T,sbins==bincenter)].squeeze()
            X = X.T
            X = zscore(X, axis=0)

            # Calculate the average decoding performance across folds
            sperf[ises,ibin] = my_classifier_wrapper(X,y,model_name='LogisticRegression',kfold=5,lam=optimal_lambda)

        # Loop through each time bin
        for ibin, bincenter in enumerate(tbins):
            X = ses.tensor[np.ix_(idx_N,idx_T,tbins==bincenter)].squeeze()
            X = X.T
            X = zscore(X, axis=0)

            # Calculate the average decoding performance across folds
            tperf[ises,ibin] = my_classifier_wrapper(X,y,model_name='LogisticRegression',kfold=5,lam=optimal_lambda)

# #%% Show the decoding performance
# fig,ax = plt.subplots(1,1,figsize=(4,3))
# for i,ses in enumerate(sessions):
#     if np.any(performance[i,:]):
#         ax.plot(bincenters,performance[i,:],color='grey',alpha=0.5,linewidth=1)
# shaded_error(bincenters,performance,error='sem',ax=ax,color='b')
# ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
# ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
# ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
# ax.axvline(x=45, color='b', linestyle='--', linewidth=1)

# ax.set_xlabel('Position relative to stim (cm)')
# ax.set_ylabel('Decoding Performance \n (accuracy - shuffle)')
# ax.set_title('Decoding Performance')
# ax.set_xlim([-80,60])
# plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'Spatial', 'LogisticDecodingPerformance_LickResponse.png'), format='png')

