# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.model_selection import KFold
from scipy.stats import zscore, pearsonr,spearmanr
from rastermap import Rastermap, utils
from utils.plotting_style import * #get all the fixed color schemes

# %matplotlib inline
savedir = 'E:\\OneDrive\\PostDoc\\Figures\\PCA - Images and gratings\\'

sessions            = filter_sessions(protocols = ['GR','IM'],load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=False, calciumversion='dF')

nSessions = len(sessions)

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

################# PCA on full session  ##############################

protocols = np.unique(sessiondata.protocol)
clr_protocols = get_clr_protocols(protocols)

area = 'V1'
nPCs = 400
EVratio = np.full((nSessions,nPCs),np.nan)

for ises in range(nSessions):
    X                   = sessions[ises].calciumdata.iloc[:,np.where(sessions[ises].celldata['roi_name']==area)[0]]
    # mat_zsc             = zscore(sessions[ises].calciumdata,axis=0)
    mat_zsc             = zscore(X,axis=0)

    pca               = PCA(n_components=nPCs,svd_solver="randomized") #construct PCA object with specified number of components
    Xp                = pca.fit_transform(mat_zsc) #fit pca to response matrix
    EVratio[ises,:]        = pca.explained_variance_ratio_


fig = plt.subplots(figsize=(5,2.5))
ax = plt.subplot(121)
for ises in range(nSessions):
    sns.lineplot(data=np.cumsum(EVratio[ises,:]),
                    color=clr_protocols[np.where(protocols==sessions[ises].protocol)[0][0]])
plt.xlim([-1,nPCs])
plt.ylim([0,0.75])
ax.set_xlabel('#PCs')
ax.set_ylabel('Explained Variance')
sns.despine()
handles= []
for i,protocol in enumerate(protocols):
    handles.append(ax.add_line(plt.plot(0,0,alpha=1,linewidth=2,color=clr_protocols[i])[0]))
ax.legend(handles,protocols,loc='lower right',frameon=False)
ax2 = plt.subplot(122)
for ises in range(nSessions):
    sns.lineplot(data=np.cumsum(EVratio[ises,:]),
                    color=clr_protocols[np.where(protocols==sessions[ises].protocol)[0][0]])
plt.xlim([-1,100])
plt.ylim([0,0.35])
ax2.set_xlabel('#PCs')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(savedir,'EV_GRvsIM_6sessions_V1_FullSession' + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'EV_GRvsIM_6sessions_FullSession' + '.png'), format = 'png')

##############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s
binmethod = 'nearby'

for i in range(nSessions):
    [sessions[i].tensor,tbins] = compute_tensor(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['tOnset'], 
                                    t_pre, t_post, binsize,method=binmethod)

for i in range(nSessions):
    sessions[i].respmat         = sessions[i].tensor[:,:,np.logical_and(tbins > 0,tbins < 1.25)].mean(axis=2) #compute average poststimulus response


# #hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
# temp = pd.DataFrame(np.reshape(np.array(sessions[0].behaviordata['runspeed']),(len(sessions[0].behaviordata['runspeed']),1)))
# respmat_runspeed = compute_respmat(temp, sessions[0].behaviordata['ts'], sessions[0].trialdata['tOnset'],
#                                    t_resp_start=0,t_resp_stop=1,method='mean')
# respmat_runspeed = np.squeeze(respmat_runspeed)

################# PCA on averaged response during stimuli ##############################

area = 'V1'
nPCs = 400
EVratio = np.full((nSessions,nPCs),np.nan)

for ises in range(nSessions):
    # X                 = sessions[ises].respmat[:,np.where(sessions[ises].celldata['roi_name']==area)[0]]
    X                 = sessions[ises].respmat[np.where(sessions[ises].celldata['roi_name']==area)[0],:]
    mat_zsc           = zscore(X.T,axis=0)

    pca               = PCA(n_components=nPCs) #construct PCA object with specified number of components
    Xp                = pca.fit_transform(mat_zsc) #fit pca to response matrix
    EVratio[ises,:]        = pca.explained_variance_ratio_


fig = plt.subplots(figsize=(5,2.5))
ax = plt.subplot(121)
for ises in range(nSessions):
    sns.lineplot(data=np.cumsum(EVratio[ises,:]),
                    color=clr_protocols[np.where(protocols==sessions[ises].protocol)[0][0]])
plt.xlim([-1,nPCs])
# plt.ylim([0,0.75])
ax.set_xlabel('#PCs')
ax.set_ylabel('Explained Variance')
sns.despine()
handles= []
for i,protocol in enumerate(protocols):
    handles.append(ax.add_line(plt.plot(0,0,alpha=1,linewidth=2,color=clr_protocols[i])[0]))
ax.legend(handles,protocols,loc='lower right',frameon=False)
ax2 = plt.subplot(122)
for ises in range(nSessions):
    sns.lineplot(data=np.cumsum(EVratio[ises,:]),
                    color=clr_protocols[np.where(protocols==sessions[ises].protocol)[0][0]])
plt.xlim([-1,40])
plt.ylim([0,0.45])
ax2.set_xlabel('#PCs')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(savedir,'EV_GRvsIM_6sessions_V1_Stimuli' + '.png'), format = 'png')

################# Estimate Dimensionality of dataset: ##############################


# for ises in range(nSessions):
#     X                   = sessions[ises].respmat[np.where(sessions[ises].celldata['roi_name']==area)[0],:]
#     X                   = X[np.random.choice(np.shape(X)[0],np.min((nNeurons,np.shape(X)[0])),replace=False),:]
#     X                   = zscore(X.T,axis=0)

#     if dimreducmodel=='PCA':
#         model                 = PCA(n_components=n_components,random_state=0,svd_solver='randomized') #construct PCA object with specified number of components
#         randmodel             = PCA(n_components=n_components,random_state=0,svd_solver='randomized') #construct PCA object with specified number of components
#     elif dimreducmodel=='FA':
#         model                 = FactorAnalysis(n_components=n_components,random_state=0) #construct object with specified number of components
#         randmodel             = FactorAnalysis(n_components=n_components,random_state=0) #construct object with specified number of components

#     # Implementing cross validation
#     kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)

#     for k,(train_index, test_index) in enumerate(kf.split(X)):
#         print('Session %d, fold %d',ises,k)
#         X_train , X_test = X[train_index,:],X[test_index,:]
        
#         model.fit(X_train)

#         X_scrambled             = X_train.flatten()
#         X_scrambled             = np.random.choice(X_scrambled,size=np.shape(X_train))
#         randmodel.fit(X_scrambled)

#         for nComp in range(n_components):
#             ### Xhat = np.dot(model.transform(X_train)[:,:nComp], model.components_[:nComp,:])
#             Xhat = np.outer(model.transform(X_train)[:,nComp], model.components_[nComp,:])
#             EV[ises,k,nComp] = 1 - np.var(X_train-Xhat) / np.var(X_train)
#             ### Xhat = np.dot(randmodel.transform(X_test)[:,:nComp], randmodel.components_[:nComp,:])
#             Xhat = np.outer(randmodel.transform(X_test)[:,nComp], randmodel.components_[nComp,:])
#             EV_scr[ises,k,nComp]  = 1 - np.var(X_test-Xhat) / np.var(X_test)


def DimReduc_crossval(data,dimreducmodel='PCA',n_components=100,kFold=5):
    
    #init output vars:
    EV_train            = np.zeros((kFold,n_components))
    EV_test             = np.zeros((kFold,n_components))
    EV_scr              = np.zeros((kFold,n_components))

    X                   = zscore(data,axis=0)

    if dimreducmodel=='PCA':
        model                 = PCA(n_components=n_components,random_state=0,svd_solver='randomized') #construct PCA object with specified number of components
        randmodel             = PCA(n_components=n_components,random_state=0,svd_solver='randomized') #construct PCA object with specified number of components
    elif dimreducmodel=='FA':
        model                 = FactorAnalysis(n_components=n_components,random_state=0) #construct object with specified number of components
        randmodel             = FactorAnalysis(n_components=n_components,random_state=0) #construct object with specified number of components

    # Implementing cross validation
    kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)

    for k,(train_index, test_index) in enumerate(kf.split(X)):
        print('Session %d, fold %d' % (ises,k))
        X_train , X_test = X[train_index,:],X[test_index,:]
        
        model.fit(X_train)

        X_scrambled             = X_train.flatten()
        X_scrambled             = np.random.choice(X_scrambled,size=np.shape(X_train))
        randmodel.fit(X_scrambled)

        for nComp in range(n_components):
            Xhat = np.outer(model.transform(X_train)[:,nComp], model.components_[nComp,:])
            EV_train[k,nComp] = 1 - np.var(X_train-Xhat) / np.var(X_train)
        
            Xhat = np.outer(model.transform(X_test)[:,nComp], model.components_[nComp,:])
            EV_test[k,nComp] = 1 - np.var(X_test-Xhat) / np.var(X_test)

            ### Xhat = np.dot(randmodel.transform(X_test)[:,:nComp], randmodel.components_[:nComp,:])
            Xhat = np.outer(randmodel.transform(X_test)[:,nComp], randmodel.components_[nComp,:])
            EV_scr[k,nComp]  = 1 - np.var(X_test-Xhat) / np.var(X_test)
    
    return EV_train,EV_test,EV_scr

# Let's assume that train and test distributions are very similar and we can use the same principle components. If X_train and X_test are two pxn and pxm. (n and m are number of samples).

# import numpy as np
# X_train = X_train - np.mean(X_train,axis=1)[:,np.newaxis]
# X_test  = X_test  - np.mean(X_test, axis=1)[:,np.newaxis]
# Sigma_train = np.dot(X_train,X_train.T)/n
# V,U = np.linalg.eigh(Sigma_train)
# Using the assumption mentioned above, you can calculate the projections of your data:

# Y_test = np.dot(U.T,X_test)
# The variance of each row is the test variance along the principle components.

# Y_var = np.sum(Y_test**2,axis=1)/m


## 

area            = 'V1'
kFold           = 5
dimreducmodel   = 'PCA' # or FA
n_components    = 200
nNeurons        = 500
nTrials         = 10000

#init output vars:
EV_train            = np.zeros((nSessions,kFold,n_components))
EV_test             = np.zeros((nSessions,kFold,n_components))
EV_scr              = np.zeros((nSessions,kFold,n_components))

for ises in range(nSessions):
    X                   = sessions[ises].respmat[np.where(sessions[ises].celldata['roi_name']==area)[0],:]
    X                   = X[np.random.choice(np.shape(X)[0],np.min((nNeurons,np.shape(X)[0])),replace=False),:]
    X                   = X[:,np.random.choice(np.shape(X)[1],np.min((nTrials,np.shape(X)[1])),replace=False)]
    [EV_train[ises,:,:],EV_test[ises,:,:],EV_scr[ises,:,:]] = DimReduc_crossval(X.T,dimreducmodel,n_components,kFold)


#%% Figure of the explained variance vs scrambled across sessions:

exampleses = 1

fig = plt.subplots(figsize=(12,4))
ax1 = plt.subplot(131)

x = np.arange(n_components)
mean_1 = np.mean(EV_train[exampleses,:,:],axis=0)
std_1 = np.std(EV_train[exampleses,:,:],axis=0)
line_1, = ax1.plot(x, mean_1, 'r-')
fill_1 = ax1.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='r', alpha=0.2)

mean_2 = np.mean(EV_test[exampleses,:,:],axis=0)
std_2 = np.std(EV_test[exampleses,:,:],axis=0)
line_2, = ax1.plot(x, mean_2, 'b-')
fill_2 = ax1.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='k', alpha=0.2)

mean_3 = np.mean(EV_scr[exampleses,:,:],axis=0)
std_3 = np.std(EV_scr[exampleses,:,:],axis=0)
line_3, = ax1.plot(x, mean_3, 'k--')
fill_3 = ax1.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
ax1.margins(x=0)
ax1.set_yscale('log')
ax1.set_title('Example Session')
sns.despine()
ax1.legend([line_1,line_2,line_3],['Train','Test','Shuffle'],loc='upper right',frameon=False)
ax1.set_ylabel('Expl. Variance')

ax2 = plt.subplot(132)

for ises in range(nSessions):
    sns.lineplot(data=np.mean(EV_test[ises,:,:],axis=0),linestyle='-',
                    color=clr_protocols[np.where(protocols==sessions[ises].protocol)[0][0]])
    sns.lineplot(data=np.mean(EV_scr[ises,:,:],axis=0),linestyle=':',
                color=clr_protocols[np.where(protocols==sessions[ises].protocol)[0][0]])

handles= []
for i,protocol in enumerate(protocols):
    handles.append(ax2.add_line(plt.plot(0,0,alpha=1,linewidth=2,color=clr_protocols[i])[0]))
ax2.legend(handles,protocols,loc='upper right',frameon=False)
ax2.margins(x=0)
ax2.set_title('All sessions')
ax2.set_yscale('log')
ax2.set_xlabel('#Components')

ax3 = plt.subplot(133)
for ises in range(nSessions):
    sns.lineplot(data=np.mean(EV_test[ises,:,:],axis=0),linestyle='-',
                    color=clr_protocols[np.where(protocols==sessions[ises].protocol)[0][0]])
    sns.lineplot(data=np.mean(EV_scr[ises,:,:],axis=0),linestyle=':',
                color=clr_protocols[np.where(protocols==sessions[ises].protocol)[0][0]])

ax3.legend(handles,protocols,loc='upper right',frameon=False)
ax3.margins(x=0)
ax3.set_yscale('log')
ax3.set_title('Close up')
ax3.set_xlim([25,100])
ax3.set_ylim([0.0015,0.007])

plt.savefig(os.path.join(savedir,'EV_Dimensionality_' + dimreducmodel + '_V1_500neurons' + '.png'), format = 'png')

#%% ########################### Requirements for assessing dimensionality #############
### How many trials do I need? ###########
# Run for different numbers of trials:

area            = 'V1'
kFold           = 5
dimreducmodel   = 'PCA' # or FA
nmaxcomponents  = 200
nNeurons        = 500

nTrials_samples     = [5,10,20,50,100,200,500,1000,2000,3200,5600]
# nTrials_samples     = [5,10,20,50,100,200]

#init output vars:
EV_train            = np.zeros((nSessions,len(nTrials_samples),kFold,nmaxcomponents))
EV_train.fill(np.nan)
EV_test             = np.zeros((nSessions,len(nTrials_samples),kFold,nmaxcomponents))
EV_test.fill(np.nan)
EV_scr              = np.zeros((nSessions,len(nTrials_samples),kFold,nmaxcomponents))
EV_scr.fill(np.nan)

for ises in range(nSessions):
    for inKs,nK in enumerate(nTrials_samples):
        X                   = sessions[ises].respmat[np.where(sessions[ises].celldata['roi_name']==area)[0],:]
        # N,K = np.shape(X)
        if nK<np.shape(X)[1]: #only if #sampled neurons is lower than number of recorded neurons in area X
            print(nK)
            X                   = X[np.random.choice(np.shape(X)[0],np.min((nNeurons,np.shape(X)[0])),replace=False),:]
            X                   = X[:,np.random.choice(np.shape(X)[1],nK,replace=False)]
            N,K = np.shape(X)
            # n_components        = np.min(np.shape(X) + (n_components,))
            n_components        = np.min((N,int(np.floor(K / kFold  * (kFold-1))),nmaxcomponents))
            [EV_train[ises,inKs,:,:n_components],EV_test[ises,inKs,:,:n_components],EV_scr[ises,inKs,:,:n_components]] = DimReduc_crossval(X.T,dimreducmodel,n_components,kFold)



EV_train_cum = np.cumsum(EV_train,axis=3)
EV_test_cum = np.cumsum(EV_test,axis=3)
EV_scr_cum = np.cumsum(EV_scr,axis=3)

mean_test  = np.nanmean(np.nanmean(EV_test_cum,axis=2),axis=0)
std_test   = np.nanstd(np.nanmean(EV_test_cum,axis=2),axis=0)

mean_scr  = np.nanmean(np.nanmean(EV_scr_cum,axis=2),axis=0)
std_scr   = np.nanstd(np.nanmean(EV_scr_cum,axis=2),axis=0)

clr_trialcounts            = sns.color_palette('inferno', len(nTrials_samples))
x = np.arange(n_components)

fig = plt.subplots(figsize=(10,4))
ax1 = plt.subplot(121)

handles = []
for inKs,nK in enumerate(nTrials_samples):
    line, = ax1.plot(x, mean_test[inKs,:], color=clr_trialcounts[inKs],linestyle='-')
    handles.append(line)
    ax1.fill_between(x, mean_test[inKs,:] - std_test[inKs,:], mean_test[inKs,:] + std_test[inKs,:], 
                              color=clr_trialcounts[inKs], alpha=0.2)
    
    ax1.plot(x, mean_scr[inKs,:], color='k',linestyle=':')
    ax1.fill_between(x, mean_scr[inKs,:] - std_scr[inKs,:], mean_scr[inKs,:] + std_scr[inKs,:], 
                              color='k', alpha=0.2)

line_scr, = ax1.plot([0],[0], color='k',linestyle=':')
handles.append(line_scr)
sns.despine()

ax1.set_ylabel('Cumulative Expl. Variance')
ax1.set_xlabel('#Components')
# ax1.set_title('Cumulative Trial Counts')
# pos = ax1.get_position()
# ax1.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
# ax1.legend(handles,nTrials_samples + ['Shuffle'],frameon=False,title='#Trials',
        #    loc='center right', bbox_to_anchor=(1.3, 0.5))



mean_test  = np.nanmean(np.nanmean(EV_test,axis=2),axis=0)
std_test   = np.nanstd(np.nanmean(EV_test,axis=2),axis=0)

mean_scr  = np.nanmean(np.nanmean(EV_scr,axis=2),axis=0)
std_scr   = np.nanstd(np.nanmean(EV_scr,axis=2),axis=0)

ax2 = plt.subplot(122)

handles = []
for inKs,nK in enumerate(nTrials_samples):
    line, = ax2.plot(x, mean_test[inKs,:], color=clr_trialcounts[inKs],linestyle='-')
    handles.append(line)
    ax2.fill_between(x, mean_test[inKs,:] - std_test[inKs,:], mean_test[inKs,:] + std_test[inKs,:], 
                              color=clr_trialcounts[inKs], alpha=0.2)
    
    ax2.plot(x, mean_scr[inKs,:], color='k',linestyle=':')
    ax2.fill_between(x, mean_scr[inKs,:] - std_scr[inKs,:], mean_scr[inKs,:] + std_scr[inKs,:], 
                              color='k', alpha=0.2)

line_scr, = ax2.plot([0],[0], color='k',linestyle=':')
handles.append(line_scr)
sns.despine()

ax2.set_yscale('log')
ax2.set_ylabel('Expl. Variance')
ax2.set_xlabel('Component #')
# ax2.set_title('Required Trial Counts')
pos = ax2.get_position()
ax2.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax2.legend(handles,nTrials_samples + ['Shuffle'],frameon=False,title='#Trials',
           loc='center right', bbox_to_anchor=(1.3, 0.5))
ax2.set_xlim([0,100])
ax2.set_ylim([0.001,0.1])

plt.savefig(os.path.join(savedir,'EV_TrialCounts_' + dimreducmodel + '_V1_500neurons' + '.png'), format = 'png')

