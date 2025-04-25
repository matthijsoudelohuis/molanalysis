# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('c:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\CCA\\GR\\')

#%% 
session_list        = np.array([['LPE12223','2024_06_10'], #GR
                                ['LPE10919','2023_11_06']]) #GR
# session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                # ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%%  Load data properly:        
calciumversion = 'dF'
# calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=True)
    
#%% ########################### Compute tuning metrics: ###################################
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)


#%% 


#%% get optimal lambda
nsampleneurons  = 500
lambdas         = np.logspace(-6, 5, 10)
# lambdas         = np.array([0,0.01,0.1,1])
nlambdas        = len(lambdas)
kfold           = 5
nranks          = 50

R2_cv           = np.full((nSessions,nlambdas,nranks,kfold),np.nan)

for ises,ses in enumerate(sessions):
    idx_T               = ses.trialdata['Orientation']==0
    # idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
        
        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(Y,axis=0)

        # Explanation of steps
        # X is of shape K x N (samples by features), Y is of shape K x M
        # K is the number of samples, N is the number of neurons in area 1,
        # M is the number of neurons in area 2

        # multiple linear regression, B_hat is of shape N x M:
        # B_hat               = LM(Y,X, lam=lam) 
        #RRR: do SVD decomp of Y_hat, 
        # U is of shape K x r, S is of shape r x r, V is of shape r x M
        # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r) 

        for ilam,lam in enumerate(lambdas):
            # cross-validation version
            # R2_cv   = np.zeros(kfold)
            kf      = KFold(n_splits=kfold)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)

                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = linalg.svd(Y_hat_train)
                S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,ilam,r,ikf] = EV(Y_test,Y_hat_rr_test)

#%% 
R2data = np.full((nSessions,nlambdas),np.nan)
rankdata = np.full((nSessions,nlambdas),np.nan)

for ises,ses in enumerate(sessions):
    for ilam,lam in enumerate(lambdas):
        R2data[ises,ilam],rankdata[ises,ilam] = rank_from_R2(R2_cv[ises,ilam,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',nlambdas)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for ilam,lam in enumerate(lambdas):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(lambdas,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(lambdas,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(lambdas,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.max(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(lambdas,R2data[ises,:],color='grey',linewidth=1)
ax.plot(lambdas,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(lambdas,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.max(R2data)+.05])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'RRR_perOri_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')
# plt.savefig(os.path.join(savedir,'RRR_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')

#%% get optimal pre pCA
nsampleneurons  = 500
PCAdims         = np.array([1,2,5,10,20,50,100])
# lambdas         = np.array([0,0.01,0.1,1])
nPCAdims        = len(PCAdims)
kfold           = 5
nranks          = 50
lam             = 5000
R2_cv           = np.full((nSessions,nPCAdims,nranks,kfold),np.nan)

for ises,ses in enumerate(sessions):
    idx_T               = ses.trialdata['Orientation']==0
    # idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
        
        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(Y,axis=0)

        # Explanation of steps
        # X is of shape K x N (samples by features), Y is of shape K x M
        # K is the number of samples, N is the number of neurons in area 1,
        # M is the number of neurons in area 2

        # multiple linear regression, B_hat is of shape N x M:
        # B_hat               = LM(Y,X, lam=lam) 
        #RRR: do SVD decomp of Y_hat, 
        # U is of shape K x r, S is of shape r x r, V is of shape r x M
        # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r) 

        for ipc,PCAdim in enumerate(PCAdims):
            # cross-validation version
            # R2_cv   = np.zeros(kfold)

            Xmodel      = PCA(n_components=PCAdim)
            Xpca        = Xmodel.fit_transform(X)
            Ymodel      = PCA(n_components=PCAdim)
            Ypca        = Ymodel.fit_transform(Y)

            kf          = KFold(n_splits=kfold)

            for ikf, (idx_train, idx_test) in enumerate(kf.split(Xpca)):
                
                X_train, X_test     = Xpca[idx_train], Xpca[idx_test]
                Y_train, Y_test     = Ypca[idx_train], Ypca[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)

                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = linalg.svd(Y_hat_train)
                S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,ipc,r,ikf] = EV(Y_test,Y_hat_rr_test) * Ymodel.explained_variance_ratio_.sum()


#%% 
R2data = np.full((nSessions,nPCAdims),np.nan)
rankdata = np.full((nSessions,nPCAdims),np.nan)

for ises,ses in enumerate(sessions):
    for ipc,PCAdim in enumerate(PCAdims):
        R2data[ises,ipc],rankdata[ises,ipc] = rank_from_R2(R2_cv[ises,ipc,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',nPCAdims)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for ilam,lam in enumerate(PCAdims):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(PCAdims,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(PCAdims,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(PCAdims,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('PCAdim')
ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.max(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(PCAdims,R2data[ises,:],color='grey',linewidth=1)
ax.plot(PCAdims,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(PCAdims,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('PCAdim')
ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.max(R2data)+.05])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'RRR_PrePCA_lam5000_Rank_%dneurons.png' % nsampleneurons), format = 'png')
# plt.savefig(os.path.join(savedir,'RRR_PrePCA_lam0_Rank_%dneurons.png' % nsampleneurons), format = 'png')
plt.savefig(os.path.join(savedir,'RRR_PrePCA_lam0_perOri_Rank_%dneurons.png' % nsampleneurons), format = 'png')
