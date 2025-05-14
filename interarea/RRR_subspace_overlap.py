# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *
from utils.CCAlib import *
from utils.corr_lib import *
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\SubspaceOverlap')


#%% 
areas = ['V1','PM','AL','RSP']
nareas = len(areas)

#%% Load example sessions:
session_list        = np.array(['LPE12385_2024_06_13','LPE11998_2024_05_02']) #GN
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list)

# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
# sessions,nSessions   = filter_sessions(protocols = 'GN',only_all_areas=areas,filter_areas=areas)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_areas=areas)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

#%% Cross area subspace difference as a function of dimensionality: 
def cross_area_subspace_wrapper(sessions,arealabels,popsize,maxnoiselevel=20,nmodelfits=10,kfold=5,lam=0,nranks=25,filter_nearby=False):
    """
    Computes the cross area subspace difference as a function of dimensionality.

    Parameters:
    - sessions: list of session objects
    - arealabels: list of area labels
    - popsize: int, number of neurons in a population
    - maxnoiselevel: int, maximum noise level to include in the data
    - nmodelfits: int, number of model fits to do
    - kfold: int, number of folds for cross-validation
    - lam: float, regularization strength for linear regression
    - nranks: int, number of subspace dimensions to test
    - filter_nearby: bool, if True, only includes cells in the data that are close to labeled cells in the same area

    Returns:
    - same_pop_R2, cross_pop_R2: 6D arrays of shape (nranks,nSessions,narealabels,narealabels,narealabels,nmodelfits,kfold)
        cross_pop_R2[i,j,k,m,n,o,p] is the R2 of predicting area n using the predictive subspace of area k to m, 
        while using the i-th rank subspace of area k. This is for session j, resampling o and kfold p
        same_pop_R2[i,j,k,m,n,o,p] is the same but area n is ignored, the test performance on held out data 
        is reported for each area combination k to m
    """

    narealabels         = len(arealabels)
    same_pop_R2         = np.full((nranks,nSessions,narealabels,narealabels,narealabels,nmodelfits,kfold),np.nan)
    cross_pop_R2        = np.full((nranks,nSessions,narealabels,narealabels,narealabels,nmodelfits,kfold),np.nan)

    kf                  = KFold(n_splits=kfold,shuffle=True,random_state=None)
    for ises,ses in tqdm(enumerate(sessions),desc='Cross subspace decoding: ',total=nSessions):
        idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
        K                   = np.sum(idx_T)
        resp                = zscore(sessions[ises].respmat[:,idx_T].T,axis=0,nan_policy='omit')

        if filter_nearby:
            nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
        else: 
            nearfilter      = np.ones(len(sessions[ises].celldata)).astype(bool)

        for imf in range(nmodelfits):
            for iarea in range(narealabels):
                idx_N_i           = np.where(np.all((sessions[ises].celldata['arealabel']==arealabels[iarea],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,nearfilter	
                                        ),axis=0))[0]
                for jarea in range(narealabels):
                    idx_N_j           = np.where(np.all((sessions[ises].celldata['arealabel']==arealabels[jarea],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,nearfilter	
                                        ),axis=0))[0]
                    for karea in range(narealabels):
                        idx_N_k           = np.where(np.all((sessions[ises].celldata['arealabel']==arealabels[karea],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,nearfilter	
                                        ),axis=0))[0]

                        #SUBSAMPLE NEURONS FROM AREAS: SKIP LOOP IF NOT ENOUGH NEURONS
                        if len(idx_N_i)<popsize:
                            continue
                        idx_N_i_sub         = np.random.choice(idx_N_i,popsize,replace=False) #take random subset of neurons
                        
                        if len(np.setdiff1d(idx_N_j,idx_N_i_sub))<popsize:
                            continue
                        idx_N_j_sub         = np.random.choice(np.setdiff1d(idx_N_j,idx_N_i_sub),popsize,replace=False) #take random subset of neurons
                        
                        # if len(np.setdiff1d(idx_N_j,np.concatenate((idx_N_i_sub,idx_N_j_sub))))<popsize:
                        #     continue
                        # idx_N_j_sub2         = np.random.choice(np.setdiff1d(idx_N_j,idx_N_i_sub),popsize,replace=False) #take random subset of neurons
                        
                        if len(np.setdiff1d(idx_N_k,np.concatenate((idx_N_i_sub,idx_N_j_sub))))<popsize:
                            continue
                        idx_N_k_sub         = np.random.choice(np.setdiff1d(idx_N_k,np.concatenate((idx_N_i_sub,idx_N_j_sub))),popsize,replace=False) #take random subset of neurons
                        
                        # Assert that the number of unique values equals the total number of values
                        all_values = np.concatenate([idx_N_i_sub, idx_N_j_sub, idx_N_k_sub])
                        assert len(np.unique(all_values)) == len(all_values), "Arrays contain overlapping values"

                        X,Y,Z = resp[:,idx_N_i_sub],resp[:,idx_N_j_sub],resp[:,idx_N_k_sub]
                        # X,Y,Z,Y2 = resp[:,idx_N_i_sub],resp[:,idx_N_j_sub],resp[:,idx_N_k_sub],resp[:,idx_N_j_sub2]

                        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                            X_train, X_test = X[idx_train], X[idx_test]
                            Y_train, Y_test = Y[idx_train], Y[idx_test]
                            Z_train, Z_test = Z[idx_train], Z[idx_test]
                            # Y2_train, Y2_test = Y2[idx_train], Y2[idx_test]

                            B_hat_train         = LM(Y_train,X_train, lam=lam)

                            Y_hat_train         = X_train @ B_hat_train

                            # decomposing and low rank approximation of A
                            # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
                            U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                            for r in range(nranks):
                                B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                            
                                Y_hat_test_rr   = X_test @ B_rrr

                                same_pop_R2[r,ises,iarea,jarea,karea,imf,ikf] = EV(Y_test,Y_hat_test_rr)
        
                                B_hat           = LM(Z_train,X_train @ B_rrr, lam=lam)
        
                                Z_hat_test_rr   = X_test @ B_rrr @ B_hat
                                
                                cross_pop_R2[r,ises,iarea,jarea,karea,imf,ikf] = EV(Z_test,Z_hat_test_rr)

                                # B_hat           = LM(Y2_train,X_train @ B_rrr, lam=lam)
        
                                # Y2_hat_test_rr   = X_test @ B_rrr @ B_hat
                                
                                # subspace_R2[iarea,jarea,karea,2,r,ises,imf,ikf] = EV(Y2_test,Y2_hat_test_rr)
    return same_pop_R2, cross_pop_R2

#%% Cross area subspace difference as a function of dimensionality: 

popsize         = 100
nmodelfits      = 5
kfold           = 2
lam             = 0
nranks          = 50
maxnoiselevel   = 20

# Init output arrays:
subspace_R2         = np.full((3,nranks,nSessions,nmodelfits,kfold),np.nan)

areaX               = 'V1unl'
areaY               = 'PMunl'
areaZ               = 'ALunl'
narealabels         = 1

kf                  = KFold(n_splits=kfold,shuffle=True,random_state=None)
for ises,ses in tqdm(enumerate(sessions),desc='Cross subspace decoding: ',total=nSessions):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    K                   = np.sum(idx_T)
    resp                = zscore(sessions[ises].respmat[:,idx_T].T,axis=0,nan_policy='omit')

    for imf in range(nmodelfits):
        for iarea in range(narealabels):
            idx_N_i           = np.where(np.all((sessions[ises].celldata['arealabel']==areaX,
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,	
                                    ),axis=0))[0]
            for jarea in range(narealabels):
                idx_N_j           = np.where(np.all((sessions[ises].celldata['arealabel']==areaY,
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,	
                                    ),axis=0))[0]
                for karea in range(narealabels):
                    idx_N_k           = np.where(np.all((sessions[ises].celldata['arealabel']==areaZ,
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,	
                                    ),axis=0))[0]
                    
                    #SUBSAMPLE NEURONS FROM AREAS: SKIP LOOP IF NOT ENOUGH NEURONS
                    if len(idx_N_i)<popsize:
                        continue
                    idx_N_i_sub         = np.random.choice(idx_N_i,popsize,replace=False) #take random subset of neurons
                    if len(np.setdiff1d(idx_N_j,idx_N_i_sub))<popsize:
                        continue
                    idx_N_j_sub         = np.random.choice(np.setdiff1d(idx_N_j,idx_N_i_sub),popsize,replace=False) #take random subset of neurons
                    
                    if len(np.setdiff1d(idx_N_j,np.concatenate((idx_N_i_sub,idx_N_j_sub))))<popsize:
                        continue
                    # idx_N_j_sub2         = np.random.choice(np.setdiff1d(idx_N_j,idx_N_i_sub,idx_N_j_sub),popsize,replace=False) #take random subset of neurons
                    idx_N_j_sub2         = np.random.choice(np.setdiff1d(idx_N_j,np.concatenate((idx_N_i_sub,idx_N_j_sub))),popsize,replace=False) #take random subset of neurons
                    
                    if len(np.setdiff1d(idx_N_k,np.concatenate((idx_N_i_sub,idx_N_j_sub))))<popsize:
                        continue
                    idx_N_k_sub         = np.random.choice(np.setdiff1d(idx_N_k,np.concatenate((idx_N_i_sub,idx_N_j_sub,idx_N_j_sub2))),popsize,replace=False) #take random subset of neurons
                    
                    assert(len(np.intersect1d(idx_N_i_sub,idx_N_j_sub))==0 and len(np.intersect1d(idx_N_i_sub,idx_N_k_sub))==0 and len(np.intersect1d(idx_N_j_sub,idx_N_k_sub))==0)

                    X,Y,Z,Y2 = resp[:,idx_N_i_sub],resp[:,idx_N_j_sub],resp[:,idx_N_k_sub],resp[:,idx_N_j_sub2]

                    for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                        X_train, X_test = X[idx_train], X[idx_test]
                        Y_train, Y_test = Y[idx_train], Y[idx_test]
                        Z_train, Z_test = Z[idx_train], Z[idx_test]
                        Y2_train, Y2_test = Y2[idx_train], Y2[idx_test]

                        B_hat_train         = LM(Y_train,X_train, lam=lam)

                        Y_hat_train         = X_train @ B_hat_train

                        # decomposing and low rank approximation of A
                        # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
                        U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                        U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                        for r in range(nranks):
                            B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                           
                            Y_hat_test_rr   = X_test @ B_rrr

                            subspace_R2[0,r,ises,imf,ikf] = EV(Y_test,Y_hat_test_rr)
    
                            B_hat           = LM(Z_train,X_train @ B_rrr, lam=lam)
    
                            Z_hat_test_rr   = X_test @ B_rrr @ B_hat
                            
                            subspace_R2[1,r,ises,imf,ikf] = EV(Z_test,Z_hat_test_rr)

                            B_hat           = LM(Y2_train,X_train @ B_rrr, lam=lam)
    
                            Y2_hat_test_rr   = X_test @ B_rrr @ B_hat
                            
                            subspace_R2[2,r,ises,imf,ikf] = EV(Y2_test,Y2_hat_test_rr)

#%% Show results: 
fig,axes = plt.subplots(1,1,figsize=(4,3))
handles = []
ax = axes
meantoplot = np.nanmean(subspace_R2[0,:,:,:,:],axis=(1,2,3))
errors = np.nanstd(subspace_R2[0,:,:,:,:],axis=(1,2,3))
handles.append(shaded_error(range(nranks),meantoplot,errors,ax=ax,color='g'))

meantoplot = np.nanmean(subspace_R2[1,:,:,:,:],axis=(1,2,3))
errors = np.nanstd(subspace_R2[1,:,:,:,:],axis=(1,2,3))
handles.append(shaded_error(range(nranks),meantoplot,errors,ax=ax,color='r'))

meantoplot = np.nanmean(subspace_R2[2,:,:,:,:],axis=(1,2,3))
errors = np.nanstd(subspace_R2[1,:,:,:,:],axis=(1,2,3))
handles.append(shaded_error(range(nranks),meantoplot,errors,ax=ax,color='b'))

ax.legend(handles,('Source to Target (Same neurons)','Source to Cross','Source to Target (Diff neurons)'),
          frameon=False,fontsize=9)
ax.set_xticks(range(0,nranks+1,10))
ax.set_xlabel('Rank')
ax.set_ylabel('CV R2')
sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'RRR_cvR2_CrossVsTarget_%dsessions' % nSessions,formats=['png'])



#%% Cross area subspace predictions: 
arealabels      = np.array(['V1unl', 'PMunl', 'ALunl','RSPunl'])
# arealabels      = np.array(['V1unl', 'PMunl', 'ALunl'])
narealabels     = len(arealabels)

popsize         = 50
maxnoiselevel   = 20
nmodelfits      = 2
kfold           = 5
lam             = 0
nranks          = 10

same_pop_R2, cross_pop_R2     = cross_area_subspace_wrapper(sessions,arealabels,popsize,maxnoiselevel,nmodelfits,kfold,lam=0,nranks=nranks)

# print('Fraction of NaN elements in cross_pop_R2: %.2f' % (np.isnan(cross_pop_R2).sum() / cross_pop_R2.size))

# print('Number of cells in each area for all sessions:')
# for ises, ses in enumerate(sessions):
#     for arealabel in arealabels:
#         print('Session %d: %d in %s' % (ises+1,
#                                         np.sum(ses.celldata['arealabel']==arealabel),
#                                         arealabel))

#%% Performance on trained population always higher than on different neurons
rank        = 5

fig,axes    = plt.subplots(1,2,figsize=(6,3),sharey=False,sharex=False)

ax          = axes[0]
datatoplot  = np.nanmean(same_pop_R2,axis=(2,3,4,5,6))
meantoplot  = np.nanmean(datatoplot,axis=(1)) #one line for every session
errors      = np.nanstd(datatoplot,axis=(1)) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errors,ax=ax,color='g'))

datatoplot  = np.nanmean(cross_pop_R2,axis=(2,3,4,5,6))
meantoplot  = np.nanmean(datatoplot,axis=1) #one line for every session
errors      = np.nanstd(datatoplot,axis=1) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errors,ax=ax,color='r'))

samedata    = np.nanmean(same_pop_R2,axis=(5,6)).reshape(nranks,-1)
crossdata   = np.nanmean(cross_pop_R2,axis=(5,6)).reshape(nranks,-1)

for r in range(nranks):
    x = samedata[r,:]
    y = crossdata[r,:]
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])
    # print('Paired t-test: p=%.3f' % (p))
    if p<0.05:
        ax.text(r,ax.get_ylim()[1]-0.01,'*',fontsize=13,ha='center',va='top') #,color='r'xtoplot,ytoplot,pos=[0.8,0.1])
# ax.axvline(rank,c='k',linestyle='--')
ax.legend(handles,('Same neurons','Diff neurons'),
          frameon=False,fontsize=9)
ax.set_xticks(range(0,nranks+1,5))
ax.set_xlabel('Rank')
ax.set_ylabel('CV R2')

ax = axes[1]
ytoplot     = np.nanmean(cross_pop_R2[rank],axis=(3,4)).flatten() #one point for every session
xtoplot     = np.nanmean(same_pop_R2[rank],axis=(3,4)).flatten()
ax.scatter(xtoplot,ytoplot,c='k',s=7,alpha=0.4)
ax.set_xlim([0,0.35])
ax.set_ylim([0,0.35])
add_paired_ttest_results(ax,xtoplot,ytoplot,pos=[0.8,0.1])
ax.plot([0,1],[0,1],c='k',linestyle='--')
ax.text(0.3,0.9,'rank = %d' % rank,fontsize=9,ha='right',va='top',transform=ax.transAxes)
ax.set_ylabel('Same neurons')
ax.set_xlabel('Different neurons')
ax.set_xticks([0,0.1,0.2,0.3])
ax.set_yticks([0,0.1,0.2,0.3])
sns.despine(top=True,right=True,offset=3)
fig.tight_layout()
# my_savefig(fig,savedir,'RRR_cvR2_DiffSamePopulations_AreaAverage_%dsessions' % (nSessions))



#%% Performance on trained population always higher than on different neurons from the same area
# rank        = 10

# fig,ax = plt.subplots(1,1,figsize=(3,3),sharey=True,sharex=True)
# ytoplot     = np.nanmean(subspace_R2[:,:,:,0,:,:,:],axis=(4,5)).flatten()
# xtoplot     = np.nanmean(subspace_R2[:,:,:,1,:,:,:],axis=(4,5)).flatten() #one point for every session
# # ytoplot     = np.nanmean(subspace_R2[:,:,:,0,:,:,:],axis=(3,4,5)).flatten()
# # xtoplot     = np.nanmean(subspace_R2[:,:,:,1,:,:,:],axis=(3,4,5)).flatten() #one point for every session
# ax.scatter(xtoplot,ytoplot,c='k',s=7,alpha=0.4)
# ax.set_xlim([0,0.35])
# ax.set_ylim([0,0.35])
# add_paired_ttest_results(ax,xtoplot,ytoplot,pos=[0.8,0.1])
# # ax.set_xlim([0,np.nanmax(np.concatenate((xtoplot,ytoplot)))])
# # ax.set_ylim([0,np.nanmax(np.concatenate((xtoplot,ytoplot)))])
# ax.plot([0,1],[0,1],c='k',linestyle='--')
# # ax.set_title()
# ax.set_ylabel('Same neurons')
# ax.set_xlabel('Different neurons')
# ax.set_xticks([0,0.1,0.2,0.3])
# ax.set_yticks([0,0.1,0.2,0.3])
# sns.despine(top=True,right=True,offset=3)
# fig.tight_layout()
# my_savefig(fig,savedir,'RRR_cvR2_DiffSamePopulations_AreaAverage_%dsessions' % (nSessions))

#%% Performance on cross population prediction versus on neurons from the same area:

clrs_areapairs = get_clr_area_labeled(arealabels)
rank = 5

# Extract the diagonal entries along the 4th and 5th dimensions
sameareadata = np.nanmean(cross_pop_R2,axis=(5,6))
sameareadata = np.einsum('...ii->...i', sameareadata)

# Extract the off diagonal entries along the 4th and 5th dimensions
diffareadata = np.nanmean(cross_pop_R2,axis=(5,6))
mask = np.ones((4, 4), dtype=bool)
np.fill_diagonal(mask, False)
diffareadata = np.einsum('...ij,ij->...ij', diffareadata, mask)

rank        = 5

fig,axes    = plt.subplots(1,1,figsize=(3,3),sharey=False,sharex=False)
ax          = axes
handles = []
datatoplot  = np.nanmean(sameareadata,axis=(2,3))
meantoplot  = np.nanmean(datatoplot,axis=(1)) #one line for every session
errors      = np.nanstd(datatoplot,axis=(1)) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errors,ax=ax,color='g'))

datatoplot  = np.nanmean(diffareadata,axis=(2,3,4))
meantoplot  = np.nanmean(datatoplot,axis=1) #one line for every session
errors      = np.nanstd(datatoplot,axis=1) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errors,ax=ax,color='r'))

samedata    = sameareadata.reshape(nranks,-1)
crossdata   = diffareadata.reshape(nranks,-1)

for r in range(nranks):
    x = samedata[r,:]
    y = crossdata[r,:]
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    t,p = ttest_ind(x, y)
    # t,p = ttest_ind(x[~nas], y[~nas])
    # t,p = ttest_rel(x[~nas], y[~nas])
    # print('Paired t-test: p=%.3f' % (p))
    if p<0.05:
        ax.text(r,ax.get_ylim()[1]-0.01,'*',fontsize=13,ha='center',va='top') #,color='r'xtoplot,ytoplot,pos=[0.8,0.1])
# ax.axvline(rank,c='k',linestyle='--')
ax.legend(handles,('Same area','Diff area\n(both diff pop)'),
          frameon=False,fontsize=9,loc='lower right')
ax.set_xticks(range(0,nranks+1,5))
ax.set_xlabel('Rank')
ax.set_ylabel('CV R2')

sns.despine(top=True,right=True,offset=3)
fig.tight_layout()

# #%% Performance on cross population prediction versus on neurons from the same area:
# # If above the diagonal then predicting neurons from the same area from specific subspace is better
# clrs_areapairs = get_clr_area_labeled(arealabels)
# rank = 5

# fig,axes = plt.subplots(narealabels,narealabels,figsize=(6,6),sharey=True,sharex=True)
# for iarea in range(narealabels):
#     for jarea in range(narealabels):
#         ax          = axes[iarea,jarea]
#         # karea       = 
#         # ytoplot     = np.nanmean(subspace_R2[iarea,jarea,jarea,1,:,:,:],axis=(1,2)) #one point for every session
#         # xtoplot     = np.nanmean(subspace_R2[iarea,jarea,karea,1,:,:,:],axis=(1,2)) #one point for every session
#         # ax.scatter(xtoplot,ytoplot,c='r',s=10)
#         # if jarea==karea and iarea!=jarea:
#             # ax.scatter(xtoplot,ytoplot,c='b',s=10)
#         # kareastoplot = [k for k in range(narealabels) if k not in (iarea,jarea)]
#         kareastoplot = [k for k in range(narealabels)]
#         for karea in kareastoplot:

#             # xtoplot     = np.nanmean(cross_pop_R2[rank,:,iarea,jarea,iarea,:,:],axis=(1,2)) #one point for every session
#             xtoplot     = np.nanmean(cross_pop_R2[rank,:,iarea,jarea,jarea,:,:],axis=(1,2)) #one point for every session
#             ytoplot     = np.nanmean(cross_pop_R2[rank,:,iarea,jarea,karea,:,:],axis=(1,2)) #one point for every session
#             # xtoplot     = np.nanmean(subspace_R2[iarea,jarea,jarea,1,:,:,:],axis=(1,2)) #one point for every session
#             # ytoplot     = np.nanmean(subspace_R2[iarea,jarea,karea,1,:,:,:],axis=(1,2)) #one point for every session
#             ax.scatter(xtoplot,ytoplot,c=clrs_areapairs[karea],s=10)

#         if iarea==0 and jarea==narealabels-1: 
#             h = ax.legend(arealabels[kareastoplot],loc='lower right',frameon=False,fontsize=6)
#             for t,hnd in zip(h.get_texts(),h.legendHandles):
#                 t.set_color(hnd.get_facecolor())
#                 hnd.set_alpha(0)  # Hide the line
#         # add_paired_ttest_results(ax,xtoplot,ytoplot,pos=[0.8,0.1])

#         ax.set_xlim([0,0.4])
#         ax.set_ylim([0,0.4])
#         ax.plot([0,1],[0,1],c='k',linestyle='--')
#         ax.set_title('%s->%s' % (arealabels[iarea],arealabels[jarea]),fontsize=9)
#         # ax.set_ylabel('Same neurons')
#         # ax.set_xlabel('Different neurons')
# # ax.set_xlim([0,np.nanmax(np.nanmean(subspace_R2,axis=(5,6)))*1.1])
# # ax.set_ylim([0,np.nanmax(np.nanmean(subspace_R2,axis=(5,6)))*1.1])
# sns.despine(top=True,right=True,offset=3)
# fig.tight_layout()
# # my_savefig(fig,savedir,'R2_cross_vs_train_diffpops_%dsessions.png' % (nSessions))


#%% Cross area subspace prediction with labeled cells:
arealabels      = np.array(['V1unl', 'V1lab','PMunl', 'PMlab', 'ALunl', 'RSPunl'])
narealabels     = len(arealabels)

popsize         = 20
maxnoiselevel   = 20
nmodelfits      = 100
kfold           = 5
lam             = 0
nranks          = 19

same_pop_R2, cross_pop_R2     = cross_area_subspace_wrapper(sessions,arealabels,popsize,maxnoiselevel,nmodelfits,kfold,lam=0,nranks=nranks,filter_nearby=True)

#%% 
print('Number of cells in each area for all sessions:')
for ises, ses in enumerate(sessions):
    for arealabel in arealabels:
        idx_N           = np.where(np.all((sessions[ises].celldata['arealabel']==arealabel,
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,filter_nearlabeled(sessions[ises],radius=50)	
                                        ),axis=0))[0]
        print('Session %d: %d in %s' % (ises+1,
                                        len(idx_N),
                                        arealabel))
print('Fraction of NaN elements in cross_pop_R2 due to insufficent population size combination: %.2f' % (np.isnan(cross_pop_R2).sum() / cross_pop_R2.size))


#%% 
# TODO: get optimal rank from modelfits and kfolds and then use this average rank for the plots and statistics
#%% 
cross_pop_optimrank = np.full((nSessions,narealabels,narealabels,narealabels),np.nan)	
cross_pop_R2_optimrank = np.full((nSessions,narealabels,narealabels,narealabels),np.nan)

for ises in range(nSessions):
    for iarea in range(narealabels):
        for jarea in range(narealabels):
            for karea in range(narealabels):
                if np.all(np.isnan(cross_pop_R2[:,ises,iarea,jarea,karea,:,:])): continue
                cross_pop_R2_optimrank[ises,iarea,jarea,karea], \
                cross_pop_optimrank[ises,iarea,jarea,karea] = rank_from_R2(
                    cross_pop_R2[:,ises,iarea,jarea,karea,:,:].reshape([nranks,nmodelfits*kfold]),
                    nranks,
                    nmodelfits*kfold)

print('Optimal rank: mean %.2f, std %.2f' % (np.nanmean(cross_pop_optimrank),np.nanstd(cross_pop_optimrank)))

#%% Compare feedforward and feedback V1 and PM labeled hypotheses:

combpairstoplot = np.empty((2,2),dtype=object)

combpairstoplot[0,0] = [['V1unl','PMunl','PMunl'],
              ['V1unl','PMunl','PMlab'],
              ['V1unl','PMlab','PMlab'],
              ['V1unl','PMunl','PMunl']]

combpairstoplot[0,1] = [['V1lab','PMunl','PMunl'],
              ['V1lab','PMlab','PMlab'],
              ['V1lab','PMunl','PMlab'],
              ['V1lab','PMlab','PMunl']]

combpairstoplot[1,0] = [['PMunl','V1unl','V1lab'],
              ['PMunl','V1lab','V1unl'],
              ['PMunl','V1unl','V1lab'],
              ['PMunl','V1lab','V1unl']
              ]

combpairstoplot[1,1] = [
                ['PMlab','V1unl','V1unl'],
              ['PMlab','V1lab','V1lab'],
              ['PMlab','V1unl','V1lab'],
              ['PMlab','V1lab','V1unl']
              ]

clrs = sns.color_palette('tab10',4)

fig,axes = plt.subplots(2,2,figsize=(7,6),sharex=True,sharey=True)
for i in range(2):
    for j in range(2): 
        ax = axes[i,j]
        handles = []
        combpairs = combpairstoplot[i,j]
        for ipair in range(len(combpairs)):
            iarea = arealabels.tolist().index(combpairs[ipair][0])
            jarea = arealabels.tolist().index(combpairs[ipair][1])
            karea = arealabels.tolist().index(combpairs[ipair][2])
            datatoplot  = np.nanmean(cross_pop_R2[:,:,iarea,jarea,karea,:,:],axis=(2,3))
            meantoplot  = np.nanmean(datatoplot,axis=1) #one line for every session
            errortoplot = np.nanstd(datatoplot,axis=1) / np.sqrt(nSessions)#one line for every session
            handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color=clrs[ipair]))
        if i==0:
            ax.set_title('Feedforward')
        elif i==1:
            ax.set_title('Feedback')
        ax.legend(handles,combpairs,loc='lower right',frameon=False,fontsize=7)
        ax.set_xlim([0,nranks])
        ax.set_xlabel('Rank')
        ax.set_ylabel('R2')
sns.despine(top=True,right=True,offset=3)
fig.tight_layout()






#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
areacombs  = [['V1unl','PMunl','ALunl'],
              ['V1unl','ALunl','PMunl'],
              ['PMunl','V1unl','ALunl'],
              ['PMunl','ALunl','V1unl'],
              ['ALunl','V1unl','PMunl'],
              ['ALunl','PMunl','V1unl']]



nareacombs     = len(areacombs)

Nsub                = 50
kfold               = 5
nmodelfits          = 10
lam                 = 0

rank                = 5
nstims              = 16

R2_cv               = np.full((nareacombs,2,2,nstims,nSessions,nmodelfits,kfold),np.nan)

kf                  = KFold(n_splits=kfold,shuffle=True,random_state=None)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model'):    # iterate over sessions
    # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
    for istim,stim in enumerate([0]): # loop over orientations 
        # idx_T               = ses.trialdata['stimCond']==stim
        idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
        for icomb, (areax,areay,areaz) in enumerate(areacombs):

            idx_areax           = np.where(np.all((ses.celldata['roi_name']==areax,
                                    ses.celldata['noise_level']<20),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['roi_name']==areay,
                                    ses.celldata['noise_level']<20),axis=0))[0]
            idx_areaz           = np.where(np.all((ses.celldata['roi_name']==areaz,
                                    ses.celldata['noise_level']<20),axis=0))[0]
            
            if len(idx_areax)>=Nsub and len(idx_areay)>=Nsub and len(idx_areaz)>=Nsub:
                for irbh,regress_out_behavior in enumerate([False,True]):
                    respmat                 = ses.respmat[:,idx_T].T
                    
                    if regress_out_behavior:
                        X       = np.stack((sessions[ises].respmat_videome[idx_T],
                        sessions[ises].respmat_runspeed[idx_T],
                        sessions[ises].respmat_pupilarea[idx_T],
                        sessions[ises].respmat_pupilx[idx_T],
                        sessions[ises].respmat_pupily[idx_T]),axis=1)
                        X       = np.column_stack((X,sessions[ises].respmat_videopc[:,idx_T].T))
                        X       = zscore(X,axis=0,nan_policy='omit')

                        si      = SimpleImputer()
                        X       = si.fit_transform(X)

                        respmat,_  = regress_out_behavior_modulation(ses,X,respmat,rank=10,lam=0,perCond=False)

                    for imf in range(nmodelfits):
                        idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
                        idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)
                        idx_areaz_sub       = np.random.choice(idx_areaz,Nsub,replace=False)
                    
                        X                   = respmat[:,idx_areax_sub]
                        Y                   = respmat[:,idx_areay_sub]
                        Z                   = respmat[:,idx_areaz_sub]
                        
                        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
                        Y                   = zscore(Y,axis=0)
                        Z                   = zscore(Z,axis=0)

                        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                            X_train, X_test = X[idx_train], X[idx_test]
                            Y_train, Y_test = Y[idx_train], Y[idx_test]
                            Z_train, Z_test = Z[idx_train], Z[idx_test]

                            B_hat_train         = LM(Y_train,X_train, lam=lam)

                            Y_hat_train         = X_train @ B_hat_train

                            U, s, V = svds(Y_hat_train,k=rank,which='LM')
                            # S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                            # decomposing and low rank approximation of A
                            # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)

                            # S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                            # for r in range(nranks):
                            B_rrr               = B_hat_train @ V[:rank,:].T @ V[:rank,:] #project beta coeff into low rank subspace
                                # Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace
                                # R2_cv_folds[r,ikf] = EV(Y_test,Y_hat_rr_test)

                            Y_hat_test_rr = X_test @ B_rrr

                            # R2_cv[iapl,r,iori,ises,i,ikf] = EV(Y_test,Y_hat_test_rr)
                            R2_cv[icomb,0,irbh,istim,ises,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                            B_hat         = LM(Z_train,X_train @ B_rrr, lam=lam)

                            Z_hat_test_rr = X_test @ B_rrr @ B_hat
                            
                            R2_cv[icomb,1,irbh,istim,ises,imf,ikf] = EV(Z_test,Z_hat_test_rr)

#%% Show the data: R2
# tempdata = np.nanmean(R2_cv,axis=(2,4,5)) #if cross-validated: average across orientations, model samples and kfolds
tempdata = np.nanmean(R2_cv,axis=(2,3,5,6)) #if cross-validated: average across orientations, model samples and kfolds
tempdata = np.nanmean(R2_cv,axis=(3,5,6)) #if cross-validated: average across orientations, model samples and kfolds
fig, axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
clrs_combs = sns.color_palette('colorblind',len(areacombs))

ax = axes[0]

handles = []
for icomb, (areax,areay,areaz) in enumerate(areacombs):
    # ax = axes[np.array(iapl>3,dtype=int)]
    ax.scatter(tempdata[icomb,0,0,:],tempdata[icomb,1,0,:],s=15,color=clrs_combs[icomb],alpha=1)
    # ax.scatter(tempdata[icomb,0],tempdata[icomb,1,0,:],s=15,color=clrs_combs[icomb],alpha=1)
handles = [plt.Line2D([0], [0], marker='o', color='w', label=areacombs[icomb],
                         markerfacecolor=clrs_combs[icomb], markersize=5) for icomb in range(len(areacombs))]
ax.legend(handles=handles, loc='upper left',frameon=False,fontsize=6)
ax.plot([0,0.2],[0,0.2],linestyle='--',color='k',alpha=0.5)
ax.set_ylabel('R2 (XsubY->Z)')
ax.set_xlabel('R2 (X->Y)')

ax = axes[1]
for icomb, (areax,areay,areaz) in enumerate(areacombs):
    # ax.scatter(tempdata[icomb,0,1,:],tempdata[icomb,1,1,:],s=15,color=clrs_combs[icomb],alpha=1)
    ax.scatter(tempdata[icomb,0,1,:],tempdata[icomb,1,1,:],s=15,color=clrs_combs[icomb],alpha=1)

# ax.set_xlim([0,0.2])
# ax.set_ylim([0,0.2])
ax.set_xlabel('R2 (X->Y)')
ax.set_xticks([0,0.1,0.2])
ax.set_yticks([0,0.1,0.2])
ax.plot([0,0.2],[0,0.2],linestyle='--',color='k',alpha=0.5)
sns.despine(top=True,right=True,offset=3)

my_savefig(fig,savedir,'RRR_cvR2_V1PMAL_Cross_RegressBehav_%dsessions.png' % (nSessions))
# plt.savefig(os.path.join(savedir,'RRR_cvR2_V1PMAL_Cross_RegressBehav_%dsessions.png' % (nSessions)),
#                         bbox_inches='tight')







#%% 

# OLD CCA IMPLEMENTATION:




#%% V1-PM-AL cross subspaces
areas = ['V1','PM','AL','RSP']
nareas = len(areas)


#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,filter_areas=areas)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)



#%% How are the CCA dimensions overlapping between V1-PM and V1-AL?

oris                = np.sort(sessions[ises].trialdata['Orientation'].unique())
nOris               = len(oris)

nSessions           = len(sessions)

areax = 'V1'
areay = 'PM'
areaz = 'AL'

Nsub        = 250   #how many neurons to subsample from each area
prePCA      = 25    #perform dim reduc before fitting CCA, otherwise overfitting

ncomponents = 10

model_CCA_XY = CCA(n_components = ncomponents,scale = False, max_iter = 1000)
model_CCA_XZ = CCA(n_components = ncomponents,scale = False, max_iter = 1000)

proj_corr      = np.full((ncomponents,nOris,nSessions),np.nan)
weight_corr    = np.full((ncomponents,nOris,nSessions),np.nan)


for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA dim1'):    # iterate over sessions
    # get signal correlations:
    [N,K]           = np.shape(ses.respmat) #get dimensions of response matrix

    for iori,ori in enumerate(oris): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori
        # for ix,areax in enumerate(areas):
            # for iy,areay in enumerate(areas):

        idx_areax           = np.where(ses.celldata['roi_name']==areax)[0]
        idx_areay           = np.where(ses.celldata['roi_name']==areay)[0]
        idx_areaz           = np.where(ses.celldata['roi_name']==areaz)[0]

        if len(idx_areax)>Nsub and len(idx_areay)>Nsub and len(idx_areay)>Nsub:
            idx_areax_sub       = np.random.choice(idx_areax,np.min((len(idx_areax),Nsub)),replace=False)
            idx_areay_sub       = np.random.choice(idx_areay,np.min((len(idx_areay),Nsub)),replace=False)
            idx_areaz_sub       = np.random.choice(idx_areaz,np.min((len(idx_areaz),Nsub)),replace=False)

            X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
            Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
            Z                   = sessions[ises].respmat[np.ix_(idx_areaz_sub,idx_T)].T
            
            X                   = zscore(X,axis=0)  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0)
            Z                   = zscore(Z,axis=0)

            if prePCA and Nsub>prePCA:
                prepca      = PCA(n_components=prePCA)
                X           = prepca.fit_transform(X)
                Y           = prepca.fit_transform(Y)
                Z           = prepca.fit_transform(Z)
                
            # Compute and store canonical correlations for the first pair
            XY_c, YX_c = model_CCA_XY.fit_transform(X,Y)

            XZ_c, ZX_c = model_CCA_XZ.fit_transform(X,Z)

            # corr_CC1[ix,iy,iori,0,ises] = np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]
            # [corr_CC1_vars[ix,iy,iori,0,0,ises],_] = CCA_subsample_1dim(X.T,Y.T,resamples=5,kFold=5,prePCA=None)
            for i in range(ncomponents):
                proj_corr[i,iori,ises]      = np.corrcoef(XY_c[:,i],XZ_c[:,i])[0,1]
                weight_corr[i,iori,ises]    = np.corrcoef(model_CCA_XY.x_weights_[i,:],model_CCA_XZ.x_weights_[i,:])[0,1]
                

#%% Make the figure where correlation between projections between V1-PM and V1-AL subspace are shown

fig,ax = plt.subplots(1,1,figsize=(3,3))

# plt.plot(np.nanmean(proj_corr,axis=(1,2)),c='b')
shaded_error(np.arange(ncomponents)+1,np.nanmean(proj_corr,axis=(1,2)),np.nanstd(proj_corr,axis=(1,2))/np.sqrt(nSessions),color='b')
ax.set_xlabel('Dimension')
ax.set_ylabel('Correlation')
ax.set_ylim([-.1,1])
ax.axhline(y=0,color='k',linestyle='--')
ax.set_xlim([1,ncomponents+1])
ax.set_xticks(np.arange(ncomponents)+1)
sns.despine(top=True,right=True,offset=3)
fig.tight_layout()
my_savefig(fig,savedir,'CCA_V1PMAL_ProjCorr_%dsessions' % nSessions)


#%% 



