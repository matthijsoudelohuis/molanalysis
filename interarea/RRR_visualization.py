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
from utils.regress_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\')

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

#%%
ises = 1
ses = sessions[ises]

#%% Test wrapper function:
nN                  = 70
nM                  = 50
nranks              = 25
nmodelfits          = 1 #number of times new neurons are resampled 
kfold               = 5
lam                 = 0

kfold               = 5
nranks              = 50

R2_cv               = np.full((nranks,kfold),np.nan)
kf                  = KFold(n_splits=kfold,shuffle=True)

# idx_T               = ses.trialdata['Orientation']==0
idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
idx_T               = ses.trialdata['Orientation']<=120
# idx_T              = np.random.choice(range(len(idx_T)),1000,replace=False)
idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                        ses.celldata['noise_level']<20),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                        ses.celldata['noise_level']<20),axis=0))[0]

idx_areax_sub       = np.random.choice(idx_areax,nN,replace=False)
idx_areay_sub       = np.random.choice(idx_areay,nM,replace=False)

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

        R2_cv[r,ikf] = EV(Y_test,Y_hat_rr_test)

# repmean,rank = rank_from_R2(R2_cv.reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)

# R2_cv[ises],optim_rank[ises]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Make figure
minmax = 0.75
fig,axes = plt.subplots(2,4,figsize=(12,6))

ax = axes[0,0]
ax.imshow(X_train,cmap='bwr',aspect=0.1,vmin=-minmax,vmax=minmax)
ax.set_title('Source Activity (Train)')
ax.set_xticks([0,nN],np.array([1,nN]))
ax.set_ylabel('Trials')
ax.set_xlabel('Source Neurons (N)')

ax = axes[0,1]
ax.imshow(B_hat_train,cmap='bwr',vmin=-0.3,vmax=0.3)
ax.set_title('Weights')
ax.set_yticks([0,nN],np.array([1,nN]))
ax.set_xticks([0,nM],np.array([1,nM]))
ax.set_ylabel('Source Neurons (N)')
ax.set_xlabel('Target Neurons (M)')

ax = axes[0,2]
ax.imshow(Y_hat_train,cmap='bwr',aspect=0.1,vmin=-minmax,vmax=minmax)
ax.set_xticks([0,nM],np.array([1,nM]))
ax.set_title('Predicted Target Activity (Train)')
ax.set_ylabel('Trials')
ax.set_xlabel('Target Neurons (M)')

ax = axes[0,3]
ax.imshow(Y,cmap='bwr',aspect=0.1,vmin=-minmax,vmax=minmax)
ax.set_xticks([0,nM],np.array([1,nM]))
ax.set_title('Target Activity (Train)')
ax.set_ylabel('Trials')
ax.set_xlabel('Target Neurons (M)')

ax = axes[1,0]
ax.imshow(X_test,cmap='bwr',aspect=0.08*kfold,vmin=-minmax,vmax=minmax)
ax.set_title('Source Activity (Test)')
ax.set_xticks([0,nN],np.array([1,nN]))
ax.set_ylabel('Trials')
ax.set_xlabel('Source Neurons (N)')

r = 5
B_lr  = B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace
Y_hat_rr_test       = X_test @ B_lr #project test data onto low rank subspace

ax = axes[1,1]
ax.imshow(B_lr,cmap='bwr',vmin=-0.3,vmax=0.3)
ax.set_title('Low Rank Weights')
ax.set_yticks([0,nN],np.array([1,nN]))
ax.set_xticks([0,nM],np.array([1,nM]))
ax.set_ylabel('Source Neurons (N)')
ax.set_xlabel('Target Neurons (M)')

ax = axes[1,2]
ax.imshow(Y_hat_rr_test,cmap='bwr',aspect=0.08*kfold,vmin=-minmax,vmax=minmax)
ax.set_xticks([0,nM],np.array([1,nM]))
ax.set_title('RRR predicted target activity (Test)')
ax.set_ylabel('Trials')
ax.set_xlabel('Target Neurons (M)')

ax = axes[1,3]
ax.imshow(Y_test,cmap='bwr',aspect=0.08*kfold,vmin=-minmax,vmax=minmax)
ax.set_xticks([0,nM],np.array([1,nM]))
ax.set_title('Target Activity (Test)')
ax.set_ylabel('Trials')
ax.set_xlabel('Target Neurons (M)')

plt.tight_layout()
my_savefig(fig,savedir,'RRR_visualization_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])


#%% Make figure
minmax = 0.75
nranks = 15
fig,axes = plt.subplots(1,nranks,figsize=(nranks*1.5,2))

for r in range(nranks):
    ax = axes[r]

    B_lr  = B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace
    Y_hat_rr_test       = X_test @ B_lr #project test data onto low rank subspace

    ax.imshow(B_lr,cmap='bwr',vmin=-0.3,vmax=0.3)
    ax.set_title('Rank %d' % r)
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_ylabel('Source Neurons (N)')
    # ax.set_xlabel('Target Neurons (M)')
plt.tight_layout()
my_savefig(fig,savedir,'RRR_rankweights_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])

#%%
nranks      = 10
EVdata      = np.full((nSessions,nranks),np.nan)
rankdata    = np.full((nSessions,nranks),np.nan)
for ises,ses in enumerate(sessions):
    for rank in range(nranks):
        Y,Y_hat_rr,Y_out,rankdata[ises,rank],EVdata[ises,rank]  = regress_out_behavior_modulation(ses,rank=rank+1,lam=0,perCond=True)

#%% Plot variance regressed out by behavioral modulation
fig,axes = plt.subplots(1,1,figsize=(3,3),sharex=False)
ax = axes
ax.plot(range(nranks+1),np.concatenate(([0],np.nanmean(EVdata,axis=0))))
ax.set_title('Variance regressed out by behavioral modulation')
ax.set_ylabel('Variance Explained')
ax.set_xlabel('Rank')
ax.set_xticks(range(nranks+1))
sns.despine(top=True,right=True,offset=3)
my_savefig(fig,savedir,'BehaviorRegressedOut_V1PM_%dsessions.png' % nSessions,formats=['png'])

#%% Plot the number of dimensions per area pair
def plot_RRR_R2_regressout(R2data,rankdata,arealabelpairs,clrs_arealabelpairs):
    narealabelpairs         = len(arealabelpairs)
    # clrs_arealabelpairs     = get_clr_area_labeled(arealabelpairs)
    fig, axes = plt.subplots(2,2,figsize=(8,6))

    statpairs = [(0,1),(0,2),(0,3),
                (4,5),(4,6),(4,7)]

    # R2data[R2data==0]          = np.nan
    arealabelpairs2     = [al.replace('-','-\n') for al in arealabelpairs]

    for irbh in range(2):
        ax=axes[irbh,0]
        for iapl, arealabelpair in enumerate(arealabelpairs):
            ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3,R2data[iapl,irbh,:],color='k',marker='o',s=10)
            ax.errorbar(iapl+0.5,np.nanmean(R2data[iapl,irbh,:]),np.nanstd(R2data[iapl,irbh,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

        ax.set_ylabel('R2 (cv)')
        ax.set_ylim([0,my_ceil(np.nanmax(R2data[:,irbh,:]),2)])

        if irbh==1:
            ax.set_xlabel('Population pair')
            ax.set_xticks(range(narealabelpairs))
        else:
            ax.set_xticks(range(narealabelpairs),labels=[])

        if irbh==0:
            ax.set_title('Performance at full rank')

        testdata = R2data[:,irbh,:]
        testdata = testdata[:,~np.isnan(testdata).any(axis=0)]

        df = pd.DataFrame({'R2':  testdata.flatten(),
                        'arealabelpair':np.repeat(np.arange(narealabelpairs),np.shape(testdata)[1])})
        
        annotator = Annotator(ax, statpairs, data=df, x="arealabelpair", y='R2', order=np.arange(narealabelpairs))
        annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False)
        # annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False)
        annotator.apply_and_annotate()

        ax=axes[irbh,1]
        for iapl, arealabelpair in enumerate(arealabelpairs):
            ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3,rankdata[iapl,irbh,:],color='k',marker='o',s=10)
            ax.errorbar(iapl+0.5,np.nanmean(rankdata[iapl,irbh,:]),np.nanstd(rankdata[iapl,irbh,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

        ax.set_xticks(range(narealabelpairs))
        if irbh==1:
            ax.set_xlabel('Population pair')
            ax.set_xticks(range(narealabelpairs))
        else:
            ax.set_xticks(range(narealabelpairs),labels=[])
        
        if irbh==1:
            ax.set_ylabel('Number of dimensions')
        ax.set_yticks(np.arange(0,14,2))
        ax.set_ylim([0,my_ceil(np.nanmax(rankdata),0)+1])
        if irbh==0:
            ax.set_title('Dimensionality')

        testdata = rankdata[:,irbh,:]
        testdata = testdata[:,~np.isnan(testdata).any(axis=0)]

        df = pd.DataFrame({'R2':  testdata.flatten(),
                        'arealabelpair':np.repeat(np.arange(narealabelpairs),np.shape(testdata)[1])})

        annotator = Annotator(ax, statpairs, data=df, x="arealabelpair", y='R2', order=np.arange(narealabelpairs))
        annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False)
        # annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False)
        # annotator.apply_and_annotate()

    sns.despine(top=True,right=True,offset=3,trim=True)
    axes[1,0].set_xticklabels(arealabelpairs2,fontsize=7)
    axes[1,1].set_xticklabels(arealabelpairs2,fontsize=7)
    return fig


#%% 

#%% ###################################################




#%% Perform RRR on raw calcium data: 

ises = 1
ses = sessions[ises]




#%% Fit:
nN                  = 200
nM                  = 200
nranks              = 5
lam                 = 0

# idx_T               = ses.trialdata['Orientation']==0
excerpt_length      = 1000
t_start             = np.random.choice(ses.ts_F,1,replace=False)[0]
idx_T               = (ses.ts_F>t_start) & (ses.ts_F<t_start+excerpt_length)
ts                  = ses.ts_F[idx_T]
nTs                 = np.sum(idx_T)

idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                        ses.celldata['noise_level']<10),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                        ses.celldata['noise_level']<10),axis=0))[0]

idx_areax_sub       = np.random.choice(idx_areax,nN,replace=False)
idx_areay_sub       = np.random.choice(idx_areay,nM,replace=False)

X                   = sessions[ises].calciumdata.to_numpy()[np.ix_(idx_T,idx_areax_sub)]
Y                   = sessions[ises].calciumdata.to_numpy()[np.ix_(idx_T,idx_areay_sub)]

X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
Y                   = zscore(Y,axis=0)

Y_hat_rr               = np.full((nTs,nM,nranks),np.nan)  


B_hat         = LM(Y,X, lam=lam)

Y_hat         = X @ B_hat

# decomposing and low rank approximation of A
U, s, V = linalg.svd(Y_hat)
S = linalg.diagsvd(s,U.shape[0],s.shape[0])

for r in range(nranks):
    Y_hat_rr[:,:,r]       = X @ B_hat @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

sourceweights = np.mean(np.abs(B_hat @ V[:r,:].T @ V[:r,:]),axis=1)
targetweights = np.mean(np.abs(B_hat @ V[:r,:].T @ V[:r,:]),axis=0)

r = nranks-1
targetweights = r2_score(Y,Y_hat_rr[:,:,r],multioutput='raw_values') + (r2_score(Y,Y_hat_rr[:,:,r],multioutput='raw_values') - r2_score(Y,Y_hat_rr[:,:,1],multioutput='raw_values'))


#%% Identify a chunk in time where there is a high prediction for high rank across areas:
window_size = 10
window_step = 1
nTs = len(ts)
nWindows = int(np.ceil((nTs-window_size)/window_step))
variance_explained = np.full((nWindows,nranks),np.nan)  

t_starts = np.empty(nWindows)
for iw in range(nWindows):
    t_starts[iw] = ts[0]+iw*window_step
    t_end = t_starts[iw]+window_size
    idx_window = (ts>=t_starts[iw]) & (ts<t_end)
    Y_window = Y[idx_window]
    for r in range(nranks):
        variance_explained[iw,r] = EV(Y_window,Y_hat_rr[idx_window,:,r])

plt.plot(t_starts,variance_explained)

example_tstart = t_starts[np.nanargmax(variance_explained[:,1])]

#%% Make figure
scale = 0.15
lw = 0.75
nshowneurons = 10
nrankstoplot = 5
fig,axes = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
ax = axes[0]

source_example_neurons = np.argsort(-sourceweights)[:nshowneurons]
for i,iN in enumerate(source_example_neurons):
    ax.plot(ts,X[:,iN]*scale+i,color='k',lw=lw)
ax.set_title('Source Activity (V1)')
ax.set_ylabel('Neurons')
ax.axis('off')
ax.add_artist(AnchoredSizeBar(ax.transData, 1,
            "1 Sec", loc=4, frameon=False))

target_example_neurons = np.argsort(-targetweights)[:nshowneurons]
clrs_ranks = sns.color_palette('magma',n_colors=nrankstoplot)
ax = axes[1]
plothandles = []
for r in range(1,nrankstoplot):
    plothandles.append(plt.Line2D([0], [0], color=clrs_ranks[r], lw=lw))
ax.legend(plothandles,['1','2','3','4','5'],title='Rank Prediction',frameon=False,fontsize=8,ncol=2,loc='upper left')	
for i,iN in enumerate(target_example_neurons):
    ax.plot(ts,Y[:,iN]*scale+i,color='k',lw=lw)
ax.set_title('Target Activity (PM)')
ax.set_xlim([example_tstart-10,example_tstart+20])
ax.axis('off')

my_savefig(fig,savedir,'V1PM_LowRank_Excerpt_%s_Rank%d.png' % (ses.sessiondata['session_id'][0],0),formats=['png']) 
for r in range(1,nrankstoplot):
    for i,iN in enumerate(target_example_neurons):
        ax.plot(ts,Y_hat_rr[:,iN,r]*scale+i,color=clrs_ranks[r],lw=lw)
    my_savefig(fig,savedir,'V1PM_LowRank_Excerpt_%s_Rank%d.png' % (ses.sessiondata['session_id'][0],r+1),formats=['png']) 
