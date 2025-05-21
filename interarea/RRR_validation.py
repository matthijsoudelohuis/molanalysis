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
from sklearn.decomposition import PCA
from scipy.stats import zscore,wilcoxon,ttest_rel
from tqdm import tqdm

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\Validation\\')

#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR
# session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                # ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)


#%% 
min_cells = 1
sessions,nSessions   = filter_sessions(protocols = 'GR',min_cells=min_cells,filter_areas=['V1','PM'])
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_cells=min_cells,filter_areas=['V1','PM'])

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%% 
print('Number of cells in V1 and in PM:')
for ises,ses in enumerate(sessions):
    print('Session %d: %d cells in V1, %d cells in PM' % (ises,np.sum(ses.celldata['roi_name']=='V1'),np.sum(ses.celldata['roi_name']=='PM')))

#%%  Load data properly:        
# calciumversion = 'dF'
calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=calciumversion)

#%% get optimal lambda
nsampleneurons  = 500
lambdas         = np.logspace(-6, 5, 10)
# lambdas         = np.array([0,0.01,0.1,1])
nlambdas        = len(lambdas)
kfold           = 5
nranks          = 50

R2_cv           = np.full((nSessions,nlambdas,nranks,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different labmdas'):
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
            kf          = KFold(n_splits=kfold,shuffle=True)
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


#%% plot the results for lam = 0
lambdacolors = sns.color_palette('magma',nlambdas)

fig,ax = plt.subplots(1,1,figsize=(3,3))
for ilam,lam in enumerate([lambdas[0]]):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('Test R2')
ax.set_xticks(range(nranks+5)[::5])
sns.despine(ax=ax,top=True,right=True,trim=True)
# ax.legend(my_ceil(np.log10(lambdas),1),frameon=False,ncol=2,title='lambda (10^x)')

#%% Compute optimal rank:
R2data = np.full((nSessions,nlambdas),np.nan)
rankdata = np.full((nSessions,nlambdas),np.nan)
for ises,ses in enumerate(sessions):
    if np.sum(np.isnan(R2_cv[ises,:,:,:]))>0:
        continue
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
ax.legend(my_ceil(np.log10(lambdas),1),frameon=False,ncol=2,title='lambda (10^x)')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(lambdas,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(lambdas,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(lambdas,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.nanmax(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(lambdas,R2data[ises,:],color='grey',linewidth=1)
ax.plot(lambdas,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(lambdas,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.nanmax(R2data)+.05])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'RRR_perOri_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')
# plt.savefig(os.path.join(savedir,'RRR_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')
my_savefig(fig,savedir,'RRR_Lam_Rank_%dneurons.png' % nsampleneurons,formats=['png'])
# my_savefig(fig,savedir,'RRR_perOri_Lam_Rank_%dneurons.png' % nsampleneurons,formats=['png'])

#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20])
npopsizes           = len(popsizes)
nranks              = 40
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,npopsizes,npopsizes),np.nan)
optim_rank          = np.full((nSessions,npopsizes,npopsizes),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    for ixpop,xpop in enumerate(popsizes):
        for iypop,ypop in enumerate(popsizes):
            if len(idx_areax)>=xpop and len(idx_areay)>=ypop:
                R2_cv[ises,ixpop,iypop],optim_rank[ises,ixpop,iypop] = RRR_wrapper(Y, X, nN=xpop,nM=ypop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot R2 for different number of V1 and PM neurons
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["r","g"], N=256) 
cmap = sns.color_palette('magma', as_cmap=True)
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ax = axes[0]
s = ax.imshow(np.nanmean(R2_cv,axis=0).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(R2_cv,axis=0)),origin='lower')
ax.set_xticks(range(npopsizes),labels=popsizes)
ax.set_yticks(range(npopsizes),labels=popsizes)
fig.colorbar(s, ax=ax,shrink=0.3,orientation='vertical')
ax.set_title('R2')
ax.set_xlabel('# source neurons')
ax.set_ylabel('# target neurons')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
s = ax.imshow(np.nanmean(optim_rank,axis=0).T,cmap=cmap,vmin=0,vmax=20,origin='lower')
ax.set_xticks(range(npopsizes),labels=popsizes)
ax.set_yticks(range(npopsizes),labels=popsizes)
fig.colorbar(s, ax=ax,shrink=0.3,orientation='vertical')
ax.set_xlabel('# source neurons')
# ax.set_ylabel('# target neurons')
ax.set_title('Rank')

sns.despine(top=True,right=True,offset=3)

plt.tight_layout()
my_savefig(fig,savedir,'R2_RRR_Rank_PopSize_Both_V1PM_%dsessions.png' % nSessions,formats=['png'])


#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20,50,100])
npopsizes           = len(popsizes)
nranks              = 50
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,npopsizes),np.nan)
optim_rank          = np.full((nSessions,npopsizes),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    for ipop,pop in enumerate(popsizes):
        if len(idx_areax)>=pop and len(idx_areay)>=pop:
            R2_cv[ises,ipop],optim_rank[ises,ipop]             = RRR_wrapper(Y, X, nN=pop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)


#%% Plot R2 for different number of V1 and PM neurons
clrs_popsizes = sns.color_palette("rocket",len(popsizes))

fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=True)
ax = axes[0]
ax.scatter(popsizes, np.nanmean(R2_cv,axis=0), marker='o', color=clrs_popsizes)
# ax.plot(popsizes, np.nanmean(R2_cv,axis=0),color='k', linewidth=2)
shaded_error(popsizes,R2_cv,center='mean',error='sem',color='k',ax=ax)
ax.set_ylim([0,0.3])
# ax.set_xticks(popsizes)
ax.axhline(y=0,color='k',linestyle='--')

ax.set_xlabel('Population size')
ax.set_ylabel('RRR R2')
ax_nticks(ax,4)
# ax.set_xscale('log')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
ax.scatter(popsizes, np.nanmean(optim_rank,axis=0), marker='o', color=clrs_popsizes)
shaded_error(popsizes,optim_rank,center='mean',error='sem',color='k',ax=ax)

ax.plot(popsizes,popsizes**0.5,color='r',linestyle='--',linewidth=1)
ax.text(100,13,'$n^{1/2}$',color='r',fontsize=12)
ax.plot(popsizes,popsizes**0.3333,color='g',linestyle='--',linewidth=1)
ax.text(100,2,'$n^{1/3}$',color='g',fontsize=12)

ax.set_ylim([0,20])
ax.set_ylabel('Dimensionality')
ax.set_xlabel('Population size')
ax.set_xticks(popsizes[::2])
ax.set_xticks([10,100,200,500])

sns.despine(top=True,right=True,offset=3)

plt.tight_layout()
my_savefig(fig,savedir,'R2_RRR_Rank_PopSize_V1PM_%dsessions.png' % nSessions,formats=['png'])




#%% Show RRR performance as a function of the number of trials:
nsampleneurons  = 500
ntrials         = np.array([10,20,50,100,200,400,600,1000,2000,3200])
# ntrials         = np.array([50,100,200,400,1000])
ntrialsubsets   = len(ntrials)
kfold           = 5
nranks          = 50
lam             = 1
R2_cv           = np.full((nSessions,ntrialsubsets,nranks,kfold),np.nan)

kf              = KFold(n_splits=kfold,shuffle=True)

for ises,ses in enumerate(sessions):
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        for intrials,nt in enumerate(ntrials):
            idx_T           = np.random.choice(ses.respmat.shape[1],nt,replace=False)

            X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
            Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
            
            X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
            Y                   = zscore(Y,axis=0)

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

                    R2_cv[ises,intrials,r,ikf] = EV(Y_test,Y_hat_rr_test)


#%% 
R2data = np.full((nSessions,ntrialsubsets),np.nan)
rankdata = np.full((nSessions,ntrialsubsets),np.nan)

for ises,ses in enumerate(sessions):
    for intrials,nt in enumerate(ntrials):
        R2data[ises,intrials],rankdata[ises,intrials] = rank_from_R2(R2_cv[ises,intrials,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',ntrialsubsets)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for intrials,nt in enumerate(ntrials):
    tempdata = np.nanmean(R2_cv[:,intrials,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[intrials],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')
ax.set_ylim([-0.05,0.5])
ax.legend(ntrials,frameon=False,ncol=2,fontsize=8,title='number of trials')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(ntrials,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(ntrials,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(ntrials,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('number of trials')
# ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.max(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(ntrials,R2data[ises,:],color='grey',linewidth=1)
ax.plot(ntrials,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(ntrials,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('number of trials')
# ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.max(R2data)+.05])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'RRR_nTrials_lam%d_Rank_%dneurons.png' % (lam,nsampleneurons)), format = 'png')


#%%  Within to across area dimensionality comparison: 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross\\')


#%% 
min_cells = 1500
sessions,nSessions   = filter_sessions(protocols = 'GR',min_cells=min_cells,filter_areas=['V1','PM'])

#%% 
print('Number of cells in V1 and in PM:')
for ises,ses in enumerate(sessions):
    print('Session %d: %d cells in V1, %d cells in PM' % (ises,np.sum(ses.celldata['roi_name']=='V1'),np.sum(ses.celldata['roi_name']=='PM')))

#%%  Load data properly:        
calciumversion = 'dF'
# calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=False, load_calciumdata=True,load_videodata=False,
                                calciumversion=calciumversion,keepraw=False)

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1-V1',
                    'PM-PM',
                    'V1-PM',
                    'PM-V1'
                    ]
alp_withinacross = ['Within',
                   'Within',
                   'Across',
                   'Across'] 

clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nsampleneurons      = 250
nranks              = 50
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    if np.sum(ses.celldata['roi_name']=='V1')<(nsampleneurons*2) or np.sum(ses.celldata['roi_name']=='PM')<(nsampleneurons*2):
        continue 
    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<20
                                ),axis=0))[0]
        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<20
                                ),axis=0))[0]
        
        idx_areay           = np.setdiff1d(idx_areay,idx_areax_sub)
        
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)
    
        X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T

        R2_cv[iapl,ises],optim_rank[iapl,ises]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)


#%% Plotting:
clr = clrs_arealabelpairs[0]

fig,axes = plt.subplots(2,3,figsize=(6.5,4.5),sharey='row',sharex='row')

comps = [[0,1],[0,3],[1,2]]
for icomp,comp in enumerate(comps):
    ax = axes[0,icomp]
    ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=100,marker='.',color=clr)
    # ax.set_xlim([0,0.4])
    # ax.set_ylim([0,0.4])
    ax.plot([0,0.4],[0,0.4],':',color='grey',linewidth=1)
    ax_nticks(ax,3)
    if icomp==0:
        ax.set_ylabel('R2\n%s (%s)' % (arealabelpairs[comp[1]],alp_withinacross[comp[1]]))
    else: 
        ax.set_ylabel('%s (%s)' % (arealabelpairs[comp[1]],alp_withinacross[comp[1]]))
    
    _,pval = ttest_rel(R2_cv[comp[0],:],R2_cv[comp[1],:])
    ax.text(0.7,0.05,'p=%.3f' % pval,transform=ax.transAxes,fontsize=10)

    ax = axes[1,icomp]
    ax.scatter(optim_rank[comp[0],:],optim_rank[comp[1],:],s=100,marker='.',color=clr)
    ax.plot([0,30],[0,30],':',color='grey',linewidth=1)
    # ax.set_xlim([0,25])
    # ax.set_ylim([0,25])
    ax_nticks(ax,3)

    ax.set_xlabel('%s (%s)' % (arealabelpairs[comp[0]],alp_withinacross[comp[0]]))
    if icomp==0:
        ax.set_ylabel('Rank\n%s (%s)' % (arealabelpairs[comp[1]],alp_withinacross[comp[1]]))
    else: 
        ax.set_ylabel('%s (%s)' % (arealabelpairs[comp[1]],alp_withinacross[comp[1]]))
    
    _,pval = ttest_rel(optim_rank[comp[0],:],optim_rank[comp[1],:])
    ax.text(0.7,0.05,'p=%.3f' % pval,transform=ax.transAxes,fontsize=10)

    # ax.set_title('Rank=%d' % np.nanmean(optim_rank[comp[0],:]))
sns.despine(offset=3,top=True,right=True)

# Add a title above the first row of subplots
fig.text(0.5, 0.95, 'Variance explained', ha='center', fontsize=14)

# Add a title above the second row of subplots
fig.text(0.5, 0.48, 'Optimal rank', ha='center', fontsize=14)

# Adjust layout to make room for the titles
fig.tight_layout(rect=[0, 0.03, 1, 0.92])

plt.tight_layout()

my_savefig(fig,savedir,'R2Rank_WithinAcross_GR_%dneurons.png' % nsampleneurons,formats=['png'])





#%% Using trial averaged or using timepoint fluctuations:

#%% 
session_list        = np.array([['LPE12223','2024_06_10'], #GR
                                ['LPE10919','2023_11_06']]) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)
sessions,nSessions   = filter_sessions(protocols = 'GR')


#%%  Load data properly:        
calciumversion = 'dF'
# calciumversion = 'deconv'

## Construct tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

for ises in range(nSessions):

    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=True)

    [sessions[ises].tensor,t_axis]     = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='nearby')

#%% 


#%% 
R2_cv_ori           = np.full((nOris,nSessions,nmodelfits),np.nan)
R2_cv_all           = np.full((nSessions,nmodelfits),np.nan)
R2_cv_oriT          = np.full((nOris,nSessions,nmodelfits),np.nan)
R2_cv_allT          = np.full((nSessions,nmodelfits),np.nan)

kfold               = 10
lam                 = 500
nmodelfits          = 5
Nsub                = 50
r                   = 10 #rank for RRR

pca                 = PCA(n_components=25)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model'):    # iterate over sessions
    idx_areax       = np.where(np.all((ses.celldata['roi_name']=='V1',
                                                ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay       = np.where(np.all((ses.celldata['roi_name']=='PM',
                                            ses.celldata['noise_level']<20),axis=0))[0]

    for iori,ori in enumerate(oris): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori

        for imf in range(nmodelfits):

            idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
            idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)

            X                   = ses.respmat[np.ix_(idx_areax_sub,idx_T)].T
            Y                   = ses.respmat[np.ix_(idx_areay_sub,idx_T)].T
            
            X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
            Y                   = zscore(Y,axis=0)

            X                   = pca.fit_transform(X)
            Y                   = pca.fit_transform(Y)

            R2_kfold    = np.zeros((kfold))
            kf          = KFold(n_splits=kfold)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)
                
                B_hat_lr            = RRR(Y_train, X_train, B_hat_train, r, mode='right')

                Y_hat_test_rr       = X_test @ B_hat_lr

                R2_kfold[ikf]       = EV(Y_test,Y_hat_test_rr) * np.sum(pca.explained_variance_ratio_)
            R2_cv_ori[iori,ises,imf] = np.average(R2_kfold)

    for iori,ori in enumerate(oris): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori

        for imf in range(nmodelfits):

            idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
            idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)

            X                   = ses.tensor[np.ix_(idx_areax_sub,idx_T,np.ones(len(t_axis)).astype(bool))]
            Y                   = ses.tensor[np.ix_(idx_areay_sub,idx_T,np.ones(len(t_axis)).astype(bool))]

            X                   = zscore(np.reshape(X,[Nsub,-1]).T,axis=0)  #Z score activity for each neuron across trials/timepoints
            Y                   = zscore(np.reshape(Y,[Nsub,-1]).T,axis=0)  #Z score activity for each neuron across trials/timepoints
            
            # X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
            # Y                   = zscore(Y,axis=0)

            X                   = pca.fit_transform(X)
            Y                   = pca.fit_transform(Y)

            R2_kfold    = np.zeros((kfold))
            kf          = KFold(n_splits=kfold)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)
                
                B_hat_lr            = RRR(Y_train, X_train, B_hat_train, r, mode='right')

                Y_hat_test_rr       = X_test @ B_hat_lr

                R2_kfold[ikf]       = EV(Y_test,Y_hat_test_rr) * np.sum(pca.explained_variance_ratio_)
            R2_cv_oriT[iori,ises,imf] = np.average(R2_kfold)

    idx_T               = np.ones(ses.trialdata['Orientation'].shape[0],dtype=bool)

    for imf in range(nmodelfits):

        idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)

        # X                   = ses.respmat[np.ix_(idx_areax_sub,idx_T)].T
        # Y                   = ses.respmat[np.ix_(idx_areay_sub,idx_T)].T
        
        _,respmat_res       = mean_resp_gr(ses,trialfilter=None)

        X                   = respmat_res[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = respmat_res[np.ix_(idx_areay_sub,idx_T)].T

        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(Y,axis=0)

        X                   = pca.fit_transform(X)
        Y                   = pca.fit_transform(Y)

        R2_kfold    = np.zeros((kfold))
        kf          = KFold(n_splits=kfold)
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
            X_train, X_test     = X[idx_train], X[idx_test]
            Y_train, Y_test     = Y[idx_train], Y[idx_test]

            B_hat_train         = LM(Y_train,X_train, lam=lam)
            
            B_hat_lr            = RRR(Y_train, X_train, B_hat_train, r, mode='right')

            Y_hat_test_rr       = X_test @ B_hat_lr

            R2_kfold[ikf]       = EV(Y_test,Y_hat_test_rr) * np.sum(pca.explained_variance_ratio_)
        R2_cv_all[ises,imf] = np.average(R2_kfold)

    for imf in range(nmodelfits):

        idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)

        X                   = ses.tensor[np.ix_(idx_areax_sub,idx_T,np.ones(len(t_axis)).astype(bool))]
        Y                   = ses.tensor[np.ix_(idx_areay_sub,idx_T,np.ones(len(t_axis)).astype(bool))]

        X                   = zscore(np.reshape(X,[Nsub,-1]).T,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(np.reshape(Y,[Nsub,-1]).T,axis=0)  #Z score activity for each neuron across trials/timepoints
        
        X                   = pca.fit_transform(X)
        Y                   = pca.fit_transform(Y)

        R2_kfold    = np.zeros((kfold))
        kf          = KFold(n_splits=kfold)
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
            X_train, X_test     = X[idx_train], X[idx_test]
            Y_train, Y_test     = Y[idx_train], Y[idx_test]

            B_hat_train         = LM(Y_train,X_train, lam=lam)
            
            B_hat_lr            = RRR(Y_train, X_train, B_hat_train, r, mode='right')

            Y_hat_test_rr       = X_test @ B_hat_lr

            R2_kfold[ikf]       = EV(Y_test,Y_hat_test_rr) * np.sum(pca.explained_variance_ratio_)
        R2_cv_allT[ises,imf] = np.average(R2_kfold)

#%% Plot the results for the different data types:
plotdata = np.vstack((np.nanmean(R2_cv_ori,axis=(0,2)),np.nanmean(R2_cv_oriT,axis=(0,2)),
                      np.nanmean(R2_cv_all,axis=(1)),np.nanmean(R2_cv_allT,axis=(1)))) #
labels = ['time-averaged\n(per ori)','tensor\n(per ori)', 'time-averaged\n(all trials)','tensor\n(all trials)']
fig,axes = plt.subplots(1,1,figsize=(4,3))
ax = axes
ax.plot(plotdata,color='k',linewidth=2,marker='o')
ax.set_xticks(np.arange(len(labels)),labels,fontsize=8)
ax.set_ylabel('R2')
ax.set_ylim([0,0.2])
ax.set_title('Which data to use?')
sns.despine(fig=fig, top=True, right=True,offset=5)
plt.savefig(os.path.join(savedir,'RRR_R2_difftypes.png'), format = 'png', bbox_inches='tight')









# DEPRECATED:


#%% get optimal pre pCA
nsampleneurons  = 500
PCAdims         = np.array([1,2,5,10,20,50,100])
# lambdas         = np.array([0,0.01,0.1,1])
nPCAdims        = len(PCAdims)
kfold           = 5
nranks          = 50
lam             = 5000
R2_cv           = np.full((nSessions,nPCAdims,nranks,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different pre PCA dims'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
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

            kf          = KFold(n_splits=kfold,shuffle=True)

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

