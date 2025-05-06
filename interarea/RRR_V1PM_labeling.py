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
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\Labeling\\')

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
    
#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)









# RRR








#%% 

areas = ['V1','PM','AL','RSP']
nareas = len(areas)


# areas = ['V1','PM']
# nareas = len(areas)



# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,filter_areas=areas)
# sessions,nSessions   = filter_sessions(protocols = 'GN',only_all_areas=areas,filter_areas=areas)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_areas=areas)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

#%% 




#%% Test wrapper function:
nsampleneurons  = 20
nranks          = 25
nmodelfits      = 1 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions),np.nan)
optim_rank      = np.full((nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    
    R2_cv[ises],optim_rank[ises]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))

#%% Are the weights higher for V1lab or PMlab?

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
N                   = len(celldata)
nranks              = 10

weights_source      = np.full((N,nranks),np.nan)
weights_target      = np.full((N,nranks),np.nan)

#%% Fit:
lam                 = 0

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_ses = np.where(celldata['session_id']==ses.sessiondata['session_id'][0])[0]

    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)

    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]

    nN                  = len(idx_areax)
    nM                  = len(idx_areay)

    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T

    X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
    Y                   = zscore(Y,axis=0)

    B_hat         = LM(Y,X, lam=lam)

    U, s, V = svds(B_hat,k=nranks,which='LM')
    U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
    V       = V.T

    weights_source[idx_ses[idx_areax],:] = U
    weights_target[idx_ses[idx_areay],:] = V

    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]

    nN                  = len(idx_areax)
    nM                  = len(idx_areay)

    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T

    X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
    Y                   = zscore(Y,axis=0)

    B_hat         = LM(Y,X, lam=lam)

    U, s, V = svds(B_hat,k=nranks,which='LM')
    U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
    V       = V.T

    weights_source[idx_ses[idx_areax],:] = U
    weights_target[idx_ses[idx_areay],:] = V





#%% Plot:
fig,axes = plt.subplots(2,2,figsize=(6,6),sharex=True,sharey=True)

ax = axes[0,0]
for r in range(nranks):
    idx_N = celldata['arealabel'] == 'V1unl'
    # ax.bar(r-0.5,np.nanmean(weights_source[idx_N,r],0),color='k',width=0.2)
    ax.bar(r-0.1,np.nanmean(np.abs(weights_source[idx_N,r]),0),color='k',width=0.2)

    idx_N = celldata['arealabel'] == 'V1lab'
    ax.bar(r+0.1,np.nanmean(np.abs(weights_source[idx_N,r]),0),color='r',width=0.2)
ax.set_xticks(range(nranks))
ax.set_xlim([-0.5,nranks-0.5])
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('Source Weights V1')

ax = axes[0,1]
for r in range(nranks):
    idx_N = celldata['arealabel'] == 'PMunl'
    # ax.bar(r-0.5,np.nanmean(weights_source[idx_N,r],0),color='k',width=0.2)
    ax.bar(r-0.1,np.nanmean(np.abs(weights_target[idx_N,r]),0),color='k',width=0.2)

    idx_N = celldata['arealabel'] == 'PMlab'
    ax.bar(r+0.1,np.nanmean(np.abs(weights_target[idx_N,r]),0),color='r',width=0.2)
ax.set_xticks(range(nranks))
ax.set_xlim([-0.5,nranks-0.5])
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('Target Weights PM')

ax = axes[1,0]
for r in range(nranks):
    idx_N = celldata['arealabel'] == 'PMunl'
    # ax.bar(r-0.5,np.nanmean(weights_source[idx_N,r],0),color='k',width=0.2)
    ax.bar(r-0.1,np.nanmean(np.abs(weights_source[idx_N,r]),0),color='k',width=0.2)

    idx_N = celldata['arealabel'] == 'PMlab'
    ax.bar(r+0.1,np.nanmean(np.abs(weights_source[idx_N,r]),0),color='r',width=0.2)
ax.set_xticks(range(nranks))
ax.set_xlim([-0.5,nranks-0.5])
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('Source Weights PM')

ax = axes[1,1]
for r in range(nranks):
    idx_N = celldata['arealabel'] == 'V1unl'
    # ax.bar(r-0.5,np.nanmean(weights_source[idx_N,r],0),color='k',width=0.2)
    ax.bar(r-0.1,np.nanmean(np.abs(weights_target[idx_N,r]),0),color='k',width=0.2)

    idx_N = celldata['arealabel'] == 'V1lab'
    ax.bar(r+0.1,np.nanmean(np.abs(weights_target[idx_N,r]),0),color='r',width=0.2)
ax.set_xticks(range(nranks))
ax.set_xlim([-0.5,nranks-0.5])
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('Target Weights V1')

sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'RRR_weights_acrossranks_labeledV1PM_%dsessions.png' % (nSessions),formats=['png'])

#%% Plot:
fig,axes = plt.subplots(2,2,figsize=(6,6),sharex=True,sharey=True)

ax = axes[0,0]
for r in range(nranks):
    idx_N = celldata['arealabel'] == 'V1unl'
    # ax.bar(r-0.5,np.nanmean(weights_source[idx_N,r],0),color='k',width=0.2)
    ax.bar(r-0.1,np.nanmean(np.abs(weights_source[idx_N,r]),0),color='k',width=0.2)

    idx_N = celldata['arealabel'] == 'V1lab'
    ax.bar(r+0.1,np.nanmean(np.abs(weights_source[idx_N,r]),0),color='r',width=0.2)
ax.set_xticks(range(nranks))
ax.set_xlim([-0.5,nranks-0.5])
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('Source Weights V1')

ax = axes[0,1]
for r in range(nranks):
    idx_N = celldata['arealabel'] == 'PMunl'
    # ax.bar(r-0.5,np.nanmean(weights_source[idx_N,r],0),color='k',width=0.2)
    ax.bar(r-0.1,np.nanmean(np.abs(weights_target[idx_N,r]),0),color='k',width=0.2)

    idx_N = celldata['arealabel'] == 'PMlab'
    ax.bar(r+0.1,np.nanmean(np.abs(weights_target[idx_N,r]),0),color='r',width=0.2)
ax.set_xticks(range(nranks))
ax.set_xlim([-0.5,nranks-0.5])
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('Target Weights PM')

ax = axes[1,0]
for r in range(nranks):
    idx_N = celldata['arealabel'] == 'PMunl'
    # ax.bar(r-0.5,np.nanmean(weights_source[idx_N,r],0),color='k',width=0.2)
    ax.bar(r-0.1,np.nanmean(np.abs(weights_source[idx_N,r]),0),color='k',width=0.2)

    idx_N = celldata['arealabel'] == 'PMlab'
    ax.bar(r+0.1,np.nanmean(np.abs(weights_source[idx_N,r]),0),color='r',width=0.2)
ax.set_xticks(range(nranks))
ax.set_xlim([-0.5,nranks-0.5])
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('Source Weights PM')

ax = axes[1,1]
for r in range(nranks):
    idx_N = celldata['arealabel'] == 'V1unl'
    # ax.bar(r-0.5,np.nanmean(weights_source[idx_N,r],0),color='k',width=0.2)
    ax.bar(r-0.1,np.nanmean(np.abs(weights_target[idx_N,r]),0),color='k',width=0.2)

    idx_N = celldata['arealabel'] == 'V1lab'
    ax.bar(r+0.1,np.nanmean(np.abs(weights_target[idx_N,r]),0),color='r',width=0.2)
ax.set_xticks(range(nranks))
ax.set_xlim([-0.5,nranks-0.5])
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('Target Weights V1')

sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'RRR_weights_acrossranks_labeledV1PM_%dsessions.png' % (nSessions),formats=['png'])

#%% 


#%% Validate regressing out behavior: 
ises    = 4
ses     = sessions[ises]
idx_T   = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
# idx_T   = ses.trialdata['stimCond']==0

X       = np.stack((ses.respmat_videome[idx_T],
                ses.respmat_runspeed[idx_T],
                ses.respmat_pupilarea[idx_T],
                ses.respmat_pupilx[idx_T],
                ses.respmat_pupily[idx_T]),axis=1)
X       = np.column_stack((X,ses.respmat_videopc[:,idx_T].T))
X       = zscore(X,axis=0,nan_policy='omit')

si      = SimpleImputer()
X       = si.fit_transform(X)

Y               = ses.respmat[:,idx_T].T
Y               = zscore(Y,axis=0,nan_policy='omit')

Y_orig,Y_hat_rr,Y_out,rank,EVdata      = regress_out_behavior_modulation(sessions[ises],rank=5,lam=0,perCond=True)
print("Variance explained by behavioral modulation: %1.4f" % EVdata)

#%% Make figure
minmax = 0.75
fig,axes = plt.subplots(1,3,figsize=(6,3),sharex=False)
ax = axes[0]
ax.imshow(X,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('Behavior')
ax.set_ylabel('Trials')
ax.set_xlabel('Behavioral features')

ax = axes[1]
ax.imshow(Y_orig,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('Original')
ax.set_yticklabels('')
ax.set_xlabel('Neurons')

ax = axes[2]
ax.imshow(Y_out,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('Orig-Behavior (RRR)')
ax.set_yticklabels('')
ax.set_xlabel('Neurons')
plt.tight_layout()
my_savefig(fig,savedir,'BehaviorRegressedOut_V1PM_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])

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

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons

arealabelpairs  = ['V1unl-V1unl',
                    'V1unl-V1lab',
                    'V1lab-V1lab',
                    'PMunl-PMunl',
                    'PMunl-PMlab',
                    'PMlab-PMlab',
                    'V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1unl',
                    'PMlab-V1lab']


arealabelpairs  = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1unl',
                    'PMlab-V1lab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nsampleneurons      = 15
nranks              = 15
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 3

R2_cv               = np.full((narealabelpairs,2,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,2,nSessions),np.nan)

filter_nearby       = True

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)

    # respmat,_  = regress_out_behavior_modulation(ses,X,respmat,rank=5,lam=0,perCond=True)
    # Y_orig,Y_hat_rr,Y_out,_,_  = regress_out_behavior_modulation(ses,rank=5,lam=0,perCond=True)
    Y_orig,Y_hat_rr,Y_out,_,_  = regress_out_behavior_modulation(ses,rank=5,lam=0,perCond=False)

    neuraldata = np.stack((Y_orig,Y_out),axis=2)

    for irbhv,rbhb in enumerate([False,True]):

            # X           = np.stack((ses.respmat_videome[idx_T],
            #                 ses.respmat_runspeed[idx_T],
            #                 ses.respmat_pupilarea[idx_T]),axis=1)

            # Yall   = regress_out_behavior_modulation(ses,B,Yall,rank=3,lam=0)
            
        for iapl, arealabelpair in enumerate(arealabelpairs):
            
            alx,aly = arealabelpair.split('-')

            if filter_nearby:
                idx_nearby  = filter_nearlabeled(ses,radius=50)
            else:
                idx_nearby = np.ones(len(ses.celldata),dtype=bool)

            idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                    ses.celldata['noise_level']<20,	
                                    idx_nearby),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                    ses.celldata['noise_level']<20,	
                                    idx_nearby),axis=0))[0]
        
            X = neuraldata[:,idx_areax,irbhv]
            Y = neuraldata[:,idx_areay,irbhv]

            if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons:
                R2_cv[iapl,irbhv,ises],optim_rank[iapl,irbhv,ises]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot the R2 performance and number of dimensions per area pair
fig = plot_RRR_R2_regressout(R2_cv,optim_rank,arealabelpairs,clrs_arealabelpairs)

# my_savefig(fig,savedir,'RRR_cvR2_RegressOutBehavior_V1PM_LabUnl_%dsessions.png' % nSessions)


#%% Print how many labeled neurons there are in V1 and Pm in the loaded sessions:
print('Number of labeled neurons in V1 and PM:')
for ises, ses in enumerate(sessions):

    print('Session %d: %d in V1, %d in PM' % (ises+1,
                                              np.sum(np.all((ses.celldata['redcell']==1,
                                                             ses.celldata['roi_name']=='V1',
                                                             ses.celldata['noise_level']<20),axis=0)),
                                              np.sum(np.all((ses.celldata['redcell']==1,
                                                             ses.celldata['roi_name']=='PM',
                                                             ses.celldata['noise_level']<20),axis=0))))


#%% Validate regressing out AL RSP activity: 
for ises in range(nSessions):#
    print(np.any(sessions[ises].celldata['roi_name']=='AL'))
ises            = 6
ses             = sessions[ises]
idx_T           = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
# idx_T   = ses.trialdata['stimCond']==0

respmat             = ses.respmat[:,idx_T].T
respmat         = zscore(respmat,axis=0)

idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
                        # ses.celldata['noise_level']<20	
                        ),axis=0))[0]

idx_V1PM   = np.where(np.all((np.isin(ses.celldata['roi_name'],['V1','PM']),
                        # ses.celldata['noise_level']<20	
                        ),axis=0))[0]

Y_orig,Y_hat_rr,Y_out,rank,EVdata  = regress_out_behavior_modulation(ses,X=respmat[:,idx_ALRSP],Y=respmat[:,idx_V1PM],rank=10,lam=0)

# Y_orig,Y_hat_rr,Y_out,rank,EVdata      = regress_out_behavior_modulation(sessions[ises],rank=5,lam=0,perCond=True)
print("Variance explained by behavioral modulation: %1.4f" % EVdata)

#%% Make figure
minmax = 0.75
fig,axes = plt.subplots(1,3,figsize=(6,3),sharex=False)
ax = axes[0]
ax.imshow(X,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('AL+RSP')
ax.set_ylabel('Trials')
ax.set_xlabel('AL RSP Neurons')

ax = axes[1]
ax.imshow(Y_orig,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('Original')
ax.set_yticklabels('')
ax.set_xlabel('V1 PM Neurons')

ax = axes[2]
ax.imshow(Y_out,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('Orig - AL/RSP')
ax.set_yticklabels('')
ax.set_xlabel('V1 PM Neurons')
plt.tight_layout()
my_savefig(fig,savedir,'AL_RSP_RegressedOut_V1PM_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])

#%%
nranks      = 10
EVdata      = np.full((nSessions,nranks),np.nan)
rankdata    = np.full((nSessions,nranks),np.nan)
for ises,ses in enumerate(sessions):
    for rank in range(nranks):
        Y,Y_hat_rr,Y_out,rankdata[ises,rank],EVdata[ises,rank]  = regress_out_behavior_modulation(ses,rank=rank+1,lam=0,perCond=True)

#%% Plot variance regressed out by AL RSP modulation
fig,axes = plt.subplots(1,1,figsize=(3,3),sharex=False)
ax = axes
ax.plot(range(nranks+1),np.concatenate(([0],np.nanmean(EVdata,axis=0))))
ax.set_title('Variance regressed out by behavioral modulation')
ax.set_ylabel('Variance Explained')
ax.set_xlabel('Rank')
ax.set_xticks(range(nranks+1))
sns.despine(top=True,right=True,offset=3)
my_savefig(fig,savedir,'BehaviorRegressedOut_V1PM_%dsessions.png' % nSessions,formats=['png'])




#%% Parameters for RRR between size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1unl',
                    'PMlab-V1lab']

#external areas to include:
regress_out_neural = True

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nsampleneurons      = 20
nranks              = 20
nmodelfits          = 20 #number of times new neurons are resampled 
kfold               = 5
ALRSP_rank          = 15

R2_cv               = np.full((narealabelpairs,2,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,2,nSessions),np.nan)

filter_nearby       = True

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    
    if np.any(sessions[ises].celldata['roi_name']=='AL'):
        idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
        
        respmat             = ses.respmat[:,idx_T].T
        respmat             = zscore(respmat,axis=0)

        idx_ALRSP           = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
                                ses.celldata['noise_level']<20	
                                ),axis=0))[0]

        idx_V1PM            = np.where(np.all((np.isin(ses.celldata['roi_name'],['V1','PM']),
                                ses.celldata['noise_level']<20	
                                ),axis=0))[0]

        Y_orig,Y_hat_rr,Y_out,rank,EVdata  = regress_out_behavior_modulation(ses,X=respmat[:,idx_ALRSP],Y=respmat[:,idx_V1PM],rank=ALRSP_rank,lam=0)

        neuraldata = np.stack((respmat,respmat),axis=2)
        neuraldata[:,idx_V1PM,0] = Y_orig
        neuraldata[:,idx_V1PM,1] = Y_out

        for irbhv,rbhb in enumerate([False,True]):
            for iapl, arealabelpair in enumerate(arealabelpairs):
                alx,aly = arealabelpair.split('-')

                if filter_nearby:
                    idx_nearby  = filter_nearlabeled(ses,radius=50)
                else:
                    idx_nearby = np.ones(len(ses.celldata),dtype=bool)

                idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                        # ses.celldata['noise_level']<20,	
                                        idx_nearby),axis=0))[0]
                idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                        # ses.celldata['noise_level']<20,	
                                        idx_nearby),axis=0))[0]
            
                X = neuraldata[:,idx_areax,irbhv]
                Y = neuraldata[:,idx_areay,irbhv]

                if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons:
                    R2_cv[iapl,irbhv,ises],optim_rank[iapl,irbhv,ises]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%%
fig = plot_RRR_R2_regressout(R2_cv,optim_rank,arealabelpairs,clrs_arealabelpairs)
my_savefig(fig,savedir,'RRR_V1PM_regressoutneuralALRSP_%dsessions.png' % (nSessions))

#%% Fraction of R2 explained by shared activity with AL and RSP:
datatoplot          = (R2_cv[:,0,:] - R2_cv[:,1,:]) / R2_cv[:,0,:]
# datatoplot          = R2_cv[:,1,:]  / R2_cv[:,0,:]
arealabelpairs2     = [al.replace('-','-\n') for al in arealabelpairs]

fig, axes = plt.subplots(1,1,figsize=(4,3))
ax = axes
for iapl, arealabelpair in enumerate(arealabelpairs):
    ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3,datatoplot[iapl,:],color='k',marker='o',s=10)
    ax.errorbar(iapl+0.5,np.nanmean(datatoplot[iapl,:]),np.nanstd(datatoplot[iapl,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

testdata = datatoplot[:,:]
testdata = testdata[:,~np.isnan(testdata).any(axis=0)]
# ax.plot(testdata,color='k',lw=0.5)

df = pd.DataFrame({'R2':  testdata.flatten(),
                       'arealabelpair':np.repeat(np.arange(narealabelpairs),np.shape(testdata)[1])})

annotator = Annotator(ax, statpairs, data=df, x="arealabelpair", y='R2', order=np.arange(narealabelpairs))
annotator.configure(test='Wilcoxon', text_format='star', loc='outside',verbose=False)
# annotator.configure(test='t-test_paired', text_format='star', loc='outside',verbose=False)
annotator.apply_and_annotate()

ttest,pval = stats.ttest_rel(datatoplot[4,:],datatoplot[7,:],nan_policy='omit')

# ax.set_title('Fraction of V1-PM R2 (RRR)\n explained by activity shared with AL and RSP')
# ax.set_ylabel('Fraction of R2')
ax.set_ylabel('V1-PM variance \nnot explained by AL and RSP')
ax.set_ylim([0,my_ceil(np.nanmax(datatoplot[:,:]),2)])

ax.set_xlabel('Population pair')
ax.set_xticks(range(narealabelpairs))
ax.set_ylim([0.5,1])
# ax.set_ylim([0,0.5])
sns.despine(top=True,right=True,offset=3)
ax.set_xticklabels(arealabelpairs2,fontsize=7)

my_savefig(fig,savedir,'RRR_V1PM_regressoutneural_Frac_var_shared_ALRSP_%dsessions.png' % (nSessions))












#%% Deprecated, earlier code below:



# #%% get optimal lambda
# nsampleneurons  = 100
# lambdas         = np.logspace(-6, 5, 20)
# nlambdas        = len(lambdas)
# R2_cv_lams      = np.zeros((nSessions,nlambdas))
# r               = 15
# prePCA          = 25
# # 
# pca             = PCA(n_components=prePCA)
# for ises,ses in enumerate(sessions):
#     idx_T               = ses.trialdata['Orientation']==0
#     idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
#                             ses.celldata['noise_level']<20),axis=0))[0]
#     idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
#                             ses.celldata['noise_level']<20),axis=0))[0]

#     if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

#         idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
#         idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

#         X                   = ses.respmat[np.ix_(idx_areax_sub,idx_T)].T
#         Y                   = ses.respmat[np.ix_(idx_areay_sub,idx_T)].T
        
#         X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
#         Y                   = zscore(Y,axis=0)

#         X                       = pca.fit_transform(X)
#         Y                       = pca.fit_transform(Y)

#         for ilam,lam in enumerate(lambdas):
#             # cross-validation version
#             R2_cv = np.zeros(kfold)
#             kf = KFold(n_splits=kfold)
#             for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
#                 X_train, X_test = X[idx_train], X[idx_test]
#                 Y_train, Y_test = Y[idx_train], Y[idx_test]
#                 B_hat_train         = LM(Y_train,X_train, lam=lam)

#                 B_hat_lr = RRR(Y_train, X_train, B_hat_train, r, mode='right')
#                 Y_hat_test_rr = X_test @ B_hat_lr

#                 R2_cv[ikf] = EV(Y_test,Y_hat_test_rr)
#             R2_cv_lams[ises,ilam] = np.average(R2_cv)


# #%% 
# plt.plot(lambdas,np.nanmean(R2_cv_lams,axis=0))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('R2')

# #optimal labmda:
# lam = lambdas[np.argmax(np.nanmean(R2_cv_lams,axis=0))]
# print('Optimal lam for %d neurons: %.3f' % (nsampleneurons,lam))
# plt.axvline(lam,linestyle='--',color='k')
# plt.text(lam,0,'lam=%.3f' % lam,ha='right',va='center',fontsize=9)
# plt.savefig(os.path.join(savedir,'RRR_Lam_%dneurons.png' % nsampleneurons), format = 'png')
# # plt.savefig(os.path.join(savedir,'RRR_Lam_prePCA_%dneurons.png' % nsampleneurons), format = 'png')


#%% Are CCA and RRR capturing the same signal?
corr_weights_CCA_RRR  = np.full((nOris,2,nSessions),np.nan)
corr_projs_CCA_RRR    = np.full((nOris,2,nSessions),np.nan)

lam                 = 500
Nsub                = 250
prePCA              = 25

model_CCA           = CCA(n_components=10,scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model'):    # iterate over sessions
    idx_areax       = np.where(np.all((ses.celldata['roi_name']=='V1',
                                                ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay       = np.where(np.all((ses.celldata['roi_name']=='PM',
                                            ses.celldata['noise_level']<20),axis=0))[0]
    
    for iori,ori in enumerate(oris): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori

        if len(idx_areax)>Nsub and len(idx_areay)>Nsub:
            idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
            idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)

            X                   = ses.respmat[np.ix_(idx_areax_sub,idx_T)].T
            Y                   = ses.respmat[np.ix_(idx_areay_sub,idx_T)].T
            
            X                   = zscore(X,axis=0)  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0)

            if prePCA and Nsub>prePCA:
                prepca      = PCA(n_components=prePCA)
                X           = prepca.fit_transform(X)
                Y           = prepca.fit_transform(Y)
                
            # Compute and store canonical correlations for the first pair
            X_c, Y_c        = model_CCA.fit_transform(X,Y)

            B_hat           = LM(Y,X, lam=lam)

            L, W            = low_rank_approx(B_hat,1, mode='right')

            B_hat_lr        = RRR(Y, X, B_hat, r=1, mode='right')
            # B_hat_lr = RRR(Y_train, X_train, B_hat_train, r, mode='left')

            Y_hat           = X @ L

            corr_weights_CCA_RRR[iori,0,ises] = np.corrcoef(model_CCA.x_weights_[:,0],L.flatten())[0,1]
            corr_weights_CCA_RRR[iori,1,ises] = np.corrcoef(model_CCA.y_weights_[:,0],W.flatten())[0,1]

            corr_projs_CCA_RRR[iori,0,ises] = np.corrcoef(X_c[:,0],np.array([X @ L]).flatten())[0,1]
            corr_projs_CCA_RRR[iori,1,ises] = np.corrcoef(Y_c[:,0],np.array([Y @ W.T]).flatten())[0,1]

#%% Plot correlation between CCA and RRR weights
fig,axes = plt.subplots(1,2,figsize=(5,3),sharey=True,sharex=True)

ax = axes[0]

for i in range(2):
    data = corr_weights_CCA_RRR[:,i,:].flatten()
    ax.scatter(np.zeros(len(data))+np.random.randn(len(data))*0.15+i,data,marker='.',color='k')
ax.set_title('Weights')
ax.set_ylabel('Correlation')

ax = axes[1]
for i in range(2):
    data = corr_projs_CCA_RRR[:,i,:].flatten()
    ax.scatter(np.zeros(len(data))+np.random.randn(len(data))*0.15+i,data,marker='.',color='k')
ax.set_title('Projections')
ax.set_xticks([0,1],areas)
ax.set_xlabel('Area')
ax.set_ylim([-1.1,1.1])
sns.despine(fig=fig, top=True, right=True,offset=5)

fig.suptitle('Correlation between CCA and RRR:')
fig.tight_layout()
fig.savefig(os.path.join(savedir,'Corr_CCA_RRR_weights.png'), format = 'png')

#%% TO DO:
# chose the value of λ using X-fold cross-validation
# Identify the optimal kfold

#%% 
kfoldsplits         = np.array([2,3,5,8,10,20,50])

R2_cv               = np.full((nOris,nSessions,nmodelfits,len(kfoldsplits)),np.nan)

lam                 = 500
nmodelfits          = 5
Nsub                = 250

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

            for ikfn, kfold in enumerate(kfoldsplits):
                # cross-validation version
                R2_kfold    = np.zeros((kfold))
                kf          = KFold(n_splits=kfold)
                for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                    X_train, X_test = X[idx_train], X[idx_test]
                    Y_train, Y_test = Y[idx_train], Y[idx_test]

                    B_hat_train         = LM(Y_train,X_train, lam=lam)
                    
                    r                   = np.min(popsizes) #rank for RRR

                    B_hat_lr            = RRR(Y_train, X_train, B_hat_train, r, mode='right')
                    # B_hat_lr = RRR(Y_train, X_train, B_hat_train, r, mode='left')

                    Y_hat_test_rr       = X_test @ B_hat_lr

                    R2_kfold[ikf]       = EV(Y_test,Y_hat_test_rr) *  np.sum(pca.explained_variance_ratio_)
                R2_cv[iori,ises,imf,ikfn] = np.average(R2_kfold)

#%% Plot the results for the optimal kFOLD
R2_cv_mean = np.nanmean(R2_cv,axis=(0,1,2))
fig,axes = plt.subplots(1,1,figsize=(4,3))
ax = axes
ax.plot(kfoldsplits,R2_cv_mean,color='k',linewidth=2)
optimal_kfold = kfoldsplits[np.argmax(R2_cv_mean)]
ax.text(optimal_kfold,np.max(R2_cv_mean),f'Optimal kfold: {optimal_kfold}',fontsize=12)
ax.set_title('Mean R2 across sessions')
ax.set_xlabel('Kfold')
ax.set_ylabel('R2')
ax.set_ylim([0,0.2])
sns.despine(fig=fig, top=True, right=True,offset=5)
plt.savefig(os.path.join(savedir,'RRR_R2_kfold.png'), format = 'png', bbox_inches='tight')



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






#%% Using trial averaged or using timepoint fluctuations:

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=['V1','PM','AL'],filter_areas=['V1','PM','AL'])

#%%  Load data properly:        
calciumversion = 'dF'
# calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

    # [sessions[ises].tensor,t_axis]     = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
    #                              t_pre, t_post, binsize,method='nearby')
    

#%%
print('Number of cells in AL per session:')
for ises in range(nSessions):
    print('%d: %d' % (ises,np.sum(np.all((sessions[ises].celldata['roi_name']=='AL',
                                           sessions[ises].celldata['noise_level']<20),axis=0))))


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
areacombs  = [['V1','PM','AL'],
              ['V1','AL','PM'],
              ['PM','V1','AL'],
              ['PM','AL','V1'],
              ['AL','V1','PM'],
              ['AL','PM','V1']]

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

                            # decomposing and low rank approximation of A
                            U, s, V = linalg.svd(Y_hat_train, full_matrices=False)

                            S = linalg.diagsvd(s,U.shape[0],s.shape[0])

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
