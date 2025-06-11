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
from utils.RRRlib import *
from utils.regress_lib import *
from preprocessing.preprocesslib import assign_layer

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross')

#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR

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


# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_areas=areas)

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
calciumversion = 'deconv'
# calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)


#%% Test wrapper function: (should be fast, just one model fit etc. EV around .08 or 0.1)
nsampleneurons  = 20
nranks          = 25
nmodelfits      = 1 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions),np.nan)
optim_rank      = np.full((nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    
    R2_cv[ises],optim_rank[ises],_      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))


#%% Perform RRR: 
ises = 2
ses = sessions[ises]

#%% Fit:
nN                  = 250
nM                  = 250
nranks              = 10
lam                 = 0

idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)

idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                                       ses.celldata['labeled']=='unl',
                        ses.celldata['noise_level']<20),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                                       ses.celldata['labeled']=='unl',
                        ses.celldata['noise_level']<20),axis=0))[0]

idx_areax_sub       = np.random.choice(idx_areax,nN,replace=False)
idx_areay_sub       = np.random.choice(idx_areay,nM,replace=False)

X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T

X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
Y                   = zscore(Y,axis=0)

B_hat_rr               = np.full((nM,nN),np.nan)  
# Y_hat_rr               = np.full((nM,nranks),np.nan)  

B_hat         = LM(Y,X, lam=lam)

Y_hat         = X @ B_hat

U, s, V = svds(B_hat,k=nranks,which='LM')
U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
V       = V.T

for r in range(nranks):
    if np.mean(U[:,r])<0 and np.mean(V[:,r])<0:
        U[:,r] = -U[:,r]
        V[:,r] = -V[:,r]

U_sorted = U[np.argsort(-U[:,0]),:]
V_sorted = V[np.argsort(-V[:,0]),:]


#%% Make figure:
nrankstoshow = 5 
fig,axes = plt.subplots(nrankstoshow,2,figsize=(6,nrankstoshow),sharex=False,sharey=True)
for r in range(nrankstoshow):
    ax = axes[r,0]
    ax.bar(range(nN),U_sorted[:,r],color='k')
    ax.set_xlim([0,nN])
    ax.set_ylim([-0.3,0.3])
    if r==0: 
        ax.set_title('Source Area weights (V1)\n(sorted by dim. 1)',fontsize=10)
    ax.axis('off')

    ax = axes[r,1]
    ax.bar(range(nM),V_sorted[:,r],color='k')
    if r==0: 
        ax.set_title('Target Area weights (PM)\n(sorted by dim. 1)',fontsize=10)
    ax.axis('off')
    ax.set_xlim([0,nM])
    ax.set_ylim([-0.3,0.3])

my_savefig(fig,savedir,'RRR_weights_acrossranks_V1PM_%s' % ses.sessiondata['session_id'][0],formats=['png'])

#%% Plot the mean across ranks: 
fig,axes = plt.subplots(1,1,figsize=(4,3))
ax = axes
ax.plot(range(1,nranks+1),np.mean(U[:,:nranks],axis=0),label='Source',color='k',linestyle='--')
ax.plot(range(1,nranks+1),np.mean(V[:,:nranks],axis=0),label='Target',color='k',linestyle=':')
ax.set_xticks(range(1,nranks+1))
ax.axhline(0,linestyle='--',color='grey')
ax.legend(frameon=False,fontsize=8)
ax.set_xlabel('Dimension')
ax.set_ylabel('Mean weight')
sns.despine(right=True,top=True,offset=3,trim=True)
my_savefig(fig,savedir,'RRR_Meanweights_acrossranks_V1PM_%s' % ses.sessiondata['session_id'][0],formats=['png'])


#%% 

U_sig = np.logical_or(zscore(U,axis=0)>2,zscore(U,axis=0)<-2)
V_sig = np.logical_or(zscore(V,axis=0)>2,zscore(V,axis=0)<-2)

# U_sig = U_sig[np.argsort(-U[:,0]),:]
# V_sig = V_sig[np.argsort(-V[:,0]),:]

U_overlap = np.empty((nranks,nranks))
V_overlap = np.empty((nranks,nranks))

for i in range(nranks):
    for j in range(nranks):
        U_overlap[i,j] = np.sum(np.logical_and(U_sig[:,i],U_sig[:,j])) / np.sum(np.logical_or(U_sig[:,i],U_sig[:,j]))
        V_overlap[i,j] = np.sum(np.logical_and(V_sig[:,i],V_sig[:,j])) / np.sum(np.logical_or(V_sig[:,i],V_sig[:,j]))

fig,axes = plt.subplots(1,2,figsize=(6,3))
ax = axes[0]
ax.imshow(U_overlap,vmin=0,vmax=1)
ax.set_title('Source Area',fontsize=10)
ax.set_xticks(range(nranks))
ax.set_yticks(range(nranks))
ax.set_xticklabels(range(1,nranks+1))
ax.set_yticklabels(range(1,nranks+1))
ax.set_xlabel('Rank')
ax.set_ylabel('Rank')
# print(np.mean(U_overlap[np.triu_indices(nranks,k=1)]))
print('%1.2f average overlap of significant source neurons across pairs of dimensions' % np.mean(U_overlap[np.triu_indices(nranks,k=1)]))

ax = axes[1]
ax.imshow(V_overlap,vmin=0,vmax=1)
ax.set_title('Target Area',fontsize=10)
ax.set_xticks(range(nranks))
ax.set_yticks(range(nranks))
ax.set_xticklabels(range(1,nranks+1))
ax.set_yticklabels(range(1,nranks+1))
ax.set_xlabel('Rank')
ax.set_ylabel('Rank')
plt.suptitle('Sign. weight overlap\nacross pairs of dimensions',fontsize=10)
plt.tight_layout()
print('%1.2f average overlap of significant target neurons across pairs of dimensions' % np.mean(V_overlap[np.triu_indices(nranks,k=1)]))

sns.despine(right=True,top=True,offset=3,trim=True)
my_savefig(fig,savedir,'RRR_SigWeightOverlap_acrossranks_V1PM_%s' % ses.sessiondata['session_id'][0],formats=['png'])


#%% Perform RRR on all neurons in V1 to PM for one session and show labeled weights:
ises = 2
ses = sessions[ises]

#%% Fit:
nranks              = 10
lam                 = 0

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

B_hat_rr               = np.full((nM,nN),np.nan)  
# Y_hat_rr               = np.full((nM,nranks),np.nan)  

B_hat         = LM(Y,X, lam=lam)

Y_hat         = X @ B_hat

U, s, V = svds(B_hat,k=nranks,which='LM')
U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
V       = V.T

for r in range(nranks):
    if np.mean(U[:,r])<0 and np.mean(V[:,r])<0:
        U[:,r] = -U[:,r]
        V[:,r] = -V[:,r]

U_sorted = U[np.argsort(-U[:,0]),:]
V_sorted = V[np.argsort(-V[:,0]),:]

#%% Make figure similar to above but now for labeled cells
nrankstoshow = 5
minmax = 0.2
fig,axes = plt.subplots(nrankstoshow,2,figsize=(6,nrankstoshow),sharex=False,sharey=True)
for r in range(nrankstoshow):
    ax = axes[r,0]
    idx_lab = sessions[ises].celldata['redcell'][idx_areax]== 1
    ax.bar(range(np.sum(idx_lab)),U_sorted[idx_lab,r],color='r')
    # ax.bar(range(nN),U_sig[:,r],color='r')
    ax.set_xlim([0,np.sum(idx_lab)])
    ax.set_ylim([-minmax,minmax])
    if r==0: 
        ax.set_title('V1$_{PM}$ weights\n(sorted by dim. 1)',fontsize=10)
    ax.axis('off')

    ax = axes[r,1]
    idx_lab = sessions[ises].celldata['redcell'][idx_areay]== 1
    ax.bar(range(np.sum(idx_lab)),V_sorted[idx_lab,r],color='r')
    # ax.bar(range(nM),V_sorted[:,r],color='k')
    # ax.bar(range(nM),V_sig[:,r],color='r')
    if r==0: 
        ax.set_title('PM$_{V1}$ weights\n(sorted by dim. 1)',fontsize=10)
    ax.axis('off')
    ax.set_xlim([0,np.sum(idx_lab)])
# plt.tight_layout()
my_savefig(fig,savedir,'RRR_weights_acrossranks_V1PMlabeled_%s' % ses.sessiondata['session_id'][0],formats=['png'])






#%% Do RRR in FF and FB direction and compare performance:
nsampleneurons  = 100
nranks          = 25
nmodelfits      = 10 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions,2),np.nan)
optim_rank      = np.full((nSessions,2),np.nan)
R2_ranks        = np.full((nSessions,2,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    
    if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
        continue
    R2_cv[ises,0],optim_rank[ises,0],R2_ranks[ises,0,:,:,:]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

    R2_cv[ises,1],optim_rank[ises,1],R2_ranks[ises,1,:,:,:]      = RRR_wrapper(X, Y, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot the performance across sessions as a function of rank: 
clrs_areapairs = get_clr_area_pairs(['V1-PM','PM-V1'])
datatoplot = np.nanmean(R2_ranks,axis=(3,4))

fig,axes = plt.subplots(1,3,figsize=(6.5,2.2))
ax = axes[0]
handles = []
handles.append(shaded_error(np.arange(nranks),datatoplot[:,0,:],color=clrs_areapairs[0],error='sem',ax=ax))
handles.append(shaded_error(np.arange(nranks),datatoplot[:,1,:],color=clrs_areapairs[1],error='sem',ax=ax))

ax.legend(handles,['V1->PM','PM->V1'],frameon=False,fontsize=8,loc='lower right')
ax.set_xlabel('Rank')
ax.set_ylabel('R2')
ax.set_yticks([0,0.1,0.2,0.3])
ax.set_ylim([0,0.2])
ax.set_xticks(np.arange(0,nranks+1,5))
ax.set_xlim([0,nranks])

ax = axes[1]
ax.scatter(R2_cv[:,0],R2_cv[:,1],color='k',s=10)
ax.set_xlabel('V1->PM')
ax.set_ylabel('PM->V1')
ax.plot([0,1],[0,1],color='k',linestyle='--',linewidth=0.5)
ax.set_xlim([0,0.4])
ax.set_ylim([0,0.4])
t,p = ttest_rel(R2_cv[:,0],R2_cv[:,1],nan_policy='omit')
print('Paired t-test (R2): p=%.3f' % p)
if p<0.05:
    ax.text(0.6,0.1,'p=%.3f' % p,transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')
else: 
    ax.text(0.6,0.1,'p=%.3f' % p,transform=ax.transAxes,ha='center',va='center',fontsize=10,color='k')

ax = axes[2]
ax.scatter(optim_rank[:,0],optim_rank[:,1],color='k',s=10)
ax.plot([0,20],[0,20],color='k',linestyle='--',linewidth=0.5)
ax.set_xlabel('V1->PM')
ax.set_ylabel('PM->V1')
ax.set_xlim([0,20])
ax.set_ylim([0,20])
ax_nticks(ax,3)
t,p = ttest_rel(optim_rank[:,0],optim_rank[:,1],nan_policy='omit')
print('Paired t-test (R2): p=%.3f' % p)
if p<0.05:
    ax.text(0.6,0.1,'p=%.3f' % p,transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')
else:
    ax.text(0.6,0.1,'p=%.3f' % p,transform=ax.transAxes,ha='center',va='center',fontsize=10,color='k')
    
# ax.set_ylim([0,0.3])
plt.tight_layout()
sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'RRR_R2_acrossranks_V1PM_%dsessions' % nSessions,formats=['png'])



#%% Do RRR in FF and FB direction and compare performance:
nsampleneurons  = 100
nranks          = 25
nmodelfits      = 20 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions,2,nmodelfits),np.nan)
optim_rank      = np.full((nSessions,2,nmodelfits),np.nan)
dims            = np.full((nSessions,2,nmodelfits),np.nan)

dimmethod = 'participation_ratio'
dimmethod = 'parallel_analysis' #very slow
dimmethod = 'pca_shuffle' 

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
        continue    

    for imf in range(nmodelfits):
        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)
    
        X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T

        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(Y,axis=0)

        R2_cv[ises,0,imf],optim_rank[ises,0,imf],_      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=1)

        R2_cv[ises,1,imf],optim_rank[ises,1,imf],_      = RRR_wrapper(X, Y, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=1)

        dims[ises,0,imf] = estimate_dimensionality(X,method=dimmethod)
        dims[ises,1,imf] = estimate_dimensionality(Y,method=dimmethod)

#%% 
fig,axes = plt.subplots(2,2,figsize=(5,5),sharex=True,sharey='row')
ax = axes[0,0]
ax.scatter(dims[:,0,:],optim_rank[:,0,:],color='r',s=10,alpha=0.5)
ax.scatter(dims[:,1,:],optim_rank[:,1,:],color='b',s=10,alpha=0.5)
ax.set_xlabel('Source Dimensionality')
ax.set_ylabel('Interarea Dimensionality')
ax.set_ylim([np.nanmin(optim_rank)*0.8,np.nanmax(optim_rank)*1.2])
ax.set_xlim([np.nanmin(dims)*0.8,np.nanmax(dims)*1.2])
ax_nticks(ax,3)
ax.legend(['V1->PM','PM->V1'],fontsize=8,frameon=True,loc='upper left')
add_corr_results(ax,dims.flatten(),optim_rank.flatten())

ax = axes[0,1]
ax.scatter(dims[:,1,:],optim_rank[:,0,:],color='r',s=10,alpha=0.5)
ax.scatter(dims[:,0,:],optim_rank[:,1,:],color='b',s=10,alpha=0.5)
ax.set_xlabel('Target Dimensionality')
ax_nticks(ax,3)

add_corr_results(ax,dims.flatten(),np.flip(optim_rank,axis=1).flatten())

ax = axes[1,0]
ax.scatter(dims[:,0,:],R2_cv[:,0,:],color='r',s=10,alpha=0.5)
ax.scatter(dims[:,1,:],R2_cv[:,1,:],color='b',s=10,alpha=0.5)
ax.set_xlabel('Source Dimensionality')
ax.set_ylabel('R2')
ax_nticks(ax,3)

ax.set_ylim([np.nanmin(R2_cv)*0.8,np.nanmax(R2_cv)*1.2])
ax.set_xlim([np.nanmin(dims)*0.8,np.nanmax(dims)*1.2])

add_corr_results(ax,dims.flatten(),R2_cv.flatten())

ax = axes[1,1]
ax.scatter(dims[:,0,:],R2_cv[:,1,:],color='r',s=10,alpha=0.5)
ax.scatter(dims[:,1,:],R2_cv[:,0,:],color='b',s=10,alpha=0.5)
ax.set_xlabel('Target Dimensionality')
ax_nticks(ax,3)

add_corr_results(ax,dims.flatten(),np.flip(R2_cv,axis=1).flatten())

plt.tight_layout()
sns.despine(top=True,right=True,offset=3)

my_savefig(fig,savedir,'RRR_Perf_WithinAcross_Dimensionality_%dsessions' % nSessions)

#%% 
fig,axes = plt.subplots(1,1,figsize=(3.5,3.5),sharex=True,sharey='row')
ax = axes
ax.scatter(dims[:,0,:],dims[:,1,:],color='r',s=10,alpha=0.5)
# ax.scatter(dims[:,1,:],optim_rank[:,1,:],color='b',s=10,alpha=0.5)
ax.set_xlabel('Source Dimensionality')
ax.set_ylabel('Target Dimensionality')
ax.set_xlim([np.nanmin(dims)*0.8,np.nanmax(dims)*1.2])
ax.set_ylim([np.nanmin(dims)*0.8,np.nanmax(dims)*1.2])
ax_nticks(ax,3)
add_corr_results(ax,dims.flatten(),optim_rank.flatten())

#%% Is the difference in feedforward vs feedback (V1-PM vs PM-V1) due to different dimensionality?




#%% 
######  ####### #     # ### #     #    #    #     # #######                                
#     # #     # ##   ##  #  ##    #   # #   ##    #    #                                   
#     # #     # # # # #  #  # #   #  #   #  # #   #    #                                   
#     # #     # #  #  #  #  #  #  # #     # #  #  #    #                                   
#     # #     # #     #  #  #   # # ####### #   # #    #                                   
#     # #     # #     #  #  #    ## #     # #    ##    #                                   
######  ####### #     # ### #     # #     # #     #    #    
                               
#     #  #####     ######  ######  ####### ######  ###  #####  ####### ### #     # ####### 
#     # #     #    #     # #     # #       #     #  #  #     #    #     #  #     # #       
#     # #          #     # #     # #       #     #  #  #          #     #  #     # #       
#     #  #####     ######  ######  #####   #     #  #  #          #     #  #     # #####   
 #   #        #    #       #   #   #       #     #  #  #          #     #   #   #  #       
  # #   #     #    #       #    #  #       #     #  #  #     #    #     #    # #   #       
   #     #####     #       #     # ####### ######  ###  #####     #    ###    #    ####### 

#%% Dominant vs predictive dimensions: 
# from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import FactorAnalysis as FA
kf                  = KFold(n_splits=kfold,shuffle=True,random_state=None)

nsampleneurons  = 100
nranks          = 40
nmodelfits      = 10 #number of times new neurons are resampled 
kfold           = 5
# R2_cv           = np.full((nSessions,2),np.nan)
# optim_rank      = np.full((nSessions,2),np.nan)
R2_ranks        = np.full((nSessions,4,nranks,nmodelfits,kfold),np.nan)
# 0: V1-PM RRR predictive dimensions
# 1: V1-PM linear regression from dominant FA dimensions in V1
# 2: V1-V1 from RRR predictive dimensions
# 3: V1-V1 from linear regression from dominant FA dimensions in V1

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    
    if len(idx_areax)<(nsampleneurons*2) or len(idx_areay)<nsampleneurons:
        continue

    for imf in range(nmodelfits):
        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areax_sub2      = np.random.choice(list(set(idx_areax) - set(idx_areax_sub)),nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        X_sub               = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        X_sub2               = sessions[ises].respmat[np.ix_(idx_areax_sub2,idx_T)].T
        Y_sub               = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
        
        X_sub               = zscore(X_sub,axis=0)
        X_sub2              = zscore(X_sub2,axis=0)
        Y_sub               = zscore(Y_sub,axis=0)
        
        #RRR from V1 to PM:
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X_sub)):
            X_train, X_test     = X_sub[idx_train], X_sub[idx_test]
            Y_train, Y_test     = Y_sub[idx_train], Y_sub[idx_test]
            B_hat_train         = LM(Y_train,X_train, lam=lam)
            Y_hat_train         = X_train @ B_hat_train

            # decomposing and low rank approximation of Yhat
            U, s, V = svds(Y_hat_train,k=np.min((nranks,nN,nM))-1,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            for r in range(nranks):
                B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace
                R2_ranks[ises,0,r,imf,ikf] = EV(Y_test,Y_hat_rr_test)

        #FA regression from V1 to PM:
        fa                  = FA(n_components=nranks)
        X_fa                = fa.fit_transform(X_sub)

        for ikf, (idx_train, idx_test) in enumerate(kf.split(X_fa)):
            for r in range(1,nranks):
                X_train, X_test     = X_fa[np.ix_(idx_train,range(r))], X_fa[np.ix_(idx_test,range(r))]
                Y_train, Y_test     = Y_sub[idx_train], Y_sub[idx_test]

                B_hat               = LM(Y_train,X_train, lam=0)

                Y_hat_test          = X_test @ B_hat

                R2_ranks[ises,1,r,imf,ikf] = EV(Y_test,Y_hat_test)
        
        #RRR from V1 to V1:
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X_sub)):
            X_train, X_test     = X_sub[idx_train], X_sub[idx_test]
            Y_train, Y_test     = X_sub2[idx_train], X_sub2[idx_test]
            B_hat_train         = LM(Y_train,X_train, lam=lam)
            Y_hat_train         = X_train @ B_hat_train

            # decomposing and low rank approximation of Yhat
            U, s, V = svds(Y_hat_train,k=np.min((nranks,nN,nM))-1,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            for r in range(nranks):
                B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace
                R2_ranks[ises,2,r,imf,ikf] = EV(Y_test,Y_hat_rr_test)

        #FA regression from V1 to V1:
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X_fa)):
            for r in range(1,nranks):
                X_train, X_test     = X_fa[np.ix_(idx_train,range(r))], X_fa[np.ix_(idx_test,range(r))]
                Y_train, Y_test     = X_sub2[idx_train], X_sub2[idx_test]

                B_hat               = LM(Y_train,X_train, lam=0)

                Y_hat_test          = X_test @ B_hat

                R2_ranks[ises,3,r,imf,ikf] = EV(Y_test,Y_hat_test)

#%% Rank from R2
R2_cv           = np.full((nSessions,4),np.nan)
optim_rank      = np.full((nSessions,4),np.nan)
for ises in range(nSessions):
    for i in range(4):
        R2_cv[ises,i],optim_rank[ises,i] = rank_from_R2(R2_ranks[ises,i,:,:,:].reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)

#%% Rank from R2
R2_cv           = np.full((4),np.nan)
optim_rank      = np.full((4),np.nan)
for i in range(4):
    R2_cv[i],optim_rank[i] = rank_from_R2(np.nanmean(R2_ranks[:,i,:,:,:],axis=0).reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)


#%% Make figure: show predictive performance for dominant vs predictive dimensions in predicting PM or another V1 population:
fig,axes = plt.subplots(1,2,figsize=(5.5,2.5),sharex=True,sharey=True)
ax = axes[0]
ax.set_title('V1-PM')
idx0 = 0
idx1 = 1
clr1 = 'k'
clr2 = 'g'
handles = []
handles.append(shaded_error(np.arange(nranks),np.nanmean(R2_ranks[:,idx0,:,:,:],axis=(-1,-2)),color=clr1,error='sem',ax=ax))
handles.append(shaded_error(np.arange(nranks),np.nanmean(R2_ranks[:,idx1,:,:,:],axis=(-1,-2)),color=clr2,error='sem',ax=ax))
ax.axhline(R2_cv[idx0],linestyle='--',color=clr1,linewidth=0.5)
ax.axhline(R2_cv[idx1],linestyle='--',color=clr2,linewidth=0.5)
ax.plot(optim_rank[idx0],R2_cv[idx0]+0.01,'v',color=clr1)
ax.plot(optim_rank[idx1],R2_cv[idx1]+0.01,'v',color=clr2)
ax.legend(handles=handles,labels=['PM-Predictive','V1-Dominant'],
          loc='lower right',frameon=False,fontsize=8)
ax.set_ylabel('Fraction of variance explained')

ax = axes[1]
ax.set_title('V1-V1')
idx0 = 2
idx1 = 3
clr1 = 'k'
clr2 = 'b'
handles = []
handles.append(shaded_error(np.arange(nranks),np.nanmean(R2_ranks[:,idx0,:,:,:],axis=(-1,-2)),color=clr1,error='sem',ax=ax))
handles.append(shaded_error(np.arange(nranks),np.nanmean(R2_ranks[:,idx1,:,:,:],axis=(-1,-2)),color=clr2,error='sem',ax=ax))
ax.axhline(R2_cv[idx0],linestyle='--',color=clr1,linewidth=0.5)
ax.axhline(R2_cv[idx1],linestyle='--',color=clr2,linewidth=0.5)
ax.plot(optim_rank[idx0],R2_cv[idx0]+0.01,'v',color=clr1)
ax.plot(optim_rank[idx1],R2_cv[idx1]+0.01,'v',color=clr2)

ax.set_xticks(range(0,nranks+1,5))
ax.set_yticks(np.arange(0,0.4,0.05))
ax.set_xlabel('Rank')
ax.set_xlim([0,nranks-5])
ax.set_ylim([0,0.22])
# ax.legend(handles=handles,labels=['V1-Dominant','PM-Predictive'],frameon=False,fontsize=8)
ax.legend(handles=handles,labels=['V1-Predictive','V1-Dominant'],
                    loc='lower right',frameon=False,fontsize=8)
sns.despine(offset=5,top=True,right=True)
fig.tight_layout()
my_savefig(fig,savedir,'V1PM_dominant_predictive_ranks_%dsessions' % (nSessions),formats=['png'])


#%% 
#     #    #         ######  #     #    #          #    #     # ####### ######   #####  
#     #   ##         #     # ##   ##    #         # #    #   #  #       #     # #     # 
#     #  # #         #     # # # # #    #        #   #    # #   #       #     # #       
#     #    #   ##### ######  #  #  #    #       #     #    #    #####   ######   #####  
 #   #     #         #       #     #    #       #######    #    #       #   #         # 
  # #      #         #       #     #    #       #     #    #    #       #    #  #     # 
   #     #####       #       #     #    ####### #     #    #    ####### #     #  #####  


#%%  #assign arealayerlabel
for ises in range(nSessions):   
    sessions[ises].celldata = assign_layer(sessions[ises].celldata)

#%%

for ses in sessions:
    ses.celldata['arealayer'] = ses.celldata['roi_name'] + ses.celldata['layer']

for ses in sessions:
    print(np.sum(ses.celldata['arealayer']=='V1L5'))


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealayerpairs  = ['V1L2/3-PML2/3',
                   'V1L2/3-PML5',
                   'PML2/3-V1L2/3',
                   'PML5-V1L2/3',
                   'PML2/3-PML5',
                   'PML5-PML2/3'                   
                   ]

arealayerpairs  = ['V1L2/3-PML2/3',
                   'V1L2/3-PML5',
                   'V1L2/3-V1L5',
                   'V1L5-V1L2/3',
                   'PML2/3-V1L2/3',
                   'PML5-V1L2/3',
                   'PML2/3-PML5',
                   'PML5-PML2/3'                   
                   ]

clrs_arealayerpairs = get_clr_arealayerpairs(arealayerpairs)
narealayerpairs     = len(arealayerpairs)

lam                 = 0
nsampleneurons      = 100
nranks              = 25
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((narealayerpairs,nSessions),np.nan)
optim_rank          = np.full((narealayerpairs,nSessions),np.nan)
R2_cv_folds         = np.full((narealayerpairs,nSessions,nranks,nmodelfits,kfold),np.nan)
filter_nearby       = False

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for populations'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
        
    for ialp, arealayerpair in enumerate(arealayerpairs):
        
        alx,aly = arealayerpair.split('-')

        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealayer']==alx,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealayer']==aly,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
    
        #Neural data:
        X           = ses.respmat[np.ix_(idx_areax,idx_T)].T
        Y           = ses.respmat[np.ix_(idx_areay,idx_T)].T

        if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons:
            R2_cv[ialp,ises],optim_rank[ialp,ises],R2_cv_folds[ialp,ises,:,:,:]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot the performance across sessions as a function of rank: 
datatoplot = np.nanmean(R2_cv_folds,axis=(3,4))

fig,axes = plt.subplots(1,3,figsize=(7.5,2.5))
ax = axes[0]
handles = []
for ialp, arealayerpair in enumerate(arealayerpairs):
    handles.append(shaded_error(np.arange(nranks),datatoplot[ialp,:,:],color=clrs_arealayerpairs[ialp],error='sem',ax=ax))
# handles.append(shaded_error(np.arange(nranks),datatoplot[:,1,:],color=clrs_areapairs[1],error='sem',ax=ax))

ax.legend(handles,arealayerpairs,frameon=False,fontsize=6,loc='upper left',ncol=2)
my_legend_strip(ax)
# ax.legend(handles,arealayerpairs,frameon=False,fontsize=8,bbox_to_anchor=(1.05, 1),loc='upper left')
ax.set_xlabel('Rank')
ax.set_xlim([0,nranks])
ax.set_ylabel('R2')
ax.set_ylim([0,0.35])
ax.set_yticks([0,0.1,0.2,0.3])
ax.set_xlim([0,nranks])

datatoplot = R2_cv
# datatoplot  = datatoplot / np.nanmean(datatoplot,axis=0)[np.newaxis,:] #normalize to whole dataset
ax = axes[1]
for ialp, arealayerpair in enumerate(arealayerpairs):
    ax.scatter(np.ones(nSessions)*ialp,datatoplot[ialp,:],color=clrs_arealayerpairs[ialp],s=4)
    ax.errorbar(ialp+0.25,np.nanmean(datatoplot[ialp,:]),np.nanstd(datatoplot[ialp,:])/np.sqrt(np.sum(~np.isnan(datatoplot[ialp,:]))),marker='.',markersize=10,color=clrs_arealayerpairs[ialp],capsize=0)
ax.set_ylim([0,ax.get_ylim()[1]])
ax.set_ylabel('R2')

datatotest = datatoplot[:,~np.any(np.isnan(datatoplot),axis=0)]
pairs = list(itertools.combinations(arealayerpairs,2))

df = pd.DataFrame({'R2':  datatotest.flatten(),
                       'arealayerpair':np.repeat(arealayerpairs,np.shape(datatotest)[1])})

annotator = Annotator(ax, pairs, data=df, x='arealayerpair', y='R2', order=arealayerpairs)
annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False,
                    comparisons_correction=None,alpha=0.05,
                    # hide_non_significant=True,
                    pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"]]
                    )
# annotator.apply_and_annotate()

ax = axes[2]
datatoplot = optim_rank
for ialp, arealayerpair in enumerate(arealayerpairs):
    ax.scatter(np.ones(nSessions)*ialp,datatoplot[ialp,:],color=clrs_arealayerpairs[ialp],s=4)
    ax.errorbar(ialp+0.25,np.nanmean(datatoplot[ialp,:]),np.nanstd(datatoplot[ialp,:])/np.sqrt(np.sum(~np.isnan(datatoplot[ialp,:]))),marker='.',markersize=10,color=clrs_arealayerpairs[ialp],capsize=0)
ax.set_ylim([0,ax.get_ylim()[1]])
ax.set_ylabel('Rank')

sns.despine(top=True,right=True,offset=3)
axes[1].set_xticks(range(narealayerpairs),arealayerpairs,rotation=45,ha='right',fontsize=6)
axes[2].set_xticks(range(narealayerpairs),arealayerpairs,rotation=45,ha='right',fontsize=6)

plt.tight_layout()
my_savefig(fig,savedir,'RRR_V1PM_Layers_%dsessions' % (nSessions),formats=['png'])
# my_savefig(fig,savedir,'RRR_V1PM_Layers_%dsessions_stats' % (nSessions),formats=['png'])



#%% 

#     #    #      ######  #     #       #    #          ######   #####  ######  
#     #   ##      #     # ##   ##      # #   #          #     # #     # #     # 
#     #  # #      #     # # # # #     #   #  #          #     # #       #     # 
#     #    #      ######  #  #  #    #     # #          ######   #####  ######  
 #   #     #      #       #     #    ####### #          #   #         # #       
  # #      #      #       #     #    #     # #          #    #  #     # #       
   #     #####    #       #     #    #     # #######    #     #  #####  #       

#%% 




#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
lam                 = 0
nsampleneurons      = 100
nranks              = 25
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((nareas,nareas,nSessions),np.nan)
optim_rank          = np.full((nareas,nareas,nSessions),np.nan)
R2_cv_folds         = np.full((nareas,nareas,nSessions,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for populations'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
        
    for iax, areax in enumerate(areas):

        for iay, areay in enumerate(areas):
            if areax==areay:
                continue
            idx_areax           = np.where(np.all((ses.celldata['roi_name']==areax,
                                    ses.celldata['noise_level']<20),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['roi_name']==areay,
                                    ses.celldata['noise_level']<20),axis=0))[0]
        
            #Neural data:
            X           = ses.respmat[np.ix_(idx_areax,idx_T)].T
            Y           = ses.respmat[np.ix_(idx_areay,idx_T)].T

            if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons:
                R2_cv[iax,iay,ises],optim_rank[iax,iay,ises],R2_cv_folds[iax,iay,ises,:,:,:]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% 
tempR2 = copy.deepcopy(R2_cv)
temprank = copy.deepcopy(optim_rank)

#Filter only sessions in which all combinations are present:
for ises in range(nSessions):
    tempR2[:,:,ises][np.eye(nareas,dtype=bool)] = 1
idx_ses = np.any(np.isnan(tempR2),axis=(0,1))
tempR2[:,:,idx_ses] = np.nan
temprank[:,:,idx_ses] = np.nan

fig,axes = plt.subplots(1,2,figsize=(6,2.7),sharey=False,sharex=False)
sns.heatmap(np.nanmean(tempR2,axis=2),annot=True,ax=axes[0],cmap='magma',vmin=0,vmax=0.2,
            xticklabels=areas,yticklabels=areas)
axes[0].set_title('Avg R2')
sns.heatmap(np.nanmean(temprank,axis=2),annot=True,ax=axes[1],cmap='magma',vmin=5,vmax=11,
            xticklabels=areas,yticklabels=areas)
axes[1].set_title('Avg optimal rank')
axes[0].set_xlabel('Target area')
axes[0].set_ylabel('Source area')
axes[1].set_xlabel('Target area')
axes[1].set_ylabel('Source area')
fig.tight_layout()
my_savefig(fig,savedir,'RRR_R2_rank_allareas_%d_onlyall' % nSessions, formats=['png'])
# my_savefig(fig,savedir,'RRR_R2_rank_allareas_%d' % nSessions, formats=['png'])



#%% 
######  ######  ######     ######  #######  #####  ####### #     # ######  
#     # #     # #     #    #     # #       #     # #     # ##   ## #     # 
#     # #     # #     #    #     # #       #       #     # # # # # #     # 
######  ######  ######     #     # #####   #       #     # #  #  # ######  
#   #   #   #   #   #      #     # #       #       #     # #     # #       
#    #  #    #  #    #     #     # #       #     # #     # #     # #       
#     # #     # #     #    ######  #######  #####  ####### #     # #       

#%% 


#%% Compare mean response and taking the residuals:
nsampleneurons  = 100
nranks          = 25
nmodelfits      = 1 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions,2,2),np.nan)
optim_rank      = np.full((nSessions,2,2),np.nan)
R2_ranks        = np.full((nSessions,2,2,nranks,nmodelfits,kfold),np.nan)
lam             = 500

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    for imethod in range(2):
        if imethod==0:
            data = ses.respmat
        else:
            if ses.sessiondata['protocol'][0]=='GR':
                mean_resp,data = mean_resp_gr(ses,trialfilter=None)
                data = np.tile(mean_resp,(1,200))
            elif ses.sessiondata['protocol'][0]=='GN':
                mean_resp,data = mean_resp_gn(ses,trialfilter=None)
                data = np.tile(mean_resp.reshape(np.shape(mean_resp)[0],-1),(1,300))

        # idx_T               = ses.trialdata['stimCond']==0
        idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
        idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                                ses.celldata['noise_level']<20),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                                ses.celldata['noise_level']<20),axis=0))[0]
        
        X                   = data[np.ix_(idx_areax,idx_T)].T
        Y                   = data[np.ix_(idx_areay,idx_T)].T
        
        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue
        R2_cv[ises,0,imethod],optim_rank[ises,0,imethod],R2_ranks[ises,0,imethod,:,:,:]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

        R2_cv[ises,1,imethod],optim_rank[ises,1,imethod],R2_ranks[ises,1,imethod,:,:,:]      = RRR_wrapper(X, Y, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot the performance across sessions as a function of rank: 

clrs_areapairs = get_clr_area_pairs(['V1-PM','PM-V1'])

fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharey=True,sharex=True)

for iarea in range(2):
    ax = axes[iarea]
    handles = []
    for imethod in range(2):
        datatoplot = np.nanmean(R2_ranks[:,iarea,imethod,:,:,:],axis=(-1,-2))
        handles.append(shaded_error(np.arange(nranks),datatoplot,color=clrs_areapairs[iarea],error='sem',ax=ax,linestyle=['-','--'][imethod]))
    ax.set_xlabel('Rank')
    ax.set_ylabel('R2')
    ax.set_xlim([0,nranks])
    # ax.set_ylim([0,0.25])
    ax.set_title(['V1-PM','PM-V1'][iarea])
    ax_nticks(ax,3)
    ax.legend(handles=handles,labels=['Original','Residual'],frameon=False,fontsize=8,loc='lower right')
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
# my_savefig(fig,savedir,'RRR_V1PM_ranks_mean_vs_residual')

#%% RRR decomposition of variance:

# arealabelpairs  = ['V1unl-V1unl',
#                     'V1unl-V1lab',
#                     'V1lab-V1lab',
#                     'PMunl-PMunl',
#                     'PMunl-PMlab',
#                     'PMlab-PMlab',
#                     'V1unl-PMunl',
#                     'V1unl-PMlab',
#                     'V1lab-PMunl',
#                     'V1lab-PMlab',
#                     'PMunl-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1unl',
#                     'PMlab-V1lab']

arealabelpairs  = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1unl',
                    'PMlab-V1lab']
filter_nearby       = True

arealabelpairs  = ['V1unl-PMunl',
                    'PMunl-V1unl']
filter_nearby       = False

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nsampleneurons      = 250
nranks              = 25
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv_folds         = np.full((narealabelpairs,nSessions,4,nranks,nmodelfits,kfold),np.nan)
# optim_rank          = np.full((narealabelpairs,2,nSessions),np.nan)


for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    #Stimulus data:
    S           = ses.trialdata['stimCond'][idx_T].to_numpy()

    #Behavioral data:
    B           = np.stack((ses.respmat_videome,
                    ses.respmat_runspeed,
                    ses.respmat_pupilarea,
                    ses.respmat_pupilx,
                    ses.respmat_pupily),axis=1)
    B           = np.column_stack((B,ses.respmat_videopc.T))
    B           = zscore(B[idx_T,:],axis=0,nan_policy='omit')

    si          = SimpleImputer()
    B           = si.fit_transform(B)
    
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
    
        #Neural data:
        X           = ses.respmat[np.ix_(idx_areax,idx_T)].T
        Y           = ses.respmat[np.ix_(idx_areay,idx_T)].T

        if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons:
            R2_cv_folds[iapl,ises,:,:,:,:]  = RRR_decompose(Y, X, B, S,nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% How much variance of each dimension is explained by each subspace?
rankdata    = np.nanmean(R2_cv_folds[:,:,0,:,:,:],axis=(0,1))
# _,optimrank =  rank_from_R2(rankdata.reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)
# optimrank = 12
optimrank = np.argmax(np.nanmean(R2_cv_folds[:,:,0,:,:,:],axis=(0,1,3,4)))-1

datatoplot = np.nanmean(R2_cv_folds,axis=(0,4,5))
labelnames = ['Full','Stimulus','Behavior','Unknown']
fig, axes = plt.subplots(1,2,figsize=(7,3))

clrs_decomp = sns.color_palette('tab10',n_colors=4)
ax = axes[0]
handles = []
for i in range(4):
    # ax.plot(datatoplot[i,:])
    handles.append(shaded_error(np.arange(nranks),datatoplot[:,i,:],ax=ax,linestyle='-',color=clrs_decomp[i],error='sem'))
    # ax.plot(np.arange(1,nranks),datatoplot[i,1:])
ax.legend(handles=handles,labels=labelnames,frameon=False,fontsize=8)
ax.axvline(x=optimrank,color = 'gray', linestyle = '--')
ax.set_ylabel('Fraction of variance explained')
ax.set_xlabel('Rank')
ax.set_xticks(np.arange(0,nranks+1,5))

datatoplot = np.nanmean(R2_cv_folds,axis=(0,4,5))
datatoplot = np.diff(datatoplot,axis=2)
datatoplot = datatoplot[:,1:,:] / datatoplot[:,0,:][:,np.newaxis,:]

datatoplot = np.nanmean(R2_cv_folds,axis=(0,1,4,5))
datatoplot = np.diff(datatoplot,axis=1)
datatoplot = datatoplot[1:,:] / datatoplot[0,:]

ax = axes[1]
handles = []
for i in range(3):
    # handles.append(shaded_error(np.arange(1,nranks),datatoplot[:,i,:],ax=ax,linestyle='-',color=clrs_decomp[i+1],
                #  error='sem'))
    ax.plot(np.arange(1,nranks),datatoplot[i,:],linewidth=2,color=clrs_decomp[i+1])
    # ax.plot(datatoplot[i,:])
ax.legend(labels=labelnames[1:],frameon=False,fontsize=8)

ax.axvline(x=optimrank,color = 'gray', linestyle = '--')
ax.add_patch(Rectangle((optimrank,0),nranks-optimrank,1,linewidth=0,edgecolor=None,facecolor='gray',alpha=0.5))
ax.set_ylabel('Fraction of explainable variance')
ax.set_xlabel('Rank')
ax.set_xticks(np.arange(0,nranks+1,5))
ax.set_ylim([0,1])
plt.tight_layout()
sns.despine(top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_V1PM_decomposition_%dneurons' % nsampleneurons,formats=['png'])


#%% 


#%% How much variance of each dimension is explained by each subspace?
rankdata    = np.nanmean(R2_cv_folds[:,:,0,:,:,:],axis=(0,1))
# _,optimrank =  rank_from_R2(rankdata.reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)
# optimrank = 12
optimrank   = np.argmax(np.nanmean(R2_cv_folds[:,:,0,:,:,:],axis=(0,1,3,4)))-1
optimrank   = 12

datatoplot  = np.nanmean(R2_cv_folds[:,:,:,optimrank,:,:],axis=(-1,-2))
labelnames  = ['Full','Stimulus','Behavior','Unknown']

fig, axes   = plt.subplots(1,2,figsize=(narealabelpairs*1+3,3.5),sharex=True)

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
ax = axes[0]
handles = []
for i in range(4):
    # ax.plot(datatoplot[i,:])
    handles.append(ax.plot(range(narealabelpairs),np.nanmean(datatoplot[:,:,i],axis=1),marker='o',linestyle='-',
    color=clrs_decomp[i])[0])
    for iapl in range(narealabelpairs):
        ax.scatter(np.ones(nSessions)*iapl-0.1+np.random.rand(nSessions)*0.03,datatoplot[iapl,:,i],marker='o',s=3,
                #    color=clrs_arealabelpairs[iapl])
                   color=clrs_decomp[i])
ax.legend(handles=handles,labels=labelnames,bbox_to_anchor=(1.05, 1),loc='upper left',frameon=False,fontsize=8)
# ax.legend(handles=handles,labels=labelnames,frameon=False,fontsize=8)
ax.set_ylabel('R2 (decomposed)')
ax.set_xlabel('Population Pair')
ax.set_ylim([0,my_ceil(np.nanmax(datatoplot),1)])
ax_nticks(ax,5)
# ax.set_xticks(np.arange(0,nranks+1,5))

datatoplot = np.nanmean(R2_cv_folds[:,:,:,optimrank,:,:],axis=(3,4))
datatoplot = datatoplot[:,:,1:] / datatoplot[:,:,0][:,:,np.newaxis]

pairs = list(itertools.combinations(arealabelpairs,2))

ax = axes[1]
handles = []
for i in range(3):
    # handles.append(shaded_error(np.arange(1,nranks),datatoplot[:,i,:],ax=ax,linestyle='-',color=clrs_decomp[i+1],
                #  error='sem'))
    handles.append(ax.plot(range(narealabelpairs),np.nanmean(datatoplot[:,:,i],axis=1),marker='o',linestyle='-',
    color=clrs_decomp[i+1])[0])

    for iapl in range(narealabelpairs):
        ax.scatter(np.ones(nSessions)*iapl-0.1,datatoplot[iapl,:,i],marker='o',s=3,
                #    color=clrs_arealabelpairs[iapl])
                   color=clrs_decomp[i+1])

    if narealabelpairs == 2:
        ttest,pval = stats.ttest_rel(datatoplot[0,:,i],datatoplot[1,:,i],nan_policy='omit')

        add_stat_annotation(ax, 0.1, 0.9, np.nanmean(datatoplot[0,:,i],axis=0)+0.05, pval, h=0.0, color=clrs_decomp[i+1])
    else:
        if i == 2: 
            datatotest = datatoplot[:,~np.any(np.isnan(datatoplot),axis=(0,2)),i]

            df = pd.DataFrame({'R2':  datatotest.flatten(),
                                'arealabelpair':np.repeat(arealabelpairs,np.shape(datatotest)[1])})

            annotator = Annotator(ax, pairs, data=df, x='arealabelpair', y='R2', order=arealabelpairs)
            annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False,
                                comparisons_correction=None,alpha=0.05,
                                # hide_non_significant=True,
                                pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"]]
                                )
            # annotator.apply_and_annotate()

# handles = ax.lines
# ax.legend(handles=handles,labels=labelnames[1:],bbox_to_anchor=(1.05, 1),loc='upper left',frameon=False,fontsize=8)

ax.set_ylim([-0.05,0.8])
ax.set_ylabel('Frac. explainable variance')
sns.despine(top=True,right=True,offset=3)
axes[0].set_xticks(range(narealabelpairs),arealabelpairs,rotation=45,ha='right',fontsize=7)
axes[1].set_xticks(range(narealabelpairs),arealabelpairs,rotation=45,ha='right',fontsize=7)
plt.tight_layout()
my_savefig(fig,savedir,'RRR_V1PM_decomposition_%dneurons' % nsampleneurons,formats=['png'])



#%% 

data                = ses.respmat

# idx_T               = ses.trialdata['stimCond']==0
idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                        ses.celldata['noise_level']<20),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                        ses.celldata['noise_level']<20),axis=0))[0]

X                   = data[np.ix_(idx_areax,idx_T)].T
Y                   = data[np.ix_(idx_areay,idx_T)].T

Y = zscore(Y,axis=0)
X = zscore(X,axis=0)

stim = ses.trialdata['stimCond'][idx_T].to_numpy()
# Data format: 
K,N     = np.shape(X)
M            = np.shape(Y)[1]

nmodelfits = 1

nN = 250
nM = 250
r = 25

idx_areax_sub           = np.random.choice(N,nN,replace=False)
idx_areay_sub           = np.random.choice(M,nM,replace=False)

X                   = X[:,idx_areax_sub]
Y                   = Y[:,idx_areay_sub]

S           = np.stack((ses.respmat_videome[idx_T],
                ses.respmat_runspeed[idx_T],
                ses.respmat_pupilarea[idx_T],
                ses.respmat_pupilx[idx_T],
                ses.respmat_pupily[idx_T]),axis=1)
S           = np.column_stack((S,ses.respmat_videopc[:,idx_T].T))
S           = zscore(S,axis=0,nan_policy='omit')

si          = SimpleImputer()
S           = si.fit_transform(S)

U_stim,pca          = compute_stim_subspace(Y, stim, n_components=5)

U_behav             = compute_behavior_subspace_linear(Y, S, n_components=5)

U_stim_orth, U_behav_orth = orthogonalize_subspaces(U_stim.T, U_behav.T)
U_stim_orth, U_behav_orth = U_stim_orth.T, U_behav_orth.T

#RRR of area 1 onto 2:
B_hat_train         = LM(Y,X, lam=lam)

Y_hat_train         = X @ B_hat_train

U, s, V = svds(Y_hat_train,k=np.min((nranks,nN,nM))-1,which='LM')
U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

S = linalg.diagsvd(s,U.shape[0],s.shape[0])

cvR2_folds = np.full((4,nranks),np.nan)

for r in range(nranks):

    B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
    
    Y_hat               = X @ B_rrr #project test data onto low rank predictive subspace
    
    cvR2_folds[0,r]     = EV(Y,Y_hat)

    # Y_hat_stim          = project_onto_subspace(Y_hat, U_stim)
    Y_hat_stim          = project_onto_subspace(Y_hat, U_stim_orth)
    cvR2_folds[1,r]     = EV(Y,Y_hat_stim)

    # print(f"Fraction of predicted variance that is stimulus-related: {ev_stim / ev_total:.3f}")

    Y_hat_behav         = project_onto_subspace(Y_hat, U_behav)
    # Y_hat_behav         = project_onto_subspace(Y_hat, U_behav_orth)
    cvR2_folds[2,r]     = EV(Y,Y_hat_behav)

    # print(f"Fraction of predicted variance that is behavior-related: {ev_behav / ev_total:.3f}")

cvR2_folds[3,:]     = cvR2_folds[0,:] - cvR2_folds[1,:] - cvR2_folds[2,:]



#%%

overlap = compute_subspace_overlap(U_stim, U_behav)

print(f"Mean cosine of principal angles: {overlap['mean_cosine']:.3f}")
print(f"Subspace overlap (squared sum of cosines): {overlap['squared_overlap']:.3f}")
print(f"All singular values (cosines): {overlap['cosines']}")


#%% 

U_stim_orth, U_behav_orth = orthogonalize_subspaces(U_stim.T, U_behav.T)
U_stim_orth, U_behav_orth = U_stim_orth.T, U_behav_orth.T

overlap = compute_subspace_overlap(U_stim_orth, U_behav_orth)

print(f"Mean cosine of principal angles: {overlap['mean_cosine']:.3f}")
print(f"Subspace overlap (squared sum of cosines): {overlap['squared_overlap']:.3f}")
print(f"All singular values (cosines): {overlap['cosines']}")

#%% 
# Given: stim_basis, behav_basis (orthonormal subspace bases)
# U_stim_orth, U_behav_orth = orthogonalize_subspaces(U_stim, U_behav)

# Example usage
Y_hat_stim          = project_onto_subspace(Y_hat, U_stim_orth.T)
ev_stim_unique      = EV(Y,Y_hat_stim)

# fraction_stim_in_Yhat       = compute_fraction_variance_in_subspace(Y_hat, U_stim)

print(f"Fraction of unique predicted variance that is stimulus-related: {ev_stim_unique / ev_total:.3f}")
# print(f"Fraction of predicted variance that is stimulus-related: {ev_stim:.3f}")

Y_hat_behav         = project_onto_subspace(Y_hat, U_behav)
ev_behav_unique     = EV(Y,Y_hat_behav)

print(f"Fraction of unique predicted variance that is behavior-related: {ev_behav_unique / ev_total:.3f}")

#%% the order matters: 

U_stim_orth1, U_behav_orth2 = orthogonalize_subspaces(U_stim.T, U_behav.T)
U_stim_orth1, U_behav_orth2 = U_stim_orth1.T, U_behav_orth2.T

U_stim_orth2, U_behav_orth1 = orthogonalize_subspaces(U_behav.T,U_stim.T)
U_stim_orth2, U_behav_orth1 = U_stim_orth2.T, U_behav_orth1.T

r = 24

B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace

Y_hat               = X @ B_rrr #project test data onto low rank predictive subspace

cvR2_folds[0,r]     = EV(Y,Y_hat)

Y_hat_stim          = project_onto_subspace(Y_hat, U_stim_orth1)
print('ev_stim, order 1: %.3f' % EV(Y,Y_hat_stim))

Y_hat_stim          = project_onto_subspace(Y_hat, U_stim_orth2)
print('ev_stim, order 2: %.3f' % EV(Y,Y_hat_stim))

Y_hat_behav          = project_onto_subspace(Y_hat, U_behav_orth1)
print('ev_behav, order 1: %.3f' % EV(Y,Y_hat_behav))

Y_hat_behav          = project_onto_subspace(Y_hat, U_behav_orth2)
print('ev_behav, order 2: %.3f' % EV(Y,Y_hat_behav))


#%% 


























#%% 

######  #######  #####  ######  #######  #####   #####     ######  ####### #     #    #    #     # 
#     # #       #     # #     # #       #     # #     #    #     # #       #     #   # #   #     # 
#     # #       #       #     # #       #       #          #     # #       #     #  #   #  #     # 
######  #####   #  #### ######  #####    #####   #####     ######  #####   ####### #     # #     # 
#   #   #       #     # #   #   #             #       #    #     # #       #     # #######  #   #  
#    #  #       #     # #    #  #       #     # #     #    #     # #       #     # #     #   # #   
#     # #######  #####  #     # #######  #####   #####     ######  ####### #     # #     #    #    



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
my_savefig(fig,savedir,'BehaviorRegressedOut_V1PM_%s' % ses.sessiondata['session_id'][0],formats=['png'])

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
my_savefig(fig,savedir,'BehaviorRegressedOut_V1PM_%dsessions' % nSessions,formats=['png'])

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

# my_savefig(fig,savedir,'RRR_cvR2_RegressOutBehavior_V1PM_LabUnl_%dsessions' % nSessions)


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
my_savefig(fig,savedir,'AL_RSP_RegressedOut_V1PM_%s' % ses.sessiondata['session_id'][0],formats=['png'])

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
my_savefig(fig,savedir,'BehaviorRegressedOut_V1PM_%dsessions' % nSessions,formats=['png'])




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
my_savefig(fig,savedir,'RRR_V1PM_regressoutneuralALRSP_%dsessions' % (nSessions))

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

my_savefig(fig,savedir,'RRR_V1PM_regressoutneural_Frac_var_shared_ALRSP_%dsessions' % (nSessions))


