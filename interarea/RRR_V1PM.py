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

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross')

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

my_savefig(fig,savedir,'RRR_weights_acrossranks_V1PM_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])

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
my_savefig(fig,savedir,'RRR_Meanweights_acrossranks_V1PM_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])


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
my_savefig(fig,savedir,'RRR_SigWeightOverlap_acrossranks_V1PM_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])


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
my_savefig(fig,savedir,'RRR_weights_acrossranks_V1PMlabeled_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])




#%% Do RRR in FF and FB direction and compare performance:
nsampleneurons  = 250
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
ax.set_ylim([0,0.3])
ax.set_yticks([0,0.1,0.2,0.3])
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
    ax.text(0.6,0.2,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='red')
else: 
    ax.text(0.6,0.2,'p=%.3f' % p,transform=ax.transAxes,ha='center',va='center',fontsize=12,color='k')

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
    ax.text(0.6,0.2,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='red')
else: 
    ax.text(0.6,0.2,'p=n.s.',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='k')
    
# ax.set_ylim([0,0.3])
plt.tight_layout()
sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'RRR_R2_acrossranks_V1PM_%dsessions.png' % nSessions,formats=['png'])




#%% Do RRR in FF and FB direction and compare performance:
nsampleneurons  = 250
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
def add_corr_results(ax, x,y):
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    r,p = pearsonr(x[~nas], y[~nas])

    print('Correlation (r=%1.2f): p=%.3f' % (r,p))
    if p<0.05:
        ax.text(0.2,0.1,'r=%1.2f\np<0.05' % r,transform=ax.transAxes,ha='center',va='center',fontsize=8,color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')
    else: 
        ax.text(0.2,0.1,'p=n.s.',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='k')

from scipy.stats import pearsonr

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

my_savefig(fig,savedir,'RRR_Perf_WithinAcross_Dimensionality_%dsessions.png' % nSessions)

#%% Is the difference in feedforward vs feedback (V1-PM vs PM-V1) due to different dimensionality?


















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










#%% Cross area predictions: 

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
