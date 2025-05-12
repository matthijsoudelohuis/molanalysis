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

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\CCA\\')

#%% 
session_list        = np.array([['LPE12223','2024_06_10'], #GR
                                ['LPE10919','2023_11_06']]) #GR
# session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                # ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%%  Load data properly:        
calciumversion = 'dF'
calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=True)
    
#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% 
 #####   #####     #    
#     # #     #   # #   
#       #        #   #  
#       #       #     # 
#       #       ####### 
#     # #     # #     # 
 #####   #####  #     # 

oris         = np.sort(sessions[0].trialdata['Orientation'].unique())
nOris = 16


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









#%% 

areas = ['V1','PM','AL','RSP']
nareas = len(areas)


areas = ['V1','PM']
nareas = len(areas)



# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,filter_areas=areas)
# sessions,nSessions   = filter_sessions(protocols = 'GN',only_all_areas=areas,filter_areas=areas)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_areas=areas)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

#%% 





#%% 
# To improve: run multiple iterations, subsampling different neurons / trials
# show errorbars across sessions
# take pop rate and pc1 of the subsample??
# Does this depend on the variability in population activity? What if only still trials are subsampled?

oris                = np.sort(sessions[ises].trialdata['Orientation'].unique())
nOris               = len(oris)

nSessions           = len(sessions)

CC1_to_var_labels       = ['test trials','pop. rate','global PC1','locomotion','videoME']

nvars                    = len(CC1_to_var_labels)
corr_CC1_vars            = np.full((nareas,nareas,nOris,2,nvars,nSessions),np.nan)

areapairmat = np.empty((nareas,nareas),dtype='object')
for ix,areax in enumerate(areas):
    for iy,areay in enumerate(areas):
        areapairmat[ix,iy] = areax + '-' + areay
nareapairs = nareas**2

Nsub        = 250   #how many neurons to subsample from each area
prePCA      = 25    #perform dim reduc before fitting CCA, otherwise overfitting

model_CCA = CCA(n_components = 1,scale = False, max_iter = 1000)
model_PCA = PCA(n_components = 1)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA dim1'):    # iterate over sessions
    zmat            = zscore(ses.respmat.T,axis=0)
    poprate         = np.nanmean(zmat,axis=1)

    gPC1            = model_PCA.fit_transform(zmat).squeeze()

    for iori,ori in enumerate(oris): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori
        for ix,areax in enumerate(areas):
            for iy,areay in enumerate(areas):

                idx_areax           = np.where(ses.celldata['roi_name']==areax)[0]
                idx_areay           = np.where(ses.celldata['roi_name']==areay)[0]

                # N1                  = np.sum(idx_areax)
                # N2                  = np.sum(idx_areay)

                if len(idx_areax)>Nsub*2 and len(idx_areay)>Nsub*2:
                    idx_areax_sub       = np.random.choice(idx_areax,np.min((len(idx_areax),Nsub)),replace=False)
                    idx_areay_sub       = np.random.choice(idx_areay[~np.isin(idx_areay,idx_areax_sub)],np.min((len(idx_areay),Nsub)),replace=False)

                    X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
                    Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
                    
                    X                   = zscore(X,axis=0)  #Z score activity for each neuron
                    Y                   = zscore(Y,axis=0)

                    if prePCA and Nsub>prePCA:
                        prepca      = PCA(n_components=prePCA)
                        X           = prepca.fit_transform(X)
                        Y           = prepca.fit_transform(Y)
                        
                    # Compute and store canonical correlations for the first pair
                    X_c, Y_c = model_CCA.fit_transform(X,Y)

                    # corr_CC1[ix,iy,iori,0,ises] = np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]
                    [corr_CC1_vars[ix,iy,iori,0,0,ises],_] = CCA_subsample_1dim(X.T,Y.T,resamples=5,kFold=5,prePCA=None)

                    corr_CC1_vars[ix,iy,iori,0,1,ises] = np.corrcoef(X_c[:,0],poprate[idx_T])[0,1]
                    corr_CC1_vars[ix,iy,iori,1,1,ises] = np.corrcoef(Y_c[:,0],poprate[idx_T])[0,1]

                    corr_CC1_vars[ix,iy,iori,0,2,ises] = np.corrcoef(X_c[:,0],gPC1[idx_T])[0,1]
                    corr_CC1_vars[ix,iy,iori,1,2,ises] = np.corrcoef(Y_c[:,0],gPC1[idx_T])[0,1]

                    corr_CC1_vars[ix,iy,iori,0,3,ises] = np.corrcoef(X_c[:,0],ses.respmat_runspeed[idx_T])[0,1]
                    corr_CC1_vars[ix,iy,iori,1,3,ises] = np.corrcoef(Y_c[:,0],ses.respmat_runspeed[idx_T])[0,1]

                    corr_CC1_vars[ix,iy,iori,0,4,ises] = np.corrcoef(X_c[:,0],ses.respmat_videome[idx_T])[0,1]
                    corr_CC1_vars[ix,iy,iori,1,4,ises] = np.corrcoef(Y_c[:,0],ses.respmat_videome[idx_T])[0,1]

#%% 
fig, axes = plt.subplots(1,nvars,figsize=(nvars*3,3),sharex=True,sharey=True)
for ivar,var in enumerate(CC1_to_var_labels):
    ax = axes[ivar]
    data = np.nanmean(corr_CC1_vars[:,:,:,:,ivar,:],axis=(2,3,4))

    ax.imshow(data,cmap='bwr',clim=(-1,1))
    ax.set_xticks(np.arange(0,nareas))
    ax.set_xticklabels(areas)
    ax.set_yticks(np.arange(0,nareas))
    ax.set_yticklabels(areas)
    ax.set_title(var)
cbar = add_colorbar_outside(ax.images[0],ax)
cbar.set_label('Correlation', rotation=90, labelpad=-40)
fig.suptitle('CC1 correlation to:')
# plt.tight_layout()

# plt.savefig(os.path.join(savedir,'CC1_CorrVars_Heatmap_V1PM_%dsessions.png' % nSessions),
            #  bbox_inches='tight',  format = 'png')
my_savefig(fig,savedir,'CC1_CorrVars_Heatmap_V1PM_%dsessions.png' % nSessions,formats=['png'])

#%% 
colorvars                = sns.color_palette("husl",nvars)

fig, axes = plt.subplots(1,1,figsize=(4,4))
ax       = axes
handles  = []
data     = np.nanmean(corr_CC1_vars,axis=(2,3,5))

for ivar,var in enumerate(CC1_to_var_labels):
    data            = np.nanmean(corr_CC1_vars[:,:,:,:,ivar,:],axis=(2,3))
    meantoplot      = np.nanmean(data,axis=2)
    errortoplot     = np.nanstd(data,axis=2) / np.sqrt(nSessions)
    
    # ax.plot(np.arange(nareapairs),meantoplot.flatten(),color=colorvars[ivar],linewidth=1.5)
    # data = np.nanmean(corr_CC1_vars[:,:,:,:,ivar,:],axis=(2,3,4))
    handles.append(shaded_error(np.arange(nareapairs),meantoplot.flatten(),yerror=errortoplot.flatten(),color=colorvars[ivar],alpha=0.25,ax=ax))

ax.set_xticks(np.arange(nareapairs),areapairmat.flatten())
ax.set_ylim([0,1])
ax.set_yticks([0,0.25,0.5,0.75,1])
ax.legend(handles,CC1_to_var_labels,frameon=False,fontsize=10,loc='lower left')
ax.set_title('CC1 correlation to:')
ax.set_xlabel('Area pairs')
ax.set_ylabel('CC1 correlation with\n variable of interest')
# plt.tight_layout()
plt.savefig(os.path.join(savedir,'CC1_CorrVars_Lineplot_V1PM_%dsessions.png' % nSessions),
             bbox_inches='tight',  format = 'png')


#%% 
oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
nOris = 16
Nsub = 50

areax = 'V1'
areay = 'PM'

corr_popcoupling = np.zeros((nSessions,nOris,2))
prePCA = 25
model_CCA = CCA(n_components=2,scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA dim1'):
    zmat            = zscore(ses.respmat.T,axis=0)
    poprate         = np.nanmean(zmat,axis=1)
    N = np.shape(sessions[ises].respmat)[0]
    popcorr = np.zeros(N)

    for iN in range(N):
        popcorr[iN] = np.corrcoef(zmat[:,iN],poprate)[0,1]

    idx_areax           = np.where(sessions[ises].celldata['roi_name']==areax)[0]
    idx_areay           = np.where(sessions[ises].celldata['roi_name']==areay)[0]

    for i,ori in enumerate(oris): # loop over orientations 
        idx_T               = sessions[ises].trialdata['Orientation']==ori
        # idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
        if len(idx_areax)>Nsub and len(idx_areay)>Nsub:
            idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
            idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)

            X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
            Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
            
            X                   = zscore(X,axis=0)  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0)

            if prePCA and Nsub>prePCA:
                # prepca      = PCA(n_components=prePCA)
                # X           = prepca.fit_transform(X)
                # Y           = prepca.fit_transform(Y)
            
                prepca      = PCA(n_components=prePCA)
                X           = prepca.fit_transform(X)
                X           = prepca.inverse_transform(X)
                
                Y           = prepca.fit_transform(Y)
                Y           = prepca.inverse_transform(Y)

            # Compute and store canonical correlations for the first pair
            X_c, Y_c = model_CCA.fit_transform(X,Y)

            corr_popcoupling[ises,i,0] = np.corrcoef(model_CCA.x_weights_[:,0],popcorr[idx_areax_sub])[0,1]
            corr_popcoupling[ises,i,1] = np.corrcoef(model_CCA.y_weights_[:,0],popcorr[idx_areay_sub])[0,1]

#%% 
clrs_areas = get_clr_areas(['V1','PM'])

corr_popcoupling = np.abs(corr_popcoupling)

fig, axes = plt.subplots(1,1,figsize=(1,3))
ax       = axes
for iarea,area in enumerate(['V1','PM']):
    ax.errorbar(iarea,np.nanmean(corr_popcoupling[:,:,iarea],axis=(0,1)),np.nanstd(corr_popcoupling[:,:,iarea],axis=(0,1))/np.sqrt(nSessions),
                color=clrs_areas[iarea],marker='o',zorder=10)
ax.set_xticks([0,1])
ax.set_xticklabels(['V1','PM'])
ax.set_ylim([0,1])
ax.set_yticks([0,0.25,0.5,0.75,1])
ax.set_xlabel('Area')
ax.set_ylabel('Correlation CC1 weights\nto pop. correlation')
sns.despine(top=True,right=True,offset=3)
my_savefig(fig,savedir,'CC1_CorrPopCorr_V1PM_%dsessions' % nSessions)

#%% 
oris                = np.sort(sessions[ises].trialdata['Orientation'].unique())
nOris               = len(oris)
nSessions           = len(sessions)

#%%
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
arealabels      = ['V1lab','PMlab']
narealabels     = len(arealabels)
ncells = np.empty((nSessions,narealabels))
for i,ses in enumerate(sessions):
    for ial, arealabel in enumerate(arealabels):
        ncells[i,ial] = np.sum(ses.celldata['arealabel']==arealabel)
# plt.hist(ncells.flatten(),np.arange(0,1100,25))
plt.hist(ncells.flatten(),np.arange(0,220,5))

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)

# clrs_arealabels = get_clr_area_labeled(arealabels)
narealabelpairs = len(arealabelpairs)

nccadims            = 10

kfold               = 5
# lam             = 0.08
# lam             = 1
nmodelfits          = 5
filter_nearby       = True

CCA_corrtest            = np.full((narealabelpairs,nccadims,nOris,nSessions,nmodelfits),np.nan)

model_CCA               = CCA(n_components = nccadims,scale = False, max_iter = 1000)

prePCA                  = 25 #perform dim reduc before fitting CCA, otherwise overfitting

nminneurons             = 15 #how many neurons in a population to include the session
nsampleneurons          = 15

regress_out_behavior = True

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA dim1'):    # iterate over sessions
    # get signal correlations:
    [N,K]           = np.shape(ses.respmat) #get dimensions of response matrix
    for iori,ori in enumerate(oris): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori
        for iapl, arealabelpair in enumerate(arealabelpairs):
            
            alx,aly = arealabelpair.split('-')

            if filter_nearby:
                idx_nearby  = filter_nearlabeled(ses,radius=50)
            else:
                idx_nearby = np.ones(len(ses.celldata),dtype=bool)

            # idx_N       = np.all((ses.celldata['arealabel']==arealabel,
            #                         ses.celldata['noise_level']<100,	
            #                         idx_nearby),axis=0) 
            idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                    ses.celldata['noise_level']<100,	
                                    idx_nearby),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                    ses.celldata['noise_level']<100,	
                                    idx_nearby),axis=0))[0]
            # idx_areay           = np.where(ses.celldata['arealabel']==aly)[0]

            if len(idx_areax)>nminneurons and len(idx_areay)>nminneurons:
                for i in range(nmodelfits):

                    idx_areax_sub       = np.random.choice(idx_areax,np.min((np.sum(idx_areax),nsampleneurons)),replace=False)
                    idx_areay_sub       = np.random.choice(idx_areay[~np.isin(idx_areay,idx_areax_sub)],np.min((np.sum(idx_areay),nsampleneurons)),replace=False)

                    X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
                    Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
                    
                    X   = zscore(X,axis=0)  #Z score activity for each neuron
                    Y   = zscore(Y,axis=0)

                    if prePCA and nsampleneurons>prePCA:
                        prepca      = PCA(n_components=prePCA)
                        X           = prepca.fit_transform(X)
                        Y           = prepca.fit_transform(Y)

                    # X = my_shuffle(X,method='random')
                    B = np.stack((sessions[ises].respmat_videome[idx_T],
                                      sessions[ises].respmat_runspeed[idx_T],
                                      sessions[ises].respmat_pupilarea[idx_T]),axis=1)

                    if regress_out_behavior:
                        X   = regress_out_behavior_modulation(sessions[ises],B,X,rank=3,lam=0)
                        Y   = regress_out_behavior_modulation(sessions[ises],B,Y,rank=3,lam=0)
                        # Y   = regress_out_behavior_modulation(Y,sessions[ises].trialdata[idx_T],sessions[ises].trialdata['Orientation']==ori)

                    [CCA_corrtest[iapl,:,iori,ises,i],_] = CCA_subsample(X.T,Y.T,resamples=5,kFold=5,prePCA=None,n_components=nccadims)

#%%

fig, axes = plt.subplots(1,1,figsize=(4,4))

ax = axes

for iapl, arealabelpair in enumerate(arealabelpairs):
    ax.plot(np.arange(nccadims),np.nanmean(CCA_corrtest[iapl,:,:,:,:],axis=(1,2,3)),
            color=clrs_arealabelpairs[iapl],linewidth=2)
# plt.plot(np.arange(nccadims),np.nanmean(CCA_corrtest[:,:,0,:,0],axis=(0,1,2)),color='k',linewidth=2)
ax.set_xticks(np.arange(nccadims))
ax.set_xticklabels(np.arange(nccadims)+1)
ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest,axis=(2,3,4))),1)])
# ax.set_yticks([0,ax.get_ylim()[1]])
ax.set_yticks([0,ax.get_ylim()[1]/2,ax.get_ylim()[1]])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel('Correlation')
ax.legend(arealabelpairs,loc='upper right',frameon=False,fontsize=9)
sns.despine(top=True,right=True,offset=3)
# plt.savefig(os.path.join(savedir,'CCA_cvShuffleTestCorr_Dim_V1PM_LabUnl_%dsessions.png' % nSessions), format = 'png')
# plt.savefig(os.path.join(savedir,'CCA_cvTestCorr_Dim_V1PM_LabUnl_%dsessions.png' % nSessions), 
            # format = 'png', bbox_inches='tight')

#%% Make one for different number of neurons per population:

#%% 

#To improve: run multiple iterations, subsampling different neurons / trials
# show errorbars across sessions

#%% RUN CCA for different population sizes: 
prePCA              = 25    #perform dim reduc before fitting CCA, otherwise overfitting
ndims               = 200

popsizes            = np.array([5,10,20,50,100,200,500])

corr_CC1_vars            = np.full((nOris,ndims,len(popsizes),nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA for different population sizes'):    # iterate over sessions
    # get signal correlations:
    [N,K]           = np.shape(ses.respmat) #get dimensions of response matrix

    for iori,ori in enumerate(oris): # loop over orientations 
    # for iori,ori in enumerate(oris[:2]): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori
        idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                                                ses.celldata['noise_level']<20),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                                                ses.celldata['noise_level']<20),axis=0))[0]
        
        X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T 
        
        for ipopsize,popsize in enumerate(popsizes):

            if len(idx_areax)>popsize and len(idx_areay)>popsize:
                # [corr_CC1_vars[ix,iy,iori,0,0,ises],_] = CCA_subsample_1dim(X.T,Y.T,resamples=5,kFold=5,prePCA=None)
                [temp,_] = CCA_subsample(X.T,Y.T,nN=popsize,nK=np.sum(idx_T),resamples=5,kFold=5,prePCA=prePCA,n_components=popsize)
                corr_CC1_vars[iori,:np.min((prePCA,popsize)),ipopsize,ises] = temp

#%% Plot:
clrs_popsizes = sns.color_palette("rocket",len(popsizes))

fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ax = axes[0]
for ipopsize,popsize in enumerate(popsizes):
    tempdata = np.nanmean(corr_CC1_vars[:,:,ipopsize,:],axis=(0))
    ax.plot(np.arange(0,ndims)+1,np.nanmean(tempdata,axis=1),color=clrs_popsizes[ipopsize],label='%d' % popsize)
    # plt.plot(range(1,np.min((prePCA,popsize))+1),np.nanmean(corr_CC1_vars[:,:np.min((prePCA,popsize)),ipopsize,:],axis=(0,3)),color=clrs_popsizes[ipopsize],label='%d' % popsize)
ax.set_xlabel('Number of dimensions')
ax.set_ylabel('Canoncial Correlation \n(cross-validated)')
ax.set_ylim([-0.1,1])
ax.set_title('Cross-validated CCA')
ax.axhline(y=0,color='k',linestyle='--')
ax.legend(title='Population size',loc='best',frameon=False,fontsize=9,ncol=2)

ax = axes[1]
for ipopsize,popsize in enumerate(popsizes):
    tempdata = np.nanmean(corr_CC1_vars[:,:,ipopsize,:],axis=(0))
    tempdata = tempdata / tempdata[0,:]
    ax.plot(np.arange(0,ndims)+1,np.nanmean(tempdata,axis=1),color=clrs_popsizes[ipopsize],label='%d' % popsize)
    # plt.plot(range(1,np.min((prePCA,popsize))+1),np.nanmean(corr_CC1_vars[:,:np.min((prePCA,popsize)),ipopsize,:],axis=(0,3)),color=clrs_popsizes[ipopsize],label='%d' % popsize)
ax.set_xlabel('Number of dimensions')
ax.set_title('Normalized to first dimension')
ax.axhline(y=0,color='k',linestyle='--')

sns.despine(top=True,right=True,offset=3)

fig.tight_layout()
plt.savefig(os.path.join(savedir,'CCA_PopulationSizes_CrossVal_%dsessions.png' % nSessions), format = 'png')

#%% Crossvalidated CCA for different number of population sizes with and without regularization/PCA:


#%% RUN CCA for different population sizes: 
prePCA              = 25    #perform dim reduc before fitting CCA, otherwise overfitting
ndims               = 200
ndims               = 25

popsizes            = np.array([5,10,20,50,100,200,500])

CCA_cvcorr          = np.full((nOris,ndims,len(popsizes),nSessions),np.nan)
CCA_cvcorr_wPCA     = np.full((nOris,ndims,len(popsizes),nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA for different population sizes'):    # iterate over sessions
    # get signal correlations:
    [N,K]           = np.shape(ses.respmat) #get dimensions of response matrix

    for iori,ori in enumerate(oris): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori
        idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                                                ses.celldata['noise_level']<20),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                                                ses.celldata['noise_level']<20),axis=0))[0]
        
        X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T 
        
        for ipopsize,popsize in enumerate(popsizes):
            if len(idx_areax)>popsize and len(idx_areay)>popsize:

                [temp,_] = CCA_subsample(X.T,Y.T,nN=popsize,nK=np.sum(idx_T),resamples=1,kFold=5,prePCA=None,n_components=np.min((ndims,popsize)))
                CCA_cvcorr[iori,:len(temp),ipopsize,ises] = temp

                [temp,_] = CCA_subsample(X.T,Y.T,nN=popsize,nK=np.sum(idx_T),resamples=1,kFold=5,prePCA=prePCA,n_components=popsize)
                CCA_cvcorr_wPCA[iori,:len(temp),ipopsize,ises] = temp

#%% Plot:
clrs_popsizes = sns.color_palette("rocket",len(popsizes))

fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ax = axes[0]
for ipopsize,popsize in enumerate(popsizes):
    tempdata = np.nanmean(CCA_cvcorr[:,:,ipopsize,:],axis=(0))
    ax.plot(np.arange(0,ndims)+1,np.nanmean(tempdata,axis=1),color=clrs_popsizes[ipopsize],label='%d' % popsize)
ax.set_xlabel('Number of dimensions')
ax.set_ylabel('Canoncial Correlation \n(cross-validated)')
ax.set_ylim([-0.1,1])
ax.set_title('Cross-validated CCA')
ax.axhline(y=0,color='k',linestyle='--')
ax.legend(title='Population size',loc='best',frameon=False,fontsize=9,ncol=2)

ax = axes[1]
for ipopsize,popsize in enumerate(popsizes):
    tempdata = np.nanmean(CCA_cvcorr_wPCA[:,:,ipopsize,:],axis=(0))
    ax.plot(np.arange(0,ndims)+1,np.nanmean(tempdata,axis=1),color=clrs_popsizes[ipopsize],label='%d' % popsize)
ax.set_xlabel('Number of dimensions')
ax.set_title('with PCA first')
ax.axhline(y=0,color='k',linestyle='--')

sns.despine(top=True,right=True,offset=3)
plt.savefig(os.path.join(savedir,'CCA_corr_PCARegularization_%dsessions.png' % nSessions), format = 'png')

#%% Run CCA and PCA and estimate the fraction of power that CCA captures for each dimension:
ndims               = 25    #perform dim reduc before fitting CCA, otherwise overfitting

popsize             = 50

CCA_EV              = np.full((nOris,ndims,2,nSessions),np.nan)
PCA_EV              = np.full((nOris,ndims,2,nSessions),np.nan)

pca_X               = PCA(n_components=ndims)
pca_Y               = PCA(n_components=ndims)
model_CCA           = CCA(n_components=ndims,scale = False, max_iter = 1000)
# model_CCA           = CCA(n_components=popsize,scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting PCA/CCA'):    # iterate over sessions
    # get signal correlations:
    [N,K]           = np.shape(ses.respmat) #get dimensions of response matrix

    for iori,ori in enumerate(oris): # loop over orientations 
    # for iori,ori in enumerate(oris[:2]): # loop over orientations 
        idx_T               = ses.trialdata['Orientation']==ori
        idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                                                ses.celldata['noise_level']<20),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                                                ses.celldata['noise_level']<20),axis=0))[0]
        
        if len(idx_areax) > popsize and len(idx_areay) > popsize:
            idx_areax_sub       = np.random.choice(idx_areax,popsize,replace=False)
            idx_areay_sub       = np.random.choice(idx_areay,popsize,replace=False)

            X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
            Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T 

            X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
            Y                   = zscore(Y,axis=0)

            X_pca                       = pca_X.fit_transform(X)
            Y_pca                       = pca_Y.fit_transform(Y)

            model_CCA.fit(X_pca,Y_pca)

            # Variance of each component: 
            PCA_EV[iori,:,0,ises]       = np.array([var_along_dim(X,pca_X.components_[idim,:]) for idim in range(ndims)])
            PCA_EV[iori,:,1,ises]       = np.array([var_along_dim(Y,pca_Y.components_[idim,:]) for idim in range(ndims)])

            var_X_c = np.array([var_along_dim(X_pca,model_CCA.x_weights_[:,idim]) for idim in range(ndims)])
            var_Y_c = np.array([var_along_dim(Y_pca,model_CCA.y_weights_[:,idim]) for idim in range(ndims)])

            CCA_EV[iori,:,0,ises] = var_X_c * np.sum(PCA_EV[iori,:,0,ises])
            CCA_EV[iori,:,1,ises] = var_Y_c * np.sum(PCA_EV[iori,:,1,ises])

            explained_variance_X_pca = np.sum(pca_X.explained_variance_ratio_)
            explained_variance_Y_pca = np.sum(pca_Y.explained_variance_ratio_)

#%% Plot explained variance along dimensions for PCA and CCA:
fig,ax = plt.subplots(1,1,figsize=(3,3))
ax.plot(range(1,ndims+1),np.nanmean(PCA_EV,axis=(0,2)),color='r',linewidth=0.25)
ax.plot(range(1,ndims+1),np.nanmean(CCA_EV,axis=(0,2)),color='g',linewidth=0.25)
ax.plot(range(1,ndims+1),np.nanmean(CCA_EV,axis=(0,2)) / np.nanmean(PCA_EV,axis=(0,2)),
        color='b',linewidth=0.25)

ax.plot(range(1,ndims+1),np.nanmean(PCA_EV,axis=(0,2,3)),color='r',label='PCA',linewidth=2)
ax.plot(range(1,ndims+1),np.nanmean(CCA_EV,axis=(0,2,3)),color='g',label='CCA',linewidth=2)
ax.plot(range(1,ndims+1),np.nanmean(CCA_EV,axis=(0,2,3)) / np.nanmean(PCA_EV,axis=(0,2,3)),
        color='b',label='CCA/PCA',linewidth=2)

ax.set_xlabel('Dimension')
ax.set_ylabel('Explained Variance')
ax.set_ylim([0,1])
ax.legend(loc='best',frameon=False)
sns.despine(top=True,right=True,offset=3)

fig.tight_layout()
plt.savefig(os.path.join(savedir,'PCA_CCA_EV_Ratio_%dsessions.png' % nSessions), format = 'png')

















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








