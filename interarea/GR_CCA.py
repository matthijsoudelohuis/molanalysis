# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
from loaddata.get_data_folder import get_local_drive
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import *
from utils.plot_lib import shaded_error
from utils.CCAlib import CCA_sample_2areas_v3
from utils.corr_lib import *
from utils.tuning import compute_tuning_wrapper

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\CCA\\GR\\')

#%% 
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR
# session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                # ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)


#%%  Load data properly:        
# calciumversion = 'dF'
calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=True)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    # sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.2,uppertriangular=False)[0]
    # delattr(sessions[ises],'videodata')
    # delattr(sessions[ises],'behaviordata')
    # delattr(sessions[ises],'calciumdata')

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%%  

sesidx = 0
ori    = 90

[N,K]           = np.shape(sessions[sesidx].respmat) #get dimensions of response matrix

oris            = np.sort(sessions[sesidx].trialdata['Orientation'].unique())
ori_counts      = sessions[sesidx].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
assert(len(ori_counts) == 16 or len(ori_counts) == 8)
# resp_meanori    = np.empty([N,len(oris)])

idx_V1 = sessions[sesidx].celldata['roi_name']=='V1'
idx_PM = sessions[sesidx].celldata['roi_name']=='PM'

ori_idx             = sessions[sesidx].trialdata['Orientation']==ori

resp_meanori        = np.nanmean(sessions[sesidx].respmat[:,ori_idx],axis=1,keepdims=True)
resp_res            = sessions[sesidx].respmat[:,ori_idx] - resp_meanori

##   split into area 1 and area 2:
DATA1               = resp_res[idx_V1,:]
DATA2               = resp_res[idx_PM,:]

DATA1_z         = zscore(DATA1,axis=1) # zscore for each neuron across trial responses
DATA2_z         = zscore(DATA2,axis=1) # zscore for each neuron across trial responses

pca             = PCA(n_components=15) #construct PCA object with specified number of components
Xp_1            = pca.fit_transform(DATA1_z.T).T #fit pca to response matrix (n_samples by n_features)
Xp_2            = pca.fit_transform(DATA2_z.T).T #fit pca to response matrix (n_samples by n_features)

plt.subplots(figsize=(3,3))
plt.scatter(Xp_1[0,:], Xp_2[0,:],s=10,color=sns.color_palette('husl',8)[4])
plt.xlabel('PCA 1 (V1)')
plt.ylabel('PCA 1 (PM)')
plt.text(5,40,'r=%1.2f' % np.corrcoef(Xp_1[0,:],Xp_2[0,:], rowvar = False)[0,1],fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'PCA_corr_example' + '.png'), format = 'png')


#%% 

nOris = 16
corr_test = np.zeros((nSessions,nOris))
corr_train = np.zeros((nSessions,nOris))
for ises in range(nSessions):
    # get signal correlations:
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
    ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)

    idx_V1 = sessions[ises].celldata['roi_name']=='V1'
    idx_PM = sessions[ises].celldata['roi_name']=='PM'

    for i,ori in enumerate(oris): # loop over orientations 
        ori_idx             = sessions[ises].trialdata['Orientation']==ori
        resp_meanori        = np.nanmean(sessions[ises].respmat[:,ori_idx],axis=1,keepdims=True)
        resp_res            = sessions[ises].respmat[:,ori_idx] - resp_meanori
        
        ## Split data into area 1 and area 2:
        DATA1               = resp_res[idx_V1,:]
        DATA2               = resp_res[idx_PM,:]

        corr_test[ises,i],corr_train[ises,i] = CCA_subsample_1dim(DATA1,DATA2,resamples=5,kFold=5,prePCA=25)

fig,ax = plt.subplots(figsize=(3,3))
shaded_error(oris,corr_test,error='std',color='blue',ax=ax)
ax.set_ylim([0,1])
ax.set_xlabel('Orientation')
ax.set_ylabel('First canonical correlation')
ax.set_xticks(oris[::2])
ax.set_xticklabels(oris[::2].astype('int'),rotation = 45)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'CCA1_Gratings_%dsessions' % nSessions  + '.png'), format = 'png')

#%% 
plt.figure()
plt.scatter(Xp_1[0,:], Xp_2[0,:],s=10,color=sns.color_palette('husl',1))

np.corrcoef(Xp_1[0,:],Xp_2[0,:], rowvar = False)[0,1]


model = CCA(n_components = 1,scale = False, max_iter = 1000)

model.fit(Xp_1.T,Xp_2.T)

X_c, Y_c = model.transform(Xp_1.T,Xp_2.T)
corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]

plt.figure()
plt.scatter(X_c, Y_c)

#%%
plt.figure()
plt.scatter(X_c, Xp_1[0,:])



#%% 

areas = ['V1','PM','AL','RSP']
nareas = len(areas)


areas = ['V1','PM']
nareas = len(areas)

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas)

#%%  Load data properly:        
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)
    

#%% 

#To improve: run multiple iterations, subsampling different neurons / trials
# show errorbars across sessions

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

Nsub        = 250 #how many neurons to subsample from each area
prePCA      = 25 #perform dim reduc before fitting CCA, otherwise overfitting

model_CCA = CCA(n_components = 1,scale = False, max_iter = 1000)
model_PCA = PCA(n_components = 1)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA dim1'):    # iterate over sessions
    # get signal correlations:
    [N,K]           = np.shape(ses.respmat) #get dimensions of response matrix

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

                idx_areax_sub       = np.random.choice(idx_areax,np.min((np.sum(idx_areax),Nsub)),replace=False)
                idx_areay_sub       = np.random.choice(idx_areay[~np.isin(idx_areay,idx_areax_sub)],np.min((np.sum(idx_areay),Nsub)),replace=False)

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

                corr_CC1_vars[ix,iy,iori,0,4,ises] = np.corrcoef(X_c[:,0],sessions[0].respmat_videome[idx_T])[0,1]
                corr_CC1_vars[ix,iy,iori,1,4,ises] = np.corrcoef(Y_c[:,0],sessions[0].respmat_videome[idx_T])[0,1]

#%% 
#function to add colorbar for imshow data and axis
def add_colorbar_outside(im,ax):
    fig = ax.get_figure()
    bbox = ax.get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    width = 0.01
    eps = 0.01 #margin between plot and colorbar
    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(im, cax=cax)
    return cbar

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
plt.savefig(os.path.join(savedir,'CC1_CorrVars_Heatmap_V1PM_%dsessions.png' % nSessions),
             bbox_inches='tight',  format = 'png')

#%% 
colorvars                = sns.color_palette("husl",nvars)

fig, axes = plt.subplots(1,1,figsize=(4,4))
ax = axes
for ivar,var in enumerate(CC1_to_var_labels):
    data = np.nanmean(corr_CC1_vars[:,:,:,:,ivar,:],axis=(2,3,4))
    ax.plot(np.arange(nareapairs),data.flatten(),color=colorvars[ivar],linewidth=1.5)
    ax.set_xticks(np.arange(nareapairs),areapairmat.flatten())
ax.set_ylim([0,1])
ax.set_yticks([0,0.25,0.5,0.75,1])
ax.legend(CC1_to_var_labels,frameon=False,fontsize=10,loc='lower left')
ax.set_title('CC1 correlation to:')
ax.set_xlabel('Area pairs')
ax.set_ylabel('CC1 correlation with\n variable of interest')
# plt.tight_layout()
plt.savefig(os.path.join(savedir,'CC1_CorrVars_Lineplot_V1PM_%dsessions.png' % nSessions),
             bbox_inches='tight',  format = 'png')


#%% 
oris                = np.sort(sessions[ises].trialdata['Orientation'].unique())
nOris               = len(oris)
nSessions           = len(sessions)


#%%

for ises,ses in enumerate(sessions):    # iterate over sessions
    ses.celldata['labeled'] = np.where(ses.celldata['redcell']==1, 'lab', 'unl')
    ses.celldata['arealabel'] = ses.celldata['roi_name'] + ses.celldata['labeled']

#%%
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
narealabels     = len(arealabels)
ncells = np.empty((nSessions,narealabels))
for i,ses in enumerate(sessions):
    for ial, arealabel in enumerate(arealabels):
        ncells[i,ial] = np.sum(ses.celldata['arealabel']==arealabel)
plt.hist(ncells.flatten(),np.arange(0,1100,25))

from utils.regress_lib import *


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
# arealabelpairs  = [['V1unl','PMunl'],
#                    ['V1lab','PMunl'],
#                    ['V1unl','PMlab'],
#                    ['V1lab','PMlab']]

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

#%%

# Make one for different number of neurons per population: