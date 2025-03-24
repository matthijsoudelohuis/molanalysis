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

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)


#%%  Load data properly:        
# calciumversion = 'dF'
calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)
    
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

        corr_test[ises,i],corr_train[ises,i] = CCA_sample_2areas_v3(DATA1,DATA2,resamples=5,kFold=5,prePCA=25)

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

areas = ['V1','PM','AL','RSP']
nareas = len(areas)
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas)

#%%  Load data properly:        
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)
    

#%% 
oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())

nOris = 16
# corr_test = np.zeros((nSessions,nOris))
# corr_train = np.zeros((nSessions,nOris))
# for ises in range(nSessions):
#     # get signal correlations:
#     [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

#     oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
#     ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
#     assert(len(ori_counts) == 16 or len(ori_counts) == 8)

#     idx_V1 = sessions[ises].celldata['roi_name']=='V1'
#     idx_PM = sessions[ises].celldata['roi_name']=='PM'

#     for i,ori in enumerate(oris): # loop over orientations 
#         ori_idx             = sessions[ises].trialdata['Orientation']==ori
#         resp_meanori        = np.nanmean(sessions[ises].respmat[:,ori_idx],axis=1,keepdims=True)
#         resp_res            = sessions[ises].respmat[:,ori_idx] - resp_meanori
        
#         ## Split data into area 1 and area 2:
#         DATA1               = resp_res[idx_V1,:]
#         DATA2               = resp_res[idx_PM,:]

#         corr_test[ises,i],corr_train[ises,i] = CCA_sample_2areas_v3(DATA1,DATA2,resamples=5,kFold=5,prePCA=25)
# nSessions       = 1
# oris            = [0,90]
nSessions       = len(sessions)

corr_CC1_poprate    = np.empty((nareas,nareas,nOris,2,nSessions))
corr_CC1_PC1        = np.empty((nareas,nareas,nOris,2,nSessions))
corr_CC1_run        = np.empty((nareas,nareas,nOris,2,nSessions))

areapairmat = np.empty((nareas,nareas),dtype='object')
for ix,areax in enumerate(areas):
    for iy,areay in enumerate(areas):
        areapairmat[ix,iy] = areax + '-' + areay

Nsub = 50

model_CCA = CCA(n_components = 1,scale = False, max_iter = 1000)
model_PCA = PCA(n_components = 1)

# CC1data = np.empty((nareas,nareas,nOris,2,nSessions))
for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA dim1'):    # iterate over sessions
    # get signal correlations:
    [N,K]           = np.shape(ses.respmat) #get dimensions of response matrix

    zmat            = zscore(ses.respmat.T,axis=0)
    poprate         = np.nanmean(zmat,axis=1)

    gPC1            = model_PCA.fit_transform(zmat).squeeze()

    for iori,ori in enumerate(oris): # loop over orientations 
        for ix,areax in enumerate(areas):
            for iy,areay in enumerate(areas):
                idx_N_x             = ses.celldata['roi_name']==areax
                idx_N_y             = ses.celldata['roi_name']==areay

                N1                  = np.sum(idx_N_x)
                N2                  = np.sum(idx_N_y)

                idx_T               = ses.trialdata['Orientation']==ori

                ori_idx             = sessions[ises].trialdata['Orientation']==ori
                resp_meanori        = np.nanmean(sessions[ises].respmat[:,ori_idx],axis=1,keepdims=True)
                resp_res            = sessions[ises].respmat[:,ori_idx] - resp_meanori
                
                resp_res = resp_res.T
                # ## Split data into area 1 and area 2:
                # DATA1               = resp_res[idx_V1,:]
                # DATA2               = resp_res[idx_PM,:]

                ## Split data into area 1 and area 2:
                # DATA1               = resp_res[np.ix_(idx_T,idx_N_x)]
                # DATA2               = resp_res[np.ix_(idx_T,idx_N_y)]

                ## Split data into area 1 and area 2:
                DATA1               = resp_res[:,idx_N_x]
                DATA2               = resp_res[:,idx_N_y]

                # DATA1               = ses.respmat[np.ix_(idx_N_x,idx_T)].T
                # DATA2               = ses.respmat[np.ix_(idx_N_y,idx_T)].T

                # DATA1               = zmat[np.ix_(idx_T,idx_N_x)]
                # DATA2               = zmat[np.ix_(idx_T,idx_N_y)]

                # corr_test[ises,i],corr_train[ises,i] = CCA_sample_2areas_v3(DATA1,DATA2,resamples=5,kFold=5,prePCA=25)

                # randtimepoints = np.random.choice(K,nK,replace=False)
                X = DATA1[:,np.random.choice(N1,np.min((N1,Nsub)),replace=False)]
                Y = DATA2[:,np.random.choice(N2,np.min((N2,Nsub)),replace=False)]
                # Y = DATA2[np.ix_(randtimepoints,np.random.choice(N2,nN,replace=False))]

                # X = np.reshape(X,(nN,-1),order='F').T #concatenate time bins from each trial and transpose (samples by features now)
                # Y = np.reshape(Y,(nN,-1),order='F').T

                # X = zscore(X,axis=0)  #Z score activity for each neuron
                # Y = zscore(Y,axis=0)

                # Compute and store canonical correlations for the first pair
                X_c, Y_c = model_CCA.fit_transform(X,Y)

                # np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]

                corr_CC1_run[ix,iy,iori,0,ises] = np.corrcoef(X_c[:,0],ses.respmat_runspeed[idx_T])[0,1]
                corr_CC1_run[ix,iy,iori,1,ises] = np.corrcoef(Y_c[:,0],ses.respmat_runspeed[idx_T])[0,1]

                corr_CC1_PC1[ix,iy,iori,0,ises] = np.corrcoef(X_c[:,0],gPC1[idx_T])[0,1]
                corr_CC1_PC1[ix,iy,iori,1,ises] = np.corrcoef(Y_c[:,0],gPC1[idx_T])[0,1]

                corr_CC1_poprate[ix,iy,iori,0,ises] = np.corrcoef(X_c[:,0],poprate[idx_T])[0,1]
                corr_CC1_poprate[ix,iy,iori,1,ises] = np.corrcoef(Y_c[:,0],poprate[idx_T])[0,1]

#%% 
data = np.nanmean(corr_CC1_run,axis=(2,3,4))
# data = np.nanmean(np.abs(corr_CC1_run),axis=(2,3,4))

fig, axes = plt.subplots(1,3,figsize=(9,3))
ax = axes[0]
ax.imshow(data,cmap='bwr',clim=(-1,1))
ax.set_xticks(np.arange(0,nareas))
ax.set_xticklabels(areas)
ax.set_yticks(np.arange(0,nareas))
ax.set_yticklabels(areas)
# fig.colorbar(ax.images[0], ax=ax)

ax = axes[1]
data = np.nanmean(corr_CC1_PC1,axis=(2,3,4))
ax.imshow(data,cmap='bwr',clim=(-1,1))
ax.set_xticks(np.arange(0,nareas))
ax.set_xticklabels(areas)
ax.set_yticks(np.arange(0,nareas))
ax.set_yticklabels(areas)
# fig.colorbar(ax.images[0], ax=ax)

data = np.nanmean(corr_CC1_poprate,axis=(2,3,4))

ax = axes[2]
ax.imshow(data,cmap='bwr',clim=(-1,1))
ax.set_xticks(np.arange(0,nareas))
ax.set_xticklabels(areas)
ax.set_yticks(np.arange(0,nareas))
ax.set_yticklabels(areas)
fig.colorbar(ax.images[0], ax=ax)

#%% 
data = np.nanmean(corr_CC1_run,axis=(2,3))
data = np.transpose(data,(2,0,1)).reshape((nSessions,-1))

fig, axes = plt.subplots(1,1,figsize=(9,3))
ax = axes
apflat = areapairmat.flatten()
nareapairs = nareas*(nareas-1)/2
# plt.nanmean(data,axis=1)
ax.plot(np.arange(len(apflat)),np.nanmean(data,axis=0))
ax.set_xticks(np.arange(len(apflat)),apflat)
