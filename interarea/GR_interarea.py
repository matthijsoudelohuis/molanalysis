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
    
#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% 
 #####   #####     #    
#     # #     #   # #   
#       #        #   #  
#       #       #     # 
#       #       ####### 
#     # #     # #     # 
 #####   #####  #     # 

#%%  
sesidx = 1
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

fig,axes = plt.subplots(figsize=(3.5,3))
plt.scatter(Xp_1[0,:], Xp_2[0,:],s=10,color=sns.color_palette('husl',8)[4])
plt.xlabel('PCA 1 (V1)')
plt.ylabel('PCA 1 (PM)')
plt.text(5,-5,'r=%1.2f' % np.corrcoef(Xp_1[0,:],Xp_2[0,:], rowvar = False)[0,1],fontsize=12)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3)
# plt.savefig(os.path.join(savedir,'PCA_corr_example' + '.png'), format = 'png')
my_savefig(fig,savedir,'PCA_V1PM_corr_example')

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
# fig.savefig(os.path.join(savedir,'CCA1_Gratings_%dsessions' % nSessions  + '.png'), format = 'png')
my_savefig(fig,savedir,'CCA1_Gratings_%dsessions' % nSessions,formats=['png'])

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


areas = ['V1','PM']
nareas = len(areas)




# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,filter_areas=areas)
# sessions,nSessions   = filter_sessions(protocols = 'GN',only_all_areas=areas,filter_areas=areas)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)

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
oris                = np.sort(sessions[ises].trialdata['Orientation'].unique())
nOris               = len(oris)
nSessions           = len(sessions)

#%%
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
narealabels     = len(arealabels)
ncells = np.empty((nSessions,narealabels))
for i,ses in enumerate(sessions):
    for ial, arealabel in enumerate(arealabels):
        ncells[i,ial] = np.sum(ses.celldata['arealabel']==arealabel)
plt.hist(ncells.flatten(),np.arange(0,1100,25))

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











#%% 

######  ######  ######  
#     # #     # #     # 
#     # #     # #     # 
######  ######  ######  
#   #   #   #   #   #   
#    #  #    #  #    #  
#     # #     # #     # 

#%% 












#%% get optimal lambda
nsampleneurons  = 200
lambdas         = np.logspace(-6, 5, 10)
# lambdas         = np.array([0,0.01,0.1,1])
nlambdas        = len(lambdas)
r               = 15
kfold           = 10
nranks          = 50

R2_cv           = np.full((nSessions,nlambdas,nranks,kfold),np.nan)

for ises,ses in enumerate(sessions):
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

                # for r in range(nranks):
                    # Y_hat_rr_train = U[:,:r] @ S[:r,:r] @ V[:r,:]
                    # print(EV(Y_train,Y_hat_rr_train))

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,ilam,r,ikf] = EV(Y_test,Y_hat_rr_test)

#%% 
rankdata = np.full((nSessions,nlambdas),np.nan)
for ises,ses in enumerate(sessions):
    for ilam,lam in enumerate(lambdas):
        rankdata[ises,ilam] = rank_from_R2(R2_cv[ises,ilam,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',nlambdas)

fig,axes = plt.subplots(1,2,figsize=(6,3))
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
# for ilam,lam in enumerate(lambdas):
#     tempdata = np.nanmean(rankdata,axis=0)
#     ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.max(rankdata)+1])

plt.tight_layout()
plt.savefig(os.path.join(savedir,'RRR_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')

#%% Test wrapper function:
nsampleneurons  = 20
nranks          = 25
nmodelfits      = 10 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions),np.nan)
optim_rank      = np.full((nSessions),np.nan)

for ises,ses in enumerate(sessions):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    
    R2_cv[ises],optim_rank[ises]             = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)


#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons

popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20,50,100])
npopsizes           = len(popsizes)
nranks              = 25
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
ax.set_ylim([0,0.25])
ax.set_xticks(popsizes)
ax.axhline(y=0,color='k',linestyle='--')

ax.set_xlabel('Population size')
ax.set_ylabel('RRR R2')
# ax.set_xscale('log')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
ax.scatter(popsizes, np.nanmean(optim_rank,axis=0), marker='o', color=clrs_popsizes)
shaded_error(popsizes,optim_rank,center='mean',error='sem',color='k',ax=ax)

ax.plot(popsizes,popsizes**0.5,color='r',linestyle='--',linewidth=1)
ax.text(50,10,'$n^{1/2}$',color='r',fontsize=12)
ax.plot(popsizes,popsizes**0.3333,color='g',linestyle='--',linewidth=1)
ax.text(50,2,'$n^{1/3}$',color='g',fontsize=12)

ax.set_ylim([0,20])
ax.set_ylabel('Dimensionality')
ax.set_xlabel('Population size')
ax.set_xticks(popsizes)

sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'R2_RRR_Rank_PopSize_V1PM_%dsessions_log.png' % nSessions), format = 'png')
fig.savefig(os.path.join(savedir,'R2_RRR_Rank_PopSize_V1PM_%dsessions.png' % nSessions), format = 'png')


#%% Validate regressing out behavior: 
ises    = 2
ses     = sessions[ises]
idx_T   = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
# idx_T   = ses.trialdata['stimCond']==0

X       = np.stack((sessions[ises].respmat_videome[idx_T],
                sessions[ises].respmat_runspeed[idx_T],
                sessions[ises].respmat_pupilarea[idx_T],
                sessions[ises].respmat_pupilx[idx_T],
                sessions[ises].respmat_pupily[idx_T]),axis=1)
X       = np.column_stack((X,sessions[ises].respmat_videopc[:,idx_T].T))
X       = zscore(X,axis=0,nan_policy='omit')

si      = SimpleImputer()
X       = si.fit_transform(X)

idx_areay       = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]

idx_areay       = ses.celldata['noise_level']<20

idx_areay       = np.isin(ses.celldata['roi_name'],['V1','PM'])

# X              = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
# X               = zscore(X,axis=0,nan_policy='omit')

Y               = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
Y               = zscore(Y,axis=0,nan_policy='omit')
# Y               = zscore(Y,axis=1,nan_policy='omit')

Y_sub,rank      = regress_out_behavior_modulation(sessions[ises],X,Y,rank=5,lam=0)

#%% Make figure
minmax = 0.75
fig,axes = plt.subplots(1,3,figsize=(6,3),sharex=False)
ax = axes[0]
ax.imshow(X,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('Behavior')
ax.set_ylabel('Trials')
ax.set_xlabel('Behavioral features')

ax = axes[1]
ax.imshow(Y,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('Original')
ax.set_yticklabels('')
ax.set_xlabel('Neurons')

ax = axes[2]
ax.imshow(Y_sub,cmap='bwr',aspect='auto',vmin=-minmax,vmax=minmax)
ax.set_title('Orig-Behavior (RRR)')
ax.set_yticklabels('')
ax.set_xlabel('Neurons')
plt.tight_layout()
my_savefig(fig,savedir,'BehaviorRegressedOut_V1PM_%s.png' % ses.sessiondata['session_id'][0],formats=['png'])

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
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
nsampleneurons      = 20
nranks              = 25
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((narealabelpairs,2,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,2,nSessions),np.nan)

filter_nearby       = True

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)

    for irbhv,rbhb in enumerate([False,True]):
        respmat                 = ses.respmat[:,idx_T].T
        # Yall                   = ses.respmat[:,idx_T].T

        if rbhb:
            # X           = np.stack((ses.respmat_videome[idx_T],
            #                 ses.respmat_runspeed[idx_T],
            #                 ses.respmat_pupilarea[idx_T]),axis=1)

            X       = np.stack((sessions[ises].respmat_videome[idx_T],
                            sessions[ises].respmat_runspeed[idx_T],
                            sessions[ises].respmat_pupilarea[idx_T],
                            sessions[ises].respmat_pupilx[idx_T],
                            sessions[ises].respmat_pupily[idx_T]),axis=1)
            X       = np.column_stack((X,sessions[ises].respmat_videopc[:,idx_T].T))
            X       = zscore(X,axis=0,nan_policy='omit')

            si      = SimpleImputer()
            X       = si.fit_transform(X)

            respmat,_  = regress_out_behavior_modulation(ses,X,respmat,rank=5,lam=0,perCond=False)
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
        
            X = respmat[:,idx_areax]
            Y = respmat[:,idx_areay]

            if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons:
                R2_cv[iapl,irbhv,ises],optim_rank[iapl,irbhv,ises]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%%



#%% 

# #%% Show the data: R2
# fig, axes = plt.subplots(1,2,figsize=(8,3),sharey=True,sharex=True)

# ax = axes[0]

# handles = []
# for iapl, arealabelpair in enumerate(arealabelpairs):

#     ax = axes[np.array(iapl>3,dtype=int)]
#     ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3,R2_cv[:,iapl],color='k',marker='o',s=10)
#     # handles.append(shaded_error(np.arange(ndims),tempdata[iapl,:,:].T,center='mean',error='sem',
#             # color=clrs_arealabelpairs[iapl],linewidth=2,ax=ax))
# # 
#     for ises in range(nSessions):
#         ax.plot(np.arange(ndims),tempdata[iapl,:,ises],
#                 color=clrs_arealabelpairs[iapl],alpha=0.5,linewidth=0.5)
#     # plt.plot(np.arange(ndims),np.nanmean(R2[:,:,0,:,0],axis=(0,1,2)),color='k',linewidth=2)

# for iax in range(2):
#     ax = axes[iax]
#     ax.legend([handles[i] for i in np.arange(0,4)+iax*4],[arealabelpairs[i] for i in np.arange(0,4)+iax*4],
#               loc='upper left',frameon=False,fontsize=9)
#     ax.set_xlabel('Rank')
# ax.set_xticks(np.arange(0,ndims,5))
# ax.set_ylim([0,my_ceil(np.nanmax(tempdata),1)])
# # ax.set_yticks([0,ax.get_ylim()[1]])
# ax.set_yticks([0,ax.get_ylim()[1]/2,ax.get_ylim()[1]])
# axes[0].set_ylabel('R2 (RRR)')

# sns.despine(top=True,right=True,offset=3)
# # fig.savefig(os.path.join(savedir,'RRR_cvR2_V1PM_LabUnl_RegressBehav%s_%dsessions.png' % (regress_out_behavior,nSessions)),
            # bbox_inches='tight')
# fig.savefig(os.path.join(savedir,'RRR_cvR2_V1PM_LabUnl_%dsessions.png' % nSessions))

#%% Plot the R2 performance and number of dimensions per area pair
fig, axes = plt.subplots(2,2,figsize=(8,6))

datatoplot = R2_cv
arealabelpairs2 = [al.replace('-','-\n') for al in arealabelpairs]

for irbh in range(2):
    ax=axes[irbh,0]
    for iapl, arealabelpair in enumerate(arealabelpairs):
        ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3,datatoplot[iapl,irbh,:],color='k',marker='o',s=10)
        ax.errorbar(iapl+0.5,np.nanmean(datatoplot[iapl,irbh,:]),np.nanstd(datatoplot[iapl,irbh,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

    ax.set_xticks(range(narealabelpairs))
    ax.set_ylabel('R2 (cv)')
    ax.set_ylim([0,my_ceil(np.nanmax(datatoplot),2)])
    # ax.set_yticks(np.arange(0,7,2))
    ax.set_xlabel('Population pair')
    ax.set_title('Performance at full rank')

    ax=axes[irbh,1]
    for iapl, arealabelpair in enumerate(arealabelpairs):
        ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3,optim_rank[iapl,irbh,:],color='k',marker='o',s=10)
        ax.errorbar(iapl+0.5,np.nanmean(optim_rank[iapl,irbh,:]),np.nanstd(optim_rank[iapl,irbh,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

    ax.set_xticks(range(narealabelpairs))
    ax.set_ylabel('Number of dimensions')
    ax.set_ylim([0,my_ceil(np.nanmax(optim_rank),0)+1])
    ax.set_yticks(np.arange(0,10,2))
    ax.set_xlabel('Population pair')
    ax.set_title('Dimensionality')

sns.despine(top=True,right=True,offset=3)
axes[1,0].set_xticklabels(arealabelpairs2,fontsize=7)
axes[1,1].set_xticklabels(arealabelpairs2,fontsize=7)

# fig.savefig(os.path.join(savedir,'RRR_cvR2_Rank_V1PM_LabUnl_RegressBehav%s_%dsessions.png' % (regress_out_behavior,nSessions)),
            # bbox_inches='tight')

#%% Print how many labeled neurons there are in V1 and Pm in the loaded sessions:


print('Number of labeled neurons in V1 and PM:')
for ises, ses in enumerate(sessions):

    print('Session %d: %d in V1, %d in PM' % (ises+1,
                                              np.sum(np.all((ses.celldata['redcell']==1,
                                                             ses.celldata['roi_name']=='V1',
                                                             ses.celldata['noise_level']<100),axis=0)),
                                              np.sum(np.all((ses.celldata['redcell']==1,
                                                             ses.celldata['roi_name']=='PM',
                                                             ses.celldata['noise_level']<100),axis=0))))


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
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
nsampleneurons      = 15
nranks              = 25
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((narealabelpairs,2,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,2,nSessions),np.nan)

filter_nearby       = True

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)

    for irbhv,rbhb in enumerate([False,True]):
        respmat                 = ses.respmat[:,idx_T].T

        if rbhb:
            # X           = np.stack((ses.respmat_videome[idx_T],
            #                 ses.respmat_runspeed[idx_T],
            #                 ses.respmat_pupilarea[idx_T]),axis=1)

            # idx_ALRSP   = np.isin(ses.celldata['roi_name'],['AL','RSP'])
            idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
                                    # ses.celldata['noise_level']<20	
                                    ),axis=0))[0]
            
            X           = respmat[:,idx_ALRSP]
            X           = zscore(X,axis=0)
            
            # idx_V1PM = np.isin(ses.celldata['roi_name'],['V1','PM'])
            idx_V1PM   = np.where(np.all((np.isin(ses.celldata['roi_name'],['V1','PM']),
                                    # ses.celldata['noise_level']<20	
                                    ),axis=0))[0]
            
            respmat[:,idx_V1PM],_  = regress_out_behavior_modulation(ses,X,respmat[:,idx_V1PM],rank=10,lam=0)
            
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
        
            X = respmat[:,idx_areax]
            Y = respmat[:,idx_areay]

            if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons:
                R2_cv[iapl,irbhv,ises],optim_rank[iapl,irbhv,ises]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot the number of dimensions per area pair
fig, axes = plt.subplots(2,2,figsize=(8,6))

statpairs = [(0,1),(0,2),(0,3),
             (4,5),(4,6),(4,7)]

# order = ['non','flp','cre'] #for statistical testing purposes
# pairs = [('non','flp'),('non','cre')]

# fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF"]

datatoplot          = R2_cv
datatoplot[datatoplot==0]          = np.nan
arealabelpairs2     = [al.replace('-','-\n') for al in arealabelpairs]

for irbh in range(2):
    ax=axes[irbh,0]
    for iapl, arealabelpair in enumerate(arealabelpairs):
        ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3,datatoplot[iapl,irbh,:],color='k',marker='o',s=10)
        ax.errorbar(iapl+0.5,np.nanmean(datatoplot[iapl,irbh,:]),np.nanstd(datatoplot[iapl,irbh,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

    ax.set_ylabel('R2 (cv)')
    ax.set_ylim([0,my_ceil(np.nanmax(datatoplot[:,irbh,:]),2)])

    if irbh==1:
        ax.set_xlabel('Population pair')
        ax.set_xticks(range(narealabelpairs))
    else:
        ax.set_xticks(range(narealabelpairs),labels=[])

    if irbh==0:
        ax.set_title('Performance at full rank')

    # df = pd.DataFrame(data=datatoplot[:,irbh,:].T,columns=np.arange(narealabelpairs))
    # df = pd.DataFrame(data=datatoplot[:,irbh,:].flatten(),columns=np.arange(narealabelpairs))
    testdata = datatoplot[:,irbh,:]
    testdata = testdata[:,~np.isnan(testdata).any(axis=0)]

    df = pd.DataFrame({'R2':  testdata.flatten(),
                       'arealabelpair':np.repeat(np.arange(narealabelpairs),np.shape(testdata)[1])})
    
    annotator = Annotator(ax, statpairs, data=df, x="arealabelpair", y='R2', order=np.arange(narealabelpairs))
    annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False)
    # annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False)
    annotator.apply_and_annotate()

    ax=axes[irbh,1]
    for iapl, arealabelpair in enumerate(arealabelpairs):
        ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3,optim_rank[iapl,irbh,:],color='k',marker='o',s=10)
        ax.errorbar(iapl+0.5,np.nanmean(optim_rank[iapl,irbh,:]),np.nanstd(optim_rank[iapl,irbh,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

    ax.set_xticks(range(narealabelpairs))
    if irbh==1:
        ax.set_xlabel('Population pair')
        ax.set_xticks(range(narealabelpairs))
    else:
        ax.set_xticks(range(narealabelpairs),labels=[])
    
    if irbh==1:
        ax.set_ylabel('Number of dimensions')
    ax.set_yticks(np.arange(0,14,2))
    ax.set_ylim([0,my_ceil(np.nanmax(optim_rank),0)+1])
    if irbh==0:
        ax.set_title('Dimensionality')

sns.despine(top=True,right=True,offset=3)
axes[1,0].set_xticklabels(arealabelpairs2,fontsize=7)
axes[1,1].set_xticklabels(arealabelpairs2,fontsize=7)

# my_savefig(fig,savedir,'RRR_V1PM_regressoutneural_%dsessions.png' % (nSessions))

#%% Fraction of R2 explained by shared activity with AL and RSP:
datatoplot          = (R2_cv[:,0,:] - R2_cv[:,1,:]) / R2_cv[:,0,:]
datatoplot          = R2_cv[:,1,:]  / R2_cv[:,0,:]
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
ax.set_ylim([0,0.5])
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
