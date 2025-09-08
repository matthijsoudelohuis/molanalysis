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
from utils.tuning import compute_tuning,compute_tuning_wrapper
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *
from utils.CCAlib import *
from utils.corr_lib import *
from utils.regress_lib import *
from utils.gain_lib import *
from utils.RRRlib import estimate_dimensionality

# savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\CCA\\')
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\PCA\\')

#%% Load an example session: 
session_list        = np.array(['LPE12223_2024_06_10']) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%%  Load data properly:        
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)
    
#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)


#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)

# #%% Add how neurons are coupled to the population rate: 
# for ses in tqdm(sessions,desc='Computing population coupling for each session'):
#     resp        = zscore(ses.respmat.T,axis=0)
#     poprate     = np.mean(resp, axis=1)
#     ses.celldata['pop_coupling']                          = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(len(ses.celldata))]

#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


#%% From Humphries (2023) ArXiv:
# When we eliminate (many) higher dimensions as noise, we inevitably run the risk
# of removing elements of neural activity that could be crucial for understanding 
# the coding or computation of the neural population. For example, population
# activity projected into a low dimensional space is unlikely to contain a meaningful 
# contribution from any “soloist” neurons [Okun et al. 2015] as by definition their activity
# is independent from the majority. Recent modelling work [Sweeney & Clopath 2020] suggests 
# that in visual cortex such soloist neurons are those with the least variable stimulus
# responses, and thus by eliminating them dimension reduction would potentially eliminate 
# the most consistent response to a given stimulus.

######   #####     #        #####  ####### #     # ######  #       ### #     #  #####  
#     # #     #   # #      #     # #     # #     # #     # #        #  ##    # #     # 
#     # #        #   #     #       #     # #     # #     # #        #  # #   # #       
######  #       #     #    #       #     # #     # ######  #        #  #  #  # #  #### 
#       #       #######    #       #     # #     # #       #        #  #   # # #     # 
#       #     # #     #    #     # #     # #     # #       #        #  #    ## #     # 
#        #####  #     #     #####  #######  #####  #       ####### ### #     #  #####  

#%% Are the PCA weights higher for coupled or uncoupled neuron?
n_components        = 20
nStim               = 16
nmodelfits          = 5
nsampleneurons      = 250
maxnoiselevel       = 20

areas               = np.array(['V1', 'PM'])
nareas              = len(areas)
weights_PCA         = np.full((nareas,nsampleneurons,n_components,nSessions,nStim),np.nan)
popcoupling_PCA     = np.full((nareas,nsampleneurons,nSessions,nStim),np.nan)


#%% 
model_PCA       = PCA(n_components=n_components)

for ises,ses in enumerate(sessions):    # iterate over sessions
    for iarea,area in enumerate(areas):
        idx_area               = np.where(np.all((ses.celldata['roi_name']==area,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        ),axis=0))[0]
        # print(len(idx_area))
        if len(idx_area)>nsampleneurons:
            
            idx_area_sub       = np.random.choice(idx_area,nsampleneurons,replace=False)
        
            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            # for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),total=nStim,desc='Fitting CCA model'):    # iterate over sessions
                idx_T               = ses.trialdata['stimCond']==stim
                idx_T               = ses.trialdata['stimCond']>-1

                X                   = sessions[ises].respmat[np.ix_(idx_area_sub,idx_T)].T
                
                X                   = zscore(X,axis=0)  #Z score activity for each neuron

                # Fit CCA MODEL:
                model_PCA.fit(X)

                weights_PCA[iarea,:,:,ises,istim]   = model_PCA.components_.T

                popcoupling_PCA[iarea,:,ises,istim] = ses.celldata.loc[idx_area_sub,'pop_coupling']

#%% 
weight_popcouple_corr = np.empty((nareas,n_components,nSessions,nStim))
for ises in range(nSessions):
    for iarea in range(nareas):
        for istim in range(nStim):
            for icomp in range(n_components):
                weight_popcouple_corr[iarea,icomp,ises,istim] = np.corrcoef(weights_PCA[iarea,:,icomp,ises,istim],popcoupling_PCA[iarea,:,ises,istim])[0,1]
                # weight_popcouple_corr[iarea,icomp,ises,istim] = np.abs(np.corrcoef(weights_PCA[iarea,:,icomp,ises,istim],popcoupling_PCA[iarea,:,ises,istim])[0,1])

#%% Show correlation between weights and pop. coupling
#Test whether lower dimensional components are more correlated with higher pop. coupling. 
dimthr = 4
fig,ax = plt.subplots(1,1,figsize=(3.5,3.5))
ax.errorbar(range(n_components),np.nanmean(weight_popcouple_corr,axis=(0,2,3)),
            np.nanstd(weight_popcouple_corr,axis=(0,2,3)) / np.sqrt(nSessions),
            fmt='o',color='k')
# ax.plot(range(n_components),np.nanmean(weight_popcouple_corr,axis=(0,2,3)))
ax.set_xlabel('PC dimension')
ax.set_ylabel('Correlation (PC weights vs. pop. coupling)')
from scipy.stats import ttest_1samp
testdata = np.nanmean(weight_popcouple_corr,axis=(0,3))
# [ttest,pval] = ttest_1samp(weight_popcouple_corr[:,:dimthr,:,:].flatten(),0,nan_policy='omit')
[ttest,pval] = ttest_1samp(testdata[:dimthr,:].flatten(),0,nan_policy='omit')
add_stat_annotation(ax, 0, dimthr, 0.8, pval, h=0.0, color='red')
# [ttest,pval] = ttest_1samp(weight_popcouple_corr[:,dimthr:,:,:].flatten(),0,nan_policy='omit')
[ttest,pval] = ttest_1samp(testdata[dimthr:,:].flatten(),0,nan_policy='omit')
add_stat_annotation(ax, dimthr, n_components, -0.15, pval, h=0.0, color='blue')

ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,savedir,'Corr_PopCoupling_PCA_weights_%dsessions' % nSessions,formats=['png'])

#%% Is dimensionality different for soloists and choristers
n_components        = 50
nStim               = 16
nsampleneurons      = 100
maxnoiselevel       = 20

nbins_popcoupling   = 5

areas               = np.array(['V1', 'PM'])
nareas              = len(areas)
PCA_ev              = np.full((nareas,nbins_popcoupling,n_components,nSessions,nStim),np.nan)
PCA_dim             = np.full((nareas,nbins_popcoupling,nSessions,nStim),np.nan)
# weights_PCA         = np.full((nareas,nsampleneurons,n_components,nSessions,nStim),np.nan)
# popcoupling_PCA     = np.full((nareas,nsampleneurons,nSessions,nStim),np.nan)
dim_metric          = 'Participation Ratio'

#%% 
model_PCA       = PCA(n_components=n_components)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting PCA model'):
# for ises,ses in enumerate(sessions):    # iterate over sessions
    binedges_popcoupling   = np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,nbins_popcoupling+1))
    for iarea,area in enumerate(areas):
        for ibin in range(nbins_popcoupling):
            idx_N               = np.where(np.all((ses.celldata['roi_name']==area,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        ses.celldata['pop_coupling']>=binedges_popcoupling[ibin],
                                        ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1],	
                                        ),axis=0))[0]

            # print(len(idx_N))
            if len(idx_N)>nsampleneurons:
                
                idx_N_sub       = np.random.choice(idx_N,nsampleneurons,replace=False)
            
                # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
                # # for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),total=nStim,desc='Fitting CCA model'):    # iterate over sessions
                #     # idx_T               = ses.trialdata['stimCond']==stim
                #     idx_T               = ses.trialdata['stimCond']>-1
                
                #     X                   = sessions[ises].respmat[np.ix_(idx_N_sub,idx_T)].T
                    
                #     X                   = zscore(X,axis=0)  #Z score activity for each neuron

                #     # Fit CCA MODEL:
                #     model_PCA.fit(X)

                #     # PCA_ev[iarea,ibin,:,ises,istim]   = model_PCA.explained_variance_ratio_
                #     PCA_ev[iarea,ibin,:,ises,istim]   = np.cumsum(model_PCA.explained_variance_ratio_)
                #     # weights_PCA[iarea,:,:,ises,istim]   = model_PCA.components_.T
                    
                #     PCA_dim[iarea,ibin,ises,istim]      = (np.sum(model_PCA.explained_variance_) ** 2) / np.sum(model_PCA.explained_variance_ ** 2)
                #         # PCA_dim[iarea,ibin,ises,istim]      = estimate_dimensionality(X,method='participation_ratio')
                #         # popcoupling_PCA[iarea,:,ises,istim] = ses.celldata.loc[idx_area_sub,'pop_coupling']

                idx_T               = ses.trialdata['stimCond']>-1
                X                   = sessions[ises].respmat[np.ix_(idx_N_sub,idx_T)].T
                
                X                   = zscore(X,axis=0)  #Z score activity for each neuron

                # Fit CCA MODEL:
                model_PCA.fit(X)

                # PCA_ev[iarea,ibin,:,ises,istim]   = model_PCA.explained_variance_ratio_
                PCA_ev[iarea,ibin,:,ises,istim]   = np.cumsum(model_PCA.explained_variance_ratio_)
                # weights_PCA[iarea,:,:,ises,istim]   = model_PCA.components_.T
                
                PCA_dim[iarea,ibin,ises,istim]      = (np.sum(model_PCA.explained_variance_) ** 2) / np.sum(model_PCA.explained_variance_ ** 2)
                # PCA_dim[iarea,ibin,ises,istim]      = estimate_dimensionality(X,method='pca_ev')
                    # popcoupling_PCA[iarea,:,ises,istim] = ses.celldata.loc[idx_area_sub,'pop_coupling']

#%% 
clrs_popcoupling    = sns.color_palette('magma',nbins_popcoupling)

fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharey=True,sharex=True)
for iarea,area in enumerate(areas):
    ax = axes[iarea]
    handles = []
    for ibin in range(nbins_popcoupling):
        handles.append(shaded_error(x=np.arange(n_components),y=np.nanmean(PCA_ev[iarea,ibin,:,:,:],axis=(2)).T,ax=ax,color=clrs_popcoupling[ibin]))
    ax.set_title(area,fontsize=12)
    if iarea==0: 
        ax.legend(handles,['0-20%','20-40%','40-60%','60-80%','80-100%'],
                    reverse=True,fontsize=7,frameon=False,title='pop. coupling', loc='lower right')
ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)
ax.set_xlabel('PCA Dimension')
axes[0].set_ylabel('Explained Variance')
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,savedir,'PCA_ev_PopCoupling_%dsessions' % nSessions,formats=['png'])

#%% 
fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharey=True,sharex=True)
for iarea,area in enumerate(areas):
    ax = axes[iarea]
    ax.plot(np.arange(nbins_popcoupling),np.nanmean(PCA_dim[iarea,:,:,:],axis=(1,2)),color='k',
            linewidth=1)
    ax.scatter(np.arange(nbins_popcoupling),np.nanmean(PCA_dim[iarea,:,:,:],axis=(1,2)),
               c=clrs_popcoupling,s=50)
    ax.set_xticks(np.arange(0,nbins_popcoupling))
    # ax.set_xticks(np.arange(0,nbins_popcoupling),['0-20%','20-40%','40-60%','60-80%','80-100%'])
    ax.set_ylim([0,np.nanmax(np.nanmean(PCA_dim[iarea,:,:,:],axis=(1,2))*1.2)])
    ax.set_xlabel('Population Coupling')
    ax.set_title(area,fontsize=12)
    if iarea==0:
        ax.set_ylabel('Dimensionality (%s)' % dim_metric)
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,savedir,'PCA_dim_PopCoupling_%dsessions' % nSessions,formats=['png'])


#%% 
 #####   #####     #       #     # ####### ###  #####  #     # #######  #####  
#     # #     #   # #      #  #  # #        #  #     # #     #    #    #     # 
#       #        #   #     #  #  # #        #  #       #     #    #    #       
#       #       #     #    #  #  # #####    #  #  #### #######    #     #####  
#       #       #######    #  #  # #        #  #     # #     #    #          # 
#     # #     # #     #    #  #  # #        #  #     # #     #    #    #     # 
 #####   #####  #     #     ## ##  ####### ###  #####  #     #    #     #####  

#%% 


#%% Are the weights higher for coupled or uncoupled neuron?
n_components        = 20
nStim               = 16
nmodelfits          = 5
nsampleneurons      = 50
maxnoiselevel       = 20

areas               = np.array(['V1', 'PM'])
weights_CCA         = np.full((len(areas),nsampleneurons,n_components,nSessions,nStim,nmodelfits),np.nan)
popcoupling_CCA     = np.full((len(areas),nsampleneurons,nSessions,nStim,nmodelfits),np.nan)
do_cv_cca           = False

#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
for ises,ses in enumerate(sessions):    # iterate over sessions
    
    idx_areax           = np.where(np.all((ses.celldata['roi_name']==areas[0],
                                    ses.celldata['noise_level']<maxnoiselevel,	
                                    ),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']==areas[1],
                                ses.celldata['noise_level']<maxnoiselevel,	
                                ),axis=0))[0]
    
    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:
        
        # for imf in range(nmodelfits):
        for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting CCA model session %d/%d' % (ises+1,nSessions)):    # iterate over sessions

            idx_areax_sub       = np.random.choice(idx_areax,np.min((np.sum(idx_areax),nsampleneurons)),replace=False)
            idx_areay_sub       = np.random.choice(idx_areay[~np.isin(idx_areay,idx_areax_sub)],np.min((np.sum(idx_areay),nsampleneurons)),replace=False)
           
            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            # for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),total=nStim,desc='Fitting CCA model'):    # iterate over sessions
                idx_T               = ses.trialdata['stimCond']==stim
            
                X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
                Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
                
                X                   = zscore(X,axis=0)  #Z score activity for each neuron
                Y                   = zscore(Y,axis=0)

                # Fit CCA MODEL:
                model_CCA.fit(X,Y)

                weights_CCA[0,:,:,ises,istim,imf]   = model_CCA.x_loadings_
                weights_CCA[1,:,:,ises,istim,imf]   = model_CCA.y_loadings_

                popcoupling_CCA[0,:,ises,istim,imf] = ses.celldata.loc[idx_areax_sub,'pop_coupling']
                popcoupling_CCA[1,:,ises,istim,imf] = ses.celldata.loc[idx_areay_sub,'pop_coupling']

#%% 
n_components_toplot = 5
fig,axes = plt.subplots(1,n_components_toplot,figsize=(n_components_toplot*2,2),sharey=True,sharex=True)
for icomponent in range(n_components_toplot):
    ax = axes[icomponent]
    for iarea in range(len(areas)):
        for istim in range(nStim):
            for imf in range(nmodelfits):
                ax.scatter(popcoupling_CCA[iarea,:,ises,istim,imf],weights_CCA[iarea,:,icomponent,ises,istim,imf],
                   s=8,marker='.',color='k',alpha=0.25)
    ax.set_title('Component %d' % icomponent)
    ax.set_xlabel('Pop coupling')
    if icomponent==0: 
        ax.set_ylabel('CCA Weight')
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,savedir,'Corr_PopCoupling_CCA_weights_Scatter_%dsessions' % nSessions,formats=['png'])

#%% 
corrmat = np.full((len(areas),n_components,nSessions,nStim,nmodelfits),np.nan)
for icomp in range(n_components):
    for iarea in range(len(areas)):
        for ises in range(nSessions):
            for istim in range(nStim):
                for imf in range(nmodelfits):
                    corrmat[iarea,icomp,ises,istim,imf] = np.corrcoef(popcoupling_CCA[iarea,:,ises,istim,imf],
                            weights_CCA[iarea,:,icomp,ises,istim,imf])[0,1]

#%% 
fig,axes = plt.subplots(1,1,figsize=(3,3),sharey=True,sharex=True)
clrs_areas = get_clr_areas(areas)
ax = axes
handles = []
for iarea in range(len(areas)):
    # shaded_error(x=np.arange(n_components),y=np.nanmean(corrmat[iarea,:,:,:,:],axis=(1,2,3)),ax=ax)
    handles.append(shaded_error(x=np.arange(n_components),y=np.reshape(corrmat[iarea,:,:,:,:],(n_components,-1)).T,
                 error='std',color=clrs_areas[iarea],ax=ax))
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_ylim([-0.3,1])
    ax.set_xlabel('CCA Dimension')
    ax.set_ylabel('Correlation')
    ax_nticks(ax,5)
    ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)
ax.legend(handles,areas,loc='upper right',fontsize=12)
my_legend_strip(ax)
ax.set_title('Correlation between \npop. coupling and CCA weights',fontsize=11)
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,savedir,'Corr_PopCoupling_CCA_weights_%dsessions' % nSessions,formats=['png'])


#%% 
 #####   #####     #       ######  ####### ######   #####  
#     # #     #   # #      #     # #     # #     # #     # 
#       #        #   #     #     # #     # #     # #       
#       #       #     #    ######  #     # ######   #####  
#       #       #######    #       #     # #             # 
#     # #     # #     #    #       #     # #       #     # 
 #####   #####  #     #    #       ####### #        #####  

#%% 

#%% Are the CCA correlations higher for choristers or soloists across areas?
kFold               = 5
n_components        = 20
nStim               = 16
nmodelfits          = 5
nsampleneurons      = 50
# nsampleneurons      = 100
maxnoiselevel       = 20
nbins_popcoupling   = 5
couplinglabels      = ['0-20%','20-40%','40-60%','60-80%','80-100%']
# couplinglabels      = ['Soloists','Choristers']
areas               = np.array(['V1', 'PM'])
CCA_corrtest        = np.full((nbins_popcoupling,n_components,nSessions,nStim),np.nan)
prePCA              = 25
# prePCA              = None

#%% Fit:
for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
    binedges_popcoupling   = np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,nbins_popcoupling+1))

    for ibin in range(nbins_popcoupling):
        idx_areax           = np.where(np.all((ses.celldata['roi_name']==areas[0],
                                        ses.celldata['pop_coupling']>binedges_popcoupling[ibin],
                                        ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1],
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==areas[1],
                                        ses.celldata['pop_coupling']>binedges_popcoupling[ibin],
                                        ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1],
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            
        if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:
            # for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),total=nStim,desc='Fitting CCA model'):    # iterate over sessions
            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations
                idx_T               = ses.trialdata['stimCond']==stim
            
                X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
                Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
                
                [g,_] = CCA_subsample(X,Y,nN=nsampleneurons,resamples=nmodelfits,kFold=kFold,prePCA=prePCA,
                    n_components=np.min([n_components,nsampleneurons]))
                CCA_corrtest[ibin,:,ises,istim] = g

                # [g,_] = CCA_subsample_it(X,Y,nN=nsampleneurons,resamples=nmodelfits,kFold=kFold,prePCA=prePCA,
                #                          n_components=np.min([n_components,nsampleneurons]))
                # CCA_corrtest[ibin,:,ises,istim] = g

# CCA_validate_iterative(X,Y,prePCA=prePCA,n_components=n_components)

#%%
clrs_couplingbins = sns.color_palette('magma',nbins_popcoupling)
fig, axes = plt.subplots(1,1,figsize=(3,3))

ax = axes
handles = []
# for iapl, arealabelpair in enumerate(ncouplingpairs):
for icpl in range(nbins_popcoupling):
    # ax.plot(np.arange(n_components),np.nanmean(CCA_corrtest[iapl,:,:,:],axis=(1,2)),
            # color=clrs_arealabelpairs[iapl],linewidth=2)
    iapldata = CCA_corrtest[icpl,:,:,:].reshape(n_components,-1)
    handles.append(shaded_error(x=np.arange(n_components),
                                y=iapldata.T,
                                # error='sem',color='k',alpha=0.3,ax=ax))
                                error='sem',color=clrs_couplingbins[icpl],alpha=0.3,ax=ax))
ax.set_xticks(np.arange(0,n_components+5,5))
ax.set_xticklabels(np.arange(0,n_components+5,5)+1)
ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest,axis=(2,3))),1)])
ax.set_yticks([0,ax.get_ylim()[1]/2,ax.get_ylim()[1]])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel('Correlation')
ax.legend(handles,couplinglabels,loc='upper right',reverse=True,fontsize=7,frameon=True,title='pop. coupling')
sns.despine(top=True,right=True,offset=1,trim=True)
my_savefig(fig,savedir,'CCA_V1PM_PopCoupling_testcorr_%dsessions' % (nSessions),formats=['png'])
