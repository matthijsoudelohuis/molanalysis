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

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\CCA\\')

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

#%% 
 #####   #####     #       #     # ####### ###  #####  #     # #######  #####  
#     # #     #   # #      #  #  # #        #  #     # #     #    #    #     # 
#       #        #   #     #  #  # #        #  #       #     #    #    #       
#       #       #     #    #  #  # #####    #  #  #### #######    #     #####  
#       #       #######    #  #  # #        #  #     # #     #    #          # 
#     # #     # #     #    #  #  # #        #  #     # #     #    #    #     # 
 #####   #####  #     #     ## ##  ####### ###  #####  #     #    #     #####  

#%% 


#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
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
nmodelfits          = 1
nsampleneurons      = 50
# nsampleneurons      = 100
maxnoiselevel       = 20
ncouplingpairs      = 2
# couplinglabels      = ['Choristers','Soloists']
couplinglabels      = ['Soloists','Choristers']
areas               = np.array(['V1', 'PM'])
CCA_corrtest        = np.full((ncouplingpairs,n_components,nSessions,nStim),np.nan)
prePCA              = 25
# prePCA              = None

#%% Fit:
# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
for ises,ses in enumerate(sessions):    # iterate over sessions
    binedges_popcoupling   = np.percentile(ses.celldata['pop_coupling'],[25,75])

    for icouple in range(ncouplingpairs):

        if icouple==0:
            idx_areax           = np.where(np.all((ses.celldata['roi_name']==areas[0],
                                            ses.celldata['pop_coupling']<binedges_popcoupling[0],
                                            ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['roi_name']==areas[1],
                                        ses.celldata['pop_coupling']<binedges_popcoupling[0],
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
        elif icouple==1:
            idx_areax           = np.where(np.all((ses.celldata['roi_name']==areas[0],
                                            ses.celldata['pop_coupling']>binedges_popcoupling[1],
                                            ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['roi_name']==areas[1],
                                        ses.celldata['pop_coupling']>binedges_popcoupling[1],
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]

        if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:
            for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),total=nStim,desc='Fitting CCA model'):    # iterate over sessions
                idx_T               = ses.trialdata['stimCond']==stim
            
                X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
                Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
                
                X                   = zscore(X,axis=0)  #Z score activity for each neuron
                Y                   = zscore(Y,axis=0)

                # [g,_] = CCA_subsample(X,Y,nN=nsampleneurons,resamples=nmodelfits,kFold=kFold,prePCA=prePCA,n_components=np.min([n_components,nsampleneurons]))
                # CCA_corrtest[icouple,:,ises,istim] = g

                [g,_] = CCA_subsample_it(X,Y,nN=nsampleneurons,resamples=nmodelfits,kFold=kFold,prePCA=prePCA,n_components=np.min([n_components,nsampleneurons]))
                CCA_corrtest[icouple,:,ises,istim] = g

# CCA_validate_iterative(X,Y,prePCA=prePCA,n_components=n_components)

#%%
clrs_couplingbins = sns.color_palette('magma',ncouplingpairs)
fig, axes = plt.subplots(1,1,figsize=(4,4))

ax = axes
handles = []
# for iapl, arealabelpair in enumerate(ncouplingpairs):
for icpl in range(ncouplingpairs):
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
ax.legend(handles,couplinglabels,loc='upper right',frameon=False,fontsize=9)
sns.despine(top=True,right=True,offset=1,trim=True)
# my_savefig(fig,savedir,'CCA_V1PM_pops_labeled_testcorr_%dsessions' % (nSessions),formats=['png'])
