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
from scipy import stats
from tqdm import tqdm
from sklearn.cross_decomposition import CCA
from sklearn.impute import SimpleImputer

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.rf_lib import filter_nearlabeled
from utils.regress_lib import *
from utils.CCAlib import *
from utils.RRRlib import *
from utils.pair_lib import value_matching

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\CCA\\Labeling\\')

#%% 
areas       = ['V1','PM']
nareas      = len(areas)

# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
calciumversion = 'dF'
for ises in range(nSessions):
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion=calciumversion,keepraw=False)
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,
                            #    keepraw=False,filter_hp=0.01)
                               keepraw=False)

t_axis = sessions[0].t_axis

#%% 
 #####   #####     #       #     #    #   ######  #     #    #     # ####### ###  #####  #     # #######  #####  
#     # #     #   # #      #     #   ##   #     # ##   ##    #  #  # #        #  #     # #     #    #    #     # 
#       #        #   #     #     #  # #   #     # # # # #    #  #  # #        #  #       #     #    #    #       
#       #       #     #    #     #    #   ######  #  #  #    #  #  # #####    #  #  #### #######    #     #####  
#       #       #######     #   #     #   #       #     #    #  #  # #        #  #     # #     #    #          # 
#     # #     # #     #      # #      #   #       #     #    #  #  # #        #  #     # #     #    #    #     # 
 #####   #####  #     #       #     ##### #       #     #     ## ##  ####### ###  #####  #     #    #     #####  


#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
n_components        = 20
nStim               = 16
nmodelfits          = 10
minsampleneurons    = 10
maxnoiselevel       = 20
filter_nearby       = True
kFold               = 5
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
weights_CCA         = np.full((n_components,len(arealabels),nSessions,nStim,nmodelfits),np.nan)
cancorr_CCA         = np.full((n_components,nSessions,nStim,nmodelfits),np.nan)
do_cv_cca           = False

#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)
# model_CCA           = PLSCanonical(n_components=n_components,scale = False, max_iter = 1000)
for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=25)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<maxnoiselevel,
                                          idx_nearby),axis=0)) for i in arealabels])
    
    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    idx_N_all = np.empty(len(arealabels),dtype=object)
    for ial, al in enumerate(arealabels):
        idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    
    for imf in range(nmodelfits):
        idx_areax           = np.concatenate((np.random.choice(idx_N_all[0],nsampleneurons,replace=False),
                                            np.random.choice(idx_N_all[1],nsampleneurons,replace=False)))
        idx_areay           = np.concatenate((np.random.choice(idx_N_all[2],nsampleneurons,replace=False),
                                            np.random.choice(idx_N_all[3],nsampleneurons,replace=False)))
        assert len(idx_areax)==2*nsampleneurons and len(idx_areay)==2*nsampleneurons

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
        
            #on residual tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            X                   -= np.mean(X,axis=1,keepdims=True)
            Y                   -= np.mean(Y,axis=1,keepdims=True)

            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0,nan_policy='omit')

            # X       = si.fit_transform(X)
            # Y       = si.fit_transform(Y)

            # Fit CCA MODEL:
            model_CCA.fit(X,Y)
            
            weights_CCA[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[:nsampleneurons,:]),axis=0)
            weights_CCA[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[nsampleneurons:,:]),axis=0)

            weights_CCA[:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[:nsampleneurons,:]),axis=0)
            weights_CCA[:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[nsampleneurons:,:]),axis=0)

            # weights_CCA[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
            # weights_CCA[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

            # weights_CCA[:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[:nsampleneurons,:]),axis=0)
            # weights_CCA[:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[nsampleneurons:,:]),axis=0)

            if do_cv_cca:#Implementing cross validation
                kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)
                corr_test = np.full((n_components,kFold),np.nan)
                # for train_index, test_index in kf.split(X):
                for ikf,(train_index, test_index) in enumerate(kf.split(X)):
                    X_train , X_test = X[train_index,:],X[test_index,:]
                    Y_train , Y_test = Y[train_index,:],Y[test_index,:]
                    
                    model_CCA.fit(X_train,Y_train)

                    X_c, Y_c = model_CCA.transform(X_test,Y_test)
                    for icomp in range(n_components):
                        corr_test[icomp,ikf] = np.corrcoef(X_c[:,icomp],Y_c[:,icomp], rowvar = False)[0,1]
                cancorr_CCA[:,ises,istim,imf] = np.nanmean(corr_test,axis=1)

#%%
clrs_areas = get_clr_areas(areas)
clrs_arealabels = get_clr_area_labeled(arealabels)

#%% 

#%% 
fig,axes = plt.subplots(1,1,figsize=(3,2.5),sharex=True,sharey=True)

varversion = 'stim'
# varversion = 'session'

ax = axes
ialdata = np.nanmean(cancorr_CCA[:,:,:,:],axis=(-1,-2))
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(nSessions)

mindim = np.where(meantoplot<0.1)[0][0]-1

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
        elinewidth=1,markersize=8,color='grey')

ax.set_xlabel('Dimension')
ax.set_ylabel('Canonical Correlation')

ax.set_yticks(np.arange(0,1.1,0.2))
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)
ax.axhline(0.1,linestyle='--',color='k')
sns.despine(top=True,right=True,offset=2,trim=False)
# my_savefig(fig,savedir,'CCA_V1PM_labeled_testcorr_%dsessions_%s' % (nSessions,varversion),formats=['png'])


#%% 
fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)

varversion = 'stim'
# varversion = 'session'

ax = axes[0]
for ial,al in enumerate(arealabels[:2]):
    
    if varversion == 'session':
        ialdata = np.nanmean(weights_CCA[:,ial,:,:,:],axis=(-1,-2))
    elif varversion == 'stim':
        ialdata = np.nanmean(weights_CCA[:,ial,:,:,:],axis=(-1))
        ialdata = ialdata.reshape(n_components,nSessions*nStim)
    meantoplot = np.nanmean(ialdata,axis=1)
    errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

    ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=clrs_arealabels[ial])
    # for ises in range(nSessions):
        # ax.plot(ialdata[:,ises],color=clrs_arealabels[ial],alpha=1)
ax.axvline(x=mindim,color='grey',linestyle='--')
ax.text(mindim+0.5,ax.get_ylim()[1],'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_ylabel('|Loadings|')
ax.legend(frameon=False,loc='lower center')
ax.set_title('V1')

ax = axes[1]
for ial,al in enumerate(arealabels[2:]):
    if varversion == 'session':
        ialdata = np.nanmean(weights_CCA[:,ial+2,:,:,:],axis=(-1,-2))
    elif varversion == 'stim':
        ialdata = np.nanmean(weights_CCA[:,ial+2,:,:,:],axis=(-1))
        ialdata = ialdata.reshape(n_components,nSessions*nStim)
    meantoplot = np.nanmean(ialdata,axis=1)
    errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

    ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=clrs_arealabels[ial+2])
    # ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,color=clrs_arealabels[ial+2])
    ax.set_xticks(range(n_components))
ax.axvline(x=mindim,color='grey',linestyle='--')
ax.text(mindim+0.5,ax.get_ylim()[1],'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_title('PM')

ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)

ax.legend(frameon=False,loc='lower center')
sns.despine(top=True,right=True,offset=2,trim=False)
my_savefig(fig,savedir,'CCA_V1PM_labeled_loadings_%dsessions_%s' % (nSessions,varversion),formats=['png'])
# my_savefig(fig,savedir,'CCA_V1PM_labeled_weights_v2_%dsessions' % nSessions,formats=['png'])


#%% 
varversion = 'stim'
# varversion = 'session'

fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)

ax = axes[0]
if varversion == 'session':
    ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1,-2))
elif varversion == 'stim':
    ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(n_components,nSessions*nStim)
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['V1']))
for icomp in range(n_components):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
    # pval = pval*n_components (to correct for multiple comparisons)
    # print(pval)
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)
# ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,color=get_clr_areas(['V1']))
ax.axvline(x=mindim,color='grey',linestyle='--')
ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_ylabel(r'$\Delta$|Loadings|   (Lab-Unl)')
ax.set_xlabel('Dimension')
ax.set_title('V1')
ax.axhline(y=0,color='k',linestyle='--')

ax = axes[1]
if varversion == 'session':
    ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1,-2))
elif varversion == 'stim':
    ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(n_components,nSessions*nStim)
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['PM']))
for icomp in range(n_components):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)

ax.axvline(x=mindim,color='grey',linestyle='--')
ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_title('PM')
ax.axhline(y=0,color='k',linestyle='--')

ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)

sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'CCA_V1PM_labeled_deltaloadings_%dsessions_%s' % (nSessions,varversion),formats=['png'])


#%% 

 #####  ####### #     # ####### ######  ####### #        #####  
#     # #     # ##    #    #    #     # #     # #       #     # 
#       #     # # #   #    #    #     # #     # #       #       
#       #     # #  #  #    #    ######  #     # #        #####  
#       #     # #   # #    #    #   #   #     # #             # 
#     # #     # #    ##    #    #    #  #     # #       #     # 
 #####  ####### #     #    #    #     # ####### #######  #####  


#%% Controls: Are the weight difference due to some confounding difference between the populations?
arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
valuematchingvars   = np.array([None,'noise_level','event_rate','skew', 'meanF'])
valuematchinglabels = np.array(['Control','Noise Level\n(Rupprecht et al. 2021)','Activity Level','Skewness', 'Mean F0'])
nvaluevars          = len(valuematchingvars)
nmatchbins          = 10 #number of hist bins for value matching (resolution of matching)

nmodelfits          = 2
weights_CCA         = np.full((nvaluevars,n_components,len(arealabels),nSessions,nStim,nmodelfits),np.nan)

#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=25)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<maxnoiselevel,
                                          idx_nearby),axis=0)) for i in arealabels])
    
    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    for ival, valuematching in enumerate(valuematchingvars):
        idx_N_all = np.empty(len(arealabels),dtype=object)
        for ial, al in enumerate(arealabels):
            idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
                                    ses.celldata['noise_level']<maxnoiselevel,	
                                    idx_nearby),axis=0))[0]
            
        if valuematching is not None:
            #Get value to match from celldata for V1 matching
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_N_all[0],idx_N_all[1]))
            group       = np.concatenate((np.zeros(len(idx_N_all[0])),np.ones(len(idx_N_all[1]))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_N_all[0]   = np.intersect1d(idx_N_all[0],idx_sub) #recover subset from idx_joint
            idx_N_all[1]   = np.intersect1d(idx_N_all[1],idx_sub)

            #Get value to match from celldata for PM matching
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_N_all[2],idx_N_all[3]))
            group       = np.concatenate((np.zeros(len(idx_N_all[2])),np.ones(len(idx_N_all[3]))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_N_all[2]   = np.intersect1d(idx_N_all[2],idx_sub) #recover subset from idx_joint
            idx_N_all[3]   = np.intersect1d(idx_N_all[3],idx_sub)

        nsampleneurons      = np.min([len(idx_N_all[i]) for i in range(len(idx_N_all))])
        if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
            continue

        for imf in range(nmodelfits):
            idx_areax           = np.concatenate((np.random.choice(idx_N_all[0],nsampleneurons,replace=False),
                                                np.random.choice(idx_N_all[1],nsampleneurons,replace=False)))
            idx_areay           = np.concatenate((np.random.choice(idx_N_all[2],nsampleneurons,replace=False),
                                                np.random.choice(idx_N_all[3],nsampleneurons,replace=False)))
            assert len(idx_areax)==2*nsampleneurons and len(idx_areay)==2*nsampleneurons

            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
                idx_T               = ses.trialdata['stimCond']==stim
            
                #on residual tensor during the response:
                X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
                Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
                
                X                   -= np.mean(X,axis=1,keepdims=True)
                Y                   -= np.mean(Y,axis=1,keepdims=True)

                X                   = X.reshape(len(idx_areax),-1).T
                Y                   = Y.reshape(len(idx_areay),-1).T

                X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
                Y                   = zscore(Y,axis=0,nan_policy='omit')

                X       = si.fit_transform(X)
                Y       = si.fit_transform(Y)

                # Fit CCA MODEL:
                model_CCA.fit(X,Y)
                
                weights_CCA[ival,:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[:nsampleneurons,:]),axis=0)
                weights_CCA[ival,:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[nsampleneurons:,:]),axis=0)

                weights_CCA[ival,:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[:nsampleneurons,:]),axis=0)
                weights_CCA[ival,:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[nsampleneurons:,:]),axis=0)

                # weights_CCA[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
                # weights_CCA[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

                # weights_CCA[:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[:nsampleneurons,:]),axis=0)
                # weights_CCA[:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[nsampleneurons:,:]),axis=0)

#%% 
#%% 
clrs = sns.color_palette("colorblind", 8)
sumdim = 4

datatoplot1 = weights_CCA[:,:,1,:,:,:] - weights_CCA[:,:,0,:,:,:]
datatoplot2 = weights_CCA[:,:,3,:,:,:] - weights_CCA[:,:,2,:,:,:]

datatoplot1 = weights_CCA[:,:,1,:,:,:] / weights_CCA[:,:,0,:,:,:]
datatoplot2 = weights_CCA[:,:,3,:,:,:] / weights_CCA[:,:,2,:,:,:]

datatoplot = np.concatenate((datatoplot1[:,:,np.newaxis,:,:,:],datatoplot2[:,:,np.newaxis,:,:,:]),axis=2) #Get the difference between labeled and unlabeled
datatoplot = np.nanmean(datatoplot,axis=(2)) #average across areas
datatoplot = np.nanmean(datatoplot,axis=(-1)) #average modelfits
# datatoplot = np.nansum(datatoplot[:,:sumdim,:,:],axis=1) #sum weight diff for first sumdim dims
datatoplot = np.nanmean(datatoplot[:,:sumdim,:,:],axis=1) #sum weight diff for first sumdim dims
datatoplot = np.reshape(datatoplot,(nvaluevars,nSessions*nStim)) #stretch sessions * stim
# Dim1: valuematching, dim2: v1/PM, dim3: datasets
datatoplot[datatoplot == 0] = np.nan

N       = datatoplot.shape[1]
N       = nSessions

fig,axes = plt.subplots(1,1,figsize=(3.5,3.5),sharex=True,sharey=True)
ax = axes
for ival,valuematching in enumerate(valuematchingvars):
    ax.errorbar(ival,np.nanmean(datatoplot[ival,:]),yerr=np.nanstd(datatoplot[ival,:])/np.sqrt(N),fmt='o',markerfacecolor='k',
            elinewidth=2,markersize=8,color=clrs[ival])

ax.set_ylabel('|Loadings|   (Ratio Lab-Unl)')
ax.set_title('Subsampling control for:')
ax.set_yticks([1,1.1,1.2])
ax.set_xticks(np.arange(len(valuematchingvars)))
sns.despine(top=True,right=True,offset=1,trim=True)
ax.set_xticklabels(valuematchinglabels,fontsize=8,rotation=60,ha='right',va='top')
if ival==0:
    ax.set_ylabel('|Weight|   (Lab-Unl)')
ax.axhline(y=1,color='k',linestyle='--')
plt.tight_layout()
my_savefig(fig,savedir,'CCA_V1PM_labeled_weights_%dsessions_Controls' % (nSessions),formats=['png'])


#%% 
# sumdim = [0,2,3,4,5,6]
sumdim = 5

datatoplot1 = weights_CCA[:,:,1,:,:,:] - weights_CCA[:,:,0,:,:,:]
datatoplot2 = weights_CCA[:,:,3,:,:,:] - weights_CCA[:,:,2,:,:,:]

# datatoplot1 = weights_CCA[:,:,1,:,:,:] / weights_CCA[:,:,0,:,:,:]
# datatoplot2 = weights_CCA[:,:,3,:,:,:] / weights_CCA[:,:,2,:,:,:]

# datatoplot = np.diff(weights_CCA,axis=2)[:,:,[0,2],:,:,:] #Get the difference between labeled and unlabeled
datatoplot = np.concatenate((datatoplot1[:,:,np.newaxis,:,:,:],datatoplot2[:,:,np.newaxis,:,:,:]),axis=2) #Get the difference between labeled and unlabeled

datatoplot = np.nanmean(datatoplot,axis=(-1)) #average modelfits
datatoplot = np.nansum(datatoplot[:,:sumdim,:,:,:],axis=1) #sum weight diff for first sumdim dims
# datatoplot = np.nanmean(datatoplot[:,:sumdim,:,:,:],axis=1) #sum weight diff for first sumdim dims
datatoplot = np.reshape(datatoplot,(nvaluevars,2,nSessions*nStim)) #stretch sessions * stim
# Dim1: valuematching, dim2: v1/PM, dim3: datasets
datatoplot[datatoplot == 0] = np.nan

N       = datatoplot.shape[2]

fig,axes = plt.subplots(1,2,figsize=(4,2.5),sharex=True,sharey=True)

for iarea,area in enumerate(areas):
    ax = axes[iarea]
    # for ival,valuematching in enumerate(valuematchingvars):
    #     data = datatoplot[ival,iarea,:]
    #     data = data[~np.isnan(data)]
    #     ax.boxplot(data,positions=[ival],whis=[5,95],showfliers=False,color=clrs_arealabels[iarea])
    #     ax.boxplot(data,positions=[ival],whis=[5,95],showfliers=False)
    sns.boxplot(datatoplot[:,iarea,:].T,ax=ax,whis=[5,95],showfliers=False)
        # ax.scatter(np.zeros(N)+np.random.randn(N)*0.1+ival,datatoplot[ival,iarea,:],s=8,color=clrs_arealabels[iarea])
        # ax.errorbar(ival,np.nanmean(datatoplot[ival,iarea,:]),yerr=np.nanstd(datatoplot[ival,iarea,:])/np.sqrt(N),fmt='o',markerfacecolor='k',
        #             elinewidth=2,markersize=8,color=clrs_arealabels[iarea])
    # t,p = stats.ttest_rel(datatoplot[0,iarea,:],datatoplot[1,iarea,:],nan_policy='omit')
    # # add_stat_annotation(ax, 0, 1, 0.15, p, h=None)
    # add_stat_annotation(ax, 0, 1, 1.15, p, h=None)
    # print('With vs Without behavior (%s): t=%1.3f, p=%1.3f' % (area,t,p))

    ax.set_xticks(np.arange(len(valuematchingvars)))
    ax.set_xticklabels(valuematchinglabels,fontsize=8,rotation=45)
    if ival==0:
        ax.set_ylabel('|Weight|   (Lab-Unl)')
    ax.set_ylim([-0.01,0.08])
    # ax.set_ylim([-0.1,0.25])
    ax.set_title(area)
    ax.axhline(y=0,color='k',linestyle='--')
    # ax.axhline(y=1,color='k',linestyle='--')
# sns.despine(top=True,right=True,offset=1,trim=True)
#  ax.scatter(1,PMlabdiff,s=20,color='k')
plt.tight_layout()
# my_savefig(fig,savedir,'CCA_V1PM_labeled_weights_%dsessions_Controls' % (nSessions),formats=['png'])



#%% Dimensionality of the labeled and unlabeled populations: (estimated using PCA)

#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
n_components        = 20
nmodelfits          = 10
arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
pca_ev              = np.full((n_components,len(arealabels),nSessions,nStim,nmodelfits),np.nan)
pca                 = PCA(n_components=n_components)
minsampleneurons    = 20

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=25)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<maxnoiselevel,
                                          idx_nearby),axis=0)) for i in arealabels])
    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue
    nsampleneurons = 20

    for ial, al in enumerate(arealabels):
        idx_area           = np.where(np.all((ses.celldata['arealabel']==al,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        for imf in range(nmodelfits):
            idx_area_sub = np.random.choice(idx_area,nsampleneurons,replace=False)
            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
                idx_T               = ses.trialdata['stimCond']==stim
                
                #on residual tensor during the response:
                X                   = sessions[ises].tensor[np.ix_(idx_area_sub,idx_T,idx_resp)]
                X                   -= np.mean(X,axis=1,keepdims=True)
                X                   = X.reshape(len(idx_area_sub),-1).T
                X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron

                pca                 = PCA(n_components=np.min([nsampleneurons,n_components]))
                pca.fit(X)
                pca_ev[:np.min([nsampleneurons,n_components]),ial,ises,istim,imf] = pca.explained_variance_ratio_

#%% Show PCA dimensionality: 

fig,ax = plt.subplots(1,1,figsize=(4,3))
for ial,al in enumerate(arealabels):
    # ax.plot(np.nanmean(pca_ev[:,ial,:,:,:],axis=(-1,-2,-3)),label=al,color=clrs_arealabels[ial])
    ax.plot(np.nanmean(np.cumsum(pca_ev[:,ial,:,:,:],axis=0),axis=(-1,-2,-3)),label=al,color=clrs_arealabels[ial])

ax.set_xlim([0,n_components])
ax.set_ylim([0,1])
ax.legend(frameon=False,loc='best')
ax.set_xlabel('PC Dimension')
ax.set_ylabel('Explained variance')
sns.despine(top=True,right=True,offset=1,trim=True)
plt.tight_layout()
# my_savefig(fig,savedir,'PCA_V1PM_labeled_GRGN_%dsessions' % (nSessions),formats=['png'])


#%% Now control for the dimensionality: 

#%% 
def equalize_eigenspectra(X, Y, mode='geometric'):
    # Center the data
    Xc = X - np.mean(X, axis=0)
    Yc = Y - np.mean(Y, axis=0)

    # Compute SVD (PCA basis)
    Ux, Sx, VxT = np.linalg.svd(Xc, full_matrices=False)
    Uy, Sy, VyT = np.linalg.svd(Yc, full_matrices=False)

    # Variances (squared singular values)
    var_x = Sx**2
    var_y = Sy**2

    # Compute target eigenspectrum
    if mode == 'geometric':
        target_var = np.sqrt(var_x * var_y)
    elif mode == 'arithmetic':
        target_var = 0.5 * (var_x + var_y)
    else:
        raise ValueError("Unsupported mode. Choose 'geometric' or 'arithmetic'.")

    # New singular values to match target variance
    Sx_new = np.sqrt(target_var)
    Sy_new = np.sqrt(target_var)

    # Reconstruct matrices with matched eigenspectra
    X_new = (Ux * Sx_new) @ VxT
    Y_new = (Uy * Sy_new) @ VyT

    # Re-add the original mean
    X_new += np.mean(X, axis=0)
    Y_new += np.mean(Y, axis=0)

    return X_new, Y_new

#%% 

# Generate example data
np.random.seed(0)
M, N = 100, 20

# X with strong signal
X1 = np.random.randn(M, N) @ np.diag(np.linspace(5, 1, N))

# X2 with more uniform variance
X2 = np.random.randn(M, N)

# Compute eigenspectra before
def get_eigenspectrum(A):
    Ac = A - np.mean(A, axis=0)
    _, S, _ = np.linalg.svd(Ac, full_matrices=False)
    return S**2

var_X1_before = get_eigenspectrum(X1)
var_X2_before = get_eigenspectrum(X2)

# Equalize eigenspectra
X1_eq, X2_eq = equalize_eigenspectra(X1, X2)

# Compute eigenspectra after
var_X1_after = get_eigenspectrum(X1_eq)
var_X2_after = get_eigenspectrum(X2_eq)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(var_X1_before, label='X Before', marker='o')
plt.plot(var_X2_before, label='Y Before', marker='o')
plt.plot(var_X1_after, label='X After', marker='x')
plt.plot(var_X2_after, label='Y After', marker='x')

plt.title('Eigenspectra Before and After Equalization')
plt.xlabel('Principal Component')
plt.ylabel('Variance (Eigenvalue)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% 

minsampleneurons    = 10
pca                 = PCA(n_components=n_components)
ev_thr              = 0.8
nmodelfits          = 1

model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)
# model_CCA           = PLSCanonical(n_components=n_components,scale = False, max_iter = 1000)
for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=25)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<maxnoiselevel,
                                          idx_nearby),axis=0)) for i in arealabels])
    
    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    idx_N_all = np.empty(len(arealabels),dtype=object)
    for ial, al in enumerate(arealabels):
        idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    
    for imf in range(nmodelfits):
        
        idx_areax1           = np.random.choice(idx_N_all[0],nsampleneurons,replace=False)
        idx_areax2           = np.random.choice(idx_N_all[1],nsampleneurons,replace=False)
        idx_areay1           = np.random.choice(idx_N_all[2],nsampleneurons,replace=False)
        idx_areay2           = np.random.choice(idx_N_all[3],nsampleneurons,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        # for istim,stim in enumerate([np.unique(ses.trialdata['stimCond'])[0]]): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
        
            X1              = sessions[ises].tensor[np.ix_(idx_areax1,idx_T,idx_resp)]
            X2              = sessions[ises].tensor[np.ix_(idx_areax2,idx_T,idx_resp)]
            Y1              = sessions[ises].tensor[np.ix_(idx_areay1,idx_T,idx_resp)]
            Y2              = sessions[ises].tensor[np.ix_(idx_areay2,idx_T,idx_resp)]

            X1              -= np.mean(X1,axis=1,keepdims=True)
            X2              -= np.mean(X2,axis=1,keepdims=True)
            Y1              -= np.mean(Y1,axis=1,keepdims=True)
            Y2              -= np.mean(Y2,axis=1,keepdims=True)

            X1              = X1.reshape(len(idx_areax1),-1).T
            X2              = X2.reshape(len(idx_areax2),-1).T
            Y1              = Y1.reshape(len(idx_areay1),-1).T
            Y2              = Y2.reshape(len(idx_areay2),-1).T

            X1              = zscore(X1,axis=0,nan_policy='omit')  #Z score activity for each neuron
            X2              = zscore(X2,axis=0,nan_policy='omit')
            Y1              = zscore(Y1,axis=0,nan_policy='omit')
            Y2              = zscore(Y2,axis=0,nan_policy='omit')

            [X1, X2]    = equalize_eigenspectra(X1, X2, mode='geometric')
            [Y1, Y2]    = equalize_eigenspectra(Y1, Y2, mode='geometric')

            #on residual tensor during the response:
            X                   = np.concatenate((X1,X2),axis=1)
            Y                   = np.concatenate((Y1,Y2),axis=1)
            
            # Fit CCA MODEL:
            model_CCA.fit(X,Y)
            
            weights_CCA[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[:nsampleneurons,:]),axis=0)
            weights_CCA[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[nsampleneurons:,:]),axis=0)

            weights_CCA[:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[:nsampleneurons,:]),axis=0)
            weights_CCA[:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[nsampleneurons:,:]),axis=0)

#%% 

#%% Plot the results: 
varversion = 'stim'
# varversion = 'session'

fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)

ax = axes[0]
if varversion == 'session':
    ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1,-2))
elif varversion == 'stim':
    ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(n_components,nSessions*nStim)
    
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['V1']))
for icomp in range(n_components):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
    # pval = pval*n_components (to correct for multiple comparisons)
    # print(pval)
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)
# ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,color=get_clr_areas(['V1']))
ax.axvline(x=mindim,color='grey',linestyle='--')
ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_ylabel(r'$\Delta$|Loadings|   (Lab-Unl)')
ax.set_xlabel('Dimension')
ax.set_title('V1')
ax.axhline(y=0,color='k',linestyle='--')

ax = axes[1]
if varversion == 'session':
    ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1,-2))
elif varversion == 'stim':
    ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(n_components,nSessions*nStim)
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['PM']))
for icomp in range(n_components):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)

ax.axvline(x=mindim,color='grey',linestyle='--')
ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_title('PM')
ax.axhline(y=0,color='k',linestyle='--')

ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)

sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'CCA_V1PM_labeled_deltaweights_%dsessions_%s_equaleigenspectrum' % (nSessions,varversion),formats=['png'])


#%% 
   #    #          #     # ####### ###  #####  #     # #######  #####  
  # #   #          #  #  # #        #  #     # #     #    #    #     # 
 #   #  #          #  #  # #        #  #       #     #    #    #       
#     # #          #  #  # #####    #  #  #### #######    #     #####  
####### #          #  #  # #        #  #     # #     #    #          # 
#     # #          #  #  # #        #  #     # #     #    #    #     # 
#     # #######     ## ##  ####### ###  #####  #     #    #     #####  

#%% 
# areas       = ['V1','PM']
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

# %% 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas)

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion=calciumversion,keepraw=False)
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,keepraw=False)

t_axis = sessions[0].t_axis



#%% Are the weights higher for V1lab or PMlab than unlabeled neurons to the other area?
n_components        = 20
nStim               = 16
nmodelfits          = 2
minsampleneurons    = 10
maxnoiselevel       = 20
filter_nearby       = True
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

diffarea = 'RSPunl'
arealabels          = np.array(['V1unl', 'V1lab', 'ALunl'])
arealabels          = np.array(['V1unl', 'V1lab', 'RSPunl'])
weights_CCA_V1AL    = np.full((n_components,len(arealabels),nSessions,nStim,nmodelfits),np.nan)

#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=50)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    popsizes = [np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<maxnoiselevel,
                                          idx_nearby),axis=0)) for i in arealabels]
    popsizes[2] /= 2 #make sure that ALunl has twice the number of nsampleneurons
    nsampleneurons      = int(np.min(popsizes))
    
    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    idx_N_all = np.empty(len(arealabels),dtype=object)
    for ial, al in enumerate(arealabels):
        idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    
    for imf in range(nmodelfits):
        idx_areax           = np.concatenate((np.random.choice(idx_N_all[0],nsampleneurons,replace=False),
                                            np.random.choice(idx_N_all[1],nsampleneurons,replace=False)))
        idx_areay           = np.random.choice(idx_N_all[2],nsampleneurons*2,replace=False)
        assert len(idx_areax)==2*nsampleneurons and len(idx_areay)==2*nsampleneurons

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
        
            #on tensor during the response:
            # X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)].reshape(len(idx_areax),-1).T
            # Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)].reshape(len(idx_areay),-1).T
            
            #on residual tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            X                   -= np.mean(X,axis=1,keepdims=True)
            Y                   -= np.mean(Y,axis=1,keepdims=True)

            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            X                   = zscore(X,axis=0)  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0)

            # Fit CCA MODEL:
            model_CCA.fit(X,Y)
            
            weights_CCA_V1AL[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
            weights_CCA_V1AL[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)


#%% Are the weights higher for V1lab or PMlab than unlabeled neurons to the other area?
arealabels          = np.array(['PMunl', 'PMlab', 'ALunl'])
arealabels          = np.array(['PMunl', 'PMlab', 'RSPunl'])
weights_CCA_PMAL    = np.full((n_components,len(arealabels),nSessions,nStim,nmodelfits),np.nan)

#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=50)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    popsizes = [np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<maxnoiselevel,
                                          idx_nearby),axis=0)) for i in arealabels]
    popsizes[2] /= 2
    nsampleneurons      = int(np.min(popsizes))
    
    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    idx_N_all = np.empty(len(arealabels),dtype=object)
    for ial, al in enumerate(arealabels):
        idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    
    for imf in range(nmodelfits):
        idx_areax           = np.concatenate((np.random.choice(idx_N_all[0],nsampleneurons,replace=False),
                                            np.random.choice(idx_N_all[1],nsampleneurons,replace=False)))
        idx_areay           = np.random.choice(idx_N_all[2],nsampleneurons*2,replace=False)
        assert len(idx_areax)==2*nsampleneurons and len(idx_areay)==2*nsampleneurons

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
        
            #on tensor during the response:
            # X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)].reshape(len(idx_areax),-1).T
            # Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)].reshape(len(idx_areay),-1).T
            
            #on residual tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            X                   -= np.mean(X,axis=1,keepdims=True)
            Y                   -= np.mean(Y,axis=1,keepdims=True)

            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            X                   = zscore(X,axis=0)  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0)

            # Fit CCA MODEL:
            model_CCA.fit(X,Y)
            
            weights_CCA_PMAL[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
            weights_CCA_PMAL[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)


#%% 
varversion = 'stim'
# varversion = 'session'

fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)

ax = axes[0]
if varversion == 'session':
    ialdata = np.nanmean(weights_CCA_V1AL[:,1,:,:,:] - weights_CCA_V1AL[:,0,:,:,:],axis=(-1,-2))
elif varversion == 'stim':
    ialdata = np.nanmean(weights_CCA_V1AL[:,1,:,:,:] - weights_CCA_V1AL[:,0,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(n_components,nSessions*nStim)
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['V1']))
for icomp in range(n_components):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
    # pval = pval*n_components #(to correct for multiple comparisons)
    # print(pval)
    if pval < 0.05:
        # ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.005*math.copysign(1, ttest)-0.002,'*',color='k',markersize=8)
# ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_ylabel(r'$\Delta$|Weight|   (V1Lab-V1Unl)')
ax.set_xlabel('Dimension')
ax.set_title('V1<->%s' % diffarea)
ax.axhline(y=0,color='k',linestyle='--')

ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)

ax = axes[1]
if varversion == 'session':
    ialdata = np.nanmean(weights_CCA_PMAL[:,1,:,:,:] - weights_CCA_PMAL[:,0,:,:,:],axis=(-1,-2))
elif varversion == 'stim':
    ialdata = np.nanmean(weights_CCA_PMAL[:,1,:,:,:] - weights_CCA_PMAL[:,0,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(n_components,nSessions*nStim)
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['PM']))
for icomp in range(n_components):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
    pval = pval*n_components #(to correct for multiple comparisons)
    # print(pval)
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.005*math.copysign(1, ttest)-0.002,'*',color='k',markersize=8)
        # ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002*math.copysign(1, ttest),'*',color='k',markersize=8)
# ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_ylabel(r'$\Delta$|Weight|   (PMLab-PMUnl)')
ax.set_xlabel('Dimension')
ax.set_title('PM<->%s' % diffarea)
ax.axhline(y=0,color='k',linestyle='--')

ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)

sns.despine(top=True,right=True,offset=3,trim=True)
# my_savefig(fig,savedir,'CCA_V1PM_labeled_to%s_deltaweights_%dsessions_%s' % (diffarea,nSessions,varversion),formats=['png'])

#%% 
######  #######  #####  ######  #######  #####   #####     ######  ####### #     #    #    #     # 
#     # #       #     # #     # #       #     # #     #    #     # #       #     #   # #   #     # 
#     # #       #       #     # #       #       #          #     # #       #     #  #   #  #     # 
######  #####   #  #### ######  #####    #####   #####     ######  #####   ####### #     # #     # 
#   #   #       #     # #   #   #             #       #    #     # #       #     # #######  #   #  
#    #  #       #     # #    #  #       #     # #     #    #     # #       #     # #     #   # #   
#     # #######  #####  #     # #######  #####   #####     ######  ####### #     # #     #    #    

#%% 

#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR
# session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                # ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%%  Load data properly with behavior as well as tensor:     
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 1.9     #post s
binsize     = 0.2
calciumversion = 'dF'
vidfields = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
    [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'videodata')

#%% 


#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
n_components        = 20
nStim               = 16
nmodelfits          = 10
minsampleneurons    = 10
maxnoiselevel       = 20
filter_nearby       = True
kFold               = 5
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
weights_CCA         = np.full((2,n_components,len(arealabels),nSessions,nStim,nmodelfits),np.nan)

#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)
si                  = SimpleImputer()

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=25)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<maxnoiselevel,
                                          idx_nearby),axis=0)) for i in arealabels])
    
    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    idx_N_all = np.empty(len(arealabels),dtype=object)
    for ial, al in enumerate(arealabels):
        idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    for imf in range(nmodelfits):
        idx_areax           = np.concatenate((np.random.choice(idx_N_all[0],nsampleneurons,replace=False),
                                                np.random.choice(idx_N_all[1],nsampleneurons,replace=False)))
        idx_areay           = np.concatenate((np.random.choice(idx_N_all[2],nsampleneurons,replace=False),
                                                np.random.choice(idx_N_all[3],nsampleneurons,replace=False)))
        assert len(idx_areax)==2*nsampleneurons and len(idx_areay)==2*nsampleneurons

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
            
             #on residual tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            X                   -= np.nanmean(X,axis=1,keepdims=True)
            Y                   -= np.nanmean(Y,axis=1,keepdims=True)

            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0,nan_policy='omit')

            #Get behavioral matrix: 
            B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
                                        sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
            B                   = B.reshape(np.shape(B)[0],-1).T
            B                   = zscore(B,axis=0,nan_policy='omit')

            X       = si.fit_transform(X)
            Y       = si.fit_transform(Y)
            B       = si.fit_transform(B)

            for irbh in range(2):   
                if irbh:
                    _,_,X,_,_ = regress_out_behavior_modulation(ses,B,X,rank=6,lam=0,perCond=False)
                    _,_,Y,_,_ = regress_out_behavior_modulation(ses,B,Y,rank=6,lam=0,perCond=False)

                # Y2,Y_hat_rr,Y_out,rank,EV = regress_out_behavior_modulation(ses,B,Y,rank=10,lam=0,perCond=False)

                # plt.imshow(Y,aspect='auto',vmin=-0.5,vmax=0.5)
                # plt.imshow(Y2,aspect='auto',vmin=-0.5,vmax=0.5)
                # plt.imshow(Y_out,aspect='auto',vmin=-0.5,vmax=0.5)
                # plt.imshow(Y_hat_rr,aspect='auto',vmin=-0.5,vmax=0.5)
                # plt.imshow(B,aspect='auto',vmin=-0.5,vmax=0.5)
                # print(EV)

                # Fit CCA MODEL:
                model_CCA.fit(X,Y)
                
                # weights_CCA[irbh,:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
                # weights_CCA[irbh,:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

                # weights_CCA[irbh,:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[:nsampleneurons,:]),axis=0)
                # weights_CCA[irbh,:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[nsampleneurons:,:]),axis=0)

                weights_CCA[irbh,:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
                weights_CCA[irbh,:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

                weights_CCA[irbh,:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[:nsampleneurons,:]),axis=0)
                weights_CCA[irbh,:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[nsampleneurons:,:]),axis=0)


#%% 
plt.scatter(model_CCA.x_weights_,model_CCA.x_loadings_)

# weights_CCA[irbh,:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
# weights_CCA[irbh,:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

#%% 
# sumdim = [0,2,3,4,5,6]
sumdim = 5

datatoplot1 = weights_CCA[:,:,1,:,:,:] - weights_CCA[:,:,0,:,:,:]
datatoplot2 = weights_CCA[:,:,3,:,:,:] - weights_CCA[:,:,2,:,:,:]

datatoplot1 = weights_CCA[:,:,1,:,:,:] / weights_CCA[:,:,0,:,:,:]
datatoplot2 = weights_CCA[:,:,3,:,:,:] / weights_CCA[:,:,2,:,:,:]

# datatoplot = np.diff(weights_CCA,axis=2)[:,:,[0,2],:,:,:] #Get the difference between labeled and unlabeled
datatoplot = np.concatenate((datatoplot1[:,:,np.newaxis,:,:,:],datatoplot2[:,:,np.newaxis,:,:,:]),axis=2) #Get the difference between labeled and unlabeled


datatoplot = np.nanmean(datatoplot,axis=(-1)) #average modelfits
# datatoplot = np.nansum(datatoplot[:,:sumdim,:,:,:],axis=1) #sum weight diff for first sumdim dims
datatoplot = np.nanmean(datatoplot[:,:sumdim,:,:,:],axis=1) #sum weight diff for first sumdim dims
datatoplot = np.reshape(datatoplot,(2,2,nSessions*nStim)) #stretch sessions * stim
# Dim1: with/wo behavior, dim2: v1/PM, dim3: datasets
datatoplot[datatoplot == 0] = np.nan

N       = datatoplot.shape[2]

fig,axes = plt.subplots(1,2,figsize=(4,2.5),sharex=True,sharey=True)

for iarea,area in enumerate(areas):
    ax = axes[iarea]
    for irbh in range(2):
        ax.scatter(np.zeros(N)+np.random.randn(N)*0.1+irbh,datatoplot[irbh,iarea,:],s=8,color=clrs_arealabels[iarea])
        ax.errorbar(irbh,np.nanmean(datatoplot[irbh,iarea,:]),yerr=np.nanstd(datatoplot[irbh,iarea,:])/np.sqrt(N),label='Orig',fmt='o',markerfacecolor='k',
                    elinewidth=2,markersize=8,color=clrs_arealabels[iarea])
    t,p = stats.ttest_rel(datatoplot[0,iarea,:],datatoplot[1,iarea,:],nan_policy='omit')
    # add_stat_annotation(ax, 0, 1, 0.15, p, h=None)
    add_stat_annotation(ax, 0, 1, 1.15, p, h=None)

    print('With vs Without behavior (%s): t=%1.3f, p=%1.3f' % (area,t,p))
    # add_star(ax,p)
        # ax.errorbar(1,np.nanmean(datatplot[0,0,:]),yerr=np.nanstd(datatplot[1,0,:])/np.sqrt(N),label='MinBeh',fmt='o',markerfacecolor='k',
                    # elinewidth=2,markersize=8,color=clrs_arealabels[0])

# ax.errorbar(1,np.nanmean(PMlabdiff),yerr=np.nanstd(PMlabdiff)/np.sqrt(N),label='PM',fmt='o',markerfacecolor='k',
            # elinewidth=2,markersize=8,color=clrs_arealabels[2])
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Orig','-Beh(RRR)'])
    if irbh==0:
        ax.set_ylabel('|Weight|   (Lab-Unl)')
    # ax.set_ylim([-0.1,0.25])
    ax.set_title(area)
    # ax.axhline(y=0,color='k',linestyle='--')
    ax.axhline(y=1,color='k',linestyle='--')
sns.despine(top=True,right=True,offset=1,trim=True)
#  ax.scatter(1,PMlabdiff,s=20,color='k')
plt.tight_layout()
# my_savefig(fig,savedir,'CCA_V1PM_labeled_sumdeltaweights_behav_%dsessions_%s' % (nSessions,varversion),formats=['png'])


#%% 

 #####   #####     #       ######  ####### ######     ######  ####### ######     ######     #    ### ######  
#     # #     #   # #      #     # #       #     #    #     # #     # #     #    #     #   # #    #  #     # 
#       #        #   #     #     # #       #     #    #     # #     # #     #    #     #  #   #   #  #     # 
#       #       #     #    ######  #####   ######     ######  #     # ######     ######  #     #  #  ######  
#       #       #######    #       #       #   #      #       #     # #          #       #######  #  #   #   
#     # #     # #     #    #       #       #    #     #       #     # #          #       #     #  #  #    #  
 #####   #####  #     #    #       ####### #     #    #       ####### #          #       #     # ### #     # 


#%%



#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
n_components        = 20
nStim               = 16
nmodelfits          = 10
minsampleneurons    = 10
maxnoiselevel       = 20
filter_nearby       = True
kFold               = 5
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

arealabelpairs      = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

CCA_corrtest        = np.full((narealabelpairs,n_components,nSessions,nStim),np.nan)


#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=25)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    #take the smallest sample size
    allpops             = np.array([i.split('-') for i in arealabelpairs]).flatten()
    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<maxnoiselevel,
                                          idx_nearby),axis=0)) for i in allpops])
    
    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue
    
    for iapl, arealabelpair in enumerate(arealabelpairs):
        alx,aly = arealabelpair.split('-')

        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<100,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<100,	
                                idx_nearby),axis=0))[0]
    
        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
        
            #on tensor during the response:
            # X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)].reshape(len(idx_areax),-1).T
            # Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)].reshape(len(idx_areay),-1).T
            
            #on residual tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            X                   -= np.mean(X,axis=1,keepdims=True)
            Y                   -= np.mean(Y,axis=1,keepdims=True)

            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            X                   = zscore(X,axis=0)  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0)

            [g,_] = CCA_subsample(X,Y,nN=nsampleneurons,resamples=nmodelfits,kFold=kFold,prePCA=None,n_components=np.min([n_components,nsampleneurons]))
            CCA_corrtest[iapl,:len(g),ises,istim] = g

#%%
fig, axes = plt.subplots(1,1,figsize=(4,4))

ax = axes
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    # ax.plot(np.arange(n_components),np.nanmean(CCA_corrtest[iapl,:,:,:],axis=(1,2)),
            # color=clrs_arealabelpairs[iapl],linewidth=2)
    iapldata = CCA_corrtest[iapl,:,:,:].reshape(n_components,-1)
    handles.append(shaded_error(x=np.arange(n_components),
                                # y=np.nanmean(CCA_corrtest_norm[iapl,:,:,:],axis=(1)).T,
                                y=iapldata.T,
                                error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_xticks(np.arange(0,n_components+5,5))
ax.set_xticklabels(np.arange(0,n_components+5,5)+1)
ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest,axis=(2,3))),1)])
ax.set_yticks([0,ax.get_ylim()[1]/2,ax.get_ylim()[1]])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel('Correlation')
ax.legend(handles,arealabelpairs,loc='upper right',frameon=False,fontsize=9)
sns.despine(top=True,right=True,offset=1,trim=True)
my_savefig(fig,savedir,'CCA_V1PM_pops_labeled_testcorr_%dsessions' % (nSessions),formats=['png'])

#%%
fig, axes = plt.subplots(1,1,figsize=(4,4))

ax = axes
CCA_corrtest_norm = CCA_corrtest / CCA_corrtest[0,0,:,:][np.newaxis,np.newaxis,:,:]
# CCA_corrtest_norm = CCA_corrtest / CCA_corrtest[:,0,:,:][:,np.newaxis,:,:]

for iapl, arealabelpair in enumerate(arealabelpairs):
    ax.plot(np.arange(n_components),np.nanmean(CCA_corrtest_norm[iapl,:,:,:],axis=(1,2)),
            color=clrs_arealabelpairs[iapl],linewidth=2)
ax.set_xticks(np.arange(0,n_components+5,5))
ax.set_xticklabels(np.arange(0,n_components+5,5)+1)
ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest_norm,axis=(2,3))),1)])
ax.set_yticks([0,ax.get_ylim()[1]/2,ax.get_ylim()[1]])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel('Correlation')
ax.legend(arealabelpairs,loc='upper right',frameon=False,fontsize=9)
sns.despine(top=True,right=True,offset=1,trim=True)
# my_savefig(fig,savedir,'CCA_V1PM_pops_labeled_testcorr_normdim1_%dsessions_%s' % (nSessions,varversion),formats=['png'])

#%%
fig, axes = plt.subplots(1,1,figsize=(4,4))
ax = axes
# CCA_corrtest_norm = CCA_corrtest / CCA_corrtest[0,:,:,:][np.newaxis,:,:,:]
CCA_corrtest_norm = CCA_corrtest - CCA_corrtest[0,:,:,:][np.newaxis,:,:,:]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    # ax.plot(np.arange(n_components),np.nanmean(CCA_corrtest_norm[iapl,:,:,:],axis=(1,2)),
            # color=clrs_arealabelpairs[iapl],linewidth=2)
    iapldata = CCA_corrtest_norm[iapl,:,:,:].reshape(n_components,-1)
    handles.append(shaded_error(x=np.arange(n_components),
                                # y=np.nanmean(CCA_corrtest_norm[iapl,:,:,:],axis=(1)).T,
                                y=iapldata.T,
                                error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
    for icomp in range(n_components):
        # ttest,pval = stats.ttest_1samp(iapldata[icomp],0,nan_policy='omit',alternative='greater')
        ttest,pval = stats.ttest_1samp(iapldata[icomp],0,nan_policy='omit',alternative='two-sided')
        if pval < 0.05:
            ax.plot(icomp,0.02+0.04*math.copysign(1, ttest) + iapl*0.003,'*',color=clrs_arealabelpairs[iapl],markersize=8)

# ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            # elinewidth=1,markersize=8,color=get_clr_areas(['PM']))

ax.set_xticks(np.arange(0,n_components+5,5))
ax.set_xticklabels(np.arange(0,n_components+5,5)+1)
ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest_norm,axis=(2,3))),1)])
ax.set_yticks([-0.025,0,0.025,0.05])
ax.set_ylim([-0.025,0.075])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel(u'Δ Correlation')
ax.legend(handles,arealabelpairs,loc='upper right',frameon=False,fontsize=9)
sns.despine(top=True,right=True,offset=1,trim=True)
my_savefig(fig,savedir,'CCA_V1PM_pops_labeled_testcorr_normUnl_%dsessions' % (nSessions),formats=['png'])


#%% 
