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
from scipy import stats
from tqdm import tqdm
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *
from utils.RRRlib import *
from utils.CCAlib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\CCA\\Labeling\\')

#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR
# session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                # ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)


#%% 
areas       = ['V1','PM']
# areas       = ['V1','PM','AL','RSP']
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
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion=calciumversion,keepraw=False)
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,keepraw=False)

t_axis = sessions[0].t_axis





#%% 
from sklearn.cross_decomposition import CCA


#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
# celldata            = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
# N                   = len(celldata)
n_components        = 20
nStim               = 16
arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
nmodelfits          = 5

weights_CCA         = np.full((n_components,len(arealabels),nSessions,nStim,nmodelfits),np.nan)
cancorr_CCA         = np.full((n_components,nSessions,nStim,nmodelfits),np.nan)

minsampleneurons    = 10
maxnoiselevel       = 20
filter_nearby       = True

idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]


#%% Fit:

# prePCA              = 25
kFold               = 5
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
            
            weights_CCA[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
            weights_CCA[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

            weights_CCA[:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[:nsampleneurons,:]),axis=0)
            weights_CCA[:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[nsampleneurons:,:]),axis=0)

            #Implementing cross validation
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
clrs_arealabels = get_clr_arealabels(arealabels)

#%% 
fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)

ax = axes[0]
for ial,al in enumerate(arealabels[:2]):
    ialdata = np.nanmean(weights_CCA[:,ial,:,:,:],axis=(-1,-2))
    meantoplot = np.nanmean(ialdata,axis=1)
    errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(nSessions)

    ialdata = np.nanmean(weights_CCA[:,ial,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(n_components,nSessions*nStim)
    meantoplot = np.nanmean(ialdata,axis=1)
    errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

    ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=clrs_arealabels[ial])
    # ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,color=clrs_arealabels[ial])
    # for ises in range(nSessions):
        # ax.plot(ialdata[:,ises],color=clrs_arealabels[ial],alpha=1)

ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.legend(frameon=False,loc='lower right')
ax.set_title('V1')

ax = axes[1]
for ial,al in enumerate(arealabels[2:]):
    # ialdata = np.nanmean(weights_CCA[:,ial+2,:,:,:],axis=(-1,-2))
    # meantoplot = np.nanmean(ialdata,axis=1)
    # errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(nSessions)
    
    ialdata = np.nanmean(weights_CCA[:,ial+2,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(n_components,nSessions*nStim)
    meantoplot = np.nanmean(ialdata,axis=1)
    errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

    ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=clrs_arealabels[ial+2])
    # ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,color=clrs_arealabels[ial+2])
    ax.set_xticks(range(n_components))
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.set_title('PM')

ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)

ax.legend(frameon=False,loc='lower right')
sns.despine(top=True,right=True,offset=2,trim=False)
my_savefig(fig,savedir,'CCA_V1PM_labeled_weights_%dsessions' % nSessions,formats=['png'])


#%% 
fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)

ax = axes[0]
# ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1,-2))
# meantoplot = np.nanmean(ialdata,axis=1)
# errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(nSessions)

ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1))
ialdata = ialdata.reshape(n_components,nSessions*nStim)
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['V1']))
for icomp in range(n_components):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')

    # print(pval)
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)
# ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,color=get_clr_areas(['V1']))
# for ises in range(nSessions):
    # ax.plot(ialdata[:,ises],color=clrs_arealabels[ial],alpha=1)
ax.set_xlabel('Dimension')
ax.set_ylabel(r'$\Delta$|Weight|   (Lab-Unl)')
ax.set_title('V1')
ax.axhline(y=0,color='k',linestyle='--')

ax = axes[1]
# ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1,-2))
# meantoplot = np.nanmean(ialdata,axis=1)
# errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(nSessions)

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
# for ises in range(nSessions):
    # ax.plot(ialdata[:,ises],color=clrs_arealabels[ial],alpha=1)
ax.set_xlabel('Dimension')
ax.set_title('PM')
ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)
ax.axhline(y=0,color='k',linestyle='--')

sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'CCA_V1PM_labeled_deltaweights_%dsessions' % nSessions,formats=['png'])

#%% 




