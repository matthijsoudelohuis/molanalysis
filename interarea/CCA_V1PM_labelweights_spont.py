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
from skimage.measure import block_reduce

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.rf_lib import filter_nearlabeled
from utils.regress_lib import *
from utils.CCAlib import *
from utils.RRRlib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\CCA\\Labeling\\')

#%% 
areas       = ['V1','PM']
nareas      = len(areas)
protocol    = 'SP'

# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
# sessions,nSessions   = filter_sessions(protocols = ['SP'],filter_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
sessions,nSessions   = filter_sessions(protocols = ['SP'],filter_areas=areas)

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_data(load_calciumdata=True)


#%% 
 #####   #####     #       #     #    #   ######  #     #    #     # ####### ###  #####  #     # #######  #####  
#     # #     #   # #      #     #   ##   #     # ##   ##    #  #  # #        #  #     # #     #    #    #     # 
#       #        #   #     #     #  # #   #     # # # # #    #  #  # #        #  #       #     #    #    #       
#       #       #     #    #     #    #   ######  #  #  #    #  #  # #####    #  #  #### #######    #     #####  
#       #       #######     #   #     #   #       #     #    #  #  # #        #  #     # #     #    #          # 
#     # #     # #     #      # #      #   #       #     #    #  #  # #        #  #     # #     #    #    #     # 
 #####   #####  #     #       #     ##### #       #     #     ## ##  ####### ###  #####  #     #    #     #####  

# same results with deconv, radius doesn't impact much
# temporal binning affects the noise
# should have comparison of correlations of SP vs GR / GN

#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
n_components        = 20
nmodelfits          = 20
minsampleneurons    = 10
maxnoiselevel       = 20
filter_nearby       = True
kFold               = 5

arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
weights_CCA         = np.full((n_components,len(arealabels),nSessions,nmodelfits),np.nan)
cancorr_CCA         = np.full((n_components,nSessions,nmodelfits),np.nan)
do_cv_cca           = False

temporalbin         = 0.5
nbin                = int(temporalbin * sessions[0].sessiondata['fs'][0])

#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=50)
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

        X           = sessions[ises].calciumdata.iloc[:,idx_areax].to_numpy()
        Y           = sessions[ises].calciumdata.iloc[:,idx_areay].to_numpy()

        if nbin>1:
            X           = block_reduce(X, block_size=(nbin,1), func=np.mean, cval=np.mean(X))
            Y           = block_reduce(Y, block_size=(nbin,1), func=np.mean, cval=np.mean(Y))

        X           = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
        Y           = zscore(Y,axis=0,nan_policy='omit')

        # Fit CCA MODEL:
        model_CCA.fit(X,Y)
        
        weights_CCA[:,0,ises,imf] = np.mean(np.abs(model_CCA.x_loadings_[:nsampleneurons,:]),axis=0)
        weights_CCA[:,1,ises,imf] = np.mean(np.abs(model_CCA.x_loadings_[nsampleneurons:,:]),axis=0)

        weights_CCA[:,2,ises,imf] = np.mean(np.abs(model_CCA.y_loadings_[:nsampleneurons,:]),axis=0)
        weights_CCA[:,3,ises,imf] = np.mean(np.abs(model_CCA.y_loadings_[nsampleneurons:,:]),axis=0)

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
            cancorr_CCA[:,ises,imf] = np.nanmean(corr_test,axis=1)

#%%
clrs_areas = get_clr_areas(areas)
clrs_arealabels = get_clr_area_labeled(arealabels)

#%% 
fig,axes = plt.subplots(1,1,figsize=(3,2.5),sharex=True,sharey=True)
ax = axes
ialdata = np.nanmean(cancorr_CCA[:,:,:,:],axis=(-1))
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
# my_savefig(fig,savedir,'CCA_V1PM_labeled_testcorr_%dsessions_%s' % (nSessions,protocol),formats=['png'])

#%% 
fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)
ax = axes[0]
for ial,al in enumerate(arealabels[:2]):
    ialdata         = np.nanmean(weights_CCA[:,ial,:,:],axis=(-1))
    meantoplot      = np.nanmean(ialdata,axis=1)
    errortoplot     = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

    ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=clrs_arealabels[ial])
ax.axvline(x=mindim,color='grey',linestyle='--')
ax.text(mindim+0.5,ax.get_ylim()[1],'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_ylabel('|Weight|')
ax.legend(frameon=False,loc='lower center')
ax.set_title('V1')

ax = axes[1]
for ial,al in enumerate(arealabels[2:]):
    ialdata         = np.nanmean(weights_CCA[:,ial+2,:,:],axis=(-1))
    meantoplot      = np.nanmean(ialdata,axis=1)
    errortoplot     = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

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
# my_savefig(fig,savedir,'CCA_V1PM_labeled_weights_%dsessions_%s' % (nSessions,protocol),formats=['png'])


#%% 
mindim = 8

fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)
ax = axes[0]
# ialdata = np.nanmean(weights_CCA[:,1,:,:] - weights_CCA[:,0,:,:],axis=(-1))
# ialdata = np.nanmean(weights_CCA[:,1,:,:] / weights_CCA[:,0,:,:],axis=(-1))
ialdata = np.nanmean(weights_CCA,axis=(-1))
ialdata = ialdata[:,1,:] / ialdata[:,0,:]
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['V1']))
for icomp in range(n_components):
    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    ttest,pval = stats.ttest_1samp(ialdata[icomp],1,nan_policy='omit',alternative='two-sided')
    # pval = pval*n_components (to correct for multiple comparisons)
    # print(pval)
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)
# ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,color=get_clr_areas(['V1']))
ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
# ax.set_ylabel(r'$\Delta$|Weight|   (Lab-Unl)')
ax.set_ylabel('|Loadings|   (Lab / Unl)')
ax.set_xlabel('Dimension')
ax.set_title('V1')
# ax.axhline(y=0,color='k',linestyle='--')
ax.axhline(y=1,color='k',linestyle='--')

ax = axes[1]
# ialdata = np.nanmean(weights_CCA[:,3,:,:] - weights_CCA[:,2,:,:],axis=(-1))
# ialdata = np.nanmean(weights_CCA[:,3,:,:] / weights_CCA[:,2,:,:],axis=(-1))
ialdata = np.nanmean(weights_CCA,axis=(-1))
ialdata = ialdata[:,3,:] / ialdata[:,2,:]
# ialdata = ialdata[:,2,:] / ialdata[:,3,:]

meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['PM']))
for icomp in range(n_components):
    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    ttest,pval = stats.ttest_1samp(ialdata[icomp],1,nan_policy='omit',alternative='two-sided')
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)

ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_title('PM')
# ax.axhline(y=0,color='k',linestyle='--')
ax.axhline(y=1,color='k',linestyle='--')

ax_nticks(ax,5)
ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)

sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'CCA_V1PM_labeled_deltaweights_%dsessions_%s' % (nSessions,protocol),formats=['png'])

