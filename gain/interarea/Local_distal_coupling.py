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
# from scipy.signal import medfilt
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
from scipy.stats import zscore
# from sklearn.cross_decomposition import CCA
from tqdm import tqdm

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning,compute_tuning_wrapper
# from utils.plot_lib import * #get all the fixed color schemes
# from utils.explorefigs import *
# from utils.CCAlib import *
# from utils.corr_lib import *
# from utils.regress_lib import *
from utils.gain_lib import *
# from utils.RRRlib import estimate_dimensionality
from utils.rf_lib import filter_nearlabeled

# savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\CCA\\')
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\')

#%% Load an example session: 
session_list        = np.array(['LPE12223_2024_06_10']) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% Remove two sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
calciumversion = 'deconv'
for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)
    
#%% How are neurons are coupled to the population rate of different areas:
areas = np.array(['V1', 'PM','AL','RSP'])
nareas = len(areas)

for ises,ses in enumerate(sessions):
    resp        = zscore(ses.respmat,axis=1)
    N           = len(ses.celldata)
    for iarea,area in enumerate(areas):
        idx_N = np.where(np.all((ses.celldata['roi_name']==area,
                        ses.celldata['noise_level']<20,
                        # ses.celldata['redcell']
                        ),axis=0))[0]
        poprate = np.nanmean(resp[idx_N,:],axis=0)

        for iN in range(N):
            if iN in idx_N:
                ses.celldata.loc[iN,'pop_coupling_%s' % area]  = np.corrcoef(resp[iN,:],np.mean(resp[np.setdiff1d(idx_N,iN),:], axis=0))[0,1]
            else:
                ses.celldata.loc[iN,'pop_coupling_%s' % area]  = np.corrcoef(resp[iN,:],poprate)[0,1]

        # ses.celldata['pop_coupling_%s' % area]  = [np.corrcoef(resp[i,:],poprate)[0,1] for i in range(len(ses.celldata))]

#%% How are neurons are coupled to the population rate of different areas:
arealabels = np.array(['V1unl','V1lab','PMunl','PMlab'])
narealabels = len(arealabels)

for ises,ses in enumerate(sessions):
    resp        = zscore(ses.respmat,axis=1)
    N           = len(ses.celldata)
    for ial,al in enumerate(arealabels):
        idx_N = np.where(np.all((ses.celldata['arealabel']==al,
                        ses.celldata['noise_level']<20,
                        ),axis=0))[0]
        poprate = np.nanmean(resp[idx_N,:],axis=0)

        for iN in range(N):
            if iN in idx_N:
                ses.celldata.loc[iN,'pop_coupling_%s' % al]  = np.corrcoef(resp[iN,:],np.mean(resp[np.setdiff1d(idx_N,iN),:], axis=0))[0,1]
            else:
                ses.celldata.loc[iN,'pop_coupling_%s' % al]  = np.corrcoef(resp[iN,:],poprate)[0,1]

        # poprate = np.nanmean(resp[idx_N,:],axis=0)

        # ses.celldata['pop_coupling_%s' % al]  = [np.corrcoef(resp[i,:],poprate)[0,1] for i in range(len(ses.celldata))]

#%% Filter out cells that are close to a labeled cell:
for ises,ses in enumerate(sessions):
    ses.celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=30)

#%% Concatenate data:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% 

#%% 
areas = np.array(['V1', 'PM'])
arealabels = np.array(['V1unl','V1lab','PMunl','PMlab'])
narealabels = len(arealabels)

nbins = 15
bins = np.linspace(-0.2,0.6,nbins)
bincenters = (bins[1:]+bins[:-1])/2

fig,axes = plt.subplots(len(areas),3,figsize=(3*2,len(areas)*2))
for ixarea,xarea in enumerate(areas):
    idx_unl = np.all((celldata['roi_name']==xarea,
                        celldata['noise_level']<20,
                        # celldata['nearby'],
                        celldata['redcell']==0,
                        ),axis=0)
    idx_lab = np.all((celldata['roi_name']==xarea,
                        celldata['noise_level']<20,
                        # celldata['nearby'],
                        celldata['redcell']==1,
                        ),axis=0)
    
    histmaps = np.full((3,len(bins)-1,len(bins)-1),np.nan)
    # histmap = np.histogram2d(celldata['pop_coupling_%s' % xarea][idx_unl],celldata['pop_coupling_%s' % xarea][idx_unl],bins=bins)
    histmaps[0] = np.histogram2d(celldata['pop_coupling_V1'][idx_unl],
                             celldata['pop_coupling_PM'][idx_unl],bins=bins,density=True)[0]
    histmaps[1] = np.histogram2d(celldata['pop_coupling_V1'][idx_lab],
                             celldata['pop_coupling_PM'][idx_lab],bins=bins,density=True)[0]
    histmaps[2] =  histmaps[1] - histmaps[0]
    cmaps = ['hot','hot','bwr']
    # ranges
    for i in range(3):
        ax = axes[ixarea,i]
        ax.imshow(histmaps[i],cmap=cmaps[i])
        # ax.pcolor(histmaps[i],cmap='bwr')
        # ax.imshow(histmaps[i],cmap='bwr',vmin=-0.1,vmax=0.1)
        # ax.set_xticks([0,nbins-1],labels=np.array2string(bincenters[[0,-1]], precision=2, floatmode='fixed'))
        # ax.set_xticks([0,nbins-1],labels=bincenters[[0,-1]])
        # ax.set_xticks(np.arange(nbins-1),labels=bincenters)
        # ax.set_yticks(range(nbins),labels=bincenters)
        ax.invert_yaxis()
        ax.set_title('%s' % xarea)
        ax.set_xlabel('PM coupling')
        ax.set_ylabel('V1 coupling')
        # ax.plot([-0.1,0.5],[-0.1,0.5],color='k',linestyle='--')
     
plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=2)   



#%%
clrs_labeled = get_clr_labeled()
fig,axes = plt.subplots(nareas,nareas,figsize=(8,8))
for ixarea,xarea in enumerate(areas):
    for iyarea,yarea in enumerate(areas):
        ax = axes[ixarea,iyarea]
        idx_N = np.all((celldata['roi_name']==xarea,
                        celldata['noise_level']<20,
                        celldata['nearby']),axis=0)
        
        # idx_N = celldata['roi_name']==xarea
        # sns.displot(data=celldata[idx_N],x='pop_coupling_%s' % xarea,
                        # y='pop_coupling_%s' % yarea,legend=False,
                        # ax=ax,hue='labeled',palette=clrs_labeled)
        
        # # idx_N = celldata['roi_name']==xarea
        sns.scatterplot(data=celldata[idx_N],x='pop_coupling_%s' % xarea,s=7,
                        y='pop_coupling_%s' % yarea,legend=False,
                        ax=ax,hue='labeled',palette=clrs_labeled)
        # sns.histplot(data=celldata,x='pop_coupling_%s' % areas[xarea],color='green',element="step",
        #              common_norm=False,ax=ax,stat="density",hue='labeled')
        ax.set_title('%s neurons' % xarea)
        ax.set_xlabel(xarea)
        ax.set_ylabel(yarea)
        ax.plot([-0.1,0.5],[-0.1,0.5],color='k',linestyle='--')
        ax.set_xlim(-0.1,0.6)
        ax.set_ylim(-0.1,0.6)

plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=2)
# my_savefig(fig,savedir,'PopCouplingHist_%ssession' % sessions[ises].session_id,formats=['png'])

#%% 
nbins = 15
bins = np.linspace(-0.2,0.6,nbins)
areas = np.array(['V1', 'PM'])
bincenters = (bins[1:]+bins[:-1])/2

fig,axes = plt.subplots(2,3,figsize=(8,6))
for ixarea,xarea in enumerate(areas):
    idx_unl = np.all((celldata['roi_name']==xarea,
                        celldata['noise_level']<20,
                        celldata['nearby'],
                        celldata['redcell']==0,
                        ),axis=0)
    idx_lab = np.all((celldata['roi_name']==xarea,
                        celldata['noise_level']<20,
                        celldata['nearby'],
                        celldata['redcell']==1,
                        ),axis=0)
    
    histmaps = np.full((3,len(bins)-1,len(bins)-1),np.nan)
    # histmap = np.histogram2d(celldata['pop_coupling_%s' % xarea][idx_unl],celldata['pop_coupling_%s' % xarea][idx_unl],bins=bins)
    histmaps[0] = np.histogram2d(celldata['pop_coupling_V1unl'][idx_unl],
                             celldata['pop_coupling_PMunl'][idx_unl],bins=bins,density=True)[0]
    histmaps[1] = np.histogram2d(celldata['pop_coupling_V1unl'][idx_lab],
                             celldata['pop_coupling_PMunl'][idx_lab],bins=bins,density=True)[0]
    histmaps[2] =  histmaps[1] - histmaps[0]
    cmaps = ['hot','hot','bwr']
    # ranges
    for i in range(3):
        ax = axes[ixarea,i]
        ax.imshow(histmaps[i],cmap=cmaps[i])
        # ax.set_xticks([0,nbins-1],labels=np.array2string(bincenters[[0,-1]], precision=2, floatmode='fixed'))
        # ax.set_xticks([0,nbins-1],labels=bincenters[[0,-1]])
        # ax.set_xticks(np.arange(nbins-1),labels=bincenters)
        # ax.set_yticks(range(nbins),labels=bincenters)
        ax.invert_yaxis()
        ax.set_title('%s' % xarea)
        ax.set_xlabel('V1unl coupling')
        ax.set_ylabel('PMunl coupling')
     
plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=2)   

#%% 
nbins = 15
bins = np.linspace(-0.2,0.6,nbins)
areas = np.array(['V1', 'PM'])
bincenters = (bins[1:]+bins[:-1])/2

fig,axes = plt.subplots(2,3,figsize=(8,6))
for ixarea,xarea in enumerate(areas):
    idx_unl = np.all((celldata['roi_name']==xarea,
                        celldata['noise_level']<20,
                        # celldata['nearby'],
                        celldata['redcell']==0,
                        ),axis=0)
    idx_lab = np.all((celldata['roi_name']==xarea,
                        celldata['noise_level']<20,
                        # celldata['nearby'],
                        celldata['redcell']==1,
                        ),axis=0)
    
    histmaps = np.full((3,len(bins)-1,len(bins)-1),np.nan)
    # histmap = np.histogram2d(celldata['pop_coupling_%s' % xarea][idx_unl],celldata['pop_coupling_%s' % xarea][idx_unl],bins=bins)
    histmaps[0] = np.histogram2d(celldata['pop_coupling_PMunl'][idx_unl],
                             celldata['pop_coupling_PMlab'][idx_unl],bins=bins,density=True)[0]
    histmaps[1] = np.histogram2d(celldata['pop_coupling_PMunl'][idx_lab],
                             celldata['pop_coupling_PMlab'][idx_lab],bins=bins,density=True)[0]
    
    histmaps[0] = np.histogram2d(celldata['pop_coupling_V1unl'][idx_unl],
                             celldata['pop_coupling_V1lab'][idx_unl],bins=bins,density=True)[0]
    histmaps[1] = np.histogram2d(celldata['pop_coupling_V1unl'][idx_lab],
                             celldata['pop_coupling_V1lab'][idx_lab],bins=bins,density=True)[0]
    histmaps[2] =  histmaps[1] - histmaps[0]
    cmaps = ['hot','hot','bwr']

    # histmaps[0,0,:] = 10
    for i in range(3):
        ax = axes[ixarea,i]
        if i<2:
            ax.imshow(histmaps[i].T,cmap=cmaps[i])
        if i==2: 
            ax.imshow(histmaps[i].T,cmap=cmaps[i],vmin=-2,vmax=2)
        # ax.set_xticks([0,nbins-1],labels=np.array2string(bincenters[[0,-1]], precision=2, floatmode='fixed'))
        # ax.set_xticks([0,nbins-1],labels=bincenters[[0,-1]])
        # ax.set_xticks(np.arange(nbins-1),labels=bincenters)
        # ax.set_yticks(range(nbins),labels=bincenters)
        ax.invert_yaxis()
        ax.set_title('%s' % xarea)
        ax.set_xlabel('V1lab coupling')
        ax.set_ylabel('V1unl coupling')
        # ax.set_ylabel('PMlab coupling')
     
plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=2)   

#%%
areas = np.array(['V1', 'PM'])
clrs_labeled = get_clr_labeled()

idx_N = np.all((celldata['roi_name']=='V1',
                celldata['noise_level']<20,
                celldata['nearby']
                ),axis=0)
# sns.displot(data=celldata[idx_N],x='pop_coupling_V1',
#                 y='pop_coupling_PM',legend=False,
#                 ax=ax,hue='labeled',palette=clrs_labeled)
sns.jointplot(data=celldata[idx_N],x='pop_coupling_V1',
                y='pop_coupling_PM',legend=False,
                kind="kde",
                hue='labeled',palette=clrs_labeled)

idx_N = np.all((celldata['roi_name']=='PM',
                celldata['noise_level']<20,
                celldata['nearby']
                ),axis=0)
sns.jointplot(data=celldata[idx_N],x='pop_coupling_PM',
                y='pop_coupling_V1',legend=False,
                kind="kde",common_norm=False,
                hue='labeled',palette=clrs_labeled)

fig,axes = plt.subplots(nareas,nareas,figsize=(8,8))
for ixarea,xarea in enumerate(areas):
    for iyarea,yarea in enumerate(areas):
        ax = axes[ixarea,iyarea]
        idx_N = np.all((celldata['roi_name']==xarea,
                        celldata['noise_level']<20,
                        celldata['nearby']
                        ),axis=0)
        
        # idx_N = celldata['roi_name']==xarea
        sns.displot(data=celldata[idx_N],x='pop_coupling_%s' % xarea,
                        y='pop_coupling_%s' % yarea,legend=False,
                        ax=ax,hue='labeled',palette=clrs_labeled)
        # ax = sns.displot(data=celldata[idx_N],x='pop_coupling_%s' % xarea,
                        # legend=False,stat="probability",common_norm=False,
                        # ax=ax,hue='labeled',palette=clrs_labeled)
        # hue="species", stat="probability")

        # # idx_N = celldata['roi_name']==xarea
        # sns.scatterplot(data=celldata[idx_N],x='pop_coupling_%s' % xarea,s=7,
        #                 y='pop_coupling_%s' % yarea,legend=False,
        #                 ax=ax,hue='labeled',palette=clrs_labeled)
        # sns.histplot(data=celldata,x='pop_coupling_%s' % areas[xarea],color='green',element="step",
        #              common_norm=False,ax=ax,stat="density",hue='labeled')
        ax.set_title('%s neurons' % xarea)
        ax.set_xlabel(xarea)
        ax.set_ylabel(yarea)
        ax.plot([-0.1,0.5],[-0.1,0.5],color='k',linestyle='--')
        ax.set_xlim(-0.1,0.6)
        ax.set_ylim(-0.1,0.6)

plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=2)

# sns.displot(penguins, x="flipper_length_mm", hue="species", stat="probability")



