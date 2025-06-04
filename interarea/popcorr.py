# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:53:22 2022

@author: joana
@author: Matthijs Oude Lohuis, 2023, Champalimaud Research
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

# import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from tqdm import tqdm

from loaddata.session_info import filter_sessions
from utils.psth import compute_tensor
from preprocessing.preprocesslib import assign_layer
from utils.plot_lib import * #get all the fixed color schemes

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\PopRateCorr\\')

#%% ################################################
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR
# session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                # ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list,
                                       load_calciumdata=True,calciumversion='deconv')

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,min_lab_cells_PM=20,
                                       load_calciumdata=True,calciumversion='deconv')

sessions,nSessions   = filter_sessions(protocols = ['SP'],min_lab_cells_V1=20,min_lab_cells_PM=20,
                                       load_calciumdata=True,calciumversion='deconv')

#%% 
for ises in range(nSessions):
    # Normalize each column between 0 and 1
    sessions[ises].calciumdata = sessions[ises].calciumdata.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

for ises in range(nSessions):
    sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 

#%% Compute average rate for different populations: 
ises        = 5

arealayerlabels = ['V1unlL2/3',
                    'V1labL2/3',
                    'PMunlL2/3',
                    'PMlabL2/3',
                    'PMunlL5',
                    'PMlabL5',
                   ]

nArealayerlabels = len(arealayerlabels)

nS              = len(sessions[ises].calciumdata)
datamat         = np.full((nArealayerlabels,nS),np.nan)

minNneurons = 10

poprate = np.nanmean(sessions[ises].calciumdata,axis=1)
for iall,arealayerlabel in enumerate(arealayerlabels):
    idx_N               = np.where(sessions[ises].celldata['arealayerlabel']==arealayerlabel)[0]
    if len(idx_N)<minNneurons:
        continue
    datamat[iall,:]     = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_N],axis=1)
    # datamat[iall,:]     = np.nanmean(sessions[ises].calciumdata[:,idx_N],axis=1) / poprate

#%% 
clrs_arealayerlabels = sns.color_palette('tab10',nArealayerlabels)

fig, axes = plt.subplots(1,1,figsize=(6,3))

idx_T = np.arange(100,200) #take random stretch of timepoints
idx_T = np.arange(500,600) #take random stretch of timepoints

ax = axes
ax.plot(poprate[idx_T],color='k',linewidth=1.5)
for iall,arealayerlabel in enumerate(arealayerlabels):
    ax.plot(datamat[iall,idx_T],color=clrs_arealayerlabels[iall],linewidth=0.5)
ax.legend(['poprate'] + list(arealayerlabels),loc='upper right',frameon=False,fontsize=8,ncol=3)

# ax.set_xticks(range(nArealayerlabels))
# ax.set_yticks(range(nS))
sns.despine(top=True,right=True,offset=3)

#%% 
datamat_diff = datamat / (poprate[np.newaxis,:]+1e-8) #look at fluctuations relative to the total population

corrmat     = np.corrcoef(datamat_diff)

fig, axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
ax.imshow(corrmat,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(nArealayerlabels),labels=arealayerlabels,rotation=90)
ax.set_yticks(range(nArealayerlabels),labels=arealayerlabels)
cbar = fig.colorbar(ax.imshow(corrmat,cmap='RdBu_r',vmin=-1,vmax=1), ax=ax,shrink=0.5)
cbar.set_label('Correlation')

#%% 


#%% Compute average rate for different populations: 
arealayerlabels = ['V1unlL2/3',
                    'V1labL2/3',
                    'PMunlL2/3',
                    'PMlabL2/3',
                    'PMunlL5',
                    'PMlabL5',
                   ]

nArealayerlabels = len(arealayerlabels)

minNneurons         = 10
corrmat_raw         = np.full((nArealayerlabels,nArealayerlabels,nSessions),np.nan)
corrmat_diff        = np.full((nArealayerlabels,nArealayerlabels,nSessions),np.nan)
lags                = np.arange(-7,8)
lags                = np.arange(-17,18)
nLags               = len(lags)
crosscorrmat_raw    = np.full((nArealayerlabels,nArealayerlabels,nLags,nSessions),np.nan)
crosscorrmat_diff   = np.full((nArealayerlabels,nArealayerlabels,nLags,nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Compute crosscorrelations between population rates:'):
    ses.celldata['arealayerlabel'] = ses.celldata['arealabel'] + ses.celldata['layer'] 
    
    nS          = len(sessions[ises].calciumdata)
    poprate     = np.nanmean(sessions[ises].calciumdata,axis=1)
    datamat     = np.full((nArealayerlabels,nS),np.nan)

    for iall,arealayerlabel in enumerate(arealayerlabels):
        idx_N               = np.where(sessions[ises].celldata['arealayerlabel']==arealayerlabel)[0]
        idx_N               = np.where(np.all((sessions[ises].celldata['arealayerlabel']==arealayerlabel,
                                                sessions[ises].celldata['noise_level']<20),axis=0))[0]
        if len(idx_N)<minNneurons:
            continue
        idx_N_sub           = np.random.choice(idx_N,25,replace=True)
        datamat[iall,:]     = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_N_sub],axis=1)

    corrmat_raw[:,:,ises] = np.corrcoef(datamat)
    
    datamat_diff = datamat / poprate[np.newaxis,:] #look at fluctuations relative to the total population
    corrmat_diff[:,:,ises] = np.corrcoef(datamat_diff)

    for i in range(nArealayerlabels):
        for j in range(nArealayerlabels):
            for k, d in enumerate(lags):
                # Definition of lag: 
                # A postive correlation at negative lag means that population i is leading population j, 
                # while a positive lag means that population j is leading population i.
                # For example, if d = -1, then x is from datamat[i, -1:] and y is from datamat[j, :-1]
                # The x and y are then correlated with each other. 
                # If d = 1, then x is from datamat[i, :-1] and y is from datamat[j, 1:]
                # The x and y are then correlated with each other. 
                if d < 0:
                    xraw = datamat[i, -d:]
                    yraw = datamat[j, :d]

                    xdiff = datamat_diff[i, -d:]
                    ydiff = datamat_diff[j, :d]

                elif d > 0:
                    xraw = datamat[i, :-d]
                    yraw = datamat[j, d:]

                    xdiff = datamat_diff[i, :-d]
                    ydiff = datamat_diff[j, d:]
                else:
                    xraw = datamat[i]
                    yraw = datamat[j]

                    xdiff = datamat_diff[i]
                    ydiff = datamat_diff[j]

                # Normalize: Pearson correlation
                xraw_centered   = xraw - xraw.mean()
                yraw_centered   = yraw - yraw.mean()
                norm            = np.std(xraw) * np.std(yraw) * len(xraw)

                if norm > 0:
                    crosscorrmat_raw[i, j, k, ises] = np.dot(xraw_centered, yraw_centered) / norm

                # Normalize: Pearson correlation
                xdiff_centered  = xdiff - xdiff.mean()
                ydiff_centered  = ydiff - ydiff.mean()
                norm            = np.std(xdiff) * np.std(ydiff) * len(xdiff)

                if norm > 0:
                    crosscorrmat_diff[i, j, k, ises] = np.dot(xdiff_centered, ydiff_centered) / norm

#%% 
fig,axes = plt.subplots(1,2,figsize=(6,3))

vmin        = 0.1
vmax        = 0.4
datatoplot  = np.nanmean(corrmat_raw,axis=2)
np.fill_diagonal(datatoplot, np.nan)

ax = axes[0]
ax.imshow(datatoplot,cmap='Reds',vmin=vmin,vmax=vmax)
ax.set_xticks(range(nArealayerlabels),labels=arealayerlabels,rotation=90)
ax.set_yticks(range(nArealayerlabels),labels=arealayerlabels)
cbar = fig.colorbar(ax.imshow(datatoplot,cmap='Reds',vmin=vmin,vmax=vmax), ax=ax,shrink=0.5)
# cbar.set_label('Pop. Rate Correlation')
ax.set_title('Pop. Rate Correlation')

vrange = 0.1
datatoplot = np.nanmean(corrmat_diff,axis=2)
np.fill_diagonal(datatoplot, np.nan)

ax = axes[1]
ax.imshow(datatoplot,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(nArealayerlabels),labels=arealayerlabels,rotation=90)
ax.set_yticks(range(nArealayerlabels),labels=arealayerlabels)
cbar = fig.colorbar(ax.imshow(datatoplot,cmap='RdBu_r',vmin=-vrange,vmax=vrange), ax=ax,shrink=0.5)
ax.set_title('Residual Pop. Rate Correlation')

plt.tight_layout()

my_savefig(fig,savedir,'Poprate_corr_V1PMlabeled_GRGN_%dsessions.png' % (nSessions),formats=['png'])

#%% 
vrange = 1
fig,axes = plt.subplots(1,nLags,figsize=(nLags*3,3))
for i in range(nLags):
    ax = axes[i]
    im = ax.imshow(np.nanmean(crosscorrmat_raw[:,:,i,:],axis=-1),cmap='RdBu_r',vmin=-1,vmax=vrange)
    # if i==0:
    ax.set_yticks(range(nArealayerlabels),labels=arealayerlabels,fontsize=6)
    # if i==nLags//2:
    ax.set_xticks(range(nArealayerlabels),labels=arealayerlabels,rotation=90,fontsize=6)
    ax.set_title('lag=%1.2f sec' % (lags[i] * 1/ses.sessiondata['fs'][0]))

cbar = fig.colorbar(ax.imshow(np.nanmean(crosscorrmat_raw[:,:,i,:],axis=-1),cmap='RdBu_r',vmin=-1,vmax=vrange), ax=ax,shrink=0.5)
cbar.set_label('Correlation')
plt.tight_layout()
my_savefig(fig,savedir,'Poprate_corr_lags_V1PMlabeled_GRGN_%dsessions.png' % (nSessions),formats=['png'])

#%% 
vrange = 0.1
fig,axes = plt.subplots(1,nLags,figsize=(nLags*3,3))
for i in range(nLags):
    ax = axes[i]
    datatoplot = np.nanmean(crosscorrmat_diff[:,:,i,:],axis=-1)
    np.fill_diagonal(datatoplot, np.nan)
    im = ax.imshow(datatoplot,cmap='RdBu_r',vmin=-vrange,vmax=vrange)
    # if i==0:
    ax.set_yticks(range(nArealayerlabels),labels=arealayerlabels,fontsize=6)
    # if i==nLags//2:
    ax.set_xticks(range(nArealayerlabels),labels=arealayerlabels,rotation=90,fontsize=6)
    ax.set_title('lag=%1.2f sec' % (lags[i] * 1/ses.sessiondata['fs'][0]))

cbar = fig.colorbar(ax.imshow(np.nanmean(crosscorrmat_diff[:,:,i,:],axis=-1),cmap='RdBu_r',vmin=-vrange,vmax=vrange), ax=ax,shrink=0.5)
cbar.set_label('Correlation')
plt.tight_layout()
my_savefig(fig,savedir,'Poprate_diff_corr_lags_V1PMlabeled_GRGN_%dsessions.png' % (nSessions),formats=['png'])

#%% Show temporal cross-correlation:
titles = np.array(['Within V1','Within PM','V1-PM'])
pop_pairs =[np.array([[0,0],[0,1],[1,1]]),
                      np.array([[2,2],[2,3],[3,3],[2,4],[2,5],[3,4],[3,5],[4,4],[4,5],[5,5]]),
                      np.array([[0,2],[0,3],[1,2],[1,3],[0,4],[0,5],[1,4],[1,5]])]

#%% Show temporal cross-correlation:
titles = np.array(['Within V1','Within PM','V1-PM'])
pop_pairs =[np.array([[0,1]]),
                      np.array([[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]]),
                      np.array([[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5]])]

#%% Compute average rate for different populations: 
fig,axes = plt.subplots(1,3,figsize=(11,3),sharey=True,sharex=True)
for i in range(3):
    ax = axes[i]
    pop_pair_array = pop_pairs[i]
    for ipop,jpop in pop_pair_array:
        datatoplot = crosscorrmat_raw[ipop,jpop]
        ax.plot(lags * 1/ses.sessiondata['fs'][0],np.nanmean(datatoplot,axis=-1),label='%s-%s' % (arealayerlabels[ipop],arealayerlabels[jpop]))
    ax.set_title(titles[i])
    ax.set_xlabel('Lag (sec)')
    ax.axhline(0,linestyle='--',color='k')
    ax.axvline(0,linestyle='--',color='grey')
    if i==0:
        ax.set_ylabel('Correlation')
    # ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=-.1,frameon=False,fontsize=6)
    ax.legend(loc='upper center', frameon=False,fontsize=6,ncol=2)
    ax.set_ylim([-0.05,0.35])
    ax.set_xlim([-2,2])

sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'Poprate_corr_Xlags_V1PMlabeled_GRGN_%dsessions.png' % (nSessions),formats=['png'])

#%% Compute average rate for different populations: 
clrs = sns.color_palette('Set1',n_colors=16)
clrs = sns.color_palette('tab10',n_colors=16)

fig,axes = plt.subplots(1,3,figsize=(11,3),sharey=False,sharex=True)
for i in range(3):
    ax = axes[i]
    pop_pair_array = pop_pairs[i]
    handles = []
    for ipair,(ipop,jpop) in enumerate(pop_pair_array):
        # print('%s-%s' % (arealayerlabels[ipop],arealayerlabels[jpop]))
        datatoplot = crosscorrmat_diff[ipop,jpop]
        # ax.plot(lags * 1/ses.sessiondata['fs'][0],np.nanmean(datatoplot,axis=-1),label='%s-%s' % (arealayerlabels[ipop],arealayerlabels[jpop]))
        handles.append(shaded_error(lags * 1/ses.sessiondata['fs'][0],datatoplot.T,ax=ax,error='sem',
                     color=clrs[ipair],
                     label='%s-%s' % (arealayerlabels[ipop],arealayerlabels[jpop])))
    ax.set_title(titles[i])
    ax.set_xlabel('Lag (sec)')
    ax.axhline(0,linestyle='--',color='k')
    ax.axvline(0,linestyle='--',color='grey')
    if i==0:
        ax.set_ylabel('Correlation')
    # ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=-.1,frameon=False,fontsize=6)
    ax.legend(loc='upper center', frameon=False,fontsize=6,ncol=2)
    ax.set_ylim([-0.06,0.12])
    ax.set_xlim([-2,2])
sns.despine(top=True,right=True,offset=3)

plt.tight_layout()
my_savefig(fig,savedir,'Poprate_diff_corr_Xlags_V1PMlabeled_GRGN_%dsessions.png' % (nSessions),formats=['png'])
