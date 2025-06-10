# -*- coding: utf-8 -*-
"""
@author: joana
@author: Matthijs Oude Lohuis, 2023, Champalimaud Research
"""

#%%
#Todo:
# take the asymmetry window smaller
# rescale the activity of datamat to be on the same scale
# take the crosscorrelation with a function instead of with for loop
# make sure the asymmetry score does not take into account negative values


#%% ###################################################
import math, os
os.chdir('c:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

# import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from tqdm import tqdm
from scipy.stats import zscore

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

protocols           = ['GR']
protocols           = ['GR','GN']
# protocols           = ['SP']
protocols           = ['IM']

sessions,nSessions   = filter_sessions(protocols = protocols,min_lab_cells_V1=20,min_lab_cells_PM=20,
                                       load_calciumdata=True,calciumversion='deconv')

#%% Interpolate timestamps such that activity was pseudosimultaneous:
nplanes = 8
for ises in range(nSessions):
    ts_F = sessions[ises].ts_F
    for iN in range(len(sessions[ises].celldata)):
        if sessions[ises].celldata['plane_idx'][iN] != 0:
            offset = sessions[ises].celldata['plane_idx'][iN] / nplanes / sessions[ises].sessiondata['fs'][0]
            #print(offset)
            sessions[ises].calciumdata.iloc[:,iN] = np.interp(ts_F - offset,ts_F,sessions[ises].calciumdata.iloc[:,iN])

#%%  # Normalize each column (neuron activity) between 0 and 1
for ises in range(nSessions):   
    #sessions[ises].calciumdata = sessions[ises].calciumdata.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    sessions[ises].calciumdata = sessions[ises].calciumdata.apply(zscore)

#%%  #assign arealayerlabel
for ises in range(nSessions):   
    sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 

#%%
for ises in range(nSessions):
    for iplane in range(8):
        idx_N = np.where(sessions[ises].celldata['plane_idx']==iplane)[0]
        ts_F = sessions[ises].ts_F
        ts_F_interp = sessions[ises].ts_F - iplane/8

        sessions[ises].calciumdata.iloc[:,idx_N] = np.interp(ts_F_interp,ts_F,sessions[ises].calciumdata.iloc[:,idx_N])
        #= compute_tensor(sessions[ises].calciumdata.iloc[:,idx_N],ts_F_interp)[sessions[ises].celldata['plane_idx']==iplane]


#%% Compute average rate for different populations: 
ises        = 4

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

minNneurons     = 10

poprate = np.nanmean(sessions[ises].calciumdata,axis=1)
for iall,arealayerlabel in enumerate(arealayerlabels):
    idx_N               = np.where(sessions[ises].celldata['arealayerlabel']==arealayerlabel)[0]
    if len(idx_N)<minNneurons:
        continue
    datamat[iall,:]     = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_N],axis=1)
    # datamat[iall,:]     = np.nanmean(sessions[ises].calciumdata[:,idx_N],axis=1) / poprate

clrs_arealayerlabels = sns.color_palette('tab10',nArealayerlabels)

fig, axes = plt.subplots(1,1,figsize=(6,3))

idx_T = np.arange(100,200) #take random stretch of timepoints
# idx_T = np.arange(500,600) #take random stretch of timepoints

ax = axes
ax.plot(np.arange(len(idx_T))/sessions[ises].sessiondata['fs'][0],poprate[idx_T],color='k',linewidth=1.5)
for iall,arealayerlabel in enumerate(arealayerlabels):
    ax.plot(np.arange(len(idx_T))/sessions[ises].sessiondata['fs'][0],datamat[iall,idx_T],color=clrs_arealayerlabels[iall],linewidth=0.75)
ax.legend(['poprate'] + list(arealayerlabels),loc='upper right',frameon=False,fontsize=8,ncol=3)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Population rate (normalized)')

sns.despine(top=True,right=True,offset=3)
#my_savefig(fig,savedir,'Poprate_Excertp_%s_%s' % ('_'.join(protocols),sessions[ises].session_id),formats=['png'])

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
def plot_crosscorr_poppairs(crosscorr_data,timelags,arealayerlabels,pop_pairs,titles):
    npop_pairs = len(pop_pairs)
    fig,axes = plt.subplots(1,npop_pairs,figsize=(npop_pairs*3+1,3),sharey=True,sharex=True)
    clrs = sns.color_palette('tab10',n_colors=16)

    for i in range(npop_pairs):
        ax = axes[i]
        pop_pair_array = pop_pairs[i]
        handles = []
        for ipair,(ipop,jpop) in enumerate(pop_pair_array):
            datatoplot = crosscorr_data[ipop,jpop]
            # ax.plot(lags * 1/ses.sessiondata['fs'][0],np.nanmean(datatoplot,axis=-1),label='%s-%s' % (arealayerlabels[ipop],arealayerlabels[jpop]))
            handles.append(shaded_error(timelags,datatoplot.T,ax=ax,error='sem',
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
        # ax.set_ylim(np.nanpercentile(datatoplot,[0,100]))
        ax.set_xlim(np.percentile(timelags,[0,100]))

    sns.despine(top=True,right=True,offset=3)
    plt.tight_layout()
    return fig

#%% 
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR


#%% Compute average rate for different populations: 
popversion = 'V1PM'
arealayerlabels = ['V1unlL2/3',
                    'V1unlL5',
                    'PMunlL2/3',
                    'PMunlL5',
                   ]
minNneurons         = 100
sampleNneurons      = 200

# arealayerlabels = ['V1unlL2/3',
#                     'V1labL2/3',
#                     'PMunlL2/3',
#                     'PMlabL2/3',
#                     'PMunlL5',
#                     'PMlabL5',
#                    ]

popversion = 'V1PMlabeled'
arealayerlabels = ['V1unlL2/3',
                    'V1labL2/3',
                    'V1unlL5',
                    'V1labL5',
                    'PMunlL2/3',
                    'PMlabL2/3',
                    'PMunlL5',
                    'PMlabL5',
                   ]

minNneurons         = 20
sampleNneurons      = 50

nresamples          = 10
nArealayerlabels    = len(arealayerlabels)
corrmat_raw         = np.full((nArealayerlabels,nArealayerlabels,nSessions,nresamples),np.nan)
corrmat_diff        = np.full((nArealayerlabels,nArealayerlabels,nSessions,nresamples),np.nan)
lags                = np.arange(-7,8)
lags                = np.arange(-4,5)
# lags                = np.arange(-12,13)
nLags               = len(lags)
timelags            = lags * 1/sessions[0].sessiondata['fs'][0]
crosscorrmat_raw    = np.full((nArealayerlabels,nArealayerlabels,nLags,nSessions,nresamples),np.nan)
crosscorrmat_diff   = np.full((nArealayerlabels,nArealayerlabels,nLags,nSessions,nresamples),np.nan)

gc_lagorder         = 5
gc_raw              = np.full((nArealayerlabels,nArealayerlabels,nSessions,nresamples),np.nan)
VAR_raw             = np.full((nArealayerlabels,nArealayerlabels,gc_lagorder,nSessions,nresamples),np.nan)
# gc_diff             = np.full((nArealayerlabels,nArealayerlabels,nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Compute crosscorrelations between population rates:'):
    ses.celldata['arealayerlabel'] = ses.celldata['arealabel'] + ses.celldata['layer'] 
    
    nS          = len(sessions[ises].calciumdata)
    poprate     = np.nanmean(sessions[ises].calciumdata,axis=1)
    
    for irs in range(nresamples):
        datamat     = np.full((nArealayerlabels,nS),np.nan)

        for iall,arealayerlabel in enumerate(arealayerlabels):
            # idx_N               = np.where(sessions[ises].celldata['arealayerlabel']==arealayerlabel)[0]
            idx_N               = np.where(np.all((sessions[ises].celldata['arealayerlabel']==arealayerlabel,
                                                    sessions[ises].celldata['noise_level']<20),axis=0))[0]
            if len(idx_N)<minNneurons:
                continue
            idx_N_sub           = np.random.choice(idx_N,sampleNneurons,replace=True)
            datamat[iall,:]     = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_N_sub],axis=1)

        corrmat_raw[:,:,ises,irs] = np.corrcoef(datamat)
        
        datamat_diff = zscore(datamat,axis=1) / (zscore(poprate)[np.newaxis,:]+1e-6) #look at fluctuations relative to the total population
        # datamat_diff = datamat / (poprate[np.newaxis,:]+1e-6) #look at fluctuations relative to the total population
        corrmat_diff[:,:,ises,irs] = np.corrcoef(datamat_diff)

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

                    # if norm > 0:
                    crosscorrmat_raw[i, j, k, ises,irs] = np.dot(xraw_centered, yraw_centered) / norm

                    # Normalize: Pearson correlation
                    xdiff_centered  = xdiff - xdiff.mean()
                    ydiff_centered  = ydiff - ydiff.mean()
                    norm            = np.std(xdiff) * np.std(ydiff) * len(xdiff)

                    # if norm > 0:
                    crosscorrmat_diff[i, j, k, ises,irs] = np.dot(xdiff_centered, ydiff_centered) / norm

                # x = np.column_stack((datamat[i,:],datamat[j,:]))
                # if np.isnan(x).any():
                #     continue
                # test_result = grangercausalitytests(x, maxlag=gc_lagorder,verbose=False)
                # gc_raw[i,j,ises]      = test_result[1][0]['params_ftest'][0]

        idx_nanpops     = np.isnan(datamat).any(axis=1)
        if np.sum(~idx_nanpops)<2:
            continue

        datamat         = np.diff(datamat,axis=1)
        model           = VAR(datamat[~idx_nanpops,:].T)
        results         = model.fit(maxlags=gc_lagorder)

        coefficients = np.transpose(results.coefs,[1,2,0]) #reorder such that lags are 3rd dim.

        VAR_raw[np.ix_(~idx_nanpops,~idx_nanpops,np.arange(gc_lagorder),[ises],[irs])] = coefficients[:,:,:,np.newaxis,np.newaxis]

#Take the average across resamples:
corrmat_raw         = np.nanmean(corrmat_raw,axis=-1)
corrmat_diff        = np.nanmean(corrmat_diff,axis=-1)
crosscorrmat_raw    = np.nanmean(crosscorrmat_raw,axis=-1)
crosscorrmat_diff   = np.nanmean(crosscorrmat_diff,axis=-1)
VAR_raw             = np.nanmean(VAR_raw,axis=-1)

#%% Plot the results:
fig,axes = plt.subplots(1,2,figsize=(6,3))

vmin        = 0.1
vmax        = 0.6
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

my_savefig(fig,savedir,'Poprate_corr_%s_%s_%dsessions' % (popversion,'_'.join(protocols),nSessions),formats=['png'])

#%% 
vrange = 0.3
fig,axes = plt.subplots(1,nLags,figsize=(nLags*3,3))
for i in range(nLags):
    ax = axes[i]
    im = ax.imshow(np.nanmean(crosscorrmat_raw[:,:,i,:],axis=-1),cmap='RdBu_r',vmin=-vrange,vmax=vrange)
    # if i==0:
    ax.set_yticks(range(nArealayerlabels),labels=arealayerlabels,fontsize=6)
    # if i==nLags//2:
    ax.set_xticks(range(nArealayerlabels),labels=arealayerlabels,rotation=90,fontsize=6)
    ax.set_title('lag=%1.2f sec' % (timelags[i]))

cbar = fig.colorbar(ax.imshow(np.nanmean(crosscorrmat_raw[:,:,i,:],axis=-1),cmap='RdBu_r',vmin=-vrange,vmax=vrange), ax=ax,shrink=0.5)
cbar.set_label('Correlation')
plt.tight_layout()
my_savefig(fig,savedir,'Poprate_corr_lags_%s_%s_%dsessions' % (popversion,'_'.join(protocols),nSessions),formats=['png'])

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
    ax.set_title('lag=%1.2f sec' % (timelags[i]))

cbar = fig.colorbar(ax.imshow(np.nanmean(crosscorrmat_diff[:,:,i,:],axis=-1),cmap='RdBu_r',vmin=-vrange,vmax=vrange), ax=ax,shrink=0.5)
cbar.set_label('Correlation')
plt.tight_layout()
my_savefig(fig,savedir,'Poprate_diff_corr_lags_%s_%s_%dsessions' % (popversion,'_'.join(protocols),nSessions),formats=['png'])

#%% Show temporal cross-correlation:
titles = np.array(['Within V1','Within PM','V1-PM'])
pop_pairs =[np.array([[0,0],[0,1],[1,1],[0,2],[0,3],[1,2],[1,3],[2,2],[2,3],[3,3]]),
                      np.array([[4,4],[4,5],[4,6],[4,7],[5,5],[5,6],[5,7],[6,6],[6,7],[7,7]]),
                      np.array([[0,4],[0,5],[0,6],[0,7],[1,4],[1,5],[1,6],[1,7],[2,4],[2,5],[2,6],[2,7],[3,4],[3,5],[3,6],[3,7]])]

pop_pairs =[np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]),
                      np.array([[4,5],[4,6],[4,7],[5,6],[5,7],[6,7]]),
                      np.array([[0,4],[0,5],[0,6],[0,7],[1,4],[1,5],[1,6],[1,7],[2,4],[2,5],[2,6],[2,7],[3,4],[3,5],[3,6],[3,7]])]

titles = np.array(['Within V1','Within PM','V1L2/3-PM','V1L5-PM'])
pop_pairs =[np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]),
                      np.array([[4,5],[4,6],[4,7],[5,6],[5,7],[6,7]]),
                      np.array([[0,4],[0,5],[0,6],[0,7],[1,4],[1,5],[1,6],[1,7]]),
                      np.array([[2,4],[2,5],[2,6],[2,7],[3,4],[3,5],[3,6],[3,7]])]

#%% Show temporal cross-correlation:
titles = np.array(['Within V1','Within PM','V1-PM'])
pop_pairs =[np.array([[0,1]]),
                      np.array([[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]]),
                      np.array([[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5]])]

#%% Show temporal cross-correlation:
titles = np.array(['Within V1','Within PM','V1-PM'])
pop_pairs =[np.array([[0,1]]),
            np.array([[2,3]]),
            np.array([[0,2],[0,3],[1,2],[1,3]])]

#%% Compute average rate for different populations: 
fig = plot_crosscorr_poppairs(crosscorrmat_raw,timelags,arealayerlabels,pop_pairs,titles)
# my_savefig(fig,savedir,'Poprate_corr_Xlags_V1PM_%s_%dsessions' % ('_'.join(protocols),nSessions),formats=['png'])
#my_savefig(fig,savedir,'Poprate_corr_Xlags_%s_%s_%dsessions' % (popversion,'_'.join(protocols),nSessions),formats=['png'])

#%% Compute average rate for different populations: 
fig = plot_crosscorr_poppairs(crosscorrmat_diff,timelags,arealayerlabels,pop_pairs,titles)
my_savefig(fig,savedir,'Poprate_corr_diff_Xlags_%s_%s_%dsessions' % (popversion,'_'.join(protocols),nSessions),formats=['png'])

#%% Compute an asymmetry score of the crosscorrelation:

crosscorrmat_norm = crosscorrmat_raw - np.nanmin(crosscorrmat_raw,axis=2,keepdims=True)
crosscorrmat_norm = crosscorrmat_norm / np.nanmax(crosscorrmat_raw,axis=2,keepdims=True)

asymm_score_raw = (np.sum(crosscorrmat_norm[:,:,lags<0,:],axis=2) - np.sum(crosscorrmat_norm[:,:,lags>0,:],axis=2)) / np.sum(crosscorrmat_raw[:,:,lags!=0,:],axis=2)
# asymm_score_raw = (np.sum(crosscorrmat_raw[:,:,lags<0,:],axis=2) - np.sum(crosscorrmat_raw[:,:,lags>0,:],axis=2)) / np.sum(crosscorrmat_raw[:,:,lags!=0,:],axis=2)
asymm_score_diff = (np.sum(crosscorrmat_diff[:,:,lags<0,:],axis=2) - np.sum(crosscorrmat_diff[:,:,lags>0,:],axis=2)) / np.sum(crosscorrmat_diff[:,:,lags!=0,:],axis=2)

fig,axes = plt.subplots(1,2,figsize=(6,3))

vmin        = -0.5
vmax        = 0.5

#vmin        = -0.2
#vmax        = 0.2
datatoplot  = np.nanmean(asymm_score_raw,axis=2)
np.fill_diagonal(datatoplot, np.nan)

ax = axes[0]
ax.imshow(datatoplot,cmap='RdBu_r',vmin=vmin,vmax=vmax,origin='upper')
ax.set_facecolor("lightgray")
ax.set_xticks(range(nArealayerlabels),labels=arealayerlabels,rotation=90)
ax.set_yticks(range(nArealayerlabels),labels=arealayerlabels)
# cbar = fig.colorbar(ax.imshow(datatoplot,cmap='RdBu_r',vmin=vmin,vmax=vmax), ax=ax,shrink=0.5)
ax.set_title('Asymmetry Score Raw')

if popversion == 'V1PMlabeled':
    #Stats comparing labeled as a source pop
    for i in np.arange(0,nArealayerlabels,step=2):
        for j in np.arange(0,nArealayerlabels,step=1):
            if i==j or i+1==j:
                continue
            p = stats.ttest_rel(asymm_score_raw[i,j,:],asymm_score_raw[i+1,j,:],nan_policy='omit')[1]
            if p < 0.05:
                ax.plot([j,j],[i+0.1,i+0.8],'k-',linewidth=1.5)

    #Stats comparing labeled as a target pop
    for i in np.arange(0,nArealayerlabels,step=1):
        for j in np.arange(0,nArealayerlabels,step=2):
            if i==j or i==j+1:
                continue
            p = stats.ttest_rel(asymm_score_raw[i,j,:],asymm_score_raw[i,j+1,:],nan_policy='omit')[1]
            if p < 0.05:
                # ax.plot([j,j],[i+0.1,i+0.8],'k-',linewidth=1.5)
                ax.plot([j+0.1,j+0.8],[i,i],'k-',linewidth=1.5)

datatoplot  = np.nanmean(asymm_score_diff,axis=2)
np.fill_diagonal(datatoplot, np.nan)
vmin        = -0.5
vmax        = 0.5
ax = axes[1]
ax.imshow(datatoplot,cmap='RdBu_r',vmin=vmin,vmax=vmax)
ax.set_facecolor("lightgray")
ax.set_xticks(range(nArealayerlabels),labels=arealayerlabels,rotation=90)
ax.set_yticks(range(nArealayerlabels),labels=arealayerlabels)
cbar = fig.colorbar(ax.imshow(datatoplot,cmap='RdBu_r',vmin=vmin,vmax=vmax), ax=ax,shrink=0.5)
ax.set_title('Asymmetry Score Diff')
plt.tight_layout()
my_savefig(fig,savedir,'Poprate_asymm_score_%s_%s_%dsessions' % (popversion,'_'.join(protocols),nSessions),formats=['png'])

#%% 
fig, axes = plt.subplots(1, gc_lagorder, figsize=(gc_lagorder*4, 4), sharey=True)

vmin = -0.2
vmax = 0.2
for i in range(gc_lagorder):
    ax = axes[i]
    datatoplot = np.nanmean(VAR_raw[:,:,i,:],axis=-1)
    np.fill_diagonal(datatoplot, np.nan)

    ax.imshow(datatoplot, cmap='PiYG', interpolation='nearest',vmin=vmin,vmax=vmax)
    ax.set_title(f'Lag {i+1}')
    ax.set_xticks(range(nArealayerlabels))
    ax.set_xticklabels(arealayerlabels, rotation=90)
    ax.set_yticks(range(nArealayerlabels))
    ax.set_yticklabels(arealayerlabels)
    ax.set_xlabel('Target Pop')
    if i == 0:
        ax.set_ylabel('Source Pop')

my_savefig(fig,savedir,'VAR_%s_%s_%dsessions' % (popversion,'_'.join(protocols),nSessions),formats=['png'])

# fig.colorbar(ax.imshow(coefficients[0], cmap='hot', interpolation='nearest'), ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
# fig.suptitle('VAR Model Coefficients for Each Lag')
# plt.tight_layout()
# plt.show()





# VAR_raw

#%%
x = datamat[~np.isnan(datamat).any(axis=1),:].T
model = VAR(x)
nLags = 4
results = model.fit(maxlags=nLags)

coefficients = results.coefs

# covariance_matrix = results.cov_params
# coefficients = results.tvalues
fig, axes = plt.subplots(1, nLags, figsize=(nLags*4, 4), sharey=True)

for i in range(nLags):
    ax = axes[i]
    ax.imshow(coefficients[i], cmap='hot', interpolation='nearest',vmin=0,vmax=0.1)
    ax.set_title(f'Lag {i+1}')
    ax.set_xticks(range(nArealayerlabels))
    ax.set_xticklabels(arealayerlabels, rotation=90)
    ax.set_yticks(range(nArealayerlabels))
    ax.set_yticklabels(arealayerlabels)
    ax.set_xlabel('Target Pop')
    if i == 0:
        ax.set_ylabel('Source Pop')

fig.colorbar(ax.imshow(coefficients[0], cmap='hot', interpolation='nearest'), ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
fig.suptitle('VAR Model Coefficients for Each Lag')
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.imshow(covariance_matrix, cmap='hot', interpolation='nearest')
# plt.xlabel('Signal')
# plt.ylabel('Signal')
# plt.title('VAR Model Covariance Matrix')
# plt.show()


# %%
