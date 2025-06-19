# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023-2025, Champalimaud Center
"""

#%% 
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.imagelib import load_natural_images #
from utils.tuning import *
from utils.corr_lib import compute_signal_noise_correlation

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\')


#%% ################################################
session_list        = np.array([['LPE11086_2023_12_16']])
session_list        = np.array([['LPE13959_2025_02_24']])

session_list        = np.array([['LPE13959_2025_02_24',
                                 'LPE11086_2023_12_16']])

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['IM'],only_session_id=session_list)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_lab_cells_V1=50,min_lab_cells_PM=50)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_cells=2000)
sessions,nSessions   = filter_sessions(protocols = ['IM'],im_ses_with_repeats=True)
sessions,nSessions   = filter_sessions(protocols = ['IM'])

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    # sessions[ises].load_respmat(calciumversion='dF',keepraw=False)
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

#%% ### Load the natural images:
# natimgdata = load_natural_images(onlyright=False)
natimgdata = load_natural_images(onlyright=True)

#%% Compute tuning metrics of natural images:
for ses in tqdm(sessions,desc='Computing tuning metrics for each session'): 
    ses.celldata['tuning_SNR']                          = compute_tuning_SNR(ses)
    ses.celldata['corr_half'],ses.celldata['rel_half']  = compute_splithalf_reliability(ses)
    ses.celldata['sparseness']          = compute_sparseness(ses.respmat)
    ses.celldata['selectivity_index']   = compute_selectivity_index(ses.respmat)
    ses.celldata['fano_factor']         = compute_fano_factor(ses.respmat)
    ses.celldata['gini_coefficient']    = compute_gini_coefficient(ses.respmat)

#%% Add how neurons are coupled to the population rate: 
for ses in tqdm(sessions,desc='Computing tuning metrics for each session'):
    resp        = stats.zscore(ses.respmat.T,axis=0)
    poprate     = np.mean(resp, axis=1)
    # popcoupling = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(N)]

    ses.celldata['pop_coupling']                          = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(len(ses.celldata))]

#%% Plot the response across individual trials for some example neurons
# Color the response by the population rate
sesidx      = 0
ses         = sessions[sesidx]
resp        = stats.zscore(ses.respmat.T,axis=0)
poprate     = np.mean(resp, axis=1)

idx_N       = np.where((ses.celldata['pop_coupling'] > 0.3) & (ses.celldata['tuning_SNR']>0.3))[0]

idx_N       = np.random.choice(idx_N,8,replace=False)

im_repeats  = ses.trialdata['ImageNumber'].value_counts()[:100].index.to_numpy()

im_repeats = np.random.choice(im_repeats,20,replace=False)

poprate     -= np.min(poprate)
poprate     /= np.max(poprate)

fig,axes = plt.subplots(2,4,figsize=(15,5))
for iN,N in enumerate(idx_N):
    ax = axes.flatten()[iN]

    for iIM,IM in enumerate(im_repeats):
        idx_T = ses.trialdata['ImageNumber']==IM
        ax.scatter(np.ones(np.sum(idx_T))*iIM,resp[idx_T,N],c=poprate[idx_T],
                   s=(poprate[idx_T]-0.05)*50,cmap='crest',vmin=0.1,vmax=0.3)
sns.despine(fig=fig, top=True, right=True)
fig.tight_layout()


#%% 
nActBins = 10

knn_dec = np.full((nActBins,nSessions),np.nan)

for ises,ses in enumerate(sessions):

    resp    = ses.respmat.T
    istim   = np.array(ses.trialdata['ImageNumber'])
    nimg    = istim.max() + 1 # these are blank stims (exclude them)

    # mean center each neuron
    resp -= resp.mean(axis=0)
    resp = resp / (resp.std(axis=0) + 1e-6)

    # compute population response
    poprate     = np.mean(resp, axis=1)

    ### sanity check - decent signal variance ?
    # split stimuli into two repeats
    NN = resp.shape[1]
    sresp = np.zeros((2, nimg, NN), np.float64)
    inan = np.zeros((nimg,)).astype(bool)
    spopresp = np.zeros((nimg,))
    for n in range(nimg):
        ist = (istim==n).nonzero()[0]
        i1 = ist[:int(ist.size/2)]
        i2 = ist[int(ist.size/2):]
        # check if two repeats of stim
        if np.logical_or(i2.size < 1, i1.size < 1):
            inan[n] = 1
        else:
            sresp[0, n, :] = resp[i1, :].mean(axis=0)
            sresp[1, n, :] = resp[i2, :].mean(axis=0)
            spopresp[n] = poprate[[i1,i2]].mean()
    
    # remove image responses without two repeats
    sresp = sresp[:,~inan,:]
    spopresp = spopresp[~inan]

    ### KNN decoding
    # 1 nearest neighbor decoder    
    # (mean already subtracted)
    # sresp = snorm
    cc  = sresp[0] @ sresp[1].T
    cc /= (sresp[0]**2).sum()
    cc /= (sresp[1]**2).sum()
    nstims = sresp.shape[1]
    acc = (cc.argmax(axis=1)==np.arange(0,nstims,1,int)).mean()
    print('decoding accuracy: %2.3f'%acc)

    binedges        = np.percentile(spopresp,np.linspace(0,100,nActBins+1))
    bincenters      = (binedges[1:]+binedges[:-1])/2

    for iap in range(nActBins):
        idx_T       = np.all((spopresp >= binedges[iap],
                            spopresp <= binedges[iap+1]), axis=0)
        
        cc  = sresp[0,idx_T,:] @ sresp[1,idx_T,:].T
        cc /= (sresp[0,idx_T,:]**2).sum()
        cc /= (sresp[1,idx_T,:]**2).sum()
        nstims = np.sum(idx_T)
        acc = (cc.argmax(axis=1)==np.arange(0,nstims,1,int)).mean()
        # print('decoding accuracy: %2.3f'%acc)
        knn_dec[iap,ises] = acc

#%% Plot the accuracy as a function of the population activity
fig,ax = plt.subplots(1,1,figsize=(3.3,3))
for i in range(nSessions):
    ax.plot(np.arange(1,nActBins+1),knn_dec[:,i],c='k',linewidth=0.5,alpha=0.5)

# shaded_error(np.arange(1,nActBins+1),np.nanmean(knn_dec,axis=1),np.nanstd(knn_dec,axis=1)/np.sqrt(nSessions))
shaded_error(np.arange(1,nActBins+1),knn_dec.T,error='sem',ax=ax,linewidth=3)
ax.set_xlabel('Population activity bins')
ax.set_ylabel('Decoding accuracy')
ax.set_title('KNN Natural Image Decoding',fontsize=11)
ax.set_ylim([0,1])
ax.set_xticks(np.arange(1,nActBins+1),np.arange(1,nActBins+1))
ax.axhline(1/2400/nActBins, color='grey', linewidth=2, linestyle='--')
ax.text(1,1/2400/nActBins+0.02,'Chance',color='k',fontsize=8)
sns.despine(fig=fig, top=True, right=True,offset=3)
fig.tight_layout()
my_savefig(fig,savedir,'KNN_decoding_ActBins_%dsessions' % (nSessions), formats = ['png'])
