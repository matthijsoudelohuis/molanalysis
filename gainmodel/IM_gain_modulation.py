# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023-2025, Champalimaud Center
"""

#%% 
import os
os.chdir('c:\\Python\\molanalysis')
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
session_list        = np.array([['LPE11086','2023_12_16']])
session_list        = np.array([['LPE13959','2025_02_24']])

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['IM'],only_session_id=session_list)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_lab_cells_V1=50,min_lab_cells_PM=50)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_cells=2000)
sessions,nSessions   = filter_sessions(protocols = ['IM'],im_ses_with_repeats=True)

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
    resp = stats.zscore(ses.respmat.T,axis=0)
    poprate = np.mean(resp, axis=1)
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
