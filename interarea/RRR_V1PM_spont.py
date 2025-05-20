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
from scipy.stats import zscore
from skimage.measure import block_reduce

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *

#%% 
areas = ['V1','PM']
nareas = len(areas)
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross')

#%% Load data: 
# session_list        = np.array([['LPE12223_2024_06_10'], #GR
#                                 ['LPE10919_2023_11_06']]) #GR
# sessions,nSessions   = filter_sessions(protocols = 'SP',only_session_id=session_list)
sessions,nSessions   = filter_sessions(protocols = ['SP'],filter_areas=areas)

#%%  Load data properly:        
calciumversion = 'deconv'
# calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_data(load_calciumdata=True)

#%% Do RRR in FF and FB direction and compare performance:
nsampleneurons  = 250
nranks          = 25
nmodelfits      = 5 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions,2),np.nan)
optim_rank      = np.full((nSessions,2),np.nan)
R2_ranks        = np.full((nSessions,2,nranks,nmodelfits,kfold),np.nan)

temporalbin     = 0.5
nbin            = int(temporalbin * sessions[0].sessiondata['fs'][0])

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].calciumdata.iloc[:,idx_areax].to_numpy()
    Y                   = sessions[ises].calciumdata.iloc[:,idx_areay].to_numpy()

    X_r     = block_reduce(X, block_size=(nbin,1), func=np.mean, cval=np.mean(X))
    Y_r     = block_reduce(Y, block_size=(nbin,1), func=np.mean, cval=np.mean(Y))

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:
        R2_cv[ises,0],optim_rank[ises,0],R2_ranks[ises,0,:,:,:]      = RRR_wrapper(Y_r, X_r, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
    
        R2_cv[ises,1],optim_rank[ises,1],R2_ranks[ises,1,:,:,:]      = RRR_wrapper(X_r, Y_r, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot the performance across sessions as a function of rank: 
clrs_areapairs = get_clr_area_pairs(['V1-PM','PM-V1'])
datatoplot = np.nanmean(R2_ranks,axis=(3,4))
axlim = np.nanmax(np.nanmean(datatoplot,axis=0))*1.2

fig,axes = plt.subplots(1,3,figsize=(6.5,2.2))
ax = axes[0]
handles = []
handles.append(shaded_error(np.arange(nranks),datatoplot[:,0,:],color=clrs_areapairs[0],error='sem',ax=ax))
handles.append(shaded_error(np.arange(nranks),datatoplot[:,1,:],color=clrs_areapairs[1],error='sem',ax=ax))

ax.legend(handles,['V1->PM','PM->V1'],frameon=False,fontsize=8,loc='lower right')
ax.set_xlabel('Rank')
ax.set_ylabel('R2')
# ax.set_yticks([0,0.05,0.1])
ax.set_ylim([0,axlim])
ax.set_xlim([0,nranks])
ax_nticks(ax,5)

ax = axes[1]
axlim = my_ceil(np.nanmax(datatoplot)*1.1,1)
ax.scatter(R2_cv[:,0],R2_cv[:,1],color='k',s=10)
ax.set_xlabel('V1->PM')
ax.set_ylabel('PM->V1')
ax.plot([0,1],[0,1],color='k',linestyle='--',linewidth=0.5)
ax.set_xlim([0,axlim])
ax.set_ylim([0,axlim])
ax_nticks(ax,3)
t,p = ttest_rel(R2_cv[:,0],R2_cv[:,1],nan_policy='omit')
print('Paired t-test (R2): p=%.3f' % p)
if p<0.05:
    ax.text(0.6,0.2,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='red')
else: 
    ax.text(0.6,0.2,'p=%.3f' % p,transform=ax.transAxes,ha='center',va='center',fontsize=12,color='k')

ax = axes[2]
axlim = my_ceil(np.nanmax(optim_rank)*1.2)
ax.scatter(optim_rank[:,0],optim_rank[:,1],color='k',s=10)
ax.plot([0,20],[0,20],color='k',linestyle='--',linewidth=0.5)
ax.set_xlabel('V1->PM')
ax.set_ylabel('PM->V1')
ax.set_xlim([0,axlim])
ax.set_ylim([0,axlim])
ax_nticks(ax,3)
t,p = ttest_rel(optim_rank[:,0],optim_rank[:,1],nan_policy='omit')
print('Paired t-test (R2): p=%.3f' % p)
if p<0.05:
    ax.text(0.6,0.2,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='red')
else: 
    ax.text(0.6,0.2,'p=n.s.',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='k')
    
# ax.set_ylim([0,0.3])
plt.tight_layout()
sns.despine(top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_R2_acrossranks_V1PM_SPONT_%dsessions_%dsec.png' % (nSessions,temporalbin),formats=['png'])
