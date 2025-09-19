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
from utils.gain_lib import *

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
sessions = compute_pop_coupling(sessions)

# #%% Add how neurons are coupled to the population rate: 
# for ses in tqdm(sessions,desc='Computing population coupling for each session'):
#     resp        = stats.zscore(ses.respmat.T,axis=0)
#     poprate     = np.mean(resp, axis=1)
#     # popcoupling = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(N)]

#     ses.celldata['pop_coupling']   = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(len(ses.celldata))]

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
#     # ####### ###  #####  #######     #####  ####### ######  ######  ####### #          #    ####### ### ####### #     #  #####  
##    # #     #  #  #     # #          #     # #     # #     # #     # #       #         # #      #     #  #     # ##    # #     # 
# #   # #     #  #  #       #          #       #     # #     # #     # #       #        #   #     #     #  #     # # #   # #       
#  #  # #     #  #   #####  #####      #       #     # ######  ######  #####   #       #     #    #     #  #     # #  #  #  #####  
#   # # #     #  #        # #          #       #     # #   #   #   #   #       #       #######    #     #  #     # #   # #       # 
#    ## #     #  #  #     # #          #     # #     # #    #  #    #  #       #       #     #    #     #  #     # #    ## #     # 
#     # ####### ###  #####  #######     #####  ####### #     # #     # ####### ####### #     #    #    ### ####### #     #  #####  



#%% ########################## Compute signal and noise correlations: ###################################
sessions_orig       = compute_signal_noise_correlation(sessions,filter_stationary=False,uppertriangular=False)
# sessions_nogain     = compute_sign

#%% 

for ises,ses in enumerate(sessions):
    # sessions[ises].joint_coupling = np.outer(ses.celldata['pop_coupling'].values,ses.celldata['pop_coupling'].values)
    sessions[ises].joint_coupling = np.outer(ses.celldata['pop_coupling'].values,ses.celldata['pop_coupling'].values)

# x = np.array([1, 2, 3])
# np.subtract.outer(x, x)

#%% 
plt.imshow(sessions[ises].joint_coupling,vmin=-0.1,vmax=0.2)

#%% 
from scipy.stats import binned_statistic_2d

#%% 
fig,axes = plt.subplots(1,3,figsize=(11,4))
subsample = 100
ax = axes[0]
ax.scatter(sessions[ises].joint_coupling.flatten()[::subsample],sessions[ises].noise_corr.flatten()[::subsample],c='k',s=1)
ax.set_xlabel('Joint coupling')
ax.set_ylabel('Noise correlation')
ax.set_title('Population coupling vs noise correlation')

ax = axes[1]
ax.scatter(sessions[ises].sig_corr.flatten()[::subsample],sessions[ises].noise_corr.flatten()[::subsample],c='k',s=1)
ax.set_ylabel('Noise correlation')
ax.set_xlabel('Signal correlation')
ax.set_title('Signal corr vs noise correlation')
xdata = sessions[ises].sig_corr.flatten()
ydata = sessions[ises].joint_coupling.flatten()
vdata = sessions[ises].noise_corr.flatten()

# Problem with dimensions here! 
#be careful with x and y dimensions here! 
#Need to copy from NC tuning GR with pcolormesh etc.

shared_idx = ~np.isnan(xdata) & ~np.isnan(ydata) & ~np.isnan(vdata)
xdata = xdata[shared_idx]
ydata = ydata[shared_idx]
vdata = vdata[shared_idx]

ax = axes[2]
g = binned_statistic_2d(xdata, ydata, vdata, statistic='mean', bins=10, range=None, expand_binnumbers=False)[0]
ax.imshow(g,origin='lower',extent=[-1,1,-1,1],vmin=-0.8,vmax=0.8,cmap='bwr',)
ax.set_facecolor('gray')
ax.set_xlabel('Signal correlation')
ax.set_ylabel('Joint coupling')
ax.set_title('Joint influence on noise correlation')
cb = fig.colorbar(ax.images[0], ax=ax, shrink=0.5)
cb.set_label('Noise correlation',fontsize=10,loc='center')
sns.despine(fig=fig, top=True, right=True,offset=3)
fig.tight_layout()
my_savefig(fig,savedir,'IM_Signal_Coupling_Noise_correlation_%s' % (sessions[ises].session_id), formats = ['png'])



## COPIED FROM GR: NEED TO ADAPT TO IM: !!!


# #%% 
# ises = 8
# tunethr = 0.5

# fig,axes = plt.subplots(1,2,figsize=(8,4))

# # idx_N = np.outer(sessions[ises].celldata['gOSI']>tunethr,sessions[ises].celldata['gOSI']>tunethr)
# idx_N = np.outer(sessions[ises].celldata['OSI']>tunethr,sessions[ises].celldata['OSI']>tunethr)
# # idx_N = np.outer(sessions[ises].celldata['tuning_var']>0.1,sessions[ises].celldata['tuning_var']>0.1)
# # idx_N = np.outer(sessions[ises].celldata['tuning_var']>0.1,sessions[ises].celldata['tuning_var']>0.1)

# subsample = 100
# ax = axes[0]
# ax.scatter(sessions[ises].joint_coupling[idx_N].flatten()[::subsample],sessions[ises].noise_corr[idx_N].flatten()[::subsample],c='k',s=1)
# ax.set_xlabel('Joint coupling')
# ax.set_ylabel('Noise correlation')
# ax.set_title('Population coupling vs noise correlation')

# ax = axes[1]
# ax.scatter(sessions[ises].sig_corr[idx_N].flatten()[::subsample],sessions[ises].noise_corr[idx_N].flatten()[::subsample],c='k',s=1)
# ax.set_ylabel('Noise correlation')
# ax.set_xlabel('Signal correlation')
# ax.set_title('Signal corr vs noise correlation')

# #%% 
# # Get 2D binned map data: 
# areapairs   = ['V1-V1','PM-PM','V1-PM']
# nAreaPairs  = len(areapairs)
# tunethr     = 0.5
# x,y         = [np.linspace(-1,1,20),np.linspace(-0.15,0.5,20)] #be careful with x and y dimensions here! 
# X,Y         = np.meshgrid(y,x)

# data = np.empty((nSessions,nAreaPairs,len(x)-1,len(y)-1)) #for each session, combination of delta pref store the mean noise corr for all and for the top and bottom tuned percentages

# for ises in range(nSessions):
#     for iareapair,areapair in enumerate(areapairs):
        
#         areafilter      = filter_2d_areapair(sessions[ises],areapair)

#         # tunefilter      = np.outer(sessions[ises].celldata['gOSI']>tunethr,sessions[ises].celldata['gOSI']>tunethr)
#         tunefilter      = np.outer(sessions[ises].celldata['OSI']>tunethr,sessions[ises].celldata['OSI']>tunethr)
#         # tunefilter      = np.outer(sessions[ises].celldata['tuning_var']>tunethr,sessions[ises].celldata['tuning_var']>tunethr)

#         nanfilter         = np.all((~np.isnan(sessions[ises].sig_corr),
#                                     ~np.isnan(sessions[ises].joint_coupling),
#                                     ~np.isnan(sessions[ises].noise_corr),
#                                               ),axis=0)

#         cellfilter       = np.all((areafilter,tunefilter,nanfilter),axis=0)

#         xdata = sessions[ises].sig_corr[cellfilter].flatten()
#         ydata = sessions[ises].joint_coupling[cellfilter].flatten()
#         vdata = sessions[ises].noise_corr[cellfilter].flatten()

#         data[ises,iareapair,:,:] = binned_statistic_2d(xdata, ydata, vdata, statistic='mean', bins=[x,y], range=None)[0]

# #%% 
# cmaplim = 0.5
# fig,axes = plt.subplots(1,nAreaPairs,figsize=(nAreaPairs*3,3))
# for iareapair,areapair in enumerate(areapairs):
#     ax = axes[iareapair]
#     pcm = ax.pcolor(X,Y,np.nanmean(data[:,iareapair,:,:],axis=0),vmin=-cmaplim,vmax=cmaplim,cmap='bwr') #np.nanmean(data[:,iareapair,:,:],vmin=-0.7,vmax=0.7,cmap='bwr')
#     ax.set_facecolor('gray')
#     if iareapair==0: 
#         ax.set_ylabel('Signal correlation')
#     ax.set_xlabel('Joint coupling')
#     ax.set_title('%s' % areapair)
#     ax.axhline(0,c='k',lw=0.25,ls='--')
#     ax.axvline(0,c='k',lw=0.25,ls='--')
#     if iareapair==nAreaPairs-1:
#         cb_ax = fig.add_axes([0.98, 0.5, 0.01, 0.3])
#         cb = fig.colorbar(pcm, cax=cb_ax, shrink=0.5)
#         cb.ax.tick_params(labelsize=7) 
#         # cb.set_label('Noise correlation',fontsize=10,loc='center')
#         # cb.set_label('Noise correlation',fontsize=6,loc='center')
# sns.despine(fig=fig, top=True, right=True,offset=3)
# fig.tight_layout()
# my_savefig(fig,savedir,'IM_Signal_Coupling_NC_Areapairs_%dsessions' % (nSessions), formats = ['png'])

