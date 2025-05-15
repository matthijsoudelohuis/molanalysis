# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('c:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *
from utils.CCAlib import *
from utils.corr_lib import *
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *
from utils.gain_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\CCA\\')

#%% Load an example session: 
session_list        = np.array(['LPE12223_2024_06_10']) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%%  Load data properly:        
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)
    
#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Make the 3D figure the cone data:
fig = plot_PCA_gratings_3D(sessions[0],thr_tuning=0)
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=0, roll=-10)
axes[0].set_zlim([-5,45])
# fig.savefig(os.path.join(savedir,'Cone_3D_V1_Original_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')



#%% 
 #####   #####     #    
#     # #     #   # #   
#       #        #   #  
#       #       #     # 
#       #       ####### 
#     # #     # #     # 
 #####   #####  #     # 

#%% 
ises        = 0
prePCA      = 200

nOris       = 16

ori         = sessions[ises].trialdata['Orientation']
oris        = np.sort(pd.Series.unique(sessions[ises].trialdata['Orientation']))
ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

pal         = np.tile(sns.color_palette('husl', int(len(oris)/2)), (2, 1))
        
idx_V1      = sessions[ises].celldata['roi_name']=='V1'
idx_PM      = sessions[ises].celldata['roi_name']=='PM'

## Split data into area 1 and area 2:
X               = sessions[ises].respmat[idx_V1,:].T
Y               = sessions[ises].respmat[idx_PM,:].T

X               = zscore(X,axis=0)  #Z score activity for each neuron
Y               = zscore(Y,axis=0)

if prePCA:
    pca         = PCA(n_components=prePCA)
    X           = pca.fit_transform(X)
    Y           = pca.fit_transform(Y)

model = CCA(n_components = 3,scale = False, max_iter = 1000)

Xp,Yp = model.fit_transform(X,Y)

# fig,ax = plt.subplots(1,2,figsize=(8,6))
fig = plt.figure(figsize=(8,6))

for iarea, (area,ccaproj) in enumerate(zip(['V1','PM'],[Xp,Yp])):

    ax = fig.add_subplot(1, 2, iarea+1, projection='3d')

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        # get all data points for this ori along first PC or projection pairs
        x = ccaproj.T[0, ori_ind[t]]
        y = ccaproj.T[1, ori_ind[t]]  # and the second
        z = ccaproj.T[2, ori_ind[t]]  # and the second
        # each trial is one dot
        # ax.scatter(x, y, z, color=pal[t], s=sizes[ori_ind[t]]*6, alpha=0.8)
        ax.scatter(x, y, z, color=pal[t], s=2, alpha=0.8)
    # if plotgainaxis:
        # ax.plot(Xg[0,:],Xg[1,:],Xg[2,:],color='k',linewidth=1)
    ax.set_xlabel('PC 1')  # give labels to axes
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    ax.set_title(area)
    nticks = 5
    ax.grid(True)
    ax.set_facecolor('white')
    ax.set_xticks(np.linspace(np.percentile(ccaproj[:,0],1),np.percentile(ccaproj[:,0],99),nticks))
    ax.set_yticks(np.linspace(np.percentile(ccaproj[:,1],1),np.percentile(ccaproj[:,1],99),nticks))
    ax.set_zticks(np.linspace(np.percentile(ccaproj[:,2],1),np.percentile(ccaproj[:,2],99),nticks))
    
    ax.set_xlim(np.percentile(ccaproj[:,0],[1,99]))
    ax.set_ylim(np.percentile(ccaproj[:,1],[1,99]))
    ax.set_zlim(np.percentile(ccaproj[:,2],[1,99]))
    # ax.locator_params(axis='x', nbins=4)
    # ax.locator_params(axis='y', nbins=4)
    # ax.locator_params(axis='z', nbins=4)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Get rid of colored axes planes, remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # ax.view_init(elev=30, azim=-15, roll=45)
# plt.tight_layout()
my_savefig(fig,savedir,'CCA_GR_V1PM_3D_%s' % sessions[ises].sessiondata['session_id'][0],formats=['png'])


#%% Fit population gain model:
ises = 0
orientations        = sessions[ises].trialdata['Orientation']
data                = sessions[ises].respmat
prefori             = sessions[ises].celldata['pref_ori']
data_hat_poprate    = pop_rate_gain_model(data, orientations)

datasets            = (data,data_hat_poprate)
fig = plot_respmat(orientations, datasets, ['original','pop rate gain'],prefori)

#%% 

#%% Make a plot of the CCA structure without any sorting of the trials: 
ises1           = 8
ises2           = 9
prePCA          = 200

sortmethod      = 'origain'

pal             = np.tile(sns.color_palette('husl', int(len(oris)/2)), (2, 1))
        
idx_V1          = sessions[ises1].celldata['roi_name']=='V1'
X               = sessions[ises1].respmat[idx_V1,:].T

idx_V1          = sessions[ises2].celldata['roi_name']=='V1'
Y               = sessions[ises2].respmat[idx_V1,:].T

X               = zscore(X,axis=0)  #Z score activity for each neuron
Y               = zscore(Y,axis=0)

if sortmethod == 'random':
    sort1 = np.random.permutation(X.shape[0])
    sort2 = np.random.permutation(X.shape[0])
elif sortmethod == 'ori':
    sort1 = np.argsort(sessions[ises1].trialdata['Orientation'])
    sort2 = np.argsort(sessions[ises2].trialdata['Orientation'])
elif sortmethod == 'gain':
    sort1 = np.argsort(np.nanmean(X,axis=1))
    sort2 = np.argsort(np.nanmean(Y,axis=1))
elif sortmethod == 'origain':
    sort1           = np.lexsort((np.nanmean(X,axis=1), sessions[ises1].trialdata['Orientation']))[::-1]
    sort2           = np.lexsort((np.nanmean(Y,axis=1), sessions[ises2].trialdata['Orientation']))[::-1]

X = X[sort1,:]
Y = Y[sort2,:]
orisort = np.stack((sessions[ises1].trialdata['Orientation'][sort1],
                 sessions[ises2].trialdata['Orientation'][sort2]))

if prePCA:
    pca         = PCA(n_components=prePCA)
    X           = pca.fit_transform(X)
    Y           = pca.fit_transform(Y)

model = CCA(n_components = 3,scale = False, max_iter = 1000)
Xp,Yp = model.fit_transform(X,Y)

# # the default is no regularization meaning this is SUMCORR-AVGVAR MCCA
# cca = CCAmv(n_components=3,regs=0)
# cca = MCCA(n_components=3, regs=0)
# Xs = [X,Y]
# the fit-transform method outputs the scores for each view
# cca_scores = cca.fit_transform(Xs[:2])
# Xp,Yp = cca_scores

# fig,ax = plt.subplots(1,2,figsize=(8,6))
fig = plt.figure(figsize=(8,6))

# for iarea, (area,ccaproj) in enumerate(zip(['V1','PM'],[Xp,Yp])):
for ises, (plottitle,ccaproj) in enumerate(zip(['V1-Session1','V1-Session2'],[Xp,Yp])):

    ax = fig.add_subplot(1, 2, ises+1, projection='3d')

    ori         = orisort[ises]
    oris        = np.sort(pd.Series.unique(sessions[ises].trialdata['Orientation']))
    ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        # get all data points for this ori along first PC or projection pairs
        x = ccaproj.T[0, ori_ind[t]]
        y = ccaproj.T[1, ori_ind[t]]  # and the second
        z = ccaproj.T[2, ori_ind[t]]  # and the second
        # each trial is one dot
        # ax.scatter(x, y, z, color=pal[t], s=sizes[ori_ind[t]]*6, alpha=0.8)
        ax.scatter(x, y, z, color=pal[t], s=1, alpha=0.8)
    # if plotgainaxis:
        # ax.plot(Xg[0,:],Xg[1,:],Xg[2,:],color='k',linewidth=1)
    ax.set_xlabel('PC 1')  # give labels to axes
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    ax.set_title(plottitle)
    nticks = 5
    ax.grid(True)
    ax.set_facecolor('white')
    ax.set_xticks(np.linspace(np.percentile(ccaproj[:,0],1),np.percentile(ccaproj[:,0],99),nticks))
    ax.set_yticks(np.linspace(np.percentile(ccaproj[:,1],1),np.percentile(ccaproj[:,1],99),nticks))
    ax.set_zticks(np.linspace(np.percentile(ccaproj[:,2],1),np.percentile(ccaproj[:,2],99),nticks))
    
    ax.set_xlim(np.percentile(ccaproj[:,0],[1,99]))
    ax.set_ylim(np.percentile(ccaproj[:,1],[1,99]))
    ax.set_zlim(np.percentile(ccaproj[:,2],[1,99]))
    # ax.locator_params(axis='x', nbins=4)
    # ax.locator_params(axis='y', nbins=4)
    # ax.locator_params(axis='z', nbins=4)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Get rid of colored axes planes, remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # ax.view_init(elev=30, azim=-15, roll=45)
# plt.tight_layout()
# my_savefig(fig,savedir,'CCA_GR_V1PM_3D_%s' % sessions[ises].sessiondata['session_id'][0],formats=['png'])
my_savefig(fig,savedir,'CCA_GR_AcrossSession_V1_%s_with_%s' % (sessions[ises1].sessiondata['session_id'][0],
                                                                sessions[ises2].sessiondata['session_id'][0]),formats=['png'])


#%%

for ises in range(nSessions):
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion=calciumversion,keepraw=True)
    fig = plot_PCA_gratings_3D(sessions[ises],thr_tuning=0)




#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',min_trials=3200)

# sessions = sessions[:3]
# nSessions = len(sessions)

#%%  Load data properly:        
calciumversion = 'dF'
calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=True)
    
#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


#%%


from mvlearn.datasets import sample_joint_factor_model
from mvlearn.embed import CCA as CCAmv
from mvlearn.embed import MCCA, KMCCA
from mvlearn.plotting import crossviews_plot
from mvlearn.decomposition import GroupPCA


n_views = 3
n_samples = 1000
n_features = [10, 20, 30]
joint_rank = 3

# sample 3 views of data from a joint factor model
# m, noise_std control the difficulty of the problem
Xs, U_true, Ws_true = sample_joint_factor_model(
    n_views=n_views, n_samples=n_samples, n_features=n_features,
    joint_rank=joint_rank, m=5, noise_std=1, random_state=23,
    return_decomp=True)

#%%
# the default is no regularization meaning this is SUMCORR-AVGVAR MCCA
cca = CCAmv(n_components=joint_rank)

# the fit-transform method outputs the scores for each view
cca_scores = cca.fit_transform(Xs[:2])
crossviews_plot(cca_scores,
                title='CCA scores (first two views fitted)',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})

# In the 2 view setting, a variety of interpretable statistics can be
# calculated. We assess the canonical correlations achieved and
# their significance using the p-values from a Wilk's Lambda test

stats = cca.stats(cca_scores)
print(f'Canonical Correlations: {stats["r"]}')
print(f'Wilk\'s Lambda Test pvalues: {stats["pF"]}')

#%%

prePCA= 200
# from mvlearn.embed import MultiviewCCA
sortmethod = 'random'
sortmethod = 'gain'
sortmethod = 'origain'

Xs          = []
trialsort   = []
for ises,ses in enumerate(sessions):
    idx_N           = ses.celldata['roi_name'] == 'V1'
    # idx_N           = np.all((ses.celldata['roi_name'] == 'V1',
    #                           ses.celldata['noise_level']<100,
    #                           ses.celldata['tuning_var']>0.0),axis=0)

    X               = ses.respmat[idx_N,:].T
    X               = zscore(X,axis=0)  #Z score activity for each neuron

    if sortmethod == 'random':
        sort    = np.random.permutation(X.shape[0])
    elif sortmethod == 'ori':
        sort    = np.argsort(ses.trialdata['Orientation'])
    elif sortmethod == 'gain':
        sort    = np.argsort(np.nanmean(X,axis=1))
    elif sortmethod == 'origain':
        sort    = np.lexsort((np.nanmean(X,axis=1), ses.trialdata['Orientation']))[::-1]
    X   = X[sort,:]

    if prePCA:
        pca         = PCA(n_components=np.min([prePCA,X.shape[1]]))
        X           = pca.fit_transform(X)

    Xs.append(X)
    trialsort.append(ses.trialdata['Orientation'][sort])

#%% Multi views:
reglam = 0.5
# regularization value of .5 for each view
mcca = MCCA(n_components=3, regs=reglam)

# the fit-transform method outputs the scores for each view
cca_scores = mcca.fit_transform(Xs)
# cca_scores = mcca.fit_transform(Xs[8:])
crossviews_plot(cca_scores[[0, 1]],
                title='MCCA scores with regularization (first 2 views shown)',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})

print('Canonical Correlations:')
print(mcca.canon_corrs(cca_scores))

#%% 
nviews = np.shape(cca_scores)[0]
fig = plt.figure(figsize=(nviews*4,6))

for ises in range(nviews):
    ccaproj     = cca_scores[ises]

    ax          = fig.add_subplot(1, nviews, ises+1, projection='3d')
    # ax          = fig.add_subplot(2, 5, ises+1, projection='3d')

    ori         = trialsort[ises]
    oris        = np.sort(pd.Series.unique(sessions[ises].trialdata['Orientation']))
    ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        # get all data points for this ori along first PC or projection pairs
        x = ccaproj.T[0, ori_ind[t]]
        y = ccaproj.T[1, ori_ind[t]]  # and the second
        z = ccaproj.T[2, ori_ind[t]]  # and the third
        # each trial is one dot
        ax.scatter(x, y, z, color=pal[t], s=1, alpha=0.8)
    # if plotgainaxis:
        # ax.plot(Xg[0,:],Xg[1,:],Xg[2,:],color='k',linewidth=1)
    ax.set_xlabel('CCA 1')  # give labels to axes
    ax.set_ylabel('CCA 2')
    ax.set_zlabel('CCA 3')

    ax.set_title('Session '+str(ises+1))
    nticks = 5
    ax.grid(True)
    ax.set_facecolor('white')
    ax.set_xticks(np.linspace(np.percentile(ccaproj[:,0],1),np.percentile(ccaproj[:,0],99),nticks))
    ax.set_yticks(np.linspace(np.percentile(ccaproj[:,1],1),np.percentile(ccaproj[:,1],99),nticks))
    ax.set_zticks(np.linspace(np.percentile(ccaproj[:,2],1),np.percentile(ccaproj[:,2],99),nticks))
    
    ax.set_xlim(np.percentile(ccaproj[:,0],[1,99]))
    ax.set_ylim(np.percentile(ccaproj[:,1],[1,99]))
    ax.set_zlim(np.percentile(ccaproj[:,2],[1,99]))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Get rid of colored axes planes, remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # ax.view_init(elev=30, azim=-15, roll=45)
plt.tight_layout()
my_savefig(fig,savedir,'GR_CCA_interarea_intersession_%s_%dsessions.png' % (sortmethod,nSessions),formats=['png'])

#%%
fig,axes = plt.subplots(1,3,figsize=(11,3))

cc = mcca.canon_corrs(cca_scores)
for icomp in range(3):
    ax = axes[icomp]
    sns.heatmap(cc[icomp,:,:],ax=ax,cmap='bwr',vmin=-1,vmax=1)
    ax.set_xlabel('Session')
    ax.set_ylabel('Session')
    ax.set_title('View %d' % icomp)
    ax.set_xticks(np.arange(nSessions)+0.5)
    ax.set_yticks(np.arange(nSessions)+0.5)
    ax.set_xticklabels(np.arange(nSessions)+1)
    ax.set_yticklabels(np.arange(nSessions)+1)
plt.tight_layout()
my_savefig(fig,savedir,'GR_CCA_CanonCorr_%s_%dsessions.png' % (sortmethod,nSessions),formats=['png'])
