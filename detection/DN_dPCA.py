# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% IMPORT LIBS
import os
os.chdir('c:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd

from loaddata.session_info import filter_sessions,load_sessions

from scipy import stats
from scipy.stats import zscore
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score as AUC

from dPCA import dPCA

import seaborn as sns
import matplotlib.pyplot as plt
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *
from loaddata.get_data_folder import get_local_drive

#%% ###############################################################

protocol            = 'DN'

session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE11998', '2024_04_23']])
session_list = np.array([['LPE11997', '2024_04_16']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list        = np.array([['LPE12013','2024_04_25']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion='deconv') #Load specified list of sessions
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE12385'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion='dF') #load sessions that meet criteria:
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE12013'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion='dF') #load sessions that meet criteria:

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\dPCA\\')

#%% ### Show for all sessions which region of the psychometric curve the noise spans #############
sessions = noise_to_psy(sessions,filter_engaged=True)

idx_inclthr = np.empty(nSessions).astype('int')
for ises,ses in enumerate(sessions):
    idx_inclthr[ises] = int(np.logical_and(np.any(sessions[ises].trialdata['signal_psy']<=0),np.any(sessions[ises].trialdata['signal_psy']>=0)))
    ses.sessiondata['incl_thr'] = idx_inclthr[ises]

sessions = [ses for ises,ses in enumerate(sessions) if ses.sessiondata['incl_thr'][0]]
nSessions = len(sessions)

#%% Zscore all the calciumdata:
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################## Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 80   #post cm
binsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')

#%% Compute average response in stimulus response zone:
for i in range(nSessions):
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

for i in range(nSessions):
    temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
                                    sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

# # not working yet, need spatial interpolation of video frames:
# for i in range(nSessions):
#     temp = pd.DataFrame(np.reshape(np.array(sessions[i].videodata['motionenergy']),(len(sessions[0].videodata['motionenergy']),1)))
#     sessions[i].respmat_videome     = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
#                                     sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

#%% dPCA on session tensor:
# ises        = 0 #selected session to plot this for

# sessions[ises].stensor[np.isnan(sessions[ises].stensor)] = 0

# # number of neurons, time-points and stimuli
# [N,t,S]     = np.shape(sessions[ises].stensor) #get dimensions of tensor

# # stimtypes   = sorted(sessions[ises].trialdata['stimcat'].unique()) # Catch, Noise and Max trials if correct
# C = 2
# stimtypes   = ['C','M']
# stimlabels  = ['catch','max']

# c_ind      = np.array([np.array(sessions[ises].trialdata['stimcat']) == stim for stim in stimtypes])

# n_trials    = np.min(np.sum(c_ind,axis=1))

# trialR = np.empty((n_trials,N,C,S))

# for iC in range(C):
#     idx = np.random.choice(np.argwhere(c_ind[iC,:]).squeeze(), size=n_trials, replace=False)  
#     trialR[:,:,iC,:] = sessions[ises].stensor[:,idx,:].transpose((1,0,2))

# print(np.shape(trialR))

# # trial-average data
# R = np.mean(trialR,0)

# # center data
# R -= np.mean(R.reshape((N,-1)),1)[:,None,None]

# dpca = dPCA.dPCA(labels='st',regularizer='auto')
# dpca.protect = ['t']


# #%% 

# plt.figure()
# plt.plot(np.mean(R[:,0,:],axis=0))
# plt.plot(np.mean(R[:,1,:],axis=0))

# #%% 
# Z = dpca.fit_transform(R,trialR)

# #%% 

# plt.figure(figsize=(16,7))
# plt.subplot(131)

# for s in range(C):
#     plt.plot(sbins,Z['t'][0,s])

# plt.title('1st time component')
    
# plt.subplot(132)

# for s in range(C):
#     plt.plot(sbins,Z['s'][0,s])
    
# plt.title('1st stimulus component')
    
# plt.subplot(133)

# for s in range(C):
#     plt.plot(sbins,Z['st'][0,s])
    
# plt.title('1st mixing component')
# plt.show()


#%% dPCA on session tensor:
ises        = 0 #selected session to plot this for

# sessions[ises].stensor[np.isnan(sessions[ises].stensor)] = 0

data = copy.deepcopy(sessions[ises].stensor)
data[np.isnan(data)] = 0

idx_N = sessions[ises].celldata['roi_name']=='V1'
idx_N = sessions[ises].celldata['roi_name']=='PM'

idx_N  = np.all((sessions[ises].celldata['roi_name']=='PM',
                 sessions[ises].celldata['noise_level']<20),axis=0)
data = data[idx_N,:,:]

# number of neurons, time-points and stimuli
[N,t,S]     = np.shape(data) #get dimensions of tensor

# stimtypes   = sorted(sessions[ises].trialdata['stimcat'].unique()) # Catch, Noise and Max trials if correct
C = 2
stimtypes   = ['C','M']
stimlabels  = ['catch','max']

C = 2
stimtypes   = ['C','N']
stimlabels  = ['catch','noise']

C = 3
stimtypes   = ['C','N','M']
stimlabels  = ['catch','noise','max']

D = 2
dectypes    = [0,1]
declabels  = ['no lick','lick']

c_ind      = np.array([np.array(sessions[ises].trialdata['stimcat']) == stim for stim in stimtypes])

d_ind      = np.array([np.array(sessions[ises].trialdata['lickResponse']) == dec for dec in dectypes])

n_trials = np.empty((C,D))
for iC in range(C):
    for iD in range(D):
        n_trials[iC,iD] = np.sum(np.logical_and(c_ind[iC,:],d_ind[iD,:]))

n_min_trials = np.min(n_trials).astype('int')
n_min_trials = 50

trialR = np.empty((n_min_trials,N,C,D,S))

for iC in range(C):
    for iD in range(D):
        # idx = np.random.choice(np.argwhere(np.logical_and(c_ind[iC,:],d_ind[iD,:])).squeeze(), size=n_min_trials, replace=False)  
        idx = np.random.choice(np.argwhere(np.logical_and(c_ind[iC,:],d_ind[iD,:])).squeeze(), size=n_min_trials, replace=True)  
        trialR[:,:,iC,iD,:] = data[:,idx,:].transpose((1,0,2))
        # trialR[:,:,iC,iD,:] = data[:,idx,:]

print(np.shape(trialR))

# trial-average data
R = np.mean(trialR,0)

# center data
R -= np.mean(R.reshape((N,-1)),1)[:,None,None,None]
# center trialR data:
# trialR -= np.mean(trialR.reshape((n_min_trials,N,-1)),1)[:,None,None,None]

#%% 
regval = 0.003
regval = 0.001
ncomponents = 3

dpca = dPCA.dPCA(labels='sdt',regularizer=regval,n_components=ncomponents)
# dpca = dPCA.dPCA(labels='sdt',regularizer='auto',n_components=ncomponents)
# dpca = dPCA.dPCA(labels='tsd',regularizer=regval)

dpca.protect = ['t']

Z = dpca.fit_transform(R,trialR)

#%%

# for regval in [0,0.001,0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]:
#     ncomponents = 3
#     dpca = dPCA.dPCA(labels='sdt',regularizer=regval,n_components=ncomponents)
#     Z = dpca.fit_transform(R,trialR)
#     values = dpca.explained_variance_ratio_.values()
#     print(regval)

#     print(np.sum(list(values)))

#%% Plot:

linecolors_c = ['grey','green','blue']
linestyles_d = ['-','--',':']

# plt.figure(figsize=(16,4))
fig,axes = plt.subplots(ncomponents,4,figsize=(16,ncomponents*4)) 

labels = [stimlabels[i]+'-'+declabels[j] for i in range(C) for j in range(D)]

for icomponent in range(ncomponents):
    ax = axes[icomponent,0]
    for c in range(C):
        for d in range(D):
            ax.plot(sbins,Z['t'][icomponent,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])

            # ax.plot(sbins,Z[labels[icomponent]][0,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
    ax.legend(labels,frameon=False)
    # ax.set_title(labels[icomponent])
    ax.set_title('Dim %d - Time component\nEV: %.5f' % (icomponent,dpca.explained_variance_ratio_['t'][icomponent]))

    ax = axes[icomponent,1]
    for c in range(C):
        for d in range(D):
            ax.plot(sbins,Z['st'][icomponent,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
    # ax.set_title('Dim %d - Stimulus component' % icomponent)
    ax.set_title('Dim %d - Stimulus component\nEV: %.5f' % (icomponent,dpca.explained_variance_ratio_['st'][icomponent]))

    ax = axes[icomponent,2]
    for c in range(C):
        for d in range(D):
            ax.plot(sbins,Z['dt'][icomponent,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
    # ax.set_title('Dim %d - Decision component' % icomponent)
    ax.set_title('Dim %d - Decision component\nEV: %.5f' % (icomponent,dpca.explained_variance_ratio_['dt'][icomponent]))

    ax = axes[icomponent,3]
    for c in range(C):
        for d in range(D):
            ax.plot(sbins,Z['sdt'][icomponent,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
    # ax.set_title('Dim %d - Mixing component' % icomponent)
    ax.set_title('Dim %d - Mixing component\nEV: %.5f' % (icomponent,dpca.explained_variance_ratio_['sdt'][icomponent]))

plt.tight_layout()



#%% 

# number of neurons, time-points and stimuli and decisions
N,T,S,D = 50,250,3,2

# noise-level and number of trials in each condition
noise, n_samples = 0.2, 10

# build two latent factors
zd = (np.arange(D)/float(D))
zt = (np.arange(T)/float(T))
zs = (np.arange(S)/float(S))

# build trial-by trial data
trialR = noise*np.random.randn(n_samples,N,S,D,T)
trialR += np.random.randn(N)[None,:,None,None,None]*zt[None,None,None,None,:]
trialR += np.random.randn(N)[None,:,None,None,None]*zs[None,None,:,None,None]
trialR += np.random.randn(N)[None,:,None,None,None]*zd[None,None,None,:,None]

# trial-average data
R = np.mean(trialR,0)

# center data
R -= np.mean(R.reshape((N,-1)),1)[:,None,None,None]

plt.figure()
plt.subplot(1,2,1)
plt.plot(np.mean(np.mean(R[:,0,:,:],axis=0),axis=0))
plt.plot(np.mean(np.mean(R[:,1,:,:],axis=0),axis=0))
plt.plot(np.mean(np.mean(R[:,2,:,:],axis=0),axis=0))
plt.subplot(1,2,2)
plt.plot(np.mean(np.mean(R[:,:,0,:],axis=0),axis=0))
plt.plot(np.mean(np.mean(R[:,:,1,:],axis=0),axis=0))


regval = 0.035

dpca = dPCA.dPCA(labels='sdt',regularizer=regval)
dpca.protect = ['t']


#%% 
Z = dpca.fit_transform(R,trialR)

time = np.arange(T)

plt.figure(figsize=(16,7))
plt.subplot(141)

for s in range(S):
    for d in range(D):
        plt.plot(time,Z['t'][0,s,d])

plt.title('1st time component')
    
plt.subplot(142)

for s in range(S):
    for d in range(D):
        plt.plot(time,Z['s'][0,s,d])
    
plt.title('1st stimulus component')

plt.subplot(143)

for s in range(S):
    for d in range(D):
        plt.plot(time,Z['d'][0,s,d])
    
plt.title('1st decision component')

plt.subplot(144)

for s in range(S):
    for d in range(D):
        plt.plot(time,Z['sd'][0,s,d])
    
plt.title('1st mixing component')
plt.show()

#%% ####################### PCA to understand variability at the population level ####################

def pca_scatter_stimresp(respmat,ses,colorversion='stimresp'):
    stimtypes   = sorted(ses.trialdata['stimcat'].unique()) # stim
    resptypes   = sorted(ses.trialdata['lickResponse'].unique()) # licking resp [0,1]

    X           = zscore(respmat,axis=1)

    pca         = PCA(n_components=15)
    Xp          = pca.fit_transform(X.T).T

    s_type_ind      = [np.argwhere(np.array(ses.trialdata['stimcat']) == stimtype)[:, 0] for stimtype in stimtypes]
    r_type_ind      = [np.argwhere(np.array(ses.trialdata['lickResponse']) == resptype)[:, 0] for resptype in resptypes]

    pal             = sns.color_palette('husl', 4)
    fc              = ['w','k']
    # cmap            = plt.get_cmap('viridis')
    cmap = plt.get_cmap('gist_rainbow')
    cmap = plt.get_cmap('jet')

    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=[12, 4], sharey='row', sharex='row')
    for ax, proj in zip(axes, projections):

        if colorversion=='stimresp':
            for s in range(len(stimtypes)):
                for r in range(len(resptypes)):
                    x = Xp[proj[0], np.intersect1d(s_type_ind[s],r_type_ind[r])]
                    y = Xp[proj[1], np.intersect1d(s_type_ind[s],r_type_ind[r])]
                    # x = Xp[proj[0], s_type_ind[s]]
                    # y = Xp[proj[1], s_type_ind[s]]
                    # ax.scatter(x, y, c=pal[s], s=20, alpha=alp[r],marker='o')
                    # if colorversion=='stimtype':
                    ax.scatter(x, y, s=20, alpha=0.8,marker='o',facecolors=pal[s],edgecolors=fc[r],linewidths=1)
                    # elif colorversion=='runspeed':
                    #     c = cmap(minmax_scale(trialdata['runspeed'][np.intersect1d(s_type_ind[s],r_type_ind[r])], feature_range=(0, 1)))[:,:3]
                    #     ax.scatter(x, y, s=20, alpha=0.8,marker='o',facecolors=c,edgecolors=fc[r],linewidths=1)

        elif colorversion=='runspeed':
            # for r in range(len(resptypes)):
            #     x = Xp[proj[0],r_type_ind[r]]
            #     y = Xp[proj[1],r_type_ind[r]]
                
            #     c = cmap(minmax_scale(np.squeeze(ses.respmat_runspeed[:,r_type_ind[r]]), feature_range=(0, 1)))[:,:3]

            #     ax.scatter(x, y, s=20, c=c, alpha=0.8,marker='o',edgecolors=fc[r],linewidths=1)
            x = Xp[proj[0],:]
            y = Xp[proj[1],:]

            c = cmap(minmax_scale(np.squeeze(ses.respmat_runspeed), feature_range=(0, 1)))[:,:3]

            ax.scatter(x, y, s=20, c=c, alpha=0.8,marker='o',edgecolors='w',linewidths=1)
           
        elif colorversion=='signal':
            # for r in range(len(resptypes)):
            #     x = Xp[proj[0],r_type_ind[r]]
            #     y = Xp[proj[1],r_type_ind[r]]
                
            #     c = cmap(minmax_scale(np.squeeze(ses.trialdata['signal'][r_type_ind[r]]), feature_range=(0, 1)))[:,:3]

            #     ax.scatter(x, y, s=20, c=c, alpha=0.8,marker='o',edgecolors=fc[r],linewidths=1)
            x = Xp[proj[0],:]
            y = Xp[proj[1],:]

            c = cmap(minmax_scale(ses.trialdata['signal'], feature_range=(0, 1)))[:,:3]

            ax.scatter(x, y, s=20, c=c, alpha=0.8,marker='o',edgecolors='w',linewidths=1)
            
    ax.set_xlabel('PC {}'.format(proj[0]+1))
    ax.set_ylabel('PC {}'.format(proj[1]+1))

    sns.despine(fig=fig, top=True, right=True)

    custom_lines = [Line2D([0], [0], color=pal[k], lw=0,markersize=10,marker='o') for
                    k in range(len(stimtypes))]
    labels = stimtypes
    ax.legend(custom_lines, labels,title='Stim',
            frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])

    return fig

#%% 
sesidx = 1
#For all areas:
fig = pca_scatter_stimresp(sessions[sesidx].respmat,sessions[sesidx],colorversion='stimresp')
plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_stimResp_allAreas_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = pca_scatter_stimresp(sessions[sesidx].respmat,sessions[sesidx],colorversion='runspeed')
plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_runspeed_allAreas_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = pca_scatter_stimresp(sessions[sesidx].respmat,sessions[sesidx],colorversion='signal')
plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_signal_allAreas_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#For each area:
for iarea,area in enumerate(areas):
    idx         = sessions[sesidx].celldata['roi_name'] == area
    # respmat     = np.nanmean(sessions[sesidx].stensor[np.ix_(idx,range(K),(sbins>0) & (sbins<20))],axis=2) 
    respmat     = sessions[sesidx].respmat[idx,:]

    fig = pca_scatter_stimresp(respmat,sessions[sesidx],colorversion='stimresp')
    plt.suptitle(area,fontsize=14)
    plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_stimResp_' + area + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    fig = pca_scatter_stimresp(respmat,sessions[sesidx],colorversion='runspeed')
    plt.suptitle(area,fontsize=14)
    plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_runspeed_' + area + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    fig = pca_scatter_stimresp(respmat,sessions[sesidx],colorversion='signal')
    plt.suptitle(area,fontsize=14)
    plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_signal_' + area + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    # pca_scatter_stimresp(respmat,sessions[sesidx])
    # plt.suptitle(area,fontsize=14)
    # plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_stimResponse_' + area + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

################################################################


#%% ################## PCA unsupervised display of noise around center for each condition #################
# split into areas:

# idx_V1_tuned = np.logical_and(sessions[sesidx].celldata['roi_name']=='V1',sessions[sesidx].celldata['tuning']>0.4)
# idx_PM_tuned = np.logical_and(sessions[sesidx].celldata['roi_name']=='PM',sessions[sesidx].celldata['tuning']>0.4)

A1 = sessions[sesidx].respmat
# A2 = sessions[sesidx].respmat[idx_PM_tuned,:]

idx_V1 = np.where(sessions[sesidx].celldata['roi_name']=='V1')[0]
# idx_V1 = np.where(sessions[sesidx].celldata['roi_name']=='AL')[0]
# idx_PM = np.where(sessions[sesidx].celldata['roi_name']=='PM')[0]

A1 = sessions[sesidx].respmat[:,:]
# A1 = sessions[sesidx].respmat[idx_V1,:]
# A2 = sessions[sesidx].respmat[idx_PM,:]

S   = np.vstack((sessions[sesidx].trialdata['signal'],
                 sessions[sesidx].trialdata['lickResponse'],
               sessions[sesidx].respmat_runspeed))
            #    sessions[sesidx].respmat_motionenergy))
# S = np.vstack((S,np.random.randn(1,K)))
slabels     = ['Signal','Licking','Running']

df = pd.DataFrame(data=S.T, columns=slabels)

sns.heatmap(df.corr())
# arealabels  = ['V1','PM']


# Define neural data parameters
N1,K        = np.shape(A1)
# N2          = np.shape(A2)[0]
NS          = np.shape(S)[0]

# Filter only noisy threshold trials:
# idx = sessions[sesidx].trialdata['stimcat']=='N'
idx = sessions[sesidx].trialdata['signal']>-5

cmap = plt.get_cmap('viridis')
# cmap = plt.get_cmap('plasma')
cmap = plt.get_cmap('gist_rainbow')
projections = [(0, 1), (1, 2), (0, 2)]
# projections = [(0, 3), (3, 4), (2, 3)]

# fig, axes = plt.subplots(NS, len(projections), figsize=[9, 9])

for iSvar in range(NS):
    fig, axes = plt.subplots(1, len(projections), figsize=[9, 3])
    # proj = (0, 1)
    # proj = (1, 2)
    # proj = (3, 4)
    # idx         = np.intersect1d(ori_ind[iO],speed_ind[iS])

    X = zscore(A1[:,idx],axis=1)
    X = A1[:,idx]
    # X = zscore(A1[:,idx],axis=1)

    pca         = PCA(n_components=15) #construct PCA object with specified number of components

    # Xp          = pca.fit_transform(respmat_zsc[:,idx].T).T #fit pca to response matrix (n_samples by n_features)
    Xp          = pca.fit_transform(X.T).T #fit pca to response matrix (n_samples by n_features)
    #dimensionality is now reduced from N by K to ncomp by K
    
    for ax, proj in zip(axes, projections):

        x = Xp[proj[0],:]                          #get all data points for this ori along first PC or projection pairs
        y = Xp[proj[1],:]                          #get all data points for this ori along first PC or projection pairs
        
        c = cmap(minmax_scale(S[iSvar,idx], feature_range=(0, 1)))[:,:3]
        # c = cmap(minmax_scale(np.log10(1+S[iSvar,idx]), feature_range=(0, 1)))[:,:3]
        # c = cmap(minmax_scale(np.unique(S[iSvar,idx]), feature_range=(0, 1)))[:,:3]

        sns.scatterplot(x=x, y=y, c=c,ax = ax,s=10,legend = False,edgecolor =None)
        # plt.title(slabels[iSvar])

        ax.set_xlabel('PC {}'.format(proj[0]+1))
        ax.set_ylabel('PC {}'.format(proj[1]+1))
        
    plt.suptitle(slabels[iSvar],fontsize=15)
    sns.despine(fig=fig, top=True, right=True)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,'PCA' + str(proj) + '_perStim_color' + slabels[iSvar] + '.png'), format = 'png')





############################## Trial-concatenated PCA ########################################

def pca_line_stimresp(data,trialdata,spatbins):
    [N,K,S]         = np.shape(data) #get dimensions of tensor

    # collapse to 2d: N x K*T (neurons by timebins of different trials concatenated)
    X               = np.reshape(data,(N,-1))
    
    #Impute missing nan data, otherwise problems with PCA
    imp_mean        = SimpleImputer(missing_values=np.nan, strategy='mean')
    #apply imputation, replacing nan with mean of that neurons' activity
    X               = imp_mean.fit_transform(X.T).T 

    X               = zscore(X,axis=1) #score each neurons activity (along rows)

    pca             = PCA(n_components=15) #construct PCA
    Xp              = pca.fit_transform(X.T).T #PCA function assumes (samples x features)

    Xp              = np.reshape(Xp,(15,K,S)) #reshape back to trials

    #Get indices of trialtypes and responses:
    stimtypes       = sorted(trialdata['stimcat'].unique()) # stim ['A','B','C','D']
    resptypes       = sorted(trialdata['lickResponse'].unique()) # licking resp [0,1]

    s_type_ind      = [np.argwhere(np.array(trialdata['stimcat']) == stimtype)[:, 0] for stimtype in stimtypes]
    r_type_ind      = [np.argwhere(np.array(trialdata['lickResponse']) == resptype)[:, 0] for resptype in resptypes]

    #For line make-up:
    pal             = sns.color_palette('husl', 4)
    sty             = [':','-']
    patchcols       = ["cyan","green"]

    nPlotPCs        = 5 #how many subplots to create for diff PC projections

    fig, axes = plt.subplots(nPlotPCs, 1, figsize=[8, 7], sharey='row', sharex='row')
    projections = np.arange(nPlotPCs)
    for ax, proj in zip(axes, projections):
        for s in range(len(stimtypes)):
            for r in range(len(resptypes)):
                #Take the average PC projection across all indexed trials:
                y   = np.mean(Xp[proj, np.intersect1d(s_type_ind[s],r_type_ind[r]),:],axis=0)
                ax.plot(spatbins,y,c=pal[s],linestyle=sty[r])
                if proj == nPlotPCs-1:
                    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
                ax.set_ylabel('PC {}'.format(proj + 1))
        
        ax.set_xticks(np.linspace(-50,50,5))
        ax.add_patch(matplotlib.patches.Rectangle((0,ax.get_xlim()[0]),20,np.diff(ax.get_xlim())[0], 
                    fill = True, alpha=0.2,
                    color = patchcols[0], linewidth = 0))
        ax.add_patch(matplotlib.patches.Rectangle((25,ax.get_xlim()[0]),20,np.diff(ax.get_xlim())[0], 
                    fill = True, alpha=0.2,
                    color = patchcols[1], linewidth = 0))

    sns.despine(fig=fig, top=True, right=True)

    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(stimtypes))]
    labels = stimtypes
    ax.legend(custom_lines, labels,title='Stim',
            frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])


ises            = 1 #selected session to plot this for
[N,K,S]         = np.shape(sessions[ises].stensor) #get dimensions of tensor

#For all areas:
binsubidx   = (sbins>-60) & (sbins<=60)
binsub      = sbins[binsubidx]
data        = sessions[ises].stensor[:,:,binsubidx]
pca_line_stimresp(data,sessions[ises].trialdata,binsub)
# plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_allAreas_' + sessions[0].sessiondata['session_id'][0] + '.svg'), format = 'svg')
plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_allAreas_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#For each area:
for iarea,area in enumerate(areas):
    idx         = sessions[ises].celldata['roi_name'] == area
    data        = sessions[ises].stensor[np.ix_(idx,range(K),binsubidx)]
    pca_line_stimresp(data,sessions[ises].trialdata,binsub)
    plt.suptitle(area,fontsize=14)
    # plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.svg'), format = 'svg')
    plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
    # plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_Left_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')


##### PCA on different stimuli, conditioned on the other corridor stimulus:


################################################ LDA ##################################################













############################## Trial-concatenated sliding LDA  ########################################
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.rad2deg(angle_rad)

def lda_line_stimresp(data,trialdata,spatbins):
    [N,K,S]         = np.shape(data) #get dimensions of tensor

    # collapse to 2d: N x K*T (neurons by timebins of different trials concatenated)
    X               = np.reshape(data,(N,-1))
    # Impute missing nan data, otherwise problems with LDA
    imp_mean        = SimpleImputer(missing_values=np.nan, strategy='mean')
    # apply imputation, replacing nan with mean of that neurons' activity
    X               = imp_mean.fit_transform(X.T).T 
    #Z-score each neurons activity (along rows)
    X               = zscore(X,axis=1)

    respmat_stim        = np.nanmean(data[:,:,(spatbins>=0) & (spatbins<20)],axis=2) 
    respmat_dec         = np.nanmean(data[:,:,(spatbins>=25) & (spatbins<35)],axis=2) 

    vec_stim            = trialdata['stimcat']     == 'M'
    vec_dec             = trialdata['lickResponse']  == 1

    lda_stim            = LDA(n_components=1)
    lda_stim.fit(respmat_stim.T, vec_stim)
    Xp_stim             = lda_stim.transform(X.T)
    Xp_stim             = np.reshape(Xp_stim,(K,S)) #reshape back to trials by spatial bins

    lda_dec             = LDA(n_components=1)
    lda_dec.fit(respmat_dec.T, vec_dec)
    Xp_dec              = lda_dec.transform(X.T)
    Xp_dec              = np.reshape(Xp_dec,(K,S)) #reshape back to trials by spatial bins

    stim_axis     = unit_vector(lda_stim.coef_[0])
    dec_axis      = unit_vector(lda_dec.coef_[0])

    print('%f degrees between STIM and DEC axes' % angle_between(stim_axis, dec_axis).round(2))

    #Get indices of trialtypes and responses:
    stimtypes       = sorted(trialdata['stimcat'].unique()) # stim ['A','B','C','D']
    # stimtypes       = np.array(['C','M'])
    resptypes       = sorted(trialdata['lickResponse'].unique()) # licking resp [0,1]

    s_type_ind      = [np.argwhere(np.array(trialdata['stimRight']) == stimtype)[:, 0] for stimtype in stimtypes]
    r_type_ind      = [np.argwhere(np.array(trialdata['lickResponse']) == resptype)[:, 0] for resptype in resptypes]

    #For line make-up:
    pal             = sns.color_palette('muted', 5)
    sty             = [':','-']
    patchcols       = ["cyan","green"]

    fig, axes = plt.subplots(2, 1, figsize=[8, 7], sharey='row', sharex='row')
    for ax,data in zip(axes,[Xp_stim,Xp_dec]):
        for s in range(len(stimtypes)):
            for r in range(len(resptypes)):
                #Take the average LDA projection across all indexed trials:
                # ax.plot(spatbins,Xp_stim[np.intersect1d(s_type_ind[s],r_type_ind[r]),:])
                y           = np.mean(data[np.intersect1d(s_type_ind[s],r_type_ind[r]),:],axis=0)
                y_err       = np.std(data[np.intersect1d(s_type_ind[s],r_type_ind[r]),:],axis=0) / np.sqrt(len(np.intersect1d(s_type_ind[s],r_type_ind[r])))
                ax.plot(spatbins,y,c=pal[s],linestyle=sty[r])
                ax.fill_between(spatbins,y-y_err,y+y_err,color=pal[s],alpha=0.4)
        
        ax.set_xticks(np.linspace(-50,50,5))
        ax.add_patch(matplotlib.patches.Rectangle((0,ax.get_ylim()[0]),25,np.diff(ax.get_ylim())[0], 
                    fill = True, alpha=0.2,
                    color = patchcols[0], linewidth = 0))
        ax.add_patch(matplotlib.patches.Rectangle((25,ax.get_ylim()[0]),25,np.diff(ax.get_ylim())[0], 
                    fill = True, alpha=0.2,
                    color = patchcols[1], linewidth = 0))
 
    axes[0].set_ylabel(r'Proj. $LDA_{STIM}$')
    axes[1].set_ylabel(r'Proj. $LDA_{DEC}$')

    sns.despine(fig=fig, top=True, right=True)

    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(stimtypes))]
    labels = stimtypes
    ax.legend(custom_lines, labels,title='Stim',
            frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,0.9])


ises            = 0 #selected session to plot this for
[N,K,S]         = np.shape(sessions[ises].stensor) #get dimensions of tensor

## For all areas:
binsubidx   = (sbins>-60) & (sbins<=40)
binsub      = sbins[binsubidx]
trialidx    = np.isin(sessions[ises].trialdata['stimcat'],['C','M'])

trialidx    = np.logical_and(trialidx,sessions[ises].trialdata['engaged']==1)
# trialidx    = np.isin(sessions[ises].trialdata['stimcat'],['C','M'])

data        = sessions[ises].stensor[np.ix_(np.arange(N),trialidx,binsubidx)]

lda_line_stimresp(data,sessions[ises].trialdata[trialidx],binsub)
plt.savefig(os.path.join(savedir,'LDA_Line_stimResponse_allAreas_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_allAreas_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#For each area:
for iarea,area in enumerate(areas):
    idx         = sessions[ises].celldata['roi_name'] == area
    data        = sessions[ises].stensor[np.ix_(idx,trialidx,binsubidx)]
    lda_line_stimresp(data,sessions[ises].trialdata[trialidx],binsub)
    plt.suptitle(area,fontsize=14)
    # plt.savefig(os.path.join(savedir,'LDA_Line_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.svg'), format = 'svg')
    plt.savefig(os.path.join(savedir,'LDA_Line_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
    # plt.savefig(os.path.join(savedir,'LDA_Line_deconv_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

##### LDA Context #################################

def lda_scatterses_context(data,trialdata):
    [N,K]         = np.shape(data) #get dimensions of tensor

    # Impute missing nan data, otherwise problems with LDA
    imp_mean            = SimpleImputer(missing_values=np.nan, strategy='mean')
    # apply imputation, replacing nan with mean of that neurons' activity
    X                   = imp_mean.fit_transform(data.T).T 
    #Z-score each neurons activity (along rows)
    X                   = zscore(X,axis=1)

    vec_ctx             = trialdata['context'] == 1

    lda_ctx             = LDA(n_components=1)
    lda_ctx.fit(X.T, vec_ctx)
    Xp_ctx              = lda_ctx.transform(X.T)

    #For line make-up:
    pal             = sns.color_palette('muted', 5)
    sty             = [':','-']
    patchcols       = ["cyan","green"]

    fig,ax = plt.subplots(figsize=[8, 3.5])
    
    plt.scatter(x=trialdata['TrialNumber'], y=Xp_ctx,s=10,c='k')
    plt.xlim([trialdata['TrialNumber'].min(),trialdata['TrialNumber'].max()])
    plt.xlabel('Trial number')
    plt.ylabel(r'Proj. $LDA_{CTX}$')

    colors = ["green","purple"]
    for iblock in np.arange(0,trialdata['TrialNumber'].max(),100):
        ax.add_patch(matplotlib.patches.Rectangle((iblock,-50),50,100, 
                            fill = True, alpha=0.2,
                            color = colors[0], linewidth = 0))
    for iblock in np.arange(50,trialdata['TrialNumber'].max(),100):
        ax.add_patch(matplotlib.patches.Rectangle((iblock,-50),50,100, 
                            fill = True, alpha=0.2,
                            color = colors[1], linewidth = 0))

    sns.despine(fig=fig, top=True, right=True)

### plot context lda figure:
ises            = 2 #selected session to plot this for
[N,K,S]         = np.shape(sessions[ises].stensor) #get dimensions of tensor

## For all areas:
binsubidx   = (sbins>=-75) & (sbins<50)
binsub      = sbins[binsubidx]
trialidx    = np.isin(sessions[ises].trialdata['stimRight'],['A','B','C','D'])

data        = sessions[ises].stensor[np.ix_(np.arange(N),trialidx,binsubidx)]

lda_scatterses_context(data,sessions[ises].trialdata[trialidx],binsub)
plt.savefig(os.path.join(savedir,'LDA','LDA_Line_stimResponse_allAreas_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_allAreas_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#For each area:
for iarea,area in enumerate(areas):
    idx         = sessions[ises].celldata['roi_name'] == area
    data        = sessions[ises].stensor[np.ix_(idx,trialidx,binsubidx)]
    lda_scatterses_context(data,sessions[ises].trialdata[trialidx],binsub)
    plt.suptitle(area,fontsize=14)
    plt.savefig(os.path.join(savedir,'LDA_Line_context_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
    # plt.savefig(os.path.join(savedir,'LDA_Line_deconv_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')


binsubidx   = (sbins>=-75) & (sbins<-40)
# binsubidx   = (sbins>=0) & (sbins<25)

#For each area:
for iarea,area in enumerate(areas):
    idx         = sessions[ises].celldata['roi_name'] == area
    data        = np.nanmean(sessions[ises].stensor[np.ix_(idx,trialidx,binsubidx)],axis=2)
    lda_scatterses_context(data,sessions[ises].trialdata[trialidx])
    plt.suptitle(area,fontsize=14)
    # plt.savefig(os.path.join(savedir,'LDA','LDA_Scatterses_context_atstim_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
    plt.savefig(os.path.join(savedir,'LDA','LDA_Scatterses_context_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
    

#################### LDA correlation in projection across areas #################

#take mean response from trials with contralateral A / B stimuli:
respmat_stim     = np.nanmean(sessions[0].stensor[np.ix_(range(N),trialidx,(sbins>0) & (sbins<25))],axis=2) 
respmat_dec      = np.nanmean(sessions[0].stensor[np.ix_(range(N),trialidx,(sbins>25) & (sbins<50))],axis=2) 
trialdata        = sessions[0].trialdata[trialidx]

stim_vec         = trialdata['stimRight'] == 'A'
dec_vec          = trialdata['lickResponse'] == 1

LDAstim_proj_A   = np.empty((np.sum(stim_vec==True),len(areas)))
LDAstim_proj_B   = np.empty((np.sum(stim_vec==False),len(areas)))
LDAdec_proj_0    = np.empty((np.sum(dec_vec==False),len(areas)))
LDAdec_proj_1    = np.empty((np.sum(dec_vec==True),len(areas)))

#For each area:
for iarea,area in enumerate(areas):
    idx                     = sessions[0].celldata['roi_name'] == area
    data                    = respmat_stim[idx,:]
    data                    = zscore(data,axis=1) #score each neurons activity (along rows)

    lda_stim                = LDA(n_components=1)
    lda_stim.fit(data.T, stim_vec)
    LDAstim_proj_A[:,iarea]   = lda_stim.transform(data[:,stim_vec==True].T).reshape(1,-1)
    LDAstim_proj_B[:,iarea]   = lda_stim.transform(data[:,stim_vec==False].T).reshape(1,-1)

    data                    = respmat_dec[idx,:]
    data                    = zscore(data,axis=1) #score each neurons activity (along rows)

    lda_dec                = LDA(n_components=1)
    lda_dec.fit(data.T, dec_vec)
    LDAdec_proj_0[:,iarea]   = lda_dec.transform(data[:,dec_vec==False].T).reshape(1,-1)
    LDAdec_proj_1[:,iarea]   = lda_dec.transform(data[:,dec_vec==True].T).reshape(1,-1)


df_stim_A     = pd.DataFrame(data=LDAstim_proj_A,columns=areas)
df_stim_B     = pd.DataFrame(data=LDAstim_proj_B,columns=areas)
df_dec_0      = pd.DataFrame(data=LDAdec_proj_0,columns=areas)
df_dec_1      = pd.DataFrame(data=LDAdec_proj_1,columns=areas)

sns.scatterplot(data = df_stim_A,x='V1',y='PM')
plt.title(r'$LDA_{STIM-A}$ projection interarea correlation')
# to do index based on area
plt.text(x=np.percentile(LDAstim_proj_A[:,0],90),y=np.percentile(LDAstim_proj_A[:,0],5),s='r = %.2f' % np.corrcoef(LDAstim_proj_A[:,0],LDAstim_proj_A[:,1])[0,1])
plt.savefig(os.path.join(savedir,'LDA_STIMA_proj_scatter_V1PM_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

fig,axes = plt.subplots(2,2,figsize=(9,6))

sns.heatmap(df_stim_A.corr(),vmin=-1,vmax=1,cmap="vlag",ax=axes[0,0])
axes[0,0].set_title(r'$LDA_{STIM-A}$')

sns.heatmap(df_stim_B.corr(),vmin=-1,vmax=1,cmap="vlag",ax=axes[0,1])
axes[0,1].set_title(r'$LDA_{STIM-B}$')

sns.heatmap(df_dec_0.corr(),vmin=-1,vmax=1,cmap="vlag",ax=axes[1,0])
axes[1,0].set_title(r'$LDA_{DEC-0}$')

sns.heatmap(df_dec_1.corr(),vmin=-1,vmax=1,cmap="vlag",ax=axes[1,1])
axes[1,1].set_title(r'$LDA_{DEC-1}$')

plt.suptitle('LDA projection interarea cross correlation')
# plt.savefig(os.path.join(savedir,'LDA_proj_corr_interarea_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'LDA_proj_deconv_corr_interarea_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

##################################

