# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% IMPORT LIBS
import os
os.chdir('e:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd

from loaddata.session_info import filter_sessions,load_sessions

from scipy import stats
from scipy.stats import zscore
from utils.psth import *
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
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11622', '2024_02_21']])
# session_list    = np.array([['LPE10884', '2023_12_14']])
# session_list    = np.array([['LPE12013','2024_04_25']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion='deconv') #Load specified list of sessions
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE12385'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion='dF') #load sessions that meet criteria:
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE12013'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion='dF') #load sessions that meet criteria:

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\dPCA\\')

#%% Zscore all the calciumdata:
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################## Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 60   #post cm
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

#%% dPCA on session tensor:
ises        = 0 #selected session to plot this for

# sessions[ises].stensor[np.isnan(sessions[ises].stensor)] = 0
area = 'AL'
# area = 'RSP'
# area = 'V1'
area = 'PM'

data = copy.deepcopy(sessions[ises].stensor)
data[np.isnan(data)] = 0

idx_N = sessions[ises].celldata['roi_name']==area

# idx_N  = np.all((sessions[ises].celldata['roi_name']=='PM',
#                  sessions[ises].celldata['noise_level']<20),axis=0)
data = data[idx_N,:,:]

# number of neurons, time-points and stimuli
[N,t,S]     = np.shape(data) #get dimensions of tensor

# stimtypes   = sorted(sessions[ises].trialdata['stimcat'].unique()) # Catch, Noise and Max trials if correct
C = 2
stimtypes   = ['C','M']
stimlabels  = ['catch','max']

# C = 2
# stimtypes   = ['C','N']
# stimlabels  = ['catch','noise']

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
R = np.nanmean(trialR,0)

# center data
R -= np.mean(R.reshape((N,-1)),1)[:,None,None,None]
# center trialR data:
trialR -= np.mean(trialR.reshape((n_min_trials,N,-1)),2)[:,:,None,None,None]

#%% 
regval = 0.003
# regval = 0.1
# regval = 0.003
# regval = 'auto'
# regval = 1.5556809555781208e-05
ncomponents = 3


# dpca = dPCA.dPCA(labels='sdt',regularizer=regval,n_components=ncomponents)
dpca = dPCA.dPCA(labels='sdt',regularizer=regval,n_components=ncomponents,join={'s' : ['s','st'],
                                                                                'd' : ['d','dt'],
                                                                                'sd' : ['sd','sdt']})
# dpca = dPCA.dPCA(labels='tsd',regularizer=regval)

dpca.protect = ['t']

Z = dpca.fit_transform(R,trialR)

#%%

significance_masks = dpca.significance_analysis(R,  trialR, n_shuffles=10, n_splits=10, n_consecutive=10,axis=True)


#%% Plot:

linecolors_c = ['grey','green','blue']
linestyles_d = ['--','-',':']

# plt.figure(figsize=(16,4))
fig,axes = plt.subplots(ncomponents,4,figsize=(12,ncomponents*3)) 

labels = [stimlabels[i]+'-'+declabels[j] for i in range(C) for j in range(D)]

for icomponent in range(ncomponents):
    ax = axes[icomponent,0]
    for c in range(C):
        for d in range(D):
            ax.plot(sbins,Z['t'][icomponent,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
            # ax.plot(sbins,Z[labels[icomponent]][0,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
    if icomponent == 0: 
        ax.legend(labels,frameon=False,fontsize=8)
    # ax.set_title(labels[icomponent])
    ax.set_title('Dim %d - Time component\nEV: %.5f' % (icomponent,dpca.explained_variance_ratio_['t'][icomponent]))

    ax = axes[icomponent,1]
    for c in range(C):
        for d in range(D):
            ax.plot(sbins,Z['s'][icomponent,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
    # ax.plot(sbins[significance_masks['st'][icomponent,:]],np.max(Z['st'][icomponent,:,:])*np.ones(significance_masks['st'][icomponent,:].sum()),color='k',linewidth=2,alpha=1)
    # ax.set_title('Dim %d - Stimulus component' % icomponent)
    ax.set_title('Dim %d - Stimulus component\nEV: %.5f' % (icomponent,dpca.explained_variance_ratio_['s'][icomponent]))

    ax = axes[icomponent,2]
    for c in range(C):
        for d in range(D):
            ax.plot(sbins,Z['d'][icomponent,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
    # ax.plot(sbins[significance_masks['dt'][icomponent,:]],np.max(Z['dt'][icomponent,:,:])*np.ones(significance_masks['dt'][icomponent,:].sum()),color='k',linewidth=2,alpha=1)
    # ax.set_title('Dim %d - Decision component' % icomponent)
    ax.set_title('Dim %d - Decision component\nEV: %.5f' % (icomponent,dpca.explained_variance_ratio_['d'][icomponent]))

    ax = axes[icomponent,3]
    for c in range(C):
        for d in range(D):
            ax.plot(sbins,Z['sd'][icomponent,c,d],color=linecolors_c[c],linestyle=linestyles_d[d])
    # ax.plot(sbins[significance_masks['sdt'][icomponent,:]],np.max(Z['sdt'][icomponent,:,:])*np.ones(significance_masks['sdt'][icomponent,:].sum()),color='k',linewidth=2,alpha=1)
    # ax.set_title('Dim %d - Mixing component' % icomponent)
    ax.set_title('Dim %d - Mixing component\nEV: %.5f' % (icomponent,dpca.explained_variance_ratio_['sd'][icomponent]))

plt.tight_layout()
fig.savefig(os.path.join(savedir,'dPCA_%s_%s_trialtypes_%s.png' % (sessions[ises].sessiondata['session_id'][0],area,''.join(stimtypes))), format = 'png')

#%%

#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

#%% dPCA on session tensor:
ises        = 0 #selected session to plot this for
data = copy.deepcopy(sessions[ises].stensor)
data[np.isnan(data)] = 0

idx_N = np.ones(len(sessions[ises].celldata['roi_name'])).astype(bool)
# idx_N = np.logical_or(sessions[ises].celldata['sig_N'],sessions[ises].celldata['sig_M'])
# idx_N = np.ones(len(sessions[ises].celldata['roi_name'])).astype(bool)

data = data[idx_N,:,:]

# number of neurons, time-points and stimuli
[N,t,S]     = np.shape(data) #get dimensions of tensor

# stimtypes   = sorted(sessions[ises].trialdata['stimcat'].unique()) # Catch, Noise and Max trials if correct
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
n_min_trials = 40

trialR = np.empty((n_min_trials,N,C,D,S))

for iC in range(C):
    for iD in range(D):
        # idx = np.random.choice(np.argwhere(np.logical_and(c_ind[iC,:],d_ind[iD,:])).squeeze(), size=n_min_trials, replace=False)  
        idx = np.random.choice(np.argwhere(np.logical_and(c_ind[iC,:],d_ind[iD,:])).squeeze(), size=n_min_trials, replace=True)  
        trialR[:,:,iC,iD,:] = data[:,idx,:].transpose((1,0,2))
        # trialR[:,:,iC,iD,:] = data[:,idx,:]

print(np.shape(trialR))

# trial-average data
R = np.nanmean(trialR,0)

# center data
R -= np.mean(R.reshape((N,-1)),1)[:,None,None,None]
# center trialR data:
trialR -= np.mean(trialR.reshape((n_min_trials,N,-1)),2)[:,:,None,None,None]

#%% 
regval = 'auto'
regval = 0.0018
regval = 0.0035
ncomponents = 3
# dpca = dPCA.dPCA(labels='sdt',regularizer=regval,n_components=ncomponents)
# dpca = dPCA.dPCA(labels='sdt',regularizer=regval,n_components=ncomponents)
dpca = dPCA.dPCA(labels='sdt',regularizer=regval,n_components=ncomponents,join={'s' : ['s','st'],
                                                                                'd' : ['d','dt'],
                                                                                'sd' : ['sd','sdt']})
# dpca = dPCA.dPCA(labels='tsd',regularizer=regval)

dpca.protect = ['t']
Z = dpca.fit_transform(R,trialR)
D = dpca.D

#%% Some vars for plotting:
margs = ['s','d','sd']
nmargs = len(margs)
areas = ['V1','PM','AL','RSP']
clrs_areas = get_clr_areas(areas)
area_idx_N = np.array(sessions[ises].celldata['roi_name'][idx_N])

#%% 
histedges = np.arange(-0.05,0.05,0.0025)
fig,axes = plt.subplots(ncomponents,nmargs,figsize=(nmargs*3,ncomponents*2.5),sharex=True)
for icomponent in range(ncomponents):
    for imarg,marg in enumerate(margs):
        ax = axes[icomponent,imarg]
        for iarea,area in enumerate(areas):
            idx = area_idx_N == area
            ax.plot(histedges[:-1],np.histogram(dpca.D[marg][idx,icomponent],histedges,density=True)[0],color=clrs_areas[iarea],label=area)
            # ax.plot(dpca.D[marg][icomponent,idx],label=area)
        if imarg == 0 and icomponent == 0:
            ax.legend(frameon=False,fontsize=8)
        ax.set_title('%s (Dim %d)' % (marg,icomponent+1))
        ax.set_xlim(histedges[0],histedges[-1])
        # ax.set_ylim(0,1.5)
        ax.set_xlabel('Weight')
fig.tight_layout()
fig.savefig(os.path.join(savedir,'dPCA_WeightHist_%s_trialtypes_%s.png' % (sessions[ises].sessiondata['session_id'][0],''.join(stimtypes))), format = 'png')

#%% 
dim = 0
df = pd.DataFrame()
df['stim'] = dpca.D['s'][:,dim]
df['choice'] = dpca.D['d'][:,dim]
df['stim x choice'] = dpca.D['sd'][:,dim]
df['area'] = sessions[ises].celldata['roi_name'][idx_N]

# sns.pairplot(df,hue='area',diag_kind="hist",height=2.5,plot_kws={"s": 3, "alpha": 0.5, "color": "k"},hue_order=areas,palette=clrs_areas)
fig = sns.pairplot(df,hue='area',diag_kind="kde",height=2.5,plot_kws={"s": 4, "alpha": 0.5},hue_order=areas,palette=clrs_areas)
plt.suptitle('dPCA weights_dim%d_%s' % (dim+1,sessions[ises].sessiondata['session_id'][0]),fontsize=14)

fig.tight_layout()
fig.savefig(os.path.join(savedir,'dPCA_Weight_AreaPair_%s_trialtypes_%s.png' % (sessions[ises].sessiondata['session_id'][0],''.join(stimtypes))), format = 'png')

#%% 










#%%

# for regval in [0,0.001,0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]:
#     ncomponents = 3
#     dpca = dPCA.dPCA(labels='sdt',regularizer=regval,n_components=ncomponents)
#     Z = dpca.fit_transform(R,trialR)
#     values = dpca.explained_variance_ratio_.values()
#     print(regval)

#     print(np.sum(list(values)))

#%% 

# number of neurons, time-points and stimuli and decisions
N,T,S,D = 50,250,3,2

# noise-level and number of trials in each condition
noise, n_samples = 0.2, 10

# build two latent factors
zs = (np.arange(S)/float(S))
zd = (np.arange(D)/float(D))
zt = (np.arange(T)/float(T))

# build trial-by trial data
trialR = noise*np.random.randn(n_samples,N,S,D,T)
trialR += np.random.randn(N)[None,:,None,None,None]*zs[None,None,:,None,None]
trialR += np.random.randn(N)[None,:,None,None,None]*zd[None,None,None,:,None]
trialR += np.random.randn(N)[None,:,None,None,None]*zt[None,None,None,None,:]

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