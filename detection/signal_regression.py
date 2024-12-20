# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
os.chdir('E:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd

# from loaddata import * #get all the loading data functions (filter_sessions,load_sessions)
from loaddata.session_info import filter_sessions,load_sessions

from scipy import stats
from scipy.stats import zscore
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score as AUC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches
from utils.plotting_style import * #get all the fixed color schemes
from matplotlib.lines import Line2D
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

session_list = np.array([['LPE12385', '2024_06_15']])
session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11998', '2024_04_23']])
session_list = np.array([['LPE10884', '2023_12_14']])
session_list = np.array([['LPE10884', '2023_12_14']])
session_list        = np.array([['LPE12013','2024_04_25']])
# session_list = np.array([['LPE09829', '2023_03_29'],
#                         ['LPE09829', '2023_03_30'],
#                         ['LPE09829', '2023_03_31']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE12385'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE12013'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:




#%% ### Show for all sessions which region of the psychometric curve the noise spans #############
sessions = noise_to_psy(sessions,filter_engaged=True)

idx_inclthr = np.empty(nSessions).astype('int')
for ises,ses in enumerate(sessions):
    idx_inclthr[ises] = int(np.logical_and(np.any(sessions[ises].trialdata['signal_psy']<=0),np.any(sessions[ises].trialdata['signal_psy']>=0)))
    ses.sessiondata['incl_thr'] = idx_inclthr[ises]

sessions = [ses for ises,ses in enumerate(sessions) if ses.sessiondata['incl_thr'][0]]
nSessions = len(sessions)

#%% 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

################################ Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -100  #pre cm
s_post      = 100   #post cm
binsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')

## Compute average response in stimulus response zone:
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


#%% #################### Compute activity for each stimulus type for one session ##################
ises        = 0 #selected session to plot this for
[N,K,S]     = np.shape(sessions[ises].stensor) #get dimensions of tensor

stimtypes   = sorted(sessions[ises].trialdata['stimcat'].unique()) # Catch, Noise and Max trials if correct
stimtypes   = ['C','N','M']
stimlabels  = ['catch','noise','max']

snakeplots  = np.empty([N,S,len(stimtypes)])

for iTT in range(len(stimtypes)):
    idx = sessions[ises].trialdata['stimcat'] == stimtypes[iTT]
    # idx = np.logical_and(idx,sessions[ises].trialdata['lickResponse']==0)
    snakeplots[:,:,iTT] = np.nanmean(sessions[ises].stensor[:,idx,:],axis=1)

#%% ################################## Plot for one session: ####################################
areas   = sessions[ises].celldata['roi_name'].unique()
for iarea,area in enumerate(areas):
    idx         = sessions[ises].celldata['roi_name'] == area
    areasnake   = snakeplots[idx,:,:]
    plot_snake_area(snakeplots[idx,:,:],sbins,stimtypes=stimlabels)
    plt.suptitle(area,fontsize=16,y=0.96)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,'ActivityInCorridor_perStim_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ############################ Plot the activity during  one session: ####################################
ises = 1

for iN in range(1000,1200):
    plot_snake_neuron_stimtypes(sessions[ises].stensor[iN,:,:],sbins,sessions[ises].trialdata)
    plt.suptitle(sessions[ises].celldata['cell_id'][iN],fontsize=16,y=0.96)

ises = 0
example_cell_ids = ['LPE12385_2024_06_15_0_0075',
'LPE12385_2024_06_15_0_0126',
'LPE12385_2024_06_15_0_0105', # reduction upon stimulus zone
'LPE12385_2024_06_15_0_0114', # noise trial specific response
'LPE12385_2024_06_15_0_0031', # noise trial specific response
'LPE12385_2024_06_15_0_0158',
'LPE12385_2024_06_15_1_0001',
'LPE12385_2024_06_15_0_0046',
'LPE12385_2024_06_15_2_0248', #non stationary activity (drift?)
'LPE12385_2024_06_15_2_0499',
'LPE12385_2024_06_15_6_0089',#non stationary activity (drift?)
'LPE12385_2024_06_15_2_0353'] #variable responsiveness

for iN in np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]:
    plot_snake_neuron_stimtypes(sessions[ises].stensor[iN,:,:],sbins,sessions[ises].trialdata)
    plt.suptitle(sessions[ises].celldata['cell_id'][iN],fontsize=16,y=0.96)
    plt.savefig(os.path.join(savedir,'SingleNeuron','ActivityInCorridor_perStim_' + sessions[ises].celldata['cell_id'][iN] + '.png'), format = 'png',bbox_inches='tight')

#%% Show example neurons that are correlated either to the stimulus signal, lickresponse or to running speed:
ises = 1
for iN in range(900,1100):
    plot_snake_neuron_sortnoise(sessions[ises].stensor[iN,:,:],sbins,sessions[ises])
    plt.suptitle(sessions[ises].celldata['cell_id'][iN],fontsize=16,y=0.96)

ises = 0
example_cell_ids = ['LPE12385_2024_06_15_0_0098', #hit specific activity?
'LPE12385_2024_06_15_1_0075'] #some structure

ises = 1
example_cell_ids = ['LPE12385_2024_06_16_0_0166', #runspeed specific activity?
                    'LPE12385_2024_06_16_0_0151', #runspeed specific activity?
                    'LPE12385_2024_06_16_0_0070', #signal correlation
                    'LPE12385_2024_06_16_0_0075', #runspeed specific activity
                    'LPE12385_2024_06_16_0_0124', #signal correlation
                    'LPE12385_2024_06_16_0_0109', #signal correlation
                    'LPE12385_2024_06_16_0_0177', #signal correlation
                    'LPE12385_2024_06_16_0_0136', #signal correlation and lick response
                    'LPE12385_2024_06_16_0_0195', #runspeed specific activity?
                    'LPE12385_2024_06_16_2_0301', #runspeed and signal
                    'LPE12385_2024_06_16_2_0209', #lick response
                    'LPE12385_2024_06_16_0_0071'] #some structure

for iN in np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]:
    plot_snake_neuron_sortnoise(sessions[ises].stensor[iN,:,:],sbins,sessions[ises])
    plt.suptitle(sessions[ises].celldata['cell_id'][iN],fontsize=16,y=0.96)
    plt.savefig(os.path.join(savedir,'SingleNeuron','ActivityInCorridor_sortNoise_' + sessions[ises].celldata['cell_id'][iN] + '.png'), format = 'png',bbox_inches='tight')


#%% ############################ Snakeplot for sessions combined ##################################
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
trialdata = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

N           = len(celldata) #get number of cells total
S           = np.shape(sessions[0].stensor)[2] #get number of spatial bins

snakeplots  = np.empty([N,S,len(stimtypes)])

for ises in range(nSessions):
    idx = celldata['session_id']==sessions[ises].sessiondata['session_id'][0]
    for iTT in range(len(stimtypes)):
        # snakeplots[idx,:,iTT] = np.nanmean(sessions[ises].stensor[:,sessions[ises].trialdata['stimcat'] == stimtypes[iTT],:],axis=1)

        trialidx = sessions[ises].trialdata['stimcat'] == stimtypes[iTT]
        # trialidx = np.logical_and(trialidx,sessions[ises].trialdata['engaged']==1)
        # trialidx = np.logical_and(trialidx,sessions[ises].trialdata['lickResponse']==1)
        snakeplots[idx,:,iTT] = np.nanmean(sessions[ises].stensor[:,trialidx,:],axis=1)

#Plot for all loaded sessions together:
areas   = celldata['roi_name'].unique()
for iarea,area in enumerate(areas):
    idx         = celldata['roi_name'] == area
    areasnake   = snakeplots[idx,:,:]
    plot_snake_area(snakeplots[idx,:,:],sbins,stimtypes=stimlabels)
    plt.suptitle(area,fontsize=16,y=0.96)
    # plt.tight_layout()
    # plt.savefig(os.path.join(savedir,'ActivityInCorridor_perStim_' + area + '_' + '.svg'), format = 'svg')
    plt.savefig(os.path.join(savedir,'ActivityInCorridor_perStim_' + area + '.png'), format = 'png',bbox_inches='tight')

#%% ############################### Plot neuron-average per stim per area #################################

ises = 0 #selected session to plot this for
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(10,10))

areas = sessions[ises].celldata['roi_name'].unique()
for iarea,area in enumerate(areas):
    idx = sessions[ises].celldata['roi_name'] == area
    plt.subplot(2,2,iarea+1)
    for iTT in range(len(stimtypes)):
        plt.plot(sbins,np.nanmean(snakeplots[idx,:,iTT],axis=0),linewidth=2)
    plt.legend(labels=stimtypes)
    plt.title(area,fontsize=10)
    plt.ylabel('dF/F',fontsize=9)
    plt.xlabel('Pos. relative to stim (cm)',fontsize=9)
    plt.xlim([-80,50])
    # plt.ylim([0.1,0.5])
# plt.savefig(os.path.join(savedir,'ActivityInCorridor_neuronAverage_perStim_' + sessions[0].sessiondata['session_id'][0] + '.svg'), format = 'svg')
plt.savefig(os.path.join(savedir,'ActivityInCorridor_neuronAverage_perStim_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'ActivityInCorridor_deconv_neuronAverage_perStim_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ################## Number of responsive neurons per stimulus #################################

sesidx          = 0 #selected session to plot this for
[N,K,S]         = np.shape(sessions[sesidx].stensor) #get dimensions of tensor

#Get indices of trialtypes and responses:
stimtypes       = sorted(sessions[sesidx].trialdata['stimRight'].unique()) # stim ['A','B','C','D']
resptypes       = sorted(sessions[sesidx].trialdata['lickResponse'].unique()) # licking resp [0,1]

D               = len(stimtypes)
sigmat          = np.empty((N,D))

binidx_base     = (sbins>=-75) & (sbins<-25)
binidx_stim     = (sbins>=0) & (sbins<20)
# binidx_stim     = (sbins>=-50) & (sbins<-25)

for n in range(N):
    print(f"\rComputing significant response for neuron {n+1} / {N}",end='\r')
    for s,stimtype in enumerate(stimtypes):
        b = np.nanmean(sessions[sesidx].stensor[np.ix_(np.array([n]),sessions[sesidx].trialdata['stimRight'] == stimtype,binidx_base)],axis=2)
        r = np.nanmean(sessions[sesidx].stensor[np.ix_(np.array([n]),sessions[sesidx].trialdata['stimRight'] == stimtype,binidx_stim)],axis=2)

        # stat,sigmat[n,s] = stats.ttest_ind(b.flatten(), r.flatten(),nan_policy='omit')
        stat,sigmat[n,s] = stats.ttest_rel(b.flatten(), r.flatten(),nan_policy='omit')

df = pd.concat([sessions[sesidx].celldata, pd.DataFrame(data=sigmat<0.01,columns=stimtypes)], axis=1)

## Plot number of responsive neurons per stimulus per area:
fig,axes = plt.subplots(2,2,figsize=(8,8),sharey=True, sharex=True)
sns.barplot(x='roi_name', y='G', data=df, estimator=lambda x: sum(x==1)*100.0/len(x),ax=axes[0,0]).set(title='A')
# sns.barplot(x='roi_name', y='B', data=df, estimator=lambda y: sum(y==1)*100.0/len(y),ax=axes[0,1]).set(title='B')
# sns.barplot(x='roi_name', y='C', data=df, estimator=lambda y: sum(y==1)*100.0/len(y),ax=axes[1,0]).set(title='C')
# sns.barplot(x='roi_name', y='D', data=df, estimator=lambda y: sum(y==1)*100.0/len(y),ax=axes[1,1]).set(title='D')

## Plot number of responsive neurons per stimulus across areas:
sns.barplot(x=stimtypes,y=np.sum(sigmat<0.01,axis=0) / N)
plt.title('Frac per stim all areas')
plt.xlabel('Stimulus')

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


# sessions[0].behaviordata['trialnum']==t and sessions[0].behaviordata['zpos']>sessions[0].trialdata['StimStart'][t]+sloc[0]

# sample 1000 random location averages in the tensor: 

# respmat         = np.nanmean(sessions[0].stensor[:,:,(sbins>-10) & (sbins<30)],axis=2) 

# # sbinidx         = np.logical_or((sbins>-80) & (sbins<-10),(sbins>20) & (sbins<50)) #omit nans and stim test window
# # permbinidx      = sbins[sbinidx]
# # permbinidx      = (permbinidx>-70) & (permbinidx<-40)
# # stensor_trim    = sessions[0].stensor[:,:,sbinidx]

# nperms      = 100
# # perm        = np.empty((N,nperms))
# stensor_roll    = np.empty((N,S,nperms))

# for p in range(nperms): #for N circular shuffles of the data:
#     stensor         = sessions[0].stensor.copy()
#     print(f"\rShuffling tensor {p+1} / {nperms}",end='\r')
#     for t in range(np.shape(stensor)[1]): #loop across trials, each one different circular roll:
#         stensor[:,t,:]     = np.roll(stensor[:,t,:],np.random.randint(2,np.shape(stensor)[2]-2,1))
    
#     stensor_roll[:,:,p]     = np.nanmean(stensor,axis=1) #average across trials

# # Get the 95% percentile activity with random locations (each spatial bin):
# permmax         = np.percentile(stensor_roll,95,axis=2)
# sigmat          = np.nanmean(sessions[0].stensor,axis=1) > permmax

# plt.figure()
# plt.plot(sbins,np.sum(sigmat,axis=0) / N)


# from sklearn.feature_selection import mutual_info_classif as MI
# from sklearn.feature_selection import mutual_info_regression as MI

# X = sessions[0].zpos_F.reshape(-1,1)
# y = sessions[0].calciumdata.iloc[:,0].to_numpy()
# g = MI(X,y)


# for s in range(1000):
#     np.roll

# r = -np.random.randint(1,np.shape(stensor_trim)[2],(1,len(sessions[0].trialdata)))
# A = stensor_trim

# rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

# # Use always a negative shift, so that column_indices are valid.
# # (could also use module operation)
# r[r < 0] += A.shape[1]
# column_indices = column_indices - r[:, np.newaxis]

# result = A[rows, column_indices]

# plt.figure(figsize=(12,5))
# # plt.imshow(np.nanmean(A,axis=1),interpolation='nearest',aspect='auto',vmin=-0.25,vmax=1.25)
# plt.imshow(np.nanmean(result,axis=1),interpolation='nearest',aspect='auto',vmin=-0.25,vmax=1.25)




# %%
