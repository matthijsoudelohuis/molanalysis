# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat


##################################################
session_list        = np.array([['LPE09830','2023_04_10']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=False, calciumversion='dF')


##############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], t_pre, t_post, binsize,method='binmean')

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
#                                  t_pre, t_post, binsize,method='interp_lin')
# [N,K,T]         = np.shape(tensor) #get dimensions of tensor
# respmat         = tensor[:,:,np.logical_and(t_axis > 0,t_axis < 1)].mean(axis=2)

#Alternative method, much faster:
respmat         = compute_respmat(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=True)
[N,K]           = np.shape(respmat) #get dimensions of response matrix

################## Noise correlations: ##############################################

# get signal correlations:
resp_meanori = np.empty([N,16])
oris = np.sort(pd.Series.unique(sessions[0].trialdata['Orientation']))

for i,ori in enumerate(oris):
    resp_meanori[:,i] = np.nanmean(respmat[:,sessions[0].trialdata['Orientation']==ori],axis=1)

sig_corr = np.corrcoef(resp_meanori)
sig_corr[np.eye(N)==1] = np.nan
plt.figure(figsize=(8,5))
# plt.imshow(sig_corr, cmap='coolwarm', vmin=-1,vmax=1)
plt.imshow(sig_corr, cmap='coolwarm',vmin=np.nanpercentile(sig_corr,15),vmax=np.nanpercentile(sig_corr,85))

# get noise correlations:
oris        = sorted(sessions[0].trialdata['Orientation'].unique())
ori_counts  = sessions[0].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
assert(len(ori_counts) == 16 or len(ori_counts) == 8)
# assert(np.all(ori_counts == 200) or np.all(ori_counts == 400))

respmat_res           = respmat.copy()

## Compute residuals:
for ori in oris:
    ori_idx     = np.where(sessions[0].trialdata['Orientation']==ori)[0]
    temp        = np.mean(respmat_res[:,ori_idx],axis=1)
    respmat_res[:,ori_idx] = respmat_res[:,ori_idx] - np.repeat(temp[:, np.newaxis], len(ori_idx), axis=1)

noise_corr = np.corrcoef(respmat_res)
noise_corr[np.eye(N)==1] = np.nan
plt.figure(figsize=(8,5))
# plt.imshow(sig_corr, cmap='coolwarm', vmin=-1,vmax=1)
plt.imshow(noise_corr, cmap='coolwarm',vmin=np.nanpercentile(noise_corr,5),vmax=np.nanpercentile(noise_corr,95))

## Compute euclidean distance matrix based on soma center:
distmat     = np.zeros((N,N))
areamat     = np.empty((N,N),dtype=object)
labelmat    = np.empty((N,N),dtype=object)

for i in range(N):
    for j in range(N):
        problem also z distmat[i,j] = math.dist([sessions[0].celldata['xloc'][i],sessions[0].celldata['yloc'][i]],
                [sessions[0].celldata['xloc'][j],sessions[0].celldata['yloc'][j]])
        areamat[i,j] = sessions[0].celldata['roi_name'][i] + '-' + sessions[0].celldata['roi_name'][j]
        labelmat[i,j] = str(int(sessions[0].celldata['redcell'][i])) + '-' + str(int(sessions[0].celldata['redcell'][j]))
distmat[np.eye(N)==1] = np.nan

df = pd.DataFrame({'NoiseCorrelation': noise_corr.flatten(),
                'AreaPair': areamat.flatten(),
                'DistPair': distmat.flatten(),
                'LabelPair': labelmat.flatten()})

###################### Noise correlations within and across areas: #########################
plt.figure(figsize=(8,5))
sns.barplot(data=df,x='AreaPair',y='NoiseCorrelation')

###################### Noise correlations as a function of pairwise distance: ####################
plt.figure(figsize=(8,5))
# sns.scatterplot(data=df[df['AreaPair']=='V1-V1'],x='DistPair',y='NoiseCorrelation',size=5)
dfV1 = df[df['AreaPair']=='V1-V1']
sns.lineplot(x=np.round(dfV1['DistPair'],-1),y=dfV1['NoiseCorrelation'],color='b')

dfPM = df[df['AreaPair']=='PM-PM']
sns.lineplot(x=np.round(dfPM['DistPair'],-1),y=dfPM['NoiseCorrelation'],color='g')

plt.xlabel="Pairwise distance (um)"
plt.legend(labels=['V1-V1','PM-PM'])
plt.xlim([-10,600])
plt.ylim([0,0.13])

########################### Noise correlations as a function of pairwise distance: ####################
######################################## Labeled vs unlabeled neurons #################################

plt.figure(figsize=(8,5))
fig, axes = plt.subplots(2,2,figsize=(8,5))

areas = ['V1','PM']

for i,iarea in enumerate(areas):
    for j,jarea in enumerate(areas):
        dfarea = df[df['AreaPair']==iarea + '-' + jarea]
        sns.lineplot(ax=axes[i,j],x=np.round(dfarea['DistPair'],-1),y=dfarea['NoiseCorrelation'],hue=dfarea['LabelPair'])
        axes[i,j].set_xlabel="Pairwise distance (um)"
        # plt.legend(labels=['V1-V1','PM-PM'])
        axes[i,j].set_xlim([-10,200])
        axes[i,j].set_ylim([-0.05,0.2])

