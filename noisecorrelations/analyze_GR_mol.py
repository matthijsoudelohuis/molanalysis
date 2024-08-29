# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%%  ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis\\')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from sklearn.decomposition import PCA
from scipy.stats import zscore, pearsonr,spearmanr
from utils.explorefigs import plot_excerpt,plot_PCA_gratings,plot_tuned_response

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

#%% #################################################
session_list        = np.array([['LPE09830','2023_04_10']])
session_list        = np.array([['LPE11086','2024_01_10']])
sessions,nSessions  = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')

sesidx      = 0
randomseed  = 5

#%% #####################################
#Show some traces and some stimuli to see responses:
example_cells   = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM
fig = plot_excerpt(sessions[0])

#%% #############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

for ises in range(nSessions):
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='nearby')
    
for ises in range(nSessions):
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                        t_resp_start=0,t_resp_stop=1,subtr_baseline=False)
    sessions[ises].respmat_runspeed = compute_respmat(sessions[ises].behaviordata['runspeed'], sessions[ises].behaviordata['ts'],
                                    sessions[ises].trialdata['tOnset'],t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)
    sessions[ises].respmat_videome  = compute_respmat(sessions[ises].videodata['motionenergy'], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'],
                                        t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

#%% Plot the averaged response for some tuned neurons: 

example_cells = [3,56,58,62,70]
fig = plot_tuned_response(sessions[sesidx].tensor,sessions[sesidx].trialdata,t_axis,example_cells)
# save the figure
fig.savefig(os.path.join(savedir,'ExploreFigs','TunedResponse_dF_%s.png' % sessions[sesidx].session_id))
# fig.savefig(os.path.join(savedir,'ExploreFigs','TunedResponse_deconv_%s.png' % sessions[sesidx].session_id))

#%% ############################################################################

resp_meanori    = np.empty([N,16])
oris            = np.sort(pd.Series.unique(sessions[ises].trialdata['Orientation']))

for i,ori in enumerate(oris):
    resp_meanori[:,i] = np.nanmean(sessions[ises].respmat[:,sessions[ises].trialdata['Orientation']==ori],axis=1)

prefori  = np.argmax(resp_meanori,axis=1)

resp_meanori_pref = resp_meanori.copy()
for n in range(N):
    resp_meanori_pref[n,:] = np.roll(resp_meanori[n,:],-prefori[n])

#Sort based on response magnitude:
magresp                 = np.max(resp_meanori,axis=1) - np.min(resp_meanori,axis=1)
arr1inds                = magresp.argsort()
resp_meanori_pref       = resp_meanori_pref[arr1inds[::-1],:]

##### Plot orientation tuned response:
fig, ax = plt.subplots(figsize=(4, 7))
# ax.imshow(resp_meanori_pref, aspect='auto',extent=[0,360,0,N],vmin=-150,vmax=700) 
ax.imshow(resp_meanori_pref, aspect='auto',extent=[0,360,0,N],vmin=np.percentile(resp_meanori_pref,5),vmax=np.percentile(resp_meanori_pref,98)) 

plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
ax.set_xlabel('Orientation (deg)')
ax.set_ylabel('Neuron')

#%% PCA plot: 
plot_PCA_gratings(sessions[sesidx],apply_zscore=False)

#%% 
maxdF = np.max(sessions[sesidx].calciumdata,axis=0)
sns.histplot(maxdF,stat='count',color='g',alpha=0.5)#,binwidth=0.3
maxdFTrials = np.max(sessions[sesidx].respmat,axis=1)
sns.histplot(maxdFTrials,stat='count',color='k',alpha=0.5)#,,binwidth=0.3
# plt.xlim([0,15])
plt.xlabel('Max dF/F')
plt.ylabel('Count')
plt.legend(['Full session','Trials'])

#%% PCA plot: 
plot_PCA_gratings(sessions[sesidx],apply_zscore=False,cellfilter=maxdF>0)
plot_PCA_gratings(sessions[sesidx],apply_zscore=True)
plot_PCA_gratings(sessions[sesidx],apply_zscore=True,cellfilter=maxdF>np.percentile(maxdF,75))

#%% ################# PCA on full session neural data and correlate with running speed

X           = zscore(sessions[0].calciumdata,axis=0)

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(X) #fit pca to response matrix (n_samples by n_features)
#dimensionality is now reduced from time by N neurons to time by ncomp

## Get interpolated values for behavioral variables at imaging frame rate:
runspeed_F  = np.interp(x=sessions[sesidx].ts_F,xp=sessions[sesidx].behaviordata['ts'],
                        fp=sessions[sesidx].behaviordata['runspeed'])

plotncomps  = 5
Xp_norm     = preprocessing.MinMaxScaler().fit_transform(Xp)
Rs_norm     = preprocessing.MinMaxScaler().fit_transform(runspeed_F.reshape(-1,1))

cmat = np.empty((plotncomps))
for icomp in range(plotncomps):
    cmat[icomp] = pearsonr(x=runspeed_F,y=Xp_norm[:,icomp])[0]

plt.figure()
for icomp in range(plotncomps):
    sns.lineplot(x=sessions[sesidx].ts_F,y=Xp_norm[:,icomp]+icomp,linewidth=0.5)
sns.lineplot(x=sessions[sesidx].ts_F,y=Rs_norm.reshape(-1)+plotncomps,linewidth=0.5,color='k')

plt.xlim([sessions[sesidx].trialdata['tOnset'][300],sessions[sesidx].trialdata['tOnset'][800]])
for icomp in range(plotncomps):
    plt.text(x=sessions[sesidx].trialdata['tOnset'][500],y=icomp+0.25,s='r=%1.3f' %cmat[icomp])

plt.ylim([0,plotncomps+1])

#%%#############################
# PCA on trial-concatenated matrix:
# Reshape tensor to N by KxT (each row is now the activity of all trials over time concatenated for one neuron)

N,K,T = np.shape(sessions[sesidx].tensor)
mat_zsc     = sessions[sesidx].tensor.reshape(N,K*T,order='F') 
mat_zsc     = zscore(mat_zsc,axis=1)

pca               = PCA(n_components=100) #construct PCA object with specified number of components
Xp                = pca.fit_transform(mat_zsc) #fit pca to response matrix

# [U,S,Vt]          = pca._fit_full(mat_zsc,100) #fit pca to response matrix

# [U,S,Vt]          = pca._fit_truncated(mat_zsc,100,"arpack") #fit pca to response matrix

plt.figure()
sns.lineplot(data=pca.explained_variance_ratio_)
plt.xlim([-1,100])
plt.ylim([0,0.15])
# plt.xscale('log')
# plt.yscale('log')

##############################
## Make dataframe of tensor with all trial, time information etc.

# mat_zsc     = tensor.transpose((0,2,1)).reshape(K*T,N,order='F') 
# mat_zsc     = zscore(mat_zsc,axis=4)

# tracedata = pd.DataFrame(data=mat_zsc, columns=calciumdata.columns)

# tracedata['time']   = np.tile(t_axis,K)
# tracedata['ori']    = np.repeat(trialdata['Orientation'].to_numpy(),T)

# sns.lineplot(
#     data=tracedata, x="time", y=tracedata.columns[2], hue="ori", 
# )

# h = 2



