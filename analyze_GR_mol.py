# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math, os
try:
    os.chdir('t:\\Python\\molanalysis\\')
except:
    os.chdir('e:\\Python\\molanalysis\\')
os.chdir('c:\\Python\\molanalysis\\')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from sklearn.decomposition import PCA
from scipy.stats import zscore, pearsonr,spearmanr
from utils.explorefigs import plot_excerpt

savedir = 'C:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\'

##################################################
session_list        = np.array([['LPE09830','2023_04_10']])
session_list        = np.array([['LPE11086','2024_01_10']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=False, calciumversion='dF')
# sessions            = filter_sessions(protocols = ['GR'])
nSessions = len(sessions)

sesidx      = 0
randomseed  = 5

######################################
#Show some traces and some stimuli to see responses:
example_cells   = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM
fig = plot_excerpt(sessions[0])

##############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

for i in range(nSessions):
    [sessions[i].tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='interp_lin')

# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], t_pre, t_post, binsize,method='binmean')

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
[tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='interp_lin')
[N,K,T]         = np.shape(tensor) #get dimensions of tensor
respmat         = tensor[:,:,np.logical_and(t_axis > 0,t_axis < 1)].mean(axis=2)

#Alternative method, much faster:
respmat         = compute_respmat(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=True)
[N,K]           = np.shape(respmat) #get dimensions of response matrix

#hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
temp = pd.DataFrame(np.reshape(np.array(sessions[0].behaviordata['runspeed']),(len(sessions[0].behaviordata['runspeed']),1)))
respmat_runspeed = compute_respmat(temp, sessions[0].behaviordata['ts'], sessions[0].trialdata['tOnset'],
                                   t_resp_start=0,t_resp_stop=1,method='mean')
respmat_runspeed = np.squeeze(respmat_runspeed)

#############################################################################
resp_meanori    = np.empty([N,16])
oris            = np.sort(pd.Series.unique(sessions[0].trialdata['Orientation']))

for i,ori in enumerate(oris):
    resp_meanori[:,i] = np.nanmean(respmat[:,sessions[0].trialdata['Orientation']==ori],axis=1)

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

# plt.close('all')


########### PCA on trial-averaged responses ############
######### plot result as scatter by orientation ########

respmat_zsc = zscore(respmat,axis=1) # zscore for each neuron across trial responses

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(respmat_zsc.T).T #fit pca to response matrix (n_samples by n_features)
#dimensionality is now reduced from N by K to ncomp by K

ori         = sessions[0].trialdata['Orientation']
oris        = np.sort(pd.Series.unique(sessions[0].trialdata['Orientation']))

ori_ind      = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

shade_alpha      = 0.2
lines_alpha      = 0.8
pal = sns.color_palette('husl', len(oris))
pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

projections = [(0, 1), (1, 2), (0, 2)]
fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
for ax, proj in zip(axes, projections):
    for t, t_type in enumerate(oris):                       #plot orientation separately with diff colors
        x = Xp[proj[0],ori_ind[t]]                          #get all data points for this ori along first PC or projection pairs
        y = Xp[proj[1],ori_ind[t]]                          #and the second
        # ax.scatter(x, y, color=pal[t], s=25, alpha=0.8)     #each trial is one dot
        ax.scatter(x, y, color=pal[t], s=respmat_runspeed[ori_ind[t]], alpha=0.8)     #each trial is one dot
        ax.set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
        ax.set_ylabel('PC {}'.format(proj[1]+1))
        
sns.despine(fig=fig, top=True, right=True)
ax.legend(oris,title='Ori')

################## PCA on full session neural data and correlate with running speed

X           = zscore(sessions[0].calciumdata,axis=0)

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(X) #fit pca to response matrix (n_samples by n_features)
#dimensionality is now reduced from time by N to time by ncomp


## Get interpolated values for behavioral variables at imaging frame rate:
runspeed_F  = np.interp(x=sessions[0].ts_F,xp=sessions[0].behaviordata['ts'],
                        fp=sessions[0].behaviordata['runspeed'])

plotncomps  = 5
Xp_norm     = preprocessing.MinMaxScaler().fit_transform(Xp)
Rs_norm     = preprocessing.MinMaxScaler().fit_transform(runspeed_F.reshape(-1,1))

cmat = np.empty((plotncomps))
for icomp in range(plotncomps):
    cmat[icomp] = pearsonr(x=runspeed_F,y=Xp_norm[:,icomp])[0]

plt.figure()
for icomp in range(plotncomps):
    sns.lineplot(x=sessions[0].ts_F,y=Xp_norm[:,icomp]+icomp,linewidth=0.5)
sns.lineplot(x=sessions[0].ts_F,y=Rs_norm.reshape(-1)+plotncomps,linewidth=0.5,color='k')

plt.xlim([sessions[0].trialdata['tOnset'][500],sessions[0].trialdata['tOnset'][800]])
for icomp in range(plotncomps):
    plt.text(x=sessions[0].trialdata['tOnset'][700],y=icomp+0.25,s='r=%1.3f' %cmat[icomp])

plt.ylim([0,plotncomps+1])

########################################

#### 

# spks is neurons by time
spks = np.load("W:\\Users\\Matthijs\\temp\\spks.npy").astype("float32")
spks = zscore(spks, axis=1)

# fit rastermap
model = Rastermap(n_PCs=200, n_clusters=100, 
                  locality=0.75, time_lag_window=5).fit(spks)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
X_embedding = zscore(utils.bin1d(spks, bin_size=25, axis=0), axis=1)

# plot
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
ax.axis('off')

##############################
# PCA on trial-concatenated matrix:
# Reorder such that tensor is N by K x T (not K by N by T)
# then reshape to N by KxT (each row is now the activity of all trials over time concatenated for one neuron)

mat_zsc     = tensor.transpose((1,0,2)).reshape(N,K*T,order='F') 
mat_zsc     = zscore(mat_zsc,axis=4)

pca               = PCA(n_components=100) #construct PCA object with specified number of components
Xp                = pca.fit_transform(mat_zsc) #fit pca to response matrix

# [U,S,Vt]          = pca._fit_full(mat_zsc,100) #fit pca to response matrix

# [U,S,Vt]          = pca._fit_truncated(mat_zsc,100,"arpack") #fit pca to response matrix

plt.figure()
sns.lineplot(data=pca.explained_variance_ratio_)
plt.xlim([-1,100])
plt.ylim([0,0.15])

##############################
## Make dataframe of tensor with all trial, time information etc.

mat_zsc     = tensor.transpose((0,2,1)).reshape(K*T,N,order='F') 
mat_zsc     = zscore(mat_zsc,axis=4)

tracedata = pd.DataFrame(data=mat_zsc, columns=calciumdata.columns)

tracedata['time']   = np.tile(t_axis,K)
tracedata['ori']    = np.repeat(trialdata['Orientation'].to_numpy(),T)

sns.lineplot(
    data=tracedata, x="time", y=tracedata.columns[2], hue="ori", 
)

h = 2


# PCA on trial-averaged data: 


# snakeselec = np.array(snakeplots[:,(bincenters>-60) & (bincenters<30),:])
# nBins = np.shape(snakeselec)[1]
# Xa = np.reshape(snakeselec, [N,nBins*4])

# n_components = 15
# pca = PCA(n_components=n_components)
# Xa_p = pca.fit_transform(Xa).T

# trial_size = nBins
# space = bincenters[(bincenters>-60) & (bincenters<30)]

# fig, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey='row')
# for comp in range(3):
#     ax = axes[comp]
#     for kk, type in enumerate(trial_types):
#         x = Xa_p[comp, kk * trial_size :(kk+1) * trial_size]
#         # x = gaussian_filter1d(x, sigma=3)
#         ax.plot(space, x, c=pal[kk])
#     # add_stim_to_plot(ax)
#     ax.set_ylabel('PC {}'.format(comp+1))
# # add_orientation_legend(axes[2])
# axes[1].set_xlabel('Time (s)')
# sns.despine(fig=fig, right=True, top=True)
# plt.tight_layout(rect=[0, 0, 0.9, 1])

# plt.imshow(Xa)



