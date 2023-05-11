# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import binned_statistic
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from sklearn.decomposition import PCA

%matplotlib inline

##################################################
session_list        = np.array([['LPE09830','2023_04_10']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list)

sessions[0].load_data(load_behaviordata=True,load_calciumdata=True)

sessiondata         = sessions[0].sessiondata
celldata            = sessions[0].celldata
calciumdata         = sessions[0].calciumdata
trialdata           = sessions[0].trialdata
behaviordata        = sessions[0].behaviordata


#Get timestamps and remove from dataframe:
ts_F                = np.array(calciumdata['timestamps'])
calciumdata         = calciumdata.drop(columns=['timestamps'],axis=1)

calciumdata         = calciumdata.drop(calciumdata.columns[100:],axis=1)

# zscore all the calcium traces:
calciumdata_z      = st.zscore(calciumdata.copy(),axis=1)

######################################
#Show some traces and some stimuli to see responses:

example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

example_cells = [6,23,130,99,361,177,153,413,435]

trialsel = np.array([50,90])

example_tstart = trialdata['tOnset'][trialsel[0]-1]

example_tstop = trialdata['tOnset'][trialsel[1]-1]

excerpt = np.array(calciumdata.loc[np.logical_and(ts_F>example_tstart,ts_F<example_tstop)])
excerpt = excerpt[:,example_cells]

min_max_scaler = preprocessing.MinMaxScaler()
excerpt = min_max_scaler.fit_transform(excerpt)

# spksselec = spksselec 
[nframes,ncells] = np.shape(excerpt)

for i in range(ncells):
    excerpt[:,i] =  excerpt[:,i] + i

oris = np.unique(trialdata['Orientation'])
rgba_color = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(oris)))  
  
fig, ax = plt.subplots(figsize=[12, 6])
plt.plot(ts_F[np.logical_and(ts_F>example_tstart,ts_F<example_tstop)],excerpt,linewidth=0.5,color='black')
plt.show()

for i in np.arange(trialsel[0],trialsel[1]):
    ax.add_patch(plt.Rectangle([trialdata['tOnset'][i],0],1,ncells,alpha=0.3,linewidth=0,
                               facecolor=rgba_color[np.where(oris==trialdata['Orientation'][i])]))
    
handles= []
for i,ori in enumerate(oris):
    handles.append(ax.add_patch(plt.Rectangle([0,0],1,ncells,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))

pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax.legend(handles,oris,loc='center right', bbox_to_anchor=(1.25, 0.5))

ax.set_xlim([example_tstart,example_tstop])

ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
ax.axis('off')


# plt.close('all')

##############################################################################
## Construct tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='binmean')

[tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
respmat         = tensor[:,:,np.logical_and(t_axis > 0,t_axis < 1)].mean(axis=2)
[K,N,T]         = np.shape(tensor) #get dimensions of tensor

#Alternative method, much faster:
respmat         = compute_respmat(calciumdata, ts_F, trialdata['tOnset'],t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=True)

[K,N]           = np.shape(respmat) #get dimensions of response matrix

#hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
temp = pd.DataFrame(np.reshape(np.array(behaviordata['runspeed']),(len(behaviordata['runspeed']),1)))
respmat_runspeed = compute_respmat(temp, behaviordata['ts'], trialdata['tOnset'],t_resp_start=0,t_resp_stop=1,method='mean')

#############################################################################
resp_meanori = np.empty([N,16])
oris = np.sort(pd.Series.unique(trialdata['Orientation']))

for n in range(N):
    for i,ori in enumerate(oris):
        resp_meanori[n,i] = np.nanmean(respmat[trialdata['Orientation']==ori,n],axis=0)

prefori  = np.argmax(resp_meanori,axis=1)
# prefori  = np.argmax(resp_meanori_z,axis=1)

resp_meanori_pref = resp_meanori.copy()
for n in range(N):
    resp_meanori_pref[n,:] = np.roll(resp_meanori[n,:],-prefori[n])

#Sort based on response magnitude:
magresp                 = np.max(resp_meanori,axis=1) - np.min(resp_meanori,axis=1)
arr1inds                = magresp.argsort()
resp_meanori_pref       = resp_meanori_pref[arr1inds[::-1],:]

fig, ax = plt.subplots(figsize=(4, 7))
# ax.imshow(resp_meanori_pref, aspect='auto',extent=[0,360,0,N],vmin=-150,vmax=700) 
ax.imshow(resp_meanori_pref, aspect='auto',extent=[0,360,0,N],vmin=-0.5,vmax=100) 

plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
ax.set_xlabel('Orientation (deg)')
ax.set_ylabel('Neuron')

# plt.close('all')

############################

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X)
    return Xz

############################
# PCA on trial-averaged responses: and plot result as scatter by orientation:
respmat_zsc   = z_score(respmat)

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(respmat_zsc) #fit pca to response matrix

ori         = trialdata['Orientation']
oris        = np.sort(pd.Series.unique(trialdata['Orientation']))

ori_ind      = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

shade_alpha      = 0.2
lines_alpha      = 0.8
pal = sns.color_palette('husl', len(oris))
pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

projections = [(0, 1), (1, 2), (0, 2)]
fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
for ax, proj in zip(axes, projections):
    for t, t_type in enumerate(oris):                       #plot orientation separately with diff colors
        x = Xp[ori_ind[t],proj[0]]                          #get all data points for this ori along first PC or projection pairs
        y = Xp[ori_ind[t],proj[1]]                          #and the second
        # ax.scatter(x, y, color=pal[t], s=25, alpha=0.8)     #each trial is one dot
        ax.scatter(x, y, color=pal[t], s=respmat_runspeed[ori_ind[t]], alpha=0.8)     #each trial is one dot
        ax.set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
        ax.set_ylabel('PC {}'.format(proj[1]+1))
        
sns.despine(fig=fig, top=True, right=True)
ax.legend(oris,title='Ori')

##############################
# PCA on trial-concatenated matrix:


#reorder such that tensor is K by T y N (not K by N by T
 # then reshape to KxT by N (each column is now the activity of all trials over time concatenated for one neuron)
mat_zsc     = tensor.transpose((0,2,1)).reshape(K*T,N,order='F') 
mat_zsc     = z_score(mat_zsc)

pca               = PCA(n_components=100) #construct PCA object with specified number of components
Xp                = pca.fit_transform(respmat_zsc) #fit pca to response matrix

[U,S,Vt]          = pca._fit_full(respmat_zsc,100) #fit pca to response matrix

[U,S,Vt]          = pca._fit_truncated(respmat_zsc,100,"arpack") #fit pca to response matrix

sns.lineplot(data=pca.explained_variance_ratio_, x="components", y="EV")
plt.figure()
sns.lineplot(data=pca.explained_variance_ratio_)

##############################
## Make dataframe of tensor with all trial, time information etc.

mat_zsc     = tensor.transpose((0,2,1)).reshape(K*T,N,order='F') 
mat_zsc     = z_score(mat_zsc)

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



