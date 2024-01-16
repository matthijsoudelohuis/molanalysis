# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from sklearn.decomposition import PCA
from scipy.stats import zscore, pearsonr,spearmanr

# %matplotlib inline

sessions            = filter_sessions(protocols = ['IM'])


##################################################
session_list        = np.array([['LPE09830','2023_04_10']])
session_list        = np.array([['NSH07422','2023_03_13']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=False, calciumversion='dF')

# session_list        = np.array([['LPE09665','2023_03_15']])
# sessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
#                                     load_calciumdata=True, load_videodata=False, calciumversion='dF')

# #Get n neurons from V1 and from PM:
# n                   = 100
# V1_selec            = np.random.choice(np.where(sessions[0].celldata['roi_name']=='V1')[0],n)
# PM_selec            = np.random.choice(np.where(sessions[0].celldata['roi_name']=='PM')[0],n)
# sessions[0].calciumdata     = sessions[0].calciumdata.iloc[:,np.concatenate((V1_selec,PM_selec))]
# sessions[0].celldata        = sessions[0].celldata.iloc[np.concatenate((V1_selec,PM_selec)),:]

# zscore all the calcium traces:
# calciumdata_z      = st.zscore(calciumdata.copy(),axis=1)

######################################
#Show some traces and some stimuli to see responses:

def show_excerpt_traces_gratings(Session,example_cells=None,trialsel=None):
    
    if example_cells is None:
        example_cells = np.random.choice(Session.calciumdata.shape[1],10)

    if trialsel is None:
        trialsel = [np.random.randint(low=0,high=len(Session.trialdata)-400)]
        trialsel.append(trialsel[0]+40)

    example_tstart  = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    excerpt         = np.array(Session.calciumdata.loc[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)])
    excerpt         = excerpt[:,example_cells]

    min_max_scaler = preprocessing.MinMaxScaler()
    excerpt = min_max_scaler.fit_transform(excerpt)

    # spksselec = spksselec 
    [nframes,ncells] = np.shape(excerpt)

    for i in range(ncells):
        excerpt[:,i] =  excerpt[:,i] + i

    oris        = np.unique(Session.trialdata['Orientation'])
    rgba_color  = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(oris)))  
    
    fig, ax = plt.subplots(figsize=[12, 6])
    plt.plot(Session.ts_F[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)],excerpt,linewidth=0.5,color='black')
    # plt.show()

    for i in np.arange(trialsel[0],trialsel[1]):
        ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i],0],1,ncells,alpha=0.3,linewidth=0,
                                facecolor=rgba_color[np.where(oris==Session.trialdata['Orientation'][i])]))

    handles= []
    for i,ori in enumerate(oris):
        handles.append(ax.add_patch(plt.Rectangle([0,0],1,ncells,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(handles,oris,loc='center right', bbox_to_anchor=(1.25, 0.5))

    ax.set_xlim([example_tstart,example_tstop])

    ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
    ax.axis('off')


example_cells   = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM
example_cells   = [6,23,130,99,361,177,153,413,435]

show_excerpt_traces_gratings(sessions[0],example_cells=example_cells,trialsel=[50,90])
show_excerpt_traces_gratings(sessions[0])

# plt.close('all')

##############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

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

def show_excerpt_rastermap(Session,neuronsel=None,trialsel=None):
    
    if trialsel is None:
        trialsel = [np.random.randint(low=0,high=len(Session.trialdata)-400)]
        trialsel.append(trialsel[0]+40)

    example_tstart  = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    excerpt         = np.array(Session.calciumdata.loc[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)])

    if neuronsel is None:
        example_cells = np.random.choice(Session.calciumdata.shape[1],10)

    datamat         = zscore(excerpt.T, axis=1)

    # fit rastermap
    model           = Rastermap(n_PCs=100, n_clusters=50, 
                    locality=0.75, time_lag_window=5).fit(datamat)
    y = model.embedding # neurons x 1
    isort = model.isort

    # bin over neurons
    X_embedding = zscore(utils.bin1d(datamat, bin_size=10, axis=0), axis=1)

    # plot
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")

    if Session.protocol == 'GR':
        stims        = np.unique(Session.trialdata['Orientation'])
        rgba_color  = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(stims)))  
        
        for i in np.arange(trialsel[0],trialsel[1]):
            idx = np.where(Session.ts_F>Session.trialdata['tOnset'][i])[0][0] - np.where(Session.ts_F>example_tstart)[0][0]
            ax.add_patch(plt.Rectangle([idx,0],3,np.shape(X_embedding)[0],alpha=0.3,linewidth=0,
                                    facecolor=rgba_color[np.where(stims==Session.trialdata['Orientation'][i])]))

        handles= []
        for i,stim in enumerate(stims):
            handles.append(ax.add_patch(plt.Rectangle([0,0],1,np.shape(X_embedding)[0],alpha=0.3,linewidth=0,facecolor=rgba_color[i])))

        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(handles,stims,loc='center right', bbox_to_anchor=(1.25, 0.5))

    elif Session.protocol == 'IM':
        stims       = np.unique(Session.trialdata['ImageNumber'])
        rgba_color  = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(stims)))  
        
        for i in np.arange(trialsel[0],trialsel[1]):
            idx = np.where(Session.ts_F>Session.trialdata['tOnset'][i])[0][0] - np.where(Session.ts_F>example_tstart)[0][0]
            ax.add_patch(plt.Rectangle([idx,0],3,np.shape(X_embedding)[0],alpha=0.3,linewidth=0,
                                    facecolor=rgba_color[np.where(stims==Session.trialdata['ImageNumber'][i])]))

        # handles= []
        # for i,im in enumerate(stims):
            # handles.append(ax.add_patch(plt.Rectangle([0,0],1,np.shape(X_embedding)[0],alpha=0.3,linewidth=0,facecolor=rgba_color[i])))


    ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
    ax.axis('off')

trialsel = [100,150]
trialsel = [1,5600]

show_excerpt_rastermap(sessions[0],trialsel=trialsel)

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



