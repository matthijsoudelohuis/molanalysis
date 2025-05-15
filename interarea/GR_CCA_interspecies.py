# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os, sys
os.chdir('e:\\Python\\molanalysis')
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

#%% 
#        #####     #    ######     ######     #    #######    #    
#       #     #   # #   #     #    #     #   # #      #      # #   
#       #     #  #   #  #     #    #     #  #   #     #     #   #  
#       #     # #     # #     #    #     # #     #    #    #     # 
#       #     # ####### #     #    #     # #######    #    ####### 
#       #     # #     # #     #    #     # #     #    #    #     # 
#######  #####  #     # ######     ######  #     #    #    #     # 

#%% Load an example mouse session: 
session_list            = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions      = filter_sessions(protocols = 'GR',only_session_id=session_list)


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


#%% Load Monkey Data: 
# import functions 
import scipy.io
import utils.fct_data as dat

#%% ### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Open data
ii_session = 2

path = 'E:\\Python\\AminData\\'
data = scipy.io.loadmat(path+'MatlabData/mat_neural_data/'+dat.session_names[ii_session]+'.mat')['neuralData'][0][0]

# Import the data:
spikesV1_array, spikesV2_array, stimID, trialID = dat.GetData(ii_session, 'Stim', '')

#%% Show some of the data: #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#Show two example neurons: 
fig,axes = plt.subplots(1,2,figsize=(10,10))
axes[0].imshow(spikesV1_array[0,:,:],vmin=0,vmax=.1)
axes[1].imshow(spikesV2_array[3,:,:],vmin=0,vmax=.1)

#%% Make a session object to directly relate to mouse data: 
ses         = Session(protocol='GR', animal_id=dat.session_names[ii_session], sessiondate=dat.session_names[ii_session])

Nstimuli    = 8
oris        = np.arange(0,180,180/Nstimuli)
NV1         = np.shape(spikesV1_array)[0]
NV2         = np.shape(spikesV2_array)[0]
Nrepet      = int(spikesV1_array.shape[1]/Nstimuli)
Ntrials     = spikesV1_array.shape[1]

ses.celldata  = pd.DataFrame({'roi_name': np.concatenate((np.tile(['V1'],NV1),np.tile(['V2'],NV2))),
                              'tuning_var': np.zeros(NV1+NV2)
                              })
                            #    'area':np.tile(['V1'],NV1), 'area':['V1','V2']})

ses.trialdata = pd.DataFrame({'Orientation': oris[stimID[::2]-1]})
# ses.trialdata['Orientation'] = oris[stimID[::2]-1]

idx_time = np.arange(0,spikesV1_array.shape[2],1)
idx_time = (idx_time>100) & (idx_time<1000)

V1resp                  = np.mean(spikesV1_array[:,:,idx_time], axis=2)
V2resp                  = np.mean(spikesV2_array[:,:,idx_time], axis=2)
ses.respmat             = np.concatenate((V1resp,V2resp),axis=0)

ses.respmat_videome     = np.ones(Ntrials)
ses.respmat_runspeed    = np.ones(Ntrials)
sessions.append(ses)

#%% Make the 3D figure for original data:
fig = plot_PCA_gratings_3D(sessions[1],size='uniform',thr_tuning=0)
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=0, roll=-10)
axes[0].set_zlim([-5,45])
my_savefig(fig,savedir,'Cone_3D_V1_V2_KohnData_%s' % sessions[1].session_id,formats=['png'])


#%% 







#%% 
 #####  ######  #######  #####   #####      #####  ######  #######  #####  ### #######  #####  
#     # #     # #     # #     # #     #    #     # #     # #       #     #  #  #       #     # 
#       #     # #     # #       #          #       #     # #       #        #  #       #       
#       ######  #     #  #####   #####      #####  ######  #####   #        #  #####    #####  
#       #   #   #     #       #       #          # #       #       #        #  #             # 
#     # #    #  #     # #     # #     #    #     # #       #       #     #  #  #       #     # 
 #####  #     # #######  #####   #####      #####  #       #######  #####  ### #######  #####  

 #####   #####     #    
#     # #     #   # #   
#       #        #   #  
#       #       #     # 
#       #       ####### 
#     # #     # #     # 
 #####   #####  #     # 

#%% For the monkey dataset: 
# stim: stimulus presented on each trial. Stimulus IDs are: 0 - blank screen; 1 - 0º
# drifting grating; 2 - 22.5º drifting grating; 3 - 45º drifting grating; 4 - 67.5º drifting
# grating; 5 - 90º drifting grating; 6 - 112.5º drifting grating; 7 - 135º drifting
# grating; 8 - 157.5º drifting grating.

#For the mouse dataset: 16 orientations (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 
# 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5) x 200 trials

#%% Subsample the same number of trials from the same orientations:


ises1           = 0
ises2           = 1

sessions_aligned = copy.deepcopy(sessions)
sessions_aligned[ises1].trialdata['Orientation'] = np.mod(sessions_aligned[ises1].trialdata['Orientation'],180)

# ori_overlap = np.intersect1d(sessions[ises1].trialdata['Orientation'],sessions[ises2].trialdata['Orientation'])
# for ses in sessions_aligned:
#     idx_T = ses.trialdata['Orientation'].isin(ori_overlap)
#     ses.trialdata = ses.trialdata[idx_T].reset_index(drop=True)
#     ses.respmat   = ses.respmat[:,idx_T]

# minrep = int(np.min([np.shape(sessions_aligned[ises1].respmat)[1],np.shape(sessions_aligned[ises2].respmat)[1]]) / Nstimuli)
# for ses in sessions_aligned:
#     idx_T           = [np.random.choice(np.where(ses.trialdata['Orientation']==ori)[0],minrep,replace=False) for ori in ori_overlap]
#     idx_T           = np.concatenate(idx_T)
#     ses.trialdata   = ses.trialdata.iloc[idx_T].reset_index(drop=True)
#     ses.respmat     = ses.respmat[:,idx_T]

assert(np.shape(sessions_aligned[ises1].respmat)[1]==np.shape(sessions_aligned[ises2].respmat)[1])


#%% Subsample the same number of trials from the same orientations:
ises1           = 0
ises2           = 1

sessions_aligned = copy.deepcopy(sessions)

ori_overlap = np.intersect1d(sessions[ises1].trialdata['Orientation'],sessions[ises2].trialdata['Orientation'])
for ses in sessions_aligned:
    idx_T = ses.trialdata['Orientation'].isin(ori_overlap)
    ses.trialdata = ses.trialdata[idx_T].reset_index(drop=True)
    ses.respmat   = ses.respmat[:,idx_T]

minrep = int(np.min([np.shape(sessions_aligned[ises1].respmat)[1],np.shape(sessions_aligned[ises2].respmat)[1]]) / Nstimuli)
for ses in sessions_aligned:
    idx_T           = [np.random.choice(np.where(ses.trialdata['Orientation']==ori)[0],minrep,replace=False) for ori in ori_overlap]
    idx_T           = np.concatenate(idx_T)
    ses.trialdata   = ses.trialdata.iloc[idx_T].reset_index(drop=True)
    ses.respmat     = ses.respmat[:,idx_T]

assert(np.shape(sessions_aligned[ises1].respmat)[1]==np.shape(sessions_aligned[ises2].respmat)[1])

#%% Make a plot of the CCA structure without any sorting of the trials: 
prePCA          = 200

sortmethod      = 'origain'

pal             = np.tile(sns.color_palette('husl', int(len(oris))), (2, 1))

idx_V1          = sessions_aligned[ises1].celldata['roi_name']=='V1'
X               = sessions_aligned[ises1].respmat[idx_V1,:].T

idx_V1          = sessions_aligned[ises2].celldata['roi_name']=='V1'
Y               = sessions_aligned[ises2].respmat[idx_V1,:].T

# X               = zscore(X,axis=0)  #Z score activity for each neuron
# Y               = zscore(Y,axis=0)

if sortmethod == 'random':
    sort1 = np.random.permutation(X.shape[0])
    sort2 = np.random.permutation(X.shape[0])
elif sortmethod == 'ori':
    sort1 = np.argsort(sessions_aligned[ises1].trialdata['Orientation'])
    sort2 = np.argsort(sessions_aligned[ises2].trialdata['Orientation'])
elif sortmethod == 'gain':
    sort1 = np.argsort(np.nanmean(X,axis=1))
    sort2 = np.argsort(np.nanmean(Y,axis=1))
elif sortmethod == 'origain':
    sort1           = np.lexsort((np.nanmean(X,axis=1), sessions_aligned[ises1].trialdata['Orientation']))[::-1]
    sort2           = np.lexsort((np.nanmean(Y,axis=1), sessions_aligned[ises2].trialdata['Orientation']))[::-1]

X           = X[sort1,:]
Y           = Y[sort2,:]
orisort     = np.stack((sessions_aligned[ises1].trialdata['Orientation'][sort1],
                 sessions_aligned[ises2].trialdata['Orientation'][sort2]))

if prePCA:
    pca         = PCA(n_components=np.min([X.shape[1],Y.shape[1],prePCA]))
    X           = pca.fit_transform(X)
    Y           = pca.fit_transform(Y)

model = CCA(n_components = 6,scale = False, max_iter = 1000)
Xp,Yp = model.fit_transform(X,Y)

# fig,ax = plt.subplots(1,2,figsize=(8,6))
fig = plt.figure(figsize=(8,6))

projs = np.array([0,1,2])
# projs = np.array([3,4,5])
# for iarea, (area,ccaproj) in enumerate(zip(['V1','PM'],[Xp,Yp])):
for ises, (plottitle,ccaproj) in enumerate(zip(['V1-Mouse','V1-Monkey'],[Xp,Yp])):

    ax = fig.add_subplot(1, 2, ises+1, projection='3d')

    ori         = orisort[ises]
    oris        = np.sort(pd.Series.unique(sessions_aligned[ises].trialdata['Orientation']))
    ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        # get all data points for this ori along first PC or projection pairs
        x = ccaproj.T[projs[0], ori_ind[t]]
        y = ccaproj.T[projs[1], ori_ind[t]]  # and the second
        z = ccaproj.T[projs[2], ori_ind[t]]  # and the second
        # each trial is one dot
        # ax.scatter(x, y, z, color=pal[t], s=sizes[ori_ind[t]]*6, alpha=0.8)
        ax.scatter(x, y, z, color=pal[t], s=1, alpha=0.8)
    # if plotgainaxis:
        # ax.plot(Xg[0,:],Xg[1,:],Xg[2,:],color='k',linewidth=1)
    ax.set_xlabel('CCA %d' % (projs[0]+1))  # give labels to axes
    ax.set_ylabel('CCA %d' % (projs[1]+1))
    ax.set_zlabel('CCA %d' % (projs[2]+1))

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
    # ax.view_init(elev=0, azim=0, roll=0)

# my_savefig(fig,savedir,'CCA_highdim_GR_AcrossSpecies_%ssort_V1_%s_with_%s' % (sortmethod,sessions[ises1].session_id,
                                                                # sessions[ises2].session_id),formats=['png'])

my_savefig(fig,savedir,'CCA_GR_AcrossSpecies_%ssort_V1_%s_with_%s' % (sortmethod,sessions[ises1].session_id,
                                                                sessions[ises2].session_id),formats=['png'])

#%% 
