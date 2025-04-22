# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import preprocessing
from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes

from utils.imagelib import load_natural_images #
from utils.explorefigs import *
from utils.psth import compute_tensor,compute_respmat,construct_behav_matrix_ts_F
from loaddata.get_data_folder import get_local_drive
from utils.corr_lib import mean_resp_image,compute_signal_correlation, compute_pairwise_metrics
from utils.plot_lib import shaded_error
from utils.RRRlib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\')

#################################################
# session_list        = np.array([['LPE09665','2023_03_15']])
session_list        = np.array([['LPE11086','2023_12_16']])
sessions,nSessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')

#%% Load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'IM',session_list=session_list)
# sessions,nSessions   = filter_sessions(protocols = ['GR'],load_behaviordata=True, 

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=True)

#### Load the natural images:
natimgdata = load_natural_images(onlyright=True)

################################################################
#Show some traces and some stimuli to see responses:

sesidx = 0

fig = plot_excerpt(sessions[sesidx],trialsel=None,plot_neural=True,plot_behavioral=False)

trialsel = [3294, 3374]
fig = plot_excerpt(sessions[sesidx],trialsel=trialsel,plot_neural=True,plot_behavioral=True,neural_version='traces')
# fig.savefig(os.path.join(savedir,'TraceExcerpt_dF_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'Excerpt_Traces_deconv_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_excerpt(sessions[sesidx],trialsel=None,plot_neural=True,plot_behavioral=True,neural_version='raster')
fig = plot_excerpt(sessions[sesidx],trialsel=trialsel,plot_neural=True,plot_behavioral=True,neural_version='raster')
fig.savefig(os.path.join(savedir,'Excerpt_Raster_dF_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


########################### Show PCA ##########################
sesidx = 0
# fig = PCA_gratings_3D(sessions[sesidx])
# fig.savefig(os.path.join(savedir,'PCA','PCA_3D_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_PCA_images(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA','PCA_Gratings_All_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plt.figure()
sns.histplot(sessions[sesidx].respmat_runspeed,binwidth=0.5)
plt.ylim([0,100])


sesidx = 0


sessions[sesidx].trialdata['repetition'] = np.r_[np.zeros([2800]),np.ones([2800])]

#Sort based on image number:
arr1inds                = sessions[sesidx].trialdata['ImageNumber'][:2800].argsort()
arr2inds                = sessions[sesidx].trialdata['ImageNumber'][2800:5600].argsort()

respmat = sessions[sesidx].respmat[:,np.r_[arr1inds,arr2inds+2800]]
# respmat_sort = sessions[sesidx].respmat_z[:,np.r_[arr1inds,arr2inds+2800]]

from sklearn.preprocessing import normalize

min_max_scaler = preprocessing.MinMaxScaler()
respmat_sort = preprocessing.minmax_scale(respmat, feature_range=(0, 1), axis=0, copy=True)

respmat_sort = normalize(respmat, 'l2', axis=1)

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
# fig, axes = plt.subplots(2, 1, figsize=(7, 17))

# axes[0].imshow(respmat_sort[:,:2800], aspect='auto',vmin=-100,vmax=200) 
axes[0].imshow(respmat_sort[:,:2800], aspect='auto',vmin=np.percentile(respmat_sort,5),vmax=np.percentile(respmat_sort,95))
axes[0].set_xlabel('Image #')
axes[0].set_ylabel('Neuron')
axes[0].set_title('Repetition 1')
# axes[1].imshow(respmat_sort[:,2800:], aspect='auto',vmin=-100,vmax=200) 
axes[1].imshow(respmat_sort[:,2800:], aspect='auto',vmin=np.percentile(respmat_sort,5),vmax=np.percentile(respmat_sort,95)) 
axes[1].set_xlabel('Image #')
axes[1].set_ylabel('Neuron')
plt.tight_layout(rect=[0, 0, 1, 1])
axes[1].set_title('Repetition 2')


#%% 



#%% ##### Show response-triggered frame for cells:
for ises in range(nSessions):
    nImages = len(np.unique(sessions[ises].trialdata['ImageNumber']))
    nNeurons = np.shape(sessions[ises].respmat)[0]
    sessions[ises].respmat_image = np.empty((nNeurons,nImages))
    for iIm in range(nImages):
        sessions[ises].respmat_image[:,iIm] = np.mean(sessions[sesidx].respmat[:,sessions[sesidx].trialdata['ImageNumber']==iIm],axis=1)
    
    #Compute response triggered average image:
    sessions[ises].RTA = np.empty((*np.shape(natimgdata)[:2],nNeurons))

    for iN in range(nNeurons):
        print(iN)
        sessions[ises].RTA[:,:,iN] = np.average(natimgdata, axis=2, weights=sessions[ises].respmat_image[iN,:])

#%% #### Plot X examples from V1 and PM with high variance in the average image (capturing some consistent preference): ####
RTA_var         = np.var(sessions[ises].RTA,axis=(0,1))

nExamples       = 25
areas           = ['V1', 'PM']

for area in areas: 
    temp = RTA_var
    temp[sessions[ises].celldata['roi_name'] != area] = 0
    
    # temp_ranked     = np.argsort(RTA_var)
    # temp_ranked     = temp_ranked[np.intersect1d(temp_ranked,
    #                                  np.where(sessions[ises].celldata['roi_name'] == area)[0],
    #                                  return_indices=True)[1]]

    example_cells   = np.argsort(temp)[-nExamples:]

    Rows        = int(np.floor(np.sqrt(nExamples)))
    Cols        = nExamples // Rows # Compute Rows required
    if nExamples % Rows != 0: #If one additional row is necessary -> add one:
        Cols += 1
    Position = range(1,nExamples + 1) # Create a Position index

    fig = plt.figure(figsize=[18, 9])
    for i,n in enumerate(example_cells):
        # add every single subplot to the figure with a for loop
        ax = fig.add_subplot(Rows,Cols,Position[i])
        plt.imshow(sessions[ises].RTA[:,:,n],cmap='gray')#,vmin=100,vmax=150)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        ax.set_title("%d" % n)
    plt.suptitle(area, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(os.path.join(savedir,'ResponseTriggeredAverageImage_%s' % area + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% ##################### Plot control figure of signal corrs ##############################
sesidx = 0
fig = plt.subplots(figsize=(8,5))
plt.imshow(sessions[sesidx].sig_corr, cmap='coolwarm',vmin=-0.02,vmax=0.04)
plt.savefig(os.path.join(savedir,'SignalCorrelations','Signal_Correlation_Images_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% ###### Regress out behavioral state related activity  #################################

X = np.column_stack((sessions[sesidx].respmat_runspeed,sessions[sesidx].respmat_videome))
Y = sessions[sesidx].respmat.T

fig = plot_PCA_images(sessions[sesidx])

sessions[sesidx].respmat = regress_out_behavior_modulation(sessions[sesidx],X,Y,nvideoPCs = 30,rank=2).T

sessions[sesidx].respmat[sessions[sesidx].respmat>np.percentile(sessions[sesidx].respmat,99.5)] = np.percentile(sessions[sesidx].respmat,99.5)
sessions[sesidx].respmat[sessions[sesidx].respmat<np.percentile(sessions[sesidx].respmat,0.5)] = np.percentile(sessions[sesidx].respmat,0.5)
EV(sessions[sesidx].calciumdata,sessions[sesidx].calciumdata2)
fig = plot_PCA_images(sessions[sesidx])

# sessions[sesidx].respmat = regress_out_behavior_modulation(sessions[sesidx],X,Y,nvideoPCs = 30,rank=2).T
sessions[sesidx].respmat = regress_out_behavior_modulation(sessions[sesidx],nvideoPCs = 30,rank=5).T


sessions[sesidx].calciumdata2 = regress_out_behavior_modulation(sessions[sesidx],nvideoPCs = 30,rank=15)
#Compute average response per trial:
for ises in range(nSessions):
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata2, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

fig = plot_PCA_images(sessions[sesidx])



####



respmean = mean_resp_image(sessions[sesidx])



# %% LM model run
B_hat = LM(Y=D, X=S, lam=10)
# B_hat = LM(X, Y, lam=0.01)
D_hat = S @ B_hat

B_hat_lr = RRR(D, S, B_hat, r=2, mode='left')
B_hat_lr = RRR(D, S, B_hat, r=2, mode='right')
D_hat_lr = S @ B_hat_lr

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(8,4))
ax1.imshow(D[:1000,:100].T,vmin=0,vmax=1000,aspect='auto')
ax2.imshow(D_hat[:1000,:100].T,vmin=0,vmax=1000,aspect='auto')
ax3.imshow(D_hat_lr[:1000,:100].T,vmin=0,vmax=1000,aspect='auto')

# %% xval lambda
n = 1000
k = 5
lam = xval_ridge_reg_lambda(Y[:n,:], X[:n,:], k)




# %% cheat
lam = 35

# %% LM model run
B_hat = LM(Y, X, lam=lam)
Y_hat = X @ B_hat

print("LM model error:")
print("LM: %5.3f " % Rss(Y,Y_hat))


S,Slabels = construct_behav_matrix_ts_F(sessions[sesidx],nvideoPCs=nvideoPCs)

sns.heatmap(np.corrcoef(S,rowvar=False),xticklabels=Slabels,yticklabels=Slabels)

