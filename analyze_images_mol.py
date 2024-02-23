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
from scipy.signal import medfilt
from utils.plotting_style import * #get all the fixed color schemes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from utils.explorefigs import *
from utils.psth import compute_tensor,compute_respmat


# from sklearn.decomposition import PCA

savedir = 'C:\\OneDrive\\PostDoc\\Figures\\Images\\'
# savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Images\\ExploreFigs\\'

#################################################
session_list        = np.array([['LPE09665','2023_03_15']])
session_list        = np.array([['LPE11086','2023_12_16']])
sessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')
sessions            = filter_sessions(protocols = ['IM'],load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')
nSessions = len(sessions)
# sessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
#                                     load_calciumdata=True, load_videodata=True, calciumversion='deconv')


for ises in range(nSessions):
    sessions[ises].videodata['pupil_area']    = medfilt(sessions[ises].videodata['pupil_area'] , kernel_size=25)
    sessions[ises].videodata['motionenergy']  = medfilt(sessions[ises].videodata['motionenergy'] , kernel_size=25)
    sessions[ises].behaviordata['runspeed']   = medfilt(sessions[ises].behaviordata['runspeed'] , kernel_size=51)

################################################################
#Show some traces and some stimuli to see responses:

fig = plot_excerpt(sessions[0],trialsel=None,plot_neural=True,plot_behavioral=False)

trialsel = [3294, 3374]
fig = plot_excerpt(sessions[0],trialsel=trialsel,plot_neural=True,plot_behavioral=True,neural_version='traces')
# fig.savefig(os.path.join(savedir,'TraceExcerpt_dF_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'Excerpt_Traces_deconv_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_excerpt(sessions[0],trialsel=None,plot_neural=True,plot_behavioral=True,neural_version='raster')
fig = plot_excerpt(sessions[0],trialsel=trialsel,plot_neural=True,plot_behavioral=True,neural_version='raster')
fig.savefig(os.path.join(savedir,'Excerpt_Raster_dF_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')



#Compute average response per trial:
for ises in range(nSessions):
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)
    # delattr(sessions[ises],'calciumdata')

    #hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
    temp = pd.DataFrame(np.reshape(np.array(sessions[ises].behaviordata['runspeed']),(len(sessions[ises].behaviordata['runspeed']),1)))
    sessions[ises].respmat_runspeed = compute_respmat(temp, sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,method='mean')
    sessions[ises].respmat_runspeed = np.squeeze(sessions[ises].respmat_runspeed)

    #hacky way to create dataframe of the video motion with F x 1 with F number of samples:
    temp = pd.DataFrame(np.reshape(np.array(sessions[ises].videodata['motionenergy']),(len(sessions[ises].videodata['motionenergy']),1)))
    sessions[ises].respmat_videome = compute_respmat(temp, sessions[ises].videodata['timestamps'], sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,method='mean')
    sessions[ises].respmat_videome = np.squeeze(sessions[ises].respmat_videome)

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





