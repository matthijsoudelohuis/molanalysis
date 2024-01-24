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

from utils.explorefigs import excerpt_neural_behavioral


# from sklearn.decomposition import PCA

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Images\\'


#################################################
session_list        = np.array([['LPE09665','2023_03_15']])
session_list        = np.array([['LPE11086','2023_12_16']])
sessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='dF')

sesidx      = 0

sessions[sesidx].videodata['pupil_area'] = medfilt(sessions[sesidx].videodata['pupil_area'] , kernel_size=25)
sessions[sesidx].videodata['motionenergy'] = medfilt(sessions[sesidx].videodata['motionenergy'] , kernel_size=25)
sessions[sesidx].behaviordata['runspeed'] = medfilt(sessions[sesidx].behaviordata['runspeed'] , kernel_size=51)


# zscore all the calcium traces:
# calciumdata_z      = st.zscore(sessions[sesidx].calciumdata.copy(),axis=1)


################################################################
#Show some traces and some stimuli to see responses:

example_cells   = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

V1_labeled      = 
V1_unlabeled    = 
PM_labeled      = 
PM_unlabeled    = 

excerpt_neural_behavioral(sessions[0],trialsel=[50,90])





## Construct response matrix of N neurons by K trials
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_resp_start     = 0        #pre s
t_resp_stop      = 0.8      #post s
t_base_start     = -0.5     #pre s
t_base_stop      = 0        #post s

N           = celldata.shape[0]
K           = trialdata.shape[0]

respmat      = np.empty([N,K])
respmat_z    = np.empty([N,K])

for k in range(K):
    print(f"\rComputing response vector for trial {k+1} / {K}")

    temp    = np.logical_and(ts_F > trialdata['tOnset'][k]+t_base_start,ts_F < trialdata['tOnset'][k]+t_base_stop)
    base    = calciumdata.iloc[temp,:].mean()
    temp    = np.logical_and(ts_F > trialdata['tOnset'][k]+t_resp_start,ts_F < trialdata['tOnset'][k]+t_resp_stop)
    resp    = calciumdata.iloc[temp,:].mean()

    respmat[:,k] = resp - base

    respmat_z[:,k] = calciumdata_z.iloc[temp,:].mean()


trialdata['repetition'] = np.r_[np.zeros([2800]),np.ones([2800])]

#Sort based on image number:
arr1inds                = trialdata['ImageNumber'][:2800].argsort()
arr2inds                = trialdata['ImageNumber'][2800:5600].argsort()

respmat_sort = respmat[:,np.r_[arr1inds,arr2inds+2800]]
respmat_sort = respmat_z[:,np.r_[arr1inds,arr2inds+2800]]

min_max_scaler = preprocessing.MinMaxScaler()
respmat_sort = preprocessing.minmax_scale(respmat_sort, feature_range=(0, 1), axis=1, copy=True)

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

axes[0].imshow(respmat_sort[:,:2800], aspect='auto',vmin=-100,vmax=200) 
# axes[0].imshow(respmat_sort[:,:2800], aspect='auto',vmin=0.1,vmax=1) 
axes[0].set_xlabel('Image #')
axes[0].set_ylabel('Neuron')
axes[0].set_title('Repetition 1')
axes[1].imshow(respmat_sort[:,2800:], aspect='auto',vmin=-100,vmax=200) 
# axes[1].imshow(respmat_sort[:,2800:], aspect='auto',vmin=0.1,vmax=1) 
axes[1].set_xlabel('Image #')
axes[1].set_ylabel('Neuron')
plt.tight_layout(rect=[0, 0, 1, 1])
axes[1].set_title('Repetition 2')


plt.close('all')



