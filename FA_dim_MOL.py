# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:53:22 2022

@author: joana
@author: Matthijs Oude Lohuis, 2023, Champalimaud Research
"""

#%% IMPORT RELEVANT PACKAGES

# import h5py
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy.io
from scipy.sparse import csr_matrix
from scipy import stats
import random
import math
import matplotlib.font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sklearn.metrics

from loaddata.session_info import load_sessions
from utils.psth import compute_tensor

from applyFA_dim import apply_FA

##################################################
session_list        = np.array([['LPE09830','2023_04_12']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,
                                    load_behaviordata=True, load_calciumdata=True, load_videodata=False, calciumversion='deconv')

#Get n neurons from V1 and from PM:
n                   = 100
V1_selec            = np.random.choice(np.where(sessions[0].celldata['roi_name']=='V1')[0],n)
PM_selec            = np.random.choice(np.where(sessions[0].celldata['roi_name']=='PM')[0],n)
sessions[0].calciumdata     = sessions[0].calciumdata.iloc[:,np.concatenate((V1_selec,PM_selec))]
sessions[0].celldata        = sessions[0].celldata.iloc[np.concatenate((V1_selec,PM_selec)),:]

##############################################################################
## Construct tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], t_pre, t_post, binsize,method='binmean')

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
[tensor,t_axis]     = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='interp_lin')

tensor              = tensor.transpose((1,2,0))
[N,T,allK]          = np.shape(tensor) #get dimensions of tensor
# [K,N,allT]         = np.shape(tensor) #get dimensions of tensor

# Reshape tensor to neurons (N) by time (T) by trial repetitions (K) by orientations (O)
oris        = sorted(sessions[0].trialdata['Orientation'].unique())
ori_counts  = sessions[0].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
assert(len(ori_counts) == 16 or len(ori_counts) == 8)
assert(np.all(ori_counts == 200) or np.all(ori_counts == 400))

O = len(ori_counts)
K = int(np.mean(ori_counts))

idx_V1 = np.where(sessions[0].celldata['roi_name']=='V1')[0]
idx_PM = np.where(sessions[0].celldata['roi_name']=='PM')[0]

N_V1    = len(idx_V1)
N_PM    = len(idx_PM)

V1_data = np.zeros((N_V1, T, K, O))
PM_data = np.zeros((N_PM, T, K, O))

# idx_V1 = sessions[0].celldata['roi_name']=='V1'

for i,ori in enumerate(oris):
    idx_ori = sessions[0].trialdata['Orientation']==ori
    # V1_data[:,:,:,i] = tensor[sessions[0].celldata['roi_name']=='V1', :, sessions[0].trialdata['Orientation']==ori]
    V1_data[:,:,:,i] = tensor[np.ix_(idx_V1,np.arange(T),idx_ori)]


bin_width = 1
subset = []

V1_data = 

FA_output = apply_FA(V1_data, t_axis, bin_width, subset)


        
#%% APPLY FA TO ALL SESSIONS

bin_width = 80

'E:\OneDrive\PostDoc\Analysis\FA_Analysis'

for sess in sessions:
    
    FA_output = apply_FA(sess, bin_width, [])
    
    file_name = 'FA_output_' + sess.session_id
    
    output_path = 'E:\\OneDrive\\PostDoc\\Analysis\\FA_Analysis\\' + file_name + '.npy'

    np.save(output_path, FA_output, allow_pickle = True)



# Plotting definitions
plt.rcParams.update({'font.size':16})
plt.rcParams['font.family'] = 'Arial'

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()