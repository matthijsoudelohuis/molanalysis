# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:53:24 2023

@author: USER
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import binned_statistic
from sklearn import preprocessing


# from sklearn.decomposition import PCA

procdatadir         = "V:\\Procdata\\"

animal_ids          = ['LPE09665'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_03_15']
protocol            = ['IM']

sesfolder = os.path.join(procdatadir,protocol[0],animal_ids[0],sessiondates[0],)

#load the data:
sessiondata         = pd.read_csv(os.path.join(sesfolder,"sessiondata.csv"), sep=',', index_col=0)
behaviordata        = pd.read_csv(os.path.join(sesfolder,"behaviordata.csv"), sep=',', index_col=0)
celldata            = pd.read_csv(os.path.join(sesfolder,"celldata.csv"), sep=',', index_col=0)
calciumdata         = pd.read_csv(os.path.join(sesfolder,"calciumdata.csv"), sep=',', index_col=0)
trialdata           = pd.read_csv(os.path.join(sesfolder,"trialdata.csv"), sep=',', index_col=0)

#get only good cells:
idx = celldata['iscell'] == 1
celldata            = celldata[idx].reset_index(drop=True)
calciumdata         = calciumdata.drop(calciumdata.columns[~idx.append(pd.Series([True]),ignore_index = True)],axis=1)

#Get timestamps and remove from dataframe:
ts_F                = calciumdata['timestamps'].to_numpy()
calciumdata         = calciumdata.drop(columns=['timestamps'],axis=1)

# zscore all the calcium traces:
calciumdata_z      = st.zscore(calciumdata.copy(),axis=1)


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



