# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:40:54 2023

@author: USER
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
# from scipy.stats import binned_statistic

# from sklearn.decomposition import PCA

procdatadir         = "V:\\Procdata\\"

animal_ids          = ['LPE09665'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_03_14']
protocol            = ['RF']

sesfolder = os.path.join(procdatadir,protocol[0],animal_ids[0],sessiondates[0],)

#load the data:
sessiondata         = pd.read_csv(os.path.join(sesfolder,"sessiondata.csv"), sep=',', index_col=0)
behaviordata        = pd.read_csv(os.path.join(sesfolder,"behaviordata.csv"), sep=',', index_col=0)
celldata            = pd.read_csv(os.path.join(sesfolder,"celldata.csv"), sep=',', index_col=0)
calciumdata         = pd.read_csv(os.path.join(sesfolder,"calciumdata.csv"), sep=',', index_col=0)

with np.load(os.path.join(sesfolder,"trialdata.npz")) as data:
    grid_array = data['x']
    RF_timestamps = data['y']
    
#get only good cells:
idx = celldata['iscell'] == 1
celldata            = celldata[idx].reset_index(drop=True)
calciumdata         = calciumdata.drop(calciumdata.columns[~idx.append(pd.Series([True]),ignore_index = True)],axis=1)

#Get timestamps and remove from dataframe:
ts_F                = calciumdata['timestamps'].to_numpy()
calciumdata         = calciumdata.drop(columns=['timestamps'],axis=1)

# zscore all the calcium traces:
calciumdata_z      = st.zscore(calciumdata.copy(),axis=1)

[xGrid,yGrid ,nGrids] = np.shape(grid_array)

N               = celldata.shape[0]
rfmaps          = np.zeros([xGrid,yGrid,N])

t_resp_start     = 0.2        #pre s
t_resp_stop      = 0.8        #post s
t_base_start     = -0.5     #pre s
t_base_stop      = 0        #post s

with np.load(os.path.join(sesfolder,"trialdata.npz")) as data:
    RF_timestamps = data['y'] +0.25
    
for n in range(N):
    for g in range(nGrids):
        # temp = np.logical_and(ts_F > RF_timestamps[g]+t_resp_start,ts_F < RF_timestamps[g]+t_resp_stop)
        # resp = calciumdata.iloc[temp,n].mean()
        # temp = np.logical_and(ts_F > RF_timestamps[g]+t_base_start,ts_F < RF_timestamps[g]+t_base_stop)
        # base = calciumdata.iloc[temp,n].mean()
        
        # rfmaps[:,:,n] = rfmaps[:,:,n] + (resp-base) * grid_array[:,:,g]
        
        temp = np.logical_and(ts_F > RF_timestamps[g]+t_base_start,ts_F < RF_timestamps[g]+t_base_stop)
        resp = calciumdata_z.iloc[temp,n].mean()
        rfmaps[:,:,n] = rfmaps[:,:,n] + resp * grid_array[:,:,g]


fig, axes = plt.subplots(5, 5, figsize=[17, 8], sharey='row')
for i in range(5):
    for j in range(5):
        n = np.random.randint(0,N)
        ax = axes[i,j]
        ax.imshow(rfmaps[:,:,n].transpose(),cmap='gray')
plt.tight_layout(rect=[0, 0, 0.9, 1])

plt.close('all')
