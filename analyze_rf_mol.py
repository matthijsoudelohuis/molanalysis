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

# animal_ids          = ['LPE09665'] #If empty than all animals in folder will be processed
# sessiondates        = ['2023_03_14']
animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_04_10']
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

N = 200

rfmaps          = np.zeros([xGrid,yGrid,N])

t_resp_start     = 0.2        #pre s
t_resp_stop      = 0.6        #post s
t_base_start     = -0.5     #pre s
t_base_stop      = 0        #post s

with np.load(os.path.join(sesfolder,"trialdata.npz")) as data:
    RF_timestamps = data['y']
    
  
for n in range(N):
    print(f"\rComputing RF for neuron {n+1} / {N}")

    for g in range(nGrids):
        temp = np.logical_and(ts_F > RF_timestamps[g]+t_resp_start,ts_F < RF_timestamps[g]+t_resp_stop)
        resp = calciumdata.iloc[temp,n].mean()
        temp = np.logical_and(ts_F > RF_timestamps[g]+t_base_start,ts_F < RF_timestamps[g]+t_base_stop)
        base = calciumdata.iloc[temp,n].mean()
        
        # rfmaps[:,:,n] = rfmaps[:,:,n] + (resp-base) * grid_array[:,:,g]
        rfmaps[:,:,n] = np.nansum(np.dstack((rfmaps[:,:,n],np.max([resp-base,0]) * grid_array[:,:,g])),2)
        # rfmaps[:,:,n] = np.nansum(np.dstack((rfmaps[:,:,n],np.max([resp-base,0]) * grid_array[:,:,g])),2)

        # temp = np.logical_and(ts_F > RF_timestamps[g]+t_base_start,ts_F < RF_timestamps[g]+t_base_stop)
        # resp = calciumdata.iloc[temp,n].mean()
        # rfmaps[:,:,n] = rfmaps[:,:,n] + resp * grid_array[:,:,g]

fig, axes = plt.subplots(10, 20, figsize=[17, 8], sharey='row')
for i in range(10):
    for j in range(20):
        # n = np.random.randint(0,N)
        n = i*10 + j
        ax = axes[i,j]
        # ax.imshow(rfmaps[:,:,n],cmap='gray')
        ax.imshow(rfmaps[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps[:,:,n])),vmax=np.max(abs(rfmaps[:,:,n])))

plt.tight_layout(rect=[0, 0, 0.9, 1])

plt.close('all')

fig, axes = plt.subplots(figsize=[17, 8])
axes.imshow(calciumdata[:,:,n],cmap='gray')



t_resp_start     = 0.0        #pre s
t_resp_stop      = 0.5        #post s



n = 24
n = 0
t_base_start     = -0.5     #pre s
t_base_stop      = 0        #post s

fig, axes = plt.subplots(4, 4, figsize=[17, 8], sharey='row')

t_starts = np.array([0, 0.1, 0.2, 0.3])
t_stops = np.array([0.4,0.5,0.6,0.7])
  
# for i,t_resp_start in np.array([0, 0.1, 0.2, 0.3]):
for i,t_resp_start in enumerate(t_starts):
    for j,t_resp_stop in enumerate(t_stops):
        rfmap = np.zeros([xGrid,yGrid])

        for g in range(nGrids):
            temp = np.logical_and(ts_F >= RF_timestamps[g]+t_resp_start,ts_F <= RF_timestamps[g]+t_resp_stop)
            resp = calciumdata.iloc[temp,n].mean()
            temp = np.logical_and(ts_F >= RF_timestamps[g]+t_base_start,ts_F <= RF_timestamps[g]+t_base_stop)
            base = calciumdata.iloc[temp,n].mean()

            # base = 0

            # rfmap = np.nansum(rfmap,np.max([resp-base,0]) * grid_array[:,:,g])
            rfmap = np.nansum(np.dstack((rfmap,np.max([resp-base,0]) * grid_array[:,:,g])),2)
            # rfmap = rfmap + (resp-base) * grid_array[:,:,g]

        ax = axes[i,j]
        ax.imshow(rfmap,cmap='gray',vmin=-np.max(abs(rfmap)),vmax=np.max(abs(rfmap)))
        # ax.imshow(rfmap,cmap='gray',vmin=-30000,vmax=30000)

