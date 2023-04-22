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

with np.load(os.path.join(sesfolder,"trialdata.npz")) as data:
    RF_timestamps = data['y']
    
    

### get parameters
[xGrid , yGrid , nGrids] = np.shape(grid_array)

N               = celldata.shape[0]

rfmaps          = np.zeros([xGrid,yGrid,N])

t_resp_start     = 0.2        #pre s
t_resp_stop      = 0.6        #post s
t_base_start     = -0.5     #pre s
t_base_stop      = 0        #post s


### Compute RF maps:
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

#### Zscored version:
rfmaps_z          = np.zeros([xGrid,yGrid,N])

for n in range(N):
    print(f"\rZscoring RF for neuron {n+1} / {N}")
    rfmaps_z[:,:,n] = st.zscore(rfmaps[:,:,n],axis=None)

## Show example cell RF maps:
example_cells = [0,24,285,335,377,496,417,551,430,543,696,689,617,612,924] #V1
example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

example_cells = range(900,1000) #PM

Tot         = len(example_cells)
Rows        = int(np.floor(np.sqrt(Tot)))
Cols        = Tot // Rows # Compute Rows required
if Tot % Rows != 0: #If one additional row is necessary -> add one:
    Cols += 1
Position = range(1,Tot + 1) # Create a Position index

fig = plt.figure(figsize=[18, 9])
for i,n in enumerate(example_cells):
    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[i])
    ax.imshow(rfmaps[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps[:,:,n])),vmax=np.max(abs(rfmaps[:,:,n])))
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title(n)
  
plt.tight_layout(rect=[0, 0, 1, 1])


#### 
fig, axes = plt.subplots(7, 13, figsize=[17, 8])
for i in range(np.shape(axes)[0]):
    for j in range(np.shape(axes)[1]):
        n = i*np.shape(axes)[1] + j
        ax = axes[i,j]
        # ax.imshow(rfmaps[:,:,n],cmap='gray')
        ax.imshow(rfmaps[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps[:,:,n])),vmax=np.max(abs(rfmaps[:,:,n])))
        ax.set_axis_off()
        ax.set_aspect('auto')
        ax.set_title(n)

### 
plt.close('all')

####################### Population Receptive Field

depths,ind  = np.unique(celldata['depth'], return_index=True)
depths      = depths[np.argsort(ind)]
areas       = ['V1','V1','V1','V1','PM','PM','PM','PM']

Rows        = 2
Cols        = 4 
Position    = range(1,8 + 1) # Create a Position index

# fig, axes = plt.subplots(2, 4, figsize=[17, 8])
fig = plt.figure()

for iplane,depth in enumerate(depths):
    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[iplane])
    idx = celldata['depth']==depth
    # popmap = np.nanmean(abs(rfmaps_z[:,:,idx]),axis=2)
    popmap = np.nanmean(abs(rfmaps[:,:,idx]),axis=2)
    ax.imshow(popmap,cmap='OrRd')
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title(areas[iplane])
    
plt.tight_layout(rect=[0, 0, 1, 1])

###










## old code to find optimal response window size:
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

