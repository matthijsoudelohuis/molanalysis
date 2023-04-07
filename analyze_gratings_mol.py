# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:24:24 2023

@author: USER
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import binned_statistic

# from sklearn.decomposition import PCA

procdatadir         = "V:\\Procdata\\"

animal_ids          = ['LPE09665'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_03_14']
protocol            = ['GR']

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
ts_F                = calciumdata['timestamps']
calciumdata         = calciumdata.drop(columns=['timestamps'],axis=1)

# zscore all the calcium traces:
calciumdata_z      = st.zscore(calciumdata.copy(),axis=1)




## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2  #temporal binsize in s

binedges    = np.arange(t_pre-binsize/2,t_post+binsize+binsize/2,binsize)
bincenters  = np.arange(t_pre,t_post+binsize,binsize)

N           = celldata.shape[0]
K           = trialdata.shape[0]
T           = len(bincenters)

tensor      = np.empty([N,K,T])
tensor_z    = np.empty([N,K,T])

for n in range(N):
    # print('Computing tensor for neuron %d/%d',[n,N)
    print(f"\rComputing tensor for neuron {n+1} / {N}")
    for k in range(K):
        tensor[n,k,:]       = binned_statistic(ts_F-trialdata['tOnset'][k],calciumdata.iloc[:,n], statistic='mean', bins=binedges)[0]
        tensor_z[n,k,:]     = binned_statistic(ts_F-trialdata['tOnset'][k],calciumdata_z.iloc[:,n], statistic='mean', bins=binedges)[0]


resp_meantime_z = tensor_z[:,:,np.logical_and(bincenters>0,bincenters<1.5)].mean(axis=2)

resp_meantime_z = tensor[:,:,np.logical_and(bincenters>0,bincenters<1.5)].mean(axis=2) - tensor[:,:,np.logical_and(bincenters>-1,bincenters<0)].mean(axis=2)

resp_meanori_z = np.empty([N,16])
oris = np.sort(pd.Series.unique(trialdata['Orientation']))

for n in range(N):
    for i,ori in enumerate(oris):
        resp_meanori_z[n,i] = np.nanmean(resp_meantime_z[n,trialdata['Orientation']==ori],axis=0)

prefori  = np.argmax(resp_meanori_z,axis=1)

resp_meanori_z_pref = resp_meanori_z.copy()
for n in range(N):
    resp_meanori_z_pref[n,:] = np.roll(resp_meanori_z[n,:],-prefori[n])

#Sort based on response magnitude:
magresp                 = np.max(resp_meanori_z,axis=1) - np.min(resp_meanori_z,axis=1)
arr1inds                = magresp.argsort()
resp_meanori_z_pref     = resp_meanori_z_pref[arr1inds[::-1],:]

fig, ax = plt.subplots(figsize=(4, 7))
ax.imshow(resp_meanori_z_pref, aspect='auto',vmin=-100,vmax=750) 
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
ax.set_xlabel('Orientation (s)')
ax.set_ylabel('Neuron')

plt.close('all')



