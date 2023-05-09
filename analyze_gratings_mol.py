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
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import utils.py


# from sklearn.decomposition import PCA

procdatadir         = "V:\\Procdata\\"

# animal_ids          = ['LPE09665'] #If empty than all animals in folder will be processed
# sessiondates        = ['2023_03_14']
animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_04_10']
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
ts_F                = np.array(calciumdata['timestamps'])
calciumdata         = calciumdata.drop(columns=['timestamps'],axis=1)

# zscore all the calcium traces:
calciumdata_z      = st.zscore(calciumdata.copy(),axis=1)

######################################
#Show some traces and some stimuli to see responses:

example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

example_cells = [6,23,130,99,361,177,153,413,435]

trialsel = np.array([50,90])

example_tstart = trialdata['tOnset'][trialsel[0]-1]

example_tstop = trialdata['tOnset'][trialsel[1]-1]

excerpt = np.array(calciumdata.loc[np.logical_and(ts_F>example_tstart,ts_F<example_tstop)])
excerpt = excerpt[:,example_cells]

min_max_scaler = preprocessing.MinMaxScaler()
excerpt = min_max_scaler.fit_transform(excerpt)

# spksselec = spksselec 
[nframes,ncells] = np.shape(excerpt)

for i in range(ncells):
    excerpt[:,i] =  excerpt[:,i] + i


oris = np.unique(trialdata['Orientation'])
rgba_color = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(oris)))  
  
fig, ax = plt.subplots(figsize=[12, 6])
plt.plot(ts_F[np.logical_and(ts_F>example_tstart,ts_F<example_tstop)],excerpt,linewidth=0.5,color='black')
plt.show()

for i in np.arange(trialsel[0],trialsel[1]):
    ax.add_patch(plt.Rectangle([trialdata['tOnset'][i],0],1,ncells,alpha=0.3,linewidth=0,
                               facecolor=rgba_color[np.where(oris==trialdata['Orientation'][i])]))
    
handles= []
for i,ori in enumerate(oris):
    handles.append(ax.add_patch(plt.Rectangle([0,0],1,ncells,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))

pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax.legend(handles,oris,loc='center right', bbox_to_anchor=(1.25, 0.5))

ax.set_xlim([example_tstart,example_tstop])

ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
ax.axis('off')


# plt.close('all')

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

# tensor      = np.empty([N,K,T])
# tensor_z    = np.empty([N,K,T])

# for n in range(N):
#     # print('Computing tensor for neuron %d/%d',[n,N)
#     print(f"\rComputing tensor for neuron {n+1} / {N}")
#     for k in range(K):
#         tensor[n,k,:]       = binned_statistic(ts_F-trialdata['tOnset'][k],calciumdata.iloc[:,n], statistic='mean', bins=binedges)[0]
#         tensor_z[n,k,:]     = binned_statistic(ts_F-trialdata['tOnset'][k],calciumdata_z.iloc[:,n], statistic='mean', bins=binedges)[0]

# #Compute mean response during response window:
# resp_meantime_z = tensor_z[:,:,np.logical_and(bincenters>0,bincenters<1.5)].mean(axis=2)

# #Compute mean response during response window - baseline:
# resp_meantime_z = tensor[:,:,np.logical_and(bincenters>0,bincenters<1.5)].mean(axis=2) - tensor[:,:,np.logical_and(bincenters>-1,bincenters<0)].mean(axis=2)

############Alternative method, much faster:
resp_meantime       = np.empty([N,K])
resp_meantime_z     = np.empty([N,K])

for k in range(K):
    print(f"\rComputing response for trial {k+1} / {K}")
    # resp_meantime[:,k]      = calciumdata[np.logical_and(ts_F>trialdata['tOnset'][k],ts_F<trialdata['tOnset'][k]+2)].to_numpy().mean(axis=0)


    resp_meantime[:,k]      = np.subtract(calciumdata[np.logical_and(ts_F>trialdata['tOnset'][k],ts_F<trialdata['tOnset'][k]+2)].to_numpy().mean(axis=0), 
                                          calciumdata[np.logical_and(ts_F>trialdata['tOnset'][k]-2,ts_F<trialdata['tOnset'][k])].to_numpy().mean(axis=0))


    # resp_meantime[:,k]      = calciumdata_z[np.logical_and(ts_F>trialdata['tOnset'][k],ts_F<trialdata['tOnset'][k]+2)].to_numpy().mean(axis=0)


resp_meanori = np.empty([N,16])
oris = np.sort(pd.Series.unique(trialdata['Orientation']))

for n in range(N):
    for i,ori in enumerate(oris):
        resp_meanori[n,i] = np.nanmean(resp_meantime[n,trialdata['Orientation']==ori],axis=0)
        # resp_meanori_z[n,i] = np.nanmean(resp_meantime_z[n,trialdata['Orientation']==ori],axis=0)

prefori  = np.argmax(resp_meanori,axis=1)
# prefori  = np.argmax(resp_meanori_z,axis=1)

resp_meanori_pref = resp_meanori.copy()
for n in range(N):
    resp_meanori_pref[n,:] = np.roll(resp_meanori[n,:],-prefori[n])

#Sort based on response magnitude:
magresp                 = np.max(resp_meanori,axis=1) - np.min(resp_meanori,axis=1)
arr1inds                = magresp.argsort()
resp_meanori_pref       = resp_meanori_pref[arr1inds[::-1],:]

fig, ax = plt.subplots(figsize=(4, 7))
# ax.imshow(resp_meanori_pref, aspect='auto',extent=[0,360,0,N],vmin=-150,vmax=700) 
ax.imshow(resp_meanori_pref, aspect='auto',extent=[0,360,0,N],vmin=-0.5,vmax=100) 

plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
ax.set_xlabel('Orientation (deg)')
ax.set_ylabel('Neuron')

# plt.close('all')

####



