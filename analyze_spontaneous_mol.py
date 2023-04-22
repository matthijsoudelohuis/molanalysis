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

procdatadir         = "V:\\Procdata\\"

# animal_ids          = ['LPE09665'] #If empty than all animals in folder will be processed
# sessiondates        = ['2023_03_14']
animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_04_10']
protocol            = ['SP']

sesfolder = os.path.join(procdatadir,protocol[0],animal_ids[0],sessiondates[0],)

#load the data:
sessiondata         = pd.read_csv(os.path.join(sesfolder,"sessiondata.csv"), sep=',', index_col=0)
behaviordata        = pd.read_csv(os.path.join(sesfolder,"behaviordata.csv"), sep=',', index_col=0)
celldata            = pd.read_csv(os.path.join(sesfolder,"celldata.csv"), sep=',', index_col=0)
calciumdata         = pd.read_csv(os.path.join(sesfolder,"calciumdata.csv"), sep=',', index_col=0)

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
#Show some traces and some behavioral data alongside it:

# example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

example_cells = [6,23,130,99,361,177,153,413,435]
example_cells = np.arange(0,50)

trialsel = np.array([50,90])

example_tstart = ts_F[0]

example_tstop = ts_F[500]

example_tstart = 1.96993e7

example_tstop = example_tstart+100

excerpt = np.array(calciumdata.loc[np.logical_and(ts_F>example_tstart,ts_F<example_tstop)])
excerpt = excerpt[:,example_cells]

min_max_scaler = preprocessing.MinMaxScaler()
excerpt = min_max_scaler.fit_transform(excerpt)

# spksselec = spksselec 
[nframes,ncells] = np.shape(excerpt)

for i in range(ncells):
    excerpt[:,i] =  excerpt[:,i] + i


fig, ax = plt.subplots(figsize=[12, 6])
plt.plot(ts_F[np.logical_and(ts_F>example_tstart,ts_F<example_tstop)],excerpt,linewidth=0.5)
plt.show()

idx = np.logical_and(behaviordata['ts']>example_tstart,behaviordata['ts']<example_tstop).to_numpy()
plt.plot(behaviordata['ts'][idx],behaviordata['runspeed'][idx],linewidth=0.5,color='black')


# pos = ax.get_position()
# ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
# ax.legend(handles,oris,loc='center right', bbox_to_anchor=(1.25, 0.5))

# ax.set_xlim([example_tstart,example_tstop])

# ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
# ax.axis('off')




