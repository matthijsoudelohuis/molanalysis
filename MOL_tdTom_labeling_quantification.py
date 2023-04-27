# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:13:45 2023

@author: USER
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import scipy.stats as st
# from scipy.stats import binned_statistic
# from sklearn import preprocessing
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

procdatadir         = "V:\\Procdata\\"

animal_ids          = [] #If empty than all animals in folder will be processed
sessiondates        = [] #If empty than all sessions in folder will be processed
protocol            = ['RF']

sessiondata         = pd.DataFrame()
behaviordata        = pd.DataFrame()
celldata            = pd.DataFrame()
calciumdata         = pd.DataFrame()
trialdata           = pd.DataFrame()
        
## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = os.listdir(os.path.join(procdatadir,protocol[0]))

for animal_id in animal_ids: #for each animal
    sessiondates = []
    if len(sessiondates) == 0:
        sessiondates = os.listdir(os.path.join(procdatadir,protocol[0],animal_id)) 

    for sessiondate in sessiondates: #for each of the sessions for this animal
        print(sessiondate)
        sesfolder       = os.path.join(procdatadir,protocol[0],animal_id,sessiondate)

        #load the data:
        sessiondata_load         = pd.read_csv(os.path.join(sesfolder,"sessiondata.csv"), sep=',', index_col=0)
        behaviordata_load        = pd.read_csv(os.path.join(sesfolder,"behaviordata.csv"), sep=',', index_col=0)
        celldata_load            = pd.read_csv(os.path.join(sesfolder,"celldata.csv"), sep=',', index_col=0)
        # calciumdata_load         = pd.read_csv(os.path.join(sesfolder,"calciumdata.csv"), sep=',', index_col=0)
        # trialdata_load           = pd.read_csv(os.path.join(sesfolder,"trialdata.csv"), sep=',', index_col=0)
        
         #get only good cells:
        idx = celldata_load['iscell'] == 1
        celldata_load            = celldata_load[idx].reset_index(drop=True)
        # calciumdata         = calciumdata.drop(calciumdata.columns[~idx.append(pd.Series([True]),ignore_index = True)],axis=1)
    
        if np.shape(sessiondata)[0] == 0:
            sessiondata         = sessiondata_load.copy()
            behaviordata        = behaviordata_load.copy()
            celldata            = celldata_load.copy()
            # calciumdata         = calciumdata_load.copy()
            # trialdata           = trialdata_load.copy()
        else: 
            celldata            = celldata.append(celldata_load)
            sessiondata         = sessiondata.append(sessiondata_load)
            behaviordata        = behaviordata.append(behaviordata_load)


       

#Show total counts across all planes:
fig = plt.figure(figsize=[5, 4])
sns.histplot(data=celldata,x="redcell_prob",hue='redcell',stat='count',linestyle='solid',binwidth=0.025,element='step')

#Show for each plane separately, hack = split by depths assuming each plane has different micrometer of depth
fig = plt.figure(figsize=[5, 4])

for i,depth in enumerate(np.unique(celldata['depth'])):
    sns.kdeplot(data=celldata[celldata['depth'] == depth],x="redcell_prob",hue='redcell',linestyle='solid')









    
            