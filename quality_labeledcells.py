# -*- coding: utf-8 -*-
"""
This script analyzes the quality of the recordings and their relation 
to various factors such as depth of recording, being labeled etc.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loaddata.session_info import filter_sessions,load_sessions,report_sessions

protocol            = 'VR'

sessions            = filter_sessions(protocol)

## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


###################### Calcium trace skewness for labeled vs unlabeled cells:
sns.violinplot(data = celldata,y = "skew",x="redcell")

###################### Noise level for labeled vs unlabeled cells:

## plot precentage of labeled cells as a function of depth:
sns.barplot(x='depth', y='redcell', data=celldata[celldata['roi_name'].isin(['V1','PM'])], estimator=lambda y: sum(y==1)*100.0/len(y))
sns.lineplot(data=celldata[celldata['roi_name'].isin(['V1','PM'])],x='depth', y='redcell', estimator=lambda y: sum(y==1)*100.0/len(y))

sns.lineplot(data=celldata[celldata['roi_name'].isin(['V1','PM'])],x=np.round(celldata['depth'],-1), y='redcell', estimator=lambda y: sum(y==1)*100.0/len(y))

#Plot fraction of labeled cells across areas of recordings: 
sns.barplot(x='roi_name', y='redcell', data=celldata, estimator=lambda x: sum(x==1)*100.0/len(x))

#Probability of being a labeled cell per area: 
sns.barplot(x='roi_name', y='redcell_prob', data=celldata)

## plot number of cells per plane across depths:
sns.histplot(data=celldata, x='depth',hue='roi_name')

## plot quality of cells per plane across depths:
sns.lineplot(data=celldata, x="depth",y=celldata['skew'],estimator='mean')

##### 

#Show total counts across all planes:
fig = plt.figure(figsize=[5, 4])
sns.histplot(data=celldata,x="redcell_prob",hue='redcell',stat='count',linestyle='solid',binwidth=0.025,element='step')

#Show for each plane separately, hack = split by depths assuming each plane has different micrometer of depth
fig = plt.figure(figsize=[5, 4])

for i,depth in enumerate(np.unique(celldata['depth'])):
    sns.kdeplot(data=celldata[celldata['depth'] == depth],x="redcell_prob",hue='redcell',linestyle='solid')


#################### 

plt.figure(figsize=(5,4))
plt.scatter(redcell[iscell[:,0]==1,1],noise_level[iscell[:,0]==1],s=8,c='k')
plt.xlabel('red cell probability')
plt.ylabel('noise level ')

#Count the number of events by taking stretches with z-scored activity above 2:
nEvents         = np.sum(np.diff(np.ndarray.astype(zF > 2,dtype='uint8'))==1,axis=1)
event_rate     = nEvents / (ops['nframes'] / ops['fs'])

plt.figure(figsize=(5,4))
plt.scatter(event_rate[iscell[:,0]==1],noise_level[iscell[:,0]==1],s=8,c='k')
plt.xlabel('event rate')
plt.ylabel('noise level ')


# df = sns.load_dataset("tips")
# x, y, hue = "day", "proportion", "sex"
# hue_order = ["Male", "Female"]

# (df[x]
#  .groupby(df[hue])
#  .value_counts(normalize=True)
#  .rename(y)
#  .reset_index()
#  .pipe((sns.barplot, "data"), x=x, y=y, hue=hue))