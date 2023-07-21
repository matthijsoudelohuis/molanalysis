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
from statannot import add_stat_annotation

protocol            = 'VR'
protocol            = 'GR'

sessions            = filter_sessions(protocol,only_animal_id=['LPE09830','LPE09665'])

report_sessions(sessions)

## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

###################### Calcium trace skewness for labeled vs unlabeled cells:
sns.violinplot(data = celldata,y = "skew",x="redcell")
ax = sns.violinplot(y = celldata[celldata["noise_level"]<1.5]["noise_level"],x = celldata["redcell"]) 

# 

df = sns.load_dataset("tips")
x = "redcell"
y = "noise_level"
order = ['Sun', 'Thur', 'Fri', 'Sat']
order = ['0','1']
order = ['V1','PM']

# ax = sns.violinplot(y = celldata[celldata["noise_level"]<1.5]["noise_level"],x = celldata["redcell"]) 

ax = sns.boxplot(y = celldata[celldata["noise_level"]<1.5]["noise_level"],x = celldata["roi_name"]) 

# ax = sns.boxplot(data=df, x=x, y=y, order=order)
test_results = add_stat_annotation(ax, data=celldata, x=x, y=y, order=order,
                                #    box_pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")],
                                #    box_pairs=[('0','1')],
                                #    box_pairs=[("V1","PM")],
                                   box_pairs=["V1","PM"],
                                   test='Mann-Whitney', text_format='star',
                                   loc='inside', verbose=2)

import statannot
statannot.add_stat_annotation(
    ax,
    data=celldata,
    x="redcell",
    y="noise_level",
    hue=hue,
    box_pairs=(0,1),
    # box_pairs=[
        # (("Biscoe", "Male"), ("Torgersen", "Female")),
        # (("Dream", "Male"), ("Dream", "Female")),
    # ],
    test="t-test_ind",
    text_format="star",
    loc="outside",
)

statannotations 

###################### Noise level for labeled vs unlabeled cells:

## plot precentage of labeled cells as a function of depth:
sns.barplot(x='depth', y='redcell', data=celldata[celldata['roi_name'].isin(['V1','PM'])], estimator=lambda y: sum(y==1)*100.0/len(y))
sns.lineplot(data=celldata[celldata['roi_name'].isin(['V1','PM'])],x='depth', y='redcell', estimator=lambda y: sum(y==1)*100.0/len(y))

sns.lineplot(data=celldata[celldata['roi_name'].isin(['V1','PM'])],x=np.round(celldata['depth'],-1), y='redcell', estimator=lambda y: sum(y==1)*100.0/len(y))

#Plot fraction of labeled cells across areas of recordings: 
sns.barplot(x='roi_name', y='redcell', data=celldata, estimator=lambda x: sum(x==1)*100.0/len(x),palette='Accent')
plt.ylabel('% labeled cells')

#Probability of being a labeled cell per area: 
sns.barplot(x='roi_name', y='redcell_prob', data=celldata)

## plot number of cells per plane across depths:
sns.histplot(data=celldata, x='depth',hue='roi_name')

## plot quality of cells per plane across depths with skew:
sns.lineplot(data=celldata, x="depth",y=celldata['skew'],estimator='mean')

## plot quality of cells per plane across depths with noise level:
sns.lineplot(data=celldata, x="depth",y=celldata['noise_level'],estimator='mean')

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
# plt.scatter(redcell[iscell[:,0]==1,1],noise_level[iscell[:,0]==1],s=8,c='k')
sns.scatterplot(data=celldata,x='redcell_prob',y='noise_level',s=8,c='k')
plt.xlabel('red cell probability')
plt.ylabel('noise level ')

plt.figure(figsize=(5,4))
sns.scatterplot(data=celldata,x='event_rate',y='noise_level',s=8,c='k')
plt.xlabel('event rate')
plt.ylabel('noise level ')





