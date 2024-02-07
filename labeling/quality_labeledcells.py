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
from loaddata.session_info import filter_sessions,load_sessions
from statannotations.Annotator import Annotator

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn import preprocessing

protocol            = ['VR']
# protocol            = ['GR']
protocol            = ['GR','VR','IM','RF','SP']
protocol            = ['GR','VR','IM']
protocol            = ['GR','IM']

# sessions            = filter_sessions(protocol,only_animal_id=['LPE09830','LPE09665'])
# sessions            = filter_sessions(protocol,only_animal_id=['LPE09829'])
sessions            = filter_sessions(protocol,only_animal_id=['LPE10919','LPE10885','LPE10883','LPE11086'],
                                      min_cells=100)

session_list        = np.array([['LPE11086','2023_12_16']])
sessions            = load_sessions(protocol = 'IM',session_list=session_list)

## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

## remove any double cells (for example recorded in both GR and RF)
celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

celldata['redcell'] = celldata['redcell_prob']>0.4
celldata.loc[celldata['redcell']==0,'recombinase'] = 'non'

celldata['noise_level'].values[celldata['noise_level'] > 5] = 0


sns.histplot(data=celldata,x='redcell_prob',stat='probability',hue='redcell',binwidth=0.05)
plt.ylim([0,0.01])

###################### Calcium trace skewness for labeled vs unlabeled cells:
# order = [0,1] #for statistical testing purposes
# pairs = [(0,1)]

order = ['non','flp','cre'] #for statistical testing purposes
pairs = [('non','flp'),('non','cre')]

# fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF","meanF_chan2"]
fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF"]

nfields = len(fields)
fig,axes   = plt.subplots(1,nfields,figsize=(12,4))

celldata = celldata[celldata['meanF_chan2']<1000]
celldata = celldata[celldata['noise_level']<1]


for i in range(nfields):
    # sns.violinplot(data=celldata,y=fields[i],x="redcell",palette=['gray','red'],ax=axes[i])
    sns.violinplot(data=celldata,y=fields[i],x="recombinase",palette=['gray','orangered','indianred'],ax=axes[i])
    axes[i].set_ylim(np.nanpercentile(celldata[fields[i]],[0.1,99.9]))

    annotator = Annotator(axes[i], pairs, data=celldata, x="recombinase", y=fields[i], order=order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    axes[i].set_xlabel('labeled')
    axes[i].set_ylabel('')
    axes[i].set_title(fields[i])

df2 = celldata.groupby(['redcell'])['redcell'].count()

# labelcounts = celldata.groupby(['redcell'])['redcell'].count().to_numpy()
# labelcounts = celldata.groupby(['recombinase'])['recombinase'].count().to_numpy()
labelcounts = celldata.groupby(['recombinase'])['recombinase'].count()
plt.suptitle('Quality comparison non-labeled ({0}), cre-labeled ({1}) and flp-labeled ({2}) cells'.format(
    labelcounts[labelcounts.index=='non'][0],labelcounts[labelcounts.index=='cre'][0],labelcounts[labelcounts.index=='flp'][0]))
plt.tight_layout()

## Scatter of all crosscombinations (seaborn pairplot):
df = celldata[["skew","noise_level","npix_soma",
               "meanF","meanF_chan2","event_rate","redcell"]]
sns.pairplot(data=df, hue="redcell")

sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap='bwr')

## 


# ###################### Calcium trace skewness for labeled vs unlabeled cells:
# sns.violinplot(data=celldata,y="skew",x="redcell",palette='Accent',ax=ax1)
# annotator = Annotator(ax1, pairs, data=celldata, x="redcell", y="skew", order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
# annotator.apply_and_annotate()

# ###################### Calcium trace noise level for labeled vs unlabeled cells:
# sns.violinplot(y = celldata[celldata["noise_level"]<1.5]["noise_level"],x = celldata["redcell"],palette='Accent',ax=ax2)
# annotator = Annotator(ax2, pairs, data=celldata, x="redcell", y="noise_level", order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
# annotator.apply_and_annotate()

# ###################### Calcium trace skewness for labeled vs unlabeled cells:
# sns.violinplot(data=celldata,y="event_rate",x = "redcell",palette='Accent',ax=ax3)
# annotator = Annotator(ax3, pairs, data=celldata, x="redcell", y="event_rate", order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
# annotator.apply_and_annotate()



###################### Noise level for labeled vs unlabeled cells:

## plot precentage of labeled cells as a function of depth:
# sns.barplot(x='depth', y='redcell', data=celldata[celldata['roi_name'].isin(['V1','PM'])], estimator=lambda y: sum(y==1)*100.0/len(y))
sns.lineplot(data=celldata[celldata['roi_name'].isin(['V1','PM'])],x='depth', y='redcell', estimator=lambda y: sum(y==1)*100.0/len(y))
# sns.lineplot(data=celldata,x='depth', y='redcell', hue='roi_name',estimator=lambda y: sum(y==1)*100.0/len(y),palette='Accent')
plt.ylabel('% labeled cells')

#Plot fraction of labeled cells across areas of recordings: 
sns.barplot(x='roi_name', y='redcell', data=celldata, estimator=lambda x: sum(x==1)*100.0/len(x),palette='Accent')
plt.ylabel('% labeled cells')

## plot number of cells per plane across depths:
sns.histplot(data=celldata, x='depth',hue='roi_name',palette='Accent')

## plot quality of cells per plane across depths with skew:
# sns.lineplot(data=celldata, x="depth",y=celldata['skew'],estimator='mean')
sns.lineplot(x=np.round(celldata["depth"],-1),y=celldata['skew'],estimator='mean')

## plot quality of cells per plane across depths with noise level:
# sns.lineplot(data=celldata, x="depth",y=celldata['noise_level'],estimator='mean')
sns.lineplot(x=np.round(celldata["depth"],-1),y=celldata['noise_level'],estimator='mean')
plt.ylim([0,0.3])

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


