# -*- coding: utf-8 -*-
"""
This script analyzes the quality of the recordings and their relation 
to various factors such as depth of recording, being labeled etc.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loaddata.session_info import filter_sessions,load_sessions
from statannotations.Annotator import Annotator

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn import preprocessing
from labeling.label_lib import reset_label_threshold
from utils.plotting_style import *

protocol            = ['VR']
# protocol            = ['GR']
protocol            = ['GR','VR','IM','RF','SP']
protocol            = ['GR','VR','IM']
protocol            = ['GR','GN','IM']

# sessions            = filter_sessions(protocol,only_animal_id=['LPE09830','LPE09665'])
sessions,nsessions    = filter_sessions(protocol,only_animal_id=['LPE10919','LPE10885','LPE10883','LPE11086'],
                                      min_cells=100)

# session_list        = np.array([['LPE11086','2023_12_16']])
# sessions            = load_sessions(protocol = 'IM',session_list=session_list)

savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Labeling\\'

############## ############
## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

## remove any double cells (for example recorded in both GR and RF)
celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

threshold = 0.4
sessions = reset_label_threshold(sessions,threshold)

celldata.loc[celldata['redcell']==0,'recombinase'] = 'non'

celldata['noise_level'].values[celldata['noise_level'] > 5] = 0

######## Show histogram of ROI overlaps: #######################
fig, ax = plt.subplots(figsize=(3.5,3))
sns.histplot(data=celldata,x='redcell_prob',stat='probability',hue='redcell',
             palette=get_clr_labeled(),binwidth=0.05,ax=ax)
ax.get_legend().remove()

plt.xlim([0,1])
plt.axvline(threshold,color='grey',linestyle=':')
plt.xlabel('ROI Overlap')
plt.ylabel('Fraction of cells')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Overlap_Dist_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')
plt.ylim([0,0.01])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Overlap_Dist_Zoom_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')


#### 
# Get the number of labeled cells, cre / flp, depth, area etc. for each plane:

planedata = pd.DataFrame()

planedata['depth']  = celldata.groupby(["session_id","plane_idx"])['depth'].unique()
planedata['roi_name']  = celldata.groupby(["session_id","plane_idx"])['roi_name'].unique()
planedata['recombinase']  = celldata[celldata['recombinase'].isin(['cre','flp'])].groupby(["session_id","plane_idx"])['recombinase'].unique()

planedata = planedata.applymap(lambda x: x[0],na_action='ignore')

planedata['frac_labeled']  = celldata.groupby(["session_id","plane_idx"])['redcell'].sum() / celldata.groupby(["session_id","plane_idx"])['redcell'].count()

clrs_areas = get_clr_areas(['V1','PM'])

fig, ax = plt.subplots(figsize=(4,3))
sns.scatterplot(data=planedata,x='depth',y='frac_labeled',hue='roi_name',palette=clrs_areas,ax=ax,s=12)
plt.ylabel('Fraction labeled in plane')
plt.xlabel(r'Cortical depth ($\mu$m)')
plt.xlim([50,500])
plt.tight_layout()
sns.lineplot(x=planedata['depth'].round(-2),y=planedata['frac_labeled'],
             hue=planedata['roi_name'],palette=clrs_areas,ax=ax)
plt.legend(['V1','PM'])
plt.savefig(os.path.join(savedir,'Frac_labeled_depth_area_%dplanes' % len(planedata) + '.png'), format = 'png')


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

labelcounts = celldata.groupby(['recombinase'])['recombinase'].count()
plt.suptitle('Quality comparison non-labeled ({0}), cre-labeled ({1}) and flp-labeled ({2}) cells'.format(
    labelcounts[labelcounts.index=='non'][0],labelcounts[labelcounts.index=='cre'][0],labelcounts[labelcounts.index=='flp'][0]))
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Quality_Metrics_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

###################### ###################### ######################
## Scatter of all crosscombinations (seaborn pairplot):
df = celldata[["depth","skew","noise_level","npix_soma",
               "meanF","meanF_chan2","event_rate","redcell"]]
sns.pairplot(data=df, hue="redcell")

ax = sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap='bwr')
plt.savefig(os.path.join(savedir,'Quality_Metrics_Heatmap_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

## 


# ###################### Noise level for labeled vs unlabeled cells:

# ## plot precentage of labeled cells as a function of depth:
# # sns.barplot(x='depth', y='redcell', data=celldata[celldata['roi_name'].isin(['V1','PM'])], estimator=lambda y: sum(y==1)*100.0/len(y))
# sns.lineplot(data=celldata[celldata['roi_name'].isin(['V1','PM'])],x='depth', y='redcell', estimator=lambda y: sum(y==1)*100.0/len(y))
# # sns.lineplot(data=celldata,x='depth', y='redcell', hue='roi_name',estimator=lambda y: sum(y==1)*100.0/len(y),palette='Accent')
# plt.ylabel('% labeled cells')

# #Plot fraction of labeled cells across areas of recordings: 
# sns.barplot(x='roi_name', y='redcell', data=celldata, estimator=lambda x: sum(x==1)*100.0/len(x),palette='Accent')
# plt.ylabel('% labeled cells')

# ## plot number of cells per plane across depths:
# sns.histplot(data=celldata, x='depth',hue='roi_name',palette='Accent')

# ## plot quality of cells per plane across depths with skew:
# # sns.lineplot(data=celldata, x="depth",y=celldata['skew'],estimator='mean')
# sns.lineplot(x=np.round(celldata["depth"],-1),y=celldata['skew'],estimator='mean')

# ## plot quality of cells per plane across depths with noise level:
# # sns.lineplot(data=celldata, x="depth",y=celldata['noise_level'],estimator='mean')
# sns.lineplot(x=np.round(celldata["depth"],-1),y=celldata['noise_level'],estimator='mean')
# plt.ylim([0,0.3])

