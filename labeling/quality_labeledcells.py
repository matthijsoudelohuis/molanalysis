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
from utils.plotting_style import *
from utils.rf_lib import filter_nearlabeled

#%% Load the data from all passive protocols:
protocols            = ['GR','GN','IM']

sessions,nsessions            = filter_sessions(protocols)

# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE11086','2024_01_05']])
# sessions,nsessions  = load_sessions(protocol = 'GR',session_list=session_list)

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Labeling\\'

#%% ############ ############
## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

## remove any double cells (for example recorded in both GR and RF)
celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

threshold = 0.5
sessions = reset_label_threshold(sessions,threshold)
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

celldata.loc[celldata['redcell']==0,'recombinase'] = 'non'

# celldata['noise_level'].values[celldata['noise_level'] > 5] = 0

#%% ####### Show histogram of ROI overlaps: #######################
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(3.5,3),sharex=True)

sns.histplot(data=celldata,x='frac_red_in_ROI',stat='probability',hue='redcell',
             palette=get_clr_labeled(),binwidth=0.05,ax=ax1)
             
sns.histplot(data=celldata,x='frac_red_in_ROI',stat='probability',hue='redcell',
             palette=get_clr_labeled(),binwidth=0.05,ax=ax2)
fig.subplots_adjust(hspace=0.05)

ax2.get_legend().remove()

ax1.set_xlim([0,1])
ax1.set_ylim([0.8,1])
ax2.set_ylim([0,0.02])

ax1.axvline(threshold,color='grey',linestyle=':')
ax2.axvline(threshold,color='grey',linestyle=':')

ax1.set_xlabel('ROI Overlap')
ax1.set_ylabel('Fraction of cells')
ax2.set_ylabel('')
plt.tight_layout()
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

d = 0.5
kwargs = dict(marker=[(-1,-d),(1,d)],markersize=12,linestyle="none",color='k',mec='k',mew=1,clip_on=False)
ax1.plot([0,1],[0,0],transform=ax1.transAxes,**kwargs)
ax2.plot([0,1],[1,1],transform=ax2.transAxes,**kwargs)

plt.tight_layout()
plt.savefig(os.path.join(savedir,'Overlap_Dist_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

#%% ####### Show scatter of chan2prob from suite2p and frac red in ROI #######################
fig, ax = plt.subplots(figsize=(3.5,3))
sns.scatterplot(data=celldata,x='frac_red_in_ROI',y='frac_of_ROI_red',hue='redcell',ax=ax,
                palette=get_clr_labeled(),s=5)
ax.get_legend().remove()

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('frac_red_in_ROI')
plt.ylabel('frac_of_ROI_red')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Scatter_Overlap_Twoways_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

#%% ####### Show scatter of chan2prob from suite2p and frac red in ROI #######################
fig, ax = plt.subplots(figsize=(3.5,3))
sns.scatterplot(data=celldata,x='frac_red_in_ROI',y='chan2_prob',hue='redcell',ax=ax,
                palette=get_clr_labeled(),s=5)
ax.get_legend().remove()

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('ROI Overlap')
plt.ylabel('Channel 2 probability')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Scatter_Overlap_Chan2Prob_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

#%% Find some cells that should be labeled according to the metric of suite2p, but not through cellpose:
idx = np.logical_and(celldata['frac_red_in_ROI']<0.1,celldata['chan2_prob']>0.9)
celldata['cell_id'][idx] 

#%% Get the colors and names of the areas:
areas = celldata['roi_name'].unique()
clrs_areas = get_clr_areas(areas)

#%% Get information about labeled cells per session per area: 
sesdata = pd.DataFrame()
sesdata['roi_name']         = celldata.groupby(["session_id","roi_name"])['roi_name'].unique()
sesdata['recombinase']      = celldata[celldata['recombinase'].isin(['cre','flp'])].groupby(["session_id","roi_name"])['recombinase'].unique()
sesdata = sesdata.applymap(lambda x: x[0],na_action='ignore')
sesdata['ncells']           = celldata.groupby(["session_id","roi_name"])['nredcells'].count()
sesdata['nredcells']        = celldata.groupby(["session_id","roi_name"])['nredcells'].unique().apply(sum)
sesdata['nlabeled']         = celldata.groupby(["session_id","roi_name"])['redcell'].sum()
sesdata['frac_responsive']  = sesdata['nlabeled'] / sesdata['nredcells'] 
sesdata['frac_labeled']     = sesdata['nlabeled'] / sesdata['ncells'] 

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='nredcells',palette=clrs_areas,ax=ax,errorbar='se')
sns.stripplot(data=sesdata,x='roi_name',y='nredcells',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2)
plt.title('# cellpose cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'nCellpose_area_%dsessions' % len(sesdata) + '.png'), format = 'png')

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='ncells',palette=clrs_areas,ax=ax,errorbar='se')
sns.stripplot(data=sesdata,x='roi_name',y='ncells',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2)
plt.title('# suite2p cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'nSuite2p_area_%dsessions' % len(sesdata) + '.png'), format = 'png')

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='frac_responsive',palette=clrs_areas,ax=ax,errorbar='se')
sns.stripplot(data=sesdata,x='roi_name',y='frac_responsive',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2)
plt.title('# Frac. responsive cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Frac_responsive_area_%dsessions' % len(sesdata) + '.png'), format = 'png')

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='frac_labeled',palette=clrs_areas,ax=ax,errorbar='se')
sns.stripplot(data=sesdata,x='roi_name',y='frac_labeled',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2)
plt.title('# Frac. labeled cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Frac_labeled_area_%dsessions' % len(sesdata) + '.png'), format = 'png')

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='nlabeled',palette=clrs_areas,ax=ax,errorbar='se')
sns.stripplot(data=sesdata,x='roi_name',y='nlabeled',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2)
plt.title('# Labeled cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'nLabeled_area_%dsessions' % len(sesdata) + '.png'), format = 'png')

#%% ### Get the number of labeled cells, cre / flp, depth, area etc. for each plane :
planedata = pd.DataFrame()
planedata['depth']          = celldata.groupby(["session_id","plane_idx"])['depth'].unique()
planedata['roi_name']       = celldata.groupby(["session_id","plane_idx"])['roi_name'].unique()
planedata['recombinase']    = celldata[celldata['recombinase'].isin(['cre','flp'])].groupby(["session_id","plane_idx"])['recombinase'].unique()
planedata = planedata.applymap(lambda x: x[0],na_action='ignore')
planedata['ncells']         = celldata.groupby(["session_id","plane_idx"])['depth'].count()
planedata['nlabeled']       = celldata.groupby(["session_id","plane_idx"])['redcell'].sum()
planedata['frac_labeled']   = celldata.groupby(["session_id","plane_idx"])['redcell'].sum() / celldata.groupby(["session_id","plane_idx"])['redcell'].count()
planedata['nredcells']      = celldata.groupby(["session_id","plane_idx"])['nredcells'].mean().astype(int)
planedata['frac_responsive']  = celldata.groupby(["session_id","plane_idx"])['redcell'].sum() / planedata['nredcells'] 

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=planedata,x='roi_name',y='frac_labeled',palette=clrs_areas,ax=ax,errorbar='se')
sns.stripplot(data=planedata,x='roi_name',y='frac_labeled',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2)
plt.ylabel('Fraction labeled in plane')
plt.xlabel(r'Area')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Frac_labeled_area_%dplanes' % len(planedata) + '.png'), format = 'png')

#%% Bar plot of difference between cre and flp:
enzymes = ['cre','flp']
clrs_enzymes = get_clr_recombinase(enzymes)

fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=planedata,x='recombinase',y='frac_labeled',palette=clrs_enzymes,ax=ax,errorbar='se')
plt.ylabel('Fraction labeled in plane')
plt.xlabel(r'Recombinase')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Frac_labeled_enzymes_%dplanes' % len(planedata) + '.png'), format = 'png')

#%% Scatter plot as a function of depth:
fig, ax = plt.subplots(figsize=(5,4))
sns.scatterplot(data=planedata,x='depth',y='frac_labeled',hue='roi_name',palette=clrs_areas,ax=ax,s=14)
plt.ylabel('Fraction labeled in plane')
plt.xlabel(r'Cortical depth ($\mu$m)')
plt.xlim([50,500])
plt.tight_layout()
sns.lineplot(x=planedata['depth'].round(-2),y=planedata['frac_labeled'],
             hue=planedata['roi_name'],palette=clrs_areas,ax=ax)
plt.legend(ax.get_legend_handles_labels()[0][:4],areas, loc='best')
plt.savefig(os.path.join(savedir,'Frac_labeled_depth_area_%dplanes' % len(planedata) + '.png'), format = 'png')

#%% Number of red cellpose cells as a function of depth (not per se suite2p calcium trace detected):
fig, ax = plt.subplots(figsize=(5,4))
sns.scatterplot(data=planedata,x='depth',y='nredcells',hue='roi_name',palette=clrs_areas,ax=ax,s=14)
plt.ylabel('Number labeled in plane')
plt.xlabel(r'Cortical depth ($\mu$m)')
plt.xlim([50,500])
plt.tight_layout()
sns.lineplot(x=planedata['depth'].round(-2),y=planedata['nredcells'],
             hue=planedata['roi_name'],palette=clrs_areas,ax=ax)
plt.legend(ax.get_legend_handles_labels()[0][:4],areas, loc='best')
plt.savefig(os.path.join(savedir,'Frac_cellpose_depth_area_%dplanes' % len(planedata) + '.png'), format = 'png')

#%% Select only cells nearby labeled cells to ensure fair comparison of quality metrics:

celldata = pd.concat([ses.celldata[filter_nearlabeled(ses,radius=50)] for ses in sessions]).reset_index(drop=True)
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% ##################### Calcium trace skewness for labeled vs unlabeled cells:
# order = [0,1] #for statistical testing purposes
# pairs = [(0,1)]

order = ['non','flp','cre'] #for statistical testing purposes
pairs = [('non','flp'),('non','cre')]

# fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF","meanF_chan2"]
fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF"]

nfields = len(fields)
fig,axes   = plt.subplots(1,nfields,figsize=(12,4))

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

labelcounts = celldata.groupby(['recombinase'])['recombinase'].count()
plt.suptitle('Quality comparison non-labeled ({0}), cre-labeled ({1}) and flp-labeled ({2}) cells'.format(
    labelcounts[labelcounts.index=='non'][0],labelcounts[labelcounts.index=='cre'][0],labelcounts[labelcounts.index=='flp'][0]))
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'Quality_Metrics_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'Quality_Metrics_%dnearbycells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

#%% ##################### ###################### ######################
## Scatter of all crosscombinations (seaborn pairplot):
df = celldata[["depth","skew","noise_level","npix_soma",
               "meanF","meanF_chan2","event_rate","redcell"]]
# sns.pairplot(data=df, hue="redcell")

ax = sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap='bwr')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Quality_Metrics_Heatmap_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')


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

