# -*- coding: utf-8 -*-
"""
This script analyzes the quality of the recordings and their relation 
to various factors such as depth of recording, being labeled etc.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""
#%% Import the libraries and functions
import os
os.chdir('e:\\Python\\molanalysis')

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
from utils.psth import compute_tensor,compute_respmat

#%% Load the data from all passive protocols:
protocols            = ['GR','GN','IM']
protocols            = ['DN']

sessions,nsessions            = filter_sessions(protocols)

# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE11086','2024_01_05']])
# sessions,nsessions  = load_sessions(protocol = 'GR',session_list=session_list)

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Labeling\\'
#%% #### reset threshold if necessary:
threshold = 0.5
for ses in sessions:
    ses.reset_label_threshold(threshold)

#%% ######## ############
## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

## remove any double cells (for example recorded in both GR and RF)
celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

celldata.loc[celldata['redcell']==0,'recombinase'] = 'non'

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

#%% ####### Show scatter of the two overlap metrics: frac red in ROI and frac of ROI red #######################
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

#%% Find some cells that have overlap of red in ROI, but low ROI is red:
idx = np.logical_and(celldata['frac_red_in_ROI']>0.9,celldata['frac_of_ROI_red']<0.4)
celldata['cell_id'][idx] 

#%% Find some cells that have full tdtomato within, but only little of the suite2p ROI:
idx = np.where((celldata['frac_of_ROI_red']>0.9) & (celldata['frac_red_in_ROI']<0.5))[0]
celldata['cell_id'][idx] 

#%% Find some cells that have full tdtomato within, but only little of the suite2p ROI:
idx = np.where((celldata['frac_of_ROI_red']<0.4) & (celldata['frac_red_in_ROI']>0.9))[0]
celldata['cell_id'][idx] 

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
idx = np.where((celldata['frac_red_in_ROI']<0.5) & (celldata['chan2_prob']>0.7) 
               & (celldata['chan2_prob']<1) & (celldata['skew']>3) & 
               (celldata['session_id'] != 'LPE10885_2023_10_12') & 
               (celldata['session_id'] != 'LPE10883_2023_10_23') & 
               (celldata['session_id'] != 'LPE10919_2023_11_06'))[0]
# idx = np.logical_and(celldata['frac_red_in_ROI']<0.5,celldata['chan2_prob']>0.8)
g = celldata['cell_id'][idx] 
g[:25]
g[45:75]

idx = celldata['cell_id']=='LPE11998_2024_05_10_1_0132'
idx = celldata['cell_id']=='LPE11086_2024_01_10_2_0094'
idx = celldata['cell_id']=='LPE12013_2024_05_07_0_0177'
idx = celldata['cell_id']=='LPE11495_2024_02_27_5_0028'
celldata['chan2_prob'][idx] 
celldata['frac_red_in_ROI'][idx]

celldata['cell_id'][1341]
celldata['frac_red_in_ROI'][1341]

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
enzymelabels = ['retroAAV-pgk-Cre + \n AAV5-CAG-Flex-tdTomato','retroAAV-EF1a-Flpo + \n AAV1-Ef1a-fDIO-tdTomato']

fig, ax = plt.subplots(figsize=(4,3))
sns.barplot(data=planedata,x='recombinase',y='frac_labeled',palette=clrs_enzymes,ax=ax,errorbar='se')
plt.ylabel('Fraction labeled in plane')
plt.xlabel(r'Recombinase')
ax.set_xticks([0,1])
ax.set_xticklabels(['\n\nCre','\n\nFlp'], fontsize=11)
ax.set_xticks([0.01,1.01],  minor=True)
ax.set_xticklabels(enzymelabels,fontsize=6, minor=True)

# ax.set_xticklabels(enzymelabels,fontsize=6)
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
# celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

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


#%% 


#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM']) 

session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_23']])
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

#%% Remove two sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
# sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE09665_2023_03_21']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%% #############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

for ises in range(nSessions):
    sessions[ises].load_data(load_calciumdata=True,calciumversion='deconv')
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post,method='nearby')
    sessions[ises].respmat = np.mean(sessions[ises].tensor[:,:,np.logical_and(t_axis>-0, t_axis<=1)] ,axis=2)

#%% 
from utils.explorefigs import plot_excerpt,plot_PCA_gratings,plot_tuned_response

#%% Show some tuned responses with calcium and deconvolved traces across orientations:
example_cells = [3,100,58,62,70]
fig = plot_tuned_response(sessions[0].tensor,sessions[0].trialdata,t_axis,example_cells)
fig.suptitle('%s - Deconvolved' % sessions[0].sessiondata['session_id'][0],fontsize=12)
# save the figure
# fig.savefig(os.path.join(savedir,'TunedResponse_deconv_%s.png' % sessions[0].sessiondata['session_id']))

#%% Figure of complete average response for dF/F and deconv:
from utils.tuning import compute_tuning, compute_prefori
from utils.plot_lib import shaded_error,my_ceil,my_floor,get_sig_asterisks
from scipy.stats import ttest_ind

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'])
    sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='tuning_var')

#%% 
tensor_avgall = np.empty((0,len(t_axis)))
respmat_avgall = np.empty((0))

tensor_avgpref = np.empty((0,len(t_axis)))
respmat_avgpref = np.empty((0))
for ises in range(nSessions):
    tensor_avgall = np.concatenate((tensor_avgall,np.mean(sessions[ises].tensor,axis=1)))
    
    temp = np.mean(sessions[ises].respmat,axis=1)
    respmat_avgall  = np.concatenate((respmat_avgall,temp))

    for iN in range(len(sessions[ises].celldata)):
        trialidx = sessions[ises].trialdata['Orientation'] == sessions[ises].celldata['pref_ori'][iN]
        temp = np.mean(sessions[ises].tensor[iN,trialidx,:],axis=0)
        tensor_avgpref = np.concatenate((tensor_avgpref,temp[np.newaxis,:]))

        temp = np.mean(sessions[ises].respmat[iN,trialidx],axis=0)
        respmat_avgpref  = np.concatenate((respmat_avgpref,[temp]))

#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# tensor_avgpref = tensor_avgpref - np.nanmean(tensor_avgpref[:,t_axis<0],axis=1,keepdims=True)

areas = ['V1','PM']
clrs_areas = get_clr_areas(areas)
projs = ['unl','lab']
clrs_projs = get_clr_labeled()

fig,axes = plt.subplots(1,2,figsize=(5.5,2.5),sharey=True)
for iarea,area in enumerate(areas):
    ax = axes[iarea]
    handles = []
    for iproj,proj in enumerate(projs):
        # idx = np.logical_and(celldata['roi_name']==area,celldata['labeled']==proj)
        # idx = np.all((celldata['roi_name']==area,celldata['labeled']==proj,celldata['tuning_var']>0.10),axis=0)
        idx = np.all((celldata['roi_name']==area,celldata['labeled']==proj),axis=0)
        handles.append(shaded_error(x=t_axis,y=tensor_avgpref[idx,:],error='sem',color=clrs_projs[iproj],ax=ax))
    ax.set_title(area,color=clrs_areas[iarea])
    ax.plot([0,1],[10,10],color='black',linewidth=3)
    if iarea==0:
        ax.set_ylabel('Deconvolved Activity')
    ax.set_xlabel('Time (s)')
    # ax.set_ylim([0,210])
    xdata = respmat_avgpref[np.logical_and(celldata['roi_name']==area,celldata['labeled']==projs[0])]
    ydata = respmat_avgpref[np.logical_and(celldata['roi_name']==area,celldata['labeled']==projs[1])]
    t_stat,p_val = ttest_ind(xdata,ydata)
    cohens_d = np.abs(np.mean(xdata)-np.mean(ydata))/np.sqrt((np.var(xdata)+np.var(ydata))/2)
    ax.text(0.5,0.08,'%s p: %.3f' % (get_sig_asterisks(p_val),p_val),ha='center',transform=ax.transAxes,fontsize=8)
    ax.legend(handles=handles,labels=projs,frameon=False,fontsize=8)
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'CalciumTracesComparison','Resp_avgall_deconv' + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'CalciumTracesComparison','Resp_avgpref_deconv' + '.png'), format = 'png')

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

