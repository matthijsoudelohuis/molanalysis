# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% 
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.imagelib import load_natural_images #
from utils.tuning import *

# from utils.plot_lib import shaded_error

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\')

#%% ################################################
session_list        = np.array([['LPE11086','2023_12_16']])

#%% Load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'IM',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['IM']) 

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

# #%% ### Load the natural images:
# natimgdata = load_natural_images(onlyright=True)


#%% Compute tuning metrics:
for ses in sessions: 
    ses.respmean,imageids = mean_resp_image(ses)

# respmean,imageids = mean_resp_image(sessions[0])

#%% Compute tuning metrics of natural images:
for ses in tqdm(sessions,desc='Computing tuning metrics for each session'): 
    ses.celldata['tuning_SNR']                          = compute_tuning_SNR(ses)
    ses.celldata['corr_half'],ses.celldata['rel_half']  = compute_splithalf_reliability(ses)
    ses.celldata['sparseness']          = compute_sparseness(ses.respmat)
    ses.celldata['selectivity_index']   = compute_selectivity_index(ses.respmat)
    ses.celldata['fano_factor']         = compute_fano_factor(ses.respmat)
    ses.celldata['gini_coefficient']    = compute_gini_coefficient(ses.respmat)

#%% Figure of the distribution of tuning metric values:
for metric in ['tuning_SNR','corr_half','rel_half','sparseness','selectivity_index','fano_factor','gini_coefficient']: 
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    for ises,ses in enumerate(sessions):
        sns.histplot(ses.celldata[metric],ax=ax,element='step',fill=False,alpha=0.5,stat='count',bins=np.arange(-0.1,1.5,0.05),label='%s' %ses.sessiondata['session_id'][0])
    plt.xlim([-0.1,1.2])
    plt.legend(loc='upper right',frameon=False,fontsize=5)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'Distribution_%s_%d_IMsessions' % (metric,nSessions) + '.png'), format = 'png')

#%% Make a barplot 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Get information about cells per session per area: 
sesdata = pd.DataFrame()
sesdata['roi_name']         = celldata.groupby(["session_id","roi_name"])['roi_name'].unique()
sesdata['recombinase']      = celldata[celldata['recombinase'].isin(['cre','flp'])].groupby(["session_id","roi_name"])['recombinase'].unique()
sesdata = sesdata.applymap(lambda x: x[0],na_action='ignore')
sesdata['ncells']           = celldata.groupby(["session_id","roi_name"])['nredcells'].count()

for metric in ['tuning_SNR','corr_half','rel_half','sparseness','selectivity_index','fano_factor','gini_coefficient']:
    sesdata[metric] = celldata.groupby(["session_id","roi_name"])[metric].mean().values
sesdata.reset_index(drop=True,inplace=True)

#%% Plot the figure of mean tuning metrics per area:
areas = celldata['roi_name'].unique()
colors_areas = get_clr_areas(areas)

for metric in ['tuning_SNR','corr_half','rel_half','sparseness','selectivity_index','fano_factor','gini_coefficient']: 
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    sns.barplot(data=sesdata,x='roi_name',y=metric,hue='roi_name',palette=colors_areas,order=areas,hue_order=areas)
    sns.stripplot(data=sesdata,x='roi_name',y=metric,color='k',ax=ax,size=4,alpha=0.8,jitter=0.1,order=areas,hue_order=areas)
    ax.get_legend().remove()
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'Area_Mean_%s_%d_IMsessions' % (metric,nSessions) + '.png'), format = 'png')

#%% Get information about labeled cells per session per area: 
sesdata = pd.DataFrame()
sesdata['session_id']         = celldata.groupby(["session_id","roi_name","redcell"])['session_id'].unique()
sesdata['roi_name']         = celldata.groupby(["session_id","roi_name","redcell"])['roi_name'].unique()
sesdata['redcell']         = celldata.groupby(["session_id","roi_name","redcell"])['redcell'].unique()
sesdata = sesdata.applymap(lambda x: x[0],na_action='ignore')
# sesdata['ncells']           = celldata.groupby(["session_id","roi_name","redcell"])['nredcells'].count()

for metric in ['tuning_SNR','corr_half','rel_half','sparseness','selectivity_index','fano_factor','gini_coefficient']:
    sesdata[metric] = celldata.groupby(["session_id","roi_name","redcell"])[metric].mean().values
sesdata.reset_index(drop=True,inplace=True)

labeled = np.array(['UNL','LAB'])

sesdata['area_label'] = sesdata['roi_name'] + '_' + labeled[sesdata['redcell'].astype(int)]

#%% Make the figure per tuning metric:
area_labeled            = sesdata['area_label'].unique()
colors_area_labeled     = get_clr_area_labeled(area_labeled)

for metric in ['tuning_SNR','corr_half','rel_half','sparseness','gini_coefficient']: 
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    sns.barplot(data=sesdata,x='area_label',y=metric,palette=colors_area_labeled,
                order=area_labeled,hue_order=area_labeled,width=0.8,dodge=True)
    sns.stripplot(data=sesdata,x='area_label',y=metric,color='k',ax=ax,size=4,alpha=0.8,jitter=0.1,
                  order=area_labeled,hue_order=area_labeled,dodge=True,ec='k',linewidth=0.5)
    # ax.get_legend().remove()
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'Area_Labeled_Mean_%s_%d_IMsessions' % (metric,nSessions) + '.png'), format = 'png')



#Sort based on image number:
# arr1inds                = sessions[sesidx].trialdata['ImageNumber'][:2800].argsort()
# arr2inds                = sessions[sesidx].trialdata['ImageNumber'][2800:5600].argsort()

# respmat = sessions[sesidx].respmat[:,np.r_[arr1inds,arr2inds+2800]]
# # respmat_sort = sessions[sesidx].respmat_z[:,np.r_[arr1inds,arr2inds+2800]]

# from sklearn.preprocessing import normalize

# min_max_scaler = preprocessing.MinMaxScaler()
# respmat_sort = preprocessing.minmax_scale(respmat, feature_range=(0, 1), axis=0, copy=True)

# respmat_sort = normalize(respmat, 'l2', axis=1)

# fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# # axes[0].imshow(respmat_sort[:,:2800], aspect='auto',vmin=-100,vmax=200) 
# axes[0].imshow(respmat_sort[:,:2800], aspect='auto',vmin=np.percentile(respmat_sort,5),vmax=np.percentile(respmat_sort,95))
# axes[0].set_xlabel('Image #')
# axes[0].set_ylabel('Neuron')
# axes[0].set_title('Repetition 1')
# # axes[1].imshow(respmat_sort[:,2800:], aspect='auto',vmin=-100,vmax=200) 
# axes[1].imshow(respmat_sort[:,2800:], aspect='auto',vmin=np.percentile(respmat_sort,5),vmax=np.percentile(respmat_sort,95)) 
# axes[1].set_xlabel('Image #')
# axes[1].set_ylabel('Neuron')
# plt.tight_layout(rect=[0, 0, 1, 1])
# axes[1].set_title('Repetition 2')


#%% 



