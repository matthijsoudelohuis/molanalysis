# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import binned_statistic,binned_statistic_2d

from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error
# from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])

# load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GN','RF'],filter_areas=['V1','PM']) 

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% ########################### Compute trace correlations: ###################################
# sessions = compute_trace_correlation(sessions,binwidth=0.25,uppertriangular=False)

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% ##################### Compute pairwise neuronal distances: ##############################
# sessions = compute_pairwise_metrics(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ##################### Compute pairwise receptive field distances: ##############################
sessions = exclude_outlier_rf(sessions,radius=50,rf_thr=50) 
sessions = smooth_rf(sessions,radius=50)
# sessions = compute_pairwise_delta_rf(sessions,rf_type='F')
sessions = compute_pairwise_delta_rf(sessions,rf_type='F')

# np.save(os.path.join('e:\\Procdata\\','AllProtocols_corrdata_n84sessions.npy'),sessions,allow_pickle = True)
np.save(os.path.join('e:\\Procdata\\','GN_RF_SP_Protocols_corrdata_n%sSessions.npy' % nSessions),sessions,allow_pickle = True)

#%% ##########################################################################################################
# DELTA ANATOMICAL DISTANCE :
# ##########################################################################################################

#%% Define the areapairs:
areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

#%% Compute pairwise trace correlations as a function of pairwise anatomical distance ###################################################################

[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='trace_corr')

#%% Make the figure per protocol:

fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='trace_corr')
fig.savefig(os.path.join(savedir,'TraceCorr_XYZdist_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')

#%% Compute pairwise signal correlations as a function of pairwise anatomical distance ###################################################################
[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='sig_corr')

#%% Make the figure per protocol:
fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='sig_corr')
fig.savefig(os.path.join(savedir,'SignalCorr_XYZdist_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')

#%% Compute pairwise noise correlations as a function of pairwise anatomical distance ###################################################################
[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='noise_corr')

#%% Make the figure per protocol:
fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='noise_corr')
fig.savefig(os.path.join(savedir,'NoiseCorr_XYZdist_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')

# %% #######################################################################################################
# DELTA RECEPTIVE FIELD:
# ##########################################################################################################

#%% Show distribution of delta receptive fields across areas: 

sessions = compute_pairwise_delta_rf(sessions,rf_type='F')

#Make a figure with each session is one line for each of the areapairs a histogram of distmat_rf:
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

fig = plot_delta_rf_across_sessions(sessions,areapairs)
fig.savefig(os.path.join(savedir,'DeltaRF_Areapairs_%dsessions_' % nSessions + '.png'), format = 'png')


#%%
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                ['LPE11998','2024_05_02'], #GN
                                ['LPE12013','2024_05_02'], #GN
                                ['LPE12013','2024_05_07'], #GN
                                ['LPE11086','2023_12_15'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['session_id'].isin([x[0] + '_' + x[1] for x in session_list]))[0]
sessions_subset = [sessions[i] for i in sessions_in_list]

#%% ################ Pairwise trace correlations as a function of pairwise delta RF: #####################
areapairs           = ['V1-V1','PM-PM','V1-PM']
rf_type             = 'Fsmooth'
sessions            = compute_pairwise_delta_rf(sessions,rf_type=rf_type)

[binmean,binedges]  =  bin_corr_deltarf_areapairs(sessions_subset,areapairs,corr_type='trace_corr',normalize=False,
                                       sig_thr = 0.001,rf_type=rf_type)

#%% Make the figure:
fig = plot_bin_corr_deltarf_protocols(sessions_subset,binmean,binedges,areapairs,corr_type='trace_corr',normalize=False)

# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_Protocols_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_GN_SP_RF_750dF_F_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_GN_SP_RF_0.75dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'NoiseCorr_distRF_GN_SP_RF_0.75dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')

#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['protocol'].isin(['GR','GN','IM']))[0]
sessions_subset = [sessions[i] for i in sessions_in_list]

#%% #########################################################################################
# Contrast: across areas
areas   = ['V1','PM']

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap_areas(sessions_subset,corr_type='trace_corr',binresolution=7.5,
                                                                 rotate_prefori=False,thr_tuned=0.0,rf_type='F',
                                                                 thr_rf_p=0.001)

min_counts = 100
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(2,2,figsize=(10,7))
for i in range(2):
    for j in range(2):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],5),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(areas[i] + '-' + areas[j])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Interarea_AllProt_%dsessions' %nSessions  + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Interarea_RF_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(2,2,figsize=(10,7))
for i in range(2):
    for j in range(2):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),
                         cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(areas[i] + '-' + areas[j])
plt.tight_layout()

# fig,axes = plt.subplots(4,4,figsize=(10,7))
# for i in range(4):
#     for j in range(4):
#         axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
#         axes[i,j].set_title(legendlabels[i,j])
# plt.tight_layout()

#%% #########################################################################################
# Contrasts: across areas and projection identity      

[noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions_subset,corr_type='trace_corr',
                                                                binresolution=10,rotate_prefori=False,thr_tuned=0.0,
                                                                thr_rf_p=0.001,rf_type='F')

min_counts = 50
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_AllProt_%dsessions' %nSessions  + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_GN_F_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_Map_smooth_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')


#%% ##################### Noise correlations within and across areas: #########################

#%% ################## Noise correlations between labeled and unlabeled cells:  #########################

df = mean_corr_areas_labeling(sessions,corr_type='trace_corr',absolute=True,minNcells=10)

#%% Filter certain protocols:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['protocol'].isin(['GR','GN','IM','SP']))[0]
df = df.loc[sessions_in_list,:]

print('%d areapairs interpolated due to missing data' % np.sum(df.isna().sum(axis=1)/4))
df = df.fillna(df.mean())

#%% Make a barplot with error bars of the mean corr across sessions conditioned on area pairs:
fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se')
plt.plot(df.T,linewidth=0.25,c='k',alpha=0.5)	
sns.stripplot(data=df,palette='dark:k',ax=ax,size=3,alpha=0.5,jitter=0.1)
ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
ax.set_ylim([0,0.15])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'TraceCorr_labeling_areas_Indiv%dsessions' %nSessions + '.png'), format = 'png')

#%% With stats:
fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se')#,labels=legendlabels_upper_tri)
ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
pairs = [('V1unl-V1unl','V1lab-V1lab'),
         ('V1unl-V1unl','V1unl-V1lab'),
         ('V1unl-V1lab','V1lab-V1lab'),
         ('PMunl-PMunl','PMunl-PMlab'),
         ('PMunl-PMunl','PMlab-PMlab'),
         ('PMunl-PMlab','PMlab-PMlab'),
         ('V1unl-PMlab','V1lab-PMlab'),
         ('V1lab-PMunl','V1lab-PMlab'),
         ('V1unl-PMunl','V1lab-PMlab'),
         ] #for statistics

annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_width=1,line_offset_to_group = 0,text_offset=0)
annotator.apply_and_annotate()
ax.set_ylim([0,0.13])
ax.set_ylabel('Absolute Trace Correlation')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'TraceCorr_NearFilter_labeling_areas_%dsessions' %nSessions + '.png'), format = 'png')

# %% Now for signal correlations:
df = mean_corr_areas_labeling(sessions,corr_type='sig_corr',absolute=True,minNcells=10)

# #%% Filter certain protocols:
# sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list = np.where(sessiondata['protocol'].isin(['GR','GN','IM','SP']))[0]
# # sessions_in_list = np.where(sessiondata['protocol'].isin(['GN']))[0]
# df = df.loc[sessions_in_list,:]


#%% With stats:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

fig,ax = plt.subplots(figsize=(5,4))
for prot in ['GR','GN','IM']:
    sessions_in_list = np.where(sessiondata['protocol'].isin([prot]))[0]
    plt.plot(df.iloc[sessions_in_list,:].T,linewidth=0.5,c=get_clr_protocols([prot]),alpha=1)	

df2 = df.dropna() #% Drop missing data
sns.barplot(data=df2,estimator="mean",errorbar='se')#,labels=legendlabels_upper_tri)
ax.set_xticklabels(labels=df2.columns,rotation=90,fontsize=8)

annotator = Annotator(ax, pairs, data=df2,order=list(df.columns))
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_width=1,
                    line_height=0,line_offset_to_group=0.01,text_offset=0)
annotator.apply_and_annotate()
ax.set_ylim([0.15,0.37])
ax.set_ylim([0,0.45])
ax.set_ylabel('Absolute Signal Correlation')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'SigCorr_NearFilter_labeling_areas_%dsessions' %nSessions + '.png'), format = 'png')

#%% Same but now for noise correlations:
df = mean_corr_areas_labeling(sessions,corr_type='noise_corr',absolute=True,minNcells=10)

#%% Filter certain protocols:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['protocol'].isin(['GR','GN','IM','SP']))[0]
df = df.loc[sessions_in_list,:]

#%% Drop missing data:
df = df.dropna()


#%% With stats:
fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se')#,labels=legendlabels_upper_tri)
ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)

annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_width=1,
                    line_height=0,line_offset_to_group=0,text_offset=0)
annotator.apply_and_annotate()
ax.set_ylim([0,0.17])
ax.set_ylabel('Absolute Noise Correlation')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'NoiseCorr_NearFilter_labeling_areas_%dsessions' %nSessions + '.png'), format = 'png')

# %%
