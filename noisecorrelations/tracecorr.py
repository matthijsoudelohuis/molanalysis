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
from scipy.signal import detrend
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')
plt.rcParams['axes.spines.right']   = False
plt.rcParams['axes.spines.top']     = False

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%% Load all sessions from certain protocols: 
sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GN','RF'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['IM'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM']) 

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)
                                # calciumversion='dF',keepraw=True)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False,filtersig=True)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=True)

#%% ##################### Compute pairwise neuronal distances: ##############################
# sessions = compute_pairwise_metrics(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ##################### Compute pairwise receptive field distances: ##############################
sessions = smooth_rf(sessions,radius=75,rf_type='Favg')
sessions = exclude_outlier_rf(sessions) 
sessions = replace_smoothed_rf(sessions) 
# sessions = compute_pairwise_delta_rf(sessions,rf_type='Fsmooth')

# np.save(os.path.join('e:\\Procdata\\','AllProtocols_corrdata_n84sessions.npy'),sessions,allow_pickle = True)
# np.load(os.path.join('e:\\Procdata\\','AllProtocols_corrdata_n60sessions.npy'),allow_pickle = False)
np.save(os.path.join('e:\\Procdata\\','GN_RF_SP_Protocols_corrdata_n%sSessions.npy' % nSessions),sessions,allow_pickle = True)

#%% ##########################################################################################################
# DELTA ANATOMICAL DISTANCE :
# ##########################################################################################################

#%% Define the areapairs:
areapairs       = ['V1-V1','PM-PM']
clrs_areapairs  = get_clr_area_pairs(areapairs)

#%% Compute pairwise correlations as a function of pairwise anatomical distance ###################################################################
for corr_type in ['trace_corr','sig_corr','noise_corr']:
    [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type=corr_type)

    #Make the figure per protocol:
    fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type=corr_type)
    fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.pdf'), format = 'pdf')

# #%% Compute pairwise trace correlations as a function of pairwise anatomical distance ###################################################################
# [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='trace_corr')

#%% ########################################################################################################
# ##################### Noise correlations within and across areas: ########################################
# ##########################################################################################################

dfses = mean_corr_areas_labeling([sessions[0]],corr_type='trace_corr',absolute=True,minNcells=100)[0]
clrs_area_labelpairs = get_clr_area_labelpairs(list(dfses.columns))

pairs = [('V1unl-V1unl','V1lab-V1lab'),
         ('V1unl-V1unl','V1unl-V1lab'),
         ('V1unl-V1lab','V1lab-V1lab'),
         ('PMunl-PMunl','PMunl-PMlab'),
         ('PMunl-PMunl','PMlab-PMlab'),
         ('PMunl-PMlab','PMlab-PMlab'),
         ('V1unl-PMunl','V1lab-PMunl'),
         ('V1unl-PMunl','V1unl-PMlab'),
         ('V1unl-PMunl','V1lab-PMlab'),
         ('V1unl-PMlab','V1lab-PMunl'),
         ('V1unl-PMlab','V1lab-PMlab'),
         ('V1lab-PMunl','V1lab-PMlab'),
         ] #for statistics

#%% ################## Noise correlations between labeled and unlabeled cells:  #########################
df = mean_corr_areas_labeling(sessions,corr_type='trace_corr',absolute=False,filternear=True,minNcells=10)

#%% Make a barplot with error bars of the mean corr across sessions conditioned on area pairs:
fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
plt.plot(df.T,linewidth=0.25,c='k',alpha=0.5)	
sns.stripplot(data=df,palette='dark:k',ax=ax,size=3,alpha=0.5,jitter=0.1)
ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
ax.set_ylabel('Correlation')
# ax.set_ylim([0,0.15])
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','TraceCorr_labeling_areas_Indiv%dsessions' %nSessions + '.png'), format = 'png')

#%% With stats:
fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
# ax.invert_yaxis()
annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                    line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False)
annotator.apply_and_annotate()
# ax.invert_yaxis()
# ax.set_ylim([0,0.13])
ax.set_ylabel('Absolute Trace Correlation')
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','TraceCorr_NearFilter_labeling_areas_%dsessions' %nSessions + '.png'), format = 'png')


#%% Plot mean absolute correlation across sessions conditioned on area pairs and per protocol:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharex=True,sharey='row')
    for iprot,prot in enumerate(['GR','GN','IM']):
        ax                  = axes[iprot]
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
        df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,absolute=True,filternear=True,minNcells=10)
        df                  = df_mean
        df                  = df.fillna(df.mean()) #interpolate occasional missing data
        
        if df.any(axis=None):
            sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
            ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
            annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
            annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                                line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                                correction_format='replace')
            annotator.apply_and_annotate()
            ax.set_ylabel('Correlation')
            ax.set_title('%s' %(prot),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

#%% Plot mean correlation across sessions conditioned on area pairs and per protocol:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,3,figsize=(12,6),sharex=True,sharey='row')
    for iprot,prot in enumerate(['GR','GN','IM']):
        for isign,sign in enumerate(['pos','neg']):
            ax                  = axes[isign,iprot]
            ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
            df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=True,minNcells=10)
            df                  = df_mean
            df                  = df.fillna(df.mean()) #interpolate occasional missing data
            
            if df.any(axis=None):
                sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
                if isign==1:
                    ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
                else: ax.set_xticks([])
                if isign==1: 
                    ax.invert_yaxis()
                annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
                annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                                    line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                                    correction_format='replace')
                annotator.apply_and_annotate()
                if isign==1:
                    ax.invert_yaxis()
                ax.set_ylabel('%s correlation' % sign)
                ax.set_title('%s' %(prot),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

#%% Plot fraction of correlated units across sessions conditioned on area pairs and per protocol:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,3,figsize=(12,6),sharex=True,sharey='row')
    for iprot,prot in enumerate(['GR','GN','IM']):
        for isign,sign in enumerate(['pos','neg']):
            ax                  = axes[isign,iprot]
            ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
            df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=True,minNcells=10)
            df                  = df_frac
            df                  = df.fillna(df.mean()) #interpolate occasional missing data
            
            if df.any(axis=None):
                sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
                if isign==1:
                    ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
                else: ax.set_xticks([])

                annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
                annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                                    line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                                    correction_format='replace')
                annotator.apply_and_annotate()
                ax.set_ylabel('Fraction of %s correlated units' % sign)
                ax.set_title('%s' %(prot),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

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
rf_type             = 'F'
sessions            = compute_pairwise_delta_rf(sessions,rf_type=rf_type)

[binmean,binedges]  =  bin_corr_deltarf_areapairs(sessions,areapairs,corr_type='trace_corr',normalize=False,
                                       sig_thr = 0.001,rf_type=rf_type)

#%% Make the figure:
fig = plot_bin_corr_deltarf_protocols(sessions,binmean,binedges,areapairs,corr_type='trace_corr',normalize=False)

# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_IM_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_GN_SP_RF_750dF_F_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_GN_SP_RF_0.75dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'NoiseCorr_distRF_IM_GN_SP_RF_0.5dF_Fneu_F0.001_%dsessions_' %nSessions + '.png'), format = 'png')

#%% Give redcells a string label

redcelllabels = np.array(['unl','lab'])
for ses in sessions:
    ses.celldata['labeled'] = ses.celldata['redcell']
    ses.celldata['labeled'] = ses.celldata['labeled'].astype(int).apply(lambda x: redcelllabels[x])

#%% 
rf_type             = 'Fsmooth'
sessions            = compute_pairwise_delta_rf(sessions,rf_type=rf_type)

#%% 
# protocols = ['IM']
protocols = ['SP']
# protocols = ['GR']

sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['protocol'].isin(protocols))[0]
sessions_subset = [sessions[i] for i in sessions_in_list]

#%% ################ Pairwise correlations as a function of pairwise delta RF: #####################
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

#If you override any of these then these pairs will be ignored:
layerpairs          = ' '
# areapairs           = ' '
# projpairs           = ' '
absolute            = False
corr_type           = 'trace_corr'

[binmean,binedges]  =  bin_corr_deltarf(sessions_subset,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs,
                                        corr_type=corr_type,normalize=False,binres=5,
                                       sig_thr = 0.001,rf_type=rf_type,mincount=10,absolute=absolute)

fig = plot_bin_corr_deltarf_flex(sessions_subset,binmean,binedges,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs,corr_type='trace_corr',normalize=False)

fig.savefig(os.path.join(savedir,'Corr_1d_arealabel_%s_%s_abs%s' % (protocols[0],corr_type,absolute) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_GN_SP_RF_0.75dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_Areas_Layers_IM_0.5dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')

#%% ##########################################################################################################
#   CENTER VERSUS SURROUND 
# ##########################################################################################################

#%% ################ Pairwise correlations as a function of pairwise delta RF: #####################
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

#If you override any of these then these pairs will be ignored:
# layerpairs          = ' '
# areapairs           = ' '
projpairs           = ' '

binres = 5
# binres = 'centersurround'

[binmean,binpos]  =  bin_corr_deltarf(sessions_subset,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs,
                                        corr_type='trace_corr',normalize=False,binres=binres,
                                       sig_thr = 0.001,rf_type=rf_type,mincount=10,absolute=False)

fig = plot_bin_corr_deltarf_flex(sessions_subset,binmean,binpos,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs,corr_type='trace_corr',normalize=False)

#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################
protocols = ['GR','GN']
# protocols = ['SP']
# protocols = ['IM']

sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['protocol'].isin(protocols))[0]
sessions_subset = [sessions[i] for i in sessions_in_list]

#%% 
rf_type             = 'F'
rf_type             = 'Fsmooth'
sessions            = compute_pairwise_delta_rf(sessions,rf_type=rf_type)

#%% #########################################################################################
# Contrast: across areas
# areas               = ['V1','PM']

areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

clrs_areapairs      = get_clr_area_pairs(areapairs)
clrs_projpairs      = get_clr_labelpairs(projpairs)
clrs_layerpairs     = get_clr_layerpairs(layerpairs)

#If you override any of these then these pairs will be ignored:
layerpairs          = ' '
# areapairs           = ' '
# projpairs           = ' '

# deltaori            = [-45,45]
deltaori            = None
rotate_prefori      = False
binres              = 5

[binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type='trace_corr',binresolution=binres,rotate_prefori=rotate_prefori,deltaori=deltaori,rf_type=rf_type,
                            sig_thr = 0.001)

#%%
binmean = np.nanmean(binmean_ses,axis=5)
bincounts = np.nansum(bincounts_ses,axis=5)

binmean = binmean_ses
bincounts = bincounts_ses

#%% Make the figure:
delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
deltarf             = np.sqrt(delta_az**2 + delta_el**2)
min_counts          = 50

#%% Make the figure:
fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax                                              = axes[ipp,iap]
            data                                            = binmean[:,:,iap,ilp,ipp].copy()
            data[bincounts[:,:,iap,ilp,ipp]<min_counts]     = np.nan
            ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95),cmap="hot")
            # ax.imshow(binmean[:,:,iap,ilp,ipp],vmin=np.nanpercentile(binmean[:,:,iap,ilp,ipp],5),
            #                     vmax=np.nanpercentile(binmean[:,:,iap,ilp,ipp],99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([-50,50])
            ax.set_ylim([-50,50])
            ax.set_xlabel(u'Δ Azimuth')
            ax.set_ylabel(u'Δ Elevation')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_%s_proj' % (corr_type,protocols[0]) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_%s_proj' % (corr_type,protocols[0]) + '.pdf'), format = 'pdf')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_%s_sesmean' % (corr_type,protocols[0]) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_%s_sesmean' % (corr_type,protocols[0]) + '.pdf'), format = 'pdf')

#%% Make the figure of the counts per bin:
fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax                                              = axes[ipp,iap]
            data                                            = np.log10(bincounts[:,:,iap,ilp,ipp])
            ax.pcolor(delta_az,delta_el,data,vmin=1.5,
                                vmax=np.nanpercentile(np.log10(bincounts),99.9),cmap="hot")
            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([-50,50])
            ax.set_ylim([-50,50])
            ax.set_xlabel(u'Δ Azimuth')
            ax.set_ylabel(u'Δ Elevation')
plt.tight_layout()

#%% 
fig,ax = plt.subplots(1,1,figsize=(3,3.5))

binedges = np.arange(0,100,5)
for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            rfdata      = deltarf.flatten()
            corrdata    = binmean[:,:,iap,ilp,ipp].flatten()
            nanfilter   = ~np.isnan(rfdata) & ~np.isnan(corrdata)
            corrdata    = corrdata[nanfilter]
            rfdata      = rfdata[nanfilter]
            bindata     = binned_statistic(x=rfdata,
                                        values= corrdata,
                                        statistic='median', bins=binedges)[0]
            ax.plot(binedges[:-1],bindata,c=clrs_areapairs[iap],label=areapair+projpair,linewidth=2)
            ax.set_title('%s\n%s' % (areapair, layerpair))
            ax.set_yticks(ticks=[0.025,0.035,0.045,0.055])
            ax.set_xlim([0,50])
            ax.set_ylim([np.round(np.nanpercentile(binmean[:,:,:,:,:].flatten(),3)*0.9,2),
                np.round(np.nanpercentile(binmean[:,:,:,:,:].flatten(),90)*1.1,2)])
            yl = ax.get_ylim()
            ax.set_yticks(ticks=[yl[0],(yl[0]+yl[1])/2,yl[1]])
            # ax.set_ylim([0.03,0.065])
            ax.set_xlabel(u'Δ RF')
            ax.set_ylabel(u'Correlation')
            ax.legend(loc='upper right',frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(savedir,'DeltaRF_1D_%s_%s_proj' % (corr_type,protocols[0]) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'DeltaRF_1D_%s_%s_proj' % (corr_type,protocols[0]) + '.pdf'), format = 'pdf')

#%% 

binmean = np.nanmean(binmean_ses,axis=5)
bincounts = np.nansum(bincounts_ses,axis=5)

binmean_ses[bincounts_ses<10]     = np.nan


#%% Make the figure:
delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
deltarf             = np.sqrt(delta_az**2 + delta_el**2)
min_counts          = 100


#%% 
fig,axes = plt.subplots(1,len(areapairs),figsize=(8,3))

binedges    = np.arange(0,100,binres)
bin1dcenters  = binedges[:-1] + binres/2

for iap,areapair in enumerate(areapairs):
    ax          = axes[iap]
    handles = []
    labels = []
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax          = ax
            rfdata      = deltarf.flatten()
            corrdata    = binmean[:,:,iap,ilp,ipp].flatten()
            nanfilter   = ~np.isnan(rfdata) & ~np.isnan(corrdata)
            corrdata    = corrdata[nanfilter]
            rfdata      = rfdata[nanfilter]
            bindata     = binned_statistic(x=rfdata,
                                        values= corrdata,
                                        statistic='median', bins=binedges)[0]
            
            if len(projpairs)>1: 
                ax.plot(binedges[:-1],bindata,c=clrs_projpairs[ipp],label=f'{areapair}\n{projpair}',linewidth=2)
            elif len(layerpairs)>1:
                ax.plot(binedges[:-1],bindata,c=clrs_layerpairs[ilp],label=f'{areapair}\n{layerpair}',linewidth=2)

            # rfdata      = deltarf.flatten()
            # bindata     = np.empty((np.shape(binmean_ses)[4],len(bin1dcenters)))
            # for ises in range(np.shape(binmean_ses)[4]):
            #     corrdata    = binmean_ses[:,:,iap,ilp,ipp,ises].flatten()
            #     nanfilter   = ~np.isnan(rfdata) & ~np.isnan(corrdata)
            #     corrdata    = corrdata[nanfilter]
            #     rfdata2      = rfdata.copy()
            #     rfdata2      = rfdata2[nanfilter]
            #     bindata[ises,:]     = binned_statistic(x=rfdata2,
            #                                 values= corrdata,
            #                                 statistic='mean', bins=binedges)[0]
            # if len(projpairs)>1: 
            #     # ax.plot(bincenters,bindata,c=clrs_projpairs[ipp],label=f'{areapair}\n{projpair}',linewidth=2)
            #     handles.append(shaded_error(ax,bin1dcenters,bindata,color=clrs_projpairs[ipp],error='sem'))
            #     labels.append(f'{areapair}\n{projpair}')
            # elif len(layerpairs)>1:
            #     ax.plot(bincenters,bindata,c=clrs_layerpairs[ilp],label=f'{areapair}\n{layerpair}',linewidth=2)
            #     # ax.plot(binedges[:-1],bindata,c=clrs_layerpairs[ilp],label=f'{areapair}\n{layerpair}',linewidth=2)

    ax.set_title('%s' % (areapair),c=clrs_areapairs[iap])
    ax.set_xlim([0,50])
    ax.set_ylim([np.round(np.nanpercentile(binmean[:,:,iap,:,:].flatten(),5)*0.9,2),
                    np.round(np.nanpercentile(binmean[:,:,iap,:,:].flatten(),90)*1.1,2)])
    yl = ax.get_ylim()
    ax.set_yticks(ticks=[yl[0],(yl[0]+yl[1])/2,yl[1]])
    ax.set_xlabel(u'Δ RF')
    ax.set_ylabel(u'Correlation')
    ax.legend(handles=handles,labels=labels,loc='lower right',frameon=False,fontsize=7,ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(savedir,'DeltaRF_1D_%s_%s_proj' % (corr_type,protocols[0]) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'DeltaRF_1D_%s_%s_proj' % (corr_type,protocols[0]) + '.pdf'), format = 'pdf')
# fig.savefig(os.path.join(savedir,'DeltaRF_1D_%s_%s_sem' % (corr_type,protocols[0]) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_1D_%s_%s_sem' % (corr_type,protocols[0]) + '.pdf'), format = 'pdf')

# #%% ##########################################################################################################
# #   2D     DELTA RECEPTIVE FIELD                 2D
# # ##########################################################################################################

# sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list = np.where(sessiondata['protocol'].isin(['GR','GN','IM']))[0]
# sessions_subset = [sessions[i] for i in sessions_in_list]

# #%% #########################################################################################
# # Contrast: across areas
# areas   = ['V1','PM']

# [noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap_areas(sessions_subset,corr_type='trace_corr',binresolution=7.5,
#                                                                  rotate_prefori=False,thr_tuned=0.0,rf_type='F',
#                                                                  thr_rf_p=0.001)

# min_counts = 100
# noiseRFmat_mean[countsRFmat<min_counts] = np.nan

# fig,axes = plt.subplots(2,2,figsize=(10,7))
# for i in range(2):
#     for j in range(2):
#         axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],5),
#                          vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
#         axes[i,j].set_title(areas[i] + '-' + areas[j])
# plt.tight_layout()
# # plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Interarea_AllProt_%dsessions' %nSessions  + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Interarea_RF_%dsessions' %nSessions  + '.png'), format = 'png')

# fig,axes = plt.subplots(2,2,figsize=(10,7))
# for i in range(2):
#     for j in range(2):
#         axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),
#                          cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
#         axes[i,j].set_title(areas[i] + '-' + areas[j])
# plt.tight_layout()

# # fig,axes = plt.subplots(4,4,figsize=(10,7))
# # for i in range(4):
# #     for j in range(4):
# #         axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
# #         axes[i,j].set_title(legendlabels[i,j])
# # plt.tight_layout()

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




