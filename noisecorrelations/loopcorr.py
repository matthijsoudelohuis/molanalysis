# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2022-2026, Champalimaud Center, Lisbon
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
from scipy.signal import detrend
from statannotations.Annotator import Annotator
from scipy.optimize import curve_fit

from loaddata.session_info import filter_sessions,load_sessions
from preprocessing.preprocesslib import assign_layer
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error,my_ceil,my_floor
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import compute_tuning_wrapper
from utils.explorefigs import plot_excerpt
from utils.shuffle_lib import my_shuffle, corr_shuffle
from utils.gain_lib import * 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

# First plot the mean and the std. This could already indicate some things. For each dataset make a scatter of the mean 
# and std with color the V1-PM and within area datasets as well.
# To understand what is different between the interactions the population metrics of Umakantha et al. 2021 could be computed as well: 
# Loading similarity, percent shared variance, and population dimensionality. 
# Factor analysis: loading similarity (all weights similarly positive or distributed?), percent shared variance (how much of the
# variability of each neuron can be accounted for by other nuerons and cofluctuations, and lastly the dimensionality. Fit FA model 
# maximizing the cross validated log likelihood. Then take d dimensions that explain 95% of explainable CV variance. 
# Make function of this FA model fitting (my_fa_fit.py)

# What is the objective here? The question is to understand what type of cofluctuations lead to the observed differences in 
# spike count correlations?


#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])

session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GR','GN','IM'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['RF'],filter_areas=['V1','PM'],session_rf=True)  

#%% Remove two sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
# sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE09665_2023_03_21']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='dF',keepraw=True)
                                # calciumversion='deconv',keepraw=True)
                                calciumversion='deconv',keepraw=True,filter_hp=0.01)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.25,uppertriangular=False)[0]
    # sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% print number of pairs:
npairs = np.zeros(nSessions)
for ises,ses in enumerate(sessions):
    npairs[ises] = np.sum(~np.isnan(ses.trace_corr))/2
print('Number of pairs: %d (mean: %d, std : %d across n=%d sessions)' % 
            (np.sum(npairs),np.mean(npairs),np.std(npairs),nSessions))

#%% 
redcelllabels = np.array(['unl','lab'])
for ses in sessions:
    ses.celldata['labeled'] = ses.celldata['redcell']
    ses.celldata['labeled'] = ses.celldata['labeled'].astype(int).apply(lambda x: redcelllabels[x])

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=True)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='GM')
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False,remove_method='RRR',remove_rank=2)

# plt.imshow(sessions[27].noise_corr,vmin=-0.03,vmax=0.05)
# plt.imshow(sessions[0].noise_corr,vmin=-0.03,vmax=0.05)

#%% #########################################################################################
# Contrast: across areas, layers and projection pairs:
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
#If you override any of these with input to the deltarf bin function as ' ', then these pairs will be ignored

clrs_areapairs      = get_clr_area_pairs(areapairs)
clrs_layerpairs     = get_clr_layerpairs(layerpairs)
clrs_projpairs      = get_clr_labelpairs(projpairs)

# clrs_area_labelpairs = get_clr_area_labelpairs(areapairs+projpairs)
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% #####################################################################################################
# DELTA ANATOMICAL DISTANCE :
# #######################################################################################################

#%% Define the areapairs:
areapairs       = ['V1-V1','PM-PM']
clrs_areapairs  = get_clr_area_pairs(areapairs)

#%% Compute pairwise correlations as a function of pairwise anatomical distance ###################################################################
# for corr_type in ['trace_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type=corr_type)

    #Make the figure per protocol:
    fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type=corr_type)
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.pdf'), format = 'pdf')

#%% #########################################################################################
protocols           = ['GR','GN']
# protocols           = ['IM']
ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

areapairs           = ['V1-V1','PM-PM']
layerpairs          = ' '
projpairs           = ['unl-unl','unl-lab','lab-lab']
clrs_projpairs      = get_clr_labelpairs(projpairs)

corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'
# corr_thr            = 0.025 #thr in percentile of total corr for significant pos or neg

[bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
bin_angle_surr_count,bincenters_angle] = bin_corr_deltaxy(ses,onlysameplane=False,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,binresolution=10,noise_thr=20)

#%% Plot:
binidx = bincenters_dist>10
fig,axes = plt.subplots(1,2,sharey=True,figsize=(6.5,3.5))
for iap,areapair in enumerate(['V1','PM']):
    ax = axes[iap]
    dim12label = 'XY (um)'
    handles = []
    for ipp,projpair in enumerate(projpairs):
        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
        handles.append(shaded_error(x=bincenters_dist[binidx],y=bin_dist_mean[binidx,iap,0,ipp],yerror=bin_dist_error[binidx,iap,0,ipp],
                        ax = ax,color=clrs_projpairs[ipp],label=projpair))
        # ax.plot(bincenters_dist,bin_dist_mean[:,iap,0,ipp],color=clrs_projpairs[ipp],label=projpair)
    ax.set_title(areapair)
    if iap==0:
        ax.set_ylabel('Correlation')
    ax.legend(handles=handles,labels=projpairs,frameon=False)
    ax.set_xlabel(u'Δ %s' % dim12label)   
    ax.set_yticks(np.arange(0,0.051,0.01))
    ax.set_xlim([-10,500])
    ax.set_ylim([0.0,0.023])
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','DistXY_MeanCorr_WithinArea_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'MeanCorr','DistXY_MeanCorr_WithinArea_%s_PCA1_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')

#%% ########################################################################################################
# ##################### Noise correlations within and across areas: ########################################
# ##########################################################################################################

dfses = mean_corr_areas_labeling([sessions[0]],corr_type='trace_corr',absolute=True,minNcells=100)[0]
clrs_area_labelpairs = get_clr_area_labelpairs(list(dfses.columns))

#%% Compute the variance across trials for each cell:
# for ses in sessions:
#     if ses.sessiondata['protocol'][0]=='GR':
#         resp_meanori,respmat_res        = mean_resp_gr(ses)
#     elif ses.sessiondata['protocol'][0]=='GN':
#         resp_meanori,respmat_res        = mean_resp_gn(ses)
#     ses.celldata['noise_variance']  = np.var(respmat_res,axis=1)

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:
protocols           = ['GR','GN']
# protocols           = ['IM']
# corr_type           = 'trace_corr'
corr_type           = 'noise_corr'
projpairs           = ['unl-unl','unl-lab','lab-lab']
clrs_projpairs      = get_clr_labelpairs(projpairs)

ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

bincenters,_,meancorrV1,varcorrV1,fraccorrV1 = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                    areapairs=['V1-V1'],layerpairs=' ',minNcells=10)
bincenters,_,meancorrPM,varcorrPM,fraccorrPM = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                    areapairs=['PM-PM'],layerpairs=' ',minNcells=10)

#%% Plot within area mean pairwise correlations:
#combine V1 and PM, and filter out duplicate pair (unl-lab is same as lab-unl):
# meancorr            = np.stack([meancorrV1[:,:,:,[0,1,3]],meancorrPM[:,:,:,[0,1,3]]],axis=0) 
meancorr            = np.stack([meancorrV1,meancorrPM],axis=0)
# meancorr             = np.stack([fraccorrV1[0,:,:,:,:],fraccorrPM[0,:,:,:,:]],axis=0)
# meancorr             = np.stack([fraccorrV1[1,:,:,:,:],fraccorrPM[1,:,:,:,:]],axis=0)

projpairs_areas = [['V1unl-V1unl','V1unl-V1lab','V1lab-V1lab'],
             ['PMunl-PMunl','PMunl-PMlab','PMlab-PMlab']]

statpairs_areas = [[('V1unl-V1unl','V1lab-V1lab'),
         ('V1unl-V1unl','V1unl-V1lab'),
         ('V1unl-V1lab','V1lab-V1lab')],
           [('PMunl-PMunl','PMunl-PMlab'),
         ('PMunl-PMunl','PMlab-PMlab'),
         ('PMunl-PMlab','PMlab-PMlab'),
         ]] #for statistics

clrs_projpairs      = get_clr_labelpairs(['unl-unl','unl-lab','lab-lab'])

fig,axes = plt.subplots(1,2,figsize=(5,3.5),sharey=True)
# for isign,sign in enumerate(['pos','neg']):
for iarea,area in enumerate(['V1','PM']):
    df                  = pd.DataFrame(data=meancorr[iarea,:,:,:,:].squeeze(),columns=projpairs_areas[iarea])
    df                  = df.dropna(axis=0,thresh=2).reset_index(drop=True) #drop occasional missing data

    ax                  = axes[iarea]
    if df.any(axis=None):
        ax.scatter(np.arange(3),df.mean(axis=0),marker='o',s=15,color='k')
        ax.plot(np.arange(3),df.mean(axis=0),linestyle='-',color='k')
        sns.stripplot(ax=ax,data=df,color='grey',size=3,palette=clrs_projpairs,jitter=0.15)
        ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
        annotator = Annotator(ax, statpairs_areas[iarea], data=df,order=list(df.columns))
        annotator.configure(test='Wilcoxon', text_format='star', loc='inside',line_height=0,text_offset=-0.5,fontsize=7,	
                            line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=0,
                            correction_format='replace')
        annotator.apply_and_annotate()
        ax.set_ylabel('Correlation')
        # ax.set_title('%s' % '_'.join(protocols),fontsize=12)
        ax.set_ylim([my_floor(df.min(axis=None)*0.9,3),my_ceil(df.max(axis=None)*1.1,3)])
        # ax.set_ylim([0,my_ceil(df.max(axis=None)*1.2,2)])
    ax.set_title('%s' % (area),fontsize=12)
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_WithinArea_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:
protocols           = ['GR','GN']
# protocols           = ['IM']
# corr_type           = 'trace_corr'
corr_type           = 'noise_corr'
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
clrs_projpairs      = get_clr_labelpairs(projpairs)

ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

bincenters,_,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                    areapairs=['V1-PM'],layerpairs=' ',minNcells=10)

#%% 
dfses = mean_corr_areas_labeling([sessions[0]],corr_type='trace_corr',absolute=True,minNcells=1000)[0]

statpairs = [('V1unl-PMunl','V1lab-PMunl'),
         ('V1unl-PMunl','V1unl-PMlab'),
         ('V1unl-PMunl','V1lab-PMlab'),
         ('V1unl-PMlab','V1lab-PMunl'),
         ('V1unl-PMlab','V1lab-PMlab'),
         ('V1lab-PMunl','V1lab-PMlab'),
         ] #for statistics

fig,axes = plt.subplots(1,1,figsize=(3.5,3.5))
df                  = pd.DataFrame(data=meancorr[:,:,:,:].squeeze(),columns=dfses.columns[-4:])
df                  = df.dropna(axis=0,thresh=2).reset_index(drop=True) #drop occasional missing data
df                  = df.fillna(df.mean()) #interpolate occasional missing data

ax                  = axes
if df.any(axis=None):
    ax.scatter(np.arange(4),df.mean(axis=0),marker='o',s=15,color='k')
    ax.plot(np.arange(4),df.mean(axis=0),linestyle='-',color='k')
    sns.stripplot(ax=ax,data=df,color='grey',size=3,palette=clrs_projpairs)
    ax.set_xticklabels(labels=df.columns,rotation=60,fontsize=8)
    annotator = Annotator(ax, statpairs, data=df,order=list(df.columns))
    annotator.configure(test='Wilcoxon', text_format='star', loc='inside',line_height=0,text_offset=3,fontsize=8,	
                        line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                        correction_format='replace')
    annotator.apply_and_annotate()
    ax.set_ylabel('Correlation')
    ax.set_title('%s' % '_'.join(protocols),fontsize=12)
    ax.set_ylim([my_floor(df.min(axis=None)*0.95,3),my_ceil(df.max(axis=None)*1.4,2)])
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_InterArea_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')

#%% 
dfses = mean_corr_areas_labeling([sessions[0]],corr_type='trace_corr',absolute=True,minNcells=1000)[0]

statpairs = [('V1unl-PMunl','V1lab-PMunl'),
         ('V1unl-PMunl','V1unl-PMlab'),
         ('V1unl-PMunl','V1lab-PMlab'),
         ('V1unl-PMlab','V1lab-PMunl'),
         ('V1unl-PMlab','V1lab-PMlab'),
         ('V1lab-PMunl','V1lab-PMlab'),
         ] #for statistics

fig,axes = plt.subplots(1,2,figsize=(6.5,3.5))
for isign,sign in enumerate(['pos','neg']):
    df                  = pd.DataFrame(data=fraccorr[isign,:,:,:,:].squeeze(),columns=dfses.columns[-4:])
    df                  = df.dropna(axis=0,thresh=2).reset_index(drop=True) #drop occasional missing data
    df                  = df.fillna(df.mean()) #interpolate occasional missing data

    ax                  = axes[isign]
    if df.any(axis=None):
        ax.scatter(np.arange(4),df.mean(axis=0),marker='o',s=15,color='k')
        ax.plot(np.arange(4),df.mean(axis=0),linestyle='-',color='k')
        sns.stripplot(ax=ax,data=df,color='grey',size=3,palette=clrs_projpairs,jitter=0.15)
        ax.set_xticklabels(labels=df.columns,rotation=60,fontsize=8)
        annotator = Annotator(ax, statpairs, data=df,order=list(df.columns))
        annotator.configure(test='Wilcoxon', text_format='star', loc='inside',line_height=0,text_offset=2,fontsize=8,	
                            line_width=1,verbose=True,
                            correction_format='replace')
        annotator.apply_and_annotate()
        ax.set_ylabel('Frac. correlated pairs')
        # ax.set_title('%s' % '_'.join(protocols),fontsize=12)
        ax.set_ylim([my_floor(df.min(axis=None)*0.9,1),my_ceil(df.max(axis=None)*1.4,1)])
    ax.set_title('%s' % (sign),fontsize=12)
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_PosNeg_InterArea_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')

#%%  Find session with largest effect:
# areapairs           = ['V1-PM']
# corr_type           = 'noise_corr'
# ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
# bincenters,histcorr,meancorr,varcorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                    # areapairs=['V1-PM'],layerpairs=' ',minNcells=10)

sesidx = np.argmax(np.nanvar(meancorr[:,0,0,:],axis=-1))
sesidx = np.nanargmax(meancorr[:,0,0,3]-meancorr[:,0,0,0])
print('Session with largest difference in mean %s by labeling across areas is:\n%s' % (corr_type,ses[sesidx].sessiondata['session_id'][0]))

sesidx = np.argmax(np.nanvar(varcorr[:,0,0,:],axis=-1))
# sesidx = np.nanargmax(meancorr[:,0,0,3]-meancorr[:,0,0,0])

print('Session with largest difference in %s variance by labeling across areas is:\n%s' % (corr_type,ses[sesidx].sessiondata['session_id'][0]))


#%% Show correlation matrix for this session.
# sort by area and labeling identity:
# sesidx = 8
ses = sessions[sesidx]
corr_type = 'trace_corr'
data = getattr(ses,corr_type)
arealabels = ['V1unl','V1lab','PMunl','PMlab']

# sort rows and columns by area and label:
sortidx     = np.flip(np.argsort(ses.celldata['arealabel']))
sortlabels  = np.flip(np.sort(ses.celldata['arealabel']))

# sort rows and columns by area and label, and sort by mean correlation to all other cells:
avgcorrcells = np.nanmean(data,axis=0)
sortidx     = np.flip(np.lexsort((avgcorrcells,ses.celldata['arealabel'])))
sortlabels  = ses.celldata['arealabel'][sortidx]

data        = data[sortidx,:][:,sortidx]
# data        = data[sortidx,:]#[:,sortidx]

fig,ax = plt.subplots(figsize=(6,6))
pcm = ax.imshow(data,cmap='bwr',clim=(-my_ceil(np.nanpercentile(data,95),2),my_ceil(np.nanpercentile(data,95),2)))
# pcm = ax.imshow(data,cmap='bwr',clim=(-my_ceil(np.nanpercentile(data,95),2),my_ceil(np.nanpercentile(data,95),2)))
for al in arealabels:
    ax.axhline(y=np.where(sortlabels==al)[0][0],color='k',linestyle='-',linewidth=0.5)
    ax.axvline(x=np.where(sortlabels==al)[0][0],color='k',linestyle='-',linewidth=0.5)
ax.set_xticks([]); ax.set_yticks([])
# fig.colorbar(pcm, ax=ax)

cb = fig.colorbar(pcm, ax=ax,shrink=0.3)
cb.set_label('Correlation',fontsize=10,loc='center')
cb.set_ticks([cb.vmin,0,cb.vmax])
fig.savefig(os.path.join(savedir,'CorrMat','CorrMat_%s_%s' % (corr_type,sessions[sesidx].sessiondata['session_id'][0]) + '.png'), format = 'png')


#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:
protocols           = ['GR','GN']
# protocols           = ['GN']

areapairs           = ['V1-V1','PM-PM','V1-PM']
# areapairs           = ['V1-V1']
# areapairs           = ['PM-PM']
areapairs           = ['V1-PM']

plt.rcParams['axes.spines.right']   = True
plt.rcParams['axes.spines.top']     = True
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
clrs_projpairs = get_clr_labelpairs(projpairs)
zscoreflag = False
# for corr_type in ['trace_corr','sig_corr','noise_corr']:
# for corr_type in ['sig_corr']:
for corr_type in ['noise_corr']:
# for corr_type in ['trace_corr']:
    for areapair in areapairs:
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

        bincenters,histcorr,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                            # areapairs=[areapair],layerpairs=['L2/3-L5'],minNcells=5,zscore=zscoreflag)
                                                            # areapairs=[areapair],layerpairs=['L2/3-L2/3'],minNcells=5,zscore=zscoreflag)
                                                            # areapairs=[areapair],layerpairs=['L5-L5'],minNcells=5,zscore=zscoreflag)
                                                            areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag,valuematching=None)
        
        bincenters_sh,histcorr_sh,meancorr_sh,varcorr_sh,_ = hist_corr_areas_labeling(ses,corr_type='corr_shuffle',filternear=False,projpairs=' ',noise_thr=20,
                                                            areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag,valuematching=None)
        print('%d/%d sessions with lab-lab populations for %s'
              % (np.sum(~np.any(np.isnan(histcorr[:,:,0,0,-1]),axis=0)),len(ses),areapair))
        
        areaprojpairs = projpairs.copy()
        for ipp,projpair in enumerate(projpairs):
            areaprojpairs[ipp]       = areapair.split('-')[0] + projpair.split('-')[0] + '-' + areapair.split('-')[1] + projpair.split('-')[1] 
    
        fig         = plt.figure(figsize=(8, 4))
        gspec       = fig.add_gridspec(nrows=2, ncols=3)
        
        histdata    = np.cumsum(histcorr,axis=0)/100 #get cumulative distribution
        # histdata    = histcorr/100 #get cumulative distribution
        histmean    = np.nanmean(histdata,axis=1) #get mean across sessions
        histerror   = np.nanstd(histdata,axis=1) / np.sqrt(len(ses)) #compute SEM
       
        histdata_sh  = np.cumsum(histcorr_sh,axis=0)/100 #get cumulative distribution
        histmean_sh = np.nanmean(histdata_sh,axis=1) #get mean across sessions
        histerror_sh = np.nanstd(histdata_sh,axis=1) / np.sqrt(len(ses)) #compute SEM

        ax0         = fig.add_subplot(gspec[:2, :2]) #bigger subplot for the cum dist
        
        xpos = bincenters[np.where(np.nanmean(histmean,axis=3).squeeze()<0.1)[0][-1]]
        axins1 = ax0.inset_axes([0.05, 0.25, 0.3, 0.4],xlim=([xpos-0.05,xpos+0.025]),ylim=[0,0.2],xticklabels=[], yticklabels=[])
        ax0.indicate_inset_zoom(axins1, edgecolor="black")
        axins1.tick_params(axis='both', which='both', length=0)
        for axis in ['top','bottom','left','right']:
            axins1.spines[axis].set_color('gray')
            axins1.spines[axis].set_linewidth(1)

        xpos = bincenters[np.where(np.nanmean(histmean,axis=3).squeeze()>0.9)[0][0]]
        axins2 = ax0.inset_axes([0.65, 0.25, 0.3, 0.4],xlim=([xpos-0.05,xpos+0.05]),ylim=[0.8,1],xticklabels=[], yticklabels=[])
        ax0.indicate_inset_zoom(axins2, edgecolor="gray")
        axins2.tick_params(axis='both', which='both', length=0)
        for axis in ['top','bottom','left','right']:
            axins2.spines[axis].set_color('gray')
            axins2.spines[axis].set_linewidth(1)

        handles = []
        for ipp,projpair in enumerate(projpairs): #show for each projection identity pair:
            ax0.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp],linewidth=0.3)
            # handles.append(shaded_error(x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
                            # yerror=np.squeeze(histerror[:,0,0,ipp]),ax=ax0,color=clrs_projpairs[ipp]))
            # for ises in range(len(ses)):
                # ax0.plot(bincenters,np.squeeze(histdata[:,ises,0,0,ipp]),color=clrs_projpairs[ipp],linewidth=0.3)
            axins1.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp])
            axins2.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # shaded_error(axins1,x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
            #                 yerror=np.squeeze(histerror[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # shaded_error(axins2,x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
            #                 yerror=np.squeeze(histerror[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # plot triangle for mean:
            ax0.plot(np.nanmean(meancorr[:,0,0,ipp],axis=None),0.9+ipp/50,'v',color=clrs_projpairs[ipp],markersize=5)
        
        handles.append(shaded_error(x=bincenters,y=np.squeeze(histmean_sh),
                            yerror=np.squeeze(histerror_sh),ax=ax0,color='k'))
        axins1.plot(bincenters,np.squeeze(histmean_sh),color='k')
        axins2.plot(bincenters,np.squeeze(histmean_sh),color='k')  

        ax0.set_xlabel('Correlation')
        ax0.set_ylabel('Cumulative Fraction')
        ax0.legend(handles=handles,labels=areaprojpairs,frameon=False,loc='upper left',fontsize=8)
        ax0.set_xlim([-0.25,0.35])
        ax0.set_xlim([-0.15,0.25])
        ax0.set_xlim([-0.1,0.20])
        if zscoreflag:
            ax0.set_xlim([-2,2])
        ax0.axvline(0,linewidth=0.5,linestyle=':',color='k') #add line at zero for ref
        ax0.set_ylim([0,1])
        # ax0.set_ylim([0,0.15])
        ax0.set_title('%s %s' % (areapair,corr_type),fontsize=12)

        #  Now show a heatmap of the meancorr data averaged over sessions (first dimension). 
        #  Between each projpair a paired t-test is done of the mean across sesssions and if significant a line is 
        #  drawn from the center of that entry of the heatmap and other one with an asterisk on top of the line. 
        #  For subplot 3 the same is done but then with varcorr.
        data        = np.squeeze(np.nanmean(meancorr[:,0,:,:],axis=0))
        data        = np.reshape(data,(2,2))

        xlabels     = [areapair.split('-')[1] + 'unl',areapair.split('-')[1] + 'lab'] 
        ylabels     = [areapair.split('-')[0] + 'unl',areapair.split('-')[0] + 'lab'] 
        xlocs        = np.array([0,1,0,1])
        ylocs        = np.array([0,0,1,1])
        if areapair=='V1-PM':
            test_indices = np.array([[0,1],[0,2],[1,2],[2,3],[0,3],[1,3]])
        else: 
            test_indices = np.array([[0,1],[0,3],[1,3]])
        
        ax1 = fig.add_subplot(gspec[0, 2])
        pcm = ax1.imshow(data,cmap='hot',vmin=my_floor(np.min(data)-0.002,2),vmax=my_ceil(np.max(data),2))
        ax1.set_xticks([0,1],labels=xlabels)
        ax1.xaxis.tick_top()
        ax1.set_yticks([0,1],labels=ylabels)
        ax1.set_title('Mean')
        fig.colorbar(pcm, ax=ax1)
        
        for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
            data1 = meancorr[:,0,0,ix]
            data2 = meancorr[:,0,0,iy]
            pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = stats.wilcoxon(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = pval * 3 #bonferroni correction
            if pval<0.05:
                ax1.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
                ax1.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                                    weight='bold',fontsize=10) #

        # Now the same but for the std of the pairwise correlations:
        data        = np.squeeze(np.nanmean(varcorr[:,0,:,:],axis=0))
        data        = np.reshape(data,(2,2))

        ax2 = fig.add_subplot(gspec[1, 2])
        pcm = ax2.imshow(data,cmap='hot',vmin=my_floor(np.min(data)-0.002,2),vmax=my_ceil(np.max(data),2))
        ax2.set_xticks([0,1],labels=xlabels)
        ax2.xaxis.tick_top()
        ax2.set_yticks([0,1],labels=ylabels)
        ax2.set_title('Std')
        fig.colorbar(pcm, ax=ax2)
        
        for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
            data1 = varcorr[:,0,0,ix]
            data2 = varcorr[:,0,0,iy]
            pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = stats.wilcoxon(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = pval * 6 #bonferroni correction
            if pval<0.05:
                ax2.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
                ax2.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                                    weight='bold',fontsize=10) #
        
        # plt.suptitle('%s %s' % (areapair,corr_type),fontsize=12)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_PCA1_L5L23_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_L23L5_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_MatchOSI_Proj_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')



#%% 
for ses in sessions:
    ses.celldata = assign_layer(ses.celldata)

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

protocols           = ['GR','GN']

# areapairs           = ['V1-V1']
areapairs           = ['V1-V1','PM-PM','V1-PM']

zscoreflag = False
# for corr_type in ['trace_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    for areapair in areapairs:
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
        
        bincenters,histcorr,meancorr,varcorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=' ',
                                                            areapairs=[areapair],layerpairs=layerpairs,minNcells=10,zscore=zscoreflag)
        
        arealayerpairs = layerpairs.copy()
        for ilp,layerpair in enumerate(layerpairs):
            arealayerpairs[ilp]       = areapair.split('-')[0] + layerpair.split('-')[0] + '-' + areapair.split('-')[1] + layerpair.split('-')[1] 

        fig         = plt.figure(figsize=(8, 4))
        gspec       = fig.add_gridspec(nrows=2, ncols=3)
        
        histdata    = np.cumsum(histcorr,axis=0)/100 #get cumulative distribution
        histmean    = np.nanmean(histdata,axis=1) #get mean across sessions
        histerror   = np.nanstd(histdata,axis=1) / np.sqrt(len(ses)) #compute SEM
        
        ax0         = fig.add_subplot(gspec[:2, :2]) #bigger subplot for the cum dist
        
        axins1 = ax0.inset_axes([0.05, 0.25, 0.3, 0.4],xlim=([-0.1,-0.025]),ylim=[0,0.2],xticklabels=[], yticklabels=[])
        ax0.indicate_inset_zoom(axins1, edgecolor="black")
        axins1.tick_params(axis='both', which='both', length=0)
        for axis in ['top','bottom','left','right']:
            axins1.spines[axis].set_color('gray')
            axins1.spines[axis].set_linewidth(1)
            
        axins2 = ax0.inset_axes([0.65, 0.25, 0.3, 0.4],xlim=([0.075,0.125]),ylim=[0.8,0.9],xticklabels=[], yticklabels=[])
        ax0.indicate_inset_zoom(axins2, edgecolor="gray")
        axins2.tick_params(axis='both', which='both', length=0)
        for axis in ['top','bottom','left','right']:
            axins2.spines[axis].set_color('gray')
            axins2.spines[axis].set_linewidth(1)

        handles = []
        for ilp,layerpair in enumerate(layerpairs): #show for each layer pair:
            handles.append(shaded_error(ax0,x=bincenters,y=np.squeeze(histmean[:,0,ilp,0]),
                            yerror=np.squeeze(histerror[:,0,ilp,0]),color=clrs_layerpairs[ilp]))
            
            axins1.plot(bincenters,np.squeeze(histmean[:,0,ilp,0]),color=clrs_layerpairs[ilp])
            axins2.plot(bincenters,np.squeeze(histmean[:,0,ilp,0]),color=clrs_layerpairs[ilp])
            #plot triangle for mean:
            ax0.plot(np.nanmean(meancorr[:,0,ilp,0],axis=None),0.9+ilp/50,'v',color=clrs_layerpairs[ilp],markersize=5)

        ax0.set_xlabel('Correlation')
        ax0.set_ylabel('Cumulative Fraction')
        ax0.legend(handles=handles,labels=arealayerpairs,frameon=False,loc='upper left',fontsize=8)
        ax0.set_xlim([-0.25,0.35])
        if zscoreflag:
            ax0.set_xlim([-2,2])
        ax0.axvline(0,linewidth=0.5,linestyle=':',color='k') #add line at zero for ref
        ax0.set_ylim([0,1])
        ax0.set_title('%s %s' % (areapair,corr_type),fontsize=12)

        #  Now show a heatmap of the meancorr data averaged over sessions (first dimension). 
        #  Between each layerpair a paired t-test is done of the mean across sesssions and if significant a line is 
        #  drawn from the center of that entry of the heatmap and other one with an asterisk on top of the line. 
        #  For subplot 3 the same is done but then with varcorr.
        data        = np.squeeze(np.nanmean(meancorr[:,0,:,0],axis=0))
        data        = np.reshape(data,(2,2))

        xlabels     = [areapair.split('-')[1] + 'L2/3',areapair.split('-')[1] + 'L5']
        ylabels     = [areapair.split('-')[0] + 'L2/3',areapair.split('-')[0] + 'L5']
        xlocs        = np.array([0,1,0,1])
        ylocs        = np.array([0,0,1,1])
        
        if areapair=='V1-PM':
            test_indices = np.array([[0,1],[0,2],[1,2],[2,3],[0,3],[1,3]])
        else: 
            test_indices = np.array([[0,1],[0,3],[1,3]])
        
        ax1 = fig.add_subplot(gspec[0, 2])
        pcm = ax1.imshow(data,cmap='plasma',vmin=my_floor(np.min(data),2),vmax=my_ceil(np.max(data),2))
        ax1.set_xticks([0,1],labels=xlabels)
        ax1.xaxis.tick_top()
        ax1.set_yticks([0,1],labels=ylabels)
        ax1.set_title('Mean')
        fig.colorbar(pcm, ax=ax1)
        
        for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
            data1 = meancorr[:,0,ix,0]
            data2 = meancorr[:,0,iy,0]
            pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = pval * 3 #bonferroni correction
            if pval<0.05:
                ax1.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
                ax1.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                                    weight='bold',fontsize=10) #

        # Now the same but for the std of the pairwise correlations:
        data        = np.squeeze(np.nanmean(varcorr[:,0,:,0],axis=0))
        data        = np.reshape(data,(2,2))

        ax2 = fig.add_subplot(gspec[1, 2])
        pcm = ax2.imshow(data,cmap='plasma',vmin=my_floor(np.min(data),2),vmax=my_ceil(np.max(data),2))
        ax2.set_xticks([0,1],labels=xlabels)
        ax2.xaxis.tick_top()
        ax2.set_yticks([0,1],labels=ylabels)
        ax2.set_title('Std')
        fig.colorbar(pcm, ax=ax2)
        
        for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
            data1 = varcorr[:,0,ix,0]
            data2 = varcorr[:,0,iy,0]
            pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = pval * 6 #bonferroni correction
            if pval<0.05:
                ax2.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
                ax2.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                                    weight='bold',fontsize=10) #
        
        # plt.suptitle('%s %s' % (areapair,corr_type),fontsize=12)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Layer_PCA1_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Layer_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')

#%% Plot mean vs standard deviation for labeling across areapairs:
# Umakantha et al. 2023: might signal different population activity fluctuations that are shared

areapairs           = ['V1-V1','PM-PM','V1-PM']
zscoreflag      = True
circres         = 0.25
tickres         = 0.2
lim             = 1.7

# for corr_type in ['trace_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    fig,axes = plt.subplots(1,3,figsize=(9,3))
    for iap,areapair in enumerate(areapairs):
        ax                  = axes[iap]
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
        
        bincenters,histcorr,meancorr,varcorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=True,projpairs=projpairs,
                                                            areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag)

        for ipp,projpair in enumerate(projpairs):
            # ax.scatter(meancorr[:,0,0,ipp],varcorr[:,0,0,ipp],c=clrs_projpairs[ipp],s=4,alpha=0.7)
            ax.errorbar(np.nanmean(meancorr[:,0,0,ipp]),np.nanmean(varcorr[:,0,0,ipp]),
                        np.nanstd(meancorr[:,0,0,ipp]) / np.sqrt(len(ses)),np.nanstd(varcorr[:,0,0,ipp])/ np.sqrt(len(ses)),
                        ecolor=clrs_projpairs[ipp],elinewidth=1,capsize=3)
        ax.set_xlabel('Mean')
        ax.set_ylabel('Std')

        ax.set_xticks(np.arange(0,lim,tickres))
        ax.set_yticks(np.arange(0,lim,tickres))
        ax.set_xlim([0,lim])
        ax.set_ylim([0,lim])
        # ax.set_xlim([0,my_ceil(np.nanmax(varcorr),2)])
        # ax.set_ylim([0,my_ceil(np.nanmax(varcorr),2)])
        ax.set_title(areapair)

        for radius in np.arange(0,lim*2,circres):
            Drawing_uncolored_circle = plt.Circle( (0, 0), radius, linestyle=':',fill=False)
            ax.add_artist(Drawing_uncolored_circle)
        # ax0.legend(frameon=False,loc='upper left',fontsize=8)
        # ax0.set_xlim([-0.5,0.5])
        # ax0.set_ylim([0,1.1])
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_Z_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdMean_Z_PCA1_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')
        
# clrs_areapairs      = get_clr_area_pairs(areapairs)
# clrs_layerpairs     = get_clr_layerpairs(layerpairs)
# clrs_projpairs      = get_clr_labelpairs(projpairs)

#%% Plot mean absolute correlation across sessions conditioned on area pairs:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

protocols = ['GR','GN']# protocols = ['IM']
protocols = ['IM']

for corr_type in ['trace_corr','sig_corr','noise_corr']:
# for corr_type in ['trace_corr']:
    ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
    df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,absolute=True,filternear=False,minNcells=10)
    df                  = df_mean
    df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
    df                  = df.fillna(df.mean()) #interpolate occasional missing data
    
    fig,axes = plt.subplots(1,1,figsize=(4,4))
    ax                  = axes
    if df.any(axis=None):
        sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
        ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
        annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
        annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                            line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                            correction_format='replace')
        annotator.apply_and_annotate()
        ax.set_ylabel('Abs. correlation')
        ax.set_title('%s' % '_'.join(protocols),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_stationary_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_stationary_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_dF_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_dF_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.pdf'), format = 'pdf')

#%% Plot mean correlation across sessions conditioned on area pairs for pos and neg separately:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

protocols = ['GR','GN']
protocols = ['IM']

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,1,figsize=(4,4))
    for isign,sign in enumerate(['pos','neg']):
        ax                  = axes[isign]
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
        df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=False,minNcells=10)
        df                  = df_mean
        df                  = df_frac
        # df                  = df_mean

        df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
        df                  = df.fillna(df.mean()) #interpolate occasional missing data

        if df.any(axis=None):
            sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
            if isign==1:
                ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
            else: ax.set_xticks([])
            if isign==1: 
                ax.invert_yaxis()
            annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
            annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=0.05,text_offset=0, 
                                line_width=0.5,comparisons_correction='Benjamini-Hochberg',verbose=False,fontsize=7,
                                correction_format='replace')
            annotator.apply_and_annotate()
            if isign==1:
                ax.invert_yaxis()
            ax.set_ylabel('%s correlation' % sign)
            ax.set_title('%s' % '_'.join(protocols),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()

    # fig.savefig(os.path.join(savedir,'MeanCorr','PosNegCorr_sigOnly_dF_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','PosNegCorr_sigOnly_dF_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','PosNegCorr_sigOnly_dF_stationary_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','PosNegCorr_sigOnly_dF_stationary_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.pdf'), format = 'pdf')
    
#%% Plot mean absolute correlation across sessions conditioned on area pairs and per protocol:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharex=True,sharey='row')
    for iprot,prot in enumerate(['GR','GN','IM']):
        ax                  = axes[iprot]
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
        df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,absolute=True,filternear=True,minNcells=10)
        df                  = df_mean
        df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
        # df                  = df.dropna() #drop sessions with occasional missing data
        df                  = df.fillna(df.mean()) #interpolate occasional missing data
        
        if df.any(axis=None):
            sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
            ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
            annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
            annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                                line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                                # line_width=1,comparisons_correction=None,verbose=False,
                                correction_format='replace')
            annotator.apply_and_annotate()
            ax.set_ylabel('Correlation')
            ax.set_title('%s' %(prot),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

#%% Plot mean correlation across sessions conditioned on area pairs and per protocol for pos and neg separately:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,3,figsize=(12,6),sharex=True,sharey='row')
    # for iprot,prot in enumerate(['GR','GN','IM']):
    for iprot,prot in enumerate(['SP']):
        for isign,sign in enumerate(['pos','neg']):
            ax                  = axes[isign,iprot]
            ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
            df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=True,minNcells=10)
            df                  = df_mean
            df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
            df                  = df.fillna(df.mean()) #interpolate occasional missing data
            # df                  = df.dropna() #drop sessions with occasional missing data

            if df.any(axis=None):
                sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
                if isign==1:
                    ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
                else: ax.set_xticks([])
                if isign==1: 
                    ax.invert_yaxis()
                annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
                annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=0.05,text_offset=0, 
                                    line_width=0.5,comparisons_correction='Benjamini-Hochberg',verbose=False,fontsize=7,
                                    correction_format='replace')
                annotator.apply_and_annotate()
                if isign==1:
                    ax.invert_yaxis()
                ax.set_ylabel('%s correlation' % sign)
                ax.set_title('%s' %(prot),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

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
            df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
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
            # ax.set_ylim([0,1])
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
    fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
