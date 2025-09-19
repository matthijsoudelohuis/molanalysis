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

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import FactorAnalysis as FA
from statsmodels.formula.api import ols

from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import *
from utils.gain_lib import * 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')
calciumversion = 'dF'

#%% ###### Load SP sessions: ###############################
sessions,nSessions  = filter_sessions(protocols = ['SP'])
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% 
session_list        = np.array([['LPE11086_2024_01_05']])
session_list        = np.array([['LPE12223_2024_06_10','LPE11086_2024_01_05']])

#%%  Load data and transfer GR tuning to SP celldata info:      
idx_ses             = np.zeros(len(sessiondata),dtype=bool)
tuning_metrics      = ['OSI','gOSI','tuning_var','pref_ori','DSI']
for ises in tqdm(range(nSessions),total=nSessions,desc='Transfer GR tuning info to SP celldata'):
    GRsessions,nGRsessions  = filter_sessions(protocols = ['GR'],only_session_id=[sessiondata['session_id'][ises]])

    if nGRsessions:
        idx_ses[ises]   = True # SP session has corresponding GR session
        #load per trial response and compute tuning metrics
        GRsessions[0].load_respmat(load_calciumdata=True,calciumversion=calciumversion) 
        GRsessions      = compute_tuning_wrapper(GRsessions)
        #Load SP data:
        sessions[ises].load_data(load_calciumdata=True,calciumversion=calciumversion)

        for metric in tuning_metrics:
            sessions[ises].celldata[metric] = GRsessions[0].celldata[metric]

#%% Keep only those SP sessions that have a corresponding GR session:
sessions        = [ses for ises,ses in enumerate(sessions) if idx_ses[ises]]
sessiondata     = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
nSessions       = len(sessions)
report_sessions(sessions)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions        = compute_pairwise_anatomical_distance(sessions)

# #%% ########################## Compute signal and noise correlations: ###################################
# sessions       = compute_signal_noise_correlation(sessions,filter_stationary=False)

#%% Compute trace correlations:
sessions = compute_trace_correlation(sessions,binwidth=0.5,uppertriangular=False)

#%% Compute noise correlations / covariance after subtracting X modes of covariance:
areas = ['V1','PM']
n_components = 20
fa = FA(n_components=n_components)
zthr = 3 # zscore threshold, remove outliers - Kohn and Smith, 2005, Ruff and Cohen, 2016 
# import numpy.ma as ma

# comps = np.array([0,1,2,3,4,5,6,7,8,9])
comps = np.arange(1,n_components)
# comps = np.array(0,)
# comps = np.arange(2,n_components)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Computing noise correlations'):
    # Compute noise correlations from spontaneous activity:
    data                            = zscore(sessions[ises].calciumdata,axis=0)
    data                            = np.clip(data,-zthr,zthr)
    sessions[ises].trace_corr       = np.corrcoef(data.T)

    fa.fit(data)
    data_T                          = fa.transform(data)
    data_hat                        = np.dot(data_T[:,comps], fa.components_[comps,:])       # Reconstruct data
    sessions[ises].noise_cov        = np.cov(data_hat.T)

#%% #########################################################################################
# Compute average correlation as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

areapairs       = ['V1-V1','PM-PM','V1-PM']
# tuning_metric = 'OSI'
tuning_metric   = 'gOSI'
# tuning_metric   = 'tuning_var'
corr_type       = 'trace_corr'
# corr_type       = 'noise_cov'
extremefrac     = 25

# binedges        = np.arange(0,360+22.5,22.5)-22.5/2
binedges      = np.arange(0,90+45,22.5)-22.5/2
bincenters      = (binedges[1:]+binedges[:-1])/2
data            = np.full((nSessions,len(areapairs),len(binedges)-1,3),np.nan) #for each session, combination of delta pref store the mean noise corr for all and for the top and bottom tuned percentages

# fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    # sessions[ises].delta_pref = np.abs(np.subtract.outer(sessions[ises].celldata['pref_ori'].values,sessions[ises].celldata['pref_ori'].values))

    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    # sessions[ises].delta_pref = np.abs(np.mod(dpref,180))
    # sessions[ises].delta_pref[dpref == 180] = 180
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)
    
    corrdata = getattr(sessions[ises],corr_type).copy()

    for iap,areapair in enumerate(areapairs):
        areafilter      = filter_2d_areapair(sessions[ises],areapair)

        tunefilter_all    = np.ones(areafilter.shape).astype(bool)

        tunefilter_unt   = np.meshgrid(sessions[ises].celldata[tuning_metric]<np.percentile(sessions[ises].celldata[tuning_metric],extremefrac),
                                        sessions[ises].celldata[tuning_metric]<np.percentile(sessions[ises].celldata[tuning_metric],extremefrac))
        tunefilter_unt    = np.logical_and(tunefilter_unt[0],tunefilter_unt[1])

        tunefilter_tune    = np.meshgrid(sessions[ises].celldata[tuning_metric]>np.percentile(sessions[ises].celldata[tuning_metric],100-extremefrac),
                                        sessions[ises].celldata[tuning_metric]>np.percentile(sessions[ises].celldata[tuning_metric],100-extremefrac))
        tunefilter_tune    = np.logical_and(tunefilter_tune[0],tunefilter_tune[1])

        nanfilter         = np.all((~np.isnan(corrdata),~np.isnan(sessions[ises].delta_pref)),axis=0)

        cellfilter = np.all((areafilter,tunefilter_all,nanfilter),axis=0)
        data[ises,iap,:,0] = binned_statistic(x=sessions[ises].delta_pref[cellfilter].flatten(),
                                            values=corrdata[cellfilter].flatten(),statistic='mean',bins=binedges)[0]
        
        cellfilter = np.all((areafilter,tunefilter_unt,nanfilter),axis=0)
        data[ises,iap,:,1] = binned_statistic(x=sessions[ises].delta_pref[cellfilter].flatten(),
                                                values=corrdata[cellfilter].flatten(),statistic='mean',bins=binedges)[0]

        cellfilter = np.all((areafilter,tunefilter_tune,nanfilter),axis=0)
        data[ises,iap,:,2] = binned_statistic(x=sessions[ises].delta_pref[cellfilter].flatten(),
                                                values=corrdata[cellfilter].flatten(),statistic='mean',bins=binedges)[0]


#%% Show tuning dependent noise correlations:
# clrs    = sns.color_palette('inferno', 3)
clrs            = ['black','blue','red']
perc_labels     = ['All','Bottom ' + str(extremefrac) + '%\n tuned','Top ' + str(extremefrac) + '%\n tuned',]

fig,axes = plt.subplots(1,3,figsize=(9.5,3),sharex=True,sharey=True)
for iap,areapair in enumerate(areapairs):
    ax = axes[iap]
    handles = []
    handles.append(shaded_error(ax=ax,x=bincenters,y=data[:,iap,:,0].squeeze(),center='mean',error='sem',color=clrs[0]))
    handles.append(shaded_error(ax=ax,x=bincenters,y=data[:,iap,:,1].squeeze(),center='mean',error='sem',color=clrs[1]))
    handles.append(shaded_error(ax=ax,x=bincenters,y=data[:,iap,:,2].squeeze(),center='mean',error='sem',color=clrs[2]))
    ax.set_xlabel(r'$\Delta$ Pref. Ori')
    ax.set_ylabel('NoiseCorrelation')
    ax.set_title('')
    ax.legend(handles,perc_labels,frameon=False,loc='upper right',fontsize=8)
sns.despine(trim=False,top=True,right=True,offset=3)
ax.set_xticks(bincenters[::2],bincenters[::2].astype(int),rotation=45)
ax.set_ylim([0,0.05])
ax.set_xlim([0,np.max(bincenters)])
plt.tight_layout()
# my_savefig(fig, savedir, 'NC_deltaOri_tuningperc_SP', formats = ['png'])
# plt.savefig(os.path.join(savedir,'PairwiseCorrelations','NC_deltaOri_V1_tuningperc' + '.png'), format = 'png')


#%% #########################################################################################
# Average correlations as a function of anatomical distance per difference in preferred orientation
# for different percentiles of how strongly tuned neurons are
areapairs = ['V1-V1','PM-PM']

extremefrac     = 25
# tuning_metric = 'OSI'
tuning_metric   = 'gOSI'
# corr_type       = 'noise_cov'
corr_type       = 'trace_corr'

# extremefrac     = 75
# tuning_metric   = 'event_rate'

# tuning_metric = 'tuning_var'
dprefs          = np.arange(0,90+22.5,22.5)
binedges        = np.arange(0,600,25)
bincenters      = (binedges[1:]+binedges[:-1])/2

data            = np.full((nSessions,len(areapairs),len(dprefs),len(binedges)-1,3),np.nan) #for each session, combination of delta pref store the mean noise corr for all and for the top and bottom tuned percentages

# fig = plt.subplots(1,3,figsize=(12,4))
for ises in tqdm(range(nSessions),total=nSessions,desc='Averaging correlations across sessions'):
    # sessions[ises].delta_pref = np.abs(np.subtract.outer(sessions[ises].celldata['pref_ori'].values,sessions[ises].celldata['pref_ori'].values))

    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    # sessions[ises].delta_pref = np.abs(np.mod(dpref,180))
    # sessions[ises].delta_pref[dpref == 180] = 180
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)

    corrdata = getattr(sessions[ises],corr_type).copy()

    for idpref,dpref in enumerate(dprefs):

        for iap,areapair in enumerate(areapairs):
            areafilter      = filter_2d_areapair(sessions[ises],areapair)

            tunefilter_all    = np.ones(areafilter.shape).astype(bool)

            tunefilter_unt   = np.meshgrid(sessions[ises].celldata[tuning_metric]<np.percentile(sessions[ises].celldata[tuning_metric],extremefrac),
                                        sessions[ises].celldata[tuning_metric]<np.percentile(sessions[ises].celldata[tuning_metric],extremefrac))
            tunefilter_unt    = np.logical_and(tunefilter_unt[0],tunefilter_unt[1])

            tunefilter_tune    = np.meshgrid(sessions[ises].celldata[tuning_metric]>np.percentile(sessions[ises].celldata[tuning_metric],100-extremefrac),
                                        sessions[ises].celldata[tuning_metric]>np.percentile(sessions[ises].celldata[tuning_metric],100-extremefrac))
            tunefilter_tune    = np.logical_and(tunefilter_tune[0],tunefilter_tune[1])

            dpreffilter         = sessions[ises].delta_pref==dpref

            nanfilter         = np.all((~np.isnan(corrdata),~np.isnan(sessions[ises].delta_pref)),axis=0)

            cellfilter = np.all((areafilter,tunefilter_all,dpreffilter,nanfilter),axis=0)
            data[ises,iap,idpref,:,0] = binned_statistic(x=sessions[ises].distmat_xy[cellfilter].flatten(),
                                                values=corrdata[cellfilter].flatten(),statistic='mean',bins=binedges)[0]
            
            cellfilter = np.all((areafilter,tunefilter_unt,dpreffilter,nanfilter),axis=0)
            data[ises,iap,idpref,:,1] = binned_statistic(x=sessions[ises].distmat_xy[cellfilter].flatten(),
                                                values=corrdata[cellfilter].flatten(),statistic='mean',bins=binedges)[0]

            cellfilter = np.all((areafilter,tunefilter_tune,dpreffilter,nanfilter),axis=0)
            data[ises,iap,idpref,:,2] = binned_statistic(x=sessions[ises].distmat_xy[cellfilter].flatten(),
                                                values=corrdata[cellfilter].flatten(),statistic='mean',bins=binedges)[0]


#%% Show tuning and distance dependent trace correlations:
clrs            = sns.color_palette('inferno_r', len(dprefs))

fig,axes = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*3,3),sharex=True,sharey=True)
for iap,areapair in enumerate(areapairs):
    ax = axes[iap]
    handles = []
    for idpref,dpref in enumerate(dprefs):
        handles.append(shaded_error(ax=ax,x=bincenters,y=data[:,iap,idpref,:,2].squeeze(),
                                    center='mean',error='sem',color=clrs[idpref]))

    # handles.append(shaded_error(ax=ax,x=bincenters,y=data[:,iap,:,1].squeeze(),center='mean',error='std',color=clrs[1]))
    # handles.append(shaded_error(ax=ax,x=bincenters,y=data[:,iap,:,2].squeeze(),center='mean',error='std',color=clrs[2]))
    ax.set_xlabel(r'$\Delta$ XY distance [$\mu m$]')
    if iap==0:
        ax.set_ylabel('Correlation')
    ax.set_title(areapair)
    ax.legend(handles,dprefs,frameon=False,loc='upper right',fontsize=8,title='delta PO')

    datatotest = data[:,iap,:,:,2].squeeze()
    datatotest = datatotest[~np.isnan(datatotest[:,0,0])]
    df = pd.DataFrame({'correlation':   datatotest.flatten(),
                        'session_id':   np.tile(np.arange(np.shape(datatotest)[0]),len(dprefs)*len(bincenters)),
                        'dpref':        np.tile(np.repeat(dprefs,len(bincenters)),np.shape(datatotest)[0]),
                        'distance':     np.repeat(bincenters,np.shape(datatotest)[0]*len(dprefs))
                        })

    model = ols('correlation ~ C(distance) + C(dpref) + C(distance):C(dpref)', data=df).fit()
    result = sm.stats.anova_lm(model, type=2)
    # print(result)
    for itest,test in enumerate(result.index[:-1]):
        pval = result.loc[test,'PR(>F)']
        print('%s (F=%2.2f,p=%1.3f)' % (test,result.loc[test,'F'],pval))
        ax.text(0.4,0.7-0.05*itest,'%s (F=%2.2f,p=%1.3f)' % (test,result.loc[test,'F'],pval),ha='center',va='center',
                transform=ax.transAxes,fontsize=5)
sns.despine(trim=False,top=True,right=True,offset=3)
ax.set_xticks(np.arange(0,1000,100),np.arange(0,1000,100),rotation=45)
ax.set_ylim([0,0.05])
ax.set_xlim([0,np.max(bincenters)])
plt.tight_layout()
# my_savefig(fig, savedir, 'NC_deltaXY_deltaOri_tuningperc_SP_%s' % (tuning_metric), formats = ['png'])
# plt.savefig(os.path.join(savedir,'PairwiseCorrelations','NC_deltaOri_V1_tuningperc' + '.png'), format = 'png')


#%% #########################################################################################
# Contrast: across areas
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ' '
projpairs           = ' '

rf_type             = 'F'
# rf_type             = 'Fsmooth'
corr_type           = 'trace_corr'
# corr_type           = 'noise_cov'
binresolution       = 10
normalize           = False
# normalize = True
absolute            = False
# absolute            = True
shufflefield        = None

[bins2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,binsdRF,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,binsangle] = bin_corr_deltarf_ses(sessions,rf_type=rf_type,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,noise_thr=20,
                        binresolution=binresolution,normalize=normalize,absolute=absolute,
                        shufflefield=shufflefield,r2_thr=0)

#%% Plot radial tuning:
fig = plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses,bin_dist_mean_ses,
                                             areapairs,layerpairs,projpairs,min_counts=500)

# my_savefig(fig,savedir,'RadialTuning_Areas_SP_%s' % (corr_type),formats = ['png'])


#%% Loop over all delta preferred orientations and store mean correlations as well as distribution of pos and neg correlations:
areapairs           = ['V1-V1','PM-PM','V1-PM','PM-V1']
# areapairs           = ['V1-V1']
layerpairs          = ' '
projpairs           = ' '

# deltaoris           = np.array([0,22.5,45,67.5,90])
for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    # sessions[ises].delta_pref = np.abs(np.mod(dpref,180))
    # sessions[ises].delta_pref[dpref == 180] = 180
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)

deltaoris           = np.unique(sessions[0].delta_pref[~np.isnan(sessions[0].delta_pref)])
# deltaoris           = np.array([0,22.5])
ndeltaoris          = len(deltaoris)

rotate_prefori      = True
# tuned_thr           = 0.025
tuned_thr           = 50 #percentile of neurons with gOSI values above this value that are included
rf_type             = 'F'
corr_type           = 'trace_corr'
# corr_type           = 'noise_cov'
noise_thr           = 20
r2_thr              = 0.1
binresolution       = 10

[bincenters_2d,bin_2d_mean_ses,bin_2d_count_ses,bin_dist_mean_ses,bin_dist_count_ses,bincenters_dist,
bin_angle_cent_mean_ses,bin_angle_cent_count_ses,bin_angle_surr_mean_ses,
bin_angle_surr_count_ses,bincenters_angle] = bin_corr_deltarf_ses(sessions,method='mean',areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                            corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                            r2_thr=1,deltaori=None,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

#Init output arrays:
bin_2d_mean_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_posf_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_negf_oris_ses        = np.full((ndeltaoris,*np.shape(bin_2d_mean_ses)),np.nan)
bin_2d_count_oris_ses       = np.full((ndeltaoris,*np.shape(bin_2d_count_ses)),np.nan)

bin_dist_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_dist_mean_ses)),np.nan)
bin_dist_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_dist_count_ses)),np.nan)

bin_angle_cent_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_cent_mean_ses)),np.nan)
bin_angle_cent_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_angle_cent_count_ses)),np.nan)

bin_angle_surr_mean_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_posf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_negf_oris_ses      = np.full((ndeltaoris,*np.shape(bin_angle_surr_mean_ses)),np.nan)
bin_angle_surr_count_oris_ses     = np.full((ndeltaoris,*np.shape(bin_angle_surr_count_ses)),np.nan)

for idOri,deltaori in enumerate(deltaoris):
# for idOri,centerori in enumerate(centeroris):
    [_,bin_2d_mean_oris_ses[idOri,:,:,:,:,:],bin_2d_count_oris_ses[idOri,:,:,:,:,:],
     bin_dist_mean_oris_ses[idOri,:,:,:,:],bin_dist_count_oris_ses[idOri,:,:,:],_,
     bin_angle_cent_mean_oris_ses[idOri,:,:,:,:],bin_angle_cent_count_oris_ses[idOri,:,:,:,:],
     bin_angle_surr_mean_oris_ses[idOri,:,:,:,:],bin_angle_surr_count_oris_ses[idOri,:,:,:,:],_] = bin_corr_deltarf_ses(sessions,
                                                    method='mean',filtersign=None,areapairs=areapairs,layerpairs=layerpairs,
                                                    projpairs=projpairs,corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

    [_,bin_2d_posf_oris_ses[idOri,:,:,:,:,:],_,
     bin_dist_posf_oris_ses[idOri,:,:,:,:],_,_,
     bin_angle_cent_posf_oris_ses[idOri,:,:,:,:],_,
     bin_angle_surr_posf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='pos',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)

    [_,bin_2d_negf_oris_ses[idOri,:,:,:,:,:],_,
     bin_dist_negf_oris_ses[idOri,:,:,:,:],_,_,
     bin_angle_cent_negf_oris_ses[idOri,:,:,:,:],_,
     bin_angle_surr_negf_oris_ses[idOri,:,:,:,:],_,_] = bin_corr_deltarf_ses(sessions,method='frac',filtersign='neg',
                                                    areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                                                    corr_type=corr_type,binresolution=binresolution,rotate_prefori=rotate_prefori,
                                                    r2_thr=r2_thr,deltaori=deltaori,rf_type=rf_type,noise_thr=noise_thr,tuned_thr=tuned_thr)


#%% Compute mean over sessions for each orientation
bin_2d_count_oris = np.nansum(bin_2d_count_oris_ses,1)
bin_2d_mean_oris = np.nanmean(bin_2d_mean_oris_ses,1)
bin_2d_posf_oris = np.nanmean(bin_2d_posf_oris_ses,1)
bin_2d_negf_oris = np.nanmean(bin_2d_negf_oris_ses,1)

bin_dist_count_oris = np.nansum(bin_dist_count_oris_ses,1)
bin_dist_mean_oris = np.nanmean(bin_dist_mean_oris_ses,1)
bin_dist_posf_oris = np.nanmean(bin_dist_posf_oris_ses,1)
bin_dist_negf_oris = np.nanmean(bin_dist_negf_oris_ses,1)

bin_angle_cent_mean_oris = np.nanmean(bin_angle_cent_mean_oris_ses,1)
bin_angle_cent_posf_oris = np.nanmean(bin_angle_cent_posf_oris_ses,1)    
bin_angle_cent_negf_oris = np.nanmean(bin_angle_cent_negf_oris_ses,1)

bin_angle_surr_mean_oris = np.nanmean(bin_angle_surr_mean_oris_ses,1)
bin_angle_surr_posf_oris = np.nanmean(bin_angle_surr_posf_oris_ses,1)    
bin_angle_surr_negf_oris = np.nanmean(bin_angle_surr_negf_oris_ses,1)

#%% Compute mean over sessions for each orientation
bin_2d_count_oris = np.nansum(bin_2d_count_oris_ses,1)
bin_2d_mean_oris = nanweightedaverage(bin_2d_mean_oris_ses,weights=bin_2d_count_oris_ses,axis=1)
bin_2d_posf_oris = nanweightedaverage(bin_2d_posf_oris_ses,weights=bin_2d_count_oris_ses,axis=1)
bin_2d_negf_oris = nanweightedaverage(bin_2d_negf_oris_ses,weights=bin_2d_count_oris_ses,axis=1)

bin_dist_count_oris = np.nansum(bin_dist_count_oris_ses,1)
bin_dist_mean_oris = nanweightedaverage(bin_dist_mean_oris_ses,weights=bin_dist_count_oris_ses,axis=1)
bin_dist_posf_oris = nanweightedaverage(bin_dist_posf_oris_ses,weights=bin_dist_count_oris_ses,axis=1)
bin_dist_negf_oris = nanweightedaverage(bin_dist_negf_oris_ses,weights=bin_dist_count_oris_ses,axis=1)

bin_angle_cent_count_oris = np.nansum(bin_angle_cent_count_oris_ses,1)
bin_angle_cent_mean_oris = nanweightedaverage(bin_angle_cent_mean_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)
bin_angle_cent_posf_oris = nanweightedaverage(bin_angle_cent_posf_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)    
bin_angle_cent_negf_oris = nanweightedaverage(bin_angle_cent_negf_oris_ses,weights=bin_angle_cent_count_oris_ses,axis=1)

bin_angle_surr_count_oris = np.nansum(bin_angle_surr_count_oris_ses,1)
bin_angle_surr_mean_oris = nanweightedaverage(bin_angle_surr_mean_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)
bin_angle_surr_posf_oris = nanweightedaverage(bin_angle_surr_posf_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)    
bin_angle_surr_negf_oris = nanweightedaverage(bin_angle_surr_negf_oris_ses,weights=bin_angle_surr_count_oris_ses,axis=1)

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_mean_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)

axes = fig.get_axes()
#report stats: 
for iap,areapair in enumerate(areapairs):
    ax = axes[iap]
    datatotest = bin_dist_mean_oris_ses[:,:,:,iap,:,:].squeeze()
    datatotest = np.transpose(datatotest,(1,0,2))
    #Now datatotest is an array of shape (nSessions,len(deltaoris),len(bincenters_dist))
    #put it in a dataframe with the right indices for the ANOVA
    df = pd.DataFrame({'correlation':   datatotest.flatten(),
                        'session_id':   np.tile(np.arange(nSessions),len(deltaoris)*len(bincenters_dist)),
                        'dpref':        np.tile(np.repeat(deltaoris,len(bincenters_dist)),nSessions),
                        'distance':     np.repeat(bincenters_dist,nSessions*len(deltaoris))
                        })
    model = ols('correlation ~ C(distance) + C(dpref) + C(distance):C(dpref)', data=df).fit()
    result = sm.stats.anova_lm(model, type=2)
    print(result)

    for itest,test in enumerate(result.index[:-1]):
        pval = result.loc[test,'PR(>F)']
        print('%s (F=%2.2f,p=%1.3f)' % (test,result.loc[test,'F'],pval))
        ax.text(0.4,0.2-0.05*itest,'%s (F=%2.2f,p=%1.3f)' % (test,result.loc[test,'F'],pval),ha='center',va='center',
                transform=ax.transAxes,fontsize=5)
        
my_savefig(fig,savedir,'Collinear_Radial_Tuning_areas_%s_mean_SP' % (corr_type), formats = ['png'])

#%% Show radial tuning for each delta ori:
fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_posf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)

fig = plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count_oris,bin_dist_negf_oris,deltaoris,	
                           areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs)

#%% 
gaussian_sigma      = 1.2
min_counts          = 250
centerthr           = [20,20,20,20]

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_mean_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
my_savefig(fig,savedir,'Collinear_DeltaRF_2D_areas_%s_mean_SP' % (corr_type), formats = ['png'])

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_posf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
my_savefig(fig,savedir,'Collinear_DeltaRF_2D_areas_%s_posf_SP' % (corr_type), formats = ['png'])

#%% Show spatial maps per delta ori for the mean correlation
fig = plot_2D_mean_corr_dori(bin_2d_negf_oris,bin_2d_count_oris,bincenters_2d,deltaoris,areapairs=areapairs,layerpairs=layerpairs,
                        projpairs=projpairs,centerthr=centerthr,min_counts=min_counts,gaussian_sigma=gaussian_sigma,cmap='magma')
my_savefig(fig,savedir,'Collinear_DeltaRF_2D_areas_%s_negf_SP' % (corr_type), formats = ['png'])

#%% Compute collinear selectivity index:
csi_cent_mean_oris =  collinear_selectivity_index(bin_angle_cent_mean_oris,bincenters_angle)
csi_surr_mean_oris =  collinear_selectivity_index(bin_angle_surr_mean_oris,bincenters_angle)    

csi_cent_posf_oris =  collinear_selectivity_index(bin_angle_cent_posf_oris,bincenters_angle)
csi_surr_posf_oris =  collinear_selectivity_index(bin_angle_surr_posf_oris,bincenters_angle)

csi_cent_negf_oris =  collinear_selectivity_index(bin_angle_cent_negf_oris,bincenters_angle)
csi_surr_negf_oris =  collinear_selectivity_index(bin_angle_surr_negf_oris,bincenters_angle)


#%% Plot the CSI values as function of delta ori for the three different areapairs
# fig = plot_csi_deltaori_areas(csi_cent_mean_oris,csi_cent_posf_oris,csi_cent_negf_oris,deltaoris,areapairs)
# fig.savefig(os.path.join(savedir,'Collinear_CSI_Cent_areas_%s' % (corr_type) + '.png'), format = 'png')
fig = plot_csi_deltaori_areas(csi_surr_mean_oris,csi_surr_posf_oris,csi_surr_negf_oris,deltaoris,areapairs)
fig.savefig(os.path.join(savedir,'Collinear_CSI_Surr_areas_%s' % (corr_type) + '.png'), format = 'png')
