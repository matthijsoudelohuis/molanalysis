# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math, os
from loaddata.get_data_folder import get_local_drive
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from scipy.stats import binned_statistic,binned_statistic_2d

from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
from utils.plot_lib import shaded_error
from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *
from utils.rf_lib import smooth_rf

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

##############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE10885','2023_10_23']])
session_list        = np.array([['LPE09830','2023_04_10']])
session_list        = np.array([['LPE09830','2023_04_10'],
                                ['LPE09830','2023_04_12']])
session_list        = np.array([['LPE11086','2024_01_05']])
session_list        = np.array([['LPE09830','2023_04_10'],
                                ['LPE09830','2023_04_12'],
                                ['LPE11086','2024_01_05'],
                                ['LPE10884','2023_10_20'],
                                ['LPE10885','2023_10_19'],
                                ['LPE10885','2023_10_20'],
                                ['LPE10919','2023_11_06']])

#%% Load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['GR'])

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    sessions[ises].load_respmat(calciumversion='deconv')


#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='OSI')
    sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='gOSI')
    sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='tuning_var')
    sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'])

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% ########################### Compute noise correlations: ###################################
sessions = compute_noise_correlation(sessions)

#TODO: make noise corr and pairwise functions attributes of session classes

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_metrics(sessions)

# construct dataframe with all pairwise measurements:
df_allpairs  = pd.DataFrame()

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    tempdf  = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr.flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref.flatten(),
                    'AreaPair': sessions[ises].areamat.flatten(),
                    'DistXYPair': sessions[ises].distmat_xy.flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz.flatten(),
                    'DistRfPair': sessions[ises].distmat_rf.flatten(),
                    'AreaLabelPair': sessions[ises].arealabelmat.flatten(),
                    'LabelPair': sessions[ises].labelmat.flatten()}).dropna(how='all') 
                    #drop all rows that have all nan (diagonal + repeat below daig)
    df_allpairs  = pd.concat([df_allpairs, tempdf], ignore_index=True).reset_index(drop=True)


#%% #####################################
#Show some traces and some stimuli to see responses:
sesidx = 0

# fig = plot_excerpt(sessions[0],trialsel=[2220, 2310],neural_version='raster')
fig = plot_excerpt(sessions[sesidx],trialsel=[2200, 2315],neural_version='traces')
fig.savefig(os.path.join(savedir,'ExploreFigs','Neural_Behavioral_Traces_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

# fig = plot_excerpt(sessions[sesidx],neural_version='traces')
# fig = plot_excerpt(sessions[sesidx],trialsel=None,plot_neural=True,plot_behavioral=True,neural_version='traces')



# idx = filter_nearlabeled(sessions[ises],radius=radius)



#%% #### 

sesidx = 0
fig = plot_PCA_gratings_3D(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA','PCA_3D_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_PCA_gratings_3D(sessions[sesidx],export_animation=True)

fig = plot_PCA_gratings(sessions[sesidx],filter=sessions[sesidx].celldata['redcell'].to_numpy()==1)
fig.savefig(os.path.join(savedir,'PCA','PCA_Gratings_All_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ############################### Show response with and without running #################

thr_still   = 0.5 #max running speed for still trials
thr_moving  = 1 #min running speed for moving trials

nOris       = 16
nCells      = len(celldata)
mean_resp_speedsplit = np.empty((nCells,nOris,2))

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    idx_trials_still = sessions[ises].respmat_runspeed<thr_still
    idx_trials_moving = sessions[ises].respmat_runspeed>thr_moving

    # compute meanresp
    oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
    ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)
    # resp_meanori    = np.empty([N,len(oris)])

    resp_meanori    = np.empty([N,len(oris),2])
    for i,ori in enumerate(oris):
        resp_meanori[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_still)],axis=1)
        resp_meanori[:,i,1] =  np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_moving)],axis=1)
        
    # prefori             = np.argmax(resp_meanori.mean(axis=2),axis=1)
    prefori             = np.argmax(resp_meanori[:,:,0],axis=1)

    resp_meanori_pref       = resp_meanori.copy()
    for n in range(N):
        resp_meanori_pref[n,:,0] = np.roll(resp_meanori[n,:,0],-prefori[n])
        resp_meanori_pref[n,:,1] = np.roll(resp_meanori[n,:,1],-prefori[n])

    # normalize by peak response during still trials
    tempmin,tempmax = resp_meanori_pref[:,:,0].min(axis=1,keepdims=True),resp_meanori_pref[:,:,0].max(axis=1,keepdims=True)
    resp_meanori_pref[:,:,0] = (resp_meanori_pref[:,:,0] - tempmin) / (tempmax - tempmin)
    resp_meanori_pref[:,:,1] = (resp_meanori_pref[:,:,1] - tempmin) / (tempmax - tempmin)

    # resp_meanori_pref
    idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'])
    mean_resp_speedsplit[idx_ses,:,:] = resp_meanori_pref

# ########### Make the figure ##################################################################

redcells    = np.unique(celldata['redcell'])
redcell_labels = ['Nonlabeled','Labeled']
areas       = np.unique(celldata['roi_name'])

fig,axes = plt.subplots(2,2,figsize=(8,8),sharex=True,sharey=True)

for iarea,area in enumerate(areas):
    for ired,redcell in enumerate(redcells):
        ax = axes[iarea,ired]
        idx_neurons = celldata['redcell']==redcell
        idx_neurons = np.logical_and(idx_neurons,celldata['roi_name']==area)
        idx_neurons = np.logical_and(idx_neurons,celldata['tuning_var']>0.05)
        handles = []
        # handles = handles + shaded_error(ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,0],center='mean',error='sem',color='black')
        handles.append(shaded_error(ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,0],center='mean',error='sem',color='black'))
        handles.append(shaded_error(ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,1],center='mean',error='sem',color='red'))
        # ax.plot(oris,np.nanmean(mean_resp_speedsplit[idx_neurons,:,0],axis=0),color='black')
        # plt.fill_between(oris, y-error, y+error)

        # ax.errorbar(oris,np.nanmean(mean_resp_speedsplit[idx_neurons,:,0],axis=0),
                    #  np.nanstd(mean_resp_speedsplit[idx_neurons,:,0],axis=0),color='black')
        # ax.plot(np.nanmean(mean_resp_speedsplit[idx_neurons,:,1],axis=0),color='red')
        ax.legend(handles=handles,labels=['Still','Running'])
        ax.set_xlabel(u'Δ Pref Ori')
        ax.set_xticks(oris)
        ax.set_xticklabels(labels=oris,fontsize=8,rotation='vertical')
        ax.set_ylabel('Normalized Response')
        ax.set_title('%s - %s' % (area,redcell_labels[ired]))
plt.tight_layout()

fig.savefig(os.path.join(savedir,'Tuning','Locomotion_V1PM_LabNonLab_' + str(nSessions) + 'sessions.png'), format = 'png')

#%% ##################### Plot control figure of signal and noise corrs ##############################
sesidx = 0
fig = plt.subplots(figsize=(8,5))
plt.imshow(sessions[sesidx].sig_corr, cmap='coolwarm',
           vmin=np.nanpercentile(sessions[sesidx].sig_corr,15),
           vmax=np.nanpercentile(sessions[sesidx].sig_corr,85))
# plt.xlabel = 'Neurons'
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Signal_Correlation_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plt.figure(figsize=(8,5))
plt.imshow(sessions[sesidx].noise_corr, cmap='coolwarm',
           vmin=np.nanpercentile(sessions[sesidx].noise_corr,5),
           vmax=np.nanpercentile(sessions[sesidx].noise_corr,95))
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Noise_Correlation_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% Plotting Noise Correlation distribution across all pairs:
plt.figure(figsize=(4,3))
handles = sns.kdeplot(data=df_allpairs,x='NoiseCorrelation')
plt.xlim([-0.15,0.4])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorr_distribution_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ##################### Noise correlations within and across areas: #########################
areapairs = ['V1-V1','V1-PM','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)
pairs = [('V1-V1','V1-PM'),('PM-PM','V1-PM')] #for statistics

fig,ax = plt.subplots(figsize=(3,3))
center = df_allpairs.groupby('AreaPair', as_index=False)['NoiseCorrelation'].mean()
handles = sns.barplot(data=center,x='AreaPair',y='NoiseCorrelation',errorbar='ci',palette=clrs_areapairs,
            order = areapairs,hue_order=areapairs)
annotator = Annotator(ax, pairs, data=df_allpairs, x="AreaPair", y='NoiseCorrelation', order=areapairs)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.yticks(np.arange(0, 1, step=0.01)) 
plt.ylim([0,0.05])  
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorr_average_%dsessions' %nSessions + '.png'), format = 'png')

############### Relationship anatomical distance and receptive field distance: ##################
df_withinarea = df_allpairs[(df_allpairs['AreaPair'].isin(['V1-V1','PM-PM'])) & (df_allpairs['DistRfPair'].notna()) & (df_allpairs['DistXYPair'] < 1000)]

g = sns.displot(df_withinarea, x="DistXYPair", y="DistRfPair", binwidth=(2, 2), cbar=True,col="AreaPair")
plt.xlim([0,650])
plt.ylim([0,250])
g.set_axis_labels("Anatomical distance \n (approx um)", "RF distance (deg)")

plt.savefig(os.path.join(savedir,'Corr_anat_rf_distance' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ###################################################################
####### Noise correlations as a function of anatomical distance ####
areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

fig,ax=plt.subplots(figsize=(5,4))
binedges = np.arange(0,1000,10) 
nbins= len(binedges)-1      
binmean = np.zeros((nSessions,len(areapairs),nbins))
handles = []
for iap,areapair in enumerate(areapairs):
    for ises in range(nSessions):
        # filter = sessions[ises].celldata['tuning_var']>0
        # filter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
        filter = sessions[ises].areamat==areapair
        # filter = np.logical_and(filter)
        binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_xy[filter].flatten(),
                                               values=sessions[ises].noise_corr[filter].flatten(),
                        statistic='mean', bins=binedges)[0]
    handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
plt.xlabel(r'Anatomical distance ($\mu$m)')
plt.ylabel('Noise Correlation')
plt.yticks(np.arange(0, 1, step=0.005))
plt.legend(handles,areapairs)
plt.xlim([20,600])
plt.ylim([0.025,0.05])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_anatomdistance_perArea_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ###################################################################

fig,ax = plt.subplots(figsize=(5,4))
sns.histplot(sessions[ises].distmat_rf.flatten(),ax=ax)
ax.set(xlabel='delta RF')
fig.savefig(os.path.join(savedir,'Distribution_deltaRF_%dsessions' %nSessions + '.png'), format = 'png')

#%% ###################################################################
# areapairs = ['V1-V1','V1-PM','PM-PM']
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

fig,ax = plt.subplots(figsize=(5,4))
binedges = np.arange(0,120,5) 
# bincenters = np.arange(0,120,10) 
nbins= len(binedges)-1
binmean = np.zeros((nSessions,len(areapairs),nbins))
handles = []
for iap,areapair in enumerate(areapairs):
    for ises in range(nSessions):
        # filter = sessions[ises].celldata['tuning_var']>0
        # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
        # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0,sessions[ises].celldata['tuning_var']>0)
        signalfilter = np.meshgrid(sessions[ises].celldata['skew']>3,sessions[ises].celldata['skew']>3)
        # signalfilter = np.meshgrid(sessions[ises].celldata['redcell']==0,sessions[ises].celldata['redcell']==0)
        # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
        signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
        areafilter = sessions[ises].areamat==areapair
        filter = np.logical_and(signalfilter,areafilter)
        binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_rf[filter].flatten(),
                                               values=sessions[ises].noise_corr[filter].flatten(),
                        statistic='mean', bins=binedges)[0]

plt.xlabel('Delta RF')
plt.ylabel('NoiseCorrelation')
plt.xlim([-2,120])
plt.ylim([0.025,0.07])
plt.tight_layout()
for iap,areapair in enumerate(areapairs):
    handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
    plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_distRF_%dsessions_' %nSessions + areapair + '.png'), format = 'png')
    # plt.savefig(os.path.join(savedir,'NoiseCorr_distRF_RegressOut_' + areapair + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

plt.legend(handles,areapairs)
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_distRF_%dsessions_' %nSessions + '.png'), format = 'png')

#%% ###################################################################

fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    # filter = sessions[ises].celldata['noise_level']<1
    filter = sessions[ises].celldata['tuning_var']>0.05
    # filter = np.logical_and(filter,sessions[ises].celldata['tuning_var']>0.1)
    # filter = sessions[ises].celldata['tuning_var']>0.1
    df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
                    'AreaPair': sessions[ises].areamat[filter,:].flatten(),
                    'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
                    'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
                    'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

    df['DeltaPrefOri'] = np.mod(df['DeltaPrefOri'],180)
    # df = df[np.isin(df['DeltaPrefOri'],[0,90])]

    deltapreforis = np.sort(df['DeltaPrefOri'].unique())
    clrs_deltaoris = sns.color_palette('inferno', len(deltapreforis))

    df['DistRfPair']    = df['DistRfPair'].round(-1)

    histdata            = df.groupby(['DeltaPrefOri','DistRfPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    for iap,areapair in enumerate(areapairs):
        plt.subplot(1,3,iap+1)
        for idf,deltaori in enumerate(deltapreforis):
            # plt.plot(histdata['DistRfPair'][histdata['AreaPair']==areapair], 
            #      histdata['NoiseCorrelation'][histdata['AreaPair']==areapair],
            #      color=[iap])
            idx = np.logical_and(histdata['AreaPair']==areapair,histdata['DeltaPrefOri']==deltaori)
            plt.plot(histdata['DistRfPair'][idx], 
                 histdata['NoiseCorrelation'][idx],
                 color=clrs_deltaoris[idf])
            
        plt.xlabel('Delta RF')
        plt.ylabel('NoiseCorrelation')
        plt.xlim([-2,120])
        plt.ylim([0.02,0.11])
        plt.title(areapair)
            
    plt.legend(deltapreforis)




##########################################################################################
# Plot noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

# fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    fig = plt.subplots(1,3,figsize=(12,4))
    tuning_perc_labels = np.linspace(0,100,11)
    tuning_percentiles  = np.percentile(sessions[ises].celldata['tuning_var'],tuning_perc_labels)
    clrs_percentiles    = sns.color_palette('inferno', len(tuning_percentiles))

    histdata            = df_allpairs.groupby(['DeltaPrefOri','DistRfPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    for iap,areapair in enumerate(areapairs):
        plt.subplot(1,3,iap+1)
        for ip in range(len(tuning_percentiles)-1):

            # filter = tuning_percentiles[ip] <= sessions[ises].celldata['tuning_var'] <= tuning_percentiles[ip+1]
            filter = np.logical_and(tuning_percentiles[ip] <= sessions[ises].celldata['tuning_var'],
                                    sessions[ises].celldata['tuning_var'] <= tuning_percentiles[ip+1])
            
            df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[np.ix_(filter,filter)].flatten(),
                            'DeltaPrefOri': sessions[ises].delta_pref[np.ix_(filter,filter)].flatten(),
                            'AreaPair': sessions[ises].areamat[np.ix_(filter,filter)].flatten(),
                            'DistXYPair': sessions[ises].distmat_xy[np.ix_(filter,filter)].flatten(),
                            'DistXYZPair': sessions[ises].distmat_xyz[np.ix_(filter,filter)].flatten(),
                            'DistRfPair': sessions[ises].distmat_rf[np.ix_(filter,filter)].flatten(),
                            'LabelPair': sessions[ises].labelmat[np.ix_(filter,filter)].flatten()}).dropna(how='all')

            # # filter = sessions[ises].celldata['tuning_var']>0.1
            # df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
            #                 'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
            #                 'AreaPair': sessions[ises].areamat[filter,:].flatten(),
            #                 'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
            #                 'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
            #                 'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
            #                 'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

            # df['DeltaPrefOri'] = np.mod(df['DeltaPrefOri'],180)

            deltapreforis = np.sort(df['DeltaPrefOri'].unique())
            histdata            = df.groupby(['DeltaPrefOri','AreaPair'], as_index=False)['NoiseCorrelation'].mean()

            # plt.plot(histdata['DistRfPair'][histdata['AreaPair']==areapair], 
            #      histdata['NoiseCorrelation'][histdata['AreaPair']==areapair],
            #      color=[iap])
            # idx = np.logical_and(histdata['AreaPair']==areapair,histdata['DeltaPrefOri']==deltaori)
            idx = histdata['AreaPair']==areapair
            plt.plot(histdata['DeltaPrefOri'][idx], 
                    histdata['NoiseCorrelation'][idx],
                    color=clrs_percentiles[ip])
            
        plt.xlabel('Delta Ori')
        plt.ylabel('NoiseCorrelation')
        # plt.xlim([0,200])
        plt.ylim([-0.02,0.4])
        plt.title(areapair)
        
    plt.legend(tuning_perc_labels[1:])

    plt.savefig(os.path.join(savedir,'NC_deltatuning_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

# plt.hist(df['DistRfPair'])
# sns.histplot(df['DistRfPair'])

#%%  ## Plot negative correlation between dissimilarly strongly tuned neurons 

def plot_noise_pair(ses,sourcell,targetcell):
    oris = np.arange(0,360,22.5)
    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

    fig,axes = plt.subplots(1,3,figsize=(11,4))

    axes[0].plot(oris,ses.resp_meanori[sourcecell],c='blue',linewidth=2)
    axes[0].plot(oris,ses.resp_meanori[targetcell],c='red',linewidth=2)
    axes[0].set_xticks(oris)

    for iori,ori in enumerate(oris):
        idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
        axes[0].scatter(ses.trialdata['Orientation'][idx_ori],ses.respmat[sourcecell,idx_ori],
                        color='blue',s=5,alpha=0.2)
        axes[0].scatter(ses.trialdata['Orientation'][idx_ori],ses.respmat[targetcell,idx_ori],
                        color='red',s=5,alpha=0.2)
        axes[0].get_xticklabels()[iori].set_color(pal[iori])
    axes[0].tick_params(axis='x', labelrotation=90)
    axes[0].set_xlabel('Ori')
    axes[0].set_ylabel('Deconvolved activity')
    axes[0].set_title('Tuning Curves')

    for iori,ori in enumerate(oris):
        idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
        axes[1].scatter(ses.respmat[sourcecell,idx_ori],ses.respmat[targetcell,idx_ori],
                        c=pal[iori],s=5,alpha=0.2)

    ses.respmat[sourcecell,:]
    axes[1].set_xlabel('Activity Neuron 1')
    axes[1].set_ylabel('Activity Neuron 2')
    axes[1].set_title('Act. relationship')
    # axes[1].text(250,250,r'NC = %1.2f' % ses.noise_corr[sourcecell,targetcell])

    for iori,ori in enumerate(oris):
        idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
        axes[2].scatter(ses.respmat_res[sourcecell,idx_ori],ses.respmat_res[targetcell,idx_ori],
                        c=pal[iori],s=5,alpha=0.2)

    ses.respmat[sourcecell,:]
    axes[2].set_xlabel('Residual Neuron 1')
    axes[2].set_ylabel('Residual Neuron 2')
    axes[2].set_title('Noise correlation (r= %1.2f)' % ses.noise_corr[sourcecell,targetcell])
    # axes[2].text(250,250,r'NC = %1.2f' % ses.noise_corr[sourcecell,targetcell])

    return fig

# Find a neuron pair that is strongly tuned, has opposite tuning pref and has negative correlation
ises = 0
idx = sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],95)
N = len(sessions[ises].celldata)
signal_filter = np.full((N,N),False)
signal_filter[np.ix_(idx,idx)] = True
idx = np.all((sessions[ises].noise_corr < -0.1,sessions[ises].delta_pref == 90,signal_filter),axis=0)
sourcecells,targetcells = np.where(idx)
random_cell = np.random.choice(len(sourcecells))
sourcecell,targetcell = sourcecells[random_cell],targetcells[random_cell]

[sessions[ises].resp_meanori,sessions[ises].respmat_res] = mean_resp_oris(sessions[ises])

fig = plot_noise_pair(sessions[ises],sourcecell,targetcell)
fig.savefig(os.path.join(savedir,'NoiseCorrelations','NC_example_orthotuning' + '.png'), format = 'png')

# Find a neuron pair that is strongly tuned, has similar tuning pref and has negative correlation
ises = 0
idx = sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],95)
N = len(sessions[ises].celldata)
signal_filter = np.full((N,N),False)
signal_filter[np.ix_(idx,idx)] = True
idx = np.all((sessions[ises].noise_corr < - 0.1,sessions[ises].delta_pref == 0,signal_filter),axis=0)
sourcecells,targetcells = np.where(idx)
random_cell = np.random.choice(len(sourcecells))
sourcecell,targetcell = sourcecells[random_cell],targetcells[random_cell]

fig = plot_noise_pair(sessions[ises],sourcecell,targetcell)
fig.savefig(os.path.join(savedir,'NoiseCorrelations','NC_example_isotuning2' + '.png'), format = 'png')

#%% Interpolate or smooth RF to get estimate for non perfect fits:

smooth_rf(sessions,sig_thr=0.001,radius=100)


#%% #########################################################################################
# Plot 2D noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

# from utils.RRRlib import *

# X = np.column_stack((sessions[ises].respmat_runspeed,sessions[ises].respmat_videome))
# Y = sessions[ises].respmat.T

# sessions[ises].respmat = regress_out_behavior_modulation(sessions[ises],X,Y,nvideoPCs = 30,rank=2).T

# Recompute noise correlations without setting half triangle to nan
sessions =  compute_noise_correlation(sessions,uppertriangular=False)

rotate_prefori  = True
min_counts      = 500 # minimum pairwise observation to include bin

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap(sessions,binresolution=5,
                                                         rotate_prefori=rotate_prefori,
                                                         rotate_deltaprefori=False)
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

## Show the counts of pairs:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(countsRFmat,vmin=np.percentile(countsRFmat,5),vmax=np.percentile(countsRFmat,99),interpolation='none',extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='counts')
if not rotate_prefori:
    plt.xlabel('delta Azimuth')
    plt.ylabel('delta Elevation')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Counts_%dsessions' %nSessions  + '.png'), format = 'png')
else:
    plt.xlabel('Collinear')
    plt.ylabel('Orthogonal')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Counts_Rotated_%dsessions' %nSessions  + '.png'), format = 'png')

## Show the noise correlation map:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(noiseRFmat_mean,vmin=np.nanpercentile(noiseRFmat_mean,5),vmax=np.nanpercentile(noiseRFmat_mean,95),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='noise correlation')
if not rotate_prefori:
    plt.xlabel('delta Azimuth')
    plt.ylabel('delta Elevation')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_%dsessions' %nSessions  + '.png'), format = 'png')
else:
    plt.xlabel('Collinear')
    plt.ylabel('Orthogonal')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Rotated_%dsessions' %nSessions  + '.png'), format = 'png')

# sns.histplot(celldata['pref_ori'],bins=oris)

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap_perori(sessions,binresolution=5,
                                                                rotate_prefori=True)

## Show the noise correlation map:
fig,axes = plt.subplots(4,4,figsize=(10,10))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(noiseRFmat_mean[i*4+j,:,:],vmin=0.02,vmax=0.07,cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())

noiseRFmat_mean = np.nanmean(noiseRFmat_mean,axis=0)

min_counts = 500
countsRFmat = np.sum(countsRFmat,axis=0)
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

## Show the noise correlation map:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(noiseRFmat_mean,vmin=0.034,vmax=0.082,cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='noise correlation')

#%% #########################################################################################
# Contrast: across areas

areas   = ['V1','PM']

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap_areas(sessions,binresolution=5,
                                                                 rotate_prefori=False,thr_tuned=0.0,
                                                                 thr_rf_p=0.01)

min_counts = 250
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(2,2,figsize=(10,7))
for i in range(2):
    for j in range(2):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean,10),
                         vmax=np.nanpercentile(noiseRFmat_mean,99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(areas[i] + '-' + areas[j])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Interarea_%dsessions' %nSessions  + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Interarea_smooth_%dsessions' %nSessions  + '.png'), format = 'png')

#%% Plot Circular tuning:

# binYaxis = np.arange(start=binrange[0,0],stop=binrange[0,1],step=5)
# binXaxis = np.arange(start=binrange[1,0],stop=binrange[1,1],step=5)
# X,Y = np.meshgrid(binXaxis,binYaxis)

# polar_r         = np.mod(np.arctan2(X,Y)+np.pi/2,np.pi*2)
# polar_theta     = np.sqrt(X**2 + Y**2)

# # plt.imshow(polar_r)
# # plt.imshow(polar_theta)
# polardata = polar_r.flatten()
# noisedata = noiseRFmat_mean[1,1,:,:].flatten()

# [NC_circbin,bin_edges,y] = binned_statistic(x=polardata[~np.isnan(noisedata)],
#                         values = noisedata[~np.isnan(noisedata)],
#                         statistic='mean',bins=np.deg2rad(np.arange(0,360,step=10)))
# plt.plot(bin_edges[:-1]+5/2,NC_circbin)

#%% #########################################################################################
# Contrasts: across areas and projection identity      

[noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions,binresolution=7.5,
                                                                 rotate_prefori=False,thr_tuned=0.00,
                                                                 thr_rf_p=0.05)

min_counts = 250
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],96),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        # axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean,10),
                        #  vmax=np.nanpercentile(noiseRFmat_mean,98),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
        axes[i,j].set_xlim([-75,75])
        axes[i,j].set_ylim([-75,75])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Area_Proj_%dsessions' %nSessions  + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_rotate_smooth_Area_Proj_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Area_rotate_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')

####################################### ####################################### #######################
#################################### LABELED AND UNLABELED ############################################

#%% Plot the tuning parameters for labeled and unlabeled cells

recombinases = celldata['recombinase'].unique()[::-1]
clr_labeled = get_clr_recombinase(recombinases)
pairs = [('non','flp'),('non','cre')] #for statistics

# recombinases = [0,1]
# clr_labeled = get_clr_labeled()
# pairs = [('non','flp'),('non','cre')] #for statistics
# pairs = [(0,1)] #for statistics

tuning_metric = 'tuning_var'
fig,ax = plt.subplots(figsize=(3,3))
# handles = sns.kdeplot(data=celldata,x='tuning_var',hue='redcell',common_norm=False,clip=[0,1],
#                       hue_order=[0,1],palette=clr_labeled)

# handles = sns.barplot(data=celldata,x='redcell',y='OSI',order=recombinases,
                    #   hue_order=recombinases,palette=clr_labeled)
# annotator = Annotator(ax, pairs, data=celldata, x="redcell", y='OSI', order=recombinases)

handles = sns.barplot(data=celldata,x='recombinase',y=tuning_metric,order=recombinases,
                      hue_order=recombinases,palette=clr_labeled)
annotator = Annotator(ax, pairs, data=celldata, x="recombinase", y=tuning_metric, order=recombinases)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.ylabel('Orientation Selectivity')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Labeling_tuning_%dsessions' %nSessions + '.png'), format = 'png')


#%% ################## Noise correlations between labeled and unlabeled cells:  #########################
labelpairs = df_allpairs['LabelPair'].unique()
labelpairs_legend = ['unl-unl','unl-lab','lab-lab']
clrs_labelpairs = get_clr_labelpairs(labelpairs)

areapairs = ['V1-V1','V1-PM','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

plt.figure(figsize=(9,4))
for iap,areapair in enumerate(areapairs):
    ax = plt.subplot(1,3,iap+1)
    areafilter      = df_allpairs['AreaPair']==areapair
    # signalfilter    = np.meshgrid(sessions[ises].celldata['skew']>3,sessions[ises].celldata['skew']>3)
    # signalfilter = np.meshgrid(sessions[ises].celldata['redcell']==0,sessions[ises].celldata['redcell']==0)
    # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
    # signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])
    # filter          = np.logical_and(areafilter,signalfilter)
    filter          = areafilter
    center          = df_allpairs[filter].groupby('LabelPair', as_index=False)['NoiseCorrelation'].mean()['NoiseCorrelation']
    err             = df_allpairs[filter].groupby('LabelPair', as_index=False)['NoiseCorrelation'].sem()['NoiseCorrelation']
    # sns.barplot(data=center,x='LabelPair',y='NoiseCorrelation')
    ax.bar(x=labelpairs,height=center,yerr=err,label=labelpairs_legend,color=clrs_labelpairs)
    ax.set_yticks(np.arange(0, 0.1, step=0.01))
    ax.set_xticklabels(labelpairs_legend)
    ax.set_ylim([0,0.075])
    ax.set_title(areapair)
    ax.set_ylabel('Noise Correlation')

plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorr_labeling_%dsessions' %nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_labeling_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% Cell showing negative noise correlations with increasing fluo in Chan 2 if not curated session
sesidx = 0
filter_area = sessions[sesidx].celldata['roi_name']=='V1'
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,3))
ax1.scatter(sessions[sesidx].celldata['meanF_chan2'][filter_area],
            np.nanmean(sessions[sesidx].noise_corr[filter_area,:],axis=1),alpha = 0.7,c=clrs_areapairs[0],s=5)
ax1.set_xlabel('F Chan2')
ax1.set_ylabel('Noise Correlations')
filter_area = sessions[sesidx].celldata['roi_name']=='PM'
ax2.scatter(sessions[sesidx].celldata['meanF_chan2'][filter_area],
            np.nanmean(sessions[sesidx].noise_corr[filter_area,:],axis=1),alpha = 0.7,c=clrs_areapairs[1],s=5)
ax2.set_xlabel('F Chan2')
# ax2.set_ylabel('Noise Correlations')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorr_FChan2_curated_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%%
sesidx = 0
plt.figure(figsize=(3,3))
plt.scatter(sessions[sesidx].celldata['skew'],
            np.nanmean(sessions[sesidx].noise_corr,axis=0),s=5)

#%% ################### Noise correlations distribution  #########################
labelpairs = df_allpairs['LabelPair'].unique()
labelpairs_legend = ['unl-unl','unl-lab','lab-lab']
clrs_labelpairs = get_clr_labelpairs(labelpairs)

# sns.histplot(data=df,x='NoiseCorrelation',hue='LabelPair',stat='probability',common_bins=False,common_norm=False,fill=False)
plt.figure(figsize=(4,3))
handles = sns.kdeplot(data=df_allpairs,x='NoiseCorrelation',hue='LabelPair',common_norm=False,
            hue_order=labelpairs,palette=clrs_labelpairs)
plt.xlim([-0.15,0.4])
# plt.legend(labelpairs_legend,frameon=False)
plt.legend(labelpairs,frameon=False)
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'NoiseCorr_distribution_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorr_distribution_%d' %nSessions + '.png'), format = 'png')

#%% ############################################################################################
################### Noise correlations as a function of pairwise distance: ####################
############################# Labeled vs unlabeled neurons #######################################

areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

labelpairs = df_allpairs['LabelPair'].unique()
clrs_labelpairs = get_clr_labelpairs(labelpairs)

binedges = np.arange(0,1000,50) 
nbins= len(binedges)-1      
binmean = np.empty((nSessions,len(areapairs),len(labelpairs),nbins))
handles = []
tuningthr = 0
for iap,areapair in enumerate(areapairs):
    for ilp,labelpair in enumerate(labelpairs):
        for ises in range(nSessions):
            areafilter = sessions[ises].areamat==areapair
            labelfilter = sessions[ises].labelmat==labelpair
            # filter = sessions[ises].celldata['tuning_var']>0
            signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>tuningthr,sessions[ises].celldata['tuning_var']>tuningthr)
            signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
            # filter = np.logical_and(areafilter,labelfilter)
            filter = np.all((signalfilter,areafilter,labelfilter),axis=0)
            if filter.any():
                binmean[ises,iap,ilp,:] = binned_statistic(x=sessions[ises].distmat_xy[filter].flatten(),
                                                values=sessions[ises].noise_corr[filter].flatten(),
                            statistic='mean', bins=binedges)[0]

plt.figure(figsize=(6,3))
for iap,areapair in enumerate(areapairs):
    ax = plt.subplot(1,len(areapairs),iap+1)
    for ilp,labelpair in enumerate(labelpairs):
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
        handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_labelpairs[ilp]))
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),
                                    # yerror=binmean[:,iap,ilp,:].squeeze()/5,color=clrs_labelpairs[ilp]))
    ax.set(xlabel=r'Anatomical distance ($\mu$m)',ylabel='Noise Correlation',
           yticks=np.arange(0, 1, step=0.01),xticks=np.arange(0, 600, step=100))
    ax.set(xlim=[10,500],ylim=[0,0.075])
    ax.legend(handles,labelpairs,frameon=False,loc='upper right')
    plt.tight_layout()
    ax.set_title(areapair)
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_anatomdistance_perArea_Labeled_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% ############################################################################################
################### Noise correlations as a function of pairwise delta RF : ####################
############################# Labeled vs unlabeled neurons #######################################

areapairs = ['V1-V1','V1-PM','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

labelpairs = df_allpairs['LabelPair'].unique()
clrs_labelpairs = get_clr_labelpairs(labelpairs)

binedges = np.arange(0,120,10) 
nbins= len(binedges)-1      
binmean = np.zeros((nSessions,len(areapairs),len(labelpairs),nbins))
handles = []
for iap,areapair in enumerate(areapairs):
    for ilp,labelpair in enumerate(labelpairs):
        for ises in range(nSessions):
            areafilter = sessions[ises].areamat==areapair
            labelfilter = sessions[ises].labelmat==labelpair
            filter = np.logical_and(areafilter,labelfilter)

            # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
            # signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
            # filter = np.logical_and(np.logical_and(signalfilter,areafilter),labelfilter)
            binmean[ises,iap,ilp,:] = binned_statistic(x=sessions[ises].distmat_rf[filter].flatten(),
                                                values=sessions[ises].noise_corr[filter].flatten(),
                            statistic='mean', bins=binedges)[0]

plt.figure(figsize=(9,4))
for iap,areapair in enumerate(areapairs):
    ax = plt.subplot(1,3,iap+1)
    for ilp,labelpair in enumerate(labelpairs):
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),
                                    # yerror=binmean[:,iap,ilp,:].squeeze()/5,color=clrs_labelpairs[ilp]))
        handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_labelpairs[ilp]))
    ax.set(xlabel='Delta RF',ylabel='Noise Correlation',
           yticks=np.arange(0, 1, step=0.01),xticks=np.arange(0, 120, step=10))
    ax.set(xlim=[0,120],ylim=[0,0.07])
    ax.legend(handles,labelpairs,frameon=False,loc='upper right')
    plt.tight_layout()
    ax.set_title(areapair)
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_deltaRF_Labeled_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')



sessions[0].celldata['rf_azimuth']
sessions[0].celldata['rf_p']

