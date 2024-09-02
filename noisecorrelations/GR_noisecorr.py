# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic,binned_statistic_2d
from tqdm import tqdm
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
from utils.plot_lib import shaded_error
from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *
from utils.rf_lib import smooth_rf, filter_nearlabeled

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE09830','2023_04_10']])
# session_list        = np.array([['LPE09830','2023_04_10'],
#                                 ['LPE09830','2023_04_12']])
# session_list        = np.array([['LPE11086','2024_01_05']])

#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                # ['LPE11998','2024_05_02'], #GN
                                # ['LPE12013','2024_05_02'], #GN
                                # ['LPE12013','2024_05_07'], #GN
                                ['LPE10919','2023_11_06']]) #GR

#%% Load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM'],session_rf=True)

#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF')

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
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_metrics(sessions)

#%% How are noise correlations distributed

# Compute the average noise correlation for each neuron and store in cell data, loop over sessions:
for ises in range(nSessions):
    sessions[ises].celldata['noise_corr_avg'] = np.nanmean(sessions[ises].noise_corr,axis=1) 
    # sessions[ises].celldata['noise_corr_avg'] = np.nanmean(np.abs(sessions[ises].noise_corr),axis=1) 

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%%  Scatter plot of average noise correlations versus skew:
def plot_corr_NC_var(sessions,vartoplot):
    cdata = []
    plt.figure(figsize=(4,4))
    for ses in sessions:
        plt.scatter(ses.celldata[vartoplot],ses.celldata['noise_corr_avg'],s=6,marker='.',alpha=0.5)
        cdata.append(np.corrcoef(ses.celldata[vartoplot],ses.celldata['noise_corr_avg'])[0,1])
    plt.xlabel(vartoplot)
    plt.ylabel('Avg. NC')
    plt.text(x=np.percentile(ses.celldata[vartoplot],25),y=0.12,s='Mean correlation: %1.3f +- %1.3f' % (np.mean(cdata),np.std(cdata)))
    plt.savefig(os.path.join(savedir,'NoiseCorrelations','%s_vs_NC' % vartoplot + '.png'), format = 'png')

#%%  Scatter plot of average noise correlations versus skew:
plot_corr_NC_var(sessions,vartoplot = 'skew')

#%%  Scatter plot of average noise correlations versus depth:
plot_corr_NC_var(sessions,vartoplot = 'depth')

#%%  Scatter plot of average noise correlations versus tuning variance:
plot_corr_NC_var(sessions,vartoplot = 'tuning_var')

#%%  Scatter plot of average noise correlations versus noise level:
plot_corr_NC_var(sessions,vartoplot = 'noise_level')

#%%  Scatter plot of average noise correlations versus event rate level:
plot_corr_NC_var(sessions,vartoplot = 'event_rate')

#%%  Scatter plot of average noise correlations versus fluorescence channel 2:
plot_corr_NC_var(sessions,vartoplot = 'meanF_chan2')

#%% Show relationships between multiple cell data properties at the same time:
plotvars = ['noise_corr_avg','skew','chan2_prob','depth','redcell','tuning_var','event_rate']
sns.pairplot(data=celldata[plotvars],hue='redcell')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Pairplot_NC' + '.png'), format = 'png')

sns.heatmap(data=celldata[plotvars].corr(),vmin=-1,vmax=1,cmap='bwr')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Heatmap_NC_features' + '.png'), format = 'png')

#%% Show tuning variance vs noise correlation:
clr_labeled = get_clr_labeled()
sns.histplot(data=celldata,x='tuning_var',y='noise_corr_avg',hue='redcell',palette=clr_labeled,
             bins=40,alpha=0.5,cbar=True,cbar_kws={'norm': 1})
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Tuning Variance vs NC' + '.png'), format = 'png')

#%% Plot fraction of visuall responsive: 



# #%%  construct dataframe with all pairwise measurements:
# df_allpairs  = pd.DataFrame()

# for ises in range(nSessions):
#     [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

#     tempdf  = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr.flatten(),
#                     'DeltaPrefOri': sessions[ises].delta_pref.flatten(),
#                     'AreaPair': sessions[ises].areamat.flatten(),
#                     'DistXYPair': sessions[ises].distmat_xy.flatten(),
#                     'DistXYZPair': sessions[ises].distmat_xyz.flatten(),
#                     'DistRfPair': sessions[ises].distmat_rf.flatten(),
#                     'AreaLabelPair': sessions[ises].arealabelmat.flatten(),
#                     'LabelPair': sessions[ises].labelmat.flatten()}).dropna(how='all') 
#                     #drop all rows that have all nan (diagonal + repeat below daig)
#     df_allpairs  = pd.concat([df_allpairs, tempdf], ignore_index=True).reset_index(drop=True)


#%% #### 
sesidx = 3
sesidx = np.where([ses.sessiondata['session_id'][0] == 'LPE09830_2023_04_12' for ses in sessions])[0][0]
fig = plot_PCA_gratings_3D(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA','PCA_3D_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_PCA_gratings_3D(sessions[sesidx],export_animation=True)

fig = plot_PCA_gratings(sessions[sesidx],cellfilter=sessions[sesidx].celldata['redcell'].to_numpy()==1)
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

#%% ########### Make the figure ##################################################################

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
fig,ax = plt.subplots(figsize=(5,4))
for ses in tqdm(sessions,total=len(sessions),desc= 'Kernel Density Estimation for each session: '):
    sns.kdeplot(data=ses.noise_corr.flatten(),ax=ax,label=ses.sessiondata['session_id'][0])
plt.xlim([-0.15,0.4])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_distribution_allSessions.png'), format = 'png')

#%% ##################### Noise correlations within and across areas: #########################
areapairs = ['V1-V1','V1-PM','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)
pairs = [('V1-V1','V1-PM'),('PM-PM','V1-PM')] #for statistics

center = np.zeros((nSessions,len(areapairs)))
for ises in tqdm(range(nSessions),desc= 'Averaging noise correlations within and across areas: '):
    for iap,areapair in enumerate(areapairs):
        areafilter =  filter_2d_areapair(sessions[ises],areapair)
  
        # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.1,sessions[ises].celldata['tuning_var']>0.1)
        # signalfilter = np.meshgrid(sessions[ises].celldata['skew']>3,sessions[ises].celldata['skew']>3)
        # signalfilter = np.meshgrid(sessions[ises].celldata['event_rate']>0.05,sessions[ises].celldata['skew']>0.05)
        # signalfilter = np.meshgrid(sessions[ises].celldata['noise_level']<0.075,sessions[ises].celldata['noise_level']<0.075)
        # signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
        # cellfilter = np.logical_and(areafilter,signalfilter)
        cellfilter = np.logical_and(areafilter,areafilter)
        # center[ises,iap] = np.nanmean(sessions[ises].noise_corr[cellfilter])
        center[ises,iap] = np.nanmean(sessions[ises].trace_corr[cellfilter])
df = pd.DataFrame(data=center,columns=areapairs)

#%% Make a barplot with error bars of the mean NC across sessions conditioned on area pairs:
fig,ax = plt.subplots(figsize=(3,3))
sns.barplot(data=df,errorbar='se',palette=clrs_areapairs,
            order = areapairs,hue_order=areapairs)
sns.stripplot(data=df,color='k',ax=ax,size=3,alpha=0.5,jitter=0.2)

# annotator = Annotator(ax, pairs, data=center, x="AreaPair", y='NoiseCorrelation', order=areapairs)
annotator = Annotator(ax, pairs, data=df.dropna(inplace=False),order=areapairs)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.configure(test='t-test_paired', text_format='star', loc='inside')
annotator.apply_and_annotate()

# plt.yticks(np.arange(0, 1, step=0.01)) 
# plt.ylim([0,0.07])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_average_%dsessions' %nSessions + '.png'), format = 'png')

#%% ###################################################################
####### Noise correlations as a function of anatomical distance ####
areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='noise_corr',normalize=False)

#%% Make the figure per protocol:
fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='noise_corr')
fig.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_anatomdistance_perArea_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ###################################################################
# Show distribution of delta receptive fields across sessions:
# fig,ax = plt.subplots(1,2,figsize=(5,4))
fig,ax = plt.subplots(figsize=(5,4))
for ses in tqdm(sessions,total=len(sessions),desc= 'Histogram of delta RF for each session: '):
    if 'rf_p_Fneu' in ses.celldata:
        sns.histplot(ses.distmat_rf.flatten(),bins=np.arange(-5,250,step=5),ax=ax,
                     fill=False,element="step",stat="percent",alpha=0.8,label=ses.sessiondata['session_id'][0])
ax.set(xlabel='delta RF')
ax.legend(loc='upper right',frameon=False,fontsize=7)
fig.savefig(os.path.join(savedir,'Distribution_deltaRF_%dsessions' %nSessions + '.png'), format = 'png')

#%% ###################################################################
# areapairs = ['V1-V1','V1-PM','PM-PM']
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='noise_corr',normalize=False)

#%% Make the figure per protocol:

fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='noise_corr')


#%% ################ Pairwise trace correlations as a function of pairwise delta RF: #####################
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

[binmean,binedges] =  bin_corr_deltarf(sessions,areapairs,corr_type='noise_corr',normalize=True)

#%% Make the figure:
fig = plot_bin_corr_deltarf(sessions,binmean,binedges,areapairs,corr_type='noise_corr')

fig.savefig(os.path.join(savedir,'TraceCorr_distRF_Protocols_%dsessions_' %nSessions + areapair + '.png'), format = 'png')

#%% 
# idx = filter_nearlabeled(sessions[ises],radius=radius)

#%% ###################################################################

fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    # filter = sessions[ises].celldata['noise_level']<1
    cellfilter = sessions[ises].celldata['tuning_var']>0.05
    # filter = np.logical_and(filter,sessions[ises].celldata['tuning_var']>0.1)
    # filter = sessions[ises].celldata['tuning_var']>0.1
    df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[cellfilter,:].flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref[cellfilter,:].flatten(),
                    'AreaPair': sessions[ises].areamat[cellfilter,:].flatten(),
                    'DistXYPair': sessions[ises].distmat_xy[cellfilter,:].flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz[cellfilter,:].flatten(),
                    'DistRfPair': sessions[ises].distmat_rf[cellfilter,:].flatten(),
                    'LabelPair': sessions[ises].labelmat[cellfilter,:].flatten()}).dropna(how='all')

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
            cellfilter = np.logical_and(tuning_percentiles[ip] <= sessions[ises].celldata['tuning_var'],
                                    sessions[ises].celldata['tuning_var'] <= tuning_percentiles[ip+1])
            
            df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[np.ix_(cellfilter,cellfilter)].flatten(),
                            'DeltaPrefOri': sessions[ises].delta_pref[np.ix_(cellfilter,cellfilter)].flatten(),
                            'AreaPair': sessions[ises].areamat[np.ix_(cellfilter,cellfilter)].flatten(),
                            'DistXYPair': sessions[ises].distmat_xy[np.ix_(cellfilter,cellfilter)].flatten(),
                            'DistXYZPair': sessions[ises].distmat_xyz[np.ix_(cellfilter,cellfilter)].flatten(),
                            'DistRfPair': sessions[ises].distmat_rf[np.ix_(cellfilter,cellfilter)].flatten(),
                            'LabelPair': sessions[ises].labelmat[np.ix_(cellfilter,cellfilter)].flatten()}).dropna(how='all')

            # # filter = sessions[ises].celldata['tuning_var']>0.1
            # df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
            #                 'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
            #                 'AreaPair': sessions[ises].areamat[filter,:].flatten(),
            #                 'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
            #                 'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
            #                 'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
            #                 'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

            # df['DeltaPrefOri'] = np.mod(df['DeltaPrefOri'],180)

            deltapreforis       = np.sort(df['DeltaPrefOri'].unique())
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

# smooth_rf(sessions,sig_thr=0.001,radius=100)


#%% #########################################################################################
# Plot 2D noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

# from utils.RRRlib import *

# X = np.column_stack((sessions[ises].respmat_runspeed,sessions[ises].respmat_videome))
# Y = sessions[ises].respmat.T

# sessions[ises].respmat = regress_out_behavior_modulation(sessions[ises],X,Y,nvideoPCs = 30,rank=2).T

# Recompute noise correlations without setting half triangle to nan
sessions =  compute_signal_noise_correlation(sessions,uppertriangular=False)

rotate_prefori  = False
min_counts      = 500 # minimum pairwise observation to include bin

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap(sessions,binresolution=5,
                                                         rotate_prefori=rotate_prefori,
                                                         rotate_deltaprefori=False)
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

## Show the counts of pairs:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(countsRFmat,vmin=np.percentile(countsRFmat,5),vmax=np.percentile(countsRFmat,99),
               interpolation='none',extent=np.flipud(binrange).flatten())
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
IM = ax.imshow(noiseRFmat_mean,vmin=np.nanpercentile(noiseRFmat_mean,5),vmax=np.nanpercentile(noiseRFmat_mean,95),
               cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
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
        axes[i,j].imshow(noiseRFmat_mean[i*4+j,:,:],vmin=0.02,vmax=0.07,cmap="hot",
                         interpolation="none",extent=np.flipud(binrange).flatten())

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

tuning_metric = 'tuning_var'
fig,ax = plt.subplots(figsize=(3,3))

handles = sns.barplot(data=celldata,x='recombinase',y=tuning_metric,order=recombinases,
                      hue_order=recombinases,palette=clr_labeled)
annotator = Annotator(ax, pairs, data=celldata, x="recombinase", y=tuning_metric, order=recombinases)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.ylabel('Orientation Selectivity')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Labeling_tuning_%dsessions' %nSessions + '.png'), format = 'png')

#%% Plot the tuning parameters for labeled and unlabeled cells, per area

labs = np.array(['unl','lab'])
celldata['area_labeled'] = celldata['roi_name'] + labs[celldata['redcell'].to_numpy().astype(int)]
celldata['area_recombinase'] = celldata['roi_name'] + celldata['recombinase']

area_labeled = celldata['area_labeled'].unique()[::-1]
clr_labeled = get_clr_area_labeled(area_labeled)
pairs = [('V1unl','V1lab'),('PMunl','PMlab')] #for statistics

tuning_metric = 'OSI'
fig,ax = plt.subplots(figsize=(3,3))

handles = sns.barplot(data=celldata,x='area_labeled',y=tuning_metric,order=area_labeled,
                      hue_order=area_labeled,palette=clr_labeled)
annotator = Annotator(ax, pairs, data=celldata, x="area_labeled", y=tuning_metric, order=area_labeled)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.ylabel('Orientation Selectivity')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Area_Labeling_TuningVar_%dsessions' %nSessions + '.png'), format = 'png')


#%% Same but for recombinase with area:
labs = np.array(['unl','lab'])
celldata['area_labeled'] = celldata['roi_name'] + labs[celldata['redcell'].to_numpy().astype(int)]
celldata['area_recombinase'] = celldata['roi_name'] + celldata['recombinase']

area_recombinase = np.sort(celldata['area_recombinase'].unique()[::-1])
recombinases = celldata['recombinase'].unique()[::-1]
clr_labeled = get_clr_recombinase(recombinases)
pairs = [('V1cre','V1non'),('PMcre','PMnon'),('PMflp','PMnon'),('V1flp','V1non')] #for statistics

tuning_metric = 'OSI'
fig,ax = plt.subplots(figsize=(4,3))

handles = sns.barplot(data=celldata,x='area_recombinase',y=tuning_metric,order=area_recombinase,
                      hue_order=area_recombinase,palette=clr_labeled)
annotator = Annotator(ax, pairs, data=celldata, x="area_recombinase", y=tuning_metric, order=area_recombinase)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.ylabel('Orientation Selectivity')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Area_Recombinase_TuningVar_%dsessions' %nSessions + '.png'), format = 'png')

#%% ##################### Noise correlations within and across areas: #########################


#%% ################## Noise correlations between labeled and unlabeled cells:  #########################
areas               = ['V1','PM']
redcells            = [0,1]
redcelllabels       = ['unl','lab']
legendlabels        = np.empty((4,4),dtype='object')

minNcells           = 0

noisemat            = np.full((4,4,nSessions),np.nan)

for ises in tqdm(range(nSessions),desc='Averaging correlations across sessions'):
    # idx_nearfilter = filter_nearlabeled(sessions[ises],radius=100)
    for ixArea,xArea in enumerate(areas):
        for iyArea,yArea in enumerate(areas):
            for ixRed,xRed in enumerate(redcells):
                for iyRed,yRed in enumerate(redcells):

                        idx_source = sessions[ises].celldata['roi_name']==xArea
                        idx_target = sessions[ises].celldata['roi_name']==yArea

                        idx_source = np.logical_and(idx_source,sessions[ises].celldata['redcell']==xRed)
                        idx_target = np.logical_and(idx_target,sessions[ises].celldata['redcell']==yRed)

                        # idx_source = np.logical_and(idx_source,idx_nearfilter)
                        # idx_target = np.logical_and(idx_target,idx_nearfilter)

                        if np.sum(idx_source)>minNcells and np.sum(idx_target)>minNcells:	
                            # noisemat[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(sessions[ises].noise_corr[np.ix_(idx_source, idx_target)])
                            noisemat[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(sessions[ises].trace_corr[np.ix_(idx_source, idx_target)])
                            # noisemat[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(np.abs(sessions[ises].trace_corr[np.ix_(idx_source, idx_target)]))
                        # noisemat[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(sessions[ises].noise_cov[np.ix_(idx_source, idx_target)])
                        
                        legendlabels[ixArea*2 + ixRed,iyArea*2 + iyRed]  = areas[ixArea] + redcelllabels[ixRed] + '-' + areas[iyArea] + redcelllabels[iyRed]

# assuming legendlabels is a 4x4 array
legendlabels_upper_tri = legendlabels[np.triu_indices(4, k=0)]

# assuming noisemat is a 4x4xnSessions array
upper_tri_indices = np.triu_indices(4, k=0)
noisemat_upper_tri = noisemat[upper_tri_indices[0], upper_tri_indices[1], :]

df = pd.DataFrame(data=noisemat_upper_tri.T,columns=legendlabels_upper_tri)

colorder = [0,1,4,7,8,9,2,3,5,6]
legendlabels_upper_tri = legendlabels_upper_tri[colorder]
df = df[legendlabels_upper_tri]

#%% Filter certain protocols:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['protocol'].isin(['GR','GN','IM','SP']))[0]
# sessions_subset = [sessions[i] for i in sessions_in_list]
df = df.loc[sessions_in_list,:]

print('%d areapairs interpolated due to missing data' % np.sum(df.isna().sum(axis=1)/4))
df = df.fillna(df.mean())

#%% Make a barplot with error bars of the mean NC across sessions conditioned on area pairs:

fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se')#,labels=legendlabels_upper_tri)
plt.plot(df.T,linewidth=0.25,c='k',alpha=0.5)	
sns.stripplot(data=df,palette='dark:k',ax=ax,size=3,alpha=0.5,jitter=0.1)
ax.set_xticklabels(labels=legendlabels_upper_tri,rotation=90,fontsize=8)
ax.set_ylim([0,0.15])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'TraceCorr_labeling_areas_Indiv%dsessions' %nSessions + '.png'), format = 'png')

#%% With stats:
fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se')#,labels=legendlabels_upper_tri)
ax.set_xticklabels(labels=legendlabels_upper_tri,rotation=90,fontsize=8)
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

annotator = Annotator(ax, pairs, data=df,order=list(legendlabels_upper_tri))
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.configure(test='t-test_paired', text_format='star', loc='inside')
annotator.apply_and_annotate()
ax.set_ylim([0,0.13])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'TraceCorr_labeling_areas_%dsessions' %nSessions + '.png'), format = 'png')

#%% ############################################################################################
################### Noise correlations as a function of pairwise distance: ####################
############################# Labeled vs unlabeled neurons #######################################

areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

labelpairs = df_allpairs['LabelPair'].unique()
clrs_labelpairs = get_clr_labelpairs(labelpairs)

binedges = np.arange(0,1000,50) 
nbins= len(binedges)-1      
# binmean = np.empty((nSessions,len(areapairs),len(labelpairs),nbins))
binmean = np.full((nSessions,len(areapairs),len(labelpairs),nbins),np.nan)

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

labelpairs = np.unique(sessions[ises].labelmat[sessions[ises].labelmat != ''])
clrs_labelpairs = get_clr_labelpairs(labelpairs)

binedges = np.arange(0,120,10) 
nbins= len(binedges)-1      
# binmean = np.zeros((nSessions,len(areapairs),len(labelpairs),nbins))
binmean = np.full((nSessions,len(areapairs),len(labelpairs),nbins),np.nan)

handles = []
for iap,areapair in enumerate(areapairs):
    for ilp,labelpair in enumerate(labelpairs):
        for ises in range(nSessions):
            areafilter = sessions[ises].areamat==areapair
            labelfilter = sessions[ises].labelmat==labelpair
            cellfilter = np.logical_and(areafilter,labelfilter)

            # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
            # signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
            # filter = np.logical_and(np.logical_and(signalfilter,areafilter),labelfilter)
            if np.any(cellfilter):
                binmean[ises,iap,ilp,:] = binned_statistic(x=sessions[ises].distmat_rf[cellfilter].flatten(),
                                                values=sessions[ises].noise_corr[cellfilter].flatten(),
                            statistic='mean', bins=binedges)[0]


#%% Make the figure:
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
    ax.set(xlim=[0,60],ylim=[0,0.11])
    ax.legend(handles,labelpairs,frameon=False,loc='upper right')
    plt.tight_layout()
    ax.set_title(areapair)
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_deltaRF_Labeled_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')



sessions[0].celldata['rf_azimuth']
sessions[0].celldata['rf_p']

# %% @#















### Legacy plots: 


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

