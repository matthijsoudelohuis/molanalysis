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
try:
    os.chdir('t:\\Python\\molanalysis\\')
except:
    os.chdir('e:\\Python\\molanalysis\\')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from scipy.stats import binned_statistic
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
from utils.plot_lib import shaded_error
from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import compute_noise_correlation

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

##############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE10885','2023_10_23']])
session_list        = np.array([['LPE11086','2024_01_05']])
# session_list        = np.array([['LPE11086','2024_01_05'],
#                                 ['LPE10884','2023_10_20'],
#                                 ['LPE10885','2023_10_19'],
#                                 ['LPE10885','2023_10_23'],
#                                 ['LPE10919','2023_11_06']])

# load_respmat: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=False, 
                                    load_calciumdata=False, load_videodata=False, calciumversion='deconv')
# sessions,nSessions   = filter_sessions(protocols = ['GR'],load_behaviordata=True, 
                                    
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    
    if 'pupil_area' in sessions[ises].videodata:
        sessions[ises].videodata['pupil_area']    = medfilt(sessions[ises].videodata['pupil_area'] , kernel_size=25)
    if 'motionenergy' in sessions[ises].videodata:
        sessions[ises].videodata['motionenergy']  = medfilt(sessions[ises].videodata['motionenergy'] , kernel_size=25)
    sessions[ises].behaviordata['runspeed']   = medfilt(sessions[ises].behaviordata['runspeed'] , kernel_size=51)

    ##############################################################################
    ## Construct trial response matrix:  N neurons by K trials
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

    sessions[ises].respmat_runspeed = compute_respmat(sessions[ises].behaviordata['runspeed'],
                                                      sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
                                                    t_resp_start=0,t_resp_stop=1,method='mean')

    if 'motionenergy' in sessions[ises].videodata:
        sessions[ises].respmat_videome = compute_respmat(sessions[ises].videodata['motionenergy'],
                                                    sessions[ises].videodata['timestamps'],sessions[ises].trialdata['tOnset'],
                                                    t_resp_start=0,t_resp_stop=1,method='mean')

    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')


# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
#                                     load_calciumdata=True, load_videodata=True, calciumversion='deconv')
# # sessions,nSessions   = filter_sessions(protocols = ['GR'],load_behaviordata=True, 
#                                     # load_calciumdata=True, load_videodata=True, calciumversion='deconv')

# for ises in range(nSessions):
#     sessions[ises].videodata['pupil_area']    = medfilt(sessions[ises].videodata['pupil_area'] , kernel_size=25)
#     sessions[ises].videodata['motionenergy']  = medfilt(sessions[ises].videodata['motionenergy'] , kernel_size=25)
#     sessions[ises].behaviordata['runspeed']   = medfilt(sessions[ises].behaviordata['runspeed'] , kernel_size=51)

# ##############################################################################
# ## Construct trial response matrix:  N neurons by K trials
# for ises in range(nSessions):
#     sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
#                                   t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

#     sessions[ises].respmat_runspeed = compute_respmat(sessions[ises].behaviordata['runspeed'],
#                                                       sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
#                                                     t_resp_start=0,t_resp_stop=1,method='mean')

#     sessions[ises].respmat_videome = compute_respmat(sessions[ises].videodata['motionenergy'],
#                                                     sessions[ises].videodata['timestamps'],sessions[ises].trialdata['tOnset'],
#                                                     t_resp_start=0,t_resp_stop=1,method='mean')

#     # delattr(sessions[ises],'calciumdata')
#     # delattr(sessions[ises],'videodata')
#     # delattr(sessions[ises],'behaviordata')

############################ Compute tuning metrics: ###################################

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

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


######################################
#Show some traces and some stimuli to see responses:
sesidx = 0

# fig = plot_excerpt(sessions[0],trialsel=[2220, 2310],neural_version='raster')
fig = plot_excerpt(sessions[sesidx],trialsel=[2200, 2315],neural_version='traces')
fig.savefig(os.path.join(savedir,'ExploreFigs','Neural_Behavioral_Traces_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

# fig = plot_excerpt(sessions[sesidx],neural_version='traces')
# fig = plot_excerpt(sessions[sesidx],trialsel=None,plot_neural=True,plot_behavioral=True,neural_version='traces')

##### 

sesidx = 0
fig = plot_PCA_gratings_3D(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA','PCA_3D_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_PCA_gratings_3D(sessions[sesidx],export_animation=True)

fig = plot_PCA_gratings(sessions[sesidx],filter=sessions[sesidx].celldata['redcell'].to_numpy()==1)
fig.savefig(os.path.join(savedir,'PCA','PCA_Gratings_All_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

################################ Show response with and without running #################

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

############################ Compute noise correlations: ###################################
sessions = compute_noise_correlation(sessions)

###################### Plot control figure of signal and noise corrs ##############################
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
                    'LabelPair': sessions[ises].labelmat.flatten()}).dropna(how='all') 
                    #drop all rows that have all nan (diagonal + repeat below daig)
    df_allpairs  = pd.concat([df_allpairs, tempdf], ignore_index=True).reset_index(drop=True)


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

    # df['DeltaPrefOri'] = np.mod(df['DeltaPrefOri'],180)

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

fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    tuning_perc_labels = np.linspace(0,100,11)
    tuning_percentiles  = np.percentile(sessions[ises].celldata['tuning_var'],tuning_perc_labels)
    clrs_percentiles    = sns.color_palette('inferno', len(tuning_percentiles))

    histdata            = df.groupby(['DeltaPrefOri','DistRfPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    for iap,areapair in enumerate(areapairs):
        plt.subplot(1,3,iap+1)
        for ip in range(len(tuning_percentiles)-1):

            # filter = tuning_percentiles[ip] <= sessions[ises].celldata['tuning_var'] <= tuning_percentiles[ip+1]
            filter = np.logical_and(tuning_percentiles[ip] <= sessions[ises].celldata['tuning_var'],
                                    sessions[ises].celldata['tuning_var'] <= tuning_percentiles[ip+1])
            # filter = sessions[ises].celldata['tuning_var']>0.1
            df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
                            'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
                            'AreaPair': sessions[ises].areamat[filter,:].flatten(),
                            'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
                            'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
                            'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
                            'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

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
        plt.ylim([-0.02,0.12])
        plt.title(areapair)
            
    plt.legend(tuning_perc_labels[1:])

plt.savefig(os.path.join(savedir,'NoiseCorr_tuning_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
# plt.hist(df['DistRfPair'])
# sns.histplot(df['DistRfPair'])

################################################################
plt.figure()
for ises in range(nSessions):
    # filter = sessions[ises].celldata['noise_level']<1
    filter = sessions[ises].celldata['tuning_var']>0.05
    # filter = sessions[ises].celldata['tuning_var']>0.1
    df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
                    'AreaPair': sessions[ises].areamat[filter,:].flatten(),
                    'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
                    'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
                    'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

    df['DistRfPair']    = df['DistRfPair'].round(-1)
    histdata            = df.groupby('DistRfPair', as_index=False)['NoiseCorrelation'].mean()
    histdata            = df.groupby(['DistRfPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    plt.plot(histdata['DistRfPair'][histdata['AreaPair']=='V1-V1'], histdata['NoiseCorrelation'][histdata['AreaPair']=='V1-V1'])
    plt.plot(histdata['DistRfPair'][histdata['AreaPair']=='PM-V1'], histdata['NoiseCorrelation'][histdata['AreaPair']=='PM-V1'])
    plt.plot(histdata['DistRfPair'][histdata['AreaPair']=='PM-PM'], histdata['NoiseCorrelation'][histdata['AreaPair']=='PM-PM'])

plt.xlabel('Delta RF')
plt.ylabel('NoiseCorrelation')
plt.xlim([0,200])
plt.ylim([0,0.1])
plt.savefig(os.path.join(savedir,'NoiseCorr_tuning_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

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

fig,ax = plt.subplots(figsize=(3,3))
# handles = sns.kdeplot(data=celldata,x='tuning_var',hue='redcell',common_norm=False,clip=[0,1],
#                       hue_order=[0,1],palette=clr_labeled)

# handles = sns.barplot(data=celldata,x='redcell',y='OSI',order=recombinases,
                    #   hue_order=recombinases,palette=clr_labeled)
# annotator = Annotator(ax, pairs, data=celldata, x="redcell", y='OSI', order=recombinases)

handles = sns.barplot(data=celldata,x='recombinase',y='OSI',order=recombinases,
                      hue_order=recombinases,palette=clr_labeled)
annotator = Annotator(ax, pairs, data=celldata, x="recombinase", y='OSI', order=recombinases)
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
# plt.savefig(os.path.join(savedir,'NoiseCorr_labeling_%dsessions' %nSessions + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorr_labeling_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

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
for iap,areapair in enumerate(areapairs):
    for ilp,labelpair in enumerate(labelpairs):
        for ises in range(nSessions):
            areafilter = sessions[ises].areamat==areapair
            labelfilter = sessions[ises].labelmat==labelpair
            # filter = sessions[ises].celldata['tuning_var']>0
            # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
            # signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
            filter = np.logical_and(areafilter,labelfilter)
            # filter = np.logical_and(signalfilter,areafilter,labelfilter)
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
    ax.set(xlim=[10,500],ylim=[0,0.05])
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


#%% ##################### Noise correlations within and across areas: #########################

# sns.scatterplot(data=df[df['AreaPair']=='V1-V1'],x='DistPair',y='NoiseCorrelation',size=5)
dfV1 = df[df['AreaPair']=='V1-V1']
sns.lineplot(x=np.round(dfV1['DistPair'],-1),y=dfV1['NoiseCorrelation'],color='b')

dfPM = df[df['AreaPair']=='PM-PM']
sns.lineplot(x=np.round(dfPM['DistPair'],-1),y=dfPM['NoiseCorrelation'],color='g')

plt.xlabel="Pairwise distance (um)"
plt.legend(labels=['V1-V1','PM-PM'])
plt.xlim([-10,600])
plt.ylim([0,0.13])

########################### Noise correlations as a function of pairwise distance: ####################
######################################## Labeled vs unlabeled neurons #################################

plt.figure(figsize=(8,5))
fig, axes = plt.subplots(2,2,figsize=(8,5))

areas = ['V1','PM']

for i,iarea in enumerate(areas):
    for j,jarea in enumerate(areas):
        dfarea = df[df['AreaPair']==iarea + '-' + jarea]
        sns.lineplot(ax=axes[i,j],x=np.round(dfarea['DistPair'],-1),y=dfarea['NoiseCorrelation'],hue=dfarea['LabelPair'])
        axes[i,j].set_xlabel="Pairwise distance (um)"
        # plt.legend(labels=['V1-V1','PM-PM'])
        axes[i,j].set_xlim([-10,200])
        axes[i,j].set_ylim([-0.05,0.2])



