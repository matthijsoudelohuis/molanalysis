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

from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.corr_lib import *
from utils.tuning import *
from utils.gain_lib import * 
from scipy.stats import binned_statistic,binned_statistic_2d

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR

session_list        = np.array([['LPE12223','2024_06_10'], #GR
                                ['LPE10884','2023_10_20']]) #GR
# session_list        = np.array([['LPE12223','2024_06_10']])


sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_areas=['V1','PM']) 

sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM']) 

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% Keep separate instances of the original data and the gain subtracted data:
sessions_orig       = copy.deepcopy(sessions)
sessions_nogain     = copy.deepcopy(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
sessions_orig       = compute_signal_noise_correlation(sessions_orig,filter_stationary=False,uppertriangular=False)
# sessions_nogain     = compute_signal_noise_correlation(sessions_nogain,filter_stationary=False,remove_method='PCA',remove_rank=2)
sessions_nogain     = compute_signal_noise_correlation(sessions_nogain,filter_stationary=False,uppertriangular=False,remove_method='GM')

#%% Show reduction in noise correlation matrix with gain subtraction:
fig,axes = plt.subplots(1,2,figsize=(11,4))
axes[0].imshow(sessions_orig[0].noise_corr,vmin=0,vmax=0.05)
axes[1].imshow(sessions_nogain[0].noise_corr,vmin=0,vmax=0.05)

#%% ########################## Subtract gain model from respmat ###################################
for ises in range(nSessions):
    data_hat                        = pop_rate_gain_model(sessions_nogain[ises].respmat, sessions_nogain[ises].trialdata['Orientation'])
    sessions_nogain[ises].respmat   = sessions_nogain[ises].respmat - data_hat

#%% ############################### Show response with and without running #################

celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

thr_still   = 0.5 #max running speed for still trials
thr_moving  = 1 #min running speed for moving trials

nOris       = 16
nCells      = len(celldata)
mean_resp_speedsplit = np.empty((nCells,nOris,2,2))

for ises in range(nSessions):
    [N,K]           = np.shape(sessions_orig[ises].respmat) #get dimensions of response matrix

    idx_trials_still = sessions_orig[ises].respmat_runspeed<thr_still
    idx_trials_moving = sessions_orig[ises].respmat_runspeed>thr_moving

    # compute meanresp
    oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
    ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)

    meanresp_orig    = np.empty([N,len(oris),2])
    meanresp_nogain  = np.empty([N,len(oris),2])
    for i,ori in enumerate(oris):
        meanresp_orig[:,i,0] = np.nanmean(sessions_orig[ises].respmat[:,np.logical_and(sessions_orig[ises].trialdata['Orientation']==ori,idx_trials_still)],axis=1)
        meanresp_orig[:,i,1] = np.nanmean(sessions_orig[ises].respmat[:,np.logical_and(sessions_orig[ises].trialdata['Orientation']==ori,idx_trials_moving)],axis=1)
        meanresp_nogain[:,i,0] = np.nanmean(sessions_nogain[ises].respmat[:,np.logical_and(sessions_nogain[ises].trialdata['Orientation']==ori,idx_trials_still)],axis=1)
        meanresp_nogain[:,i,1] = np.nanmean(sessions_nogain[ises].respmat[:,np.logical_and(sessions_nogain[ises].trialdata['Orientation']==ori,idx_trials_moving)],axis=1)
        
    prefori                     = np.argmax(meanresp_orig[:,:,0],axis=1)
    meanresp_orig_pref          = meanresp_orig.copy()
    meanresp_nogain_pref        = meanresp_nogain.copy()
    for n in range(N):
        meanresp_orig_pref[n,:,0] = np.roll(meanresp_orig[n,:,0],-prefori[n])
        meanresp_orig_pref[n,:,1] = np.roll(meanresp_orig[n,:,1],-prefori[n])
        meanresp_nogain_pref[n,:,0] = np.roll(meanresp_nogain[n,:,0],-prefori[n])
        meanresp_nogain_pref[n,:,1] = np.roll(meanresp_nogain[n,:,1],-prefori[n])

    # normalize by peak response during still trials
    tempmin,tempmax = meanresp_orig_pref[:,:,0].min(axis=1,keepdims=True),meanresp_orig_pref[:,:,0].max(axis=1,keepdims=True)
    meanresp_orig_pref[:,:,0] = (meanresp_orig_pref[:,:,0] - tempmin) / (tempmax - tempmin)
    meanresp_orig_pref[:,:,1] = (meanresp_orig_pref[:,:,1] - tempmin) / (tempmax - tempmin)
    meanresp_nogain_pref[:,:,0] = (meanresp_nogain_pref[:,:,0] - tempmin) / (tempmax - tempmin)
    meanresp_nogain_pref[:,:,1] = (meanresp_nogain_pref[:,:,1] - tempmin) / (tempmax - tempmin)

    # meanresp_orig_pref
    idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'])
    mean_resp_speedsplit[idx_ses,:,:,0] = meanresp_orig_pref
    mean_resp_speedsplit[idx_ses,:,:,1] = meanresp_nogain_pref

#%% ########### Make the figure ##################################################################
redcells            = np.unique(celldata['redcell'])
redcell_labels      = ['Unl','Lab']
areas               = ['V1','PM']
clrs_areas          = get_clr_areas(areas)

fig,axes = plt.subplots(2,2,figsize=(5,5),sharex=True,sharey=True)
for iarea,area in enumerate(areas):
    for ired,redcell in enumerate(redcells):
        ax = axes[iarea,ired]
        idx_neurons = celldata['redcell']==redcell
        idx_neurons = np.logical_and(idx_neurons,celldata['roi_name']==area)
        idx_neurons = np.logical_and(idx_neurons,celldata['tuning_var']>0.05)
        handles = []
        handles.append(shaded_error(ax=ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,0,0],center='mean',error='sem',color='black'))
        handles.append(shaded_error(ax=ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,1,0],center='mean',error='sem',color='red'))
        if ired==0 and iarea==0:
            ax.legend(handles=handles,labels=['Still','Running'],frameon=False,loc='upper right')
        if iarea==1: 
            ax.set_xlabel(u'Δ Pref Ori')
        ax.set_xticks(oris[::2],oris[::2],rotation=45)
        if ired==0: 
            ax.set_ylabel('Normalized Response')
        ax.set_title('%s - %s' % (area,redcell_labels[ired]))
        ax.set_ylim([0,3.2])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'SharedGain','RunningModulation_V1PM_LabUnl_' + str(nSessions) + 'sessions.png'), format = 'png')

#%% Is the gain modulation similar for labeled and unlabeled, and for V1 and PM?
redcells            = np.unique(celldata['redcell'])
redcell_labels      = ['Unl','Lab']
areas               = ['V1','PM']
mrkrs_redcells      = ['o','+']

fig,ax = plt.subplots(1,1,figsize=(3,3))
for iarea,area in enumerate(areas):
    for ired,redcell in enumerate(redcells):
        # ax = axes[iarea,ired]
        idx_neurons = celldata['redcell']==redcell
        idx_neurons = np.logical_and(idx_neurons,celldata['roi_name']==area)
        idx_neurons = np.logical_and(idx_neurons,celldata['tuning_var']>0.05)
        xdata = np.nanmean(mean_resp_speedsplit[idx_neurons,:,0,0],axis=0)
        ydata = np.nanmean(mean_resp_speedsplit[idx_neurons,:,1,0],axis=0)
        plt.scatter(xdata,ydata,color=clrs_areas[iarea],marker=mrkrs_redcells[ired],label=area + redcell_labels[ired],s=25)
        plt.plot([0,3],[0,3],'grey',ls='--',linewidth=1)
ax.legend(frameon=False,loc='lower right')
ax.set_xlabel('Still (Norm. Response)')
ax.set_ylabel('Running (Norm. Response)')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'SharedGain','Gain_V1PM_LabUnl_' + str(nSessions) + 'sessions.png'), format = 'png')

#%% Subtracting gain removes tuned gain modulation in mean response:
redcells            = np.unique(celldata['redcell'])
redcell_labels      = ['Unl','Lab']
areas               = ['V1','PM']
clrs_areas          = get_clr_areas(areas)

fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharex=True,sharey=True)
for imodel,model in enumerate(['orig','nogain']):
    ax = axes[imodel]
    idx_neurons = celldata['tuning_var']>0.05
    handles = []
    handles.append(shaded_error(ax=ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,0,imodel],center='mean',error='sem',color='black'))
    handles.append(shaded_error(ax=ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,1,imodel],center='mean',error='sem',color='red'))
    if imodel==1:
        ax.legend(handles=handles,labels=['Still','Running'],fontsize=9,frameon=False,loc='upper right')
    ax.set_xlabel(u'Δ Pref Ori')
    ax.set_xticks(oris[::2],oris[::2],rotation=45)
    ax.set_ylabel('Normalized Response')
    ax.set_title(model)
    ax.set_ylim([-0.6,3])
    ax.axhline(0,ls='--',color='grey')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'SharedGain','RunningMod_Gainsub_' + str(nSessions) + 'sessions.png'), format = 'png')

#%% #########################################################################################
# Plot noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

areapairs = ['V1-V1','PM-PM','V1-PM']
data = np.empty((nSessions,len(areapairs),len(oris),3)) #for each session, combination of delta pref store the mean noise corr for all and for the top and bottom tuned percentages

extremefrac = 10
binedges = np.arange(0,360+22.5,22.5)-22.5/2
# fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    sessions_orig[ises].delta_pref = np.abs(np.subtract.outer(sessions_orig[ises].celldata['pref_ori'].values,sessions_orig[ises].celldata['pref_ori'].values))

    for iap,areapair in enumerate(areapairs):
        areafilter      = filter_2d_areapair(sessions_orig[ises],areapair)

        tunefilter_all    = np.ones(areafilter.shape).astype(bool)

        tunefilter_unt   = np.meshgrid(sessions_orig[ises].celldata['tuning_var']<np.percentile(sessions_orig[ises].celldata['tuning_var'],extremefrac),
                                      sessions_orig[ises].celldata['tuning_var']<np.percentile(sessions_orig[ises].celldata['tuning_var'],extremefrac))
        tunefilter_unt    = np.logical_and(tunefilter_unt[0],tunefilter_unt[1])

        tunefilter_tune    = np.meshgrid(sessions_orig[ises].celldata['tuning_var']>np.percentile(sessions_orig[ises].celldata['tuning_var'],100-extremefrac),
                                      sessions_orig[ises].celldata['tuning_var']>np.percentile(sessions_orig[ises].celldata['tuning_var'],100-extremefrac))
        tunefilter_tune    = np.logical_and(tunefilter_tune[0],tunefilter_tune[1])

        nanfilter         = np.all((~np.isnan(sessions_orig[ises].noise_corr),~np.isnan(sessions_orig[ises].delta_pref)),axis=0)

        cellfilter = np.all((areafilter,tunefilter_all,nanfilter),axis=0)
        data[ises,iap,:,0] = binned_statistic(x=sessions_orig[ises].delta_pref[cellfilter].flatten(),
                                              values=sessions_orig[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]
        
        cellfilter = np.all((areafilter,tunefilter_unt,nanfilter),axis=0)
        data[ises,iap,:,1] = binned_statistic(x=sessions_orig[ises].delta_pref[cellfilter].flatten(),
                                              values=sessions_orig[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]

        cellfilter = np.all((areafilter,tunefilter_tune,nanfilter),axis=0)
        data[ises,iap,:,2] = binned_statistic(x=sessions_orig[ises].delta_pref[cellfilter].flatten(),
                                              values=sessions_orig[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]

#%% Show tuning dependent noise correlations:
# clrs    = sns.color_palette('inferno', 3)
clrs    = ['black','blue','red']
perc_labels = ['All','Bottom ' + str(extremefrac) + '%\n tuned','Top ' + str(extremefrac) + '%\n tuned',]

fig,ax = plt.subplots(1,1,figsize=(3.5,3))
iap = 0
handles = []
handles.append(shaded_error(ax=ax,x=oris,y=data[:,iap,:,0].squeeze(),center='mean',error='std',color=clrs[0]))
handles.append(shaded_error(ax=ax,x=oris,y=data[:,iap,:,1].squeeze(),center='mean',error='std',color=clrs[1]))
handles.append(shaded_error(ax=ax,x=oris,y=data[:,iap,:,2].squeeze(),center='mean',error='std',color=clrs[2]))
ax.set_xticks(oris[::2],oris[::2],rotation=45)
ax.set_xlabel('Delta Pref. Ori')
ax.set_ylabel('NoiseCorrelation')
ax.set_ylim([0,my_ceil(np.nanmax(data[:,iap,:,:]),2)])
ax.set_title('')
ax.legend(handles,perc_labels,frameon=False,loc='upper right',fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'PairwiseCorrelations','NC_deltaOri_V1_tuningperc' + '.png'), format = 'png')

#%% Show within and across area tuning dependent correlations:
clrs_areapairs = get_clr_area_pairs(areapairs)

fig,ax = plt.subplots(1,1,figsize=(3.5,3))
handles  = []
for iap,areapair in enumerate(areapairs):
    handles.append(shaded_error(ax=ax,x=oris,y=data[:,iap,:,0].squeeze(),center='mean',error='sem',color=clrs_areapairs[iap]))
ax.set_xlabel('Delta Ori')
ax.set_xticks(oris[::2],oris[::2],rotation=45)
ax.set_ylabel('NoiseCorrelation')
ax.set_ylim([0.01,my_ceil(np.nanmax(data[:,:,:,0]),2)])
ax.set_ylim([0.02,0.08])
ax.set_title('')
ax.legend(handles,areapairs,frameon=False,loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'PairwiseCorrelations','NC_deltaOri_areapairs' + '.png'), format = 'png')

#%% #########################################################################################
data = np.empty((nSessions,len(oris),2)) #for each session, combination of delta pref store the mean noise corr for all and for with and without gain subtraction
binedges = np.arange(0,360+22.5,22.5)-22.5/2
for ises in range(nSessions):
    sessions_orig[ises].delta_pref = np.abs(np.subtract.outer(sessions_orig[ises].celldata['pref_ori'].values,sessions_orig[ises].celldata['pref_ori'].values))
    sessions_nogain[ises].delta_pref = np.abs(np.subtract.outer(sessions_orig[ises].celldata['pref_ori'].values,sessions_orig[ises].celldata['pref_ori'].values))
    nanfilter         = np.all((~np.isnan(sessions_orig[ises].noise_corr),~np.isnan(sessions_orig[ises].delta_pref)),axis=0)

    # cellfilter = np.all((nanfilter),axis=0)
    cellfilter = nanfilter
    data[ises,:,0] = binned_statistic(x=sessions_orig[ises].delta_pref[cellfilter].flatten(),
                                            values=sessions_orig[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]
    data[ises,:,1] = binned_statistic(x=sessions_nogain[ises].delta_pref[cellfilter].flatten(),
                                            values=sessions_nogain[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]

#%% Subtracting gain removes tuning-dependent noise correlations:
clrs = sns.color_palette('husl', 2)
clrs = ['black','grey']
fig,ax = plt.subplots(1,1,figsize=(3.5,3))

handles = []
for imodel,model in enumerate(['orig','no gain']):
    handles.append(shaded_error(ax=ax,x=oris,y=data[:,:,imodel].squeeze(),center='mean',error='std',
                                color=clrs[imodel],label=model))
ax.set_xlabel('Delta Ori')
ax.set_xticks(oris[::2],oris[::2],rotation=45)
ax.set_ylabel('NoiseCorrelation')
ax.set_ylim([0,my_ceil(np.nanmax(data),2)])
ax.set_title('')
ax.legend(frameon=False,loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'PairwiseCorrelations','NC_deltaOri_subgainmodel' + '.png'), format = 'png')

#%% 






# #%%  ## Plot negative correlation between dissimilarly strongly tuned neurons 

# def plot_noise_pair(ses,sourcell,targetcell):
#     oris = np.arange(0,360,22.5)
#     pal = sns.color_palette('husl', len(oris))
#     pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

#     fig,axes = plt.subplots(1,3,figsize=(11,4))

#     axes[0].plot(oris,ses.meanresp_orig[sourcecell],c='blue',linewidth=2)
#     axes[0].plot(oris,ses.meanresp_orig[targetcell],c='red',linewidth=2)
#     axes[0].set_xticks(oris)

#     for iori,ori in enumerate(oris):
#         idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
#         axes[0].scatter(ses.trialdata['Orientation'][idx_ori],ses.respmat[sourcecell,idx_ori],
#                         color='blue',s=5,alpha=0.2)
#         axes[0].scatter(ses.trialdata['Orientation'][idx_ori],ses.respmat[targetcell,idx_ori],
#                         color='red',s=5,alpha=0.2)
#         axes[0].get_xticklabels()[iori].set_color(pal[iori])
#     axes[0].tick_params(axis='x', labelrotation=90)
#     axes[0].set_xlabel('Ori')
#     axes[0].set_ylabel('Deconvolved activity')
#     axes[0].set_title('Tuning Curves')

#     for iori,ori in enumerate(oris):
#         idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
#         axes[1].scatter(ses.respmat[sourcecell,idx_ori],ses.respmat[targetcell,idx_ori],
#                         c=pal[iori],s=5,alpha=0.2)

#     ses.respmat[sourcecell,:]
#     axes[1].set_xlabel('Activity Neuron 1')
#     axes[1].set_ylabel('Activity Neuron 2')
#     axes[1].set_title('Act. relationship')
#     # axes[1].text(250,250,r'NC = %1.2f' % ses.noise_corr[sourcecell,targetcell])

#     for iori,ori in enumerate(oris):
#         idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
#         axes[2].scatter(ses.respmat_res[sourcecell,idx_ori],ses.respmat_res[targetcell,idx_ori],
#                         c=pal[iori],s=5,alpha=0.2)

#     ses.respmat[sourcecell,:]
#     axes[2].set_xlabel('Residual Neuron 1')
#     axes[2].set_ylabel('Residual Neuron 2')
#     axes[2].set_title('Noise correlation (r= %1.2f)' % ses.noise_corr[sourcecell,targetcell])
#     # axes[2].text(250,250,r'NC = %1.2f' % ses.noise_corr[sourcecell,targetcell])

#     return fig

# # Find a neuron pair that is strongly tuned, has opposite tuning pref and has negative correlation
# ises = 0
# idx = sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],95)
# N = len(sessions[ises].celldata)
# signal_filter = np.full((N,N),False)
# signal_filter[np.ix_(idx,idx)] = True
# idx = np.all((sessions[ises].noise_corr < -0.1,sessions[ises].delta_pref == 90,signal_filter),axis=0)
# sourcecells,targetcells = np.where(idx)
# random_cell = np.random.choice(len(sourcecells))
# sourcecell,targetcell = sourcecells[random_cell],targetcells[random_cell]

# [sessions[ises].meanresp_orig,sessions[ises].respmat_res] = mean_resp_oris(sessions[ises])

# fig = plot_noise_pair(sessions[ises],sourcecell,targetcell)
# fig.savefig(os.path.join(savedir,'NoiseCorrelations','NC_example_orthotuning' + '.png'), format = 'png')

# # Find a neuron pair that is strongly tuned, has similar tuning pref and has negative correlation
# ises = 0
# idx = sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],95)
# N = len(sessions[ises].celldata)
# signal_filter = np.full((N,N),False)
# signal_filter[np.ix_(idx,idx)] = True
# idx = np.all((sessions[ises].noise_corr < - 0.1,sessions[ises].delta_pref == 0,signal_filter),axis=0)
# sourcecells,targetcells = np.where(idx)
# random_cell = np.random.choice(len(sourcecells))
# sourcecell,targetcell = sourcecells[random_cell],targetcells[random_cell]

# fig = plot_noise_pair(sessions[ises],sourcecell,targetcell)
# fig.savefig(os.path.join(savedir,'NoiseCorrelations','NC_example_isotuning2' + '.png'), format = 'png')


