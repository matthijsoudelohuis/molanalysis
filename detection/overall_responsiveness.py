# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
os.chdir('e:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

#import personal modules
from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive
from utils.psth import *
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.plot_lib import * # get support functions for plotting
from detection.plot_neural_activity_lib import *
from detection.example_cells import get_example_cells
plt.rcParams['svg.fonttype'] = 'none'

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\')

#%% ###############################################################
protocol            = 'DN'
calciumversion      = 'deconv'
# calciumversion      = 'dF'

sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE11998'],min_cells=100,
                           load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

sessions,nSessions = filter_sessions(protocol,min_cells=100,
                           load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE10884'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

#%% Z-score calcium data:
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 60   #post cm
binsize     = 5     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')

    ## Compute average response in stimulus response zone:
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    # temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    # sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
    #                                 sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

#%% #################### Compute activity for each stimulus type for all session ##################
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

N           = len(celldata) #get number of cells total
S           = len(sbins) #get number of spatial bins

stimtypes   = ['C','N','M']
stimlabels  = ['catch','noise','max']
tt_mean     = np.empty([N,S,len(stimtypes)])

for ises,ses in enumerate(sessions):
    idx = celldata['session_id']==ses.sessiondata['session_id'][0]
    for iTT in range(len(stimtypes)):
        # tt_mean[idx,:,iTT] = np.nanmean(sessions[ises].stensor[:,sessions[ises].trialdata['stimcat'] == stimtypes[iTT],:],axis=1)

        trialidx = ses.trialdata['stimcat'] == stimtypes[iTT]
        # trialidx = np.logical_and(trialidx,ses.trialdata['engaged']==1)
        tt_mean[idx,:,iTT] = np.nanmean(sessions[ises].stensor[:,trialidx,:],axis=1)

#%% get session info
uanimals        = np.unique(sessiondata['animal_id'])
nanimals        = len(uanimals)
celldata['animal_id'] = celldata['session_id'].str[:8]

#%% Plot for all loaded sessions together:
fig = plot_snake_allareas(tt_mean,sbins,celldata['roi_name'],trialtypes=stimlabels,sort='stimwin')
fig.savefig(os.path.join(savedir,'SpatialActivity','ActivityInCorridor_perArea_%d' % nanimals + '.png'), format = 'png',bbox_inches='tight')

#%% Plot for different animals:
for ianimal,uanimal in enumerate(uanimals):
    idx     = celldata['animal_id']==uanimal

    fig = plot_snake_allareas(tt_mean[idx,:,:],sbins,celldata['roi_name'][idx],trialtypes=stimlabels,sort='stimwin')
    plt.suptitle(uanimal,fontsize=15,y=0.96)
    # plt.savefig(os.path.join(savedir,'ActivityInCorridor_perStim_' + uanimal + '.svg'), format = 'svg')
    fig.savefig(os.path.join(savedir,'SpatialActivity','ActivityInCorridor_perStim_' + uanimal + '.png'), format = 'png',bbox_inches='tight')

#%% Plot for all animals:
fig = plot_snake_allanimals(tt_mean,sbins,celldata['animal_id'],trialtypes=stimlabels,sort='stimwin')
fig.savefig(os.path.join(savedir,'SpatialActivity','ActivityInCorridor_perAnimal_%d' % nanimals + '.png'), format = 'png',bbox_inches='tight')



#%% ################## Number of responsive neurons per stimulus #################################

sessions        = calc_stimresponsive_neurons(sessions,sbins)
celldata        = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Plot number of responsive neurons per stimulus per area:

areas = ['V1','PM','AL','RSP']
clr_areas = get_clr_areas(areas)

frac_N = celldata.groupby(['roi_name','session_id'])['sig_N'].sum() / celldata.groupby(['roi_name','session_id'])['sig_N'].count()
frac_N = frac_N.reset_index()

frac_M = celldata.groupby(['roi_name','session_id'])['sig_M'].sum() / celldata.groupby(['roi_name','session_id'])['sig_M'].count()
frac_M = frac_M.reset_index()


fig,axes = plt.subplots(1,2,figsize=(5,3),sharey=True, sharex=True)
ax = axes[0]
sns.stripplot(x='roi_name', y='sig_N', data=frac_N, order=areas,hue_order=areas,
              color='k',s=2,ax=ax)
sns.pointplot(x='roi_name', y='sig_N', data=frac_N, order=areas,hue_order=areas,
              errorbar=('ci', 95),palette=clr_areas,ax=ax,capsize=.0,estimator=np.mean)
ax.set_title('Noise Stimulus')
ax.set_xlabel('Area')
ax.set_ylabel('% responsive')

ax = axes[1]
sns.stripplot(x='roi_name', y='sig_M', data=frac_M, order=areas,hue_order=areas,
              color='k',s=2,ax=ax)
sns.pointplot(x='roi_name', y='sig_M', data=frac_M, order=areas,hue_order=areas,
              errorbar=('ci', 95),palette=clr_areas,ax=ax,capsize=.0,estimator=np.mean)
ax.set_title('Max Stimulus')
ax.set_xlabel('Area')
ax.set_ylabel('% responsive')
plt.tight_layout()

fig.savefig(os.path.join(savedir,'SpatialActivity','FracResponsive_perStim_%d' % nanimals + '.png'), format = 'png',bbox_inches='tight')


#%% Plot number of responsive neurons per stimulus per area for labeled and unlabeled:

ax_areas        = ['V1','V1','PM','PM']
ax_stim         = ['N','M','N','M']
ax_stimlabels   = ['Noise','Max','Noise','Max']

labeled         = ['unl','lab']
clr_labeled     = get_clr_labeled()
min_nlabcells   = 5

fig,axes = plt.subplots(1,4,figsize=(6,3),sharey=True, sharex=True)

for iax,(ax,ar,st) in enumerate(zip(axes,ax_areas,ax_stim)):

    nlabcells = celldata[np.logical_and(celldata['roi_name']==ar,celldata['labeled']=='lab')].groupby('session_id')['sig_'+st].count()
    nlabcells = nlabcells.reset_index()
    idx_ses = nlabcells['session_id'][nlabcells['sig_'+st]>=min_nlabcells]

    frac = celldata[np.logical_and(celldata['roi_name']==ar,celldata['session_id'].isin(idx_ses))].groupby(['session_id','labeled'])['sig_'+st].sum() / \
        celldata[np.logical_and(celldata['roi_name']==ar,celldata['session_id'].isin(idx_ses))].groupby(['session_id','labeled'])['sig_'+st].count().unstack(fill_value=0).stack()
    frac = frac.reset_index()
    frac.columns = ['session_id','labeled','sig']

    sns.stripplot(x='labeled', y='sig' ,hue='labeled', data=frac, s=3,jitter=True, 
                  palette=clr_labeled,ax=ax,order=labeled,legend=None,hue_order=labeled)
    g = sns.pointplot(x='labeled', y='sig', hue='labeled',data=frac, ax=ax,order=labeled,
                  hue_order=labeled,palette=clr_labeled,
                  errorbar=('ci', 95),capsize=.0,estimator=np.mean,markers=['o', 'o'])
    
    # g = sns.pointplot(x='labeled', y='sig', color='k',data=frac, ax=ax,order=labeled,
    #             #   hue_order=labeled,palette='grey',
    #               errorbar=('ci', 95),capsize=.0)
    g.get_legend().remove()
    
    stat,pval = stats.ttest_rel(frac[frac['labeled']=='unl']['sig'],frac[frac['labeled']=='lab']['sig'])
    # stat,pval = stats.wilcoxon(frac[frac['labeled']=='unl']['sig'],frac[frac['labeled']=='lab']['sig'])
    ax.annotate('%sp = %0.3f' % (get_sig_asterisks(pval),pval), xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center', size=9)
    ax.set_title('%s - %s' % (ar,ax_stimlabels[iax]),fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('% responsive')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(savedir,'SpatialActivity','FracResponsive_Labeled_%dsessions' % nSessions + '.png'), format = 'png',bbox_inches='tight')

#%% Get signal as relative to psychometric curve for all sessions:
sessions = noise_to_psy(sessions,filter_engaged=True)

#%% #################### Compute mean activity for saliency trial bins for all sessions ##################

labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

lickresp    = [0,1]
nlickresp   = len(lickresp)

nbins_noise = 5
Z           = nbins_noise + 2

edges       = np.linspace(-2,2,nbins_noise+1)
centers     = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
plotcenters = np.hstack((centers[0]-2*np.mean(np.diff(centers)),centers,centers[-1]+2*np.mean(np.diff(centers))))

S           = len(sbins)

data_spatial = np.full((S,Z,nlickresp,nareas,nlabels,nSessions),np.nan)
data_mean    = np.full((Z,nlickresp,nareas,nlabels,nSessions),np.nan)

min_ncells  = 5
min_ntrials = 5
for ises,ses in enumerate(sessions):
    print(f"\rComputing mean activity for noise trial bins for session {ises+1} / {len(sessions)}",end='\r')
    for ilab,label in enumerate(labeled):
        for iarea, area in enumerate(areas):
            # idx_N_resp = np.logical_or(sessions[ises].celldata['sig_N'],sessions[ises].celldata['sig_M'])
            # idx_N     = np.all((sessions[ises].celldata['roi_name']==area,
            #                     sessions[ises].celldata['labeled']==label,
            #                     idx_N_resp), axis=0)
            idx_N     = np.all((sessions[ises].celldata['roi_name']==area,
                    sessions[ises].celldata['labeled']==label), axis=0)
            if np.sum(idx_N)>=min_ncells:
                for ilr,lr in enumerate(lickresp):
                    #Catch trials
                    idx_T           = np.all((sessions[ises].trialdata['signal']==0, 
                                              sessions[ises].trialdata['lickResponse']==lr,
                                              sessions[ises].trialdata['engaged']==1), axis=0)
                    data_spatial[:,0,ilr,iarea,ilab,ises]   = np.nanmean(np.nanmean(sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)
                    data_mean[0,ilr,iarea,ilab,ises]        = np.nanmean(np.nanmean(sessions[ises].respmat[np.ix_(idx_N,idx_T)],axis=0),axis=0)
                    #Max trials
                    idx_T           = np.all((sessions[ises].trialdata['signal']==100,
                                              sessions[ises].trialdata['lickResponse']==lr,
                                              sessions[ises].trialdata['engaged']==1), axis=0)
                    data_spatial[:,-1,ilr,iarea,ilab,ises]  = np.nanmean(np.nanmean(sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)
                    data_mean[-1,ilr,iarea,ilab,ises]       = np.nanmean(np.nanmean(sessions[ises].respmat[np.ix_(idx_N,idx_T)],axis=0),axis=0)

                    for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                        idx_T           = np.all((sessions[ises].trialdata['signal_psy']>=low,
                                                sessions[ises].trialdata['signal_psy']<=high,
                                                sessions[ises].trialdata['lickResponse']==lr,
                                              sessions[ises].trialdata['engaged']==1), axis=0)
                        if np.sum(idx_T)>=min_ntrials:
                            data_spatial[:,ibin+1,ilr,iarea,ilab,ises]  = np.nanmean(np.nanmean(sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)
                            
                            data_mean[ibin+1,ilr,iarea,ilab,ises]       = np.nanmean(np.nanmean(sessions[ises].respmat[np.ix_(idx_N,idx_T)],axis=0),axis=0)

#%% Construct color panel for saliency trial bins
plotlabels = ['catch'] + [str(x) for x in np.round(centers,2)] + ['max']
plotcolors = ['black']  # Start with black
plotcolors += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
plotcolors.append('orange')  # Add orange at the end
plotlines = ['--','-']


#%% ############################### Plot neuron-average activity per stim #################################

plotdata = np.nanmean(data_spatial[:,:,:,:,0,:],axis=(2,3,4))

fig,ax = plt.subplots(1,1,figsize=(1*3,1*2.5),sharex=True,sharey=True)
for iZ in range(Z):
    ax.plot(sbins, plotdata[:,iZ], color=plotcolors[iZ], label=plotlabels[iZ],linewidth=2)
ax.set_ylim([-0.05,0.25])
# ax.axhline(0, color='grey', linewidth=1, linestyle='--')
ax.legend(frameon=False,fontsize=8)
ax.set_xlim([-60,60])
ax.set_title('Average activity per stim')
ax.set_xlabel('Position relative to stim (cm)')
ax.set_ylabel('Activity (z)')
ax.set_yticks([0,0.1,0.2])
# add_stim_resp_win(ax)
plt.savefig(os.path.join(savedir,'Spatial_perSaliency_allAreas_%dsessions' % nSessions + '.png'), format = 'png')

#%% ############################### Plot spatial neuron-average per stim per area #################################

fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*3,nlabels*2.5),sharex=True,sharey=True)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax          = axes[ilab,iarea]
        for ilr,lr in enumerate(lickresp):
            plotdata    = np.nanmean(data_spatial[:,:,ilr,iarea,ilab,:],axis=(2))
            if np.any(~np.isnan(plotdata)):
                for iZ in range(Z):
                    ax.plot(sbins, plotdata[:,iZ], color=plotcolors[iZ], label=plotlabels[iZ],linewidth=2,linestyle=plotlines[ilr])

        if not np.any(ax.get_legend_handles_labels()):
            ax.axis('off')
        else: 
            add_stim_resp_win(ax)
        # ax.set_ylim([-0.05,0.35])
        # if ilab == 0 and iarea == 0:
            # ax.legend(frameon=False,fontsize=6)
        ax.set_xlim([-60,60])
        if ilab == 0:
            ax.set_title(area)
        if ilab == 1:
            ax.set_xlabel('Position relative to stim (cm)')
        if iarea==0:
            ax.set_ylabel('Activity (z)')
            ax.set_yticks([0,0.1,0.2,0.3])
        if iarea == 0 and ilab == 0: 
            leg1 = ax.legend([plt.Line2D([0], [0], color=c, lw=1.5) for c in plotcolors], 
                         plotlabels, frameon=False,fontsize=7,loc='upper left',title='Saliency')
            ax.add_artist(leg1)
        if iarea == 0 and ilab == 1: 
            leg2 = ax.legend([plt.Line2D([0], [0], color='k', lw=1.5,ls=l) for l in plotlines],
                                ['Hit','Miss'], frameon=False,fontsize=7,loc='upper left',title='Response')
            # ax.add_artist(leg1)
        
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Spatial_perSaliency_responsiveNeurons_arealabels_%dsessions' % nSessions + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'Spatial_perSaliency_allNeurons_arealabels_%dsessions' % nSessions + '.png'), format = 'png')

#%% ############################### Plot stimwin neuron-average per stim per area #################################

linecolors = ['red','blue']
fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*3,nlabels*2.5),sharex=True,sharey=True)
clrs_areas = get_clr_areas(areas)

for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax          = axes[ilab,iarea]
        for ilr,lr in enumerate(lickresp):
            # plotdata    = np.nanmean(data_mean[:,ilr,iarea,ilab,:],axis=1)
            # ax.plot(plotcenters, plotdata, linewidth=2,linestyle=plotlines[ilr],color='k')
            plotdata    = data_mean[:,ilr,iarea,ilab,:]
            if np.any(~np.isnan(plotdata)):
                x = plotcenters[np.any(~np.isnan(plotdata),axis=1)]
                y = plotdata[np.any(~np.isnan(plotdata),axis=1),:].T
                h = shaded_error(x, y, error='sem',linestyle=plotlines[ilr],color=clrs_areas[iarea],ax=ax)
            # ax.plot(plotcenters, plotdata[:], color=plotcolors[iZ], label=plotlabels[iZ],linewidth=2,linestyle=plotlines[ilr])
        if not np.any(~np.isnan(plotdata)):
            ax.axis('off')
        
        ax.set_ylim([-0.025,0.3])
        ax.set_xticks(plotcenters,plotlabels)
        if ilab == 0:
            ax.set_title(area,fontsize=12)
        if np.any(~np.isnan(plotdata)):
            ax.text(0.5, 0.9, label, ha='center', transform=ax.transAxes,fontsize=10)
        
        if iarea==0:
            ax.set_ylabel('Activity (z)')
            ax.set_yticks([0,0.1,0.2,0.3])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'StimResponse_Saliency_neuronAverage_arealabels_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'ActivityInCorridor_deconv_neuronAverage_perStim_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% 
linecolors = ['red','blue']
fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*2.5,nlabels*2.5),sharex=True,sharey=True)

for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax          = axes[ilab,iarea]
        for ises, session in enumerate(sessions):
            for ilr,lr in enumerate(lickresp):
                # plotdata    = np.nanmean(data_mean[:,ilr,iarea,ilab,:],axis=1)
                plotdata    = data_mean[:,ilr,iarea,ilab,ises]
                x = plotcenters[~np.isnan(plotdata)]
                y = plotdata[~np.isnan(plotdata)]
                ax.plot(x, y, linewidth=1,linestyle=plotlines[ilr],color=clrs_areas[iarea])

        ax.set_ylim([-0.1,0.45])
        # if ilab == 0 and iarea == 0:
            # ax.legend(frameon=False,fontsize=6)
        # ax.set_xlim([-60,60])
        ax.set_xticks(plotcenters,plotlabels,rotation=45)
        if ilab == 0:
            ax.set_title(area)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if iarea==0:
            ax.set_ylabel('Activity (z)')
            ax.set_yticks([0,0.1,0.2,0.3,0.4])
plt.tight_layout()

#%% 
fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*2.5,nlabels*2.5),sharex=True,sharey=True)

for iarea, area in enumerate(areas):
    for ilab,label in enumerate(labeled):
        ax          = axes[ilab,iarea]
        for ises, session in enumerate(sessions):
            plotdata    = data_mean[:,1,iarea,ilab,ises] - data_mean[:,0,iarea,ilab,ises]

            x = plotcenters[~np.isnan(plotdata)]
            y = plotdata[~np.isnan(plotdata)]
            ax.plot(x, y, linewidth=0.5,color='grey')

        plotdata    = data_mean[:,1,iarea,ilab,:] - data_mean[:,0,iarea,ilab,:]
        x = plotcenters[np.any(~np.isnan(plotdata),axis=1)]
        y = plotdata[np.any(~np.isnan(plotdata),axis=1),:].T
        h = shaded_error(x, y, error='sem',linestyle=plotlines[ilr],color=clrs_areas[iarea],ax=ax)
        
        # t_stat,p_val = stats.ttest_rel(plotdata[:,0],plotdata[:,1],nan_policy='omit')
        t_stat,p_values = stats.ttest_rel(data_mean[:,1,iarea,ilab,:],data_mean[:,0,iarea,ilab,:],axis=1,nan_policy='omit')
        for i,p_val in enumerate(p_values): 
            ax.text(plotcenters[i], 0.2, '%s' % (get_sig_asterisks(p_val)), fontsize=12)
        
        if not np.any(~np.isnan(plotdata)):
            ax.axis('off')
        ax.set_ylim([-0.25,0.25])
        ax.set_xticks(plotcenters,plotlabels,rotation=45)
        if ilab == 0:
            ax.set_title(area)
        if iarea==0:
            ax.set_ylabel('Activity (z)\n(Hit - Miss)')
            ax.set_yticks([-0.2,-0.1,0,0.1,0.2])
        if np.any(~np.isnan(plotdata)):
            ax.axhline(0,linestyle='--',color='black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(savedir,'StimResponse_Saliency_HitMinusMiss_allNeurons_arealabels_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'StimResponse_Saliency_HitMinusMiss_neuronAverage_arealabels_%dsessions' % nSessions + '.png'), format = 'png')

#%% 
