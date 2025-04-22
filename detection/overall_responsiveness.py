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
from utils.explorefigs import plot_excerpt
from utils.psth import *
from utils.plot_lib import * #get all the fixed color schemes
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

# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE12013'],min_cells=100,load_videodata=True,
                        #    load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

sessions,nSessions = filter_sessions(protocol,min_cells=100,
                           load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE10884'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

#%% Get signal as relative to psychometric curve for all sessions:
sessions,nSessions = filter_sessions(protocols=protocol,min_cells=100) #Load specified list of sessions
sessions = noise_to_psy(sessions,filter_engaged=True,bootstrap=True)

#%% Include sessions based on performance: psychometric curve for the noise #############
sessiondata = pd.concat([ses.sessiondata for ses in sessions])
# zmin_thr = -0.5
# zmax_thr = 0.5

zmin_thr = -0.3
zmax_thr = 0.3
guess_thr = 0.4
idx_ses = np.all((sessiondata['noise_zmin']<=zmin_thr,
                  sessiondata['noise_zmax']>=zmax_thr,
                  sessiondata['guess_rate']<=guess_thr),axis=0)
print('Filtered %d/%d sessions based on performance' % (np.sum(idx_ses),len(idx_ses)))

#%%
sessions = [sessions[i] for i in np.where(idx_ses)[0]]
nSessions = len(sessions)
plot_psycurve(sessions,filter_engaged=True)

#%% Load the data:           
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)


#%% Z-score calcium data:
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)


#%%  Show raster or traces plot for some example session:
ises = 10

fig = plot_excerpt(sessions[ises],trialsel=[100,150],plot_neural=True,plot_behavioral=True,neural_version='traces')
# fig = plot_excerpt(sessions[ises],trialsel=[100,110],plot_neural=True,plot_behavioral=True,neural_version='traces')
# fig.savefig(os.path.join(savedir,'ExampleTraces','ExTraces_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')

#%%  Show raster or traces plot for some example session:
ises = 1
trialsel = [162,188]
neuronsel = np.all((sessions[ises].celldata['redcell'] == 0,
                    np.isin(sessions[ises].celldata['roi_name'],['V1','PM'])),axis=0)
neuronsel = np.isin(sessions[ises].celldata['roi_name'],['V1','PM'])

fig = plot_excerpt(sessions[ises],trialsel=trialsel,neuronsel=neuronsel,plot_neural=True,plot_behavioral=True,neural_version='raster')
fig.savefig(os.path.join(savedir,'ExampleTraces','ExRaster_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -75  #pre cm
s_post      = 75   #post cm
binsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')

    ## Compute average response in stimulus response zone:
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    # temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    # sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
    #                                 sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

# #%% ### Show for all sessions which region of the psychometric curve the noise spans #############
# sessions = noise_to_psy(sessions,filter_engaged=True)

# idx_inclthr = np.empty(nSessions).astype('int')
# for ises,ses in enumerate(sessions):
#     idx_inclthr[ises] = int(np.logical_and(np.any(sessions[ises].trialdata['signal_psy']<=0),np.any(sessions[ises].trialdata['signal_psy']>=0)))
#     ses.sessiondata['incl_thr'] = idx_inclthr[ises]

# sessions = [ses for ses in sessions if ses.sessiondata['incl_thr'][0]]
# nSessions = len(sessions)

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

#%% Plot number of responsive neurons per stimulus per area:

areas = ['V1','PM','AL','RSP']
clr_areas = get_clr_areas(areas)

frac_MN = celldata.groupby(['roi_name','session_id'])['sig_MN'].sum() / celldata.groupby(['roi_name','session_id'])['sig_MN'].count()
frac_MN = frac_MN.reset_index()


fig,ax = plt.subplots(1,1,figsize=(2.5,3),sharey=True, sharex=True)
sns.stripplot(x='roi_name', y='sig_MN', data=frac_MN, order=areas,hue_order=areas,
              color='k',s=2,ax=ax)
sns.pointplot(x='roi_name', y='sig_MN', data=frac_MN, order=areas,hue_order=areas,
              errorbar=('ci', 95),palette=clr_areas,ax=ax,capsize=.0,estimator=np.mean)
ax.set_title('Either Stimulus')
ax.set_xlabel('Area')
ax.set_ylabel('% responsive')

# fig.savefig(os.path.join(savedir,'SpatialActivity','FracResponsive_perStim_%d' % nanimals + '.png'), format = 'png',bbox_inches='tight')




#%%

sessel = sessions[:2]
sessel = sessions[:2]
perc_N,frac_M,frac_MN        = calc_spatial_responsive_neurons(sessel,sbins)

celldata        = pd.concat([ses.celldata for ses in sessel]).reset_index(drop=True)

frac_N,frac_M,frac_MN        = calc_spatial_responsive_neurons(sessions,sbins,nshuffle=100)
celldata        = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%%
np.save('frac_N.npy',frac_N)
np.save('frac_M.npy',frac_M)
np.save('frac_MN.npy',frac_MN)

#%% Plot number of responsive neurons per stimulus per area for labeled and unlabeled:

areas           = ['V1','PM','AL','RSP']
nareas          = len(areas)
clrs_areas      = get_clr_areas(areas)
# ax_areas        = ['V1','V1','PM','PM']
# ax_stim         = ['N','M','N','M']
# ax_stimlabels   = ['Noise','Max','Noise','Max']

labeled         = ['unl','lab']
nlabeled = len(labeled)
clr_labeled     = get_clr_labeled()
lines_labeled = ['-',':']
min_nlabcells   = 5

# data = copy.deepcopy(frac_N)
data = copy.deepcopy(frac_MN)

thr  = 0.01

# fig,axes = plt.subplots(1,nareas,figsize=(15,3),sharey=True, sharex=True)
fig,axes = plt.subplots(2,nareas,figsize=(15,6),sharey=True, sharex=True)

# for iax,(ax,ar,st) in enumerate(zip(axes,ax_areas,ax_stim)):
for iarea,area in enumerate(areas):
    for ilab,lab in enumerate(labeled):
        idx_neurons = np.logical_and(celldata['roi_name']==area,celldata['labeled']==lab)
        if np.sum(idx_neurons)>min_nlabcells:
            ax = axes[0,iarea]

            plot_data = np.sum(data[idx_neurons,:]>(1-thr),axis=0) / np.sum(idx_neurons)
            ax.plot(sbins[:-1],plot_data,color=clrs_areas[iarea],linestyle=lines_labeled[ilab])
            add_stim_resp_win(ax,linewidth=0.5)
            ax.set_xlabel('Pos. relative to stim')
            ax.set_ylabel('% responsive')

            ax = axes[1,iarea]
            plot_data = np.sum(data[idx_neurons,:]<thr,axis=0) / np.sum(idx_neurons)
            ax.plot(sbins[:-1],plot_data,color=clrs_areas[iarea],linestyle=lines_labeled[ilab])

            add_stim_resp_win(ax,linewidth=0.5)
            ax.set_xlabel('Pos. relative to stim')
            ax.set_ylabel('% responsive')

    # nlabcells = celldata[np.logical_and(celldata['roi_name']==ar,celldata['labeled']=='lab')].groupby('session_id')['sig_'+st].count()
    # nlabcells = nlabcells.reset_index()
    # idx_ses = nlabcells['session_id'][nlabcells['sig_'+st]>=min_nlabcells]

    # frac = celldata[np.logical_and(celldata['roi_name']==ar,celldata['session_id'].isin(idx_ses))].groupby(['session_id','labeled'])['sig_'+st].sum() / \
    #     celldata[np.logical_and(celldata['roi_name']==ar,celldata['session_id'].isin(idx_ses))].groupby(['session_id','labeled'])['sig_'+st].count().unstack(fill_value=0).stack()
    # frac = frac.reset_index()
    # frac.columns = ['session_id','labeled','sig']

    # sns.stripplot(x='labeled', y='sig' ,hue='labeled', data=frac, s=3,jitter=True, 
    #               palette=clr_labeled,ax=ax,order=labeled,legend=None,hue_order=labeled)
    # g = sns.pointplot(x='labeled', y='sig', hue='labeled',data=frac, ax=ax,order=labeled,
    #               hue_order=labeled,palette=clr_labeled,
    #               errorbar=('ci', 95),capsize=.0,estimator=np.mean,markers=['o', 'o'])
    
    # g = sns.pointplot(x='labeled', y='sig', color='k',data=frac, ax=ax,order=labeled,
    #             #   hue_order=labeled,palette='grey',
    #               errorbar=('ci', 95),capsize=.0)
    # g.get_legend().remove()
    
    # stat,pval = stats.ttest_rel(frac[frac['labeled']=='unl']['sig'],frac[frac['labeled']=='lab']['sig'])
    # stat,pval = stats.wilcoxon(frac[frac['labeled']=='unl']['sig'],frac[frac['labeled']=='lab']['sig'])
    # ax.annotate('%sp = %0.3f' % (get_sig_asterisks(pval),pval), xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center', size=9)
    # ax.set_title('%s - %s' % (ar,ax_stimlabels[iax]),fontsize=10)
    

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(savedir,'SpatialActivity','Spatial_FracResponsive_Labeled_%dsessions' % nSessions + '.png'), format = 'png',bbox_inches='tight')





#%% Get signal as relative to psychometric curve for all sessions:
sessions = noise_to_psy(sessions,filter_engaged=True,bootstrap=True)


# fig     = plot_psycurve(sessions,filter_engaged=True)
# fig     = plot_psycurve([sessions[21]],filter_engaged=True)
# fig     = plot_all_psycurve(sessions,filter_engaged=True)


#%% #################### Compute mean activity for saliency trial bins for all sessions ##################

labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

lickresp    = [0,1]
nlickresp   = len(lickresp)


sigtype     = 'signal'
zmin        = 5
zmax        = 20
nbins_noise = 3

# sigtype     = 'signal_psy'
# zmin        = -1
# zmax        = 1
# nbins_noise = 3

data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True)
data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True)

#%% 
plt.figure()
range_min = np.array([ses.sessiondata['signal_center'][0] - ses.sessiondata['signal_range'][0]/2 for ses in sessions])
range_max = np.array([ses.sessiondata['signal_center'][0] + ses.sessiondata['signal_range'][0]/2 for ses in sessions])
plt.hist(range_min,bins=np.arange(30),color='b',alpha=0.5)
plt.hist(range_max,bins=np.arange(30),color='r',alpha=0.5)


#%% Construct color panel for saliency trial bins
# plotlabels = ['catch'] + [str(x) for x in np.round(plotcenters,2)] + ['max']
# plotlabels = ['catch'] + [str(x) for x in np.round(plotcenters,2)] + ['max']
plotlabels = plotcenters
plotcolors = ['black']  # Start with black
plotcolors += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
plotcolors.append('orange')  # Add orange at the end
plotlines = ['--','-']


#%% ############################### Plot neuron-average activity per stim #################################

plotdata = np.nanmean(data_mean_spatial_hitmiss,axis=(0,3)).T
Z = np.shape(data_mean_spatial_hitmiss)[1]

fig,ax = plt.subplots(1,1,figsize=(1*3,1*2.5),sharex=True,sharey=True)
for iZ in range(Z):
    ax.plot(sbins, plotdata[:,iZ], color=plotcolors[iZ], label=plotlabels[iZ],linewidth=2)
ax.set_yticks(np.arange(0,2,0.1))
ax.set_ylim(np.nanmin(plotdata)*0.9,np.nanmax(plotdata)*1.1)
# ax.axhline(0, color='grey', linewidth=1, linestyle='--')
ax.legend(frameon=False,fontsize=8)
ax.set_xlim([-60,60])
ax.set_title('Average activity per stim')
ax.set_xlabel('Position relative to stim (cm)')
ax.set_ylabel('Activity (z)')
add_stim_resp_win(ax)
# plt.savefig(os.path.join(savedir,'Spatial_perSaliency_allAreas_%dsessions' % nSessions + '.png'), format = 'png')

#%% ############################### Plot spatial neuron-average per stim per area #################################
min_nlabcells = 10

labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

lickresponses = ['Miss','Hit']
lickresp    = [0,1]
nlickresp   = len(lickresp)

fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*3,nlabels*2.5),sharex=True,sharey=True)
for ilab,lab in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax          = axes[ilab,iarea]
        # idx_N = np.logical_and(celldata['roi_name']==area,celldata['labeled']==lab)
        idx_N = np.all((celldata['roi_name']==area,
                        # celldata['sig_MN']==1,
                        # celldata['sig_MN']==0,
                        celldata['labeled']==lab),axis=0)
        if np.sum(idx_N)>min_nlabcells:

            for ilr,lr in enumerate(lickresp):
                plotdata    = np.nanmean(data_mean_spatial_hitmiss[idx_N,:,:,ilr],axis=(0))
                for iZ in range(Z):
                    ax.plot(sbins, plotdata[iZ,:], color=plotcolors[iZ], label=plotlabels[iZ],linewidth=2,linestyle=plotlines[ilr])

        if not np.any(ax.get_legend_handles_labels()):
            ax.axis('off')
        else: 
            add_stim_resp_win(ax)
        # ax.set_ylim([-0.05,0.35])
        # if ilab == 0 and iarea == 0:
            # ax.legend(frameon=False,fontsize=6)
        ax.set_xlim([-75,75])
        if ilab == 0:
            ax.set_title(area)
        if ilab == 1:
            ax.set_xlabel('Position relative to stim (cm)')
        if iarea==0:
            ax.set_ylabel('Activity (z)')
            # ax.set_yticks([0,0.1,0.2,0.3])
            ax.set_yticks([0,0.1,0.2,0.3])
        if iarea == 0 and ilab == 0: 
            leg1 = ax.legend([plt.Line2D([0], [0], color=c, lw=1.5) for c in plotcolors], 
                         plotlabels, frameon=False,fontsize=7,loc='upper left',title='Saliency')
            ax.add_artist(leg1)
        if iarea == 0 and ilab == 1: 
            leg2 = ax.legend([plt.Line2D([0], [0], color='k', lw=1.5,ls=l) for l in plotlines],
                                lickresponses, frameon=False,fontsize=7,loc='upper left',title='Response')
            # ax.add_artist(leg1)
        
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Spatial_perSaliency_responsiveNeurons_arealabels_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'Spatial_perSaliency_allNeurons_arealabels_%dsessions' % nSessions + '.png'), format = 'png')

#%% ############################### Plot stimwin neuron-average per stim per area #################################

linecolors      = ['red','blue']
areas           = ['V1','PM']
# areas           = ['V1','PM','AL','RSP']
clrs_areas      = get_clr_areas(areas)
nareas          = len(areas)

fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*3,nlabels*2.5),sharex=True,sharey=True)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax          = axes[ilab,iarea]
        idx_N = np.all((celldata['roi_name']==area,
                        celldata['sig_N']==1,
                        # celldata['sig_MN']==0,
                        celldata['labeled']==label),axis=0)
        if np.sum(idx_N)>min_nlabcells:
            for ilr,lr in enumerate(lickresp):
                # plotdata    = np.nanmean(data_mean[:,ilr,iarea,ilab,:],axis=1)
                # ax.plot(plotcenters, plotdata, linewidth=2,linestyle=plotlines[ilr],color='k')
                # plotdata    = data_mean[:,ilr,iarea,ilab,:]
                # plotdata    = np.nanmean(data_mean_hitmiss[idx_N,:,ilr],axis=(0))
                plotdata    = data_mean_hitmiss[idx_N,:,ilr]

                if np.any(~np.isnan(plotdata)):
                    x = plotcenters[np.any(~np.isnan(plotdata),axis=0)]
                    y = plotdata[:,np.any(~np.isnan(plotdata),axis=0)]
                    # h = shaded_error(x, y, error='sem',linestyle=plotlines[ilr],color=clrs_areas[iarea],ax=ax)
                    h = shaded_error(np.arange(Z), y, error='sem',linestyle=plotlines[ilr],color=clrs_areas[iarea],ax=ax)
                # ax.plot(plotcenters, plotdata[:], color=plotcolors[iZ], label=plotlabels[iZ],linewidth=2,linestyle=plotlines[ilr])
        else:
            ax.axis('off')
        
        # ax.set_ylim([-0.025,0.3])
        # ax.set_xticks(plotcenters,plotlabels)
        # ax.set_xticks(plotcenters,plotcenters)
        ax.set_xticks(np.arange(Z))
        if ilab == 0:
            ax.set_title(area,fontsize=12)
        if np.any(~np.isnan(plotdata)):
            ax.text(0.5, 0.9, label, ha='center', transform=ax.transAxes,fontsize=10)
        
        if iarea==0:
            ax.set_ylabel('Activity (z)')
            # ax.set_yticks([0,0.1,0.2,0.3])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'StimResponse_Saliency_neuronAverage_arealabels_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'ActivityInCorridor_deconv_neuronAverage_perStim_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% 
min_nlabcells = 5
fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*2.5,nlabels*2.5),sharex=True,sharey=True)
plotcenters = np.arange(Z)
for iarea, area in enumerate(areas):
    for ilab,label in enumerate(labeled):
        ax          = axes[ilab,iarea]
        for ises, session in enumerate(sessions):
            idx_N = np.all((celldata['roi_name']==area,
                        celldata['sig_N']==1,
                        # celldata['sig_MN']==1,
                        # celldata['sig_MN']==0,
                        celldata['session_id']==session.sessiondata['session_id'][0],
                        celldata['labeled']==label),axis=0)
            
            if np.sum(idx_N)>min_nlabcells:
            
                plotdata    = data_mean_hitmiss[idx_N,:,1] - data_mean_hitmiss[idx_N,:,0]

                x = plotcenters[np.any(~np.isnan(plotdata),axis=0)]
                y = np.nanmean(plotdata[:,np.any(~np.isnan(plotdata),axis=0)],axis=0)
                # h = shaded_error(x, y, error='sem',linestyle=plotlines[ilr],color=clrs_areas[iarea],ax=ax)
                # h = shaded_error(np.arange(Z), y, error='sem',linestyle=plotlines[ilr],color=clrs_areas[iarea],ax=ax)

                # x = plotcenters[~np.isnan(plotdata)]
                # y = plotdata[~np.isnan(plotdata)]
                ax.plot(x, y, linewidth=0.5,color='grey')

        idx_N = np.all((celldata['roi_name']==area,
                        # celldata['sig_MN']==1,
                        celldata['sig_N']==1,
                        # celldata['sig_MN']==0,
                        celldata['labeled']==label),axis=0)
        # plotdata    = data_mean_hitmiss[:,1,iarea,ilab,:] - data_mean_hitmiss[:,0,iarea,ilab,:]
        if np.sum(idx_N)>min_nlabcells:

            plotdata    = data_mean_hitmiss[idx_N,:,1] - data_mean_hitmiss[idx_N,:,0]

            x = plotcenters[np.any(~np.isnan(plotdata),axis=0)]
            y = plotdata[:,np.any(~np.isnan(plotdata),axis=0)]
            h = shaded_error(x, y, error='sem',linestyle=plotlines[ilr],color=clrs_areas[iarea],ax=ax)
        
        # t_stat,p_val = stats.ttest_rel(plotdata[:,0],plotdata[:,1],nan_policy='omit')
        # t_stat,p_values = stats.ttest_rel(data_mean_hitmiss[:,1,iarea,ilab,:],data_mean_hitmiss[:,0,iarea,ilab,:],axis=1,nan_policy='omit')
        # for i,p_val in enumerate(p_values): 
        #     ax.text(plotcenters[i], 0.2, '%s' % (get_sig_asterisks(p_val)), fontsize=12)
        
        else:
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
# plt.savefig(os.path.join(savedir,'StimResponse_Saliency_HitMinusMiss_allNeurons_arealabels_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'StimResponse_Saliency_HitMinusMiss_neuronAverage_arealabels_%dsessions' % nSessions + '.png'), format = 'png')

#%% 

sigtype     = 'signal'
zmin        = 5
zmax        = 20
nbins_noise = 5

# zmin        = 5
# zmax        = 20
# nbins_noise = 5

# data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True)
data_mean,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True)

#%% Data Ratio:
areas           = ['V1','PM','AL','RSP']
# areas           = ['V1','PM']
clrs_areas      = get_clr_areas(areas)

nareas          = len(areas)
depth_border    = 250
nmincells       = -1
layerdata       = np.full((nSessions,nareas,len(sbins),2,2),np.nan)
diffdata        = np.full((nSessions,nareas,len(sbins),2),np.nan)

for ises,ses in enumerate(sessions):
    for iarea, area in enumerate(areas):
        idx_N_sup = np.all((celldata['session_id']==ses.sessiondata['session_id'][0],
                            celldata['depth']<depth_border,
                            celldata['roi_name']==area), axis=0)
        idx_N_deep = np.all((celldata['session_id']==ses.sessiondata['session_id'][0],
                            celldata['depth']>=depth_border,
                            celldata['roi_name']==area), axis=0)

        # idx_N_sup = np.all((celldata['session_id']==ses.sessiondata['session_id'][0],
        #                     celldata['depth']<depth_border,
        #                     celldata['sig_MN']==1,
        #                     celldata['roi_name']==area), axis=0)
        # idx_N_deep = np.all((celldata['session_id']==ses.sessiondata['session_id'][0],
        #                     celldata['depth']>=depth_border,
        #                     celldata['sig_MN']==1,
        #                     celldata['roi_name']==area), axis=0)
        

        if np.sum(idx_N_sup) > nmincells and np.sum(idx_N_deep) > nmincells:
            
            layerdata[ises,iarea,:,:,0] = np.nanmean(data_mean[idx_N_sup,1:-1,:,:],axis=(0,1))
            layerdata[ises,iarea,:,:,1] = np.nanmean(data_mean[idx_N_deep,1:-1,:,:],axis=(0,1))

            # temp = np.nanmean(data_mean[idx_N_sup,1:-1,:,:],axis=(0,1)) / np.nanmean(data_mean[idx_N_deep,1:-1,:,:],axis=(0,1))
            # temp =  / np.nanmean(data_mean[idx_N_deep,1:-1,:,:],axis=0)

            # plotdata[ises,iarea,:] = np.nanmean(temp)
            # plotdata[ises,iarea,:] = np.nanmean(temp,axis=0)

            temp  = np.nanmean(data_mean[idx_N_sup,1:-1,:,:],axis=0) - np.nanmean(data_mean[idx_N_deep,1:-1,:,:],axis=0)
            diffdata[ises,iarea,:,:]  = np.nanmean(temp,axis=0)


# diffdata = layerdata[:,:,:,:,0] - layerdata[:,:,:,:,1]

#%% Plot the layerdata of feedforward to feedback activity
# fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*2.5,nlabels*2.5),sharex=True,sharey=True)
linestyles = ['-','--']
layerlabels =['Superficial','Deep']
lickresponses = ['Miss','Hit']
fig,axes = plt.subplots(2,nareas,figsize=(nareas*2.5,2*2.5),sharex=True,sharey=True)
clrs_areas = get_clr_areas(areas)
for iarea, area in enumerate(areas):
    for ilr,lr in enumerate(lickresponses):
        ax          = axes[ilr,iarea]
        handles = []
        for ilayer,layer in enumerate(layerlabels):

            for ises, session in enumerate(sessions):
                ax.plot(sbins,layerdata[ises,iarea,:,ilr,ilayer],color=clrs_areas[iarea],linestyle=linestyles[ilayer],
                        linewidth=0.3,label=session.sessiondata['session_id'][0])
                # ax.plot(sbins,ratiodata[ises,iarea,:,ilr],color=clrs_areas[iarea],linewidth=0.3,label=session.sessiondata['session_id'][0])
        # ax.plot(sbins,plotdata[ises,iarea,:],color=clrs_areas[iarea],linewidth=2)
            handles.append(shaded_error(sbins,layerdata[:,iarea,:,ilr,ilayer],color=clrs_areas[iarea],
                     linestyle=linestyles[ilayer],error='sem',ax=ax))
        ax.set_title(area + '-' + lr)
        ax.legend(handles,layerlabels,frameon=False)

        add_stim_resp_win(ax)
    # ax.set_ylim([-1,1])
    # if ilab == 0 and iarea == 0:
        # ax.legend(frameon=False,fontsize=6)
        
plt.tight_layout()

#%% Plot the ratio of feedforward to feedback activity
# fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*2.5,nlabels*2.5),sharex=True,sharey=True)
linestyles = ['--','-']
fig,axes = plt.subplots(1,nareas,figsize=(nareas*2.5,1*2.5),sharex=True,sharey=True)
clrs_areas = get_clr_areas(areas)
for iarea, area in enumerate(areas):
    ax          = axes[iarea]
    handles = []
    for ilr,lr in enumerate([0,1]):
        for ises, session in enumerate(sessions):
            ax.plot(sbins,diffdata[ises,iarea,:,ilr],color=clrs_areas[iarea],linestyle=linestyles[ilr],
                    linewidth=0.3,label=session.sessiondata['session_id'][0])
        # ax.plot(sbins,plotdata[ises,iarea,:],color=clrs_areas[iarea],linewidth=2)
        handles.append(shaded_error(sbins,diffdata[:,iarea,:,ilr],color=clrs_areas[iarea],linewidth=2,
                     linestyle=linestyles[ilr],error='sem',ax=ax))
    ax.legend(handles,lickresponses,frameon=False,fontsize=9)
    add_stim_resp_win(ax)
    ax.set_ylabel('Superficial - Deep')
    # ax.set_ylim([-1,1])
    # if ilab == 0 and iarea == 0:
        # ax.legend(frameon=False,fontsize=6)
        
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'StimResponse_Saliency_HitMinusMiss_allNeurons_arealabels_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'StimResponse_Saliency_HitMinusMiss_neuronAverage_arealabels_%dsessions' % nSessions + '.png'), format = 'png')



