# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Import packages
import math
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import medfilt
from scipy.stats import zscore

os.chdir('e:\\Python\\molanalysis\\')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.psth import compute_tensor_space,compute_respmat_space
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.plot_lib import *
from utils.regress_lib import *
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\')


#%% ########################## Load psychometric data #######################
protocol            = ['DP']
sessions,nSessions  = filter_sessions(protocol,load_behaviordata=True)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# Apply median filter to smooth runspeed
for ises,ses in enumerate(sessions):
    sessions[ises].behaviordata['runspeed'] = medfilt(sessions[ises].behaviordata['runspeed'], kernel_size=51)

#%% #################### Spatial runspeed plots ####################################
for ises,ses in enumerate(sessions):
    ### running across the trial:
    [sessions[ises].runPSTH,bincenters] = calc_runPSTH(sessions[ises],binsize=5)
    fig = plot_run_corridor_outcome(sessions[ises].trialdata,sessions[ises].runPSTH,bincenters,
                                    plot_mean=True,plot_trials=True)
    fig.savefig(os.path.join(savedir,'Spatial','ExampleSessions','RunSpeed_Outcome_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
    
#%% #################### Spatial lick rate plots ####################################
for ises,ses in enumerate(sessions):
    ### running across the trial:
    [sessions[ises].lickPSTH,bincenters] = calc_lickPSTH(sessions[ises],binsize=5)
    # fig = plot_lick_corridor_outcome(sessions[ises].trialdata,sessions[ises].runPSTH,bincenters,
    fig = plot_lick_corridor_outcome(sessions[ises].trialdata,sessions[ises].lickPSTH,bincenters)
    # fig.savefig(os.path.join(savedir,'Spatial','ExampleSessions','LickRate_Outcome_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

# # Behavior as a function of distance within the corridor:
# sesidx = 0
# print(sessions[sesidx].sessiondata['session_id'])
# ### licking across the trial:
# [sessions[sesidx].lickPSTH,bincenters] = calc_lickPSTH(sessions[sesidx],binsize=5)










#%% ########################## Load data #######################
protocol            = ['DM','DP','DN']
sessions,nSessions  = filter_sessions(protocol,load_behaviordata=True,load_videodata=True)

# Remove sessions LPE10884 that are too bad:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE10884_2023_12_14','LPE10884_2023_12_15','LPE10884_2024_01_11','LPE10884_2024_01_16']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# Only sessions that have rewardZoneOffset == 25
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(sessiondata['rewardZoneOffset'] == 25)[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% #################### Define the stimulus window ############################
s_min       = -5   #cm, start of stimulus window
s_max       = 15
sbinsize    = 5

#%% #################### Compute spatial runspeed ####################################
for ises,ses in enumerate(sessions): # running across the trial:
    sessions[ises].behaviordata['runspeed']     = medfilt(sessions[ises].behaviordata['runspeed'], kernel_size=51)
    [sessions[ises].runPSTH,bincenters]         = calc_runPSTH(sessions[ises],binsize=sbinsize)
    sessions[ises].trialdata['runspeed_stim']   = np.mean(sessions[ises].runPSTH[:,(bincenters>=s_min) & (bincenters<=s_max)],axis=1)
    [sessions[ises].pupilPSTH,bincenters]       = calc_pupilPSTH(sessions[ises],binsize=sbinsize)
    [sessions[ises].videomePSTH,bincenters]     = calc_videomePSTH(sessions[ises],binsize=sbinsize)
    [sessions[ises].lickPSTH,bincenters]        = calc_lickPSTH(sessions[ises],binsize=sbinsize)
    sessions[ises].trialdata['lickrate_stim']   = np.mean(sessions[ises].lickPSTH[:,(bincenters>=s_min) & (bincenters<=s_max)],axis=1)

#%% Show histograms of running speed during the stimulus window:
speedres    = 2.5 #cm/s bins
sp_min      = -5    #lowest speed bin
sp_max      = 80    #highest speed bin

speedbinedges    = np.arange(sp_min-speedres/2,sp_max+speedres+speedres/2,speedres)
speedbincenters  = np.arange(sp_min,sp_max+speedres,speedres)
nspeedbins      = len(speedbincenters)

u_animals       = np.unique(sessiondata['animal_id'])
clrs_animals    = get_clr_animal_id(u_animals)
runstimspeed    = np.empty((nSessions,nspeedbins)) #init array for hist data

filter_engaged   = True

for ises,ses in enumerate(sessions):
    if filter_engaged:
        idx = ses.trialdata['engaged']==1
    runstimspeed[ises,:] = np.histogram(sessions[ises].trialdata['runspeed_stim'][idx],
                                        bins=speedbinedges,density=False)[0]

fig, ax = plt.subplots(figsize=(4,3))
handles = [] #Just for the labels, not shown:
for ianimal, animal in enumerate(u_animals):
    sesidx = np.where(sessiondata['animal_id']==animal)[0]
    handles.append(ax.plot(speedbincenters, np.nanmean(runstimspeed[sesidx,:],axis=0), label=animal,
            color=clrs_animals[ianimal], linewidth=0)[0])

#Plot histogram for each session:
for ises, ses in enumerate(sessions):
    ax.plot(speedbincenters, runstimspeed[ises,:], label=ses.sessiondata['animal_id'][0],
            linewidth=0.5,color=clrs_animals[np.where(u_animals==ses.sessiondata['animal_id'][0])[0][0]])
#Figure make up:
ax.set_ylabel('Trial count')
ax.set_xlabel('Running speed in stim window (cm/s)')
leg = ax.legend(handles,u_animals,frameon=False,fontsize=7,loc='upper right',title='Animal')
for i, text in enumerate(leg.get_texts()):
    text.set_color(clrs_animals[i])
plt.tight_layout()
#Save figure:
fig.savefig(os.path.join(savedir, 'Spatial', 'RunSpeed_Hist_AllSessions_%dsessions' % nSessions + '.png'), format='png')

#%% Distribution of running speed during the stimulus window:
tres      = 0.1 #cm/s bins
t_min      = 0    #lowest speed bin
t_max      = 5    #highest speed bin

tbinedges       = np.arange(t_min-tres/2,t_max+tres+tres/2,tres)
tbincenters     = np.arange(t_min,t_max+tres,tres)
nbins           = len(tbincenters)

u_animals       = np.unique(sessiondata['animal_id'])
clrs_animals    = get_clr_animal_id(u_animals)
stimdur         = np.empty((nSessions,nbins)) #init array for hist data

filter_engaged   = True

for ises,ses in enumerate(sessions):
    if filter_engaged:
        idx = ses.trialdata['engaged']==1
    stimdur[ises,:] = np.histogram(sessions[ises].trialdata['tStimEnd'][idx] - sessions[ises].trialdata['tStimStart'][idx],
                                        bins=tbinedges,density=False)[0]

fig, ax = plt.subplots(figsize=(4,3))

handles = [] #Just for the labels, not shown:
for ianimal, animal in enumerate(u_animals):
    sesidx = np.where(sessiondata['animal_id']==animal)[0]
    handles.append(ax.plot(tbincenters, np.nanmean(stimdur[sesidx,:],axis=0), label=animal,
            color=clrs_animals[ianimal], linewidth=0)[0])

#Plot histogram for each session:
for ises, ses in enumerate(sessions):
    ax.plot(tbincenters, stimdur[ises,:], label=ses.sessiondata['animal_id'][0],
            linewidth=0.5,color=clrs_animals[np.where(u_animals==ses.sessiondata['animal_id'][0])[0][0]])
#Figure make up:
ax.set_ylabel('Trial count')
ax.set_xlabel('Stim duration (s)')
leg = ax.legend(handles,u_animals,frameon=False,fontsize=7,loc='upper right',title='Animal')
for i, text in enumerate(leg.get_texts()):
    text.set_color(clrs_animals[i])
plt.tight_layout()
#Save figure:
fig.savefig(os.path.join(savedir, 'Spatial', 'StimDur_Hist_AllSessions_%dsessions' % nSessions + '.png'), format='png')


#%% Show 2D histogram of running speed for hist and misses:
speedres    = 0.1 #cm/s bins
sp_min      = 0    #lowest speed bin
sp_max      = 1    #highest speed bin

speedbinedges    = np.arange(sp_min-speedres/2,sp_max+speedres,speedres)
speedbincenters  = np.arange(sp_min,sp_max+speedres,speedres)
nspeedbins      = len(speedbincenters)

nspatbins       = len(bincenters)

datamat = np.full((nSessions, nspeedbins, nspatbins, 2), np.nan)  # init array with NaN for hist data
for ises,ses in enumerate(sessions):
    for isp,sp in enumerate(bincenters):
        for ihit, hit in enumerate([0,1]):
            idx = np.all((ses.trialdata['engaged']==1,ses.trialdata['stimcat']=='N',ses.trialdata['lickResponse']==hit), axis=0)
            
            if np.sum(idx) > 25:
                datamat[ises,:,isp,ihit] = np.histogram(sessions[ises].runPSTH[idx,isp] / np.nanmax(sessions[ises].runPSTH),
                                                bins=speedbinedges,density=True)[0]

example_sessions = np.where(np.all(np.any(~np.isnan(datamat), axis=(1, 2)),axis=1))[0]
datamat = np.concatenate((datamat, datamat[:,:,:,1,None] - datamat[:,:,:,0,None]), axis=3)
# datamat = np.concatenate((datamat, datamat[:,:,:,1,None] / datamat[:,:,:,0,None]), axis=3)

#%% Show the 2D speed over space histograms for some example sessions: 
nexamples = 3
fig, axes = plt.subplots(nexamples, 3, figsize=(3*3, nexamples*1.7))
for i, ises in enumerate(example_sessions[:nexamples]):  # Loop over 5 example sessions
    for ihit, hit in enumerate(['Miss','Hit','Diff']):
        # data = np.nanmean(datamat[i, :, :, ihit], axis=1)
        data = datamat[ises, :, :, ihit]
        ax = axes[i, ihit]
        if ihit == 0 or ihit == 1: 
            ax.pcolor(bincenters, speedbincenters, data, cmap='magma')
        else:
            ax.pcolor(bincenters, speedbincenters, data, cmap='RdBu_r',
                      vmin=-np.max(np.abs(np.percentile(data,[2,99]))),vmax=np.max(np.abs(np.percentile(data,[2,99]))))
            # ax.pcolor(bincenters, speedbincenters, data, cmap='RdBu_r')
        if ihit == 2:
            ax.axvline(x=0, color='k', linestyle='--')
            ax.axvline(x=20, color='k', linestyle='--')
        else:
            ax.axvline(x=0, color='w', linestyle='--')
            ax.axvline(x=20, color='w', linestyle='--')

        if i == 0:
            ax.set_title(hit)
        if i != nexamples-1:
            ax.set_xticklabels([])
        if ihit == 1 and i == nexamples-1:
            ax.set_xlabel('Position relative to stim (cm)')
        if ihit !=0:
            ax.set_yticklabels([])
        if ihit == 0 and i == 1:
            ax.set_ylabel('Running Speed (norm)')
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Spatial', 'RunSpeed_Space_Heatmap_ExampleSessions.png'), format='png')

#%% Show the 2D speed over space histograms for the average: 
fig, axes = plt.subplots(1, 3, figsize=(3*3, 2))
for ihit, hit in enumerate(['Miss','Hit','Diff']):
    data = np.nanmean(datamat[:, :, :, ihit],axis=0)
    ax = axes[ihit]
    if ihit == 0 or ihit == 1: 
        ax.pcolor(bincenters, speedbincenters, data, cmap='magma')
    else:
        ax.pcolor(bincenters, speedbincenters, data, cmap='RdBu_r',
                              vmin=-np.max(np.abs(np.percentile(data,[1,99]))),vmax=np.max(np.abs(np.percentile(data,[1,99]))))

    if ihit == 2:
        ax.axvline(x=0, color='k', linestyle='--')
        ax.axvline(x=20, color='k', linestyle='--')
    else:
        ax.axvline(x=0, color='w', linestyle='--')
        ax.axvline(x=20, color='w', linestyle='--')
    ax.set_title(hit)

    if ihit == 1:
        ax.set_xlabel('Position relative to stim (cm)')
    if ihit == 0:
        ax.set_ylabel('Running Speed (norm)')
    if ihit !=0:
        ax.set_yticklabels([])

plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Spatial', 'RunSpeed_Space_Heatmap_Average.png'), format='png')

#%% 




#%% Decoding of choice from behavioral variables:

# Define the variables to use for decoding
variables = ['runspeed', 'pupil_area', 'videoME', 'lick_rate']

# Define the number of folds for cross-validation
kfold = 10

# Initialize an array to store the decoding performance
performance = np.full((nSessions,len(bincenters)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),desc='Decoding response across sessions'):
    #Correct setting: stimulus trials during engaged part of the session:
    idx = np.all((ses.trialdata['engaged']==1,np.isin(ses.trialdata['stimcat'],['M','N'])), axis=0)
    
    if np.sum(idx) > 50:
        # Get the lickresponse data for this session
        y = ses.trialdata['lickResponse'][idx].to_numpy()

        X = np.stack((ses.runPSTH[idx,:], ses.pupilPSTH[idx,:], ses.videomePSTH[idx,:], ses.lickPSTH[idx,:]), axis=2)
        #take the mean during the response window to determine optimal lambda
        with np.errstate(invalid='ignore'):
            X = np.nanmean(X[:, (bincenters>=25) & (bincenters<=45), :], axis=1)
        # X = zscore(X, axis=0)
        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        # idx_notnan = np.all(np.all((~np.isnan(X),~np.isinf(X),~(X==0)),axis=0),axis=0)
        # X[:, ~idx_notnan] = 0
        # X[np.isnan(X)] = 0
        
        optimal_lambda = find_optimal_lambda(X,y,model_name='LOGR',kfold=kfold)

        # Loop through each spatial bin
        for ibin, bincenter in enumerate(bincenters):
            y = ses.trialdata['lickResponse'][idx].to_numpy()

            # Define the X and y variables
            X = np.stack((ses.runPSTH[idx,ibin], ses.pupilPSTH[idx,ibin], ses.videomePSTH[idx,ibin], ses.lickPSTH[idx,ibin]), axis=1)
            
            X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            # X = zscore(X, axis=0)
        
            # idx_notnan = np.all(np.all((~np.isnan(X),~np.isinf(X),~(X==0)),axis=0),axis=0)
            # X[:, ~idx_notnan] = 0
            # X[np.isnan(X)] = 0

            # Calculate the average decoding performance across folds
            performance[ises,ibin] = my_classifier_wrapper(X,y,model_name='LOGR',kfold=kfold,lam=optimal_lambda,norm_out=True)

#%% Show the decoding performance
fig,ax = plt.subplots(1,1,figsize=(4,3))
for i,ses in enumerate(sessions):
    if np.any(performance[i,:]):
        ax.plot(bincenters,performance[i,:],color='grey',alpha=0.5,linewidth=1)
shaded_error(bincenters,performance,error='sem',ax=ax,color='b')
add_stim_resp_win(ax)
ax.set_xlabel('Position relative to stim (cm)')
ax.set_ylabel('Decoding Performance \n (accuracy - shuffle)')
ax.set_title('Decoding Performance')
ax.set_xlim([-60,60])
ax.set_ylim([-0.1,1])
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'Spatial', 'LogisticDecodingPerformance_LickResponse.png'), format='png')
# plt.savefig(os.path.join(savedir, 'Spatial', 'LogisticDecodingPerformance_LickResponse_engaged.png'), format='png')
plt.savefig(os.path.join(savedir, 'Spatial', 'LogisticDecodingPerformance_LickResponse_engaged_stimonly.png'), format='png')


#%% 









#%% Remove non DN sessions:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(sessiondata['protocol'].isin(['DN']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Show licking across the tunnel and show distribution across trials: 
# Distribution of licking during the stimulus window:
lrres       = 1  #licks/cm bins
lr_min      = 0    #lowest lick rate bin
lr_max      = 30    #highest lick rate bin

lrbinedges    = np.arange(lr_min-lrres/2,lr_max+lrres+lrres/2,lrres)
lrbincenters  = (lrbinedges[1:]+lrbinedges[:-1])/2
nlrbins       = len(lrbincenters)

u_animals     = np.unique(sessiondata['animal_id'])
clrs_animals  = get_clr_animal_id(u_animals)
lrstimdata    = np.empty((nSessions,nlrbins)) #init array for hist data

for ises,ses in enumerate(sessions):
    idx = ses.trialdata['engaged']==1
    lrstimdata[ises,:] = np.histogram(sessions[ises].trialdata['lickrate_stim'][idx]/0.04,
                                      bins=lrbinedges,density=True)[0]
    lrstimdata[ises,:] = np.cumsum(lrstimdata[ises,:])/np.sum(lrstimdata[ises,:])

fig, ax = plt.subplots(figsize=(4,3))
handles = [] #Just for the labels, not shown:
for ianimal, animal in enumerate(u_animals):
    sesidx = np.where(sessiondata['animal_id']==animal)[0]
    handles.append(ax.plot(lrbincenters, np.nanmean(lrstimdata[sesidx,:],axis=0), label=animal,
            color=clrs_animals[ianimal], linewidth=0)[0])

#Plot histogram for each session:
for ises, ses in enumerate(sessions):
    ax.plot(lrbincenters, lrstimdata[ises,:], label=ses.sessiondata['animal_id'][0],
            linewidth=0.4,color=clrs_animals[np.where(u_animals==ses.sessiondata['animal_id'][0])[0][0]])
ax.plot(lrbincenters, np.median(lrstimdata,axis=0),
            linewidth=1.5,color='k')

#Figure make up:
ax.set_ylabel('Cumulative fraction of trials')
ax.set_xlabel('#licks in stimulus window (-5 to +15 cm)')
leg = ax.legend(handles,u_animals,frameon=False,fontsize=7,loc='lower right',title='Animal')
for i, text in enumerate(leg.get_texts()):
    text.set_color(clrs_animals[i])
plt.ylim([0,1])
plt.tight_layout()

#Save figure:
fig.savefig(os.path.join(savedir, 'Spatial', 'Lickrate_Hist_DNSessions_%dsessions' % nSessions + '.png'), format='png')


#%% 
for ises in range(nSessions):
    [sessions[ises].lickPSTH,bincenters] = calc_lickPSTH(sessions[ises],binsize=5)
    fig = plot_lick_corridor_raster(sessions[ises].trialdata,sessions[ises].lickPSTH/0.04,bincenters,version='trialNumber',filter_engaged=False)
    fig.savefig(os.path.join(savedir,'Spatial','LickRaster','LickRaster_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% 
# fig = plot_lick_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
# sessions[sesidx].lickPSTH[-1,:] = 0
sesidx = 0
[sessions[sesidx].lickPSTH,bincenters] = calc_lickPSTH(sessions[sesidx],binsize=5)
fig = plot_lick_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
# fig.savefig(os.path.join(savedir,'Performance','LickRate_Psy_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% running across the trial:
[sessions[sesidx].runPSTH,bincenters] = calc_runPSTH(sessions[sesidx],binsize=2.5)
fig = plot_run_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters,
                                plot_mean=True,plot_trials=True)

fig.savefig(os.path.join(savedir,'Performance','RunSpeed_Outcome_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig = plot_run_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','RunSpeed_Psy_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

##################### Plot psychometric curve #########################

fig = plot_psycurve([sessions[sesidx]])
fig = plot_psycurve(sessions)












#%% 



#%% Load behavior of DM protocols:
protocol                = ['DM']
sessions,nSessions      = filter_sessions(protocol,load_behaviordata=False,has_pupil=False)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
nanimals                = len(np.unique(sessiondata['animal_id']))

#%% ###############################################################
#### The hit rate and performance as function of trial in session:
sessions        = smooth_rate_dprime(sessions,sigma=25)

#### construct concatenated trialdata DataFrame by appending all sessions:
trialdata       = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

fig,ax = plt.subplots(figsize=(7,4))
sns.lineplot(data=trialdata,x='trialNumber',y='smooth_hitrate',color='g')
sns.lineplot(data=trialdata,x='trialNumber',y='smooth_farate',color='r')
plt.ylabel('HIT / FA rate')

plt.savefig(os.path.join(savedir,'Performance','HITFA_rate_acrosssession_%danimals' % nanimals + '.png'), format = 'png')

### individual sessions
fig,ax = plt.subplots(figsize=(7,4))
for i,ses in enumerate(sessions):
    plt.plot(ses.trialdata['trialNumber'],ses.trialdata['smooth_hitrate'],color='g')
    plt.plot(ses.trialdata['trialNumber'],ses.trialdata['smooth_farate'],color='r')
plt.ylabel('HIT / FA rate')
plt.xlabel('Trial Number')

plt.savefig(os.path.join(savedir,'Performance','HITFA_rate_acrosssession_indiv_%danimals' %nanimals + '.png'), format = 'png')

### Dprime:
fig,ax = plt.subplots(figsize=(7,4))
sns.lineplot(data=trialdata,x='trialNumber',y='smooth_dprime',color='k')
plt.ylabel('Dprime')
plt.ylim([0,7])
plt.savefig(os.path.join(savedir,'Performance','Dprime_acrosssession_%danimals' %nanimals + '.png'), format = 'png')

fig,ax = plt.subplots(figsize=(7,4))
for i,ses in enumerate(sessions):
    plt.plot(ses.trialdata['trialNumber'],ses.trialdata['smooth_dprime'],color='k')
plt.ylabel('Dprime')
plt.xlabel('Trial Number')

ax.set_ylim([-0.5,ax.get_ylim()[1]])
ax.axhline(0,color='k',linestyle=':')
plt.savefig(os.path.join(savedir,'Performance','Dprime_acrosssession_indiv' + '.png'), format = 'png')


################ Spatial plots ##############################################
# Behavior as a function of distance within the corridor:

sesidx = 1
### licking across the trial:
[sessions[sesidx].lickPSTH,bincenters] = calc_lickPSTH(sessions[sesidx],binsize=5)

# fig = plot_lick_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
# sessions[sesidx].lickPSTH[-1,:] = 0
fig = plot_lick_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','LickRate_Max_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig = plot_lick_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','LickRate_Outcome_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

### running across the trial:
[sessions[sesidx].runPSTH,bincenters] = calc_runPSTH(sessions[sesidx],binsize=5)

fig = plot_run_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','RunSpeed_Outcome_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig = plot_run_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','RunSpeed_Psy_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

################################ 



##################### Spatial plots ####################################
# Behavior as a function of distance within the corridor:
sesidx = 0
print(sessions[sesidx].sessiondata['session_id'])
### licking across the trial:
[sessions[sesidx].lickPSTH,bincenters] = calc_lickPSTH(sessions[sesidx],binsize=5)

# fig = plot_lick_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
# sessions[sesidx].lickPSTH[-1,:] = 0
fig = plot_lick_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','LickRate_Psy_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

### running across the trial:
[sessions[sesidx].runPSTH,bincenters] = calc_runPSTH(sessions[sesidx],binsize=5)

fig = plot_run_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','RunSpeed_Outcome_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig = plot_run_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','RunSpeed_Psy_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

##################### Plot psychometric curve #########################

fig = plot_psycurve([sessions[sesidx]])
fig = plot_psycurve(sessions)
fig.savefig(os.path.join(savedir,'Psychometric','Psy_%s.png' % sessions[sesidx].session_id))

fig = plot_psycurve(sessions,filter_engaged=True)
fig.savefig(os.path.join(savedir,'Psychometric','Psy_%s_Engaged.png' % sessions[sesidx].session_id))

# df = sessions[sesidx].trialdata[sessions[0].trialdata['trialOutcome']=='CR']

fig = plt.figure()
plt.scatter(sessions[sesidx].lickPSTH.flatten(),sessions[sesidx].runPSTH.flatten(),s=6,alpha=0.2)
plt.xlabel('Lick Rate')
plt.ylabel('Running Speed')
fig.savefig(os.path.join(savedir,'Psychometric','LickRate_vs_RunningSpeed_%s.png' % sessions[sesidx].session_id))

