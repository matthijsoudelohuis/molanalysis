# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#TODO
# filter runspeed
# plot individual trials locomotion, get sense of variance
# split script for diff purposes (psy vs max vs noise)
# allow psy protocol to fit DP and DN with same function

import math
import pandas as pd
import os
os.chdir('T:\\Python\\molanalysis\\')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.stats as st
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.psth import compute_tensor_space,compute_respmat_space
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import compute_dprime,smooth_rate_dprime,plot_psycurve #get support functions for beh analysis 

savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Behavior\\Detection\\'

############## Load the data - MaxOnly ####################################
protocol            = ['DM']
sessions            = filter_sessions(protocol,load_behaviordata=True)

nsessions = len(sessions)

################################################################
### Show the overall dprime for each animal across sessions:

dp_ses      = np.zeros([nsessions])
dp_ses_eng  = np.zeros([nsessions])
cr_ses      = np.zeros([nsessions])
cr_ses_eng  = np.zeros([nsessions])
for i,ses in enumerate(sessions):
    dp_ses[i],cr_ses[i]     = compute_dprime(ses.trialdata['signal']>0,ses.trialdata['lickResponse'])

    #Engaged only:
    df = ses.trialdata[ses.trialdata['engaged']==1]
    dp_ses_eng[i],cr_ses_eng[i]     = compute_dprime(df['signal']>0,df['lickResponse'])

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessiondata['dprime']    = dp_ses #dp_target_eng
sessiondata['dprime_eng']    = dp_ses_eng #dp_target_eng

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,3),sharey=True)

sns.stripplot(data = sessiondata,x='animal_id',y='dprime',palette='Dark2',size=10,ax=ax1)
ax1.axhline(y = 0, color = 'k', linestyle = ':')
ax1.set_title('Total Session', fontsize=11)
sns.stripplot(data = sessiondata,x='animal_id',y='dprime_eng',palette='Dark2',size=10,ax=ax2)
ax2.axhline(y = 0, color = 'k', linestyle = ':')
ax2.set_title('Engaged Only', fontsize=11)
plt.tight_layout()

plt.savefig(os.path.join(savedir,'Performance','Dprime_LPE10884_1ses' + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'Performance','Dprime_2animals_duringrecordings_engagedonly' + '.png'), format = 'png')

# # plot the mean line
# sns.boxplot(showmeans=True, 
#             meanline=True,
#             meanprops={'color': 'k', 'ls': '-', 'lw': 2},
#             medianprops={'visible': False},
#             whiskerprops={'visible': False},
#             zorder=10,
#             x="animal_id",
#             y="dprime_target",
#             data=sessiondata,
#             showfliers=False,
#             showbox=False,
#             showcaps=False,
#             palette='Dark2',
#             ax=ax1)

################################################################
#### The hit rate and performance as function of trial in session:
sessions        = smooth_rate_dprime(sessions,sigma=25)

#### construct concatenated trialdata DataFrame by appending all sessions:
trialdata       = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

fig,ax = plt.subplots(figsize=(7,4))
sns.lineplot(data=trialdata,x='trialNumber',y='smooth_hitrate',color='g')
sns.lineplot(data=trialdata,x='trialNumber',y='smooth_farate',color='r')
plt.ylabel('HIT / FA rate')

plt.savefig(os.path.join(savedir,'Performance','HITFA_rate_acrosssession_LPE10884' + '.png'), format = 'png')

### individual sessions
fig,ax = plt.subplots(figsize=(7,4))
for i,ses in enumerate(sessions):
    plt.plot(ses.trialdata['trialNumber'],ses.trialdata['smooth_hitrate'],color='g')
    plt.plot(ses.trialdata['trialNumber'],ses.trialdata['smooth_farate'],color='r')
plt.ylabel('HIT / FA rate')
plt.xlabel('Trial Number')

plt.savefig(os.path.join(savedir,'Performance','HITFA_rate_acrosssession_indiv' + '.png'), format = 'png')

### Dprime:
fig,ax = plt.subplots(figsize=(7,4))
sns.lineplot(data=trialdata,x='trialNumber',y='smooth_dprime',color='k')
plt.ylabel('Dprime')
plt.savefig(os.path.join(savedir,'Performance','Dprime_acrosssession' + '.png'), format = 'png')

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

## Parameters for spatial binning
s_pre       = -100  #pre cm
s_post      = 50   #post cm
binsize     = 5     #spatial binning in cm

binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
bincenters  = np.arange(s_pre,s_post+binsize,binsize)

trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

runPSTH     = np.empty((len(trialdata),len(bincenters)))
lickPSTH     = np.empty((len(trialdata),len(bincenters)))
                   
# ts_harp     = sessions[0].behaviordata['ts'].to_numpy()

for ises,ses in enumerate(sessions):

    ntrials     = len(ses.trialdata)

    runPSTH_ses     = np.empty(shape=(ntrials, len(bincenters)))

    for itrial in range(ntrials):
        idx = ses.behaviordata['trialnumber']==itrial+1
        runPSTH_ses[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][0],
                                            ses.behaviordata['runspeed'][idx], statistic='mean', bins=binedges)[0]

    runPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = runPSTH_ses

    lickPSTH_ses    = np.empty(shape=(ntrials, len(bincenters)))

    for itrial in range(ntrials-1):
        idx = ses.behaviordata['trialnumber']==itrial+1
        lickPSTH_ses[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][0],
                                            ses.behaviordata['lick'][idx], statistic='sum', bins=binedges)[0]
    lickPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = lickPSTH_ses


### Plot running speed as a function of trial type:

fig, ax = plt.subplots()

ttypes = pd.unique(trialdata['trialOutcome'])
ttypes = ['CR', 'MISS', 'HIT','FA']

for i,ttype in enumerate(ttypes):
    idx = np.logical_and(trialdata['trialOutcome']==ttype,trialdata['trialNumber']<1000)
    data_mean = np.nanmean(runPSTH[idx,:],axis=0)
    data_error = np.nanstd(runPSTH[idx,:],axis=0)# / math.sqrt(sum(idx))
    ax.plot(bincenters,data_mean,label=ttype)
    ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.5, linewidth=0)

ax.legend()
ax.set_ylim(0,50)
ax.set_xlim(-80,80)
ax.set_xlabel('Position rel. to stimulus onset (cm)')
ax.set_ylabel('Running speed (cm/s)')
ax.add_patch(matplotlib.patches.Rectangle((0,0),20,50, 
                        fill = True, alpha=0.2,
                        color = "blue",
                        linewidth = 0))
ax.add_patch(matplotlib.patches.Rectangle((20,0),20,50, 
                        fill = True, alpha=0.2,
                        color = "green",
                        linewidth = 0))

plt.text(5, 45, 'Stim',fontsize=12)
plt.text(25, 45, 'Reward',fontsize=12)


################################################################
### Plot licking rate as a function of trial type:

fig, ax = plt.subplots()

ttypes = pd.unique(trialdata['trialOutcome'])
ttypes = ['CR', 'MISS', 'HIT','FA']

for i,ttype in enumerate(ttypes):
    idx = np.logical_and(trialdata['trialOutcome']==ttype,trialdata['trialNumber']<300)
    data_mean = np.nanmean(lickPSTH[idx,:],axis=0)
    data_error = np.nanstd(lickPSTH[idx,:],axis=0) / math.sqrt(sum(idx))
    ax.plot(bincenters,data_mean,label=ttype)
    ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.5, linewidth=0)

ax.legend()
ax.set_ylim(0,5.6)
ax.set_xlim(-80,80)
ax.set_xlabel('Position rel. to stimulus onset (cm)')
ax.set_ylabel('Lick Rate (Hz)')
# ax.fill_between([0,30], [0,50], [0,50],alpha=0.5)
ax.add_patch(matplotlib.patches.Rectangle((0,0),20,5.6, 
                        fill = True, alpha=0.2,
                        color = "blue",
                        linewidth = 0))
ax.add_patch(matplotlib.patches.Rectangle((20,0),20,5.6, 
                        fill = True, alpha=0.2,
                        color = "green",
                        linewidth = 0))

plt.text(5, 5.2, 'Stim',fontsize=12)
plt.text(25, 5.2, 'Reward',fontsize=12)

################################ 

