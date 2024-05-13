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
from loaddata.get_data_folder import get_local_drive

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
# from utils.behaviorlib import compute_dprime,smooth_rate_dprime,plot_psycurve #get support functions for beh analysis 
from utils.behaviorlib import * # get support functions for beh analysis 

# savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Behavior\\Detection\\'
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\')

############## Load the data - MaxOnly ####################################
protocol                = ['DM']
sessions,nsessions      = filter_sessions(protocol,load_behaviordata=True,load_videodata=True,has_pupil=True)

protocol                = 'DM'
session_list            = np.array([['LPE11495', '2024_02_16'],['LPE11622', '2024_02_16']])
sessions,nsessions      = load_sessions(protocol,session_list,load_behaviordata=True) #no behav or ca data

sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
nanimals                = len(np.unique(sessiondata['animal_id']))

#### 
sesidx = 1
fig         = plot_psycurve([sessions[sesidx]])
fig.savefig(os.path.join(savedir,'Performance','PsyCurve_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

################################################################
### Show the overall dprime for each animal across sessions:

dp_ses      = np.zeros([nsessions])
dp_ses_eng  = np.zeros([nsessions])
cr_ses      = np.zeros([nsessions])
cr_ses_eng  = np.zeros([nsessions])
for i,ses in enumerate(sessions):
    idx = np.isin(ses.trialdata['signal'],[0,100])
    df = ses.trialdata[idx]
    dp_ses[i],cr_ses[i]     = compute_dprime(df['signal']>0,df['lickResponse'])

    #Engaged only:
    idx = np.isin(ses.trialdata['signal'],[0,100])
    idx = np.logical_and(idx,ses.trialdata['engaged']==1)
    df = ses.trialdata[idx]
    dp_ses_eng[i],cr_ses_eng[i]     = compute_dprime(df['signal']>0,df['lickResponse'])
    # dp_ses_eng[i],cr_ses_eng[i]     = compute_dprime(df['signal']==100,df['lickResponse'])

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessiondata['dprime']    = dp_ses #dp_target_eng
sessiondata['dprime_eng']    = dp_ses_eng #dp_target_eng

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,3),sharey=True)

sns.stripplot(data = sessiondata,x='animal_id',y='dprime',hue='animal_id',palette='Dark2',size=6,ax=ax1)
ax1.axhline(y = 0, color = 'k', linestyle = ':')
ax1.set_title('Total Session', fontsize=11)
sns.stripplot(data = sessiondata,x='animal_id',y='dprime_eng',hue='animal_id',palette='Dark2',size=6,ax=ax2)
ax2.axhline(y = 0, color = 'k', linestyle = ':')
ax2.set_title('Engaged Only', fontsize=11)
plt.tight_layout()

# plt.savefig(os.path.join(savedir,'Performance','Dprime_LPE10884_1ses' + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'Performance','Dprime_%danimals' % nanimals + '.png'), format = 'png')

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

