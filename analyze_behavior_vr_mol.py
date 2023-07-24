# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import math
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.stats as st
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
from loaddata.session_info import filter_sessions,load_sessions,report_sessions

from utils import compute_dprime

############## Load the data ####################################
protocol            = ['VR']
sessions            = filter_sessions(protocol)


def compute_dprime(signal,response):
    
    ntrials             = len(signal)
    hit_rate            = sum((signal == 1) & (response == True)) / ntrials
    falsealarm_rate     = sum((signal == 0) & (response == True)) / ntrials
    
    dprime             = st.norm.ppf(hit_rate) - st.norm.ppf(falsealarm_rate)
    return dprime

compute_dprime(sessions[0].trialdata['trialType']=='G',sessions[0].trialdata['lickResponse'])

signal = sessions[3].trialdata['trialType']=='G'
response = sessions[3].trialdata['lickResponse']

compute_dprime(signal,response)

sessions[3].trialdata.head(25)
sessions[3].sessiondata.head()

nsessions = len(sessions)

### Show the overall dprime for each animal across sessions:

dp_ses = np.zeros([nsessions])
for i,ses in enumerate(sessions):
    dp_ses[i] = compute_dprime(ses.trialdata['trialType']=='G',ses.trialdata['lickResponse'])

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessiondata['dprime'] = dp_ses

plt.figure(figsize=(4,5))
ax = sns.stripplot(data = sessiondata,x='animal_id',y='dprime',palette='Dark2',size=10)

# plot the mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="animal_id",
            y="dprime",
            data=sessiondata,
            showfliers=False,
            showbox=False,
            showcaps=False,
            palette='Dark2',
            ax=ax)


### The dprime for left vs right context blocks:

dp_blocks = np.zeros((nsessions,2))
for i,ses in enumerate(sessions):
    dp_blocks[i,0] = compute_dprime(ses.trialdata['trialType'][ses.trialdata['context']==0]=='G',ses.trialdata['lickResponse'])
    dp_blocks[i,1] = compute_dprime(ses.trialdata['trialType']=='G',ses.trialdata['lickResponse'])

trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

### The hit rate as function of trial in block:
sns.lineplot(data=trialdata,x='n_in_block',y='lickResponse',hue='trialType',palette='Set1')




ntrials             = len(trd)
hit_rate            = sum((trd.rewardtrial == 1) & (trd.lickresponse == True)) / ntrials
falsealarm_rate     = sum((trd.rewardtrial == 0) & (trd.lickresponse == True)) / ntrials

d_prime             = st.norm.ppf(hit_rate) - st.norm.ppf(falsealarm_rate)

hit_rate_L          = sum((trd.rewardtrial == 1) & (trd.lickresponse == True) & (trd.stimleft=='stimA')) / ntrials
falsealarm_rate_L   = sum((trd.rewardtrial == 0) & (trd.lickresponse == True) & (trd.stimleft=='stimB')) / ntrials
hit_rate_R          = sum((trd.rewardtrial == 1) & (trd.lickresponse == True) & (trd.stimright=='stimA')) / ntrials
falsealarm_rate_R   = sum((trd.rewardtrial == 0) & (trd.lickresponse == True) & (trd.stimright=='stimB')) / ntrials

d_prime_L           = st.norm.ppf(hit_rate_L) - st.norm.ppf(falsealarm_rate_L)
d_prime_R           = st.norm.ppf(hit_rate_R) - st.norm.ppf(falsealarm_rate_R)



temp        = trd['lickresponse'].astype(float)
tempfilt    = gaussian_filter(temp, sigma=15)

plt.plot(trd['trialnum'],tempfilt)
plt.xlabel('trial number')
plt.ylabel('lickresponse')
plt.show()

## Parameters for spatial binning
s_pre       = -100 #pre cm
s_post      = 100   #post cm

binsize     = 2 #spatial binning in cm

binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
bincenters  = np.arange(s_pre,s_post+binsize,binsize)

harptrialtemp   = nwbfile.acquisition['TrialNumber'].data[:]
runspeedtemp    = nwbfile.acquisition['RunningSpeed'].data[:]
zpostemp        = nwbfile.acquisition['CorridorPosition'].data[:]
ts              = nwbfile.acquisition['CorridorPosition'].timestamps[:]

runPSTH     = np.empty(shape=(ntrials, len(bincenters)))

for itrial in range(ntrials):
    idx = harptrialtemp==itrial+1
    runPSTH[itrial,:] = binned_statistic(zpostemp[idx]-trd.loc[0, 'stimstart'],runspeedtemp[idx], statistic='mean', bins=binedges)[0]

    
licks_ts        = nwbfile.processing["behavior"]["Licks"]["Licks"].timestamps[:]

lickhistses     = np.histogram(licks_ts,np.append(ts,ts[-1]))[0]

lickPSTH        = np.empty(shape=(ntrials, len(bincenters)))

for itrial in range(ntrials-1):
    idx = harptrialtemp==itrial+1
    lickPSTH[itrial,:] = binned_statistic(zpostemp[idx]-trd.loc[0, 'stimstart'],lickhistses[idx], statistic='sum', bins=binedges)[0]


idx_gonogo          = np.empty(shape=(ntrials, 4),dtype=bool)
idx_gonogo[:,0]     = (trd['rewardtrial'] == 1) & (trd.lickresponse == True)
idx_gonogo[:,1]     = (trd['rewardtrial'] == 1) & (trd.lickresponse == False)
idx_gonogo[:,2]     = (trd['rewardtrial'] == 0) & (trd.lickresponse == True)
idx_gonogo[:,3]     = (trd['rewardtrial'] == 0) & (trd.lickresponse == False)

idx_gonogo

labels_gonogo       = ['HIT', 'MISS', 'FA', 'CR']

###
fig, ax = plt.subplots()

for i in range(4):
    # ax.plot(bincenters,np.nanmean(runPSTH[idx_gonogo[:,i],:],axis=0))
    data_mean = np.nanmean(runPSTH[idx_gonogo[:,i] & (trd.trialnum<300),:],axis=0)
    data_error = np.nanstd(runPSTH[idx_gonogo[:,i] & (trd.trialnum<300),:],axis=0) / math.sqrt(sum(idx_gonogo[:,i] & (trd.trialnum<300)))
    ax.plot(bincenters,data_mean,label=labels_gonogo[i])
    ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.5, linewidth=0)


ax.legend()
ax.set_ylim(0,50)
ax.set_xlim(-80,80)
ax.set_xlabel('Position rel. to stimulus onset (cm)')
ax.set_ylabel('Running speed (cm/s)')
# ax.fill_between([0,30], [0,50], [0,50],alpha=0.5)
ax.add_patch(matplotlib.patches.Rectangle((0,0),30,50, 
                        fill = True, alpha=0.2,
                        color = "blue",
                        linewidth = 0))
ax.add_patch(matplotlib.patches.Rectangle((30,0),30,50, 
                        fill = True, alpha=0.2,
                        color = "green",
                        linewidth = 0))

plt.text(5, 45, 'Stim',fontsize=12)
plt.text(35, 45, 'Reward',fontsize=12)

##### 

fig, ax = plt.subplots()

for i in range(4):
    # ax.plot(bincenters,np.nanmean(runPSTH[idx_gonogo[:,i],:],axis=0))
    data_mean   = np.nanmean(lickPSTH[idx_gonogo[:,i] & (trd.trialnum<300),:],axis=0)
    data_error  = np.nanstd(lickPSTH[idx_gonogo[:,i] & (trd.trialnum<300),:],axis=0) / math.sqrt(sum(idx_gonogo[:,i] & (trd.trialnum<300)))
    ax.plot(bincenters,data_mean,label=labels_gonogo[i])
    ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.5, linewidth=0)

ax.legend()
ax.set_ylim(0,1.6)
ax.set_xlim(-80,80)
ax.set_xlabel('Position rel. to stimulus onset (cm)')
ax.set_ylabel('Lick Rate (Hz)')
ax.add_patch(matplotlib.patches.Rectangle((0,0),30,1.6, 
                        fill = True, alpha=0.2,
                        color = "blue",
                        linewidth = 0))
ax.add_patch(matplotlib.patches.Rectangle((30,0),30,1.6, 
                        fill = True, alpha=0.2,
                        color = "green",
                        linewidth = 0))

plt.text(5, 1.4, 'Stim',fontsize=12)
plt.text(35, 1.4, 'Reward',fontsize=12)

#####

idx_gonogo          = np.empty(shape=(ntrials, 4),dtype=bool)
idx_gonogo[:,0]     = (trd['rewardtrial'] == 1) & (trd.lickresponse == True)
idx_gonogo[:,1]     = (trd['rewardtrial'] == 1) & (trd.lickresponse == False)
idx_gonogo[:,2]     = (trd['rewardtrial'] == 0) & (trd.lickresponse == True)
idx_gonogo[:,3]     = (trd['rewardtrial'] == 0) & (trd.lickresponse == False)

smooth_hitrate        = np.empty(shape=(ntrials, 1))
smooth_farate         = np.empty(shape=(ntrials, 1))

window_size = 30;

for itrial in range(window_size,ntrials):
    smooth_hitrate[itrial,0] = sum(idx_gonogo[itrial-window_size:itrial,0]) / (sum(idx_gonogo[itrial-window_size:itrial,0]) + sum(idx_gonogo[itrial-window_size:itrial,1]))
    smooth_farate[itrial,0] = sum(idx_gonogo[itrial-window_size:itrial,2]) / (sum(idx_gonogo[itrial-window_size:itrial,2]) + sum(idx_gonogo[itrial-window_size:itrial,3]))

smooth_hitrate[smooth_hitrate<0.001] = 0.001
smooth_hitrate[smooth_hitrate>0.999] = 0.999
smooth_farate[smooth_farate<0.001] = 0.001
smooth_farate[smooth_farate>0.999] = 0.999

smooth_d_prime           = st.norm.ppf(smooth_hitrate) - st.norm.ppf(smooth_farate)

fig, ax = plt.subplots()

plt.plot(trd['trialnum'],smooth_hitrate,color="green")
plt.plot(trd['trialnum'],smooth_farate,color="brown")
plt.xlabel('trial number')
plt.ylabel('HITrate / FArate')
# plt.ylim(0,50)
plt.xlim(window_size,)
plt.legend(['HIT','FA'])
colors = ["cyan","pink"]
for iblock in np.arange(0,ntrials,100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,1.0, 
                        fill = True, alpha=0.2,
                        color = colors[0], linewidth = 0))
for iblock in np.arange(50,ntrials,100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,1.0, 
                        fill = True, alpha=0.2,
                        color = colors[1], linewidth = 0))
    
fig, ax = plt.subplots()
plt.plot(trd['trialnum'],smooth_d_prime,color="blue")
plt.xlabel('trial number')
plt.ylabel('Dprime')
plt.ylim(0,5)
plt.xlim(window_size,)
plt.legend(['Dprime'])
colors = ["cyan","pink"]
for iblock in np.arange(0,ntrials,100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,5.0, 
                        fill = True, alpha=0.2,
                        color = colors[0], linewidth = 0))
for iblock in np.arange(50,ntrials,100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,5.0, 
                        fill = True, alpha=0.2,
                        color = colors[1], linewidth = 0))