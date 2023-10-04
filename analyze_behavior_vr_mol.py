# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

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

# from utils import compute_dprime
savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Behavior - VR\\'

############## Load the data ####################################
protocol            = ['VR']
sessions            = filter_sessions(protocol,load_behaviordata=True)


def compute_dprime(signal,response):
    assert(len(signal)==len(response))
    ntrials             = len(signal)
    hit_rate            = sum((signal == 1) & (response == True)) / sum(signal==1)
    falsealarm_rate     = sum((signal == 0) & (response == True)) / sum(signal==0)
    
    dprime             = st.norm.ppf(hit_rate) - st.norm.ppf(falsealarm_rate)
    return dprime


nsessions = len(sessions)

################################################################
### Show the overall dprime for each animal across sessions:

dp_target = np.zeros([nsessions])
dp_distr = np.zeros([nsessions])
for i,ses in enumerate(sessions):
    # dp_ses[i] = compute_dprime(ses.trialdata['trialType']=='G',ses.trialdata['lickResponse'])
    # dp_target[i]    = compute_dprime(ses.trialdata['stimRight_norm']==0,ses.trialdata['lickResponse'])
    # dp_target[i]    = compute_dprime(ses.trialdata['trialType']=='G',ses.trialdata['lickResponse'])
    
    dp_target[i]     = compute_dprime(np.logical_or(ses.trialdata['stimRight_norm']==0,ses.trialdata['stimLeft_norm']==0),ses.trialdata['lickResponse'])
    dp_distr[i]      = compute_dprime(np.logical_or(ses.trialdata['stimRight_norm']==2,ses.trialdata['stimLeft_norm']==2),ses.trialdata['lickResponse'])

    #Engaged only:
    df = ses.trialdata[ses.trialdata['engaged']==1]
    dp_target_eng[i]     = compute_dprime(np.logical_or(df['stimRight_norm']==0,df['stimLeft_norm']==0),df['lickResponse'])
    dp_distr_eng[i]      = compute_dprime(np.logical_or(df['stimRight_norm']==2,df['stimLeft_norm']==2),df['lickResponse'])

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessiondata['dprime_target']    = dp_target #dp_target_eng
sessiondata['dprime_distr']     = dp_distr #dp_distr_eng

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,3),sharey=True)

sns.stripplot(data = sessiondata,x='animal_id',y='dprime_target',palette='Dark2',size=10,ax=ax1)
ax1.axhline(y = 0, color = 'k', linestyle = ':')
ax1.set_title('Target', fontsize=11)
sns.stripplot(data = sessiondata,x='animal_id',y='dprime_distr',palette='Dark2',size=10,ax=ax2)
ax2.axhline(y = 0, color = 'k', linestyle = ':')
ax2.set_title('Distractor', fontsize=11)
plt.tight_layout()

plt.savefig(os.path.join(savedir,'Performance','Dprime_2animals_duringrecordings' + '.png'), format = 'png')
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
######## The dprime for left vs right context blocks:
dp_blocks = np.zeros((nsessions,2))
for ises,ses in enumerate(sessions):
    for iblock in pd.unique(ses.trialdata['context']):
        df = ses.trialdata[ses.trialdata['context']==iblock]
        dp_blocks[ises,iblock] = compute_dprime(df['rewardAvailable'],df['lickResponse'])

### Plotting:
df = pd.DataFrame({'dprime' : dp_blocks.flatten(order='F'), 
                   'context' : np.repeat(pd.unique(ses.trialdata['context']),nsessions), 
                   'animal_id' : np.tile(sessiondata.animal_id,2)})
plt.figure(figsize=(5,4))
sns.stripplot(data=df,x='context',y='dprime',size=10,hue='animal_id',palette='Dark2')
sns.barplot(data=df,x='context',y='dprime',palette='Greys')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Performance','Dprime_blocks' + '.png'), format = 'png')

#### construct concatenated trialdata DataFrame by appending all sessions:
trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

# ### The lick rate as function of trial in session:
# sns.lineplot(data=trialdata,x='TrialNumber',y='lickResponse',hue='trialType',palette='Set1')
sns.lineplot(x=np.round(trialdata['TrialNumber'],-1),y=trialdata['lickResponse'],hue=trialdata['trialType'],palette='Set1')
# sns.lineplot(x=np.floor(trialdata['TrialNumber']/50)*50,y=trialdata['lickResponse'],hue=trialdata['trialType'],palette='Set1')
# plt.savefig(os.path.join(savedir,'Licking','LickRate_acrosssession' + '.png'), format = 'png')

#### Smooth hit and fa rate and smooth dprime:
sigma = 25

for i,ses in enumerate(sessions):

    a       = np.empty((len(ses.trialdata)))
    a[:]    = np.nan
    x       = np.where(ses.trialdata['trialType']=='G')[0]
    y       = ses.trialdata['lickResponse'][x]
    f       = interp1d(x,y,fill_value="extrapolate")
    xnew    = np.arange(len(ses.trialdata))
    ynew    = f(xnew)   # use interpolation function returned by `interp1d`

    ses.trialdata['smooth_hitrate'] = gaussian_filter(ynew,sigma=sigma)

    a       = np.empty((len(ses.trialdata)))
    a[:]    = np.nan
    x       = np.where(ses.trialdata['trialType']=='N')[0]
    y       = ses.trialdata['lickResponse'][x]
    f       = interp1d(x,y,fill_value="extrapolate")
    xnew    = np.arange(len(ses.trialdata))
    ynew    = f(xnew)   # use interpolation function returned by `interp1d`

    ses.trialdata['smooth_farate'] = gaussian_filter(ynew,sigma=sigma)

    ses.trialdata['smooth_dprime'] = [st.norm.ppf(ses.trialdata['smooth_hitrate'][t]) - st.norm.ppf(ses.trialdata['smooth_farate'][t]) 
      for t in range(len(ses.trialdata))]

trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

################################################################
#### The hit rate as function of trial in session:

clrs_blocks = get_clr_blocks([])

fig,ax = plt.subplots(figsize=(7,4))
sns.lineplot(data=trialdata,x='TrialNumber',y='smooth_hitrate',color='g')
sns.lineplot(data=trialdata,x='TrialNumber',y='smooth_farate',color='r')
plt.ylabel('HIT / FA rate')

for iblock in np.arange(0,ax.get_xlim()[1],100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,1.0, 
                        fill = True, alpha=0.2, color=clrs_blocks[0], linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((iblock+50,0),50,1.0, 
                        fill = True, alpha=0.2, color=clrs_blocks[1], linewidth = 0))

plt.savefig(os.path.join(savedir,'Performance','HITFA_rate_acrosssession' + '.png'), format = 'png')

### individual sessions
fig,ax = plt.subplots(figsize=(7,4))
for i,ses in enumerate(sessions):
    plt.plot(ses.trialdata['TrialNumber'],ses.trialdata['smooth_hitrate'],color='g')
    plt.plot(ses.trialdata['TrialNumber'],ses.trialdata['smooth_farate'],color='r')
plt.ylabel('HIT / FA rate')
plt.xlabel('Trial Number')
for iblock in np.arange(0,ax.get_xlim()[1],100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,1.0, 
                        fill = True, alpha=0.2, color=clrs_blocks[0], linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((iblock+50,0),50,1.0, 
                        fill = True, alpha=0.2, color=clrs_blocks[1], linewidth = 0))
plt.savefig(os.path.join(savedir,'Performance','HITFA_rate_acrosssession_indiv' + '.png'), format = 'png')

### Dprime:
fig,ax = plt.subplots(figsize=(7,4))
sns.lineplot(data=trialdata,x='TrialNumber',y='smooth_dprime',color='k')
plt.ylabel('Dprime')
for iblock in np.arange(0,ax.get_xlim()[1],100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,0),50,ax.get_ylim()[1], 
                        fill = True, alpha=0.2, color=clrs_blocks[0], linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((iblock+50,0),50,ax.get_ylim()[1], 
                        fill = True, alpha=0.2, color=clrs_blocks[1], linewidth = 0))
plt.savefig(os.path.join(savedir,'Performance','Dprime_acrosssession' + '.png'), format = 'png')


fig,ax = plt.subplots(figsize=(7,4))
for i,ses in enumerate(sessions):
    plt.plot(ses.trialdata['TrialNumber'],ses.trialdata['smooth_dprime'],color='k')
plt.ylabel('Dprime')
plt.xlabel('Trial Number')
for iblock in np.arange(0,ax.get_xlim()[1],100):
    ax.add_patch(matplotlib.patches.Rectangle((iblock,-10),50,20, 
                        fill = True, alpha=0.2, color=clrs_blocks[0], linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((iblock+50,-10),50,20, 
                        fill = True, alpha=0.2, color=clrs_blocks[1], linewidth = 0))
ax.set_ylim([-0.5,ax.get_ylim()[1]])
ax.axhline(0,color='k',linestyle=':')
plt.savefig(os.path.join(savedir,'Performance','Dprime_acrosssession_indiv' + '.png'), format = 'png')

### The hit rate as function of trial in block:
ax = sns.lineplot(data=trialdata,x='n_in_block',y='lickResponse',hue='trialType',palette='Set1',
                           legend=['FA rate','HIT rate'])
h, l = ax.get_legend_handles_labels()
# plt.legend(labels=['FA rate','HIT rate'])
plt.legend(h,labels=['FA rate','HIT rate'])

### The dprime as function of trial in block:
ax = sns.lineplot(data=trialdata,x='n_in_block',y='smooth_dprime')
ax.axhline(0,color='k',linestyle=':')
plt.ylim([-0.2,1.5])
plt.xlim([0,50])


############ The dprime for blocks across the session:
trialdata['nblock'] = np.floor(trialdata['TrialNumber']/50)
nblocks     = int(np.max(trialdata['nblock']))
dp_blockid  = np.zeros(nblocks)
hr_blockid  = np.zeros(nblocks)
fa_blockid  = np.zeros(nblocks)

dp_ctxid    = np.tile([2,1],nblocks)
dp_ctxid    = dp_ctxid[:nblocks]
for iblock in range(nblocks):
    df                  = trialdata[trialdata['nblock']==iblock]
    dp_blockid[iblock]  = compute_dprime(df['rewardAvailable'],df['lickResponse'])
    hr_blockid[iblock]  = np.sum(np.logical_and(df['trialType']=='G',df['lickResponse']==1)) / np.sum(df['trialType']=='G')
    fa_blockid[iblock]  = np.sum(np.logical_and(df['trialType']=='N',df['lickResponse']==1)) / np.sum(df['trialType']=='N')

fig,ax = plt.subplots(figsize=(5,4))
sns.scatterplot(x=range(0,nblocks),y=dp_blockid,hue=dp_ctxid,palette=clrs_blocks)
plt.xlabel('Block number')
plt.ylabel('Dprime')
plt.savefig(os.path.join(savedir,'Performance','Dprime_acrossblocks' + '.png'), format = 'png')

fig,ax = plt.subplots(figsize=(5,4))
sns.scatterplot(x=range(0,nblocks),y=hr_blockid,hue=dp_ctxid,palette=['lightgreen','green'])
sns.scatterplot(x=range(0,nblocks),y=fa_blockid,hue=dp_ctxid,palette=['lightcoral','red'])
plt.xlabel('Block number')
plt.ylabel('HIT / FA Rate')
plt.savefig(os.path.join(savedir,'Performance','HR_FA_acrossblocks' + '.png'), format = 'png')


################ Spatial plots ##############################################
# Behavior as a function of distance within the corridor:

## Parameters for spatial binning
s_pre       = -100  #pre cm
s_post      = 100   #post cm
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
        idx = ses.behaviordata['trialnum']==itrial+1
        runPSTH_ses[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['StimStart'][0],
                                            ses.behaviordata['runspeed'][idx], statistic='mean', bins=binedges)[0]

    runPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = runPSTH_ses

    lickPSTH_ses    = np.empty(shape=(ntrials, len(bincenters)))

    for itrial in range(ntrials-1):
        idx = ses.behaviordata['trialnum']==itrial+1
        lickPSTH_ses[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['StimStart'][0],
                                            ses.behaviordata['lick'][idx], statistic='sum', bins=binedges)[0]
    lickPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = lickPSTH_ses


### Plot running speed as a function of trial type:

fig, ax = plt.subplots()

ttypes = pd.unique(trialdata['trialOutcome'])
ttypes = ['CR', 'MISS', 'HIT','FA']

for i,ttype in enumerate(ttypes):
    # ax.plot(bincenters,np.nanmean(runPSTH[idx_gonogo[:,i],:],axis=0))
    idx = np.logical_and(trialdata['trialOutcome']==ttype,trialdata['TrialNumber']<300)
    data_mean = np.nanmean(runPSTH[idx,:],axis=0)
    data_error = np.nanstd(runPSTH[idx,:],axis=0) / math.sqrt(sum(idx))
    ax.plot(bincenters,data_mean,label=ttype)
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



################################################################
### Plot licking rate as a function of trial type:

fig, ax = plt.subplots()

ttypes = pd.unique(trialdata['trialOutcome'])
ttypes = ['CR', 'MISS', 'HIT','FA']

for i,ttype in enumerate(ttypes):
    idx = np.logical_and(trialdata['trialOutcome']==ttype,trialdata['TrialNumber']<300)
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
ax.add_patch(matplotlib.patches.Rectangle((0,0),30,5.6, 
                        fill = True, alpha=0.2,
                        color = "blue",
                        linewidth = 0))
ax.add_patch(matplotlib.patches.Rectangle((30,0),30,5.6, 
                        fill = True, alpha=0.2,
                        color = "green",
                        linewidth = 0))

plt.text(5, 5.2, 'Stim',fontsize=12)
plt.text(35, 5.2, 'Reward',fontsize=12)

################################ 


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