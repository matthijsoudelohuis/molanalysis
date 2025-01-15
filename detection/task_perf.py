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
os.chdir('e:\\Python\\molanalysis\\')
from scipy.signal import medfilt

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.psth import compute_tensor_space,compute_respmat_space
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\')

#%% Load behavior of all protocols:
protocol                = ['DM','DP','DN']
sessions,nsessions      = filter_sessions(protocol)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
nanimals                = len(np.unique(sessiondata['animal_id']))

#%% Remove sessions LPE10884 that are too bad:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE10884_2023_12_14','LPE10884_2023_12_15','LPE10884_2024_01_11','LPE10884_2024_01_16']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nsessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Report on number of trials per session:
print('%3.1f +- %3.1f trials per session' % (np.mean(sessiondata['ntrials']),np.std(sessiondata['ntrials'])))
print(f"{len(sessiondata[sessiondata['protocol']=='DM'])} DM, \n"
      f"{len(sessiondata[sessiondata['protocol']=='DP'])} DP, \n"
      f"{len(sessiondata[sessiondata['protocol']=='DN'])} DN sessions\n")

#%% ###############################################################
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

#%% Show figure of dprime for all animals:
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessiondata['dprime']       = dp_ses #dp_target_eng
sessiondata['dprime_eng']   = dp_ses_eng #dp_target_eng

fig, ax = plt.subplots(1,1,figsize=(4,3))

sns.stripplot(data = sessiondata,x='animal_id',y='dprime_eng',hue='animal_id',palette='Dark2',size=6,ax=ax)
# sns.stripplot(data = sessiondata,x='animal_id',y='dprime_eng',hue='animal_id',palette='Dark2',size=6,ax=ax,legend=False)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False,fontsize=8)
plt.subplots_adjust(right=0.7)
ax.axhline(y = 0, color = 'k', linestyle = ':')

# ax.set_title('Total Session', fontsize=11)
ax.set_xticks(range(nanimals))
ax.set_xticklabels(range(1,nanimals+1))
ax.errorbar(x=nanimals,y=sessiondata['dprime_eng'].mean(),yerr=sessiondata['dprime_eng'].std(),fmt='o',color='k',capsize=3)
ax.set_xticks(range(nanimals+1))
ax.set_xticklabels([str(x) for x in range(1,nanimals+1)] + [u'\u03BC'])
ax.set_ylabel('Dprime')
plt.tight_layout()
print('Mean Dprime: %.2f +/- %.2f' % (sessiondata['dprime_eng'].mean(),sessiondata['dprime_eng'].std()))

plt.savefig(os.path.join(savedir,'Performance','Dprime_%danimals' % nanimals + '.png'), format = 'png')


#%% ########################## Load psychometric data #######################
protocol            = ['DP','DN']
sessions,nsessions  = filter_sessions(protocol,load_behaviordata=False)
sessions            = stim_remapping(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# Remove sessions LPE10884 that are too bad:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE10884_2023_12_14','LPE10884_2023_12_15','LPE10884_2024_01_11','LPE10884_2024_01_16']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nsessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)


#%% ########################## Show psychometric curve for an example session #######################
# fig = plot_psycurve(sessions,filter_engaged=True)
example_session = 'LPE11998_2024_04_16'
# example_session = 'LPE11998_2024_04_18'
sesidx  = np.where(np.array([ses.sessiondata['session_id'][0] for ses in sessions]) == example_session)[0][0]
fig     = plot_psycurve([sessions[sesidx]],filter_engaged=True)
fig.savefig(os.path.join(savedir,'Performance','Psycurve_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ########################## Show psychometric curve for an example session #######################
fig     = plot_all_psycurve(sessions,filter_engaged=True)
fig.savefig(os.path.join(savedir,'Performance','Psycurve_%dsessions' % nsessions + '.png'), format = 'png')

#%% For the different stimuli:
print('Number of sessions per stim:')
print(sessiondata['stim'].value_counts())

filter_engaged = True
params = np.empty((nsessions,4))
r2_ses = np.empty((nsessions))
for ises,ses in enumerate(sessions):
    trialdata = ses.trialdata
    if filter_engaged:
        trialdata = trialdata[trialdata['engaged']==1]
    params[ises,:],r2_ses[ises] = fit_psycurve(trialdata,printoutput=False)

sessiondata[['mu', 'sigma', 'lapse_rate', 'guess_rate']] = params

#%%
fig,ax = plt.subplots(1,1,figsize=(3,3))
sns.stripplot(data=sessiondata,x='stim',y='mu',hue='stim',palette='tab10',ax=ax,
              legend=False,order=np.sort(sessiondata['stim'].unique()))
ax.set_ylabel('Threshold (% signal)')
ax.set_xlabel('Stim')

for istim,stim in enumerate(sessiondata['stim'].unique()):
    stimdata = sessiondata[sessiondata['stim']==stim]
    mean = np.nanmean(stimdata['mu'])
    error = np.nanstd(stimdata['mu']) / np.sqrt(len(stimdata))
    error = np.nanstd(stimdata['mu'])
    ax.errorbar(x=istim+0.3,y=mean,yerr=error,fmt='o',color='k',capsize=0)
ax.set_ylim([0,35])
ax.set_yticks([0,10,20,30])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Performance','Threshold_per_stim_%danimals' % nanimals + '.png'), format = 'png')

#%% 
df = pd.DataFrame(params,columns=['mu','sigma','lapse_rate','guess_rate'])
df['r2'] = r2_ses
for i,ses in enumerate(sessions):
    df.loc[i,'session_id'] = ses.sessiondata['session_id'][0]
    df.loc[i,'animal_id'] = ses.sessiondata['animal_id'][0]

#%% 
fig,axes = plt.subplots(2,3,figsize=(8,6),sharex=True)
for i,param in enumerate(df.columns[:5]):
    ax = axes[i//3,i%3]
    if i==4:
        sns.stripplot(data=df,y=param,ax=ax,hue='animal_id',palette='tab10',legend=True)
    else:
        sns.stripplot(data=df,y=param,ax=ax,hue='animal_id',palette='tab10',legend=False)

    # sns.stripplot(data=df,x=param,y='animal_id',ax=axes[i],palette='tab10',order=np.sort(sessiondata['animal_id'].unique()),hue='animal_id',legend=False)
    ax.set_xlabel('')
    ax.set_title(param)
    # axes[i].set_xticks([])
    if i==4:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False,fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Performance','Params_per_session_%danimals' % nanimals + '.png'), format = 'png')

# fig,axes = plt.subplots(1,4,figsize=(12,3),sharex=True)
# sns.stripplot(data=df,y=param,ax=axes[i],hue='animal_id',legend=False)


#%% Correlations between psy fit parameters:
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df.corr(), ax=ax, 
            cmap='coolwarm', 
            annot=True, 
            linewidths=0.5, 
            square=True,
            cbar_kws={'shrink': 0.5})
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.title('Correlation between psy fit parameters')
plt.savefig(os.path.join(savedir,'Performance','Corrmat_params_%danimals' % nanimals + '.png'), format = 'png')
