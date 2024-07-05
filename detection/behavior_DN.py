# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#TODO
# filter runspeed
# plot individual trials locomotion, get sense of variance

import math
import pandas as pd
import os
os.chdir('D:\\Python\\molanalysis\\')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.plotting_style import * # get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 

savedir = 'D:\\OneDrive\\PostDoc\\Figures\\Detection\\'

########################### Load the data  #######################
protocol            = ['DN']
sessions,nsessions  = filter_sessions(protocol,load_behaviordata=True)
sessions,nsessions  = filter_sessions(protocol,load_behaviordata=True,only_animal_id='LPE11622')
sessions,nsessions  = filter_sessions(protocol,load_behaviordata=True,only_animal_id='LPE12385')

protocol            = 'DN'
session_list = np.array([['LPE10884', '2024_01_16']])
session_list = np.array([['LPE11622', '2024_02_20']])
session_list = np.array([['LPE12013', '2024_04_25']])
sessions,nsessions = load_sessions(protocol,session_list,load_behaviordata=True) #no behav or ca data

#%% ### Show for all sessions which region of the psychometric curve the noise spans #############
sessions = noise_to_psy(sessions,filter_engaged=True)

idx_inclthr = np.empty(nsessions).astype('int')
for ises,ses in enumerate(sessions):
    idx_inclthr[ises] = int(np.logical_and(np.any(sessions[ises].trialdata['signal_psy']<=0),np.any(sessions[ises].trialdata['signal_psy']>=0)))
    ses.sessiondata['incl_thr'] = idx_inclthr[ises]

stepsize = 0.5
binedges = np.arange(-3,3+stepsize/2,step=stepsize)
bincenters = binedges[:-1]+stepsize/2
data_occ = np.empty((nsessions,len(bincenters)))
for ises,ses in enumerate(sessions):
    data_occ[ises,:] = np.histogram(sessions[ises].trialdata['signal_psy'],bins=binedges)[0]>0
data_occ = np.sum(data_occ[idx_inclthr==1,:],axis=0)
data_occ /= np.sum(idx_inclthr)

fig,axes = plt.subplots(2,1,figsize=(5,7),gridspec_kw={'height_ratios': [1, 3]})

clrs_incl = ['red','green']

# fig.add_subplot(1,3,[])
# axes[0].plot(bincenters,data_occ)
axes[0].fill_between(bincenters,data_occ,where=data_occ>=0, color='green')
axes[0].set_xticks([-2,-1,0,1,2])
axes[0].set_xticklabels([])
axes[0].xaxis.grid() # vertical lines
axes[0].set_ylabel('Fraction of sessions')
axes[0].set_ylim([0,1])

for ises,ses in enumerate(sessions):
    axes[1].plot(np.nanpercentile(sessions[ises].trialdata['signal_psy'],[0,100]),[ises,ises],
            c=clrs_incl[idx_inclthr[ises]],linewidth=8)
axes[1].set_xlim([-2.9,2.9])
axes[1].set_ylim([-0.5,nsessions-0.5])
axes[1].set_xticks([-2,-1,0,1,2])
axes[1].set_xticklabels(['-2 std','-1 std','thr','+1 std','+2 std'])
axes[1].xaxis.grid() # vertical lines
axes[1].set_yticks(np.arange(nsessions))
axes[1].set_yticklabels([s.sessiondata['session_id'][0] for s in sessions])
plt.tight_layout()

fig.savefig(os.path.join(savedir,'Noise','Signal_Psy_Span_%d.png' % nsessions))
sessions = [ses for ises,ses in enumerate(sessions) if ses.sessiondata['incl_thr'][0]]
nsessions = len(sessions)

#%% ############################ Spatial plots: Licking ###################################
#### Behavior as a function of distance within the corridor

trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)
sessions    = noise_to_psy(sessions,filter_engaged=True)

### licking across the trial:
for ises,ses in enumerate(sessions):
    [ses.lickPSTH,bincenters] = calc_lickPSTH(ses,binsize=2.5)

sesidx = 15
fig = plot_lick_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Noise','LickRate_Outcome_%s.png' % sessions[sesidx].session_id))

fig = plot_lick_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Noise','LickRate_Psy_%s.png' % sessions[sesidx].session_id))

#Concatenate sessions:
lickPSTH    = np.concatenate([ses.lickPSTH for ses in sessions],axis=0)
lickPSTH[lickPSTH>10] = np.nan

idx = trialdata['engaged']==1

fig = plot_lick_corridor_outcome(trialdata.loc[idx,:],lickPSTH[idx,:],bincenters)
fig.savefig(os.path.join(savedir,'Noise','LickRate_Outcome_%ssessions.png' % nsessions),bbox_inches='tight')
fig = plot_lick_corridor_psy(trialdata.loc[idx,:],lickPSTH[idx,:],bincenters,version='signal_psy',hitonly=True)
fig.savefig(os.path.join(savedir,'Noise','LickRate_Psy_%ssessions.png' % nsessions),bbox_inches='tight')

############################# Spatial plots: Running ###################################
### running across the trial:
for ises,ses in enumerate(sessions):
    [ses.runPSTH,bincenters] = calc_runPSTH(ses,binsize=2.5)

sesidx = 3

fig = plot_run_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Noise','RunSpeed_Outcome_%s.png' % sessions[sesidx].session_id))

fig = plot_run_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Noise','RunSpeed_Psy_%s.png' % sessions[sesidx].session_id))

#Concatenate sessions:
runPSTH    = np.concatenate([ses.runPSTH for ses in sessions],axis=0)

idx = trialdata['engaged']==1

fig = plot_run_corridor_outcome(trialdata.loc[idx,:],runPSTH[idx,:],bincenters)
fig.savefig(os.path.join(savedir,'Noise','RunSpeed_Outcome_%ssessions.png' % nsessions))
fig = plot_run_corridor_psy(trialdata.loc[idx,:],runPSTH[idx,:],bincenters,version='signal')
fig.savefig(os.path.join(savedir,'Noise','RunSpeed_Psy_%ssessions.png' % nsessions),bbox_inches='tight')

fig = plot_run_corridor_psy(trialdata.loc[idx,:],runPSTH[idx,:],bincenters,version='signal_psy',hitonly=True)
fig.savefig(os.path.join(savedir,'Noise','RunSpeed_SignalPsy_%ssessions.png' % nsessions),bbox_inches='tight')

#%% ################### Plot psychometric curve #########################
sesidx = np.where([ses.sessiondata['session_id'][0]=='LPE12013_2024_04_29' for ses in sessions])[0][0]
sesidx = np.where([ses.sessiondata['session_id'][0]=='LPE11622_2024_02_27' for ses in sessions])[0][0]
fig = plot_psycurve([sessions[sesidx]],filter_engaged=True)
fig.savefig(os.path.join(savedir,'Noise','Psy_%s.png' % sessions[sesidx].session_id))

fig = plot_psycurve(sessions,filter_engaged=True)
fig.savefig(os.path.join(savedir,'Noise','Psy_%s_Engaged.png' % sessions[sesidx].session_id))

#%% 


# %%
