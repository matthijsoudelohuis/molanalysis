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

########################### Load the data - Psy #######################
protocol            = ['DN']
sessions,nsessions  = filter_sessions(protocol,load_behaviordata=True)
sessions,nsessions  = filter_sessions(protocol,load_behaviordata=True,only_animal_id='LPE11622')

protocol            = 'DN'
session_list = np.array([['LPE10884', '2024_01_16']])
session_list = np.array([['LPE11622', '2024_02_20']])
session_list = np.array([['LPE12013', '2024_04_25']])
sessions,nsessions = load_sessions(protocol,session_list,load_behaviordata=True) #no behav or ca data


##################### Spatial plots ####################################
# Behavior as a function of distance within the corridor:
sesidx = 11

### licking across the trial:
[sessions[sesidx].lickPSTH,bincenters] = lickPSTH(sessions[sesidx],binsize=5)

fig = plot_lick_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Noise','LickRate_Outcome_%s.png' % sessions[sesidx].session_id))

fig = plot_lick_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Noise','LickRate_Psy_%s.png' % sessions[sesidx].session_id))

### running across the trial:
[sessions[sesidx].runPSTH,bincenters] = runPSTH(sessions[sesidx],binsize=5)

fig = plot_run_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Noise','RunSpeed_Outcome_%s.png' % sessions[sesidx].session_id))
fig = plot_run_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].runPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Noise','RunSpeed_Psy_%s.png' % sessions[sesidx].session_id))

##################### Plot psychometric curve #########################

fig = plot_psycurve([sessions[12]])
fig = plot_psycurve(sessions,filter_engaged=True)
fig.savefig(os.path.join(savedir,'Noise','Psy_%s.png' % sessions[sesidx].session_id))

fig = plot_psycurve(sessions,filter_engaged=True)
fig.savefig(os.path.join(savedir,'Noise','Psy_%s_Engaged.png' % sessions[sesidx].session_id))

#%% ### Show for all sessions which region of the psychometric curve the noise spans #############
sessions = noise_to_psy(sessions,filter_engaged=True)

fig,ax = plt.subplots(1,1,figsize=(6,6))
clrs_incl = ['red','blue']

for ises,ses in enumerate(sessions):
    c_ind = int(np.logical_and(np.any(sessions[ises].trialdata['signal_psy']<0),np.any(sessions[ises].trialdata['signal_psy']>0)))
    ax.plot(np.nanpercentile(sessions[ises].trialdata['signal_psy'],[0,100]),[ises,ises],
            c=clrs_incl[c_ind],linewidth=10)
ax.set_xlim([-2.9,2.9])
ax.set_ylim([-0.5,nsessions-0.5])
ax.set_xticks([-2,-1,0,1,2])
ax.set_xticklabels(['-2 std','-1 std','thr','+1 std','+2 std'])
ax.xaxis.grid() # vertical lines
ax.set_yticks(np.arange(nsessions))
ax.set_yticklabels([s.sessiondata['session_id'][0] for s in sessions])

fig.savefig(os.path.join(savedir,'Noise','Signal_Psy_Span_%d.png' % nsessions))


# %%
