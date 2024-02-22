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
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.plotting_style import * # get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 

savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Behavior\\Detection\\'

########################### Load the data - Psy #######################
protocol            = ['DP']
sessions            = filter_sessions(protocol,load_behaviordata=True)

protocol            = 'DP'
session_list = np.array([['LPE11623', '2024_02_20'],
                         ['LPE11495', '2024_02_20']])
sessions = load_sessions(protocol,session_list,load_behaviordata=True) #no behav or ca data

nsessions = len(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
nanimals    = len(np.unique(sessiondata['animal_id']))


##################### Spatial plots ####################################
# Behavior as a function of distance within the corridor:
sesidx = 0
print(sessions[sesidx].sessiondata['session_id'])
### licking across the trial:
[sessions[sesidx].lickPSTH,bincenters] = lickPSTH(sessions[sesidx],binsize=5)

# fig = plot_lick_corridor_outcome(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
# sessions[sesidx].lickPSTH[-1,:] = 0
fig = plot_lick_corridor_psy(sessions[sesidx].trialdata,sessions[sesidx].lickPSTH,bincenters)
fig.savefig(os.path.join(savedir,'Performance','LickRate_Psy_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

### running across the trial:
[sessions[sesidx].runPSTH,bincenters] = runPSTH(sessions[sesidx],binsize=5)

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



