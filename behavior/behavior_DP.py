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

############## Load the data - Psy ####################################
protocol            = ['DP']
sessions            = filter_sessions(protocol,load_behaviordata=True)

nsessions = len(sessions)





# def psy curve: 

fig, ax = plt.subplots()

conds = np.sort(pd.unique(trialdata['signal']))
nconds = len(conds)

for ises,ses in enumerate(sessions):
    psy = ses.trialdata.groupby(['signal'])['lickResponse'].mean()
    plt.plot(psy,'k',marker='o',markersize=12)

ax.set_ylim(0,1)
ax.set_xlim(0,100)
ax.set_xlabel('Signal')
ax.set_ylabel('Response Rate')

if nconds>4:

    fit 


# Generate some example data
np.random.seed(42)
x_data = np.linspace(-5, 5, 100)
true_params = [0.5, 1.5, 0.05, 0.02]  # Example parameters (mu, sigma, lapse_rate, guess_rate)
y_data_true = psychometric_function(x_data, *true_params)
y_data_noise = y_data_true + np.random.normal(0, 0.02, size=len(x_data))  # Add some noise

# Fit the psychometric curve to the data using curve_fit
initial_guess = [0, 1, 0.1, 0.01]  # Initial guess for parameters


import numpy as np
from scipy.optimize import curve_fit
import scipy as sy
import matplotlib.pyplot as plt

d = np.array([75, 80, 90, 95, 100, 105, 110, 115, 120, 125], dtype=float)
p2 = np.array([6, 13, 25, 29, 29, 29, 30, 29, 30, 30], dtype=float) / 30. # scale to 0..1

# psychometric function
def pf(x, alpha, beta):
    return 1. / (1 + np.exp( -(x-alpha)/beta ))

# fitting
par0 = sy.array([100., 1.]) # use some good starting values, reasonable default is [0., 1.]
par, mcov = curve_fit(pf, d, p2, par0)
print(par)
plt.plot(d, p2, 'ro')
plt.plot(d, pf(d, par[0], par[1]))
plt.show()


################ Spatial plots ##############################################
# Behavior as a function of distance within the corridor:

def runPSTH(ses,s_pre = -100, s_post = 50, binsize = 5):

    ## Parameters for spatial binning
    # s_pre       #pre cm
    # s_post      #post cm
    # binsize     #spatial binning in cm
    binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(s_pre,s_post+binsize,binsize)

    # trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

    trialdata   = ses.trialdata
    runPSTH     = np.empty((len(ses.trialdata),len(bincenters)))
                    
    for ises,ses in enumerate(sessions):
        ntrials     = len(ses.trialdata)
        runPSTH_ses     = np.empty(shape=(ntrials, len(bincenters)))

        for itrial in range(ntrials):
            idx = ses.behaviordata['trialnumber']==itrial+1
            runPSTH_ses[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][0],
                                                ses.behaviordata['runspeed'][idx], statistic='mean', bins=binedges)[0]
        runPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = runPSTH_ses
        

    return runPSTH, bincenters


def lickPSTH(ses,s_pre = -100, s_post = 50, binsize = 5):

    ## Parameters for spatial binning
    # s_pre       #pre cm
    # s_post      #post cm
    # binsize     #spatial binning in cm

    binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(s_pre,s_post+binsize,binsize)
    # trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)
    trialdata   = ses.trialdata

    lickPSTH     = np.empty((len(ses.trialdata),len(bincenters)))
                    
    for ises,ses in enumerate(sessions):
        ntrials     = len(ses.trialdata)
        lickPSTH_ses    = np.empty(shape=(ntrials, len(bincenters)))

        for itrial in range(ntrials-1):
            idx = ses.behaviordata['trialnumber']==itrial+1
            lickPSTH_ses[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][0],
                                                ses.behaviordata['lick'][idx], statistic='sum', bins=binedges)[0]
        lickPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = lickPSTH_ses
    lickPSTH /= binsize 

    return lickPSTH, bincenters

def plot_lick_corridor_outcome(trialdata,lickPSTH,bincenters):
    ### Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()

    ttypes = pd.unique(trialdata['trialOutcome'])
    ttypes = ['CR', 'MISS', 'HIT','FA']

    for i,ttype in enumerate(ttypes):
        # idx = np.logical_and(trialdata['trialOutcome']==ttype)
        idx = trialdata['trialOutcome']==ttype
        data_mean = np.nanmean(lickPSTH[idx,:],axis=0)
        data_error = np.nanstd(lickPSTH[idx,:],axis=0) / math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype)
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.5, linewidth=0)

    ax.legend()
    ax.set_ylim(0,2)
    ax.set_xlim(-90,60)
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Lick Rate (licks/cm)')
    # ax.fill_between([0,30], [0,50], [0,50],alpha=0.5)
    ax.add_patch(matplotlib.patches.Rectangle((0,0),20,1.8, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((20,0),20,1.8, 
                            fill = True, alpha=0.2,
                            color = "green",
                            linewidth = 0))

    plt.text(5, 5.2, 'Stim',fontsize=12)
    plt.text(25, 5.2, 'Reward',fontsize=12)

    return 

def plot_lick_corridor_psy(trialdata,lickPSTH,bincenters):

### Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()
    ttypes = np.sort(pd.unique(trialdata['signal']))

    for i,ttype in enumerate(ttypes):
        # idx = np.logical_and(trialdata['signal']==ttype,trialdata['lickResponse']==1)
        idx = trialdata['signal']==ttype
        data_mean = np.nanmean(lickPSTH[idx,:],axis=0)
        data_error = np.nanstd(lickPSTH[idx,:],axis=0) / math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype)
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.5, linewidth=0)

    ax.legend()
    ax.set_ylim(0,2)
    ax.set_xlim(-90,60)
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Lick Rate (Hz)')
    # ax.fill_between([0,30], [0,50], [0,50],alpha=0.5)
    ax.add_patch(matplotlib.patches.Rectangle((0,0),20,2, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((20,0),20,2, 
                            fill = True, alpha=0.2,
                            color = "green",
                            linewidth = 0))

    plt.text(5, 1.8, 'Stim',fontsize=12)
    plt.text(25, 1.8, 'Reward',fontsize=12)

    return 


def plot_run_corridor_psy(trialdata,lickPSTH,bincenters):
    ### Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()
    ttypes = np.sort(pd.unique(trialdata['signal']))

    for i,ttype in enumerate(ttypes):
        # idx = np.logical_and(trialdata['signal']==ttype,trialdata['lickResponse']==1)
        idx = trialdata['signal']==ttype
        data_mean = np.nanmean(lickPSTH[idx,:],axis=0)
        data_error = np.nanstd(lickPSTH[idx,:],axis=0) / math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype)
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.5, linewidth=0)

    ax.legend()
    ax.set_ylim(0,2)
    ax.set_xlim(-90,60)
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Lick Rate (Hz)')
    # ax.fill_between([0,30], [0,50], [0,50],alpha=0.5)
    ax.add_patch(matplotlib.patches.Rectangle((0,0),20,2, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((20,0),20,2, 
                            fill = True, alpha=0.2,
                            color = "green",
                            linewidth = 0))

    plt.text(5, 1.8, 'Stim',fontsize=12)
    plt.text(25, 1.8, 'Reward',fontsize=12)

    return 



[sessions[0].lickPSTH,bincenters] = lickPSTH(sessions[0],binsize=10)

plot_lick_corridor_outcome(sessions[0].trialdata,sessions[0].lickPSTH,bincenters)
plot_lick_corridor_psy(sessions[0].trialdata,sessions[0].lickPSTH,bincenters)

[sessions[0].runPSTH,bincenters] = runPSTH(sessions[0],binsize=10)

plot_run_corridor_outcome(sessions[0].trialdata,sessions[0].lickPSTH,bincenters)
plot_run_corridor_psy(sessions[0].trialdata,sessions[0].lickPSTH,bincenters)

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


