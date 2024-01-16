# -*- coding: utf-8 -*-
"""
Set of function used for analysis of mouse behavior in visual navigation task
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025
"""

import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import scipy.stats as st
from scipy import special
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

from utils.plotting_style import * # get all the fixed color schemes

def compute_dprime(signal,response):
    
    ntrials             = len(signal)
    hit_rate            = sum((signal == 1) & (response == True)) / ntrials
    falsealarm_rate     = sum((signal == 0) & (response == True)) / ntrials
    
    dprime              = st.norm.ppf(hit_rate) - st.norm.ppf(falsealarm_rate)
    criterion           = -0.5 * (st.norm.ppf(hit_rate) + st.norm.ppf(falsealarm_rate))
    return dprime,criterion


def smooth_rate_dprime(sessions,sigma=25): #Smooth hit and fa rate and smooth dprime

    for i,ses in enumerate(sessions):

        a       = np.empty((len(ses.trialdata)))
        a[:]    = np.nan
        x       = np.where(ses.trialdata['signal']>0)[0]
        y       = ses.trialdata['lickResponse'][x]
        f       = interp1d(x,y,fill_value="extrapolate")
        xnew    = np.arange(len(ses.trialdata))
        ynew    = f(xnew)   # use interpolation function returned by `interp1d`

        ses.trialdata['smooth_hitrate'] = gaussian_filter(ynew,sigma=sigma)

        a       = np.empty((len(ses.trialdata)))
        a[:]    = np.nan
        x       = np.where(ses.trialdata['signal']==0)[0]
        y       = ses.trialdata['lickResponse'][x]
        f       = interp1d(x,y,fill_value="extrapolate")
        xnew    = np.arange(len(ses.trialdata))
        ynew    = f(xnew)   # use interpolation function returned by `interp1d`

        ses.trialdata['smooth_farate'] = gaussian_filter(ynew,sigma=sigma)

        HR_maxed,FR_maxed = ses.trialdata['smooth_hitrate'].copy(),ses.trialdata['smooth_farate'].copy()
        HR_maxed[HR_maxed>0.9999] = 0.9999
        FR_maxed[FR_maxed>0.9999] = 0.9999
        
        #Compute dprime and criterion:
        ses.trialdata['smooth_dprime']      = [st.norm.ppf(HR_maxed[t]) - st.norm.ppf(FR_maxed[t]) 
                for t in range(len(ses.trialdata))]
        ses.trialdata['smooth_criterion']   =  -0.5 * np.array([st.norm.ppf(HR_maxed[t]) + st.norm.ppf(FR_maxed[t]) 
                        for t in range(len(ses.trialdata))])

    return sessions


# Psychometric function (cumulative Gaussian)
def psychometric_function(x, mu, sigma, lapse_rate, guess_rate):
    """
    Parameters:
    - mu: mean or threshold
    - sigma: standard deviation or slope
    - lapse_rate: rate of lapses or false positives/negatives
    - guess_rate: rate of guessing
    Wichmann & Hill, 2001
    """
    # return guess_rate + (1 - guess_rate - lapse_rate) * 0.5 * (1 + np.erf((x - mu) / (np.sqrt(2) * sigma)))
    return guess_rate + (1 - guess_rate - lapse_rate) * 0.5 * (1 + special.erf((x - mu) / (np.sqrt(2) * sigma)))


def plot_psycurve(sessions,filter_engaged=False):

    for ises,ses in enumerate(sessions):
        trialdata = ses.trialdata.copy()
        if filter_engaged:
            trialdata = trialdata[trialdata['engaged']==1]

        psydata = trialdata.groupby(['signal'])['lickResponse'].sum() / trialdata.groupby(['signal'])['lickResponse'].count()

        x = psydata.keys().to_numpy()
        y = psydata.to_numpy()

        # # Plot the results
        fig, ax = plt.subplots()

        ax.scatter(x,y) 

        X = trialdata['signal']
        Y = trialdata['lickResponse']
        initial_guess           = [20, 15, 0.1, 0.1]  # Initial guess for parameters (mu,sigma,lapse_rate,guess_rate)
        bounds                  = ([0,4,0,0],[100,40,0.5,0.5])
        params, covariance      = curve_fit(psychometric_function, x, y, p0=initial_guess,bounds=bounds)
        params, covariance      = curve_fit(psychometric_function, X, Y, p0=initial_guess,bounds=bounds)

        # Plot the results
        ax.scatter(x, y, label='data',c='k')
        x_highres = np.linspace(np.min(x),np.max(x),1000)
        ax.plot(x_highres, psychometric_function(x_highres, *params), label='fit', color='blue')
        ax.set_xlabel('Stimulus Intensity')
        ax.set_ylabel('Probability of Response')
        ax.legend()
        ax.set_xlim([np.min(x),np.max(x)])
        ax.set_ylim([0,1])
        
        # Print the fitted parameters
        print("Fitted Parameters:")
        print("mu:", '%2.2f' % params[0])
        print("sigma:", '%2.2f' % params[1])
        print("lapse_rate:", '%2.2f' % params[2])
        print("guess_rate:", '%2.2f' % params[3])
 
    return fig


def fit_psycurve(ses):
    # Fit the psychometric curve to the data using curve_fit
    initial_guess = [0, 1, 0.1, 0.01]  # Initial guess for parameters
    params, covariance = curve_fit(psychometric_function, x_data, y_data_noise, p0=initial_guess)

    return

def runPSTH(ses,s_pre = -80, s_post = 60, binsize = 5):

    ## Parameters for spatial binning
    # s_pre       #pre cm
    # s_post      #post cm
    # binsize     #spatial binning in cm
    binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(s_pre,s_post+binsize,binsize)

    # trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

    trialdata   = ses.trialdata
    # runPSTH     = np.empty((len(ses.trialdata),len(bincenters)))

    # for ises,ses in enumerate(sessions):
    ntrials     = len(ses.trialdata)
    runPSTH     = np.empty(shape=(ntrials, len(bincenters)))

    for itrial in range(ntrials):
        idx = ses.behaviordata['trialNumber']==itrial+1
        runPSTH[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][0],
                                            ses.behaviordata['runSpeed'][idx], statistic='mean', bins=binedges)[0]
    # runPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = runPSTH_ses

    return runPSTH, bincenters


def lickPSTH(ses,s_pre = -80, s_post = 60, binsize = 5):

    ## Parameters for spatial binning
    # s_pre       #pre cm
    # s_post      #post cm
    # binsize     #spatial binning in cm

    binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(s_pre,s_post+binsize,binsize)
    # trialdata   = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)
    trialdata   = ses.trialdata

    # lickPSTH     = np.empty((len(ses.trialdata),len(bincenters)))
                    
    # for ises,ses in enumerate(sessions):
    ntrials     = len(ses.trialdata)
    lickPSTH    = np.empty(shape=(ntrials, len(bincenters)))

    for itrial in range(ntrials-1):
        idx = ses.behaviordata['trialNumber']==itrial+1
        lickPSTH[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][0],
                                            ses.behaviordata['lick'][idx], statistic='sum', bins=binedges)[0]
    # lickPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = lickPSTH_ses
    lickPSTH /= binsize 

    return lickPSTH, bincenters

def plot_lick_corridor_outcome(trialdata,lickPSTH,bincenters):
    ### Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()

    ttypes = pd.unique(trialdata['trialOutcome'])
    # ttypes = ['CR', 'MISS', 'HIT','FA']
    colors = get_clr_outcome(ttypes)

    for i,ttype in enumerate(ttypes):
        idx = trialdata['trialOutcome']==ttype
        data_mean = np.nanmean(lickPSTH[idx,:],axis=0)
        data_error = np.nanstd(lickPSTH[idx,:],axis=0) / math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype,color=colors[i])
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])

    ax.legend()
    ax.set_ylim(0,2)
    ax.set_xlim(bincenters[0],bincenters[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Lick Rate (licks/cm)')
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

    return fig

def plot_lick_corridor_psy(trialdata,lickPSTH,bincenters):
    # Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()
    ttypes = np.sort(pd.unique(trialdata['signal']))
    colors = get_clr_psy(ttypes)
    for i,ttype in enumerate(ttypes):
        # idx = np.logical_and(trialdata['signal']==ttype,trialdata['lickResponse']==1)
        idx = trialdata['signal']==ttype
        data_mean = np.nanmean(lickPSTH[idx,:],axis=0)
        data_error = np.nanstd(lickPSTH[idx,:],axis=0) / math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype,color=colors[i])
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])

    ax.legend()
    ax.set_ylim(0,2)
    ax.set_xlim(bincenters[0],bincenters[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Lick Rate (licks/cm)')
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

    return fig

def plot_run_corridor_psy(trialdata,runPSTH,bincenters):
    ### Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()
    ttypes = np.sort(pd.unique(trialdata['signal']))
    colors = get_clr_psy(ttypes)

    for i,ttype in enumerate(ttypes):
        # idx = np.logical_and(trialdata['signal']==ttype,trialdata['lickResponse']==1)
        idx = trialdata['signal']==ttype
        data_mean = np.nanmean(runPSTH[idx,:],axis=0)
        data_error = np.nanstd(runPSTH[idx,:],axis=0) #/ math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype,color=colors[i])
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])

    ax.legend()
    ax.set_ylim(0,50)
    ax.set_xlim(bincenters[0],bincenters[-1])
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

    return fig

def plot_run_corridor_outcome(trialdata,runPSTH,bincenters):
    ### Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()
    
    ttypes = pd.unique(trialdata['trialOutcome'])
    colors = get_clr_outcome(ttypes)

    for i,ttype in enumerate(ttypes):
        idx = trialdata['trialOutcome']==ttype
        data_mean = np.nanmean(runPSTH[idx,:],axis=0)
        data_error = np.nanstd(runPSTH[idx,:],axis=0)# / math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype,color=colors[i])
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])

    ax.legend()
    ax.set_ylim(0,50)
    ax.set_xlim(bincenters[0],bincenters[-1])
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

    return fig


# Alternative psychometric curve function: 
# d = np.array([75, 80, 90, 95, 100, 105, 110, 115, 120, 125], dtype=float)
# p2 = np.array([6, 13, 25, 29, 29, 29, 30, 29, 30, 30], dtype=float) / 30. # scale to 0..1

# # psychometric function
# def pf(x, alpha, beta):
#     return 1. / (1 + np.exp( -(x-alpha)/beta ))

# # fitting
# par0 = sy.array([100., 1.]) # use some good starting values, reasonable default is [0., 1.]
# par, mcov = curve_fit(pf, d, p2, par0)
# print(par)
# plt.plot(d, p2, 'ro')
# plt.plot(d, pf(d, par[0], par[1]))
# plt.show()