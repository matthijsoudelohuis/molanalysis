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
    hit_rate            = sum((signal == 1) & (response == 1)) / sum(signal == 1)
    falsealarm_rate     = sum((signal == 0) & (response == 1)) / sum(signal == 0)
    
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
       
        params = fit_psycurve(trialdata,printoutput=True)

        ## Plot the results
        fig, ax = plt.subplots()
        ax.scatter(x, y, label='data',c='k')
        x_highres = np.linspace(np.min(x),np.max(x),1000)
        ax.plot(x_highres, psychometric_function(x_highres, *params), label='fit', color='blue')
        ax.set_xlabel('Stimulus Intensity')
        ax.set_ylabel('Probability of Response')
        ax.legend()
        ax.set_xlim([np.min(x),np.max(x)])
        ax.set_ylim([0,1])
        ax.set_title(ses.sessiondata['session_id'][0])
 
    return fig


def fit_psycurve(trialdata,printoutput=False):

    psydata = trialdata.groupby(['signal'])['lickResponse'].sum() / trialdata.groupby(['signal'])['lickResponse'].count()
    x = psydata.keys().to_numpy()
    y = psydata.to_numpy()

    X = trialdata['signal'] #Fit with actual trials, not averages per condition
    Y = trialdata['lickResponse']
    initial_guess           = [20, 15, 1-y[-1], y[0]]  # Initial guess for parameters (mu,sigma,lapse_rate,guess_rate)
    # set guess rate and lapse rate to be within 10% of actual response rates at catch and max trials:
    bounds                  = ([0,2,(1-y[-1])*0.9,y[0]*0.9-0.01],[100,40,(1-y[-1])*1.1+0.01,y[0]*1.1])
    # bounds                  = ([0,4,0,0],[100,40,0.5,0.5])
    
    # Fit the psychometric curve to the data using curve_fit
    # params, covariance      = curve_fit(psychometric_function, x, y, p0=initial_guess,bounds=bounds)
    params, covariance      = curve_fit(psychometric_function, X, Y, p0=initial_guess,bounds=bounds)
    
    if printoutput: 
        # Print the fitted parameters
        print("Fitted Parameters:")
        print("mu:", '%2.2f' % params[0])
        print("sigma:", '%2.2f' % params[1])
        print("lapse_rate:", '%2.2f' % params[2])
        print("guess_rate:", '%2.2f' % params[3])
    
    return params


def noise_to_psy(sessions,filter_engaged=True):

    for ises,ses in enumerate(sessions):
        trialdata = ses.trialdata.copy()
        if filter_engaged:
            trialdata = trialdata[trialdata['engaged']==1]

        params = fit_psycurve(trialdata,printoutput=False)

        idx = ses.trialdata['stimcat']=='N'
        ses.trialdata['signal_psy'] = pd.Series(dtype='float')

        ses.trialdata.loc[idx,'signal_psy'] = (ses.trialdata.loc[idx,'signal'] - params[0]) / params[1]

    return sessions


def calc_runPSTH(ses,s_pre = -80, s_post = 60, binsize = 5):

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
        idx = np.logical_and(itrial-1 <= ses.behaviordata['trialNumber'], ses.behaviordata['trialNumber'] <= itrial+2)
        runPSTH[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][itrial],
                                            ses.behaviordata['runspeed'][idx], statistic='mean', bins=binedges)[0]
        
        # idx = ses.behaviordata['trialNumber']==itrial+1
        # runPSTH[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][itrial],
        #                                     ses.behaviordata['runspeed'][idx], statistic='mean', bins=binedges)[0]
    # runPSTH[trialdata['session_id']==ses.sessiondata['session_id'][0],:] = runPSTH_ses

    return runPSTH, bincenters


def calc_lickPSTH(ses,s_pre = -80, s_post = 60, binsize = 5):

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
        idx = np.logical_and(itrial-1 <= ses.behaviordata['trialNumber'], ses.behaviordata['trialNumber'] <= itrial+2)
        lickPSTH[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][itrial],
                                            ses.behaviordata['lick'][idx], statistic='sum', bins=binedges)[0]
        
        # lickPSTH[itrial,:] = binned_statistic(ses.behaviordata['zpos'][idx]-ses.trialdata['stimStart'][itrial],
                                            # ses.behaviordata['lick'][idx], statistic='sum', bins=binedges)[0]
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
        ax.plot(bincenters,data_mean,label=ttype,color=colors[i],linewidth=2)
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])

    rewzonestart = np.mean(trialdata['rewardZoneStart'] - trialdata['stimStart'])
    rewzonelength = np.mean(trialdata['rewardZoneEnd'] - trialdata['rewardZoneStart'])

    ax.legend()
    ax.set_ylim(0,1.5)
    ax.set_xlim(bincenters[0],bincenters[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Lick Rate (licks/cm)')
    # ax.fill_between([0,30], [0,50], [0,50],alpha=0.5)
    ax.add_patch(matplotlib.patches.Rectangle((0,0),20,2, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((rewzonestart,0),rewzonelength,3, 
                            fill = True, alpha=0.2,
                            color = "grey",
                            linewidth = 0))

    plt.text(5, 1.4, 'Stim',fontsize=11)
    plt.text(27, 1.4, 'Reward',fontsize=11)
    plt.tight_layout()

    return fig

def plot_lick_corridor_psy(trialdata,lickPSTH,bincenters,version='signal',hitonly=False):
    # Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()
    if version=='signal':
        ttypes = np.sort(pd.unique(trialdata['signal']))
        colors = get_clr_psy(ttypes)
        for i,ttype in enumerate(ttypes):
            idx = trialdata[version]==ttype
            
            if hitonly:
                idx = np.logical_and(idx,trialdata['lickResponse']==1)

            data_mean = np.nanmean(lickPSTH[idx,:],axis=0)
            data_error = np.nanstd(lickPSTH[idx,:],axis=0) #/ math.sqrt(sum(idx))
            ax.plot(bincenters,data_mean,label=ttype,color=colors[i])
            ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])

    elif version=='signal_psy':
        resolution=0.4
        edges = np.hstack((-10,np.arange(start=-2-resolution/2,stop=2+resolution/2,step=resolution),10))
        colors = get_clr_psy(edges[:-1])

        for i,lims in enumerate(zip(edges[:-1],edges[1:])):
            idx = np.logical_and(trialdata[version]>lims[0],trialdata[version]<lims[1])

            if hitonly:
                idx = np.logical_and(idx,trialdata['lickResponse']==1)

            data_mean = np.nanmean(lickPSTH[idx,:],axis=0)
            data_error = np.nanstd(lickPSTH[idx,:],axis=0) #/ math.sqrt(sum(idx))
            ax.plot(bincenters,data_mean,label=np.mean(lims).round(1),color=colors[i])
            ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])


    rewzonestart = np.mean(trialdata['rewardZoneStart'] - trialdata['stimStart'])
    rewzonelength = np.mean(trialdata['rewardZoneEnd'] - trialdata['rewardZoneStart'])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),title=version)
    
    ax.set_ylim(0,1.75)
    ax.set_xlim(bincenters[0],bincenters[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Lick Rate (licks/cm)')
    # ax.fill_between([0,30], [0,50], [0,50],alpha=0.5)
    ax.add_patch(matplotlib.patches.Rectangle((0,0),20,2, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((rewzonestart,0),rewzonelength,3, 
                            fill = True, alpha=0.2,
                            color = "grey",
                            linewidth = 0))


    plt.text(5, 1.6, 'Stim',fontsize=11)
    plt.text(27, 1.6, 'Reward',fontsize=11)
    plt.tight_layout()
    return fig

def plot_run_corridor_psy(trialdata,runPSTH,bincenters,version='signal',hitonly=False):
    ### Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()
    if version=='signal':
        ttypes = np.sort(pd.unique(trialdata['signal']))
        colors = get_clr_psy(ttypes)
        for i,ttype in enumerate(ttypes):
            idx = trialdata[version]==ttype
            
            if hitonly:
                idx = np.logical_and(idx,trialdata['lickResponse']==1)

            data_mean = np.nanmean(runPSTH[idx,:],axis=0)
            data_error = np.nanstd(runPSTH[idx,:],axis=0) #/ math.sqrt(sum(idx))
            ax.plot(bincenters,data_mean,label=ttype,color=colors[i])
            ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])

    elif version=='signal_psy':
        resolution=0.4
        edges = np.hstack((-10,np.arange(start=-2-resolution/2,stop=2+resolution/2,step=resolution),10))
        colors = get_clr_psy(edges[:-1])

        for i,lims in enumerate(zip(edges[:-1],edges[1:])):
            idx = np.logical_and(trialdata[version]>lims[0],trialdata[version]<lims[1])

            if hitonly:
                idx = np.logical_and(idx,trialdata['lickResponse']==1)

            data_mean = np.nanmean(runPSTH[idx,:],axis=0)
            data_error = np.nanstd(runPSTH[idx,:],axis=0) #/ math.sqrt(sum(idx))
            ax.plot(bincenters,data_mean,label=np.mean(lims).round(1),color=colors[i])
            ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.3, linewidth=0,color=colors[i])

    rewzonestart = np.mean(trialdata['rewardZoneStart'] - trialdata['stimStart'])
    rewzonelength = np.mean(trialdata['rewardZoneEnd'] - trialdata['rewardZoneStart'])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),title=version)

    ax.set_ylim(0,50)
    ax.set_xlim(bincenters[0],bincenters[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Running speed (cm/s)')
    ax.add_patch(matplotlib.patches.Rectangle((0,0),20,50, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((rewzonestart,0),rewzonelength,50, 
                            fill = True, alpha=0.2,
                            color = "grey",
                            linewidth = 0))

    plt.text(5, 45, 'Stim',fontsize=11)
    plt.text(27, 45, 'Reward',fontsize=11)
    plt.tight_layout()
    
    return fig

def plot_run_corridor_outcome(trialdata,runPSTH,bincenters):
    ### Plot licking rate as a function of trial type:

    fig, ax = plt.subplots()
    
    ttypes = pd.unique(trialdata['trialOutcome'])
    colors = get_clr_outcome(ttypes)

    for i,ttype in enumerate(ttypes):
        idx = trialdata['trialOutcome']==ttype
        data_mean = np.nanmean(runPSTH[idx,:],axis=0)
        data_error = np.nanstd(runPSTH[idx,:],axis=0) #/ math.sqrt(sum(idx))
        ax.plot(bincenters,data_mean,label=ttype,color=colors[i],linewidth=2)
        ax.fill_between(bincenters, data_mean+data_error,  data_mean-data_error, alpha=.2, linewidth=0,color=colors[i])

    rewzonestart = np.mean(trialdata['rewardZoneStart'] - trialdata['stimStart'])
    rewzonelength = np.mean(trialdata['rewardZoneEnd'] - trialdata['rewardZoneStart'])

    ax.legend()
    ax.set_ylim(0,50)
    ax.set_xlim(bincenters[0],bincenters[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Running speed (cm/s)')
    ax.add_patch(matplotlib.patches.Rectangle((0,0),20,50, 
                            fill = True, alpha=0.2,
                            color = "blue",
                            linewidth = 0))
    ax.add_patch(matplotlib.patches.Rectangle((rewzonestart,0),rewzonelength,50, 
                            fill = True, alpha=0.2,
                            color = "grey",
                            linewidth = 0))

    plt.text(5, 45, 'Stim',fontsize=11)
    plt.text(27, 45, 'Reward',fontsize=11)
    plt.tight_layout()

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