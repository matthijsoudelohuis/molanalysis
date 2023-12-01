# -*- coding: utf-8 -*-
"""
Set of function used for analysis of mouse behavior in visual navigation task
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025
"""

import scipy.stats as st
import math
import pandas as pd
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


def compute_dprime(signal,response):
    
    ntrials             = len(signal)
    hit_rate            = sum((signal == 1) & (response == True)) / ntrials
    falsealarm_rate     = sum((signal == 0) & (response == True)) / ntrials
    
    dprime              = st.norm.ppf(hit_rate) - st.norm.ppf(falsealarm_rate)
    criterion           = -0.5 * (st.norm.ppf(hit_rate) + st.norm.ppf(falsealarm_rate))
    return dprime,criterion


def smooth_rate_dprime(sessions,sigma=25):
    #### Smooth hit and fa rate and smooth dprime:

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