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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special


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

def plot_psycurve(sessions):




    return

# Psychometric function (cumulative Gaussian)
def psychometric_function(x, mu, sigma, lapse_rate, guess_rate):
    """
    Parameters:
    - mu: mean or threshold
    - sigma: standard deviation or slope
    - lapse_rate: rate of lapses or false positives/negatives
    - guess_rate: rate of guessing
    """
    # return guess_rate + (1 - guess_rate - lapse_rate) * 0.5 * (1 + np.erf((x - mu) / (np.sqrt(2) * sigma)))
    return guess_rate + (1 - guess_rate - lapse_rate) * 0.5 * (1 + special.erf((x - mu) / (np.sqrt(2) * sigma)))



def fit_psycurve(ses):
    




params, covariance = curve_fit(psychometric_function, x_data, y_data_noise, p0=initial_guess)

# Plot the results
plt.scatter(x_data, y_data_noise, label='Noisy Data')
plt.plot(x_data, psychometric_function(x_data, *params), label='Fitted Curve', color='red')
plt.title('Fitted Psychometric Curve')
plt.xlabel('Stimulus Intensity')
plt.ylabel('Probability of Response')
plt.legend()
plt.show()

# Print the fitted parameters
print("Fitted Parameters:")
print("mu:", params[0])
print("sigma:", params[1])
print("lapse_rate:", params[2])
print("guess_rate:", params[3])
