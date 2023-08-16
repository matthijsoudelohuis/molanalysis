# -*- coding: utf-8 -*-
"""
Set of function used for analysis of mouse behavior in visual navigation task
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025
"""

import scipy.stats as st

def compute_dprime(signal,response):
    
    ntrials             = len(signal)
    hit_rate            = sum((signal == 1) & (response == True)) / ntrials
    falsealarm_rate     = sum((signal == 0) & (response == True)) / ntrials
    
    dprime              = st.norm.ppf(hit_rate) - st.norm.ppf(falsealarm_rate)
    return dprime