# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:47:23 2023

@author: USER
"""

import scipy.stats as st

def compute_dprime(signal,response):
    
    ntrials             = len(signal)
    hit_rate            = sum((signal == 1) & (response == True)) / ntrials
    falsealarm_rate     = sum((signal == 0) & (response == True)) / ntrials
    
    dprime             = st.norm.ppf(hit_rate) - st.norm.ppf(falsealarm_rate)
    return dprime