# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic,binned_statistic_2d
from tqdm import tqdm
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
from utils.plot_lib import shaded_error
from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *
from utils.rf_lib import smooth_rf, filter_nearlabeled

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE09830','2023_04_10']])
# session_list        = np.array([['LPE09830','2023_04_10'],
#                                 ['LPE09830','2023_04_12']])
# session_list        = np.array([['LPE11086','2024_01_05']])
# session_list        = np.array([['LPE09830','2023_04_10'],
#                                 ['LPE09830','2023_04_12'],
#                                 ['LPE11086','2024_01_05'],
#                                 ['LPE10884','2023_10_20'],
#                                 ['LPE10885','2023_10_19'],
#                                 ['LPE10885','2023_10_20'],
#                                 ['LPE10919','2023_11_06']])

#%% Load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM'],session_rf=True)

#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF')

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='OSI')
    sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='gOSI')
    sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='tuning_var')
    sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'])

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% ########################### Compute noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#TODO: make noise corr and pairwise functions attributes of session classes

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_metrics(sessions)

#%% How are noise correlations distributed

# Compute the average noise correlation for each neuron and store in cell data, loop over sessions:
for ises in range(nSessions):
    sessions[ises].celldata['noise_corr_avg'] = np.nanmean(sessions[ises].noise_corr,axis=1) 
    # sessions[ises].celldata['noise_corr_avg'] = np.nanmean(np.abs(sessions[ises].noise_corr),axis=1) 

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%%  Scatter plot of average noise correlations versus skew:
def plot_corr_NC_var(sessions,vartoplot):
    cdata = []
    plt.figure(figsize=(4,4))
    for ses in sessions:
        plt.scatter(ses.celldata[vartoplot],ses.celldata['noise_corr_avg'],s=6,marker='.',alpha=0.5)
        cdata.append(np.corrcoef(ses.celldata[vartoplot],ses.celldata['noise_corr_avg'])[0,1])
    plt.xlabel(vartoplot)
    plt.ylabel('Avg. NC')
    plt.text(x=np.percentile(ses.celldata[vartoplot],25),y=0.12,s='Mean correlation: %1.3f +- %1.3f' % (np.mean(cdata),np.std(cdata)))
    plt.savefig(os.path.join(savedir,'NoiseCorrelations','%s_vs_NC' % vartoplot + '.png'), format = 'png')

#%%  Scatter plot of average noise correlations versus skew:
plot_corr_NC_var(sessions,vartoplot = 'skew')

#%%  Scatter plot of average noise correlations versus depth:
plot_corr_NC_var(sessions,vartoplot = 'depth')

#%%  Scatter plot of average noise correlations versus tuning variance:
plot_corr_NC_var(sessions,vartoplot = 'tuning_var')

#%%  Scatter plot of average noise correlations versus noise level:
plot_corr_NC_var(sessions,vartoplot = 'noise_level')

#%%  Scatter plot of average noise correlations versus event rate level:
plot_corr_NC_var(sessions,vartoplot = 'event_rate')

#%%  Scatter plot of average noise correlations versus fluorescence channel 2:
plot_corr_NC_var(sessions,vartoplot = 'meanF_chan2')
