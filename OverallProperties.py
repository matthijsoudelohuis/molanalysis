
#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.signal import medfilt
# from scipy.stats import binned_statistic,binned_statistic_2d

# from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
# from utils.tuning import compute_tuning, compute_prefori
# from utils.plotting_style import * #get all the fixed color schemes
# from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
# from utils.plot_lib import shaded_error
# from utils.RRRlib import regress_out_behavior_modulation
# from utils.corr_lib import *
# from utils.rf_lib import smooth_rf

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\DescriptiveStatisticsSessions\\')

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['GR'])

sessions,nSessions   = filter_sessions(protocols = ['IM'])

sessions,nSessions   = filter_sessions(protocols = ['GN'])

#%% Show number of trials per session 
sessions,nSessions   = filter_sessions(protocols = ['GR','IM','GN'])
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

sesdata = pd.DataFrame()
sesdata['ntrials']         = sessiondata.groupby(["protocol"])['ntrials']

sns.barplot(sessiondata,x='protocol',y='ntrials',hue='protocol')
sns.scatterplot(sessiondata,x='protocol',y='ntrials',hue='protocol')
plt.ylim([0,5800])
plt.savefig(os.path.join(savedir,'nTrials_perSession.png'), format = 'png')

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['GR'])

#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')

#%% Normalize the data to the maximum across trials:
def avg_norm_smooth(data,normalize=True):
    # Assuming data is the N by K array
    # Step 1: Normalize each row by its maximum value
    if data.ndim == 1:
        data = data[np.newaxis,:]
    
    if normalize:
        normalized_respmat = data / np.max(data, axis=1, keepdims=True)
    else: 
        normalized_respmat = data

    # Step 2: Average across neurons (rows)
    average_response = np.mean(normalized_respmat, axis=0)
    
    # Step 3: smooth the data a bit
    nsmoothing = 100
    boxcar = np.ones(nsmoothing) / nsmoothing
    pad_width = len(boxcar) // 2
    padded_response = np.pad(average_response, pad_width=pad_width, mode='edge')
    convolved_response = np.convolve(padded_response, boxcar, mode='valid')
    return convolved_response

#%% store avg response across neurons across trials
avg_resp = np.empty((nSessions, 3201))
avg_resp[:] = np.nan

for ises in range(nSessions):    # iterate over sessions
    convolved_response = avg_norm_smooth(sessions[ises].respmat)
    avg_resp[ises,:len(convolved_response)] = convolved_response #store the convolved response
    
#%% Plot the mean neural response to the gratings across sessions
plt.subplots(figsize=(5,5))
for ises in range(nSessions):    # iterate over sessions
    plt.plot(avg_resp[ises,:],linewidth=1,
             label=sessions[ises].sessiondata['session_id'][0])
plt.plot(np.nanmean(avg_resp,axis=0),linewidth=2,c='k' ,
            label='mean'),
plt.xlim([-50,3200])
plt.ylim([0.02,0.15])
plt.xlabel('Trials')
plt.ylabel('Normalized deconv. response')
plt.legend(frameon=False,fontsize=6,loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'AveragedNormalizedResponses.png'), format = 'png')

#%% Normalize the data to the maximum across trials:
avg_resp = np.empty((nSessions, 3201))
avg_resp[:] = np.nan

for ises in range(nSessions):    # iterate over sessions
    convolved_response = avg_norm_smooth(sessions[ises].respmat_videome)
    avg_resp[ises,:len(convolved_response)] = convolved_response #store the convolved response
    
#%% Plot the mean neural response to the gratings across sessions
plt.subplots(figsize=(5,5))
for ises in range(nSessions):    # iterate over sessions
    plt.plot(avg_resp[ises,:],linewidth=1,
             label=sessions[ises].sessiondata['session_id'][0])
plt.plot(np.nanmean(avg_resp,axis=0),linewidth=2,c='k' ,
            label='mean'),
plt.xlim([-50,3200])
# plt.ylim([0.02,0.15])
plt.xlabel('Trials')
plt.ylabel('Normalized video motion energy')
plt.legend(frameon=False,fontsize=6,loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'AveragedNormalizedResponse_videoME.png'), format = 'png')

#%% Normalize the data to the maximum across trials:
avg_resp = np.empty((nSessions, 3201))
avg_resp[:] = np.nan

for ises in range(nSessions):    # iterate over sessions
    convolved_response = avg_norm_smooth(sessions[ises].respmat_runspeed,normalize=False)
    avg_resp[ises,:len(convolved_response)] = convolved_response #store the convolved response
    
#%% Plot the mean neural response to the gratings across sessions
plt.subplots(figsize=(5,5))
for ises in range(nSessions):    # iterate over sessions
    plt.plot(avg_resp[ises,:],linewidth=1,
             label=sessions[ises].sessiondata['session_id'][0])
plt.plot(np.nanmean(avg_resp,axis=0),linewidth=2,c='k' ,
            label='mean'),
plt.xlim([-50,3200])
# plt.ylim([0.02,0.15])
plt.xlabel('Trials')
plt.ylabel('Running speed')
plt.legend(frameon=False,fontsize=6,loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Averaged_runspeed.png'), format = 'png')

# %%
