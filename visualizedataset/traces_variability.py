# -*- coding: utf-8 -*-
"""
This script analyzes responses to visual gratings in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('c:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import *
from utils.psth import compute_respmat
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import plot_excerpt,plot_PCA_gratings,plot_tuned_response

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

#%% Load an example session: 
session_list        = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion=calciumversion,keepraw=False)
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,keepraw=False)

t_axis = sessions[0].t_axis

#%% compute tuning metrics:
idx_resp = (t_axis>=0.5) & (t_axis<=1.5)
sessions[0].respmat = np.nanmean(sessions[0].tensor[:,:,idx_resp],axis=2)
sessions = compute_tuning_wrapper(sessions)

#%% Concatenate celldata across sessions:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

n_example_cells = 10
example_cells = np.random.choice(np.where(
    sessions[0].celldata['tuning_var'] > np.percentile(sessions[0].celldata['tuning_var'],90))[0],n_example_cells)
    # sessions[0].celldata['gOSI'] > np.percentile(sessions[0].celldata['gOSI'],90))[0],n_example_cells)

# %%
#%% Show some tuned responses with calcium and deconvolved traces across orientations:
fig = plot_tuned_response(sessions[0].tensor,sessions[0].trialdata,t_axis,example_cells,plot_n_trials=10)
fig.suptitle('%s - dF/F' % sessions[0].session_id,fontsize=12)
# save the figure
fig.savefig(os.path.join(savedir,'TunedResponse_dF_%s.png' % sessions[0].session_id))


#%% Load an example session: 
sessions,nSessions   = filter_sessions(protocols = 'SP',min_cells=2000)

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='deconv',keepraw=True)
    sessions[ises].load_data(load_calciumdata=True,calciumversion='deconv',load_behaviordata=True,load_videodata=True)

#%% #####################################

for ises in range(nSessions):
    #Show some traces to see responses:
    example_cells   = np.arange(len(sessions[ises].celldata)) #all cells
    sessions[ises].celldata['redcell'] = 0
    fig = plot_excerpt(sessions[ises],trialsel=(10,20),neural_version='raster',neuronsel=example_cells)
    my_savefig(fig,savedir,'Excerpt_SP_%s' % (sessions[ises].session_id))

#%% 
sessions[1].celldata['redcell'] = 0
example_cells   = np.arange(len(sessions[1].celldata)) #all cells
fig = plot_excerpt(sessions[1],trialsel=(10,20),neural_version='raster',neuronsel=example_cells)
my_savefig(fig,savedir,'AffineModel_R2_MultAddSep_PredictorsOverall_%dsessions' % (nSessions), formats = ['png'])

# %%
