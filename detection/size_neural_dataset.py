# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Import packages
import math
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis\\')
from scipy.signal import medfilt

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.psth import compute_tensor_space,compute_respmat_space
from utils.plot_lib import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\OverallDataset\\')

#%% Load behavior of all protocols:
protocol                = ['DN']
sessions,nsessions      = filter_sessions(protocol,min_cells=100)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
nanimals                = len(np.unique(sessiondata['animal_id']))

#%% Remove sessions LPE10884 that are too bad:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE10884_2023_12_14','LPE10884_2023_12_15','LPE10884_2024_01_11','LPE10884_2024_01_16']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nsessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% 
nunique_animals = len(np.unique(sessiondata['animal_id']))
print(f"{nunique_animals} unique animals in the dataset\n")

#%% Minimum and maximum number of different sessions per animal
min_nsess_per_animal = np.min(sessiondata.groupby('animal_id')['session_id'].nunique())
max_nsess_per_animal = np.max(sessiondata.groupby('animal_id')['session_id'].nunique())
print(f"{min_nsess_per_animal} - {max_nsess_per_animal} sessions per animal")

#%% Get number of engaged and disengaged trials
for ises,ses in enumerate(sessions): 
    ses.sessiondata['ntrials_eng'] = np.sum(ses.trialdata['engaged']==1)
    ses.sessiondata['ntrials_dis'] = np.sum(ses.trialdata['engaged']==0)
    ses.sessiondata['ntrials_eng_noise'] = np.sum(np.logical_and(ses.trialdata['engaged']==1,ses.trialdata['stimcat']=='N'))
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Report on number of trials per session:
print(f"{len(sessiondata[sessiondata['protocol']=='DN'])} DN sessions with recordings \n")
print('%3.1f +- %3.1f trials per session' % (np.mean(sessiondata['ntrials']),np.std(sessiondata['ntrials'])))
print('%3.1f +- %3.1f engaged trials per session' % (np.mean(sessiondata['ntrials_eng']),np.std(sessiondata['ntrials_eng'])))
print('%3.1f +- %3.1f disengaged trials per session' % (np.mean(sessiondata['ntrials_dis']),np.std(sessiondata['ntrials_dis'])))
print('%3.1f +- %3.1f engaged noise trials per session' % (np.mean(sessiondata['ntrials_eng_noise']),np.std(sessiondata['ntrials_eng_noise'])))

#%% Report on number of cells per session:
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
cell_counts = celldata.groupby(['session_id']).count().reset_index()
print('%3d cells total' % len(celldata))
print('%3d labeled cells total' % np.sum(celldata['redcell']))
print('%3.1f +- %3.1f cells per session' % (np.mean(cell_counts['iscell']),np.std(cell_counts['iscell'])))

#%% Show the number of cells per layer across sessions:
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

areas = ['V1','PM','AL','RSP']
clr_areas = get_clr_areas(areas)

# Group celldata by session_id and layer, and count the number of cells
cell_counts = celldata.groupby(['session_id', 'roi_name']).count().reset_index()

fig = plt.figure(figsize=(3,3))
# sns.barplot(data=cell_counts,x='layer',y='iscell',hue='layer',palette='tab10')
sns.stripplot(data=cell_counts,x='roi_name',y='iscell',hue='roi_name',palette=clr_areas,
              order=areas,legend=None,hue_order=areas)
# sns.barplot(data=cell_counts,x='roi_name',y='iscell',hue='roi_name',palette=clr_areas,order=areas)
plt.xlabel('Area')
plt.ylabel('#Cells')
plt.title('#Cells per area')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'CellCountsPerArea_%dsessions_' %nsessions + '.png'), format = 'png')

