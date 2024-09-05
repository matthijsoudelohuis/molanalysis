# -*- coding: utf-8 -*-
"""
This script analyzes responses to visual gratings in a multi-area calcium imaging
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

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import comp_grating_responsive
from utils.psth import compute_respmat
from utils.plotting_style import * #get all the fixed color schemes

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_23']])

#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                # ['LPE11998','2024_05_02'], #GN
                                # ['LPE12013','2024_05_02'], #GN
                                # ['LPE12013','2024_05_07'], #GN
                                ['LPE10919','2023_11_06']]) #GR

#%% Load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=['V1','PM'])
# sessions,nSessions   = filter_sessions(protocols = ['GR','GN'])


#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='deconv')
    #Alternative method, much faster:
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,t_base_start=-1,t_base_stop=0,method='mean',subtr_baseline=True)
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    # delattr(sessions[ises],'videodata')
    # delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

# %% Compute visually responsive or not: 

sessions = comp_grating_responsive(sessions,pthr = 0.0001)

#%% Concatenate celldata across sessions:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Show fraction of visually responsive for each session and area: 
fracdata = pd.DataFrame()
fracdata['vis_resp']         = celldata.groupby(["session_id","roi_name"])['vis_resp'].sum() / celldata.groupby(["session_id","roi_name"])['iscell'].count()

clrs_areas = get_clr_areas(['PM','V1'])
fig,ax = plt.subplots(1,1,figsize=(4,4))
# sns.barplot(data=fracdata,x='roi_name',y='vis_resp',palette=clrs_areas,ax=ax)
sns.stripplot(data=fracdata,x='roi_name',y='vis_resp',hue='roi_name',palette=clrs_areas,jitter=0.1,s=8,ax=ax)
ax.set_xlim([-0.25,1.25])
ax.set_ylim([0,1])
ax.set_ylabel('Fraction visually responsive')
ax.set_xlabel('Area')
ax.set_yticks([0,0.25,0.5,0.75,1])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Frac_Responsive_GR_GN_RankSum_%sSessions' % nSessions + '.png'), format = 'png')

# %%
