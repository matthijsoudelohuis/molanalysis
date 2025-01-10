# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
os.chdir('e:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import copy

# from loaddata import * #get all the loading data functions (filter_sessions,load_sessions)
from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive

from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import * # get support functions for plotting
plt.rcParams['svg.fonttype'] = 'none'

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\')

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'
# calciumversion      = 'dF'

session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE12013', '2024_04_25']])
session_list = np.array([['LPE11622', '2024_02_22']])
# session_list = np.array([['LPE11997', '2024_04_16']])
session_list = np.array([['LPE11998', '2024_04_30']])
# session_list = np.array([['LPE10884', '2023_12_15']])
# session_list = np.array([['LPE10884', '2024_01_16']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE11998'],min_cells=100,
#                            load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE10884'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

#%% 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = 0  #pre cm
s_post      = 200   #post cm
binsize     = 10     #spatial binning in cm

z_T = sessions[i].trialdata['stimStart']
z_T = np.arange(start=0,stop=np.nanmax(sessions[i].zpos_F),step=200)

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,z_T,
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')


#%% ################################## Plot for one session: ####################################
ises        = 0
# areas       = sessions[ises].celldata['roi_name'].unique()
areas       = ['V1','PM','AL','RSP']

fig, axes = plt.subplots(nrows=1,ncols=4,figsize=(12,3))
for iarea,area in enumerate(areas):
    ax          = axes[iarea]

    idx_N       = sessions[ises].celldata['roi_name'] == area

    data        = copy.deepcopy(sessions[ises].stensor[idx_N,:,:])

    [N,K,S]     = np.shape(data)
    idx_T_sort  = np.random.choice(K,int(K/2),replace=False)
    idx_T_not_sort = np.setdiff1d(np.arange(K),idx_T_sort)

    sortdata    = np.nanmean(data[:,idx_T_sort,:],axis=1)

    # sortidx     = np.argsort(-np.nanargmax(np.nanmean(data[],axis=2),axis=1))
    sortidx     = np.argsort(-np.nanargmax(sortdata,axis=1))
    
    # plotdata    = np.nanmean(sessions[ises].stensor[:,idx_T_sort,:],axis=1)
    plotdata    = np.nanmean(data[:,idx_T_not_sort,:],axis=1)

    plotdata        = plotdata[sortidx,:]
    Narea       = np.shape(data)[0]
    X, Y        = np.meshgrid(sbins, range(Narea)) #Construct X Y positions of the heatmaps:

    c = ax.pcolormesh(X,Y,plotdata, cmap = 'bwr',
                        vmin=-np.percentile(plotdata,99),vmax=np.percentile(plotdata,99))
    # c = plt.pcolormesh(X,Y,data[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.25)
    # c = plt.pcolormesh(X,Y,data[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.5)
    ax.set_title(area,fontsize=11)

    ax.set_yticks([0,N])
    ax.set_xlabel('Background position (cm)',fontsize=9)
    # ax.set_xlim([-80,60])
    ax.set_ylim([0,Narea])
    if iarea==0:
        ax.set_ylabel('nNeurons',fontsize=10)
    
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.3])
fig.colorbar(c, cax=cbar_ax,label='Activity (z)')
fig.tight_layout()

plt.savefig(os.path.join(savedir,'BackgroundActivity_crosssorted_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
