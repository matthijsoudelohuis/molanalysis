# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

# %% # Imports
# Import general libs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
os.chdir('e:\\Python\\molanalysis')

# os.chdir('../')  # set working directory to the root of the git repo

# Import personal lib funcs
from loaddata.session_info import load_sessions
from utils.plotting_style import *  # get all the fixed color schemes
from utils.imagelib import load_natural_images
from loaddata.get_data_folder import get_local_drive
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.rf_lib import estimate_rf_IM, exclude_outlier_rf, smooth_rf

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - RF\\')

# %% Load IM session with receptive field mapping ################################################
session_list = np.array([['LPE10885', '2023_10_20']])
# session_list = np.array([['LPE09665', '2023_03_15']])

# Load sessions lazy: (no calciumdata, behaviordata etc.,)
sessions, nSessions = load_sessions(protocol='IM', session_list=session_list)

# Load proper data and compute average trial responses:
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv', keepraw=False)

# %% ### Load the natural images:
natimgdata = load_natural_images(onlyright=True)

#%% Interpolation of receptive fields:
sessions = compute_pairwise_anatomical_distance(sessions)

sessions = exclude_outlier_rf(sessions,radius=50,rf_thr=50) 

sessions = smooth_rf(sessions,sig_thr=0.001,radius=50)

#%% Save session rf cell data as a copy to preserve estimated rf from sparse noise mapping
old_celldata    = pd.DataFrame({'rf_az_F': sessions[0].celldata['rf_az_F'],
                                 'rf_el_F': sessions[0].celldata['rf_el_F'], 
                                 'rf_p_F': sessions[0].celldata['rf_p_F'] })

#%% Save session rf cell data as a copy to preserve estimated rf from sparse noise mapping
old_celldata    = pd.DataFrame({'rf_az_F': sessions[0].celldata['rf_az_Fneu'],
                                 'rf_el_F': sessions[0].celldata['rf_el_Fneu'], 
                                 'rf_p_F': sessions[0].celldata['rf_p_Fneu'] })

#%% Get response-triggered frame for cells and then estimate receptive field from that:
sessions    = estimate_rf_IM(sessions,show_fig=False)

# #%% Flip elevation axis:
# vec_elevation       = [-16.7,50.2] #bottom and top of screen displays
# sessions[0].celldata['rf_el_F'] = -(sessions[0].celldata['rf_el_F']-np.mean(vec_elevation)) + np.mean(vec_elevation)

#%% Make a ascatter of azimuth estimated through rf mapping and by linear model of average triggered image:
areas       = ['V1', 'PM']
spat_dims   = ['az', 'el']
clrs_areas  = get_clr_areas(areas)
sig_thr     = 0.001
# sig_thr     = 0.05

fig,axes    = plt.subplots(2,2,figsize=(10,10))
for iarea,area in enumerate(areas):
    for ispat_dim,spat_dim in enumerate(spat_dims):
        idx = (sessions[0].celldata['roi_name'] == area) & (old_celldata['rf_p_F'] < sig_thr)
        # idx = (sessions2[0].celldata['roi_name'] == area) & (sessions[0].celldata['rf_p_F'] < 0.001)
        x = old_celldata['rf_' + spat_dim + '_F'][idx]
        y = sessions[0].celldata['rf_' + spat_dim + '_F'][idx]
        sns.scatterplot(ax=axes[iarea,ispat_dim],x=x,y=y,s=5,c=clrs_areas[iarea],alpha=0.5)
        #plot diagonal line
        axes[iarea,ispat_dim].plot([-180, 180], [-180, 180], color='black',linewidth=0.5)
        axes[iarea,ispat_dim].set_title(area + ' ' + spat_dim,fontsize=15)
        axes[iarea,ispat_dim].set_xlabel('Sparse Noise (deg)',fontsize=9)
        axes[iarea,ispat_dim].set_ylabel('Linear Model IM (deg)',fontsize=9)
        if spat_dim == 'az':
            axes[iarea,ispat_dim].set_xlim([-135,135])
            axes[iarea,ispat_dim].set_ylim([-135,135])
        elif spat_dim == 'el':
            axes[iarea,ispat_dim].set_xlim([-16.7,50.2])
            axes[iarea,ispat_dim].set_ylim([-16.7,50.2])
        idx = (~np.isnan(x)) & (~np.isnan(y))
        x =  x[idx]
        y =  y[idx]
        axes[iarea,ispat_dim].text(x=0,y=-10,s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Alignment_IM_RF_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

# %%

x = -np.log10(old_celldata['rf_p_F'])
y = sessions[0].celldata['rf_p_F']

fig,axes    = plt.subplots(1,1,figsize=(10,10))

sns.scatterplot(ax=axes,x=x,y=y,s=5,c='k',alpha=0.5)

#%% ##################### Retinotopic mapping within V1 and PM #####################

# from utils.rf_lib import plot_rf_plane,plot_rf_screen

# oldp = sessions[ises].celldata['rf_p_F']

# g = -np.log10(sessions[ises].celldata['rf_p_F'])
# g = 10**-sessions[ises].celldata['rf_p_F']
# g = 1.01**-sessions[ises].celldata['rf_p_F']
# plt.hist(g,bins=np.arange(0,0.1,0.001))

# sessions[ises].celldata['rf_p_F'] = 1.015**-oldp

sig_thr = 0.01
rf_type = 'F'
for ises in range(nSessions):
    fig = plot_rf_plane(sessions[ises].celldata,sig_thr=sig_thr,rf_type=rf_type) 
    # fig.savefig(os.path.join(savedir,'RF_planes','V1_PM_plane_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '.png'), format = 'png')


#%% ########### Plot locations of receptive fields as on the screen ##############################
rf_type = 'F'
for ises in range(nSessions):
    fig = plot_rf_screen(sessions[ises].celldata,sig_thr=sig_thr,rf_type=rf_type) 
    # fig.savefig(os.path.join(savedir,'RF_planes','V1_PM_rf_screen_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '_smooth.png'), format = 'png')
