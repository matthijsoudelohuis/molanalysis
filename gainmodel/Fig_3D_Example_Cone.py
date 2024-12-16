
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE12223','2024_06_10']])

# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,filter_areas=['V1','PM'])

#   Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)


#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% Make the 3D figure:
fig = plot_PCA_gratings_3D(sessions[0],thr_tuning=0.05,plotgainaxis=True)
axes = fig.get_axes()
axes[0].view_init(elev=-30, azim=25, roll=40)
axes[1].view_init(elev=15, azim=0, roll=-10)
axes[0].set_xlim([-2,35])
axes[0].set_ylim([-2,35])
axes[1].set_zlim([-5,45])
for ax in axes:
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Get rid of colored axes planes, remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

plt.tight_layout()
fig.savefig(os.path.join(savedir,'Example_Cone_3D_V1_PM_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% SHOW AL as well: #############################################################################

session_list        = np.array([['LPE12223','2024_06_10']])

# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,filter_areas=['AL'])

# Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)

#%% ########################### Compute tuning metrics: ###################################
sessions[0].celldata['tuning_var'] = compute_tuning(sessions[0].respmat,
                                        sessions[0].trialdata['Orientation'],tuning_metric='tuning_var')

#%% 
fig = plot_PCA_gratings_3D(sessions[0],thr_tuning=0.025)
axes = fig.get_axes()
axes[0].view_init(elev=25, azim=45, roll=-45)
axes[0].set_zlim([-5,15])
for ax in axes:
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

# plt.tight_layout()
fig.savefig(os.path.join(savedir,'Example_Cone_3D_AL_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% 









#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE12223','2024_06_10']])

# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,filter_areas=['V1'])

#   Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)

#%% ########################### Compute tuning metrics: ###################################
sessions[0].celldata['tuning_var'] = compute_tuning(sessions[0].respmat,
                                        sessions[0].trialdata['Orientation'],tuning_metric='tuning_var')


#%% Fit population gain model:
orientations        = sessions[0].trialdata['Orientation']
data                = sessions[0].respmat
data_hat_poprate    = pop_rate_gain_model(data, orientations)

datasets            = (data,data_hat_poprate)
fig = plot_respmat(orientations, datasets, ['original','pop rate gain'])

#%% Make session objects with only gain, or no gain at all:
sessions_onlygain   = copy.deepcopy(sessions)
sessions_nogain     = copy.deepcopy(sessions)

sessions_onlygain[0].respmat = data_hat_poprate
sessions_nogain[0].respmat = data - data_hat_poprate

#%% Make the 3D figure for original data:
fig = plot_PCA_gratings_3D(sessions[0],thr_tuning=0)
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=0, roll=-10)
axes[0].set_zlim([-5,45])
fig.savefig(os.path.join(savedir,'Cone_3D_V1_Original_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% Make the 3D figure for only gain data:
fig = plot_PCA_gratings_3D(sessions_onlygain[0],thr_tuning=0)
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=-15, roll=-35)
fig.savefig(os.path.join(savedir,'Cone_3D_V1_Gainonly_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% Make the 3D figure for gain-subtracted data:
fig = plot_PCA_gratings_3D(sessions_nogain[0],thr_tuning=0)
axes = fig.get_axes()
axes[0].view_init(elev=65, azim=-135, roll=0)
fig.savefig(os.path.join(savedir,'Cone_3D_V1_Nogain_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

# #%% #############################################################################

# sessions,nSessions = filter_sessions(protocols = ['GR'],only_all_areas=['V1','PM','AL'])
# sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# session_list        = np.array([['LPE11622','2024_03_26']])
# # session_list        = np.array([['LPE11998','2024_05_10']])
# # session_list        = np.array([['LPE12223','2024_06_10']])

# # load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

# #   Load proper data and compute average trial responses:                      
# for ises in range(nSessions):    # iterate over sessions
#     sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
#                                 calciumversion='deconv',keepraw=True)

# #%% ########################### Compute tuning metrics: ###################################
# for ises in range(nSessions):
#     if sessions[ises].sessiondata['protocol'].isin(['GR','GN'])[0]:
#         sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
#                                                     sessions[ises].trialdata['Orientation'],
#                                                     tuning_metric='OSI')
#         sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
#                                                         sessions[ises].trialdata['Orientation'],
#                                                         tuning_metric='gOSI')
#         sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
#                                                         sessions[ises].trialdata['Orientation'],
#                                                         tuning_metric='tuning_var')
#         sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
#                                                         sessions[ises].trialdata['Orientation'])
# #%% 
# fig = plot_PCA_gratings_3D(sessions[ises])
