"""
This script analyzes receptive field position across V1 and PM in 2P Mesoscope recordings
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from loaddata.session_info import filter_sessions,load_sessions
from utils.rf_lib import *
from loaddata.get_data_folder import get_local_drive
from utils.corr_lib import compute_pairwise_metrics

### TODO:
# append for multiple sessions to dataframe with pairwise measurements
# filter and compute only for cells with receptive field? Probably not

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - RF\\RF_quantification\\')

#################### Loading the data ##############################
# 
# sessions            = filter_sessions(protocols = ['SP'])

session_list        = np.array([['LPE09830','2023_04_10']])
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE10884','2023_10_20']])
session_list        = np.array([['LPE10885','2023_10_19']])
sessions,nSessions = load_sessions(protocol = 'SP',session_list=session_list)
sessions,nSessions = filter_sessions(protocols = ['GR'],only_animal_id='LPE09830')
sessions,nSessions = filter_sessions(protocols = ['GR'],only_animal_id=['LPE09665','LPE09830'],session_rf=True)
# sessions,nSessions = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=False, 
                                    # load_calciumdata=False, load_videodata=False, calciumversion='dF')
sessions,nSessions = filter_sessions(protocols = ['SP'])

sig_thr = 0.001 #cumulative significance of receptive fields clusters

###################### Retinotopic mapping within V1 and PM #####################

for ises in range(nSessions):
    fig = plot_rf_plane(sessions[ises].celldata,sig_thr=sig_thr) 
    fig.savefig(os.path.join(savedir,'V1_PM_azimuth_elevation_inplane_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

###### Plot locations of receptive fields and scale by probability ##############################

for ises in range(nSessions):
    fig = plot_rf_screen(sessions[ises].celldata,sig_thr=0.01) 


sessions = compute_pairwise_metrics(sessions)

###### Fit gradient of RF as a function of spatial location of somata:

# r2 = interp_rf(sessions,sig_thr=0.01,show_fit=True)

###### Smooth RF with local good fits (spatial location of somata): ######

for ises in range(nSessions):
    fig = plot_rf_plane(sessions[ises].celldata,sig_thr=sig_thr) 
    fig.savefig(os.path.join(savedir,'V1_PM_azimuth_elevation_inplane_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

smooth_rf(sessions,sig_thr=0.001,radius=100)

for ises in range(nSessions):
    fig = plot_rf_plane(sessions[ises].celldata,sig_thr=1) 
    fig.savefig(os.path.join(savedir,'V1_PM_azimuth_elevation_inplane_smooth_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

###

## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# ## remove any double cells (for example recorded in both GR and RF)
# celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

###################### Retinotopic mapping within V1 and PM #####################

fig        = plt.subplots(figsize=(12,12))

fracs = celldata.groupby('roi_name').count()['rf_azimuth'] / celldata.groupby('roi_name').count()['iscell']

sns.barplot(data = fracs,x = 'roi_name',y=fracs)
plt.savefig(os.path.join(savedir,'V1_PM_azimuth_elevation_inplane_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

###################### RF size difference between V1 and PM #####################
order = [0,1] #for statistical testing purposes
pairs = [(0,1)]

order = ['V1','PM'] #for statistical testing purposes
pairs = [('V1','PM')]
fig,ax   = plt.subplots(1,1,figsize=(3,4))

sns.violinplot(data=celldata,y="rf_size",x="roi_name",palette=['blue','red'],ax=ax)

annotator = Annotator(ax, pairs, data=celldata, x="roi_name", y="rf_size", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

ax.set_xlabel('area')
ax.set_ylabel('RF size\n(squared degrees)')

plt.savefig(os.path.join(savedir,'V1_PM_rf_size_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')
