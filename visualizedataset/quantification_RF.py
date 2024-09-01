"""
This script analyzes receptive field position across V1 and PM in 2P Mesoscope recordings
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from loaddata.session_info import filter_sessions,load_sessions
from utils.rf_lib import *
from loaddata.get_data_folder import get_local_drive
from utils.corr_lib import compute_pairwise_anatomical_distance
from utils.plotting_style import * #get all the fixed color schemes

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - RF\\')

#%% ################### Loading the data ##############################

session_list        = np.array([['LPE11086','2024_01_08']])
session_list        = np.array([['LPE12013','2024_05_02']])

#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                # ['LPE11998','2024_05_02'], #GN
                                # ['LPE12013','2024_05_02'], #GN
                                # ['LPE12013','2024_05_07'], #GN
                                # ['LPE11086','2023_12_15'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessions,nSessions = load_sessions(protocol = 'SP',session_list=session_list)

# sessions,nSessions = filter_sessions(protocols = ['GR'],only_animal_id=['LPE09665','LPE09830'],session_rf=True)
# # sessions,nSessions = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=False, 
                                    # load_calciumdata=False, load_videodata=False, calciumversion='dF')
sessions,nSessions = filter_sessions(protocols = ['SP','GR','IM','GN'],session_rf=True,filter_areas=['V1','PM'])

sig_thr = 0.05 #cumulative significance of receptive fields clusters
sig_thr = 0.001 #cumulative significance of receptive fields clusters

#%% Show fraction of receptive fields per session before any corrections:
[fig,rf_frac_F] = plot_RF_frac(sessions,rf_type='F',sig_thr=sig_thr)
fig.savefig(os.path.join(savedir,'RF_quantification','RF_fraction_F' + '.png'), format = 'png')

#%% ##################### Retinotopic mapping within V1 and PM #####################
rf_type = 'F'
for ises in range(nSessions):
    fig = plot_rf_plane(sessions[ises].celldata,sig_thr=sig_thr,rf_type=rf_type) 
    fig.savefig(os.path.join(savedir,'RF_planes','V1_PM_plane_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '.png'), format = 'png')

#%% Interpolation of receptive fields:
sessions = compute_pairwise_anatomical_distance(sessions)

sessions = exclude_outlier_rf(sessions,radius=50,rf_thr=50) 

#Show fraction of receptive fields per session after filtering out scattered neurons: 
[fig,rf_frac_F] = plot_RF_frac(sessions,rf_type='F',sig_thr=sig_thr)
fig.savefig(os.path.join(savedir,'RF_quantification','RF_fraction_F_filter' + '.png'), format = 'png')

sessions = smooth_rf(sessions,sig_thr=0.001,radius=50)

#%% Show fraction of receptive fields per session after smoothed interpolation and filtering:
[fig,rf_frac_F] = plot_RF_frac(sessions,rf_type='Fsmooth',sig_thr=sig_thr)
fig.savefig(os.path.join(savedir,'RF_quantification','RF_fraction_Fsmooth' + '.png'), format = 'png')

#%% ##################### Retinotopic mapping within V1 and PM #####################
rf_type = 'F'
rf_type = 'Fneu'
# rf_type = 'Fsmooth'
for ises in range(nSessions):
    fig = plot_rf_plane(sessions[ises].celldata,sig_thr=sig_thr,rf_type=rf_type) 
    fig.savefig(os.path.join(savedir,'RF_planes','V1_PM_plane_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '_smooth.png'), format = 'png')

#%% ########### Plot locations of receptive fields as on the screen ##############################
rf_type = 'F'
for ises in range(nSessions):
    fig = plot_rf_screen(sessions[ises].celldata,sig_thr=sig_thr,rf_type=rf_type) 
    fig.savefig(os.path.join(savedir,'RF_planes','V1_PM_rf_screen_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '_smooth.png'), format = 'png')

#%% ##################### Scatter between individual RF location and neuropil estimation #####################
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# ## remove any double cells (for example recorded in both GR and RF)
celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

logpdata   = -np.log10(celldata['rf_p_F'])
dev_az      = np.abs(celldata['rf_az_F'] - celldata['rf_az_Fneu'])
dev_el      = np.abs(celldata['rf_el_F'] - celldata['rf_el_Fneu'])

pticks = 1/np.power(10,np.arange(1,10))
alphaval = 0.1
areas   = ['V1','PM']
spat_dims = ['az','el']
clrs_areas = get_clr_areas(areas)
fig,axes   = plt.subplots(2,2,figsize=(8,8))

for iarea,area in enumerate(areas):
    for ispat_dim,spat_dim in enumerate(spat_dims):
        idx = celldata['roi_name'] == area

        if spat_dim == 'az':
            axes[iarea,ispat_dim].scatter(logpdata[idx],dev_az[idx],s=3,alpha=alphaval,c=clrs_areas[iarea])
            axes[iarea,ispat_dim].set_ylim([0,250])

        elif spat_dim == 'el':
            axes[iarea,ispat_dim].scatter(logpdata[idx],dev_el[idx],s=3,alpha=alphaval,c=clrs_areas[iarea])
            axes[iarea,ispat_dim].set_ylim([0,60])

        axes[iarea,ispat_dim].set_xticks(-np.log10(pticks),labels=pticks,fontsize=6)
        axes[iarea,ispat_dim].set_title(area + ' ' + spat_dim)
        axes[iarea,ispat_dim].set_xlim([1,10])
        axes[iarea,ispat_dim].set_ylabel('RF Scatter')

plt.tight_layout()

# fig.savefig(os.path.join(savedir,'RF_quantification','RF_scatter' + '.png'), format = 'png')

#%% ##### Plot locations of receptive fields and scale by probability ##############################
rf_type = 'F'
for ises in range(nSessions):
    fig = plot_rf_screen(sessions[ises].celldata,sig_thr=sig_thr,rf_type=rf_type) 
    fig.savefig(os.path.join(savedir,'RF_planes','V1_PM_rf_screen_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '.png'), format = 'png')

#%% 
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


###################### Include only neurons nearby labeled cells #####################
rads = np.arange(600)
fracincl = np.empty((nSessions,600))

for ises in range(nSessions):
    for radius in rads:
        idx = filter_nearlabeled(sessions[ises],radius=radius)
        fracincl[ises,radius] = np.sum(idx) / len(sessions[ises].celldata)

fig = plt.figure(figsize=(4,3))
plt.plot(rads,fracincl.T,linewidth=2,alpha=0.5)
plt.plot(rads,np.median(fracincl,axis=0),linewidth=3,color='black')
plt.xlabel(u"Dist. from labeled neuron \u03bcm")
plt.ylabel('Included data')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Filter_NearLabeled_%d_Sessions' % nSessions + '.png'))

#%% ##################### How far are individual cells from neuropil #####################
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

fig,axes = plt.subplots(1,5,figsize=(10,3))

axes[0].hist(celldata['rf_az_F'][celldata['roi_name']=='V1'] - celldata['rf_az_Fneu'][celldata['roi_name']=='V1'],
             bins=np.arange(-200,200,step=5),density=True,color='green')
axes[0].set_title('V1 - Azimuth jitter')
axes[1].hist(celldata['rf_el_F'][celldata['roi_name']=='V1'] - celldata['rf_el_Fneu'][celldata['roi_name']=='V1'],
             bins=np.arange(-200,200,step=5),density=True,color='green')
axes[1].set_title('V1 - Elevation jitter')

axes[2].hist(celldata['rf_az_F'][celldata['roi_name']=='PM'] - celldata['rf_az_Fneu'][celldata['roi_name']=='PM'],
             bins=np.arange(-200,200,step=5),density=True,color='purple')
axes[2].set_title('PM - Azimuth jitter')
axes[3].hist(celldata['rf_el_F'][celldata['roi_name']=='PM'] - celldata['rf_el_Fneu'][celldata['roi_name']=='PM'],
             bins=np.arange(-200,200,step=5),density=True,color='purple')
axes[3].set_title('PM - Elevation jitter')

pvals = [0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]
jitterdata = np.empty((2,2,len(pvals)))
for ip,p in enumerate(pvals):
    idx = np.logical_and(celldata['roi_name']=='V1',celldata['rf_p_F']<p)
    jitterdata[0,0,ip] = np.std(celldata['rf_az_F'][idx] - celldata['rf_az_Fneu'][idx])
    idx = np.logical_and(celldata['roi_name']=='PM',celldata['rf_p_F']<p)
    jitterdata[0,1,ip] = np.std(celldata['rf_az_F'][idx] - celldata['rf_az_Fneu'][idx])
    idx = np.logical_and(celldata['roi_name']=='V1',celldata['rf_p_F']<p)
    jitterdata[1,0,ip] = np.std(celldata['rf_el_F'][idx] - celldata['rf_el_Fneu'][idx])
    idx = np.logical_and(celldata['roi_name']=='PM',celldata['rf_p_F']<p)
    jitterdata[1,1,ip] = np.std(celldata['rf_el_F'][idx] - celldata['rf_el_Fneu'][idx])


axes[4].plot(-np.log10(pvals),jitterdata[0,0,:],linewidth=2,linestyle=':',color='purple')
axes[4].plot(-np.log10(pvals),jitterdata[0,1,:],linewidth=2,linestyle='-',color='purple')
axes[4].plot(-np.log10(pvals),jitterdata[1,0,:],linewidth=2,linestyle=':',color='green')
axes[4].plot(-np.log10(pvals),jitterdata[1,1,:],linewidth=2,linestyle='-',color='green')
axes[4].set_xticks(-np.log10(pvals),labels=pvals)
axes[4].set_xlabel("P-value threshold")
axes[4].set_ylabel('RF jitter')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'RF_jitter' + '.png'), format = 'png')
