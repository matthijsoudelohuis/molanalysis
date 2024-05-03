
####################################################
import os
from loaddata.get_data_folder import get_local_drive
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic,binned_statistic_2d

from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
# from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes
# from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
# from utils.plot_lib import shaded_error
# from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

##############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])

# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['SP']) 

#   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')


##############################################################################
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%%%% 

areas   = ['V1','PM']
rf_frac = np.empty((nSessions,len(areas)))
for ises in range(nSessions):    # iterate over sessions
    for iarea in range(len(areas)):    # iterate over sessions
        idx = sessions[ises].celldata['roi_name'] == areas[iarea]
        rf_frac[ises,iarea] = np.sum(sessions[ises].celldata['rf_p'][idx]<0.0001) / np.sum(idx)

fig,ax = plt.subplots(figsize=(4,4))
# plt.scatter([0,1],rf_frac)
sns.scatterplot(rf_frac.T,color='black',s=50)
plt.xlim([-0.5,1.5])
plt.ylim([0,0.5])
plt.xticks([0,1],labels=areas)
# plt.xticks(areas)
plt.xlabel('Area')
plt.ylabel('Fraction receptive fields')
# plt.legend()
ax.get_legend().remove()
plt.savefig(os.path.join(savedir,'RF_fraction' + '.png'), format = 'png')


############################ Compute noise correlations: ###################################
sessions = compute_noise_correlation(sessions)

###################### Plot control figure of signal and noise corrs ##############################
sesidx = 0
fig = plt.figure(figsize=(8,5))
plt.imshow(sessions[sesidx].noise_corr, cmap='coolwarm',
           vmin=np.nanpercentile(sessions[sesidx].noise_corr,5),
           vmax=np.nanpercentile(sessions[sesidx].noise_corr,95))
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NC_SP_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')



#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_metrics(sessions)

# # construct dataframe with all pairwise measurements:
# df_allpairs  = pd.DataFrame()

# for ises in range(nSessions):
#     tempdf  = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr.flatten(),
#                     # 'DeltaPrefOri': sessions[ises].delta_pref.flatten(),
#                     'AreaPair': sessions[ises].areamat.flatten(),
#                     'DistXYPair': sessions[ises].distmat_xy.flatten(),
#                     'DistXYZPair': sessions[ises].distmat_xyz.flatten(),
#                     'DistRfPair': sessions[ises].distmat_rf.flatten(),
#                     'AreaLabelPair': sessions[ises].arealabelmat.flatten(),
#                     'LabelPair': sessions[ises].labelmat.flatten()}).dropna(how='all') 
#                     #drop all rows that have all nan (diagonal + repeat below daig)
#     df_allpairs  = pd.concat([df_allpairs, tempdf], ignore_index=True).reset_index(drop=True)

smooth_rf(sessions,sig_thr=0.01,show_result=False,radius=100)

# Recompute noise correlations without setting half triangle to nan
sessions =  compute_noise_correlation(sessions,uppertriangular=False)

rotate_prefori  = False
min_counts      = 500 # minimum pairwise observation to include bin

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap(sessions,binresolution=5,
                                                         rotate_prefori=rotate_prefori,
                                                         rotate_deltaprefori=False)
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

## Show the counts of pairs:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(countsRFmat,vmin=np.percentile(countsRFmat,5),vmax=np.percentile(countsRFmat,99),interpolation='none',extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='counts')
if not rotate_prefori:
    plt.xlabel('delta Azimuth')
    plt.ylabel('delta Elevation')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Counts_%dsessions' %nSessions  + '.png'), format = 'png')
else:
    plt.xlabel('Collinear')
    plt.ylabel('Orthogonal')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Counts_Rotated_%dsessions' %nSessions  + '.png'), format = 'png')

## Show the noise correlation map:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(noiseRFmat_mean,vmin=np.nanpercentile(noiseRFmat_mean,5),vmax=np.nanpercentile(noiseRFmat_mean,95),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='noise correlation')
if not rotate_prefori:
    plt.xlabel('delta Azimuth')
    plt.ylabel('delta Elevation')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_%dsessions' %nSessions  + '.png'), format = 'png')
else:
    plt.xlabel('Collinear')
    plt.ylabel('Orthogonal')


#%% #########################################################################################
# Contrast: across areas
areas   = ['V1','PM']

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap_areas(sessions,binresolution=5,
                                                                 rotate_prefori=False,thr_tuned=0.0,
                                                                 thr_rf_p=1)

min_counts = 1000
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(2,2,figsize=(10,7))
for i in range(2):
    for j in range(2):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(areas[i] + '-' + areas[j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_smooth_Map_Interarea_SP_%dsessions' %nSessions  + '.png'), format = 'png')

#%% #########################################################################################
# Contrasts: across areas and projection identity      

[noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions,binresolution=5,
                                                                 rotate_prefori=False,thr_tuned=0.00,
                                                                 thr_rf_p=1)

min_counts = 50
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_smooth_Map_Area_Proj_SP_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_smooth_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')

