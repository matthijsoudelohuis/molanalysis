# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statannotations.Annotator import Annotator

from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
from scipy.stats import zscore
from scipy.stats import binned_statistic,binned_statistic_2d

from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled
from utils.tuning import compute_tuning, compute_prefori
from utils.RRRlib import *

#%% 
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
session_list        = np.array([['LPE12013','2024_05_02']])
#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE11998','2024_05_02'], #GN
                                ['LPE12013','2024_05_02']]) #GN
sessions,nSessions   = load_sessions(protocol = 'GN',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)


#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GN'],filter_areas=['V1','PM'],session_rf=True) 

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=False)


#%% ########################## Compute signal and noise correlations: ###################################
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ##################### Compute pairwise receptive field distances: ##############################
sessions = smooth_rf(sessions,rf_type='Fneu')
sessions = exclude_outlier_rf(sessions)
sessions = compute_pairwise_delta_rf(sessions,rf_type='Fsmooth')

#%% ########################################################################################################
# ##################### Noise correlations within and across areas: ########################################
# ##########################################################################################################

# #Define the areapairs:
# areapairs       = ['V1-V1','PM-PM']
# clrs_areapairs  = get_clr_area_pairs(areapairs)


# dfses = mean_corr_areas_labeling([sessions[0]],corr_type='trace_corr',absolute=True,minNcells=100)
# clrs_area_labelpairs = get_clr_area_labelpairs(list(dfses.columns))

# pairs = [('V1unl-V1unl','V1lab-V1lab'),
#          ('V1unl-V1unl','V1unl-V1lab'),
#          ('V1unl-V1lab','V1lab-V1lab'),
#          ('PMunl-PMunl','PMunl-PMlab'),
#          ('PMunl-PMunl','PMlab-PMlab'),
#          ('PMunl-PMlab','PMlab-PMlab'),
#          ('V1unl-PMlab','V1lab-PMlab'),
#          ('V1lab-PMunl','V1lab-PMlab'),
#          ('V1unl-PMunl','V1lab-PMlab'),
#          ] #for statistics

#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

# #%% #########################################################################################
# # Contrast: across areas
# # areas               = ['V1','PM']

# areapairs           = ['V1-V1']

# areapairs           = ['V1-V1','PM-PM','V1-PM']
# layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
# projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

# #If you override any of these then these pairs will be ignored:
# layerpairs          = ' '
# # areapairs           = ' '
# # projpairs           = ' '

# deltaori            = [-15,15]
# # deltaori            = [80,100]
# # deltaori            = None
# rotate_prefori      = True
# rf_type             = 'Fsmooth'

# [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions,areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
#                             corr_type='trace_corr',binresolution=5,rotate_prefori=rotate_prefori,deltaori=deltaori,rf_type=rf_type,
#                             sig_thr = 0.001,noise_thr=1,tuned_thr=0.00,absolute=False,normalize=False)
# binmean[bincounts<min_counts]           = np.nan

# #%% Definitions of azimuth, elevation and delta RF 2D space:
# delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
# deltarf             = np.sqrt(delta_az**2 + delta_el**2)
# anglerf             = np.mod(np.arctan2(delta_az,delta_el)+np.pi/2,np.pi*2)

# #%% Make the figure:
# centerthr           = [15,25,25]
# min_counts          = 10

#%% 2D CCA maps:

binres              = 5 #deg steps in azimuth and elevation to select target neurons

vec_elevation       = [-16.7,50.2] #bottom and top of screen displays
vec_azimuth         = [-135,135] #left and right of screen displays

binedges_az         = np.arange(vec_azimuth[0],vec_azimuth[1]+binres,binres)
binedges_el         = np.arange(vec_elevation[0],vec_elevation[1]+binres,binres)

deltabinres         = 5 #deg steps in azimuth and elevation to bin weights of source neurons

delta_rfbinedges    = np.arange(-75,75,deltabinres)
delta_rfbincenters  = delta_rfbinedges[:-1] + deltabinres/2

# target_area         = 'V1'
# source_area         = 'PM'
# 
target_area         = 'PM'
source_area         = 'V1'

min_target_neurons  = 10

binmean             = np.zeros((len(delta_rfbincenters),len(delta_rfbincenters)))
bincounts           = np.zeros((len(delta_rfbincenters),len(delta_rfbincenters)))

# method              = 'CCA'
method              = 'RRR'

n_components        = 5
lambda_reg          = 1

absolute            = True

for ises,ses in enumerate(sessions):
    for iaz,az in enumerate(binedges_az[:-1]):
        for iel,el in enumerate(binedges_el[:-1]):
            idx_in_bin = np.where((ses.celldata['roi_name']==target_area) & 
                                (ses.celldata['rf_az_Fsmooth']>=binedges_az[iaz]) & 
                                (ses.celldata['rf_az_Fsmooth']<binedges_az[iaz+1]) & 
                                (ses.celldata['rf_el_Fsmooth']>=binedges_el[iel]) & 
                                (ses.celldata['rf_el_Fsmooth']<binedges_el[iel+1]))[0]
            
            if len(idx_in_bin)>min_target_neurons:
                X       = ses.respmat[idx_in_bin,:].T
                Y       = ses.respmat[ses.celldata['roi_name']==source_area,:].T
                
                if method=='CCA':
                    cca     = CCA(n_components=n_components,copy=False)
                    cca.fit(X,Y)
                    # weights = np.mean(cca.x_weights_,axis=1)
                    if absolute:
                        cca.y_weights_     = np.abs(cca.y_weights_)
                    weights     = np.mean(cca.y_weights_,axis=1)
                elif method=='RRR':
                    ## LM model run
                    B_hat = LM(Y, X, lam=lambda_reg)

                    B_hat_rr = RRR(Y, X, B_hat, r=n_components, mode='left')
                    if absolute:
                        B_hat_rr     = np.abs(B_hat_rr)
                    weights     = np.mean(B_hat_rr,axis=0)

                xdata       = ses.celldata['rf_az_Fsmooth'][ses.celldata['roi_name']==source_area] - np.mean(binedges_az[iaz:iaz+2])
                ydata       = ses.celldata['rf_el_Fsmooth'][ses.celldata['roi_name']==source_area] - np.mean(binedges_el[iel:iel+2])

                #Take the sum of the weights in each bin:
                binmean[:,:]   += binned_statistic_2d(x=xdata, y=ydata, values=weights,
                                                                    bins=delta_rfbinedges, statistic='sum')[0]
                bincounts[:,:] += np.histogram2d(x=xdata, y=ydata, bins=delta_rfbinedges)[0]

#Get the mean by dividing by the number of paired neurons in each bin:
binmean = binmean/bincounts

#%% 
deglim = 75
delta_az,delta_el = np.meshgrid(delta_rfbincenters,delta_rfbincenters)

fig,ax = plt.subplots(figsize=(5,4))
data = binmean
s = ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95),cmap="hot")
# ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
ax.set_xlim([-deglim,deglim])
ax.set_ylim([-deglim,deglim])
ax.set_xlabel(u'Δ Azimuth')
ax.set_ylabel(u'Δ Elevation')
cbar = fig.colorbar(s)


#%% 
# fig,axes    = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
# if len(projpairs)==1 and len(areapairs)==1:
#     axes = np.array([axes])
# axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax                                              = axes[ipp,iap]
            data                                            = copy.deepcopy(binmean[:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bincounts[:,:,iap,ilp,ipp]<min_counts]     = np.nan

            ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95),cmap="hot")
            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([-deglim,deglim])
            ax.set_ylim([-deglim,deglim])
            ax.set_xlabel(u'Δ deg Collinear')
            ax.set_ylabel(u'Δ deg Orthogonal')
            circle=plt.Circle((0,0),centerthr[iap], color='g', fill=False,linestyle='--',linewidth=1)
            ax.add_patch(circle)

#%% Show distribution of delta receptive fields across areas: 

sessions = compute_pairwise_delta_rf(sessions,rf_type='F')

#Make a figure with each session is one line for each of the areapairs a histogram of distmat_rf:
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

#%%
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                ['LPE11086','2023_12_15'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['session_id'].isin([x[0] + '_' + x[1] for x in session_list]))[0]
sessions_subset = [sessions[i] for i in sessions_in_list]

#%% Give redcells a string label
redcelllabels = np.array(['unl','lab'])
for ses in sessions:
    ses.celldata['labeled'] = ses.celldata['redcell']
    ses.celldata['labeled'] = ses.celldata['labeled'].astype(int).apply(lambda x: redcelllabels[x])



fig,axes    = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax                                              = axes[ipp,iap]
            data                                            = copy.deepcopy(binmean[:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bincounts[:,:,iap,ilp,ipp]<min_counts]     = np.nan

            ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95),cmap="hot")
            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([-deglim,deglim])
            ax.set_ylim([-deglim,deglim])
            ax.set_xlabel(u'Δ deg Collinear')
            ax.set_ylabel(u'Δ deg Orthogonal')
            circle=plt.Circle((0,0),centerthr[iap], color='g', fill=False,linestyle='--',linewidth=1)
            ax.add_patch(circle)

plt.tight_layout()
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Collinear' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Collinear_labeled' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Orthogonal' % (corr_type) + '.png'), format = 'png')

#%% Average correlation values based on circular tuning:
polarbinres         = 45
polarbinedges       = np.deg2rad(np.arange(0,360,step=polarbinres))
polarbincenters     = polarbinedges[:-1]+np.deg2rad(polarbinres/2)
polardata           = np.empty((len(polarbincenters),2,*np.shape(binmean)[2:]))
for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            data = binmean[:,:,iap,ilp,ipp].copy()
            data[deltarf>centerthr[iap]] = np.nan
            polardata[:,0,iap,ilp,ipp] = binned_statistic(x=anglerf[~np.isnan(data)],
                                    values=data[~np.isnan(data)],
                                    statistic='mean',bins=polarbinedges)[0]
            data = binmean[:,:,iap,ilp,ipp].copy()
            data[deltarf<=centerthr[iap]] = np.nan
            polardata[:,1,iap,ilp,ipp]  = binned_statistic(x=anglerf[~np.isnan(data)],
                                    values=data[~np.isnan(data)],
                                    statistic='mean',bins=polarbinedges)[0]

# Make the figure:
deglim      = 2*np.pi
fig,axes    = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax                                          = axes[ipp,iap]
            ax.plot(polarbincenters,polardata[:,0,iap,ilp,ipp],color='k',label='center')
            ax.plot(polarbincenters,polardata[:,1,iap,ilp,ipp],color='g',label='surround')
            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([0,deglim])
            # ax.set_ylim([0.04,0.1])
            ax.set_xticks(np.arange(0,2*np.pi,step = np.deg2rad(45)),labels=np.arange(0,360,step = 45))
            ax.set_xlabel(u'Angle (deg)')
            ax.set_ylabel(u'Correlation')
            ax.legend(frameon=False,fontsize=8,loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(savedir,'DeltaRF_1D_Polar_%s_GR_Collinear_labeled' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_1D_Polar_%s_GR_Collinear' % (corr_type) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'DeltaRF_2D_%s_GR_Orthogonal' % (corr_type) + '.png'), format = 'png')

#%% 
fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(10,5))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(areapairs),len(projpairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax = axes[iap,ipp]
            ax.imshow(np.log10(bincounts[:,:,iap,ilp,ipp]),vmax=np.nanpercentile(np.log10(bincounts),99.9),
                cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
            # ax.imshow(binmean[:,:,iap,ilp,ipp],vmin=np.nanpercentile(binmean[:,:,iap,ilp,ipp],5),
            #                     vmax=np.nanpercentile(binmean[:,:,iap,ilp,ipp],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
            ax.set_title('%s\n%s' % (areapair, layerpair))
            ax.set_xlim([-75,75])
            ax.set_ylim([-75,75])
plt.tight_layout()

#%% #########################################################################################
# Contrasts: across areas and projection identity      

[noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions_subset,corr_type='trace_corr',
                                                                binresolution=10,rotate_prefori=False,thr_tuned=0.0,
                                                                thr_rf_p=0.001,rf_type='F')

min_counts = 50
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_AllProt_%dsessions' %nSessions  + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'2D_NC_smooth_Map_Area_Proj_GN_F_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'2D_NC_Map_smooth_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')




