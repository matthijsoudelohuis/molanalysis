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
from utils.plot_lib import * #get all the fixed color schemes
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled
from utils.tuning import compute_tuning, compute_prefori
from utils.RRRlib import *


#%% 
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE12013_2024_05_02']])
#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE11998_2024_05_02'], #GN
                                ['LPE12013_2024_05_02']]) #GN
sessions,nSessions   = filter_sessions(protocols = 'GN',only_session_id=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)


#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_areas=['V1','PM'],session_rf=True) 

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:  
calciumversion = 'deconv'                    
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)


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


# delta_rfbinedges    = np.arange(-75,75,deltabinres)
# delta_rfbincenters  = delta_rfbinedges[:-1] + deltabinres/2

# target_area         = 'V1'
# source_area         = 'PM'
# 

# min_target_neurons  = 10

# binmean             = np.zeros((len(delta_rfbincenters),len(delta_rfbincenters)))
# bincounts           = np.zeros((len(delta_rfbincenters),len(delta_rfbincenters)))

# method              = 'CCA'
# method              = 'RRR'

# n_components        = 5
# lambda_reg          = 1

#%%
for ses in sessions:
    if 'rf_az_Fsmooth' in ses.celldata.columns:
        # print(np.sum(~np.isnan( ses.celldata['rf_az_Fsmooth'])) / len(ses.celldata))
        print(np.sum(~np.isnan( ses.celldata['rf_az_Fneu'])) / len(ses.celldata))
        # print(np.sum(ses.celldata['rf_r2_Fneu']>0.2) / len(ses.celldata))

#%% Matched and mismatched receptive fields across areas: 

binres              = 10 #deg steps in azimuth and elevation to select target neurons

vec_elevation       = [-16.7,50.2] #bottom and top of screen displays
vec_azimuth         = [-135,135] #left and right of screen displays

binedges_az         = np.arange(vec_azimuth[0],vec_azimuth[1]+binres,binres)
binedges_el         = np.arange(vec_elevation[0],vec_elevation[1]+binres,binres)
nbins_az            = len(binedges_az)
nbins_el            = len(binedges_el)

radius_match        = 15 #deg, radius of receptive field to match
radius_mismatch     = 15 #deg, radius of receptive field to mismatch, if within this radius then excluded

# arealabelpairs      = ['PMunl-V1unl']
arealabelpairs      = ['V1unl-PMunl','PMunl-V1unl']
# arealabelpairs      = ['PMunl-V1unl','PMunl-V1lab']
# arealabelpairs      = ['PMunl-V1unl','PMlab-V1unl']
# arealabelpairs      = ['V1unl-PMunl','V1lab-PMunl']

arealabelpairs  = ['V1unl-PMunl',
                    'V1lab-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1unl',
                    'PMlab-V1lab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

nsampleneurons       = 20

lam                 = 0
nranks              = 20
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5
maxnoiselevel       = 20

R2_cv               = np.full((nbins_az,nbins_el,narealabelpairs,2,nSessions),np.nan)
optim_rank          = np.full((nbins_az,nbins_el,narealabelpairs,2,nSessions),np.nan)
R2_ranks            = np.full((nbins_az,nbins_el,narealabelpairs,2,nSessions,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for match/mismatch RF:'):
    if 'rf_az_Fsmooth' not in ses.celldata.columns:
        continue
    # sesaz = ses.celldata['rf_az_Fsmooth'].to_numpy()
    # sesel = ses.celldata['rf_el_Fsmooth'].to_numpy()
    sesaz = ses.celldata['rf_az_Fneu'].to_numpy()
    sesel = ses.celldata['rf_el_Fneu'].to_numpy()

    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        for iaz,az in enumerate(binedges_az):
            for iel,el in enumerate(binedges_el):
                idx_match = np.all((sesaz>=az-radius_match,
                                    sesaz<az+radius_match,
                                    sesel>=el-radius_match,
                                    sesel<el+radius_match),axis=0)
                
                idx_mismatch = ~np.all((sesaz>=az-radius_mismatch,
                                    sesaz<az+radius_mismatch,
                                    sesel>=el-radius_mismatch,
                                    sesel<el+radius_mismatch),axis=0)
                
                idx_x       = np.where(np.all((ses.celldata['arealabel']==alx,
                                idx_match,
                                ses.celldata['rf_r2_Fneu']>0.2,
                                ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            
                idx_y_match  = np.where(np.all((ses.celldata['arealabel']==aly,
                                idx_match,
                                ses.celldata['rf_r2_Fneu']>0.2,
                                ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            
                idx_y_mismatch  = np.where(np.all((ses.celldata['arealabel']==aly,
                                idx_mismatch,
                                ses.celldata['rf_r2_Fneu']>0.2,
                                ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]

                if len(idx_x)<nsampleneurons or len(idx_y_match)<nsampleneurons or len(idx_y_mismatch)<nsampleneurons: #skip exec if not enough neurons in one of the populations
                    continue
                
                X                   = sessions[ises].respmat[np.ix_(idx_x,idx_T)].T #Get activity and transpose to samples x features
                Y                   = sessions[ises].respmat[np.ix_(idx_y_match,idx_T)].T
                Z                   = sessions[ises].respmat[np.ix_(idx_y_mismatch,idx_T)].T

                # R2_cv[iaz,iel,iapl,0,ises] = 1
                # R2_cv[iaz,iel,iapl,1,ises] = 1
                
                R2_cv[iaz,iel,iapl,0,ises],optim_rank[iaz,iel,iapl,0,ises],R2_ranks[iaz,iel,iapl,0,ises,:,:,:]  = \
                    RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

                R2_cv[iaz,iel,iapl,1,ises],optim_rank[iaz,iel,iapl,1,ises],R2_ranks[iaz,iel,iapl,1,ises,:,:,:]  = \
                    RRR_wrapper(Z, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%%
print('Fraction of array filled with data: %.2f' % (np.sum(~np.isnan(R2_cv)) / R2_cv.size))

#%% Plot the results: 
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*2,3),sharey=True,sharex=True)
if narealabelpairs == 1:
    axes = np.array([axes])

for iapl, arealabelpair in enumerate(arealabelpairs):
    ax = axes[iapl]

    datatoplot = np.column_stack((R2_cv[:,:,iapl,0,:].flatten(),R2_cv[:,:,iapl,1,:].flatten())) 
    datatoplot = datatoplot[~np.isnan(datatoplot).any(axis=1)]

    ax.scatter(np.zeros(len(datatoplot))+np.random.randn(len(datatoplot))*0.05,datatoplot[:,0],color='k',marker='o',s=10)
    ax.errorbar(0.2,np.nanmean(datatoplot[:,0]),np.nanstd(datatoplot[:,0])/np.sqrt(nSessions),color='g',marker='o',zorder=10)

    ax.scatter(np.ones(len(datatoplot))+np.random.randn(len(datatoplot))*0.05,datatoplot[:,1],color='k',marker='o',s=10)
    ax.errorbar(1.2,np.nanmean(datatoplot[:,1]),np.nanstd(datatoplot[:,1])/np.sqrt(nSessions),color='r',marker='o',zorder=10)

    ax.set_xticks([0,1],['Match','Mismatch'])
    ax.set_ylabel('R2')
    ax.set_title('%s' % arealabelpair,fontsize=8)
ax.set_ylim([0,0.25])
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
# my_savefig(fig,savedir,'RRR_R2_MatchMismatch_RF_%dsessions' % (nSessions),formats = ['png'])

#%% Plot the results: 
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*1.3,3),sharey=True,sharex=True)
if narealabelpairs == 1:
    axes = np.array([axes])

for iapl, arealabelpair in enumerate(arealabelpairs):
    ax = axes[iapl]

    datatoplot = np.column_stack((optim_rank[:,:,iapl,0,:].flatten(),optim_rank[:,:,iapl,1,:].flatten())) 
    datatoplot = np.column_stack((np.nanmean(optim_rank[:,:,iapl,0,:],axis=(0,1)).flatten(),np.nanmean(optim_rank[:,:,iapl,1,:],axis=(0,1)).flatten())) 
    datatoplot = datatoplot[~np.isnan(datatoplot).any(axis=1)]

    ax.scatter(np.zeros(len(datatoplot))+np.random.randn(len(datatoplot))*0.05,datatoplot[:,0],color='k',marker='o',s=10)
    ax.errorbar(0.2,np.nanmean(datatoplot[:,0]),np.nanstd(datatoplot[:,0])/np.sqrt(nSessions),color='g',marker='o',zorder=10)

    ax.scatter(np.ones(len(datatoplot))+np.random.randn(len(datatoplot))*0.05,datatoplot[:,1],color='k',marker='o',s=10)
    ax.errorbar(1.2,np.nanmean(datatoplot[:,1]),np.nanstd(datatoplot[:,1])/np.sqrt(nSessions),color='r',marker='o',zorder=10)

    ax.set_xticks([0,1],['Match','Mismatch'])
    ax.set_title('%s' % arealabelpair,fontsize=8)
# ax.set_ylim([0,0.25])
axes[0].set_ylabel('Rank')
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'RRR_Rank_MatchMismatch_RF_%dsessions' % (nSessions),formats = ['png'])


#%%  Show percentage difference between match and mismatch:

fig,axes = plt.subplots(1,1,figsize=(2,3),sharey=True,sharex=True)

datatoplot = np.column_stack([R2_cv[:,:,iapl,0,:].flatten() / R2_cv[:,:,iapl,1,:].flatten() for iapl in range(narealabelpairs)]) 
ax = axes

for iapl, arealabelpair in enumerate(arealabelpairs):
    # ax.errorbar(iapl+0.5,np.nanmean(datatoplot[iapl,:]),np.nanstd(datatoplot[iapl,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)
    ax.errorbar(iapl,np.nanmean(datatoplot[:,iapl]),np.nanstd(datatoplot[:,iapl])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)
ax.axhline(1, color='k', linewidth=0.5, linestyle='--')
ax.set_ylabel('Ratio R2 (match/mismatch)')
ax_nticks(ax,4)
sns.despine(top=True,right=True,offset=3)	
ax.set_xticks(range(narealabelpairs))
ax.set_xticklabels(arealabelpairs,rotation=45,ha='right',fontsize=8)
my_savefig(fig,savedir,'R2_Ratio_MatchMismatch_RF_%dsessions' % (nSessions),formats = ['png'])


#%% 


#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

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

method              = 'CCA'
# method              = 'RRR'

n_components        = 5
lambda_reg          = 1

absolute            = False

for ises,ses in tqdm(enumerate(sessions),desc='Computing CCA maps',total=nSessions):
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




