# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from skimage.measure import block_reduce
from tqdm import tqdm

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *
from utils.RRRlib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\TimeResolution\\')

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% Remove two sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 1.9     #post s

# calciumversion = 'dF'
calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,load_videodata=False,
                                calciumversion=calciumversion)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='nearby')
    delattr(sessions[ises],'calciumdata')

#%% #############################################################################

# RRR


#%% Time res wrapper function:
nsampleneurons      = 250
nranks              = 25
nmodelfits          = 50 #number of times new neurons are resampled 
kfold               = 5
filter_nearby       = False

arealabelpairs      = ['V1-PM',
                        'PM-V1',
                        ]

clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

idx_time        = (t_axis>=-1) & (t_axis<=2) #axis=1
binframes       = np.arange(1,np.sum(idx_time),2)
# binframes       = np.array([1,5])

nbinframes      = len(binframes)
R2_cv           = np.full((nSessions,narealabelpairs,nbinframes),np.nan)
optim_rank      = np.full((nSessions,narealabelpairs,nbinframes),np.nan)
R2_ranks        = np.full((nSessions,narealabelpairs,nbinframes,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        Xfull = ses.tensor[np.ix_(idx_areax,idx_T,idx_time)]
        Yfull = ses.tensor[np.ix_(idx_areay,idx_T,idx_time)]
        for ibin,nbin in enumerate(binframes):
            X_r     = block_reduce(Xfull, block_size=(1,1,nbin), func=np.mean, cval=np.mean(Xfull)).reshape(Xfull.shape[0],-1).T
            Y_r     = block_reduce(Yfull, block_size=(1,1,nbin), func=np.mean, cval=np.mean(Yfull)).reshape(Yfull.shape[0],-1).T

            R2_cv[ises,iapl,ibin],optim_rank[ises,iapl,ibin],R2_ranks[ises,iapl,ibin,:,:,:]      = RRR_wrapper(Y_r, X_r, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))

#%% Plotting:
fig,axes = plt.subplots(1,2,figsize=(5.5,2.5))
ax = axes[0]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs[:4]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(binframes,R2_cv[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=arealabelpairs,loc='upper left',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(binframes)
ax.set_xticklabels(binframes)
ax.set_xlabel('Time binning (frames)')
ax.set_ylabel('R2')
ax_nticks(ax,3)

ax = axes[1]
for iapl, arealabelpair in enumerate(arealabelpairs[:4]):
    shaded_error(binframes,optim_rank[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax)
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.set_xticks(binframes)
ax.set_xticklabels(binframes)
ax.set_xlabel('Time binning (frames)')
ax.set_ylabel('Optimal rank')
ax_nticks(ax,3)

sns.despine(fig,top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'RRR_V1PM_timeres')

#%% # Now with labeled cells:








#%% Time res wrapper function:
nsampleneurons      = 20
nranks              = 25
nmodelfits          = 50 #number of times new neurons are resampled 
kfold               = 5
filter_nearby       = True

arealabelpairs      = ['V1unl-PMunl',
                        'V1unl-PMlab',
                        'V1lab-PMunl',
                        'V1lab-PMlab',
                        'PMunl-V1unl',
                        'PMunl-V1lab',
                        'PMlab-V1unl',
                        'PMlab-V1lab'
                        ]

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

idx_time        = (t_axis>=-1) & (t_axis<=2) #axis=1
binframes       = np.arange(1,np.sum(idx_time),2)
nbinframes      = len(binframes)
R2_cv           = np.full((nSessions,narealabelpairs,nbinframes),np.nan)
optim_rank      = np.full((nSessions,narealabelpairs,nbinframes),np.nan)
R2_ranks        = np.full((nSessions,narealabelpairs,nbinframes,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        Xfull = ses.tensor[np.ix_(idx_areax,idx_T,idx_time)]
        Yfull = ses.tensor[np.ix_(idx_areay,idx_T,idx_time)]
        for ibin,nbin in enumerate(binframes):
            X_r     = block_reduce(Xfull, block_size=(1,1,nbin), func=np.mean, cval=np.mean(Xfull)).reshape(Xfull.shape[0],-1).T
            Y_r     = block_reduce(Yfull, block_size=(1,1,nbin), func=np.mean, cval=np.mean(Yfull)).reshape(Yfull.shape[0],-1).T

            R2_cv[ises,iapl,ibin],optim_rank[ises,iapl,ibin],R2_ranks[ises,iapl,ibin,:,:,:]      = RRR_wrapper(Y_r, X_r, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))


#%% Plotting:
fig,axes = plt.subplots(2,2,figsize=(5.5,5.5))
ax = axes[0,0]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs[:4]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(binframes,R2_cv[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=arealabelpairs,loc='upper left',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(binframes)
ax.set_xticklabels(binframes)
ax.set_xlabel('Time binning (frames)')
ax.set_ylabel('R2')
ax_nticks(ax,3)

ax = axes[0,1]
for iapl, arealabelpair in enumerate(arealabelpairs[:4]):
    shaded_error(binframes,optim_rank[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax)
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.set_xticks(binframes)
ax.set_xticklabels(binframes)
ax.set_xlabel('Time binning (frames)')
ax.set_ylabel('Optimal rank')
ax_nticks(ax,3)

ax = axes[1,0]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs[4:]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(binframes,R2_cv[:,iapl+4,:],error='sem',color=clrs_arealabelpairs[iapl+4],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=arealabelpairs[4:],loc='upper left',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(binframes)
ax.set_xticklabels(binframes)
ax.set_xlabel('Time binning (frames)')
ax.set_ylabel('R2')
ax_nticks(ax,3)

ax = axes[1,1]
for iapl, arealabelpair in enumerate(arealabelpairs[4:]):
    shaded_error(binframes,optim_rank[:,iapl+4,:],error='sem',color=clrs_arealabelpairs[iapl+4],alpha=0.3,ax=ax)
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.set_xticks(binframes)
ax.set_xticklabels(binframes)
ax.set_xlabel('Time binning (frames)')
ax.set_ylabel('Optimal rank')
ax_nticks(ax,3)

sns.despine(fig=fig, top=True, right=True, offset = 3)
plt.tight_layout()
my_savefig(fig,savedir,'RRR_V1PM_labeled_timeres')







#%% RRR across time relative to stimulus onset:
nranks              = 20
nmodelfits          = 50 #number of times new neurons are resampled 
kfold               = 5
lam                 = 0

arealabelpairs      = ['V1unl-PMunl',
                        'V1unl-PMlab',
                        'V1lab-PMunl',
                        'V1lab-PMlab',
                        'PMunl-V1unl',
                        'PMunl-V1lab',
                        'PMlab-V1unl',
                        'PMlab-V1lab'
                        ]
nsampleneurons      = 25
filter_nearby       = True

# arealabelpairs      = ['V1unl-PMunl',
#                         'PMunl-V1unl',
#                         ]
# nsampleneurons      = 200
# filter_nearby       = False

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

ntimebins           = len(t_axis)
R2_cv               = np.full((nSessions,narealabelpairs,ntimebins),np.nan)
optim_rank          = np.full((nSessions,narealabelpairs,ntimebins),np.nan)
R2_ranks            = np.full((nSessions,narealabelpairs,ntimebins,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)

    for iapl, arealabelpair in enumerate(arealabelpairs):
        alx,aly = arealabelpair.split('-')
        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        for ibin in range(ntimebins):
            X = ses.tensor[np.ix_(idx_areax,idx_T,[ibin])].squeeze().T
            Y = ses.tensor[np.ix_(idx_areay,idx_T,[ibin])].squeeze().T

            R2_cv[ises,iapl,ibin],optim_rank[ises,iapl,ibin],R2_ranks[ises,iapl,ibin,:,:,:]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))



#%% RRR across time relative to stimulus onset:
nranks              = 20
nmodelfits          = 1 #number of times new neurons are resampled 
kfold               = 5
lam                 = 500

arealabelpairs      = ['V1unl-PMunl',
                        'V1unl-PMlab',
                        'V1lab-PMunl',
                        'V1lab-PMlab',
                        'PMunl-V1unl',
                        'PMunl-V1lab',
                        'PMlab-V1unl',
                        'PMlab-V1lab'
                        ]
nsampleneurons      = 25
filter_nearby       = True

arealabelpairs      = ['V1unl-PMunl',
                        'PMunl-V1unl',
                        ]
nsampleneurons      = 50
filter_nearby       = False

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

oris                = np.unique(np.mod(sessions[0].trialdata['Orientation'],180))
nOris               = len(oris)
ntimebins           = len(t_axis)
R2_cv               = np.full((nSessions,nOris,narealabelpairs,ntimebins),np.nan)
optim_rank          = np.full((nSessions,nOris,narealabelpairs,ntimebins),np.nan)
R2_ranks            = np.full((nSessions,nOris,narealabelpairs,ntimebins,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    ses.trialdata['Orientation'] = np.mod(ses.trialdata['Orientation'],180)
    for iapl, arealabelpair in enumerate(arealabelpairs):
        alx,aly = arealabelpair.split('-')
        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        for iori,ori in enumerate(oris):
            idx_T               = ses.trialdata['Orientation']==ori
            for ibin in range(ntimebins):
                X = ses.tensor[np.ix_(idx_areax,idx_T,[ibin])].squeeze().T
                Y = ses.tensor[np.ix_(idx_areay,idx_T,[ibin])].squeeze().T

                R2_cv[ises,iori,iapl,ibin],optim_rank[ises,iori,iapl,ibin],R2_ranks[ises,iori,iapl,ibin,:,:,:]      = RRR_wrapper(
                                                                        Y, X, nN=nsampleneurons,lam=lam,
                                                                       nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))
R2_cv               = np.nanmean(R2_cv,axis=1)
optim_rank          = np.nanmean(optim_rank,axis=1)
R2_ranks            = np.nanmean(R2_ranks,axis=1)

#%% Plotting:
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(2,2,figsize=(5.5,5.5))
ax = axes[0,0]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs[:4]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,R2_cv[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealabelpairs,loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('R2')

ax = axes[1,0]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs[4:]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,R2_cv[:,iapl+4,:],error='sem',color=clrs_arealabelpairs[iapl+4],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealabelpairs[4:],loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('R2')

ax = axes[0,1]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs[:4]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealabelpairs,loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Optimal rank')

ax = axes[1,1]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs[4:]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl+4],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealabelpairs[4:],loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Optimal rank')

sns.despine(fig=fig, top=True, right=True, offset = 3)
plt.tight_layout()
# my_savefig(fig,savedir,'RRR_V1PM_labeled_timeres')
# my_savefig(fig,savedir,'RRR_V1PM_labeled_GR_across_time',formats=['png'])
my_savefig(fig,savedir,'RRR_V1PM_labeled_GR_perOri_across_time',formats=['png'])


#%% Plotting:
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(1,2,figsize=(6,2.5))
ax = axes[0]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,R2_cv[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealabelpairs,loc='upper left',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('R2')

ax = axes[1]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealabelpairs,loc='upper left',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Optimal rank')

sns.despine(fig=fig, top=True, right=True, offset = 3)
plt.tight_layout()
my_savefig(fig,savedir,'RRR_V1PM_GR_across_time',formats=['png'])


#%%  #assign arealayerlabel
from preprocessing.preprocesslib import assign_layer

for ises in range(nSessions):   
    sessions[ises].celldata = assign_layer(sessions[ises].celldata)

#%%
for ses in sessions:
    ses.celldata['arealayer'] = ses.celldata['roi_name'] + ses.celldata['layer']
    #assign arealayerlabel
    ses.celldata['arealayerlabel'] = ses.celldata['arealabel'] + ses.celldata['layer'] 

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealayerpairs  = ['V1L2/3-PML2/3',
                   'V1L2/3-PML5',
                   'PML2/3-V1L2/3',
                   'PML5-V1L2/3',
                   'PML2/3-PML5',
                   'PML5-PML2/3'                   
                   ]

arealayerpairs  = ['V1unlL2/3-PMunlL2/3',
                   'V1labL2/3-PMunlL2/3',
                    'V1unlL2/3-PMlabL2/3',
                   'V1labL2/3-PMlabL2/3',
                   ]

# arealayerlabels = ['V1unlL2/3',
#                     'V1labL2/3',
#                     'PMunlL2/3',
#                     'PMlabL2/3',
#                     'PMunlL5',
#                     'PMlabL5',
#                    ]



#%% RRR across time relative to stimulus onset:
nranks              = 20
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
lam                 = 0

arealayerpairs  = ['V1unlL2/3-PMunlL2/3',
                   'V1labL2/3-PMunlL2/3',
                    'V1unlL2/3-PMlabL2/3',
                   'V1labL2/3-PMlabL2/3',
                   'V1unlL2/3-PMunlL5',
                   'V1labL2/3-PMunlL5',
                    'V1unlL2/3-PMlabL5',
                   'V1labL2/3-PMlabL5',
                   ]

arealayerpairs  = ['PMunlL2/3-V1unlL2/3',
                   'PMunlL2/3-V1labL2/3',
                    'PMlabL2/3-V1unlL2/3',
                   'PMlabL2/3-V1labL2/3',
                   'PMunlL5-V1unlL2/3',
                   'PMunlL5-V1labL2/3',
                    'PMlabL5-V1unlL2/3',
                   'PMlabL5-V1labL2/3',
                   ]

narealayerpairs     = len(arealayerpairs)

nsampleneurons      = 20
filter_nearby       = True

# arealabelpairs      = ['V1unl-PMunl',
#                         'PMunl-V1unl',
#                         ]
# nsampleneurons      = 200
# filter_nearby       = False

# clrs_arealabelpairs = get_clr_area_labelpairs(arealayerpairs)
clrs_arealayerlabels = sns.color_palette('tab10',narealayerpairs)


ntimebins           = len(t_axis)
R2_cv               = np.full((nSessions,narealayerpairs,ntimebins),np.nan)
optim_rank          = np.full((nSessions,narealayerpairs,ntimebins),np.nan)
R2_ranks            = np.full((nSessions,narealayerpairs,ntimebins,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)

    for iapl, arealayerpair in enumerate(arealayerpairs):
        alx,aly = arealayerpair.split('-')
        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealayerlabel']==alx,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealayerlabel']==aly,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        for ibin in range(ntimebins):
            X = ses.tensor[np.ix_(idx_areax,idx_T,[ibin])].squeeze().T
            Y = ses.tensor[np.ix_(idx_areay,idx_T,[ibin])].squeeze().T

            R2_cv[ises,iapl,ibin],optim_rank[ises,iapl,ibin],R2_ranks[ises,iapl,ibin,:,:,:]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))

#%% Plotting:
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(2,2,figsize=(5.5,5.5))
ax = axes[0,0]
handles = []
for iapl, arealabelpair in enumerate(arealayerpairs[:4]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,R2_cv[:,iapl,:],error='sem',color=clrs_arealayerlabels[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealayerpairs,loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('R2')

ax = axes[1,0]
handles = []
for iapl, arealabelpair in enumerate(arealayerpairs[4:]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,R2_cv[:,iapl+4,:],error='sem',color=clrs_arealayerlabels[iapl+4],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealayerpairs[4:],loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('R2')

ax = axes[0,1]
handles = []
for iapl, arealabelpair in enumerate(arealayerpairs[:4]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealayerlabels[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealayerpairs,loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Optimal rank')

ax = axes[1,1]
handles = []
for iapl, arealabelpair in enumerate(arealayerpairs[4:]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealayerlabels[iapl+4],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealayerpairs[4:],loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Optimal rank')

sns.despine(fig=fig, top=True, right=True, offset = 3)
plt.tight_layout()
# my_savefig(fig,savedir,'RRR_V1PM_labeled_timeres')
# my_savefig(fig,savedir,'RRR_V1PM_labeled_GR_across_time',formats=['png'])
my_savefig(fig,savedir,'RRR_V1PM_layer_labeled_GR_across_time',formats=['png'])
