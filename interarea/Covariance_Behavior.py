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

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\Behavior\\')

#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR
# session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                # ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% 
areas = ['V1','PM']
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)

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
binsize     = 0.2
calciumversion = 'dF'
# calciumversion = 'deconv'
# vidfields = np.array(['videoPC_%d'%i for i in range(30)])
vidfields = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
    [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'videodata')

#%% #############################################################################





#%% Show example covariance matrix predicted by behavior:  
ises        = 0 #which session to compute covariance matrix for
stim        = 0 #which stimulus to compute covariance matrix for

idx_T               = sessions[ises].trialdata['stimCond']==stim
idx_N               = np.ones(len(sessions[ises].celldata),dtype=bool)
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

#on residual tensor during the response:
Y                   = sessions[ises].tensor[np.ix_(idx_N,idx_T,idx_resp)]
Y                   -= np.mean(Y,axis=1,keepdims=True)
Y                   = Y.reshape(len(idx_N),-1).T
Y                   = zscore(Y,axis=0,nan_policy='omit')  #Z score activity for each neuron

#Get behavioral matrix: 
B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
                        sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
B                   = B.reshape(np.shape(B)[0],-1).T
B                   = zscore(B,axis=0,nan_policy='omit')

si                  = SimpleImputer()
Y                   = si.fit_transform(Y)
B                   = si.fit_transform(B)

#Reduced rank regression: 
B_hat               = LM(Y,B,lam=0)
Y_hat               = B @ B_hat

# decomposing and low rank approximation of Y_hat
rank                = 5
U, s, V             = svds(Y_hat,k=rank)
U, s, V             = U[:, ::-1], s[::-1], V[::-1, :]

S                   = linalg.diagsvd(s,U.shape[0],s.shape[0])

Y_cov               = np.cov(Y.T)
np.fill_diagonal(Y_cov,np.nan)

Y_hat_rr            = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]
Y_cov_rrr           = np.cov(Y_hat_rr.T)
np.fill_diagonal(Y_cov_rrr,np.nan)

#%% Plot: 
vmin,vmax       = np.nanpercentile(Y_cov,5),np.nanpercentile(Y_cov,95)
arealabeled     = np.array(['V1unl','V1lab','PMunl','PMlab'])

idx_sort       = np.argsort(sessions[ises].celldata['arealabel'],order=arealabeled)
al_sorted      = np.sort(np.array(sessions[ises].celldata['arealabel']))[::-1]

Y_cov_sort      = copy.deepcopy(Y_cov)
Y_cov_sort      = Y_cov_sort[idx_sort,:]
Y_cov_sort      = Y_cov_sort[:,idx_sort]

Y_cov_rrr_sort  = copy.deepcopy(Y_cov_rrr)
Y_cov_rrr_sort  = Y_cov_rrr_sort[idx_sort,:]
Y_cov_rrr_sort  = Y_cov_rrr_sort[:,idx_sort]

fig,ax = plt.subplots(1,2,figsize=(6,3))
ax[0].imshow(Y_cov,vmin=vmin,vmax=vmax,cmap='magma')
ax[0].set_title('Covariance\n(original)')
ax[0].set_yticks([])
for ial,arealabel in enumerate(arealabeled):
    start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
    ax[0].plot([-5,-5],[start,stop],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
    ax[0].text(-85,(start+stop)/2,arealabel,fontsize=9,color=get_clr_area_labeled([arealabel]),
               rotation=45,ha='center',va='center')
for ial,arealabel in enumerate(arealabeled):
    start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
    ax[0].plot([start,stop],[-5,-5],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
    ax[0].text((start+stop)/2,-85,arealabel,fontsize=9,color=get_clr_area_labeled([arealabel]),
               rotation=45,ha='center',va='center')
ax[0].set_xticks([0,np.shape(Y_cov)[0]-1])

ax[1].imshow(Y_cov_rrr,vmin=vmin,vmax=vmax,cmap='magma')
ax[1].set_title('Covariance\n(predicted from behavior)')
ax[1].set_xticks([0,np.shape(Y_cov)[0]-1])
for ial,arealabel in enumerate(arealabeled):
    start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
    ax[1].plot([-5,-5],[start,stop],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
    ax[1].text(-85,(start+stop)/2,arealabel,fontsize=9,color=get_clr_area_labeled([arealabel]),
               rotation=45,ha='center',va='center')
for ial,arealabel in enumerate(arealabeled):
    start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
    ax[1].plot([start,stop],[-5,-5],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
    ax[1].text((start+stop)/2,-85,arealabel,fontsize=9,color=get_clr_area_labeled([arealabel]),
               rotation=45,ha='center',va='center')
for axi in ax:
    # w = ax.get_xaxis()
    # w.set_visible(False)
    # axi.axis["left"].set_visible(False)
    # axi.axis["top"].set_visible(False)
    # axi.axis["right"].set_visible(False)
    axi.set_axis_off()
plt.tight_layout()
my_savefig(fig,savedir,'CovarianceMatrix_V1PM_%s' % sessions[ises].session_id,formats=['png'])




#%% Compute the variance and covariance explained by the behavior: 

# Variance: 
arealabeled         = np.array(['V1unl','V1lab','PMunl','PMlab'])
clrs_arealabels     = get_clr_area_labeled(arealabeled)
narealabels         = len(arealabeled)

# Covariance:
arealabelpairs  = np.array(['V1unl-V1unl',
                    'V1unl-V1lab',
                    'V1lab-V1lab',
                    'PMunl-PMunl',
                    'PMunl-PMlab',
                    'PMlab-PMlab',
                    'V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab'])
clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

#Parameters:
lam                 = 0
nranks              = 20
kfold               = 5
maxnoiselevel       = 20
nStim               = 16
filter_nearby       = True

idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
ntimebins           = len(idx_resp)

#Explained (co)variance
EV_pops             = np.full((narealabels,nranks,nStim,nSessions,kfold),np.nan)
EC_poppairs         = np.full((narealabelpairs,nranks,nStim,nSessions,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Covariance by behavior, fitting models'):
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Covariance by behavior, fitting models'):

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        idx_T               = ses.trialdata['stimCond']==stim

        idx_N               = np.ones(len(ses.celldata),dtype=bool)

        #on residual tensor during the response:
        Y                   = sessions[ises].tensor[np.ix_(idx_N,idx_T,idx_resp)]
        Y                   -= np.mean(Y,axis=1,keepdims=True)
        Y                   = Y.reshape(len(idx_N),-1).T
        Y                   = zscore(Y,axis=0,nan_policy='omit')  #Z score activity for each neuron

        #Get behavioral matrix: 
        B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
                                sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
        B                   = B.reshape(np.shape(B)[0],-1).T
        B                   = zscore(B,axis=0,nan_policy='omit')

        si      = SimpleImputer()
        Y       = si.fit_transform(Y)
        B       = si.fit_transform(B)

        #Reduced rank regression: 
        B_hat           = LM(Y,B,lam=lam)

        Y_hat           = B @ B_hat

        # decomposing and low rank approximation of Y_hat
        U, s, V = svds(Y_hat,k=nranks)
        U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

        S = linalg.diagsvd(s,U.shape[0],s.shape[0])

        Y_cov = np.cov(Y.T)

        for irank,rank in enumerate(range(1,nranks+1)):
            #construct low rank subspace prediction
            Y_hat_rr       = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]

            Y_out           = Y - Y_hat_rr #subtract prediction

            Y_cov_rrr = np.cov(Y_hat_rr.T)

            for ial,al in enumerate(arealabeled):
                idx_N           = np.where(np.all((ses.celldata['arealabel']==al,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        idx_nearby),axis=0))[0]
                
                EV_pops[ial,irank,istim,ises,0] = EV(Y[:,idx_N],Y_hat_rr[:,idx_N])
            
            for ialp,arealabelpair in enumerate(arealabelpairs):

                alx,aly             = arealabelpair.split('-')

                idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        idx_nearby),axis=0))[0]
                idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        idx_nearby
                                        ),axis=0))[0]
                
                EC_poppairs[ialp,irank,istim,ises,0] = EV(Y_cov[np.ix_(idx_areax,idx_areay)],Y_cov_rrr[np.ix_(idx_areax,idx_areay)])


#%% Plotting:
fig,axes = plt.subplots(1,3,figsize=(7.5,2.5))
ax = axes[0]
handles = []

for ial, arealabel in enumerate(arealabeled):
    ialdata = np.reshape(EV_pops[ial,:,:,:],(nranks,-1))
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(range(nranks),ialdata.T,error='sem',color=clrs_arealabels[ial],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=list(arealabeled),loc='lower right',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(np.arange(0,nranks+5,5))
ax.set_xticklabels(np.arange(0,nranks+5,5)+1)
ax.set_xlabel('Rank')
ax.set_ylabel('Variance explained')
ax.set_title('Variance explained',fontsize=10)

idx = [0,3,6]
ax = axes[1]
handles = []
for ialp in idx:
    ialpdata = np.reshape(EC_poppairs[ialp,:,:,:],(nranks,-1))
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(range(nranks),ialpdata.T,error='sem',color=clrs_arealabelpairs[ialp],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=list(arealabelpairs[idx]),loc='lower right',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(np.arange(0,nranks+5,5))
ax.set_xticklabels(np.arange(0,nranks+5,5)+1)
ax.set_xlabel('Rank')
ax.set_ylabel('Covariance explained')
ax.set_title('Covariance explained',fontsize=10)

idx = [6,7,8,9]
ax = axes[2]
handles = []
for ialp in idx:
    ialpdata = np.reshape(EC_poppairs[ialp,:,:,:],(nranks,-1))
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(range(nranks),ialpdata.T,error='sem',color=clrs_arealabelpairs[ialp],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=list(arealabelpairs[idx]),loc='lower right',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(np.arange(0,nranks+5,5))
ax.set_xticklabels(np.arange(0,nranks+5,5)+1)
ax.set_xlabel('Rank')
ax.set_ylabel('Covariance explained')
ax.set_title('Labeled covariance explained',fontsize=10)
sns.despine(fig,top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'CoVarianceExplained_V1PM_%dsessions' % nSessions,formats=['png'])

#%% 


