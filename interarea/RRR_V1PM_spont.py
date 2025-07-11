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
from scipy.stats import zscore
from skimage.measure import block_reduce
from tqdm import tqdm

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *
from utils.RRRlib import *

#%% 
protocol = 'SP'
areas = ['V1','PM']
nareas = len(areas)
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross')

#%% Load data: 
# session_list        = np.array([['LPE12223_2024_06_10'], #GR
#                                 ['LPE10919_2023_11_06']]) #GR
# sessions,nSessions   = filter_sessions(protocols = 'SP',only_session_id=session_list)
sessions,nSessions   = filter_sessions(protocols = ['SP'],filter_areas=areas)

# #%% Remove two sessions with too much drift in them:
# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20']))[0]
# sessions            = [sessions[i] for i in sessions_in_list]
# nSessions           = len(sessions)

# #%%  Load data properly:        
# # calciumversion = 'deconv'
# calciumversion = 'dF'
# for ises in range(nSessions):
#     sessions[ises].load_data(load_calciumdata=True)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF')
    
#%% Do RRR in FF and FB direction and compare performance:
nsampleneurons  = 250
nranks          = 25
nmodelfits      = 5 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions,2),np.nan)
optim_rank      = np.full((nSessions,2),np.nan)
R2_ranks        = np.full((nSessions,2,nranks,nmodelfits,kfold),np.nan)

temporalbin     = 0.5
nbin            = int(temporalbin * sessions[0].sessiondata['fs'][0])

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].calciumdata.iloc[:,idx_areax].to_numpy()
    Y                   = sessions[ises].calciumdata.iloc[:,idx_areay].to_numpy()

    X_r     = block_reduce(X, block_size=(nbin,1), func=np.mean, cval=np.mean(X))
    Y_r     = block_reduce(Y, block_size=(nbin,1), func=np.mean, cval=np.mean(Y))

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:
        R2_cv[ises,0],optim_rank[ises,0],R2_ranks[ises,0,:,:,:]      = RRR_wrapper(Y_r, X_r, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
    
        R2_cv[ises,1],optim_rank[ises,1],R2_ranks[ises,1,:,:,:]      = RRR_wrapper(X_r, Y_r, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot the performance across sessions as a function of rank: 
clrs_areapairs = get_clr_area_pairs(['V1-PM','PM-V1'])
datatoplot = np.nanmean(R2_ranks,axis=(3,4))
axlim = np.nanmax(np.nanmean(datatoplot,axis=0))*1.2

fig,axes = plt.subplots(1,3,figsize=(6.5,2.2))
ax = axes[0]
handles = []
handles.append(shaded_error(np.arange(nranks),datatoplot[:,0,:],color=clrs_areapairs[0],error='sem',ax=ax))
handles.append(shaded_error(np.arange(nranks),datatoplot[:,1,:],color=clrs_areapairs[1],error='sem',ax=ax))

ax.legend(handles,['V1->PM','PM->V1'],frameon=False,fontsize=8,loc='lower right')
ax.set_xlabel('Rank')
ax.set_ylabel('R2')
# ax.set_yticks([0,0.05,0.1])
ax.set_ylim([0,axlim])
ax.set_xlim([0,nranks])
ax_nticks(ax,5)

ax = axes[1]
axlim = my_ceil(np.nanmax(datatoplot)*1.1,1)
ax.scatter(R2_cv[:,0],R2_cv[:,1],color='k',s=10)
ax.set_xlabel('V1->PM')
ax.set_ylabel('PM->V1')
ax.plot([0,1],[0,1],color='k',linestyle='--',linewidth=0.5)
ax.set_xlim([0,axlim])
ax.set_ylim([0,axlim])
ax_nticks(ax,3)
t,p = ttest_rel(R2_cv[:,0],R2_cv[:,1],nan_policy='omit')
print('Paired t-test (R2): p=%.3f' % p)
if p<0.05:
    ax.text(0.6,0.2,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='red')
else: 
    ax.text(0.6,0.2,'p=%.3f' % p,transform=ax.transAxes,ha='center',va='center',fontsize=12,color='k')

ax = axes[2]
axlim = my_ceil(np.nanmax(optim_rank)*1.2)
ax.scatter(optim_rank[:,0],optim_rank[:,1],color='k',s=10)
ax.plot([0,20],[0,20],color='k',linestyle='--',linewidth=0.5)
ax.set_xlabel('V1->PM')
ax.set_ylabel('PM->V1')
ax.set_xlim([0,axlim])
ax.set_ylim([0,axlim])
ax_nticks(ax,3)
t,p = ttest_rel(optim_rank[:,0],optim_rank[:,1],nan_policy='omit')
print('Paired t-test (R2): p=%.3f' % p)
if p<0.05:
    ax.text(0.6,0.2,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='red')
else: 
    ax.text(0.6,0.2,'p=n.s.',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='k')
    
# ax.set_ylim([0,0.3])
plt.tight_layout()
sns.despine(top=True,right=True,offset=3)
# my_savefig(fig,savedir,'RRR_R2_acrossranks_V1PM_SPONT_%dsessions_%dsec.png' % (nSessions,temporalbin),formats=['png'])

#%% 

######  ######  ######     #          #    ######     #     #  #####     #     # #     # #       
#     # #     # #     #    #         # #   #     #    #     # #     #    #     # ##    # #       
#     # #     # #     #    #        #   #  #     #    #     # #          #     # # #   # #       
######  ######  ######     #       #     # ######     #     #  #####     #     # #  #  # #       
#   #   #   #   #   #      #       ####### #     #     #   #        #    #     # #   # # #       
#    #  #    #  #    #     #       #     # #     #      # #   #     #    #     # #    ## #       
#     # #     # #     #    ####### #     # ######        #     #####      #####  #     # ####### 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\Labeling')


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons

# arealabelpairs  = ['V1unl-V1unl',
#                     'V1unl-V1lab',
#                     'V1lab-V1lab',
#                     'PMunl-PMunl',
#                     'PMunl-PMlab',
#                     'PMlab-PMlab',
#                     'V1unl-PMunl',
#                     'V1unl-PMlab',
#                     'V1lab-PMunl',
#                     'V1lab-PMlab',
#                     'PMunl-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1unl',
#                     'PMlab-V1lab']

# arealabelpairs  = ['V1unl-PMunl',
#                     'V1lab-PMunl',
#                     'V1unl-PMlab',
#                     'V1lab-PMlab',
#                     'PMunl-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1unl',
#                     'PMlab-V1lab']

arealabelpairs  = [
                    'V1unl-PMunl', #Feedforward pairs
                    'V1lab-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMlab',

                    'PMunl-V1unl', #Feedback pairs
                    'PMlab-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1lab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nranks              = 20
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5
maxnoiselevel       = 20

R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions),np.nan)
R2_ranks            = np.full((narealabelpairs,nSessions,nranks,nmodelfits,kfold),np.nan)

filter_nearby       = True
# filter_nearby       = False

minsampleneurons    = 20

regress_behavior = False
vidfields       = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)
si              = SimpleImputer()

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):

    allpops             = np.array([i.split('-') for i in arealabelpairs]).flatten()
    nsampleneurons      = np.min([np.sum((ses.celldata['arealabel']==i) & (ses.celldata['noise_level']<maxnoiselevel)) for i in allpops])
    #take the smallest sample size

    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    neuraldata          = zscore(sessions[ises].calciumdata.to_numpy(),axis=0,nan_policy='omit')

    if regress_behavior:
        B = np.empty((len(sessions[ises].ts_F),len(vidfields)+1))
        for ifie,field in enumerate(vidfields):
            B[:,ifie] = np.interp(x=sessions[ises].ts_F, xp=sessions[ises].videodata['ts'],
                                            # fp=sessions[ises].videodata['runspeed'])
                                            fp=sessions[ises].videodata[field])
        B[:,-1] = np.interp(x=sessions[ises].ts_F, xp=sessions[ises].behaviordata['ts'],
                                            fp=sessions[ises].behaviordata['runspeed'])

        B                   = si.fit_transform(B)
        B                   = zscore(B,axis=0,nan_policy='omit')

        #Reduced rank regression: 
        B_hat               = LM(neuraldata,B,lam=0)
        Y_hat               = B @ B_hat

        # decomposing and low rank approximation of Y_hat
        rank                = 3
        U, s, V             = svds(Y_hat,k=rank)
        U, s, V             = U[:, ::-1], s[::-1], V[::-1, :]

        S                   = linalg.diagsvd(s,U.shape[0],s.shape[0])

        Y_hat_rr            = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]

        # plt.imshow(neuraldata,vmin=-0.5,vmax=0.5)
        # plt.imshow(Y_hat,vmin=-0.5,vmax=0.5)
        # plt.imshow(Y_hat_rr,vmin=-0.5,vmax=0.5)

        neuraldata          -= Y_hat_rr

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=30)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    
        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue
    
        X                   = neuraldata[:,idx_areax]
        Y                   = neuraldata[:,idx_areay]

        R2_cv[iapl,ises],optim_rank[iapl,ises],R2_ranks[iapl,ises,:,:,:]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
        #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS

#%% Plot the R2 performance and number of dimensions per area pair
# fig         = plot_RRR_R2_arealabels(R2_cv,optim_rank,R2_ranks,arealabelpairs,clrs_arealabelpairs)
# my_savefig(fig,savedir,'RRR_cvR2_RegressOutBehavior_V1PM_LabUnl_%dsessions' % nSessions)
# fig         = plot_RRR_R2_arealabels(R2_cv[:4],optim_rank[:4],R2_ranks[:4],arealabelpairs[:4],clrs_arealabelpairs[:4])
# my_savefig(fig,savedir,'RRR_cvR2_V1PM_LabUnl_%dsessions_spont' % nSessions)
# fig         = plot_RRR_R2_arealabels(R2_cv[4:],optim_rank[4:],R2_ranks[4:],arealabelpairs[4:],clrs_arealabelpairs[4:])
# my_savefig(fig,savedir,'RRR_cvR2_PMV1_LabUnl_%dsessions_spont' % nSessions)

#%% 
normalize   = False
#Same target population:
for idx in np.array([[0,1],[2,3],[4,5],[6,7]]):
    clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
    # fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs,normalize=normalize)
    fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs,normalize=normalize)
    my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],protocol,nSessions))

for idx in np.array([[0,1],[2,3],[4,5],[6,7]]):
    mean,sd = np.nanmean(R2_cv[idx[1]] / R2_cv[idx[0]])*100-100,np.nanstd(R2_cv[idx[1]] / R2_cv[idx[0]])
    print('%s vs %s: %2.1f %% +/- %2.1f' % (arealabelpairs[idx[1]],arealabelpairs[idx[0]],mean,sd))

#Different target population:
for idx in np.array([[0,3],[4,7]]):
    clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
    fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs,normalize=normalize)
    my_savefig(fig,savedir,'RRR_cvR2_diffTarget_%s_%s_%dsessions' % (arealabelpairs[idx[1]],protocol,nSessions))

for idx in np.array([[0,3],[4,7]]):
    mean,sd = np.nanmean(R2_cv[idx[1]] / R2_cv[idx[0]])*100-100,np.nanstd(R2_cv[idx[1]] / R2_cv[idx[0]])
    print('%s vs %s: %2.1f %% +/- %2.1f' % (arealabelpairs[idx[1]],arealabelpairs[idx[0]],mean,sd))
