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

#%% 
areas = ['V1','PM']
nareas = len(areas)
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross')

#%% Load data: 
# session_list        = np.array([['LPE12223_2024_06_10'], #GR
#                                 ['LPE10919_2023_11_06']]) #GR
# sessions,nSessions   = filter_sessions(protocols = 'SP',only_session_id=session_list)
sessions,nSessions   = filter_sessions(protocols = ['SP'],filter_areas=areas)

#%%  Load data properly:        
calciumversion = 'deconv'
# calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_data(load_calciumdata=True)

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
my_savefig(fig,savedir,'RRR_R2_acrossranks_V1PM_SPONT_%dsessions_%dsec.png' % (nSessions,temporalbin),formats=['png'])

#%% 

######  ######  ######     #          #    ######     #     #  #####     #     # #     # #       
#     # #     # #     #    #         # #   #     #    #     # #     #    #     # ##    # #       
#     # #     # #     #    #        #   #  #     #    #     # #          #     # # #   # #       
######  ######  ######     #       #     # ######     #     #  #####     #     # #  #  # #       
#   #   #   #   #   #      #       ####### #     #     #   #        #    #     # #   # # #       
#    #  #    #  #    #     #       #     # #     #      # #   #     #    #     # #    ## #       
#     # #     # #     #    ####### #     # ######        #     #####      #####  #     # ####### 

#%

from utils.pair_lib import value_matching

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

lam                 = 0
nranks              = 20
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
maxnoiselevel       = 20

R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions),np.nan)
R2_ranks            = np.full((narealabelpairs,nSessions,nranks,nmodelfits,kfold),np.nan)

filter_nearby       = True
# filter_nearby       = False

valuematching       = None
# valuematching       = 'noise_level'
# valuematching       = 'event_rate'
# valuematching       = 'skew'
# valuematching       = 'meanF'
nmatchbins          = 10
minsampleneurons    = 10

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    # idx_T               = ses.trialdata['stimCond']==0

    allpops             = np.array([i.split('-') for i in arealabelpairs]).flatten()
    nsampleneurons      = np.min([np.sum((ses.celldata['arealabel']==i) & (ses.celldata['noise_level']<maxnoiselevel)) for i in allpops])
    #take the smallest sample size

    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    
        if valuematching is not None:
            #Get value to match from celldata:
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_areax,idx_areay))
            group       = np.concatenate((np.zeros(len(idx_areax)),np.ones(len(idx_areay))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_areax   = np.intersect1d(idx_areax,idx_sub) #recover subset from idx_joint
            idx_areay   = np.intersect1d(idx_areay,idx_sub)

        X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T #Get activity and transpose to samples x features
        Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue
        R2_cv[iapl,ises],optim_rank[iapl,ises],R2_ranks[iapl,ises,:,:,:]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
        #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS

#%% Plot the R2 performance and number of dimensions per area pair
# fig         = plot_RRR_R2_arealabels(R2_cv,optim_rank,R2_ranks,arealabelpairs,clrs_arealabelpairs)
# my_savefig(fig,savedir,'RRR_cvR2_RegressOutBehavior_V1PM_LabUnl_%dsessions' % nSessions)
fig         = plot_RRR_R2_arealabels(R2_cv[:4],optim_rank[:4],R2_ranks[:4],arealabelpairs[:4],clrs_arealabelpairs[:4])
my_savefig(fig,savedir,'RRR_cvR2_V1PM_LabUnl_%dsessions' % nSessions)
fig         = plot_RRR_R2_arealabels(R2_cv[4:],optim_rank[4:],R2_ranks[4:],arealabelpairs[4:],clrs_arealabelpairs[4:])
my_savefig(fig,savedir,'RRR_cvR2_PMV1_LabUnl_%dsessions' % nSessions)
