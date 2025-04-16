# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
os.chdir('e:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
from tqdm import tqdm

# from loaddata import * #get all the loading data functions (filter_sessions,load_sessions)
from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive

from scipy import stats
from scipy.stats import zscore
from utils.psth import *
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score as AUC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches
from utils.plotting_style import * #get all the fixed color schemes
from matplotlib.lines import Line2D
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *
from detection.example_cells import get_example_cells
from utils.plot_lib import * # get support functions for plotting
from utils.regress_lib import * # get support functions for regression
from utils.dimreduc_lib import * # get support functions for dimensionality reduction

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\MultiAreaRegression\\')

#%% ###############################################################
protocol            = 'DN'
# calciumversion      = 'deconv'
calciumversion      = 'dF'

session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE12013', '2024_04_25']])
# session_list = np.array([['LPE11997', '2024_04_16']])
# session_list = np.array([['LPE11998', '2024_04_30']])
# session_list = np.array([['LPE11622', '2024_02_22']])
# session_list = np.array([['LPE10884', '2023_12_15']])
# session_list = np.array([['LPE10884', '2024_01_16']])
session_list = np.array([['LPE11997', '2024_04_16'],
                         ['LPE11622', '2024_02_21'],
                         ['LPE11998', '2024_04_30'],
                         ['LPE12013','2024_04_25']])

sessions,nSessions = filter_sessions(protocol,only_session_id=session_list,load_behaviordata=True,load_videodata=True,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

#%%
minlabcells = 25
sessions,nSessions = filter_sessions(protocol,min_lab_cells_PM=minlabcells,min_lab_cells_V1=minlabcells,load_behaviordata=True,load_videodata=True,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

#%% 
# sessions,nSessions,sbins = load_neural_performing_sessions()

#%% ### Show for all sessions which region of the psychometric curve the noise spans #############
sessions = noise_to_psy(sessions,filter_engaged=True)

#%% 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)


#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 80   #post cm
binsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')
    sessions[i].stensor[np.isnan(sessions[i].stensor)] = np.nanmean(sessions[i].stensor)
    
    # Compute average response in stimulus response zone:
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
                                    sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    # temp = pd.DataFrame(np.reshape(np.array(sessions[i].videodata['motionenergy']),(len(sessions[0].videodata['motionenergy']),1)))
    # sessions[i].respmat_videome     = compute_respmat_space(temp, sessions[i].videodata['ts'], sessions[i].trialdata['stimStart'],
    #                                 sessions[i].videodata['zpos'],sessions[i].videodata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)


#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

#%% ############################### Plot neuron-average per stim per area #################################
ises = 1 #selected session to plot this for

fig = plot_mean_stim_spatial(sessions[ises], sbins, labeled= ['unl','lab'], areas= ['V1','PM','AL','RSP'])
# plt.savefig(os.path.join(savedir,'ActivityInCorridor_neuronAverage_arealabels_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'ActivityInCorridor_deconv_neuronAverage_perStim_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% 

######  ######  ######  
#     # #     # #     # 
#     # #     # #     # 
######  ######  ######  
#   #   #   #   #   #   
#    #  #    #  #    #  
#     # #     # #     # 


#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsize             = 50
nranks              = 25
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions),np.nan)
optim_rank          = np.full((nSessions),np.nan)

idx_sbin            = (sbins>=-10) & (sbins<=30)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = ses.trialdata['Orientation']==0
    # idx_T               = ses.trialdata['stimcat']=='M'
    idx_T               = np.ones(len(ses.trialdata['stimcat']),dtype=bool)
    idx_T               = np.random.choice(np.arange(len(ses.trialdata)),int(len(ses.trialdata)*0.8),replace=False)

    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].stensor[np.ix_(idx_areax,idx_T,idx_sbin)].reshape(len(idx_areax),-1).T
    Y                   = sessions[ises].stensor[np.ix_(idx_areay,idx_T,idx_sbin)].reshape(len(idx_areay),-1).T

    if len(idx_areax)>popsize and len(idx_areay)>popsize:
        R2_cv[ises],optim_rank[ises]             = RRR_wrapper(Y, X, nN=popsize,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))

#%% 
labeled     = ['unl','lab']
nlabels     = 2
areas       = ['V1','PM','AL','RSP']
areas       = ['V1','PM']
nareas      = len(areas)
clrs_areas  = get_clr_areas(areas)
clrs_vars   = sns.color_palette('inferno', 3)

#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200])
# popsizes            = np.array([5,10,20,50,100])
npopsizes           = len(popsizes)
nranks              = 25
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,npopsizes),np.nan)
optim_rank          = np.full((nSessions,npopsizes),np.nan)

idx_sbin            = (sbins>-20) & (sbins<50)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = ses.trialdata['Orientation']==0
    # idx_T               = ses.trialdata['stimcat']=='M'
    idx_T               = np.ones(len(ses.trialdata['stimcat']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    # X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    # Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    X                   = sessions[ises].stensor[np.ix_(idx_areax,idx_T,idx_sbin)].reshape(len(idx_areax),-1).T
    Y                   = sessions[ises].stensor[np.ix_(idx_areay,idx_T,idx_sbin)].reshape(len(idx_areay),-1).T

    for ipop,pop in enumerate(popsizes):
        if len(idx_areax)>pop and len(idx_areay)>pop:
            R2_cv[ises,ipop],optim_rank[ises,ipop]             = RRR_wrapper(Y, X, nN=pop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot R2 for different number of V1 and PM neurons
clrs_popsizes = sns.color_palette("rocket",len(popsizes))

fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=True)
ax = axes[0]
ax.scatter(popsizes, np.nanmean(R2_cv,axis=0), marker='o', color=clrs_popsizes)
# ax.plot(popsizes, np.nanmean(R2_cv,axis=0),color='k', linewidth=2)
ax.plot(popsizes,R2_cv.T,color='k', linewidth=0.5)
shaded_error(popsizes,R2_cv,center='mean',error='sem',color='k',ax=ax)
ax.set_ylim([0,0.25])
ax.set_xticks(popsizes)
ax.axhline(y=0,color='k',linestyle='--')

ax.set_xlabel('Population size')
ax.set_ylabel('RRR R2')
# ax.set_xscale('log')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
ax.scatter(popsizes, np.nanmean(optim_rank,axis=0), marker='o', color=clrs_popsizes)
shaded_error(popsizes,optim_rank,center='mean',error='sem',color='k',ax=ax)

ax.plot(popsizes,popsizes**0.5,color='r',linestyle='--',linewidth=1)
ax.text(50,10,'$n^{1/2}$',color='r',fontsize=12)
ax.plot(popsizes,popsizes**0.3333,color='g',linestyle='--',linewidth=1)
ax.text(50,2,'$n^{1/3}$',color='g',fontsize=12)

ax.set_ylim([0,20])
ax.set_ylabel('Dimensionality')
ax.set_xlabel('Population size')
ax.set_xticks(popsizes)

plt.tight_layout()

#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons

npopsize            = 100
nranks              = 25
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,2),np.nan)
optim_rank          = np.full((nSessions,2),np.nan)

# idx_sbin            = (sbins>=-10) & (sbins<=30)
idx_sbin            = (sbins>=-20) & (sbins<=50)
min_trials          = 50 #minimum number of trials for each category to be included

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    for iresp in [0,1]:
        # idx_T               = ses.trialdata['Orientation']==0

        idx_T               = np.all((ses.trialdata['stimcat']=='N',
                                ses.trialdata['lickResponse']==iresp),axis=0)
        
        idx_T               = np.all((np.isin(ses.trialdata['stimcat'],['N','M']),
                            ses.trialdata['lickResponse']==iresp),axis=0)
        
        # idx_T               = ses.trialdata['lickResponse']==iresp

        idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                                ses.celldata['noise_level']<20),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                                ses.celldata['noise_level']<20),axis=0))[0]
        
        # X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
        # Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
        X                   = sessions[ises].stensor[np.ix_(idx_areax,idx_T,idx_sbin)].reshape(len(idx_areax),-1).T
        Y                   = sessions[ises].stensor[np.ix_(idx_areay,idx_T,idx_sbin)].reshape(len(idx_areay),-1).T

        if len(idx_areax)>npopsize and len(idx_areay)>npopsize and np.sum(idx_T)>=min_trials:
            R2_cv[ises,iresp],optim_rank[ises,iresp]             = RRR_wrapper(Y, X, nN=npopsize,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot R2 for different number of V1 and PM neurons
fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=False)
ax = axes[0]
ax.scatter(R2_cv[:,0],R2_cv[:,1], marker='o', color='k')
ax.plot([0,1],[0,1],color='grey',linestyle='--')
ax.set_xlim([0,0.25])
ax.set_ylim([0,0.25])
ax.set_title('R2 (RRR V1->PM)')
ax.set_xlabel('Misses)')
ax.set_ylabel('Hits)')

ax = axes[1]
ax.scatter(optim_rank[:,0],optim_rank[:,1], marker='o', color='k')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.plot([0,10],[0,10],color='grey',linestyle='--')
ax.set_title('Rank (RRR V1->PM)')
ax.set_xlabel('Misses')
ax.set_ylabel('Hits')

#%% print how many V1lab and PMlab cells there are in the loaded sessions:
print('V1 labeled cells:\n')
for ses in sessions:
    print('%d' % np.sum(ses.celldata['arealabel']=='V1lab'))

print('PM labeled cells:\n')
for ses in sessions:
    print('%d' % np.sum(ses.celldata['arealabel']=='PMlab'))

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1unl',
                    'PMlab-V1lab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

nsampleneurons      = 25

lam                 = 0
nranks              = 25
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions),np.nan)

filter_nearby       = True

idx_sbin            = (sbins>=-20) & (sbins<=50)
min_trials          = 50 #minimum number of trials for each category to be included

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
        idx_T               = np.isin(ses.trialdata['stimcat'],['N','M'])
        # idx_T               = np.all((np.isin(ses.trialdata['stimcat'],['N','M'])),axis=0)
        
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
        
            X                   = sessions[ises].stensor[np.ix_(idx_areax,idx_T,idx_sbin)].reshape(len(idx_areax),-1).T
            Y                   = sessions[ises].stensor[np.ix_(idx_areay,idx_T,idx_sbin)].reshape(len(idx_areay),-1).T

            if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons and np.sum(idx_T)>=min_trials:
                R2_cv[iapl,ises],optim_rank[iapl,ises]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)


#%% Plot the number of dimensions per area pair

datatoplot = R2_cv
arealabelpairs2 = [al.replace('-','-\n') for al in arealabelpairs]

fig, axes = plt.subplots(1,2,figsize=(8,3.5))
ax=axes[0]
for iapl, arealabelpair in enumerate(arealabelpairs):
    ax.scatter(np.ones(nSessions)*iapl + np.random.randn(nSessions)*0.1,datatoplot[iapl,:],color='k',marker='o',s=10)
    ax.errorbar(iapl+0.25,np.nanmean(datatoplot[iapl,:]),np.nanstd(datatoplot[iapl,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)
ax.plot(datatoplot,'k',linewidth=0.15,alpha=0.5)
ax.set_xticks(range(narealabelpairs))
ax.set_ylabel('R2 (cv)')
ax.set_ylim([0,my_ceil(np.nanmax(datatoplot),2)])
ax.set_xlabel('Population pair')
ax.set_title('Performance at optimal rank')

ax=axes[1]
for iapl, arealabelpair in enumerate(arealabelpairs):
    ax.scatter(np.ones(nSessions)*iapl + np.random.randn(nSessions)*0.2,
               optim_rank[iapl,:],color='k',marker='o',s=10)
    ax.errorbar(iapl+0.25,np.nanmean(optim_rank[iapl,:]),np.nanstd(optim_rank[iapl,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)
ax.plot(optim_rank,'k',linewidth=0.15,alpha=0.5)
ax.set_xticks(range(narealabelpairs))
ax.set_ylabel('Number of dimensions')
ax.set_ylim([0,my_ceil(np.nanmax(optim_rank),0)+1])
ax.set_yticks(np.arange(0,10,2))
ax.set_xlabel('Population pair')
ax.set_title('Dimensionality')

sns.despine(top=True,right=True,offset=3)
axes[0].set_xticklabels(arealabelpairs2,fontsize=7)
axes[1].set_xticklabels(arealabelpairs2,fontsize=7)

fig.savefig(os.path.join(savedir,'RRR_cvR2_V1PM_LabUnl_%dsessions.png' % nSessions))

#%% Plot the number of dimensions per area pair

arealabelpairs2 = [al.replace('-','-\n') for al in arealabelpairs]

fig, axes = plt.subplots(1,2,figsize=(7,3),sharey=True,sharex=True)
ax=axes[0]

for iapl in [1,2,3]:
    ax.scatter(R2_cv[0,:],R2_cv[iapl,:],marker='o', color=clrs_arealabelpairs[iapl],s=20)
    t,p = ttest_rel(R2_cv[0,:],R2_cv[iapl,:],nan_policy='omit')
    if p<0.05:
        # ax.text(0.6,0.1+0.01*iapl,'%s (%s)' % (get_sig_asterisks(p),arealabelpairs2[iapl]),transform=ax.transAxes,
        ax.text(0.6,0.1+0.1*iapl,'%s' % (get_sig_asterisks(p)),transform=ax.transAxes,
                ha='center',va='center',fontsize=25,color=clrs_arealabelpairs[iapl])
ax.legend(arealabelpairs[1:4],frameon=True,loc='upper right',fontsize=7,ncol=1,title=arealabelpairs[0] + ' vs.')
ax.plot([0,1],[0,1],color='grey',linestyle='--')
ax.set_xlim([0,my_ceil(np.nanmax(R2_cv),2)])
ax.set_ylim([0,my_ceil(np.nanmax(R2_cv),2)])

ax.set_xlabel('Unlabeled \n(%s)' % arealabelpairs[0])
ax.set_ylabel('labeled population')
ax.set_title('Feedforward (V1->PM)')

ax=axes[1]
for iapl in [5,6,7]:
    ax.scatter(R2_cv[4,:],R2_cv[iapl,:],marker='o', color=clrs_arealabelpairs[iapl],s=20)
    t,p = ttest_rel(R2_cv[4,:],R2_cv[iapl,:],nan_policy='omit')
    if p<0.05:
        ax.text(0.6,0.1+0.1*(iapl-4),'%s' % (get_sig_asterisks(p)),transform=ax.transAxes,
                ha='center',va='center',fontsize=25,color=clrs_arealabelpairs[iapl])
ax.legend(arealabelpairs[5:],frameon=True,loc='upper right',fontsize=7,ncol=1,title=arealabelpairs[4] + 'vs.')
ax.plot([0,1],[0,1],color='grey',linestyle='--')
ax.set_xlim([0,my_ceil(np.nanmax(R2_cv),2)])
ax.set_ylim([0,my_ceil(np.nanmax(R2_cv),2)])
ax_nticks(ax,3)
ax.set_xlabel('Unlabeled \n(%s)' % arealabelpairs[4])
ax.set_ylabel('labeled population')
ax.set_title('Performance at optimal rank')
ax.set_title('Feedback (PM->V1)')

# ax=axes[1]
# for iapl in [1,2,3]:
#     ax.scatter(optim_rank[0,:],optim_rank[iapl,:],marker='o', color=clrs_arealabelpairs[iapl],s=15)
#     t,p = ttest_rel(optim_rank[0,:],optim_rank[iapl,:],nan_policy='omit')
#     print(p)
#     if p<0.05:
#         # ax.text(0.6,0.1+0.01*iapl,'%s (%s)' % (get_sig_asterisks(p),arealabelpairs2[iapl]),transform=ax.transAxes,
#         ax.text(0.6,0.1+0.01*iapl,'%s' % (get_sig_asterisks(p)),transform=ax.transAxes,
#                 ha='center',va='center',fontsize=25,color=clrs_arealabelpairs[iapl])

# ax.plot([0,10],[0,10],color='grey',linestyle='--')
# ax.set_xlim([0,np.nanmax(optim_rank)+1])
# ax.set_ylim([0,np.nanmax(optim_rank)+1])

# ax.set_xlabel(arealabelpairs[0])
# ax.set_ylabel(arealabelpairs[1:4])
# ax.set_title('Dimensionality')

sns.despine(top=True,right=True,offset=3)
fig.savefig(os.path.join(savedir,'RRR_cvR2_V1PM_LabUnl_%dsessions_scatter.png' % nSessions))

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1unl',
                    'PMlab-V1lab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

nsampleneurons      = 25

lam                 = 0
nranks              = 25
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((narealabelpairs,2,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,2,nSessions),np.nan)

filter_nearby       = True

idx_sbin            = (sbins>=-20) & (sbins<=50)
min_trials          = 50 #minimum number of trials for each category to be included

def min_pop_size(ses):
    arealabels      = ['V1lab','PMlab']

    popsizes        = []
    for ial, al in enumerate(arealabels):
        popsizes.append(np.sum(np.all((ses.celldata['arealabel']==al,
                            ses.celldata['noise_level']<20),axis=0)))
    return np.min(popsizes)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    
    # Sampling the same minimal number of neurons from each population (v1lab, v1unl, pmunl, pmlab)
    nsampleneurons  = min_pop_size(ses)
    # print(nsampleneurons)
    
    #Subsample the same number of trials/samples for hits and misses
    idx_T_resp = np.empty((len(ses.trialdata['stimcat']),2),dtype=bool)
    for iresp in [0,1]: #get the number of trials for both responses
        idx_T_resp[:,iresp]               = np.all((np.isin(ses.trialdata['stimcat'],['N','M']),
                                    ses.trialdata['lickResponse']==iresp),axis=0)
    nsamples = np.min(np.sum(idx_T_resp,axis=0)) * idx_sbin.sum() #number of samples (trials x timebins)

    for iresp in [0,1]:
        # idx_T               = np.all((ses.trialdata['stimcat']=='N',
                                # ses.trialdata['lickResponse']==iresp),axis=0)
        
        idx_T               = np.all((np.isin(ses.trialdata['stimcat'],['N','M']),
                            ses.trialdata['lickResponse']==iresp),axis=0)
        
        for iapl, arealabelpair in enumerate(arealabelpairs):
            
            alx,aly = arealabelpair.split('-') #get source and target area label

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
        
            X                   = sessions[ises].stensor[np.ix_(idx_areax,idx_T,idx_sbin)].reshape(len(idx_areax),-1).T
            Y                   = sessions[ises].stensor[np.ix_(idx_areay,idx_T,idx_sbin)].reshape(len(idx_areay),-1).T

            if len(idx_areax)>=nsampleneurons and len(idx_areay)>=nsampleneurons and np.sum(idx_T)>=min_trials:
                R2_cv[iapl,iresp,ises],optim_rank[iapl,iresp,ises]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=nsamples,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% 
R2_cv[iapl,iresp,ises],optim_rank[iapl,iresp,ises]  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot R2 for different number of V1 and PM neurons
arealabelpairs2     = [al.replace('-','-\n') for al in arealabelpairs]

fig,axes = plt.subplots(1,2,figsize=(8,4))
ax = axes[0]
for iapl, arealabelpair in enumerate(arealabelpairs):
    ax.scatter(R2_cv[iapl,0,:],R2_cv[iapl,1,:], marker='o', color=clrs_arealabelpairs[iapl])
    # ax.scatter(R2_cv[:,0],R2_cv[:,1], marker='o', color='k')
    t,p = ttest_rel(R2_cv[iapl,0,:],R2_cv[iapl,1,:],nan_policy='omit')
    print(p)
    if p<0.05:
        ax.text(0.6,0.1+0.01*iapl,'%s' % get_sig_asterisks(p),transform=ax.transAxes,
                ha='center',va='center',fontsize=25,color=clrs_arealabelpairs[iapl])
ax.legend(arealabelpairs2,frameon=False,fontsize=7,ncol=2)
ax.plot([0,1],[0,1],color='grey',linestyle='--')
ax.set_xlim([0,my_ceil(np.nanmax(R2_cv),1)])
ax.set_ylim([0,my_ceil(np.nanmax(R2_cv),1)])
ax.set_title('R2 (RRR)')
ax.set_xlabel('Misses')
ax.set_ylabel('Hits')

ax = axes[1]
for iapl, arealabelpair in enumerate(arealabelpairs):
    ax.scatter(optim_rank[iapl,0,:]+np.random.rand(nSessions)*0.3,optim_rank[iapl,1,:]+np.random.rand(nSessions)*0.3, marker='o', color=clrs_arealabelpairs[iapl])
    # ax.scatter(R2_cv[:,0],R2_cv[:,1], marker='o', color='k')
    t,p = ttest_rel(optim_rank[iapl,0,:],optim_rank[iapl,1,:],nan_policy='omit')
    if p<0.05:
        ax.text(0.8,0.1+0.05*iapl,'%s' % get_sig_asterisks(p),transform=ax.transAxes,
                ha='center',va='center',fontsize=15,color=clrs_arealabelpairs[iapl])
ax.legend(arealabelpairs2,frameon=False,fontsize=7,ncol=2)
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.plot([0,10],[0,10],color='grey',linestyle='--')
ax.set_title('Rank (RRR)')
ax.set_xlabel('Misses')
ax.set_ylabel('Hits')

#%% Time-resolved RRR across time bins (decoding communication strength over time)
window_size = 3       # number of time bins per window
step_size   = 1       # step between windows
timebins    = np.arange(len(sbins))
bin_starts  = np.arange(0, len(timebins) - window_size + 1, step_size)

arealabelpairs = ['V1unl-PMunl',
                    'PMunl-V1unl']
arealabelpairs  = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1unl',
                    'PMlab-V1lab']
narealabelpairs = len(arealabelpairs)
nsampleneurons  = 25

# shape: [narealabelpairs, 2 (hit/miss), nSessions, nTimeWindows]
R2_timecv       = np.full((narealabelpairs, 2, nSessions, len(bin_starts)), np.nan)
Rank_timecv   = np.full((narealabelpairs, 2, nSessions, len(bin_starts)), np.nan)

for ises, ses in tqdm(enumerate(sessions), total=nSessions, desc='Time-resolved RRR decoding'):
    for iresp in [0,1]:  # 0=miss, 1=hit

        idx_T = np.all((np.isin(ses.trialdata['stimcat'],['N','M']),
                        ses.trialdata['lickResponse']==iresp), axis=0)

        if np.sum(idx_T) < min_trials:
            continue  # Skip sessions with too few trials

        for iapl, arealabelpair in enumerate(arealabelpairs):
            # print('Processing %s' % arealabelpair)
            alx, aly = arealabelpair.split('-')

            if filter_nearby:
                idx_nearby = filter_nearlabeled(ses, radius=50)
            else:
                idx_nearby = np.ones(len(ses.celldata), dtype=bool)

            idx_areax = np.where(np.all((ses.celldata['arealabel']==alx,
                                         ses.celldata['noise_level']<20,
                                         idx_nearby), axis=0))[0]
            idx_areay = np.where(np.all((ses.celldata['arealabel']==aly,
                                         ses.celldata['noise_level']<20,
                                         idx_nearby), axis=0))[0]

            if len(idx_areax) < nsampleneurons or len(idx_areay) < nsampleneurons:
                continue

            for itw, start_idx in enumerate(bin_starts):
                idx_bin = timebins[start_idx:start_idx+window_size]

                X = ses.stensor[np.ix_(idx_areax, idx_T, idx_bin)].reshape(len(idx_areax), -1).T
                Y = ses.stensor[np.ix_(idx_areay, idx_T, idx_bin)].reshape(len(idx_areay), -1).T

                R2_rep, rank = RRR_wrapper(Y, X,
                                        nN=nsampleneurons,
                                        nK=None,
                                        lam=lam,
                                        nranks=nranks,
                                        kfold=kfold,
                                        nmodelfits=nmodelfits)

                # Store max R2 across ranks (or select based on rank strategy)
                R2_timecv[iapl, iresp, ises, itw] = R2_rep
                Rank_timecv[iapl, iresp, ises, itw] = rank

#%% plot the results:
narealabelpairs = len(arealabelpairs)

# set to nan all sessions of which either hit or miss has nan
hasnan = np.logical_or(np.isnan(R2_timecv[0,0,:,0]),np.isnan(R2_timecv[0,1,:,0]))
R2_timecv[:, :, hasnan,:] = np.nan

# R2_timecv  # shape = [arealabels x hit/miss x sessions x timebins]
bincenters = sbins[(bin_starts+window_size/2).astype(int)]
# fig,axes = plt.subplots(narealabelpairs,1,figsize=(3.5,2*narealabelpairs))
fig,axes = plt.subplots(2,4,figsize=(10,4))
axes = axes.flatten()
for iapl, arealabelpair in enumerate(arealabelpairs):
    ax = axes[iapl]
    handles = []
    handles.append(ax.plot(bincenters,np.nanmean(R2_timecv[iapl,0,:],axis=0),label=arealabelpair, color=clrs_arealabelpairs[iapl],linestyle='--')[0])
    handles.append(ax.plot(bincenters,np.nanmean(R2_timecv[iapl,1,:],axis=0),label=arealabelpair, color=clrs_arealabelpairs[iapl],linestyle='-')[0])
    if iapl==0:
        ax.set_ylabel('R2 (RRR)')
        ax.legend(handles=handles,labels=['Miss','Hit'],frameon=False,fontsize=8,loc='upper left')
    ax.fill_between(bincenters,np.nanmean(R2_timecv[iapl,0,:],axis=0)-np.nanstd(R2_timecv[iapl,0,:],axis=0),np.nanmean(R2_timecv[iapl,0,:],axis=0)+np.nanstd(R2_timecv[iapl,0,:],axis=0),alpha=0.3,color=clrs_arealabelpairs[iapl])
    ax.fill_between(bincenters,np.nanmean(R2_timecv[iapl,1,:],axis=0)-np.nanstd(R2_timecv[iapl,1,:],axis=0),np.nanmean(R2_timecv[iapl,0,:],axis=0)+np.nanstd(R2_timecv[iapl,0,:],axis=0),alpha=0.3,color=clrs_arealabelpairs[iapl])
    ax.set_title(arealabelpair,fontsize=10,color=clrs_arealabelpairs[iapl])
    if iapl==narealabelpairs-1:
        ax.set_xticks(bincenters[::2])
        ax.set_xlabel('Position (cm)')
    else:
        ax.set_xticks(bincenters[::2],labels=[])
    add_stim_resp_win(ax)
    ax.set_ylim([0,0.06])
    ax.set_xlim(np.percentile(bincenters,[0,100]))
sns.despine(top=True,right=True,offset=3)

# fig.suptitle('Time-resolved decoding of communication strength')
fig.tight_layout()
fig.savefig(os.path.join(savedir,'RRR_R2_timecv.png'), 
            format = 'png', bbox_inches='tight')  
