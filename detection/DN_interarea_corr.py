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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive
from utils.plot_lib import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.psth import *
from detection.plot_neural_activity_lib import *
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
minlabcells = 20
minlabcells = 0
sessions,nSessions = filter_sessions(protocol,min_lab_cells_PM=minlabcells,min_lab_cells_V1=minlabcells,min_cells=1,
                        load_behaviordata=True,load_videodata=True,load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

#%% 
# sessions,nSessions,sbins = load_neural_performing_sessions()
idx_ses = get_idx_performing_sessions(sessions,zmin_thr=0,zmax_thr=0,guess_thr=0.4,filter_engaged=True)()

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
binsize     = 10   #spatial binning in cm

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
ises = 2 #selected session to plot this for

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
areas       = ['V1','PM','AL','RSP']
print('Number of cells in each area for all the sessions:')
for ises,ses in enumerate(sessions):
    for area in areas:
        print('%d: %s: %d' % (ises,area,np.sum(ses.celldata['roi_name']==area)))

#%% 
areas       = ['V1','PM','AL','RSP']
areas       = ['V1','PM']
nareas      = len(areas)
clrs_areas  = get_clr_areas(areas)
clrs_vars   = sns.color_palette('inferno', 3)

#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20,50,100])
npopsizes           = len(popsizes)
nranks              = 25
nmodelfits          = 20 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,npopsizes),np.nan)
optim_rank          = np.full((nSessions,npopsizes),np.nan)

idx_sbin            = (sbins>=-20) & (sbins<=50)

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
fig.savefig(os.path.join(savedir,'RRR_R2_and_Rank_PopSize_V1PM_%dsessions.png' % nSessions), format = 'png')


#%% Does performance or rank differ for hits and misses:
npopsize            = 100
nranks              = 25
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,2),np.nan)
optim_rank          = np.full((nSessions,2),np.nan)

# idx_sbin            = (sbins>=-10) & (sbins<=30)
idx_sbin            = sbins>-1000
min_trials          = 50 #minimum number of trials for each category to be included
source_area = 'V1'
target_area = 'PM'
# source_area = 'PM'
# target_area = 'AL'
# target_area = 'V1'

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='RRR for hits and misses'):
    #Subsample the same number of trials/samples for hits and misses
    idx_T_resp = np.empty((len(ses.trialdata['stimcat']),2),dtype=bool)
    for iresp in [0,1]: #get the number of trials for both responses
        # idx_T_resp[:,iresp]               = np.all((np.isin(ses.trialdata['stimcat'],['N','M']),
        idx_T_resp[:,iresp]               = np.all((np.isin(ses.trialdata['stimcat'],['N']),
                                    ses.trialdata['lickResponse']==iresp),axis=0)
    nsamples = np.min(np.sum(idx_T_resp,axis=0)) * idx_sbin.sum() #number of samples (trials x timebins)

    for iresp in [0,1]:
        # idx_T               = ses.trialdata['Orientation']==0

        idx_T               = np.all((ses.trialdata['stimcat']=='N',
                                ses.trialdata['lickResponse']==iresp),axis=0)
        
        # idx_T               = np.all((np.isin(ses.trialdata['stimcat'],['N','M']),
                            # ses.trialdata['lickResponse']==iresp),axis=0)
        
        # idx_T               = ses.trialdata['lickResponse']==iresp

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==source_area,
                                ses.celldata['noise_level']<20),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==target_area,
                                ses.celldata['noise_level']<20),axis=0))[0]
        
        # X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
        # Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
        if len(idx_areax)>npopsize and len(idx_areay)>npopsize and np.sum(idx_T)>=min_trials:
            X                   = sessions[ises].stensor[np.ix_(idx_areax,idx_T,idx_sbin)].reshape(len(idx_areax),-1).T
            Y                   = sessions[ises].stensor[np.ix_(idx_areay,idx_T,idx_sbin)].reshape(len(idx_areay),-1).T

            R2_cv[ises,iresp],optim_rank[ises,iresp]             = RRR_wrapper(Y, X, nN=npopsize,nK=nsamples,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot R2 for hits and misses:
fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=False)
ax = axes[0]
ax.scatter(R2_cv[:,0],R2_cv[:,1], marker='o', color='k')
ax.plot([0,1],[0,1],color='grey',linestyle='--')
ax.set_ylim([0,my_ceil(np.nanmax(R2_cv),1)] )
ax.set_xlim([0,my_ceil(np.nanmax(R2_cv),1)] )
ax.set_title('R2 (RRR %s->%s)' % (source_area,target_area))
ax.set_xlabel('Misses)')
ax.set_ylabel('Hits)')

t,p = ttest_rel(R2_cv[:,0],R2_cv[:,1],nan_policy='omit')
print('Paired t-test (R2): p=%.3f' % p)
if p<0.05:
    ax.text(0.1,0.2,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='red')

ax = axes[1]
ax.scatter(optim_rank[:,0],optim_rank[:,1], marker='o', color='k')
ax.set_xlim([0,np.nanmax(optim_rank)+1])
ax.set_ylim([0,np.nanmax(optim_rank)+1])
ax.plot([0,20],[0,20],color='grey',linestyle='--')
ax.set_title('Rank (RRR %s->%s)' % (source_area,target_area))
ax.set_xlabel('Misses')
ax.set_ylabel('Hits')

t,p = ttest_rel(optim_rank[:,0],optim_rank[:,1],nan_policy='omit')
print('Paired t-test (rank): p=%.3f' % p)
if p<0.05:
    ax.text(0.1,0.2,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='red')

# fig.savefig(os.path.join(savedir,'RRR_R2_Rank_%s%s_HitsMisses_Stim_%dsessions.png' % 
fig.savefig(os.path.join(savedir,'RRR_R2_Rank_%s%s_HitsMisses_Thr_%dsessions.png' % 
                         (source_area,target_area,len(sessions))), 
            format = 'png',bbox_inches='tight',dpi=300)



#%% Time-resolved RRR across time bins (decoding communication strength over time)
window_size         = 3       # number of time bins per window
step_size           = 1       # step between windows
timebins            = np.arange(len(sbins))
bin_starts          = np.arange(0, len(timebins) - window_size + 1, step_size)

arealabelpairs      = ['V1-PM','PM-V1']
narealabelpairs     = len(arealabelpairs)
nsampleneurons      = 100
min_trials          = 50
lam                 = 0
nranks              = 25
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5

nhitmiss            = 2
# shape: [narealabelpairs, 2 (hit/miss), nSessions, nTimeWindows]
RRR_time_R2       = np.full((narealabelpairs, nhitmiss, nSessions, len(bin_starts)), np.nan)
RRR_time_Rank     = np.full((narealabelpairs, nhitmiss, nSessions, len(bin_starts)), np.nan)

# shape: [narealabelpairs, 2 (source/target), 2 (hit/miss), nSessions, nTimeWindows]
dec_time_signalR2       = np.full((narealabelpairs, 2, nhitmiss, nSessions, len(bin_starts)), np.nan)
dec_time_neuralR2       = np.full((narealabelpairs, 2,nhitmiss, nSessions, len(bin_starts)), np.nan)
dec_time_projcorr       = np.full((narealabelpairs, nhitmiss, nSessions, len(bin_starts)), np.nan)
dec_model_name          = 'Ridge'
dec_lam                 = 100

for ises, ses in tqdm(enumerate(sessions), total=nSessions, desc='Time-resolved RRR + decoding'):
    for iresp in [0,1]:  # 0=miss, 1=hit

        idx_T = np.all((np.isin(ses.trialdata['stimcat'],['N']),
        # idx_T = np.all((np.isin(ses.trialdata['stimcat'],['N','M']),
                        ses.trialdata['lickResponse']==iresp), axis=0)

        s_idx_T           = ses.trialdata['signal'][idx_T].to_numpy()

        if np.sum(idx_T) < min_trials:
            continue  # Skip sessions with too few trials

        for iapl, arealabelpair in enumerate(arealabelpairs):
            # print('Processing %s' % arealabelpair)
            alx, aly    = arealabelpair.split('-')

            idx_areax   = np.where(np.all((ses.celldata['roi_name']==alx,
                                         ses.celldata['noise_level']<20,
                                         ), axis=0))[0]
            idx_areay   = np.where(np.all((ses.celldata['roi_name']==aly,
                                         ses.celldata['noise_level']<20,
                                         ), axis=0))[0]

            if len(idx_areax) < nsampleneurons or len(idx_areay) < nsampleneurons:
                continue

            for itw, start_idx in enumerate(bin_starts):
                idx_bin = timebins[start_idx:start_idx+window_size]

                X = ses.stensor[np.ix_(idx_areax, idx_T, idx_bin)].reshape(len(idx_areax), -1).T
                Y = ses.stensor[np.ix_(idx_areay, idx_T, idx_bin)].reshape(len(idx_areay), -1).T

                # R2_rep, rank = RRR_wrapper(Y, X,
                #                         nN=nsampleneurons,
                #                         nK=None,
                #                         lam=lam,
                #                         nranks=nranks,
                #                         kfold=kfold,
                #                         nmodelfits=nmodelfits)

                # Store max R2 across ranks (or select based on rank strategy)
                RRR_time_R2[iapl, iresp, ises, itw] = R2_rep
                RRR_time_Rank[iapl, iresp, ises, itw] = rank

                stemp   = copy.deepcopy(s_idx_T)

                # stemp = np.tile(stemp, len(idx_bin))
                stemp = np.repeat(stemp, len(idx_bin))

                idx_areax_sub  = np.random.choice(np.where(idx_areax)[0],size=nsampleneurons,replace=False)
                idx_areay_sub  = np.random.choice(np.where(idx_areay)[0],size=nsampleneurons,replace=False)
                
                X = ses.stensor[np.ix_(idx_areax_sub, idx_T, idx_bin)].reshape(len(idx_areax_sub), -1).T
                Y = ses.stensor[np.ix_(idx_areay_sub, idx_T, idx_bin)].reshape(len(idx_areay_sub), -1).T

                # X               = np.nanmean(ses.stensor[np.ix_(idx_areax_sub, idx_T, idx_bin)], axis=2).T
                # Y               = np.nanmean(ses.stensor[np.ix_(idx_areay_sub, idx_T, idx_bin)], axis=2).T

                X,stemp,_           = prep_Xpredictor(X,stemp) #zscore, set columns with all nans to 0, set nans to 0
                Y,stemp,_           = prep_Xpredictor(Y,stemp) #zscore, set columns with all nans to 0, set nans to 0
                
                r2_x,w_x,p_x,ev_x   = my_decoder_wrapper(X,stemp,model_name=dec_model_name,kfold=kfold,lam=dec_lam,
                                                            subtract_shuffle=True,norm_out=True)
                r2_y,w_y,p_y,ev_y   = my_decoder_wrapper(Y,stemp,model_name=dec_model_name,kfold=kfold,lam=dec_lam,
                                                            subtract_shuffle=True,norm_out=True)
                                
                dec_time_signalR2[iapl,0,iresp, ises, itw] = r2_x
                dec_time_signalR2[iapl,1,iresp, ises, itw] = r2_y
                
                dec_time_neuralR2[iapl,0,iresp, ises, itw] = ev_x
                dec_time_neuralR2[iapl,1,iresp, ises, itw] = ev_y

                dec_time_projcorr[iapl,iresp, ises, itw]   = np.corrcoef(p_x,p_y)[0,1]


#%% plot the results:
narealabelpairs             = len(arealabelpairs)
clrs_arealabelpairs         = get_clr_area_pairs(arealabelpairs)

# set to nan all sessions of which either hit or miss has nan
hasnan = np.logical_or(np.isnan(RRR_time_R2[0,0,:,0]),np.isnan(RRR_time_R2[0,1,:,0]))
RRR_time_R2[:, :, hasnan,:] = np.nan

bincenters = sbins[(bin_starts+window_size/2).astype(int)]
# fig,axes = plt.subplots(narealabelpairs,1,figsize=(3.5,2*narealabelpairs))
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3))
axes = axes.flatten()
for iapl, arealabelpair in enumerate(arealabelpairs):
    ax = axes[iapl]
    handles = []
    handles.append(ax.plot(bincenters,np.nanmean(RRR_time_R2[iapl,0,:],axis=0),label=arealabelpair, color=clrs_arealabelpairs[iapl],linestyle='--')[0])
    handles.append(ax.plot(bincenters,np.nanmean(RRR_time_R2[iapl,1,:],axis=0),label=arealabelpair, color=clrs_arealabelpairs[iapl],linestyle='-')[0])
    if iapl==0:
        ax.set_ylabel('R2 (RRR)')
        ax.legend(handles=handles,labels=['Miss','Hit'],frameon=False,fontsize=8,loc='upper left')
    ax.fill_between(bincenters,np.nanmean(RRR_time_R2[iapl,0,:],axis=0)-np.nanstd(RRR_time_R2[iapl,0,:],axis=0),np.nanmean(RRR_time_R2[iapl,0,:],axis=0)+np.nanstd(RRR_time_R2[iapl,0,:],axis=0),alpha=0.3,color=clrs_arealabelpairs[iapl])
    ax.fill_between(bincenters,np.nanmean(RRR_time_R2[iapl,1,:],axis=0)-np.nanstd(RRR_time_R2[iapl,1,:],axis=0),np.nanmean(RRR_time_R2[iapl,1,:],axis=0)+np.nanstd(RRR_time_R2[iapl,1,:],axis=0),alpha=0.3,color=clrs_arealabelpairs[iapl])
    ax.set_title(arealabelpair,fontsize=10,color=clrs_arealabelpairs[iapl])
    if iapl==narealabelpairs-1:
        ax.set_xticks(bincenters[::2])
        ax.set_xlabel('Position (cm)')
    else:
        ax.set_xticks(bincenters[::2],labels=[])
    add_stim_resp_win(ax)
    ax.set_ylim([0,0.1])
    ax.set_xlim(np.percentile(bincenters,[0,100]))
sns.despine(top=True,right=True,offset=3)

# fig.suptitle('Time-resolved decoding of communication strength')
fig.tight_layout()
# fig.savefig(os.path.join(savedir,'RRR_RRR_time_R2.png'), 
            # format = 'png', bbox_inches='tight')  

            
#%% plot the results:
narealabelpairs             = len(arealabelpairs)
clrs_arealabelpairs         = get_clr_area_pairs(arealabelpairs)
lstyle_hitmiss              = ['--','-']

bincenters = sbins[(bin_starts+window_size/2).astype(int)]
# fig,axes = plt.subplots(narealabelpairs,1,figsize=(3.5,2*narealabelpairs))
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3))
axes = axes.flatten()
for iapl, arealabelpair in enumerate(arealabelpairs):
    ax = axes[iapl]
    for ihitmiss in [0,1]:
        tempdata = dec_time_projcorr[iapl,ihitmiss,:]
        handles.append(ax.plot(bincenters,np.nanmean(tempdata,axis=0),
                                label=area, color=get_clr_areas([area]),linestyle=lstyle_hitmiss[ihitmiss])[0])
        ax.fill_between(bincenters,np.nanmean(tempdata,axis=0)-np.nanstd(tempdata,axis=0),
                        np.nanmean(tempdata,axis=0)+np.nanstd(tempdata,axis=0),alpha=0.3,color=get_clr_areas([area]))
    if iapl==0:
        ax.set_ylabel('R2 (RRR)')
        ax.legend(handles=handles,labels=['Miss','Hit'],frameon=False,fontsize=8,loc='upper left')
    ax.set_title(arealabelpair,fontsize=10,color=clrs_arealabelpairs[iapl])
    if iapl==narealabelpairs-1:
        ax.set_xticks(bincenters[::2])
        ax.set_xlabel('Position (cm)')
    else:
        ax.set_xticks(bincenters[::2],labels=[])
    add_stim_resp_win(ax)
    # ax.set_ylim([0,0.1])
    ax.set_xlim(np.percentile(bincenters,[0,100]))
sns.despine(top=True,right=True,offset=3)

# fig.suptitle('Time-resolved decoding of communication strength')
fig.tight_layout()
# fig.savefig(os.path.join(savedir,'RRR_RRR_time_R2.png'), 
            # format = 'png', bbox_inches='tight')  

#%% plot the results:
narealabelpairs             = len(arealabelpairs)
clrs_arealabelpairs         = get_clr_area_pairs(arealabelpairs)
lstyle_hitmiss              = ['--','-']
bincenters = sbins[(bin_starts+window_size/2).astype(int)]

fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,2*3),sharex=True,sharey=True)
# axes = axes.flatten()
for iapl, arealabelpair in enumerate(arealabelpairs):
    areasplit    = arealabelpair.split('-')
    for iarea,area in enumerate(areasplit):
        ax = axes[iarea,iapl]
        handles = []
        for ihitmiss in [0,1]:
            tempdata = dec_time_signalR2[iapl,iarea,ihitmiss,:]
            handles.append(ax.plot(bincenters,np.nanmean(tempdata,axis=0),
                                    label=area, color=get_clr_areas([area]),linestyle=lstyle_hitmiss[ihitmiss])[0])
            ax.fill_between(bincenters,np.nanmean(tempdata,axis=0)-np.nanstd(tempdata,axis=0),
                            np.nanmean(tempdata,axis=0)+np.nanstd(tempdata,axis=0),alpha=0.3,color=get_clr_areas([area]))
        if iapl==0:
            ax.set_ylabel('Signal R2')
            ax.legend(handles=handles,labels=['Miss','Hit'],frameon=False,fontsize=8,loc='upper left')
        ax.set_title(area,fontsize=10,color=clrs_arealabelpairs[iapl])
        if iapl==narealabelpairs-1:
            ax.set_xticks(bincenters[::2])
            ax.set_xlabel('Position (cm)')
        else:
            ax.set_xticks(bincenters[::2],labels=[])
        add_stim_resp_win(ax)
        # ax.set_ylim([0,0.1])
        ax.set_xlim(np.percentile(bincenters,[0,100]))
sns.despine(top=True,right=True,offset=3)

# fig.suptitle('Time-resolved decoding of communication strength')
fig.tight_layout()
# fig.savefig(os.path.join(savedir,'RRR_RRR_time_R2.png'), 
            # format = 'png', bbox_inches='tight')  

#%%


        #   dec_time_signalR2[iapl,0,iresp, ises, itw] = r2_x
        #             dec_time_neuralR2[iapl,0,iresp, ises, itw] = r2_y

        #             dec_time_signalR2[iapl,1,iresp, ises, itw] = ev_x
        #             dec_time_neuralR2[iapl,1,iresp, ises, itw] = ev_y

        #             dec_time_projcorr[iapl,iresp, ises, itw]   = np.corrcoef(p_x,p_y)[0,1]

