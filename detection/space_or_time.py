# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script analyzes whether spatial or temporal alignment of the neural 
activity captures the relevant feature encoding better.
"""

#%% Import packages
import os
os.chdir('c:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import zscore, ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
from loaddata.get_data_folder import get_local_drive
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import *
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.decode_lib import * # get support functions for decoding

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - VR\\Stim\\'
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Alignment\\')
# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - DN regression\\'

#%% 
session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
# session_list = np.array([['LPE12013', '2024_04_26']])
# session_list = np.array([['LPE11997', '2024_04_16']])
# session_list = np.array([['LPE12013', '2024_04_25']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions


#%% 
sessions,nSessions  = filter_sessions(protocol,load_behaviordata=True,load_calciumdata=True,
                                      calciumversion=calciumversion,min_cells=100)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Remove sessions LPE10884 that are too bad:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE10884_2023_12_14','LPE10884_2023_12_15','LPE10884_2024_01_11',
                                                                'LPE10884_2024_01_16','LPE11622_2024_02_22']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% 
# for i in range(nSessions):
#     sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -60  #pre cm
s_post      = 60   #post cm
sbinsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=sbinsize,method='binmean')

#%% #################### Spatial runspeed  ####################################
for ises,ses in enumerate(sessions):
    [sessions[ises].runPSTH,_] = calc_runPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=sbinsize)
   

#%% ############################### Time Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for spatial binning

# t_pre       = -5  #pre sec
# t_post      = 5   #post sec
# tbinsize     = 0.36  #spatial binning in cm

for ises in range(nSessions):

    mu_runspeed = np.nanmean(sessions[ises].runPSTH[:,(sbins>-10) & (sbins<50)],axis=None)
    t_pre       = s_pre/mu_runspeed  #pre sec
    t_post      = s_post/mu_runspeed   #post sec
    tbinsize     = sbinsize/mu_runspeed  #spatial binning in cm

    if len(np.arange(t_pre-tbinsize/2, t_post + tbinsize+tbinsize/2, tbinsize))-1 != len(sbins):
        tbinsize = tbinsize * 1.05

    print('Mean running speed: %1.2f cm/s' % mu_runspeed)
    print('%2.1f cm bins would correspond to %1.2f sec bins' % (sbinsize,tbinsize))

    sessions[ises].tensor,sessions[ises].tbins    = compute_tensor(sessions[ises].calciumdata,sessions[ises].ts_F,sessions[ises].trialdata['tStimStart'],
                                       t_pre=t_pre,t_post=t_post,binsize=tbinsize,method='binmean')

# len(sbins)
# for ises,ses in enumerate(sessions):
#     print(len(sessions[ises].tbins))

#%% 
def plot_neuron_spacetime_alignment(ses,cell_id,sbins,tbins):
    ### Plot activity over space or over time, side by side
    # for the same trials, same neuron
    iN = np.where(ses.celldata['cell_id']==cell_id)[0][0]
    K = np.shape(ses.stensor)[1]

    fig, axes = plt.subplots(1,2,figsize=(5,2.5))
    
    ax = axes[0]

    ttypes = pd.unique(ses.trialdata['trialOutcome'])
    colors = get_clr_outcome(ttypes)

    for k in range(K):
        ax.plot(sbins,ses.stensor[iN,k,:],color='grey',alpha=0.5,linewidth=0.5)

    for i,ttype in enumerate(ttypes):
        idx = ses.trialdata['trialOutcome']==ttype
        data_mean = np.nanmean(ses.stensor[iN,idx,:],axis=0)
        data_error = np.nanstd(ses.stensor[iN,idx,:],axis=0) #/ math.sqrt(sum(idx))
        ax.plot(sbins,data_mean,label=ttype,color=colors[i],linewidth=2)
        ax.fill_between(sbins, data_mean+data_error,  data_mean-data_error, alpha=.2, linewidth=0,color=colors[i])

    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax.set_xlim(sbins[0],sbins[-1])
    ax.set_xlabel('Position rel. to stimulus onset (cm)')
    ax.set_ylabel('Activity')

    ax = axes[1]

    for k in range(K):
        ax.plot(tbins,ses.tensor[iN,k,:],color='grey',alpha=0.5,linewidth=0.5)

    for i,ttype in enumerate(ttypes):
        idx = ses.trialdata['trialOutcome']==ttype
        data_mean = np.nanmean(ses.tensor[iN,idx,:],axis=0)
        data_error = np.nanstd(ses.tensor[iN,idx,:],axis=0) #/ math.sqrt(sum(idx))
        ax.plot(tbins,data_mean,label=ttype,color=colors[i],linewidth=2)
        ax.fill_between(tbins, data_mean+data_error,  data_mean-data_error, alpha=.2, linewidth=0,color=colors[i])

    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax.set_xlim(tbins[0],tbins[-1])
    ax.set_xlabel('Time rel. to stimulus onset (s)')
    # ax.set_ylabel('Activity')

    for ax in axes:
        # ylim = my_ceil(np.nanmax(ses.tensor[iN,:,:]),-1)
        smax = np.nanmax(ses.stensor[iN,:,:])
        tmax = np.nanmax(ses.tensor[iN,:,:])
        ylim = my_ceil(np.nanmax([smax,tmax]),-1)
        ax.set_ylim(0,ylim)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
        ax.axvline(x=45, color='b', linestyle='--', linewidth=1)
    # plt.text(3, ylim-3, 'Stim',fontsize=10)
    # plt.text(rewzonestart+3, ylim-3, 'Rew',fontsize=10)
    # plt.tight_layout()
    # plt.title(trialdata['session_id'][0],fontsize=10)
    plt.suptitle(cell_id,fontsize=11,y=0.96)
    plt.tight_layout()
    return fig


#%%
ises = 0

example_cell_ids = ['LPE12385_2024_06_15_0_0075',
'LPE12385_2024_06_15_0_0126',
'LPE12385_2024_06_15_0_0105', # reduction upon stimulus zone
'LPE12385_2024_06_15_0_0114', # noise trial specific response, very nice one
'LPE12385_2024_06_15_0_0183',
'LPE12385_2024_06_15_3_0016',
'LPE12385_2024_06_15_0_0031', # noise trial specific response
'LPE12385_2024_06_15_1_0075', # hit specific activity?
'LPE12385_2024_06_15_1_0475', # very clean response
'LPE12385_2024_06_15_2_0099', # nice responses
'LPE12385_2024_06_15_2_0499'] #variable responsiveness

example_cell_ids = ['LPE12013_2024_04_25_4_0187',
                    'LPE12013_2024_04_25_0_0007',
                    'LPE12013_2024_04_25_1_0046',
                    'LPE12013_2024_04_25_7_0161'] #

example_cell_ids = ['LPE12013_2024_04_26_2_0259',
                    'LPE12013_2024_04_26_2_0250',
                    'LPE12013_2024_04_26_0_0016',
                    'LPE12013_2024_04_26_7_0310',
                    'LPE12013_2024_04_26_0_0017'] #

example_cell_ids = ['LPE11997_2024_04_16_0_0013',
                    'LPE11997_2024_04_16_1_0001',
                    'LPE11997_2024_04_16_2_0047',
                    'LPE11997_2024_04_16_0_0134',
                    'LPE11997_2024_04_16_4_0115'] #


#%% 
example_cell_ids = np.random.choice(sessions[ises].celldata['cell_id'],size=8,replace=False)

#%% Show example neurons that are correlated either to the stimulus signal, lickresponse or to running speed:
# for iN,cell_id in np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]:
for icell,cell_id in enumerate(example_cell_ids):
    if np.isin(cell_id,sessions[ises].celldata['cell_id']):
        plot_neuron_spacetime_alignment(sessions[ises],cell_id,sbins,sessions[ises].tbins)
        plt.savefig(os.path.join(savedir,'ActivityInCorridor_SpaceVsTime_' + cell_id + '.png'), format = 'png')

#%% 
idx_N = np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]
# idx_N = sessions[ises].celldata['noise_level']<20

idx_T = np.isin(sessions[ises].trialdata['stimcat'],['C'])
idx_T = np.isin(sessions[ises].trialdata['stimcat'],['M'])
# idx_T = np.isin(sessions[ises].trialdata['stimcat'],['N'])
# idx_T = np.isin(sessions[ises].trialdata['lickResponse'],[1])

data = sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
std_space = np.nanmean(np.nanstd(data,axis=1),axis=0)
data = sessions[ises].tensor[np.ix_(idx_N,idx_T,np.ones(len(sessions[ises].tbins)).astype(bool))]
std_time = np.nanmean(np.nanstd(data,axis=1),axis=0)
# std_time = np.nanmean(np.nanstd(sessions[ises].tensor[idx,:,:],axis=1),axis=0)


# std_space = np.nanmean(np.nanstd(sessions[ises].stensor[idx,:,:],axis=1),axis=0)
# std_time = np.nanmean(np.nanstd(sessions[ises].tensor[idx,:,:],axis=1),axis=0)

#%% Compare the variability in responses across spatial or temporal bins:
fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharey=True)
ax = axes[0]
ax.plot(sbins,std_space,color='r')
ax.set_xlim(sbins[0],sbins[-1])
ax.set_xlabel('Stimulus start (cm)')
ax.set_ylabel('Std. trials')
ax.set_title('Space')
ax.grid(True)

ax = axes[1]
ax.plot(sessions[ises].tbins,std_time,color='b')
# ax.legend(frameon=False,fontsize=8,loc='upper left')
ax.set_xlim(sessions[ises].tbins[0],sessions[ises].tbins[-1])
ax.set_xlabel('Stimulus passing (sec)')
ax.set_yticklabels(axes[1].get_yticks())
ax.set_title('Time')
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'Var_Comparison_%s.png') % sessions[ises].sessiondata['session_id'][0], format='png')


#%% Decoding performance across space or across time:

sperf = np.full((nSessions,len(sbins)), np.nan)
tperf = np.full((nSessions,len(sbins)), np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding response across sessions'):
    # idx_T = np.isin(ses.trialdata['stimcat'],['C','M'])
    idx_T = np.isin(ses.trialdata['stimcat'],['C','N'])
    idx_N = ses.celldata['roi_name']=='V1'

    if np.sum(idx_T) > 50:
        # Get the maximum signal vs catch for this session
        y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()

        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        X = np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,((sbins>-5) & (sbins<20)).astype(bool))],axis=2)
        X = X.T

        X = X[:,~np.all(np.isnan(X),axis=0)] #
        idx_nan = ~np.all(np.isnan(X),axis=1)
        X = X[idx_nan,:]
        y = y[idx_nan]
        X[np.isnan(X)] = np.nanmean(X, axis=None)
        X = zscore(X, axis=1)
        X[np.isnan(X)] = np.nanmean(X, axis=None)

        optimal_lambda = find_optimal_lambda(X,y,model_name='LogisticRegression',kfold=5)
        # Loop through each spatial bin
        for ibin, bincenter in enumerate(sbins):
            y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
            X = ses.stensor[np.ix_(idx_N,idx_T,sbins==bincenter)].squeeze()
            X = X.T

            X = X[:,~np.all(np.isnan(X),axis=0)] #
            idx_nan = ~np.all(np.isnan(X),axis=1)
            X = X[idx_nan,:]
            y = y[idx_nan]
            X[np.isnan(X)] = np.nanmean(X, axis=None)
            X = zscore(X, axis=1)
            X[np.isnan(X)] = np.nanmean(X, axis=None)

            # Calculate the average decoding performance across folds
            sperf[ises,ibin] = my_classifier_wrapper(X,y,model_name='LogisticRegression',kfold=5,lam=optimal_lambda,norm_out=True)

        # Loop through each time bin
        for ibin, bincenter in enumerate(sessions[ises].tbins):
            y = (ses.trialdata['stimcat'][idx_T] == 'C').to_numpy()
            X = ses.tensor[np.ix_(idx_N,idx_T,sessions[ises].tbins==bincenter)].squeeze()
            X = X.T

            X = X[:,~np.all(np.isnan(X),axis=0)] #
            idx_nan = ~np.all(np.isnan(X),axis=1)
            X = X[idx_nan,:]
            y = y[idx_nan]
            X[np.isnan(X)] = np.nanmean(X, axis=None)
            X = zscore(X, axis=1)
            X[np.isnan(X)] = np.nanmean(X, axis=None)

            # Calculate the average decoding performance across folds
            tperf[ises,ibin] = my_classifier_wrapper(X,y,model_name='LogisticRegression',kfold=5,lam=optimal_lambda,norm_out=True)


#%% Show the decoding performance
fig,axes = plt.subplots(1,3,figsize=(8,3))
ax = axes[0]
for i,ses in enumerate(sessions):
    if np.any(sperf[i,:]):
        # ax.plot(sbins,sperf[i,:],color='grey',alpha=0.5,linewidth=1)
        ax.plot(sbins,sperf[i,:],alpha=0.5,linewidth=1)
# shaded_error(sbins,sperf,error='sem',ax=ax,color='grey')
ax.plot(sbins,np.nanmean(sperf,axis=0),alpha=1,linewidth=2,color='k')
ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Position relative to stim (cm)')
ax.set_ylabel('Decoding Performance \n (accuracy - shuffle)')
ax.set_title('Space')
ax.set_xlim([-60,60])
ax.set_ylim([-0.1,1])

ax = axes[1]
for i,ses in enumerate(sessions):
    if np.any(tperf[i,:]):
        # ax.plot(sbins,sperf[i,:],color='grey',alpha=0.5,linewidth=1)
        ax.plot(np.arange(len(sbins)),tperf[i,:],alpha=0.5,linewidth=1)
ax.plot(np.arange(len(sbins)),np.nanmean(tperf,axis=0),alpha=1,linewidth=2,color='k')
# shaded_error(np.arange(len(sbins)),tperf,error='sem',ax=ax,color='grey')
ax.axvline(x=np.where(sbins==0)[0], color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Time relative to stim (sec)')
ax.set_ylabel('Decoding Performance \n (accuracy - shuffle)')
ax.set_title('Time')
ax.set_ylim([-0.1,1])
ax.set_xticks(np.arange(len(sbins))[::3])
ax.set_xticklabels(np.round(sessions[ises].tbins[::3]))

ax = axes[2]
# data = np.vstack((np.nanmean(sperf[:,[6,7]],axis=1),np.nanmean(tperf[:,[6,7]],axis=1)))
data = np.vstack((np.nanmean(sperf[:,[7,8]],axis=1),np.nanmean(tperf[:,[7,8]],axis=1)))
ax.plot(data,marker='o',linestyle='-',markersize=6)
ax.set_xticks([0,1])
ax.set_xticklabels(['Space','Time'])
ax.set_yticks(np.arange(0,1.1,0.1))
ax.set_ylim([0.3,1])

t, p = ttest_rel(data[0,:], data[1,:])
# ax.text(0.5,0.8,f'p={p:.2f}',ha='center',va='center',fontsize=9)
ax.text(0.5,0.95,f'p={p:.2f}',ha='center',va='center',fontsize=9)
# ax.set_ylabel('Decoding Performance \n (accuracy - shuffle)')
ax.set_title('Performance Stim Window',fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'DecodingPerformance_NoiseVsCatch.png'), format='png')
# plt.savefig(os.path.join(savedir, 'DecodingPerformance_MaxVsCatch.png'), format='png')

#%% ############################### Spatial Resp Mat #################################
for i in range(nSessions):
    sessions[i].srespmat = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                                sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)
    sessions[i].trespmat = compute_respmat(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['tStimStart'],
                                                t_resp_start=0,t_resp_stop=0.6,method='mean',subtr_baseline=False)

#%% ############################### Correlation Matrix###############################
for i in range(nSessions):


    sessions[i].noise_corr_s = np.corrcoef(sessions[i].srespmat)
    sessions[i].noise_corr_t = np.corrcoef(sessions[i].trespmat)

#%% 
sesidx = 2
# show the correlation matrix for spatial and temporal binning side by side for one session:
fig,ax = plt.subplots(1,2,figsize=(8,4))
ax[0].imshow(sessions[sesidx].noise_corr_s,vmin=-0.1,vmax=0.1,cmap='bwr')
ax[0].set_title('Spatial')
ax[1].imshow(sessions[sesidx].noise_corr_t,vmin=-0.1,vmax=0.1,cmap='bwr')
ax[1].set_title('Temporal')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'CorrelationMatrix_SpatialVsTemporal.png'), format='png')

#%% 
r_st = np.empty(nSessions)
for i in range(nSessions):
    r_st[i] = np.corrcoef(sessions[i].noise_corr_s.flatten(),sessions[i].noise_corr_t.flatten())[0,1]
print('Correlation between spatial and temporal noise correlation: %1.2f' % np.mean(r_st))

