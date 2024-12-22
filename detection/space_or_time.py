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

# from loaddata import * #get all the loading data functions (filter_sessions,load_sessions)
from loaddata.session_info import filter_sessions,load_sessions

# from scipy import stats
from scipy.stats import zscore
from utils.psth import compute_tensor,compute_respmat,compute_tensor_space,compute_respmat_space
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import roc_auc_score as AUC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# from sklearn import preprocessing
# from sklearn import linear_model
# from sklearn.preprocessing import minmax_scale
# from scipy.signal import medfilt
# from sklearn.preprocessing import StandardScaler

from loaddata.get_data_folder import get_local_drive
# import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.patches
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import *
from utils.behaviorlib import * # get support functions for beh analysis 

# from matplotlib.lines import Line2D
# from utils.behaviorlib import * # get support functions for beh analysis 
# from detection.plot_neural_activity_lib import *

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - VR\\Stim\\'
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Alignment\\')
# savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - DN regression\\'

#%% 
# for i in range(nSessions):
#     sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -60  #pre cm
s_post      = 60   #post cm
binsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')

#%% #################### Spatial runspeed  ####################################
for ises,ses in enumerate(sessions):
    [sessions[ises].runPSTH,bincenters] = calc_runPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)
   
mu_runspeed = np.nanmean(sessions[ises].runPSTH[:,(bincenters>-10) & (bincenters<50)],axis=None)
print('Mean running speed: %1.2f cm/s' % mu_runspeed)
print('%2.1f cm bins would correspond to %1.2f sec bins' % (binsize,binsize/mu_runspeed))

#%% ############################### Time Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for spatial binning
t_pre       = -5  #pre sec
t_post      = 5   #post sec
binsize     = 0.6  #spatial binning in cm

for i in range(nSessions):
    sessions[i].tensor,tbins    = compute_tensor(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['tStimStart'],
                                       t_pre=t_pre,t_post=t_post,binsize=binsize,method='binmean')




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
'LPE12385_2024_06_15_7_0212', # hit specific activity?
'LPE12385_2024_06_15_1_0475', # very clean response
'LPE12385_2024_06_15_2_0099', # nice responses
'LPE12385_2024_06_15_2_0499'] #variable responsiveness

#%% 
example_cell_ids = np.random.choice(sessions[ises].celldata['cell_id'],size=8,replace=False)

#%% Show example neurons that are correlated either to the stimulus signal, lickresponse or to running speed:
# for iN,cell_id in np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]:
for icell,cell_id in enumerate(example_cell_ids):
    if np.isin(cell_id,sessions[ises].celldata['cell_id']):
        plot_neuron_spacetime_alignment(sessions[ises],cell_id,sbins,tbins)
        

#%% 
idx_N = np.where(np.isin(sessions[ises].celldata['cell_id'],example_cell_ids))[0]
# idx_N = sessions[ises].celldata['noise_level']<20

idx_T = np.isin(sessions[ises].trialdata['stimcat'],['C'])
idx_T = np.isin(sessions[ises].trialdata['stimcat'],['M'])
# idx_T = np.isin(sessions[ises].trialdata['stimcat'],['N'])
# idx_T = np.isin(sessions[ises].trialdata['lickResponse'],[1])

data = sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
std_space = np.nanmean(np.nanstd(data,axis=1),axis=0)
data = sessions[ises].tensor[np.ix_(idx_N,idx_T,np.ones(len(tbins)).astype(bool))]
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
ax.plot(tbins,std_time,color='b')
# ax.legend(frameon=False,fontsize=8,loc='upper left')
ax.set_xlim(tbins[0],tbins[-1])
ax.set_xlabel('Stimulus passing (sec)')
ax.set_yticklabels(axes[1].get_yticks())
ax.set_title('Time')
ax.grid(True)

