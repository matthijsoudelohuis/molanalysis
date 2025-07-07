# -*- coding: utf-8 -*-
"""
This script analyzes responses to visual gratings in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\')

#%% Load an example session: 
session_list        = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion=calciumversion,keepraw=False)
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,keepraw=True)

t_axis = sessions[0].t_axis

#%% compute tuning metrics:
idx_resp = (t_axis>=0.5) & (t_axis<=1.5)
sessions[0].respmat = np.nanmean(sessions[0].tensor[:,:,idx_resp],axis=2)
sessions = compute_tuning_wrapper(sessions)

#%% Add how neurons are coupled to the population rate: 
for ses in tqdm(sessions,desc='Computing tuning metrics for each session'):
    resp        = zscore(ses.respmat.T,axis=0)
    poprate     = np.mean(resp, axis=1)
    # popcoupling = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(N)]

    ses.celldata['pop_coupling']                          = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(len(ses.celldata))]

#%% Concatenate celldata across sessions:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

n_example_cells = 10
# idx_N = np.all((sessions[0].celldata['OSI'] > np.percentile(sessions[0].celldata['OSI'],60),
idx_N = np.all((sessions[0].celldata['tuning_var'] > np.percentile(sessions[0].celldata['tuning_var'],60),
    sessions[0].celldata['pop_coupling'] > np.percentile(sessions[0].celldata['pop_coupling'],80)),axis=0)
neuronsel = np.random.choice(np.where(idx_N)[0],n_example_cells,replace=False)

idx_N = np.all((sessions[0].celldata['noise_level'] < np.percentile(sessions[0].celldata['noise_level'],20),
    sessions[0].celldata['pop_coupling'] > np.percentile(sessions[0].celldata['pop_coupling'],80)),axis=0)
neuronsel = np.random.choice(np.where(idx_N)[0],n_example_cells,replace=False)


#%%



#%% Show some tuned responses with calcium and deconvolved traces across orientations:
fig = plot_tuned_response(sessions[0].tensor,sessions[0].trialdata,t_axis,neuronsel,plot_n_trials=10)
fig.suptitle('%s - dF/F' % sessions[0].session_id,fontsize=12)
# save the figure
fig.savefig(os.path.join(savedir,'TunedResponse_dF_%s.png' % sessions[0].session_id))

#%% 
poprate = np.mean(sessions[0].calciumdata, axis=1)
trialsel = [10,50]
trialsel = [60,80]
trialsel = [680,700]
trialsel = [690,750]
trialsel = [2510,2560]

tstart  = ses.trialdata['tOffset'][trialsel[0]-1]
tstop   = ses.trialdata['tOnset'][trialsel[1]-1]

idx_T = np.logical_and(
        ses.ts_F > tstart, ses.ts_F < tstop)

fig = plt.figure(figsize=(5,2))
plt.plot(poprate[idx_T],linewidth=0.5)
my_savefig(fig,savedir,'Poprate_example_%s' % sessions[0].session_id,formats=['pdf'])

#%% 
fig = plot_excerpt(sessions[0],trialsel=trialsel,neuronsel=neuronsel,plot_neural=True,
             plot_behavioral=False,neural_version='traces')
my_savefig(fig,savedir,'Rate_exampleneurons_%s' % sessions[0].session_id,formats=['pdf'])


#%% 

# resp        = zscore(ses.respmat.T,axis=0)
# poprate     = np.mean(resp, axis=1)
nPopRateBins = 10

poprate = np.mean(sessions[0].calciumdata.apply(zscore), axis=1)
popratequantiles = np.quantile(poprate,np.arange(0,1.1,1/nPopRateBins))


#%% 
clrs_popcoupling = sns.color_palette('magma',nPopRateBins)
binwidth = 0.01

fig,ax = plt.subplots(1,1,figsize=(4,2.5))
for iqrpopcoupling in range(len(clrs_popcoupling)):
    # sns.histplot(poprate[(poprate>popratequantiles[iqrpopcoupling]],edgecolor='none',color=clrs_popcoupling[iqrpopcoupling],
    #              bins=np.arange(-0.5,1,binwidth),ax=ax,fill=True)
    sns.histplot(poprate,edgecolor='none',color=clrs_popcoupling[iqrpopcoupling],
                #  bins=np.arange(popratequantiles[iqrpopcoupling]-binwidth/2,popratequantiles[iqrpopcoupling+1]+binwidth/2,binwidth),ax=ax,fill=True)
                 bins=np.arange(popratequantiles[iqrpopcoupling],popratequantiles[iqrpopcoupling+1]+binwidth/2,binwidth),ax=ax,fill=True)
                #  bins=np.arange(popratequantiles[iqrpopcoupling],popratequantiles[iqrpopcoupling+1],bins=20),ax=ax,fill=True)

plt.xlabel('Z-score')
plt.ylabel('Count')
plt.xlim([-0.5,1])
ax_nticks(ax,4)
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
                    reverse=True,fontsize=7,frameon=False,title='pop. rate bins',bbox_to_anchor=(0.7,1), loc='upper left')

my_savefig(fig,savedir,'Poprate_quantiles_%s' % sessions[0].session_id,formats=['png'])
