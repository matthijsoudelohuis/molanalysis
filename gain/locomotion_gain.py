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
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.corr_lib import *
from utils.tuning import *
from utils.gain_lib import * 
from scipy.stats import binned_statistic,binned_statistic_2d

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\')

#%% #############################################################################
session_list        = np.array([['LPE10919_2023_11_06']])
session_list        = np.array([['LPE09665_2023_03_21'], #GR
                                ['LPE10919_2023_11_06']]) #GR

session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10884_2023_10_20']]) #GR
# session_list        = np.array([['LPE12223','2024_06_10']])

sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_areas=['V1','PM']) 

sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM']) 

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)


#%% ############################### Show response with and without running #################

celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

thr_still   = 0.5 #max running speed for still trials
thr_moving  = 1   #min running speed for moving trials

nOris       = 16
nCells      = len(celldata)
mean_resp_speedsplit = np.empty((nCells,nOris,2))

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    idx_trials_still    = sessions[ises].respmat_runspeed<thr_still
    idx_trials_moving   = sessions[ises].respmat_runspeed>thr_moving

    # compute meanresp
    oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
    ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)

    meanresp    = np.empty([N,len(oris),2])
    meanresp_nogain  = np.empty([N,len(oris),2])
    for i,ori in enumerate(oris):
        meanresp[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_still)],axis=1)
        meanresp[:,i,1] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_moving)],axis=1)
        
    prefori                     = np.argmax(meanresp[:,:,0],axis=1)
    # prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

    meanresp_pref          = meanresp.copy()
    for n in range(N):
        meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
        meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

    # normalize by peak response during still trials
    tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
    meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
    meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

    # meanresp_pref
    idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'])
    mean_resp_speedsplit[idx_ses,:,:] = meanresp_pref

#%% ########### Make the figure ##################################################################
redcells            = np.unique(celldata['redcell'])
redcell_labels      = ['unl','lab']
areas               = ['V1','PM']
clrs_areas          = get_clr_areas(areas)

fig,axes = plt.subplots(2,2,figsize=(4,4),sharex=True,sharey=True)
for iarea,area in enumerate(areas):
    for ired,redcell in enumerate(redcells):
        ax = axes[iarea,ired]
        idx_neurons = celldata['redcell']==redcell
        idx_neurons = np.logical_and(idx_neurons,celldata['roi_name']==area)
        idx_neurons = np.logical_and(idx_neurons,celldata['tuning_var']>0.05)
        handles = []
        handles.append(shaded_error(ax=ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,0],center='mean',error='sem',color='black'))
        handles.append(shaded_error(ax=ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,1],center='mean',error='sem',color='red'))
        if ired==0 and iarea==0:
            ax.legend(handles=handles,labels=['Still','Running'],frameon=False,loc='upper right')
        if iarea==1: 
            ax.set_xlabel(u'Δ Pref Ori')
        if ired==0: 
            ax.set_ylabel('Normalized Response')
        ax.set_title('%s%s' % (area,redcell_labels[ired]))
        ax.set_ylim([0,3.2])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
axes[1,0].set_xticks(oris[::2],oris[::2],rotation=45,fontsize=6)
axes[1,1].set_xticks(oris[::2],oris[::2],rotation=45,fontsize=6)

# my_savefig(fig,savedir,'RunningModulation_V1PM_LabUnl_' + str(nSessions) + 'sessions')

#%% Is the gain modulation similar for labeled and unlabeled, and for V1 and PM?
arealabels = ['V1unl','V1lab','PMunl','PMlab']
clrs_arealabels = get_clr_area_labeled(arealabels)
narealabels = len(arealabels)
# 
# redcells            = np.unique(celldata['redcell'])
# redcell_labels      = ['Unl','Lab']
# areas               = ['V1','PM']
mrkrs_arealabels      = ['o','+','o','+']
mrkrs_arealabels      = ['o','o','+','+']
data_gainregress_mean = np.full((narealabels,3),np.nan)

fig,ax = plt.subplots(1,1,figsize=(3,3))
for ial,arealabel in enumerate(arealabels):
    # ax = axes[iarea,ired]
    idx_N = np.all((celldata['arealabel']==arealabel,
                    celldata['tuning_var']>0.05),axis=0)

    xdata = np.nanmean(mean_resp_speedsplit[idx_N,:,0],axis=0)
    ydata = np.nanmean(mean_resp_speedsplit[idx_N,:,1],axis=0)
    b = linregress(xdata,ydata)
    data_gainregress_mean[ial,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = data_gainregress_mean[ial,0]*xvals + data_gainregress_mean[ial,1]
    ax.plot(xvals,yvals,color=clrs_arealabels[ial],linewidth=0.3)
    ax.scatter(xdata,ydata,color=clrs_arealabels[ial],marker=mrkrs_arealabels[ial],label=arealabel,alpha=0.6,s=25)
    ax.plot([0,3],[0,3],'grey',ls='--',linewidth=1)
ax.legend(frameon=False,loc='lower right')
ax.set_xlabel('Still (Norm. Response)')
ax.set_ylabel('Running (Norm. Response)')
ax.set_xlim([0,3.5])
ax.set_ylim([0,3.5])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# fig.savefig(os.path.join(savedir,'SharedGain','Gain_V1PM_LabUnl_' + str(nSessions) + 'sessions.png'), format = 'png')

#%% Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
N = len(celldata)
data_gainregress = np.full((N,3),np.nan)
for iN in tqdm(range(N),total=N,desc='Fitting gain for each neuron'):
    b = linregress(mean_resp_speedsplit[iN,:,0],mean_resp_speedsplit[iN,:,1])
    data_gainregress[iN,:] = b[:3]

#%% Show some neurons
nbyn = 3
# neuronsel = np.random.choice(np.where(celldata['tuning_var']>0.05)[0],size=25,replace=False)
neuronsel = np.random.choice(np.where(data_gainregress[:,2]>0.75)[0],size=nbyn**2,replace=False)
fig, axes = plt.subplots(nbyn,nbyn,figsize=(nbyn*1.5,nbyn*1.5),sharex=True,sharey=True)
for iN,N in enumerate(neuronsel):
    ax = axes[iN//nbyn,iN%nbyn]
    ax.plot(mean_resp_speedsplit[N,:,0],mean_resp_speedsplit[N,:,1],'.',color='black',alpha=0.8)
    ax.plot([0,3],[0,3],'grey',ls='--',linewidth=1)
    xvals = np.arange(-1,3,0.1)
    yvals = data_gainregress[N,0]*xvals + data_gainregress[N,1]
    ax.plot(xvals,yvals,linewidth=0.3,color='blue')
ax.set_xlim([-0.25,3])
ax.set_ylim([-0.25,3])
ax.set_xticks([0,3])
ax.set_yticks([0,3])
axes[nbyn//2,0].set_ylabel('Running (Norm. Response)')
axes[nbyn-1,nbyn//2].set_xlabel('Still (Norm. Response)')
sns.despine(fig=fig, top=True, right=True,offset=3)
fig.suptitle('Gain Modulation - Individual neurons')
fig.savefig(os.path.join(savedir,'Gain_ExampleNeurons' + str(nSessions) + 'sessions.png'), 
            bbox_inches='tight',format = 'png')

#%%
arealabels          = ['V1unl','V1lab','PMunl','PMlab']
clrs_arealabels     = get_clr_area_labeled(arealabels)
minrvalue           = 0.2
mintuningvar        = 0.0

df                  = pd.DataFrame(np.c_[data_gainregress, celldata[['arealabel', 'session_id']]], columns=['slope', 'intercept', 'rvalue', 'arealabel', 'session_id'])

idx_N               = np.all((celldata['tuning_var']>mintuningvar,
                    np.isin(celldata['arealabel'],arealabels),
                        data_gainregress[:,2]>minrvalue),axis=0)

df = df[idx_N]

# Convert categorical variables to categorical type
df['arealabel'] = df['arealabel'].astype('category')
df['session_id'] = df['session_id'].astype('category')
df['slope'] = pd.to_numeric(df['slope'], errors='coerce')
df['intercept'] = pd.to_numeric(df['intercept'], errors='coerce')
df['rvalue'] = pd.to_numeric(df['rvalue'], errors='coerce')

testpairs   = [('V1unl','V1lab'),
             ('PMunl','PMlab'),
             ('V1unl','PMunl')]
fig,axes = plt.subplots(1,2,figsize=(6,4))
for ivar,var in enumerate(['slope','intercept']):
    ax = axes[ivar]
    sns.violinplot(data=df,x='arealabel',y=var,ax=ax,palette=clrs_arealabels,order=arealabels,hueorder=arealabels,inner=None)

    for ial,arealabel in enumerate(arealabels):
        xdata = df[var][df['arealabel']==arealabel]

        # # sns.violinplot(x=[ial] * len(xdata),y=xdata,ax=ax,color=clrs_arealabels[ial],inner=None,position=ial)
        # sns.violinplot(x=ial,y=xdata,ax=ax,color=clrs_arealabels[ial],inner=None)
        median = np.median(xdata)
        q25    = np.percentile(xdata,25)
        q75    = np.percentile(xdata,75)
        ax.plot([ial,ial],[q25,q75],linestyle='-',color='k',alpha=0.5)
        ax.plot([ial],[median],marker='o',color='k',alpha=0.5)
    if ivar==0:
        ax.axhline(1,color='k',ls='--',alpha=0.5)
    if ivar==1:
        ax.axhline(0,color='k',ls='--',alpha=0.5)
    ax.set_title(var)
    ax.set_xticks(np.arange(4),arealabels,fontsize=8,rotation=45)
    ax.set_ylim([-1,np.percentile(df[var],98)])
    ax.set_ylabel('')

    for itp,(area1,area2) in enumerate(testpairs):
        xdata = df[var][df['arealabel'].isin([area1,area2])]
        t,p = stats.ttest_ind(xdata[df['arealabel']==area1],xdata[df['arealabel']==area2])
        area1_loc = arealabels.index(area1)
        area2_loc = arealabels.index(area2)
        add_stat_annotation(ax, area1_loc, area2_loc, np.percentile(df[var],95)+itp*0.25, p, h=0.25, 
                            size = 12,color='k')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,savedir,'GainPopulation_V1PM_LabUnl_' + str(nSessions) + 'sessions',formats=['png'])


#%% Is the gain modulation fully mediated by increase in population rate: 
# Show mean affine modulation for solists and choristers: 
nbins                   = 5
binedges_popcoupling    = np.percentile(celldata['pop_coupling'],np.linspace(0,100,nbins+1))
clrs_popcoupling        = sns.color_palette('magma',nbins)

handles = np.empty(nbins,dtype=object)
fig, ax = plt.subplots(1,1,figsize=(4,3.5))
for ibin in range(len(binedges_popcoupling)-1):
    idx_N = np.all((celldata['pop_coupling']>binedges_popcoupling[ibin],
                  celldata['pop_coupling']<binedges_popcoupling[ibin+1]),axis=0)
    
    # ax = axes[iN//nbyn,iN%nbyn]
    xdata = np.mean(mean_resp_speedsplit[idx_N,:,0],axis=0)
    ydata = np.mean(mean_resp_speedsplit[idx_N,:,1],axis=0)
    ax.plot(xdata,ydata,'.',color=clrs_popcoupling[ibin],alpha=0.8)
    ax.plot([-1,13],[-1,13],'grey',ls='--',linewidth=1)
    b = linregress(xdata,ydata)
    # data_gainregress[iN,:] = b[:3]
    # yvals = data_gainregress[N,0]*xvals + data_gainregress[N,1]

    xvals = np.arange(-1,1.5,0.1)
    yvals = b[0]*xvals + b[1]
    handles[ibin] = ax.plot(xvals,yvals,linewidth=0.75,color=clrs_popcoupling[ibin])[0]
ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))
ax.set_xlim([-0.25,4])
ax.set_ylim([-0.25,4])
ax.set_xlabel('Still (Norm. Response)')
ax.set_ylabel('Running (Norm. Response)')
ax.legend(handles,['0-20%','20-40%','40-60%','60-80%','80-100%'],
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
                    reverse=True,fontsize=9,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')

sns.despine(fig=fig, top=True, right=True,offset=0,trim=True)
fig.suptitle('Gain Modulation - Soloists vs Choristers')
my_savefig(fig,savedir,'AffineModulation_Locomotion' + str(nSessions),formats = ['png'])

# fig.savefig(os.path.join(savedir,'Gain_ExampleNeurons' + str(nSessions) + 'sessions.png'), 
#             bbox_inches='tight',format = 'png')

# from utils.pair_lib import value_matching

#%% ############################### Show response with and without running #################

celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

thr_still   = 0.5 #max running speed for still trials
thr_moving  = 1   #min running speed for moving trials

nOris       = 16
nCells      = len(celldata)
mean_resp_speedsplit_ratebalanced = np.empty((nCells,nOris,2))

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
    
    idx_N           = sessions[ises].celldata['roi_name'] == 'V1'
    resp            = zscore(sessions[ises].respmat[idx_N,:].T,axis=0)
    poprate         = np.mean(resp, axis=1)

    # idx_N = sessions[ises].celldata['roi_name'] == 'V1'
    # resp                = zscore(sessions[ises].respmat.T,axis=0)
    # poprate             = np.mean(resp, axis=1)
    
    idx_trials_still    = sessions[ises].respmat_runspeed<thr_still
    idx_trials_moving   = sessions[ises].respmat_runspeed>thr_moving
    idx                 = np.concatenate((np.where(idx_trials_still)[0],np.where(idx_trials_moving)[0]))
    group               = np.concatenate((np.zeros(np.sum(idx_trials_still)),np.ones(np.sum(idx_trials_moving))))
    values              = np.concatenate((poprate[idx_trials_still],poprate[idx_trials_moving]))
    idx_subsampled      = value_matching(idx,group,values,bins=50,showFig=True)
    
    idx_trials_still[np.setdiff1d(np.arange(K),idx_subsampled)] = False
    idx_trials_moving[np.setdiff1d(np.arange(K),idx_subsampled)] = False

    meanresp    = np.empty([N,len(oris),2])
    meanresp_nogain  = np.empty([N,len(oris),2])
    for i,ori in enumerate(oris):
        meanresp[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_still)],axis=1)
        meanresp[:,i,1] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_moving)],axis=1)
        # meanresp[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_still)],axis=1)
        # meanresp[:,i,1] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_moving)],axis=1)
        
    prefori                     = np.argmax(meanresp[:,:,0],axis=1)
    # prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)
    meanresp_pref               = meanresp.copy()
    for n in range(N):
        meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
        meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

    # normalize by peak response during still trials
    tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
    meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
    meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

    # meanresp_pref
    idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'])
    mean_resp_speedsplit_ratebalanced[idx_ses,:,:] = meanresp_pref

#%% Is the gain modulation fully mediated by increase in population rate: 
# Show mean affine modulation for solists and choristers: 
nbins                   = 5
binedges_popcoupling    = np.percentile(celldata['pop_coupling'],np.linspace(0,100,nbins+1))
clrs_popcoupling        = sns.color_palette('magma',nbins)

handles = np.empty(nbins,dtype=object)
fig, ax = plt.subplots(1,1,figsize=(4,3.5))
for ibin in range(len(binedges_popcoupling)-1):
    idx_N = np.all((celldata['pop_coupling']>binedges_popcoupling[ibin],
                  celldata['pop_coupling']<binedges_popcoupling[ibin+1]),axis=0)
    
    # ax = axes[iN//nbyn,iN%nbyn]
    xdata = np.mean(mean_resp_speedsplit_ratebalanced[idx_N,:,0],axis=0)
    ydata = np.mean(mean_resp_speedsplit_ratebalanced[idx_N,:,1],axis=0)
    ax.plot(xdata,ydata,'.',color=clrs_popcoupling[ibin],alpha=0.8)
    ax.plot([-1,13],[-1,13],'grey',ls='--',linewidth=1)
    b = linregress(xdata,ydata)
    # data_gainregress[iN,:] = b[:3]
    # yvals = data_gainregress[N,0]*xvals + data_gainregress[N,1]

    xvals = np.arange(-1,1.5,0.1)
    yvals = b[0]*xvals + b[1]
    handles[ibin] = ax.plot(xvals,yvals,linewidth=0.75,color=clrs_popcoupling[ibin])[0]
ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))
ax.set_xlim([-0.25,4])
ax.set_ylim([-0.25,4])
ax.set_xlabel('Still (Norm. Response)')
ax.set_ylabel('Running (Norm. Response)')
ax.legend(handles,['0-20%','20-40%','40-60%','60-80%','80-100%'],
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
                    reverse=True,fontsize=9,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')

sns.despine(fig=fig, top=True, right=True,offset=0,trim=True)
fig.suptitle('Gain Modulation - Soloists vs Choristers')
my_savefig(fig,savedir,'AffineModulation_Locomotion_BalanceRate' + str(nSessions),formats = ['png'])

#%% 
idx_N = np.all((celldata['roi_name']=='V1',
                  celldata['noise_level']<20),axis=0)

xdata = np.mean(mean_resp_speedsplit[idx_N,:,0],axis=0)
ydata = np.mean(mean_resp_speedsplit[idx_N,:,1],axis=0)
b0 = linregress(xdata,ydata)

xdata = np.mean(mean_resp_speedsplit_ratebalanced[idx_N,:,0],axis=0)
ydata = np.mean(mean_resp_speedsplit_ratebalanced[idx_N,:,1],axis=0)
b1 = linregress(xdata,ydata)


print('By controlling for population activity, the locomotion gain modulation is reduced from:\n\
      Slope(original): %1.3f\nSlope(rate balanced): %1.3f\n\
      Offset(original): %1.3f\nOffset(rate balanced): %1.3f\n'
      % (b0[0],b1[0],b0[1],b1[1]))

#%% 

#%% is affine modulation only present if a neuron is responsive / tuned: 

celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

celldata['gain_slope']      = data_gainregress[:,0]
celldata['gain_intercept']  = data_gainregress[:,1]
celldata['gain_rvalue']     = data_gainregress[:,2]

# df = celldata[celldata['gain_rvalue']>0.1]
df = celldata[celldata['noise_level']<20]

fig,axes = plt.subplots(1,3,figsize=(9,3),sharex=True)
# xfield = 'tuning_var'
xfield = 'gOSI'

ax = axes[0]
# sns.scatterplot(data=df,x=xfield,y='gain_slope',ax=ax,marker='.',color='black',alpha=0.1)
sns.regplot(data=df,x=xfield,y='gain_slope',ax=ax,marker='o',color='black',
            scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'},robust=True,ci=None)
ax.set_ylim([0,np.nanpercentile(df['gain_slope'],97)])
ax.axhline(1,ls='--',color='red',alpha=0.5)

ax = axes[1]
sns.regplot(data=df,x=xfield,y='gain_intercept',ax=ax,marker='o',color='black',
            scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'},robust=True,ci=None)
# sns.scatterplot(data=df,x=xfield,y='gain_intercept',ax=ax,marker='.',color='black',alpha=0.1)
ax.set_ylim(np.nanpercentile(df['gain_intercept'],[1,97]))
ax.axhline(0,ls='--',color='red',alpha=0.5)

ax = axes[2]
sns.regplot(data=df,x=xfield,y='gain_rvalue',ax=ax,marker='o',color='black',
            scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'},logx=True,ci=None)
            # scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'},lowess=True,ci=None)
# sns.scatterplot(data=df,x=xfield,y='gain_rvalue',ax=ax,marker='.',color='black',alpha=0.1)
ax.set_ylim([0,1])

ax.set_xlim([0,1])
ax.set_xticks([0,0.5,1])

sns.despine(fig=fig, top=True, right=True, offset=5,trim=True)
plt.tight_layout()
my_savefig(fig,savedir,'GainModulation_vs_Tuning_%s_%dsessions' % (xfield,nSessions) ,formats=['png'])

#%% Subtracting gain removes tuned gain modulation in mean response:
redcells            = np.unique(celldata['redcell'])
redcell_labels      = ['Unl','Lab']
areas               = ['V1','PM']
clrs_areas          = get_clr_areas(areas)

fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharex=True,sharey=True)
for imodel,model in enumerate(['orig','nogain']):
    ax = axes[imodel]
    idx_neurons = celldata['tuning_var']>0.05
    handles = []
    handles.append(shaded_error(ax=ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,0,imodel],center='mean',error='sem',color='black'))
    handles.append(shaded_error(ax=ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,1,imodel],center='mean',error='sem',color='red'))
    if imodel==1:
        ax.legend(handles=handles,labels=['Still','Running'],fontsize=9,frameon=False,loc='upper right')
    ax.set_xlabel(u'Δ Pref Ori')
    ax.set_xticks(oris[::2],oris[::2],rotation=45)
    ax.set_ylabel('Normalized Response')
    ax.set_title(model)
    ax.set_ylim([-0.6,3])
    ax.axhline(0,ls='--',color='grey')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'SharedGain','RunningMod_Gainsub_' + str(nSessions) + 'sessions.png'), format = 'png')




