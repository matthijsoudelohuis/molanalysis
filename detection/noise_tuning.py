# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import sklearn
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn import svm as SVM
# from sklearn.metrics import accuracy_score, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score
from scipy.signal import medfilt
from scipy.stats import zscore, wilcoxon, ranksums, ttest_rel

from loaddata.session_info import *
from loaddata.get_data_folder import get_local_drive
import seaborn as sns
import matplotlib.pyplot as plt
from utils.psth import *
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *
from detection.example_cells import get_example_cells
from utils.regress_lib import *

plt.rcParams['svg.fonttype'] = 'none'

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\Encoding\\')


#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

# session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11622', '2024_02_21']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11997', '2024_04_16'],
                         ['LPE11622', '2024_02_21'],
                         ['LPE11998', '2024_04_30'],
                         ['LPE12013','2024_04_25']])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list        = np.array([['LPE12013','2024_04_25']])
# session_list        = np.array([['LPE12013','2024_04_26']])

sessions,nSessions = filter_sessions(protocols=protocol,only_session_id=session_list,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion,min_cells=100) #Load specified list of sessions

sessions,nSessions = filter_sessions(protocols=protocol,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion,min_cells=100) #Load specified list of sessions

#%% Z-score the calciumdata: 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -60  #pre cm
s_post      = 80   #post cm
binsize     = 10     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')
    # Compute average response in stimulus response zone:
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
                                    sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)


#%% 
sessions,nSessions,sbins = load_neural_performing_sessions()

sessions = noise_to_psy(sessions,filter_engaged=True,bootstrap=True)

#%% #################### Compute spatial runspeed ####################################
for ises,ses in enumerate(sessions): # running across the trial:
    sessions[ises].behaviordata['runspeed'] = medfilt(sessions[ises].behaviordata['runspeed'], kernel_size=51)
    [sessions[ises].runPSTH,_]     = calc_runPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)
    [sessions[ises].lickPSTH,_]    = calc_lickPSTH(sessions[ises],s_pre=s_pre,s_post=s_post,binsize=binsize)

#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

#%%
ises = 6
example_cell_ids = get_example_cells(sessions[ises].sessiondata['session_id'][0])

fig = plot_mean_activity_example_neurons(sessions[ises].stensor,sbins,sessions[ises],example_cell_ids)
# fig.savefig(os.path.join(savedir,'ExampleNeuronActivity_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')

#%%
sesid = 'LPE11997_2024_04_12'
# sesid = 'LPE11997_2024_04_16'
# sesid = 'LPE12385_2024_06_16'
sesid = 'LPE12013_2024_04_25'

sessiondata = pd.concat([ses.sessiondata for ses in sessions])
ises = np.where(sessiondata['session_id']==sesid)[0][0]

# ises = 1
example_cell_ids = get_example_cells(sessions[ises].sessiondata['session_id'][0])

# get some responsive cells: 
# idx                 = np.nanmean(sessions[ises].respmat,axis=1)>0.5
# idx                 = sessions[ises].celldata['sig_N']==1
# example_cell_ids = np.random.choice(sessions[ises].celldata['cell_id'][idx], size=9, replace=False)

fig = plot_noise_activity_example_neurons(sessions[ises],example_cell_ids)
# fig.savefig(os.path.join(savedir,'HitMiss_ExampleNeuronActivity_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png',bbox_inches='tight')



#%%  Include sessions based on performance: psychometric curve for the noise #############
sessiondata = pd.concat([ses.sessiondata for ses in sessions])
zmin_thr    = -0.3
zmax_thr    = 0.3
zmin_thr    = 0
zmax_thr    = 0
# guess_thr   = 0.5
guess_thr   = 0.4

idx_ses     = np.all((sessiondata['noise_zmin']<=zmin_thr,
                sessiondata['noise_zmax']>=zmax_thr,
                sessiondata['guess_rate']<=guess_thr),axis=0)
print('Filtered %d/%d DN sessions based on performance' % (np.sum(idx_ses),len(idx_ses)))

idx_N_perf = np.isin(celldata['session_id'],sessiondata['session_id'][idx_ses])



#%% 


# celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
# N           = len(celldata)

# lickresp    = [0,1]
# D           = len(lickresp)

# sigtype     = 'signal_psy'
# zmin        = -1
# zmax        = 1
# nbins_noise = 5
# Z           = nbins_noise + 2

# sigtype     = 'signal'
# zmin        = 7
# zmax        = 17
# nbins_noise = 5
# Z           = nbins_noise + 2

# edges       = np.linspace(zmin,zmax,nbins_noise+1)
# centers     = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
# plotcenters = np.hstack((centers[0]-2*np.mean(np.diff(centers)),centers,centers[-1]+2*np.mean(np.diff(centers))))

# S           = len(sbins)

# data_sig_spatial = np.full((N,Z,S),np.nan)
# data_sig_mean    = np.full((N,Z),np.nan)

# data_sig_hit_spatial = np.full((N,Z,D,S),np.nan)
# data_mean_hitmiss    = np.full((N,Z,D),np.nan)

# data_frac_spatial = np.full((N,Z,S),np.nan)
# data_frac_mean    = np.full((N,Z),np.nan)

# min_ntrials     = 5

sigtype = 'signal_psy'
zmin        = -0.5
zmax        = 0.5
nbins_noise = 3

sigtype     = 'signal'
zmin        = 5
zmax        = 20
nbins_noise = 5

data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,
                                                    splithitmiss=True,min_ntrials=5)


# #Do statistical test between hit and miss
# for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
#     idx_T           = np.all((sessions[ises].trialdata[sigtype]>=low,
#                             sessions[ises].trialdata[sigtype]<=high,
#                             sessions[ises].trialdata['lickResponse']==lr,
#                             sessions[ises].trialdata['engaged']==1), axis=0)
#         if np.sum(idx_T)>=min_ntrials:
#             data_mean_hitmiss_diff[idx_N_ses,ibin+1,ilr]        = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)


#%% 


#%% 
plt.plot(np.sum(np.isnan(data_mean_hitmiss[:,:,0]),axis=1))

plt.imshow(np.isnan(data_mean_hitmiss[:,:,0]),aspect='auto')
plt.imshow(np.isnan(data_mean_hitmiss[:,:,1]),aspect='auto')

#%% 
data = copy.deepcopy(data_mean_hitmiss.T)

print(np.shape(data))

# data = data - np.nanmin(data,axis=0,keepdims=True)
data = data - data[:,0][:,np.newaxis]

data = data / np.nanmax(data,axis=0,keepdims=True)

# plt.imshow(data,aspect='auto',cmap='Reds')
# plt.imshow(data,aspect='auto',cmap='bwr',vmin=-0.25,vmax=1.25)

#%% 
# plt.plot(plotcenters,np.nanmean(data_sig_mean,axis=0),color='k')

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

lickresp        = [0,1]
D               = len(lickresp)

plotcolors  	= ['blue','red']  # Start with black
plotlabels      = ['miss','hit']
markerstyles    = ['o','o']
plotlocs        = np.arange(np.shape(data_mean_hitmiss)[1])
posfilter = np.array([0,1,3,5,6])
posfilter = np.array([0,2,4])
siglabels = ['nonresponsive','responsive']
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)

for sig in [0,1]:
    ax = axes[sig]
    # idx_N = celldata['sig_N']==1
    # idx_N = celldata['sig_N']!=1

    idx_N = celldata['sig_MN']==sig
    # idx_N = celldata['sig_N']==sig
    for ilr,lr in enumerate(lickresp):
        # ax.plot(plotcenters[1:-1],np.nanmean(data_mean_hitmiss[idx_N,1:-1,ilr],axis=0),
                # marker='.',markersize=15,color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
        # ax.plot(plotcenters,np.nanmean(data_mean_hitmiss[idx_N,:,ilr],axis=0),color=plotcolors[ilr], 
                #  marker='.',markersize=15,label=plotlabels[ilr],linewidth=2)
        ax.plot(plotlocs,np.nanmean(data_mean_hitmiss[idx_N,:,ilr],axis=0),color=plotcolors[ilr], 
                 marker='.',markersize=15,label=plotlabels[ilr],linewidth=2)
    
    pvals = ttest_rel(data_mean_hitmiss[idx_N,:,0],data_mean_hitmiss[idx_N,:,1],nan_policy='omit')[1] * 5
    # pvals = ranksums(data_mean_hitmiss[idx_N,:,0],data_mean_hitmiss[idx_N,:,1],nan_policy='omit')[1]
    for ipval,pval in enumerate(pvals):
        ax.text(ipval,np.nanmean(data_mean_hitmiss[idx_N,ipval,:],axis=None)+0.05,get_sig_asterisks(pval,return_ns=True),
                color='k',fontsize=10,ha='center')
    
    ax.set_xticks(plotlocs[posfilter],plotcenters[posfilter],rotation=45)
    ax.legend(plotlabels,loc='upper left',fontsize=11,frameon=False,reverse=True)
        # plt.plot(plotcenters,np.nanmean(data_mean_hitmiss[:,0],axis=0),color='k')
    ax.set_xlabel('Signal Strength (%)')
    ax.set_ylabel('Mean Activity (z)')
    ax.set_ylim([-0.05,0.55])
    ax.set_title(siglabels[sig])

# fig.savefig(os.path.join(savedir, 'HitMiss_Mean_Resp_vs_NonRespNeurons_%dsessions.png') % (nSessions), format='png')
# fig.savefig(os.path.join(savedir, 'HitMiss_Mean_NoiseResponsiveNeurons_%dsessions.png') % (nSessions), format='png')
# fig.savefig(os.path.join(savedir, 'HitMiss_Noise_Mean_NoiseResponsiveNeurons_%dsessions.png') % (nSessions), format='png',bbox_inches='tight')

#%% 
labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)
clrs_areas  = get_clr_areas(areas)

data = copy.deepcopy(data_mean_hitmiss)
#normalize data:
# data = data - np.nanmin(data,axis=1,keepdims=True)
# data = data / np.nanmax(data,axis=1,keepdims=True)

fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2),sharey=True,sharex=True)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax = axes[ilab,iarea]
        handles = []

        idx_N = np.all((celldata['roi_name']==area, 
                        # celldata['sig_MN']==1,
                        # celldata['sig_N']==1,
                        celldata['sig_N']!=1,
                        # celldata['sig_MN']!=1,
                        # celldata['layer']=='L2/3',
                        # celldata['layer']=='L4',
                        # celldata['layer']=='L5',
                        # ~np.any(np.isnan(data),axis=(1,2)),
                        celldata['labeled']==label), axis=0)

        if np.sum(idx_N) > 5:
            for ilr,lr in enumerate(lickresp):
                # plt.plot(plotcenters,np.nanmean(data_mean_hitmiss[idx_N,:,ilr],axis=0),color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
                # ax.plot(plotcenters[1:-1],np.nanmean(data[idx_N,1:-1,ilr],axis=0),color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
                # h =shaded_error(plotcenters,data[idx_N,:,ilr],color=plotcolors[ilr],
                            #  label=plotlabels[ilr],ax=ax,error='sem')
                h =shaded_error(plotcenters[1:-1],data[idx_N,1:-1,ilr],color=plotcolors[ilr],
                             label=plotlabels[ilr],ax=ax,error='sem')
                handles.append(h)
        
        if ilab==0 and iarea==0:
            ax.legend(handles,plotlabels,loc='upper left',fontsize=11,frameon=False)

        # ax.axhline(0,color='k',linestyle='--',linewidth=1)
        if ilab==1:
            ax.set_xlabel('Signal Strength')
        ax.set_xticks(np.round(plotcenters[1:-1],1))
        if iarea==0:
            ax.set_ylabel('Mean Activity (z)')
        # ax.set_ylim([-0.1,0.6])
        #     ax.legend(frameon=False,fontsize=6)
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'HitMiss_Noise_Mean_NoiseResponsiveNeurons_RawSignal_%dsessions_Arealabels.png') % (nSessions), format='png')
# plt.savefig(os.path.join(savedir, 'HitMiss_Noise_Mean_NonNoiseResponsiveNeurons_RawSignal_%dsessions_Arealabels.png') % (nSessions), format='png')
# plt.savefig(os.path.join(savedir, 'EncodingModel_%s_cvR2_Areas_Labels_%dsessions.png') % (version,nSessions), format='png')

#%% 
# diffdata = np.diff(data_mean_hitmiss,axis=2).squeeze()
zmin=7
zmax=17
data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,min_ntrials=5,
                                                    splithitmiss=True,autobin=True)
diffdata = np.nanmean(np.diff(data_mean_hitmiss,axis=2)[:,1:-1,:],axis=1).squeeze()
# diffdata = np.nanmean(np.diff(data_mean_hitmiss,axis=2)[:,:,:],axis=1).squeeze()

#%% 
labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM']
nareas      = len(areas)
clrs_areas  = get_clr_areas(areas)
ndepthbins  = 5

fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2),sharey=False,sharex=True)
for iarea, area in enumerate(areas):
    for ilab,label in enumerate(labeled):
        ax = axes[ilab,iarea]
        handles = []
        idx_N = np.all((celldata['roi_name']==area, 
                        celldata['sig_MN']==1,
                        # celldata['sig_N']==1,
                        # celldata['sig_N']!=1,
                        # celldata['sig_MN']!=1,
                        idx_N_perf,
                        # idx_nearby,
                        celldata['labeled']==label), axis=0)

        df = pd.DataFrame(data = {'depth':celldata['depth'][idx_N], 'diff': diffdata[idx_N]})
        
        # sns.scatterplot(data=df,y='depth',x='diff',ax=ax,color=clrs_areas[iarea],alpha=0.5,s=5)
        sns.scatterplot(data=df,y='depth',x='diff',ax=ax,color='k',alpha=0.5,s=5)
        depth_edges = np.nanpercentile(df['depth'],np.linspace(0,100,ndepthbins))

        # depth_edges = np.arange(0,500,100)
        if np.all(np.isnan(depth_edges)):
            depth_edges = [0,1000]

        centers     = np.stack((depth_edges[:-1],depth_edges[1:]),axis=1).mean(axis=1)

        depth_diff = np.full(len(centers),np.nan)
        depth_stats = np.full((2,len(centers)),np.nan)

        for ibin,(bd1,bd2) in enumerate(zip(depth_edges[:-1],depth_edges[1:])): 
            idx = idx_N & (celldata['depth']>=bd1) & (celldata['depth']<bd2)
            if sum(idx) >= 5:
                depth_diff[ibin] = np.nanmean(diffdata[idx])
                depth_stats[:,ibin] = wilcoxon(diffdata[idx])

            # depth_diff = np.array([np.nanmean(diffdata[idx_N & (celldata['depth']>=bd1) & (celldata['depth']<bd2)]) if sum(idx_N & (celldata['depth']>=bd1) & (celldata['depth']<bd2))>=10 else np.nan for bd1,bd2 in zip(depth_edges[:-1],depth_edges[1:])])
        
        ax.plot(depth_diff,centers,color=clrs_areas[iarea],linewidth=4)
        for ibin in range(len(depth_diff)):
            if depth_stats[1,ibin] < 0.05:
                sign = int(depth_diff[ibin]>0) * 2 - 1
                ax.text(0.35*sign,centers[ibin],get_sig_asterisks(depth_stats[1,ibin]),
                        ha='center',va='center',color=clrs_areas[iarea],fontsize=14,fontweight='bold')
        # ax.plot(depth_diff,centers,color='k',linewidth=3)
        ax.axvline(x=0,color='k',linestyle='--',linewidth=0.5)
        ax.grid(True)
        ax.set_xlim([-0.5,0.5])
        # ax.set_xlim([-1,1])
        ax.set_ylim([0,500])

        ax.invert_yaxis()
        ax.set_yticks(np.arange(0,600,100))
        # ax.set_xticks(np.arange(-0.5,0.5,0.25))
        ax.set_title('%s-%s' % (area,label),fontsize=12,color=clrs_areas[iarea])
        if ilab==0:
            ax.set_ylabel('Depth (um)')
        if ilab==1:
            ax.set_xlabel(r'hit-miss ($\Delta$z)')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'DeltaHitMiss_Depth_ResponsiveNeurons_Arealabels_%dsessions.png') % (nSessions), format='png')
# plt.savefig(os.path.join(savedir,'DeltaHitMiss_Depth_NonResponsiveNeurons_Arealabels_%dsessions.png') % (nSessions), format='png')

#%% Is the hit-miss difference related to difference in running speed correlation?

data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True)
diffdata = np.nanmean(np.diff(data_mean_hitmiss,axis=2)[:,1:-1,:],axis=1).squeeze()

sessiondata = pd.concat([ses.sessiondata for ses in sessions])
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

N = len(celldata)
runcorr = np.full(N,np.nan)

for iN in tqdm(range(N),total=N,desc='Computing running speed correlation'):
    sesidx = np.where(sessiondata['session_id']==celldata['session_id'][iN])[0][0]
    neuronidx = np.where(sessions[sesidx].celldata['cell_id']==celldata['cell_id'][iN])[0][0]
    idx_T = np.isin(sessions[sesidx].trialdata['stimcat'],['N'])
    # runcorr[iN] = np.corrcoef(sessions[sesidx].respmat[np.ix_(neuronidx,idx_T)],sessions[sesidx].respmat_runspeed[idx_T].squeeze())[0,1]
    runcorr[iN] = np.corrcoef(sessions[sesidx].respmat[neuronidx,idx_T],sessions[sesidx].respmat_runspeed.squeeze()[idx_T])[0,1]

#%% Plot running speed correlation vs hit-miss difference

fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ax = axes[0]
ax.scatter(runcorr,diffdata,c='k',marker = '.',s=5, alpha=0.25)
from scipy.stats import linregress

notnan = ~np.isnan(runcorr) & ~np.isnan(diffdata)

res = linregress(runcorr[notnan],diffdata[notnan])
xfit = np.linspace(np.nanmin(runcorr),np.nanmax(runcorr),100)

yfit = res.intercept + res.slope*xfit
ax.plot(xfit,yfit,'-',color='grey')
ax.fill_between(xfit,res.intercept + res.slope*xfit + res.stderr*1.96,res.intercept + res.slope*xfit - res.stderr*1.96,alpha=0.2,color='k')
ax.text(0.1,0.1,'p = %.2e' % res.pvalue,ha='left',va='bottom',transform=ax.transAxes,fontsize=10)
ax.set_xlabel('Running speed correlation')
ax.set_ylabel('Hit-miss difference')

ax = axes[1]

hist_matrix, x_edges, y_edges = np.histogram2d(runcorr[notnan], diffdata[notnan], bins=50)
# hist_matrix, x_edges, y_edges = np.histogram2d(diffdata[notnan],runcorr[notnan], bins=50)
hist_matrix = hist_matrix.T
hist_matrix = hist_matrix / np.sum(hist_matrix)
# hist_matrix = np.log10(hist_matrix + 1e-10)
hist_matrix = np.log10(hist_matrix + np.min(hist_matrix[hist_matrix>0]))
ax.pcolormesh(x_edges, y_edges, hist_matrix, cmap='magma')
# ax.pcolormesh(x_edges, y_edges, hist_matrix.T, cmap='viridis', norm=colors.LogNorm())
ax.plot(xfit,yfit,'w-')
ax.set_ylim([-1,1])
ax.set_xlim([-0.55,0.55])
ax.set_xlabel('Running speed correlation')
# cbar = fig.colorbar(ax.collections[0], ax=ax, shrink=0.3)

#%% 

sigtype     = 'signal'
zmin        = 5
zmax        = 17
nbins_noise = 5


sigtype     = 'signal_psy'
zmin        = -2
zmax        = 2
nbins_noise = 5

min_ntrials = 10

# diffdata = np.diff(data_mean_hitmiss,axis=2).squeeze()
data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=min_ntrials)
diffdata = np.nanmean(np.diff(data_mean_hitmiss,axis=2)[:,1:-1,:],axis=1).squeeze()
data_mean,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=False,min_ntrials=min_ntrials)
meandata = np.nanmean(data_mean[:,1:-1],axis=1).squeeze()

labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM']
nareas      = len(areas)
clrs_areas  = get_clr_areas(areas)

fig,axes    = plt.subplots(nlabels,nareas,figsize=(nareas*2,nlabels*2),sharey=True,sharex=True)
for iarea, area in enumerate(areas):
    for ilab,label in enumerate(labeled):
        ax = axes[ilab,iarea]
        handles = []
        idx_N = np.all((celldata['roi_name']==area, 
                        # celldata['layer']=='L2/3',
                        # celldata['layer']=='L4',
                        # celldata['layer']=='L5',
                        celldata['labeled']==label), axis=0)
        X = meandata[idx_N]
        Y = diffdata[idx_N]
        idx_nan = ~np.isnan(X) & ~np.isnan(Y)
        X = X[idx_nan]
        Y = Y[idx_nan]
        ax.scatter(X,Y,color=clrs_areas[iarea],alpha=0.35,s=5)
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
        
        # Create x values for the regression line
        x_reg = np.linspace(np.min(meandata[idx_N]), np.max(meandata[idx_N]), 100)
        
        # Plot regression line
        ax.plot(x_reg, slope * x_reg + intercept, color='k', linestyle='--', linewidth=2)
        
        
        # Add R-squared value to the plot
        # ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}', transform=ax.transAxes, 
        ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=10)
        
        # ax.set_xlabel('Mean Response')
        if iarea==0:
            ax.set_ylabel('Hit-Miss Response')
        ax.set_title(f'{area} - {label}')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.grid(True)
        ax.set_xlim([np.min(meandata[idx_N]),np.max(meandata[idx_N])])
        ax.set_ylim([-1,1])
        
# plt.savefig(os.path.join(savedir,'Corr_Response_Hitmiss_arealabels_L5_%dsessions.png') % (nSessions), format='png')
# plt.savefig(os.path.join(savedir,'Corr_Response_Hitmiss_arealabels_L23_%dsessions.png') % (nSessions), format='png')
# plt.savefig(os.path.join(savedir,'Corr_Response_Hitmiss_arealabels_%dsessions.png') % (nSessions), format='png')

#%% 
sigtype = 'signal'
zmin = 7
zmax = 17
nbins_noise = 5

# diffdata = np.diff(data_mean_hitmiss,axis=2).squeeze()
data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=min_ntrials)
diffdata = np.nanmean(np.diff(data_mean_hitmiss,axis=2)[:,1:-1,:],axis=1).squeeze()
data_mean,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=False,min_ntrials=min_ntrials)
meandata = np.nanmean(data_mean[:,1:-1],axis=1).squeeze()

celldata['runcorr'] = np.nan

for ises,ses in enumerate(sessions):
    idx_N_ses = celldata['session_id']==ses.sessiondata['session_id'][0]
    g = [np.corrcoef(ses.respmat[iN,:],ses.respmat_runspeed.squeeze())[0,1] for iN in range(len(ses.celldata))]
    celldata.loc[idx_N_ses,'runcorr'] = g

#%%
layers = ['L2/3', 'L4', 'L5']
areas = ['V1', 'PM']

fig, axes = plt.subplots(len(layers), len(areas), figsize=(3*len(areas), 3*len(layers)), sharex=True, sharey=True)

for i, layer in enumerate(layers):
    for j, area in enumerate(areas):
        ax = axes[i, j]

        idx_N = np.all((celldata['roi_name']==area, 
                        # celldata['sig_MN']==1,

                        # celldata['layer']=='L2/3',
                        # celldata['layer']=='L4',
                        # celldata['layer']=='L5',
                        # celldata['labeled']==label,
                        celldata['layer'] == layer), axis=0)

        # idx_N = np.all((celldata['roi_name'] == area, celldata['layer'] == layer), axis=0)

        if np.sum(idx_N)>5:
            # X = meandata[idx_N]
            X = diffdata[idx_N]
            Y = celldata['runcorr'][idx_N]

            idx_nan = ~np.isnan(X) & ~np.isnan(Y)
            X = X[idx_nan]
            Y = Y[idx_nan]

            ax.scatter(X, Y, alpha=0.5, s=5)

            slope, intercept, r_value, p_value, _ = stats.linregress(X, Y)
            ax.plot(X, slope*X + intercept, color='r', linestyle='--')

        ax.set_title(f"{area} - {layer}")
        ax.text(0.95, 0.15, f'r = {r_value:.3f}\np = {p_value:.3f}', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=10)

        if i == len(layers) - 1:
            # ax.set_xlabel('Mean response')
            ax.set_xlabel('Hit/miss difference')
        if j == 0:
            ax.set_ylabel('Runspeed correlation')

plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'Corr_Response_Runspeed_Layers_%dsessions.png' % nSessions), format='png')
plt.savefig(os.path.join(savedir, 'Corr_Hitmiss_Runspeed_Layers_%dsessions.png' % nSessions), format='png')

#%% 

#%% Create index of nearby cells to compare to:
from utils.rf_lib import filter_nearlabeled

celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
N           = len(celldata)
idx_nearby = np.zeros(N,dtype=bool)
for ses in sessions:
    idx_ses = np.where(celldata['session_id']==ses.sessiondata['session_id'][0])[0]
    idx_nearby[idx_ses] = filter_nearlabeled(ses,radius=50)
    # idx_nearby[idx_ses] = filter_nearlabeled(ses,radius=50)

#%% 
sigtype     = 'signal'
zmin        = 5
zmax        = 15

nbins_noise = 3
plotlabels      = ['Catch','Sub','Thr','Sup','Max']

# nbins_noise = 5
# plotlabels      = ['Catch','Imp','Sub','Thr','Sup','Lar','Max']

min_ntrials = 2
ncompstoplot = 2
ncomponents = nbins_noise+2
areas       = ['V1','PM']
clrs_areas = get_clr_areas(areas)

# data,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,
                                    #    splithitmiss=False,min_ntrials=min_ntrials)

data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=min_ntrials)

data = np.nanmean(data_mean_hitmiss,axis=-1)

idx_N = np.all((
                # celldata['roi_name']==area, 
                np.isin(celldata['roi_name'],['V1','PM']),
                # celldata['sig_MN']==1,
                celldata['noise_level']<20,
                # celldata['depth']>200,
                # celldata['sig_N']==1,
                # celldata['sig_N']!=1,
                idx_nearby,
                ), axis=0)
# idx_N = np.ones(len(celldata),dtype=bool)

data = data[idx_N,:]
idx_nan = ~np.any(np.isnan(data),axis=1)
data = data[idx_nan,:]
print(np.shape(data))

pca = PCA(n_components=ncomponents)
pca.fit(data)

data_pca = pca.transform(data)

fig,axes = plt.subplots(1,3,figsize=(10,3))
ax = axes[0]
ax.plot(pca.explained_variance_ratio_,linewidth=2,marker='o',color='k')
ax.set_xlabel('PC index')
ax.set_ylabel('Explained Variance')
ax.set_title('Explained Variance')

plotlocs        = np.arange(np.shape(data)[1])
# clrs = ['blue','red']
clrs = sns.color_palette('husl',n_colors=ncompstoplot)
ax = axes[1]
for icomp in range(ncompstoplot):
    ax.plot(plotlocs,pca.components_[icomp,:].T,linewidth=3,color=clrs[icomp])
    # ax.plot(plotcenters,pca.components_[icomp,:].T,linewidth=5/(icomp+1))
# ax.plot(plotcenters,pca.components_[:2,:].T)
ax.set_title('PC components')
ax.set_xticks(plotlocs,plotlabels)
ax.legend(['PC%d' % (icomp+1) for icomp in range(ncompstoplot)],frameon=False,loc='best',fontsize=10)

ax = axes[2]
for iarea,area in enumerate(areas):
    idx = celldata['roi_name']==area
    idx = idx[idx_N][idx_nan]
    # idx_N = celldata['roi_name']==area
    # X = data_pca_re[0,idx_N,icomp]
    # Y = data_pca_re[1,idx_N,icomp]
    ax.scatter(data_pca[idx,0], data_pca[idx,1], c=clrs_areas[iarea], marker = '.',s=10, alpha=0.25)
l = ax.legend(areas, loc='upper right',frameon=False,fontsize=11)
for it,text in enumerate(l.get_texts()):
    text.set_color(clrs_areas[it])

ax.set_title('Neuron locations in PC space')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'PCA_TuningCurve_%dsessions.png') % (nSessions), format='png')

#%% Tuning curves for labeled and unlabeled cells:
nclusters = 3
#perform k-means clustering in PC space with k=3
from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3, random_state=0).fit(data_pca)
kmeans = KMeans(n_clusters=nclusters).fit(data_pca[:,:2])
# kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(data_pca[:,:3])

#Resort categories to have consistent definition:
kmeans_cat = np.zeros(np.shape(data_pca)[0])
kmeans_cat[kmeans.labels_==np.argmax(kmeans.cluster_centers_[:,0])] = 1
kmeans_cat[kmeans.labels_==np.argmax(kmeans.cluster_centers_[:,1])] = 2

# plt.hist(kmeans_cat,bins=nclusters)

# kmeans_labels = ['Max-tuned','Thr-tuned','Non-tuned']
kmeans_labels = ['Non-tuned','Max-tuned','Thr-tuned']
clrs_cluster = ['grey','#1f77b4', '#ff7f0e']


#%% Make clusters manually:
thr = 0.3
# thr = 0.2
kmeans_cat = np.argmax(data_pca[:,:2], axis=1)+1
kmeans_cat[(data_pca[:,0]<thr) & (data_pca[:,1]<thr)] = 0

#%% Show the KMEans clusters in PC space:
fig,axes = plt.subplots(1,3,figsize=(9,3))
ax = axes[0]
for iarea,area in enumerate(areas):
    idx = celldata['roi_name']==area
    idx = idx[idx_N]
    idx = idx[idx_nan]
    # idx_N = celldata['roi_name']==area
    # X = data_pca_re[0,idx_N,icomp]
    # Y = data_pca_re[1,idx_N,icomp]
    ax.scatter(data_pca[idx,0], data_pca[idx,1], c=clrs_areas[iarea], marker = '.',s=10, alpha=0.25)
l = ax.legend(areas, loc='upper right',frameon=False,fontsize=11)
for it,text in enumerate(l.get_texts()):
    text.set_color(clrs_areas[it])
    
ax = axes[1]
for icluster in range(nclusters):

    idx_cluster = kmeans_cat==icluster
    ax.scatter(data_pca[idx_cluster,0], data_pca[idx_cluster,1], c=clrs_cluster[icluster], marker = '.',s=10, alpha=0.5)
ax.set_title('K-Means clusters in PC space')
ax.legend(kmeans_labels,title="Category",frameon=False)
for itext,text in enumerate(ax.legend_.texts):
    text.set_color(clrs_cluster[itext])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.tight_layout()


ax = axes[2]
arealabels = ['V1unl','V1lab','PMunl','PMlab']

df = pd.DataFrame({'arealabel':celldata['arealabel'][idx_N][idx_nan],
                   'klabel':kmeans_cat})
df = df[df['arealabel'].isin(arealabels)]

count_data = df.groupby(["arealabel", "klabel"]).size().unstack(fill_value=0)
# Normalize to ensure each bar sums to 1
normalized_data = count_data.div(count_data.sum(axis=1), axis=0)
# Plot stacked bars using Matplotlib with Seaborn styling
bottom = np.zeros(len(normalized_data))
# Loop through each unique value in "Value" to stack the bars
for idx, value in enumerate(normalized_data.columns):
    ax.bar(normalized_data.index, normalized_data[value], label=kmeans_labels[idx], bottom=bottom, color=clrs_cluster[idx])
    bottom += normalized_data[value]  # Update bottom for stacking

# Labels and legend
ax.set_ylabel("Fraction of Total")
# plt.title("")
ax.legend(title="Category",frameon=False,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_ylim(0, 1)  # Ensure y-axis goes from 0 to 1
fig.savefig(os.path.join(savedir,'KMeans_PCA_%dsessions.png') % (nSessions), format='png')

#%% 

arealabels = ['V1unl','V1lab','PMunl','PMlab']
narealabels = len(arealabels)

normalized_data = np.full((narealabels, nclusters,nSessions), np.nan)  # init array with NaN 
for ises,ses in enumerate(sessions):
    for ial,arealabel in enumerate(arealabels):
        idx = np.all((celldata['arealabel'][idx_N][idx_nan]==arealabel,
                          celldata['session_id'][idx_N][idx_nan]==ses.sessiondata['session_id'][0])
                          ,axis=0)
        if np.sum(idx)>5:
            for icluster in range(nclusters):
                normalized_data[ial,icluster,ises] = np.sum(kmeans_cat[idx]==icluster) / np.sum(idx)

#%% 
fs = 12
clrs_arealabels = get_clr_area_labeled(arealabels)
fig,axes = plt.subplots(1,2,figsize=(5,3),sharey=True,sharex=True)
for iax,icl in enumerate([1,2]):
    ax = axes[iax]
    for ial,arealabel in enumerate(arealabels):
        ax.bar(ial,np.nanmean(normalized_data[ial,icl,:],axis=-1),color=clrs_arealabels[ial],
               label=arealabel,width=0.5)
        for ises,ses in enumerate(sessions):
                ax.plot([ial],normalized_data[ial,icl,ises],'.',color='k',markersize=7)

    ax.plot([0,1],normalized_data[[0,1],icl,:],'-',color='k',linewidth=0.3)
    ax.plot([2,3],normalized_data[[2,3],icl,:],'-',color='k',linewidth=0.3)

    ax.set_xticks(np.arange(narealabels),arealabels)
    ax.set_ylim([0,ax.get_ylim()[1]])

    data = normalized_data[:,icl,:].T
    t,p = ttest_rel(data[:,0],data[:,1],nan_policy='omit')
    ax.text(0.25,0.7,'%s' % get_sig_asterisks(p,return_ns=True),transform=ax.transAxes,ha='center',fontsize=fs)
    t,p = ttest_rel(data[:,2],data[:,3],nan_policy='omit')
    ax.text(0.75,0.7,'%s' % get_sig_asterisks(p,return_ns=True),transform=ax.transAxes,ha='center',fontsize=fs)

    # t,p = ttest_rel(data[:,0],data[:,2],nan_policy='omit')
    # ax.text(0.5,0.2,'%s' % get_sig_asterisks(p,return_ns=True),transform=ax.transAxes,ha='center',color='k',fontsize=fs)

    ax.set_title(kmeans_labels[icl])
    if iax==1:
        ax.set_xlabel('Area + Proj.')
    if iax==0: 
        ax.set_ylabel('Fraction tuned')
    # ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Tuning_FractionTotal_V1PM_Labeled_%dsessions.png') % (nSessions), format='png')

#%% 
fig,axes = plt.subplots(1,2,figsize=(3,3),sharey=True,sharex=True)
for iax,icl in enumerate([1,2]):
    ax = axes[iax]

    ial = 0
    ax.bar(ial,np.nanmean(normalized_data[ial,icl,:],axis=-1),color=clrs_arealabels[ial],
            label=arealabel,width=0.5)
    for ises,ses in enumerate(sessions):
        ax.plot([0],normalized_data[ial,icl,ises],'.',color='k',markersize=7)

    ial = 2
    ax.bar(1,np.nanmean(normalized_data[ial,icl,:],axis=-1),color=clrs_arealabels[ial],
            label=arealabel,width=0.5)
    for ises,ses in enumerate(sessions):
        ax.plot([1],normalized_data[ial,icl,ises],'.',color='k',markersize=7)

    ax.plot([0,1],normalized_data[[0,2],icl,:],'-',color='k',linewidth=0.3)
    # ax.plot([2,3],normalized_data[[2,3],icl,:],'-',color='k',linewidth=0.3)

    ax.set_xticks([0,1],arealabels[0::2])
    ax.set_ylim([0,ax.get_ylim()[1]])

    data = normalized_data[:,icl,:].T

    t,p = ttest_rel(data[:,0],data[:,2],nan_policy='omit')
    ax.text(0.5,0.8,'%s' % get_sig_asterisks(p,return_ns=True),transform=ax.transAxes,ha='center',color='k',fontsize=fs)

    ax.set_title(kmeans_labels[icl])
    # if iax==1:
        # ax.set_xlabel('Area + Proj.')
    if iax==0: 
        ax.set_ylabel('Fraction of total')
    # ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Tuning_FractionTotal_V1PM_%dsessions.png') % (nSessions), format='png')


#%% Now project the data of hits and misses in the PCA space
# data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=min_ntrials)

# data = np.nanmean(data_mean_hitmiss,axis=-1)

# data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=min_ntrials)
# 
data = data_mean_hitmiss[idx_N,:,:][idx_nan,:,:]

idx_nan_hitmiss = ~np.any(np.isnan(data),axis=(1,2))

misses = pca.transform(data[idx_nan_hitmiss,:,0])[:,:2]
hits = pca.transform(data[idx_nan_hitmiss,:,1])[:,:2]
# hits = pca.transform(data_mean_hitmiss)

kmeans_cat_hitmiss = kmeans_cat[idx_nan_hitmiss]

#%% 
fig,axes = plt.subplots(1,3,figsize=(10,3))
ax = axes[0]
for icl in range(nclusters):
    idx = kmeans_cat_hitmiss==icl
    ax.scatter(hits[idx,0],hits[idx,1],c=clrs_cluster[icl], marker = 'o',s=10, alpha=0.25)
    ax.scatter(misses[idx,0],misses[idx,1],c=clrs_cluster[icl], marker = 'x',s=10, alpha=0.25)
    # ax.scatter(hits[:,0],hits[:,1],c='r', marker = '.',s=10, alpha=0.25)
    # ax.scatter(misses[:,0],misses[:,1],c='g', marker = '.',s=10, alpha=0.25)

ax = axes[1]
for icl in range(nclusters):
    idx = kmeans_cat_hitmiss==icl
    # ax.scatter(hits[idx,0],hits[idx,1],c=clrs_cluster[icl], marker = 'o',s=10, alpha=0.25)
    # ax.scatter(misses[idx,0],misses[idx,1],c=clrs_cluster[icl], marker = 'x',s=10, alpha=0.25)
    ax.scatter(misses[idx,0],hits[idx,0],c=clrs_cluster[icl], marker = '.',s=10, alpha=0.5)

ax.plot([0,1],[0,1],color='k',linewidth=1,alpha=1,linestyle='--',transform=ax.transAxes)
ax.set_xlabel('Misses')
ax.set_ylabel('Hits')
ax.set_title(f'PC{0} space: Hits vs Misses')
# ax.legend(handles,areas, loc='lower right',frameon=False)

ax = axes[2]
for icl in range(nclusters):
    idx = kmeans_cat_hitmiss==icl
    # ax.scatter(hits[idx,0],hits[idx,1],c=clrs_cluster[icl], marker = 'o',s=10, alpha=0.25)
    # ax.scatter(misses[idx,0],misses[idx,1],c=clrs_cluster[icl], marker = 'x',s=10, alpha=0.25)
    ax.scatter(misses[idx,1],hits[idx,1],c=clrs_cluster[icl], marker = '.',s=10, alpha=0.5)

ax.plot([0,1],[0,1],color='k',linewidth=1,alpha=1,linestyle='--',transform=ax.transAxes)
ax.set_xlabel('Misses')
ax.set_ylabel('Hits')
ax.set_title(f'PC{1} space: Hits vs Misses')
# ax.legend(handles,areas, loc='lower right',frameon=False)

#%% 
data = data_mean_hitmiss[idx_N,:,:]
data = data[idx_nan,:,:]
idx_nan_hitmiss = ~np.any(np.isnan(data),axis=(1,2))

fig,axes = plt.subplots(2,3,figsize=(10,6),sharey=True)
# fig,axes = plt.subplots(2,3,figsize=(10,6),sharey=False)
for iarea,area in enumerate(areas):
    for icl in range(nclusters):
        ax      = axes[iarea,icl]
        idx     = kmeans_cat==icl
        idx     = np.all((kmeans_cat==icl,
                      idx_nan_hitmiss,
                       celldata['roi_name'][idx_N][idx_nan]==area),axis=0)
        
        shaded_error(plotlocs,data[idx,:,0],error='sem',color='grey',ax=ax)
        shaded_error(plotlocs,data[idx,:,1],error='sem',color='darkgreen',ax=ax)

        t,p = ttest_rel(np.nanmean(data[idx,:,0],axis=1),np.nanmean(data[idx,:,1],axis=1))
        # t,p = ttest_rel(data[idx,:,0].flatten(),data[idx,:,1].flatten())

        ax.text(0.5,0.8,'p=%s' % get_sig_asterisks(p,return_ns=True),transform=ax.transAxes,ha='center',color='k',fontsize=fs)
        ax.set_title('%s-%s' % (area,kmeans_labels[icl]))
        # ax.set_ticks(np.arange())
        ax.set_xticks(plotlocs,plotlabels)

plt.tight_layout()
fig.savefig(os.path.join(savedir,'HitMiss_KmeansTuned_V1PM_%dsessions.png') % (nSessions), format='png')

#%% 
data = data_mean_hitmiss[idx_N,:,:]
data = data[idx_nan,:,:]
idx_nan_hitmiss = ~np.any(np.isnan(data),axis=(1,2))

fig,axes = plt.subplots(4,3,figsize=(10,10),sharey=True)
# fig,axes = plt.subplots(2,3,figsize=(10,6),sharey=False)
for ial,arealabel in enumerate(arealabels):
    for icl in range(nclusters):
        ax      = axes[ial,icl]
        idx     = kmeans_cat==icl
        idx     = np.all((kmeans_cat==icl,
                      idx_nan_hitmiss,
                       celldata['arealabel'][idx_N][idx_nan]==arealabel),axis=0)
        
        shaded_error(plotlocs,data[idx,:,0],error='sem',color='grey',ax=ax)
        shaded_error(plotlocs,data[idx,:,1],error='sem',color='darkgreen',ax=ax)

        t,p = ttest_rel(np.nanmean(data[idx,:,0],axis=1),np.nanmean(data[idx,:,1],axis=1))
        
        # t,p = ttest_rel(data[idx,:,0].flatten(),data[idx,:,1].flatten())
        
        ax.text(0.5,0.8,'p=%s' % get_sig_asterisks(p,return_ns=True),transform=ax.transAxes,ha='center',color='k',fontsize=fs)
        ax.set_title('%s-%s' % (arealabel,kmeans_labels[icl]))
        # ax.set_ticks(np.arange())
        ax.set_xticks(plotlocs,plotlabels)

plt.tight_layout()
fig.savefig(os.path.join(savedir,'HitMiss_KmeansTuned_V1PM_Labeled_%dsessions.png') % (nSessions), format='png')


#%% 
data = data_mean_hitmiss[idx_N,:,:]
data = data[idx_nan,:,:]
idx_nan_hitmiss = ~np.any(np.isnan(data),axis=(1,2))

fig,axes = plt.subplots(2,3,figsize=(10,6),sharey=True)
# fig,axes = plt.subplots(2,3,figsize=(10,6),sharey=False)
for iarea,area in enumerate(areas):
    for icl in range(nclusters):
        ax      = axes[iarea,icl]
        idx     = kmeans_cat==icl
        idx     = np.all((
                        kmeans_cat<10,
                      idx_nan_hitmiss,
                       celldata['roi_name'][idx_N][idx_nan]==area),axis=0)
        
        ax.plot(np.nanmean(data[idx,:,0],axis=0),color='grey')
        ax.plot(np.nanmean(data[idx,:,1],axis=0),color='darkgreen')
        ax.set_title('%s-%s' % (area,kmeans_labels[icl]))
        # ax.set_ticks(np.arange())
        ax.set_xticks(plotlocs,plotlabels)

plt.tight_layout()

#%%
# areas = ['V1','PM','AL','RSP']
areas = ['V1','PM']

df = pd.DataFrame(data_pca[:,:2],columns=['PC1','PC2'])
fig,axes = plt.subplots(1,3,figsize=(10,3))
df['roi_name'] = np.array(celldata['roi_name'][idx_N][idx_nan])
clrs_areas  = get_clr_areas(areas)
sns.jointplot(x='PC1',y='PC2',data= df[df['roi_name'].isin(areas)],hue='roi_name',ax=axes,
              hue_order=areas,s=7,
              palette=clrs_areas,marginal_kws={'common_norm':False})

#%% 
sigtype     = 'signal'
zmin        = 5
zmax        = 15
nbins_noise = 3
min_ntrials = 5

data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=min_ntrials)
# diffdata = np.nanmean(np.diff(data_mean_hitmiss,axis=2)[:,1:-1,:],axis=1).squeeze()

idx_N = np.all((
                # celldata['roi_name']==area, 
                # celldata['sig_MN']==1,
                # celldata['runcorr']>0.1,
                celldata['noise_level']<20,
                # celldata['depth']>200,
                # celldata['sig_N']==1,
                # celldata['sig_N']!=1,
                # idx_nearby,
                ), axis=0)
# idx_N = np.ones(len(celldata),dtype=bool)

hits        = np.repeat([0,1],np.sum(idx_N))
cell_ids    = np.tile(np.arange(np.sum(idx_N)),2)
data        = np.transpose(data_mean_hitmiss[idx_N,:,:],(2,0,1))
data        = np.reshape(data,(np.sum(idx_N)*2,-1))

print(np.shape(data))

idx_nan = ~np.any(np.isnan(data),axis=1)
data = data[idx_nan,:]

ncomponents = nbins_noise+2

pca = PCA(n_components=ncomponents)
pca.fit(data)

data_pca = pca.transform(data)
data_pca_full = np.full((np.sum(idx_N)*2,ncomponents),np.nan)
data_pca_full[idx_nan,:] = data_pca
data_pca_re = np.reshape(data_pca_full,(2,np.sum(idx_N),ncomponents))

fig,axes = plt.subplots(1,3,figsize=(10,3))
ax = axes[0]
ax.plot(pca.explained_variance_ratio_,linewidth=2,marker='o',color='k')
ax.set_xlabel('PC index')
ax.set_ylabel('Explained Variance')
ax.set_title('Explained Variance')

plotlocs        = np.arange(np.shape(data)[1])

clrs = ['blue','red','green']
ax = axes[1]
for icomp in range(ncompstoplot):
    ax.plot(plotlocs,pca.components_[icomp,:].T,linewidth=3,color=clrs[icomp])
    # ax.plot(plotcenters,pca.components_[icomp,:].T,linewidth=5/(icomp+1))
# ax.plot(plotcenters,pca.components_[:2,:].T)
ax.set_title('PC components')
ax.legend(['PC1','PC2'],frameon=False,loc='best',fontsize=11)

# ax = axes[2]
# ax.scatter(data_pca[:,0],data_pca[:,1],c='k', marker='o',s=10,alpha=0.25)
ax = axes[2]
for iarea,area in enumerate(['V1','PM']):
    idx = celldata['roi_name']==area
    idx = idx[idx_N]
    X = data_pca_re[0,idx,0]
    Y = data_pca_re[0,idx,1]

    # ax.scatter(X, Y, c=clrs_areas[iarea], marker = '.',s=10, alpha=0.25)
    ax.scatter(X, Y, c='r', marker = '.',s=10, alpha=0.25)

    X = data_pca_re[1,idx,0]
    Y = data_pca_re[1,idx,1]
    # ax.scatter(X, Y, c=clrs_areas[iarea], marker = '.',s=10, alpha=0.25)
    ax.scatter(X, Y, c='g', marker = '.',s=10, alpha=0.25)

ax.set_title('Neuron locations in PC space')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
# fig.savefig(os.path.join(savedir,'PCA_TuningCurve_HITMISS_%dsessions.png') % (nSessions), format='png')

#%% 

fig,axes = plt.subplots(1,ncompstoplot,figsize=(ncompstoplot*3.5,3))
areas = ['V1']
# areas = ['PM']

clrs_areas = get_clr_areas(areas)
clrs_areas = [clrs_areas]

areas = ['V1','PM']
# areas = ['V1','PM','AL','RSP']
clrs_areas = get_clr_areas(areas)

for icomp in range(ncompstoplot):
    ax = axes[icomp]
    handles = []
    for iarea,area in enumerate(areas):
        idx_N_area = celldata['roi_name'][idx_N]==area
        X = data_pca_re[0,idx_N_area,icomp]
        Y = data_pca_re[1,idx_N_area,icomp]
        handles.append(ax.scatter(X,Y,s=4,color=clrs_areas[iarea],
                                linewidth=0.5,alpha=0.5,linestyle='--'))
        ax.set_xlim([np.nanmin((X,Y)),np.nanmax((X,Y))])
        ax.set_ylim([np.nanmin((X,Y)),np.nanmax((X,Y))])
        t_stat, p_value = stats.wilcoxon(X[~np.isnan(X) & ~np.isnan(Y)], Y[~np.isnan(X) & ~np.isnan(Y)])
        sign = np.sign(np.nanmean(Y)-np.nanmean(X))
        # t_stat, p_value = stats.ttest_re1l(X[~np.isnan(X) & ~np.isnan(Y)], Y[~np.isnan(X) & ~np.isnan(Y)])
        ax.text(0.05, 0.95-0.1*iarea, f'{area}: t = {sign*t_stat:.3f}\np = {p_value:1.3f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=8)
        ax.plot(np.nanmean(X),np.nanmean(Y),'+',color='k',markersize=10)

    ax.plot([0,1],[0,1],color='k',linewidth=1,alpha=1,linestyle='--',transform=ax.transAxes)
    ax.set_xlabel('Misses')
    ax.set_ylabel('Hits')
    ax.set_title(f'PC{icomp} space: Hits vs Misses')
    ax.legend(handles,areas, loc='lower right',frameon=False)

fig.savefig(os.path.join(savedir,'PCA_Diff_HITMISS_%dsessions.png') % (nSessions), format='png')

#%% 

