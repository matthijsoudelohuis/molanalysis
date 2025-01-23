# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
In particular pairwise correlations between neurons
"""

#%% Import packages
import os
os.chdir('e:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import zscore

from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive
import seaborn as sns
import matplotlib.pyplot as plt
from utils.psth import *
from utils.plotting_style import * #get all the fixed color schemes
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *
from detection.example_cells import get_example_cells

from preprocessing.preprocesslib import assign_layer
from utils.plot_lib import shaded_error,my_ceil,my_floor
from utils.corr_lib import *
from utils.rf_lib import filter_nearlabeled
from utils.shuffle_lib import my_shuffle, corr_shuffle

plt.rcParams['svg.fonttype'] = 'none'

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'

session_list = np.array([['LPE11622', '2024_02_21']])
# session_list = np.array([['LPE12385', '2024_06_16']])
session_list = np.array([['LPE11997', '2024_04_16'],
                         ['LPE11997', '2024_04_17'],
                         ['LPE11622', '2024_02_21'],
                         ['LPE11622', '2024_02_26'],
                         ['LPE11998', '2024_04_25'],
                         ['LPE11998', '2024_04_30'],
                         ['LPE12385', '2024_06_16'],
                         ['LPE12385', '2024_06_27'],
                         ['LPE12013', '2024_04_25'],
                         ['LPE12013', '2024_04_29'],
                         ])
# session_list = np.array([['LPE10884', '2023_12_14']])
# session_list = np.array([['LPE10884', '2023_12_14']])

# sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=False,
                        #  load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions

sessions,nSessions = filter_sessions(protocols=protocol,load_behaviordata=True,load_videodata=False,
                         load_calciumdata=True,calciumversion=calciumversion,min_cells=100) #Load specified list of sessions

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\PairwiseCorr\\')

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].sessiondata['fs'] = 5.355475
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.25,uppertriangular=True)[0]

#%% Z-score the calciumdata: 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 60   #post cm
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
for ises in range(nSessions):
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

#%% Get signal as relative to psychometric curve for all sessions:
sessions = noise_to_psy(sessions,filter_engaged=True)

#%% Compute noise correlations: 
uppertriangular = False

for ises in tqdm(range(nSessions),total=nSessions,desc= 'Computing signal and noise correlations: '):
    [N,K]                       = np.shape(sessions[ises].respmat) #get dimensions of response matrix
    idx_K = np.isin(sessions[ises].trialdata['stimcat'],['N'])
    sessions[ises].resp_corr    = np.corrcoef(sessions[ises].respmat)
    sessions[ises].noise_corr   = np.corrcoef(sessions[ises].respmat[:,idx_K])
    # sessions[ises].noise_corr   = np.corrcoef(sessions[ises].respmat)

    if uppertriangular:
        idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
        sessions[ises].resp_corr[idx_triu] = np.nan
        sessions[ises].noise_corr[idx_triu] = np.nan
    else: #set only autocorrelation to nan
        np.fill_diagonal(sessions[ises].resp_corr,np.nan)
        np.fill_diagonal(sessions[ises].noise_corr,np.nan)
        

sessions = corr_shuffle(sessions,method='random')

# sessions[0].respmat[:10,:10]

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
celldata            = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
# plt.hist(celldata['noise_level'],bins=np.arange(0,40,1))

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:

areapairs           = ['V1-V1','PM-PM','V1-PM']
# areapairs           = ['V1-V1','PM-PM','AL-AL','RSP-RSP']
# areapairs           = ['V1-V1']
# areapairs           = ['V1-PM']
# areapairs           = ['PM-PM']
noise_thr           = 20
filternear          = False

projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
clrs_projpairs      = get_clr_labelpairs(projpairs)
zscoreflag          = False

# for corr_type in ['sig_corr']:
for corr_type in ['noise_corr']:
# for corr_type in ['resp_corr']:
# for corr_type in ['trace_corr']:
    for areapair in areapairs:

        bincenters,histcorr,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(sessions,corr_type=corr_type,filternear=filternear,projpairs=projpairs,noise_thr=noise_thr,
                                                            areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag,valuematching=None)
        
        # bincenters_sh,histcorr_sh,meancorr_sh,varcorr_sh,_ = hist_corr_areas_labeling(sessions,corr_type='corr_shuffle',filternear=False,projpairs=' ',noise_thr=noise_thr,
                                                            # areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag,valuematching=None)
        print('%d/%d sessions with lab-lab populations for %s'
              % (np.sum(~np.any(np.isnan(histcorr[:,:,0,0,-1]),axis=0)),nSessions,areapair))
        
        areaprojpairs = projpairs.copy()
        for ipp,projpair in enumerate(projpairs):
            areaprojpairs[ipp]       = areapair.split('-')[0] + projpair.split('-')[0] + '-' + areapair.split('-')[1] + projpair.split('-')[1] 
    
        fig         = plt.figure(figsize=(8, 4))
        gspec       = fig.add_gridspec(nrows=2, ncols=3)
        
        histdata    = np.cumsum(histcorr,axis=0)/100 #get cumulative distribution
        # histdata    = histcorr/100 #get cumulative distribution
        histmean    = np.nanmean(histdata,axis=1) #get mean across sessions
        histerror   = np.nanstd(histdata,axis=1) / np.sqrt(nSessions) #compute SEM
       
        histdata_sh  = np.cumsum(histcorr_sh,axis=0)/100 #get cumulative distribution
        histmean_sh = np.nanmean(histdata_sh,axis=1) #get mean across sessions
        histerror_sh = np.nanstd(histdata_sh,axis=1) / np.sqrt(nSessions) #compute SEM

        ax0         = fig.add_subplot(gspec[:2, :2]) #bigger subplot for the cum dist
        
        xpos = bincenters[np.where(np.nanmean(histmean,axis=3).squeeze()<0.1)[0][-1]]
        axins1 = ax0.inset_axes([0.05, 0.25, 0.3, 0.4],xlim=([xpos-0.05,xpos+0.025]),ylim=[0,0.2],xticklabels=[], yticklabels=[])
        ax0.indicate_inset_zoom(axins1, edgecolor="black")
        axins1.tick_params(axis='both', which='both', length=0)
        for axis in ['top','bottom','left','right']:
            axins1.spines[axis].set_color('gray')
            axins1.spines[axis].set_linewidth(1)

        xpos = bincenters[np.where(np.nanmean(histmean,axis=3).squeeze()>0.9)[0][0]]
        axins2 = ax0.inset_axes([0.65, 0.25, 0.3, 0.4],xlim=([xpos-0.05,xpos+0.05]),ylim=[0.8,1],xticklabels=[], yticklabels=[])
        ax0.indicate_inset_zoom(axins2, edgecolor="gray")
        axins2.tick_params(axis='both', which='both', length=0)
        for axis in ['top','bottom','left','right']:
            axins2.spines[axis].set_color('gray')
            axins2.spines[axis].set_linewidth(1)

        handles = []
        for ipp,projpair in enumerate(projpairs): #show for each projection identity pair:
            handles.append(ax0.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp],
                                    linewidth=0.3)[0])
            # handles.append(shaded_error(x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
                            # yerror=np.squeeze(histerror[:,0,0,ipp]),ax=ax0,color=clrs_projpairs[ipp]))
            for ises in range(nSessions):
                ax0.plot(bincenters,np.squeeze(histdata[:,ises,0,0,ipp]),color=clrs_projpairs[ipp],linewidth=0.3)
            axins1.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp])
            axins2.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # shaded_error(axins1,x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
            #                 yerror=np.squeeze(histerror[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # shaded_error(axins2,x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
            #                 yerror=np.squeeze(histerror[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # plot triangle for mean:
            ax0.plot(np.nanmean(meancorr[:,0,0,ipp],axis=None),0.9+ipp/50,'v',color=clrs_projpairs[ipp],markersize=5)
        
        # handles.append(shaded_error(x=bincenters,y=np.squeeze(histmean_sh),
                            # yerror=np.squeeze(histerror_sh),ax=ax0,color='k'))
        axins1.plot(bincenters,np.squeeze(histmean_sh),color='k')
        axins2.plot(bincenters,np.squeeze(histmean_sh),color='k')  

        ax0.set_xlabel('Correlation')
        ax0.set_ylabel('Cumulative Fraction')
        ax0.legend(handles=handles,labels=areaprojpairs,frameon=False,loc='upper left',fontsize=8)
        ax0.set_xlim([-0.25,0.35])
        # ax0.set_xlim([-0.15,0.25])
        # ax0.set_xlim([-0.1,0.20])
        if zscoreflag:
            ax0.set_xlim([-2,2])
        ax0.axvline(0,linewidth=0.5,linestyle=':',color='k') #add line at zero for ref
        ax0.set_ylim([0,1])
        # ax0.set_ylim([0,0.15])
        ax0.set_title('%s %s' % (areapair,corr_type),fontsize=12)

        #  Now show a heatmap of the meancorr data averaged over sessions (first dimension). 
        #  Between each projpair a paired t-test is done of the mean across sesssions and if significant a line is 
        #  drawn from the center of that entry of the heatmap and other one with an asterisk on top of the line. 
        #  For subplot 3 the same is done but then with varcorr.
        data        = np.squeeze(np.nanmean(meancorr[:,0,:,:],axis=0))
        data        = np.reshape(data,(2,2))

        xlabels     = [areapair.split('-')[1] + 'unl',areapair.split('-')[1] + 'lab'] 
        ylabels     = [areapair.split('-')[0] + 'unl',areapair.split('-')[0] + 'lab'] 
        xlocs        = np.array([0,1,0,1])
        ylocs        = np.array([0,0,1,1])
        if areapair=='V1-PM':
            test_indices = np.array([[0,1],[0,2],[1,2],[2,3],[0,3],[1,3]])
        else: 
            test_indices = np.array([[0,1],[0,3],[1,3]])
        
        ax1 = fig.add_subplot(gspec[0, 2])
        pcm = ax1.imshow(data,cmap='hot',vmin=my_floor(np.min(data)-0.002,2),vmax=my_ceil(np.max(data),2))
        ax1.set_xticks([0,1],labels=xlabels)
        ax1.xaxis.tick_top()
        ax1.set_yticks([0,1],labels=ylabels)
        ax1.set_title('Mean')
        fig.colorbar(pcm, ax=ax1)
        
        for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
            data1 = meancorr[:,0,0,ix]
            data2 = meancorr[:,0,0,iy]
            pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = stats.wilcoxon(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = pval * 3 #bonferroni correction
            if pval<0.05:
                ax1.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
                ax1.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                                    weight='bold',fontsize=10) #

        # Now the same but for the std of the pairwise correlations:
        data        = np.squeeze(np.nanmean(varcorr[:,0,:,:],axis=0))
        data        = np.reshape(data,(2,2))

        ax2 = fig.add_subplot(gspec[1, 2])
        pcm = ax2.imshow(data,cmap='hot',vmin=my_floor(np.min(data)-0.002,2),vmax=my_ceil(np.max(data),2))
        ax2.set_xticks([0,1],labels=xlabels)
        ax2.xaxis.tick_top()
        ax2.set_yticks([0,1],labels=ylabels)
        ax2.set_title('Std')
        fig.colorbar(pcm, ax=ax2)
        
        for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
            data1 = varcorr[:,0,0,ix]
            data2 = varcorr[:,0,0,iy]
            pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = stats.wilcoxon(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = pval * 6 #bonferroni correction
            if pval<0.05:
                ax2.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
                ax2.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                                    weight='bold',fontsize=10) #
        
        # plt.suptitle('%s %s' % (areapair,corr_type),fontsize=12)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_PCA1_L5L23_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_%s_%s' % (areapair,corr_type) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_MatchOSI_Proj_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')


#%% 
   

