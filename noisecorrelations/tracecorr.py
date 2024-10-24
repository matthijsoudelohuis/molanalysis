# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2022-2026, Champalimaud Center, Lisbon
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.signal import detrend
from statannotations.Annotator import Annotator
from scipy.optimize import curve_fit

from loaddata.session_info import filter_sessions,load_sessions
from preprocessing.preprocesslib import assign_layer
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error,my_ceil,my_floor
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import compute_tuning, compute_prefori
from utils.explorefigs import plot_excerpt
from utils.shuffle_lib import my_shuffle, corr_shuffle
from utils.gain_lib import * 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

# First plot the mean and the std. This could already indicate some things. For each dataset make a scatter of the mean 
# and std with color the V1-PM and within area datasets as well.
# To understand what is different between the interactions the population metrics of Umakantha et al. 2021 could be computed as well: 
# Loading similarity, percent shared variance, and population dimensionality. 
# Factor analysis: loading similarity (all weights similarly positive or distributed?), percent shared variance (how much of the
# variability of each neuron can be accounted for by other nuerons and cofluctuations, and lastly the dimensionality. Fit FA model 
# maximizing the cross validated log likelihood. Then take d dimensions that explain 95% of explainable CV variance. 
# Make function of this FA model fitting (my_fa_fit.py)

# What is the objective here? The question is to understand what type of cofluctuations lead to the observed differences in 
# spike count correlations?


#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])

session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'SP',session_list=session_list)

#%% Load all sessions from certain protocols: 
# sessions,nSessions   = filter_sessions(protocols = ['SP','GR','IM','GN','RF'],filter_areas=['V1','PM']) 
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['IM'],filter_areas=['V1','PM']) 
# sessions,nSessions   = filter_sessions(protocols = ['RF'],filter_areas=['V1','PM'],session_rf=True)  

#%% Remove two sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:                      
for ises in range(nSessions):
    
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='deconv',keepraw=True)
                                calciumversion='dF',keepraw=True,filter_hp=0.01)
    
    # detrend(sessions[ises].calciumdata,type='linear',axis=0,overwrite_data=True)
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False,filtersig=False)[0]
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'calciumdata')

#%% ##################### Compute pairwise neuronal distances: ##############################
# sessions = compute_pairwise_metrics(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

#%% 
for ses in sessions:
    if 'rf_r2_Fgauss' in ses.celldata:
        ses.celldata['rf_p_Fgauss'] = ses.celldata['rf_r2_Fgauss']<0.2
        ses.celldata['rf_p_Fneugauss'] = ses.celldata['rf_r2_Fneugauss']<0.2

#%% Copy Fgauss to F
for ses in sessions:
    if 'rf_az_Fgauss' in ses.celldata:
        ses.celldata['rf_az_F'] = ses.celldata['rf_az_Fgauss']
        ses.celldata['rf_el_F'] = ses.celldata['rf_el_Fgauss']
        ses.celldata['rf_p_F'] = ses.celldata['rf_p_Fgauss']

#%% ##################### Compute pairwise receptive field distances: ##############################
sessions = smooth_rf(sessions,radius=50,rf_type='Fneugauss',mincellsFneu=5)
sessions = exclude_outlier_rf(sessions) 
sessions = replace_smooth_with_Fsig(sessions) 
# sessions = compute_pairwise_delta_rf(sessions,rf_type='Fsmooth')

#%% print number of pairs:
npairs = np.zeros(nSessions)
for ises,ses in enumerate(sessions):
    npairs[ises] = np.sum(~np.isnan(ses.trace_corr))/2
print('Number of pairs: %d (mean: %d, std : %d across n=%d sessions)' % 
            (np.sum(npairs),np.mean(npairs),np.std(npairs),nSessions))

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    if sessions[ises].sessiondata['protocol'].isin(['GR','GN'])[0]:
        sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='OSI')
        sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='gOSI')
        sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='tuning_var')
        sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'])


#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False,remove_method='RRR',remove_rank=2)

#%% 
sessions = corr_shuffle(sessions,method='random')

#%% ##########################################################################################################
# DELTA ANATOMICAL DISTANCE :
# ##########################################################################################################

#%% Define the areapairs:
areapairs       = ['V1-V1','PM-PM']
clrs_areapairs  = get_clr_area_pairs(areapairs)

#%% Compute pairwise correlations as a function of pairwise anatomical distance ###################################################################
# for corr_type in ['trace_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type=corr_type)

    #Make the figure per protocol:
    fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type=corr_type)
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.pdf'), format = 'pdf')

# #%% Compute pairwise trace correlations as a function of pairwise anatomical distance ###################################################################
# [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='trace_corr')

#%% ########################################################################################################
# ##################### Noise correlations within and across areas: ########################################
# ##########################################################################################################

dfses = mean_corr_areas_labeling([sessions[0]],corr_type='trace_corr',absolute=True,minNcells=100)[0]
clrs_area_labelpairs = get_clr_area_labelpairs(list(dfses.columns))

pairs = [('V1unl-V1unl','V1lab-V1lab'),
         ('V1unl-V1unl','V1unl-V1lab'),
         ('V1unl-V1lab','V1lab-V1lab'),
         ('PMunl-PMunl','PMunl-PMlab'),
         ('PMunl-PMunl','PMlab-PMlab'),
         ('PMunl-PMlab','PMlab-PMlab'),
         ('V1unl-PMunl','V1lab-PMunl'),
         ('V1unl-PMunl','V1unl-PMlab'),
         ('V1unl-PMunl','V1lab-PMlab'),
         ('V1unl-PMlab','V1lab-PMunl'),
         ('V1unl-PMlab','V1lab-PMlab'),
         ('V1lab-PMunl','V1lab-PMlab'),
         ] #for statistics

#%% #########################################################################################
# Contrast: across areas, layers and projection pairs:
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
#If you override any of these with input to the deltarf bin function as ' ', then these pairs will be ignored

clrs_areapairs      = get_clr_area_pairs(areapairs)
clrs_layerpairs     = get_clr_layerpairs(layerpairs)
clrs_projpairs      = get_clr_labelpairs(projpairs)

# clrs_area_labelpairs = get_clr_area_labelpairs(areapairs+projpairs)

#%% Give redcells a string label
redcelllabels = np.array(['unl','lab'])
for ses in sessions:
    ses.celldata['labeled'] = ses.celldata['redcell']
    ses.celldata['labeled'] = ses.celldata['labeled'].astype(int).apply(lambda x: redcelllabels[x])
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# #%% Detrend the data:
# for ises in np.arange(len(sessions)):
#     sessions[ises].respmat = detrend(sessions[ises].respmat,axis=1)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% Compute the variance across trials for each cell:
for ses in sessions:
    if ses.sessiondata['protocol'][0]=='GR':
        resp_meanori,respmat_res        = mean_resp_gr(ses)
    elif ses.sessiondata['protocol'][0]=='GN':
        resp_meanori,respmat_res        = mean_resp_gn(ses)
    ses.celldata['noise_variance']  = np.var(respmat_res,axis=1)

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:
protocols           = ['GR','GN']
# protocols           = ['GN']

areapairs           = ['V1-V1']
areapairs           = ['V1-PM']
areapairs           = ['V1-V1','PM-PM','V1-PM']
areapairs           = ['PM-PM']

plt.rcParams['axes.spines.right']   = True
plt.rcParams['axes.spines.top']     = True

plt.plot(g[:,[0,3]].T)

zscoreflag = False
# for corr_type in ['trace_corr','sig_corr','noise_corr']:
# for corr_type in ['sig_corr']:
for corr_type in ['noise_corr']:
    for areapair in areapairs:
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

        bincenters,histcorr,meancorr,varcorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=True,projpairs=projpairs,noise_thr=0.2,
                                                            # areapairs=[areapair],layerpairs=['L2/3-L5'],minNcells=10,zscore=zscoreflag)
                                                            # areapairs=[areapair],layerpairs=['L2/3-L2/3'],minNcells=10,zscore=zscoreflag)
                                                            # areapairs=[areapair],layerpairs=['L5-L5'],minNcells=10,zscore=zscoreflag)
                                                            areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag,valuematching=None)
        
        bincenters_sh,histcorr_sh,meancorr_sh,varcorr_sh = hist_corr_areas_labeling(ses,corr_type='corr_shuffle',filternear=True,projpairs=' ',noise_thr=0.2,
                                                            areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag,valuematching=None)
        
        areaprojpairs = projpairs.copy()
        for ipp,projpair in enumerate(projpairs):
            areaprojpairs[ipp]       = areapair.split('-')[0] + projpair.split('-')[0] + '-' + areapair.split('-')[1] + projpair.split('-')[1] 
    
        fig         = plt.figure(figsize=(8, 4))
        gspec       = fig.add_gridspec(nrows=2, ncols=3)
        
        histdata    = np.cumsum(histcorr,axis=0)/100 #get cumulative distribution
        # histdata    = histcorr/100 #get cumulative distribution
        histmean    = np.nanmean(histdata,axis=1) #get mean across sessions
        histerror   = np.nanstd(histdata,axis=1) / np.sqrt(len(ses)) #compute SEM
       
        histdata_sh  = np.cumsum(histcorr_sh,axis=0)/100 #get cumulative distribution
        histmean_sh = np.nanmean(histdata_sh,axis=1) #get mean across sessions
        histerror_sh = np.nanstd(histdata_sh,axis=1) / np.sqrt(len(ses)) #compute SEM

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
            handles.append(shaded_error(ax0,x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
                            yerror=np.squeeze(histerror[:,0,0,ipp]),color=clrs_projpairs[ipp]))
            # for ises in range(len(sessions)):
                # ax0.plot(bincenters,np.squeeze(histdata[:,ises,0,0,ipp]),color=clrs_projpairs[ipp],linewidth=0.3)
            axins1.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp])
            axins2.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # shaded_error(axins1,x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
            #                 yerror=np.squeeze(histerror[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # shaded_error(axins2,x=bincenters,y=np.squeeze(histmean[:,0,0,ipp]),
            #                 yerror=np.squeeze(histerror[:,0,0,ipp]),color=clrs_projpairs[ipp])
            # plot triangle for mean:
            ax0.plot(np.nanmean(meancorr[:,0,0,ipp],axis=None),0.9+ipp/50,'v',color=clrs_projpairs[ipp],markersize=5)
        
        handles.append(shaded_error(ax0,x=bincenters,y=np.squeeze(histmean_sh),
                            yerror=np.squeeze(histerror_sh),color='k'))
        axins1.plot(bincenters,np.squeeze(histmean_sh),color='k')
        axins2.plot(bincenters,np.squeeze(histmean_sh),color='k')  

        ax0.set_xlabel('Correlation')
        ax0.set_ylabel('Cumulative Fraction')
        ax0.legend(handles=handles,labels=areaprojpairs,frameon=False,loc='upper left',fontsize=8)
        ax0.set_xlim([-0.25,0.35])
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
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_L23L5_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_MatchOSI_Proj_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')

#%% 
for ses in sessions:
    ses.celldata = assign_layer(ses.celldata)

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

protocols           = ['GR','GN']

# areapairs           = ['V1-V1']
areapairs           = ['V1-V1','PM-PM','V1-PM']

zscoreflag = False
# for corr_type in ['trace_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    for areapair in areapairs:
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
        
        bincenters,histcorr,meancorr,varcorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=' ',
                                                            areapairs=[areapair],layerpairs=layerpairs,minNcells=10,zscore=zscoreflag)
        
        arealayerpairs = layerpairs.copy()
        for ilp,layerpair in enumerate(layerpairs):
            arealayerpairs[ilp]       = areapair.split('-')[0] + layerpair.split('-')[0] + '-' + areapair.split('-')[1] + layerpair.split('-')[1] 

        fig         = plt.figure(figsize=(8, 4))
        gspec       = fig.add_gridspec(nrows=2, ncols=3)
        
        histdata    = np.cumsum(histcorr,axis=0)/100 #get cumulative distribution
        histmean    = np.nanmean(histdata,axis=1) #get mean across sessions
        histerror   = np.nanstd(histdata,axis=1) / np.sqrt(len(ses)) #compute SEM
        
        ax0         = fig.add_subplot(gspec[:2, :2]) #bigger subplot for the cum dist
        
        axins1 = ax0.inset_axes([0.05, 0.25, 0.3, 0.4],xlim=([-0.1,-0.025]),ylim=[0,0.2],xticklabels=[], yticklabels=[])
        ax0.indicate_inset_zoom(axins1, edgecolor="black")
        axins1.tick_params(axis='both', which='both', length=0)
        for axis in ['top','bottom','left','right']:
            axins1.spines[axis].set_color('gray')
            axins1.spines[axis].set_linewidth(1)
            
        axins2 = ax0.inset_axes([0.65, 0.25, 0.3, 0.4],xlim=([0.075,0.125]),ylim=[0.8,0.9],xticklabels=[], yticklabels=[])
        ax0.indicate_inset_zoom(axins2, edgecolor="gray")
        axins2.tick_params(axis='both', which='both', length=0)
        for axis in ['top','bottom','left','right']:
            axins2.spines[axis].set_color('gray')
            axins2.spines[axis].set_linewidth(1)

        handles = []
        for ilp,layerpair in enumerate(layerpairs): #show for each layer pair:
            handles.append(shaded_error(ax0,x=bincenters,y=np.squeeze(histmean[:,0,ilp,0]),
                            yerror=np.squeeze(histerror[:,0,ilp,0]),color=clrs_layerpairs[ilp]))
            
            axins1.plot(bincenters,np.squeeze(histmean[:,0,ilp,0]),color=clrs_layerpairs[ilp])
            axins2.plot(bincenters,np.squeeze(histmean[:,0,ilp,0]),color=clrs_layerpairs[ilp])
            #plot triangle for mean:
            ax0.plot(np.nanmean(meancorr[:,0,ilp,0],axis=None),0.9+ilp/50,'v',color=clrs_layerpairs[ilp],markersize=5)

        ax0.set_xlabel('Correlation')
        ax0.set_ylabel('Cumulative Fraction')
        ax0.legend(handles=handles,labels=arealayerpairs,frameon=False,loc='upper left',fontsize=8)
        ax0.set_xlim([-0.25,0.35])
        if zscoreflag:
            ax0.set_xlim([-2,2])
        ax0.axvline(0,linewidth=0.5,linestyle=':',color='k') #add line at zero for ref
        ax0.set_ylim([0,1])
        ax0.set_title('%s %s' % (areapair,corr_type),fontsize=12)

        #  Now show a heatmap of the meancorr data averaged over sessions (first dimension). 
        #  Between each layerpair a paired t-test is done of the mean across sesssions and if significant a line is 
        #  drawn from the center of that entry of the heatmap and other one with an asterisk on top of the line. 
        #  For subplot 3 the same is done but then with varcorr.
        data        = np.squeeze(np.nanmean(meancorr[:,0,:,0],axis=0))
        data        = np.reshape(data,(2,2))

        xlabels     = [areapair.split('-')[1] + 'L2/3',areapair.split('-')[1] + 'L5']
        ylabels     = [areapair.split('-')[0] + 'L2/3',areapair.split('-')[0] + 'L5']
        xlocs        = np.array([0,1,0,1])
        ylocs        = np.array([0,0,1,1])
        
        if areapair=='V1-PM':
            test_indices = np.array([[0,1],[0,2],[1,2],[2,3],[0,3],[1,3]])
        else: 
            test_indices = np.array([[0,1],[0,3],[1,3]])
        
        ax1 = fig.add_subplot(gspec[0, 2])
        pcm = ax1.imshow(data,cmap='plasma',vmin=my_floor(np.min(data),2),vmax=my_ceil(np.max(data),2))
        ax1.set_xticks([0,1],labels=xlabels)
        ax1.xaxis.tick_top()
        ax1.set_yticks([0,1],labels=ylabels)
        ax1.set_title('Mean')
        fig.colorbar(pcm, ax=ax1)
        
        for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
            data1 = meancorr[:,0,ix,0]
            data2 = meancorr[:,0,iy,0]
            pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = pval * 3 #bonferroni correction
            if pval<0.05:
                ax1.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
                ax1.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                                    weight='bold',fontsize=10) #

        # Now the same but for the std of the pairwise correlations:
        data        = np.squeeze(np.nanmean(varcorr[:,0,:,0],axis=0))
        data        = np.reshape(data,(2,2))

        ax2 = fig.add_subplot(gspec[1, 2])
        pcm = ax2.imshow(data,cmap='plasma',vmin=my_floor(np.min(data),2),vmax=my_ceil(np.max(data),2))
        ax2.set_xticks([0,1],labels=xlabels)
        ax2.xaxis.tick_top()
        ax2.set_yticks([0,1],labels=ylabels)
        ax2.set_title('Std')
        fig.colorbar(pcm, ax=ax2)
        
        for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
            data1 = varcorr[:,0,ix,0]
            data2 = varcorr[:,0,iy,0]
            pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
            # pval = pval * 6 #bonferroni correction
            if pval<0.05:
                ax2.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
                ax2.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                                    weight='bold',fontsize=10) #
        
        # plt.suptitle('%s %s' % (areapair,corr_type),fontsize=12)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Layer_PCA1_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Layer_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')

#%% Plot mean vs standard deviation for labeling across areapairs:
# Umakantha et al. 2023: might signal different population activity fluctuations that are shared

areapairs           = ['V1-V1','PM-PM','V1-PM']
zscoreflag      = True
circres         = 0.25
tickres         = 0.2
lim             = 1.7

# for corr_type in ['trace_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    fig,axes = plt.subplots(1,3,figsize=(9,3))
    for iap,areapair in enumerate(areapairs):
        ax                  = axes[iap]
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
        
        bincenters,histcorr,meancorr,varcorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=True,projpairs=projpairs,
                                                            areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag)

        for ipp,projpair in enumerate(projpairs):
            # ax.scatter(meancorr[:,0,0,ipp],varcorr[:,0,0,ipp],c=clrs_projpairs[ipp],s=4,alpha=0.7)
            ax.errorbar(np.nanmean(meancorr[:,0,0,ipp]),np.nanmean(varcorr[:,0,0,ipp]),
                        np.nanstd(meancorr[:,0,0,ipp]) / np.sqrt(len(ses)),np.nanstd(varcorr[:,0,0,ipp])/ np.sqrt(len(ses)),
                        ecolor=clrs_projpairs[ipp],elinewidth=1,capsize=3)
        ax.set_xlabel('Mean')
        ax.set_ylabel('Std')

        ax.set_xticks(np.arange(0,lim,tickres))
        ax.set_yticks(np.arange(0,lim,tickres))
        ax.set_xlim([0,lim])
        ax.set_ylim([0,lim])
        # ax.set_xlim([0,my_ceil(np.nanmax(varcorr),2)])
        # ax.set_ylim([0,my_ceil(np.nanmax(varcorr),2)])
        ax.set_title(areapair)

        for radius in np.arange(0,lim*2,circres):
            Drawing_uncolored_circle = plt.Circle( (0, 0), radius, linestyle=':',fill=False)
            ax.add_artist(Drawing_uncolored_circle)
        # ax0.legend(frameon=False,loc='upper left',fontsize=8)
        # ax0.set_xlim([-0.5,0.5])
        # ax0.set_ylim([0,1.1])
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_Z_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdMean_Z_PCA1_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')
        

#%% Plot mean absolute correlation across sessions conditioned on area pairs:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

protocols = ['GR','GN']# protocols = ['IM']

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
    df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,absolute=True,filternear=True,minNcells=10)
    df                  = df_mean
    df                  = df.dropna(axis=0,thresh=1).reset_index(drop=True) #drop occasional missing data
    df                  = df.fillna(df.mean()) #interpolate occasional missing data
    
    fig,axes = plt.subplots(1,1,figsize=(4,4))
    ax                  = axes
    if df.any(axis=None):
        sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
        ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
        annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
        annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                            line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                            correction_format='replace')
        annotator.apply_and_annotate()
        ax.set_ylabel('Abs. correlation')
        ax.set_title('%s' % '_'.join(protocols),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_stationary_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_stationary_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_dF_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_dF_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.pdf'), format = 'pdf')

#%% Plot mean correlation across sessions conditioned on area pairs for pos and neg separately:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

protocols = ['GR','GN']

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,1,figsize=(4,4))
    for isign,sign in enumerate(['pos','neg']):
        ax                  = axes[isign]
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
        df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=True,minNcells=10)
        df                  = df_mean

        df                  = df.dropna(axis=0,thresh=1).reset_index(drop=True) #drop occasional missing data
        df                  = df.fillna(df.mean()) #interpolate occasional missing data

        if df.any(axis=None):
            sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
            if isign==1:
                ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
            else: ax.set_xticks([])
            if isign==1: 
                ax.invert_yaxis()
            annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
            annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=0.05,text_offset=0, 
                                line_width=0.5,comparisons_correction='Benjamini-Hochberg',verbose=False,fontsize=7,
                                correction_format='replace')
            annotator.apply_and_annotate()
            if isign==1:
                ax.invert_yaxis()
            ax.set_ylabel('%s correlation' % sign)
            ax.set_title('%s' % '_'.join(protocols),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()

    # fig.savefig(os.path.join(savedir,'MeanCorr','PosNegCorr_sigOnly_dF_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','PosNegCorr_sigOnly_dF_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.pdf'), format = 'pdf')
    fig.savefig(os.path.join(savedir,'MeanCorr','PosNegCorr_sigOnly_dF_stationary_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','PosNegCorr_sigOnly_dF_stationary_Labeling_Areas_%s_%s' % ('_'.join(protocols),corr_type) + '.pdf'), format = 'pdf')
    
#%% Plot mean absolute correlation across sessions conditioned on area pairs and per protocol:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharex=True,sharey='row')
    for iprot,prot in enumerate(['GR','GN','IM']):
        ax                  = axes[iprot]
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
        df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,absolute=True,filternear=True,minNcells=10)
        df                  = df_mean
        df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
        # df                  = df.dropna() #drop sessions with occasional missing data
        df                  = df.fillna(df.mean()) #interpolate occasional missing data
        
        if df.any(axis=None):
            sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
            ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
            annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
            annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                                line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                                # line_width=1,comparisons_correction=None,verbose=False,
                                correction_format='replace')
            annotator.apply_and_annotate()
            ax.set_ylabel('Correlation')
            ax.set_title('%s' %(prot),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','AbsCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

#%% Plot mean correlation across sessions conditioned on area pairs and per protocol for pos and neg separately:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,3,figsize=(12,6),sharex=True,sharey='row')
    for iprot,prot in enumerate(['GR','GN','IM']):
        for isign,sign in enumerate(['pos','neg']):
            ax                  = axes[isign,iprot]
            ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
            df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=True,minNcells=10)
            df                  = df_mean
            df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
            df                  = df.fillna(df.mean()) #interpolate occasional missing data
            # df                  = df.dropna() #drop sessions with occasional missing data

            if df.any(axis=None):
                sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
                if isign==1:
                    ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
                else: ax.set_xticks([])
                if isign==1: 
                    ax.invert_yaxis()
                annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
                annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=0.05,text_offset=0, 
                                    line_width=0.5,comparisons_correction='Benjamini-Hochberg',verbose=False,fontsize=7,
                                    correction_format='replace')
                annotator.apply_and_annotate()
                if isign==1:
                    ax.invert_yaxis()
                ax.set_ylabel('%s correlation' % sign)
                ax.set_title('%s' %(prot),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

#%% Plot fraction of correlated units across sessions conditioned on area pairs and per protocol:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['trace_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,3,figsize=(12,6),sharex=True,sharey='row')
    for iprot,prot in enumerate(['GR','GN','IM']):
        for isign,sign in enumerate(['pos','neg']):
            ax                  = axes[isign,iprot]
            ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
            df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=True,minNcells=10)
            df                  = df_frac
            df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
            df                  = df.fillna(df.mean()) #interpolate occasional missing data
            
            if df.any(axis=None):
                sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
                if isign==1:
                    ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
                else: ax.set_xticks([])

                annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
                annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                                    line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                                    correction_format='replace')
                annotator.apply_and_annotate()
                ax.set_ylabel('Fraction of %s correlated units' % sign)
                ax.set_title('%s' %(prot),fontsize=12)
            # ax.set_ylim([0,1])
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
    fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

# %% #######################################################################################################
# DELTA RECEPTIVE FIELD:
# ##########################################################################################################

#%%
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                ['LPE11998','2024_05_02'], #GN
                                ['LPE12013','2024_05_02'], #GN
                                ['LPE12013','2024_05_07'], #GN
                                ['LPE11086','2023_12_15'], #GR
                                ['LPE10919','2023_11_06']]) #GR

sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list = np.where(sessiondata['session_id'].isin([x[0] + '_' + x[1] for x in session_list]))[0]
sessions_subset = [sessions[i] for i in sessions_in_list]

#%% 
protocols = ['IM']
protocols = ['SP']
protocols           = ['GR']
# protocols           = ['GN','GR','IM']	

# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
# sessions_in_list    = np.where(np.logical_and(sessiondata['protocol'].isin(protocols),
#                                 ~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20'])))[0]
# sessions_subset     = [sessions[i] for i in sessions_in_list]

#%% ################ Pairwise correlations as a function of pairwise delta RF: #####################
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

#If you override any of these then these pairs will be ignored:
layerpairs          = ' '
# areapairs           = ' '
# projpairs           = ' '
absolute            = False
corr_type           = 'trace_corr'
rf_type             = 'Fsmooth'
binres              = 5
filtersign          = 'pos'
filternear          = True

[binmean,binedges]  =  bin_corr_deltarf(sessions_subset,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs,
                                        corr_type=corr_type,binres=binres,filternear=filternear,
                                        filtersign=filtersign,normalize=False,
                                        sig_thr = 0.001,rf_type=rf_type,mincount=10,absolute=absolute)

fig = plot_bin_corr_deltarf_flex(sessions_subset,binmean,binedges,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs,corr_type='trace_corr',normalize=False)

# fig.savefig(os.path.join(savedir,'Corr_1d_arealabel_%s_%s_abs%s' % (protocols[0],corr_type,absolute) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_GN_SP_RF_0.75dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'TraceCorr_distRF_Areas_Layers_IM_0.5dF_F0.0001_%dsessions_' %nSessions + '.png'), format = 'png')

fig = plot_center_surround_corr(binmean,binedges,layerpairs=layerpairs,areapairs=areapairs,projpairs=projpairs)


#%% ##########################################################################################################
#   CENTER VERSUS SURROUND 
# ##########################################################################################################



#%% ##########################################################################################################
#   2D     DELTA RECEPTIVE FIELD                 2D
# ##########################################################################################################

#%% #########################################################################################
# Contrast: across areas, layers and projection pairs:
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
#If you override any of these with input to the deltarf bin function as ' ', then these pairs will be ignored

#%% Make the 2D, 1D and center surround averages for each protocol and areapair (not layerpair or projpair)
binres              = 5
rf_type             = 'Fsmooth'
filtersign          = None
filternear          = False
# corr_type           = 'trace_corr'
corr_type           = 'noise_corr'

# for ses in sessions:
#     if 'rf_az_F' in ses.celldata and 'rf_az_Fsmooth' in ses.celldata:
#         ses.celldata['rf_az_F'] = ses.celldata['rf_az_Fsmooth']
#         ses.celldata['rf_el_F'] = ses.celldata['rf_el_Fsmooth']
        
# for prot in ['GN','GR','IM','SP','RF']:
for prot in ['GN','GR']:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=' ',
                                corr_type=corr_type,binresolution=binres,rf_type=rf_type,
                                sig_thr = 0.001,filternear=filternear,filtersign=filtersign)
    
    filestring = '%s_%s_%s_azelFsmooth' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        fig = plot_2D_corr_map(binmean,bincounts,bincenters,min_counts = 50,
                                areapairs=areapairs,layerpairs=' ',projpairs=' ')
        fig.suptitle(filestring)
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')
    
        fig = plot_1D_corr_areas(binmean,bincounts,bincenters,min_counts = 50,
                                areapairs=areapairs,layerpairs=' ',projpairs=' ')
        fig.suptitle(filestring)
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.pdf'), format = 'pdf')

        fig = plot_center_surround_corr_areas(binmean,bincenters,centerthr=15,layerpairs=' ',areapairs=areapairs,projpairs=' ')
        fig.suptitle(filestring)
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.pdf'), format = 'pdf')


    # if np.any(bincounts):
    #     figs = []
    #     axs = []

    #     fig = plot_2D_corr_map(binmean,bincounts,bincenters,min_counts = 50,
    #                             areapairs=areapairs,layerpairs=' ',projpairs=' ')
    #     fig.suptitle(filestring)
    #     figs.append(fig)
    #     axs.extend(fig.axes)
        
    #     fig = plot_1D_corr_areas(binmean,bincounts,bincenters,min_counts = 50,
    #                             areapairs=areapairs,layerpairs=' ',projpairs=' ')
    #     fig.suptitle(filestring)
    #     figs.append(fig)
    #     axs.extend(fig.axes)
        
    #     fig = plot_center_surround_corr_areas(binmean,bincenters,centerthr=15,layerpairs=' ',areapairs=areapairs,projpairs=' ')
    #     fig.suptitle(filestring)
    #     figs.append(fig)
    #     axs.extend(fig.axes)
        
    #     fig = plt.figure(figsize=(15,15))
    #     for i,ax in enumerate(axs):
    #         ax.set_position([0.1 + i*0.3,0.1,0.3,0.8])
    #         fig.add_axes(ax)
    #     fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_combined_%s' % filestring + '.png'), format = 'png')
    #     fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_combined_%s' % filestring + '.pdf'), format = 'pdf')

#%%
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False,remove_method='PCA',remove_rank=1)

#%% Make the 2D, 1D and center surround averages for each protocol and areapair and projpair (not layerpair)
binres              = 5
# rf_type             = 'Fneugauss'
rf_type             = 'Fsmooth'
filtersign          = None
filternear          = True
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

# for prot in ['GN','GR','IM','SP','RF']:
for prot in [['GN','GR']]:
# for prot in ['RF']:
# for prot in ['IM']:
# for prot in ['GR']:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
                                corr_type=corr_type,binresolution=binres,rf_type=rf_type,normalize=False,
                                sig_thr = 0.001,filternear=filternear)
    
    # filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
    filestring = '%s_%s_%s_proj' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        fig = plot_2D_corr_map(binmean,bincounts,bincenters,min_counts = 1,gaussian_sigma=1,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')
    
        fig = plot_1D_corr_areas_projs(binmean,bincounts,bincenters,min_counts = 1,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_1D_%s' % filestring + '.pdf'), format = 'pdf')

        fig = plot_center_surround_corr_areas_projs(binmean,bincenters,centerthr=15,layerpairs=' ',areapairs=areapairs,projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_CS_%s' % filestring + '.pdf'), format = 'pdf')

#%% Make a 2D histogram with the distribution of correlation values for each delta RF bin
binres_rf           = 2
binres_corr         = 0.1
rf_type             = 'Fsmooth'
filternear          = True
# corr_type           = 'noise_corr'
corr_type           = 'trace_corr'

for prot in [['GN','GR']]:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [bincounts,bincenters_rf,bincenters_corr] = bin_2d_rangecorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=['L2/3-L2/3'],projpairs=projpairs,
                                corr_type=corr_type,binres_rf=binres_rf,binres_corr=binres_corr,rf_type=rf_type,
                                sig_thr = 0.001,filternear=filternear,noise_thr=0.2)

    # [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
    #                             corr_type=corr_type,binresolution=binres,rf_type=rf_type,normalize=False,
    #                             sig_thr = 0.001,filternear=filternear,filtersign=filtersign)
    
    # filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
    filestring = '%s_%s_%s_proj' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        
        # fig = plot_1D_fraccorr(bincounts,bincenters_rf,bincenters_corr,gaussian_sigma=2,
                                # areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        # fig.suptitle(filestring)
        # plt.tight_layout()
        
        fig = plot_2D_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,gaussian_sigma=2,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_rangecorr_norm034_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaXY_2D_%s' % filestring + '.png'), format = 'png')

        fig = plot_perc_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_Perc_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')

#%% 
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% Make a 2D histogram with the distribution of correlation values for each delta RF bin
binres_rf           = 5
rf_type             = 'Fsmooth'
filternear          = False
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

for prot in [['GN','GR']]:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]
    filestring          = '%s_%s_proj' % (corr_type,rf_type)

    [bincounts,binpos,binneg,bincenters_rf] = bin_1d_fraccorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
    # [bincounts,binpos,binneg,bincenters_rf] = bin_1d_fraccorr_deltarf(sessions_subset,areapairs=areapairs1,layerpairs=['L2/3-L2/3'],projpairs=projpairs,
                                corr_type=corr_type,binres_rf=binres_rf,rf_type=rf_type,
                                sig_thr = 0.001,filternear=filternear,noise_thr=0.2,corr_thr=0.05)

    fig = plot_1D_fraccorr(bincounts,binpos,binneg,bincenters_rf,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,mincounts=50)
    # fig.savefig(os.path.join(savedir,'deltaRF','1DRF','Frac_PosNeg_DeltaRF_GNGR_%s' % filestring + '.png'), format = 'png')

#%% Make a 2D histogram with the distribution of correlation values for each delta XY bin
binres_rf           = 25
binres_corr         = 0.1
rf_type             = 'Fsmooth'
filternear          = False
corr_type           = 'noise_corr'
# corr_type           = 'trace_corr'

# for prot in ['GN','GR','IM','SP','RF']:
for prot in [['GN','GR']]:
    sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # sessions_in_list    = np.where(sessiondata['protocol'].isin([prot]))[0]
    sessions_in_list    = np.where(sessiondata['protocol'].isin(prot))[0]
    sessions_subset     = [sessions[i] for i in sessions_in_list]

    [bincounts,bincenters_rf,bincenters_corr] = bin_2d_rangecorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
                                corr_type=corr_type,binres_rf=binres_rf,binres_corr=binres_corr,rf_type=rf_type,
                                sig_thr = 1,filternear=filternear,noise_thr=0.2)
    
    # [binmean,bincounts,bincenters] = bin_2d_corr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
    #                             corr_type=corr_type,binresolution=binres,rf_type=rf_type,normalize=False,
    #                             sig_thr = 0.001,filternear=filternear,filtersign=filtersign)
    
    filestring = '%s_%s_%s_PCA1_proj' % (corr_type,rf_type,prot[0]+prot[1])
    # filestring = '%s_%s_%s_proj' % (corr_type,rf_type,prot)

    if np.any(bincounts):
        fig = plot_2D_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,gaussian_sigma=1,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.png'), format = 'png')
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaXY_2D_%s' % filestring + '.png'), format = 'png')

        fig = plot_perc_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,
                                areapairs=areapairs,layerpairs=' ',projpairs=projpairs)
        fig.suptitle(filestring)
        plt.tight_layout()
        fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_Perc_%s' % filestring + '.png'), format = 'png')
        # fig.savefig(os.path.join(savedir,'deltaRF','DeltaRF_2D_%s' % filestring + '.pdf'), format = 'pdf')

#%% 

#%% Control figure of counts per bin:

# Make the figure of the counts per bin:
fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
if len(projpairs)==1 and len(areapairs)==1:
    axes = np.array([axes])
axes = axes.reshape(len(projpairs),len(areapairs))

for iap,areapair in enumerate(areapairs):
    for ilp,layerpair in enumerate(layerpairs):
        for ipp,projpair in enumerate(projpairs):
            ax                                              = axes[ipp,iap]
            data                                            = np.log10(bincounts[:,:,iap,ilp,ipp])
            ax.pcolor(delta_az,delta_el,data,vmin=1.5,
                                vmax=np.nanpercentile(np.log10(bincounts),99.9),cmap="hot")
            ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
            ax.set_xlim([-50,50])
            ax.set_ylim([-50,50])
            ax.set_xlabel(u'Δ Azimuth')
            ax.set_ylabel(u'Δ Elevation')
plt.tight_layout()


# binmean = np.nanmean(binmean_ses,axis=5)
# bincounts = np.nansum(bincounts_ses,axis=5)

# binmean_ses[bincounts_ses<10]     = np.nan
