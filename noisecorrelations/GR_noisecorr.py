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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic,binned_statistic_2d
from tqdm import tqdm
from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning, compute_prefori,compute_tuning_wrapper
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
from utils.plot_lib import shaded_error
from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *
from utils.rf_lib import smooth_rf, filter_nearlabeled

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE09830','2023_04_10']])
# session_list        = np.array([['LPE09830','2023_04_10'],
#                                 ['LPE09830','2023_04_12']])
# session_list        = np.array([['LPE11086','2024_01_05']])

#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                # ['LPE11998','2024_05_02'], #GN
                                # ['LPE12013','2024_05_02'], #GN
                                # ['LPE12013','2024_05_07'], #GN
                                # ['LPE11086','2023_12_15'], #GN
                                ['LPE10919','2023_11_06']]) #GR

#%% Load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM'],session_rf=True)

#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF')

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
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

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% ########################### Compute noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_metrics(sessions)

#%% 
sessions = compute_tuning_wrapper(sessions)

#%% How are noise correlations distributed

from scipy.stats import skew

# Compute the average noise correlation for each neuron and store in cell data, loop over sessions:
for ises in range(nSessions):
    # sessions[ises].celldata['noise_corr_avg'] = np.nanmean(sessions[ises].noise_corr,axis=1) 
    sessions[ises].celldata['noise_corr_avg'] = np.nanmean(np.abs(sessions[ises].trace_corr),axis=1) 
    if hasattr(sessions[ises],'respmat'): 
        sessions[ises].celldata['meandF']       = np.nanmean(sessions[ises].respmat,axis=1) 
        sessions[ises].celldata['mediandF']     = np.nanmedian(sessions[ises].respmat,axis=1)
        sessions[ises].celldata['skewrespmat']  = skew(sessions[ises].respmat,axis=1)
    # sessions[ises].celldata['noise_corr_avg'] = np.nanmean(np.abs(sessions[ises].noise_corr),axis=1) 

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\CellProperties\\')


#%%  Scatter plot of average noise correlations versus skew:
def plot_corr_NC_var(sessions,vartoplot):
    cdata = []
    plt.figure(figsize=(4,4))
    for ses in sessions:
        if vartoplot in ses.celldata:
            plt.scatter(ses.celldata[vartoplot],ses.celldata['noise_corr_avg'],s=6,marker='.',alpha=0.5)
            cdata.append(np.corrcoef(ses.celldata[vartoplot],ses.celldata['noise_corr_avg'],rowvar=True)[0,1])
    plt.xlim(np.nanpercentile(celldata[vartoplot],[0.1,99.9]))
    plt.xlabel(vartoplot)
    plt.ylabel('Correlation')
    plt.ylim(0,0.2)
    plt.text(x=np.nanpercentile(celldata[vartoplot],25),y=0.18,s='Mean correlation: %1.3f +- %1.3f' % (np.nanmean(cdata),np.nanstd(cdata)))
    # plt.savefig(os.path.join(savedir,'NoiseCorrelations','%s_vs_NC' % vartoplot + '.png'), format = 'png')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,'AbsTC_vs_%s' % vartoplot + '.png'), format = 'png')

#%%  Scatter plot of average correlations versus skew:
plot_corr_NC_var(sessions,vartoplot = 'skew')

#%%  Scatter plot of average correlations versus depth:
plot_corr_NC_var(sessions,vartoplot = 'depth')

#%%  Scatter plot of average correlations versus tuning variance:
plot_corr_NC_var(sessions,vartoplot = 'tuning_var')

#%%  Scatter plot of average correlations versus noise level:
plot_corr_NC_var(sessions,vartoplot = 'noise_level')

#%%  Scatter plot of average correlations versus event rate level:
plot_corr_NC_var(sessions,vartoplot = 'event_rate')

#%%  Scatter plot of average correlations versus fluorescence channel 2:
plot_corr_NC_var(sessions,vartoplot = 'meanF')

#%%  Scatter plot of average correlations versus fluorescence channel 2:
plot_corr_NC_var(sessions,vartoplot = 'meanF_chan2')

#%%  Scatter plot of average correlations versus tdTomato in ROI:
plot_corr_NC_var(sessions,vartoplot = 'frac_red_in_ROI')

#%%  Scatter plot of average correlations versus fraction of ROI that has tdtomato cell body:
plot_corr_NC_var(sessions,vartoplot = 'frac_of_ROI_red')

#%%  Scatter plot of average correlations versus receptive field fit:
plot_corr_NC_var(sessions,vartoplot = 'rf_p_F')

#%%  Scatter plot of average correlations versus mean response across trials:
plot_corr_NC_var(sessions,vartoplot = 'meandF')

#%%  Scatter plot of average correlations versus median response across trials:
plot_corr_NC_var(sessions,vartoplot = 'mediandF')

#%%  Scatter plot of average correlations versus mean response across trials:
plot_corr_NC_var(sessions,vartoplot = 'skewrespmat')




#%%  Scatter plot of average noise correlations versus skew:
def plot_corr_NC_var_2d(sessions,vartoplot):
    nbins       = 10
    binedges    = np.linspace(np.nanpercentile(sessions[0].celldata[vartoplot],2),np.nanpercentile(sessions[0].celldata[vartoplot],98),nbins+1)
    
    cdata = np.empty((len(sessions),nbins,nbins))   
    for ises,ses in enumerate(sessions):
        if vartoplot in ses.celldata:
            vdata = ses.trace_corr[~np.isnan(ses.trace_corr)].flatten()
            xdata,ydata = np.meshgrid(ses.celldata[vartoplot],ses.celldata[vartoplot])[:]
            xdata = xdata[~np.isnan(ses.trace_corr)].flatten()
            ydata = ydata[~np.isnan(ses.trace_corr)].flatten()
            cdata[ises,:,:]   = binned_statistic_2d(x=xdata, y=ydata, values=vdata,bins=binedges, statistic='mean')[0]

    fig = plt.figure(figsize=(4,4))

    plt.imshow(np.nanmean(cdata,axis=0).squeeze(),extent=[binedges[0],binedges[-1],binedges[0],binedges[-1]],
               vmin=np.nanpercentile(np.nanmean(cdata,axis=0),2),vmax=np.nanpercentile(np.nanmean(cdata,axis=0),98),origin='lower')
    plt.xticks(binedges[::2],fontsize=8)
    plt.yticks(binedges[::2],fontsize=8)
    # plt.xlabel(vartoplot)
    # plt.ylabel(vartoplot)
    plt.title(vartoplot)
    cbar = plt.colorbar(shrink=0.4)
    cbar.set_label('Mean trace corr')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,'2D_tracecorr_vs_%s' % vartoplot + '.png'), format = 'png')

#%%  2D plot of trace correlations versus skew:
plot_corr_NC_var_2d(sessions,vartoplot = 'skew')

#%%  2D plot of trace correlations versus depth:
plot_corr_NC_var_2d(sessions,vartoplot = 'depth')

#%%  2D plot of trace correlations versus tuning variance:
plot_corr_NC_var_2d(sessions,vartoplot = 'tuning_var')

#%%  2D plot of trace correlations versus noise level:
plot_corr_NC_var_2d(sessions,vartoplot = 'noise_level')

#%%  2D plot of trace correlations versus event rate level:
plot_corr_NC_var_2d(sessions,vartoplot = 'event_rate')

#%%  2D plot of trace correlations versus fluorescence:
plot_corr_NC_var_2d(sessions,vartoplot = 'meanF')

#%%  2D plot of trace correlations versus fluorescence channel 2:
plot_corr_NC_var_2d(sessions,vartoplot = 'meanF_chan2')

#%% 
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
protocols           = ['GR','GN']
ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

#%%  Scatter plot of average correlations versus receptive field fit:
plot_corr_NC_var_2d(ses,vartoplot = 'rf_r2_F')

#%%  Scatter plot of average correlations versus receptive field fit:
plot_corr_NC_var_2d(ses,vartoplot = 'rf_sy_F')

#%% 
plot_corr_NC_var_2d(ses,vartoplot = 'noise_corr_avg')


#%% 

celldata['cell_id'][celldata['meandF']<0]
celldata['cell_id'][celldata['meandF']>4]


celldata['cell_id'][celldata['noise_corr_avg']>0.19]


#%% Show relationships between multiple cell data properties at the same time:
plotvars = ['noise_corr_avg','skew','chan2_prob','depth','redcell','tuning_var','event_rate']
sns.pairplot(data=celldata[plotvars],hue='redcell')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Pairplot_NC' + '.png'), format = 'png')

sns.heatmap(data=celldata[plotvars].corr(),vmin=-1,vmax=1,cmap='bwr')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Heatmap_NC_features' + '.png'), format = 'png')

#%% Show tuning variance vs noise correlation:
clr_labeled = get_clr_labeled()
sns.histplot(data=celldata,x='tuning_var',y='noise_corr_avg',hue='redcell',palette=clr_labeled,
             bins=40,alpha=0.5,cbar=True,cbar_kws={'norm': 1})
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Tuning Variance vs NC' + '.png'), format = 'png')

#%% Plot fraction of visuall responsive: 

# #%%  construct dataframe with all pairwise measurements:
# df_allpairs  = pd.DataFrame()

# for ises in range(nSessions):
#     [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

#     tempdf  = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr.flatten(),
#                     'DeltaPrefOri': sessions[ises].delta_pref.flatten(),
#                     'AreaPair': sessions[ises].areamat.flatten(),
#                     'DistXYPair': sessions[ises].distmat_xy.flatten(),
#                     'DistXYZPair': sessions[ises].distmat_xyz.flatten(),
#                     'DistRfPair': sessions[ises].distmat_rf.flatten(),
#                     'AreaLabelPair': sessions[ises].arealabelmat.flatten(),
#                     'LabelPair': sessions[ises].labelmat.flatten()}).dropna(how='all') 
#                     #drop all rows that have all nan (diagonal + repeat below daig)
#     df_allpairs  = pd.concat([df_allpairs, tempdf], ignore_index=True).reset_index(drop=True)


#%% #### 
sesidx = 3
sesidx = np.where([ses.sessiondata['session_id'][0] == 'LPE09830_2023_04_12' for ses in sessions])[0][0]
fig = plot_PCA_gratings_3D(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA','PCA_3D_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_PCA_gratings_3D(sessions[sesidx],export_animation=True)

fig = plot_PCA_gratings(sessions[sesidx],cellfilter=sessions[sesidx].celldata['redcell'].to_numpy()==1)
fig.savefig(os.path.join(savedir,'PCA','PCA_Gratings_All_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% ##################### Plot control figure of signal and noise corrs ##############################
sesidx = 0
fig = plt.subplots(figsize=(8,5))
plt.imshow(sessions[sesidx].sig_corr, cmap='coolwarm',
           vmin=np.nanpercentile(sessions[sesidx].sig_corr,15),
           vmax=np.nanpercentile(sessions[sesidx].sig_corr,85))
# plt.xlabel = 'Neurons'
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Signal_Correlation_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plt.figure(figsize=(8,5))
plt.imshow(sessions[sesidx].noise_corr, cmap='coolwarm',
           vmin=np.nanpercentile(sessions[sesidx].noise_corr,5),
           vmax=np.nanpercentile(sessions[sesidx].noise_corr,95))
plt.savefig(os.path.join(savedir,'NoiseCorrelations','Noise_Correlation_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% Plotting Noise Correlation distribution across all pairs:
fig,ax = plt.subplots(figsize=(5,4))
for ses in tqdm(sessions,total=len(sessions),desc= 'Kernel Density Estimation for each session: '):
    sns.kdeplot(data=ses.noise_corr.flatten(),ax=ax,label=ses.sessiondata['session_id'][0])
plt.xlim([-0.15,0.4])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_distribution_allSessions.png'), format = 'png')

#%% ##################### Noise correlations within and across areas: #########################
areapairs = ['V1-V1','V1-PM','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)
pairs = [('V1-V1','V1-PM'),('PM-PM','V1-PM')] #for statistics

center = np.zeros((nSessions,len(areapairs)))
for ises in tqdm(range(nSessions),desc= 'Averaging noise correlations within and across areas: '):
    for iap,areapair in enumerate(areapairs):
        areafilter =  filter_2d_areapair(sessions[ises],areapair)
  
        # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.1,sessions[ises].celldata['tuning_var']>0.1)
        # signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
        # cellfilter = np.logical_and(areafilter,signalfilter)
        cellfilter = np.logical_and(areafilter,areafilter)
        # center[ises,iap] = np.nanmean(sessions[ises].noise_corr[cellfilter])
        center[ises,iap] = np.nanmean(sessions[ises].trace_corr[cellfilter])
df = pd.DataFrame(data=center,columns=areapairs)

#%% Make a barplot with error bars of the mean NC across sessions conditioned on area pairs:
fig,ax = plt.subplots(figsize=(3,3))
sns.barplot(data=df,errorbar='se',palette=clrs_areapairs,
            order = areapairs,hue_order=areapairs)
sns.stripplot(data=df,color='k',ax=ax,size=3,alpha=0.5,jitter=0.2)

# annotator = Annotator(ax, pairs, data=center, x="AreaPair", y='NoiseCorrelation', order=areapairs)
annotator = Annotator(ax, pairs, data=df.dropna(inplace=False),order=areapairs)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.configure(test='t-test_paired', text_format='star', loc='inside')
annotator.apply_and_annotate()

# plt.yticks(np.arange(0, 1, step=0.01)) 
# plt.ylim([0,0.07])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_average_%dsessions' %nSessions + '.png'), format = 'png')

#%% ###################################################################
####### Noise correlations as a function of anatomical distance ####
areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='noise_corr',normalize=False)

#%% Make the figure per protocol:
fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='noise_corr')
fig.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_anatomdistance_perArea_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ###################################################################
# Show distribution of delta receptive fields across sessions:
# fig,ax = plt.subplots(1,2,figsize=(5,4))
fig,ax = plt.subplots(figsize=(5,4))
for ses in tqdm(sessions,total=len(sessions),desc= 'Histogram of delta RF for each session: '):
    if 'rf_p_Fneu' in ses.celldata:
        sns.histplot(ses.distmat_rf.flatten(),bins=np.arange(-5,250,step=5),ax=ax,
                     fill=False,element="step",stat="percent",alpha=0.8,label=ses.sessiondata['session_id'][0])
ax.set(xlabel='delta RF')
ax.legend(loc='upper right',frameon=False,fontsize=7)
fig.savefig(os.path.join(savedir,'Distribution_deltaRF_%dsessions' %nSessions + '.png'), format = 'png')

#%% ###################################################################
# areapairs = ['V1-V1','V1-PM','PM-PM']
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

[binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type='noise_corr',normalize=False)

#%% Make the figure per protocol:

fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type='noise_corr')


#%% ################ Pairwise trace correlations as a function of pairwise delta RF: #####################
areapairs = ['V1-V1','PM-PM','V1-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

[binmean,binedges] =  bin_corr_deltarf(sessions,areapairs,corr_type='noise_corr',normalize=True)

#%% Make the figure:
fig = plot_bin_corr_deltarf(sessions,binmean,binedges,areapairs,corr_type='noise_corr')

fig.savefig(os.path.join(savedir,'TraceCorr_distRF_Protocols_%dsessions_' %nSessions + areapair + '.png'), format = 'png')


#%% #########################################################################################
# Plot 2D noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

# from utils.RRRlib import *

# X = np.column_stack((sessions[ises].respmat_runspeed,sessions[ises].respmat_videome))
# Y = sessions[ises].respmat.T

# sessions[ises].respmat = regress_out_behavior_modulation(sessions[ises],X,Y,nvideoPCs = 30,rank=2).T

# Recompute noise correlations without setting half triangle to nan
sessions =  compute_signal_noise_correlation(sessions,uppertriangular=False)

rotate_prefori  = False
min_counts      = 500 # minimum pairwise observation to include bin

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap(sessions,binresolution=5,
                                                         rotate_prefori=rotate_prefori,
                                                         rotate_deltaprefori=False)
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

## Show the counts of pairs:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(countsRFmat,vmin=np.percentile(countsRFmat,5),vmax=np.percentile(countsRFmat,99),
               interpolation='none',extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='counts')
if not rotate_prefori:
    plt.xlabel('delta Azimuth')
    plt.ylabel('delta Elevation')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Counts_%dsessions' %nSessions  + '.png'), format = 'png')
else:
    plt.xlabel('Collinear')
    plt.ylabel('Orthogonal')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Counts_Rotated_%dsessions' %nSessions  + '.png'), format = 'png')

## Show the noise correlation map:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(noiseRFmat_mean,vmin=np.nanpercentile(noiseRFmat_mean,5),vmax=np.nanpercentile(noiseRFmat_mean,95),
               cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='noise correlation')
if not rotate_prefori:
    plt.xlabel('delta Azimuth')
    plt.ylabel('delta Elevation')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_%dsessions' %nSessions  + '.png'), format = 'png')
else:
    plt.xlabel('Collinear')
    plt.ylabel('Orthogonal')
    # fig.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NoiseCorrMap_Rotated_%dsessions' %nSessions  + '.png'), format = 'png')

# sns.histplot(celldata['pref_ori'],bins=oris)

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap_perori(sessions,binresolution=5,
                                                                rotate_prefori=True)

## Show the noise correlation map:
fig,axes = plt.subplots(4,4,figsize=(10,10))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(noiseRFmat_mean[i*4+j,:,:],vmin=0.02,vmax=0.07,cmap="hot",
                         interpolation="none",extent=np.flipud(binrange).flatten())

noiseRFmat_mean = np.nanmean(noiseRFmat_mean,axis=0)

min_counts = 500
countsRFmat = np.sum(countsRFmat,axis=0)
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

## Show the noise correlation map:
fig,ax = plt.subplots(1,1,figsize=(7,4))
IM = ax.imshow(noiseRFmat_mean,vmin=0.034,vmax=0.082,cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
plt.colorbar(IM,fraction=0.026, pad=0.04,label='noise correlation')

#%% #########################################################################################
# Contrast: across areas

areas   = ['V1','PM']

[noiseRFmat_mean,countsRFmat,binrange] = noisecorr_rfmap_areas(sessions,binresolution=5,
                                                                 rotate_prefori=False,thr_tuned=0.0,
                                                                 thr_rf_p=0.01)

min_counts = 250
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(2,2,figsize=(10,7))
for i in range(2):
    for j in range(2):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean,10),
                         vmax=np.nanpercentile(noiseRFmat_mean,99),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(areas[i] + '-' + areas[j])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Interarea_%dsessions' %nSessions  + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Interarea_smooth_%dsessions' %nSessions  + '.png'), format = 'png')

#%% Plot Circular tuning:

# binYaxis = np.arange(start=binrange[0,0],stop=binrange[0,1],step=5)
# binXaxis = np.arange(start=binrange[1,0],stop=binrange[1,1],step=5)
# X,Y = np.meshgrid(binXaxis,binYaxis)

# polar_r         = np.mod(np.arctan2(X,Y)+np.pi/2,np.pi*2)
# polar_theta     = np.sqrt(X**2 + Y**2)

# # plt.imshow(polar_r)
# # plt.imshow(polar_theta)
# polardata = polar_r.flatten()
# noisedata = noiseRFmat_mean[1,1,:,:].flatten()

# [NC_circbin,bin_edges,y] = binned_statistic(x=polardata[~np.isnan(noisedata)],
#                         values = noisedata[~np.isnan(noisedata)],
#                         statistic='mean',bins=np.deg2rad(np.arange(0,360,step=10)))
# plt.plot(bin_edges[:-1]+5/2,NC_circbin)

#%% #########################################################################################
# Contrasts: across areas and projection identity      

[noiseRFmat_mean,countsRFmat,binrange,legendlabels] = noisecorr_rfmap_areas_projections(sessions,binresolution=7.5,
                                                                 rotate_prefori=False,thr_tuned=0.00,
                                                                 thr_rf_p=0.05)

min_counts = 250
noiseRFmat_mean[countsRFmat<min_counts] = np.nan

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean[i,j,:,:],10),
                         vmax=np.nanpercentile(noiseRFmat_mean[i,j,:,:],96),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        # axes[i,j].imshow(noiseRFmat_mean[i,j,:,:],vmin=np.nanpercentile(noiseRFmat_mean,10),
                        #  vmax=np.nanpercentile(noiseRFmat_mean,98),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
        axes[i,j].set_xlim([-75,75])
        axes[i,j].set_ylim([-75,75])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Area_Proj_%dsessions' %nSessions  + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_rotate_smooth_Area_Proj_%dsessions' %nSessions  + '.png'), format = 'png')

fig,axes = plt.subplots(4,4,figsize=(10,7))
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(np.log10(countsRFmat[i,j,:,:]),vmax=np.nanpercentile(np.log10(countsRFmat),99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
        axes[i,j].set_title(legendlabels[i,j])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Area_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','2D_NC_Map_Area_rotate_Proj_Counts_%dsessions' %nSessions  + '.png'), format = 'png')

####################################### ####################################### #######################
#################################### LABELED AND UNLABELED ############################################

#%% Plot the tuning parameters for labeled and unlabeled cells

recombinases = celldata['recombinase'].unique()[::-1]
clr_labeled = get_clr_recombinase(recombinases)
pairs = [('non','flp'),('non','cre')] #for statistics

tuning_metric = 'tuning_var'
fig,ax = plt.subplots(figsize=(3,3))

handles = sns.barplot(data=celldata,x='recombinase',y=tuning_metric,order=recombinases,
                      hue_order=recombinases,palette=clr_labeled)
annotator = Annotator(ax, pairs, data=celldata, x="recombinase", y=tuning_metric, order=recombinases)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.ylabel('Orientation Selectivity')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Labeling_tuning_%dsessions' %nSessions + '.png'), format = 'png')

#%% Plot the tuning parameters for labeled and unlabeled cells, per area

labs = np.array(['unl','lab'])
celldata['area_labeled'] = celldata['roi_name'] + labs[celldata['redcell'].to_numpy().astype(int)]
celldata['area_recombinase'] = celldata['roi_name'] + celldata['recombinase']

area_labeled = celldata['area_labeled'].unique()[::-1]
clr_labeled = get_clr_area_labeled(area_labeled)
pairs = [('V1unl','V1lab'),('PMunl','PMlab')] #for statistics

tuning_metric = 'OSI'
fig,ax = plt.subplots(figsize=(3,3))

handles = sns.barplot(data=celldata,x='area_labeled',y=tuning_metric,order=area_labeled,
                      hue_order=area_labeled,palette=clr_labeled)
annotator = Annotator(ax, pairs, data=celldata, x="area_labeled", y=tuning_metric, order=area_labeled)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.ylabel('Orientation Selectivity')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Area_Labeling_TuningVar_%dsessions' %nSessions + '.png'), format = 'png')


#%% Same but for recombinase with area:
labs = np.array(['unl','lab'])
celldata['area_labeled'] = celldata['roi_name'] + labs[celldata['redcell'].to_numpy().astype(int)]
celldata['area_recombinase'] = celldata['roi_name'] + celldata['recombinase']

area_recombinase = np.sort(celldata['area_recombinase'].unique()[::-1])
recombinases = celldata['recombinase'].unique()[::-1]
clr_labeled = get_clr_recombinase(recombinases)
pairs = [('V1cre','V1non'),('PMcre','PMnon'),('PMflp','PMnon'),('V1flp','V1non')] #for statistics

tuning_metric = 'OSI'
fig,ax = plt.subplots(figsize=(4,3))

handles = sns.barplot(data=celldata,x='area_recombinase',y=tuning_metric,order=area_recombinase,
                      hue_order=area_recombinase,palette=clr_labeled)
annotator = Annotator(ax, pairs, data=celldata, x="area_recombinase", y=tuning_metric, order=area_recombinase)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.ylabel('Orientation Selectivity')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Area_Recombinase_TuningVar_%dsessions' %nSessions + '.png'), format = 'png')

#%% ##################### Noise correlations within and across areas: #########################


#%% Make a barplot with error bars of the mean NC across sessions conditioned on area pairs:

fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se')#,labels=legendlabels_upper_tri)
plt.plot(df.T,linewidth=0.25,c='k',alpha=0.5)	
sns.stripplot(data=df,palette='dark:k',ax=ax,size=3,alpha=0.5,jitter=0.1)
ax.set_xticklabels(labels=legendlabels_upper_tri,rotation=90,fontsize=8)
ax.set_ylim([0,0.15])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'TraceCorr_labeling_areas_Indiv%dsessions' %nSessions + '.png'), format = 'png')

#%% With stats:
fig,ax = plt.subplots(figsize=(5,4))
sns.barplot(data=df,estimator="mean",errorbar='se')#,labels=legendlabels_upper_tri)
ax.set_xticklabels(labels=legendlabels_upper_tri,rotation=90,fontsize=8)
pairs = [('V1unl-V1unl','V1lab-V1lab'),
         ('V1unl-V1unl','V1unl-V1lab'),
         ('V1unl-V1lab','V1lab-V1lab'),
         ('PMunl-PMunl','PMunl-PMlab'),
         ('PMunl-PMunl','PMlab-PMlab'),
         ('PMunl-PMlab','PMlab-PMlab'),
         ('V1unl-PMlab','V1lab-PMlab'),
         ('V1lab-PMunl','V1lab-PMlab'),
         ('V1unl-PMunl','V1lab-PMlab'),
         ] #for statistics

annotator = Annotator(ax, pairs, data=df,order=list(legendlabels_upper_tri))
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.configure(test='t-test_paired', text_format='star', loc='inside')
annotator.apply_and_annotate()
ax.set_ylim([0,0.13])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'TraceCorr_labeling_areas_%dsessions' %nSessions + '.png'), format = 'png')

#%% ############################################################################################
################### Noise correlations as a function of pairwise distance: ####################
############################# Labeled vs unlabeled neurons #######################################

areapairs = ['V1-V1','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

labelpairs = df_allpairs['LabelPair'].unique()
clrs_labelpairs = get_clr_labelpairs(labelpairs)

binedges = np.arange(0,1000,50) 
nbins= len(binedges)-1      
# binmean = np.empty((nSessions,len(areapairs),len(labelpairs),nbins))
binmean = np.full((nSessions,len(areapairs),len(labelpairs),nbins),np.nan)

handles = []
tuningthr = 0
for iap,areapair in enumerate(areapairs):
    for ilp,labelpair in enumerate(labelpairs):
        for ises in range(nSessions):
            areafilter = sessions[ises].areamat==areapair
            labelfilter = sessions[ises].labelmat==labelpair
            # filter = sessions[ises].celldata['tuning_var']>0
            signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>tuningthr,sessions[ises].celldata['tuning_var']>tuningthr)
            signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
            # filter = np.logical_and(areafilter,labelfilter)
            filter = np.all((signalfilter,areafilter,labelfilter),axis=0)
            if filter.any():
                binmean[ises,iap,ilp,:] = binned_statistic(x=sessions[ises].distmat_xy[filter].flatten(),
                                                values=sessions[ises].noise_corr[filter].flatten(),
                            statistic='mean', bins=binedges)[0]

plt.figure(figsize=(6,3))
for iap,areapair in enumerate(areapairs):
    ax = plt.subplot(1,len(areapairs),iap+1)
    for ilp,labelpair in enumerate(labelpairs):
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
        handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_labelpairs[ilp]))
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),
                                    # yerror=binmean[:,iap,ilp,:].squeeze()/5,color=clrs_labelpairs[ilp]))
    ax.set(xlabel=r'Anatomical distance ($\mu$m)',ylabel='Noise Correlation',
           yticks=np.arange(0, 1, step=0.01),xticks=np.arange(0, 600, step=100))
    ax.set(xlim=[10,500],ylim=[0,0.075])
    ax.legend(handles,labelpairs,frameon=False,loc='upper right')
    plt.tight_layout()
    ax.set_title(areapair)
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_anatomdistance_perArea_Labeled_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% ############################################################################################
################### Noise correlations as a function of pairwise delta RF : ####################
############################# Labeled vs unlabeled neurons #######################################

areapairs = ['V1-V1','V1-PM','PM-PM']
clrs_areapairs = get_clr_area_pairs(areapairs)

labelpairs = np.unique(sessions[ises].labelmat[sessions[ises].labelmat != ''])
clrs_labelpairs = get_clr_labelpairs(labelpairs)

binedges = np.arange(0,120,10) 
nbins= len(binedges)-1      
# binmean = np.zeros((nSessions,len(areapairs),len(labelpairs),nbins))
binmean = np.full((nSessions,len(areapairs),len(labelpairs),nbins),np.nan)

handles = []
for iap,areapair in enumerate(areapairs):
    for ilp,labelpair in enumerate(labelpairs):
        for ises in range(nSessions):
            areafilter = sessions[ises].areamat==areapair
            labelfilter = sessions[ises].labelmat==labelpair
            cellfilter = np.logical_and(areafilter,labelfilter)

            # signalfilter = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
            # signalfilter = np.logical_and(signalfilter[0],signalfilter[1])
            # filter = np.logical_and(np.logical_and(signalfilter,areafilter),labelfilter)
            if np.any(cellfilter):
                binmean[ises,iap,ilp,:] = binned_statistic(x=sessions[ises].distmat_rf[cellfilter].flatten(),
                                                values=sessions[ises].noise_corr[cellfilter].flatten(),
                            statistic='mean', bins=binedges)[0]


#%% Make the figure:
plt.figure(figsize=(9,4))
for iap,areapair in enumerate(areapairs):
    ax = plt.subplot(1,3,iap+1)
    for ilp,labelpair in enumerate(labelpairs):
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
        # handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),
                                    # yerror=binmean[:,iap,ilp,:].squeeze()/5,color=clrs_labelpairs[ilp]))
        handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,ilp,:].squeeze(),error='sem',color=clrs_labelpairs[ilp]))
    ax.set(xlabel='Delta RF',ylabel='Noise Correlation',
           yticks=np.arange(0, 1, step=0.01),xticks=np.arange(0, 120, step=10))
    ax.set(xlim=[0,60],ylim=[0,0.11])
    ax.legend(handles,labelpairs,frameon=False,loc='upper right')
    plt.tight_layout()
    ax.set_title(areapair)
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NoiseCorr_deltaRF_Labeled_%dsessions' % nSessions + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea_regressout' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')



sessions[0].celldata['rf_azimuth']
sessions[0].celldata['rf_p']

# %% @#















### Legacy plots: 


#%% Cell showing negative noise correlations with increasing fluo in Chan 2 if not curated session
sesidx = 0
filter_area = sessions[sesidx].celldata['roi_name']=='V1'
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,3))
ax1.scatter(sessions[sesidx].celldata['meanF_chan2'][filter_area],
            np.nanmean(sessions[sesidx].noise_corr[filter_area,:],axis=1),alpha = 0.7,c=clrs_areapairs[0],s=5)
ax1.set_xlabel('F Chan2')
ax1.set_ylabel('Noise Correlations')
filter_area = sessions[sesidx].celldata['roi_name']=='PM'
ax2.scatter(sessions[sesidx].celldata['meanF_chan2'][filter_area],
            np.nanmean(sessions[sesidx].noise_corr[filter_area,:],axis=1),alpha = 0.7,c=clrs_areapairs[1],s=5)
ax2.set_xlabel('F Chan2')
# ax2.set_ylabel('Noise Correlations')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'NoiseCorr_FChan2_curated_%s' % sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

