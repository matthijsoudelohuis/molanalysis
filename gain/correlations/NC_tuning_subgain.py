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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator
from scipy.stats import binned_statistic_2d

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.corr_lib import *
from utils.tuning import *
from utils.gain_lib import * 
from scipy.stats import binned_statistic,binned_statistic_2d

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\NoiseCorrelations\\')

#%% #############################################################################
session_list        = np.array([['LPE10919_2023_11_06']])
session_list        = np.array([['LPE09665_2023_03_21'], #GR
                                ['LPE10919_2023_11_06']]) #GR

session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10884_2023_10_20']]) #GR
# session_list        = np.array([['LPE12223','2024_06_10']])

sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_areas=['V1','PM']) 

sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM']) 


#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

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









#%% Keep separate instances of the original data and the gain subtracted data:
sessions_orig       = copy.deepcopy(sessions)
sessions_nogain     = copy.deepcopy(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
sessions_orig       = compute_signal_noise_correlation(sessions_orig,filter_stationary=False,uppertriangular=False)
# sessions_nogain     = compute_signal_noise_correlation(sessions_nogain,filter_stationary=False,remove_method='PCA',remove_rank=2)
sessions_nogain     = compute_signal_noise_correlation(sessions_nogain,filter_stationary=False,uppertriangular=False,remove_method='GM')
# sessions       = compute_signal_noise_correlation(sessions,filter_stationary=False,uppertriangular=False)

#%% Show reduction in noise correlation matrix with gain subtraction:
fig,axes = plt.subplots(1,2,figsize=(11,4))
axes[0].imshow(sessions_orig[0].noise_corr,vmin=0,vmax=0.05)
axes[1].imshow(sessions_nogain[0].noise_corr,vmin=0,vmax=0.05)

#%% ########################## Subtract gain model from respmat ###################################
for ises in range(nSessions):
    data_hat                        = pop_rate_gain_model(sessions_nogain[ises].respmat, sessions_nogain[ises].trialdata['Orientation'])
    sessions_nogain[ises].respmat   = sessions_nogain[ises].respmat - data_hat


#%% ############################### Show response with and without running #################
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris       = 16
nCells      = len(celldata)
oris        = np.sort(sessions[0].trialdata['Orientation'].unique())


#%% #########################################################################################
# Plot noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

areapairs = ['V1-V1','PM-PM','V1-PM']
data = np.empty((nSessions,len(areapairs),len(oris),3)) #for each session, combination of delta pref store the mean noise corr for all and for the top and bottom tuned percentages

extremefrac = 10
binedges = np.arange(0,360+22.5,22.5)-22.5/2
# fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    sessions_orig[ises].delta_pref = np.abs(np.subtract.outer(sessions_orig[ises].celldata['pref_ori'].values,sessions_orig[ises].celldata['pref_ori'].values))

    for iap,areapair in enumerate(areapairs):
        areafilter      = filter_2d_areapair(sessions_orig[ises],areapair)

        tunefilter_all    = np.ones(areafilter.shape).astype(bool)

        tunefilter_unt   = np.meshgrid(sessions_orig[ises].celldata['tuning_var']<np.percentile(sessions_orig[ises].celldata['tuning_var'],extremefrac),
                                      sessions_orig[ises].celldata['tuning_var']<np.percentile(sessions_orig[ises].celldata['tuning_var'],extremefrac))
        tunefilter_unt    = np.logical_and(tunefilter_unt[0],tunefilter_unt[1])

        tunefilter_tune    = np.meshgrid(sessions_orig[ises].celldata['tuning_var']>np.percentile(sessions_orig[ises].celldata['tuning_var'],100-extremefrac),
                                      sessions_orig[ises].celldata['tuning_var']>np.percentile(sessions_orig[ises].celldata['tuning_var'],100-extremefrac))
        tunefilter_tune    = np.logical_and(tunefilter_tune[0],tunefilter_tune[1])

        nanfilter         = np.all((~np.isnan(sessions_orig[ises].noise_corr),~np.isnan(sessions_orig[ises].delta_pref)),axis=0)

        cellfilter = np.all((areafilter,tunefilter_all,nanfilter),axis=0)
        data[ises,iap,:,0] = binned_statistic(x=sessions_orig[ises].delta_pref[cellfilter].flatten(),
                                              values=sessions_orig[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]
        
        cellfilter = np.all((areafilter,tunefilter_unt,nanfilter),axis=0)
        data[ises,iap,:,1] = binned_statistic(x=sessions_orig[ises].delta_pref[cellfilter].flatten(),
                                              values=sessions_orig[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]

        cellfilter = np.all((areafilter,tunefilter_tune,nanfilter),axis=0)
        data[ises,iap,:,2] = binned_statistic(x=sessions_orig[ises].delta_pref[cellfilter].flatten(),
                                              values=sessions_orig[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]

#%% Show tuning dependent noise correlations:
# clrs    = sns.color_palette('inferno', 3)
clrs    = ['black','blue','red']
perc_labels = ['All','Bottom ' + str(extremefrac) + '%\n tuned','Top ' + str(extremefrac) + '%\n tuned',]

fig,ax = plt.subplots(1,1,figsize=(3.5,3))
iap = 0
handles = []
handles.append(shaded_error(ax=ax,x=oris,y=data[:,iap,:,0].squeeze(),center='mean',error='std',color=clrs[0]))
handles.append(shaded_error(ax=ax,x=oris,y=data[:,iap,:,1].squeeze(),center='mean',error='std',color=clrs[1]))
handles.append(shaded_error(ax=ax,x=oris,y=data[:,iap,:,2].squeeze(),center='mean',error='std',color=clrs[2]))
ax.set_xlabel(r'$\Delta$ Pref. Ori')
ax.set_ylabel('NoiseCorrelation')
ax.set_ylim([0,my_ceil(np.nanmax(data[:,iap,:,:]),2)])
ax.set_title('')
ax.legend(handles,perc_labels,frameon=False,loc='upper right',fontsize=8)
sns.despine(trim=False,top=True,right=True,offset=3)
ax.set_xticks(oris[::2],oris[::2].astype(int),rotation=45)
plt.tight_layout()
my_savefig(fig, savedir, 'NC_deltaOri_V1_tuningperc', formats = ['png'])
# plt.savefig(os.path.join(savedir,'PairwiseCorrelations','NC_deltaOri_V1_tuningperc' + '.png'), format = 'png')

#%% Show within and across area tuning dependent correlations:
clrs_areapairs = get_clr_area_pairs(areapairs)

fig,ax = plt.subplots(1,1,figsize=(3.5,3))
handles  = []
for iap,areapair in enumerate(areapairs):
    handles.append(shaded_error(ax=ax,x=oris,y=data[:,iap,:,0].squeeze(),center='mean',error='sem',color=clrs_areapairs[iap]))
ax.set_xlabel('Delta Ori')
ax.set_xticks(oris[::2],oris[::2],rotation=45)
ax.set_ylabel('NoiseCorrelation')
ax.set_ylim([0.01,my_ceil(np.nanmax(data[:,:,:,0]),2)])
ax.set_ylim([0.02,0.06])
ax.set_title('')
ax.legend(handles,areapairs,frameon=False,loc='upper right')
sns.despine(trim=False,top=True,right=True,offset=3)
ax.set_xticks(oris[::2],oris[::2].astype(int),rotation=45)
plt.tight_layout()
my_savefig(fig, savedir, 'NC_deltaOri_areapairs', formats = ['png'])

#%% #########################################################################################
data = np.empty((nSessions,len(oris),2)) #for each session, combination of delta pref store the mean noise corr for all and for with and without gain subtraction
binedges = np.arange(0,360+22.5,22.5)-22.5/2
for ises in range(nSessions):
    sessions_orig[ises].delta_pref = np.abs(np.subtract.outer(sessions_orig[ises].celldata['pref_ori'].values,sessions_orig[ises].celldata['pref_ori'].values))
    sessions_nogain[ises].delta_pref = np.abs(np.subtract.outer(sessions_orig[ises].celldata['pref_ori'].values,sessions_orig[ises].celldata['pref_ori'].values))
    nanfilter         = np.all((~np.isnan(sessions_orig[ises].noise_corr),~np.isnan(sessions_orig[ises].delta_pref)),axis=0)

    # cellfilter = np.all((nanfilter),axis=0)
    cellfilter = nanfilter
    data[ises,:,0] = binned_statistic(x=sessions_orig[ises].delta_pref[cellfilter].flatten(),
                                            values=sessions_orig[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]
    data[ises,:,1] = binned_statistic(x=sessions_nogain[ises].delta_pref[cellfilter].flatten(),
                                            values=sessions_nogain[ises].noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]

#%% Subtracting gain removes tuning-dependent noise correlations:
clrs = sns.color_palette('husl', 2)
clrs = ['black','grey']
fig,ax = plt.subplots(1,1,figsize=(3.5,3))

handles = []
for imodel,model in enumerate(['orig','no gain']):
    handles.append(shaded_error(ax=ax,x=oris,y=data[:,:,imodel].squeeze(),center='mean',error='std',
                                color=clrs[imodel],label=model))
ax.set_xlabel('Delta Ori')
ax.set_xticks(oris[::2],oris[::2],rotation=45)
ax.set_ylabel('NoiseCorrelation')
ax.set_ylim([0,my_ceil(np.nanmax(data),2)])
ax.set_title('')
ax.legend(frameon=False,loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'PairwiseCorrelations','NC_deltaOri_subgainmodel' + '.png'), format = 'png')

#%% 

 #####  ###  #####               #####  ####### #     # ######  #       #######             #     #  #####  
#     #  #  #     #      #      #     # #     # #     # #     # #       #                   ##    # #     # 
#        #  #            #      #       #     # #     # #     # #       #          #####    # #   # #       
 #####   #  #  ####    #####    #       #     # #     # ######  #       #####               #  #  # #       
      #  #  #     #      #      #       #     # #     # #       #       #          #####    #   # # #       
#     #  #  #     #      #      #     # #     # #     # #       #       #                   #    ## #     # 
 #####  ###  #####               #####  #######  #####  #       ####### #######             #     #  #####  

#%% #############################################################################
session_list        = np.array([['LPE10919_2023_11_06']])
session_list        = np.array([['LPE12223_2024_06_10']])
session_list        = np.array([['LPE10884_2023_10_20']])

sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_areas=['V1','PM']) 

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% ########################### Compute tuning metrics: ###################################
sessions        = compute_tuning_wrapper(sessions)

#%% Add how neurons are coupled to the population rate: 
sessions        = compute_pop_coupling(sessions)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions        = compute_pairwise_anatomical_distance(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
sessions       = compute_signal_noise_correlation(sessions,filter_stationary=False,uppertriangular=False)

#%% ########################## Compute joint coupling: ###################################
for ises,ses in enumerate(sessions):
    sessions[ises].joint_coupling = np.outer(ses.celldata['pop_coupling'].values,ses.celldata['pop_coupling'].values)

#%%
ises = 0
plt.imshow(sessions[ises].joint_coupling,vmin=-0.1,vmax=0.2)


#%%
for ises in range(nSessions):
    resp_meanori,respmat_res        = mean_resp_gr(sessions[ises],trialfilter=None)

    sessions[ises].sig_corr         = np.corrcoef(resp_meanori)
    sessions[ises].noise_corr       = np.corrcoef(respmat_res)

    [N,K]                           = np.shape(respmat_res) #get dimensions of response matrix

    # #Compute signal correlation on separate halfs of trials:
    trialfilter                     = np.random.choice([True,False],size=(K),p=[0.5,0.5])
    resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
    resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=~trialfilter)
    sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_meanori1, resp_meanori2)[:N, N:] +
                                    np.corrcoef(resp_meanori2, resp_meanori1)[:N, N:])

    # np.where(sessions[ises].sig_corr==np.max(sessions[ises].sig_corr))
    np.fill_diagonal(sessions[ises].sig_corr,np.nan)
    np.fill_diagonal(sessions[ises].delta_pref,np.nan)
    np.fill_diagonal(sessions[ises].noise_corr,np.nan)

#%% 
ises = 8
tunethr = 0.5

fig,axes = plt.subplots(1,2,figsize=(8,4))

# idx_N = np.outer(sessions[ises].celldata['gOSI']>tunethr,sessions[ises].celldata['gOSI']>tunethr)
idx_N = np.outer(sessions[ises].celldata['OSI']>tunethr,sessions[ises].celldata['OSI']>tunethr)
# idx_N = np.outer(sessions[ises].celldata['tuning_var']>0.1,sessions[ises].celldata['tuning_var']>0.1)
# idx_N = np.outer(sessions[ises].celldata['tuning_var']>0.1,sessions[ises].celldata['tuning_var']>0.1)

subsample = 100
ax = axes[0]
ax.scatter(sessions[ises].joint_coupling[idx_N].flatten()[::subsample],sessions[ises].noise_corr[idx_N].flatten()[::subsample],c='k',s=1)
ax.set_xlabel('Joint coupling')
ax.set_ylabel('Noise correlation')
ax.set_title('Population coupling vs noise correlation')

ax = axes[1]
ax.scatter(sessions[ises].sig_corr[idx_N].flatten()[::subsample],sessions[ises].noise_corr[idx_N].flatten()[::subsample],c='k',s=1)
ax.set_ylabel('Noise correlation')
ax.set_xlabel('Signal correlation')
ax.set_title('Signal corr vs noise correlation')

#%% 
# Get 2D binned map data: 
areapairs   = ['V1-V1','PM-PM','V1-PM']
nAreaPairs  = len(areapairs)
tunethr     = 0.5
x,y         = [np.linspace(-1,1,20),np.linspace(-0.15,0.5,20)] #be careful with x and y dimensions here! 
X,Y         = np.meshgrid(y,x)

data = np.empty((nSessions,nAreaPairs,len(x)-1,len(y)-1)) #for each session, combination of delta pref store the mean noise corr for all and for the top and bottom tuned percentages

for ises in range(nSessions):
    for iareapair,areapair in enumerate(areapairs):
        
        areafilter      = filter_2d_areapair(sessions[ises],areapair)

        # tunefilter      = np.outer(sessions[ises].celldata['gOSI']>tunethr,sessions[ises].celldata['gOSI']>tunethr)
        tunefilter      = np.outer(sessions[ises].celldata['OSI']>tunethr,sessions[ises].celldata['OSI']>tunethr)
        # tunefilter      = np.outer(sessions[ises].celldata['tuning_var']>tunethr,sessions[ises].celldata['tuning_var']>tunethr)

        nanfilter         = np.all((~np.isnan(sessions[ises].sig_corr),
                                    ~np.isnan(sessions[ises].joint_coupling),
                                    ~np.isnan(sessions[ises].noise_corr),
                                              ),axis=0)

        cellfilter       = np.all((areafilter,tunefilter,nanfilter),axis=0)

        xdata = sessions[ises].sig_corr[cellfilter].flatten()
        ydata = sessions[ises].joint_coupling[cellfilter].flatten()
        vdata = sessions[ises].noise_corr[cellfilter].flatten()

        data[ises,iareapair,:,:] = binned_statistic_2d(xdata, ydata, vdata, statistic='mean', bins=[x,y], range=None)[0]

#%% 
cmaplim = 0.5
fig,axes = plt.subplots(1,nAreaPairs,figsize=(nAreaPairs*3,3))
for iareapair,areapair in enumerate(areapairs):
    ax = axes[iareapair]
    pcm = ax.pcolor(X,Y,np.nanmean(data[:,iareapair,:,:],axis=0),vmin=-cmaplim,vmax=cmaplim,cmap='bwr') #np.nanmean(data[:,iareapair,:,:],vmin=-0.7,vmax=0.7,cmap='bwr')
    ax.set_facecolor('gray')
    if iareapair==0: 
        ax.set_ylabel('Signal correlation')
    ax.set_xlabel('Joint coupling')
    ax.set_title('%s' % areapair)
    ax.axhline(0,c='k',lw=0.25,ls='--')
    ax.axvline(0,c='k',lw=0.25,ls='--')
    if iareapair==nAreaPairs-1:
        cb_ax = fig.add_axes([0.98, 0.5, 0.01, 0.3])
        cb = fig.colorbar(pcm, cax=cb_ax, shrink=0.5)
        cb.ax.tick_params(labelsize=7) 
        # cb.set_label('Noise correlation',fontsize=10,loc='center')
        # cb.set_label('Noise correlation',fontsize=6,loc='center')
sns.despine(fig=fig, top=True, right=True,offset=3)
fig.tight_layout()
my_savefig(fig,savedir,'GR_Signal_Coupling_NC_Areapairs_%dsessions' % (nSessions), formats = ['png'])











#%% 
# plt.plot(data[idx_ses_2,:,0])

idx_ses_1 = 3
idx_ses_2 = 6

alloris                            = np.sort(sessions[ises].trialdata['Orientation'].unique())

oris1                           = sessions[idx_ses_1].trialdata['Orientation']
poprate1                        = np.nanmean(zscore(sessions[idx_ses_1].respmat,axis=1),axis=0)
resp_meanori1,respmat_res1      = mean_resp_gr(sessions[idx_ses_1])
prefori1                        = alloris[np.argmax(resp_meanori1,axis=1)]

oris2                           = sessions[idx_ses_2].trialdata['Orientation']
poprate2                        = np.nanmean(zscore(sessions[idx_ses_2].respmat,axis=1),axis=0)
resp_meanori2,respmat_res2      = mean_resp_gr(sessions[idx_ses_2])
prefori2                        = alloris[np.argmax(resp_meanori2,axis=1)]

sort_idx_trials = np.lexsort((poprate1, oris1))[::-1]
respmat_res1 = respmat_res1[:,sort_idx_trials]
oris1 = oris1[sort_idx_trials]

sort_idx_trials = np.lexsort((poprate2, oris2))[::-1]
respmat_res2 = respmat_res2[:,sort_idx_trials]
oris2 = oris2[sort_idx_trials]

fig,axes = plt.subplots(1,2,figsize=(8,4))
axes[0].imshow(respmat_res1,vmin=0,vmax=100,aspect='auto')
axes[1].imshow(respmat_res2,vmin=0,vmax=100,aspect='auto')


#%% Compute noise correlations from residuals:
M = np.shape(respmat_res1)[0]
N = np.shape(respmat_res2)[0]

respmat_res = np.concatenate((respmat_res1,respmat_res2),axis=0)

# noise_corr = np.empty((M+N,M+N,len(oris)))  
# for i,ori in enumerate(oris):
    # noise_corr[:,:,i] = np.corrcoef(respmat_res[:,oris1==ori])
# twoses_noise_corr       = np.mean(noise_corr,axis=2)

twoses_noise_corr       = np.corrcoef(respmat_res)

plt.imshow(twoses_noise_corr,vmin=0,vmax=0.15,aspect='auto')

#%% #########################################################################################
# Plot noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

areapairs = ['V1-V1','PM-PM','V1-PM']
data = np.empty((len(areapairs),len(oris))) #for each session, combination of delta pref store the mean noise corr for all and for the top and bottom tuned percentages

delta_pref  = np.abs(np.subtract.outer(np.concatenate((prefori1,prefori2)),np.concatenate((prefori1,prefori2))))
sesidx      = np.concatenate((np.zeros(M,dtype=bool),np.ones(N,dtype=bool))) #np.zeros(M+N,dtype=bool)
roi_names   = np.concatenate((sessions[idx_ses_1].celldata['roi_name'],sessions[idx_ses_2].celldata['roi_name']))
tunevar     = np.concatenate((sessions[idx_ses_1].celldata['tuning_var'],sessions[idx_ses_2].celldata['tuning_var'])) 

for iap,areapair in enumerate(areapairs):
    area1,area2     = areapair.split('-')
    areafilter      = np.outer(roi_names==area1, roi_names==area2)

    tunefilter      = np.ones(areafilter.shape).astype(bool)

    tunefilter    = np.meshgrid(tunevar>0.01,tunevar>0.01)
    tunefilter    = np.logical_and(tunefilter[0],tunefilter[1])

    nanfilter       = np.all((~np.isnan(twoses_noise_corr),~np.isnan(delta_pref)),axis=0)

    diffsesfilter   = np.outer(sesidx==0,sesidx==1)

    cellfilter      = np.all((areafilter,tunefilter,nanfilter,diffsesfilter),axis=0)
    # cellfilter      = np.all((diffsesfilter,nanfilter),axis=0)
    data[iap,:]     = binned_statistic(x=delta_pref[cellfilter].flatten(),
                                            values=twoses_noise_corr[cellfilter].flatten(),statistic='mean',bins=binedges)[0]
    # data[iap,:]     = binned_statistic(x=delta_pref.flatten(),
                                            # values=twoses_noise_corr.flatten(),statistic='mean',bins=binedges)[0]

plt.plot(oris,data[iap,:])

#%% Noise correlations across V1-PM from different sessions:
clrs = sns.color_palette('husl', 2)
clrs = ['black','grey']
fig,ax = plt.subplots(1,1,figsize=(3,3))
iap = 2
handles = []
ax.plot(oris,data[iap,:].squeeze(),color=clrs_areapairs[iap],lw=2)
ax.set_xlabel('$\Delta$ Pref. Ori')
ax.set_ylabel('Noise Correlation')
ax.set_ylim([0.02,my_ceil(np.nanmax(data),2)])
ax.set_title('Noise correlations across V1 and PM from different \nsessions when sorted along population rate',
             fontsize=11)
# ax.legend(frameon=False,loc='upper right')
ax_nticks(ax,3)
sns.despine(top=True,right=True,offset=5)
ax.set_xticks(oris[::2],oris[::2].astype(int),rotation=45)
plt.tight_layout()
my_savefig(fig, savedir, 'NC_deltaOri_V1-PM_diffses', formats = ['png'])

# plt.savefig(os.path.join(savedir,'PairwiseCorrelations','NC_deltaOri_subgainmodel' + '.png'), format = 'png')




#%% Get the average response for each orientation during trials with low, medium or high activity overall in the population
# Further split neurons in how strongly they are coupled to the population rate
# Align the response to the preferred orientation for the neurons

npopratequantiles       = 5
npopcouplingquantiles   = 5
ntuningquantiles        = 5
oris                    = np.arange(0,360,22.5)
noris                   = len(oris)
data                    = np.full((nSessions,npopratequantiles,npopcouplingquantiles,ntuningquantiles,noris),np.nan)
tuning_metric           = 'tuning_var'
for ises,ses in enumerate(sessions):
    resp                    = stats.zscore(ses.respmat.T,axis=0)
    # resp                    = ses.respmat.T
    N                       = np.shape(resp)[1]
    poprate                 = np.mean(resp, axis=1)
    popcoupling             = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(N)]
    pref_ori                = np.array(ses.celldata['pref_ori']/22.5).astype(int)

    popratequantiles        = np.percentile(poprate,range(0,101,100//npopratequantiles))
    popcouplingquantiles    = np.percentile(popcoupling,range(0,101,100//npopcouplingquantiles))
    tuningquantiles         = np.percentile(ses.celldata[tuning_metric],range(0,101,100//ntuningquantiles))

    for iqrpopcoupling in range(len(popcouplingquantiles)-1):
        for iqrtuning in range(len(tuningquantiles)-1):
            idx_N     = np.where(np.all((popcoupling>popcouplingquantiles[iqrpopcoupling],
                                popcoupling<=popcouplingquantiles[iqrpopcoupling+1],
                                ses.celldata[tuning_metric]>tuningquantiles[iqrtuning],
                                ses.celldata[tuning_metric]<=tuningquantiles[iqrtuning+1]),axis=0))[0]
            # idx_N     = np.where((popcoupling>popcouplingquantiles[iqrpopcoupling]) & (popcoupling<=popcouplingquantiles[iqrpopcoupling+1]))[0]
            for iqrpoprate in range(len(popratequantiles)-1):
                # idx_T       = np.where((poprate>popratequantiles[iqrpoprate-1]) & (poprate<=popratequantiles[iqrpoprate]))[0]
                # idx_T_oris  = sessions[ises].trialdata['Orientation'][idx_T]
                tunedresp = np.empty((noris,len(idx_N)))

                for iori,ori in enumerate(oris):
                    idx_T = np.all((poprate>popratequantiles[iqrpoprate],
                                    poprate<=popratequantiles[iqrpoprate+1],
                                    sessions[ises].trialdata['Orientation']==ori),axis=0
                                                    )
                    tunedresp[iori,:] = np.nanmean(resp[np.ix_(idx_T,idx_N)],axis=0)

                for iN,N in enumerate(idx_N):
                    tunedresp[:,iN] = np.roll(tunedresp[:,iN],-pref_ori[N])

                data[ises,iqrpoprate,iqrpopcoupling,iqrtuning,:] = np.nanmean(tunedresp,axis=1)

#%% ##### Plot orientation tuned response for different quantiles of population rate and coupling
clrs_popcoupling = sns.color_palette('magma',npopratequantiles)
fig, axes = plt.subplots(ntuningquantiles,npopcouplingquantiles,figsize=(npopcouplingquantiles*2.5,ntuningquantiles*2.5),sharey=True,sharex=True)
# ax = axes
for iqrpopcoupling in range(npopcouplingquantiles):
    for iqrtuning in range(len(tuningquantiles)-1):
        ax = axes[iqrtuning,iqrpopcoupling]
        for iqrpoprate in range(npopratequantiles):
            ax.plot(oris,np.nanmean(data[:,iqrpoprate,iqrpopcoupling,iqrtuning,:],axis=0),color=clrs_popcoupling[iqrpoprate],lw=2)
        # ax.set_ylabel('Neuron')
        ax.axhline(0,color='k',ls='--',lw=1)
        ax.set_title('Coupling quantile %d' % (iqrpopcoupling+1),fontsize=10)
# axes[2].set_xlabel('Orientation (deg)')
# axes[0].set_ylabel('Z-scored response')
sns.despine(top=True,right=True,offset=5)
# for ax in axes: 
    # ax.set_xticks(oris[::2],oris[::2].astype(int),rotation=45)
plt.tight_layout()
my_savefig(fig, savedir, 'GainResponse_prefOri_coupling_tuning_quantiles', formats = ['png'])


#%% 
# clrs_popcoupling = sns.color_palette('magma',npopratequantiles)
# ##### Plot orientation tuned response:
# fig, axes = plt.subplots(1,npopcouplingquantiles,figsize=(npopcouplingquantiles*2.5,2.5),sharey=True,sharex=True)
# # ax = axes
# for iqrpopcoupling in range(npopcouplingquantiles):
#     ax = axes[iqrpopcoupling]
#     for iqrpoprate in range(npopratequantiles):
#         ax.plot(oris,np.nanmean(data[:,iqrpoprate,iqrpopcoupling,:],axis=0),color=clrs_popcoupling[iqrpoprate],lw=2)
#     # ax.set_ylabel('Neuron')
#     ax.axhline(0,color='k',ls='--',lw=1)
#     ax.set_title('Coupling quantile %d' % (iqrpopcoupling+1),fontsize=10)
# axes[2].set_xlabel('Orientation (deg)')
# axes[0].set_ylabel('Z-scored response')
# sns.despine(top=True,right=True,offset=5)
# for ax in axes: 
#     ax.set_xticks(oris[::2],oris[::2].astype(int),rotation=45)
# plt.tight_layout()
# my_savefig(fig, savedir, 'GainResponse_prefOri_coupling_quantiles', formats = ['png'])






#%%  ## Plot negative correlation between dissimilarly strongly tuned neurons 
def plot_noise_pair(ses,sourcecell,targetcell):
    oris = np.arange(0,360,22.5)
    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

    # fig,axes = plt.subplots(1,3,figsize=(10,3))
    fig,axes = plt.subplots(2,2,figsize=(6,6))

    for iN,N in enumerate([sourcecell,targetcell]):
        ax = axes[iN,0]

        ax.plot(oris,ses.meanresp_orig[N],c='k',linewidth=2)

        for iori,ori in enumerate(oris):
            idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
            ax.scatter(ses.trialdata['Orientation'][idx_ori],ses.respmat[N,idx_ori],
                            color=pal[iori],s=5,alpha=0.4)
            # ax.scatter(ses.trialdata['Orientation'][idx_ori],ses.respmat[targetcell,idx_ori],
            #                 color='red',s=5,alpha=0.2)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('Ori')
        ax.set_ylabel('Deconvolved activity')
        # ax.set_title('Tuning Curves')
        ax.set_ylim([0,my_ceil(np.nanpercentile(ses.respmat[N,:],99),-1)])
        ax.set_yticks([0,ax.get_ylim()[1]])
        ax.set_xticks(oris[::2],oris[::2].astype(int),rotation=45)
    ax = axes[0,1]
    for iori,ori in enumerate(oris):
        idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
        ax.scatter(ses.respmat[sourcecell,idx_ori],ses.respmat[targetcell,idx_ori],
                        c=pal[iori],s=5,alpha=0.2)
    ax_nticks(ax,3)
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_title('Activity')
    # ax.text(250,250,r'NC = %1.2f' % ses.noise_corr[sourcecell,targetcell])

    ax = axes[1,1]
    for iori,ori in enumerate(oris):
        idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
        ax.scatter(ses.respmat_res[sourcecell,idx_ori],ses.respmat_res[targetcell,idx_ori],
                        c=pal[iori],s=5,alpha=0.2)

    ax_nticks(ax,3)
    ax.set_xlabel('Residual Neuron 1')
    ax.set_ylabel('Residual Neuron 2')
    ax.set_title('Residual activity')
    ax.text(0.1,0.8,'r= %1.2f' % ses.noise_corr[sourcecell,targetcell],transform=ax.transAxes,ha='center',va='center',fontsize=10,color='k')
    # ax.text(250,250,r'NC = %1.2f' % ses.noise_corr[sourcecell,targetcell])

    sns.despine(fig=fig, top=True, right=True, offset=5,trim=True)
    plt.tight_layout()
    return fig


#%% Find a neuron pair that is strongly tuned, has opposite tuning pref and has negative correlation
ises = 0
prctile = 90

idx = sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],prctile)
N = len(sessions[ises].celldata)
signal_filter = np.full((N,N),False)
signal_filter[np.ix_(idx,idx)] = True
idx = np.all((sessions[ises].noise_corr < -0.05,sessions[ises].delta_pref == 90,signal_filter),axis=0)
sourcecells,targetcells = np.where(idx)
random_cell = np.random.choice(len(sourcecells))
sourcecell,targetcell = sourcecells[random_cell],targetcells[random_cell]

[sessions[ises].meanresp_orig,sessions[ises].respmat_res] = mean_resp_gr(sessions[ises])

fig = plot_noise_pair(sessions[ises],sourcecell,targetcell)
my_savefig(fig, os.path.join(savedir,'NoiseCorrelations'), 'NC_example_orthotuning_%s_cell%d_%d' % (sessions[ises].session_id,sourcecell,targetcell), formats = ['png']) 

#%% Find a neuron pair that is strongly tuned, has similar tuning pref and has negative correlation
ises = 0
idx = sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],prctile)
N = len(sessions[ises].celldata)
signal_filter = np.full((N,N),False)
signal_filter[np.ix_(idx,idx)] = True
idx = np.all((sessions[ises].noise_corr > 0.5,sessions[ises].delta_pref == 0,signal_filter),axis=0)
sourcecells,targetcells = np.where(idx)
random_cell = np.random.choice(len(sourcecells))
sourcecell,targetcell = sourcecells[random_cell],targetcells[random_cell]

fig = plot_noise_pair(sessions[ises],sourcecell,targetcell)
my_savefig(fig, os.path.join(savedir,'NoiseCorrelations'), 'NC_example_isotuning_%s_cell%d_%d' % (sessions[ises].session_id,sourcecell,targetcell), formats = ['png']) 

#%%
oris = np.arange(0,360,22.5)


idx_N   = np.all((sessions[ises].noise_corr < -0.05,sessions[ises].delta_pref == 90,signal_filter),axis=0)
idx_N   = np.where((np.sum(sessions[ises].respmat>0,axis=0) / len(sessions[ises].trialdata)) > 0.9)[0]

idx_N = np.random.choice(idx_N,2,replace=False)

fig,axes = plt.subplots(2,2,figsize=(6,6))

for iN,N in enumerate(idx_N):
    for iori,ori in enumerate(oris[:2]):
        ax = axes[iori,iN]
        idx_ori = np.where(sessions[ises].trialdata['Orientation']==ori)[0]
        data = sessions[ises].respmat[N,idx_ori][:,np.newaxis]
        data = data[np.random.choice(np.shape(data)[0],size=50,replace=False)]
        data = np.sort(data,axis=0)

        ax.imshow(data,aspect=0.1,origin='lower',vmin=0,vmax=np.percentile(data,99),cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # ax.set_aspect(0.1)


