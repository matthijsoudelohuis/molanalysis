#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.stats import vonmises
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import linregress
import statsmodels.formula.api as smf
from scipy import stats

os.chdir('e:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance,value_matching
from utils.plot_lib import * #get all the fixed color schemes
from preprocessing.preprocesslib import assign_layer,assign_layer2
from utils.rf_lib import filter_nearlabeled
from utils.RRRlib import regress_out_behavior_modulation

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\Affine_FF_vs_FB\\')

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

# %% Load Data with baseline subtracted
# t_resp_start = 0
# t_resp_stop = 1
# t_base_start = -120
# t_base_stop = -0.2
# calciumversion = 'deconv'
# subtr_baseline = True
# for ises in range(nSessions):
#     sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
#                                 calciumversion=calciumversion)
#     sessions[ises].respmat = compute_respmat(sessions[ises].calciumdata,sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
#                                         t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,
#                                         t_base_start=t_base_start,t_base_stop=t_base_stop,
#                                         method='mean',subtr_baseline=subtr_baseline, label = "response matrix")
#     sessions[ises].respmat_runspeed = compute_respmat(sessions[ises].behaviordata['runspeed'],
#                                         sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
#                                         t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean', label = "runspeed")
#     sessions[ises].respmat_videome = compute_respmat(sessions[ises].videodata['motionenergy'],
#                                         sessions[ises].videodata['ts'],sessions[ises].trialdata['tOnset'],
#                                         t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean', label = "motion energy")
        
#     delattr(sessions[ises], 'calciumdata')
#     delattr(sessions[ises], 'videodata')
#     delattr(sessions[ises], 'behaviordata')

#%% Compute Tuning Metrics (gOSI, gDSI etc.)
sessions = compute_tuning_wrapper(sessions)

#%%
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=50)

#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 

    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 

# #%% 
# arealayers = np.array(['V1L2/3','PML2/3','PML5'])
# narealayers = len(arealayers)
# maxnoiselevel = 20
# celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
# clrs = ['black','red']
# fig,axes = plt.subplots(1,narealayers,figsize=(narealayers*3,3),sharey=True,sharex=True)

# for ial,al in enumerate(arealayers):
#     ax = axes[ial]
#     idx_N              = np.all((
#                                 celldata['noise_level']<maxnoiselevel,
#                                 celldata['arealayer']==al,
#                                 celldata['nearby'],
#                                     ),axis=0)

#     sns.histplot(data=celldata[idx_N],x='pop_coupling',hue='arealayerlabel',color='green',element="step",stat="density", 
#                 common_norm=False,fill=False,bins=np.linspace(-0.3,1,1000),cumulative=True,ax=ax,palette=clrs,legend=False)
#     ax.set_title(al)
# sns.despine(fig=fig, top=True, right=True,offset=0)


#%% For every session remove behavior related variability:
rank_behavout = 3
# maxnoiselevel = 20

#%%
for ises in range(nSessions):
    # Convert response_matrix and orientations_vector to numpy arrays
    # response_matrix         = np.array(response_matrix)
    conditions_vector       = np.array(sessions[ises].trialdata['stimCond'])
    conditions              = np.sort(np.unique(conditions_vector))
    C                       = len(conditions)

    resp_mean       = sessions[ises].respmat.copy()
    # resp_res        = sessions[ises].respmat.copy()

    for iC,cond in enumerate(conditions):
        tempmean                            = np.nanmean(sessions[ises].respmat[:,conditions_vector==cond],axis=1)
        # resp_mean[:,iC]                     = tempmean
        resp_mean[:,conditions_vector==cond] = tempmean[:,np.newaxis]

    Y               = sessions[ises].respmat.copy()
    Y               = Y - resp_mean
    [Y_orig,Y_hat,Y_out,rank,ev] = regress_out_behavior_modulation(sessions[ises],X=None,Y=Y.T,
                                nvideoPCs = 30,rank=rank_behavout,lam=0,perCond=True,kfold = 5)
    # print(ev)
    sessions[ises].respmat_behavout = Y_out.T + resp_mean

# plt.imshow(sessions[ises].respmat,cmap='RdBu_r',vmin=0,vmax=100)
# plt.imshow(sessions[ises].respmat_behavout,cmap='RdBu_r',vmin=0,vmax=100)

#%% 
ises = 0
fig, axes = plt.subplots(1,1,figsize=(6,2.5))
ax = axes
idx_N_FF              = np.where(np.all((sessions[ises].celldata['arealabel'] == 'V1lab',),axis=0))[0]
idx_N_FB              = np.where(np.all((sessions[ises].celldata['arealabel'] == 'PMlab',),axis=0))[0]
idx_T_still           = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                            sessions[ises].respmat_runspeed < maxrunspeed)
respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

meanpopact_FF          = np.nanmean(respdata[idx_N_FF,:],axis=0)
meanpopact_FB          = np.nanmean(respdata[idx_N_FB,:],axis=0)
ax.plot(meanpopact_FF[idx_T_still],color=clrs_arealabelpairs[0],linewidth=0.4)
ax.plot(meanpopact_FB[idx_T_still],color=clrs_arealabelpairs[1],linewidth=0.4)
ax.set_xlim([0,500])
ax.set_ylim(np.percentile([meanpopact_FF[idx_T_still][:500],meanpopact_FB[idx_T_still][:500]],[0,100]))
ax.set_xlabel('Trials',fontsize=10)
ax.set_ylabel('Population activity (deconv/dF0)',fontsize=10)
ax.set_title('Population activity',fontsize=13)
ax.text(0.1,0.8,'r = %1.2f, p = %1.2f' % (pearsonr(meanpopact_FF[idx_T_still],meanpopact_FB[idx_T_still])[0],pearsonr(meanpopact_FF[idx_T_still],meanpopact_FB[idx_T_still])[1]),transform=ax.transAxes,fontsize=8)
ax_nticks(ax,4)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
# my_savefig(fig,savedir,'PopulationActivity_Trials_FF_FB', formats = ['png'])

#%% Correlation matrix between the different populations
arealabelpairs  = ['V1unl','V1lab','PMunl','PMlab']
narealabelpairs = len(arealabelpairs)
maxnoiselevel=20
corrmat = np.full((narealabelpairs,narealabelpairs,nSessions),np.nan)
for ises in range(nSessions):
    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    # idx_T_still = np.ones(np.sum(idx_T_still),dtype=bool)
    datamat = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    nsampleneurons = 1000
    for ial,alp in enumerate(arealabelpairs):
        nsampleneurons = np.min([nsampleneurons,
                                 len(np.where(np.all((sessions[ises].celldata['arealabel'] == alp,
                                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                      sessions[ises].celldata['nearby'],
                                                      ),axis=0))[0])])
    
    for ial,alp in enumerate(arealabelpairs):
        idx_N               = np.where(np.all((sessions[ises].celldata['arealabel'] == alp,),axis=0))[0]
        idx_N               = np.where(np.all((sessions[ises].celldata['arealabel'] == alp,
                                            sessions[ises].celldata['noise_level']<maxnoiselevel),axis=0))[0]
        idx_N               = np.random.choice(idx_N, size=nsampleneurons, replace=False)
        datamat[ial,:]      = np.nanmean(sessions[ises].respmat[np.ix_(idx_N,idx_T_still)],axis=0)
    corrmat[:,:,ises] = np.corrcoef(datamat)

fig, axes = plt.subplots(1,1,figsize=(6,2.5))
ax = axes
ax.imshow(np.nanmean(corrmat,axis=2),cmap='RdBu_r',vmin=-.3,vmax=.3)
ax.set_xticks(range(narealabelpairs))
ax.set_xticklabels(arealabelpairs)
ax.set_yticks(range(narealabelpairs))
ax.set_yticklabels(arealabelpairs)
ax.set_title('Correlation matrix',fontsize=13)
cbar = fig.colorbar(ax.imshow(np.nanmean(corrmat,axis=2),cmap='RdBu_r',vmin=-.3,vmax=.3), ax=ax)
cbar.set_label('Correlation', rotation=270, labelpad=10)


#%% 
ises = 6
maxrunspeed = 0.5
maxvideome = 0.2

idx_T_still = np.where(np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed))[0]

arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']
# axislabels  = ['V1unl','V1lab','PMunl','PMlab']
# axislabels = arealabeled_to_figlabels(alp.split('-')[0])

vscale      = 0.015
# vscale      = 2
# fig,axes    = plt.subplots(1,2,figsize=(6,3))
# maxnoiselevel = 20
# fig,axes = plt.subplots(2,2,figsize=(6,6))
fig,axes = plt.subplots(2,1,figsize=(3.5,6))
for ialp,alp in enumerate(arealabelpairs):
    respdata        = sessions[ises].respmat
    # respdata        = zscore(sessions[ises].respmat, axis=1)
    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]
    idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]

    subsampleneurons    = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
    idx_source_N1       = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
    idx_source_N2       = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

    meanpopact_N1       = np.nanmean(respdata[np.ix_(idx_source_N1,idx_T_still)],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)
    meanpopact_N2       = np.nanmean(respdata[np.ix_(idx_source_N2,idx_T_still)],axis=0)

    vmin,vmax = np.percentile([meanpopact_N1 - meanpopact_N2],[5,95])
    # vmax = np.percentile([meanpopact_N2,meanpopact_N1],[99.5,99.5])
    ax=axes[ialp]
    ax.scatter(meanpopact_N2,meanpopact_N1,c=meanpopact_N1 - meanpopact_N2,s=0.2,cmap='coolwarm',vmin=vmin,vmax=vmax)
    # ax.scatter(meanpopact_N2,meanpopact_N1,c=meanpopact_N1 / meanpopact_N2,s=0.2,cmap='coolwarm',vmin=-vscale,vmax=vscale)

    ax.text(0.1,0.8,'%s > %s' % (arealabeled_to_figlabels([alp.split('-')[0]]),arealabeled_to_figlabels([alp.split('-')[1]])),transform=ax.transAxes)
    ax.text(0.7,0.2,'%s < %s' % (arealabeled_to_figlabels([alp.split('-')[0]]),arealabeled_to_figlabels([alp.split('-')[1]])),transform=ax.transAxes)
    # ax.text(0.7,0.2,'%s < %s' % (arealabeled_to_figlabels([alp.split('-')[0]]),transform=ax.transAxes)
    ax.plot([-1,1],[-1,1],color='k',linewidth=0.2)
    ax.set_xlim(np.percentile([meanpopact_N2,meanpopact_N1],[0.1,99.5]))
    ax.set_ylim(np.percentile([meanpopact_N2,meanpopact_N1],[0.1,99.5]))
    ax.set_xlabel('Mean %s activity (events/F0) ' % arealabeled_to_figlabels([alp.split('-')[1]]))
    ax.set_ylabel('Mean %s activity (events/F0) ' % arealabeled_to_figlabels([alp.split('-')[0]]))
    ax_nticks(ax, 3)
    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True,offset=1)
my_savefig(fig,savedir,'Diff_LabUnl_%s' % sessiondata['session_id'][ises])

#%% Show the distribution of mean activity of the labeled population and the selection of the
# extreme percentiles:
arealabelpairs  = ['V1lab','PMlab']
arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']

# clrs_arealabelpairs = get_clr_arealabelpairs(arealabelpairs)
legendlabels        = ['FF','FB']
clrs_arealabels_low_high = get_clr_area_low_high()

respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

perc = 25
step  = 0.001
fig, axes = plt.subplots(1,2,figsize=(6,3))
for ial,alp in enumerate(arealabelpairs):
    ax = axes[ial]
    idx_N1              = np.where(np.all((
                                        sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                        ),axis=0))[0]
    meanpopact_N1          = np.nanmean(respdata[idx_N1,:],axis=0)
    idx_N2              = np.where(np.all((
                                        sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                        ),axis=0))[0]
    meanpopact_N2          = np.nanmean(respdata[idx_N2,:],axis=0)

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    # meanpopact = meanpopact_N1
    meanpopact = meanpopact_N1 - meanpopact_N2

    allbins = np.arange(np.percentile(meanpopact[idx_T_still],0),np.percentile(meanpopact[idx_T_still],100),step)
    bins = allbins[allbins <= np.percentile(meanpopact[idx_T_still],perc)]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins[bins<np.percentile(meanpopact[idx_T_still],perc)],color=clrs_arealabels_low_high[ial,0],edgecolor='none')
    bins = allbins[np.logical_and(allbins >= np.percentile(meanpopact[idx_T_still],perc), allbins <= np.percentile(meanpopact[idx_T_still],100-perc))]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color='grey',edgecolor='none')
    bins = allbins[allbins > np.percentile(meanpopact[idx_T_still],100-perc)]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color=clrs_arealabels_low_high[ial,1],edgecolor='none')
    # ax.set_ylabel('FB (high)',fontsize=10)
    ax.set_xlabel('Population activity (deconv/dF0)',fontsize=10)
    ax.set_title(legendlabels[ial],fontsize=13)
    ax_nticks(ax,3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
# my_savefig(fig,savedir,'Hist_PopAct_FF_FB')
my_savefig(fig,savedir,'Hist_PopAct_FF_FB%s' % sessiondata['session_id'][ises])


#%% 
# ises = 0
# ialp = 0
# arealabelpairs  = [
#                     'V1lab-V1unl-PMunlL2/3',
#                     'PMlab-PMunl-V1unlL2/3',
#                     ]
# alp = arealabelpairs[ialp]
# idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
#                                             sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                             sessions[ises].celldata['nearby']
#                                             ),axis=0))[0]
# idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
#                                             sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                             sessions[ises].celldata['nearby']
#                                             ),axis=0))[0]

# meanpopact_N1          = np.nanmean(respdata[idx_source_N1,:],axis=0)
# meanpopact_N2          = np.nanmean(respdata[idx_source_N2,:],axis=0)
# vscale      = 0.015

# fig,axes    = plt.subplots(1,2,figsize=(6,3))


# # fig,axes = plt.subplots(2,2,figsize=(6,6))
# # fig,axes = plt.subplots(2,1,figsize=(3,6))
# ax=axes[0]
# ax.plot(meanpopact_N1,color='red',label=legendlabels[0],linewidth=0.1,alpha=1)
# ax.plot(meanpopact_N2,color='grey',label=legendlabels[1],linewidth=0.1,alpha=1)
# ax.set_xlabel('Trials')
# ax.set_ylabel('Population activity')
# ax.legend(['PMLab','PMUnl'],fontsize=9,frameon=False)
# ax = axes[1]
# # ax.scatter(meanpopact_N2,meanpopact_N1,color='k',s=0.1)
# ax.scatter(meanpopact_N2,meanpopact_N1,c=meanpopact_N1 - meanpopact_N2,s=0.2,cmap='coolwarm',vmin=-vscale,vmax=vscale)
# ax.text(0.1,0.8,'High FB',transform=ax.transAxes)
# ax.text(0.7,0.2,'Low FB',transform=ax.transAxes)
# ax.plot([-1,1],[-1,1],color='k',linewidth=0.2)
# ax.set_xlim(np.percentile(meanpopact_N1,[0,99.5]))
# ax.set_ylim(np.percentile(meanpopact_N1,[0,99.5]))
# ax.set_xlabel('PMUnl')
# ax.set_ylabel('PMLab')
# ax_nticks(ax, 3)
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=1)
# # my_savefig(fig,savedir,'Diff_LabUnl_GR%dsessions' % (nSessions), formats = ['png'])






#%% Show tuning curve when activity in the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
perc                    = 25

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5
alphathr                = 0.001 #threshold for correlation with cross area rate

minnneurons             = 10
maxnoiselevel           = 20
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells),np.nan)

#Regression output:
nboots                  = 100
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

#Correlation output:
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)


for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    # sessions[ises].celldata['nearby']
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    # sessions[ises].celldata['nearby']
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        # subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
        # idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
        # idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue
        
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        #Ratio:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        #Difference:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        # meanpopact          = np.nanmean(respdata[idx_N2,:],axis=0) - np.nanmean(respdata[idx_N1,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        errorresp           = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            errorresp[:,i,0]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K1)))
            errorresp[:,i,1]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K2)))

        meanresp = meanresp - np.nanmin(meanresp[:,:,:],axis=(1,2),keepdims=True)

        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp[idx_N3]
        error_resp_split[ialp,:,:,idx_ses] = errorresp[idx_N3]

        regressdata          = np.full((N,3),np.nan)
        regress_sig          = np.full((N,2),0)
        for n in range(N):
            xdata = meanresp[n,:,0]
            ydata = meanresp[n,:,1]
            regressdata[n,:] = linregress(xdata,ydata)[:3]
        params_regress[idx_ses,ialp,:] = regressdata[idx_N3]

        if nboots:
            bootregressdata  = np.full((N,nboots,3),np.nan)
            bootregress_sig  = np.full((N,2),0)
            for iboot in range(nboots):
                meanrespboot            = np.empty([N,len(oris),2])
                for i,ori in enumerate(oris):
                    idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
                    idx_K1              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
                    idx_K2              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
                    meanrespboot[:,i,0]     = np.nanmean(respdata[:,idx_K1],axis=1)
                    meanrespboot[:,i,1]     = np.nanmean(respdata[:,idx_K2],axis=1)
                for n in range(N):
                    bootregressdata[n,iboot,:] = linregress(meanrespboot[n,:,0],meanrespboot[n,:,1])[:3]

            bootregress_sig[regressdata[:,0]>np.percentile(bootregressdata[:,:,0],97.5,axis=1),0] = 1
            bootregress_sig[regressdata[:,0]<np.percentile(bootregressdata[:,:,0],2.5,axis=1),0] = -1
            bootregress_sig[regressdata[:,1]>np.percentile(bootregressdata[:,:,1],97.5,axis=1),1] = 1
            bootregress_sig[regressdata[:,1]<np.percentile(bootregressdata[:,:,1],2.5,axis=1),1] = -1

            sig_params_regress[idx_ses,ialp,:] = bootregress_sig[idx_N3]

        #Aligned:
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        mean_resp_split_aligned[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<alphathr) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# # Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
# for iN in tqdm(range(nCells),total=nCells,desc='Fitting gain for each neuron'):
#     for ialp,alp in enumerate(arealabelpairs):
#         xdata = mean_resp_split[ialp,:,0,iN]
#         ydata = mean_resp_split[ialp,:,1,iN]
#         params_regress[iN,ialp,:] = linregress(xdata,ydata)[:3]

#%% Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

#%% Show some example neurons:

#%% use 
ialp = 0
legendlabels        = ['FF','FB']

#%% Get good multiplicatively modulated cells:
#mutliplicative: 
idx_examples = np.all((params_regress[:,ialp,0]>np.nanpercentile(params_regress[:,ialp,0],90),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good divisive modulated cells:
idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                       ),axis=0)

example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good additively modulated cells: 
#additive:
idx_examples = np.all((
                        # params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],80),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],70),
                        # rangeresp>0.04,
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good subtractive modulated cells: 
idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],70),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],25),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% 

#%% Plot in two ways:
clrs_arealabelpairs = get_clr_arealabelpairs(arealabelpairs)
clrs_arealabels_low_high =  np.array([["#C489FF", "#4C0099"],  # V1lab-V1unl-PMunl
                   ["#20FFC7", "#006149"]])  # PMlab-PMunl-V1unl

example_cells = [
                    'LPE09665_2023_03_21_7_0011', #FF divisive
                    'LPE12223_2024_06_10_3_0023', #FF additive
                    'LPE11086_2024_01_05_6_0103', #FF additive
                    'LPE09830_2023_04_10_5_0065', #FF additive
                    'LPE11086_2024_01_05_4_0002', #FF additive
                    'LPE11086_2024_01_05_5_0169', #FF multiplicative
                    'LPE10919_2023_11_06_0_0322', #FF subtractive/divisive
                    'LPE11086_2024_01_10_1_0080', #FB additive
                    'LPE11086_2024_01_05_1_0035', #FB additive
                    'LPE12223_2024_06_10_1_0051', #FB multiplicative
                    'LPE11086_2024_01_10_3_0108', #FB multiplicative
                    'LPE10885_2023_10_23_1_0276', #FB divisive
                    ]

#%% Plot in two ways:
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]


for example_cell in example_cells:
    idx_N = np.where(celldata['cell_id']==example_cell)[0][0]
    ialp = np.where(~np.isnan(mean_resp_split[:,0,0,idx_N]))[0][0]
    ustim = np.unique(sessions[ises].trialdata['Orientation'])
    x = mean_resp_split[ialp,:,0,idx_N]
    y = mean_resp_split[ialp,:,1,idx_N]
    xerror = error_resp_split[ialp,:,0,idx_N]
    yerror = error_resp_split[ialp,:,1,idx_N]

    # clrs_stimuli    = sns.color_palette('viridis',8)
    fig,axes = plt.subplots(1,2,figsize=(5,2.5))
    ax = axes[0]
    ax.scatter(ustim,x,color=clrs_arealabels_low_high[ialp,0],s=20)
    ax.plot(ustim,x,color=clrs_arealabels_low_high[ialp,0],linestyle='-',linewidth=0.8)
    # ax.errorbar(ustim,x,yerr=xerror,color='k',ls='None',linewidth=1)
    ax.errorbar(ustim,x,yerr=xerror,color=clrs_arealabels_low_high[ialp,0],ls='None',linewidth=1)

    ax.scatter(ustim,y,color=clrs_arealabels_low_high[ialp,1],s=20)
    ax.plot(ustim,y,color=clrs_arealabels_low_high[ialp,1],linestyle='-',linewidth=0.8)
    # ax.errorbar(ustim,y,yerr=yerror,color='k',ls='None',linewidth=1)
    ax.errorbar(ustim,y,yerr=yerror,color=clrs_arealabels_low_high[ialp,1],ls='None',linewidth=1)
    ax.set_xlabel('Orientation',fontsize=10)
    ax.set_ylabel('Response',fontsize=10)
    ax.set_xticks([0,90,180,270])

    ax = axes[1]
    ax.scatter(x,y,color='k',s=10)
    ax.errorbar(x,y,xerr=xerror,yerr=yerror,color='k',ls='None',linewidth=1)
    b = linregress(x, y)
    xp = np.linspace(np.percentile(x,0),np.percentile(x,100)*1.1,100)
    ax.plot(xp,b[0]*xp+b[1],color=clrs_arealabelpairs[ialp],linestyle='-',linewidth=2)

    ax.plot([0,1],[0,1],color='grey',ls='--',linewidth=1)
    ax.set_xlim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
    ax.set_ylim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
    ax.set_ylabel('%s high' % legendlabels[ialp],fontsize=10)
    ax.set_xlabel('%s low' % legendlabels[ialp],fontsize=10)
    ax_nticks(ax,2)
    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
    # my_savefig(fig,savedir,'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])
    my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell)
    # my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly','BaselineCorrected'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])

#%%


#%% 
params_regress_mean = np.full((narealabelpairs,3),np.nan)
# clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    # idx_N =  np.array(celldata['gOSI']>0.5)
    # idx_N = params_regress[:,ialp,2] > 0.5
    # idx_N = corrsig_cells[ialp,:]==-1
    idx_N =  np.all((
                    # celldata['gOSI']>0.4,
                    rangeresp>0.04,
                    # sig_params_regress[:,ialp,0]==1,
                    # celldata['nearby'],
                    # corrsig_cells[ialp,:]==1,
                    # corrsig_cells[ialp,:]==1,
                    # np.any(mean_resp_split>0.5,axis=(0,1,2)),
                    # np.any(params_regress[:,:,2] > 0.5,axis=1),
                     ),axis=0)

    xdata = np.nanmean(mean_resp_split_aligned[ialp,:,0,idx_N].T,axis=1)
    ydata = np.nanmean(mean_resp_split_aligned[ialp,:,1,idx_N].T,axis=1)
    b = linregress(xdata,ydata)
    params_regress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = params_regress_mean[ialp,0]*xvals + params_regress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=1.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.7,s=35)
    ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
    ax.set_title(legendlabels[ialp],fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.text(0.1,0.8,'Slope: %1.2f\nOffest: %1.2f'%(params_regress_mean[ialp,0],params_regress_mean[ialp,1]),
            transform=ax.transAxes,fontsize=8)
    # ax.legend(frameon=False,loc='lower right')
    ax.set_xlabel('%s low (events/F0) '%(alp.split('-')[0]))
    ax.set_ylabel('%s high'%(alp.split('-')[0]))
    ax.set_xlim([0,np.nanmax([xdata,ydata])*1.1])
    ax.set_ylim([0,np.nanmax([xdata,ydata])*1.1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_%dsessions' % (nSessions), formats = ['png'])

#%% Fraction of significant multiplicative and additively modulated cells:
fig,axes = plt.subplots(1,2,figsize=(6,3))
for itype,(itype,itypelabel) in enumerate(zip([0,1],['mult','add'])):
    ax = axes[itype]
    for isign,(sign,signlabel) in enumerate(zip([1,-1],['pos','neg'])):
        for ialp,alp in enumerate(arealabelpairs):
            idx_N =  np.all((
                rangeresp>0.04,
                     ),axis=0)
            # frac = np.sum(sig_params_regress[:,ialp,itype]==sign) / np.sum(~np.isnan(sig_params_regress[:,ialp,itype]))
            Nsig = np.sum(sig_params_regress[idx_N,ialp,itype]==sign)
            Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,itype]))
            frac = Nsig/Ntotal
            xpos = isign*2 + ialp
            ax.bar(xpos,frac,width=0.8,color=clrs_arealabelpairs[ialp])
            ax.text(xpos,.01,'%d/\n%d' % (Nsig,Ntotal),
                ha='center',va='bottom',fontsize=7)
            ax.set_xticks(range(4))
            ax.set_xticks([.5,2.5],['+','-'],fontsize=15)
            ax.set_title(itypelabel)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_sig_mod_labunldiff_%dsessions' % (nSessions), formats = ['png'])

#%%
fracmat = np.full((3,3,narealabelpairs+1),np.nan)

for ialp,alp in enumerate(arealabelpairs):
    # for imult, mult in enumerate([-1,0,1]):
    for imult, mult in enumerate([1,0,-1]):
        for iadd, add in enumerate([-1,0,1]):
            idx_N =  np.all((
                rangeresp>0.05,
                     ),axis=0)
            Nsig = np.sum(np.all((
                                sig_params_regress[idx_N,ialp,0]==mult,
                                sig_params_regress[idx_N,ialp,1]==add,
                                ),axis=0))
            Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
            frac = Nsig/Ntotal
            fracmat[imult,iadd,ialp] = frac
fracmat[:,:,2] = fracmat[:,:,1] - fracmat[:,:,0]

fig,axes = plt.subplots(1,3,figsize=(9,3))
for ialp in range(narealabelpairs+1):
    ax = axes[ialp]
    if ialp < narealabelpairs:
        vmin,vmax = 0,0.25
        cmap = 'viridis'
    else:
        vmin,vmax = -0.05,0.05
        cmap = 'bwr'
    im = ax.imshow(fracmat[:,:,ialp],vmin=vmin,vmax=vmax,cmap=cmap)

    ax.set_xticks([0,1,2],['Sub','None','Add'])
    # ax.set_yticks([0,1,2],['Div','None','Mult'])
    ax.set_yticks([0,1,2],['Mult','None','Div'])
    ax.set_xlabel('Addition')
    ax.set_ylabel('Multiplicative')
    ax.set_title(legendlabels[ialp] if ialp < narealabelpairs else 'Diff (FB-FF)')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,'%1.2f' % fracmat[i,j,ialp],ha='center',va='center',color='white' if fracmat[i,j,ialp]>0.2 else 'black')
    fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Affine_sig_mod_FF_FB_heatmap_%dsessions' % (nSessions))



#%% 
legendlabels = ['FF','FB']
fig,axes = plt.subplots(1,2,figsize=(6,3))
for iparam in range(2):
    ax = axes[iparam]
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        bins = np.arange(0,2,0.015)
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        bins = np.arange(-0.025,0.04,0.0001)
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    handles = []
    for ialp,alp in enumerate(arealabelpairs):
        
        idx_N =  rangeresp>0.04
        # idx_N =  sig_params_regress[:,ialp,1]==1
        # idx_N =  params_regress[:,ialp,2]>0.3

        # idx_N =  celldata['gOSI']>0
        print(np.sum(~np.isnan(params_regress[idx_N,ialp,iparam])))
        sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],
                     alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
                    #  ,alpha=0.3linewidth=1,ax=ax,stat='probability',bins=bins,fill=True)
        # ax.axvline(0,color='grey',ls='--',linewidth=1)
        # ax.axvline(1,color='grey',ls='--',linewidth=1)
        handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.9,markersize=10,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
        # ax.axvline(np.nanmean(params_regress[idx_N,ialp,iparam]),color=clrs_arealabelpairs[ialp],ls='--',linewidth=1)
        # ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    h,p = stats.ttest_ind(params_regress[idx_N,0,iparam],
                            params_regress[idx_N,1,iparam],nan_policy='omit')
    p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
    # ax.text(0.6, 0.15, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes,fontsize=9)
    ax.text(0.6, 0.15, '%s,p<0.001' % (get_sig_asterisks(p,return_ns=True)), transform=ax.transAxes,fontsize=9)

    ax.legend(handles,legendlabels,fontsize=9,frameon=False,loc='center right')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_DiffLabUnlab_gOSI04_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_MinRespSub_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_AL_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI0_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])

my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_RangeResp_cumhistcoefs_%dGRsessions' % (nSessions))
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])

# %%
celldata['meanact']     = np.nanmean(mean_resp_split,axis=(0,1,2))
celldata['slope']       = np.nanmean(params_regress[:,:,0],axis=1)
celldata['offset']      = np.nanmean(params_regress[:,:,1],axis=1)
celldata['affine_R2']   = np.nanmean(params_regress[:,:,2],axis=1)
celldata['rangeresp']   = rangeresp

#%%
fig,axes = plt.subplots(2,3,figsize=(9,6))
for ivarx,varx in enumerate(['slope','offset','affine_R2']):
    for ivary,vary in enumerate(['gOSI','rangeresp']):
        ax = axes[ivary,ivarx]
        ax.scatter(celldata[vary],celldata[varx],s=3,alpha=0.2,color='grey')
        ax.set_xlabel(vary)
        ax.set_ylabel(varx)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

#%% Is the effect similar for the two areas, but 
# just dependent on a difference in activity levels?
# Is modulation more multiplicative for larger activity levels, stimuli with 
from scipy.stats import binned_statistic
bins = np.arange(0,1,0.025)
bins = np.nanpercentile(mean_resp_split,np.arange(0,100,1))
bins = bins[bins>0]
fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=True,sharey=False)
resp_mod = mean_resp_split[:,:,1,:] - mean_resp_split[:,:,0,:]
# resp_mod = mean_resp_split[:,:,1,:]  / mean_resp_split[:,:,0,:]

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[0]
    # xdata = mean_resp_split[ialp,:,0,:].flatten()
    # xdata = mean_resp_split[ialp,:,1,:].flatten()
    # ydata = resp_mod[ialp].flatten()

    # idx_N =  celldata['gOSI']>0.4
    idx_N =  rangeresp>0.04
    # xdata = mean_resp_split[np.ix_([ialp],range(16),[1],idx_N)].flatten()
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    ydata = resp_mod[np.ix_([ialp],range(16),idx_N)].flatten()
    ax.scatter(xdata,ydata,s=2,alpha=0.3,color=clrs_arealabelpairs[ialp])
    # ax.set_xlim([0,.8])
    # ax.set_ylim([0,.1])
    ax.set_ylabel('Modulation')
    ax.set_xlabel('Activity')

    ax = axes[1]
    ymeandata,bin_edges = binned_statistic(xdata, ydata, statistic='mean', 
                            bins=bins)[:2]
    bincenters = (bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],
            linewidth=2)
    ax.set_xlim([0,bincenters[-1]])
    ax.set_ylim([0,.03])
    ax.set_xlabel('Activity')
    ax.set_ylabel('Modulation')
    ax.legend(legendlabels,fontsize=11,frameon=False,loc='best')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_Modulation_vs_Activity_%dGRsessions' % (nSessions), formats = ['png'])



#%% Is the effect similar for the two areas, but 
# just dependent on a difference in activity levels?
# Is modulation more multiplicative for larger activity levels, stimuli with 
mincounts = 20
from scipy.stats import binned_statistic
vartocontrol = 'gOSI'
# vartocontrol = 'rangeresp'
# vartocontrol = 'meanact'
bins = np.linspace(celldata[vartocontrol].min(),np.nanpercentile(celldata[vartocontrol],99),20)
fig,axes = plt.subplots(2,2,figsize=(5,5),sharex='row',sharey='row')
for ivar,var in enumerate(['slope','offset']):
    for ialp,alp in enumerate(arealabelpairs):
        ax = axes[ivar,0]

        idx_N = celldata['arealayerlabel'] == alp.split('-')[2]
        xdata = celldata[vartocontrol][idx_N]
        ydata = celldata[var][idx_N]

        idx_notnan = np.logical_and(~np.isnan(xdata),~np.isnan(ydata))
        xdata = xdata[idx_notnan]
        ydata = ydata[idx_notnan]

        ax.set_xlim(np.percentile(bins,[0,100]))
        ax.scatter(xdata,ydata,s=3,alpha=0.2,color=clrs_arealabelpairs[ialp])
        ax.set_xlabel(vartocontrol)
        ax.set_ylabel(var)

        ax = axes[ivar,1]
        ymeandata,bin_edges = binned_statistic(xdata, ydata, statistic='mean', 
                                bins=bins)[:2]
        bincounts = np.histogram(xdata,bins=bin_edges)[0]
        ymeandata[bincounts<mincounts] = np.nan
        ax.plot(bin_edges[:-1],ymeandata,color=clrs_arealabelpairs[ialp],
                linewidth=2)
        ax.set_ylim(np.nanpercentile(ydata,[1,99]))
        ax.set_xlim(np.nanpercentile(bins,[0,100]))
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Control_AffineCoefs_%s_%dGRsessions' % (vartocontrol,nSessions), formats = ['png'])

#%% Is there a difference in the distribution of activity across the stimuli across the areas:
fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=False,sharey=True)
resp_mod = mean_resp_split[:,:,1,:] - mean_resp_split[:,:,0,:]

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[0]
    # xdata = mean_resp_split[ialp,:,0,:].flatten()
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()

    ydata = resp_mod[ialp].flatten()

    # ax.scatter(xdata,ydata,s=2,alpha=0.3,color=clrs_arealabelpairs[ialp])
    # ymeandata,bin_edges = binned_statistic(xdata, ydata, statistic='mean', 
                            # bins=np.arange(0,0.8,0.025))[:2]
    # ax.plot(bin_edges[:-1],ymeandata,color=clrs_arealabelpairs[ialp],
    #         linewidth=2)
    ax.set_ylim([0,1])
    ax.set_xlabel('Activity')

    sns.histplot(xdata,bins=np.arange(0,0.1,0.0005),fill=False,stat='probability',
                 cumulative=True,
                 color=clrs_arealabelpairs[ialp],element='step',ax=ax)
    ax = axes[1]
    sns.histplot(ydata,bins=np.arange(-0.1,0.1,0.0005),fill=False,stat='probability',
                 cumulative=True,
                 color=clrs_arealabelpairs[ialp],element='step',ax=ax)
    ax.set_ylim([0,1])
    ax.set_xlabel('Modulation')

plt.tight_layout()
# my_savefig(fig,savedir,'Control_Activity_Diff_vs_Mod_%dGRsessions' % (nSessions), formats = ['png'])

#%% Show for positive and negatively correlated neurons only:
params_regress_mean = np.full((narealabelpairs,3),np.nan)
clrs_arealabelpairs = ['green','purple']
legendlabels = ['FF','FB']
# clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(2,3,figsize=(7,4))
for isign,sign in enumerate([-1,1]):
    print(sign)
    for iparam in range(2):
        ax = axes[isign,iparam]
        if iparam == 0:
            ax.set_xlabel('Multiplicative Slope')
            bins = np.arange(-0.5,3,0.05)
            ax.axvline(1,color='grey',ls='--',linewidth=1)
        else:
            ax.set_xlabel('Additive Offset')
            bins = np.arange(-0.025,0.05,0.0015)
            ax.axvline(0,color='grey',ls='--',linewidth=1)
        handles = []
        for ialp,alp in enumerate(arealabelpairs):
            ax = axes[isign,iparam]
            # idx_N = params_regress[:,ialp,2] > 0.5
            idx_N =  np.all((
                             celldata['gOSI']>0.4,
                            #  rangeresp>0.04,
                             corrsig_cells[ialp,:]==sign),axis=0)
            # print(np.sum(idx_N))
            sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                        color=clrs_arealabelpairs[ialp],
                        alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
            handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
                    color=clrs_arealabelpairs[ialp],marker='v')[0])
            ax.legend(handles,legendlabels,fontsize=9,frameon=False)
        
            ax = axes[isign,2]
            xdata = np.nanmean(mean_resp_split_aligned[ialp,:,0,idx_N].T,axis=1)
            ydata = np.nanmean(mean_resp_split_aligned[ialp,:,1,idx_N].T,axis=1)
            b = linregress(xdata,ydata)
            params_regress_mean[ialp,:] = b[:3]
            xvals = np.arange(0,3,0.1)
            yvals = params_regress_mean[ialp,0]*xvals + params_regress_mean[ialp,1]
            ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=1.3)
            ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.7,s=25)
            ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
            # ax.set_title(,fontsize=12,color=clrs_arealabelpairs[ialp])
            ax.text(0.6,0.15*ialp,'Slope: %1.2f\nOffest: %1.2f'%(params_regress_mean[ialp,0],params_regress_mean[ialp,1]),
                    transform=ax.transAxes,fontsize=8,color=clrs_arealabelpairs[ialp])
            ax.set_xlabel('%s low (events/F0) '%(alp.split('-')[0]))
            ax.set_ylabel('%s high'%(alp.split('-')[0]))
            ax.set_xlim([0,np.nanmax([xdata,ydata])*1.1])
            ax.set_ylim([0,np.nanmax([xdata,ydata])*1.1])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_PosNeg_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])


#%% Show correlation of slope and offset
clrs_arealabelpairs = ['green','purple']
fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharey=True,sharex=True)
for ialp,alp in enumerate(legendlabels): #Mult and Add
    ax = axes[ialp]
    idx_N = np.all((
            celldata['gOSI']>0,
            rangeresp>0.04,
                ),axis=0)
    x = params_regress[idx_N,ialp,0]
    y = params_regress[idx_N,ialp,1]
    # c = celldata['gOSI'][idx_N][~np.isnan(y)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    sns.scatterplot(x=x,y=y,s=5,alpha=0.2,color=clrs_arealabelpairs[ialp],ax=ax)
    # sns.scatterplot(x=x,y=y,s=5,alpha=0.2,c=celldata['gOSI'][idx_N],ax=ax)
    # ax.scatter(x=x,y=y,s=5,alpha=0.2,c=c)
    
    ax.set_xlim(np.nanpercentile(params_regress[idx_N,:,0],[0.2,99.8]))
    ax.set_ylim(np.nanpercentile(params_regress[idx_N,:,1],[0.2,99.8]))
    ax_nticks(ax, 3)
    ax.set_xlabel(u'Slope')
    ax.set_ylabel(u'Offset')
    ax.set_title(legendlabels[ialp],fontsize=11)
    ax.text(0.5,0.8,'r = %.2f' % (stats.pearsonr(x,y)[0]),transform=ax.transAxes,fontsize=9) #print(stats.pearsonr(x,y)[0])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Corr_Mult_Add_Mean_FF_FB_MinRespSubtracted_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'Corr_Mult_Add_Mean_FF_FB_BaselineSubtracted_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'Corr_Mult_Add_Mean_FF_FB_GR%dsessions' % (nSessions), formats = ['png'])


#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
# tuning curve of PM neurons and vice versa for feedback. Because the population activity in V1 and PM cofluctuates,
#  just taking the level of V1 or PM activity would confound the analysis with local activity levels. 
# So therefore I took the 10% of trials with the labeled cells being more active than unlabeled cells vs the 10% trials 
# with unlabeled cells being more active than labeled cells (e.g. for FF: mean of V1lab - mean of V1unl). This would 
# be a proxy of epochs of particularly high FF activity, vs epochs of low FF activity (while controlling for overall 
# activity levels). Then the population tuning curve of PMunl or PMlab is plotted computed on these trials separately.
# You can see that high FF activity has very small divisive effect, while high FB activity has a clear multiplicative 
# effect. I also checked the effect on individual neurons (fitting affine modulation per neuron) but they mainly reflect 
# the mean. There are also additive effects, but the magnitude of the additive effects does not seem larger for PM cells 
# when FF ratio is high (edited) 


arealabelpairs  = [
                    'V1labL2/3-V1unlL2/3-PMunlL2/3',
                    'V1labL2/3-V1unlL2/3-PMunlL5',
                    'PMlabL2/3-PMunlL2/3-V1unlL2/3',
                    'PMlabL5-PMunlL5-V1unlL2/3',
                    ]

arealabelpairs  = [
                    'V1labL2/3-V1unlL2/3-PMunlL2/3',
                    'V1labL2/3-V1unlL2/3-PMlabL2/3',
                    'V1labL2/3-V1unlL2/3-PMunlL5',
                    'V1labL2/3-V1unlL2/3-PMlabL5',
                    'PMlabL2/3-PMunlL2/3-V1unlL2/3',
                    'PMlabL2/3-PMunlL2/3-V1labL2/3',
                    'PMlabL5-PMunlL5-V1unlL2/3',
                    'PMlabL5-PMunlL5-V1labL2/3',
                    ]

arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'V1lab-V1unl-PMlabL2/3',
                    # 'V1lab-V1unl-PMunlL5',
                    # 'V1lab-V1unl-PMlabL5',
                    'PMlab-PMunl-V1unlL2/3',
                    'PMlab-PMunl-V1labL2/3',
                    ]

# arealabelpairs  = [
                    # 'V1unl-PMunl-PMunl',
                    # 'V1unl-PMunl-PMlab',
                    # 'V1lab-PMunl-PMunl',
                    # 'V1lab-PMunl-PMlab',
                    # 'PMunl-V1unl-V1unl',
                    # 'PMunl-V1unl-V1lab',
                    # 'PMlab-V1unl-V1unl',
                    # 'PMlab-V1unl-V1lab',
                    # ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())

nboots                  = 100
perc                    = 25
minnneurons             = 10
maxnoiselevel           = 20

alphathr                = 0.001
# maxnoiselevel           = 100
# mineventrate            = 0

mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
corrdata                = np.full((narealabelpairs,nSessions),np.nan)
corrdata_boot           = np.full((narealabelpairs,nSessions,nboots),np.nan)
corrdata_confounds      = np.full((narealabelpairs,nSessions,3),np.nan)
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)

# valuematching           = 'pop_coupling'
valuematching           = None
nmatchbins              = 5

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    # respdata            = zscore(sessions[ises].respmat, axis=1)
    # respdata            = zscore(sessions[ises].respmat_behavout, axis=1)

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    # idx_nearby          = filter_nearlabeled(sessions[ises],radius=50)

    for ialp,alp in enumerate(arealabelpairs):
        # idx_N1              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0]
        # idx_N2              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1]
        # idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2]

        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    # sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                    #   sessions[ises].celldata['OSI']>0.5,
                                    #   sessions[ises].celldata['gOSI']>0.5,
                                    # idx_nearby,

                                      ),axis=0))[0]
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    # sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        # sessions[ises].celldata['gOSI']>0.5,
                                        #   sessions[ises].celldata['OSI']>0.5,
                                    # idx_nearby,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                    #   sessions[ises].celldata['gOSI']>np.nanpercentile(sessions[ises].celldata['gOSI'],50),
                                    #   sessions[ises].celldata['OSI']>0.5,
                                    #   sessions[ises].celldata['gOSI']>0.5,
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        # if len(idx_N1) < minnneurons or len(idx_N2) < minnneurons or len(idx_N3) < minnneurons:
            # continue

        # if len(idx_N1) < minnneurons:
        #     continue

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue

        if valuematching is not None:
            #Get value to match from celldata for V1 matching
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_N1,idx_N2))
            group       = np.concatenate((np.zeros(len(idx_N1)),np.ones(len(idx_N2))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_N1      = np.intersect1d(idx_N1,idx_sub) #recover subset from idx_joint
            idx_N2      = np.intersect1d(idx_N2,idx_sub)
        
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        # meanpopact          = np.nanmean(zscore(respdata[idx_N1,:],axis=1),axis=0)

        corrdata_confounds[ialp,ises,0]      = np.corrcoef(meanpopact,sessions[ises].respmat_runspeed)[0,1]
        corrdata_confounds[ialp,ises,1]      = np.corrcoef(meanpopact,sessions[ises].respmat_videome)[0,1]
        corrdata_confounds[ialp,ises,2]      = np.corrcoef(meanpopact,sessions[ises].respmat_pupilarea)[0,1]

        corrdata[ialp,ises]      = np.corrcoef(meanpopact,np.nanmean(respdata[idx_N3,:],axis=0))[0,1]

        # sampleNneurons = min(np.sum(idx_N1),np.sum(idx_N2),np.sum(idx_N3))
        sampleNneurons = min(len(idx_N1),len(idx_N2),len(idx_N3))

        for iboot in range(nboots):
            bootidx_N1          = np.random.choice(idx_N1,sampleNneurons,replace=True)
            bootidx_N2          = np.random.choice(idx_N2,sampleNneurons,replace=True)
            bootidx_N3          = np.random.choice(idx_N3,sampleNneurons,replace=True)
            # corrdata_boot[ialp,ises,iboot] = np.corrcoef(np.nanmean(respdata[bootidx_N1,:],axis=0) - np.nanmean(respdata[bootidx_N2,:],axis=0),
                                            #   np.nanmean(respdata[bootidx_N3,:],axis=0))[0,1]
            corrdata_boot[ialp,ises,iboot] = np.corrcoef(np.nanmean(respdata[bootidx_N1,:],axis=0),
                                              np.nanmean(respdata[bootidx_N3,:],axis=0))[0,1]
        
        # idx_K1              = meanpopact < np.nanpercentile(meanpopact,perc)
        # idx_K2              = meanpopact > np.nanpercentile(meanpopact,100-perc)
        # # compute meanresp for trials with low and high difference in lab-unl activation
        # meanresp            = np.empty([N,len(oris),2])
        # ori_ses             = sessions[ises].trialdata['Orientation']
        # oris                = np.unique(ori_ses)
        # for i,ori in enumerate(oris):
        #     meanresp[:,i,0] = np.nanmean(respdata[:,np.logical_and(ori_ses==ori,idx_K1)],axis=1)
        #     meanresp[:,i,1] = np.nanmean(respdata[:,np.logical_and(ori_ses==ori,idx_K2)],axis=1)
        
        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            idx_T               = ori_ses == ori
            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0] = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1] = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response during still trials
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        # meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,0] = meanresp_pref[:,:,0] / tempmax
        # meanresp_pref[:,:,1] = meanresp_pref[:,:,1] / tempmax

        # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<alphathr) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
N = len(celldata)
params_regress = np.full((N,narealabelpairs,3),np.nan)
for iN in tqdm(range(N),total=N,desc='Fitting gain for each neuron'):
    for ialp,alp in enumerate(arealabelpairs):
        xdata = mean_resp_split[ialp,:,0,iN]
        ydata = mean_resp_split[ialp,:,1,iN]
        params_regress[iN,ialp,:] = linregress(xdata,ydata)[:3]

#%%
# idx_sigN = corrsig_cells[0,:]==1
# idx_sigN = corrsig_cells[0,:]==-1
# plt.hist(corrdata_cells[0,idx_sigN].flatten())
# corrsig_cells[ialp,idx_ses]==-1


#%%
plotdata = np.nanmean(corrdata_boot,axis=2)

fig,ax = plt.subplots(1,1,figsize=(2.5,3))
df = pd.DataFrame({'correlation': plotdata.flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticklabels(arealabelpairs,rotation=90,fontsize=8)

# my_savefig(fig,savedir,'FF_FB_poprate_arealayerpairs_GR_%dsessions' % (nSessions))

#%% 

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax=axes[0]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,0].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Running speed')

ax=axes[1]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,1].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Video ME')

ax=axes[2]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,2].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Pupil size')

for ax in axes:
    ax.set_ylim([-0.3,0.3])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
axes[0].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)
axes[1].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)
axes[2].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)

my_savefig(fig,savedir,'FF_FB_poprate_confounds_GR_%dsessions' % (nSessions))

#%% 
params_regress_mean = np.full((narealabelpairs,3),np.nan)
# clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    # idx_N =  np.array(celldata['gOSI']>0.5)
    # idx_N = params_regress[:,ialp,2] > 0.5
    # idx_N = corrsig_cells[ialp,:]==-1

    idx_N =  np.all((
                    # celldata['gOSI']>0.5,
                    # celldata['nearby'],
                    # corrsig_cells[ialp,:]==1,
                    # np.any(mean_resp_split>0.5,axis=(0,1,2)),
                    np.any(params_regress[:,:,2] > 0.5,axis=1),
                     ),axis=0)

    xdata = np.nanmean(mean_resp_split[ialp,:,0,idx_N].T,axis=1)
    ydata = np.nanmean(mean_resp_split[ialp,:,1,idx_N].T,axis=1)
    b = linregress(xdata,ydata)
    params_regress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = params_regress_mean[ialp,0]*xvals + params_regress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=1.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.7,s=35)
    ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
    ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.text(0.1,0.8,'Slope: %1.2f\nOffest: %1.2f'%(params_regress_mean[ialp,0],params_regress_mean[ialp,1]),
            transform=ax.transAxes,fontsize=8)
    # ax.legend(frameon=False,loc='lower right')
    ax.set_xlabel('%s low (events/F0) '%(alp.split('-')[0]))
    ax.set_ylabel('%s high'%(alp.split('-')[0]))
    ax.set_xlim([0,np.nanmax([xdata,ydata])*1.1])
    ax.set_ylim([0,np.nanmax([xdata,ydata])*1.1])
# ax.set_xlim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_ylim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_xlim([np.nanmin(np.nanmean(mean_resp_split,axis=(3)))*1.1,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_ylim([np.nanmin(np.nanmean(mean_resp_split,axis=(3)))*1.1,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_%dsessions' % (nSessions), formats = ['png'])

#%%
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
for ialp,alp in enumerate(arealabelpairs):
    sns.histplot(params_regress[:,ialp,2],bins=np.linspace(-1,1.1,25),element='step',stat='probability',
                 color=clrs_arealabelpairs[ialp],fill=False,ax=ax)
ax.set_xlabel('R2')
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
my_savefig(fig,savedir,'AffineModel_R2_%dsessions' % (nSessions), formats = ['png'])

# #%% 
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)

# fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,6),sharey='row')
# for ialp,alp in enumerate(arealabelpairs):
#     for iparam in range(2):
#         ax = axes[iparam,ialp]
#         idx_N = params_regress[:,ialp,2] > 0.5
#         # idx_N =  celldata['gOSI']>0.5

#         sns.histplot(data=params_regress[idx_N,ialp,iparam],color=clrs_arealabelpairs[ialp],
#                      ax=ax,stat='probability',bins=np.arange(-1,3,0.1))
#         ax.axvline(0,color='grey',ls='--',linewidth=1)
#         ax.axvline(1,color='grey',ls='--',linewidth=1)
#         ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
#                 color=clrs_arealabelpairs[ialp],marker='v')
#         ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
#         if iparam == 0:
#             ax.set_xlabel('Slope')
#         else:
#             ax.set_xlabel('Offset')

# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
clrs_arealabelpairs = ['green','purple']
legendlabels = ['FF','FB']
clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(1,2,figsize=(6,3))
for iparam in range(2):
    ax = axes[iparam]
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        bins = np.arange(-0.5,3,0.1)
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        bins = np.arange(-0.15,0.15,0.01)
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    handles = []
    for ialp,alp in enumerate(arealabelpairs):
        idx_N = params_regress[:,ialp,2] > 0.5
        # idx_N =  celldata['gOSI']>0.5

        sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],alpha=0.3,fill=True,linewidth=1,
                     ax=ax,stat='probability',bins=bins)
        # ax.axvline(0,color='grey',ls='--',linewidth=1)
        # ax.axvline(1,color='grey',ls='--',linewidth=1)
        handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
        # ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.legend(handles,legendlabels)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
from scipy import stats
clrs_arealabelpairs = ['green','purple']
ticklabels = ['FF','FB']
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,2,figsize=(4.5,3))
for iparam in range(2):
    ax = axes[iparam]
    # idx_N = np.all(params_regress[:,:,2] > 0.3,axis=1)
    # idx_N = np.any(params_regress[:,:,2] > 0.3,axis=1)
    # idx_N = params_regress[:,:,2] > 0.5
    # idx_N =  celldata['OSI']>0.5
    idx_N =  np.all((
                    # celldata['gOSI']>0.5,
                    # celldata['nearby'],
                    # np.any(corrsig_cells==1,axis=0),
                    # np.any(corrsig_cells==-1,axis=0),
                    np.any(params_regress[:,:,2] > 0.5,axis=1),
                     ),axis=0)
    sns.barplot(data=params_regress[idx_N,:,iparam],palette=clrs_arealabelpairs,
                ax=ax,estimator=np.nanmean,errorbar=('ci', 95))
    if np.shape(params_regress)[1]==2:
        h,p = stats.ttest_ind(params_regress[idx_N,0,iparam],
                            params_regress[idx_N,1,iparam],nan_policy='omit')
        p = p * narealabelpairs
        add_stat_annotation(ax, 0.2, 0.8, np.nanmean(params_regress[idx_N,:,iparam],axis=0).max()*1.1, p, h=0)
    elif np.shape(params_regress)[1]==4:
        for iidx,idx in enumerate([[0,2],[0,1],[2,3]]):
            h,p = stats.ttest_ind(params_regress[idx_N,idx[0],iparam],
                                params_regress[idx_N,idx[1],iparam],nan_policy='omit')
            p = p * narealabelpairs
            add_stat_annotation(ax, idx[0], idx[1], np.nanmean(params_regress[np.ix_(np.where(idx_N)[0],idx,[iparam])],axis=0).max()*1.2+iidx*0.01, p, h=0.001)
    
        # h,p = stats.ttest_ind(params_regress[idx_N,0,iparam],
        #                     params_regress[idx_N,2,iparam],nan_policy='omit')
        # add_stat_annotation(ax, 0, 2, np.nanmean(params_regress[idx_N,:,iparam],axis=0).max()*1.1, p, h=0.0)
 
    ax.tick_params(labelsize=9,rotation=0)
    ax.set_xticklabels(ticklabels)
    if iparam == 0:
        ax.set_title('Multiplicative')
    else:
        ax.set_title('Additive')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_sigN_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_sigP_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% 
# DEPRECATED: 


#%% Check whether the affine modulation modulation depends on the activity of the other area
perc                = 10

arealabelpairs      = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    ]

# arealabelpairs  = [
#                     'PMunl-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1unl',
#                     'PMlab-V1lab',
#                     # 'PMunl-PMlab'
#                     ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respmat                = zscore(sessions[ises].respmat, axis=1)
    # poprate             = np.nanmean(data,axis=0)
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = sessions[ises].celldata['arealabel'] == alp.split('-')[0]
        idx_N2              = sessions[ises].celldata['arealabel'] == alp.split('-')[1]

        meanpopact          = np.nanmean(respmat[idx_N1,:],axis=0)
        idx_K1              = meanpopact < np.nanpercentile(meanpopact,perc)
        idx_K2              = meanpopact > np.nanpercentile(meanpopact,100-perc)

        # compute meanresp
        meanresp            = np.empty([N,len(oris),2])
        for i,ori in enumerate(oris):
            meanresp[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K1)],axis=1)
            meanresp[:,i,1] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K2)],axis=1)
        
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response during still trials
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        # meanresp_pref
        # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N2])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N2]
    
#%% 
params_regress_mean = np.full((narealabelpairs,3),np.nan)
clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    xdata = np.nanmean(mean_resp_split[ialp,:,0,:],axis=1)
    ydata = np.nanmean(mean_resp_split[ialp,:,1,:],axis=1)
    b = linregress(xdata,ydata)
    params_regress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = params_regress_mean[ialp,0]*xvals + params_regress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=0.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.6,s=25)
    ax.plot([0,3],[0,3],'grey',ls='--',linewidth=1)
    ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
ax.legend(frameon=False,loc='lower right')
ax.set_xlabel('Low (Norm. Response)')
ax.set_ylabel('High (Norm. Response)')
ax.set_xlim([0,3.5])
ax.set_ylim([0,3.5])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

# temp = mean_resp_split[ialp,i,:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K1)]
# print(alp,ori,np.nanmean(temp),np.nanstd(temp),np.nanmedian(temp),np.nanpercentile(temp,25),np.nanpercentile(temp,75))


   #    #           #####  ####### #     # ####### ######  ####### #       
  # #   #          #     # #     # ##    #    #    #     # #     # #       
 #   #  #          #       #     # # #   #    #    #     # #     # #       
#     # #          #       #     # #  #  #    #    ######  #     # #       
####### #          #       #     # #   # #    #    #   #   #     # #       
#     # #          #     # #     # #    ##    #    #    #  #     # #       
#     # #######     #####  ####### #     #    #    #     # ####### ####### 


#%% Show tuning curve when activityin the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    'V1lab-V1unl-ALunlL2/3',
                    'PMlab-PMunl-ALunlL2/3',
                    # 'ALunl-V1unl-PMunlL2/3',
                    # 'ALunl-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
perc                    = 25

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5

minnneurons             = 10
maxnoiselevel           = 20
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells),np.nan)

np.random.seed(98)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
        idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
        idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue
        
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        errorresp           = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            errorresp[:,i,0]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K1)))
            errorresp[:,i,1]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K2)))

        # meanresp -= np.nanmin(meanresp[:,:,0],axis=(1),keepdims=True)[:,:,None]

        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp[idx_N3]
        error_resp_split[ialp,:,:,idx_ses] = errorresp[idx_N3]

        # #Aligned:
        # prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)
        # # prefori                     = np.argmax(meanresp[:,:,0],axis=1)

        # meanresp_pref          = meanresp.copy()
        # for n in range(N):
        #     meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
        #     meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # # normalize by peak response
        # tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        # meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        # mean_resp_split_aligned[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        # tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        # tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        # corrdata_cells[ialp,idx_ses] = tempcorr
        # tempsig = (tempsig<alphathr) * np.sign(tempcorr)
        # corrsig_cells[ialp,idx_ses] = tempsig

# Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
N = len(celldata)
params_regress = np.full((N,narealabelpairs,3),np.nan)
for iN in tqdm(range(N),total=N,desc='Fitting gain for each neuron'):
    for ialp,alp in enumerate(arealabelpairs):
        xdata = mean_resp_split[ialp,:,0,iN]
        ydata = mean_resp_split[ialp,:,1,iN]
        params_regress[iN,ialp,:] = linregress(xdata,ydata)[:3]

#%% Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

#%%

from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols


#%% 
nresamples = 100
clrs_arealabelpairs = ['green','purple','orange','red','blue','grey']
legendlabels = ['$V1_{PM}>PM$',
                '$PM_{V1}>V1$',
                '$V1_{PM}>AL$',
                '$PM_{V1}>AL$',
                ]
narealabelpairs = len(arealabelpairs)

# clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(1,2,figsize=(6,3))
# fig,axes = plt.subplots(1,2,figsize=(12,6))
for iparam in range(2):
    ax = axes[iparam]
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        bins = np.arange(-0.5,5,0.05)
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        bins = np.arange(-0.05,0.05,0.0005)
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    handles = []

    # idx_N =  celldata['gOSI']>0.4
    idx_N =  rangeresp>0.04
    for ialp,alp in enumerate(arealabelpairs):
        if ialp<2:
            continue
        
        # print(np.sum(~np.isnan(params_regress[idx_N,ialp,iparam])))
        sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],
                     alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
        handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
    ax.legend(handles,legendlabels[2:],fontsize=9,frameon=False)

    #Test for effect in AL:
    h,p = stats.ttest_ind(params_regress[idx_N,2,iparam],
                            params_regress[idx_N,3,iparam],nan_policy='omit')
    p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
    ax.text(0.5, 0.5, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes)
    print('p=%1.2f' % (p))

    #Statistics:
    df = pd.DataFrame({'var': params_regress[idx_N,:,iparam].flatten(),
                       'arealabelpair': np.tile(arealabelpairs,np.sum(idx_N))})
    df.dropna(inplace=True)
    df['source'] = ''
    df.loc[df['arealabelpair'].str.contains('V1lab'),'source'] = 'V1lab'
    df.loc[df['arealabelpair'].str.contains('PMlab'),'source'] = 'PMlab'
    df['target'] = 'V1PM'
    df.loc[df['arealabelpair'].str.contains('AL'),'target'] = 'AL'

    formula = "var ~ source*target" #model with interaction
    lm = ols(formula, df).fit()
    table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
    print(table)
    print('Interaction effect: p=%1.2f' % (table.loc['source:target','PR(>F)']))

    #The number of recorded neurons in AL is smaller than those in V1 and PM:
    #pick nALneurons random entries from the entries that have 'target' set to 'V1PM' to match the number of target categories
    idx_V1PM = np.where(df['target'] == 'V1PM')[0] 
    idx_AL = np.where(df['target'] == 'AL')[0]
    # print(np.sum(df['target'] == 'AL'))
    # print(np.sum(df['target'] == 'V1PM'))
    fracsig = np.zeros(nresamples)
    for isub in range(nresamples):
        idx_V1PM_to_pick = np.random.choice(idx_V1PM, size=len(idx_AL), replace=False)

        df_sub = df.iloc[np.concatenate([idx_AL, idx_V1PM_to_pick]),:]

        lm = ols(formula, df_sub).fit()
        table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
        fracsig[isub] = table.loc['source:target','PR(>F)']<=0.05

    print('%2.1f%% of resamples had a significant interaction' % (np.sum(fracsig)/nresamples*100))
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'AL_HistCoefs_StillTrials_gOSI05_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])


#%% 
# nresamples = 100
# clrs_arealabelpairs = ['green','purple','orange','red','blue','grey']
# legendlabels = ['$V1_{PM}>PM$',
#                 '$PM_{V1}>V1$',
#                 '$V1_{PM}>AL$',
#                 '$PM_{V1}>AL$',
#                 ]
# narealabelpairs = len(arealabelpairs)

# #pick nALneurons random entries from the entries that have 'target' set to 'V1PM' to match the number of target categories
# nALneurons = np.sum(df['target'] == 'AL')
# nALneurons = 200

# params_regress_mean = np.zeros((narealabelpairs,2))
# for isub in range(nresamples):
#     params_regress_diff = np.stack((params_regress[:,1,:]-params_regress[:,0,:],params_regress[:,3,:]+params_regress[:,2,:]),axis=1)

# # clrs_arealabelpairs = ['grey','pink','grey','red']
# fig,axes = plt.subplots(1,2,figsize=(6,3))
# # fig,axes = plt.subplots(1,2,figsize=(12,6))
# for iparam in range(2):
#     ax = axes[iparam]
#     if iparam == 0:
#         ax.set_xlabel('Multiplicative Slope')
#         bins = np.arange(-0.5,5,0.05)
#         ax.axvline(1,color='grey',ls='--',linewidth=1)
#     else:
#         ax.set_xlabel('Additive Offset')
#         bins = np.arange(-0.05,0.05,0.0005)
#         ax.axvline(0,color='grey',ls='--',linewidth=1)
#     handles = []

#     # idx_N =  celldata['gOSI']>0.4
#     idx_N =  rangeresp>0.04
#     for ialp,alp in enumerate(arealabelpairs):
#         if ialp<2:
#             continue
        
#         # print(np.sum(~np.isnan(params_regress[idx_N,ialp,iparam])))
#         sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
#                      color=clrs_arealabelpairs[ialp],
#                      alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
#         handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
#                 color=clrs_arealabelpairs[ialp],marker='v')[0])
#     ax.legend(handles,legendlabels[2:],fontsize=9,frameon=False)

#     #Test for effect in AL:
#     h,p = stats.ttest_ind(params_regress[idx_N,2,iparam],
#                             params_regress[idx_N,3,iparam],nan_policy='omit')
#     p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
#     ax.text(0.5, 0.5, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes)
#     print('p=%1.2f' % (p))

#     #Statistics:
#     df = pd.DataFrame({'var': params_regress[idx_N,:,iparam].flatten(),
#                        'arealabelpair': np.tile(arealabelpairs,np.sum(idx_N))})
#     df.dropna(inplace=True)
#     df['source'] = ''
#     df.loc[df['arealabelpair'].str.contains('V1lab'),'source'] = 'V1lab'
#     df.loc[df['arealabelpair'].str.contains('PMlab'),'source'] = 'PMlab'
#     df['target'] = 'V1PM'
#     df.loc[df['arealabelpair'].str.contains('AL'),'target'] = 'AL'

#     formula = "var ~ source*target" #model with interaction
#     lm = ols(formula, df).fit()
#     table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
#     print(table)
#     print('Interaction effect: p=%1.2f' % (table.loc['source:target','PR(>F)']))

#     #The number of recorded neurons in AL is smaller than those in V1 and PM:
#     #pick nALneurons random entries from the entries that have 'target' set to 'V1PM' to match the number of target categories
#     idx_V1PM = np.where(df['target'] == 'V1PM')[0] 
#     idx_AL = np.where(df['target'] == 'AL')[0]
#     # print(np.sum(df['target'] == 'AL'))
#     # print(np.sum(df['target'] == 'V1PM'))
#     fracsig = np.zeros(nresamples)
#     for isub in range(nresamples):
#         idx_V1PM_to_pick = np.random.choice(idx_V1PM, size=len(idx_AL), replace=False)

#         df_sub = df.iloc[np.concatenate([idx_AL, idx_V1PM_to_pick]),:]

#         lm = ols(formula, df_sub).fit()
#         table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
#         fracsig[isub] = table.loc['source:target','PR(>F)']<=0.05

#     print('%2.1f%% of resamples had a significant interaction' % (np.sum(fracsig)/nresamples*100))
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'AL_HistCoefs_StillTrials_gOSI05_%dGRsessions' % (nSessions), formats = ['png'])
# # my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])
