# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Lib imports ###################################################
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn as sns
os.chdir('e:\\Python\\molanalysis')

#### linear approaches: regression and dimensionality reduction 
from numpy import linalg
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy.stats import zscore, pearsonr
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.regress_lib import *
from sklearn.preprocessing import LabelEncoder

#Personal libs:
from utils.RRRlib import LM,Rss,EV
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import plot_excerpt
from utils.tuning import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\NoiseRegression\\')
randomseed  = 5
calciumversion = 'dF'
calciumversion = 'deconv'

#%% #############################################################################
#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE11998','2024_05_02'], #GN
                                ['LPE12013','2024_05_02']]) #GN
#%% Load sessions lazy: 
minlabcells = 25
sessions,nSessions   = filter_sessions(protocols = ['GN'],filter_areas=['V1','PM'],
                                       min_lab_cells_V1=minlabcells,min_lab_cells_PM=minlabcells)

#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)
    
#%% Compute tuning metrics:
sessions = compute_tuning_wrapper(sessions)

#%% ## 
# idx_V1 = np.where(sessions[ises].celldata['roi_name']=='V1')[0]
# idx_PM = np.where(sessions[ises].celldata['roi_name']=='PM')[0]
idx_V1 = sessions[ises].celldata['roi_name']=='V1'
idx_PM = sessions[ises].celldata['roi_name']=='PM'

#%% ########## PCA on trial-averaged responses ############
######### plot result as scatter by orientation ########

respmat_zsc = zscore(sessions[ises].respmat,axis=1) # zscore for each neuron across trial responses

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(respmat_zsc.T).T #fit pca to response matrix (n_samples by n_features)
#dimensionality is now reduced from N by K to ncomp by K

oris, speeds    = [np.unique(sessions[ises].trialdata[col]).astype('int') for col in ('centerOrientation', 'centerSpeed')]
noris           = len(oris) 
nspeeds         = len(speeds)
areas           = np.array(['PM', 'V1'], dtype=object)
clrs,labels     = get_clr_gratingnoise_stimuli(oris,speeds)

ori_ind         = [np.argwhere(np.array(sessions[ises].trialdata['centerOrientation']) == ori)[:, 0] for ori in oris]
speed_ind       = [np.argwhere(np.array(sessions[ises].trialdata['centerSpeed']) == speed)[:, 0] for speed in speeds]

shade_alpha      = 0.2
lines_alpha      = 0.8

# handles = []
projections = [(0, 1), (1, 2), (0, 2)]
projections = [(0, 1), (1, 2), (3, 4)]
fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
for ax, proj in zip(axes, projections):
    for iO, ori in enumerate(oris):                                #plot orientation separately with diff colors
        for iS, speed in enumerate(speeds):                       #plot speed separately with diff colors
            idx = np.intersect1d(ori_ind[iO],speed_ind[iS])
            x = Xp[proj[0],idx]                          #get all data points for this ori + speed along first PC of projection pairs
            y = Xp[proj[1],idx]                          #and the second

            # x = Xp[proj[0],ori_ind[io]]                          #get all data points for this ori along first PC or projection pairs
            # y = Xp[proj[1],ori_ind[io]]                          #and the second
            # handles.append(ax.scatter(x, y, color=clrs[iO,iS,:], s=sessions[ises].respmat_runspeed[idx], alpha=0.8))     #each trial is one dot
            ax.scatter(x, y, color=clrs[iO,iS,:], s=sessions[ises].respmat_runspeed[idx], alpha=0.8)    #each trial is one dot
            ax.set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
            ax.set_ylabel('PC {}'.format(proj[1]+1))

axes[2].legend(labels.flatten(),fontsize=8,bbox_to_anchor=(1,1))
sns.despine(fig=fig, top=True, right=True)
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'GN_Stimuli','PCA_allStim_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% LDA of all trials
from sklearn.metrics import accuracy_score

colors,labels = get_clr_gratingnoise_stimuli(oris,speeds)

areas = ['V1', 'PM']
plotdim = 0
ises = 0

lda_ori = LinearDiscriminantAnalysis()
lda_spd = LinearDiscriminantAnalysis()
arealabels = ['V1unl', 'PMunl', 'V1lab', 'PMlab']
# fig, axes = plt.subplots(1, len(areas), figsize=[4*len(areas), 4])
fig, axes = plt.subplots(1, len(arealabels), figsize=[4*len(arealabels), 4])
# for iax, area in enumerate(areas):
for iax, area in enumerate(arealabels):
    idx_N           = np.where(np.all((sessions[ises].celldata['arealabel']==area,
                                    sessions[ises].celldata['noise_level']<100,	
                                    ),axis=0))[0]
    # idx     = sessions[ises].celldata['roi_name'] == area
    
    # X       = respmat_zsc[idx, :]
    X       = sessions[ises].respmat[idx_N, :]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=randomseed)
    y_pred_ori = np.zeros_like(sessions[ises].trialdata['centerOrientation'])
    y_pred_spd = np.zeros_like(sessions[ises].trialdata['centerSpeed'])
    for train_index, test_index in kf.split(X.T):
        X_train, X_test = X.T[train_index], X.T[test_index]
        y_train, y_test = sessions[ises].trialdata.loc[train_index,['centerOrientation', 'centerSpeed']],sessions[ises].trialdata.loc[test_index,['centerOrientation', 'centerSpeed']]      
        
        lda_ori.fit(X_train, y_train['centerOrientation'])
        y_pred_ori[test_index] = lda_ori.predict(X_test)

        lda_spd.fit(X_train, y_train['centerSpeed'])
        y_pred_spd[test_index] = lda_spd.predict(X_test)

    for iOri, ori in enumerate(oris):
        for iSpd, spd in enumerate(speeds):
            idx_T = np.logical_and(sessions[ises].trialdata['centerOrientation'] == ori, sessions[ises].trialdata['centerSpeed'] == spd)
            axes[iax].scatter(lda_ori.transform(X.T)[idx_T,plotdim], lda_spd.transform(X.T)[idx_T,plotdim], color=colors[iOri,iSpd,:], marker='o', alpha=0.5)
    print('Accuracy ori: {:.2f}%'.format(accuracy_score(sessions[ises].trialdata['centerOrientation'], y_pred_ori)*100))
    print('Accuracy spd: {:.2f}%'.format(accuracy_score(sessions[ises].trialdata['centerSpeed'], y_pred_spd)*100))
    axes[iax].set_title(area)
    axes[iax].set_xlabel('LDA %d Ori' % plotdim)
    axes[iax].set_ylabel('LDA %d Speed' % plotdim)
    axes[iax].legend(labels.flatten(),fontsize=8,bbox_to_anchor=(1,1))
sns.despine(fig=fig, top=True, right=True, offset = 3)
plt.tight_layout()
my_savefig(fig,savedir,'LDA_ori_speed_allStim_dim' + str(plotdim+1) + '_' + sessions[ises].sessiondata['session_id'][0])
# plt.savefig(os.path.join(savedir,'LDA_ori_speed_allStim_dim' + str(plotdim) + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')



#%% Decoding with LDA of trialdata['stimCond'] for all sessions and different neural populations
nsampleneurons  = 25
nmodelfits      = 5

areas           = ['V1', 'PM']
arealabels      = ['V1unl', 'PMunl', 'V1lab', 'PMlab']
narealabels     = len(arealabels)

dec_perf        = np.full((nSessions,len(arealabels),nmodelfits),np.nan)

for ises in range(nSessions):
    idx_T           = np.ones(len(sessions[ises].trialdata['Orientation']),dtype=bool)
    for iax, area in enumerate(arealabels):
        idx_N           = np.where(np.all((sessions[ises].celldata['arealabel']==area,
                                        # sessions[ises].celldata['noise_level']<20,	
                                        ),axis=0))[0]

        for imf in range(nmodelfits):
            idx_N_sub       = np.random.choice(idx_N,nsampleneurons,replace=False) #take random subset of neurons

            X               = sessions[ises].respmat[np.ix_(idx_N_sub,idx_T)].T #get trialaveraged activity of these neurons
            
            X               = zscore(X,axis=0)  #Z score activity for each neuron

            Y               = sessions[ises].trialdata['stimCond'] #get orixspeed center condition
            Y               = LabelEncoder().fit_transform(Y) #convert to labels
        
            # print(find_optimal_lambda(X,Y,model_name='LDA',kfold=5))

            dec_perf[ises,iax,imf],_,_,_ = my_decoder_wrapper(X,Y,model_name='LDA',kfold=5,lam=0.01,subtract_shuffle=False,
                          scoring_type='accuracy_score',norm_out=False)

#%% Plot decoding performance:
fig, ax = plt.subplots(1, 1, figsize=[3,3])
for iax, area in enumerate(arealabels):
    for ises in range(nSessions):
        ax.plot([iax], [np.nanmean(dec_perf[ises,iax,:])], 'o', color=get_clr_area_labeled([area]), alpha=0.8)
    ax.errorbar([iax+.25], [np.nanmean(dec_perf[:,iax,:])], yerr=[np.nanstd(dec_perf[:,iax,:])], color=get_clr_area_labeled([area]), alpha=0.8, fmt='o', capsize=3)
ax.set_xlabel('Population')
ax.set_ylabel('Decoding performance')
ax.set_xticks(range(narealabels),arealabels)
ax.set_ylim([0,1])
ax.axhline(1/9,linestyle='--',color='k',alpha=0.5)
sns.despine(fig=fig, top=True, right=True, offset = 3)
plt.tight_layout()
my_savefig(fig,savedir,'Decoding_OriSpeed_V1PMlabeled_%dsessions' % nSessions)

#%% Define variables for external regression dimensions:
slabels     = ['Ori','Speed','RunSpeed','videoME']
scolors     = get_clr_GN_svars(slabels)

#%% ################## PCA unsupervised display of noise around center for each condition #################
S   = np.vstack((sessions[ises].trialdata['deltaOrientation'],
               sessions[ises].trialdata['deltaSpeed'],
               sessions[ises].respmat_runspeed,
               sessions[ises].respmat_videome))
NS          = np.shape(S)[0]

cmap = plt.get_cmap('hot')
proj = (0, 1) 
# proj = (1, 2)
proj = (2, 3)
# proj = (3, 4)

for iSvar in range(NS):
    fig, axes = plt.subplots(3, 3, figsize=[9, 9])
    for iO, ori in enumerate(oris):                                #plot orientation separately with diff colors
        for iS, speed in enumerate(speeds):                       #plot speed separately with diff colors
            idx         = np.intersect1d(ori_ind[iO],speed_ind[iS])
            
            Xp          = pca.fit_transform(sessions[ises].respmat[:,idx].T).T #fit pca to response matrix (n_samples by n_features)
            # Xp          = pca.fit_transform(zscore(sessions[ises].respmat[:,idx],axis=1).T).T #fit pca to response matrix (n_samples by n_features)
            #dimensionality is now reduced from N by K to ncomp by K

            x = Xp[proj[0],:]                          #get all data points for this ori along first PC or projection pairs
            y = Xp[proj[1],:]                          #get all data points for this ori along first PC or projection pairs
            
            c = cmap(minmax_scale(S[iSvar,idx], feature_range=(0, 1)))[:,:3]

            sns.scatterplot(x=x, y=y, c=c,ax = axes[iO,iS],s=10,legend = False,edgecolor =None)
            plt.title(slabels[iSvar])
            axes[iO,iS].set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
            axes[iO,iS].set_ylabel('PC {}'.format(proj[1]+1))
    plt.suptitle(slabels[iSvar],fontsize=15)
    sns.despine(fig=fig, top=True, right=True)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,'GN_PCA','PCA' + str(proj) + '_color' + slabels[iSvar] + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ################## Regression of behavioral variables onto neural data #############
###### First identify single neurons that are correlated with behavioral variables ####
# i.e. the covariance/correlation between neuronal responses and delta orientation/speed:
# So for each stimulus condition (ori x speed combination) we compute the covariance/correlation 

ises  = 0

# Define neural data parameters
N,K         = np.shape(sessions[ises].respmat)
S           = np.vstack((sessions[ises].trialdata['deltaOrientation'],
               sessions[ises].trialdata['logdeltaSpeed'],
               sessions[ises].respmat_runspeed,
               sessions[ises].respmat_videome))
NS          = np.shape(S)[0]

ori_ind         = [np.argwhere(np.array(sessions[ises].trialdata['centerOrientation']) == ori)[:, 0] for ori in oris]
speed_ind       = [np.argwhere(np.array(sessions[ises].trialdata['centerSpeed']) == speed)[:, 0] for speed in speeds]

corrmat = np.empty((N,NS,noris,nspeeds))
for iN in tqdm(range(N),desc='Computing correlations for neuron'):
    for iSvar in range(NS):
        for iO, ori in enumerate(oris): 
            for iS, speed in enumerate(speeds): 
                idx = np.intersect1d(ori_ind[iO],speed_ind[iS])
                corrmat[iN,iSvar,iO,iS] = np.corrcoef(S[iSvar,idx],sessions[ises].respmat[iN,idx])[0,1]   

# for ises in range(nSessions):
sessions[ises].celldata['pref_ori_corr_ori'],sessions[ises].celldata['pref_speed_corr_ori'] = get_pref_orispeed(corrmat[:,0,:,:].squeeze(),oris,speeds)
sessions[ises].celldata['pref_ori_corr_speed'],sessions[ises].celldata['pref_speed_corr_speed'] = get_pref_orispeed(corrmat[:,1,:,:].squeeze(),oris,speeds)

# #%% Is the normalized response to a certain stimulus condition related to the amount of correlation to the jitter
# # This is to be expected somewhat, that a neuron is sensitive to stimulus perturbations at the stimulus it responds to,
# # HOwever, likely the sharpest part of the tuning curve is outside its preferred stimulus

# fig,axes = plt.subplots(1,2,figsize=(6,3))
# for ises in range(nSessions):
#     resp_mean,resp_res      = mean_resp_gn(sessions[ises])
#     nCells                  = np.shape(resp_mean)[0]
#     resp_mean               = np.reshape(resp_mean,(nCells,noris*nspeeds))
#     resp_mean               = minmax_scale(resp_mean,feature_range=(0,1),axis=1)
    
#     corr_ori                 = np.reshape(np.abs(sessions[ises].corr_ori),(nCells,noris*nspeeds))
#     corr_speed               = np.reshape(np.abs(sessions[ises].corr_speed),(nCells,noris*nspeeds))
    
#     axes[0].scatter(resp_mean,corr_ori,s=8,alpha=0.3)
#     axes[1].scatter(resp_mean,corr_speed,s=8,alpha=0.3)
#     axes[0].text(0.7,0.5+ises*0.05,s='r=%1.3f' % np.corrcoef(resp_mean.flatten(),corr_ori.flatten())[0,1])
#     axes[1].text(0.7,0.5+ises*0.05,s='r=%1.3f' % np.corrcoef(resp_mean.flatten(),corr_speed.flatten())[0,1])
#     axes[0].set_ylim([0,0.8])
#     axes[1].set_ylim([0,0.8])

#%% #### and plot as density #################################
fig,ax = plt.subplots(1,1,figsize=(2.5,2.5))
for iSvar in range(NS):
    sns.kdeplot(corrmat[:,iSvar,:,:].flatten(),ax=ax,color=scolors[iSvar],linewidth=1.5)
plt.legend(slabels,frameon=False,loc='upper right',fontsize=7)
plt.xlabel('Correlation (neuron to variable)')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'KDE_Correlations_Svars' + '.png'), format = 'png')

#%% Make correlation absolute and find max correlation across stimulus conditions (3 orientations x 3 speeds)
corrmat_condmax = np.max(np.max(np.abs(corrmat),axis=3),axis=2)

#%% #### and plot as density #################################
fig,ax = plt.subplots(1,1,figsize=(2.5,2.5))
for iSvar in range(NS):
    sns.kdeplot(corrmat_condmax[:,iSvar].flatten(),ax=ax,color=scolors[iSvar],linewidth=1.5)
plt.legend(slabels,frameon=False,loc='upper right',fontsize=7)
plt.xlabel('Abs. corr (neuron to variable)')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'KDE_Correlations_Svars_condMax' + '.png'), format = 'png')

#%% ## Show the activity fluctuations as a function of variability in the behavioral vars for a couple of neurons:
nexamples = 4 # per variable
plt.rcParams.update({'font.size': 7})

fig,ax = plt.subplots(NS,nexamples,figsize=(6,6))

for iSvar in range(NS):
    idxN,idxO,idxS  = np.where(np.logical_or(corrmat[:,iSvar,:,:]>np.percentile(corrmat[:,iSvar,:,:].flatten(),98),
                                             corrmat[:,iSvar,:,:]<np.percentile(corrmat[:,iSvar,:,:].flatten(),2)))
    idx_examples    = np.random.choice(idxN,nexamples)
    
    for iN in range(nexamples):
        
        idx_trials = np.intersect1d(ori_ind[idxO[idx_examples[iN]==idxN][0]],speed_ind[idxS[idx_examples[iN]==idxN][0]])
        ax = plt.subplot(NS,nexamples,iN + 1 + iSvar*nexamples)
        ax.scatter(S[iSvar,idx_trials],sessions[ises].respmat[idx_examples[iN],idx_trials],
                   s=10,alpha=0.7,marker='.',color=scolors[iSvar])
        ax.set_title(slabels[iSvar],fontsize=9)
        # ax.set_xlim([])
        ax.set_ylim(np.percentile(sessions[ises].respmat[idx_examples[iN],idx_trials],[0,100]))
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Examplecells_correlated_with_Svars_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ###### Population regression of noise around each center stimulus (for all sessions)

#%% Decoding with regression of noise around each center stimulus 
# for all sessions and different neural populations
nsampleneurons  = 25
nmodelfits      = 20
kfold           = 2
lam             = 50
NS              = 4
filter_nearby   = True

arealabels      = ['V1unl', 'V1lab', 'PMunl', 'PMlab']
# arealabels      = ['V1unl', 'PMunl']
narealabels     = len(arealabels)
stimConds       = np.unique(sessions[ises].trialdata['stimCond'])
nstims          = len(stimConds)
stimLabels      = np.array(['%ddeg/%ddegs' % (sessions[ises].trialdata['centerOrientation'][np.where(sessions[ises].trialdata['stimCond']==i)[0][0]],
                                             sessions[ises].trialdata['centerSpeed'][np.where(sessions[ises].trialdata['stimCond']==i)[0][0]]) for i in stimConds])

dec_perf = np.full((len(arealabels),NS,nSessions,nstims,nmodelfits),np.nan)

for ises in range(nSessions):
    # Define neural data parameters
    S           = np.vstack((sessions[ises].trialdata['deltaOrientation'],
                sessions[ises].trialdata['logdeltaSpeed'],
                sessions[ises].respmat_runspeed,
                sessions[ises].respmat_videome))

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(sessions[ises],radius=50)
    else:
        idx_nearby = np.ones(len(sessions[ises].celldata),dtype=bool)

    for iax, area in enumerate(arealabels):
        idx_N           = np.where(np.all((sessions[ises].celldata['arealabel']==area,
                                        sessions[ises].celldata['tuning_var']>0.01,	
                                        idx_nearby,
                                        # sessions[ises].celldata['noise_level']<20,	
                                        ),axis=0))[0]
        if len(idx_N)<nsampleneurons:
            continue
        for istim, stim in enumerate(stimConds):
            # idx_T           = sessions[ises].trialdata['stimCond']==stim
            idx_T           = (sessions[ises].trialdata['stimCond']==stim) & (sessions[ises].trialdata['TrialNumber']>10)

            for imf in range(nmodelfits):
                idx_N_sub       = np.random.choice(idx_N,nsampleneurons,replace=False)
                X               = sessions[ises].respmat[np.ix_(idx_N_sub,idx_T)].T
                X               = zscore(X,axis=0)  #Z score activity for each neuron
                
                for ivar in range(NS):
                    Y               = S[ivar,idx_T]
                    Y               = zscore(Y)
                    # print(find_optimal_lambda(X,Y,model_name='Ridge',kfold=kfold))

                    dec_perf[iax,ivar,ises,istim,imf],_,_,_ = my_decoder_wrapper(X,Y,model_name='Ridge',kfold=kfold,lam=lam,subtract_shuffle=True,
                                scoring_type='r2_score',norm_out=False)

#%% Show decoding results per stimulus condition, for vars ori and speed averaging over modelfits, sessions
fig, axes = plt.subplots(1, NS, figsize=(9,3), sharey=False,sharex=True)
for ivar, (ax, varlabel) in enumerate(zip(axes, slabels)):
    for ial, al in enumerate(arealabels): 
    # for istim, stim in enumerate(stimConds):
        mean_decoding = np.nanmean(dec_perf[ial, ivar, :, :, :], axis=(0, 2))
        ax.plot(mean_decoding, label=al, linewidth=2,color=get_clr_area_labeled([al]))
    ax.set_xlabel('Stimulus condition')
    ax.set_ylabel('Decoding R2')
    ax.set_title('Decoding %s' % varlabel)
    ax.legend()
    ax.set_ylim(-0.1,0.6)
    ax.set_xticks(range(nstims), stimLabels, rotation=45,fontsize=6)
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'Decoding_%s_per_stimcond.png' % var), format='png')
my_savefig(fig,savedir,'Decoding_Jitter_DiffStimuli_V1PMlabeled_%dsessions' % nSessions,formats=['png'])

#%% Debug code: problem with run speed decoding in one session:

# lam = 100
# ises = 5
# kfold = 3

# # Define neural data parameters
# S           = np.vstack((sessions[ises].trialdata['deltaOrientation'],
# sessions[ises].trialdata['logdeltaSpeed'],
# sessions[ises].respmat_runspeed,
# sessions[ises].respmat_videome))

# area = 'V1unl'

# idx_N           = np.where(np.all((sessions[ises].celldata['arealabel']==area,
#                     sessions[ises].celldata['noise_level']<100,	
#                     ),axis=0))[0]
# stim = 0

# idx_T           = sessions[ises].trialdata['stimCond']==stim
# idx_T           = (sessions[ises].trialdata['stimCond']==stim) & (sessions[ises].trialdata['TrialNumber']>50)

# # print(np.sum(S[2,idx_T]>2)/np.sum(idx_T))

# idx_N_sub       = np.random.choice(idx_N,nsampleneurons,replace=False)
# X               = sessions[ises].respmat[np.ix_(idx_N_sub,idx_T)].T
# X               = zscore(X,axis=0)  #Z score activity for each neuron

# ivar            = 2
# Y               = S[ivar,idx_T]
# Y               = zscore(Y)

# dectemp,_,_,_ = my_decoder_wrapper(X,Y,model_name='Ridge',kfold=kfold,lam=lam,subtract_shuffle=False,
#             scoring_type='r2_score',norm_out=False)
# print(dectemp)

# stim = 1
# idx_T           = sessions[ises].trialdata['stimCond']==stim
# idx_T           = (sessions[ises].trialdata['stimCond']==stim) & (sessions[ises].trialdata['TrialNumber']>50)

# idx_N_sub       = np.random.choice(idx_N,nsampleneurons,replace=False)
# X               = sessions[ises].respmat[np.ix_(idx_N_sub,idx_T)].T
# X               = zscore(X,axis=0)  #Z score activity for each neuron

# # print(np.sum(S[2,idx_T]>5)/np.sum(idx_T))
# ivar = 2
# Y               = S[ivar,idx_T]
# Y               = zscore(Y)

# dectemp,_,_,_ = my_decoder_wrapper(X,Y,model_name='Ridge',kfold=kfold,lam=lam,subtract_shuffle=False,
#             scoring_type='r2_score',norm_out=False)
# print(dectemp)

#%% Plot R2 scores for each arealabel and session
fig, axes = plt.subplots(1, NS, figsize=(9,3), sharey=False,sharex=True)
for iax, (ax, varlabel) in enumerate(zip(axes, slabels)):
    datatoplot = copy.deepcopy(dec_perf[:, iax, :, :, :])
    mean_R2_scores = np.nanmean(datatoplot, axis=(2, 3))  # average over modelfits and stim conds
    mean_R2_scores[mean_R2_scores<0.02] = np.nan
    for ial, al in enumerate(arealabels): 
        for ises in range(nSessions):
            ax.plot(ial, mean_R2_scores[ial, ises], 'o', color=get_clr_area_labeled([al]), alpha=0.8)
        ax.plot(np.arange(narealabels), mean_R2_scores,'-', color='k', linewidth=0.2)
        ax.plot(np.arange(narealabels)+0.25, np.nanmean(mean_R2_scores, axis=1), 'o-', color='k', linewidth=2, label='Mean')
    
    stats,pval = ttest_rel(mean_R2_scores[0,:], mean_R2_scores[1,:], nan_policy='omit')
    add_stat_annotation(ax, 0,1, np.nanmax(mean_R2_scores[:1,:]+0.1), pval, h=0.01)

    stats,pval = ttest_rel(mean_R2_scores[0,:], mean_R2_scores[2,:], nan_policy='omit')
    add_stat_annotation(ax, 0,2, np.nanmax(mean_R2_scores[:1,:]+0.2), pval, h=0.01)

    stats,pval = ttest_rel(mean_R2_scores[2,:], mean_R2_scores[3,:], nan_policy='omit')
    add_stat_annotation(ax, 2,3, np.nanmax(mean_R2_scores[2:,:]+0.1), pval, h=0.01)
    
    stats,pval = ttest_rel(mean_R2_scores[1,:], mean_R2_scores[3,:], nan_policy='omit')
    add_stat_annotation(ax, 1,3, np.nanmax(mean_R2_scores[:1,:]+0.3), pval, h=0.01)
    
    ax.set_title(varlabel)
    ax.set_xlabel('Populations')
    ax.set_xticks(range(narealabels),arealabels)
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
axes[0].set_ylabel('Decoding (R2) \n(over shuffle)')
sns.despine(top=True, right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'Decoding_Jitter_OriSpeed_V1PMlabeled_%dsessions' % nSessions,formats=['png'])
# my_savefig(fig,savedir,'Decoding_Jitter_OriSpeed_V1PMlabeled_%dsessions' % nSessions,formats=['png'])


#%% Decoding with regression of noise around each center stimulus 
# for all sessions and different neural populations
nmodelfits      = 5
kfold           = 3
lam             = 100

arealabels      = ['V1', 'PM']
narealabels     = len(arealabels)
stimConds       = np.unique(sessions[ises].trialdata['stimCond'])
nstims          = len(stimConds)
NS              = 4

dec_perf = np.full((NS,nSessions,nstims,nmodelfits),np.nan)

celldata    = pd.concat([sessions[ises].celldata for ises in range(nSessions)])
N           = len(celldata)
weights     = np.full((N,NS,nstims,nmodelfits),np.nan)

for ises in range(nSessions):
    # Define neural data parameters
    S           = np.vstack((sessions[ises].trialdata['deltaOrientation'],
                sessions[ises].trialdata['logdeltaSpeed'],
                sessions[ises].respmat_runspeed,
                sessions[ises].respmat_videome))

    idx_N           = np.ones(sessions[ises].celldata.shape[0],dtype=bool)
    # for iax, area in enumerate(arealabels):
    idx_N           = np.where(np.all((sessions[ises].celldata['noise_level']<100,	
                                    sessions[ises].celldata['tuning_var']>0.01,
                                    ),axis=0))[0]
    idx_N_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N])

    for istim, stim in enumerate(stimConds):
        # idx_T           =sessions[ises].trialdata['stimCond']==stim
        idx_T           = (sessions[ises].trialdata['stimCond']==stim) & (sessions[ises].trialdata['TrialNumber']>10)

        for imf in range(nmodelfits):
            X               = sessions[ises].respmat[np.ix_(idx_N,idx_T)].T
            X               = zscore(X,axis=0)  #Z score activity for each neuron
            
            for ivar in range(NS):
                Y               = S[ivar,idx_T]
                Y               = zscore(Y)
                # print(find_optimal_lambda(X,Y,model_name='Ridge',kfold=kfold))

                dec_perf[ivar,ises,istim,imf],tempweights,_,_ = my_decoder_wrapper(X,Y,model_name='Ridge',kfold=kfold,lam=lam,subtract_shuffle=False,
                            scoring_type='r2_score',norm_out=False)
                weights[idx_N_ses,ivar,istim,imf] = tempweights

#%% Show the weights of the neurons:
binlim          = 0.05
binres          = 0.001
arealabels      = ['V1unl', 'PMunl', 'V1lab', 'PMlab']
fig, axes = plt.subplots(1,NS,figsize=[5*NS,NS],sharex=True,sharey=True)
for iY,slabel in enumerate(slabels):
    ax = axes[iY]
    for ial,al in enumerate(arealabels):
        idx_N           = np.where(np.all((celldata['arealabel']==al,
                                        # celldata['noise_level']<20,	
                                        # celldata['meanF']<500,	
                                        ),axis=0))[0]
        ax.hist(weights[idx_N,iY,:,:].flatten(),bins=np.arange(-binlim,binlim,binres),alpha=0.5,density=True,
                histtype='stepfilled',fill=False,label=area,edgecolor=get_clr_area_labeled([al]))
    ax.set_title(slabel)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Frequency')
    ax.legend(arealabels,fontsize=10,frameon=False)
fig.tight_layout()

#%% 
for svar in range(NS):
    celldata['weight_%s' % slabels[svar]] = np.nanmean(np.abs(weights[:,svar,:,:]),axis=(1,2))
celldata['weight_all'] = np.nanmean(np.abs(weights),axis=(1,2,3))

    # fig.savefig(os.path.join(savedir,'WeightHistogram_%s_%s' % (slabel,area) + '.png'), format = 'png')

#%% plot the correlation 
fields = ['noise_level','event_rate','meanF','meanF_chan2','tuning_var']
fig, axes = plt.subplots(1,len(fields),figsize=(len(fields)*3,3))
weightfield = 'weight_videoME'
weightfield = 'weight_Ori'
# weightfield = 'weight_all'
for ifield,field in enumerate(fields):
    ax = axes[ifield]
    ax.scatter(celldata[field],celldata[weightfield],s=2,alpha=0.5)
    # ax.scatter(celldata[field],celldata['weight_all'],s=5,alpha=0.5)
    ax.set_xlabel(field)
    ax.set_ylabel('Weight')
    print(celldata[[field,weightfield]].corr().to_numpy()[0,1])
    
fig.tight_layout()


#%% Show the weights of the neurons:
weightlim       = 0.05
arealabels      = ['V1unl', 'PMunl', 'V1lab', 'PMlab']
fig, axes = plt.subplots(1,NS,figsize=[3*NS,NS],sharex=True,sharey=True)
for iY,slabel in enumerate(slabels):
    ax = axes[iY]
    for ial,al in enumerate(arealabels):
        idx_N           = np.where(np.all((celldata['arealabel']==al,
                                        celldata['noise_level']<20,	
                                        ),axis=0))[0]
        ax.hist(weights[idx_N,iY,:,:].flatten(),bins=np.arange(-weightlim,weightlim,0.001),alpha=0.5,density=True,
                histtype='stepfilled',fill=False,label=area,edgecolor=get_clr_area_labeled([al]))
    ax.set_title(slabel)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Frequency')
    ax.legend()
fig.tight_layout()
    # fig.savefig(os.path.join(savedir,'WeightHistogram_%s_%s' % (slabel,area) + '.png'), format = 'png')












#%% Are the weights correlated to the single-neuron correlation between the variables
# Should be, positive control: 
fig,axes = plt.subplots(1,NS,figsize=[8,2])
for iSvar in range(NS):
    ax = axes[iSvar]
    for iO in range(oris.shape[0]):
        for iS in range(speeds.shape[0]):
            w = weights_avg[iSvar,:,iO,iS]
            c = corrmat[:,iSvar,iO,iS]
            ax.scatter(w,c,s=10,alpha=0.2,marker='.',color=colors[iO,iS])
    ax.set_xlabel('Weight')
    ax.set_ylabel('Correlation')
    ax.set_title(slabels[iSvar])
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'WeightVsCorrelation_Svars_%s' % sessions[ises].sessiondata['session_id'][0] + '.pdf'), format = 'pdf')
fig.savefig(os.path.join(savedir,'WeightVsCorrelation_Svars_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% Correlation matrix of the weights between each of the Slabel variables for each of the stimulus conditions
weights_avg = np.nanmean(weights,axis=4) #take average across kfold

corrmat_Svars = np.zeros((NS,NS,oris.shape[0],speeds.shape[0]))
for iO in range(oris.shape[0]):
    for iS in range(speeds.shape[0]):
        # corrmat_Svars[:,:,iO,iS] = np.corrcoef(weights_avg[:,:,iO,iS])
        corrmat_Svars[:,:,iO,iS] = np.corrcoef(corrmat[:,:,iO,iS].T)

corrmat_Svars_avg = np.nanmean(corrmat_Svars,axis=(2,3))

fig,ax = plt.subplots(1,1,figsize=(3,3))
sns.heatmap(corrmat_Svars_avg,vmin=-1,vmax=1,cmap='bwr',ax=ax,xticklabels=slabels,yticklabels=slabels,
            annot=True,annot_kws={"fontsize":8},
            cbar_kws={'label': 'Correlation', 'ticks': [-1, 0, 1]})
plt.title('Correlation of correlations \n (stimulus averaged) ')
# plt.title('Regression weight correlation \n (stimulus averaged) ')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Correlation_weights_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'Correlation_singleneuroncorrelations_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% Make a plot where the absolute weight from the regression is averaged across neurons from each area and split by redcell (0 or 1). 
# Averaged across stimulus conditions, but show the wegith
 
#%% Compare weights across areas and redcells
# Plot unlabeled and labeled neurons for each area are next to each other,
# Perform an independent samples t-test between the two populations (labeled and unlabeled)
# If significant, statannotations to display an asterisk between the two groups.

clrs_labeled = get_clr_labeled()

fig,axes = plt.subplots(1,NS,figsize=(NS*2,2),sharex=True,sharey=True)
for iSvar in range(NS):
    ax = axes[iSvar]
    df = pd.DataFrame(sessions[ises].celldata[['roi_name','redcell']])
    df['weight'] = np.nanmean(np.abs(weights_avg[iSvar,:,:,:]),axis=(1,2))
    # df['weight'] = np.nanmax(np.abs(weights_avg[iSvar,:,:,:]),axis=(1,2))

    sns.barplot(data = df,x='roi_name',y='weight',hue='redcell',errorbar='se',palette=clrs_labeled,ax=ax)
    for area in areas:
        df_area = df[df['roi_name']==area]
        ttest_ind(df_area[df_area['redcell']==0]['weight'],df_area[df_area['redcell']==1]['weight'])
        if ttest_ind(df_area[df_area['redcell']==0]['weight'],df_area[df_area['redcell']==1]['weight'])[1] < 0.05:
            add_stat_annotation(ax, data=df_area, x='roi_name', y='weight', hue='redcell',
                box_pairs=[(area, area)], test='t-test_ind', text_format='star',
                pvalue_thresholds=[[0.05,'*']],
                perform_stat_test=False,
                verbose=0)
    ax.set_title(slabels[iSvar])
    ax.set_xticks([0,1],labels=areas)
    ax.set_xlabel('Area')
    ax.set_ylabel('Absolute weight')
    if iSvar==0: 
        ax.legend(frameon=False,loc='upper right',fontsize=7)
    else: ax.get_legend().remove()
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Weights_Area_Redcell_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

# sessions[ises].trialdata['deltaSpeed']

#%% ## Regression of behavioral activity onto neural data: 

# idx_tuned = np.logical_and(np.logical_or(corrmat[:,0,0,0]>np.percentile(corrmat[:,0,0,0].flatten(),95),
#     corrmat[:,0,0,0]<np.percentile(corrmat[:,0,0,0].flatten(),5)),   
#                            sessions[ises].celldata['tuning']>0.4)

idx_V1 = np.where(sessions[ises].celldata['roi_name']=='V1')[0]
idx_PM = np.where(sessions[ises].celldata['roi_name']=='PM')[0]

# idx_tuned = sessions[ises].celldata['tuning']>0.3
idx_tuned = sessions[ises].celldata['tuning_var']>-0.5

# idx_tuned = np.any(resp_selec.reshape(N,-1)>0.2,axis=1)
# idx_tuned = np.logical_or(corrmat[:,0,0,0]>np.percentile(corrmat[:,0,0,0].flatten(),95),
#     corrmat[:,0,0,0]<np.percentile(corrmat[:,0,0,0].flatten(),5))

A1 = sessions[ises].respmat[idx_tuned,:]
A1 = sessions[ises].respmat[idx_V1,:]

N1 = np.shape(A1)[0]

kfold       = 5
R2_Y        = np.empty((noris,nspeeds)) #variance explained across all neural data
R2_Y_mat    = np.empty((N1,noris,nspeeds)) #variance for each neuron separately
R2_X_mat    = np.empty((NS,noris,nspeeds)) #variance explained by each predictor separately
weights     = np.empty((N1,NS,noris,nspeeds,kfold)) 

sc = StandardScaler(with_mean=True,with_std=False)
# sc = StandardScaler(with_mean=True,with_std=True)

dimPCA = 5 #no PCA = 0

for iO, ori in enumerate(oris): 
    for iS, speed in enumerate(speeds):     

        idx     = np.intersect1d(ori_ind[iO],speed_ind[iS])
        X       = zscore(S[:,idx],axis=1).T # z-score to be able to interpret weights in uniform scale
        # X = S[:,idx].T
        # Y = zscore(A1[:,idx],axis=1).T
        # Y = A1[:,idx].T

        Y       = sc.fit_transform(A1[:,idx].T)
        
        if dimPCA: 
            pca = PCA(n_components=dimPCA)
            Y_orig = Y.copy()
            Y = pca.fit_transform(Y)

        #Implementing cross validation
        # kf  = KFold(n_splits=kfold, random_state=randomseed,shuffle=True)
        kf  = KFold(n_splits=kfold,shuffle=True)

        model = linear_model.Ridge(alpha=50)  

        Yhat        = np.empty(np.shape(Y))
        Yhat_vars   = np.empty(np.shape(Y) + (NS,))
        for (train_index, test_index),iF in zip(kf.split(X),range(kfold)):
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index,:],Y[test_index,:]
            
            model.fit(X_train,Y_train)

            Yhat[test_index,:]   = model.predict(X_test)
            for iSvar in range(NS):
                Xtemp = np.zeros(np.shape(X_test))
                Xtemp[:,iSvar] = X_test[:,iSvar]
                Yhat_vars[test_index,:,iSvar]   = model.predict(Xtemp)

            # weights[:,:,iO,iS,iF] = model.coef_

        if dimPCA: 
            Yhat = pca.inverse_transform(Yhat)
            Y   = Y_orig.copy()
            # Yhat_vars
            # for iSvar in range(NS):
            #     Yhat_vars[:,:,iSvar]   = pca.inverse_transform(Yhat_vars[:,:,iSvar])
            #     Yhat_vars[:,:,iSvar]   = pca.inverse_transform(Yhat_vars[:,:,iSvar])
            Yhat_vars = np.transpose([pca.inverse_transform(Yhat_vars[:,:,iSvar]) for iSvar in range(NS)],(1,2,0))

        R2_Y[iO,iS]         = r2_score(Y, Yhat)      
        # R2_Y_mat[:,iO,iS]   = r2_score(Y, Yhat, multioutput='raw_values')
        for iSvar in range(NS):
            R2_X_mat[iSvar,iO,iS]         = r2_score(Y, Yhat_vars[:,:,iSvar])

#%% ###################### Plot R2 for different predictors ################################ 
fig, axes   = plt.subplots(1, NS, figsize=[9, 2])
for iSvar in range(NS):
    ax = axes[iSvar]
    oris_m, speeds_m = np.meshgrid(range(oris.shape[0]), range(speeds.shape[0]), indexing='ij')
    ax.pcolor(oris_m, speeds_m, R2_X_mat[iSvar,:,:].squeeze(),vmin=0,vmax=0.05,cmap='hot')
    ax.set_xticks(range(len(oris)),labels=oris)
    ax.set_yticks(range(len(speeds)),labels=speeds)
    
    # sns.heatmap(data=R2_X_mat[iSvar,:,:],vmin=0,vmax=0.05,ax=axes[iSvar])
    ax.set_title(slabels[iSvar])
    ax.set_xticklabels(oris)
    ax.set_yticklabels(speeds)

plt.tight_layout()
# plt.savefig(os.path.join(savedir,'GN_noiseregression','Regress_StoA_Svar_R2' + '.png'), format = 'png')


#%% ### testing with specific ori and speed which showed effect: 
dimPCA = 50
iO = 1
iS = 1
idx     = np.intersect1d(ori_ind[iO],speed_ind[iS])
X       = zscore(S[:,idx],axis=1).T # z-score to be able to interpret weights in uniform scale
A1      = sessions[ises].respmat[idx_V1,:]
Y       = A1[:,idx].T
if dimPCA: 
    pca = PCA(n_components=dimPCA)
    Y_orig = Y.copy()
    Y = pca.fit_transform(Y)

fig, axes = plt.subplots(1, NS, figsize=[15, 3])
proj = (1, 2)
for iSvar in range(NS):
    x = Y[:,proj[0]]                          #get all data points for this ori along first PC or projection pairs
    y = Y[:,proj[1]]                          #get all data points for this ori along first PC or projection pairs
    c = cmap(minmax_scale(S[iSvar,idx], feature_range=(0, 1)))[:,:3]
    sns.scatterplot(x=x, y=y, c=c,ax = axes[iSvar],s=10,legend = False,edgecolor =None)
    axes[iSvar].set_title(slabels[iSvar])
    axes[iSvar].set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
    axes[iSvar].set_ylabel('PC {}'.format(proj[1]+1))

#%% ########### # ############# ############# ############# ############# 
from sklearn.linear_model import RidgeCV

model = linear_model.Ridge(alpha=np.array([0.0001,0.01,1,10,100]))  
# list of alphas to check: 100 values from 0 to 5 with
r_alphas = np.logspace(0, 5, 100)
# initiate the cross validation over alphas
ridge_model = RidgeCV(alphas=r_alphas, scoring='r2',store_cv_values=True)
# fit the model with the best alpha
ridge_model = ridge_model.fit(X_train, Y_train)

ridge_model.alpha_
ridge_model.cv_values_

fig,ax = plt.subplots(1,1,figsize=(6,6))
for iSvar in range(NS):
    sns.kdeplot(corrmat[:,iSvar,:,:].flatten(),ax=ax,color=scolors[iSvar])
plt.legend(slabels)

####################################################################
sc = StandardScaler()

model = PCA(n_components=5)

coefs = np.mean(weights,axis=4) #average over folds

for iO, ori in enumerate(oris): 
    for iS, speed in enumerate(speeds):     
        for iY in range(NS):
            X = A1[:,idx].T

            Y = zscore(S[:,idx],axis=1).T #to be able to interpret weights in uniform scale

            # EV(X,u)
            u = coefs[iY,:,iO,iS]
            u = u[:,np.newaxis]


            model.fit(sc.fit_transform(X))
            model.score
            v = model.components_[0,:]
            G = v @ v.T @ X.T @ X

            G = u @ u.T @ X.T @ X
            TSS = np.trace(X.T @ X)

            RSS = np.trace(G)





model.fit(X)

Xcov = np.cov(X.T)

TSS = np.trace(X.T @ X)

# Get variance explained by singular values
explained_variance_ = (S ** 2) / (n_samples - 1)
total_var = explained_variance_.sum()
explained_variance_ratio_ = explained_variance_ / total_var
singular_values_ = S.copy()  # Store the singular values.


for iO, ori in enumerate(oris): 
    for iS, speed in enumerate(speeds):     
        # sns.heatmap(data=R2_Y_mat[])
        # proj_ori    = X_test @ regr.coef_[0,:].T
        # regr.fit(X.T, y_speed)
        # plt.scatter(proj_ori,Yhat_test[:,0])
        # print(r2_score(y_train, Yhat_train))
        print(r2_score(y_test, Yhat_test,multioutput='raw_values'))

        c = np.mean((cmap1(minmax_scale(y_test['deltaOrientation'], feature_range=(0, 1))),
                     cmap2(minmax_scale(y_test['deltaSpeed'], feature_range=(0, 1)))),axis=0)[:,:3]
        sns.scatterplot(x=Yhat_test[:,0], y=Yhat_test[:,1],c=c,ax = axes[iO,iS],legend = False)

        # c = np.mean((cmap1(minmax_scale(y_train['deltaOrientation'], feature_range=(0, 1))),
        #              cmap2(minmax_scale(y_train['deltaSpeed'], feature_range=(0, 1)))),axis=0)[:,:3]
        # sns.scatterplot(x=Yhat_train[:,0], y=Yhat_train[:,1],color=c,ax = axes[iO,iS],legend = False)

        # sns.scatterplot(x=proj_ori, y=proj_speed,color=c,ax = axes[iO,iS],legend = False)
        axes[iO,iS].set_xlabel('delta Ori')            #give labels to axes
        axes[iO,iS].set_ylabel('delta Speed')            #give labels to axes
        axes[iO,iS].set_title('%d deg - %d deg/s' % (ori,speed))       

sns.despine()
plt.tight_layout()

################### Show noise around center for each condition #################

fig, axes = plt.subplots(3, 3, figsize=[9, 9])
proj    = (0, 1)

for iO, ori in enumerate(oris):                                #plot orientation separately with diff colors
    for iS, speed in enumerate(speeds):     
        
        idx     = np.intersect1d(ori_ind[iO],speed_ind[iS])

        X       = respmat_zsc[:,idx]
        y_ori   = sessions[0].trialdata['deltaOrientation'][idx]
        y_speed = sessions[0].trialdata['deltaSpeed'][idx]

        # regr = linear_model.LinearRegression()  
        regr = linear_model.Ridge(alpha=0.001)  
        regr.fit(X.T, y_ori)
        proj_ori    = X.T @ regr.coef_
        regr.fit(X.T, y_speed)
        proj_speed  = X.T @ regr.coef_

        # X = pd.DataFrame({'deltaOrientation': sessions[0].trialdata['deltaOrientation'][idx],
        #         'deltaSpeed': sessions[0].trialdata['deltaSpeed'][idx],
        #         'runSpeed': respmat_runspeed[idx]})
        
        # Y = respmat_zsc[:,idx].T

        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=40)

        # regr.fit(X_train,y_train)
        # Yhat_test = regr.predict(X_test)


        # c = np.mean((cmap1(minmax_scale(proj_ori, feature_range=(0, 1))),cmap2(minmax_scale(proj_speed, feature_range=(0, 1)))),axis=0)[:,:3]
        c = np.mean((cmap1(minmax_scale(y_ori, feature_range=(0, 1))),cmap2(minmax_scale(y_speed, feature_range=(0, 1)))),axis=0)[:,:3]

        sns.scatterplot(x=proj_ori, y=proj_speed,c=c,ax = axes[iO,iS],legend = False)
        # sns.scatterplot(x=proj_ori, y=proj_speed,color=c,ax = axes[iO,iS],legend = False)
        axes[iO,iS].set_xlabel('delta Ori')            #give labels to axes
        axes[iO,iS].set_ylabel('delta Speed')            #give labels to axes
        axes[iO,iS].set_title('%d deg - %d deg/s' % (ori,speed))       

sns.despine()
plt.tight_layout()

