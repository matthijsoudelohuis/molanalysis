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

#Personal libs:
from utils.RRRlib import LM,Rss,EV
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import plot_excerpt
from utils.tuning import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\NoiseRegression\\')
randomseed  = 5

#%% #############################################################################
#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE11998','2024_05_02'], #GN
                                ['LPE12013','2024_05_02']]) #GN
sessions,nSessions   = load_sessions(protocol = 'GN',session_list=session_list)

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['GN'])

#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv')

#%% ######### Compute average response, tuning metrics, and responsive fraction per session ##################
ises = 0
oris, speeds    = [np.unique(sessions[ises].trialdata[col]).astype('int') for col in ('centerOrientation', 'centerSpeed')]
noris           = len(oris) 
nspeeds         = len(speeds)
areas           = np.array(['AL', 'PM', 'RSP', 'V1'], dtype=object)
redcells        = np.array([0, 1])
redcelllabels   = np.array(['unl', 'lab'])
clrs,labels     = get_clr_gratingnoise_stimuli(oris,speeds)

for ises in range(nSessions):
    resp_mean,resp_res      = mean_resp_gn(sessions[ises])
    sessions[ises].celldata['tuning_var'] = compute_tuning_var(resp_mat=sessions[ises].respmat,resp_res=resp_res)
    sessions[ises].celldata['pref_ori'],sessions[ises].celldata['pref_speed'] = get_pref_orispeed(resp_mean,oris,speeds)

    
#%% #### Show example traces:

# THIS PART NEEDS CALCIUM DATA, MOVE TO OTHER SCRIPT, SHOW SOME TRACES FOR ONE SESSION WITH CALCIUMDATA:




# #%% #### Show the most beautifully tuned cells:
# example_cells = np.where(sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],95))[0]
# fig = plot_excerpt(sessions[ises])
# fig = plot_excerpt(sessions[ises],example_cells=example_cells)
# fig.savefig(os.path.join(savedir,'ExampleTraces_TunedOnly_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

# #%% #### Show cells tuned to certain orientation and speed to check method:
# example_cells=np.where(np.all((prefori==150, prefspeed==200,tuning>np.percentile(tuning,90)),axis=0))[0] 
# example_cells=np.where(np.all((prefori==30, prefspeed==12.5,tuning>np.percentile(tuning,90)),axis=0))[0] 
# # example_cells=np.where(np.all((prefori==30, prefspeed==12.5,resp_selec[:,0,0]>np.percentile(resp_selec[:,0,0],90)),axis=0))[0] 

# fig = show_excerpt_traces_gratings(sessions[ises],example_cells=example_cells)[0]
# fig.savefig(os.path.join(savedir,'ExampleTraces_TunedCondition_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

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
plt.savefig(os.path.join(savedir,'GN_Stimuli','PCA_allStim_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% LDA of all trials

colors,labels = get_clr_gratingnoise_stimuli(oris,speeds)

areas = ['V1', 'PM']
plotdim = 1

fig, axes = plt.subplots(1, len(areas), figsize=[12, 4])
for iax, area in enumerate(areas):
    idx     = sessions[ises].celldata['roi_name'] == area
    # X       = respmat_zsc[idx, :]
    X       = sessions[ises].respmat[idx, :]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=randomseed)
    y_pred_ori = np.zeros_like(sessions[ises].trialdata['centerOrientation'])
    y_pred_spd = np.zeros_like(sessions[ises].trialdata['centerSpeed'])
    for train_index, test_index in kf.split(X.T):
        X_train, X_test = X.T[train_index], X.T[test_index]
        y_train, y_test = sessions[ises].trialdata.loc[train_index,['centerOrientation', 'centerSpeed']],sessions[ises].trialdata.loc[test_index,['centerOrientation', 'centerSpeed']]      
          
        lda_ori = LinearDiscriminantAnalysis()
        lda_ori.fit(X_train, y_train['centerOrientation'])
        y_pred_ori[test_index] = lda_ori.predict(X_test)

        lda_spd = LinearDiscriminantAnalysis()
        lda_spd.fit(X_train, y_train['centerSpeed'])
        y_pred_spd[test_index] = lda_spd.predict(X_test)

    for iOri, ori in enumerate(oris):
        for iSpd, spd in enumerate(speeds):
            idx = np.logical_and(sessions[ises].trialdata['centerOrientation'] == ori, sessions[ises].trialdata['centerSpeed'] == spd)
            axes[iax].scatter(lda_ori.transform(X.T)[idx,plotdim], lda_spd.transform(X.T)[idx,plotdim], color=colors[iOri,iSpd,:], marker='o', alpha=0.5)
    axes[iax].set_title(area)
    axes[iax].set_xlabel('LDA %d Ori' % plotdim)
    axes[iax].set_ylabel('LDA %d Speed' % plotdim)
    axes[iax].legend(labels.flatten(),fontsize=8,bbox_to_anchor=(1,1))
sns.despine(fig=fig, top=True, right=True)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'LDA_ori_speed_allStim_dim' + str(plotdim) + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

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

#%% ###### Population regression: 

# ## split into area 1 and area 2:
# tuning_thr      = 0.05

# # idx_V1_tuned = np.logical_and(sessions[ises].celldata['roi_name']=='V1',sessions[ises].celldata['tuning_var']>tuning_thr)
# # idx_PM_tuned = np.logical_and(sessions[ises].celldata['roi_name']=='PM',sessions[ises].celldata['tuning_var']>tuning_thr)

# # A1 = sessions[ises].respmat[idx_V1_tuned,:]
# # A2 = sessions[ises].respmat[idx_PM_tuned,:]

# # idx_V1 = np.where(sessions[ises].celldata['roi_name']=='V1')[0]
# # idx_PM = np.where(sessions[ises].celldata['roi_name']=='PM')[0]

# # A1 = sessions[ises].respmat[idx_V1,:]
# # A2 = sessions[ises].respmat[idx_PM,:]

# # Define neural data parameters
# N1,K        = np.shape(A1)
# N2          = np.shape(A2)[0]
# arealabels  = ['V1','PM']

### Regression of neural variables onto behavioral data ####

areas       = ['V1', 'PM']
# areas       = ['V1', 'PM', 'RSP', 'AL']
nareas      = len(areas)
kfold       = 5
N,K         = np.shape(sessions[ises].respmat)

R2_Y_mat    = np.empty((NS,nareas,noris,nspeeds))
weights     = np.full((NS,N,noris,nspeeds,kfold),np.nan) 
lambda_reg   = 10

from sklearn.linear_model import RidgeCV

# lambda_values = 10**np.linspace(10,-2,100)*0.5

lambda_values = 10**np.linspace(10,-2,50)*0.5

lambda_reg = []

for iarea, area in enumerate(areas):
    idx_area     = sessions[ises].celldata['roi_name'] == area
    for iO, ori in enumerate(oris): 
        for iS, speed in enumerate(speeds):     
            idx_trials = np.intersect1d(ori_ind[iO],speed_ind[iS])
            X = sessions[ises].respmat[np.ix_(idx_area,idx_trials)].T #z-score activity for each neuron across these trials
            Y = zscore(S[:,idx_trials],axis=1).T #z-score to be able to interpret weights in uniform scale

            ridgecv = RidgeCV(alphas = lambda_values, scoring = "neg_mean_squared_error", cv = 10)
            ridgecv.fit(X, Y)
            lambda_reg.append(ridgecv.alpha_)

lambda_reg = np.mean(lambda_reg)

for iarea, area in enumerate(areas):
    idx_area     = sessions[ises].celldata['roi_name'] == area
    for iO, ori in enumerate(oris): 
        for iS, speed in enumerate(speeds):     
            idx_trials = np.intersect1d(ori_ind[iO],speed_ind[iS])
            # X = sessions[ises].respmat[np.ix_(idx_area,idx_trials)].T
            X = sessions[ises].respmat[np.ix_(idx_area,idx_trials)].T #z-score activity for each neuron across these trials
            # X = zscore(sessions[ises].respmat[np.ix_(idx_area,idx_trials)],axis=1).T #z-score activity for each neuron across these trials
            Y = zscore(S[:,idx_trials],axis=1).T #z-score to be able to interpret weights in uniform scale

            #Implementing cross validation
            kf      = KFold(n_splits=kfold, random_state=randomseed,shuffle=True)
            model   = linear_model.Ridge(alpha=lambda_reg)  

            Yhat    = np.empty(np.shape(Y))
            for (train_index, test_index),iF in zip(kf.split(X),range(kfold)):
                X_train , X_test = X[train_index,:],X[test_index,:]
                Y_train , Y_test = Y[train_index,:],Y[test_index,:]
                
                model.fit(X_train,Y_train)

                # Yhat_train  = model.predict(X_train)
                Yhat[test_index,:]   = model.predict(X_test)

                weights[:,idx_area,iO,iS,iF] = model.coef_

            for iY in range(NS):
                R2_Y_mat[:,iarea,iO,iS] = r2_score(Y, Yhat, multioutput='raw_values')

#%% ########### # ############# ############# ############# ############# 
fig, axes   = plt.subplots(nareas, NS, figsize=[1.5*NS, 1.5*nareas],sharex=True,sharey=True)
for iY,slabel in enumerate(slabels):
    for iarea,area in enumerate(areas):
        ax = axes[iarea,iY]
        oris_m, speeds_m = np.meshgrid(range(oris.shape[0]), range(speeds.shape[0]), indexing='ij')
        im = ax.pcolor(oris_m, speeds_m, R2_Y_mat[iY,iarea,:,:].squeeze(),vmin=0,vmax=1,cmap='hot',linewidth=0.25,edgecolor='k')
        ax.set_xticks(range(len(oris)),labels=oris)
        ax.set_yticks(range(len(speeds)),labels=speeds)
        ax.set_xlabel('Orientation (deg)')
        ax.set_ylabel('Speed (deg/s)')
        ax.set_title(area + ' - ' + slabel)
        ax.set_xticklabels(oris)
        ax.set_yticklabels(speeds)

fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, orientation='vertical',label='R2')
plt.savefig(os.path.join(savedir,'Regress_Neural_onto_Behav_R2_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
#%% 
weights_avg = np.nanmean(weights,axis=4) #take average across kfold

fig,ax = plt.subplots(1,1,figsize=[3,3])
ax.hist(weights_avg.flatten(),bins=100,color='k')
ax.set_ylabel('Count')
fig.savefig(os.path.join(savedir,'Hist_Weights_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

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

################## PCA on full session neural data and correlate with running speed

X           = zscore(sessions[0].calciumdata,axis=0)

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(X) #fit pca to response matrix (n_samples by n_features)
#dimensionality is now reduced from time by N to time by ncomp


## Get interpolated values for behavioral variables at imaging frame rate:
runspeed_F  = np.interp(x=sessions[0].ts_F,xp=sessions[0].behaviordata['ts'],
                        fp=sessions[0].behaviordata['runspeed'])

plotncomps  = 5
Xp_norm     = preprocessing.MinMaxScaler().fit_transform(Xp)
Rs_norm     = preprocessing.MinMaxScaler().fit_transform(runspeed_F.reshape(-1,1))

cmat = np.empty((plotncomps))
for icomp in range(plotncomps):
    cmat[icomp] = pearsonr(x=runspeed_F,y=Xp_norm[:,icomp])[0]

plt.figure()
for icomp in range(plotncomps):
    sns.lineplot(x=sessions[0].ts_F,y=Xp_norm[:,icomp]+icomp,linewidth=0.5)
sns.lineplot(x=sessions[0].ts_F,y=Rs_norm.reshape(-1)+plotncomps,linewidth=0.5,color='k')

plt.xlim([sessions[0].trialdata['tOnset'][500],sessions[0].trialdata['tOnset'][800]])
for icomp in range(plotncomps):
    plt.text(x=sessions[0].trialdata['tOnset'][700],y=icomp+0.25,s='r=%1.3f' %cmat[icomp])

plt.ylim([0,plotncomps+1])

########################################


# ##############################
# # PCA on trial-concatenated matrix:
# # Reorder such that tensor is N by K x T (not K by N by T)
# # then reshape to N by KxT (each row is now the activity of all trials over time concatenated for one neuron)

# mat_zsc     = tensor.transpose((1,0,2)).reshape(N,K*T,order='F') 
# mat_zsc     = zscore(mat_zsc,axis=4)

# pca               = PCA(n_components=100) #construct PCA object with specified number of components
# Xp                = pca.fit_transform(mat_zsc) #fit pca to response matrix

# # [U,S,Vt]          = pca._fit_full(mat_zsc,100) #fit pca to response matrix

# # [U,S,Vt]          = pca._fit_truncated(mat_zsc,100,"arpack") #fit pca to response matrix

# plt.figure()
# sns.lineplot(data=pca.explained_variance_ratio_)
# plt.xlim([-1,100])
# plt.ylim([0,0.15])

