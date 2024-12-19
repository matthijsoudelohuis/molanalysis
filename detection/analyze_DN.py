# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn as sns
try:
    os.chdir('t:\\Python\\molanalysis\\')
except:
    os.chdir('e:\\Python\\molanalysis\\')

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from sklearn.decomposition import PCA
from scipy.stats import zscore, pearsonr
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
from utils.plotting_style import * #get all the fixed color schemes
from scipy.signal import medfilt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from rastermap import Rastermap, utils

savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Neural - DN regression\\'

#%% ################################################
session_list        = np.array([['LPE11622','2024_02_23']])
# session_list        = np.array([['LPE10884','2024_01_12']])
sessions,nSessions  = load_sessions(protocol = 'DN',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=False, calciumversion='dF')

sesidx      = 0
randomseed  = 5

sessions[sesidx].behaviordata['runspeed'] = medfilt(sessions[sesidx].behaviordata['runspeed'] , kernel_size=51)

##############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 3     #post s
binsize     = 0.2   #temporal binsize in s

# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], t_pre, t_post, binsize,method='binmean')

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
#                                  t_pre, t_post, binsize,method='interp_lin')
# [N,K,T]         = np.shape(tensor) #get dimensions of tensor
# respmat         = tensor[:,:,np.logical_and(t_axis > 0,t_axis < 1)].mean(axis=2)

#Alternative method, much faster:
sessions[sesidx].respmat         = compute_respmat(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tStart'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)
[N,K]           = np.shape(sessions[sesidx].respmat) #get dimensions of response matrix

#hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
temp = pd.DataFrame(np.reshape(np.array(sessions[0].behaviordata['runspeed']),(len(sessions[0].behaviordata['runspeed']),1)))
sessions[sesidx].respmat_runspeed = compute_respmat(temp, sessions[0].behaviordata['ts'], sessions[0].trialdata['tStart'],
                                   t_resp_start=0,t_resp_stop=2,method='mean')
sessions[sesidx].respmat_runspeed = np.squeeze(sessions[sesidx].respmat_runspeed)

#hacky way to create dataframe of the mean motion energy with F x 1 with F number of samples:
temp = pd.DataFrame(np.reshape(np.array(sessions[0].videodata['motionenergy']),(len(sessions[0].videodata['motionenergy']),1)))
sessions[sesidx].respmat_motionenergy = compute_respmat(temp, sessions[0].videodata['timestamps'], sessions[0].trialdata['tStart'],
                                   t_resp_start=0,t_resp_stop=2,method='mean')
sessions[sesidx].respmat_motionenergy = np.squeeze(sessions[sesidx].respmat_motionenergy)


# #### Check if values make sense:
fig,ax = plt.subplots(1,1,figsize=(6,6))
sns.histplot(np.nanmin(sessions[sesidx].respmat,axis=1),ax=ax)
sns.histplot(np.nanmax(sessions[sesidx].respmat,axis=1))

#############################################################################
idx_N = sessions[sesidx].trialdata['signal']
signals         = np.sort(pd.Series.unique(sessions[sesidx].trialdata['signal']))
# speeds          = np.sort(pd.Series.unique(sessions[sesidx].trialdata['centerSpeed']))
# noris           = len(oris) 
nsignals         = len(signals)


# oris            = np.sort(pd.Series.unique(sessions[sesidx].trialdata['centerOrientation']))
# speeds          = np.sort(pd.Series.unique(sessions[sesidx].trialdata['centerSpeed']))
# noris           = len(oris) 
# nspeeds         = len(speeds)

clrs,labels     = get_clr_gratingnoise_stimuli(oris,speeds)
# clrs,labels     = get_clr_signalnoise_stimuli(oris,speeds)

### Mean response per condition:
resp_mean       = np.empty([N,noris,nspeeds])

## Compute residual response:
resp_res = sessions[sesidx].respmat.copy()
for iO,ori in enumerate(oris):
    for iS,speed in enumerate(speeds):
        
        idx_trial = np.logical_and(sessions[0].trialdata['centerOrientation']==ori,sessions[0].trialdata['centerSpeed']==speed)
        tempmean = np.nanmean(sessions[sesidx].respmat[:,idx_trial],axis=1)
        resp_res[:,idx_trial] -= tempmean[:,np.newaxis]

##### Compute tuning measure: how much of the variance is due to mean response per stimulus category:
sessions[sesidx].celldata['tuning'] = 1 - np.var(resp_res,axis=1) / np.var(sessions[sesidx].respmat,axis=1)
fig,ax = plt.subplots(1,1,figsize=(6,6))
sns.histplot(sessions[sesidx].celldata['tuning'],ax=ax)
fig.savefig(os.path.join(savedir,'Tuning_distribution' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

tuning = sessions[sesidx].celldata['tuning']

##### Compute selectivity measure: how selective is the mean response to one category versus the rest:
resp_selec  = np.empty([N,noris,nspeeds])
for iO,ori in enumerate(oris):
    for iS,speed in enumerate(speeds):
        resp_selec[:,iO,iS] = resp_mean[:,iO,iS] / np.sum(resp_mean[:,:,:].reshape(N,-1),axis=1)
assert(np.allclose(np.sum(resp_selec.reshape(N,-1),axis=1),1)), 'selectivity measure gone wrong'

##### Show the most beautifully tuned cells:
fig = show_excerpt_traces_gratings(sessions[sesidx],example_cells=np.where(tuning>np.percentile(tuning,95))[0])[0]
fig.savefig(os.path.join(savedir,'ExampleTraces_TunedOnly_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

## Get the preferred orientation and speed:
prefcond    = np.argmax(resp_mean.reshape(N,-1),axis=1)
prefori     = oris[np.mod(prefcond,3)]
prefspeed   = speeds[np.floor(prefcond / 3).astype(np.int64)]

prefori     = oris[np.floor(prefcond / 3).astype(np.int64)]
prefspeed   = speeds[np.mod(prefcond,3).astype(np.int64)]

sessions[sesidx].celldata['prefcond'] = prefcond
sessions[sesidx].celldata['prefori'] = prefori
sessions[sesidx].celldata['prefspeed'] = prefspeed

##### Show cells tuned to certain orientation and speed to check method:
example_cells=np.where(np.all((prefori==150, prefspeed==200,tuning>np.percentile(tuning,90)),axis=0))[0] 
example_cells=np.where(np.all((prefori==30, prefspeed==12.5,tuning>np.percentile(tuning,90)),axis=0))[0] 
# example_cells=np.where(np.all((prefori==30, prefspeed==12.5,resp_selec[:,0,0]>np.percentile(resp_selec[:,0,0],90)),axis=0))[0] 

fig = show_excerpt_traces_gratings(sessions[sesidx],example_cells=example_cells)[0]
fig.savefig(os.path.join(savedir,'ExampleTraces_TunedCondition_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#### 
# idx_V1 = np.where(sessions[sesidx].celldata['roi_name']=='V1')[0]
# idx_PM = np.where(sessions[sesidx].celldata['roi_name']=='PM')[0]
idx_V1 = sessions[sesidx].celldata['roi_name']=='V1'
idx_PM = sessions[sesidx].celldata['roi_name']=='PM'


### Fraction of neurons with preferred orientations and speeds across areas: 

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,3))
df = sessions[sesidx].celldata[sessions[sesidx].celldata['tuning']>0.5]
sns.histplot(data=sessions[sesidx].celldata,x='prefori',hue='roi_name',ax=ax1,stat='probability')
# sns.histplot(data=df,x='prefori',hue='roi_name',ax=ax1,stat='probability',alpha=0.3)
ax1.set_xticks(oris)
sns.histplot(data=sessions[sesidx].celldata,x='prefspeed',hue='roi_name',ax=ax2,stat='probability')
# sns.histplot(data=df,x='prefspeed',hue='roi_name',ax=ax2,stat='probability',alpha=0.3)#, 'edgecolor':'black', 
ax2.set_xticks(speeds)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Preferred_Stim_Area_Bar_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

### Fraction of neurons with preferred orientations and speeds across areas as a heatmap:

V1_prefconds = np.histogram(prefcond[idx_V1],range(10),density=True)[0].reshape(3,3,)
PM_prefconds = np.histogram(prefcond[idx_PM],range(10),density=True)[0].reshape(3,3,)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,3),sharex=True,sharey=True,)
im1 = ax1.imshow(V1_prefconds,vmin=0,vmax=0.3)
ax1.set_title('V1')
im2 = ax2.imshow(PM_prefconds,vmin=0,vmax=0.3)
ax2.set_title('PM')
plt.colorbar(im1,ax=ax1,location='right')
plt.colorbar(im2,ax=ax2,location='right')
ax1.set_xticks(range(len(speeds)))
ax1.set_xticklabels(speeds)
ax1.set_yticks(range(len(oris)))
ax1.set_yticklabels(oris)
ax1.set_ylabel('Orientations')
ax1.set_xlabel('Speeds')
plt.savefig(os.path.join(savedir,'Preferred_Stim_Area_Heatmap_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

########### PCA on trial-averaged responses ############
######### plot result as scatter by orientation ########

respmat_zsc = zscore(sessions[sesidx].respmat,axis=1) # zscore for each neuron across trial responses

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(respmat_zsc.T).T #fit pca to response matrix (n_samples by n_features)
#dimensionality is now reduced from N by K to ncomp by K

ori_ind         = [np.argwhere(np.array(sessions[sesidx].trialdata['centerOrientation']) == ori)[:, 0] for ori in oris]
speed_ind       = [np.argwhere(np.array(sessions[sesidx].trialdata['centerSpeed']) == speed)[:, 0] for speed in speeds]

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
            x = Xp[proj[0],idx]                          #get all data points for this ori along first PC or projection pairs
            y = Xp[proj[1],idx]                          #get all data points for this ori along first PC or projection pairs

            # x = Xp[proj[0],ori_ind[io]]                          #get all data points for this ori along first PC or projection pairs
            # y = Xp[proj[1],ori_ind[io]]                          #and the second
            # handles.append(ax.scatter(x, y, color=clrs[iO,iS,:], s=sessions[sesidx].respmat_runspeed[idx], alpha=0.8))     #each trial is one dot
            ax.scatter(x, y, color=clrs[iO,iS,:], s=sessions[sesidx].respmat_runspeed[idx], alpha=0.8)    #each trial is one dot
            ax.set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
            ax.set_ylabel('PC {}'.format(proj[1]+1))

axes[2].legend(labels.flatten(),fontsize=8,bbox_to_anchor=(1,1))
sns.despine(fig=fig, top=True, right=True)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'PCA_allStim_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

################### PCA unsupervised display of noise around center for each condition #################
## split into area 1 and area 2:

idx_V1_tuned = np.logical_and(sessions[sesidx].celldata['roi_name']=='V1',sessions[sesidx].celldata['tuning']>0.4)
idx_PM_tuned = np.logical_and(sessions[sesidx].celldata['roi_name']=='PM',sessions[sesidx].celldata['tuning']>0.4)

A1 = sessions[sesidx].respmat[idx_V1_tuned,:]
A2 = sessions[sesidx].respmat[idx_PM_tuned,:]

idx_V1 = np.where(sessions[sesidx].celldata['roi_name']=='V1')[0]
idx_PM = np.where(sessions[sesidx].celldata['roi_name']=='PM')[0]

A1 = sessions[sesidx].respmat[idx_V1,:]
A2 = sessions[sesidx].respmat[idx_PM,:]

# S   = np.vstack((sessions[sesidx].trialdata['deltaOrientation'],
#                sessions[sesidx].trialdata['deltaSpeed'],
#                sessions[sesidx].respmat_runspeed))
# S = np.vstack((S,np.random.randn(1,K)))
# slabels     = ['Ori','Speed','Running','Random']

S   = np.vstack((sessions[sesidx].trialdata['deltaOrientation'],
               sessions[sesidx].trialdata['deltaSpeed'],
               sessions[sesidx].respmat_runspeed,
               sessions[sesidx].respmat_motionenergy))
S = np.vstack((S,np.random.randn(1,K)))
slabels     = ['Ori','Speed','Running','MotionEnergy','Random']

arealabels  = ['V1','PM']

# Define neural data parameters
N1,K        = np.shape(A1)
N2          = np.shape(A2)[0]
NS          = np.shape(S)[0]

cmap = plt.get_cmap('hot')

for iSvar in range(NS):
    fig, axes = plt.subplots(3, 3, figsize=[9, 9])
    proj = (0, 1)
    # proj = (1, 2)
    # proj = (3, 4)
    for iO, ori in enumerate(oris):                                #plot orientation separately with diff colors
        for iS, speed in enumerate(speeds):                       #plot speed separately with diff colors
            idx         = np.intersect1d(ori_ind[iO],speed_ind[iS])
            
            # Xp          = pca.fit_transform(respmat_zsc[:,idx].T).T #fit pca to response matrix (n_samples by n_features)
            Xp          = pca.fit_transform(A1[:,idx].T).T #fit pca to response matrix (n_samples by n_features)
            #dimensionality is now reduced from N by K to ncomp by K

            x = Xp[proj[0],:]                          #get all data points for this ori along first PC or projection pairs
            y = Xp[proj[1],:]                          #get all data points for this ori along first PC or projection pairs
            
            c = cmap(minmax_scale(S[iSvar,idx], feature_range=(0, 1)))[:,:3]

            # tip_rate = tips.eval("tip / total_bill").rename("tip_rate")
            sns.scatterplot(x=x, y=y, c=c,ax = axes[iO,iS],s=10,legend = False,edgecolor =None)
            plt.title(slabels[iSvar])
            # ax.scatter(x, y, color=pal[t], s=25, alpha=0.8)     #each trial is one dot
            # ax.scatter(x, y, color=pal[(iS-1)*len(unique_oris)+iO], s=respmat_runspeed[idx], alpha=0.8)     #each trial is one dot
            axes[iO,iS].set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
            axes[iO,iS].set_ylabel('PC {}'.format(proj[1]+1))
    plt.suptitle(slabels[iSvar],fontsize=15)
    sns.despine(fig=fig, top=True, right=True)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,'PCA' + str(proj) + '_perStim_color' + slabels[iSvar] + '.png'), format = 'png')

#### linear model explaining responses: 
from numpy import linalg

def LM(Y, X, lam=0):
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    Y_hat = X @ B_hat
    return B_hat

def Rss(Y, Y_hat, normed=True):
    """ evaluate (normalized) model error """
    e = Y_hat - Y
    Rss = np.trace(e.T @ e)
    if normed:
        Rss /= Y.shape[0]
    return Rss

def EV(X, u):
    # how much of the variance lies along this dimension?
    # here X is the data matrix (samples x features) and u is the dimension

    return EV

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# from scipy.stats import spearmanr

################### Regression of behavioral variables onto neural data #############
###### First identify single neurons that are correlated with behavioral variables ####

scolors = get_clr_GN_svars(slabels)

corrmat = np.empty((N,NS,noris,nspeeds))
for iN in range(N):
    print(f"\rComputing correlations for neuron  {iN+1} / {N}",end='\r')
    for iSvar in range(NS):
        for iO, ori in enumerate(oris): 
            for iS, speed in enumerate(speeds): 
                idx = np.intersect1d(ori_ind[iO],speed_ind[iS])
                corrmat[iN,iSvar,iO,iS] = np.corrcoef(S[iSvar,idx],sessions[sesidx].respmat[iN,idx])[0,1]   
                # corrmat[iN,iSvar,iO,iS] = spearmanr(S[iSvar,idx],sessions[sesidx].respmat[iN,idx])[0]


##### and plot as density #################################
fig,ax = plt.subplots(1,1,figsize=(4,4))
for iSvar in range(NS):
    sns.kdeplot(corrmat[:,iSvar,:,:].flatten(),ax=ax,color=scolors[iSvar],linewidth=1)
plt.legend(slabels)
plt.xlabel('Correlation (neuron to variable)')
plt.savefig(os.path.join(savedir,'KDE_Correlations_Svars' + '.png'), format = 'png')


### Show the activity fluctuations as a function of variability in the behavioral vars for a couple of neurons:
nexamples = 4
plt.rcParams.update({'font.size': 7})

fig,ax = plt.subplots(NS,nexamples,figsize=(6,6))

for iSvar in range(NS):
    idxN,idxO,idxS  = np.where(np.logical_or(corrmat[:,iSvar,:,:]>np.percentile(corrmat[:,iSvar,:,:].flatten(),98),
                                             corrmat[:,iSvar,:,:]<np.percentile(corrmat[:,iSvar,:,:].flatten(),2)))
    idx_examples    = np.random.choice(idxN,nexamples)
    
    for iN in range(nexamples):
        
        idx_trials = np.intersect1d(ori_ind[idxO[idx_examples[iN]==idxN][0]],speed_ind[idxS[idx_examples[iN]==idxN][0]])
        ax = plt.subplot(NS,nexamples,iN + 1 + iSvar*nexamples)
        ax.scatter(S[iSvar,idx_trials],sessions[sesidx].respmat[idx_examples[iN],idx_trials],
                   s=10,alpha=0.7,marker='.',color=scolors[iSvar])
        # ax.set_xlabel(slabels[iSvar],fontsize=9)
        ax.set_title(slabels[iSvar],fontsize=9)
        # ax.set_xlim([])
        ax.set_ylim(np.percentile(sessions[sesidx].respmat[idx_examples[iN],idx_trials],[3,97]))
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Examplecells_correlated_with_Svars' + '.png'), format = 'png')

###### Is selectivity to particular orientations or speeds related to the correlation 
# of neural activity with variability along one of the sensory/state variables?

fig,ax = plt.subplots(1,NS,figsize=(10,2),sharex=True,sharey=True)

for iSvar in range(NS):
    ax = plt.subplot(1,NS,iSvar+1)
    ax.scatter(resp_selec.flatten(),corrmat[:,iSvar,:,:].flatten(),
                                  s=10,alpha=0.2,marker='.',color=scolors[iSvar])
    ax.set_title(slabels[iSvar])
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('Correlation')
    ax.axvline(1/9,color='k',linestyle=':',linewidth=0.5)
plt.savefig(os.path.join(savedir,'Relationship_selectivity_pairwisecorrelations' + '.png'), format = 'png')

####### Population regression: 

### Regression of neural variables onto behavioral data ####

kfold       = 5
R2_Y_mat    = np.empty((NS,noris,nspeeds))
R2_X1_mat   = np.empty((noris,nspeeds))
weights     = np.empty((NS,N1,noris,nspeeds,kfold)) 

for iO, ori in enumerate(oris): 
    for iS, speed in enumerate(speeds):     
        # ax = axes[iO,iS]
        idx = np.intersect1d(ori_ind[iO],speed_ind[iS])

        X = A1[:,idx].T

        Y = zscore(S[:,idx],axis=1).T #z-score to be able to interpret weights in uniform scale

        #Implementing cross validation
        kf  = KFold(n_splits=kfold, random_state=randomseed,shuffle=True)
        
        model = linear_model.Ridge(alpha=120)  

        Yhat = np.empty(np.shape(Y))
        for (train_index, test_index),iF in zip(kf.split(X),range(kfold)):
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index,:],Y[test_index,:]
            
            model.fit(X_train,Y_train)

            # Yhat_train  = model.predict(X_train)
            Yhat[test_index,:]   = model.predict(X_test)

            weights[:,:,iO,iS,iF] = model.coef_

        for iY in range(NS):
            R2_Y_mat[:,iO,iS] = r2_score(Y, Yhat, multioutput='raw_values')


############ # ############# ############# ############# ############# 
fig, axes   = plt.subplots(1, NS, figsize=[10, 2])
for iY in range(NS):
    sns.heatmap(data=R2_Y_mat[iY,:,:],vmin=0,vmax=1,ax=axes[iY])
    axes[iY].set_title(slabels[iY])
    axes[iY].set_xticklabels(oris)
    axes[iY].set_yticklabels(speeds)

plt.tight_layout()
plt.savefig(os.path.join(savedir,'GN_noiseregression','Regress_Behav_R2' + '.png'), format = 'png')

### Regression of behavioral activity onto neural data: 

# idx_tuned = np.logical_and(np.logical_or(corrmat[:,0,0,0]>np.percentile(corrmat[:,0,0,0].flatten(),95),
#     corrmat[:,0,0,0]<np.percentile(corrmat[:,0,0,0].flatten(),5)),   
#                            sessions[sesidx].celldata['tuning']>0.4)

# idx_tuned = sessions[sesidx].celldata['tuning']>0.3
idx_tuned = sessions[sesidx].celldata['tuning']>-0.5

# idx_tuned = np.any(resp_selec.reshape(N,-1)>0.2,axis=1)
# idx_tuned = np.logical_or(corrmat[:,0,0,0]>np.percentile(corrmat[:,0,0,0].flatten(),95),
#     corrmat[:,0,0,0]<np.percentile(corrmat[:,0,0,0].flatten(),5))

A1 = sessions[sesidx].respmat[idx_tuned,:]
A1 = sessions[sesidx].respmat[idx_V1,:]

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

####################### Plot R2 for different predictors ################################ 
fig, axes   = plt.subplots(1, NS, figsize=[9, 2])
for iSvar in range(NS):
    sns.heatmap(data=R2_X_mat[iSvar,:,:],vmin=0,vmax=0.2,ax=axes[iSvar])
    axes[iSvar].set_title(slabels[iSvar])
    axes[iSvar].set_xticklabels(oris)
    axes[iSvar].set_yticklabels(speeds)

plt.tight_layout()
plt.savefig(os.path.join(savedir,'GN_noiseregression','Regress_StoA_Svar_R2' + '.png'), format = 'png')



#### testing with specific ori and speed which showed effect: 
dimPCA = 50
iO = 1
iS = 1
idx     = np.intersect1d(ori_ind[iO],speed_ind[iS])
X       = zscore(S[:,idx],axis=1).T # z-score to be able to interpret weights in uniform scale
A1      = sessions[sesidx].respmat[idx_V1,:]
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

############ # ############# ############# ############# ############# 
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

plt.xlim([sessions[0].trialdata['tStimStart'][100],sessions[0].trialdata['tStimStart'][120]])
for icomp in range(plotncomps):
    plt.text(x=sessions[0].trialdata['tStimStart'][120],y=icomp+0.25,s='r=%1.3f' %cmat[icomp])

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

