# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Import packages
import math
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

os.chdir('e:\\Python\\molanalysis\\')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.psth import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.plot_lib import *
from utils.regress_lib import *
from detection.plot_neural_activity_lib import *

from utils.rf_lib import filter_nearlabeled
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\DecodeLabeling\\')

#%% ########################## Load data #######################
protocol            = ['DN']
calciumversion      = 'deconv'

sessions,nSessions  = filter_sessions(protocol,load_calciumdata=True,load_behaviordata=True,
                                      load_videodata=True,calciumversion=calciumversion,
                                      min_cells=1,
                                    #   min_lab_cells_V1=20,min_lab_cells_PM=20,
                                    #   min_lab_cells_V1=1,min_lab_cells_PM=1,
                                      filter_areas=['V1','PM'])

report_sessions(sessions)
# sessions,nSessions  = filter_sessions(protocol,calciumversion=calciumversion,min_lab_cells_V1=50,min_lab_cells_PM=50)

#%% Z-score the calciumdata: 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
s_pre       = -60  #pre cm
s_post      = 80   #post cm
sbinsize    = 10     #spatial binning in cm

for i in tqdm(range(nSessions),desc='Computing spatial tensor',total=nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=sbinsize,method='binmean')

#%% Pd dataframe of all cells:
celldata = pd.concat([ses.celldata for ses in sessions])
N = len(celldata)

#%% Create index of nearby cells to compare to:
idx_nearby = np.zeros(N,dtype=bool)
for ses in sessions:
    idx_ses = np.where(celldata['session_id']==ses.sessiondata['session_id'][0])[0]
    idx_nearby[idx_ses] = filter_nearlabeled(ses,radius=50)

#%% Get psychometric function data and get index of all neurons from a session with good performance
sessions = noise_to_psy(sessions,filter_engaged=True,bootstrap=True)

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
#          #    ######     #     # #     # #          ######  ####### ######  ####### #     # 
#         # #   #     #    #     # ##    # #          #     # #       #     #    #    #     # 
#        #   #  #     #    #     # # #   # #          #     # #       #     #    #    #     # 
#       #     # ######     #     # #  #  # #          #     # #####   ######     #    ####### 
#       ####### #     #    #     # #   # # #          #     # #       #          #    #     # 
#       #     # #     #    #     # #    ## #          #     # #       #          #    #     # 
####### #     # ######      #####  #     # #######    ######  ####### #          #    #     # 

#%% ################### Compute mean activity for saliency trial bins for all sessions ##################

labeled         = ['unl','lab']
nlabels         = len(labeled)
areas           = ['V1','PM']
nareas          = len(areas)
clrs_areas      = get_clr_areas(areas)

sigtype     = 'signal'
zmin        = 5
zmax        = 20
nbins_noise = 3

data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,
                                                                    splithitmiss=False,min_ntrials=10,autobin=True)
Z                       = np.shape(data_mean_spatial_hitmiss)[1]
clrs_Z,labels_Z         = get_clr_thrbins(Z)

#%%
ndepthbins      = 5
depth_edges     = np.linspace(75,500,ndepthbins+1)
depth_centers   = np.stack((depth_edges[:-1],depth_edges[1:]),axis=1).mean(axis=1)

fig,ax = plt.subplots(1,1,figsize=(3,3))
for iarea,area in enumerate(areas):
    ax.hist(celldata['depth'][np.isin(celldata['roi_name'],area)],bins=depth_edges,
            color=clrs_areas[iarea],alpha=0.5,label=area)
ax.set_xlabel('Cortical depth (um)')
ax.set_ylabel('Number of Neurons')
ax.set_xlim([0,550])
ax.set_xticks(depth_edges)
h = ax.legend(frameon=False,fontsize=11,loc='upper right')
for text in h.get_texts():
    text.set_color(clrs_areas[areas.index(text.get_text())])
    
# ax.set_xticks(depth_centers)
# ax.set_xticklabels(depth_centers)

#%% 
noise_level     = 20
nminneurons     = 20

for iarea,area in enumerate(areas):
    fig,axes = plt.subplots(ndepthbins,Z*2,figsize=(Z*2*2,ndepthbins*2),sharey=True,sharex=True)

    for iZ in range(Z):
        for ibin,(bd1,bd2) in enumerate(zip(depth_edges[:-1],depth_edges[1:])):

            idx_N_unl = np.all((celldata['noise_level']<noise_level,
                celldata['labeled']=='unl',
                np.isin(celldata['roi_name'],area),
                celldata['depth']>=bd1,
                celldata['depth']<bd2,
                idx_N_perf,
                idx_nearby
                ),axis=0)
            
            idx_N_lab = np.all((celldata['noise_level']<noise_level,
                celldata['labeled']=='lab',
                np.isin(celldata['roi_name'],area),
                celldata['depth']>=bd1,
                celldata['depth']<bd2,
                idx_N_perf,
                idx_nearby),axis=0)
            
            if sum(idx_N_lab) >= nminneurons:
                ax = axes[ibin,iZ]
                shaded_error(sbins,data_mean_spatial_hitmiss[idx_N_unl,iZ,:],color=clrs_Z[iZ], label=plotcenters[iZ],
                            center='mean',error='sem',linewidth=2,linestyle=['-','--'][0],ax=ax)
                shaded_error(sbins,data_mean_spatial_hitmiss[idx_N_lab,iZ,:],color=clrs_Z[iZ], label=plotcenters[iZ],
                            center='mean',error='sem',linewidth=2,linestyle=['-','--'][1],ax=ax)
                ax.plot([0,0],[0,0.25],color='grey',linewidth=2)
                
                if ibin==0:
                    ax.set_title(labels_Z[iZ],fontsize=15,fontweight='bold',color=clrs_Z[iZ],ha='center',va='center')

                ax = axes[ibin,iZ+Z]
                temp = np.nanmean(data_mean_spatial_hitmiss[idx_N_lab,iZ,:],axis=0) - np.nanmean(data_mean_spatial_hitmiss[idx_N_unl,iZ,:],axis=0)
                
                ax.plot(sbins,temp,color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=2,linestyle='-')
                ax.fill_between(sbins,temp,where=(temp>=0),color='gray',alpha=0.5,interpolate=True)
                ax.fill_between(sbins,temp,where=(temp<=0),color='gray',alpha=0.5,interpolate=True)

                # t,p = stats.ranksums(data_mean_spatial_hitmiss[idx_N_unl,iZ,:],data_mean_spatial_hitmiss[idx_N_lab,iZ,:])
                t,p = stats.ttest_ind(data_mean_spatial_hitmiss[idx_N_unl,iZ,:],data_mean_spatial_hitmiss[idx_N_lab,iZ,:])
                for ib,sbin in enumerate(sbins):
                    if p[ib]<0.05:
                        ax.text(sbins[ib],0.2,get_sig_asterisks(p[ib],return_ns=False),color='black',
                                fontsize=15,fontweight='bold')

                ax.set_xlim([-50,75])

                if iZ==2 and ibin==0:
                    ax = axes[ibin,iZ+Z]
                    ax.set_title('Activity difference',color='black',fontsize=15,fontweight='bold')
        
                if iZ==0:
                    ax.text(-25,0.2,'%3.0f-\n%3.0f um' % (bd1,bd2),fontsize=12,fontweight='bold',
                            rotation=0,ha='center',va='center')
                    
                if iZ == 0 and ibin == 1:
                    ax = axes[ibin,iZ]
                    leg_lines = [Line2D([0], [0], color='black', linewidth=2, linestyle='-'),
                                Line2D([0], [0], color='black', linewidth=2, linestyle='--')]
                    leg_labels = ['unlabeled','labeled']
                    ax.legend(leg_lines, leg_labels, loc='upper right', frameon=False, fontsize=13)
    for ax in axes.flatten():
        ax.set_axis_off()
        
    fig.suptitle('Area: %s' % (area),fontsize=30)
    fig.subplots_adjust(hspace=0.03,wspace=0.03)
    fig.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanAct_%s_Spatial_%dsessions_PerDepth_perfonly.png' % (area,nSessions)), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanAct_%s_Spatial_%dsessions_PerDepth.png' % (area,nSessions)), format = 'png')

#%% 
noise_level     = 20
nminneurons     = 20

plotmean = np.full((nareas,ndepthbins),np.nan)
ploterror = np.zeros((nareas,ndepthbins))

for iarea,area in enumerate(areas):
    for ibin,(bd1,bd2) in enumerate(zip(depth_edges[:-1],depth_edges[1:])):

        idx_N_unl = np.all((celldata['noise_level']<noise_level,
            celldata['labeled']=='unl',
            np.isin(celldata['roi_name'],area),
            celldata['depth']>=bd1,
            celldata['depth']<bd2,
            idx_N_perf,
            idx_nearby
            ),axis=0)
        
        idx_N_lab = np.all((celldata['noise_level']<noise_level,
            celldata['labeled']=='lab',
            np.isin(celldata['roi_name'],area),
            celldata['depth']>=bd1,
            celldata['depth']<bd2,
            idx_N_perf,
            idx_nearby
            ),axis=0)
        
        if sum(idx_N_lab) >= nminneurons:
            # tempdata = data_mean_spatial_hitmiss[idx_N_unl,:,:]
            # tempdata = data_mean_spatial_hitmiss[np.ix_(idx_N_lab,np.arange(1,Z-1),(sbins>=0)  & (sbins<=20))] - data_mean_spatial_hitmiss[np.ix_(idx_N_unl,np.arange(1,Z-1),(sbins>=0)  & (sbins<=20))]
            # tempdata = np.nanmean(tempdata,axis=(1,2))
            # plotmean[iarea,ibin] = np.nanmean(tempdata)
            # ploterror[iarea,ibin] = np.nanstd(tempdata)/np.sqrt(len(tempdata))

            temp_lab = data_mean_spatial_hitmiss[np.ix_(idx_N_lab,np.arange(1,Z-1),(sbins>=0)  & (sbins<=20))]
            temp_unl = data_mean_spatial_hitmiss[np.ix_(idx_N_unl,np.arange(1,Z-1),(sbins>=0)  & (sbins<=20))] 
            temp_lab = np.nanmean(temp_lab,axis=(1,2))
            temp_unl = np.nanmean(temp_unl,axis=(1,2))
            tempdata = np.nanmean(temp_lab) - np.nanmean(temp_unl)

            plotmean[iarea,ibin] = np.nanmean(tempdata)
            plotmean[iarea,ibin] = tempdata
            # ploterror[iarea,ibin] = np.nanstd(tempdata)/np.sqrt(len(tempdata))


fig,ax = plt.subplots(1,1,figsize=(6,4),sharey=True,sharex=True)

for iarea,area in enumerate(areas):
    # for ibin,(bd1,bd2) in enumerate(zip(depth_edges[:-1],depth_edges[1:])):
    shaded_error(depth_centers,y = plotmean[iarea,:],yerror = ploterror[iarea,:],color=clrs_areas[iarea],label=area,
                linewidth=2,linestyle=['-','--'][0],ax=ax)

ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
# fig.suptitle('Area: %s' % (area),fontsize=30)
fig.subplots_adjust(hspace=0.03,wspace=0.03)
fig.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanAct_%s_Spatial_%dsessions_PerDepth_perfonly.png' % (area,nSessions)), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanAct_%s_Spatial_%dsessions_PerDepth.png' % (area,nSessions)), format = 'png')


#%% 
#     # ### #######    #     # ###  #####   #####     ######  ####### ######  ####### #     # 
#     #  #     #       ##   ##  #  #     # #     #    #     # #       #     #    #    #     # 
#     #  #     #       # # # #  #  #       #          #     # #       #     #    #    #     # 
#######  #     #       #  #  #  #   #####   #####     #     # #####   ######     #    ####### 
#     #  #     #       #     #  #        #       #    #     # #       #          #    #     # 
#     #  #     #       #     #  #  #     # #     #    #     # #       #          #    #     # 
#     # ###    #       #     # ###  #####   #####     ######  ####### #          #    #     # 


#%% 
sigtype     = 'signal'
zmin        = 5
zmax        = 20
# zmin        = 7
# zmax        = 17
nbins_noise = 3

data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,
                                                                    splithitmiss=True,min_ntrials=5,autobin=True)
Z                       = np.shape(data_mean_spatial_hitmiss)[1]
clrs_Z,labels_Z         = get_clr_thrbins(Z)
# data_mean_spatial_hitmiss[data_mean_spatial_hitmiss>np.nanpercentile(data_mean_spatial_hitmiss,99)] = np.nanpercentile(data_mean_spatial_hitmiss,99)
# data_mean_spatial_hitmiss[data_mean_spatial_hitmiss<np.nanpercentile(data_mean_spatial_hitmiss,0.1)] = np.nanpercentile(data_mean_spatial_hitmiss,0.1)

# data_mean_spatial_hitmiss[np.any(np.isnan(data_mean_spatial_hitmiss),axis=(1,2,3)),:,:,:] = np.nan

#%%
ndepthbins      = 5
depth_edges     = np.linspace(75,500,ndepthbins+1)
depth_centers   = np.stack((depth_edges[:-1],depth_edges[1:]),axis=1).mean(axis=1)

noise_level     = 20
nminneurons     = 20

for iarea,area in enumerate(areas):
    for ilab,lab in enumerate(labeled):
        fig,axes = plt.subplots(ndepthbins,Z*2,figsize=(Z*2*2,ndepthbins*2),sharey=True,sharex=True)

        for iZ in range(Z):
            for ibin,(bd1,bd2) in enumerate(zip(depth_edges[:-1],depth_edges[1:])):

                idx_N = np.all((celldata['noise_level']<noise_level,
                    celldata['labeled']==lab,
                    np.isin(celldata['roi_name'],area),
                    celldata['depth']>=bd1,
                    celldata['depth']<bd2,
                    idx_N_perf,
                    idx_nearby
                    ),axis=0)
                
                if sum(idx_N) >= nminneurons:
                    ax = axes[ibin,iZ]
                    shaded_error(sbins,data_mean_spatial_hitmiss[idx_N,iZ,:,0],color=clrs_Z[iZ], label=plotcenters[iZ],
                                error='sem',linewidth=2,linestyle=['--','-'][0],ax=ax)
                    shaded_error(sbins,data_mean_spatial_hitmiss[idx_N,iZ,:,1],color=clrs_Z[iZ], label=plotcenters[iZ],
                                error='sem',linewidth=2,linestyle=['--','-'][1],ax=ax)
                    ax.plot([0,0],[0,0.25],color='grey',linewidth=2)
                    
                    if ibin==0:
                        ax.set_title(labels_Z[iZ],fontsize=15,fontweight='bold',color=clrs_Z[iZ],ha='center',va='center')

                    ax = axes[ibin,iZ+Z]
                    temp = np.nanmean(data_mean_spatial_hitmiss[idx_N,iZ,:,1],axis=0) - np.nanmean(data_mean_spatial_hitmiss[idx_N,iZ,:,0],axis=0)
                    
                    ax.plot(sbins,temp,color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=2,linestyle='-')
                    ax.fill_between(sbins,temp,where=(temp>=0),color='gray',alpha=0.5,interpolate=True)
                    ax.fill_between(sbins,temp,where=(temp<=0),color='gray',alpha=0.5,interpolate=True)

                    t,p = stats.ttest_rel(data_mean_spatial_hitmiss[idx_N,iZ,:,0],data_mean_spatial_hitmiss[idx_N,iZ,:,1])
                    for ib,sbin in enumerate(sbins):
                        if p[ib]<0.05:
                            ax.text(sbins[ib],0.2,get_sig_asterisks(p[ib],return_ns=False),color='black',
                                    fontsize=15,fontweight='bold')
                    
                    ax.set_xlim([-50,75])

                    if iZ==2 and ibin==0:
                        ax = axes[ibin,iZ+Z]
                        ax.set_title('Activity difference',color='black',fontsize=15,fontweight='bold')
            
                    if iZ==0:
                        ax.text(-25,0.2,'%3.0f-\n%3.0f um' % (bd1,bd2),fontsize=12,fontweight='bold',
                                rotation=0,ha='left',va='center')
                        
                    if iZ == 0 and ibin == 1:
                        ax = axes[ibin,iZ]
                        leg_lines = [Line2D([0], [0], color='black', linewidth=2, linestyle='--'),
                                    Line2D([0], [0], color='black', linewidth=2, linestyle='-')]
                        leg_labels = ['Miss','Hit']
                        ax.legend(leg_lines, leg_labels, loc='upper right', frameon=False, fontsize=13)
        for ax in axes.flatten():
            ax.set_axis_off()
            
        fig.suptitle('%s - %s' % (area, lab),fontsize=30)
        fig.subplots_adjust(hspace=0.03,wspace=0.03)
        fig.tight_layout()
        # fig.savefig(os.path.join(savedir,'ActHitMiss_Depth_%s_%s_Spatial_%dsessions_perfonly.png' % (area,lab,nSessions)), format = 'png')
        # fig.savefig(os.path.join(savedir,'ActHitMiss_Depth_%s_%s_Spatial_%dsessions.png' % (area,lab,nSessions)), format = 'png')

#%% Classify area and label from tuning profile:

for ses in sessions:
    ses.respmat = np.nanmean(ses.stensor[:,:,(sbins>=0)&(sbins<=20)],axis=2)

data_mean_hitmiss,plotcenters = get_mean_signalbins(sessions,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,
                                                    min_ntrials=5,autobin=True)


data = data_mean_hitmiss.reshape(N,-1)

#%% Decoding cell type and area from tuning profile:
# looking at misclassification of area and label
from utils.regress_lib import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# LDA on area + labeling:
model = LDA(n_components=2,solver='eigen', shrinkage='auto')
# model = LDA(n_components=3,solver='svd')

idx_N = np.all((celldata['noise_level']<noise_level,
                idx_N_perf,
                idx_nearby,
                # celldata['sig_MN']==1
                ),axis=0)

X = data[idx_N,:]
y = celldata['arealabel'][idx_N] #ses.trialdata['Orientation'][idx_T]
# y = ses.trialdata['Orientation'][idx_T]


arealabels = np.unique(celldata['arealabel'] )
clrs_arealabels = get_clr_area_labeled(arealabels)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array

X[np.isnan(X)] = np.nanmean(X)

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

# X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

# Train a classification model on the training data with regularization
LDAproj = model.fit_transform(X, y).T
fig = plt.figure(figsize=(4,3))
for ial, arealabel in enumerate(arealabels):
    idx_ial = np.where(y==ial)[0]
    plt.scatter(LDAproj[0,idx_ial],LDAproj[1,idx_ial],s=3, label=arealabel,c=clrs_arealabels[ial],alpha=0.25)

from sklearn.model_selection import cross_val_predict,cross_val_score,KFold

cv = KFold(n_splits=5,shuffle=True)
# cv = KFold(n_splits=5,shuffle=False)
y_pred = cross_val_predict(model,X,y.ravel(),cv=cv)
# y_pred = cross_val_predict(model,X,y,cv=cv)


fig = plt.figure(figsize=(4,3))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.array(y),np.array(y_pred),labels=label_encoder.transform(arealabels))
cm = cm / cm.sum(axis=1)[:, np.newaxis]
# cm = cm / cm.sum(axis=0)[:, np.newaxis]
plt.imshow(cm,interpolation='nearest',vmin=0,vmax=0.5)
plt.colorbar(shrink=0.5)
tick_marks = [i for i in range(len(arealabels))]
plt.xticks(tick_marks, arealabels, rotation=45)
plt.yticks(tick_marks, arealabels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
for i in range(len(arealabels)):
    plt.gca().add_patch(plt.Rectangle((i-.5,i-.5),1,1,edgecolor='r',facecolor='none',lw=2))

    for j in range(len(arealabels)):
        if j == i+2 or j == i-2:
            plt.gca().add_patch(plt.Rectangle((i-.5,j-.5),1,1,edgecolor='purple',facecolor='none',lw=2))
plt.tight_layout()
fig.savefig(os.path.join(savedir,'LDA_arealabel_confusion_%ssessions.png' % nSessions), format = 'png')


#plot the performance of the diagonal versus off-diagonal
fig,ax = plt.subplots(1,1,figsize=(2,3))
dat = np.diag(cm)
ax.errorbar(0,np.nanmean(dat),yerr=np.nanstd(dat),label='on-diagonal',fmt='o',color='black',ecolor='black',capsize=3)
dat = cm[~np.eye(cm.shape[0], dtype=bool)]
ax.errorbar(1,np.nanmean(dat),yerr=np.nanstd(dat),label='Off-diagonal',fmt='o',color='black',ecolor='black',capsize=3)
ax.set_xlabel('True label')
ax.set_ylabel('Predicted accuracy')
ax.set_ylim([0,0.5])
ax.legend()
fig.savefig(os.path.join(savedir,'LDA_arealabel_predacc_%ssessions.png' % nSessions), format = 'png')





#%% Decode lab vs unlab

data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=2)
# data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=False)

#%%


#%% 
model_name = 'LOGR'
# model_name = 'LDA'
lam   = 0.95
lam   = 0.0001
# lam   = 0.00001
lam = 1e6
# lam = None
lam = 'auto'

noise_level     = 20
exp_label       = 'Decoding_proj_type'
nmodelruns      = 100
subfrac         = 0.5

minneurons      = 100

lambdas = np.logspace(-6, 5, 20)
nlambdas = len(lambdas)
perf    = np.full((nareas,nmodelruns,nlambdas),np.nan)

for iarea,area in enumerate(areas):

    idx_N = np.all((celldata['noise_level']<noise_level,
                    np.isin(celldata['roi_name'],area),
                    celldata['depth']<250,
                    # celldata['depth']>250,
                    idx_N_perf,
                    idx_nearby),axis=0)
    
    y = celldata['redcell'][idx_N].to_numpy()
    X = data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)

    X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

    coefs   = np.full((np.shape(data_mean_spatial_hitmiss)[1:] + (nmodelruns,)),np.nan)
    
    if lam is None:
        lam = find_optimal_lambda(X,y,model_name=model_name,kfold=5)
    
    if model_name == 'LDA': 
        model = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))

    elif model_name == 'LOGR':
        model = LOGR(penalty='l2', solver='liblinear', C=lam)
    
    # print(np.sum(y==1))
    # np.sum(y==1)>=minneurons:
    for i in range(nmodelruns):


        idx_sub             = np.concatenate((np.random.choice(np.where(y==0)[0],size=np.min([minneurons,np.sum(y==0)]),replace=False),
                                        np.random.choice(np.where(y==1)[0],size=np.min([minneurons,np.sum(y==1)]),replace=False)))
        
        # idx_sub = np.random.choice(np.arange(np.shape(X)[0]),size=np.shape(X)[0]//(int(1/subfrac)),replace=False)
        Xsub,ysub = X[idx_sub,:], y[idx_sub]
        # X,y,idx_nan = prep_Xpredictor(Xsub,ysub) #zscore, set columns with all nans to 0, set nans to 0
        # LDAproj     = model.fit_transform(Xsub,ysub)
        # model.fit(Xsub,ysub)
        for ilam,lam in enumerate(lambdas):
            perf[iarea,i,ilam],_,_,ev = my_decoder_wrapper(Xsub,ysub,
                            model_name='LOGR',kfold=5,lam=lam,subtract_shuffle=False,
                                scoring_type=None,norm_out=False)
        # coefs[:,:,:,i] = np.reshape(model.coef_,(Z,len(sbins),2))
    # coefs = np.nanmean(coefs,axis=3)

#%%
fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.plot(lambdas,np.nanmean(perf[0,:,:],axis=0),c=clrs_areas[0],label='V1')
ax.plot(lambdas,np.nanmean(perf[1,:,:],axis=0),c=clrs_areas[1],label='PM')
shaded_error(lambdas,perf[0,:,:],color=clrs_areas[0],alpha=0.5,error='sem')
shaded_error(lambdas,perf[1,:,:],color=clrs_areas[1],alpha=0.5,error='sem')

ax.set_xscale('log')
ax.set_title('Decoding performance vs. lambda')
ax.set_xlabel('lambda')
ax.set_ylabel('performance')
ax.legend(loc='upper left',frameon=False)

#%% 
model_name = 'LOGR'
# model_name = 'LDA'
lam   = 0.95
lam   = 0.0001
# lam   = 0.00001
lam = 1e6
# lam = None

noise_level     = 20
exp_label       = 'Decoding_proj_type'
nmodelruns      = 100
subfrac         = 0.5

minneurons      = 100

perf    = np.full((nareas,nmodelruns),np.nan)
for iarea,area in enumerate(areas):

    idx_N = np.all((celldata['noise_level']<noise_level,
                    np.isin(celldata['roi_name'],area),
                    celldata['depth']<250,
                    # celldata['depth']>250,
                    idx_N_perf,
                    idx_nearby),axis=0)
    
    y = celldata['redcell'][idx_N].to_numpy()
    X = data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)

    X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

    coefs   = np.full((np.shape(data_mean_spatial_hitmiss)[1:] + (nmodelruns,)),np.nan)
    
    if lam is None:
        lam = find_optimal_lambda(X,y,model_name=model_name,kfold=5)
    
    if model_name == 'LDA': 
        model = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))

    elif model_name == 'LOGR':
        model = LOGR(penalty='l2', solver='liblinear', C=lam)
    
    # print(np.sum(y==1))
    # np.sum(y==1)>=minneurons:
    for i in range(nmodelruns):

        idx_sub             = np.concatenate((np.random.choice(np.where(y==0)[0],size=np.min([minneurons,np.sum(y==0)]),replace=False),
                                        np.random.choice(np.where(y==1)[0],size=np.min([minneurons,np.sum(y==1)]),replace=False)))
        
        # idx_sub = np.random.choice(np.arange(np.shape(X)[0]),size=np.shape(X)[0]//(int(1/subfrac)),replace=False)
        Xsub,ysub = X[idx_sub,:], y[idx_sub]
        # X,y,idx_nan = prep_Xpredictor(Xsub,ysub) #zscore, set columns with all nans to 0, set nans to 0
        # LDAproj     = model.fit_transform(Xsub,ysub)
        # model.fit(Xsub,ysub)
        perf[iarea,i],_,_,ev = my_decoder_wrapper(Xsub,ysub,
                        model_name='LOGR',kfold=5,lam=lam,subtract_shuffle=False,
                          scoring_type=None,norm_out=False)
        # coefs[:,:,:,i] = np.reshape(model.coef_,(Z,len(sbins),2))
    # coefs = np.nanmean(coefs,axis=3)

clrs_areas = get_clr_areas(areas)

fig,ax = plt.subplots(1,1,figsize=(4,3))
for iarea,area in enumerate(areas):
    ax.scatter(iarea*np.ones(nmodelruns)+np.random.normal(0,0.05,size=nmodelruns),perf[iarea,:],c=clrs_areas[iarea])
ax.scatter(range(nareas),np.nanmean(perf,axis=1),c='k',marker='x')
ax.axhline(0.5,color='k',linestyle='--')
ax.set_xticks(range(nareas))
ax.set_xticklabels(areas)
ax.set_ylabel('Decoding performance')
ax.set_xlabel('Area')
ax.set_ylim([0,1])
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'Decoding_%s_%ssessions.png' % (model_name,sessions[ises].sessiondata['session_id'][0])), format = 'png')

#%% 
fig,axes = plt.subplots(1,nareas,figsize=(4*nareas,2.5))
for iarea,area in enumerate(areas):
    ax = axes[iarea]

    # for iZ in range(Z):
    for iZ in range(Z)[1:-1]:
        ax.plot(sbins,coefs[iZ,:,0],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][0])
        ax.plot(sbins,coefs[iZ,:,1],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][1])

    # ax.plot(sbins,np.nanmean(coefs[1:-1,:,0],axis=0),color='k',linewidth=2,linestyle=['--','-'][0])
    # ax.plot(sbins,np.nanmean(coefs[1:-1,:,1],axis=0),color='k',linewidth=2,linestyle=['--','-'][1])

    # ax.plot(sbins,np.nanmean(coefs[:-1,:,0],axis=0),color='k',linewidth=2,linestyle=['--','-'][0])
    # ax.plot(sbins,np.nanmean(coefs[:-1,:,1],axis=0),color='k',linewidth=2,linestyle=['--','-'][1])

    ax.plot(sbins,np.nanmean(coefs[1:-1,:,0],axis=0),color='k',linewidth=3,linestyle=['--','-'][0])
    ax.plot(sbins,np.nanmean(coefs[1:-1,:,1],axis=0),color='k',linewidth=3,linestyle=['--','-'][1])

    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
    ax.set_xticks([-75,-50,-25,0,25,50,75])
    ax.set_xticklabels([-75,-50,-25,0,25,50,75])
    add_stim_resp_win(ax)
    ax.set_xlim([-60,80])
    ax.set_title(area)
    if iarea == 0: 
        ax.set_ylabel('Weights')
        ax.legend(labels_Z,frameon=False,fontsize=8,loc='upper left')
plt.tight_layout()






#%% 







#%% 

data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=2)
# data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=False)

# data_mean_spatial_hitmiss -= np.nanmean(data_mean_spatial_hitmiss[:,:,sbins<0,:],axis=(2),keepdims=True)
# data_mean_spatial_hitmiss -= np.nanmean(data_mean_spatial_hitmiss,axis=(2),keepdims=True)

#%%

# doesn't work!!! '
# ''
# '
areas           = ['V1','PM']
nareas          = len(areas)

noise_level     = 20
exp_label       = 'Decoding_proj_type'
nmodelruns      = 25
# subfrac         = 0.5
lam             = 0.9
minneurons      = 10 #per cat, lab or unl
coefs           = np.full((nareas,Z,len(sbins),2,nSessions),np.nan)
sigtype         = 'signal'
nbins_noise     = 3

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding label identity across sessions'):
    
    zmin        = np.min(ses.trialdata['signal'][ses.trialdata['stimcat']=='N'])
    zmax        = np.max(ses.trialdata['signal'][ses.trialdata['stimcat']=='N'])
    # # zmin        = 7
    # # zmax        = 17
    data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins([ses],sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=2)

    for iarea,area in enumerate(areas):
        idx_nearby = filter_nearlabeled(ses,radius=50)

        idx_N = np.all((ses.celldata['noise_level']<noise_level,
                        np.isin(ses.celldata['roi_name'],area),
                        # celldata['depth']<250,
                        # celldata['depth']>250,
                        idx_nearby),axis=0)
            
        y = ses.celldata['redcell'][idx_N].to_numpy()
        
        # X = ses.tensor[np.ix_(idx_N,ses.celldata['redcell'][idx_N].to_numpy(),sbins<0)].reshape(np.sum(idx_N),-1)
        # X = ses.stensor[idx_N,:,:].reshape(np.sum(idx_N),-1)
        data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)
# 
        # X = data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)

        # tempdata -= np.nanmean(tempdata,axis=(1,2),keepdims=True)
        # tempdata /= np.nanstd(tempdata[:,:,:],axis=(1,2),keepdims=True)

        # tempdata -= np.nanmean(tempdata[:,:,sbins<0],axis=(2),keepdims=True)

        # X -= np.nanmean(X,axis=1,keepdims=True)

        # X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
        X           = zscore(X, axis=1,nan_policy='omit')
        idx_nan     = ~np.all(np.isnan(X),axis=1)
        X           = X[idx_nan,:]
        y           = y[idx_nan]
        X[:,np.all(np.isnan(X),axis=0)] = 0
        X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
        y           = np.nan_to_num(y,nan=np.nanmean(y,axis=0,keepdims=True))

        model       = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))
        # model       = LOGR(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))
        # coefs = np.full((np.shape(data_mean_spatial_hitmiss) + (nmodelruns,)),np.nan)
        coeftemp    = np.full((np.shape(data_mean_spatial_hitmiss)[1:] + (nmodelruns,)),np.nan)
        if np.sum(y==0)>=minneurons and np.sum(y==1)>=minneurons:
            for i in range(nmodelruns):
                idx_sub             = np.concatenate((np.random.choice(np.where(y==0)[0],size=minneurons,replace=False),
                                                        np.random.choice(np.where(y==1)[0],size=minneurons,replace=False)))
                
                Xsub,ysub           = X[idx_sub,:], y[idx_sub]
                # X,y,idx_nan = prep_Xpredictor(Xsub,ysub) #zscore, set columns with all nans to 0, set nans to 0
                LDAproj             = model.fit_transform(Xsub,ysub)
                coeftemp[:,:,:,i]   = np.reshape(model.coef_,(Z,len(sbins),2))
            coefs[iarea,:,:,:,ises] = np.nanmean(coeftemp,axis=3)
        
#%% 

fig,axes = plt.subplots(1,nareas,figsize=(4*nareas,2.5))

for iarea,area in enumerate(areas):
    ax = axes[iarea]

    # for iZ in range(Z):
    # for iZ in range(Z)[1:-1]:
        # ax.plot(sbins,coefs[iZ,:,0],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][0])
        # ax.plot(sbins,coefs[iZ,:,1],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][1])
    handles = []
    # for iZ in range(Z)[1:-1]:
    for iZ in range(Z):
        handles.append(shaded_error(sbins,coefs[iarea,iZ,:,0,:].T,error='sem',color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,
                     linestyle=['--','-'][0],ax=ax))
        shaded_error(sbins,coefs[iarea,iZ,:,1,:].T,error='sem',color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,
                     linestyle=['--','-'][1],ax=ax)
        # for ises,ses in enumerate(sessions):
            # ax.plot(sbins,coefs[iarea,iZ,:,1,ises],color=clrs_Z[iZ],linewidth=0.25,linestyle=['--','-'][0])

    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
    ax.set_xticks([-75,-50,-25,0,25,50,75])
    ax.set_xticklabels([-75,-50,-25,0,25,50,75])
    add_stim_resp_win(ax)
    ax.set_xlim([-60,80])
    ax.set_title(area)
    if iarea == 0: 
        ax.set_ylabel('LDA weights')
        ax.legend(handles,labels_Z,frameon=False,fontsize=8,loc='upper left')
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')
# fig.savefig(os.path.


#%% 



# # doesn't work!!! '
# # ''
# # '
# areas           = ['V1','PM']
# nareas          = len(areas)

# noise_level     = 20
# exp_label       = 'Decoding_proj_type'
# nmodelruns      = 25
# # subfrac         = 0.5
# lam             = 0.9
# minneurons      = 10 #per cat, lab or unl
# coefs           = np.full((nareas,Z,len(sbins),2,nSessions),np.nan)
# sigtype         = 'signal'
# nbins_noise     = 3

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding label identity across sessions'):
    
#     zmin        = np.min(ses.trialdata['signal'][ses.trialdata['stimcat']=='N'])
#     zmax        = np.max(ses.trialdata['signal'][ses.trialdata['stimcat']=='N'])
#     # zmin        = 7
#     # zmax        = 17
#     data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins([ses],sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=2)

#     for iarea,area in enumerate(areas):
#         idx_nearby = filter_nearlabeled(ses,radius=50)

#         idx_N = np.all((ses.celldata['noise_level']<noise_level,
#                         np.isin(ses.celldata['roi_name'],area),
#                         # celldata['depth']<250,
#                         # celldata['depth']>250,
#                         idx_nearby),axis=0)
            
#         y = ses.celldata['redcell'][idx_N].to_numpy()
        
#         X = data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)

#         # tempdata -= np.nanmean(tempdata,axis=(1,2),keepdims=True)
#         # tempdata /= np.nanstd(tempdata[:,:,:],axis=(1,2),keepdims=True)

#         # tempdata -= np.nanmean(tempdata[:,:,sbins<0],axis=(2),keepdims=True)

#         # X -= np.nanmean(X,axis=1,keepdims=True)

#         # X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
#         X           = zscore(X, axis=1,nan_policy='omit')
#         idx_nan     = ~np.all(np.isnan(X),axis=1)
#         X           = X[idx_nan,:]
#         y           = y[idx_nan]
#         X[:,np.all(np.isnan(X),axis=0)] = 0
#         X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
#         y           = np.nan_to_num(y,nan=np.nanmean(y,axis=0,keepdims=True))

#         model       = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))
#         # coefs = np.full((np.shape(data_mean_spatial_hitmiss) + (nmodelruns,)),np.nan)
#         coeftemp    = np.full((np.shape(data_mean_spatial_hitmiss)[1:] + (nmodelruns,)),np.nan)
#         if np.sum(y==0)>=minneurons and np.sum(y==1)>=minneurons:
#             for i in range(nmodelruns):
#                 idx_sub             = np.concatenate((np.random.choice(np.where(y==0)[0],size=minneurons,replace=False),
#                                                         np.random.choice(np.where(y==1)[0],size=minneurons,replace=False)))
                
#                 Xsub,ysub           = X[idx_sub,:], y[idx_sub]
#                 # X,y,idx_nan = prep_Xpredictor(Xsub,ysub) #zscore, set columns with all nans to 0, set nans to 0
#                 LDAproj             = model.fit_transform(Xsub,ysub)
#                 coeftemp[:,:,:,i]   = np.reshape(model.coef_,(Z,len(sbins),2))
#             coefs[iarea,:,:,:,ises] = np.nanmean(coeftemp,axis=3)
        
# #%% 

# fig,axes = plt.subplots(1,nareas,figsize=(4*nareas,2.5))

# for iarea,area in enumerate(areas):
#     ax = axes[iarea]

#     # for iZ in range(Z):
#     # for iZ in range(Z)[1:-1]:
#         # ax.plot(sbins,coefs[iZ,:,0],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][0])
#         # ax.plot(sbins,coefs[iZ,:,1],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][1])
#     handles = []
#     # for iZ in range(Z)[1:-1]:
#     for iZ in range(Z):
#         handles.append(shaded_error(sbins,coefs[iarea,iZ,:,0,:].T,error='sem',color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,
#                      linestyle=['--','-'][0],ax=ax))
#         shaded_error(sbins,coefs[iarea,iZ,:,1,:].T,error='sem',color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,
#                      linestyle=['--','-'][1],ax=ax)
#         # for ises,ses in enumerate(sessions):
#             # ax.plot(sbins,coefs[iarea,iZ,:,1,ises],color=clrs_Z[iZ],linewidth=0.25,linestyle=['--','-'][0])

#     # ax.plot(sbins,np.nanmean(coefs[1:-1,:,0],axis=0),color='k',linewidth=2,linestyle=['--','-'][0])
#     # ax.plot(sbins,np.nanmean(coefs[1:-1,:,1],axis=0),color='k',linewidth=2,linestyle=['--','-'][1])

#     # ax.plot(sbins,np.nanmean(coefs[1:-1,:,0],axis=0),color='k',linewidth=3,linestyle=['--','-'][0])
#     # ax.plot(sbins,np.nanmean(coefs[1:-1,:,1],axis=0),color='k',linewidth=3,linestyle=['--','-'][1])

#     ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
#     ax.set_xticks([-75,-50,-25,0,25,50,75])
#     ax.set_xticklabels([-75,-50,-25,0,25,50,75])
#     add_stim_resp_win(ax)
#     ax.set_xlim([-60,80])
#     ax.set_title(area)
#     if iarea == 0: 
#         ax.set_ylabel('LDA weights')
#         ax.legend(handles,labels_Z,frameon=False,fontsize=8,loc='upper left')
# plt.tight_layout()
# # fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')
# # fig.savefig(os.path.




#%% 

fig,axes = plt.subplots(1,nareas,figsize=(4*nareas,2.5))

for iarea,area in enumerate(areas):
    ax = axes[iarea]

    # for iZ in range(Z):
    for iZ in range(Z)[1:-1]:
        ax.plot(sbins,coefs[iZ,:,0],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][0])
        ax.plot(sbins,coefs[iZ,:,1],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][1])

    # ax.plot(sbins,np.nanmean(coefs[1:-1,:,0],axis=0),color='k',linewidth=2,linestyle=['--','-'][0])
    # ax.plot(sbins,np.nanmean(coefs[1:-1,:,1],axis=0),color='k',linewidth=2,linestyle=['--','-'][1])

    ax.plot(sbins,np.nanmean(coefs[1:-1,:,0],axis=0),color='k',linewidth=3,linestyle=['--','-'][0])
    ax.plot(sbins,np.nanmean(coefs[1:-1,:,1],axis=0),color='k',linewidth=3,linestyle=['--','-'][1])

    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
    ax.set_xticks([-75,-50,-25,0,25,50,75])
    ax.set_xticklabels([-75,-50,-25,0,25,50,75])
    add_stim_resp_win(ax)
    ax.set_xlim([-60,80])
    ax.set_title(area)
    if iarea == 0: 
        ax.set_ylabel('LDA weights')
        ax.legend(labels_Z,frameon=False,fontsize=8,loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')
# fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_DepthTo250%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')
# fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_DepthFrom250%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')

#%% 
zmin        = 5
zmax        = 20
nbins_noise = 3
data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True,min_ntrials=2)
# data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=False)
Z = len(plotcenters)
clrs_Z = ['black']  # Start with black
clrs_Z += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
clrs_Z.append('orange')  # Add orange at the end
labels_Z = ['catch','sub','thr','sup','max']

from sklearn.linear_model import LogisticRegression as LOGR

#%% 
model_name = 'LOGR'
# model_name = 'LDA'
lam   = 0.95
lam   = 0.0001
# lam   = 0.00001

noise_level     = 20
exp_label       = 'Decoding_proj_type'
nmodelruns      = 100
subfrac         = 0.5

minneurons      = 50

fig,axes = plt.subplots(1,nareas,figsize=(4*nareas,2.5))

for iarea,area in enumerate(areas):
    ax = axes[iarea]

    idx_N = np.all((celldata['noise_level']<noise_level,
                    np.isin(celldata['roi_name'],area),
                    # celldata['depth']<250,
                    # celldata['depth']>250,
                    idx_N_perf,
                    idx_nearby),axis=0)
        
    y = celldata['redcell'][idx_N].to_numpy()
    X = data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)

    X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

    coefs   = np.full((np.shape(data_mean_spatial_hitmiss)[1:] + (nmodelruns,)),np.nan)
    perf    = np.full((nareas,nmodelruns),np.nan)
    
    if lam is None:
        lam = find_optimal_lambda(X,y,model_name=model_name,kfold=5)
    
    if model_name == 'LDA': 
        model = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))

    elif model_name == 'LOGR':
        model = LOGR(penalty='l2', solver='liblinear', C=lam)
    
    print(np.sum(y==1))
    # np.sum(y==1)>=minneurons:
    for i in range(nmodelruns):

        idx_sub             = np.concatenate((np.random.choice(np.where(y==0)[0],size=np.min([minneurons,np.sum(y==0)]),replace=False),
                                        np.random.choice(np.where(y==1)[0],size=np.min([minneurons,np.sum(y==1)]),replace=False)))
        
        # idx_sub = np.random.choice(np.arange(np.shape(X)[0]),size=np.shape(X)[0]//(int(1/subfrac)),replace=False)
        Xsub,ysub = X[idx_sub,:], y[idx_sub]
        # X,y,idx_nan = prep_Xpredictor(Xsub,ysub) #zscore, set columns with all nans to 0, set nans to 0
        # LDAproj     = model.fit_transform(Xsub,ysub)
        model.fit(Xsub,ysub)
        perf[iarea,i],weights,projs,ev = my_decoder_wrapper(Xsub,ysub,
                        model_name='LOGR',kfold=5,lam=lam,subtract_shuffle=False,
                          scoring_type=None,norm_out=False)
        coefs[:,:,:,i] = np.reshape(model.coef_,(Z,len(sbins),2))
    coefs = np.nanmean(coefs,axis=3)

    # LDAproj = model.fit_transform(X,y)

    # coefs = np.reshape(model.coef_,(Z,len(sbins),2))

    # for iZ in range(Z):
    for iZ in range(Z)[1:-1]:
        ax.plot(sbins,coefs[iZ,:,0],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][0])
        ax.plot(sbins,coefs[iZ,:,1],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=1,linestyle=['--','-'][1])

    # ax.plot(sbins,np.nanmean(coefs[1:-1,:,0],axis=0),color='k',linewidth=2,linestyle=['--','-'][0])
    # ax.plot(sbins,np.nanmean(coefs[1:-1,:,1],axis=0),color='k',linewidth=2,linestyle=['--','-'][1])

    # ax.plot(sbins,np.nanmean(coefs[:-1,:,0],axis=0),color='k',linewidth=2,linestyle=['--','-'][0])
    # ax.plot(sbins,np.nanmean(coefs[:-1,:,1],axis=0),color='k',linewidth=2,linestyle=['--','-'][1])

    ax.plot(sbins,np.nanmean(coefs[1:-1,:,0],axis=0),color='k',linewidth=3,linestyle=['--','-'][0])
    ax.plot(sbins,np.nanmean(coefs[1:-1,:,1],axis=0),color='k',linewidth=3,linestyle=['--','-'][1])

    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
    ax.set_xticks([-75,-50,-25,0,25,50,75])
    ax.set_xticklabels([-75,-50,-25,0,25,50,75])
    add_stim_resp_win(ax)
    ax.set_xlim([-60,80])
    ax.set_title(area)
    if iarea == 0: 
        ax.set_ylabel('Weights')
        ax.legend(labels_Z,frameon=False,fontsize=8,loc='upper left')
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')
# fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_DepthTo250%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')
# fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_DepthFrom250%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')



#%% 
lam = 0.08
kfold = 5
modelname = 'LDA'
# modelname = 'LOGR'

idx_N = np.all((celldata['noise_level']<noise_level,
                np.isin(celldata['roi_name'],area),
                # celldata['depth']<250,
                # celldata['depth']>250,
                # idx_nearby
                ),axis=0)
    
# y = celldata['redcell'][idx_N].to_numpy()
y = celldata['sig_MN'][idx_N].to_numpy()
X = data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)

# X -= np.nanmean(X,axis=0,keepdims=True)

X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

# temp,_,_,_   = my_decoder_wrapper(X,y,model_name='LOGR',kfold=kfold,lam=None,norm_out=True)
temp,_,_,_   = my_decoder_wrapper(X,y,model_name=modelname,kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=True)
print(temp)

# lam = find_optimal_lambda(X,y,model_name='LDA',kfold=kfold)







#%% 
coefs = np.reshape(model.coef_,(Z,len(sbins)))
fig,ax = plt.subplots(1,1,figsize=(5,3))
for iZ in range(Z):
    ax.plot(sbins,coefs[iZ,:],color=clrs_Z[iZ], label=plotcenters[iZ],linewidth=2)

ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
ax.set_xticks([-75,-50,-25,0,25,50,75])
ax.set_xticklabels([-75,-50,-25,0,25,50,75])
ax.legend(labels_Z,frameon=False,fontsize=8,loc='upper left')
add_stim_resp_win(ax)
ax.set_xlim([-60,80])
ax.set_ylabel('LDA weights')

#%% 
plt.plot(np.nanmean(X[y==0,:],axis=0),c='b')
plt.plot(np.nanmean(X[y==1,:],axis=0),c='r')	

#%% 

plt.plot(LDAproj)
