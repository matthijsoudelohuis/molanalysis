# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of functions that analyze activity in visual VR detection task. 
"""

#%% Import packages
import os
os.chdir('c:\\Python\\molanalysis\\')
import numpy as np
import pandas as pd
from tqdm import tqdm

# from loaddata import * #get all the loading data functions (filter_sessions,load_sessions)
from loaddata.session_info import filter_sessions,load_sessions
from loaddata.get_data_folder import get_local_drive

from scipy import stats
from scipy.stats import zscore
from utils.psth import *
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score as AUC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches
from utils.plotting_style import * #get all the fixed color schemes
from matplotlib.lines import Line2D
from utils.behaviorlib import * # get support functions for beh analysis 
from detection.plot_neural_activity_lib import *
from detection.example_cells import get_example_cells
from utils.plot_lib import * # get support functions for plotting
from utils.regress_lib import * # get support functions for regression

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Detection\\MultiAreaRegression\\')

#%% ###############################################################

protocol            = 'DN'
calciumversion      = 'deconv'
# calciumversion      = 'dF'

session_list = np.array([['LPE12385', '2024_06_15']])
# session_list = np.array([['LPE12385', '2024_06_16']])
# session_list = np.array([['LPE12013', '2024_04_25']])
session_list = np.array([['LPE11997', '2024_04_16']])
# session_list = np.array([['LPE11998', '2024_04_30']])
# session_list = np.array([['LPE11622', '2024_02_22']])
# session_list = np.array([['LPE10884', '2023_12_15']])
# session_list = np.array([['LPE10884', '2024_01_16']])

sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_videodata=True,
                         load_calciumdata=True,calciumversion=calciumversion) #Load specified list of sessions
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE11998'],min_cells=100,
                        #    load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:
# sessions,nSessions = filter_sessions(protocol,only_animal_id=['LPE10884'],
#                            load_behaviordata=True,load_calciumdata=True,calciumversion=calciumversion) #load sessions that meet criteria:

# #%% ### Show for all sessions which region of the psychometric curve the noise spans #############
# sessions = noise_to_psy(sessions,filter_engaged=True)

# idx_inclthr = np.empty(nSessions).astype('int')
# for ises,ses in enumerate(sessions):
#     idx_inclthr[ises] = int(np.logical_and(np.any(sessions[ises].trialdata['signal_psy']<=0),np.any(sessions[ises].trialdata['signal_psy']>=0)))
#     ses.sessiondata['incl_thr'] = idx_inclthr[ises]

# sessions = [ses for ises,ses in enumerate(sessions) if ses.sessiondata['incl_thr'][0]]
# nSessions = len(sessions)

#%% 
for i in range(nSessions):
    sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)


#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -80  #pre cm
s_post      = 60   #post cm
binsize     = 5     #spatial binning in cm

for i in range(nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')

    # Compute average response in stimulus response zone:
    sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                    sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
    sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
                                    sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

    # temp = pd.DataFrame(np.reshape(np.array(sessions[i].videodata['motionenergy']),(len(sessions[0].videodata['motionenergy']),1)))
    # sessions[i].respmat_videome     = compute_respmat_space(temp, sessions[i].videodata['ts'], sessions[i].trialdata['stimStart'],
    #                                 sessions[i].videodata['zpos'],sessions[i].videodata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

# data = sessions[i].videodata['motionenergy']
# ts_F = sessions[i].videodata['ts']
# z_T = sessions[i].trialdata['stimStart']
# zpos_F = sessions[i].videodata['zpos']
# trialnum_F = sessions[i].trialnum_F
# s_resp_start=0
# s_resp_stop=20
# method='mean'
# subtr_baseline=False


#%% ############################### Plot neuron-average per stim per area #################################
ises = 0 #selected session to plot this for

labeled     = ['unl','lab']
nlabels     = 2
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

clrs_vars = sns.color_palette('inferno', 3)

fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*3,nlabels*2.5),sharex=True,sharey=True)
S = len(sbins)
# plt.plot(sbins,r2_cv.mean(axis=0),color='k',linewidth=2)
for ilab,label in enumerate(labeled):
    for iarea, area in enumerate(areas):
        ax      = axes[ilab,iarea]
        idx_N     = np.all((sessions[ises].celldata['roi_name']==area, sessions[ises].celldata['labeled']==label), axis=0)
        
        nbins_noise     = 5
        C               = nbins_noise + 2
        noise_signal    = sessions[ises].trialdata['signal'][sessions[ises].trialdata['stimcat']=='N'].to_numpy()
        
        plotdata        = np.empty((C,S))
        idx_T           = sessions[ises].trialdata['signal']==0
        plotdata[0,:]   = np.nanmean(np.nanmean(sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)
        idx_T           = sessions[ises].trialdata['signal']==100
        plotdata[-1,:]   = np.nanmean(np.nanmean(sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)

        edges = np.linspace(np.min(noise_signal),np.max(noise_signal),nbins_noise+1)
        centers = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)

        for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
            
            idx_T           =  (sessions[ises].trialdata['signal']>=low) & (sessions[ises].trialdata['signal']<=high)
            plotdata[ibin+1,:]   = np.nanmean(np.nanmean(sessions[ises].stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)

            plotlabels = np.round(np.hstack((0,centers,100)))
            plotcolors = ['black']  # Start with black
            plotcolors += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
            plotcolors.append('orange')  # Add orange at the end

        for iC in range(C):
            ax.plot(sbins, plotdata[iC,:], color=plotcolors[iC], label=plotlabels[iC],linewidth=2)

        add_stim_resp_win(ax)

        ax.set_ylim([-0.1,0.75])
        if ilab == 0 and iarea == 0:
            ax.legend(frameon=False,fontsize=6)
        ax.set_xlim([-60,60])
        if ilab == 0:
            ax.set_title(area)
        if ilab == 1:
            ax.set_xlabel('Position relative to stim (cm)')
        if iarea==0:
            ax.set_ylabel('Activity (z)')
            ax.set_yticks([0,0.25,0.5])
plt.tight_layout()

# plt.savefig(os.path.join(savedir,'ActivityInCorridor_neuronAverage_arealabels_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
# plt.savefig(os.path.join(savedir,'ActivityInCorridor_deconv_neuronAverage_perStim_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% ####################### PCA to understand variability at the population level ####################

def pca_scatter_stimresp(respmat,ses,colorversion='stimresp'):
    stimtypes   = sorted(ses.trialdata['stimcat'].unique()) # stim
    resptypes   = sorted(ses.trialdata['lickResponse'].unique()) # licking resp [0,1]

    S           = copy.deepcopy(ses.trialdata['stimcat'].to_numpy())
    SIG         = copy.deepcopy(ses.trialdata['signal'].to_numpy())
    R           = copy.deepcopy(ses.trialdata['lickResponse'].to_numpy())
    C           = np.squeeze(ses.respmat_runspeed)
    X           = copy.deepcopy(respmat)

    idx_valid   = np.logical_not(np.any(np.isnan(X),axis=0))
    X           = X[:,idx_valid]
    S           = S[idx_valid]
    SIG         = SIG[idx_valid]
    R           = R[idx_valid]
    C           = C[idx_valid]
    X           = zscore(X,axis=1)

    pca         = PCA(n_components=3)
    Xp          = pca.fit_transform(X.T).T

    s_type_ind      = [np.argwhere(S == stimtype)[:, 0] for stimtype in stimtypes]
    r_type_ind      = [np.argwhere(R == resptype)[:, 0] for resptype in resptypes]

    pal             = sns.color_palette('husl', 4)
    fc              = ['w','k']
    # cmap            = plt.get_cmap('viridis')
    # cmap = plt.get_cmap('gist_rainbow')
    cmap = plt.get_cmap('jet')

    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=[12, 4], sharey='row', sharex='row')
    for ax, proj in zip(axes, projections):

        if colorversion=='stimresp':
            for s in range(len(stimtypes)):
                for r in range(len(resptypes)):
                    x = Xp[proj[0], np.intersect1d(s_type_ind[s],r_type_ind[r])]
                    y = Xp[proj[1], np.intersect1d(s_type_ind[s],r_type_ind[r])]
                  
                    ax.scatter(x, y, s=20, alpha=0.8,marker='o',facecolors=pal[s],edgecolors=fc[r],linewidths=1)
                   
            custom_lines = [Line2D([0], [0], color=pal[k], lw=0,markersize=10,marker='o') for
                                k in range(len(stimtypes))]
            labels = stimtypes
            ax.legend(custom_lines, labels,title='Stim',
                    frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        elif colorversion=='runspeed':
            
            x = Xp[proj[0],:]
            y = Xp[proj[1],:]
            
            sc = ax.scatter(x, y, c=C, vmin=np.percentile(C,1), vmax=np.percentile(C,99), s=20, marker='o',edgecolors='w',linewidths=1,cmap=cmap)
            
            # Create colorbar#
            cbar = plt.colorbar(sc, ax=ax,shrink=0.3)#+
            cbar.set_label('Runspeed (cm/s)', rotation=270, labelpad=20)#+
            # cbar.set_ticks(np.percentile(C,[1,50,99]))
            cbar.set_ticks(np.round(np.percentile(C,[1,50,99]),1))

        elif colorversion=='signal':
          
            x = Xp[proj[0],:]
            y = Xp[proj[1],:]

            # Get unique sorted values
            unique_sorted_values = np.sort(np.unique(SIG))
            # Create a mapping from value to ordinal index
            value_to_ordinal = {value: index for index, value in enumerate(unique_sorted_values)}
            # Convert the array to ordinal values
            ordinal_array = np.array([value_to_ordinal[value] for value in SIG])

            sc = ax.scatter(x, y, c=ordinal_array, vmin=np.percentile(ordinal_array,1), vmax=np.percentile(ordinal_array,99), s=20, marker='o',edgecolors='w',linewidths=1,cmap=cmap)
            
            # Create colorbar#
            cbar = plt.colorbar(sc, ax=ax,shrink=0.3)#+
            cbar.set_label('Signal (norm)', rotation=270, labelpad=20)#+
            cbar.set_ticks(np.round(np.percentile(ordinal_array,[1,50,99]),1))
            cbar.set_ticklabels(['Catch','Noise','Max'])
            
    ax.set_xlabel('PC {}'.format(proj[0]+1))
    ax.set_ylabel('PC {}'.format(proj[1]+1))

    sns.despine(fig=fig, top=True, right=True)

    plt.tight_layout(rect=[0,0,0.9,1])

    return fig

#%% 
sesidx = 0
#For all areas:
fig = pca_scatter_stimresp(sessions[sesidx].respmat,sessions[sesidx],colorversion='stimresp')
plt.suptitle('stimresp',fontsize=14)
# plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_stimResp_allAreas_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = pca_scatter_stimresp(sessions[sesidx].respmat,sessions[sesidx],colorversion='signal')
plt.suptitle('signal',fontsize=14)
# plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_signal_allAreas_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = pca_scatter_stimresp(sessions[sesidx].respmat,sessions[sesidx],colorversion='runspeed')
plt.suptitle('runspeed',fontsize=14)
# plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_runspeed_allAreas_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% 
sesidx = 0

#For each area:
for iarea,area in enumerate(areas):
    # idx         = sessions[sesidx].celldata['roi_name'] == area
    idx         = np.all((sessions[sesidx].celldata['roi_name']==area, sessions[sesidx].celldata['noise_level']<20), axis=0)
    # respmat     = np.nanmean(sessions[sesidx].stensor[np.ix_(idx,range(K),(sbins>0) & (sbins<20))],axis=2) 
    respmat     = sessions[sesidx].respmat[idx,:]
    
    fig = pca_scatter_stimresp(respmat,sessions[sesidx],colorversion='stimresp')
    plt.suptitle(area + ' stimresp',fontsize=14)
    plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_stimResp_' + area + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    fig = pca_scatter_stimresp(respmat,sessions[sesidx],colorversion='runspeed')
    plt.suptitle(area + ' runspeed',fontsize=14)
    plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_runspeed_' + area + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    fig = pca_scatter_stimresp(respmat,sessions[sesidx],colorversion='signal')
    plt.suptitle(area + ' signal',fontsize=14)
    plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_signal_' + area + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    # pca_scatter_stimresp(respmat,sessions[sesidx])
    # plt.suptitle(area,fontsize=14)
    # plt.savefig(os.path.join(savedir,'PCA','PCA_Scatter_stimResponse_' + area + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ###############################################################


#%% ################## PCA unsupervised display of noise around center for each condition #################
# Filter only noisy threshold trials:
idx_T   = sessions[sesidx].trialdata['stimcat']=='N'
idx_N   = np.where(sessions[sesidx].celldata['roi_name']=='V1')[0]

A1      = sessions[sesidx].respmat[np.ix_(idx_N,idx_T)].T

S       = np.column_stack((sessions[sesidx].trialdata['signal'],
                 sessions[sesidx].trialdata['lickResponse'],
               sessions[sesidx].respmat_runspeed.flatten()))

slabels     = ['Signal','Licking','Running']
# slabels     = ['Signal','Licking','Running','MotionEnergy']

S           = S[idx_T,:]
S           = zscore(S,axis=0,nan_policy='omit')
A1,S        = prep_Xpredictor(A1,S)


df = pd.DataFrame(data=S, columns=slabels)
fig, ax = plt.subplots(figsize=(4,4))         # Sample figsize in inches
sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap="vlag",ax=ax,annot=True)

# Define neural data parameters
N1,K        = np.shape(A1)
# N2          = np.shape(A2)[0]
NS          = np.shape(S)[1]

cmap = plt.get_cmap('viridis')
cmap = plt.get_cmap('plasma')
# cmap = plt.get_cmap('Spectral')
projections = [(0, 1), (1, 2), (0, 2)]
# projections = [(0, 3), (3, 4), (2, 3)]

fig, axes = plt.subplots(NS, len(projections), figsize=[len(projections)*2, NS*2])
for iSvar in range(NS):
    # X           = zscore(A1,axis=0)
    pca         = PCA(n_components=3) #construct PCA object with specified number of components
    Xp          = pca.fit_transform(A1) #fit pca to response matrix (n_samples by n_features)
    #dimensionality is now reduced from N by K to ncomp by K
    
    for iproj, proj in enumerate(projections):
        ax = axes[iSvar, iproj]
        x = Xp[:,proj[0]]                          #get all data points for this ori along first PC or projection pairs
        y = Xp[:,proj[1]]                          #get all data points for this ori along first PC or projection pairs
        
        temp = np.clip(S[:,iSvar],np.percentile(S[:,iSvar],1),np.percentile(S[:,iSvar],99))
        # c = cmap(minmax_scale(temp, feature_range=(0, 1)))[:,:3]
        sns.scatterplot(x=x, y=y, c=temp,cmap=cmap,vmin=np.percentile(temp,1), vmax=np.percentile(temp,99),ax = ax,s=10,legend = False,edgecolor =None)

        if iproj==0:
            ax.set_ylabel('PC {}'.format(proj[1]+1))
        if iSvar==NS-1:
            ax.set_xlabel('PC {}'.format(proj[0]+1))

        if iproj==1:
            ax.set_title(slabels[iSvar],fontsize=12)
    sns.despine(fig=fig, top=True, right=True)
plt.tight_layout()

# plt.savefig(os.path.join(savedir,'PCA' + str(proj) + '_perStim_color' + slabels[iSvar] + '.png'), format = 'png')



#%% ############################# Trial-concatenated PCA ########################################

def pca_line_stimresp(data,trialdata,spatbins):
    [N,K,S]         = np.shape(data) #get dimensions of tensor

    # collapse to 2d: N x K*T (neurons by timebins of different trials concatenated)
    X               = np.reshape(data,(N,-1))
    
    #Impute missing nan data, otherwise problems with PCA
    imp_mean        = SimpleImputer(missing_values=np.nan, strategy='mean')
    #apply imputation, replacing nan with mean of that neurons' activity
    X               = imp_mean.fit_transform(X.T).T 

    X               = zscore(X,axis=1) #score each neurons activity (along rows)

    pca             = PCA(n_components=15) #construct PCA
    Xp              = pca.fit_transform(X.T).T #PCA function assumes (samples x features)

    Xp              = np.reshape(Xp,(15,K,S)) #reshape back to trials

    #Get indices of trialtypes and responses:
    stimtypes       = sorted(trialdata['stimcat'].unique()) # stim ['A','B','C','D']
    resptypes       = sorted(trialdata['lickResponse'].unique()) # licking resp [0,1]

    s_type_ind      = [np.argwhere(np.array(trialdata['stimcat']) == stimtype)[:, 0] for stimtype in stimtypes]
    r_type_ind      = [np.argwhere(np.array(trialdata['lickResponse']) == resptype)[:, 0] for resptype in resptypes]

    #For line make-up:
    pal             = sns.color_palette('husl', 4)
    sty             = [':','-']
    patchcols       = ["cyan","green"]

    nPlotPCs        = 5 #how many subplots to create for diff PC projections

    fig, axes = plt.subplots(nPlotPCs, 1, figsize=[8, 7], sharey='row', sharex='row')
    projections = np.arange(nPlotPCs)
    for ax, proj in zip(axes, projections):
        for s in range(len(stimtypes)):
            for r in range(len(resptypes)):
                #Take the average PC projection across all indexed trials:
                y   = np.mean(Xp[proj, np.intersect1d(s_type_ind[s],r_type_ind[r]),:],axis=0)
                ax.plot(spatbins,y,c=pal[s],linestyle=sty[r])
                if proj == nPlotPCs-1:
                    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
                ax.set_ylabel('PC {}'.format(proj + 1))
        
        ax.set_xticks(np.linspace(-50,50,5))
        ax.add_patch(matplotlib.patches.Rectangle((0,ax.get_xlim()[0]),20,np.diff(ax.get_xlim())[0], 
                    fill = True, alpha=0.2,
                    color = patchcols[0], linewidth = 0))
        ax.add_patch(matplotlib.patches.Rectangle((25,ax.get_xlim()[0]),20,np.diff(ax.get_xlim())[0], 
                    fill = True, alpha=0.2,
                    color = patchcols[1], linewidth = 0))

    sns.despine(fig=fig, top=True, right=True)

    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(stimtypes))]
    labels = stimtypes
    ax.legend(custom_lines, labels,title='Stim',
            frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])


ises            = 1 #selected session to plot this for
[N,K,S]         = np.shape(sessions[ises].stensor) #get dimensions of tensor

#For all areas:
binsubidx   = (sbins>-60) & (sbins<=60)
binsub      = sbins[binsubidx]
data        = sessions[ises].stensor[:,:,binsubidx]
pca_line_stimresp(data,sessions[ises].trialdata,binsub)
# plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_allAreas_' + sessions[0].sessiondata['session_id'][0] + '.svg'), format = 'svg')
plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_allAreas_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#For each area:
for iarea,area in enumerate(areas):
    idx         = sessions[ises].celldata['roi_name'] == area
    data        = sessions[ises].stensor[np.ix_(idx,range(K),binsubidx)]
    pca_line_stimresp(data,sessions[ises].trialdata,binsub)
    plt.suptitle(area,fontsize=14)
    # plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.svg'), format = 'svg')
    plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')
    # plt.savefig(os.path.join(savedir,'PCA_Line_stimResponse_Left_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% #### PCA on different stimuli, conditioned on the other corridor stimulus:


#%% ############################################# Regression ##################################################


#%% ################## Regression display of noise around center for each condition #################

# Filter only noisy threshold trials:
idx_T   = sessions[sesidx].trialdata['stimcat']=='N'
idx_N   = np.where(sessions[sesidx].celldata['roi_name']=='V1')[0]

A1      = sessions[sesidx].respmat[np.ix_(idx_N,idx_T)].T

S       = np.column_stack((sessions[sesidx].trialdata['signal'],
                 sessions[sesidx].trialdata['lickResponse'],
               sessions[sesidx].respmat_runspeed.flatten()))

# slabels     = ['Signal','Licking','Running','MotionEnergy']

S           = S[idx_T,:]
S           = zscore(S,axis=0,nan_policy='omit')
A1,S        = prep_Xpredictor(A1,S)

df = pd.DataFrame(data=S, columns=slabels)
fig, ax = plt.subplots(figsize=(4,4))         # Sample figsize in inches
sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap="vlag",ax=ax,annot=True)


#%% Decoding variables from V1 activity across space:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# modelname       = 'Lasso' # Linear regression with Lasso (L1) regularization
model_name_cont  = 'Ridge'
model_name_disc  = 'LogisticRegression'
scoring_type    = 'r2_score'
lam             = None
kfold           = 5

slabels         = ['Signal','Hit/Miss','Running']
NS              = len(slabels)
N               = len(celldata)
nBins           = len(sbins)

weights         = np.full((NS,nBins,N),np.nan)
error_cv        = np.full((NS,nBins,nSessions),np.nan)

area            = 'V1'
# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding across sessions'):
    # idx_ses = np.where(celldata['session_id']==ses.sessiondata['session_id'][0])
    idx_T = np.isin(ses.trialdata['stimcat'],['N'])
    assert np.sum(idx_T) > 50, 'Not enough trials in session %d' % ses.sessiondata['session_id'][0]
    idx_N = ses.celldata['roi_name']==area
    idx_N = np.ones(len(ses.celldata)).astype(bool)
    # idx_N_ses = celldata['cell_id'] in ses.celldata['cell_id'][idx_N]
    idx_N_ses = np.isin(celldata['cell_id'],ses.celldata['cell_id'][idx_N])
    
    A1      = sessions[ises].respmat[np.ix_(idx_N,idx_T)].T

    S       = np.column_stack((sessions[ises].trialdata['signal'],
                    sessions[ises].trialdata['lickResponse'],
                sessions[ises].respmat_runspeed.flatten()))
    S       = S[idx_T,:]

    for iS in range(NS):
        A1,S        = prep_Xpredictor(A1,S)
        y           = S[:,iS]
        datatype = 'disc' if np.all(np.isin(y,[0,1])) else 'cont'
        if datatype=='disc':
            y           = y.astype(int)
            model_name = model_name_disc
        else:
            y           = zscore(y,axis=0,nan_policy='omit')
            model_name = model_name_cont
            
        # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
        X = np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,((sbins>-5) & (sbins<20)).astype(bool))],axis=2)
        X = X.T # Transpose to K x N (samples x features)

        X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        if lam is None:
            lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

        # Loop through each spatial bin
        for ibin, bincenter in enumerate(sbins):
            y       = S[:,iS]
            X       = ses.stensor[np.ix_(idx_N,idx_T,sbins==bincenter)].squeeze()
            X       = X.T
            X,y     = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            
            error_cv[iS,ibin,ises],weights[iS,ibin,idx_N_ses],_ = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,lam=lam,
                                                        scoring_type=scoring_type,norm_out=False,subtract_shuffle=False) 


#%% 

#%% Show the decoding performance per session 
fig,axes = plt.subplots(1,1,figsize=(3,2.5),sharex=True,sharey=True)
for i,ses in enumerate(sessions):
    ax = axes
    # ax = axes[i//nperrow,i%nperrow]
    for iS in range(NS):
        ax.plot(sbins,error_cv[iS,:,ises],alpha=0.5,linewidth=1.5,color=clrs_vars[iS])
        # ax.plot(sbins,error_cv[i,:,ises],alpha=0.5,linewidth=1.5,color=clrs_vars[iS])
        # ax.plot(sbins,dec_perf_choice[i,:],alpha=0.5,linewidth=1.5,color='g')
    # shaded_error(sbins,sperf,error='sem',ax=ax,color='grey')
    add_stim_resp_win(ax)
    ax.set_title(ses.sessiondata['session_id'][0])

    ax.set_xticks([-50,-25,0,25,50])
    ax.set_ylim([-0.6,1])
    ax.set_xlim([-60,60])
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=1)
    if i == 0:
        ax.set_ylabel('Performance \n (cv R2)')
    if i==int(nSessions/2):
        ax.set_xlabel('Position relative to stim (cm)')
    if i == 0:
        ax.legend(slabels,frameon=False,fontsize=7,title='Decoding')
plt.tight_layout()
# plt.savefig(os.path.join(savedir, 'Spatial', 'DecPerformance_Stim_Resp_indSes.png'), format='png')











#%% Decoding variables from V1 activity across space:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
for ses in sessions:
    ses.trialdata['trial_id'] = np.array([ses.sessiondata['session_id'][0] + '_' + '%04.0f' % k for k in range(0,len(ses.trialdata))])

trialdata = pd.concat([ses.trialdata for ses in sessions]).reset_index(drop=True)

modelname       = 'Lasso' # Linear regression with Lasso (L1) regularization
# model_name_cont  = 'Ridge'
model_name_disc  = 'LogisticRegression'
# model_name_disc  = 'LDA'
scoring_type    = 'r2_score'
kfold           = 5
lam             = None
lam             = 0.05

slabels         = ['Signal','Hit/Miss','Running']
NS              = len(slabels)

N               = len(celldata)
K               = len(trialdata)
areas           = ['V1','PM','AL','RSP']
nareas          = len(areas)

weights         = np.full((NS,N),np.nan)
error_cv        = np.full((NS,nareas,nSessions),np.nan)
projs           = np.full((NS,nareas,K),np.nan)

# Loop through each session
for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Decoding across sessions'):
    # idx_ses = np.where(celldata['session_id']==ses.sessiondata['session_id'][0])
    idx_T = np.isin(ses.trialdata['stimcat'],['N'])
    idx_T_ses = np.isin(trialdata['trial_id'],ses.trialdata['trial_id'][idx_T])

    assert np.sum(idx_T) > 50, 'Not enough trials in session %d' % ses.sessiondata['session_id'][0]
    for iarea,area in enumerate(areas):
        idx_N = ses.celldata['roi_name']==area
        # idx_N = np.all((ses.celldata['roi_name']==area,
                        # np.logical_or(ses.celldata['sig_N']==1,ses.celldata['sig_M']==1)),axis=0)
        # idx_N = np.ones(len(ses.celldata)).astype(bool)
        # idx_N_ses = celldata['cell_id'] in ses.celldata['cell_id'][idx_N]
        idx_N_ses = np.isin(celldata['cell_id'],ses.celldata['cell_id'][idx_N])

        A1      = sessions[ises].respmat[np.ix_(idx_N,idx_T)].T

        for iS in range(NS):
            S       = np.column_stack((sessions[ises].trialdata['signal'],
                        sessions[ises].trialdata['lickResponse'],
                    sessions[ises].respmat_runspeed.flatten()))
            S       = S[idx_T,:]

            A1,S        = prep_Xpredictor(A1,S)
            y           = S[:,iS]
            datatype = 'disc' if np.all(np.isin(y,[0,1])) else 'cont'
            if datatype=='disc':
                y           = y.astype(int)
                model_name = model_name_disc
            else:
                y           = zscore(y,axis=0,nan_policy='omit')
                model_name = model_name_cont
                
            # X = ses.stensor[np.ix_(idx_N,idx_T,np.ones(len(sbins)).astype(bool))]
            # X = np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,((sbins>-5) & (sbins<20)).astype(bool))],axis=2)
            X = ses.respmat[np.ix_(idx_N,idx_T)]
            X = X.T # Transpose to K x N (samples x features)

            X,y = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            if lam is None:
                lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

            error_cv[iS,iarea,ises],weights[iS,idx_N_ses],projs[iS,iarea,idx_T_ses] = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,lam=lam,
                                                        scoring_type=scoring_type,norm_out=False,subtract_shuffle=False) 


#%% 
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.rad2deg(angle_rad)

def angles_between(v):
    """ Returns the angle in degrees between each of the columns in vector array v:
    """
    angles = np.full((v.shape[1],v.shape[1]), np.nan)
    for i in range(v.shape[1]):
        for j in range(i+1,v.shape[1]):
            angles[i,j] = angle_between(v[:,i],v[:,j])
            angles[j,i] = angles[i,j]
    return angles

#%% Angle between weights of decoded variables for each area
fig,axes = plt.subplots(2,2,figsize=(6,5))         # Sample figsize in inches
for i,area in enumerate(areas):
    ax = axes[i//2,i%2]

    v = weights[:,celldata['roi_name']==area].T
    # v = np.abs(weights[:,celldata['roi_name']==area].T)
    
    sns.heatmap(angles_between(v),xticklabels=slabels,yticklabels=slabels,vmin=0,vmax=110,
                cmap="vlag",fmt="3.1f",ax=ax,annot=True)
    ax.set_title(area)
plt.tight_layout()
# sns.heatmap(df.corr(),vmin=-0.5,vmax=0.5,cmap="vlag",ax=ax,annot=True)
fig.savefig(os.path.join(savedir, 'AngleBetweenWeights_DecVars_%s.png') % (sessions[ises].sessiondata['session_id'][0]), format = 'png')


#%% 

# df = pd.DataFrame(weights.T,columns=slabels)
df = pd.DataFrame(np.abs(weights).T,columns=slabels)
df['area'] = celldata['roi_name']
fig,axes = plt.subplots(2,2,figsize=(6,5))         # Sample figsize in inches
for i,area in enumerate(areas):
    ax = axes[i//2,i%2]
    sns.heatmap(df.loc[df['area']==area,slabels].corr(),vmin=-0.5,vmax=0.5,cmap="vlag",ax=ax,annot=True)
    ax.set_title(area)
plt.tight_layout()
fig.savefig(os.path.join(savedir, 'CorrWeights_DecVars_%s.png') % (sessions[ises].sessiondata['session_id'][0]), format = 'png')

#%% 

fig,axes = plt.subplots(1,3,figsize=(9,2.5))         # Sample figsize in inches
for iS,slabel in enumerate(slabels):
    ax = axes[iS]

    # df = pd.DataFrame(weights.T,columns=slabels)
    df = pd.DataFrame(projs[iS,:,:].T,columns=areas)

    sns.heatmap(df.corr(),vmin=-0.5,vmax=0.5,cmap="vlag",ax=ax,annot=False)
    ax.set_title(slabel)
plt.suptitle('Correlations of trial-projected data between areas') 
plt.tight_layout()
fig.savefig(os.path.join(savedir, 'CorrProjectedData_Areas_%ssessions.png') % (sessions[ises].sessiondata['session_id'][0]), format = 'png')

#%% Plot signal dimension in PCA space
fig,axes = plt.subplots(1,3,figsize=(9,2.5))         # Sample figsize in inches
for iS,slabel in enumerate(slabels):
    ax = axes[iS]


NEED TO FINISH THIS.... 



#%% 

#%%  ############################# Trial-concatenated sliding LDA  ########################################
def lda_line_stimresp(data,trialdata,sbins):
    [N,K,S]         = np.shape(data) #get dimensions of tensor

    # collapse to 2d: N x K*T (neurons by timebins of different trials concatenated)
    X               = np.reshape(data,(N,-1))
    # Impute missing nan data, otherwise problems with LDA
    imp_mean        = SimpleImputer(missing_values=np.nan, strategy='mean')
    # apply imputation, replacing nan with mean of that neurons' activity
    X               = imp_mean.fit_transform(X.T).T 
    #Z-score each neurons activity (along rows)
    X               = zscore(X,axis=1)

    respmat_stim        = np.nanmean(data[:,:,(sbins>=0) & (sbins<20)],axis=2) 
    respmat_dec         = np.nanmean(data[:,:,(sbins>=25) & (sbins<45)],axis=2) 

    vec_stim            = trialdata['stimcat']     == 'M'
    vec_dec             = trialdata['lickResponse']  == 1

    lda_stim            = LDA(n_components=1)
    lda_stim.fit(respmat_stim.T, vec_stim)
    Xp_stim             = lda_stim.transform(X.T)
    Xp_stim             = np.reshape(Xp_stim,(K,S)) #reshape back to trials by spatial bins

    lda_dec             = LDA(n_components=1)
    lda_dec.fit(respmat_dec.T, vec_dec)
    Xp_dec              = lda_dec.transform(X.T)
    Xp_dec              = np.reshape(Xp_dec,(K,S)) #reshape back to trials by spatial bins

    stim_axis     = unit_vector(lda_stim.coef_[0])
    dec_axis      = unit_vector(lda_dec.coef_[0])

    print('%f degrees between STIM and DEC axes' % angle_between(stim_axis, dec_axis).round(2))

    #Get indices of trialtypes and responses:
    stimtypes       = sorted(trialdata['stimcat'].unique()) # stim ['A','B','C','D']
    # stimtypes       = np.array(['C','M'])
    resptypes       = sorted(trialdata['lickResponse'].unique()) # licking resp [0,1]

    s_type_ind      = [np.argwhere(np.array(trialdata['stimcat']) == stimtype)[:, 0] for stimtype in stimtypes]
    r_type_ind      = [np.argwhere(np.array(trialdata['lickResponse']) == resptype)[:, 0] for resptype in resptypes]

    #For line make-up:
    pal             = sns.color_palette('muted', 5)
    sty             = [':','-']
    patchcols       = ["cyan","green"]

    fig, axes = plt.subplots(2, 1, figsize=[5,4], sharey='row', sharex='row')
    for ax,data in zip(axes,[Xp_stim,Xp_dec]):
        for s in range(len(stimtypes)):
            for r in range(len(resptypes)):
                #Take the average LDA projection across all indexed trials:
                # ax.plot(sbins,Xp_stim[np.intersect1d(s_type_ind[s],r_type_ind[r]),:])
                y           = np.mean(data[np.intersect1d(s_type_ind[s],r_type_ind[r]),:],axis=0)
                y_err       = np.std(data[np.intersect1d(s_type_ind[s],r_type_ind[r]),:],axis=0) / np.sqrt(len(np.intersect1d(s_type_ind[s],r_type_ind[r])))
                ax.plot(sbins,y,c=pal[s],linestyle=sty[r])
                ax.fill_between(sbins,y-y_err,y+y_err,color=pal[s],alpha=0.4)
        
        ax.set_xticks(np.linspace(-50,50,5))
        add_stim_resp_win(ax)

    axes[0].set_ylabel(r'Proj. $LDA_{STIM}$')
    axes[1].set_ylabel(r'Proj. $LDA_{DEC}$')

    sns.despine(fig=fig, top=True, right=True)

    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(stimtypes))]
    labels = stimtypes
    ax.legend(custom_lines, labels,title='Stim',
            frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,0.9])


ises            = 0 #selected session to plot this for
[N,K,S]         = np.shape(sessions[ises].stensor) #get dimensions of tensor

## For all areas:
# binsubidx   = (sbins>-60) & (sbins<=40)
# binsub      = sbins[binsubidx]
idx_T       = np.isin(sessions[ises].trialdata['stimcat'],['C','M'])
idx_T       = np.isin(sessions[ises].trialdata['stimcat'],['C','N','M'])

idx_N       = np.isin(sessions[ises].celldata['roi_name'],areas)
idx_N       = np.ones(len(sessions[ises].celldata['roi_name'])).astype(bool)
idx_S       = np.ones(len(sbins)).astype(bool)
# trialidx    = np.logical_and(trialidx,sessions[ises].trialdata['engaged']==1)
# trialidx    = np.isin(sessions[ises].trialdata['stimcat'],['C','M'])

data        = sessions[ises].stensor[np.ix_(idx_N,idx_T,idx_S)]

lda_line_stimresp(data,sessions[ises].trialdata[idx_T],sbins)
# plt.savefig(os.path.join(savedir,'LDA_Line_stimResponse_allAreas_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#For each area:
for iarea,area in enumerate(areas):
    idx_N         = sessions[ises].celldata['roi_name'] == area
    data            = sessions[ises].stensor[np.ix_(idx_N,idx_T,binsubidx)]
    lda_line_stimresp(data,sessions[ises].trialdata[idx_T],binsub)
    plt.suptitle(area,fontsize=14)
    # plt.savefig(os.path.join(savedir,'LDA_Line_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.svg'), format = 'svg')
    # plt.savefig(os.path.join(savedir,'LDA_Line_stimResponse_' + area + '_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% ################### LDA correlation in projection across areas #################

#take mean response from trials with contralateral A / B stimuli:
respmat_stim     = np.nanmean(sessions[0].stensor[np.ix_(range(N),trialidx,(sbins>0) & (sbins<25))],axis=2) 
respmat_dec      = np.nanmean(sessions[0].stensor[np.ix_(range(N),trialidx,(sbins>25) & (sbins<50))],axis=2) 
trialdata        = sessions[0].trialdata[trialidx]

stim_vec         = trialdata['stimRight'] == 'A'
dec_vec          = trialdata['lickResponse'] == 1

LDAstim_proj_A   = np.empty((np.sum(stim_vec==True),len(areas)))
LDAstim_proj_B   = np.empty((np.sum(stim_vec==False),len(areas)))
LDAdec_proj_0    = np.empty((np.sum(dec_vec==False),len(areas)))
LDAdec_proj_1    = np.empty((np.sum(dec_vec==True),len(areas)))

#For each area:
for iarea,area in enumerate(areas):
    idx                     = sessions[0].celldata['roi_name'] == area
    data                    = respmat_stim[idx,:]
    data                    = zscore(data,axis=1) #score each neurons activity (along rows)

    lda_stim                = LDA(n_components=1)
    lda_stim.fit(data.T, stim_vec)
    LDAstim_proj_A[:,iarea]   = lda_stim.transform(data[:,stim_vec==True].T).reshape(1,-1)
    LDAstim_proj_B[:,iarea]   = lda_stim.transform(data[:,stim_vec==False].T).reshape(1,-1)

    data                    = respmat_dec[idx,:]
    data                    = zscore(data,axis=1) #score each neurons activity (along rows)

    lda_dec                = LDA(n_components=1)
    lda_dec.fit(data.T, dec_vec)
    LDAdec_proj_0[:,iarea]   = lda_dec.transform(data[:,dec_vec==False].T).reshape(1,-1)
    LDAdec_proj_1[:,iarea]   = lda_dec.transform(data[:,dec_vec==True].T).reshape(1,-1)


df_stim_A     = pd.DataFrame(data=LDAstim_proj_A,columns=areas)
df_stim_B     = pd.DataFrame(data=LDAstim_proj_B,columns=areas)
df_dec_0      = pd.DataFrame(data=LDAdec_proj_0,columns=areas)
df_dec_1      = pd.DataFrame(data=LDAdec_proj_1,columns=areas)

sns.scatterplot(data = df_stim_A,x='V1',y='PM')
plt.title(r'$LDA_{STIM-A}$ projection interarea correlation')
# to do index based on area
plt.text(x=np.percentile(LDAstim_proj_A[:,0],90),y=np.percentile(LDAstim_proj_A[:,0],5),s='r = %.2f' % np.corrcoef(LDAstim_proj_A[:,0],LDAstim_proj_A[:,1])[0,1])
plt.savefig(os.path.join(savedir,'LDA_STIMA_proj_scatter_V1PM_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

fig,axes = plt.subplots(2,2,figsize=(9,6))

sns.heatmap(df_stim_A.corr(),vmin=-1,vmax=1,cmap="vlag",ax=axes[0,0])
axes[0,0].set_title(r'$LDA_{STIM-A}$')

sns.heatmap(df_stim_B.corr(),vmin=-1,vmax=1,cmap="vlag",ax=axes[0,1])
axes[0,1].set_title(r'$LDA_{STIM-B}$')

sns.heatmap(df_dec_0.corr(),vmin=-1,vmax=1,cmap="vlag",ax=axes[1,0])
axes[1,0].set_title(r'$LDA_{DEC-0}$')

sns.heatmap(df_dec_1.corr(),vmin=-1,vmax=1,cmap="vlag",ax=axes[1,1])
axes[1,1].set_title(r'$LDA_{DEC-1}$')

plt.suptitle('LDA projection interarea cross correlation')
# plt.savefig(os.path.join(savedir,'LDA_proj_corr_interarea_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')
plt.savefig(os.path.join(savedir,'LDA_proj_deconv_corr_interarea_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% #################################







# %%
