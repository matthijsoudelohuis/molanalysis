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
from utils.plotting_style import * #get all the fixed color schemes
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
                                      min_lab_cells_V1=20,min_lab_cells_PM=20,
                                      filter_areas=['V1','PM'])

report_sessions(sessions)
# sessions,nSessions  = filter_sessions(protocol,calciumversion=calciumversion,min_lab_cells_V1=50,min_lab_cells_PM=50)

#%% 
# sessions,nSessions = load_neural_performing_sessions()

#%% ############################### Spatial Tensor #################################
## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
## Parameters for spatial binning
s_pre       = -60  #pre cm
s_post      = 90   #post cm
sbinsize    = 10     #spatial binning in cm

for i in tqdm(range(nSessions),desc='Computing spatial tensor',total=nSessions):
    sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                       sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=sbinsize,method='binmean')

#%% 
sessions = calc_stimresponsive_neurons(sessions,sbins)

#%% #################### Compute mean activity for saliency trial bins for all sessions ##################

labeled     = ['unl','lab']
nlabels     = len(labeled)
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

lickresp    = [0,1]
nlickresp   = len(lickresp)

sigtype     = 'signal'
zmin        = 5
zmax        = 20
nbins_noise = 3

# data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True)
data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=False)

Z = np.shape(data_mean_spatial_hitmiss)[1]
# plotcolors = [sns. sns.color_palette("inferno",C)
plotcolors = ['black']  # Start with black
plotcolors += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
plotcolors.append('orange')  # Add orange at the end
plothandles = ['catch','sub','thr','sup','max']

#%%
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
N           = len(celldata)
noise_level = 100

#%% Create index of nearby cells to compare to:
idx_nearby = np.zeros(N,dtype=bool)
for ses in sessions:
    idx_ses = np.where(celldata['session_id']==ses.sessiondata['session_id'][0])[0]
    idx_nearby[idx_ses] = filter_nearlabeled(ses,radius=50)
    # idx_nearby[idx_ses] = filter_nearlabeled(ses,radius=50)

#%% 
areas = ['V1','PM']
nareas = len(areas)
noise_level = 100

#%% 
fig,axes = plt.subplots(1,nareas,figsize=(4*nareas,2.5))

for iarea,area in enumerate(areas):
    ax = axes[iarea]

    idx_N = np.all((celldata['noise_level']<noise_level,
                    np.isin(celldata['roi_name'],area),
                    # celldata['depth']>300,
                    idx_nearby),axis=0)
    # idx_N = np.ones(len(celldata),dtype=bool)

    y = celldata['redcell'][idx_N].to_numpy()

    tempdata = copy.deepcopy(data_mean_spatial_hitmiss[idx_N,:,:])
    # tempdata -= np.nanmean(tempdata,axis=(1,2),keepdims=True)
    # tempdata /= np.nanstd(tempdata[:,:,:],axis=(1,2),keepdims=True)

    tempdata -= np.nanmean(tempdata[:,:,sbins<0],axis=(2),keepdims=True)
    # tempdata /= np.nanstd(tempdata[:,:,:],axis=(1,2),keepdims=True)

    for ilab in range(2):
        for iZ in range(Z):
            ax.plot(sbins,np.nanmean(tempdata[y==ilab,iZ,:],axis=0),color=plotcolors[iZ], label=plotcenters[iZ],linewidth=2,linestyle=['-','--'][ilab])
            # ax.plot(sbins,np.nanmean(tempdata[y==ilab,iZ,:],axis=0),color=plotcolors[iZ], label=plotcenters[iZ],linewidth=2,linestyle='-')

    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
    ax.set_xticks([-75,-50,-25,0,25,50,75])
    ax.set_xticklabels([-75,-50,-25,0,25,50,75])
    
    add_stim_resp_win(ax)
    ax.set_xlim([-60,80])
    if iarea == 0:
        ax.set_ylabel('Mean Activity (deconv.)\n(baseline subtracted)')
        first_legend = ax.legend(plothandles,frameon=False,fontsize=8,loc='upper left',title='Trial Type')
        leg_lines = [Line2D([0], [0], color='black', linewidth=2, linestyle='-'),
                    Line2D([0], [0], color='black', linewidth=2, linestyle='--')]
        leg_labels = ['unlabeled','labeled']
        ax.legend(leg_lines, leg_labels, loc='upper right', frameon=False, fontsize=8)
        # Manually add the first legend back to the plot
        ax.add_artist(first_legend)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'MeanAct_SpatialHitMiss_%dsessions.png' % (nSessions)), format = 'png')

#%% 
exp_label = 'Decoding_proj_type'

fig,axes = plt.subplots(1,nareas,figsize=(4*nareas,2.5))

for iarea,area in enumerate(areas):
    ax = axes[iarea]

    idx_N = np.all((celldata['noise_level']<noise_level,
                    np.isin(celldata['roi_name'],area),
                    # celldata['depth']>300,
                    idx_nearby),axis=0)
    
    y = celldata['redcell'][idx_N].to_numpy()
    X = data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)

    # X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

    lam   = 0.95
    model = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))

    LDAproj = model.fit_transform(X,y)

    coefs = np.reshape(model.coef_,(Z,len(sbins)))

    for iZ in range(Z):
        ax.plot(sbins,coefs[iZ,:],color=plotcolors[iZ], label=plotcenters[iZ],linewidth=2)

    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
    ax.set_xticks([-75,-50,-25,0,25,50,75])
    ax.set_xticklabels([-75,-50,-25,0,25,50,75])
    add_stim_resp_win(ax)
    ax.set_xlim([-60,80])
    ax.set_title(area)
    if iarea == 0: 
        ax.set_ylabel('LDA weights')
        ax.legend(plothandles,frameon=False,fontsize=8,loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'LDAweights_%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')

#%% 

data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=True)
# data_mean_spatial_hitmiss,plotcenters = get_spatial_mean_signalbins(sessions,sbins,sigtype,nbins_noise,zmin,zmax,splithitmiss=False)


#%% 
exp_label = 'Decoding_proj_type'

fig,axes = plt.subplots(1,nareas,figsize=(4*nareas,2.5))

for iarea,area in enumerate(areas):
    ax = axes[iarea]

    idx_N = np.all((celldata['noise_level']<noise_level,
                    np.isin(celldata['roi_name'],area),
                    # celldata['depth']>300,
                    idx_nearby),axis=0)
    
    y = celldata['redcell'][idx_N].to_numpy()
    X = data_mean_spatial_hitmiss[idx_N,:,:].reshape(np.sum(idx_N),-1)

    X,y,idx_nan = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

    lam   = 0.95
    model = LDA(n_components=1,solver='eigen', shrinkage=np.clip(lam,0,1))

    LDAproj = model.fit_transform(X,y)

    coefs = np.reshape(model.coef_,(Z,len(sbins),2))

    # for iZ in range(Z):
        # ax.plot(sbins,coefs[iZ,:,0],color=plotcolors[iZ], label=plotcenters[iZ],linewidth=2,linestyle=['--','-'][0])
        # ax.plot(sbins,coefs[iZ,:,1],color=plotcolors[iZ], label=plotcenters[iZ],linewidth=2,linestyle=['--','-'][1])

    ax.plot(sbins,np.nanmean(coefs[:,:,0],axis=0),color='k',linewidth=2,linestyle=['--','-'][0])
    ax.plot(sbins,np.nanmean(coefs[:,:,1],axis=0),color='k',linewidth=2,linestyle=['--','-'][1])

    ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
    ax.set_xticks([-75,-50,-25,0,25,50,75])
    ax.set_xticklabels([-75,-50,-25,0,25,50,75])
    add_stim_resp_win(ax)
    ax.set_xlim([-60,80])
    ax.set_title(area)
    if iarea == 0: 
        ax.set_ylabel('LDA weights')
        ax.legend(plothandles,frameon=False,fontsize=8,loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'LDAweights_HitMiss_%s_%dsessions.png' % (exp_label,nSessions)), format = 'png')


#%%
df = pd.DataFrame(data={'LDAproj':LDAproj.squeeze(),'y':y})
# sns.scatterplot(x='LDAproj',y='y',data=df)
sns.histplot(x='LDAproj',hue='y',data=df,kde=True)

#%%

# df = pd.DataFrame(data={'LDAproj':LDAproj.squeeze()})



#%% 
lam = 0.8
kfold = 5
# temp,_,_,_   = my_decoder_wrapper(X,y,model_name='LOGR',kfold=kfold,lam=None,norm_out=True)
temp,_,_,_   = my_decoder_wrapper(X,y,model_name='LDA',kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=True)
print(temp)

# lam = find_optimal_lambda(X,y,model_name='LDA',kfold=kfold)







#%% 
coefs = np.reshape(model.coef_,(Z,len(sbins)))
fig,ax = plt.subplots(1,1,figsize=(5,3))
for iZ in range(Z):
    ax.plot(sbins,coefs[iZ,:],color=plotcolors[iZ], label=plotcenters[iZ],linewidth=2)

ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
ax.set_xticks([-75,-50,-25,0,25,50,75])
ax.set_xticklabels([-75,-50,-25,0,25,50,75])
ax.legend(plothandles,frameon=False,fontsize=8,loc='upper left')
add_stim_resp_win(ax)
ax.set_xlim([-60,80])
ax.set_ylabel('LDA weights')

#%% 
plt.plot(np.nanmean(X[y==0,:],axis=0),c='b')
plt.plot(np.nanmean(X[y==1,:],axis=0),c='r')	

#%% 

plt.plot(LDAproj)
