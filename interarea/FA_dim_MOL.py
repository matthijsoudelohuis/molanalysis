# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:53:22 2022

@author: joana
@author: Matthijs Oude Lohuis, 2023, Champalimaud Research
"""

#%% IMPORT RELEVANT PACKAGES

# import h5py
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy.io
from scipy.sparse import csr_matrix
from scipy import stats
import random
import math
import matplotlib.font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sklearn.metrics

from loaddata.session_info import load_sessions
from utils.psth import compute_tensor

from interarea.applyFA_dim import apply_FA

##################################################
session_list        = np.array([['LPE09830','2023_04_12']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,
                                    load_behaviordata=True, load_calciumdata=True, load_videodata=False, calciumversion='deconv')

#Get n neurons from V1 and from PM:
n                   = 100
V1_selec            = np.random.choice(np.where(sessions[0].celldata['roi_name']=='V1')[0],n)
PM_selec            = np.random.choice(np.where(sessions[0].celldata['roi_name']=='PM')[0],n)
sessions[0].calciumdata     = sessions[0].calciumdata.iloc[:,np.concatenate((V1_selec,PM_selec))]
sessions[0].celldata        = sessions[0].celldata.iloc[np.concatenate((V1_selec,PM_selec)),:]

##############################################################################
## Construct tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], t_pre, t_post, binsize,method='binmean')

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
[tensor,t_axis]     = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='interp_lin')

tensor              = tensor.transpose((1,2,0))
[N,T,allK]          = np.shape(tensor) #get dimensions of tensor
# [K,N,allT]         = np.shape(tensor) #get dimensions of tensor

# Reshape tensor to neurons (N) by time (T) by trial repetitions (K) by orientations (O)
oris        = sorted(sessions[0].trialdata['Orientation'].unique())
ori_counts  = sessions[0].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
assert(len(ori_counts) == 16 or len(ori_counts) == 8)
assert(np.all(ori_counts == 200) or np.all(ori_counts == 400))

O = len(ori_counts)
K = int(np.mean(ori_counts))

idx_V1 = np.where(sessions[0].celldata['roi_name']=='V1')[0]
idx_PM = np.where(sessions[0].celldata['roi_name']=='PM')[0]

N_V1    = len(idx_V1)
N_PM    = len(idx_PM)

V1_data = np.zeros((N_V1, T, K, O))
PM_data = np.zeros((N_PM, T, K, O))

# idx_V1 = sessions[0].celldata['roi_name']=='V1'

for i,ori in enumerate(oris):
    idx_ori = sessions[0].trialdata['Orientation']==ori
    # V1_data[:,:,:,i] = tensor[sessions[0].celldata['roi_name']=='V1', :, sessions[0].trialdata['Orientation']==ori]
    V1_data[:,:,:,i] = tensor[np.ix_(idx_V1,np.arange(T),idx_ori)]


bin_width = 1
subset = []

FA_output = apply_FA(V1_data, t_axis, bin_width, subset)



#%%
from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings,plot_PCA_gratings_3D_traces
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from sklearn.decomposition import FactorAnalysis as FA
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA

sessions  = compute_tuning_wrapper(sessions)

#%% 

def plot_LOWDIM_gratings_3D(ses, size='runspeed', export_animation=False, savedir=None,
                            thr_tuning=0,plotgainaxis=False, version = 'FA'):

    ########### PCA on trial-averaged responses ############
    ######### plot result as scatter by orientation ########

    areas = np.unique(ses.celldata['roi_name'])

    ori = ses.trialdata['Orientation']
    oris = np.sort(pd.Series.unique(ses.trialdata['Orientation']))

    ori_ind = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', 8), (2, 1))
    if size == 'runspeed':
        sizes = (ses.respmat_runspeed - np.percentile(ses.respmat_runspeed, 5)) / \
            (np.percentile(ses.respmat_runspeed, 95) -
             np.percentile(ses.respmat_runspeed, 5))
    elif size == 'videome':
        sizes = (ses.respmat_videome - np.percentile(ses.respmat_videome, 5)) / \
            (np.percentile(ses.respmat_videome, 95) -
             np.percentile(ses.respmat_videome, 5))
    elif size == 'uniform':
        sizes = np.ones_like(ses.respmat_runspeed)*0.5

    fig = plt.figure(figsize=[len(areas)*4, 4])
    # fig,axes = plt.figure(1, len(areas), figsize=[len(areas)*3, 3])

    for iarea, area in enumerate(areas):
        # ax = axes[iarea]
        idx_area = ses.celldata['roi_name'] == area
        idx_tuned = ses.celldata['tuning_var'] >= thr_tuning
        idx = np.logical_and(idx_area, idx_tuned)
        # zscore for each neuron across trial responses
        respmat_zsc = zscore(ses.respmat[idx, :], axis=1)
        # respmat_zsc = ses.respmat[idx, :]

        if version == 'PCA':
            model = PCA(n_components=3) # construct PCA object with specified number of components
        elif version == 'FA':
            model = FA(n_components=3)
        # fit pca to response matrix (n_samples by n_features)
        
        Xp = model.fit_transform(respmat_zsc.T).T
        # dimensionality is now reduced from N by K to ncomp by K

        if plotgainaxis:
            data                = respmat_zsc
            poprate             = np.nanmean(data,axis=0)
            gain_weights        = np.array([np.corrcoef(poprate,data[n,:])[0,1] for n in range(data.shape[0])])
            gain_trials         = poprate - np.nanmean(data,axis=None)
            # g = np.outer(np.percentile(gain_trials,[0,100]),gain_weights)
            g = np.outer([0,10],gain_weights)
            # g = np.outer(np.percentile(gain_trials,[0,100])*np.percentile(poprate,[0,100]),gain_weights)
            Xg = model.transform(g).T

        ax = fig.add_subplot(1, len(areas), iarea+1, projection='3d')
        
        # plot orientation separately with diff colors
        for t, t_type in enumerate(oris):
            # get all data points for this ori along first PC or projection pairs
            x = Xp[0, ori_ind[t]]
            y = Xp[1, ori_ind[t]]  # and the second
            z = Xp[2, ori_ind[t]]  # and the second
            # ax.scatter(x, y, color=pal[t], s=25, alpha=0.8)     #each trial is one dot
            # ax.scatter(x, y, z, color=pal[t], s=ses.respmat_runspeed[ori_ind[t]], alpha=0.8)     #each trial is one dot
            # each trial is one dot
            ax.scatter(x, y, z, color=pal[t], s=sizes[ori_ind[t]]*6, alpha=0.8)
            # ax.scatter(x, y, z,marker='o')     #each trial is one dot
        if plotgainaxis:
            ax.plot(Xg[0,:],Xg[1,:],Xg[2,:],color='k',linewidth=1)
        ax.set_xlabel('PC 1')  # give labels to axes
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_title(area)
        
        ax.grid(False)
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Get rid of colored axes planes, remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

            # ax.view_init(elev=-30, azim=45, roll=-45)
        # print('Variance Explained (%s) by first 3 components: %2.2f' %
            #   (area, model.explained_variance_ratio_.cumsum()[2]))
    return fig

#%% Make the 3D figure for original data:
fig = plot_LOWDIM_gratings_3D(sessions[19],thr_tuning=0.025, version = 'FA')
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=0, roll=-10)
axes[0].set_zlim([-5,45])


#%% 

ses = sessions[19]

idx = ses.celldata['roi_name'] == 'V1'
X_orig = zscore(ses.respmat[idx, :], axis=1).T
# X_orig = ses.respmat[idx, :].T

pca = PCA(n_components=3) # construct PCA object with specified number of components
fa = FA(n_components=3)
# fit fa to response matrix (n_samples by n_features)

X_re_orig_PCA = pca.inverse_transform(pca.fit_transform(X_orig))

# Obtain factor scores
factor_scores = fa.fit_transform(X_orig)

# Estimate factor loadings
factor_loadings = fa.components_

# Reconstruct data
X_re_orig_FA = factor_scores @ factor_loadings

# show all three as heatmaps in separate panels
fig,axes = plt.subplots(1,3,figsize=(12,4))
sns.heatmap(X_orig,ax=axes[0],vmin=-3,vmax=3,cmap='RdBu_r')
axes[0].set_title('Original')
sns.heatmap(X_re_orig_PCA,ax=axes[1],vmin=-3,vmax=3,cmap='RdBu_r')
axes[1].set_title('PCA')
sns.heatmap(X_re_orig_FA,ax=axes[2],vmin=-3,vmax=3,cmap='RdBu_r')
axes[2].set_title('FA')
plt.tight_layout()




        
#%% APPLY FA TO ALL SESSIONS

bin_width = 80

'E:\OneDrive\PostDoc\Analysis\FA_Analysis'

for sess in sessions:
    
    FA_output = apply_FA(sess, bin_width, [])
    
    file_name = 'FA_output_' + sess.session_id
    
    output_path = 'E:\\OneDrive\\PostDoc\\Analysis\\FA_Analysis\\' + file_name + '.npy'

    np.save(output_path, FA_output, allow_pickle = True)



# Plotting definitions
plt.rcParams.update({'font.size':16})
plt.rcParams['font.family'] = 'Arial'

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
