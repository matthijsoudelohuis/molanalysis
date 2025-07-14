
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
from scipy.stats import zscore
from sklearn.decomposition import PCA

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################
session_list        = np.array([['LPE10919_2023_11_06']])
# session_list        = np.array([['LPE12223_2024_06_10']])

sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#   Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)


#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% Make the 3D figure:
# fig = plot_PCA_gratings_3D(sessions[0],thr_tuning=0.05,plotgainaxis=True)
fig = plot_PCA_gratings_3D(sessions[0],idx=sessions[0].celldata['tuning_var'] > 0.05,plotgainaxis=True)
axes = fig.get_axes()
axes[0].view_init(elev=-30, azim=25, roll=40)
axes[1].view_init(elev=15, azim=0, roll=-10)
axes[0].set_xlim([-2,35])
axes[0].set_ylim([-2,35])
axes[1].set_zlim([-5,45])
for ax in axes:
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

plt.tight_layout()
fig.savefig(os.path.join(savedir,'Example_Cone_3D_V1_PM_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% SHOW AL as well: #############################################################################

session_list        = np.array([['LPE12223','2024_06_10']])

# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,filter_areas=['AL'])

# Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)

#%% ########################### Compute tuning metrics: ###################################
sessions[0].celldata['tuning_var'] = compute_tuning(sessions[0].respmat,
                                        sessions[0].trialdata['Orientation'],tuning_metric='tuning_var')

#%% 
fig = plot_PCA_gratings_3D(sessions[0],thr_tuning=0.025)
axes = fig.get_axes()
axes[0].view_init(elev=25, azim=45, roll=-45)
axes[0].set_zlim([-5,15])
for ax in axes:
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

# plt.tight_layout()
fig.savefig(os.path.join(savedir,'Example_Cone_3D_AL_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% 









#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE12223','2024_06_10']])

# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,filter_areas=['V1'])

#   Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)

#%% ########################### Compute tuning metrics: ###################################
sessions[0].celldata['tuning_var'] = compute_tuning(sessions[0].respmat,
                                        sessions[0].trialdata['Orientation'],tuning_metric='tuning_var')


#%% Fit population gain model:
orientations        = sessions[0].trialdata['Orientation']
data                = sessions[0].respmat
data_hat_poprate    = pop_rate_gain_model(data, orientations)

datasets            = (data,data_hat_poprate)
fig = plot_respmat(orientations, datasets, ['original','pop rate gain'])

#%% Make session objects with only gain, or no gain at all:
sessions_onlygain   = copy.deepcopy(sessions)
sessions_nogain     = copy.deepcopy(sessions)

sessions_onlygain[0].respmat = data_hat_poprate
sessions_nogain[0].respmat = data - data_hat_poprate

#%% Make the 3D figure for original data:
fig = plot_PCA_gratings_3D(sessions[0],thr_tuning=0)
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=0, roll=-10)
axes[0].set_zlim([-5,45])
fig.savefig(os.path.join(savedir,'Cone_3D_V1_Original_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% Make the 3D figure for only gain data:
fig = plot_PCA_gratings_3D(sessions_onlygain[0],thr_tuning=0)
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=-15, roll=-35)
fig.savefig(os.path.join(savedir,'Cone_3D_V1_Gainonly_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% Make the 3D figure for gain-subtracted data:
fig = plot_PCA_gratings_3D(sessions_nogain[0],thr_tuning=0)
axes = fig.get_axes()
axes[0].view_init(elev=65, azim=-135, roll=0)
fig.savefig(os.path.join(savedir,'Cone_3D_V1_Nogain_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

# #%% #############################################################################




#%% Show for neurons with different population coupling: 
ises                = 0
data                = zscore(sessions[ises].respmat.T, axis=0)
poprate             = np.nanmean(data,axis=1)
sessions[ises].celldata['popcoupling'] = [np.corrcoef(data[:,i],poprate)[0,1] for i in range(np.shape(data)[1])]

#%% PCA for differently coupled neurons: 
nPopCouplingBins        = 5
binedges_popcoupling    = np.percentile(sessions[ises].celldata['popcoupling'],np.linspace(0,100,nPopCouplingBins+1))

ses = sessions[ises]

fig = plt.figure(figsize=(nPopCouplingBins*3,2.5))
for iPopCouplingBin in range(nPopCouplingBins):
    ax = fig.add_subplot(1, nPopCouplingBins, iPopCouplingBin+1, projection='3d')
    idx_N = np.all((
                        sessions[0].celldata['roi_name']=='V1',
                        sessions[0].celldata['noise_level']<20,
                        sessions[ises].celldata['popcoupling']>binedges_popcoupling[iPopCouplingBin],
                        sessions[ises].celldata['popcoupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)
    
    ########### PCA on trial-averaged responses ############
    ######### plot result as scatter by orientation ########
    ori         = ses.trialdata['Orientation']
    oris        = np.sort(pd.Series.unique(ses.trialdata['Orientation']))

    ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    pal         = sns.color_palette('husl', len(oris))
    pal         = np.tile(sns.color_palette('husl', 8), (2, 1))

    respmat_zsc = zscore(ses.respmat[idx_N, :], axis=1)

    # construct PCA object with specified number of components
    pca = PCA(n_components=3)
    # fit pca to response matrix (n_samples by n_features)
    Xp = pca.fit_transform(respmat_zsc.T).T
    # dimensionality is now reduced from N by K to ncomp by K

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        # get all data points for this ori along first PC or projection pairs
        x = Xp[0, ori_ind[t]]
        y = Xp[1, ori_ind[t]]  # and the second
        z = Xp[2, ori_ind[t]]  # and the second
        # each trial is one dot
        ax.scatter(x, y, z, color=pal[t], s=0.3, alpha=0.7)
    
    ax_3d_makeup(ax,Xp.T)
my_savefig(fig,savedir,'Cone_High_Low_PopCoupling_%s' % (sessions[ises].session_id),formats=['png','pdf'])

#%% PCA for differen subsets of trials with little or a lot of variance: 
nPopRateVarianceBins    = 2

ses                 = sessions[ises]
data                = zscore(sessions[ises].respmat.T, axis=0)
poprate             = np.nanmean(data,axis=1)
nTrials             = 1000

fig,ax = plt.subplots(1,1,figsize=(4,3.5))
ax.hist(poprate,bins=np.arange(-0.5,1,0.02),density=False,alpha=0.3)
p = 1e16**-np.abs(poprate) #sample according to how close the activity is to zero
p = p/np.sum(p) #normalize
idx_T_low = np.random.choice(np.arange(len(poprate)),size=nTrials,replace=False,p=p)
ax.hist(poprate[idx_T_low],bins=np.arange(-0.5,1,0.02),density=False,alpha=0.3)
p = np.abs(poprate)/np.sum(np.abs(poprate)) #sample according to how far the activity is different from zero
idx_T_high = np.random.choice(np.arange(len(poprate)),size=nTrials,replace=False,p=p)

ax.hist(poprate[idx_T_high],bins=np.arange(-0.5,1,0.02),density=False,alpha=0.3)
ax.set_ylabel('Trial Count')
ax.set_xlabel('Z-scored population activity')
ax.legend(['All','Low Variance','High Variance'],frameon=False,
          loc='upper right',fontsize=10,title='Trials')
idx_T_both = np.column_stack((idx_T_low,idx_T_high))
my_savefig(fig,savedir,'Hist_High_Low_PopRateVariance_%s' % (sessions[ises].session_id),formats=['png'])

#%% Show PCA for differen subsets of trials with little or a lot of variance:
fig = plt.figure(figsize=(6,2.5))
for iPopRateVarianceBin in range(nPopRateVarianceBins):
    ax = fig.add_subplot(1, nPopRateVarianceBins, iPopRateVarianceBin+1, projection='3d')
    idx_N   = np.all((
                        sessions[0].celldata['roi_name']=='V1',
                        sessions[0].celldata['noise_level']<20,
                        sessions[0].celldata['tuning_var']>0.02,
                        # sessions[ises].celldata['popcoupling']>binedges_popcoupling[iPopCouplingBin]
                        ),axis=0)
    
    idx_T = idx_T_both[:,iPopRateVarianceBin]

    ########### PCA on trial-averaged responses ############
    ######### plot result as scatter by orientation ########
    ori         = ses.trialdata['Orientation'][idx_T]
    oris        = np.sort(pd.Series.unique(ses.trialdata['Orientation'][idx_T]))

    ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    pal         = sns.color_palette('husl', len(oris))
    pal         = np.tile(sns.color_palette('husl', 8), (2, 1))

    respmat_zsc = zscore(ses.respmat[np.ix_(idx_N, idx_T)], axis=1)

    # construct PCA object with specified number of components
    pca = PCA(n_components=3)
    # fit pca to response matrix (n_samples by n_features)
    Xp = pca.fit_transform(respmat_zsc.T).T
    # dimensionality is now reduced from N by K to ncomp by K

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        # get all data points for this ori along first PC or projection pairs
        x = Xp[0, ori_ind[t]]
        y = Xp[1, ori_ind[t]]  # and the second
        z = Xp[2, ori_ind[t]]  # and the second
        # each trial is one dot
        ax.scatter(x, y, z, color=pal[t], s=0.6, alpha=0.7)
    
    ax_3d_makeup(ax,Xp.T)
    ax.set_title('Low rate variance' if iPopRateVarianceBin==0 else 'High rate variance')
my_savefig(fig,savedir,'Cone_High_Low_Variance%s' % (sessions[ises].session_id),formats=['png','pdf'])


#%% PCA for trials without locomotion:
nBins        = 5
binedges    = np.percentile(sessions[ises].celldata['OSI'],np.linspace(0,100,nBins+1))
tunefield = 'gOSI'
binedges    = np.percentile(sessions[ises].celldata[tunefield],np.linspace(0,100,nBins+1))

ses = sessions[ises]

fig = plt.figure(figsize=(nBins*3,2.5))
for ibin in range(nBins):
    ax = fig.add_subplot(1, nBins, ibin+1, projection='3d')
    idx_N = np.all((
                        sessions[0].celldata['roi_name']=='V1',
                        sessions[0].celldata['noise_level']<20,
                        sessions[ises].celldata[tunefield]>binedges[ibin],
                        sessions[ises].celldata[tunefield]<=binedges[ibin+1]),axis=0)
    
    ########### PCA on trial-averaged responses ############
    ######### plot result as scatter by orientation ########
    ori         = ses.trialdata['Orientation']
    oris        = np.sort(pd.Series.unique(ses.trialdata['Orientation']))

    ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    pal         = sns.color_palette('husl', len(oris))
    pal         = np.tile(sns.color_palette('husl', 8), (2, 1))

    respmat_zsc = zscore(ses.respmat[idx_N, :], axis=1)

    # construct PCA object with specified number of components
    pca = PCA(n_components=3)
    # fit pca to response matrix (n_samples by n_features)
    Xp = pca.fit_transform(respmat_zsc.T).T
    # dimensionality is now reduced from N by K to ncomp by K

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        # get all data points for this ori along first PC or projection pairs
        x = Xp[0, ori_ind[t]]
        y = Xp[1, ori_ind[t]]  # and the second
        z = Xp[2, ori_ind[t]]  # and the second
        # each trial is one dot
        ax.scatter(x, y, z, color=pal[t], s=0.3, alpha=0.7)
    
    ax_3d_makeup(ax,Xp.T)
my_savefig(fig,savedir,'Cone_High_Low_Tuning_%s' % (sessions[ises].session_id),formats=['png'])


#%% PCA for trials without locomotion:
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(1, 1, 1, projection='3d')
idx_N   = np.all((
                    sessions[0].celldata['roi_name']=='V1',
                    sessions[0].celldata['noise_level']<20,
                    # sessions[0].celldata['tuning_var']>0.02,
                    # sessions[ises].celldata['popcoupling']>binedges_popcoupling[iPopCouplingBin]
                    ),axis=0)

idx_T = sessions[ises].respmat_runspeed<0.5   
# idx_T = sessions[ises].respmat_runspeed>0.5   

########### PCA on trial-averaged responses ############
######### plot result as scatter by orientation ########
ori         = ses.trialdata['Orientation'][idx_T]
oris        = np.sort(pd.Series.unique(ses.trialdata['Orientation'][idx_T]))

ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

pal         = sns.color_palette('husl', len(oris))
pal         = np.tile(sns.color_palette('husl', 8), (2, 1))

respmat_zsc = zscore(ses.respmat[np.ix_(idx_N, idx_T)], axis=1)

# construct PCA object with specified number of components
pca = PCA(n_components=3)
# fit pca to response matrix (n_samples by n_features)
Xp = pca.fit_transform(respmat_zsc.T).T
# dimensionality is now reduced from N by K to ncomp by K

# plot orientation separately with diff colors
for t, t_type in enumerate(oris):
    # get all data points for this ori along first PC or projection pairs
    x = Xp[0, ori_ind[t]]
    y = Xp[1, ori_ind[t]]  # and the second
    z = Xp[2, ori_ind[t]]  # and the second
    # each trial is one dot
    ax.scatter(x, y, z, color=pal[t], s=0.6, alpha=0.7)

ax_3d_makeup(ax,Xp.T)
ax.set_title('No locomotion')
my_savefig(fig,savedir,'Cone_NoLocomotion_%s' % (sessions[ises].session_id),formats=['png'])

