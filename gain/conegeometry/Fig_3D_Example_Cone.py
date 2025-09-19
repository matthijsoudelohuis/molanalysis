
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
session_list        = np.array([['LPE12223_2024_06_10']])

sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#   Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)


#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% 
sessions = compute_pop_coupling(sessions)

#%% Make the 3D figure:
# fig = plot_PCA_gratings_3D(sessions[0],thr_tuning=0.05,plotgainaxis=True)
fig = plot_PCA_gratings_3D(sessions[0],idx_N=sessions[0].celldata['tuning_var'] > 0.05,
                           size='poprate',
                           plotgainaxis=True)
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
sessions = compute_tuning_wrapper(sessions)

#%% 
ises = 0
idx_N = np.all((
                sessions[0].celldata['roi_name']=='V1',
                sessions[0].celldata['noise_level']<20,
                sessions[0].celldata['tuning_var']>0.025
                ),axis=0)

ses = sessions[ises]
ori = ses.trialdata['Orientation']
oris = np.sort(pd.Series.unique(ses.trialdata['Orientation']))

ori_ind = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

pal = sns.color_palette('husl', len(oris))
pal = np.tile(sns.color_palette('husl', 8), (2, 1))
minperc = 5
maxperc = 95
poprate = np.nanmean(zscore(ses.respmat, axis=1), axis=0)
sizes = (poprate- np.percentile(poprate, minperc)) / \
    (np.percentile(poprate, maxperc) -
        np.percentile(poprate, minperc))

fig = plt.figure(figsize=[6,6])
# fig,axes = plt.figure(1, len(areas), figsize=[len(areas)*3, 3])
# 
respmat_zsc = zscore(ses.respmat[idx_N, :], axis=1)
# construct PCA object with specified number of components
pca = PCA(n_components=3)
# fit pca to response matrix (n_samples by n_features)
Xp = pca.fit_transform(respmat_zsc.T).T
# dimensionality is now reduced from N by K to ncomp by K
ax = fig.add_subplot(111, projection='3d')

# plot orientation separately with diff colors
for t, t_type in enumerate(oris):
    # get all data points for this ori along first PC or projection pairs
    x = Xp[0, ori_ind[t]]
    y = Xp[1, ori_ind[t]]  # and the second
    z = Xp[2, ori_ind[t]]  # and the second
    # each trial is one dot
    clrs = pal[t]
    xpop = poprate[ori_ind[t]]
    xpop -= np.min(xpop)
    xpop /= np.max(xpop)
    xpop = 1 - xpop

    clrs = np.tile(xpop[:,np.newaxis],3) * np.tile(pal[t][np.newaxis,:],(len(ori_ind[t]),1))

    ax.scatter(x, y, z, c=clrs, s=sizes[ori_ind[t]]*5, alpha=0.5)
    # ax.scatter(x, y, z, color=pal[t], s=sizes[ori_ind[t]]*6, alpha=0.4)

ax.set_xlabel('PC 1')  # give labels to axes
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlim(np.percentile(Xp[0,:],[1,99.5]))
ax.set_ylim(np.percentile(Xp[1,:],[1,99.5]))
ax.set_zlim(np.percentile(Xp[2,:],[1,99.5]))

axes = fig.get_axes()
ax.view_init(elev=-18, azim=25, roll=25)
# ax.set_title(plottitle)
nticks = 5
ax.grid(True)
ax.set_facecolor('white')
ax.set_xticks(np.linspace(np.percentile(Xp[0,:],1),np.percentile(Xp[0],99),nticks))
ax.set_yticks(np.linspace(np.percentile(Xp[1],1),np.percentile(Xp[1],99),nticks))
ax.set_zticks(np.linspace(np.percentile(Xp[2],1),np.percentile(Xp[2],99),nticks))

# Get rid of colored axes planes, remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# plt.tight_layout()
fig.savefig(os.path.join(savedir,'Example_Cone_3D_V1_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')



#%% Make a 3D cone where coloring is based on population rate: 

ses = sessions[0]
colormap = "magma"

########### PCA on trial-averaged responses ############
######### plot result as scatter by orientation ########
idx_N = np.all((ses.celldata['roi_name']=='V1',
                ses.celldata['noise_level']<100,
                ses.celldata['tuning_var']>0.01),axis=0)

ori = np.mod(ses.trialdata['Orientation'],180)
oris = np.sort(np.unique(ori))

fig = plt.figure(figsize=[4, 4])
 
# zscore for each neuron across trial responses
respmat_zsc = zscore(ses.respmat[idx_N, :], axis=1)
poprate             = np.nanmean(respmat_zsc,axis=0)

plotgainaxis = False

# construct PCA object with specified number of components
pca = PCA(n_components=3)
# fit pca to response matrix (n_samples by n_features)
Xp = pca.fit_transform(respmat_zsc.T).T
# dimensionality is now reduced from N by K to ncomp by K

if plotgainaxis:
    gain_weights        = np.array([np.corrcoef(poprate,respmat_zsc[n,:])[0,1] for n in range(respmat_zsc.shape[0])])
    gain_trials         = poprate - np.nanmean(respmat_zsc,axis=None)
    # g = np.outer(np.percentile(gain_trials,[0,100]),gain_weights)
    g = np.outer([0,10],gain_weights)
    # g = np.outer(np.percentile(gain_trials,[0,100])*np.percentile(poprate,[0,100]),gain_weights)
    Xg = pca.transform(g).T

ax = fig.add_subplot(111, projection='3d')

c = np.clip(poprate,np.percentile(poprate,1),np.percentile(poprate,99))
c = c-np.min(c)
c = c/np.max(c)
cmap = matplotlib.cm.get_cmap(colormap)
g = np.squeeze(cmap([c]))

ax.scatter(Xp[0,:], Xp[1,:], Xp[2,:], c=g , s=1, alpha=0.7)

nPopRateBins = 10

binedges_poprate    = np.percentile(poprate,np.linspace(1,99,nPopRateBins+1))
c = np.mean(np.column_stack((binedges_poprate[:-1],binedges_poprate[1:])),axis=1)
c = c-np.min(c)
c = c/np.max(c)
g = cmap(c)

for iPopRateBin in range(nPopRateBins):
    meandata = np.empty([len(oris),3])

    for istim,stim in enumerate(oris):
        idx_T = np.all((ori == stim,
                    poprate>binedges_poprate[iPopRateBin],
                    poprate<=binedges_poprate[iPopRateBin+1]),axis=0)
        meandata[istim,:] = np.mean(Xp[:,idx_T],axis=1)
    meandata = np.concatenate((meandata,meandata[:1,:]),axis=0)
    
    ax.plot(meandata[:,0],meandata[:,1],meandata[:,2],
            color=g[iPopRateBin],linewidth=2)

ax.set_xlim(np.percentile(Xp[0,:],[1,99.5]))
ax.set_ylim(np.percentile(Xp[1,:],[1,99.5]))
ax.set_zlim(np.percentile(Xp[2,:],[1,99.5]))

if plotgainaxis:
    ax.plot(Xg[0,:],Xg[1,:],Xg[2,:],color='k',linewidth=1)
ax.set_xlabel('PC 1')  # give labels to axes
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.grid(False)
ax.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

axes = fig.get_axes()
ax.view_init(elev=-15, azim=25, roll=15)
# ax.set_title(plottitle)
nticks = 5
ax.grid(True)
ax.set_facecolor('white')
ax.set_xticks(np.linspace(np.percentile(Xp[0,:],1),np.percentile(Xp[0],99),nticks))
ax.set_yticks(np.linspace(np.percentile(Xp[1],1),np.percentile(Xp[1],99),nticks))
ax.set_zticks(np.linspace(np.percentile(Xp[2],1),np.percentile(Xp[2],99),nticks))

# Get rid of colored axes planes, remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

print('Variance Explained by first 3 components: %2.2f' %
        (pca.explained_variance_ratio_.cumsum()[2]))
# my_savefig(fig,savedir,'Example_Cone_3D_PopRate_%s' % sessions[0].session_id,formats=['png','pdf'])


#%% Make a 3D cone where coloring is based on population rate: 

ses = sessions[0]
colormap = "magma"

########### PCA on trial-averaged responses ############
######### plot result as scatter by orientation ########
idx_N = np.all((ses.celldata['roi_name']=='V1',
                ses.celldata['noise_level']<100,
                ses.celldata['tuning_var']>0.01),axis=0)

ori = np.mod(ses.trialdata['Orientation'],180)
oris = np.sort(np.unique(ori))

fig = plt.figure(figsize=[4, 4])
 
# zscore for each neuron across trial responses
respmat_zsc = zscore(ses.respmat[idx_N, :], axis=1)
poprate             = np.nanmean(respmat_zsc,axis=0)

plotgainaxis = False

# construct PCA object with specified number of components
pca = PCA(n_components=3)
# fit pca to response matrix (n_samples by n_features)
Xp = pca.fit_transform(respmat_zsc.T).T
# dimensionality is now reduced from N by K to ncomp by K

if plotgainaxis:
    gain_weights        = np.array([np.corrcoef(poprate,respmat_zsc[n,:])[0,1] for n in range(respmat_zsc.shape[0])])
    gain_trials         = poprate - np.nanmean(respmat_zsc,axis=None)
    # g = np.outer(np.percentile(gain_trials,[0,100]),gain_weights)
    g = np.outer([0,10],gain_weights)
    # g = np.outer(np.percentile(gain_trials,[0,100])*np.percentile(poprate,[0,100]),gain_weights)
    Xg = pca.transform(g).T

ax = fig.add_subplot(111, projection='3d')

c = np.clip(poprate,np.percentile(poprate,1),np.percentile(poprate,99))
c = c-np.min(c)
c = c/np.max(c)
cmap = matplotlib.cm.get_cmap(colormap)
g = np.squeeze(cmap([c]))

ax.scatter(Xp[0,:], Xp[1,:], Xp[2,:], c=g , s=1, alpha=0.7)

nPopRateBins = 10

binedges_poprate    = np.percentile(poprate,np.linspace(1,99,nPopRateBins+1))
c = np.mean(np.column_stack((binedges_poprate[:-1],binedges_poprate[1:])),axis=1)
c = c-np.min(c)
c = c/np.max(c)
g = cmap(c)

for iPopRateBin in range(nPopRateBins):
    meandata = np.empty([len(oris),3])

    for istim,stim in enumerate(oris):
        idx_T = np.all((ori == stim,
                    poprate>binedges_poprate[iPopRateBin],
                    poprate<=binedges_poprate[iPopRateBin+1]),axis=0)
        meandata[istim,:] = np.mean(Xp[:,idx_T],axis=1)
    meandata = np.concatenate((meandata,meandata[:1,:]),axis=0)
    
    ax.plot(meandata[:,0],meandata[:,1],meandata[:,2],
            color=g[iPopRateBin],linewidth=2)

ax.set_xlim(np.percentile(Xp[0,:],[1,99.5]))
ax.set_ylim(np.percentile(Xp[1,:],[1,99.5]))
ax.set_zlim(np.percentile(Xp[2,:],[1,99.5]))

if plotgainaxis:
    ax.plot(Xg[0,:],Xg[1,:],Xg[2,:],color='k',linewidth=1)
ax.set_xlabel('PC 1')  # give labels to axes
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.grid(False)
ax.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

axes = fig.get_axes()
ax.view_init(elev=-15, azim=25, roll=15)

# ax.set_title(plottitle)
nticks = 5
ax.grid(True)
ax.set_facecolor('white')
ax.set_xticks(np.linspace(np.percentile(Xp[0,:],1),np.percentile(Xp[0],99),nticks))
ax.set_yticks(np.linspace(np.percentile(Xp[1],1),np.percentile(Xp[1],99),nticks))
ax.set_zticks(np.linspace(np.percentile(Xp[2],1),np.percentile(Xp[2],99),nticks))

# Get rid of colored axes planes, remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

print('Variance Explained by first 3 components: %2.2f' %
        (pca.explained_variance_ratio_.cumsum()[2]))
# my_savefig(fig,savedir,'Example_Cone_3D_PopRate_%s' % sessions[0].session_id,formats=['png','pdf'])

#%% 













#%% Fit affine model:
sessions = fitAffine_singleneuron(sessions,radius=500)



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




#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)


#%% PCA for differently coupled neurons: 
nPopCouplingBins        = 5
binedges_pop_coupling    = np.percentile(sessions[ises].celldata['pop_coupling'],np.linspace(0,100,nPopCouplingBins+1))

ses = sessions[ises]

fig = plt.figure(figsize=(nPopCouplingBins*3,2.5))
for iPopCouplingBin in range(nPopCouplingBins):
    ax = fig.add_subplot(1, nPopCouplingBins, iPopCouplingBin+1, projection='3d')
    idx_N = np.all((
                        sessions[0].celldata['roi_name']=='V1',
                        sessions[0].celldata['noise_level']<20,
                        sessions[ises].celldata['pop_coupling']>binedges_pop_coupling[iPopCouplingBin],
                        sessions[ises].celldata['pop_coupling']<=binedges_pop_coupling[iPopCouplingBin+1]),axis=0)
    
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
# my_savefig(fig,savedir,'Cone_High_Low_PopCoupling_%s' % (sessions[ises].session_id),formats=['png','pdf'])

#%% Get good unmodulated cells, multiplicative and additively modulated neurons: 
N = len(sessions[ises].celldata)
idx_N = np.zeros((N,3)).astype(bool)
perc = 33
tuned_thr = 0.0
OSI_thr = np.percentile(sessions[ises].celldata['gOSI'],50)
maxnoiselevel = 100
idx_N[:,0] = np.all((
                        # sessions[ises].celldata['roi_name']=='V1',
                        sessions[ises].celldata['pop_coupling']<np.percentile(sessions[ises].celldata['pop_coupling'],perc),
                        sessions[ises].celldata['noise_level']<maxnoiselevel,
                        sessions[ises].celldata['tuning_var']>tuned_thr,
                        sessions[ises].celldata['gOSI']>OSI_thr,
                        # sessions[ises].celldata['Affine_Mult']<np.percentile(sessions[ises].celldata['Affine_Mult'],perc),
                        # sessions[ises].celldata['Affine_Add']<np.percentile(sessions[ises].celldata['Affine_Add'],perc),
                        ),axis=0)
idx_N[:,1] = np.all((
                        # sessions[ises].celldata['roi_name']=='V1',
                        sessions[ises].celldata['noise_level']<maxnoiselevel,
                        sessions[ises].celldata['pop_coupling']>np.percentile(sessions[ises].celldata['pop_coupling'],100-perc),
                        sessions[ises].celldata['tuning_var']>tuned_thr,
                        sessions[ises].celldata['gOSI']>OSI_thr,
                        sessions[ises].celldata['Affine_Mult']<np.percentile(sessions[ises].celldata['Affine_Mult'],perc),
                        sessions[ises].celldata['Affine_Add']>np.percentile(sessions[ises].celldata['Affine_Add'],100-perc),
                        ),axis=0)
idx_N[:,2] = np.all((
                        # sessions[ises].celldata['roi_name']=='V1',
                        sessions[ises].celldata['noise_level']<maxnoiselevel,
                        sessions[ises].celldata['pop_coupling']>np.percentile(sessions[ises].celldata['pop_coupling'],100-perc),
                        sessions[ises].celldata['tuning_var']>tuned_thr,
                        sessions[ises].celldata['gOSI']>OSI_thr,
                        sessions[ises].celldata['Affine_Mult']>np.percentile(sessions[ises].celldata['Affine_Mult'],100-perc),
                        # sessions[ises].celldata['Affine_Add']<np.percentile(sessions[ises].celldata['Affine_Add'],perc),
                        ),axis=0)


labels = ['soloists','add. choristers','mult. choristers']
# labels = ['soloists','mult. choristers','add. choristers']

fig = plt.figure(figsize=(3*3,2.5))
for iBin in range(3):
    ax = fig.add_subplot(1, 3, iBin+1, projection='3d')
    
    ########### PCA on trial-averaged responses ############
    ######### plot result as scatter by orientation ########
    ori         = sessions[ises].trialdata['Orientation']
    oris        = np.sort(pd.Series.unique(sessions[ises].trialdata['Orientation']))

    ori_ind     = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    pal         = sns.color_palette('husl', len(oris))
    pal         = np.tile(sns.color_palette('husl', 8), (2, 1))

    respmat_zsc = zscore(sessions[ises].respmat[idx_N[:,iBin], :], axis=1)

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
    ax.set_title('%s (n=%d)' % (labels[iBin],np.sum(idx_N[:,iBin])))
    ax_3d_makeup(ax,Xp.T)
my_savefig(fig,savedir,'Cone_Affine_Pops_%s' % (sessions[ises].session_id),formats=['png','pdf'])

#%% 
# plt.scatter(sessions[ises].celldata['Affine_Mult'],sessions[ises].celldata['Affine_Add'])

#%% PCA for differen subsets of trials with little or a lot of variance: 
nPopRateVarianceBins    = 2

idx_N   = np.all((
                    sessions[0].celldata['roi_name']=='V1',
                    sessions[0].celldata['noise_level']<20,
                    # sessions[0].celldata['tuning_var']>0.05,
                    sessions[0].celldata['gOSI']>0.3,
                    ),axis=0)

ses                 = sessions[ises]
data                = zscore(sessions[ises].respmat[idx_N,:].T, axis=0)
poprate             = np.nanmean(data,axis=1)
nTrials             = 750

fig,ax = plt.subplots(1,1,figsize=(4,3.5))
ax.hist(poprate,bins=np.arange(-0.5,1,0.02),density=False,alpha=0.3)
pexp = 1e25
p = pexp**-np.abs(poprate) #sample according to how close the activity is to zero
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
                    # sessions[ises].celldata['pop_coupling']>binedges_pop_coupling[iPopCouplingBin]
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



