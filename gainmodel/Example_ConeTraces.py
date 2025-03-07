
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
from tqdm import tqdm

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings,plot_PCA_gratings_3D_traces
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.psth import compute_tensor
from scipy.stats import zscore


savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE12223','2024_06_10']])

# load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,filter_areas=['V1','PM'])
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list,filter_areas=['V1'])

#   Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_areas=['V1'])

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)

#%%  Load data properly:                      
## Parameters for temporal binning
t_pre       = -0.75    #pre s
t_post      = 2     #post s

for ises in range(nSessions):
    # Construct time tensor: 3D 'matrix' of K trials by N neurons by T time bins
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                    t_pre, t_post,method='nearby')


#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% Make the 3D figure:
ises = 0
fig = plot_PCA_gratings_3D(sessions[ises],thr_tuning=0.05,plotgainaxis=True)
axes = fig.get_axes()
axes[0].view_init(elev=-30, azim=25, roll=40)
axes[0].set_xlim([-2,35])
axes[0].set_ylim([-2,35])
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
# fig.savefig(os.path.join(savedir,'Example_Cone_3D_V1_PM_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% 

#%% Show histogram of gain model weights:
ises = 0
data                = sessions[ises].respmat
poprate             = np.nanmean(data,axis=0)
gain_trials         = poprate - np.nanmean(data,axis=None)
gain_weights        = np.array([np.corrcoef(poprate,data[n,:])[0,1] for n in range(data.shape[0])])

#%% 
fig,axes = plt.subplots(1,2,figsize=(6,3))
axes[0].hist(gain_trials,bins=25,color='grey')
axes[0].set_title('Trial gain')
axes[0].set_xlabel('Pop. rate - mean pop. rate')
axes[1].hist(gain_weights,bins=25,color='grey')
axes[1].set_title('Neuron gain ')
axes[1].set_xlabel('Correlation neuron rate to pop. rate')
plt.tight_layout()

# #%% Z-score the calciumdata: 
# for i in range(nSessions):
#     sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

# #%%  Load data properly:                      
# ## Parameters for temporal binning
# t_pre       = -2    #pre s
# t_post      = 3     #post s

# for ises in range(nSessions):
#     t_resp_start = 0
#     t_resp_stop = 1

#     # ## Construct trial response matrix:  N neurons by K trials
#     # sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
#     #                                 t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean',subtr_baseline=False, label = "response matrix")

#     # Construct time tensor: 3D 'matrix' of K trials by N neurons by T time bins
#     [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
#                                     t_pre, t_post,method='nearby')

#%% 
import matplotlib.animation as animation

#%% 
ises = 5
ises = 9
# ises = 0
fig = plot_PCA_gratings_3D_traces(sessions[ises],t_axis,thr_tuning=0.00,plotgainaxis=True,export_animation=False)
axes = fig.get_axes()
# for ax in axes:
    # ax.view_init(elev=-45, azim=25, roll=10)
# axes[0].view_init(azim=25)
fig.savefig(os.path.join(savedir,'Example_Cone_3D_V1_traces_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

# print("Making animation")
# rot_animation = animation.FuncAnimation(
#     fig, rotate, frames=np.arange(0, 364, 4), interval=100)
# rot_animation.save(os.path.join(savedir, 'rotation_%s.gif' % sessions[ises].sessiondata['session_id'][0]), dpi=80, writer='imagemagick')

#%% 
ises = 0

fig = plot_PCA_gratings_3D_traces(sessions[ises],t_axis,thr_tuning=0.00,plotgainaxis=True,export_animation=False)



#%% 
from utils.regress_lib import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_areas=['V1'])

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% Decoding performance as a function of population rate: 
nActBins = 10
kfold = 5
lam = None
lam = 1

error_cv = np.full((nSessions,nActBins),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Decoding stimulus ori across sessions',total=nSessions):
    idx_N = np.ones(len(ses.celldata)).astype(bool)

    data                = zscore(ses.respmat, axis=1)
    # data                = zscore(ses.respmat[idx, :], axis=1)
    poprate             = np.nanmean(data,axis=0)

    binedges = np.percentile(poprate,np.linspace(0,100,nActBins+1))
    bincenters = (binedges[1:]+binedges[:-1])/2

    if lam is None:
        y = ses.trialdata['Orientation']
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
        X = data.T
        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
        lam = find_optimal_lambda(X,y,model_name='LOGR',kfold=kfold)


    for iap in range(nActBins):
        idx_T = (poprate >= binedges[iap]) & (poprate <= binedges[iap+1])
        X = data[np.ix_(idx_N,idx_T)].T
        y = ses.trialdata['Orientation'][idx_T]

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array

        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name='LDA',kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=False)
        # error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name='LOGR',kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=False)

#%% Plot error as a function of population rate: 
from utils.plot_lib import shaded_error
fig,ax = plt.subplots(1,1,figsize=(3,3))
# ax.plot(np.arange(nActBins),error_cv.mean(axis=0))
shaded_error(np.arange(nActBins)+1,error_cv,error='sem',ax=ax)
ax.set_xlabel('Population rate (quantile)')
ax.set_ylabel('Decoding accuracy\n (crossval LDA)')
ax.set_ylim([0.5,1])
ax.set_xticks(np.arange(nActBins)+1)
ax.legend(['mean+-sem\nn=%d sessions' % nSessions],loc='lower right',frameon=False)
# fig.savefig(os.path.join(savedir,'Decoding_Orientation_LOGR_ActBins_%d' % nSessions + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'Decoding_Orientation_LDA_ActBins_%d' % nSessions + '.png'), format = 'png')

