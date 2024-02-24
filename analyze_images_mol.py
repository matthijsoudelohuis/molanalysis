# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import preprocessing
from loaddata.session_info import filter_sessions,load_sessions
from scipy.signal import medfilt
from utils.plotting_style import * #get all the fixed color schemes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from utils.imagelib import load_natural_images #

from utils.explorefigs import *
from utils.psth import compute_tensor,compute_respmat,construct_behav_matrix_ts_F



# from sklearn.decomposition import PCA

savedir = 'C:\\OneDrive\\PostDoc\\Figures\\Images\\'
# savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Images\\ExploreFigs\\'

#################################################
# session_list        = np.array([['LPE09665','2023_03_15']])
# session_list        = np.array([['LPE11086','2023_12_16']])
# sessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
#                                     load_calciumdata=True, load_videodata=True, calciumversion='deconv')
sessions            = filter_sessions(protocols = ['IM'],load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')
nSessions = len(sessions)
# sessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
#                                     load_calciumdata=True, load_videodata=True, calciumversion='deconv')


for ises in range(nSessions):
    sessions[ises].videodata['pupil_area']    = medfilt(sessions[ises].videodata['pupil_area'] , kernel_size=25)
    sessions[ises].videodata['motionenergy']  = medfilt(sessions[ises].videodata['motionenergy'] , kernel_size=25)
    sessions[ises].behaviordata['runspeed']   = medfilt(sessions[ises].behaviordata['runspeed'] , kernel_size=51)


#Compute average response per trial:
for ises in range(nSessions):
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)
    # delattr(sessions[ises],'calciumdata')

    #hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
    temp = pd.DataFrame(np.reshape(np.array(sessions[ises].behaviordata['runspeed']),(len(sessions[ises].behaviordata['runspeed']),1)))
    sessions[ises].respmat_runspeed = compute_respmat(temp, sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,method='mean')
    sessions[ises].respmat_runspeed = np.squeeze(sessions[ises].respmat_runspeed)

    #hacky way to create dataframe of the video motion with F x 1 with F number of samples:
    temp = pd.DataFrame(np.reshape(np.array(sessions[ises].videodata['motionenergy']),(len(sessions[ises].videodata['motionenergy']),1)))
    sessions[ises].respmat_videome = compute_respmat(temp, sessions[ises].videodata['timestamps'], sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,method='mean')
    sessions[ises].respmat_videome = np.squeeze(sessions[ises].respmat_videome)


################################################################
#Show some traces and some stimuli to see responses:

sesidx = 0

fig = plot_excerpt(sessions[sesidx],trialsel=None,plot_neural=True,plot_behavioral=False)

trialsel = [3294, 3374]
fig = plot_excerpt(sessions[sesidx],trialsel=trialsel,plot_neural=True,plot_behavioral=True,neural_version='traces')
# fig.savefig(os.path.join(savedir,'TraceExcerpt_dF_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'Excerpt_Traces_deconv_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_excerpt(sessions[sesidx],trialsel=None,plot_neural=True,plot_behavioral=True,neural_version='raster')
fig = plot_excerpt(sessions[sesidx],trialsel=trialsel,plot_neural=True,plot_behavioral=True,neural_version='raster')
fig.savefig(os.path.join(savedir,'Excerpt_Raster_dF_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


########################### Show PCA ##########################
sesidx = 0
# fig = PCA_gratings_3D(sessions[sesidx])
# fig.savefig(os.path.join(savedir,'PCA','PCA_3D_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plot_PCA_images(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA','PCA_Gratings_All_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = plt.figure()
sns.histplot(sessions[sesidx].respmat_runspeed,binwidth=0.5)
plt.ylim([0,100])


sesidx = 0


sessions[sesidx].trialdata['repetition'] = np.r_[np.zeros([2800]),np.ones([2800])]

#Sort based on image number:
arr1inds                = sessions[sesidx].trialdata['ImageNumber'][:2800].argsort()
arr2inds                = sessions[sesidx].trialdata['ImageNumber'][2800:5600].argsort()

respmat = sessions[sesidx].respmat[:,np.r_[arr1inds,arr2inds+2800]]
# respmat_sort = sessions[sesidx].respmat_z[:,np.r_[arr1inds,arr2inds+2800]]

from sklearn.preprocessing import normalize

min_max_scaler = preprocessing.MinMaxScaler()
respmat_sort = preprocessing.minmax_scale(respmat, feature_range=(0, 1), axis=0, copy=True)

respmat_sort = normalize(respmat, 'l2', axis=1)

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# axes[0].imshow(respmat_sort[:,:2800], aspect='auto',vmin=-100,vmax=200) 
axes[0].imshow(respmat_sort[:,:2800], aspect='auto',vmin=np.percentile(respmat_sort,5),vmax=np.percentile(respmat_sort,95))
axes[0].set_xlabel('Image #')
axes[0].set_ylabel('Neuron')
axes[0].set_title('Repetition 1')
# axes[1].imshow(respmat_sort[:,2800:], aspect='auto',vmin=-100,vmax=200) 
axes[1].imshow(respmat_sort[:,2800:], aspect='auto',vmin=np.percentile(respmat_sort,5),vmax=np.percentile(respmat_sort,95)) 
axes[1].set_xlabel('Image #')
axes[1].set_ylabel('Neuron')
plt.tight_layout(rect=[0, 0, 1, 1])
axes[1].set_title('Repetition 2')


natimgdata = load_natural_images(onlyright=True)


####### Regress out behavioral state related activity  #################################

from utils.RRRlib import *

def regress_out_behavior_modulation(ses,X=None,Y=None,nvideoPCs = 30,rank=2):
    if not X:
        X,Xlabels = construct_behav_matrix_ts_F(ses,nvideoPCs=nvideoPCs)

    if not Y:
        Y = ses.calciumdata.to_numpy()
        
    assert X.shape[0] == Y.shape[0],'number of samples of calcium activity and interpolated behavior data do not match'

    ## LM model run
    B_hat = LM(Y, X, lam=10)

    B_hat_rr = RRR(Y, X, B_hat, r=rank, mode='left')
    Y_hat_rr = X @ B_hat_rr

    Y_out = Y - Y_hat_rr

    return Y_out


Rss_rank = []
 ## LM model run
for i in range(np.shape(X)[1]):
    B_hat_rr = RRR(Y, X, B_hat, r=i, mode='left')
    Y_hat_rr = X @ B_hat_rr
    # Rss_rank.append(Rss(Y,Y_hat_rr))
    Rss_rank.append(EV(Y,Y_hat_rr))

plt.figure()
plt.plot(Rss_rank)




# %% LM model run
B_hat = LM(Y=D, X=S, lam=10)
# B_hat = LM(X, Y, lam=0.01)
D_hat = S @ B_hat

B_hat_lr = RRR(D, S, B_hat, r=2, mode='left')
B_hat_lr = RRR(D, S, B_hat, r=2, mode='right')
D_hat_lr = S @ B_hat_lr

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(8,4))
ax1.imshow(D[:1000,:100].T,vmin=0,vmax=1000,aspect='auto')
ax2.imshow(D_hat[:1000,:100].T,vmin=0,vmax=1000,aspect='auto')
ax3.imshow(D_hat_lr[:1000,:100].T,vmin=0,vmax=1000,aspect='auto')

# %% xval lambda
n = 1000
k = 5
lam = xval_ridge_reg_lambda(Y[:n,:], X[:n,:], k)




# %% cheat
lam = 35

# %% LM model run
B_hat = LM(Y, X, lam=lam)
Y_hat = X @ B_hat

print("LM model error:")
print("LM: %5.3f " % Rss(Y,Y_hat))


S,Slabels = construct_behav_matrix_ts_F(sessions[sesidx],nvideoPCs=nvideoPCs)

sns.heatmap(np.corrcoef(S,rowvar=False),xticklabels=Slabels,yticklabels=Slabels)

