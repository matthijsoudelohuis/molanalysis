


import os, math

from loaddata.get_data_folder import get_local_drive
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from utils.explorefigs import plot_PCA_gratings
from loaddata.session import Session
import pandas as pd
import seaborn as sns
from utils.corr_lib import compute_noise_correlation, compute_pairwise_metrics

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes

savedir = 'D:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\GainModel\\'

nNeurons        = 1000
nTrials         = 3200

noise_level     = 0.3
gain_level      = 2
offset_level    = 0

noris           = 16

oris            = np.linspace(0,360,noris+1)[:-1]
locs            = np.random.rand(nNeurons) * np.pi * 2  # circular mean
kappa           = 2  # concentration
# kappas          = np.random.rand(nNeurons) * 2  # concentration

tuning_var      = np.random.rand(nNeurons) #how strongly tuned neurons are

ori_trials      = np.random.choice(oris,nTrials)

R = np.empty((nNeurons,nTrials))
for iN in range(nNeurons):
    tuned_resp = vonmises.pdf(np.deg2rad(ori_trials), loc=locs[iN], kappa=kappa)
    R[iN,:] = (tuned_resp / np.max(tuned_resp)) * tuning_var[iN]

# plt.figure()
# plt.imshow(R)

# plt.scatter(ori_trials,R[23,:])

gain_trials = np.random.rand(nTrials)
gain_weights = np.random.randn(nNeurons) * gain_level
# gain_weights = np.random.rand(nNeurons) * gain_level

G = 1 + np.outer(gain_weights,gain_trials) 

offset_trials = np.random.rand(nTrials)
offset_weights = np.random.randn(nNeurons) * offset_level

O = np.outer(offset_weights,offset_trials) 

N = np.random.randn(nNeurons,nTrials) * noise_level

Full = R * G + O + N

model_ses = Session()
model_ses.respmat = Full
model_ses.trialdata = pd.DataFrame()
model_ses.trialdata['Orientation'] = ori_trials
model_ses.respmat_runspeed = gain_trials
model_ses.sessiondata = pd.DataFrame()
model_ses.sessiondata['protocol'] = 'GR'

fig = plot_PCA_gratings(model_ses)

fig.savefig(os.path.join(savedir,'AffineModel_Gain%1.2f_O%1.2f_noise%1.2f_N%d_K%d' % (gain_level,offset_level,noise_level,nNeurons,nTrials) + '.png'), format = 'png')

############################ Compute noise correlations: ###################################
model_ses = compute_noise_correlation([model_ses])[0]

##########################################################################################
# Plot noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

fig,ax = plt.subplots(1,1,figsize=(5,5))

tuning_perc_labels = np.linspace(0,100,11)
tuning_percentiles  = np.percentile(tuning_var,tuning_perc_labels)
clrs_percentiles    = sns.color_palette('inferno', len(tuning_percentiles))

for ip in range(len(tuning_percentiles)-1):

    filter = np.logical_and(tuning_percentiles[ip] <= tuning_var,
                            tuning_var <= tuning_percentiles[ip+1])

    df = pd.DataFrame({'NoiseCorrelation': model_ses.noise_corr[filter,:].flatten(),
                    'DeltaPrefOri': model_ses.delta_pref[filter,:].flatten()}).dropna(how='all')

    deltapreforis = np.sort(df['DeltaPrefOri'].unique())
    histdata            = df.groupby(['DeltaPrefOri'], as_index=False)['NoiseCorrelation'].mean()

    plt.plot(histdata['DeltaPrefOri'], 
            histdata['NoiseCorrelation'],
            color=clrs_percentiles[ip])
    
plt.xlabel('Delta Ori')
plt.ylabel('NoiseCorrelation')
        
plt.legend(tuning_perc_labels[1:],fontsize=9,loc='best')
plt.tight_layout()

# fig.savefig(os.path.join(savedir,'NoiseCorr_PosWeight_AffineModel_Gain%1.2f_O%1.2f_noise%1.2f_N%d_K%d' % (gain_level,offset_level,noise_level,nNeurons,nTrials) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'NoiseCorr_RandWeight_AffineModel_Gain%1.2f_O%1.2f_noise%1.2f_N%d_K%d' % (gain_level,offset_level,noise_level,nNeurons,nTrials) + '.png'), format = 'png')


#### 
# Take two neurons with strong tuning and large delta pref ori:

### Fit Affine model to data:


##############################################################################
# session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE10885','2023_10_23']])
# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=False, 
                                    load_calciumdata=False, load_videodata=False, calciumversion='deconv')

#   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    
    ##############################################################################
    ## Construct trial response matrix:  N neurons by K trials
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

    sessions[ises].respmat_runspeed = compute_respmat(sessions[ises].behaviordata['runspeed'],
                                                      sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
                                                    t_resp_start=0,t_resp_stop=1,method='mean')

    sessions[ises].respmat_videome = compute_respmat(sessions[ises].videodata['motionenergy'],
                                                    sessions[ises].videodata['timestamps'],sessions[ises].trialdata['tOnset'],
                                                    t_resp_start=0,t_resp_stop=1,method='mean')


R = sessions[ises].respmat.T

R = R[:,:1000]
data = sessions[ises].respmat

orientations = sessions[ises].trialdata['Orientation']
stims = sessions[ises].trialdata['Orientation']
istims = sessions[ises].trialdata['Orientation'].to_numpy()
ustim,istims = np.unique(sessions[ises].trialdata['Orientation'],return_index=True)
ustim,istimeses,istims = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)

istims = 

import numpy as np
from scipy.linalg import norm

def fitAffine(R, stims, estimate_additive=True):
    """
    Fit an affine model to visual cortical responses.

    Parameters:
        R (ndarray): K x N  array representing neural responses, where N is the number of neurons
                        and K is the number of trials.
        stims (ndarray): 1D array of length K representing the stimulus presented on each trial.
        estimate_additive (bool): Whether to estimate the additive component (default is True).

    Returns:
        # varexp (float): Variance explained by the model.
        # gain (ndarray): Array of length K representing the multiplicative gain for each trial.
        # Rfit (ndarray): N x K array representing the fitted responses.
        # sm (ndarray): Array representing the orientation-tuned response for each neuron.
    """
    # Normalize R
    # R = R / np.sqrt(np.sum(R**2, axis=1)[:, np.newaxis])
    R = R - R.min(axis=0)
    R = R / R.max(axis=0)

    ntrials, nneurons = R.shape
   
    # N, K = data.shape
    u_stims = np.unique(stims)
    nstim = len(u_stims)

    offset = np.ones((ntrials, 1))
    gain = np.ones((ntrials, 1))

    # Initialize stimuli with mean response
    # R = gain * sm + offset * soff

    R_mean  = np.array([np.mean(R[istims == i, :], axis=0) for i in range(nstim)])

    #Get estimate of the response matrix purely from the mean response:
    sm      = R.copy() * 1000
    for istim,stim in enumerate(u_stims):
        sm[stims==stim,:] = R_mean[istim,:]

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
    ax1.imshow(R,vmin=np.percentile(R,5),vmax=np.percentile(R,95))
    ax2.imshow(sm,vmin=np.percentile(R,5),vmax=np.percentile(R,95))

    # sm = np.array([np.mean(R[istims == i, :], axis=0) for i in range(1, nstim + 1)])
    # sm = np.array([np.mean(R[istims == i, :], axis=0) for i in range(nstim)])
    # sm = np.array([np.mean(R[istims == i, :], axis=0) for i in range(nstim)])
    soff = np.ones((1, nneurons))
    sm = np.vstack((sm,soff))

    # sm = [];
    # for i = 1:nstim
    #     sm(i,:) = mean(R(istims==i,:),1);
    # end

    gain0 = gain.copy()
    offset0 = offset.copy()

    Rfit = sm[:-1, :]
    cost = np.mean((R - Rfit)**2, axis=0)

    Rrez = R.copy()
    for _ in range(10):
        for i in range(nstim):
            if estimate_additive:
                goff = np.linalg.lstsq(sm[[i, nstim], :].T @ sm[[i, nstim], :], sm[[i, nstim], :].T @ R[istims == i, :].T, rcond=None)[0].T
                offset[istims == i] = goff[:, 1]
            else:
                goff = np.linalg.lstsq(sm[i, :].T @ sm[i, :], sm[i, :].T @ Rrez[istims == i, :].T, rcond=None)[0].T
            gain[istims == i] = goff[:, 0]

        gdesign = np.zeros((ntrials, nstim + 1))
        gdesign[:, nstim] = offset.flatten()
        for n in range(nstim):
            for i in range(nstim):
                gdesign[istims == i, i] = gain[istims == i].flatten()
            xtx = gdesign.T @ gdesign / ntrials + 1e-4 * np.eye(nstim + 1)
            xty = gdesign.T @ R[:, n] / ntrials
            sm[:, n] = np.linalg.lstsq(xtx, xty, rcond=None)[0]
            Rfit[:, n] = gdesign @ sm[:, n]

        if not estimate_additive:
            Rrez = R - sm[-1, :]

        cost = np.mean((R - Rfit)**2, axis=0)

        sm = norm(sm, axis=1, ord=1) 

    varexp = 1 - np.mean(cost / np.var(R, axis=0))
    
    Rtrain = np.empty((0, nstim))
    Rtest = np.empty((0, nstim))
    RtrainFit = np.empty((0, nstim))
    RtestFit = np.empty((0, nstim))
    for isti in range(1, 33):
        isa = np.where(istims == isti)[0]
        iss = np.random.permutation(len(isa))
        iss = isa[iss]
        ni = len(iss)
        RtrainFit = np.concatenate((RtrainFit, Rfit[iss[:ni // 2], :]), axis=0)
        RtestFit = np.concatenate((RtestFit, Rfit[iss[ni // 2:ni], :]), axis=0)
        Rtrain = np.concatenate((Rtrain, R[iss[:ni // 2], :]), axis=0)
        Rtest = np.concatenate((Rtest, R[iss[ni // 2:ni], :]), axis=0)
    
    vsignal = np.mean(np.mean((Rtrain - RtrainFit) * (Rtest - RtestFit)))
    vsignal = np.mean(np.mean(RtrainFit * RtestFit))

    return varexp, gain.flatten(), Rfit, sm


def fit_affine_model(data, orientations, estimate_additive=True):
    """
    Fit an affine model to visual cortical responses.

    Parameters:
        data (ndarray): N x K array representing neural responses, where N is the number of neurons
                        and K is the number of trials.
        orientations (ndarray): 1D array of length K representing the orientation presented on each trial.
        estimate_additive (bool): Whether to estimate the additive component (default is True).

    Returns:
        varexp (float): Variance explained by the model.
        gain (ndarray): Array of length K representing the multiplicative gain for each trial.
        Rfit (ndarray): N x K array representing the fitted responses.
        sm (ndarray): Array representing the orientation-tuned response for each neuron.
    """

    N, K = data.shape
    u_orientations = np.unique(orientations)
    n_orientations = len(u_orientations)

    # Calculate orientation-tuned response for each neuron
    r_t = np.array([np.mean(data[:, orientations == i], axis=1) for i in u_orientations])
    r_t = np.vstack((r_t, np.ones(N))).T  # Add intercept term

    # Initialize gain parameters:
    g_n = np.ones(N)
    g_w = np.ones(K)
    # Initialize additive parameters:
    a_n = np.ones(N)
    a_w = np.ones(K)

    Rfit = np.zeros((N, K))

    G = 1 + np.outer(gain_weights,gain_trials) 

    A = sm[[i, -1], :].T @ sm[[i, -1], :]
    B = sm[[i, -1], :].T @ data[:, orientations == ori].T

    A = 
    B = 
    np.linalg.lstsq(A,B,rcond=None)[0][0]

    for _ in range(10):
        # Update gain
        for i,ori in enumerate(u_orientations):
            if estimate_additive:

                gain[orientations == ori] = np.linalg.lstsq(sm[[i, -1], :].T @ sm[[i, -1], :],
                                                              sm[[i, -1], :].T @ data[:, orientations == ori].T,
                                                              rcond=None)[0][0]
                
                # gain[orientations == ori] = np.linalg.lstsq(sm[[i, -1], :].T @ sm[[i, -1], :],
                #                                               sm[[i, -1], :].T @ data[:, orientations == ori].T,
                #                                               rcond=None)[0][0]
            else:
                gain[orientations == i + 1] = np.linalg.lstsq(sm[i, :].T @ sm[i, :], sm[i, :].T @ data[:, orientations == i + 1].T,
                                                              rcond=None)[0][0]

        # Update Rfit
        for i in range(n_orientations):
            Rfit[:, orientations == i + 1] = gain[orientations == i + 1] * sm[i, :][:, np.newaxis]

        if not estimate_additive:
            data = data - sm[-1, :][:, np.newaxis]

        # Calculate variance explained
        cost = np.mean((data - Rfit)**2, axis=1)
        varexp = 1 - np.mean(cost / np.var(data, axis=1))

        sm = sm / np.linalg.norm(sm, axis=1, ord=2, keepdims=True)

    return varexp, gain, Rfit, sm

import numpy as np
from scipy.linalg import norm

# def fitAffine(R, istims, estimate_additive):
#     # Normalize R
#     R = R / np.sqrt(np.sum(R**2, axis=1)[:, np.newaxis])
    
#     ntrials, nstim = R.shape

#     offset = np.ones((ntrials, 1))
#     gain = np.ones((ntrials, 1))

#     # Initialize stimuli with mean response
#     sm = np.array([np.mean(R[istims == i, :], axis=0) for i in range(1, nstim + 1)])
#     sm = np.vstack((sm, np.ones((1, nstim))))

#     gain0 = gain.copy()
#     offset0 = offset.copy()

#     Rfit = sm[istims - 1, :]
#     cost = np.mean((R - Rfit)**2, axis=0)

#     Rrez = R.copy()
#     for _ in range(10):
#         for i in range(nstim):
#             if estimate_additive:
#                 goff = np.linalg.lstsq(sm[[i, nstim], :].T @ sm[[i, nstim], :], sm[[i, nstim], :].T @ R[istims == i, :].T, rcond=None)[0].T
#                 offset[istims == i] = goff[:, 1]
#             else:
#                 goff = np.linalg.lstsq(sm[i, :].T @ sm[i, :], sm[i, :].T @ Rrez[istims == i, :].T, rcond=None)[0].T
#             gain[istims == i] = goff[:, 0]

#         gdesign = np.zeros((ntrials, nstim + 1))
#         gdesign[:, nstim] = offset.flatten()
#         for n in range(nstim):
#             for i in range(nstim):
#                 gdesign[istims == i, i] = gain[istims == i].flatten()
#             xtx = gdesign.T @ gdesign / ntrials + 1e-4 * np.eye(nstim + 1)
#             xty = gdesign.T @ R[:, n] / ntrials
#             sm[:, n] = np.linalg.lstsq(xtx, xty, rcond=None)[0]
#             Rfit[:, n] = gdesign @ sm[:, n]

#         if not estimate_additive:
#             Rrez = R - sm[-1, :]

#         cost = np.mean((R - Rfit)**2, axis=0)

#         sm = norm(sm, axis=1, ord=1)

#     varexp = 1 - np.mean(cost / np.var(R, axis=0))
    
#     Rtrain = np.empty((0, nstim))
#     Rtest = np.empty((0, nstim))
#     RtrainFit = np.empty((0, nstim))
#     RtestFit = np.empty((0, nstim))
#     for isti in range(1, 33):
#         isa = np.where(istims == isti)[0]
#         iss = np.random.permutation(len(isa))
#         iss = isa[iss]
#         ni = len(iss)
#         RtrainFit = np.concatenate((RtrainFit, Rfit[iss[:ni // 2], :]), axis=0)
#         RtestFit = np.concatenate((RtestFit, Rfit[iss[ni // 2:ni], :]), axis=0)
#         Rtrain = np.concatenate((Rtrain, R[iss[:ni // 2], :]), axis=0)
#         Rtest = np.concatenate((Rtest, R[iss[ni // 2:ni], :]), axis=0)
    
#     vsignal = np.mean(np.mean((Rtrain - RtrainFit) * (Rtest - RtestFit)))
#     vsignal = np.mean(np.mean(RtrainFit * RtestFit))

#     return varexp, gain.flatten(), Rfit, sm
