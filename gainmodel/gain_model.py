#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.stats import vonmises
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score

os.chdir('e:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive

from scipy.stats import vonmises
from utils.explorefigs import plot_PCA_gratings
from loaddata.session import Session
from utils.corr_lib import *
from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes
from utils.gain_lib import * 

savedir = 'D:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\GainModel\\'

#%%  
nNeurons        = 1000
nTrials         = 3200

noise_level     = 15
gain_level      = 5
offset_level    = 0

noris           = 16

oris            = np.linspace(0,360,noris+1)[:-1]
locs            = np.random.rand(nNeurons) * np.pi * 2  # circular mean
kappa           = 2  # concentration

tuning_var      = np.random.rand(nNeurons) #how strongly tuned neurons are

ori_trials      = np.random.choice(oris,nTrials)

R = np.empty((nNeurons,nTrials))
for iN in range(nNeurons):
    tuned_resp = vonmises.pdf(np.deg2rad(ori_trials), loc=locs[iN], kappa=kappa)
    R[iN,:] = (tuned_resp / np.max(tuned_resp)) * tuning_var[iN]

# plt.figure()
# plt.imshow(R)
# plt.scatter(ori_trials,R[23,:])

# The distribution of gains across trials determines the distribution of points within each column of the cone
# I.e. if gain is rand <0,1> then there are trials with zero gain. If gain is strongly dependent on locomotion
# then there are many trials with large gains in sessions where the mouse is continuously moving.
# However, it seems that gain (1 + weights * trials) should be positive, otherwise tuning response is inverted

gain_trials = np.random.rand(nTrials)
gain_weights = np.random.randn(nNeurons) * gain_level

gain_trials = np.random.rand(nTrials)
gain_weights = np.random.normal(0,1,nNeurons) * gain_level

gain_trials = sessions[ises].respmat_runspeed+1
# gain_trials = np.random.lognormal(0,1,nTrials)
gain_weights = (np.random.lognormal(0,1,nNeurons)-1) * gain_level
gain_weights = np.random.lognormal(0,1,nNeurons) * gain_level

gain_weights = [np.corrcoef(np.nanmean(R,axis=0),R[n,:])[0,1] for n in range(R.shape[0])]
gain_trials = np.nanmean(R,axis=0)


G = 1 + np.outer(gain_weights,gain_trials) 

offset_trials = np.random.rand(nTrials)
offset_weights = np.random.randn(nNeurons) * offset_level

O = np.outer(offset_weights,offset_trials) 

N = np.random.randn(nNeurons,nTrials) * noise_level

# model is tuned response (R) multiplied by a gain (G) + offset (O) + noise (N)
Full = R * G + O + N 

model_ses = Session()
model_ses.respmat = Full
model_ses.trialdata = pd.DataFrame()
model_ses.trialdata['Orientation'] = ori_trials
model_ses.respmat_runspeed = gain_trials
model_ses.sessiondata = pd.DataFrame()
model_ses.sessiondata['protocol'] = ['GR']

fig = plot_PCA_gratings(model_ses,apply_zscore=True)

# fig.savefig(os.path.join(savedir,'AffineModel_Gain%1.2f_O%1.2f_noise%1.2f_N%d_K%d' % (gain_level,offset_level,noise_level,nNeurons,nTrials) + '.png'), format = 'png')

#%% ########################### Compute noise correlations: ###################################
model_ses = compute_signal_noise_correlation([model_ses],filter_stationary=False)[0]

##########################################################################################
# Plot noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

fig,ax = plt.subplots(1,1,figsize=(5,5))

tuning_perc_labels = np.linspace(0,100,11)
tuning_percentiles  = np.percentile(tuning_var,tuning_perc_labels)
clrs_percentiles    = sns.color_palette('inferno', len(tuning_percentiles))

for ip in range(len(tuning_percentiles)-1):

    filter_tuning = np.logical_and(tuning_percentiles[ip] <= tuning_var,
                            tuning_var <= tuning_percentiles[ip+1])

    df = pd.DataFrame({'NoiseCorrelation': model_ses.noise_corr[filter_tuning,:].flatten(),
                    'DeltaPrefOri': model_ses.delta_pref[filter_tuning,:].flatten()}).dropna(how='all')

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
# fig.savefig(os.path.join(savedir,'NoiseCorr_RandWeight_AffineModel_Gain%1.2f_O%1.2f_noise%1.2f_N%d_K%d' % (gain_level,offset_level,noise_level,nNeurons,nTrials) + '.png'), format = 'png')


#===============================================================================
#                     Fit Affine model to data
#===============================================================================

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE12223','2024_06_10']])
# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

#   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)


#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    if sessions[ises].sessiondata['protocol'].isin(['GR','GN'])[0]:
        sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='OSI')
        sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='gOSI')
        sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='tuning_var')
        sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'])

#%% Filter only well tuned neurons in V1:
idx                                 = np.all((sessions[ises].celldata['roi_name']=='V1',sessions[ises].celldata['tuning_var']>0.05),axis=0)
sessions[ises].respmat              = sessions[ises].respmat[idx,:]
sessions[ises].celldata             = sessions[ises].celldata[idx]
sessions[ises].calciumdata          = sessions[ises].calciumdata.iloc[:,idx]

#%% Figure of raw data correlation of population activity, video ME, runspeed, and pupil area:
lw = 0.25
fig,ax = plt.subplots(1,1,figsize=(9,3))
ax.plot(minmax_scale(np.nanmean(sessions[ises].respmat,axis=0), feature_range=(0, 1), axis=0, copy=True),color='k',linewidth=lw,label='pop. mean')
ax.plot(minmax_scale(sessions[ises].respmat_videome, feature_range=(0, 1), axis=0, copy=True),color='g',linewidth=lw,label='video ME')
ax.plot(minmax_scale(sessions[ises].respmat_runspeed, feature_range=(0, 1), axis=0, copy=True),color='r',linewidth=lw,label='runspeed')
ax.plot(minmax_scale(sessions[ises].respmat_pupilarea, feature_range=(0, 1), axis=0, copy=True),color='b',linewidth=lw,label='pupil area')
ax.set_xlim([0,3200])
ax.legend(frameon=False,loc='upper right')
ax.set_xlabel('Trials')
ax.set_ylabel('Normalized response')
plt.tight_layout()

#%% CLOSE UP Figure of raw data correlation of population activity, video ME, runspeed, and pupil area:
lw = 1
fig,ax = plt.subplots(1,1,figsize=(9,3))
ax.plot(minmax_scale(np.nanmean(sessions[ises].respmat,axis=0), feature_range=(0, 1), axis=0, copy=True),color='k',linewidth=lw*2,label='pop. mean')
ax.plot(minmax_scale(sessions[ises].respmat_videome, feature_range=(0, 1), axis=0, copy=True),color='g',linewidth=lw,label='video ME')
ax.plot(minmax_scale(sessions[ises].respmat_runspeed, feature_range=(0, 1), axis=0, copy=True),color='r',linewidth=lw,label='runspeed')
ax.plot(minmax_scale(sessions[ises].respmat_pupilarea, feature_range=(0, 1), axis=0, copy=True),color='b',linewidth=lw,label='pupil area')
ax.set_xlim([500,650])
ax.legend(frameon=False,loc='upper right')
ax.set_xlabel('Trials')
ax.set_ylabel('Normalized response')
plt.tight_layout()

#%% 
fig = plot_PCA_gratings(sessions[ises])


#%% 
data                = sessions[ises].respmat
# data                = sessions[ises].respmat - np.mean(sessions[ises].respmat, axis=1, keepdims=True)
data_hat_tuned      = tuned_resp_model(data, orientations)
data_hat_poprate    = pop_rate_gain_model(data, orientations)

# datasets            = (data,data_hat_tuned)
# fig = plot_respmat(orientations, datasets, ['original','mean tuning'])
# fig = plot_tuned_response(orientations, datasets, ['original','tuned response'])

datasets            = (data,data_hat_tuned,data_hat_poprate)
fig = plot_respmat(orientations, datasets, ['original','mean tuning','pop rate gain'])
# fig = plot_tuned_response(orientations, datasets, ['original','mean tuning','pop rate gain'])

#%% Baseline PCA:
fig = plot_PCA_gratings(sessions[ises])

#%% PCA for session without mean tuned response:
ses = copy.deepcopy(sessions[ises])
ses.respmat = sessions[ises].respmat - data_hat_tuned
fig = plot_PCA_gratings(ses)

#%% PCA for session without population rate gain model:
ses = copy.deepcopy(sessions[ises])
ses.respmat = sessions[ises].respmat - data_hat_poprate
fig = plot_PCA_gratings(ses)

#%% Baseline Noise correlations
ses = copy.deepcopy(sessions[ises])
ses = compute_signal_noise_correlation([ses],filter_stationary=False)[0]
fig = plot_noise_corr_deltaori(ses)

#%% Noise correlations for session without mean tuned response:
ses = copy.deepcopy(sessions[ises])
ses.respmat = sessions[ises].respmat - data_hat_tuned
ses = compute_signal_noise_correlation([ses],filter_stationary=False)[0]
fig = plot_noise_corr_deltaori(ses)

#%% Noise correlations for session without population rate gain model:
ses = copy.deepcopy(sessions[ises])
ses.respmat = sessions[ises].respmat - data_hat_poprate
ses = compute_signal_noise_correlation([ses],filter_stationary=False)[0]
fig = plot_noise_corr_deltaori(ses)

#%% ########################### Compute noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,filter_stationary=False)[0]
# [sessions] = compute_signal_noise_correlation([sessions],filter_stationary=False)[0]
sessions = [sessions]


#%% #########################################################################################
# Plot noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are


#%% 


# orientations            = sessions[ises].trialdata['Orientation']
# stims                   = sessions[ises].trialdata['Orientation']
# istims                  = sessions[ises].trialdata['Orientation'].to_numpy()
# ustim,istims            = np.unique(sessions[ises].trialdata['Orientation'],return_index=True)
# ustim,istimeses,istims  = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)

# [varexp, gain, data_hat, sm] = fitAffine(R, stims, estimate_additive=False)

# plt.hist(np.nanmean(R,axis=1),bins=50)


# [varexp, gain, data_hat, sm] = fitAffine(R, stims, estimate_additive=False)

# #%% 
# R                    = sessions[ises].respmat
# ustim,istimeses,istims  = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)


#%%
def fit_multiplicative_model(R, stims, estimate_additive=False):
    """
    Fit a multiplicative model to neuronal response data.

    Parameters:
        R (ndarray): K x N  array representing neural responses, where N is the number of neurons
                        and K is the number of trials.
        stims (ndarray): 1D array of length K representing the stimulus presented on each trial.
        estimate_additive (bool): Whether to estimate the additive component (default is False).

    Returns:
        gain (ndarray): Array of length K representing the multiplicative gain for each trial.
        gain_weights (ndarray): Array of length N representing the multiplicative gain weights for each neuron.
        data_hat (ndarray): N x K array representing the fitted responses.
    """
    ntrials, nneurons = R.shape
    nstim = len(np.unique(stims))

    # Calculate mean response per stimulus
    sm = np.array([np.mean(R[stims == i, :], axis=0) for i in range(nstim)])

    # Initialize gain parameters randomly
    gain = np.random.rand(ntrials)
    gain_weights = np.random.rand(nneurons)

    # Initialize fitted responses
    data_hat = np.zeros_like(R)

    # Fit the model iteratively
    for _ in range(100):  # number of iterations
        for i in range(nstim):
            stim_trials = stims == i
            data_hat[stim_trials, :] = sm[i, :] * gain[stim_trials, None] * gain_weights[None, :]

        # Calculate residuals
        residuals = R - data_hat

        # Update gain parameters    
        gain, gain_weights = update_gain(residuals, gain, gain_weights)


    return gain, gain_weights, data_hat

# Example usage:
R = sessions[ises].respmat.T
stims = sessions[ises].trialdata['Orientation']
ustim,istimeses,stims  = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)

gain, gain_weights, data_hat = fit_multiplicative_model(R, stims)

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
        # data_hat (ndarray): N x K array representing the fitted responses.
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

    data_hat = sm[:-1, :]
    cost = np.mean((R - data_hat)**2, axis=0)

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
            data_hat[:, n] = gdesign @ sm[:, n]

        if not estimate_additive:
            Rrez = R - sm[-1, :]

        cost = np.mean((R - data_hat)**2, axis=0)

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
        RtrainFit = np.concatenate((RtrainFit, data_hat[iss[:ni // 2], :]), axis=0)
        RtestFit = np.concatenate((RtestFit, data_hat[iss[ni // 2:ni], :]), axis=0)
        Rtrain = np.concatenate((Rtrain, R[iss[:ni // 2], :]), axis=0)
        Rtest = np.concatenate((Rtest, R[iss[ni // 2:ni], :]), axis=0)
    
    vsignal = np.mean(np.mean((Rtrain - RtrainFit) * (Rtest - RtestFit)))
    vsignal = np.mean(np.mean(RtrainFit * RtestFit))

    return varexp, gain.flatten(), data_hat, sm





# def fit_affine_model(data, orientations, estimate_additive=True):
#     """
#     Fit an affine model to visual cortical responses.

#     Parameters:
#         data (ndarray): N x K array representing neural responses, where N is the number of neurons and
#                         and K is the number of trials.
#         orientations (ndarray): 1D array of length K representing the orientation presented on each trial.
#         estimate_additive (bool): Whether to estimate the additive component (default is True).

#     Returns:
#         varexp (float): Variance explained by the model.
#         gain (ndarray): Array of length K representing the multiplicative gain for each trial.
#         data_hat (ndarray): N x K array representing the fitted responses.
#         sm (ndarray): Array representing the orientation-tuned response for each neuron.
#     """

#     N, K = data.shape
#     u_orientations = np.unique(orientations)
#     n_orientations = len(u_orientations)

#     # Calculate orientation-tuned response for each neuron
#     sm = np.array([np.mean(data[:, orientations == i], axis=1) for i in u_orientations])
#     sm = np.vstack((sm, np.ones(N)))  # Add intercept term

#     # Initialize gain parameters:
#     g_n = np.ones(N)
#     g_k = np.ones(K)

#     # Initialize additive parameters:
#     a_n = np.ones(N)
#     a_w = np.ones(K)

#     data_hat = np.zeros((N, K))

#     # G = 1 + np.outer(gain_weights,gain_trials) 

#     # A = r_t[[i, -1], :].T @ sm[[i, -1], :]
#     # B = r_t[[i, -1], :].T @ data[:, orientations == ori].T

#     # A = 
#     # B = 

#     np.linalg.lstsq(A,B,rcond=None)[0][0]
#     np.shape(sm[[i, -1], :].T @ sm[[i, -1], :])
#     np.shape(sm[[i, -1], :] @ sm[[i, -1], :].T)
    
#     for _ in range(10):
#         # Update gain
#         for i,ori in enumerate(u_orientations):
#             if estimate_additive:

                
#                 # g_k[orientations == ori] = np.linalg.lstsq(sm[[i, -1], :].T @ sm[[i, -1], :],
#                 #                                               sm[[i, -1], :] @ data[:, orientations == ori],
#                 #                                               rcond=None)[0][0]
                
#                 # g_k[orientations == ori] = np.linalg.lstsq(sm[[i, -1], :].T @ sm[[i, -1], :],
#                 #                                               sm[[i, -1], :].T @ data[:, orientations == ori].T,
#                 #                                               rcond=None)[0][0]
            
#             else:
#                 a @ x = b
#                 a = sm[i, :].T @ sm[i, :]
#                 np.shape(a)

#                 g_k[orientations == i + ] = np.linalg.lstsq(sm[i, :] @ sm[i, :].T, sm[i, :].T @ data[:, orientations == i + 1].T,rcond=None)
#                 [0][0]

#         # Update data_hat
#         for i in range(n_orientations):
#             data_hat[:, orientations == i + 1] = g_k[orientations == i + 1] * sm[i, :][:, np.newaxis]

#         if not estimate_additive:
#             data = data - sm[-1, :][:, np.newaxis]

#         # Calculate variance explained
#         cost = np.mean((data - data_hat)**2, axis=1)
#         varexp = 1 - np.mean(cost / np.var(data, axis=1))

#         sm = sm / np.linalg.norm(sm, axis=1, ord=2, keepdims=True)

#     return varexp, g_k, data_hat, sm


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

#     data_hat = sm[istims - 1, :]
#     cost = np.mean((R - data_hat)**2, axis=0)

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
#             data_hat[:, n] = gdesign @ sm[:, n]

#         if not estimate_additive:
#             Rrez = R - sm[-1, :]

#         cost = np.mean((R - data_hat)**2, axis=0)

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
#         RtrainFit = np.concatenate((RtrainFit, data_hat[iss[:ni // 2], :]), axis=0)
#         RtestFit = np.concatenate((RtestFit, data_hat[iss[ni // 2:ni], :]), axis=0)
#         Rtrain = np.concatenate((Rtrain, R[iss[:ni // 2], :]), axis=0)
#         Rtest = np.concatenate((Rtest, R[iss[ni // 2:ni], :]), axis=0)
    
#     vsignal = np.mean(np.mean((Rtrain - RtrainFit) * (Rtest - RtestFit)))
#     vsignal = np.mean(np.mean(RtrainFit * RtestFit))

#     return varexp, gain.flatten(), data_hat, sm
#%%

def getProjOnLine(matPoints, vecRef):
    """
    Projects points onto a reference vector, and returns projected locations, points, and norms.

    Parameters:
        matPoints (ndarray): K x N  array representing neural responses, where N is the number of neurons
        vecRef (ndarray): D x 1 array representing the reference vector

    Returns:
        vecProjectedLocation (ndarray): norm of projected points along reference vector (dimensionality-dependent)
        matProjectedPoints (ndarray): ND locations of projected points
        vecProjLocDimNorm (ndarray): norm of projected points, normalized for dimensionality (i.e., /sqrt(D))
    """

    intD = matPoints.shape[0]
    intPoints = matPoints.shape[1]
    if intPoints < intD:
        raise ValueError('Number of dimensions is larger than number of points; please make sure matrix is in form [Trials x Neurons]')

    assert vecRef.ndim == 1, 'Reference vector input is not a [D x 1] vector'
    assert vecRef.shape[0] == intD, 'Reference vector input has a different dimensionality to points matrix'

    # recenter
    matProj = np.outer(vecRef, vecRef) / np.dot(vecRef, vecRef)
    vecNormRef = vecRef / np.linalg.norm(vecRef)

    # calculate projected points
    matProjectedPoints = np.nan * np.ones(matPoints.shape)
    vecProjectedLocation = np.nan * np.ones((matPoints.shape[1],))
    for intTrial in range(matPoints.shape[1]):
        vecPoint = matPoints[:, intTrial]
        vecOrth = matProj @ vecPoint
        matProjectedPoints[:, intTrial] = vecOrth
        vecProjectedLocation[intTrial] = np.dot(vecOrth, vecNormRef)

    vecProjLocDimNorm = vecProjectedLocation / np.sqrt(intD)  # normalize for number of dimensions so 1 is the norm of the reference vector

    return vecProjectedLocation, matProjectedPoints, vecProjLocDimNorm


#%%
matCounts       = sessions[ises].respmat
# vecRealMean     = matMean[:, intScale]
# vecRealSd       = matSd[:, intScale]
vecRealMean     = np.nanmean(matCounts, axis=1)
vecRealSd       = np.nanstd(matCounts, axis=1)


# define source parameters
intNumN, intNumT = matCounts.shape
vecReqMu = np.nanmean(matCounts, axis=1)
vecReqSd = np.nanstd(matCounts, axis=1)
vecReqVar = vecReqSd ** 2
matReqCov = np.diag(vecReqVar)

# matCountsLogNormal = np.random.lognormal(vecReqMu, matReqCov, (intNumT))
# matCountsLogNormal = np.random.lognormal(vecReqMu, vecReqSd, (intNumN, intNumT))
matCountsLogNormal = np.random.lognormal(vecReqMu[:, None], vecReqSd[:, None], (intNumN, intNumT))
plt.hist(matCounts.flatten(),np.arange(-0.5,2,0.1))
plt.hist(matCountsLogNormal.flatten(),np.arange(-0.5,5,0.1))

# fit simple gain-scaling model
# gain with no off-axis noise; n+2 free params: vecGainAxis, dblGainMean, dblGainSd
vecGainAxis = vecReqMu / np.linalg.norm(vecReqMu)
vecProjectedLocation, matProjectedPoints, vecProjLocDimNorm = getProjOnLine(matCounts, vecGainAxis)
dblGainRange = np.std(vecProjectedLocation)
dblGainMean = np.mean(vecProjectedLocation)

# generate random gains
vecPopGainPerTrial = np.random.lognormal(dblGainMean, dblGainRange**2, intNumT)

# prediction is on-axis gain
# matCountsGain = np.nan * np.ones(matCountsLogNormal.shape)
matCountsGain = np.nan * np.ones(matCounts.shape)
for intT in range(intNumT):
    dblThisGain = vecPopGainPerTrial[intT]
    vecOnAxisAct = vecGainAxis * dblThisGain
    matCountsGain[:, intT] = vecOnAxisAct

#%% 
# fit gain-scaling model split by stim ori
# gain per stim with no off-axis noise; 16*(n+2) free params: vecGainAxis, dblGainMean, dblGainSd
matCountsGainStim = np.nan * np.ones(matCountsLogNormal.shape)
for intStimIdx in range(len(vecUnique)):
    vecTrials = np.where(cellStimIdx[intScale] == intStimIdx)[0]
    matSubCounts = matCounts[:, vecTrials]
    
    vecReqMu = np.mean(matSubCounts, axis=1)
    vecReqSd = np.std(matSubCounts, axis=1)
    
    # gain with no off-axis noise; n+2 free params: vecGainAxis, dblGainMean, dblGainSd
    vecGainAxis = vecReqMu / np.linalg.norm(vecReqMu)
    vecProjectedLocation, matProjectedPoints, vecProjLocDimNorm = getProjOnLine(matSubCounts, vecGainAxis)
    dblGainRange = np.std(vecProjectedLocation)
    dblGainMean = np.mean(vecProjectedLocation)
    
    # generate random gains
    intNumStimT = len(vecTrials)
    vecPopGainPerTrial = logmvnrnd(dblGainMean, dblGainRange**2, intNumStimT)
    
    # prediction is on-axis gain
    for intT in range(intNumStimT):
        dblThisGain = vecPopGainPerTrial[intT]
        vecOnAxisAct = vecGainAxis * dblThisGain
        matCountsGainStim[:, vecTrials[intT]] = vecOnAxisAct
