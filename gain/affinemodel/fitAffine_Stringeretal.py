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
from tqdm import tqdm

os.chdir('e:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive

from scipy.stats import vonmises
from utils.explorefigs import plot_PCA_gratings
from loaddata.session import Session
from utils.corr_lib import *
from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning_wrapper
from utils.plot_lib import * #get all the fixed color schemes
from utils.gain_lib import * 
from utils.plot_lib import shaded_error

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\GainModel\\')

#%% Based on Stringer et al: 
# https://github.com/MouseLand/stringer-pachitariu-et-al-2018a/tree/master/stimspont


#%% Explore this lib as well for multiplicative gain fit with alternating least squares: 
# https://github.com/jcbyts/V1Locomotion/tree/main


#===============================================================================
#                     Fit Affine model to GR data
#===============================================================================

#%% #############################################################################
#Put data in rootdir/Procdata/GR/LPE12223/2024_06_10/
#Add the rootdir on working PC (USER) in get_data_folder()
# loaddata.get_data_folder()

session_list        = np.array([['LPE12223_2024_06_10']])

# load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)
                                # calciumversion='dF',keepraw=True)

#%% 
fig = plot_PCA_gratings(sessions[ises])

#===============================================================================
#                     Fit Affine model to GN data
#===============================================================================

#%% #############################################################################
session_list        = np.array([['LPE12385','2024_06_13']])

# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GN',session_list=session_list)

#   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)




#%% #########################################################################################

# Example usage:
R = sessions[ises].respmat.T
stims = sessions[ises].trialdata['Orientation']
ustim,istimeses,stims  = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)

gain, gain_weights, data_hat = fit_multiplicative_model(R, stims)


#%% 


orientations            = sessions[ises].trialdata['Orientation']
stims                   = sessions[ises].trialdata['Orientation']
istims                  = sessions[ises].trialdata['Orientation'].to_numpy()
ustim,istims            = np.unique(sessions[ises].trialdata['Orientation'],return_index=True)
ustim,istimeses,istims  = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)

[varexp, gain, data_hat, sm] = fitAffine(R, stims, estimate_additive=False)

plt.hist(np.nanmean(R,axis=1),bins=50)

[varexp, gain, data_hat, sm] = fitAffine(R, stims, estimate_additive=False)

#%% 
R                    = sessions[ises].respmat
ustim,istimeses,istims  = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)


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


#%% From Montijn & Heimel, 2024 bioRxiv
# Also fitting a multiplicative model:

# def getProjOnLine(matPoints, vecRef):
#     """
#     Projects points onto a reference vector, and returns projected locations, points, and norms.

#     Parameters:
#         matPoints (ndarray): K x N  array representing neural responses, where N is the number of neurons
#         vecRef (ndarray): D x 1 array representing the reference vector

#     Returns:
#         vecProjectedLocation (ndarray): norm of projected points along reference vector (dimensionality-dependent)
#         matProjectedPoints (ndarray): ND locations of projected points
#         vecProjLocDimNorm (ndarray): norm of projected points, normalized for dimensionality (i.e., /sqrt(D))
#     """

#     intD = matPoints.shape[0]
#     intPoints = matPoints.shape[1]
#     if intPoints < intD:
#         raise ValueError('Number of dimensions is larger than number of points; please make sure matrix is in form [Trials x Neurons]')

#     assert vecRef.ndim == 1, 'Reference vector input is not a [D x 1] vector'
#     assert vecRef.shape[0] == intD, 'Reference vector input has a different dimensionality to points matrix'

#     # recenter
#     matProj = np.outer(vecRef, vecRef) / np.dot(vecRef, vecRef)
#     vecNormRef = vecRef / np.linalg.norm(vecRef)

#     # calculate projected points
#     matProjectedPoints = np.nan * np.ones(matPoints.shape)
#     vecProjectedLocation = np.nan * np.ones((matPoints.shape[1],))
#     for intTrial in range(matPoints.shape[1]):
#         vecPoint = matPoints[:, intTrial]
#         vecOrth = matProj @ vecPoint
#         matProjectedPoints[:, intTrial] = vecOrth
#         vecProjectedLocation[intTrial] = np.dot(vecOrth, vecNormRef)

#     vecProjLocDimNorm = vecProjectedLocation / np.sqrt(intD)  # normalize for number of dimensions so 1 is the norm of the reference vector

#     return vecProjectedLocation, matProjectedPoints, vecProjLocDimNorm


# #%%
# matCounts       = sessions[ises].respmat
# # vecRealMean     = matMean[:, intScale]
# # vecRealSd       = matSd[:, intScale]
# vecRealMean     = np.nanmean(matCounts, axis=1)
# vecRealSd       = np.nanstd(matCounts, axis=1)


# # define source parameters
# intNumN, intNumT = matCounts.shape
# vecReqMu = np.nanmean(matCounts, axis=1)
# vecReqSd = np.nanstd(matCounts, axis=1)
# vecReqVar = vecReqSd ** 2
# matReqCov = np.diag(vecReqVar)

# # matCountsLogNormal = np.random.lognormal(vecReqMu, matReqCov, (intNumT))
# # matCountsLogNormal = np.random.lognormal(vecReqMu, vecReqSd, (intNumN, intNumT))
# matCountsLogNormal = np.random.lognormal(vecReqMu[:, None], vecReqSd[:, None], (intNumN, intNumT))
# plt.hist(matCounts.flatten(),np.arange(-0.5,2,0.1))
# plt.hist(matCountsLogNormal.flatten(),np.arange(-0.5,5,0.1))

# # fit simple gain-scaling model
# # gain with no off-axis noise; n+2 free params: vecGainAxis, dblGainMean, dblGainSd
# vecGainAxis = vecReqMu / np.linalg.norm(vecReqMu)
# vecProjectedLocation, matProjectedPoints, vecProjLocDimNorm = getProjOnLine(matCounts, vecGainAxis)
# dblGainRange = np.std(vecProjectedLocation)
# dblGainMean = np.mean(vecProjectedLocation)

# # generate random gains
# vecPopGainPerTrial = np.random.lognormal(dblGainMean, dblGainRange**2, intNumT)

# # prediction is on-axis gain
# # matCountsGain = np.nan * np.ones(matCountsLogNormal.shape)
# matCountsGain = np.nan * np.ones(matCounts.shape)
# for intT in range(intNumT):
#     dblThisGain = vecPopGainPerTrial[intT]
#     vecOnAxisAct = vecGainAxis * dblThisGain
#     matCountsGain[:, intT] = vecOnAxisAct

# #%% 
# # fit gain-scaling model split by stim ori
# # gain per stim with no off-axis noise; 16*(n+2) free params: vecGainAxis, dblGainMean, dblGainSd
# matCountsGainStim = np.nan * np.ones(matCountsLogNormal.shape)
# for intStimIdx in range(len(vecUnique)):
#     vecTrials = np.where(cellStimIdx[intScale] == intStimIdx)[0]
#     matSubCounts = matCounts[:, vecTrials]
    
#     vecReqMu = np.mean(matSubCounts, axis=1)
#     vecReqSd = np.std(matSubCounts, axis=1)
    
#     # gain with no off-axis noise; n+2 free params: vecGainAxis, dblGainMean, dblGainSd
#     vecGainAxis = vecReqMu / np.linalg.norm(vecReqMu)
#     vecProjectedLocation, matProjectedPoints, vecProjLocDimNorm = getProjOnLine(matSubCounts, vecGainAxis)
#     dblGainRange = np.std(vecProjectedLocation)
#     dblGainMean = np.mean(vecProjectedLocation)
    
#     # generate random gains
#     intNumStimT = len(vecTrials)
#     vecPopGainPerTrial = logmvnrnd(dblGainMean, dblGainRange**2, intNumStimT)
    
#     # prediction is on-axis gain
#     for intT in range(intNumStimT):
#         dblThisGain = vecPopGainPerTrial[intT]
#         vecOnAxisAct = vecGainAxis * dblThisGain
#         matCountsGainStim[:, vecTrials[intT]] = vecOnAxisAct
