#%% 
import os, math
import numpy as np
from scipy.linalg import norm
from scipy.stats import vonmises, zscore
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import linregress,binned_statistic
from scipy.optimize import minimize

from utils.gain_lib import comp_poprate

#%% ###########################################################################
# NONLINEAR TRANSFER FUNCTION FITTING PIPELINE
# Model: r(t) = f( θ_k(t) + γ · P(t) + b )
#   θ_k  : stimulus drive — one free parameter per orientation (16)
#   γ    : population-rate scaling (additive input, 1 param)
#   b    : input bias (1 param)
#   f(·) : nonlinearity (with model-specific free parameters)
###########################################################################

#%% Define nonlinearities with fittable parameters
def nl_linear(u):
    return u

def nl_relu(u):
    return np.maximum(0.0, u)

def nl_softplus(u, beta):
    # f(u) = (1/β) log(1 + exp(β·u)); β controls sharpness (→ReLU as β→∞)
    b = np.abs(beta) + 1e-4
    bu = b * u
    return np.where(bu > 30.0, u, np.log1p(np.exp(np.clip(bu, -500.0, 30.0))) / b)

def nl_sigmoid(u, a):
    # maps sigmoid to [0, a]: f(u) = a · σ(u)
    return np.abs(a) / (1.0 + np.exp(5*-np.clip(u, -100.0, 100.0)))

def nl_tanh(u, a):
    # maps tanh's [-1,1] to [0, a]: f(u) = a · (1 + tanh(u)) / 2
    return np.abs(a) * 0.5 * (1.0 + np.tanh(u))

def nl_powerlaw(u, p):
    # f(u) = max(0,u)^p; p is the free exponent
    return np.power(np.maximum(0.0, u), np.abs(p) + 1e-4)

def nl_exp(u):
    # max(0, exp(u)-1), shifted so f(0)=0; output gain a is universal
    return np.maximum(0.0, np.expm1(np.clip(u, -500.0, 10.0)))

# Format: (name, nl_func, n_shape, p0_shape, bounds_shape)
# Responses are min-max normalised to [0,1] before fitting, so all nonlinearities
# operate in the same output regime without per-model gain/offset parameters.
NL_CONFIGS = [
    ('Linear',          nl_linear,   0, [],      []),
    ('ReLU',            nl_relu,     0, [],      []),
    ('Softplus',        nl_softplus, 1, [5.0],   [(0.01, 50.0)]),
    # ('Tanh',            nl_tanh,     1, [1.0],   [(0.0, None)]),
    ('Exp',             nl_exp,      0, [],      []),
    ('Power-law (p=2)', nl_powerlaw, 1, [2.0],   [(0.1,  4.0)]),
    ('Sigmoid',         nl_sigmoid,  1, [1],   [(0.0, None)]),
]

nl_names = [c[0] for c in NL_CONFIGS]
nNL      = len(NL_CONFIGS)

#%% Core fitting function

def fit_nl_models(resp, stim_ids, poprate, configs=NL_CONFIGS):
    """
    Fit all NL models to a single neuron's trial-by-trial responses.

    Model: r_norm = f( θ_k + γ·P + b )
      Responses are min-max normalised to [0,1] before fitting so all
      nonlinearities share the same output regime without per-model gain.
      Shared params: θ_k (nstim), γ, b  — warm-started via least squares.
      Per-model params: shape params only (e.g. softplus β, power-law p).

    Returns dict keyed by model name:
      r2, theta, gamma, b, nl_par, pred (in [0,1] space), u, resp_norm
    """
    nstim = int(stim_ids.max()) + 1
    nT    = len(resp)

    # Least-squares warm start on normalised responses
    X = np.zeros((nT, nstim + 2))
    for k in range(nstim):
        X[stim_ids == k, k] = 1.0
    X[:, nstim]     = poprate
    X[:, nstim + 1] = 1.0
    p_ls, _, _, _ = np.linalg.lstsq(X, resp, rcond=None)
    theta0 = p_ls[:nstim]
    gamma0 = p_ls[nstim]
    b0     = p_ls[nstim + 1]

    results = {}
    for name, nl_func, n_shape, p0_shape, bnds_shape in configs:
        p0     = np.concatenate([theta0, [gamma0, b0], p0_shape])
        bounds = [(None, None)] * (nstim + 2) + bnds_shape

        def _loss(params, _resp=resp, _sid=stim_ids, _pop=poprate,
                  _f=nl_func, _n=n_shape, _ns=nstim):
            u    = params[:_ns][_sid] + params[_ns] * _pop + params[_ns + 1]
            pred = _f(u, *params[_ns + 2: _ns + 2 + _n]) if _n else _f(u)
            return np.mean((_resp - pred) ** 2)

        try:
            opt   = minimize(_loss, p0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
            theta = opt.x[:nstim]
            gamma = opt.x[nstim]
            b     = opt.x[nstim + 1]
            shape = list(opt.x[nstim + 2: nstim + 2 + n_shape]) if n_shape else []
            u     = theta[stim_ids] + gamma * poprate + b
            pred  = nl_func(u, *shape) if n_shape else nl_func(u)
            r2    = r2_score(resp, pred)
            results[name] = dict(r2=r2, theta=theta, gamma=gamma, b=b,
                                 nl_par=shape, pred=pred, u=u,
                                 resp_norm=resp, success=opt.success)
        except Exception:
            results[name] = dict(r2=np.nan, theta=None, gamma=None, b=None,
                                 nl_par=None, pred=None, u=None,
                                 resp_norm=resp, success=False)
    return results


#%% Fit all neurons across all sessions and collect R², Gamma, Beta, theta, nl_par
def fit_nl_models_sessions(sessions, nl_configs=NL_CONFIGS, verbose=False):
    nSessions = len(sessions)
    nl_names = [c[0] for c in nl_configs]
    nNL      = len(NL_CONFIGS)

    theta_arr  = {name: [] for name in nl_names}   # (nstim,) per neuron per model
    nlpar_arr  = {name: [] for name in nl_names}   # shape params per neuron per model
    ses_idx_arr = []                               # session index for each neuron

    for ises in range(nSessions):
        ses      = sessions[ises]
        ustim_s  = np.unique(ses.trialdata['Orientation'])
        stim_ids = np.searchsorted(ustim_s, ses.trialdata['Orientation'].to_numpy())
        N        = ses.respmat.shape[0]
        nstim    = len(ustim_s)

        if not hasattr(ses,'popratemat'):
            ses = comp_poprate(ses,version='radius_500')
            
        for name in nl_names:
            ses.celldata['R2'    + name] = np.nan
            ses.celldata['Gamma' + name] = np.nan
            ses.celldata['Beta'  + name] = np.nan

        for iN in tqdm(range(N), desc=f'Session {ises+1}/{nSessions}'):
            poprate = ses.popratemat[iN,:]
            resp = ses.respmat[iN, :]
            res  = fit_nl_models(resp, stim_ids, poprate, configs=nl_configs)
            for name in nl_names:
                ses.celldata.loc[iN, 'R2'    + name] = res[name]['r2']
                ses.celldata.loc[iN, 'Gamma' + name] = res[name]['gamma']
                ses.celldata.loc[iN, 'Beta'  + name] = res[name]['b']
                theta_arr[name].append(
                    res[name]['theta'] if res[name]['theta'] is not None
                    else np.full(nstim, np.nan))
                nlpar_arr[name].append(res[name]['nl_par'] or [])
            ses_idx_arr.append(ises)

    return sessions, theta_arr, nlpar_arr, ses_idx_arr
