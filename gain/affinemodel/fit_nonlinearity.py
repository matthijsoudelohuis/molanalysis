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
from scipy.stats import linregress,binned_statistic
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator
from sklearn.decomposition import PCA

os.chdir('c:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive
from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import *

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\TransferFunctions')

#%% Define nonlinearities:

def lin(x):
    return x

def relu(x):
    return np.maximum(0, x)

def softplus(x, beta=1.0):
    return np.log1p(np.exp(beta * x)) / beta

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def exp(x):
    return np.maximum(0, np.exp(x) - 1)  # Shifted to be zero at x=0

def tanh(x):
    return np.tanh((x))+1  # Shifted to be zero at x=0

def powerlaw(x, p=2):
    return np.maximum(0, x) ** p


#%% Show transfer functions for different nonlinearities:
nonlinearities = [lin, relu, lambda x: softplus(x, beta=2), 
                sigmoid, tanh, lambda x: powerlaw(x, p=2), exp]
nonlinearity_names = ['Linear', 'ReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Power-law (p=2)', 'Exp']
nnonlinearities = len(nonlinearities)

operating_range = np.array([[0,1],
                            [-0.5,1],
                            [-3,3],
                            [-5,5],
                            [-2.5,2.5],
                            [-.5,3],
                            [-.5,2]])

fig, axes = plt.subplots(3,3,figsize=(6, 6))
axes = axes.flatten()
x = np.linspace(-10, 10, 100)
x = np.linspace(-5, 5, 100)
# x = np.linspace(-1, 1, 100)

for i, nonlinearity in enumerate(nonlinearities):
    ax = axes[i]
    y = nonlinearity(x)
    ax.plot(x, y)
    ax.set_title(nonlinearity_names[i])
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.grid()
plt.tight_layout()
sns.despine()
# my_savefig(plt.gcf(),savedir,f'{nonlinearity_names[i]}_Nonlinearity_TransferFunction')
# my_savefig(fig,savedir,f'Tranfer_functions_overview')



#%% 

session_list        = np.array([['LPE11086_2024_01_05']])
session_list        = np.array([['LPE12223_2024_06_10']])
session_list        = np.array([['LPE12223_2024_06_10','LPE11086_2024_01_05','LPE10919_2023_11_06']])

sessions,nSessions  = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_noiselevel=True)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)
sessions = ori_remapping(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ###########################################################################
# NONLINEAR TRANSFER FUNCTION FITTING PIPELINE
# Model: r(t) = f( θ_k(t) + γ · P(t) + b )
#   θ_k  : stimulus drive — one free parameter per orientation (16)
#   γ    : population-rate scaling (additive input, 1 param)
#   b    : input bias (1 param)
#   f(·) : nonlinearity (with model-specific free parameters)
###########################################################################

#%% Redefine nonlinearities with fittable parameters

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
    return np.abs(a) / (1.0 + np.exp(5*-np.clip(u, -500.0, 500.0)))

def nl_tanh(u, a):
    # maps tanh's [-1,1] to [0, a]: f(u) = a · (1 + tanh(u)) / 2
    return np.abs(a) * 0.5 * (1.0 + np.tanh(u))

def nl_powerlaw(u, p):
    # f(u) = max(0,u)^p; p is the free exponent
    return np.power(np.maximum(0.0, u), np.abs(p) + 1e-4)

def nl_exp(u):
    # max(0, exp(u)-1), shifted so f(0)=0; output gain a is universal
    return np.maximum(0.0, np.expm1(np.clip(u, -500.0, 10.0)))

# def softplus(x, beta=1.0):
#     return np.log1p(np.exp(beta * x)) / beta

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def exp(x):
#     return np.maximum(0, np.exp(x) - 1)  # Shifted to be zero at x=0

# def tanh(x):
#     return np.tanh((x))+1  # Shifted to be zero at x=0

# def powerlaw(x, p=2):
#     return np.maximum(0, x) ** p

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
clrs_nl  = sns.color_palette('tab10', nNL)

#%% Show nonlinearities at p0 initialization
x = np.linspace(-0.5, 1.25, 300)
fig, axes = plt.subplots(1, nNL, figsize=(nNL * 2.2, 3), sharey=True)
for i, (name, nl_func, n_shape, p0_shape, _) in enumerate(NL_CONFIGS):
    ax = axes[i]
    y  = nl_func(x, *p0_shape) if n_shape else nl_func(x)
    ax.plot(x, y, color=clrs_nl[i], lw=2)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('u  (θ_k + γ·P + b)')
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    if i == 0:
        ax.set_ylabel('f(u)')
    if p0_shape:
        ax.text(0.05, 0.95, ', '.join([f'{v}' for v in p0_shape]),
                transform=ax.transAxes, fontsize=7, va='top', color='gray')

sns.despine(trim=True, offset=3)

plt.suptitle('Nonlinearities at p0 initialization', fontsize=10, y=1.02)
plt.tight_layout()
# my_savefig(fig, savedir, 'NL_p0_shapes', formats=['png'])

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

#%% Pick example neuron: well-tuned with moderate–high pop coupling
ises     = 0
ses      = sessions[ises]
poprate  = np.nanmean(zscore(ses.respmat, axis=1), axis=0)   # (nTrials,)
# poprate  = np.nanmean(ses.respmat, axis=0)   # (nTrials,)
ustim    = np.unique(ses.trialdata['Orientation'])
stim_ids = np.searchsorted(ustim, ses.trialdata['Orientation'].to_numpy())
nstim    = len(ustim)

idx_good = np.where(
    # (ses.celldata['gOSI']           > 0.5) &
    (ses.celldata['gOSI']           <0.2) &
    (ses.celldata['pop_coupling']  > np.percentile(ses.celldata['pop_coupling'], 70))
    # (ses.celldata['pop_coupling']  > np.percentile(ses.celldata['pop_coupling'], 50)) &
    # (ses.celldata['noise_level']   < 20)
    )[0]
# np.random.seed(42)
ex_iN    = np.random.choice(idx_good)
# ex_iN = 0
resp_ex  = ses.respmat[ex_iN, :]
# Normalise responses to [0, 1]
r_min     = resp_ex.min()
r_max     = resp_ex.max()
r_max     = np.percentile(resp_ex, 99)
resp_ex = (resp_ex - r_min) / max(r_max - r_min, 1e-8)
# resp_ex = zscore(resp_ex)

ex_cid   = ses.celldata['cell_id'].iloc[ex_iN]
print(f'Example: {ex_cid}  OSI={ses.celldata["gOSI"].iloc[ex_iN]:.2f}  '
      f'pop_coupling={ses.celldata["pop_coupling"].iloc[ex_iN]:.2f}')

results_ex = fit_nl_models(resp_ex, stim_ids, poprate, NL_CONFIGS)

#%% Diagnostic figure for the example neuron
pref_k    = int(np.argmax(results_ex[nl_names[0]]['theta']))
orth_k    = (pref_k + 4) % nstim
pop_sweep = np.linspace(np.percentile(poprate, 1), np.percentile(poprate, 99), 200)
# u_range   = np.linspace(-1.5, 2.0, 300)
# u_range   = np.linspace(0, 1.0, 300)
best_name = max(nl_names, key=lambda n: results_ex[n]['r2']
                if not np.isnan(results_ex[n]['r2']) else -1)
best_res     = results_ex[best_name]
resp_norm_ex = best_res['resp_norm']
residuals    = resp_norm_ex - best_res['pred']

fig, axes = plt.subplots(3, 3, figsize=(14, 12))

# (0,0) Fitted nonlinearity shapes over the actual input range seen by each model
ax = axes[0, 0]
for i, (name, nl_func, n_shape, _, _) in enumerate(NL_CONFIGS):
    entry = results_ex[name]
    if entry['theta'] is None:
        continue
    u_vals  = entry['u']
    u_sweep = np.linspace(np.percentile(u_vals, 1), np.percentile(u_vals, 99), 300)
    y = nl_func(u_sweep, *entry['nl_par']) if n_shape else nl_func(u_sweep)
    ax.plot(u_sweep, y, color=clrs_nl[i], lw=2, label=name)
ax.axhline(0, color='k', lw=0.5, ls=':')
ax.axvline(0, color='k', lw=0.5, ls=':')
ax.set_xlabel('u  (θ_k + γ·P + b)')
ax.set_ylabel('f(u)  [normalised scale]')
ax.set_title('Fitted nonlinearities\n(over actual input range)')
ax.legend(fontsize=7, frameon=False)
sns.despine(ax=ax, trim=True, offset=3)

# (0,1) Fitted θ — tuning curve in input space
ax = axes[0, 1]
for i, (name, *_) in enumerate(NL_CONFIGS):
    if results_ex[name]['theta'] is None:
        continue
    ax.plot(ustim, results_ex[name]['theta'], color=clrs_nl[i], lw=1.5,
            marker='o', ms=3, label=name)
ax.set_xlabel('Orientation (°)')
ax.set_ylabel('θ_k  (input-space drive)')
ax.set_title('Fitted stimulus drive (pre-NL)')
ax.legend(fontsize=7, frameon=False)
ax.set_xticks(ustim[::2])
ax.tick_params(axis='x', labelrotation=45)
sns.despine(ax=ax, trim=True, offset=3)

# (0,2) Mean output tuning curve: observed vs all model predictions
ax = axes[0, 2]
mean_obs = np.array([np.mean(resp_norm_ex[stim_ids == k]) for k in range(nstim)])
ax.plot(ustim, mean_obs, color='k', lw=2, marker='o', ms=4, label='observed', zorder=5)
for i, (name, nl_func, n_shape, _, _) in enumerate(NL_CONFIGS):
    entry = results_ex[name]
    if entry['pred'] is None:
        continue
    mean_pred = np.array([np.mean(entry['pred'][stim_ids == k]) for k in range(nstim)])
    ax.plot(ustim, mean_pred, color=clrs_nl[i], lw=1.5, ls='--', label=name)
ax.set_xlabel('Orientation (°)')
ax.set_ylabel('Mean response (normalised)')
ax.set_title('Mean tuning curve\n(observed vs fitted)')
ax.legend(fontsize=7, frameon=False)
ax.set_xticks(ustim[::2])
ax.tick_params(axis='x', labelrotation=45)
sns.despine(ax=ax, trim=True, offset=3)

# (1,0) Response vs pop rate for preferred and orthogonal orientations
ax = axes[1, 0]
for k_ori, lbl, col in [(pref_k, 'pref', 'tab:blue'), (orth_k, 'orth', 'tab:orange')]:
    idx_T = stim_ids == k_ori
    ax.scatter(poprate[idx_T], resp_norm_ex[idx_T], s=4, alpha=0.35, color=col,
               zorder=1, label=f'data ({lbl})')
    for i, (name, nl_func, n_shape, _, _) in enumerate(NL_CONFIGS):
        entry = results_ex[name]
        if entry['theta'] is None:
            continue
        u_line    = entry['theta'][k_ori] + entry['gamma'] * pop_sweep + entry['b']
        pred_line = nl_func(u_line, *entry['nl_par']) if n_shape else nl_func(u_line)
        ax.plot(pop_sweep, pred_line, color=clrs_nl[i], lw=1.2, alpha=0.8)
ax.set_xlabel('Population rate (z)')
ax.set_ylabel('Response (normalised)')
ax.set_title('Resp vs pop rate\n(pref & orth, all models)')
ax.set_xlim([pop_sweep[0], pop_sweep[-1]])
ax.legend(fontsize=7, frameon=False)
sns.despine(ax=ax, trim=True, offset=3)

# (1,1) R² bar plot
ax = axes[1, 1]
r2s = [results_ex[n]['r2'] for n in nl_names]
ax.bar(np.arange(nNL), r2s, color=clrs_nl)
ax.set_xticks(np.arange(nNL))
ax.set_ylabel('R²')
ax.set_title(f'R² per model — {ex_cid}')
ax.set_ylim([0, max(r for r in r2s if not np.isnan(r)) * 1.25])
for i, v in enumerate(r2s):
    if not np.isnan(v):
        ax.text(i, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
sns.despine(ax=ax, trim=True, offset=3)
ax.set_xticklabels(nl_names, rotation=45, ha='right', fontsize=8)

# (1,2) Predicted vs observed (best model, normalised space)
ax = axes[1, 2]
ax.scatter(resp_norm_ex, best_res['pred'], s=2, alpha=0.3, color='k')
lims = [min(resp_norm_ex.min(), best_res['pred'].min()),
        max(resp_norm_ex.max(), best_res['pred'].max())]
ax.plot(lims, lims, 'r--', lw=1)
ax.set_xlabel('Observed (normalised)')
ax.set_ylabel('Predicted')
ax.set_title(f'Predicted vs observed\n({best_name}, R²={best_res["r2"]:.3f})')
sns.despine(ax=ax, trim=True, offset=3)

# (2,0) Distribution of fitted inputs u across models
ax = axes[2, 0]
for i, (name, *_) in enumerate(NL_CONFIGS):
    u = results_ex[name]['u']
    if u is not None:
        sns.kdeplot(u, ax=ax, color=clrs_nl[i], label=name, fill=False)
ax.axvline(0, color='k', lw=0.5, ls=':')
ax.set_xlabel('Input  u = θ_k + γ·P + b')
ax.set_ylabel('Density')
ax.set_title('Distribution of fitted inputs')
ax.legend(fontsize=7, frameon=False)
sns.despine(ax=ax, trim=True, offset=3)

# (2,1) Residuals vs pop rate (best model)
ax = axes[2, 1]
ax.scatter(poprate, residuals, s=2, alpha=0.3, color='k')
ax.axhline(0, color='r', lw=1)
_, _, rv, pv, _ = linregress(poprate, residuals)
ax.text(0.05, 0.93, f'r={rv:.2f}, p={pv:.2e}', transform=ax.transAxes, fontsize=8)
ax.set_xlabel('Population rate (z)')
ax.set_ylabel('Residual')
ax.set_title(f'Residuals vs pop rate  ({best_name})')
sns.despine(ax=ax, trim=True, offset=3)

# (2,2) Mean residuals per orientation (best model)
ax = axes[2, 2]
mean_resid = [np.mean(residuals[stim_ids == k]) for k in range(nstim)]
ax.bar(ustim, mean_resid, width=18, color='steelblue', alpha=0.8)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Orientation (°)')
ax.set_ylabel('Mean residual')
ax.set_title(f'Residuals by orientation  ({best_name})')
ax.set_xticks(ustim[::2])
ax.tick_params(axis='x', labelrotation=45)
sns.despine(ax=ax, trim=True, offset=3)

plt.suptitle(f'NL model fits — {ex_cid}', fontsize=12, y=1.01)
plt.tight_layout()
# my_savefig(fig, savedir, f'NLfit_diagnostics_{ex_cid}', formats=['png'])

#%% Fit all neurons across all sessions and collect R²
for ises in range(nSessions):
    ses      = sessions[ises]
    poprate  = np.nanmean(zscore(ses.respmat, axis=1), axis=0)
    ustim_s  = np.unique(ses.trialdata['Orientation'])
    stim_ids = np.searchsorted(ustim_s, ses.trialdata['Orientation'].to_numpy())
    N        = ses.respmat.shape[0]

    nstim    = len(ustim_s)

    for name in nl_names:
        ses.celldata['R2' + name] = np.nan
        ses.celldata['Gamma' + name] = np.nan
        ses.celldata['Beta' + name] = np.nan

    for iN in tqdm(range(N), desc=f'Session {ises+1}/{nSessions}'):
        # Normalise responses to [0, 1]
        resp = ses.respmat[iN, :]
        respmin = resp.min()
        respmax = resp.max()
        resp = (resp - r_min) / max(r_max - r_min, 1e-8)
        # resp = np.clip(resp, 0, np.percentile(resp,99))
        res = fit_nl_models(resp, stim_ids, poprate, configs=NL_CONFIGS)
        res['Sigmoid']['gamma']
        for name in nl_names:
            ses.celldata.loc[iN, 'R2' + name] = res[name]['r2']
            ses.celldata.loc[iN, 'Gamma' + name] = res[name]['gamma']
            ses.celldata.loc[iN, 'Beta' + name] = res[name]['b']
            # r2_all[name].append(res[name]['r2'])
            # gamma_all[name].append(res[name]['gamma'])
            # beta_all[name].append(res[name]['b'])

#%% Plot R² distributions across models and neurons
celldata = pd.concat([ses.celldata for ses in sessions])
bw_adjust = 0.25
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
idx_N = np.all((celldata['noise_level']<20,
                # celldata['roi_name']=='V1',
                # celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],50),
                ),axis=0)
ax = axes[0]
for i, name in enumerate(nl_names):
    vals = celldata['R2' + name][idx_N].dropna().values
    vals = vals[vals > -1]
    sns.kdeplot(vals, ax=ax, color=clrs_nl[i], label=name, 
                bw_adjust=bw_adjust, clip=[0,1],fill=False, lw=2)
ax.set_xlabel('R²')
ax.set_ylabel('Density')
ax.set_title('R² distribution across neurons')
ax.legend(fontsize=8, frameon=False)
sns.despine(ax=ax, trim=True, offset=3)

ax = axes[1]
# r2_df = pd.DataFrame(r2_all)
r2_df = celldata[['R2' + name for name in nl_names]].dropna()
r2_long = (r2_df.clip(lower=-1)
              .melt(var_name='Model', value_name='R²')
              .dropna())
r2_long['Model'] = r2_long['Model'].str.replace('R2', '')
sns.violinplot(data=r2_long, x='Model', y='R²', hue='Model',palette=clrs_nl, ax=ax,
                bw_adjust=bw_adjust,inner='quartile', cut=0, order=nl_names)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_title('R² distribution (violin)')
ax.axhline(0, color='k', lw=0.5, ls=':')
sns.despine(ax=ax, trim=True, offset=3)

plt.tight_layout()
# my_savefig(fig, savedir, f'NLfit_R2_distributions_{nSessions}sessions', formats=['png'])

#%% Scatter: pop_coupling vs fitted gamma across models
# from utils.corr_lib import filter_sharednan
idx_N = np.all((celldata['noise_level']<20,
                # celldata['roi_name']=='V1',
                # celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],50),
                ),axis=0)
# pop_coupling_all = celldata['pop_coupling'].values[idx_N]
ncols = nNL
fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3), sharex=True, sharey=False)

for i, name in enumerate(nl_names):
    ax     = axes[i]
    y  = celldata['Gamma' + name]
    x  = celldata['pop_coupling']
    # pc     = pop_coupling_all
    x,y = filter_sharednan(x,y)
    # mask = np.isfinite(gamma) & np.isfinite(pc)
    # x, y = pc[mask], gamma[mask]

    ax.scatter(x, y, s=3, alpha=0.3, color=clrs_nl[i], rasterized=True)

    slope, intercept, r, p, _ = linregress(x, y)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color='k', lw=1.5, ls='--')

    ax.set_xlabel('Pop. coupling', fontsize=9)
    if i == 0:
        ax.set_ylabel('Fitted γ', fontsize=9)
    ax.set_title(name, fontsize=9)
    ax.set_xlim([-0.2,0.6])
    ax.set_ylim(np.percentile(y, [2, 98]))
    ax.text(0.05, 0.93, f'r={r:.2f}, p={p:.1e}',
            transform=ax.transAxes, fontsize=7, va='top')
    sns.despine(ax=ax, trim=True, offset=3)

plt.suptitle('Population coupling vs fitted γ  (pop-rate scaling)', fontsize=10, y=1.02)
plt.tight_layout()
# my_savefig(fig, savedir, f'PopCoupling_vs_gamma_{nSessions}sessions', formats=['png'])

#%% Scatter of linear vs. best model R2
fig, axes = plt.subplots(1, len(nl_names)-1, figsize=((len(nl_names)-1) * 3, 3),
                         sharex=True, sharey=True)
for i, name in enumerate(nl_names[1:]):
    ax = axes[i]
    xdata = celldata['R2Linear']
    ydata = celldata['R2' + name]
    ax.scatter(xdata, ydata, s=5, alpha=0.5, color=clrs_nl[i], rasterized=True)
    add_paired_ttest_results(ax, xdata, ydata, pos=[0.2,0.9])
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.plot([0,1], [0,1], color='k', lw=0.5, ls=':')
sns.despine(trim=True, offset=3)

plt.tight_layout()
# my_savefig(fig, savedir, f'NLfit_R2_scatter_{nSessions}sessions', formats=['png'])

