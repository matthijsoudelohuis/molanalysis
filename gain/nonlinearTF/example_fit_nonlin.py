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

os.chdir('e:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive
from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import *
from utils.nonlin_lib import *

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\TransferFunctions')


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


#%% Pick example neuron: well-tuned with moderate–high pop coupling
ises     = 0
ses      = sessions[ises]
poprate  = np.nanmean(zscore(ses.respmat, axis=1), axis=0)   # (nTrials,)
# poprate  = np.nanmean(ses.respmat, axis=0)   # (nTrials,)
ustim    = np.unique(ses.trialdata['Orientation'])
stim_ids = np.searchsorted(ustim, ses.trialdata['Orientation'].to_numpy())
nstim    = len(ustim)

idx_good = np.where(
    (ses.celldata['gOSI']           > 0.5) &
    # (ses.celldata['gOSI']           <0.2) &
    (ses.celldata['pop_coupling']  > np.percentile(ses.celldata['pop_coupling'], 70))
    # (ses.celldata['pop_coupling']  > np.percentile(ses.celldata['pop_coupling'], 50)) &
    # (ses.celldata['noise_level']   < 20)
    )[0]

# sigmoidaldiff = ses.celldata['R2Sigmoid'] - ses.celldata['R2Power-law (p=2)']
# idx_good = np.where(sigmoidaldiff > np.nanpercentile(sigmoidaldiff, 80))[0]
# np.random.seed(42)
ex_iN    = np.random.choice(idx_good)
# ex_iN = 0
resp_ex  = ses.respmat[ex_iN, :]
# Normalise responses to [0, 1]
r_min     = resp_ex.min()
r_max     = resp_ex.max()
# r_max     = np.percentile(resp_ex, 99)
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
    u_norm = np.linspace(0,1,300)
    ax.plot(u_norm, y, color=clrs_nl[i], lw=2, label=name)
    # ax.plot(u_sweep, y, color=clrs_nl[i], lw=2, label=name)
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
    # ax.plot(ustim, results_ex[name]['theta'], color=clrs_nl[i], lw=1.5,
    ax.plot(ustim, zscore(results_ex[name]['theta']), color=clrs_nl[i], lw=1.5,
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


#%%

[sessions, theta_arr, nlpar_arr, ses_idx_arr] = fit_nl_models_sessions(sessions, nl_configs=NL_CONFIGS)
#  fit_nl_models(sessions, nl_configs=NL_CONFIGS, verbose=False):


#%% Plot R² distributions across models and neurons
celldata = pd.concat([ses.celldata for ses in sessions])
bw_adjust = 0.25
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
idx_N = np.all((celldata['noise_level']<20,
                # celldata['roi_name']=='V1',
                celldata['gOSI']>0.5,
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
                # celldata['gOSI']>0.5,
                # celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],50),
                ),axis=0)
ncols = nNL
fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3), sharex=True, sharey=False)

for i, name in enumerate(nl_names):
    ax     = axes[i]
    y  = celldata['Gamma' + name][idx_N]
    x  = celldata['pop_coupling'][idx_N]
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
fig, axes = plt.subplots(1, len(nl_names)-1, figsize=((len(nl_names)-1) * 2, 2),
                         sharex=True, sharey=True)
for i, name in enumerate(nl_names[1:]):
    ax = axes[i]
    xdata = celldata['R2Linear'][idx_N]
    ydata = celldata['R2' + name][idx_N]
    ax.scatter(xdata, ydata, s=5, alpha=0.5, color=clrs_nl[i+1], rasterized=True)
    add_paired_ttest_results(ax, xdata, ydata, pos=[0.2,0.9])
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Linear R²', fontsize=9)
    ax.set_ylabel(f'{name} R²', fontsize=9)
    ax.plot([0,1], [0,1], color='k', lw=0.5, ls=':')
sns.despine(trim=True, offset=3)

plt.tight_layout()
# my_savefig(fig, savedir, f'NLfit_R2_scatter_{nSessions}sessions', formats=['png'])

#%% =====================================================================
# FISHER INFORMATION ANALYSIS
# Does the TF shape (sigmoid saturation vs power law) predict how the
# information boost from population rate modulation varies across
# orientations (preferred vs adjacent)?
# =====================================================================

#%% Analytical TF derivative functions  (match NL_CONFIGS exactly)
def tfd_linear(u):
    return np.ones_like(u)

def tfd_relu(u):
    return (u > 0).astype(float)

def tfd_softplus(u, beta):
    b = np.abs(beta) + 1e-4
    return 1.0 / (1.0 + np.exp(-np.clip(b * u, -500.0, 500.0)))

def tfd_sigmoid(u, a):
    s = 1.0 / (1.0 + np.exp(-np.clip(u, -500.0, 500.0)))
    return np.abs(a) * s * (1.0 - s)

def tfd_tanh(u, a):
    return np.abs(a) * 0.5 * (1.0 - np.tanh(u) ** 2)

def tfd_powerlaw(u, p):
    exp = np.abs(p) + 1e-4
    return exp * np.power(np.maximum(u, 1e-8), exp - 1.0) * (u > 0).astype(float)

def tfd_exp(u):
    return np.maximum(0.0, np.exp(np.clip(u, -500.0, 10.0)))

TF_DERIVS = {
    'Linear':          tfd_linear,
    'ReLU':            tfd_relu,
    'Softplus':        tfd_softplus,
    'Sigmoid':         tfd_sigmoid,
    'Tanh':            tfd_tanh,
    'Power-law (p=2)': tfd_powerlaw,
    'Exp':             tfd_exp,
}

#%% Compute empirical mean_resp_all and modulation_all (rolled so idx 0 = preferred)

perc_split     = 25
mean_resp_arr  = []
modulation_arr = []

for ises in range(nSessions):
    ses     = sessions[ises]
    N       = ses.respmat.shape[0]
    stims   = ses.trialdata['Orientation'].to_numpy()
    ustim_s = np.unique(stims)
    ns      = len(ustim_s)
    pop_s   = np.nanmean(zscore(ses.respmat, axis=1), axis=0)
    thr_lo  = np.percentile(pop_s, perc_split)
    thr_hi  = np.percentile(pop_s, 100 - perc_split)

    mr = np.full((N, ns), np.nan)
    md = np.full((N, ns), np.nan)
    for istim, stim in enumerate(ustim_s):
        i_all  = stims == stim
        i_lo   = i_all & (pop_s <= thr_lo)
        i_hi   = i_all & (pop_s >= thr_hi)
        mr[:, istim] = np.nanmean(ses.respmat[:, i_all], axis=1)
        md[:, istim] = (np.nanmean(ses.respmat[:, i_hi], axis=1) -
                        np.nanmean(ses.respmat[:, i_lo], axis=1))

    pref_idx = np.argmax(mr, axis=1)
    for n in range(N):
        mr[n, :] = np.roll(mr[n, :], -pref_idx[n])
        md[n, :] = np.roll(md[n, :], -pref_idx[n])

    mean_resp_arr.append(mr)
    modulation_arr.append(md)

mean_resp_all  = np.concatenate(mean_resp_arr,  axis=0)   # (N_total, nstim)
modulation_all = np.concatenate(modulation_arr, axis=0)   # (N_total, nstim)
N_total        = len(mean_resp_all)
nstim_fi       = mean_resp_all.shape[1]

#%% Identify best-fitting TF per neuron and compute MAI

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

r2_cols  = ['R2' + n for n in nl_names]
best_TF  = celldata[r2_cols].idxmax(axis=1).str.replace('R2', '', regex=False)

# Modulation Asymmetry Index:
#   > 0 → preferred modulated more (power-law signature)
#   < 0 → adjacent orientations modulated more (sigmoid-saturation signature)
dr_pref = modulation_all[:, 0]
dr_adj  = np.nanmean(np.abs(modulation_all[:, 1:3]), axis=1)  # mean of ±22.5° & ±45°
MAI     = (dr_pref - dr_adj) / (np.abs(dr_pref) + dr_adj + 1e-8)
r_pref  = mean_resp_all[:, 0]    # mean response at preferred orientation

#%% Compute model-predicted ΔI_F per neuron at each orientation

# For each neuron use its best-fitting TF and parameters.
# ΔI_F(k) = [ f'(u_high(k))² − f'(u_low(k))² ] · s'(k)² / max(r̄(k), ε)
# where u(k) = theta[k] + gamma * P + b,  P_high / P_low are session quartiles.

dIF_all = np.full((N_total, nstim_fi), np.nan)

neuron_offset = 0
for ises in range(nSessions):
    ses   = sessions[ises]
    N     = ses.respmat.shape[0]
    pop_s = np.nanmean(zscore(ses.respmat, axis=1), axis=0)
    P_lo  = np.percentile(pop_s, perc_split)
    P_hi  = np.percentile(pop_s, 100 - perc_split)

    for iN in range(N):
        gi = neuron_offset + iN          # global neuron index
        tf_name = best_TF.iloc[gi] if gi < len(best_TF) else None
        if tf_name not in TF_DERIVS:
            continue
        tf_deriv = TF_DERIVS[tf_name]

        theta  = np.array(theta_arr[tf_name][gi])
        gamma  = celldata['Gamma' + tf_name].iloc[gi]
        b      = celldata['Beta'  + tf_name].iloc[gi]
        nl_par = nlpar_arr[tf_name][gi]

        if np.any(np.isnan(theta)) or np.isnan(gamma) or np.isnan(b):
            continue

        # Roll theta same way as mean_resp (by empirical pref index)
        pref_i = int(np.argmax(theta))
        theta_r = np.roll(theta, -pref_i)

        u_hi = theta_r + gamma * P_hi + b
        u_lo = theta_r + gamma * P_lo + b

        df_hi = tf_deriv(u_hi, *nl_par) if nl_par else tf_deriv(u_hi)
        df_lo = tf_deriv(u_lo, *nl_par) if nl_par else tf_deriv(u_lo)

        # Orientation slope: numerical gradient of the rolled theta (input tuning)
        s_prime = np.gradient(theta_r)

        r_mean  = np.maximum(mean_resp_all[gi, :], 1e-6)
        dIF_all[gi, :] = (df_hi**2 - df_lo**2) * s_prime**2 / r_mean

    neuron_offset += N

#%% --- Plot 1: MAI vs r̄_pref, colored by best-fitting TF ---

quality_mask = np.all((
    celldata['noise_level'] < 20,
    celldata['gOSI']        > 0.3,
    np.isfinite(MAI),
    np.isfinite(r_pref),
    r_pref > 0,
), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: scatter colored by best TF
ax = axes[0]
for i, name in enumerate(nl_names):
    idx = quality_mask & (best_TF == name)
    ax.scatter(r_pref[idx], MAI[idx], s=5, alpha=0.4,
               color=clrs_nl[i], label=name, rasterized=True)

ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xlabel('Mean response at preferred orientation  (r̄_pref)', fontsize=10)
ax.set_ylabel('Modulation asymmetry index  (MAI)', fontsize=10)
ax.set_title('MAI vs operating point\n> 0: pref boosted most   < 0: adjacent boosted most')
ax.legend(fontsize=7, frameon=False, markerscale=3)
ax.set_ylim([-1, 1])
sns.despine(ax=ax, trim=True, offset=3)

# Right: mean MAI ± SEM per TF, split by strongly vs weakly driven neurons
ax = axes[1]
r_pref_median = np.nanmedian(r_pref[quality_mask])

for i, name in enumerate(nl_names):
    for driven, ls, label_sfx in [(True, '-', ' high r̄'), (False, '--', ' low r̄')]:
        idx = quality_mask & (best_TF == name) & (
              (r_pref > r_pref_median) if driven else (r_pref <= r_pref_median))
        vals = MAI[idx]
        if len(vals) < 5:
            continue
        ax.errorbar(i + (0.2 if driven else -0.2),
                    np.nanmean(vals),
                    np.nanstd(vals) / np.sqrt(np.sum(np.isfinite(vals))),
                    fmt='o', color=clrs_nl[i], ls=ls, ms=6,
                    capsize=3, label=name + label_sfx)

ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xticks(np.arange(nNL))
ax.set_xticklabels(nl_names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean MAI ± SEM', fontsize=10)
ax.set_title('MAI by TF type and drive level\n(solid = high r̄_pref,  dashed = low r̄_pref)')
sns.despine(ax=ax, trim=True, offset=3)

plt.tight_layout()
my_savefig(fig, savedir, f'MAI_vs_rPref_{nSessions}sessions', formats=['png'])

#%% --- Plot 2: ΔI_F profile across orientation distance, grouped by TF and drive level ---

# Circular orientation distances 0..nstim//2  (0 = preferred, 4 = orthogonal for 16-stim)
ori_dists   = np.minimum(np.arange(nstim_fi), nstim_fi - np.arange(nstim_fi))
unique_dist = np.unique(ori_dists)   # [0,1,2,3,4,5,6,7,8] for 16 stims

# Group: high vs low r̄_pref, two TF families (sigmoid-type vs power-law-type)
sigmoid_TFs  = {'Sigmoid', 'Tanh'}
powerlaw_TFs = {'Power-law (p=2)', 'Softplus', 'ReLU', 'Exp'}

groups = [
    ('Sigmoid-type,  high r̄',  sigmoid_TFs,  True,  'tab:red',    '-'),
    ('Sigmoid-type,  low r̄',   sigmoid_TFs,  False, 'tab:red',    '--'),
    ('Power-law-type, high r̄', powerlaw_TFs, True,  'tab:blue',   '-'),
    ('Power-law-type, low r̄',  powerlaw_TFs, False, 'tab:blue',   '--'),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: ΔI_F profile averaged over orientation distance
ax = axes[0]
for label, tf_set, driven, col, ls in groups:
    idx = quality_mask & (best_TF.isin(tf_set)) & (
          (r_pref > r_pref_median) if driven else (r_pref <= r_pref_median))
    idx = idx & np.all(np.isfinite(dIF_all), axis=1)
    if idx.sum() < 5:
        continue

    # Average ΔI_F within each circular distance bin
    dIF_by_dist = np.array([
        np.nanmean(dIF_all[np.ix_(idx, ori_dists == d)]) for d in unique_dist
    ])
    sem_by_dist = np.array([
        np.nanstd(dIF_all[np.ix_(idx, ori_dists == d)]) /
        np.sqrt(np.sum(idx) * np.sum(ori_dists == d))
        for d in unique_dist
    ])
    ax.plot(unique_dist * 22.5, dIF_by_dist, color=col, ls=ls, lw=2, label=label)
    ax.fill_between(unique_dist * 22.5,
                    dIF_by_dist - sem_by_dist,
                    dIF_by_dist + sem_by_dist,
                    color=col, alpha=0.15)

ax.axhline(0, color='k', lw=0.5, ls=':')
ax.set_xlabel('Orientation distance from preferred (°)', fontsize=10)
ax.set_ylabel('ΔI_F  (high − low pop rate)', fontsize=10)
ax.set_title('Fisher information boost\nacross orientation distance')
ax.legend(fontsize=7, frameon=False)
ax.set_xticks(unique_dist * 22.5)
sns.despine(ax=ax, trim=True, offset=3)

# Right: ΔI_F at preferred vs adjacent as scatter — direct comparison
ax = axes[1]
dIF_pref = dIF_all[:, 0]
dIF_adj  = np.nanmean(dIF_all[:, 1:3], axis=1)

for i, name in enumerate(nl_names):
    idx = quality_mask & (best_TF == name) & np.isfinite(dIF_pref) & np.isfinite(dIF_adj)
    ax.scatter(dIF_pref[idx], dIF_adj[idx], s=5, alpha=0.35,
               color=clrs_nl[i], label=name, rasterized=True)

lim_max = np.nanpercentile(np.abs(np.concatenate([dIF_pref, dIF_adj])), 98)
ax.plot([-lim_max, lim_max], [-lim_max, lim_max], 'k--', lw=1)
ax.axhline(0, color='k', lw=0.5, ls=':')
ax.axvline(0, color='k', lw=0.5, ls=':')
ax.set_xlim([-lim_max, lim_max])
ax.set_ylim([-lim_max, lim_max])
ax.set_xlabel('ΔI_F at preferred orientation', fontsize=10)
ax.set_ylabel('ΔI_F at adjacent orientation (±22.5–45°)', fontsize=10)
ax.set_title('Information boost: preferred vs adjacent\n'
             'Above diagonal: adjacent benefits more (sigmoid saturation)')
ax.legend(fontsize=7, frameon=False, markerscale=3)
sns.despine(ax=ax, trim=True, offset=3)

plt.tight_layout()
# my_savefig(fig, savedir, f'dIF_profile_{nSessions}sessions', formats=['png'])

#%%
