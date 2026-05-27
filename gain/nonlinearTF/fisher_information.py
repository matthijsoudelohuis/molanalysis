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
