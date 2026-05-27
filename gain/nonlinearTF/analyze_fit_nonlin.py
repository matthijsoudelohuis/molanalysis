#%% 
from copyreg import pickle
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
from utils.corr_lib import filter_sharednan

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\TransferFunctions')

resultdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Analysis\\SharedGain')


#%% 

# session_list        = np.array([['LPE11086_2024_01_05']])
# session_list        = np.array([['LPE12223_2024_06_10']])
# session_list        = np.array([['LPE12223_2024_06_10','LPE11086_2024_01_05','LPE10919_2023_11_06']])

# sessions,nSessions  = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_noiselevel=True)
# sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# #%%  Load data properly:                      
# for ises in range(nSessions):
#     sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
#                                 calciumversion='deconv',keepraw=False)

# #%% Add how neurons are coupled to the population rate: 
# sessions = compute_pop_coupling(sessions)
# sessions = ori_remapping(sessions)
# sessions = compute_tuning_wrapper(sessions)
# sessions = compute_pairwise_anatomical_distance(sessions)


#%% Load the data:
filename = 'NonLin_Fit_allGR_sessions_2026-05-26_19-56-11'

data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    print(key)  
    exec(key+'=data[key]')


#%%

# [sessions, theta_arr, nlpar_arr, ses_idx_arr] = fit_nl_models_sessions(sessions, nl_configs=NL_CONFIGS)
#  fit_nl_models(sessions, nl_configs=NL_CONFIGS, verbose=False):

# nl_names = [c[0] for c in NL_CONFIGS]
# nNL      = len(NL_CONFIGS)
clrs_nl  = sns.color_palette('tab10', nNL)

#%% Plot R² distributions across models and neurons
celldata = pd.concat([ses.celldata for ses in sessions])
bw_adjust = 0.25
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
idx_N = np.all((celldata['noise_level']<20,
                celldata['roi_name']=='V1',
                # celldata['session_id'].isin([sessions[11].session_id]),
                # celldata['gOSI']>0.5,
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
r2_df = celldata[['R2' + name for name in nl_names]][idx_N].dropna()
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


