
# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""
# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2022-2026, Champalimaud Center, Lisbon
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import binned_statistic,binned_statistic_2d
from scipy.signal import detrend
from statannotations.Annotator import Annotator
from scipy.optimize import curve_fit

from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error,my_ceil,my_floor
from utils.corr_lib import *
from utils.rf_lib import smooth_rf,exclude_outlier_rf,filter_nearlabeled,replace_smooth_with_Fsig
from utils.tuning import compute_tuning, compute_prefori

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% Load all sessions from certain protocols: 
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=['V1','PM']) 

#%% Remove two sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=False)

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filtersig=False)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% 
for ses in sessions:
    if 'rf_r2_Fgauss' in ses.celldata:
        ses.celldata['rf_p_Fgauss'] = ses.celldata['rf_r2_Fgauss']<0.2
        ses.celldata['rf_p_Fneugauss'] = ses.celldata['rf_r2_Fneugauss']<0.2

#%% Copy Fgauss to F
for ses in sessions:
    if 'rf_az_Fgauss' in ses.celldata:
        ses.celldata['rf_az_F'] = ses.celldata['rf_az_Fgauss']
        ses.celldata['rf_el_F'] = ses.celldata['rf_el_Fgauss']
        ses.celldata['rf_p_F'] = ses.celldata['rf_p_Fgauss']

#%% ##################### Compute pairwise receptive field distances: ##############################
sessions = smooth_rf(sessions,radius=50,rf_type='Fneugauss',mincellsFneu=5)
sessions = exclude_outlier_rf(sessions) 
sessions = replace_smooth_with_Fsig(sessions) 

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    if sessions[ises].sessiondata['protocol'].isin(['GR'])[0]:
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
    else:
        sessions[ises].celldata['OSI'] = sessions[ises].celldata['tuning_var'] = np.nan

#%% Give redcells a string label
redcelllabels = np.array(['unl','lab'])
for ses in sessions:
    ses.celldata['labeled'] = ses.celldata['redcell']
    ses.celldata['labeled'] = ses.celldata['labeled'].astype(int).apply(lambda x: redcelllabels[x])
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
celldata = pd.concat([ses.celldata[filter_nearlabeled(ses,radius=50)] for ses in sessions]).reset_index(drop=True)

## remove any double cells (for example recorded in both GR and RF)
celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

celldata['area_label'] = celldata['roi_name'] + celldata['labeled']

#%% Compute the variance across trials for each cell:
for ses in sessions:
    if ses.sessiondata['protocol'][0]=='GR':
        resp_meanori,respmat_res        = mean_resp_gr(ses)
    elif ses.sessiondata['protocol'][0]=='GN':
        resp_meanori,respmat_res        = mean_resp_gn(ses)
    ses.celldata['noise_variance'] = np.var(respmat_res,axis=1)

#%% Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata[filter_nearlabeled(ses,radius=50)] for ses in sessions]).reset_index(drop=True)

celldata['area_label'] = celldata['roi_name'] + celldata['labeled']

#%% ##################### Calcium trace skewness for labeled vs unlabeled cells:
# order = [0,1] #for statistical testing purposes
# pairs = [(0,1)]

order = ['V1unl','V1lab','PMunl','PMlab'] #for statistical testing purposes
pairs = [('V1unl','V1lab'),('PMunl','PMlab')]

# fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF","meanF_chan2"]
fields = ["noise_level","meanF","tuning_var","OSI",'noise_variance','depth']

nfields = len(fields)
# fig,axes   = plt.subplots(1,nfields,figsize=(12,4))

celldata['noise_level'].clip(upper=10,inplace=True)
celldata['noise_variance'].clip(upper=0.6,inplace=True)

for i in range(nfields):
    fig,ax   = plt.subplots(1,1,figsize=(3,4))
    sns.violinplot(data=celldata,y=fields[i],x="area_label",palette=['gray','orangered','gray','indianred'],
                    ax=ax,order=order,inner='quart',cut=10)
    # sns.violinplot(data=celldata,y=fields[i],x="roi_name",palette=['gray','orangered'],order=['V1','PM'],
                #    hue='labeled',ax=ax,split=True,inner='quart')
    ax.set_ylim(np.nanpercentile(celldata[fields[i]],[0,98]))
    # ax.set_xlim(-0.5,1.5)

    annotator = Annotator(ax, pairs, data=celldata, x="area_label", y=fields[i], order=order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    ax.set_xlabel('area + label')
    ax.set_ylabel('')
    ax.set_title(fields[i])
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'Quality_Metric_GRGN_%s_%dcells_%dsessions' % (fields[i],len(celldata),nSessions) + '.png'), format = 'png')

# plt.tight_layout()
# fig.savefig(os.path.join(savedir,'Quality_Metrics_GRGN_%dcells_%dsessions' % (len(celldata),nSessions) + '.png'), format = 'png')

