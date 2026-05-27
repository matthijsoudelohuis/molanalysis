#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
import pickle

os.chdir('e:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.tuning import *
from utils.nonlin_lib import *
# from utils.params import load_params

params = dict(
                figdir          = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain'),
                resultdir       = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Analysis\\SharedGain'),
                calciumversion = 'deconv', #deconv or dF
                t_pre       = -1,         #pre s
                t_post      = 2,        #post s
                maxnoiselevel = 20, #maximum noise level to include cell
                )

#%% Load parameters and settings:
version = 'allGR_sessions'
session_list        = None

# version = 'LPE12223_2024_06_10'
# session_list        = np.array([['LPE12223_2024_06_10']])

resultdir = os.path.join(params['resultdir'])
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savefilename = os.path.join(resultdir,'NonLin_Fit_%s_%s' % (version,datetime_str))

#%% 
sessions,nSessions  = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
report_sessions(sessions)

# #%% 
# for ises in range(nSessions):
#     sessions[ises].cellfilter = np.zeros(len(sessions[ises].celldata), dtype=bool) #TEMPORARY: only fit first 100 neurons to speed up testing
#     sessions[ises].cellfilter[:10] = True

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion='deconv')

#%% Add how neurons are coupled to the population rate: 
sessions = comp_poprate(sessions,version='radius_500')
sessions = compute_pop_coupling(sessions)
# sessions = compute_pop_coupling(sessions,version='allfast')
sessions = ori_remapping(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

#%% Fit nonlinearity models to each neuron and collect results:
[sessions, theta_arr, nlpar_arr, ses_idx_arr] = fit_nl_models_sessions(sessions, nl_configs=NL_CONFIGS)

#%% Save the data:
np.savez(savefilename + '.npz',
         sessions=sessions, theta_arr=theta_arr, nlpar_arr=nlpar_arr, ses_idx_arr=ses_idx_arr,
         allow_pickle=True)

# with open(savefilename +'_params' + '.txt', "wb") as myFile:
#     pickle.dump(params, myFile)
