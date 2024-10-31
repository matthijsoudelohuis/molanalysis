
#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loaddata.session_info import filter_sessions,load_sessions
from utils.rf_lib import *
from loaddata.get_data_folder import get_local_drive
from utils.corr_lib import compute_pairwise_anatomical_distance
from utils.plotting_style import * #get all the fixed color schemes

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - RF\\')

#%% ################### Loading the data ##############################

# session_list        = np.array([['LPE11086','2024_01_10']])
# session_list        = np.array([['LPE12013','2024_05_02']])
session_list        = np.array([['LPE09665','2023_03_21']])

sessions,nSessions = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions = load_sessions(protocol = 'GN',session_list=session_list)
# sessions,nSessions = filter_sessions(protocols = ['RF'])

#%% Compute delta RF and delta XY:
sessions    = compute_pairwise_anatomical_distance(sessions)
sessions    = compute_pairwise_delta_rf(sessions,rf_type='Fsmooth')

#%% Show correlation between anatomical distance and delta receptive field distance for V1 and PM:
areapairs = ['V1-V1','PM-PM']
fig = scatter_dXY_dRF(sessions[0],areapairs)
fig.savefig(os.path.join(savedir,'dXY_dRF_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')
