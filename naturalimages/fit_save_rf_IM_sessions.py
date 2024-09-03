# -*- coding: utf-8 -*-
"""
This script fits receptive fields to the average response triggered image 
and saves the results for each session
Matthijs Oude Lohuis, 2023-2025, Champalimaud Center
"""
# %% # Imports
# Import general libs
import os
import numpy as np
import pandas as pd
os.chdir('e:\\Python\\molanalysis')

# Import personal lib funcs
from loaddata.session_info import filter_sessions
from loaddata.get_data_folder import get_local_drive
from utils.rf_lib import estimate_rf_IM

# %% Load IM session with receptive field mapping ################################################
sessions, nSessions = filter_sessions(protocols='IM')

sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions = sessions[9:]
sessiondata['session_id']
# 0    LPE11495_2024_02_29
# 1    LPE11622_2024_03_27
# 2    LPE11998_2024_05_08
# 3    LPE12223_2024_06_11

for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv', keepraw=False)

#%% Get response-triggered frame for cells and then estimate receptive field from that:
for ses in sessions:    # iterate over sessions
    rf_data    = estimate_rf_IM(ses,show_fig=False)
    outfilepath = ses.sessiondata_path.replace('sessiondata','IMrfdata')
    rf_data.to_csv(outfilepath, sep=',')


