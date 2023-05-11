# -*- coding: utf-8 -*-
"""
This script shows you how to load one session (shallow load)
It creates an instance of a session which by default loads information about
the session, trials and the cells, but does not load behavioral data traces, 
video data and calcium activity
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import numpy as np
from session_info import filter_sessions,load_sessions,report_sessions

# animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed
# sessiondates        = ['2023_04_10']
protocol            = 'GR'

session_list = np.array([['LPE09830', '2023_04_10'],
                        ['LPE09665', '2023_03_14']])

session_list = np.array([['LPE09830', '2023_04_10']])

sessions = load_sessions(protocol,session_list)

report_sessions(sessions)

sessions = filter_sessions(protocol)

sessions[0].load_data(load_behaviordata=True,load_calciumdata=True)

