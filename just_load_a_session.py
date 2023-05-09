# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:59:13 2023

@author: USER
"""

import numpy as np
from session_info import filter_sessions,load_sessions,report_sessions

procdatadir         = "V:\\Procdata\\"

animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_04_10']
protocol            = 'GR'

session_list = np.array([['LPE09830', '2023_04_10'],
                        ['LPE09665', '2023_03_14']])


sessions = load_sessions(protocol,session_list)

report_sessions(sessions)

sessions = filter_sessions(protocol)

# session = Session(animal_id=animal_id, session_id=session_id,
#                   load_lfp=False, load_spikes=True)


