"""
This script shows you how to load one session (shallow load)
It creates an instance of a session which by default loads information about
the session, trials and the cells, but does not load behavioral data traces, 
video data and calcium activity
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""
import os
os.chdir('C:\\Python\\molanalysis\\')

import numpy as np
from loaddata.session_info import load_sessions

protocol            = 'VR'

# session_list = np.array([['LPE09830', '2023_04_10']])

session_list = np.array([['LPE09829', '2023_03_29']])

sessions = load_sessions(protocol,session_list) #no behav or ca data
sessions = load_sessions(protocol,session_list,load_behaviordata=True,load_calciumdata=True,calciumversion='deconv')

# Calcium imaging data:
sessions[0].ts_F #timestamps of the imaging data
sessions[0].calciumdata #data (samples x features)

T,N = np.shape(sessions[0].calciumdata)
K = len(sessions[0].trialdata)

#information about the cells: 
sessions[0].celldata

#e.g. filter calciumdata for cells in V1:
cellidx         = sessions[0].celldata['roi_name'].to_numpy() == 'V1'
sessions[0].calciumdata.iloc[:,cellidx] #only v1 cells

# Events: 
sessions[0].trialdata['StimStart'] #spatial position in corridor of stimulus
sessions[0].trialdata['tStimStart'] #time stamp of entering stimulus zone

sessions[0].trialdata['stimRight'] #stimulus on the (contralateral) side of the corridor

#in behaviordata are position in the corridor, running speed, lick timestamps, rewards at 100 Hz
sessions[0].behaviordata['zpos'] #position in the corridor

#timestamps of licks: 
sessions[0].behaviordata.loc[sessions[0].behaviordata['lick'],'ts'] #time stamp of entering stimulus zone

#timestamps of rewards:
sessions[0].behaviordata.loc[sessions[0].behaviordata['reward'],'ts'] #time stamp of entering stimulus zone

# Some of these continuous behavioral variable also exist as interpolated
# versions at imaging sampling rate in sessiondata:
sessions[0].zpos_F
sessions[0].trialnum_F
sessions[0].runspeed_F
