"""
This script shows you how to load one session (shallow load)
It creates an instance of a session which by default loads information about
the session, trials and the cells, but does not load behavioral data traces, 
video data and calcium activity
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% 
import os
os.chdir('E:\\Python\\molanalysis\\')
import numpy as np
from loaddata.session_info import load_sessions,filter_sessions
import matplotlib.pyplot as plt

protocol            = 'GR'
session_list = np.array([['LPE11086', '2024_01_05']])

#%% Load the session in a lazy way, i.e. no heavy data such as behavioral, video or calcium imaging data:
sessions,nSessions = load_sessions(protocol,session_list) #no behav or ca data

#%% Filter and load only sessions with V1 and PM recordings:
sessions,nSessions = filter_sessions(protocol)
sessions,nSessions = filter_sessions(protocol,only_all_areas=['V1','PM'])
sessions,nSessions = filter_sessions(protocol,only_all_areas=['V1','PM'],filter_areas=['V1','PM'])

#%% Load the session thoroughly, including calcium imaging data etc.
sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_calciumdata=True,
                                   load_videodata=True,calciumversion='deconv')

#%% Task and stimulus events: 
sessions[0].trialdata['StimStart'] #spatial position in corridor of stimulus
sessions[0].trialdata['tStimStart'] #time stamp of entering stimulus zone

sessions[0].trialdata['Orientation'] # Grating orientation

K = len(sessions[0].trialdata) #K trials

#%% Calcium imaging data:
sessions[0].ts_F #timestamps of the imaging data
sessions[0].calciumdata #data (samples x features)

T,N = np.shape(sessions[0].calciumdata) #T imaging frames, N neurons

#%% information about the cells: 
sessions[0].celldata

#e.g. filter calciumdata for cells in V1:
cellidx         = sessions[0].celldata['roi_name'].to_numpy() == 'V1'
sessions[0].calciumdata.iloc[:,cellidx] #only v1 cells

#%% In behaviordata is the running speed, and in task protocols the position in the corridor, lick timestamps, rewards at 100 Hz
plt.plot(sessions[0].behaviordata['runspeed'])

#timestamps of licks: 
sessions[0].behaviordata.loc[sessions[0].behaviordata['lick'],'ts'] #time stamp of entering stimulus zone

#timestamps of rewards:
sessions[0].behaviordata.loc[sessions[0].behaviordata['reward'],'ts'] #time stamp of entering stimulus zone

# Some of these continuous behavioral variable also exist as interpolated
# versions at imaging sampling rate in sessiondata:
sessions[0].zpos_F
sessions[0].runspeed_F

#%% Video data: 
plt.plot(sessions[0].videodata['pupil_area'])
