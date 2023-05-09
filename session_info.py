import numpy as np
import pandas as pd
from session import Session
from constants import *
import os
# import subprocess
# import warnings

# def make_trial_overview():

#     dfs = []

#     for animal_id in decoding_sessions.keys():
#         for session_id in decoding_sessions[animal_id]:
#             session = Session(animal_id=animal_id, session_id=session_id)
#             session.load_data(load_spikes=False, load_lfp=False)
#             df = session.get_session_overview_trials()
#             print(df)
#             dfs.append(df)

#     df = pd.concat(dfs).reset_index().drop('index', axis=1)

#     ind = df.loc[(df['animal_id'] == '2009') &
#            (df['session_id'] == '2018-08-24_11-56-35') &
#            (df['target_name'] == 'visualOriPostNorm')].index

#     df = df.drop(ind)

#     # just to make sure
#     sel = df.loc[(df['animal_id'] == '2009') &
#            (df['session_id'] == '2018-08-24_11-56-35') &
#            (df['target_name'] == 'visualOriPostNorm')]
#     assert sel.shape[0] == 0

#     return df



# def make_units_overview():

#     dfs = []

#     for animal_id in decoding_sessions.keys():
#         for session_id in decoding_sessions[animal_id]:
#             session = Session(animal_id=animal_id, session_id=session_id)
#             # TODO we should be able to provide the unit layer without loading
#             # the LFP
#             session.load_data(load_spikes=True, load_lfp=False)
#             df = session.get_session_overview_n_units()
#             dfs.append(df)

#     return pd.concat(dfs)

def load_sessions(protocol,session_list):
    """
    This function loads and outputs the session objects that have to be loaded.
    session_list is a 2D np array with animal_id and session_id pairs (each row one session)

    sessions = load_sessions(protocol,session_list)
    """
    sessions = []
    
    # iterate over sessions in requested array:
    for i,ses in enumerate(session_list):
        ses = Session(protocol=protocol,animal_id=session_list[i,0],session_id=session_list[i,1])
        ses.load_data()
        
        sessions.append(ses)
      
    report_sessions(sessions)            
    
    return sessions

def filter_sessions(protocol,min_cells=None, min_trials=None):
        #             only_correct=False, min_units=None,
        #             min_units_per_layer=None, min_channels_per_layer=None,
        #             exclude_NA_layer=False, min_perc_correct=None):
    """
    This function outputs a list of session objects with all the sessions which
    respond to the given criteria. Usage is as follows:

    sessions = filter_sessions(protocol,min_trials=100)
    
    :param min_trials: To restrict to sessions which have minimum trials
    """
    sessions = []
    
    # iterate over files in that directory
    for animal_id in os.listdir(os.path.join(DATA_FOLDER,protocol)):
        for session_id in os.listdir(os.path.join(DATA_FOLDER,protocol,animal_id)):
            
            ses = Session(protocol=protocol,animal_id=animal_id,session_id=session_id)
            ses.load_data()
            
            ## go through specified conditions that have to be met for the session to be included:
            sesflag = True
            
            # SELECT BASED ON # TRIALS
            if min_trials is not None:
                sesflag = len(ses.trialdata) >= min_trials
            
            # SELECT BASED ON # CELLS
            if sesflag and min_cells is not None:
                sesflag = len(ses.celldata) >= min_cells
                
            if sesflag:
                sessions.append(ses)
      
    report_sessions(sessions)            
    
    return sessions

def report_sessions(sessions):

    """
    This function reports show stats about the loaded sessions 
    """
    
    sessiondata     = pd.DataFrame()
    trialdata       = pd.DataFrame()
    celldata        = pd.DataFrame()
    
    for ses in sessions:
        sessiondata     = sessiondata.append(ses.sessiondata)
        trialdata       = trialdata.append(ses.trialdata)
        celldata        = celldata.append(ses.celldata)
    
    print("{protocol} dataset: {nsessions} sessions, {ntrials} trials".format(
        protocol = pd.unique(sessiondata['protocol']),nsessions = len(sessiondata),ntrials = len(trialdata)))

    print("Neurons in area:")
    print(celldata.groupby('roi_name')['roi_name'].count())
    
    
    # print("{nneurons} dataset: {nsessions} sessions, {ntrials} trials".format(
        # protocol = sessions[0].sessiondata.protocol,nsessions = len(sessiondata),ntrials = len(trialdata))
                    
    # # SELECT BASED ON NUMBER OF UNITS PER AREA
    # if min_units is not None:
    #     units_df = make_units_overview()
    #     units_df = units_df[units_df['n_units'] >= min_units]
    #     sel_df = pd.merge(units_df, trial_df, on=['animal_id', 'session_id'])


    # # SELECT BASED ON NUMBER OF UNITS/CHANNELS PER AREA/LAYER
    # if min_units_per_layer is not None or min_channels_per_layer is not None:
    #     layers_df = make_units_and_channels_overview()
    #     #layers_df_copy = layers_df.copy()

    #     if min_units_per_layer is not None:
    #         layers_df = layers_df[layers_df['n_units'] >= min_units_per_layer]

    #     if min_channels_per_layer is not None:
    #         layers_df = layers_df[
    #             layers_df['n_channels'] >= min_channels_per_layer]

    #     if exclude_NA_layer:
    #         layers_df = layers_df[layers_df['layer'] != 'NA']

    #     sel_df = pd.merge(layers_df, trial_df, on=['animal_id', 'session_id'])

    # try:
    #     return sel_df
    # except NameError:
    #     return trial_df




# if __name__ == '__main__':
#     # performance

#     only_correct = True
#     stimulus_type = None
#     min_trials_per_stim = 10
#     min_units = None
#     min_units_per_layer = 12
#     min_channels_per_layer = 10
#     min_perc_correct = 40
#     exclude_NA_layer = True

#     if min_units is not None and min_units_per_layer is not None:
#         raise ValueError('You can select either the minimum number of (total) units'
#                          'per area, or the minimum number of units per area/layer, '
#                          'not both!')

#     trial_df = make_trial_overview()

#     if min_trials_per_stim is not None:
#         trial_df = trial_df.loc[(trial_df.only_correct == only_correct) &
#                                     (trial_df.nt_0 >= min_trials_per_stim) &
#                                     (trial_df.nt_1 >= min_trials_per_stim)]

#     if stimulus_type is not None:
#         trial_df = trial_df[trial_df['stimulus_type'] == stimulus_type]

#     if min_perc_correct is not None:
#         trial_df = trial_df[trial_df['perc_corr'] >= min_perc_correct]



#     if min_units is not None:
#         units_df = make_units_overview()

#         units_df = units_df[units_df['n_units'] >= min_units]
#         sel_df = pd.merge(units_df, trial_df, on=['animal_id', 'session_id'])


#     if min_units_per_layer is not None or min_channels_per_layer is not None:
#         layers_df = make_units_and_channels_overview()
#         layers_df_copy = layers_df.copy()

#         if min_units_per_layer is not None:
#             layers_df = layers_df[layers_df['n_units'] >= min_units_per_layer]

#         if min_channels_per_layer is not None:
#             layers_df = layers_df[layers_df['n_channels'] >= min_channels_per_layer]

#         if exclude_NA_layer:
#             layers_df = layers_df[layers_df['layer'] != 'NA']

#         sel_df = pd.merge(layers_df, trial_df, on=['animal_id', 'session_id'])



