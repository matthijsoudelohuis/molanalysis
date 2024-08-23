"""
This script has data loading functions used to get dirs, filter and select sessions
Actual loading happens as method of instances of sessions (session.py)
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os
import numpy as np
import pandas as pd
from loaddata.get_data_folder import get_data_folder
from loaddata.session import Session
import logging

logger = logging.getLogger(__name__)


def load_sessions(protocol, session_list, load_behaviordata=False, load_calciumdata=False, load_videodata=False, calciumversion='dF'):
    """
    This function loads and outputs the session objects that have to be loaded.
    session_list is a 2D np array with animal_id and session_id pairs (each row one session)

    sessions = load_sessions(protocol,session_list)
    """
    sessions = []

    assert np.shape(session_list)[
        1] == 2, 'session list does not seem to have two columns for animal and dates'

    # iterate over sessions in requested array:
    for i, ses in enumerate(session_list):
        ses = Session(
            protocol=protocol, animal_id=session_list[i, 0], session_id=session_list[i, 1])
        ses.load_data(load_behaviordata, load_calciumdata,
                      load_videodata, calciumversion)

        sessions.append(ses)

    report_sessions(sessions)

    return sessions, len(sessions)


def filter_sessions(protocols, load_behaviordata=False, load_calciumdata=False,
                    load_videodata=False, calciumversion='dF',
                    only_animal_id=None, min_cells=None, min_trials=None, session_rf=None,
                    incl_areas=None, only_areas=None, has_pupil=False):
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
    if isinstance(protocols, str):
        protocols = [protocols]

    if protocols is None:
        protocols = ['VR', 'IM', 'GR', 'GN', 'RF', 'SP', 'DM', 'DN', 'DP']

    # iterate over files in that directory
    for protocol in protocols:
        for animal_id in os.listdir(os.path.join(get_data_folder(), protocol)):
            for session_id in os.listdir(os.path.join(get_data_folder(), protocol, animal_id)):

                ses = Session(protocol=protocol,
                              animal_id=animal_id, session_id=session_id)
                ses.load_data(load_behaviordata=False,
                              load_calciumdata=False, load_videodata=False)

                # go through specified conditions that have to be met for the session to be included:
                sesflag = True

                # SELECT BASED ON # TRIALS
                if only_animal_id is not None:
                    sesflag = sesflag and animal_id in only_animal_id

                # SELECT BASED ON # TRIALS
                if min_trials is not None:
                    sesflag = sesflag and len(ses.trialdata) >= min_trials

                # SELECT BASED ON # CELLS
                if sesflag and min_cells is not None:
                    sesflag = sesflag and hasattr(
                        ses, 'celldata') and len(ses.celldata) >= min_cells

                if sesflag and session_rf is not None:
                    sesflag = sesflag and hasattr(
                        ses, 'celldata') and 'rf_p_F' in ses.celldata

                if sesflag and incl_areas is not None:
                    sesflag = sesflag and hasattr(ses, 'celldata') and np.any(
                        np.isin(incl_areas, np.unique(ses.celldata['roi_name'])))

                if sesflag and only_areas is not None:
                    sesflag = sesflag and hasattr(ses, 'celldata') and np.all(
                        np.isin(np.unique(ses.celldata['roi_name']), only_areas))

                if sesflag and has_pupil:
                    ses.load_data(load_videodata=True)
                    sesflag = sesflag and hasattr(
                        ses, 'videodata') and 'pupil_area' in ses.videodata and np.any(ses.videodata['pupil_area'])
                if sesflag:
                    ses.load_data(load_behaviordata, load_calciumdata,
                                  load_videodata, calciumversion)
                    sessions.append(ses)

    report_sessions(sessions)

    return sessions, len(sessions)


def report_sessions(sessions):
    """
    This function reports show stats about the loaded sessions 
    """

    sessiondata = pd.DataFrame()
    trialdata = pd.DataFrame()
    celldata = pd.DataFrame()

    for ses in sessions:
        sessiondata = pd.concat([sessiondata, ses.sessiondata])
        trialdata = pd.concat([trialdata, ses.trialdata])
        if hasattr(ses, 'celldata'):
            celldata = pd.concat([celldata, ses.celldata])

    logger.info(
        f'{pd.unique(sessiondata["protocol"])} dataset: {len(pd.unique(sessiondata["animal_id"]))} mice, {len(sessiondata)} sessions, {len(trialdata)} trials')

    if np.any(celldata):
        for area in np.unique(celldata['roi_name']):
            logger.info(
                f"Number of neurons in {area}: {len(celldata[celldata['roi_name'] == area])}")
        logger.info(f"Total number of neurons: {len(celldata)}")

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
