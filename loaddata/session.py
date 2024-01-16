# -*- coding: utf-8 -*-
"""
This script load one session
By default it is a shallow load which means it loads information about
the session, trials and the cells, but does not load behavioral data traces, 
video data and calcium activity.
It creates an instance of a class session
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import os
import numpy as np
import pandas as pd
from loaddata.get_data_folder import get_data_folder

class Session():

    def __init__(self, protocol='',animal_id='', session_id='', verbose=1):

        print('\nInitializing Session object for: \n- animal ID: {}'
              '\n- Session ID: {}\n'.format(animal_id, session_id))
        self.data_folder = os.path.join(get_data_folder(), protocol, animal_id, session_id)
        self.verbose = verbose
        self.protocol = protocol
        self.animal_id = animal_id
        self.session_id = session_id

    def load_data(self, load_behaviordata=False, load_calciumdata=False, load_videodata=False, calciumversion='dF'):
        #Calciumversion can be 'dF' or 'deconv'

        self.sessiondata_path   = os.path.join(self.data_folder, 'sessiondata.csv')
        self.trialdata_path     = os.path.join(self.data_folder, 'trialdata.csv')
        self.celldata_path      = os.path.join(self.data_folder, 'celldata.csv')
        self.behaviordata_path  = os.path.join(self.data_folder, 'behaviordata.csv')
        self.videodata_path     = os.path.join(self.data_folder, 'videodata.csv')
        self.calciumdata_path   = os.path.join(self.data_folder, '%sdata.csv' % calciumversion)

        assert(os.path.exists(self.sessiondata_path)), 'Could not find data in {}'.format(self.sessiondata_path)

        self.sessiondata  = pd.read_csv(self.sessiondata_path, sep=',', index_col=0)

        if not self.protocol in ['SP','RF']:
            self.trialdata  = pd.read_csv(self.trialdata_path, sep=',', index_col=0)
        else:
            self.trialdata = None
    
        if os.path.exists(self.celldata_path):
            self.celldata  = pd.read_csv(self.celldata_path, sep=',', index_col=0)
            # get only good cells (selected ROIs by suite2p):
            goodcells               = self.celldata['iscell'] == 1
            self.celldata           = self.celldata[goodcells].reset_index(drop=True)
        
        if load_behaviordata:
            # print('Loading behavior data at {}'.format(self.behaviordata_path))
            self.behaviordata  = pd.read_csv(self.behaviordata_path, sep=',', index_col=0)
        else:
            self.behaviordata = None

        if load_videodata:
            self.videodata  = pd.read_csv(self.videodata_path, sep=',', index_col=0)
        else:
            self.videodata = None

        if load_calciumdata:
            print('Loading calcium data at {}'.format(self.calciumdata_path))
            self.calciumdata         = pd.read_csv(self.calciumdata_path, sep=',', index_col=0)
            self.calciumdata         = self.calciumdata.drop('session_id',axis=1)

            self.ts_F                = self.calciumdata['timestamps']
            self.calciumdata         = self.calciumdata.drop('timestamps',axis=1)

            self.F_chan2             = self.calciumdata['F_chan2']
            self.calciumdata         = self.calciumdata.drop('F_chan2',axis=1)

            self.calciumdata         = self.calciumdata.drop(self.calciumdata.columns[~goodcells],axis=1)
            
            assert(np.shape(self.calciumdata)[1]==np.shape(self.celldata)[0])

        if load_calciumdata and load_behaviordata:
            ## Get interpolated values for behavioral variables at imaging frame rate:
            self.zpos_F      = np.interp(x=self.ts_F,xp=self.behaviordata['ts'],
                                    fp=self.behaviordata['zpos'])
            self.runspeed_F  = np.interp(x=self.ts_F,xp=self.behaviordata['ts'],
                                    fp=self.behaviordata['runSpeed'])
            if 'trialNumber' in self.behaviordata:
                self.trialnum_F  = np.interp(x=self.ts_F,xp=self.behaviordata['ts'],
                                    fp=self.behaviordata['trialNumber'])

            
#     def initialize(self, session_data, trial_data, spike_data=None, lfp_data=None,
#                    center_lfp=True):

#         self.session_data = session_data

#         # --- ADD TRIAL DATA ---
#         trial_data_df = pd.DataFrame(columns=trial_data.keys())
#         for key in trial_data.keys():
#             trial_data_df[key] = trial_data[key]
#         self.trial_data = trial_data_df

#         self.trial_data['responseSide'] = [i if isinstance(i, str) else 'n' for i in
#                                            self.trial_data.responseSide]

#         first_lick_times = []
#         for times in self.trial_data['lickTime']:
#             try:
#                 first_lick_time = times[0]
#             except IndexError:
#                 first_lick_time = np.nan
#             except TypeError:
#                 first_lick_time = times
#             first_lick_times.append(first_lick_time)

#         self.trial_data['firstlickTime'] = first_lick_times
#         self.trial_data['Lick'] = [0 if n == 1 else 1 for n in self.trial_data['noResponse']]

#         # --- ADD SPIKE DATA ---
#         self.spike_data = spike_data
#         if spike_data is not None:
#             self.spike_time_stamps = spike_data['ts']
#             # since these are originally matlab indices, turn to string to avoid
#             # that they are used here as indices
#             self.spike_data['ch'] = self.spike_data['ch'].astype(str)
#             self.cell_id = self.spike_data['cell_ID'].astype(int).astype(str)
#             self.session_t_start = np.hstack(self.spike_data['ts']).min() * pq.us
#             self.session_t_stop = np.hstack(self.spike_data['ts']).max() * pq.us

#             # RUN CHECKS
#             # no duplicate cell ids
#             assert pd.value_counts(self.spike_data['cell_ID']).max() == 1



#         # --- ADD LFP DATA ---
#         self.lfp_data = lfp_data

#         if lfp_data is not None:
#             self.sampling_rate = lfp_data['fs'][0]

#             self.session_t_start = lfp_data['t_start'][0] * pq.us
#             self.session_t_stop = lfp_data['t_end'][0] * pq.us
#             self.delta_t = (1 / self.sampling_rate) * 1e6 * pq.us

#             self.channel_id = lfp_data['channel_ID'].astype(str)
#             # self.lfp_times = np.arange(self.session_t_start,
#             #                            self.session_t_stop+self.delta_t, self.delta_t)

#             self.lfp_data['signal'] = np.vstack(self.lfp_data['signal'])

#             if center_lfp:
#                 self.lfp_data['signal'] = scipy.signal.detrend(self.lfp_data['signal'],
#                                                                 type='constant')

#             self.lfp_times = self.session_t_start + np.arange(self.lfp_data['signal'].shape[1]) / (self.sampling_rate * pq.Hz)

#             # RUN CHECKS ON LFP DATA
#             assert np.all(lfp_data['fs']==self.sampling_rate)
#             assert np.all(lfp_data['butter_applied']==1)
#             assert np.all(lfp_data['kaiser_applied'] == 0)
#             assert np.all(lfp_data['t_units']=='us')
#             assert np.all(lfp_data['signal_units']=='V')
#             assert np.all(lfp_data['t_start']==lfp_data['t_start'][0])
#             assert np.all(lfp_data['t_end']==lfp_data['t_end'][0])
#             assert self.lfp_times.shape[0] == self.lfp_data['signal'].shape[1]


#     def quick_downsample_lfp(self, factor=2):

#         self.lfp_times = self.lfp_times[::factor]
#         self.lfp_data['signal'] = self.lfp_data['signal'][:, ::factor]


#     def select_trials(self, trial_type=None, only_correct=False,
#                       visual_post_norm=None, audio_post_norm=None,
#                       visual_change=None, auditory_change=None, exclude_last=20,
#                       auditory_pre=None, auditory_post=None, response=None,
#                       response_side=None):

#         """
#         trial_type can be just a string, or a list of strings if you
#         want to include more trial types.

#         visual_change and auditory_change can be 0 (no split),
#         1 (small split), or 2 (big split).

#         auditory_pre and auditory_post are the pre and post split frequencies.

#         """

#         total_n_trials = self.trial_data.shape[0]
#         selected_trials = []

#         if trial_type is not None:
#             if isinstance(trial_type, str):
#                 trial_type = [trial_type]
#             mask = np.isin(self.trial_data['trialType'], trial_type)
#             sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if response is not None:
#             if isinstance(response, int):
#                 response = [response]
#             mask = np.isin(self.trial_data['correctResponse'], response)
#             sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if only_correct:
#             ind = self.trial_data['correctResponse'] == 1
#             sel_trials = self.trial_data.loc[ind, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if visual_change is not None:
#             if isinstance(visual_change, (float, int)):
#                 visual_change = [visual_change]
#             mask = np.isin(self.trial_data['visualOriChangeNorm'], visual_change)
#             sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if auditory_change is not None:
#             if isinstance(auditory_change, (float, int)):
#                 auditory_change = [auditory_change]
#             mask = np.isin(self.trial_data['audioFreqChangeNorm'], auditory_change)
#             sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if auditory_pre is not None:
#             ind = self.trial_data['audioFreqPreChange'] == auditory_pre
#             sel_trials = self.trial_data.loc[ind, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if auditory_post is not None:
#             if isinstance(auditory_post, (float, int)):
#                 auditory_post = [auditory_post]
#             mask = np.isin(self.trial_data['audioFreqPostChange'], auditory_post)
#             sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if audio_post_norm is not None:
#             if isinstance(audio_post_norm, (float, int)):
#                 audio_post_norm = [audio_post_norm]
#             mask = np.isin(self.trial_data['audioFreqPostChangeNorm'], audio_post_norm)
#             sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if visual_post_norm is not None:
#             if isinstance(visual_post_norm, (float, int)):
#                 visual_post_norm = [visual_post_norm]
#             mask = np.isin(self.trial_data['visualOriPostChangeNorm'], visual_post_norm)
#             sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if response_side is not None:
#             ind = self.trial_data['responseSide'] == response_side
#             sel_trials = self.trial_data.loc[ind, 'trialNum'].tolist()
#             selected_trials.append(sel_trials)

#         if len(selected_trials) > 0:
#             selected_final = list(set.intersection(*map(set, selected_trials)))
#         else:
#             selected_final = self.trial_data['trialNum'].tolist()

#         perc = 100*(len(selected_final) / total_n_trials)
#         print('> Finished trial selection'
#               '\n ---> {} out of {} trials selected '
#               '({:.1f}%)'.format(len(selected_final), total_n_trials, perc))

#         if exclude_last is not None:
#             last_trials = self.trial_data['trialNum'][-exclude_last:].tolist()
#             selected_final = list(set(selected_final)-set(last_trials))

#         selected_final.sort()

#         return selected_final


#     def get_trial_info(self, trial_number):
#         return self.trial_data[self.trial_data['trialNum'] == trial_number]


#     def get_type_of_trial(self, trial_number):
#         trial_info = self.get_trial_info(trial_number)
#         return trial_info['trialType'].iloc[0]


#     def get_response_side_of_trial(self, trial_number):
#         trial_info = self.get_trial_info(trial_number)
#         return trial_info['responseSide'].iloc[0]

#     def get_correct_response_of_trial(self, trial_number):
#         trial_info = self.get_trial_info(trial_number)
#         return trial_info['correctResponse'].iloc[0]

#     def get_stimulus_change_of_trial(self, trial_number):
#         trial_info = self.get_trial_info(trial_number)
#         trial_type = self.get_type_of_trial(trial_number)
#         if trial_type == 'X':
#             stim = trial_info['visualOriChangeNorm'].iloc[0]
#         elif trial_type == 'Y':
#             stim = trial_info['audioFreqChangeNorm'].iloc[0]
#         elif trial_type == 'P':
#             stim = 0
#         else:
#             raise ValueError('Stimulus identity for which modality?')
#         return stim

#     def get_lick_time_of_trial(self, trial_number):
#         trial_info = self.get_trial_info(trial_number)
#         return trial_info['firstlickTime'].iloc[0]

#     def get_lick_time_of_trials(self, trial_numbers):
#         return [self.get_lick_time_of_trial(t) for t in trial_numbers]

#     def get_stimulus_identity_of_trial(self, trial_number):
#         trial_info = self.get_trial_info(trial_number)
#         trial_type = self.get_type_of_trial(trial_number)
#         if trial_type == 'X':
#             stim = trial_info['visualOriPostChangeNorm'].iloc[0]
#         elif trial_type == 'Y':
#             stim = trial_info['audioFreqPostChangeNorm'].iloc[0]
#         elif trial_type == 'P':
#             stim = 0
#         else:
#             raise ValueError('Stimulus identity for which modality?')
#         return stim


#     def select_units(self, area=None, layer=None, min_isolation_distance=None,
#                      min_coverage=None, max_perc_isi_spikes=None,
#                      return_ids=False):

#         # Good units:
#         # isolation distance >= 10
#         # coverage >= 0.9
#         # ISI < 1 %

#         total_n_units = self.spike_data['klustaID'].shape[0]
#         selected_unit_indices = []

#         # if area is not None:
#         #     if isinstance(area, str):
#         #         area = [area]
#         #     mask = np.isin(self.spike_data['area'], area)
#         #     sel_trials = self.spike_data.loc[mask, 'trialNum'].tolist()
#         #     selected_trials.append(sel_trials)

#         if area is not None and area != 'all':
#             ind = np.where(self.spike_data['area'] == area)[0]
#             selected_unit_indices.append(ind)
#             perc = 100*(len(ind) / total_n_units)
#             print('> Selecting units of area {}'
#                   '\n ---> {} out of {} units selected '
#                   '({:.1f}%)'.format(area, len(ind), total_n_units, perc))

#         if layer is not None:
#             ind = np.where(self.spike_data['layer'] == layer)[0]
#             selected_unit_indices.append(ind)
#             perc = 100*(len(ind) / total_n_units)
#             print('> Selecting units of layer {}'
#                   '\n ---> {} out of {} units selected '
#                   '({:.1f}%)'.format(layer, len(ind), total_n_units, perc))


#         if min_isolation_distance is not None:
#             ind = np.where(self.spike_data['QM_IsolationDistance'] >= min_isolation_distance)[0]
#             selected_unit_indices.append(ind)
#             perc = 100*(len(ind) / total_n_units)
#             print('> Selecting units with minimum isolation distance of {}'
#                   '\n ---> {} out of {} units selected '
#                   '({:.1f}%)'.format(min_isolation_distance, len(ind), total_n_units, perc))

#         if min_coverage is not None:
#             ind = np.where(self.spike_data['coverage'] >= min_coverage)[0]
#             selected_unit_indices.append(ind)
#             perc = 100*(len(ind) / total_n_units)
#             print('> Selecting units with minimum coverage of {}'
#                   '\n ---> {} out of {} units selected '
#                   '({:.1f}%)'.format(min_coverage, len(ind), total_n_units, perc))

#         if max_perc_isi_spikes is not None:
#             perc_isi_spikes = 100 * self.spike_data['QM_ISI_FA']
#             ind = np.where(perc_isi_spikes < max_perc_isi_spikes)[0]
#             selected_unit_indices.append(ind)
#             perc = 100*(len(ind) / total_n_units)
#             print('> Selecting units with less than {}% spikes in the ISI'
#                   '\n ---> {} out of {} units selected '
#                   '({:.1f}%)'.format(max_perc_isi_spikes, len(ind), total_n_units, perc))

#         if len(selected_unit_indices) > 0:
#             selected_final = list(set.intersection(*map(set, selected_unit_indices)))
#         else:
#             selected_final = np.arange(self.spike_data['klustaID'].shape[0]).tolist()

#         perc = 100*(len(selected_final) / total_n_units)
#         print('> Finished unit selection'
#               '\n ---> {} out of {} units selected '
#               '({:.1f}%)'.format(len(selected_final), total_n_units, perc))

#         selected_final.sort()

#         if return_ids:
#             selected_final = self.get_cell_id(selected_final)

#         return selected_final


#     def get_aligned_times(self, trial_numbers, time_before_in_s=1,
#                           time_after_in_s=1, event='stimChange'):

#         sel_trial_ind = np.isin(self.trial_data['trialNum'], trial_numbers)
#         selected_trial_data = self.trial_data.loc[sel_trial_ind, :]

#         time_before_in_us = time_before_in_s * 1e6
#         time_after_in_us = time_after_in_s * 1e6

#         event_t = selected_trial_data[event].tolist()

#         aligned_trial_times = [[s - time_before_in_us, s + time_after_in_us] for
#                                s in event_t]

#         return aligned_trial_times



#     def make_frequency_bands(self, low, mid, high, spacing_low, spacing_high):
#         low_freqs = np.arange(low, mid + 1, spacing_low)
#         high_freqs = np.arange(mid + spacing_high, high + 1, spacing_high)
#         freqs = np.hstack((low_freqs, high_freqs))
#         freq_bands = [[freqs[i], freqs[i+1]] for i in range(freqs.shape[0]-1)]
#         return freq_bands


#     def set_filter_parameters(self, freq_bands=None, transition_width=5,
#                               bandpass_attenuation=60, order=5):
#         self.freq_bands = freq_bands
#         self.transition_width = transition_width
#         self.bandpass_attenuation = bandpass_attenuation
#         # NOTE: the order does not apply to Kaiser filters, where it is
#         # determined based on transition width and attenuation
#         self.filter_order = order


#     @staticmethod
#     def make_kaiser_window(lowcut, highcut, sampling_rate,
#                            bandpass_attenuation,
#                            transition_width):

#         nyquist_frequency = sampling_rate / 2
#         transition_width_normalized = transition_width / nyquist_frequency

#         M, beta = scipy.signal.kaiserord(bandpass_attenuation,
#                                          transition_width_normalized)

#         b = firwin(M, [lowcut, highcut], nyq=nyquist_frequency,
#                    pass_zero=False,
#                    window=('kaiser', beta), scale=False)
#         return b, 1


#     def make_butterworth_window(self, lowcut, highcut, sampling_rate, order):
#         nyquist_frequency = sampling_rate / 2
#         lowcut = lowcut / nyquist_frequency
#         highcut = highcut / nyquist_frequency
#         b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band')
#         return b, a


#     def _bandpass_filter(self, signal, lowcut, highcut, sampling_rate,
#                         bandpass_attenuation, transition_width):

#         b, a = self.make_kaiser_window(lowcut, highcut, sampling_rate,
#                                     bandpass_attenuation, transition_width)
#         assert a == 1
#         y = scipy.signal.filtfilt(b, a, signal)
#         return y, b


#     @staticmethod
#     def get_analytic_signal(filtered_signal):
#         analytic_signal = scipy.signal.hilbert(filtered_signal)
#         return analytic_signal

#     @staticmethod
#     def get_phase(analytic_signal):
#         instantaneous_phase = np.angle(analytic_signal)
#         return instantaneous_phase

#     @staticmethod
#     def get_energy(analytic_signal):
#         energy = np.square(np.abs(analytic_signal))
#         return energy

#     def get_lfp_signal_from_channel_index(self, channel_index):
#         return self.lfp_data['signal'][channel_index, :]

#     def get_lfp_channel_index_from_channel_id(self, channel_id):
#         ind = np.argwhere(self.channel_id == channel_id)[0]
#         assert ind.shape[0] == 1
#         return ind[0]

#     def bandpass_filter(self, channel_id):

#         print('\nBandpass filtering channel {}'.format(channel_id))
#         channel_index = self.get_lfp_channel_index_from_channel_id(channel_id)

#         print('Bandpass filtering parameters: \n - {} Hz transition width \n - {} dB '
#               'attenuation'.format(self.transition_width, self.bandpass_attenuation))
#         signal = self.get_lfp_signal_from_channel_index(channel_index)

#         filtered_signal = np.zeros([len(self.freq_bands), signal.shape[0]])
#         phase = np.zeros([len(self.freq_bands), signal.shape[0]])
#         energy = np.zeros([len(self.freq_bands), signal.shape[0]])

#         self.filters = {}

#         for j, (lowcut, highcut) in enumerate(self.freq_bands):

#             print(' -> Filtering between {} and {} Hz'.format(lowcut, highcut))
#             y, b = self._bandpass_filter(signal, lowcut, highcut, self.sampling_rate,
#                                       self.bandpass_attenuation, self.transition_width)
#             filtered_signal[j, :] = y
#             self.filters[(lowcut, highcut)] = b

#             analytic_signal = self.get_analytic_signal(y)

#             p = self.get_phase(analytic_signal)
#             phase[j, :] = p

#             e = self.get_energy(analytic_signal)
#             energy[j, :] = e

#         if hasattr(self, 'filtered_lfp'):
#             self.filtered_lfp[channel_id] = filtered_signal
#         else:
#             self.filtered_lfp = {}
#             self.filtered_lfp[channel_id] = filtered_signal

#         if hasattr(self, 'lfp_phase'):
#             self.lfp_phase[channel_id] = phase
#         else:
#             self.lfp_phase = {}
#             self.lfp_phase[channel_id] = phase

#         if hasattr(self, 'lfp_energy'):
#             self.lfp_energy[channel_id] = energy
#         else:
#             self.lfp_energy = {}
#             self.lfp_energy[channel_id] = energy


#         #return self.filtered_lfp, self.lfp_phase, self.lfp_energy




#     def make_filters(self, filter_type='kaiser'):
#         """
#         This is a shortcut to generate all the filters, useful if we want
#         to plot the impulse responses.
#         """
#         self.filter_type = filter_type
#         self.filters = {}
#         for j, (lowcut, highcut) in enumerate(self.freq_bands):
#             if filter_type == 'kaiser':
#                 b, a = self.make_kaiser_window(lowcut, highcut, self.sampling_rate,
#                                             self.bandpass_attenuation, self.transition_width)
#             elif filter_type == 'butterworth':
#                 b, a = self.make_butterworth_window(lowcut, highcut, self.sampling_rate,
#                                                     self.filter_order)
#             self.filters[(lowcut, highcut)] = [b, a]


#     def plot_frequency_response_filters(self):
#         freq_bands = list(self.filters.keys())
#         f, ax = plt.subplots(1, 1, figsize=[5, 3])
#         for (lowcut, highcut) in freq_bands:
#             b, a = self.filters[(lowcut, highcut)]
#             w, h = scipy.signal.freqz(b, a, worN=2000)
#             ax.plot((self.sampling_rate * 0.5 / np.pi) * w, abs(h),
#                     label='{}-{} Hz'.format(lowcut, highcut))
#         ax.set_xlim([0, freq_bands[-1][1] + 5])
#         ax.set_xlabel('Frequency (Hz)')
#         ax.set_ylabel('Gain')
#         ax.legend(bbox_to_anchor=(1.04, 0.5), loc='center left',
#                   frameon=False, title='')
#         filter_type = self.filter_type.capitalize()
#         ax.set_title('{} filters\nBandpass attenuation: {} dB'
#                      '\nTransition width: {} Hz'.format(filter_type,
#                                                         self.bandpass_attenuation,
#                                                         self.transition_width),
#                      fontsize=11)
#         sns.despine()
#         plt.tight_layout(rect=(0, 0, 0.7, 1))
#         return f



#     def plot_impulse_response_filters(self, artifact_attenuation=1000,
#                                       n=12000):

#         """
#         We plot the impulse response to all the filters.
#         We determine a range where filter artifacts can occur as follows:
#         we look at the maximum amplitude of the impulse response (across all
#         filters), then we look for a time point after which the maximum
#         impulse response (across filters) is less than the maximum amplitude
#         divided by artifact_attenuation. That is, for time points outside
#         the artifact range, the effect of the filter artifact is less than
#         e.g. 1 thousandth of the maximum impulse response. Setting the
#         artifact_attenuation parameter allows more or less conservative
#         estimation of 'safe' time ranges.
#         """

#         imp = scipy.signal.unit_impulse(n, 'mid')

#         responses = []
#         f, ax = plt.subplots(1, 1, figsize=[5, 3])
#         for (lowcut, highcut) in self.filters.keys():
#             b, a = self.filters[(lowcut, highcut)]
#             response = scipy.signal.filtfilt(b, a, imp)
#             responses.append(response)
#             ax.plot(np.arange(-n // 2, n // 2), response,
#                     label='{}-{} Hz'.format(lowcut, highcut))

#         max_resp = np.vstack(responses).max(axis=0)
#         max_resp = max_resp[max_resp.shape[0] // 2:]
#         future_max = [max_resp[i:].max() for i in range(max_resp.shape[0])]
#         ind = np.argwhere(future_max < max_resp.max() / artifact_attenuation)[0]

#         time = ind[0] / self.sampling_rate

#         ax.axvline(ind, c='grey', ls='--')
#         ax.axvline(-ind, c='grey', ls='--')
#         ax.set_xlabel('Time [ms]')
#         ax.set_ylabel('Amplitude')
#         ax.set_xlim([-ind - 50, ind + 50])
#         ax.legend(bbox_to_anchor=(1.04, 0.5), loc='center left',
#                   frameon=False, title='')
#         filter_type = self.filter_type.capitalize()
#         ax.set_title('{} filters\nBandpass attenuation: {} dB'
#                      '\nTransition width: {} Hz\n'
#                      'Artifact range -{} to {} s'.format(filter_type,
#                                                         self.bandpass_attenuation,
#                                                         self.transition_width,
#                                                         time, time), fontsize=11)
#         sns.despine()
#         plt.tight_layout(rect=(0, 0, 0.7, 1))
#         return f


#     def bin_spikes(self, binsize_in_ms, t_start_in_us=None, t_stop_in_us=None,
#                    sliding_window=False, slide_by_in_ms=None):

#         """
#         Wrapper for the spike binning.

#         Relies on two essentially static methods which work fully in
#         microseconds. Here the binsize and potentially the sliding window
#         are passed in milliseconds which is more intuitive.

#         t_start_in_us and t_stop_in_us can be integers OR quantities (we check for it)
#         but binsize and slide_by should be integers

#         Returns an array of binned spikes and an array of the centers
#         of the bins. The bin centers can be used to interpolate phase and
#         energy of the lfp.
#         """

#         # make everything into quantities
#         if t_start_in_us is None:
#             # session t_start and t_stop are always in microseconds
#             t_start_in_us = self.session_t_start
#         else:
#             try:
#                 t_start_in_us.rescale(pq.us)
#             except AttributeError:
#                 t_start_in_us = t_start_in_us * pq.us

#         if t_stop_in_us is None:
#             t_stop_in_us = self.session_t_stop
#         else:
#             try:
#                 t_stop_in_us.rescale(pq.us)
#             except AttributeError:
#                 t_stop_in_us = t_stop_in_us * pq.us

#         binsize_in_us = binsize_in_ms * 1000 * pq.us

#         if slide_by_in_ms is not None:
#             slide_by_in_us = slide_by_in_ms * 1000 * pq.us


#         # generate spike trains
#         spiketrains = self._make_spiketrains(t_start_in_us, t_stop_in_us)

#         # bin spikes without sliding window
#         if not sliding_window:
#             binned_spikes, spike_bin_centers = self._bin_spikes(spiketrains,
#                                                                 binsize_in_us=binsize_in_us,
#                                                                 t_start_in_us=t_start_in_us,
#                                                                 t_stop_in_us=t_stop_in_us)

#         # bin spikes with sliding window
#         if sliding_window:
#             binned_spikes, spike_bin_centers = self._bin_spikes_overlapping(spiketrains,
#                                                     binsize_in_us=binsize_in_us,
#                                                     slide_by_in_us=slide_by_in_us,
#                                                     t_start_in_us=t_start_in_us,
#                                                     t_stop_in_us=t_stop_in_us)

#         return binned_spikes, spike_bin_centers


#     def _make_spiketrains(self, t_start_in_us, t_stop_in_us):
#         """
#         Input times should have a quantity
#         """
#         spiketrains = []
#         for k in range(self.spike_time_stamps.shape[0]):
#             spike_times = self.spike_time_stamps[k]
#             spike_times = spike_times[spike_times >= t_start_in_us]
#             spike_times = spike_times[spike_times <= t_stop_in_us]
#             train = neo.SpikeTrain(times=spike_times * pq.us,
#                                    t_start=t_start_in_us,
#                                    t_stop=t_stop_in_us)
#             spiketrains.append(train)
#         return spiketrains


#     def _bin_spikes(self, spiketrains, binsize_in_us, t_start_in_us,
#                     t_stop_in_us):
#         """
#         Input times should have a quantity
#         """
#         bs = elephant.conversion.BinnedSpikeTrain(spiketrains,
#                                                   binsize=binsize_in_us,
#                                                   t_start=t_start_in_us,
#                                                   t_stop=t_stop_in_us)

#         n_spikes = np.sum([t.times.__len__() for t in spiketrains])
#         n_spikes_binned = bs.to_array().sum()
#         if n_spikes != n_spikes_binned:
#             warnings.warn('The number of binned spikes is different than '
#                           'the number of original spikes')

#         binned_spikes = bs.to_array()
#         spike_bin_centers = bs.bin_centers.rescale(pq.us)

#         return binned_spikes, spike_bin_centers


#     def _bin_spikes_overlapping(self, spiketrains, binsize_in_us, slide_by_in_us,
#                                 t_start_in_us, t_stop_in_us):
#         """
#         Input times need not be quantities
#         """
#         try:
#             # this needs to be a normal integer for teh list comprehension
#             binsize_in_us = binsize_in_us.item()
#         except:
#             pass

#         left_edges = np.arange(t_start_in_us, t_stop_in_us, slide_by_in_us)
#         bins = [(e, e + binsize_in_us) for e in left_edges]

#         spike_bin_centers = [(b1 + b2) / 2 for b1, b2 in bins]

#         # make sure that all the bin centers are within the event
#         bins = [b for b in bins if (b[0] + b[1]) / 2 <= (t_stop_in_us)]
#         # make sure that bins are fully within
#         bins = [b for b in bins if b[1] <= t_stop_in_us]


#         # prepare the bin centers (to return)
#         spike_bin_centers = [(b1 + b2) / 2 for b1, b2 in bins]

#         num_bins = len(bins)  # Number of bins
#         num_neurons = len(spiketrains)  # Number of neurons
#         binned_spikes = np.empty([num_neurons, num_bins])

#         for i, train in enumerate(spiketrains):
#             # this is just for safety
#             spike_times = train.times.rescale(pq.us)
#             for t, bin in enumerate(bins):
#                 binned_spikes[i, t] = np.histogram(spike_times, bin)[0]

#         binned_spikes = binned_spikes.astype(int)
#         spike_bin_centers = np.array(spike_bin_centers) * pq.us

#         return binned_spikes, spike_bin_centers

#     def bin_spikes_per_trial(self, binsize_in_ms, trial_times,
#                              sliding_window=False, slide_by_in_ms=None):

#         try:
#             trial_times.units
#         except AttributeError:
#             print('Trial times have no units, assuming microseconds (us)')
#             trial_times = trial_times * pq.us

#         binned_spikes, spike_bin_centers = [], []

#         for i, (t_start, t_stop) in enumerate(trial_times):

#             bs, bc = self.bin_spikes(binsize_in_ms, t_start_in_us=t_start,
#                                      t_stop_in_us=t_stop, sliding_window=sliding_window,
#                                      slide_by_in_ms=slide_by_in_ms)
#             binned_spikes.append(bs)
#             spike_bin_centers.append(bc)

#         return binned_spikes, spike_bin_centers


#     def interpolate_phase_and_energy(self, spike_bin_centers):

#         """
#         Interpolate phase and energy (for the case where spike_bin_centers
#         is just a simple array or list)
#         """

#         interp_lfp_phase, interp_lfp_energy = {}, {}

#         for channel_id in self.lfp_phase.keys():

#             lfp_phase = self.lfp_phase[channel_id]
#             lfp_energy = self.lfp_energy[channel_id]

#             interp_lfp_phase[channel_id] = np.zeros_like(lfp_phase)
#             interp_lfp_energy[channel_id] = np.zeros_like(lfp_energy)

#             for j in range(lfp_phase.shape[0]):
#                 interp_lfp_phase[channel_id][j, :] = self.interpolate(lfp_phase[j, :],
#                                                                     self.lfp_times,
#                                                                     spike_bin_centers)

#                 interp_lfp_energy[channel_id][j, :] = self.interpolate(lfp_energy[j, :],
#                                                                     self.lfp_times,
#                                                                     spike_bin_centers)
#         return interp_lfp_phase, interp_lfp_energy


#     def interpolate_phase_and_energy_per_trial(self, spike_bin_centers):

#         """
#         Interpolate phase and energy (for the case where spike_bin_centers is
#         a list of arrays indicating, with one array per trial)
#         """

#         interp_lfp_phase, interp_lfp_energy = {}, {}

#         for channel_id in self.lfp_phase.keys():

#             lfp_phase = self.lfp_phase[channel_id]
#             lfp_energy = self.lfp_energy[channel_id]

#             interp_lfp_phase[channel_id] = []
#             interp_lfp_energy[channel_id] = []

#             n_bands = lfp_phase.shape[0]

#             for i, trial_bin_centers in enumerate(spike_bin_centers):

#                 interp_phase_trial = np.zeros([n_bands, len(trial_bin_centers)])
#                 interp_energy_trial = np.zeros([n_bands, len(trial_bin_centers)])

#                 for j in range(lfp_phase.shape[0]):
#                     interp_phase_trial[j, :] = self.interpolate(lfp_phase[j, :],
#                                                                self.lfp_times,
#                                                                trial_bin_centers)

#                     interp_energy_trial[j, :] = self.interpolate(lfp_energy[j, :],
#                                                                self.lfp_times,
#                                                                trial_bin_centers)

#                 interp_lfp_phase[channel_id].append(interp_phase_trial)
#                 interp_lfp_energy[channel_id].append(interp_energy_trial)

#         return interp_lfp_phase, interp_lfp_energy


#     def interpolate_raw_lfp_per_trial(self, spike_bin_centers):
#         """
#         Returns a list of arrays of shape [n_channels, n_time_point_per_trial]
#         """

#         interp_lfp = []

#         #loop over trials
#         for i, trial_bin_centers in enumerate(spike_bin_centers):
#             interp_lfp_trial = np.zeros([len(self.channel_id), len(trial_bin_centers)])

#             # loop over channels
#             for j, channel_id in enumerate(self.channel_id):

#                 channel_index = self.get_lfp_channel_index_from_channel_id(channel_id)
#                 lfp = self.get_lfp_signal_from_channel_index(channel_index)
#                 interp_lfp_trial[j, :] = self.interpolate(lfp, self.lfp_times,
#                                                     trial_bin_centers)
#             interp_lfp.append(interp_lfp_trial)

#         return interp_lfp



#     def interpolate_filtered_lfp_per_trial(self, spike_bin_centers):

#         """
#         Interpolate the filtered lfp (for the case where spike_bin_centers is
#         a list of arrays indicating, with one array per trial)
#         """

#         interp_filtered_lfp = {}

#         for channel_id in self.filtered_lfp.keys():

#             filtered_lfp = self.filtered_lfp[channel_id]

#             interp_filtered_lfp[channel_id] = []

#             n_bands = filtered_lfp.shape[0]

#             for i, trial_bin_centers in enumerate(spike_bin_centers):

#                 interp_filtered_lfp_trial = np.zeros([n_bands, len(trial_bin_centers)])

#                 for j in range(filtered_lfp.shape[0]):
#                     interp_filtered_lfp_trial[j, :] = self.interpolate(filtered_lfp[j, :],
#                                                                self.lfp_times,
#                                                                trial_bin_centers)

#                 interp_filtered_lfp[channel_id].append(interp_filtered_lfp_trial)

#         return interp_filtered_lfp



#     @staticmethod
#     def interpolate(signal, times, new_times):
#         return np.interp(new_times, times, signal)

#     @staticmethod
#     def interpolate_angle(signal, times, new_times):
#         return np.interp(new_times, times, signal)


#     def discretize_phase(self, phase, n_bins):

#         """
#         Discretize phase into a given number of bins (between -pi and pi).
#         Handles both the case in which phase is a single lfp array
#         and the case in which it is a list of arrays (one array per trial).
#         """

#         full_range = 2 * np.pi
#         delta = full_range * 0.01
#         bin_edges = np.linspace(-np.pi - delta, np.pi + delta, n_bins + 1)
#         print('Discretizing phase in {} bins: \n- '
#               'Edges: {}'.format(n_bins, bin_edges))

#         disc_phase = {}

#         for channel_id in phase.keys():

#             X = phase[channel_id]

#             if isinstance(X, np.ndarray) and X.ndim == 2:
#                 Xt = self._discretize_array_fixed_bins(X, bin_edges=bin_edges)

#             elif isinstance(X, list):
#                 Xt = []
#                 for X_trial in X:
#                     X_trial_t = self._discretize_array_fixed_bins(X_trial, bin_edges=bin_edges)
#                     Xt.append(X_trial_t)

#             disc_phase[channel_id] = Xt

#         return disc_phase


#     def discretize_energy(self, energy, n_bins):

#         """
#         Discretize lfp energy into a given number of quantile bins.
#         Handles the following cases:
#         - energy is an array, in which case quantile bins are built on the array
#         - energy is a list of arrays (one array per trial) in which case
#         the arrays are first stacked to obtain a full array, the quantile
#         bins are constructed using the full array, then applied to each
#         individual array.
#         """
#         disc_energy = {}

#         for channel_id in energy.keys():
#             X = energy[channel_id]

#             print('Rescaling power - so that binning works')

#             kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
#                                      strategy='quantile')

#             if isinstance(X, np.ndarray) and X.ndim == 2:
#                 X = X * 1e9
#                 kbins.fit(X.T)
#                 Xt = kbins.transform(X.T)

#             elif isinstance(X, list):
#                 X = [tr * 1e9 for tr in X]
#                 Xt = []
#                 full_array = np.hstack(X)
#                 kbins.fit(full_array.T)

#                 for X_trial in X:
#                     X_trial_t = kbins.transform(X_trial.T).T
#                     Xt.append(X_trial_t)

#             disc_energy[channel_id] = Xt

#         return disc_energy


#     def _discretize_array_fixed_bins(self, X, bin_edges):
#         """
#         Array is of the shape n_features x n_samples.
#         This is the function that actually does the discretizing given
#         the bin edges.
#         """
#         Xt = np.zeros_like(X, dtype=int)
#         for k, row in enumerate(X):
#             Xt[k, :] = pd.cut(row, bins=bin_edges, right=False,
#                               include_lowest=True).codes
#         return Xt


#     def make_target(self, trial_numbers, target_name,
#                     n_time_bins_per_trial=None, coarse=True):

#         """
#         Return a target variable for the select trials. If n_time_per_bins
#         is passed, the target is repeated so that for every time point
#         you get a target value.
#         """
#         sel_trial_ind = np.isin(self.trial_data['trialNum'], trial_numbers)
#         target = self.trial_data.loc[sel_trial_ind, target_name].tolist()

#         if n_time_bins_per_trial is not None:
#             target = np.repeat(target, n_time_bins_per_trial)
#         else:
#             target = np.array(target)

#         joinable_targets = ['visualOriPostChangeNorm', 'audioFreqPostChangeNorm']
#         if np.isin(target_name, joinable_targets).sum() > 0 and coarse:
#             print('Joining labels (1, 2) and (3, 4)')
#             target[np.logical_or(target == 1, target == 2)] = 0
#             target[np.logical_or(target == 3, target == 4)] = 1

#         print('Returning target {} with {} elements and value '
#               'counts: \n{}'.format(target_name, target.shape[0], pd.value_counts(target)))

#         return target


#     def get_equispaced_channel_indices(self, total_n_channels, n_channels,
#                                        exclude_first_and_last=True):

#         if exclude_first_and_last:
#             ind = np.round(np.linspace(0, total_n_channels-1, n_channels+2)).astype(int)
#             ind = ind[1:-1]
#         else:
#             ind = np.round(np.linspace(0, total_n_channels-1, n_channels)).astype(int)
#         return ind

#     def get_nearest_channel(self, cell_id, channel_ids):

#         channel_id_cell = self.get_channel_id_of_cell(cell_id)
#         xy_cell = self.get_xy_position_of_channel(channel_id_cell)

#         channel_ids = [c for c in channel_ids if c != channel_id_cell]

#         xy_channels = []
#         for ch_id in channel_ids:
#             xy_channels.append(self.get_xy_position_of_channel(ch_id))
#         xy_channels = np.vstack(xy_channels)

#         distances = np.linalg.norm(xy_channels - xy_cell, axis=1)

#         closest_ch_id = channel_ids[np.argmin(distances)]

#         d1 = self.compute_cell_to_channel_distance(cell_id, closest_ch_id)
#         d2 = np.min(distances)
#         assert d1 == d2

#         print('For cell {} recorded on channel {} selecting closest channel {}'
#               '\n - distance: {} micrometers'.format(cell_id, channel_id_cell,
#                     closest_ch_id, d1))

#         return closest_ch_id


#     def compute_cell_to_channel_distance(self, cell_id, channel_id):

#         xy_cell = self.get_xy_position_of_cell(cell_id)
#         xy_chan = self.get_xy_position_of_channel(channel_id)

#         return np.linalg.norm(xy_cell - xy_chan)


#     def get_channel_id_of_cell(self, cell_id):
#         cell_ind = np.argwhere(self.cell_id == cell_id)[0]
#         channel_id = self.spike_data['channel_ID'][cell_ind]
#         return channel_id

#     def get_xy_position_of_channel(self, channel_id):
#         channel_ind = np.argwhere(self.channel_id == channel_id)[0]
#         x = self.lfp_data['ChannelX'][channel_ind]
#         y = self.lfp_data['ChannelY'][channel_ind]
#         return np.array([x, y])

#     def get_xy_position_of_cell(self, cell_id):
#         cell_ind = np.argwhere(self.cell_id == cell_id)[0]
#         x = self.lfp_data['ChannelX'][cell_ind]
#         y = self.lfp_data['ChannelY'][cell_ind]
#         return  np.array([x, y])


#     def get_equispaced_channel_ids(self, area):
#         try:
#             area_ind = np.where(self.lfp_data['area'] == area)[0]
#         except KeyError:
#             area_ind = np.where(self.lfp_data['Area'] == area)[0]

#         pass



#     def select_channels(self, area, layer=None, in_the_middle=False, q=0.1):

#         selected_channel_inds = []
#         try:
#             ind = np.where(self.lfp_data['area'] == area)[0]
#         except KeyError:
#             ind = np.where(self.lfp_data['Area'] == area)[0]

#         selected_channel_inds.append(ind)

#         if layer is not None:
#             ind = np.where(self.lfp_data['layer'] == layer)[0]
#             selected_channel_inds.append(ind)

#         if len(ind) == 0:
#             print('There are no channels which match the criteria!')
#             return []

#         if in_the_middle:
#             # the channel depth quantiles are defined only over the channels
#             # of the selected area
#             channel_depth = self.lfp_data['ChannelY'][ind]
#             min_depth = np.quantile(channel_depth, q)
#             max_depth = np.quantile(channel_depth, 1-q)
#             print(' - Restricting to channels with depth between {:.2f} and '
#                   '{:.2f} micrometers'.format(min_depth, max_depth))
#             # the indices are across all channels, so we can intersect them
#             # with the others
#             ind = np.where(np.logical_and(self.lfp_data['ChannelY'] >= min_depth,
#                                           self.lfp_data['ChannelY'] <= max_depth))[0]

#             selected_channel_inds.append(ind)


#         selected_inds = list(set.intersection(*map(set, selected_channel_inds)))

#         selected_ids = [self.channel_id[ind] for ind in selected_inds]

#         return selected_ids



#     def get_random_channel_id(self, area, layer=None, in_the_middle=False,
#                               q=0.1):

#         print('\nRandomly selecting a channel from area {} and '
#               'layer {}'.format(area, layer))

#         selected_ids = self.select_channels(area=area, layer=layer,
#                              in_the_middle=in_the_middle, q=q)
#         selected_id =  np.random.choice(selected_ids)
#         selected_ind = self.get_lfp_channel_index_from_channel_id(selected_id)

#         try:
#             assert self.lfp_data['area'][selected_ind] == area
#         except KeyError:
#             assert self.lfp_data['Area'][selected_ind] == area


#         if layer is not None:
#             assert self.lfp_data['layer'][selected_ind] == layer

#         else:
#             layer = self.lfp_data['layer'][selected_ind]

#         sl_ch_depth = self.lfp_data['ChannelY'][selected_ind]

#         print('\n - Selected channel {}'
#               '\n --- Area : {}'
#               '\n --- Layer : {}'
#               '\n --- Depth : {}'.format(selected_id, area, layer, sl_ch_depth))

#         return selected_id



#     def get_freq_band_index_from_freq_band(self, band):
#         ind = np.where((np.array(self.freq_bands) == band).all(axis=1))[0]
#         return ind


#     def get_data_for_time_trial_plot(self, data, lfp_channel_id=None, freq_band=None,
#                                      audio_post_norm=None,
#                                      visual_post_norm=None,
#                                      auditory_change=None, visual_change=None,
#                                      trial_type=None,
#                                      only_correct_trials=False,
#                                      discretize_phase=False,
#                                      discretize_energy=True,
#                                      n_phase_bins=4, n_energy_bins=4,
#                                      time_before_stim_in_s=1,
#                                      time_after_stim_in_s=1,
#                                      binsize_in_ms=20):
#         """
#         data can be spikes, lfp, phase, energy
#         - if it is spikes, you must pass a unit index
#         - if it is lfp, you must pass a channel id
#         - if it is filtered_lfo, phase or energy,
#         you must pass a channel_id and a frequency band
#         """

#         trial_numbers = self.select_trials(only_correct=only_correct_trials,
#                                            trial_type=trial_type,
#                                            audio_post_norm=audio_post_norm,
#                                            visual_post_norm=visual_post_norm,
#                                            visual_change=visual_change,
#                                            auditory_change=auditory_change)

#         trial_times = self.get_aligned_times(trial_numbers,
#                                              time_before_in_s=time_before_stim_in_s,
#                                              time_after_in_s=time_after_stim_in_s)

#         binned_spikes, spike_bin_centers = self.bin_spikes_per_trial(binsize_in_ms, trial_times)

#         if data == 'lfp':
#             interp_lfp = self.interpolate_raw_lfp_per_trial(spike_bin_centers)
#             ind = self.get_lfp_channel_index_from_channel_id(lfp_channel_id)
#             plot_array = np.vstack([lfp_trial[ind, :] for lfp_trial in
#                                     interp_lfp])


#         if data == 'filtered_lfp':
#             interp_filtered_lfp = self.interpolate_filtered_lfp_per_trial(spike_bin_centers)
#             ind = self.get_freq_band_index_from_freq_band(freq_band)
#             plot_array = np.vstack([filt_lfp_trial[ind, :] for filt_lfp_trial in
#                                     interp_filtered_lfp[lfp_channel_id]])


#         if data == 'phase':
#             interp_phase, interp_energy = self.interpolate_phase_and_energy_per_trial(spike_bin_centers)
#             if discretize_phase:
#                 interp_phase = self.discretize_phase(interp_phase, n_bins=n_phase_bins)

#             ind = self.get_freq_band_index_from_freq_band(freq_band)
#             plot_array = np.vstack([phase_trial[ind, :] for phase_trial in
#                                     interp_phase[lfp_channel_id]])

#         if data == 'energy':
#             interp_phase, interp_energy = self.interpolate_phase_and_energy_per_trial(spike_bin_centers)
#             if discretize_energy:
#                 interp_energy = self.discretize_energy(interp_energy, n_bins=n_energy_bins)

#             ind = self.get_freq_band_index_from_freq_band(freq_band)
#             plot_array = np.vstack([en_trial[ind, :] for en_trial in
#                                     interp_energy[lfp_channel_id]])

#         extent = (-time_before_stim_in_s, time_after_stim_in_s,
#                   plot_array.shape[0] - 0.5, -0.5)

#         return plot_array, extent



#     def slice_lfp_by_time(self, lfp_channel_id, t_start, t_stop):
#         channel_ind = self.get_lfp_channel_index_from_channel_id(lfp_channel_id)
#         signal = self.get_lfp_signal_from_channel_index(channel_ind)
#         slice = signal[(self.lfp_times >= t_start) & (self.lfp_times < t_stop)]
#         return slice


#     def prepare_lfp_for_mne(self, lfp_channel_ids, times):

#         """
#         lfp_channel_ids can be either a single (string) channel id, or a
#         list of channel ids. Times is a list of lists/tuples, corresponding
#         to e.g. trial times. The function returns an array of shape
#         len(times) x len(lfp_channel_ids) x n_times_points, which is the required
#         dimension for MNE time frequency analysis.

#         """

#         if isinstance(lfp_channel_ids, str):
#             lfp_channel_ids = [lfp_channel_ids]

#         channel_data = []
#         for lfp_channel_id in lfp_channel_ids:
#             data = []
#             for t_start, t_stop in times:
#                 trial_signal = self.slice_lfp_by_time(lfp_channel_id,
#                                                       t_start, t_stop)
#                 data.append(trial_signal)
#             data = np.vstack(data)
#             data = data[:, np.newaxis, :]
#             channel_data.append(data)
#         data = np.concatenate(channel_data, axis=1)

#         assert data.shape[0] == len(times)
#         assert data.shape[1] == len(lfp_channel_ids)

#         return data


#     def prepare_data_for_affinewarp_spikedata(self, trial_numbers,
#                                               selected_unit_ind,
#                                               time_before_in_s,
#                                               time_after_in_s):
#         """
#         Assumes spike times are in microseconds
#         Assumes that the align event is stimulus change.
#         Returns spike times in seconds!
#         """

#         trial_times = self.get_aligned_times(trial_numbers,
#                                                 time_before_in_s=time_before_in_s,
#                                                 time_after_in_s=time_after_in_s)
#         stim_ch_time_trial = self.get_stimulus_change_time(trial_numbers)

#         try:
#             trial_times.units
#         except AttributeError:
#             print('Trial times have no units, assuming microseconds (us)')
#             trial_times = trial_times * pq.us

#         stim_ch_time_trial = stim_ch_time_trial * pq.us

#         trial_inds, spiketimes, neurons = [], [], []

#         for trial_ind, (stim_ch_time, (t_start, t_stop)) in enumerate(zip(stim_ch_time_trial,
#                                                                           trial_times)):

#             spiketrains = self._make_spiketrains(t_start, t_stop)

#             for unit_ind, ind in enumerate(selected_unit_ind):
#                 # we use ind to access the correct spiketrain, but SpikeData
#                 # wants indices starting at 0, otherwise if you pass
#                 # indices [2, 3, 4] it thinks you have 5 neurons [0, 1, 2, 3, 4]
#                 # (same for the trials)
#                 spikes = spiketrains[ind].times - stim_ch_time
#                 trial = np.repeat(trial_ind, len(spikes))
#                 neuron = np.repeat(unit_ind, len(spikes))
#                 spiketimes.append(spikes)
#                 trial_inds.append(trial)
#                 neurons.append(neuron)
#         units = spiketimes[0].units
#         trial_inds = np.hstack(trial_inds)
#         spiketimes = (np.asarray(np.hstack(spiketimes)) * units).rescale(pq.s)
#         neurons = np.hstack(neurons)

#         return trial_inds, spiketimes, neurons


#     def get_within_trial_spike_bin_times(self):
#         pass

#     def get_stimulus_change_time(self, trial_numbers):
#         sel_trial_ind = np.isin(self.trial_data['trialNum'], trial_numbers)
#         selected_trial_data = self.trial_data.loc[sel_trial_ind, :]
#         stim_ch_time_trial = selected_trial_data['stimChange'].values
#         return stim_ch_time_trial

#     def get_layer_from_channel_id(self, channel_id):
#         channel_ind = self.get_lfp_channel_index_from_channel_id(channel_id)
#         return self.lfp_data['layer'][channel_ind]



#     def get_cell_id(self, cell_index, shortened_id=False):

#         if isinstance(cell_index, int):
#             cell_id = self.cell_id[cell_index]
#             if shortened_id:
#                 cell_id = cell_id[-3:]
#         else:
#             cell_id = [self.cell_id[i] for i in cell_index]
#             if shortened_id:
#                 cell_id = [c[-3:] for c in cell_id]

#         return cell_id


#     def get_cell_area_from_cell_ind(self, cell_index):
#         return self.spike_data['area'][cell_index]


#     def get_cell_area(self, cell_ids):
#         """
#         Works for both cell_ids as a single string and as an array of strings
#         """
#         cell_ind = np.where(np.isin(self.cell_id, cell_ids))[0]
#         return self.spike_data['area'][cell_ind]


#     def get_cell_layer(self, cell_ids):
#         """
#         Works for both cell_ids as a single string and as an array of strings
#         """
#         cell_ind = np.where(np.isin(self.cell_id, cell_ids))[0]
#         return self.spike_data['Layer'][cell_ind]


#     def get_cell_depth(self, cell_ids):
#         """
#         Works for both cell_ids as a single string and as an array of strings
#         """
#         cell_ind = np.where(np.isin(self.cell_id, cell_ids))[0]
#         return self.spike_data['ChannelY'][cell_ind]


#     def get_percentage_correct(self, trial_type, big_change=False):

#         if isinstance(trial_type, str):
#             trial_type = [trial_type]

#         sel = self.trial_data[np.isin(self.trial_data['trialType'], trial_type)]

#         if big_change:
#             if trial_type == 'X':
#                 sel = sel[sel['visualOriChangeNorm'] == 2]
#             elif trial_type == 'Y':
#                 sel = sel[sel['audioFreqChangeNorm'] == 2]

#         percentage_correct = 100 * sel['correctResponse'].sum() / sel.shape[0]
#         return percentage_correct
