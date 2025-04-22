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
from utils.psth import compute_respmat
import scipy
import logging
from utils.filter_lib import my_highpass_filter
logger = logging.getLogger(__name__)

class Session():

    def __init__(self, protocol='', animal_id='', session_id='', verbose=1):
        logger.debug(
            'Initializing Session object for: \n- animal ID: {}' '\n- Session ID: {}\n'.format(animal_id, session_id))
        self.data_folder = os.path.join(
            get_data_folder(), protocol, animal_id, session_id)
        self.verbose = verbose
        self.protocol = protocol
        self.animal_id = animal_id
        self.session_id = session_id
        self.cellfilter = None

    def load_data(self, load_behaviordata=False, load_calciumdata=False, load_videodata=False, 
                  calciumversion='dF', filter_hp=None):

        self.sessiondata_path   = os.path.join(self.data_folder, 'sessiondata.csv')
        self.trialdata_path     = os.path.join(self.data_folder, 'trialdata.csv')
        self.celldata_path      = os.path.join(self.data_folder, 'celldata.csv')
        self.behaviordata_path  = os.path.join(self.data_folder, 'behaviordata.csv')
        self.videodata_path     = os.path.join(self.data_folder, 'videodata.csv')
        self.calciumdata_path   = os.path.join(self.data_folder, '%sdata.csv' % calciumversion) #Calciumversion can be 'dF' or 'deconv'
        self.Ftsdata_path       = os.path.join(self.data_folder, 'Ftsdata.csv')
        self.Fchan2data_path    = os.path.join(self.data_folder, 'Fchan2data.csv')

        assert(os.path.exists(self.sessiondata_path)), 'Could not find data in {}'.format(self.sessiondata_path)

        self.sessiondata  = pd.read_csv(self.sessiondata_path, sep=',', index_col=0)

        if not self.protocol in ['SP','RF']: #These protocols are not trial-based
            self.trialdata  = pd.read_csv(self.trialdata_path, sep=',', index_col=0)
        else:
            self.trialdata = None

        if os.path.exists(self.celldata_path):

            self.celldata  = pd.read_csv(self.celldata_path, sep=',', index_col=0)

            if os.path.exists(os.path.join(self.data_folder,'IMrfdata.csv')):
                IMrfdata = pd.read_csv(os.path.join(self.data_folder,'IMrfdata.csv'), sep=',', index_col=0)
                assert np.shape(IMrfdata)[0]==np.shape(self.celldata)[0], 'dimensions of IMrfdata and celldata do not match for sessions {} and {}'.format(self.animal_id, self.session_id)
                self.celldata = self.celldata.join(IMrfdata,lsuffix='_old')
            
            # get only good cells (selected ROIs by suite2p): #not used anymore, only good cells are saved anyways
            # goodcells               = self.celldata['iscell'] == 1
            # self.celldata           = self.celldata[goodcells].reset_index(drop=True)
        
            if self.cellfilter is not None:
                if isinstance(self.cellfilter, pd.DataFrame):
                    self.cellfilter = self.cellfilter.to_numpy().squeeze()
                assert np.shape(self.celldata)[0]==len(self.cellfilter)
                assert np.array_equal(self.cellfilter, self.cellfilter.astype(bool)), 'Cell filter not boolean'
                self.celldata = self.celldata.iloc[self.cellfilter,:]
                self.celldata.reset_index(drop=True, inplace=True)

        if load_behaviordata:
            self.behaviordata  = pd.read_csv(self.behaviordata_path, sep=',', index_col=0)
        else:
            self.behaviordata = None

        if load_videodata:
            self.videodata = pd.read_csv(
                self.videodata_path, sep=',', index_col=0)
        else:
            self.videodata = None

        if load_calciumdata:

            # print('Loading calcium data at {}'.format(self.calciumdata_path))
            self.calciumdata        = pd.read_csv(self.calciumdata_path, sep=',', index_col=0)
            self.ts_F               = pd.read_csv(self.Ftsdata_path, sep=',', index_col=0).to_numpy().squeeze()
            self.F_chan2            = pd.read_csv(self.Fchan2data_path, sep=',', index_col=0).to_numpy().squeeze()

            if self.cellfilter is not None:
                if isinstance(self.cellfilter, pd.DataFrame):
                    self.cellfilter = self.cellfilter.to_numpy().squeeze()
                assert np.shape(self.calciumdata)[1]==len(self.cellfilter)
                assert np.array_equal(self.cellfilter, self.cellfilter.astype(bool)), 'Cell filter not boolean'
                
                self.calciumdata = self.calciumdata.iloc[:,self.cellfilter]
                # self.celldata = self.celldata.iloc[cellfilter,:]

            if filter_hp is not None and filter_hp > 0:
                self.calciumdata = my_highpass_filter(data = self.calciumdata, cutoff = filter_hp, fs=self.sessiondata['fs'][0])

            assert(np.shape(self.calciumdata)[1]==np.shape(self.celldata)[0]), 'Dimensions of calciumdata and celldata do not match, %s %s' % (np.shape(self.calciumdata)[1],np.shape(self.celldata)[0])

        if load_calciumdata and load_behaviordata:
            # Get interpolated values for behavioral variables at imaging frame rate:
            self.zpos_F = np.interp(x=self.ts_F, xp=self.behaviordata['ts'],
                                    fp=self.behaviordata['zpos'])
            self.runspeed_F = np.interp(x=self.ts_F, xp=self.behaviordata['ts'],
                                        fp=self.behaviordata['runspeed'])
            if 'trialNumber' in self.behaviordata:
                self.trialnum_F = np.interp(x=self.ts_F, xp=self.behaviordata['ts'],
                                            fp=self.behaviordata['trialNumber'])

        # if load_videodata and load_behaviordata:
        #     self.load_videodata['zpos'] = np.interp(x=self.videodata['ts'],xp=self.behaviordata['ts'],
        #                             fp=self.behaviordata['zpos'])

    def reset_label_threshold(self, threshold):
        logger.info(
            'Setting new labeling threshold based on %1.2f overlap' % threshold)
        # self.celldata['redcell'] = self.celldata['redcell_prob']>0.4
        self.celldata['redcell'] = self.celldata['frac_red_in_ROI'] >= threshold
        # print('Need to set cre non flp again\n')
        logger.info('put lower and upper threshold')

        # Add recombinase enzym label to red cells:
        labelareas = ['V1', 'PM']
        for area in labelareas:
            temprecombinase = area + '_recombinase'
            self.celldata.loc[self.celldata['roi_name'] == area,
                              'recombinase'] = self.sessiondata[temprecombinase].to_list()[0]
        # set all nonlabeled cells to 'non'
        self.celldata.loc[self.celldata['redcell'] == 0, 'recombinase'] = 'non'

    def load_respmat(self, load_behaviordata=True, load_calciumdata=True, load_videodata=True, calciumversion='dF',
                    keepraw=False, cellfilter=None,filter_hp=None):
        #combination to load data, then compute the average responses to the stimuli and delete the full data afterwards:

        self.load_data(load_behaviordata=load_behaviordata, load_calciumdata=load_calciumdata,
                       load_videodata=load_videodata,calciumversion=calciumversion,filter_hp=filter_hp)
        
        # if filter_hp is not None and filter_hp > 0:
        #     self.calciumdata = my_highpass_filter(data = self.calciumdata, cutoff = filter_hp, fs=self.sessiondata['fs'][0])

        if self.sessiondata['protocol'][0]=='IM':
            if calciumversion=='deconv':
                t_resp_start = 0
                t_resp_stop = 0.75
            elif calciumversion=='dF':
                t_resp_start = 0.25
                t_resp_stop = 1.25
        elif self.sessiondata['protocol'][0]=='GR':
            if calciumversion=='deconv':
                t_resp_start = 0
                t_resp_stop = 1
            elif calciumversion=='dF':
                t_resp_start = 0.25
                t_resp_stop = 1.5
        elif self.sessiondata['protocol'][0]=='GN':
            if calciumversion=='deconv':
                t_resp_start = 0
                t_resp_stop = 1
            elif calciumversion=='dF':
                t_resp_start = 0.25
                t_resp_stop = 1.5
        else:
            print('skipping mean response calculation for unknown protocol')
            return

        ##############################################################################
        ## Construct trial response matrix:  N neurons by K trials
        self.respmat         = compute_respmat(self.calciumdata, self.ts_F, self.trialdata['tOnset'],
                                        t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean',subtr_baseline=False, label = "response matrix")

        self.respmat_runspeed = compute_respmat(self.behaviordata['runspeed'],
                                        self.behaviordata['ts'], self.trialdata['tOnset'],
                                        t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean', label = "runspeed")

        self.respmat_videome = compute_respmat(self.videodata['motionenergy'],
                                        self.videodata['ts'],self.trialdata['tOnset'],
                                        t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean', label = "motion energy")
        
        self.respmat_videopc = compute_respmat(self.videodata['motionenergy'],
                                        self.videodata['ts'],self.trialdata['tOnset'],
                                        t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean', label = "motion energy")
        
        nPCs = 30
        self.respmat_videopc = compute_respmat(self.videodata[['videoPC_' + str(i) for i in range(nPCs)]],
                                        self.videodata['ts'],self.trialdata['tOnset'],
                                        t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean', label = "motion PCs")
        
        if 'pupil_xpos' in self.videodata:
            self.respmat_pupilx = compute_respmat(self.videodata['pupil_xpos'],
                                                self.videodata['ts'], self.trialdata['tOnset'],
                                                t_resp_start=0, t_resp_stop=t_resp_stop, method='mean', label='pupil x position')
            
        if 'pupil_ypos' in self.videodata:
            self.respmat_pupily = compute_respmat(self.videodata['pupil_ypos'],
                                                self.videodata['ts'], self.trialdata['tOnset'],
                                                t_resp_start=0, t_resp_stop=t_resp_stop, method='mean', label='pupil y position')
        
        if 'pupil_area' in self.videodata:
            self.respmat_pupilarea = compute_respmat(self.videodata['pupil_area'],
                                        self.videodata['ts'],self.trialdata['tOnset'],
                                        t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean', label = "pupil area")
            
            sampling_rate = 1 / np.mean(np.diff(self.ts_F))
            self.respmat_pupilareaderiv = self.lowpass_filter(
                self.respmat_pupilarea, sampling_rate, lowcut=None, highcut=0.7, order=6)
            self.respmat_pupilareaderiv = np.gradient(
                self.respmat_pupilareaderiv, axis=0)
        else: 
            self.respmat_pupilarea = None
            self.respmat_pupilareaderiv = None

        if not keepraw:
            delattr(self, 'calciumdata')
            delattr(self, 'videodata')
            delattr(self, 'behaviordata')

    # Throw respmat_pupilarea through a lowpass filter to create respmat_pupilareaderiv:
    def lowpass_filter(self, respmat, sampling_rate, lowcut=0.1, highcut=0.5, order=10):
        b, a = self._make_butterworth_window(
            lowcut, highcut, sampling_rate, order)
        respmat_filtered = self._replace_nan_with_avg(respmat)
        respmat_filtered = scipy.signal.filtfilt(
            b, a, respmat_filtered, axis=0)
        return respmat_filtered

    def _make_butterworth_window(self, lowcut, highcut, sampling_rate, order):
        nyquist_frequency = sampling_rate / 2
        if lowcut:
            lowcut = lowcut / nyquist_frequency
        if highcut:
            highcut = highcut / nyquist_frequency
        if lowcut and highcut:
            b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band')
        elif lowcut:
            b, a = scipy.signal.butter(order, lowcut, btype='highpass')
        elif highcut:
            b, a = scipy.signal.butter(order, highcut, btype='lowpass')
        else:
            raise ValueError('Either lowcut or highcut must be specified')
        return b, a

    def _replace_nan_with_avg(self, arr):
        nan_indices = np.where(np.isnan(arr))[0]  # Get indices of NaN values

        for i in nan_indices:
            # Handle cases where NaN is at the start or end of the array
            if i == 0:
                arr[i] = arr[i + 1]
            elif i == len(arr) - 1:
                arr[i] = arr[i - 1]
            else:
                # Replace NaN with the average of adjacent values
                arr[i] = np.nanmean([arr[i - 1], arr[i + 1]])

        return arr
