
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d


spon_len = 1500
stim_len = 1280

min_neurons_V1 = 88
min_neurons_V2 = 24

threshold_rate = 1 # Used to exclude silent neurons (in Hz)
threshold_FF = 1e6

stim_period = 160 # in ms

#

session_names = ['105l001p16', '106r001p26', '106r002p70', '107l002p67', '107l003p143']

#

def GetData(ii_session, period='All', path='', exclude_silent=False):
		
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### Open data

	data = scipy.io.loadmat(path+'MatlabData/mat_neural_data/'+session_names[ii_session]+'.mat')['neuralData'][0][0]

	spikesV1 = data['spikeRasters'][:,0] # Format: spikesV1[ii_trial][ii_neuron, ii_time]
	spikesV2 = data['spikeRasters'][:,1]

	stimID = data['stim'][:,0]
	trialID = data['trialId'][:,0]

	stimID_copy = np.copy(stimID)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### Reshape spikes into numpy array: Nneurons, Ntrials, Ntime

	if period == 'All': # Concatenate stimulus and spontaneous

		Ntrials = int(len(spikesV1)/2.)

		spikesV1_array = np.zeros(( spikesV1[0].shape[0], Ntrials, spikesV1[0].shape[1]+spikesV1[1].shape[1] ))
		spikesV2_array = np.zeros(( spikesV2[0].shape[0], Ntrials, spikesV2[0].shape[1]+spikesV2[1].shape[1] ))

		for ii_trial in range(Ntrials):

			spikesV1_array[:,ii_trial,:] = np.concatenate( [ scipy.sparse.csr_matrix.toarray(spikesV1[2*ii_trial]), \
											scipy.sparse.csr_matrix.toarray(spikesV1[2*ii_trial+1]) ], axis=1)

			spikesV2_array[:,ii_trial,:] = np.concatenate( [ scipy.sparse.csr_matrix.toarray(spikesV2[2*ii_trial]), \
											scipy.sparse.csr_matrix.toarray(spikesV2[2*ii_trial+1]) ], axis=1)

	if period == 'AllPre': # Concatenate spontaneous and stimulus

		Ntrials = int(len(spikesV1)/2.)

		spikesV1_array = np.zeros(( spikesV1[0].shape[0], Ntrials, spikesV1[0].shape[1]+spikesV1[1].shape[1] ))
		spikesV2_array = np.zeros(( spikesV2[0].shape[0], Ntrials, spikesV2[0].shape[1]+spikesV2[1].shape[1] ))

		# Invent spontaneous activity for first stimulus

		spikesV1_array[:,0,:] = np.concatenate( [ scipy.sparse.csr_matrix.toarray(spikesV1[1]), \
										scipy.sparse.csr_matrix.toarray(spikesV1[0]) ], axis=1)

		spikesV2_array[:,0,:] = np.concatenate( [ scipy.sparse.csr_matrix.toarray(spikesV2[1]), \
										scipy.sparse.csr_matrix.toarray(spikesV2[0]) ], axis=1)

		# Take the other ones correctly

		for ii_trial in range(Ntrials-1):

			spikesV1_array[:,ii_trial+1,:] = np.concatenate( [ scipy.sparse.csr_matrix.toarray(spikesV1[2*ii_trial+1]), \
											scipy.sparse.csr_matrix.toarray(spikesV1[2*ii_trial+2]) ], axis=1)

			spikesV2_array[:,ii_trial+1,:] = np.concatenate( [ scipy.sparse.csr_matrix.toarray(spikesV2[2*ii_trial+1]), \
											scipy.sparse.csr_matrix.toarray(spikesV2[2*ii_trial+2]) ], axis=1)

	elif period == 'Stim': # Only stimulus

		Ntrials = int(len(spikesV1)/2.)

		spikesV1_array = np.zeros(( spikesV1[0].shape[0], Ntrials, spikesV1[0].shape[1] ))
		spikesV2_array = np.zeros(( spikesV2[0].shape[0], Ntrials, spikesV2[0].shape[1] ))

		for ii_trial in range(Ntrials):

			spikesV1_array[:,ii_trial,:] = scipy.sparse.csr_matrix.toarray(spikesV1[2*ii_trial])
			spikesV2_array[:,ii_trial,:] = scipy.sparse.csr_matrix.toarray(spikesV2[2*ii_trial])


	elif period == 'Spon': # Only spontaneous

		Ntrials = int(len(spikesV1)/2.)

		spikesV1_array = np.zeros(( spikesV1[0].shape[0], Ntrials, spikesV1[1].shape[1] ))
		spikesV2_array = np.zeros(( spikesV2[0].shape[0], Ntrials, spikesV2[1].shape[1] ))

		for ii_trial in range(Ntrials):

			spikesV1_array[:,ii_trial,:] = scipy.sparse.csr_matrix.toarray(spikesV1[2*ii_trial+1])
			spikesV2_array[:,ii_trial,:] = scipy.sparse.csr_matrix.toarray(spikesV2[2*ii_trial+1])


	return spikesV1_array, spikesV2_array, stimID, trialID

#

def SortByStimulus(data1, data2, stimID):

	data1_sorted = np.zeros_like(data1)
	data2_sorted = np.zeros_like(data2)

	Nstimuli = np.max(stimID)
	Nrepet = int(data1.shape[1]/Nstimuli)

	for ii_stimulus in range(Nstimuli):

		data1_sorted[:,ii_stimulus*Nrepet:(ii_stimulus+1)*Nrepet,:] = data1[:,(np.where(stimID==(ii_stimulus+1))[0]/2).astype(int),:]
		data2_sorted[:,ii_stimulus*Nrepet:(ii_stimulus+1)*Nrepet,:] = data2[:,(np.where(stimID==(ii_stimulus+1))[0]/2).astype(int),:]
	
	return data1_sorted, data2_sorted

#

def Bin(spikes, bin_size):

	# Cut final time points you can't bin

	spikes = spikes[:,:,:(spikes.shape[2]-spikes.shape[2]%bin_size)]

	# Bin

	spikes_binned = np.zeros(( spikes.shape[0], spikes.shape[1], int(spikes.shape[2]/float(bin_size)) ))

	for ii_neuron in range(spikes.shape[0]):
		for ii_trial in range(spikes.shape[1]):

			spikes_binned[ii_neuron, ii_trial,:] = np.mean(np.reshape(spikes[ii_neuron, ii_trial,:], (-1,bin_size)), 1)

	return spikes_binned

#

def GaussianFilter(data, std):

	return gaussian_filter1d(data, sigma=std, axis=2)

#

def ExponentialFilter(data, std):

	cutoff = 5 # in units of std

	x = np.arange(cutoff*std)
	fil_x = np.exp(-(x/std))

	return convolve1d(data, fil_x, origin=-int(len(x)/2))

#

def ExcludeSilentNeurons(spikes_array, idx=False):

	# Check for mean firing rate

	threshold = threshold_rate * spikes_array.shape[2]/1e3 # len expressed in ms
	idx_rate = np.where(np.mean(np.sum(spikes_array,2),1)>threshold)[0]
	
	print ('Cut, based on rate: ', len(np.setdiff1d(np.arange(spikes_array.shape[0]), idx_rate)))

	# Check for Fano Factor

	idx_FF = np.where((np.std(np.sum(spikes_array,2),1)**2)/np.mean(np.sum(spikes_array,2),1)<threshold_FF)[0]

	print ('Cut, based on FF: ', len(np.setdiff1d(np.arange(spikes_array.shape[0]), idx_FF)))

	if idx: return spikes_array[np.intersect1d(idx_rate, idx_FF),:,:], np.intersect1d(idx_rate, idx_FF)
	else: return spikes_array[np.intersect1d(idx_rate, idx_FF),:,:]
