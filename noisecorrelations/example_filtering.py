# -*- coding: utf-8 -*-
"""
Matthijs Oude Lohuis, 2022-2026, Champalimaud Center, Lisbon
"""

#%% ###################################################
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from utils.filter_lib import my_highpass_filter,compute_power_spectra
from utils.explorefigs import *
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\DescriptiveStatisticsSessions\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']]) #GR
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

#%%  Load data properly
ses = sessions[0]
ses.load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            calciumversion='dF',keepraw=True)

#%%% Compute power spectrum on original data:
freqs,psd = compute_power_spectra(ses.sessiondata['fs'][0],data=ses.calciumdata)

#%%  Define parameters for the high-pass filter
cutoff = 0.001  # Cutoff frequency in Hz (adjust if needed)
filtered_data = my_highpass_filter(data = ses.calciumdata, cutoff = cutoff, fs=ses.sessiondata['fs'][0])
freqs,psd_filt_001Hz = compute_power_spectra(ses.sessiondata['fs'][0],data=filtered_data)

#%%  Define parameters for the high-pass filter
cutoff = 0.01  # Cutoff frequency in Hz (adjust if needed)
filtered_data = my_highpass_filter(data = ses.calciumdata, cutoff= cutoff, fs=ses.sessiondata['fs'][0])
freqs,psd_filt_01Hz = compute_power_spectra(ses.sessiondata['fs'][0],data=filtered_data)

#%% Plot the power spectrum of the original data and the filtered data
fig = plt.figure(figsize=(6,4))
plt.plot(freqs, psd, label='Original',c='k')
plt.plot(freqs, psd_filt_001Hz, label='Filtered .001Hz',c='g')
plt.plot(freqs, psd_filt_01Hz, label='Filtered .01Hz',c='b')
plt.text(0.5, 0.72e6, 'Stim. freq. = 0.5Hz', ha='center', va='center')
# plt.loglog()
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.ylim([0,1.2e6])
plt.legend(frameon=False)
plt.title('Power Spectrum dF/F %s' % ses.sessiondata['session_id'][0])
plt.show()
fig.savefig(os.path.join(savedir,'Filtering','PSD_Filtering_%s.png' % ses.sessiondata['session_id'][0]), format = 'png')


#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06'],
                                ['LPE10919','2023_11_06'],
                                ['LPE10919','2023_11_06']]) #GR

session_list        = np.array([['LPE09830','2023_04_10'],
                                ['LPE09830','2023_04_10'],
                                ['LPE09830','2023_04_10']]) #GR

sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

#%%  Load data properly
ises = 0 # without filtering
sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            calciumversion='dF',keepraw=True)
ises = 1 # with filtering
filter_hp = 0.01
sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            calciumversion='dF',keepraw=True,filter_hp=filter_hp)

ises = 2 # with filtering
filter_hp = 0.001
sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            calciumversion='dF',keepraw=True,filter_hp=filter_hp)

#%% 
sessions[0].calciumdata.head(10)
sessions[1].calciumdata.head(10)
sessions[2].calciumdata.head(10)

#%% Raster plot for the three versions:
filtertext = ['NO','01Hz','001Hz']
for ises, ses in enumerate(sessions):
    fig = plot_excerpt(sessions[ises],trialsel=[1,200],plot_neural=True,plot_behavioral=False,neural_version='raster')
    fig.savefig(os.path.join(savedir,'Filtering','ExampleRaster_First200trials_Filter%s_%s.png' % (filtertext[ises],sessions[ises].sessiondata['session_id'][0])), format = 'png')

#%% 
areas           = np.unique(sessions[0].celldata['roi_name'])
labeled         = np.unique(sessions[0].celldata['redcell'])
labeltext       = ['unlabeled', 'labeled',]

nexcells        = 10
example_cells   = np.array([])

for iarea, area in enumerate(areas):
    for ilabel, label in enumerate(labeled):
        idx             = np.where(np.logical_and(sessions[0].celldata['roi_name'] == area, 
                                    sessions[0].celldata['redcell'] == label))[0]
        temp_excells    = np.min((len(idx), nexcells))
        example_cells    = np.append(example_cells, np.random.choice(idx, temp_excells, replace=False))

filtertext = ['NO','01Hz','001Hz']
for ises, ses in enumerate(sessions):
    fig = plot_excerpt(sessions[ises],trialsel=[1,200],plot_neural=True,plot_behavioral=True,neuronsel=example_cells,neural_version='traces')
    fig.savefig(os.path.join(savedir,'Filtering','ExampleTraces_First200trials_Filter%s_%s.png' % (filtertext[ises],sessions[ises].sessiondata['session_id'][0])), format = 'png')

#%% PCA plot of trials for each session:
filtertext = ['NO','01Hz','001Hz']
for ises, ses in enumerate(sessions):
    fig = plot_PCA_gratings(sessions[ises])
    fig.savefig(os.path.join(savedir,'Filtering','PCA_2D_Filter%s_%s.png' % (filtertext[ises],sessions[ises].sessiondata['session_id'][0])), format = 'png')

#%% ########################## Compute signal and noise correlations: ###################################
from corr_lib import compute_signal_noise_correlation
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

#%% heatmap of noise correlations per session
filtertext = ['NO','01Hz','001Hz']
for ises, ses in enumerate(sessions):

    fig = plt.figure(figsize=(8,5))
    plt.imshow(sessions[ises].noise_corr, cmap='coolwarm',
            vmin=np.nanpercentile(sessions[0].noise_corr,10),
            vmax=np.nanpercentile(sessions[0].noise_corr,90))
    plt.title(sessions[ises].sessiondata['session_id'][0] + '- filter ' + filtertext[ises])
    cbar_ax = fig.add_axes([0.8, 0.2, 0.02, 0.2]) # [left, bottom, width, height]
    cbar = plt.colorbar(cax=cbar_ax, orientation='vertical')
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'Filtering','NC_map_Filter%s_%s.png' % (filtertext[ises],sessions[ises].sessiondata['session_id'][0])), format = 'png')

#%%
filtertext = ['NO','01Hz','001Hz']

fig = plt.figure(figsize=(4,3))
for ises, ses in enumerate(sessions):
    plt.hist(sessions[ises].noise_corr.flatten(),bins=np.arange(-1,1,0.01),alpha=1,linewidth=1,label=filtertext[ises],
    histtype='step')
plt.legend()
plt.xlim([-0.6,0.6])
plt.xlabel('Noise Correlation')
plt.ylabel('Count')
plt.title('Noise Correlation Histogram')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Filtering','NoiseCorr_Histogram.png'), format = 'png')

#%%


