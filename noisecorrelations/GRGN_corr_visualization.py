# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2022-2026, Champalimaud Center, Lisbon
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from loaddata.session_info import filter_sessions,load_sessions
from utils.plotting_style import * #get all the fixed color schemes
from utils.plot_lib import shaded_error
from utils.corr_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\DescriptiveStatisticsSessions\\')

#%% #############################################################################
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10919','2023_11_06']]) #GR
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

#%% 
session_list        = np.array([['LPE11998','2024_05_02']]) #GN
sessions,nSessions   = load_sessions(protocol = 'GN',session_list=session_list)


#%%  Load data properly:                      
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s

for ises in range(nSessions):
    # Construct time tensor: 3D 'matrix' of K trials by N neurons by T time bins
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=True)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                    t_pre, t_post, binsize,method='nearby')
    sessions[ises] = compute_trace_correlation([sessions[ises]],binwidth=0.5,uppertriangular=False,filtersig=False)[0]

#%%
def compute_power_spectra(fs,data):
    """
    Compute the power spectrum of the dF/F signal for each neuron in a session.
    Parameters
    ----------
    fs : float
        sampling frequency in Hz
    data : pandas DataFrame
        The dF/F data to compute the power spectrum from. The columns are the
        different neurons and the rows are the time bins.
    Returns
    -------
    freqs : numpy array
        The frequencies at which the power spectrum was computed.
    power_spectra : numpy array
        The power spectrum averaged across all neurons
    """
    T = data.shape[0]
    power_spectra = []
    for neuron in data.columns:
        fft_vals = np.fft.fft(data[neuron])
        power_spectrum = np.abs(fft_vals)**2
        power_spectra.append(power_spectrum)
    power_spectra = np.array(power_spectra)
    avg_power_spectrum = np.mean(power_spectra, axis=0)
    freqs = np.fft.fftfreq(T, d=1/fs)
    freqs_out = freqs[freqs >= 0]
    psd = avg_power_spectrum[freqs >= 0]
    return freqs_out,psd

#%%% 
freqs,psd = compute_power_spectra(ses.sessiondata['fs'][0],data=ses.calciumdata)


#%% 
from scipy.signal import butter, filtfilt
def my_highpass_filter(data, cutoff, fs, order=2): # Define a high-pass filter function
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False)  # High-pass filter
    y = filtfilt(b, a, data, axis=0)  # Apply the filter with zero-phase distortion
    return y


#%%  Define parameters for the high-pass filter
cutoff = 0.001  # Cutoff frequency in Hz (adjust if needed)
# Apply the high-pass filter to the data
filtered_data = my_highpass_filter(data =ses.calciumdata.values, cutoff= cutoff, fs=ses.sessiondata['fs'][0])

# Store the filtered data in a new DataFrame (preserving the original structure)
filtered_data = pd.DataFrame(filtered_data, 
                   columns=ses.calciumdata.columns, 
                   index=ses.calciumdata.index)

# Optionally compute power spectra for the filtered data
freqs,psd_filt_001Hz = compute_power_spectra(ses.sessiondata['fs'][0],data=filtered_data)

#%%  Define parameters for the high-pass filter
cutoff = 0.01  # Cutoff frequency in Hz (adjust if needed)
# Apply the high-pass filter to the data
filtered_data = my_highpass_filter(data = ses.calciumdata.values, cutoff= cutoff, fs=ses.sessiondata['fs'][0])

# Store the filtered data in a new DataFrame (preserving the original structure)
filtered_data = pd.DataFrame(filtered_data, 
                   columns=ses.calciumdata.columns, 
                   index=ses.calciumdata.index)

# Optionally compute power spectra for the filtered data
freqs,psd_filt_01Hz = compute_power_spectra(ses.sessiondata['fs'][0],data=filtered_data)

#%% 
# Plot the power spectrum focusing on low frequencies
fig = plt.figure(figsize=(6,4))
plt.plot(freqs, psd, label='Original',c='k')
plt.plot(freqs, psd_filt_001Hz, label='Filtered .01Hz',c='g')
plt.plot(freqs, psd_filt_01Hz, label='Filtered .1Hz',c='b')
plt.text(0.5, 0.72e6, 'Stim. freq. = 0.5Hz', ha='center', va='center')
plt.loglog()
# plt.xscale('log')
# plt.yscale('linear')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.ylim([0,1e6])
plt.legend(frameon=False)
plt.title('Power Spectrum dF/F %s' % ses.sessiondata['session_id'][0])
plt.show()
fig.savefig(os.path.join(savedir,'PSD_Filtering_loglog_%s.png' % ses.sessiondata['session_id'][0]), format = 'png')

#%% 
for ises in range(nSessions):
    sessions[ises] = high_pass_filter(sessions[ises],cutoff=0.01)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ########################### Compute tuning metrics: ###################################
for ises in range(nSessions):
    if sessions[ises].sessiondata['protocol'].isin(['GR'])[0]:
        conditions = sessions[ises].trialdata['Orientation']
        sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,conditions,
                                                        tuning_metric='tuning_var')
        sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'])
    elif sessions[ises].sessiondata['protocol'].isin(['GN'])[0]:
        resp_mean,resp_res      = mean_resp_gn(sessions[ises],filter_stationary=False)
        sessions[ises].celldata['tuning_var'] = compute_tuning_var(resp_mat=sessions[ises].respmat,resp_res=resp_res)
        # sessions[ises].celldata['pref_ori'],sessions[ises].celldata['pref_speed'] = get_pref_orispeed(resp_mean,oris,speeds)

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method='PCA',remove_rank=1)


#%% Give redcells a string label
redcelllabels = np.array(['unl','lab'])
for ses in sessions:
    ses.celldata['labeled'] = ses.celldata['redcell'].astype(int).apply(lambda x: redcelllabels[x])
    ses.celldata['area_label'] = ses.celldata['roi_name'] + ses.celldata['labeled']
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)


#%%
pair_corr = np.empty((0,4),dtype=object)
for ises,ses in enumerate(sessions):
    corr_extreme = np.logical_or(ses.noise_corr<-0.1,ses.noise_corr>0.3)
    V1_idx = np.all((ses.celldata['roi_name']=='V1',ses.celldata['noise_level']<0.1,ses.celldata['tuning_var']>0.025),axis=0)
    PM_idx = np.all((ses.celldata['roi_name']=='PM',ses.celldata['noise_level']<0.1,ses.celldata['tuning_var']>0.025),axis=0)
    row_idx,col_idx = np.where(np.all((np.outer(V1_idx,PM_idx),corr_extreme),axis=0))

    pair_corr = np.vstack((pair_corr,np.array([ses.celldata['cell_id'][row_idx],
                                                ses.celldata['cell_id'][col_idx],
                                                ses.noise_corr[row_idx,col_idx],
                                                np.repeat(ses.sessiondata['session_id'],len(row_idx))]).T))

print(pair_corr.shape)

#%% #####################################
#Show some traces and some stimuli to see responses:
example_cells   = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM
example_cells   = [3,100,58,62,70]
fig             = plot_excerpt(sessions[ises],plot_behavioral=False,trialsel=[1,len(sessions[ises].trialdata)],neuronsel=example_cells)

#%% Show average trace

ises = 1

#%% Figure of complete average response for dF/F and deconv: 
for ises,ses in enumerate(sessions):
    fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(6,3))
    area_labelpairs = np.flip(np.unique(sessions[ises].celldata['area_label']))
    clrs_area_labels = get_clr_area_labeled(area_labelpairs)

    ises = 0
    for i,area_label in enumerate(area_labelpairs):
        idx = sessions[ises].celldata['area_label']==area_label
        # idx = np.all((sessions[ises].celldata['area_label']==area_label,sessions[ises].celldata['tuning_var']>0.05),axis=0)
        data = sessions[ises].tensor[idx,:,:].mean(axis=0).mean(axis=0)
        ax[int(i>1)].plot(t_axis,data,alpha=1,label=area_label,linewidth=2,color=clrs_area_labels[i])
    ax[0].legend(frameon=False)
    ax[1].legend(frameon=False)
    ax[0].set_ylabel('dF/F')
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_ylim([0,0.5])
    fig.suptitle(ses.sessiondata['session_id'][0])
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'Averaged_response_dF_tuned_%s.png' % ses.sessiondata['session_id'][0]), format = 'png')



#%% Show calcium traces averaged across V1lab, V1unl, PMlab,PMunl populations

for ises,ses in enumerate(sessions):
    fig,ax = plt.subplots(1,1,sharex=True,figsize=(10,3))
    for i,area_label in enumerate(area_labelpairs):
        data = ses.calciumdata.iloc[:,np.where(ses.celldata['area_label']==area_label)[0]]
        ax.plot(np.linspace(0,1,len(data)),data.mean(axis=1)+i,alpha=1,label=area_label,linewidth=0.5,color=clrs_area_labels[i])
        ax.axhline(data.mean(axis=None)+i,alpha=0.5,color='black',linewidth=0.5,linestyle='--')
    ax.set_xlabel('Normalized time within session')
    ax.set_ylabel('dF/F')
    ax.set_title(ses.sessiondata['session_id'][0])
    ax.set_ylim([-0.6,4.1])
    ax.set_xlim([0,1])
    ax.legend(frameon=False,loc='lower right',ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(savedir,'SessionWideTrace','SessionTrace_%s.png' % ses.sessiondata['session_id'][0]), format = 'png')

#%% 

#%%  Compute the average fluorescence for labeled and unlabeled cells 
# ie. for each neuron and store in cell data, loop over sessions:
for ises in range(nSessions):
    if hasattr(sessions[ises],'calciumdata'): 
        sessions[ises].celldata['meandF']       = np.nanmean(sessions[ises].calciumdata,axis=0) 
        sessions[ises].celldata['mediandF']     = np.nanmedian(sessions[ises].calciumdata,axis=0) 
    # sessions[ises].celldata['noise_corr_avg'] = np.nanmean(np.abs(sessions[ises].noise_corr),axis=1) 

#%% Combine cell data from all loaded sessions to one dataframe:
# celldata = pd.concat([ses.celldata[filter_nearlabeled(ses,radius=50)] for ses in sessions]).reset_index(drop=True)
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# celldata['area_label'] = celldata['roi_name'] + celldata['labeled']

#%% ##################### Calcium trace skewness for labeled vs unlabeled cells:
# order = [0,1] #for statistical testing purposes
# pairs = [(0,1)]

order = ['V1unl','V1lab','PMunl','PMlab'] #for statistical testing purposes
pairs = [('V1unl','V1lab'),('PMunl','PMlab')]

# fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF","meanF_chan2"]
fields = ["noise_level","meanF","tuning_var",'depth','meandF']
fields = ['meandF']

nfields = len(fields)
# fig,axes   = plt.subplots(1,nfields,figsize=(12,4))

celldata['noise_level'].clip(upper=10,inplace=True)
# celldata['noise_variance'].clip(upper=0.6,inplace=True)

for i in range(nfields):
    fig,ax   = plt.subplots(1,1,figsize=(3,4))
    sns.violinplot(data=celldata,y=fields[i],x="area_label",palette=['gray','orangered','gray','indianred'],
                    ax=ax,order=order,inner='quart',cut=10)
    # sns.violinplot(data=celldata,y=fields[i],x="roi_name",palette=['gray','orangered'],order=['V1','PM'],
                #    hue='labeled',ax=ax,split=True,inner='quart')
    ax.set_ylim(np.nanpercentile(celldata[fields[i]],[0,98]))
    # ax.set_xlim(-0.5,1.5)

    annotator = Annotator(ax, pairs, data=celldata, x="area_label", y=fields[i], order=order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    ax.set_xlabel('area + label')
    ax.set_ylabel('')
    ax.set_title(fields[i])
    plt.tight_layout()

g = sessions[ises].celldata['cell_id'][sessions[ises].celldata['meandF']>np.percentile(sessions[ises].celldata['meandF'],95)]
g.head(50)

plt.scatter(sessions[ises].celldata['meanF'],sessions[ises].celldata['meandF'],alpha=0.5,c='gray',s=2)
plt.scatter(sessions[ises].celldata['meanF_chan2'][sessions[ises].celldata['redcell']==0],
            sessions[ises].celldata['meandF'][sessions[ises].celldata['redcell']==0],alpha=0.5,c='gray',s=2)
plt.scatter(sessions[ises].celldata['meanF_chan2'][sessions[ises].celldata['redcell']==1],
            sessions[ises].celldata['meandF'][sessions[ises].celldata['redcell']==1],alpha=0.5,c='red',s=2)

g = sessions[ises].celldata['cell_id'][sessions[ises].celldata['meanF_chan2']>1000]
g.head(50)
# plt.savefig(os.path.join(savedir,'Deconvolution','Resp_dF_deconv' + '.png'), format = 'png')

#%% #########################################################################################
# Contrast: across areas, layers and projection pairs:
# areapairs           = ['V1-V1','PM-PM','V1-PM']
# layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
# projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
# #If you override any of these with input to the deltarf bin function as ' ', then these pairs will be ignored

# clrs_areapairs      = get_clr_area_pairs(areapairs)
# clrs_layerpairs     = get_clr_layerpairs(layerpairs)
# clrs_projpairs      = get_clr_labelpairs(projpairs)

# # clrs_area_labelpairs = get_clr_area_labelpairs(areapairs+projpairs)

# #%% Give redcells a string label
# redcelllabels = np.array(['unl','lab'])
# for ses in sessions:
#     ses.celldata['labeled'] = ses.celldata['redcell']
#     ses.celldata['labeled'] = ses.celldata['labeled'].astype(int).apply(lambda x: redcelllabels[x])
# sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

# # #%% Detrend the data:
# # for ises in np.arange(len(sessions)):
# #     sessions[ises].respmat = detrend(sessions[ises].respmat,axis=1)
# # sessions = compute_signal_noise_correlation(sessions,uppertriangular=False)

# #%% Compute the variance across trials for each cell:
# for ses in sessions:
#     if ses.sessiondata['protocol'][0]=='GR':
#         resp_meanori,respmat_res        = mean_resp_gr(ses)
#     elif ses.sessiondata['protocol'][0]=='GN':
#         resp_meanori,respmat_res        = mean_resp_gn(ses)
#     ses.celldata['noise_variance']  = np.var(respmat_res,axis=1)

