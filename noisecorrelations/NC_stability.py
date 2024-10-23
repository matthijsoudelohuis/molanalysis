#%% ###################################################
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from corr_lib import compute_signal_noise_correlation

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\')

#%% ###################################################

session_list        = np.array([['LPE09830','2023_04_10']]) #GR
session_list        = np.array([['LPE10919','2023_11_06']])

session_list        = np.array([['LPE09665','2023_03_21']]) #GR

sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)

#%%  Load data properly
ises = 0 #
sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            # calciumversion='dF',keepraw=True,filter_hp=0.01)
                            calciumversion='deconv',keepraw=True)

respmat_backup      = sessions[ises].respmat
trialdata_backup    = sessions[ises].trialdata
[N,K]          = np.shape(respmat_backup) #get dimensions of response matrix

#%% ########################## Compute signal and noise correlations: ###################################

nchunks = 2
SCdata = np.empty([N,N,nchunks])
NCdata = np.empty([N,N,nchunks])
for ichunk in range(nchunks):
    start_trial = int(ichunk * K/nchunks)
    end_trial   = int((ichunk+1) * K/nchunks)

    sessions[ises].respmat      = respmat_backup[:,start_trial:end_trial]
    sessions[ises].trialdata    = trialdata_backup[start_trial:end_trial]

    sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=False)
    SCdata[:,:,ichunk] = sessions[ises].sig_corr
    NCdata[:,:,ichunk] = sessions[ises].noise_corr

#%% heatmap of signal correlations split half
fig,axes = plt.subplots(1,2,figsize=(8,5))
for ichunk in range(nchunks):

    plt.subplot(1,2,ichunk+1)
    plt.imshow(SCdata[:,:,ichunk], cmap='coolwarm',
            vmin=np.nanpercentile(SCdata[:,:,0],30),
            vmax=np.nanpercentile(SCdata[:,:,0],80))
    plt.title(sessions[ises].sessiondata['session_id'][0] + ' - Half %s' % (ichunk+1))
xdata = SCdata[:,:,0].flatten()
ydata = SCdata[:,:,1].flatten()
xdata = xdata[~np.isnan(xdata)]
ydata = ydata[~np.isnan(ydata)]
plt.suptitle('Signal Correlation stability r=%1.2f' % np.corrcoef(xdata,ydata)[0,1])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'SC_stability_%s.png' % sessions[ises].sessiondata['session_id'][0]), format = 'png')

#%% heatmap of noise correlations per session
fig,axes = plt.subplots(1,2,figsize=(8,5))
for ichunk in range(nchunks):

    plt.subplot(1,2,ichunk+1)
    plt.imshow(NCdata[:,:,ichunk], cmap='coolwarm',
            vmin=np.nanpercentile(NCdata[:,:,0],20),
            vmax=np.nanpercentile(NCdata[:,:,0],80))
    plt.title(sessions[ises].sessiondata['session_id'][0] + ' - Half %s' % (ichunk+1))
xdata = NCdata[:,:,0].flatten()
ydata = NCdata[:,:,1].flatten()
xdata = xdata[~np.isnan(xdata)]
ydata = ydata[~np.isnan(ydata)]
plt.suptitle('Noise Correlation stability r=%1.2f' % np.corrcoef(xdata,ydata)[0,1])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'NC_stability_deconv_%s.png' % sessions[ises].sessiondata['session_id'][0]), format = 'png')

plt.scatter(xdata,ydata,s=3,c='k',alpha=0.05)
