#%% ###################################################
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from utils.corr_lib import *
from utils.psth import compute_tensor

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

#%% Compute types of correlations and show the difference between them: 

idx         = np.zeros(np.shape(sessions[ises].ts_F)[0],dtype=bool)

for it in range(sessions[ises].trialdata.shape[0]):
    idx[np.logical_and(sessions[ises].ts_F>=sessions[ises].trialdata['tOnset'][it],
        sessions[ises].ts_F<=sessions[ises].trialdata['tOnset'][it]+0.75)] = True

N = sessions[ises].calciumdata.shape[1]
labels = ['trace','trace-act','tensor','resp','noise','noise-still','noise-PC','noise-GM']
M = len(labels)

corrmats = np.empty([M,N,N])
corrmats[0,:,:]  = np.corrcoef(sessions[ises].calciumdata.T.to_numpy())
corrmats[1,:,:]  = np.corrcoef(sessions[ises].calciumdata[idx].T.to_numpy())

[tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre=-1, t_post=2,method='nearby')

corrmats[2,:,:]  = np.corrcoef(np.reshape(tensor,(N,-1)))

corrmats[3,:,:]  = np.corrcoef(sessions[ises].respmat)

resp_meanori,respmat_res        = mean_resp_gr(sessions[ises])
# Compute noise correlations from residuals:
corrmats[4,:,:]  =  np.corrcoef(respmat_res)

sessions = compute_signal_noise_correlation(sessions,filter_stationary=True,uppertriangular=False)
corrmats[5,:,:]  =  sessions[ises].noise_corr

sessions = compute_signal_noise_correlation(sessions,filter_stationary=False,uppertriangular=False,remove_method='PCA',remove_rank=1)
corrmats[6,:,:]  =  sessions[ises].noise_corr

sessions = compute_signal_noise_correlation(sessions,filter_stationary=False,uppertriangular=False,remove_method='GM')
corrmats[7,:,:]  =  sessions[ises].noise_corr

crosscorr = np.empty([M,M])
for ix in range(M):
    for iy in range(M):
        data1 = corrmats[ix,:,:]
        data2 = corrmats[iy,:,:]
        nanfilter = np.all((~np.isnan(data1),~np.isnan(data2),np.eye(N,N)==0),axis=0)
        crosscorr[ix,iy] = np.corrcoef(data1[nanfilter].flatten(),data2[nanfilter].flatten())[0,1]

#%% heatmap of cross correlations
fig,axes = plt.subplots(1,M,figsize=(10,6))
for iM in range(M):
    ax = axes[iM]
    ax.imshow(corrmats[iM,:,:] , cmap='coolwarm',
            vmin=np.nanpercentile(corrmats,10),
            vmax=np.nanpercentile(corrmats,90))
    ax.set_title(labels[iM])
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Corr_types_%s.png' % sessions[ises].sessiondata['session_id'][0]), format = 'png')

#%%
fig,ax = plt.subplots(figsize=(6,4.5))
sns.heatmap(crosscorr,vmin=-1,vmax=1,cmap="vlag",xticklabels=labels,yticklabels=labels,ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Crosscorr_corrtypes_%s.png' % sessions[ises].sessiondata['session_id'][0]), format = 'png')
