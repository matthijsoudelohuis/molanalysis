
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from scipy.stats import zscore
from scipy.stats import linregress

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% LPE09665_2023_03_14

#%% #############################################################################

sessions,nSessions   = filter_sessions(protocols = ['SP','GR'])

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

print(sessiondata[sessiondata['protocol'].isin(['SP','GR'])].drop_duplicates('session_id')['session_id'].values)

uses =  sessiondata['session_id'].unique()

for sesid in uses:
    if np.all(np.isin(['SP','GR'],sessiondata.loc[sessiondata['session_id'] == sesid,'protocol'])):
        print(sesid)

#%% 

session_list        = np.array([['LPE11086_2024_01_05']])
# session_list        = np.array([['LPE12223_2024_06_10']])

sessions,nSessions   = filter_sessions(protocols = ['SP','GR'],only_session_id=session_list)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)


#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='deconv',keepraw=True)
    sessions[ises].load_data(load_calciumdata=True,calciumversion='deconv')
    data                = zscore(sessions[ises].calciumdata.to_numpy(), axis=0)
    poprate             = np.nanmean(data,axis=1)
    sessions[ises].celldata['popcoupling'] = [np.corrcoef(data[:,i],poprate)[0,1] for i in range(np.shape(data)[1])]

#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


#%% 
idx_GR = np.where(sessiondata['protocol'].isin(['GR']))[0][0]
idx_SP = 1-idx_GR

#%% 
##############################################################################
## Construct trial response matrix:  N neurons by K trials
sessions[idx_GR].respmat         = compute_respmat(sessions[idx_GR].calciumdata, sessions[idx_GR].ts_F, sessions[idx_GR].trialdata['tOnset'],
                                t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False, label = "response matrix")

#%% 
from utils.plot_lib import * #get all the fixed color schemes
from utils.gain_lib import * 
from utils.tuning import compute_tuning_wrapper

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)


#%% 

ises = idx_GR
orientations        = sessions[ises].trialdata['Orientation']
data                = sessions[ises].respmat
data_hat_tuned      = tuned_resp_model(data, orientations)
data_hat_poprate    = pop_rate_gain_model(data, orientations)

datasets            = (data,data_hat_tuned,data_hat_poprate)
dataset_labels      = ['original','tuning','pop rate gain']

fig = plot_respmat(orientations, datasets, dataset_labels,sessions[ises].celldata['pref_ori'].to_numpy())
fig.savefig(os.path.join(savedir,'Heatmap_respmat_modelversions_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% 
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.scatter(sessions[idx_SP].celldata['popcoupling'],sessions[idx_GR].celldata['popcoupling'],
           s=5,marker='.',c='k',alpha=0.5)
ax.set_xlabel('SP coupling')
ax.set_ylabel('GR coupling')
corr = np.corrcoef(sessions[idx_SP].celldata['popcoupling'],sessions[idx_GR].celldata['popcoupling'])[0,1]
ax.set_title('r = %1.2f' % corr)
sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)

#%% Multiplicative tuning for different coupled neurons: 
nPopCouplingBins    = 5
nPopRateBins        = 5

binedges_popcoupling   = np.percentile(sessions[idx_SP].celldata['popcoupling'],np.linspace(0,100,nPopCouplingBins+1))

poprate             = np.nanmean(zscore(sessions[ises].respmat.T, axis=0),axis=1)
binedges_poprate    = np.percentile(poprate,np.linspace(0,100,nPopRateBins+1))

stims    = sessions[idx_GR].trialdata['Orientation'].to_numpy()
ustim   = np.unique(stims)
nstim   = len(ustim)

# respmat = sessions[idx_GR].respmat
respmat = zscore(sessions[idx_GR].respmat,axis=1)

N = np.shape(sessions[idx_GR].respmat)[0]
resp_meanori    = np.empty([N,16])
for istim,stim in enumerate(ustim):
    resp_meanori[:,istim] = np.nanmean(respmat[:,sessions[ises].trialdata['Orientation']==stim],axis=1)
prefori  = np.argmax(resp_meanori,axis=1)

meandata = np.full((N,nPopRateBins,nstim),np.nan)
stddata  = np.full((N,nPopRateBins,nstim),np.nan)

for iPopRateBin in range(nPopRateBins):
# ax = axes[d]
    data    = respmat
    for istim,stim in enumerate(ustim):
        idx_T = np.all((stims == stim,
                        poprate>binedges_poprate[iPopRateBin],
                        poprate<=binedges_poprate[iPopRateBin+1]),axis=0)
        # idx_T = np.all((stims == stim,
        #                 poprate>=-1000,
        #                 poprate<=1000),axis=0)
        meandata[:,iPopRateBin,istim] = np.mean(respmat[:,idx_T],axis=1)
        stddata[:,iPopRateBin,istim] = np.std(respmat[:,idx_T],axis=1)

    # sm = np.roll(sm,shift=-prefori,axis=1)
    for n in range(N):
        meandata[n,iPopRateBin,:] = np.roll(meandata[n,iPopRateBin,:],-prefori[n])
        stddata[n,iPopRateBin,:] = np.roll(stddata[n,iPopRateBin,:],-prefori[n])

#%% 
clrs_popcoupling    = sns.color_palette('viridis',nPopCouplingBins)

fig,axes = plt.subplots(1,nPopCouplingBins,figsize=(15,2.5),sharey=True,sharex=True)
for iPopCouplingBin in range(nPopCouplingBins):
    ax = axes[iPopCouplingBin]
    idx_popcoupling = np.all((sessions[idx_GR].celldata['OSI']>0,
                            sessions[idx_SP].celldata['popcoupling']>binedges_popcoupling[iPopCouplingBin],
                            sessions[idx_SP].celldata['popcoupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)
                            # sessions[idx_GR].celldata['popcoupling']>binedges_popcoupling[iPopCouplingBin],
                            # sessions[idx_GR].celldata['popcoupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)

    # idx_popcoupling = np.where((sessions[idx_SP].celldata['popcoupling']>binedges_popcoupling[iPopCouplingBin])&
    #                         (sessions[idx_SP].celldata['popcoupling']<=binedges_popcoupling[iPopCouplingBin+1]))[0]

    for iPopRateBin in range(nPopRateBins):
        ax.plot(np.mean(meandata[idx_popcoupling,iPopRateBin,:],axis=0),color=clrs_popcoupling[iPopRateBin],
                linewidth=2)
    ax.set_xticks(np.arange(0,len(ustim),2),labels=ustim[::2],fontsize=7)
    # ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
    ax.set_xlabel('Orientation',fontsize=9)
    # ax.set_ylabel('Neuron',fontsize=9)
    ax.tick_params(axis='x', labelrotation=45)
sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)
my_savefig(fig,savedir,'SP_coupling_vs_GR_tunedresp_%s' % (sessions[idx_GR].session_id), formats = ['png'])

#%% 
clrs_popcoupling    = sns.color_palette('viridis',nPopCouplingBins)

fig,axes = plt.subplots(1,nPopCouplingBins,figsize=(15,3.5),sharey=True,sharex=True)
for iPopCouplingBin in range(nPopCouplingBins):
    ax = axes[iPopCouplingBin]
    idx_popcoupling = np.all((sessions[idx_GR].celldata['tuning_var']>0.025,
                            sessions[idx_SP].celldata['popcoupling']>binedges_popcoupling[iPopCouplingBin],
                            sessions[idx_SP].celldata['popcoupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)

    # idx_popcoupling = np.where((sessions[idx_SP].celldata['popcoupling']>binedges_popcoupling[iPopCouplingBin])&
    #                         (sessions[idx_SP].celldata['popcoupling']<=binedges_popcoupling[iPopCouplingBin+1]))[0]
    xdata = np.mean(meandata[idx_popcoupling,0,:],axis=0)
    ydata = np.mean(meandata[idx_popcoupling,-1,:],axis=0)
    ax.scatter(xdata,
               ydata,
               color=clrs_popcoupling[iPopCouplingBin],
                linewidth=2)
    
    #Fit regression line:
    slope, intercept, r_value, p_value, std_err = linregress(xdata,ydata)
    ax.plot(xdata,slope*xdata+intercept,color=clrs_popcoupling[iPopCouplingBin],linewidth=2)
    ax.text(0.2,0.2,'r=%0.2f, p=%0.2f\nslope=%0.2f, intercept=%0.2f' % (r_value,p_value,slope,intercept),fontsize=9)
    
    #Figure make up:
    ax.set_title('Bin #%d' % (iPopCouplingBin+1),fontsize=10)
    ax.plot([-1,5],[-1,5],color='k',linewidth=0.5,ls='--')
    ax.set_xlim([-0.5,2.2])
    ax.set_ylim([-0.5,2.2])
    if iPopCouplingBin==0:
        ax.set_ylabel('High Population Activity')
    if iPopCouplingBin//2:
        ax.set_xlabel('Low Population Activity')
fig.suptitle('Additive and multiplicative scaling across op coupling bins in spontaneous data',fontsize=12)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)
my_savefig(fig,savedir,'SP_coupling_vs_GR_gain_%s' % (sessions[idx_GR].session_id), formats = ['png'])

#%% 

