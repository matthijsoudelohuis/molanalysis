
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import linregress

from loaddata.session_info import filter_sessions,load_sessions
from utils.gain_lib import *
from utils.tuning import *

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################

sessions,nSessions   = filter_sessions(protocols = ['GR'])

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% 

session_list        = np.array([['LPE11086_2024_01_05']])
session_list        = np.array([['LPE12223_2024_06_10']])

sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)

#%% 
def ori_remapping(sessions):
    for ises in range(nSessions):
        if sessions[ises].sessiondata['protocol'][0] == 'GR':
            if not 'Orientation_orig' in sessions[ises].trialdata.keys():
                sessions[ises].trialdata['Orientation_orig']    = sessions[ises].trialdata['Orientation']
                sessions[ises].trialdata['Orientation']         = np.mod(270 - sessions[ises].trialdata['Orientation'],360)
    return sessions

#%%
sessions = ori_remapping(sessions)

#%%
sessions = compute_tuning_wrapper(sessions)

#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Figure
nbins = 8
uoris = np.unique(celldata['pref_ori'])
binedges_popcoupling   = np.percentile(celldata['pop_coupling'],np.linspace(0,100,nbins+1))
clrs_popcoupling = sns.color_palette('magma',nbins)

fig, ax = plt.subplots(1,1,figsize=(4,3.5))

for i in range(len(binedges_popcoupling)-1):
    idx_N = np.all((celldata['pop_coupling']>binedges_popcoupling[i],
                  celldata['pop_coupling']<binedges_popcoupling[i+1]),axis=0)
    
    ax.plot(uoris,np.histogram(celldata['pref_ori'][idx_N],bins=np.arange(0,360+22.5,22.5))[0],
            color=clrs_popcoupling[i])
  
ax.set_xlim([0,337.5])
ax.set_xticks(np.arange(0,360+22.5,45))
ax.set_xlabel('Preferred orientation')
ax.set_ylabel('Count')
ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
                    reverse=True,fontsize=7,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')

sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Popcoupling_tuning_%dGRsessions' % nSessions,formats=['png'])

#%% 
fig, ax = plt.subplots(1,1,figsize=(4,4),subplot_kw={'projection': 'polar'})

for i in range(len(binedges_popcoupling)-1):
    idx_N = np.all((celldata['pop_coupling']>binedges_popcoupling[i],
                  celldata['pop_coupling']<binedges_popcoupling[i+1]),axis=0)
    
    ax.plot(np.deg2rad(uoris),np.histogram(celldata['pref_ori'][idx_N],bins=np.arange(0,360+22.5,22.5))[0],
            color=clrs_popcoupling[i])
# ax.set_rticks([0,5,10,15,20,25])
ax.set_theta_zero_location("N")

ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.set_yticklabels([])
ax.tick_params(axis='y', which='major', pad=10)
ax.set_title('Count',pad=10)
ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
                    reverse=True,fontsize=7,frameon=False,bbox_to_anchor=(1.25,0.9), 
                    title='pop. coupling',loc='upper center')
my_savefig(fig,savedir,'Polar_Popcoupling_tuning_%dGRsessions' % nSessions,formats=['png'])





#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_tensor(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='dF',keepraw=False)
                                calciumversion='deconv',keepraw=False)

#%%
t_axis = sessions[ises].t_axis
for ises in range(nSessions):
    sessions[ises].respmat = np.nanmean(sessions[ises].tensor[:,:,(t_axis>0) & (t_axis<1.5)], axis=2)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)
sessions = compute_tuning_wrapper(sessions)

#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Figure
nbins                   = 8
binedges_popcoupling    = np.percentile(celldata['pop_coupling'],np.linspace(0,100,nbins+1))
clrs_popcoupling        = sns.color_palette('magma',nbins)

ustim, istimeses, stims  = np.unique(sessions[ises].trialdata['Orientation'], \
        return_index=True, return_inverse=True)


N                       = len(celldata)
Nstimuli                = len(ustim)
Nrepet                  = int(len(sessions[ises].trialdata)/Nstimuli)
ntimebins               = len(t_axis)
tensor_meanori          = np.empty((N,Nstimuli,ntimebins))

prefori     = (sessions[ises].celldata['pref_ori'].values//22.5).astype(int)
data        = sessions[ises].tensor[:,np.argsort(stims),:]
data        -= np.mean(data)
data        = data.reshape((N, Nrepet, Nstimuli, ntimebins), order='F')
data        = np.mean(data,axis=1)

for n in range(N):
    data[n,:,:] = np.roll(data[n,:,:],-prefori[n],axis=0)

#%% 
fig, axes = plt.subplots(nbins,Nstimuli,figsize=(Nstimuli*2,nbins*2),sharex=True,sharey=True)
# ylims = [-0.1,0.6]
for icbin in range(len(binedges_popcoupling)-1):
    idx_N   = np.all((celldata['pop_coupling']>binedges_popcoupling[icbin],
                  celldata['pop_coupling']<binedges_popcoupling[icbin+1]),axis=0)
    
    for istim in range(Nstimuli):
        ax = axes[icbin,istim]
        ax.plot(t_axis,np.mean(data[idx_N,istim,:],axis=0),linewidth=2,
                color=clrs_popcoupling[icbin])

        ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
        ax.axis('off')
    # ax.set_xlim([0,337.5])
    # ax.set_xticks(np.arange(0,360+22.5,45))
    # ax.set_xlabel('Preferred orientation'
# ax.set_ylim(ylims)
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Popcoupling_tensor_resp_tuning_%dGRsessions' % nSessions,formats=['png'])

#%% 
fig, ax = plt.subplots(1,1,figsize=(3,2),sharex=True,sharey=True)

ylims = [-0.1,0.6]
for icbin in range(len(binedges_popcoupling)-1):
    idx_N   = np.all((celldata['pop_coupling']>binedges_popcoupling[icbin],
                  celldata['pop_coupling']<binedges_popcoupling[icbin+1]),axis=0)
    
    tuning_curve = np.nanmean(data[np.ix_(idx_N,range(Nstimuli),(t_axis>0) & (t_axis<1.5))],axis=(0,2)) 
    tuning_curve = np.nanmean(data[np.ix_(idx_N,range(Nstimuli),(t_axis>0) & (t_axis<0.75))],axis=(0,2)) 
    # - np.nanmean(data[np.ix_(idx_N,range(Nstimuli),t_axis<0)],axis=(0,2))
    - np.nanmean(data)

    tuning_curve = np.nanmean(data[:,:,(t_axis>0) & (t_axis<0.75)],axis=(2)) - np.nanmean(data[:,:,(t_axis<0)],axis=(2))
    tuning_curve = np.nanmean(tuning_curve[idx_N,:],axis=0)
    # - np.nanmean(data[np.ix_(idx_N,range(Nstimuli),t_axis<0)],axis=(0,2))

    ax.plot(uoris,tuning_curve,linewidth=2,color=clrs_popcoupling[icbin])

ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
ax.set_xticks(np.arange(0,360+22.5,45))
ax.set_xlabel('Preferred orientation')
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Popcoupling_tuning_%dGRsessions' % nSessions,formats=['png'])
