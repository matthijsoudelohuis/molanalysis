
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

sessions,nSessions  = filter_sessions(protocols = ['GR'],only_session_id=session_list)
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
for ises in tqdm(range(len(sessions)),desc= 'Computing tuning metrics: '):
    if sessions[ises].sessiondata['protocol'].isin(['GR'])[0]:
        idx_K = sessions[ises].respmat_runspeed<1
        # idx_K = sessions[ises].respmat_runspeed>2
        sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat[:,idx_K],
                                                        sessions[ises].trialdata['Orientation'][idx_K])
                                                            
#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Figure
nbins = 5
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
ax.legend(['0-20%','20-40%','40-60%','60-80%','80-100%'],
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
                    reverse=True,fontsize=7,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')

sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Popcoupling_tuning_%dGRsessions' % nSessions,formats=['png'])

#%% 
fig, ax = plt.subplots(1,1,figsize=(4,4),subplot_kw={'projection': 'polar'})

#repeat 0 to 360
uoris = np.unique(celldata['pref_ori'])
uoris = np.append(uoris,360)

for i in range(len(binedges_popcoupling)-1):
    idx_N = np.all((celldata['pop_coupling']>binedges_popcoupling[i],
                  celldata['pop_coupling']<binedges_popcoupling[i+1]),axis=0)
    histdata = np.histogram(celldata['pref_ori'][idx_N],bins=np.arange(0,360+22.5,22.5))[0]
    histdata = np.append(histdata,histdata[0]) #repeat 0 to 360
    ax.plot(np.deg2rad(uoris),histdata,
            color=clrs_popcoupling[i])
# ax.set_rticks([0,5,10,15,20,25])
ax.set_theta_zero_location("N")

ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.set_yticklabels([])
ax.tick_params(axis='y', which='major', pad=10)
ax.set_title('Count',pad=10)
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
ax.legend(['0-20%','20-40%','40-60%','60-80%','80-100%'],
                    reverse=True,fontsize=7,frameon=False,bbox_to_anchor=(1.25,0.9), 
                    title='pop. coupling',loc='upper center')
my_savefig(fig,savedir,'Polar_Popcoupling_tuning_%dGRsessions' % nSessions,formats=['png'])

#%% Figure of orientation selectivity index:
nbins = 5
tuning_metric = 'OSI'
# tuning_metric = 'gOSI'
# tuning_metric = 'tuning_var'
maxnoiselevel = 20

data = np.full((nbins,nSessions),np.nan)
for ises,ses in enumerate(sessions):
    binedges_popcoupling   = np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,nbins+1))
    for ibin in range(len(binedges_popcoupling)-1):
        idx_N = np.all((ses.celldata['pop_coupling']>binedges_popcoupling[ibin],
                    ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1],
                    # ses.celldata['roi_name']=='V1',
                    ses.celldata['noise_level']<maxnoiselevel),axis=0)
        
        # idx_N = np.all((ses.celldata['pop_coupling']>binedges_popcoupling[ibin],
        #             ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1]),axis=0)
        data[ibin,ises] = np.mean(ses.celldata[tuning_metric][idx_N])

fig, ax = plt.subplots(1,1,figsize=(4,3.5))

meandata = np.mean(data,axis=1)
errordata = np.std(data,axis=1) /  np.sqrt(nSessions)
ax.errorbar(np.arange(nbins),meandata,errordata,color='k',linewidth=2)
ax.set_xticks(np.arange(nbins))
ax.set_ylim([0,0.7])
ax.set_xlabel('Coupling bins')
ax.set_ylabel(tuning_metric)
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
#                     reverse=True,fontsize=7,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Popcoupling_%s_tuning_%dGRsessions' % (tuning_metric,nSessions),formats=['png'])



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


#%%







ises = 0

#%% Multiplicative tuning for different coupled neurons: 
nPopCouplingBins    = 5
nPopRateBins        = 5

binedges_popcoupling   = np.percentile(sessions[ises].celldata['pop_coupling'],np.linspace(0,100,nPopCouplingBins+1))

poprate             = np.nanmean(zscore(sessions[ises].respmat.T, axis=0),axis=1)
binedges_poprate    = np.percentile(poprate,np.linspace(0,100,nPopRateBins+1))

stims    = sessions[ises].trialdata['Orientation'].to_numpy()
ustim   = np.unique(stims)
nstim   = len(ustim)

# respmat = sessions[idx_GR].respmat
respmat = zscore(sessions[ises].respmat,axis=1)

N = np.shape(sessions[ises].respmat)[0]
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
    idx_popcoupling = np.all((sessions[ises].celldata['OSI']>0,
                            sessions[ises].celldata['pop_coupling']>binedges_popcoupling[iPopCouplingBin],
                            sessions[ises].celldata['pop_coupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)
    for iPopRateBin in range(nPopRateBins):
        ax.plot(np.mean(meandata[idx_popcoupling,iPopRateBin,:],axis=0),color=clrs_popcoupling[iPopRateBin],
                linewidth=2)
    ax.set_xticks(np.arange(0,len(ustim),2),labels=ustim[::2],fontsize=7)
    # ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
    ax.set_xlabel('Orientation',fontsize=9)
    # ax.set_ylabel('Neuron',fontsize=9)
    ax.tick_params(axis='x', labelrotation=45)
sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)
# my_savefig(fig,savedir,'SP_coupling_vs_GR_tunedresp_%s' % (sessions[ises].session_id), formats = ['png'])

#%% 
sessions[ises].poprate = np.nanmean(zscore(sessions[ises].respmat.T, axis=0),axis=1)
resp = zscore(sessions[ises].respmat.T,axis=0)
resp = sessions[ises].respmat.T

#%% Identify example cell with high population coupling and orientation tuning:
cell_ids = ['LPE12223_2024_06_10_0_0025',
            'LPE12223_2024_06_10_1_0038']

example_cell = np.random.choice(np.where(np.all((
                    # sessions[ises].celldata['pop_coupling']<np.percentile(sessions[ises].celldata['pop_coupling'],10),
                    sessions[ises].celldata['pop_coupling']>np.percentile(sessions[ises].celldata['pop_coupling'],80),
                    sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],80),
                    # sessions[ises].celldata['OSI']>np.percentile(sessions[ises].celldata['OSI'],80),
                    ),axis=0))[0],1)[0]
print(sessions[0].celldata['cell_id'][example_cell])

#%% 
# pal = sns.color_palette('husl', len(oris))
pal = np.tile(sns.color_palette('husl', 8), (2, 1))

# clrs_stimuli    = sns.color_palette('viridis',8)
fig,ax = plt.subplots(1,1,figsize=(3.5,3))

for istim,stim in enumerate(ustim[:8]):
    
    idx_T = np.mod(sessions[ises].trialdata['Orientation'],180)==stim
    ax.scatter(sessions[ises].poprate[idx_T],resp[idx_T,example_cell],
               color=pal[istim],s=0.5)
    x = sessions[ises].poprate[idx_T]
    y = resp[idx_T,example_cell]
    b = linregress(x, y)
    
    xp = np.linspace(np.percentile(sessions[ises].poprate,0.5),
                     np.percentile(sessions[ises].poprate,99.5),100)
    ax.plot(xp,b[0]*xp+b[1],color=pal[istim],linestyle='-',linewidth=2)

ax.set_xlim(np.percentile(sessions[ises].poprate,[0.1,99.9]))
ax.set_ylim(np.percentile(resp[:,example_cell],[0.1,99.9]))
# ax.plot(np.mean(meandata[example_cell,:,:],axis=0),color='k',linewidth=2)
ax.set_ylabel('Response',fontsize=10)
# ax.set_xticks(np.arange(0,len(ustim),2),labels=ustim[::2],fontsize=7)
ax.set_xlabel('Population rate',fontsize=10)
# ax.tick_params(axis='x', labelrotation=45)
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,savedir,'Example_cell_%s' % (sessions[ises].celldata['cell_id'][example_cell]), formats = ['png'])

#%% 
nPopRateBins = 5
binedges_poprate = np.percentile(sessions[ises].poprate,np.linspace(0,99,nPopRateBins+1))
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
x = np.mod(sessions[ises].trialdata['Orientation'],180)
y = sessions[ises].poprate
z = resp[:,example_cell]
stimcond = np.array(x//22.5).astype(int)
# c = pal[sessions[ises].trialdata['stimCond'].astype(int)]
c = pal[stimcond]
ax.scatter(x,y,z,c=c,s=0.5)

# xp = np.linspace(np.percentile(sessions[ises].poprate,0.5),
#                      np.percentile(sessions[ises].poprate,99.5),100)
oris            = np.unique(np.mod(sessions[ises].trialdata['Orientation'],180))
for iqpoprate in range(nPopRateBins):
    resp_meanori    = np.empty([len(oris)])

    for i,ori in enumerate(oris):
        idx_T = np.all((
                    np.mod(sessions[ises].trialdata['Orientation'],180)==ori,
                    y>binedges_poprate[iqpoprate],
                    y<=binedges_poprate[iqpoprate+1]),axis=0)
        # resp_meanori[i] = np.nanmean(resp[example_cell,idx_T])
        resp_meanori[i] = np.nanmean(resp[idx_T,example_cell])
    ax.plot(oris,np.repeat(np.mean([binedges_poprate[iqpoprate],binedges_poprate[iqpoprate+1]]),len(oris)),resp_meanori,
                color='k',linestyle='-',linewidth=2)
    for i,ori in enumerate(oris):
        ax.plot(oris[i],np.mean([binedges_poprate[iqpoprate],binedges_poprate[iqpoprate+1]]),resp_meanori[i],
                color=pal[i,:],linewidth=0,marker='o',markersize=5)
    
    # tuning_curve = mean_resp_gr(sessions[ises],trialfilter=idx_T)[0]
    # tuning_curve = np.mean(z[idx_T],axis=0)
    # b = linregress(x[idx_T], z[idx_T])
    # ax.plot(xp,b[0]*xp+b[1],binedges_poprate[iqpoprate],color=pal[istim],linestyle='-',linewidth=2)

ax.set_xlabel('Stimulus orientation',fontsize=10)
ax.set_ylabel('Pop. rate (z-score)',fontsize=10)
ax.set_zlabel('Response (deconv.)',fontsize=10)
ax.set_xlim([0,160])
ax.set_ylim(np.percentile(sessions[ises].poprate,[0.1,99.5]))
ax.set_zlim(np.percentile(resp[:,example_cell],[0.1,99.9]))
fig.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,savedir,'Example_cell_3D_Ori_PopRate_Response_%s' % (sessions[ises].celldata['cell_id'][example_cell]), formats = ['png'])
