# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023-2025, Champalimaud Center
"""

#%% 
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
from statsmodels.stats.anova import anova_lm, AnovaRM

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.imagelib import load_natural_images #
from utils.tuning import *
from utils.corr_lib import compute_signal_noise_correlation
from utils.gain_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\')


#%% ################################################
session_list        = np.array([['LPE11086_2023_12_16']])
session_list        = np.array([['LPE13959_2025_02_24']])

session_list        = np.array([['LPE13959_2025_02_24',
                                 'LPE11086_2023_12_16']])

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['IM'],only_session_id=session_list)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_lab_cells_V1=50,min_lab_cells_PM=50)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_cells=2000)
sessions,nSessions   = filter_sessions(protocols = ['IM'],im_ses_with_repeats=True)
sessions,nSessions   = filter_sessions(protocols = ['IM'])

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    # sessions[ises].load_respmat(calciumversion='dF',keepraw=False)
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

#%% ### Load the natural images:
# natimgdata = load_natural_images(onlyright=False)
natimgdata = load_natural_images(onlyright=True)

#%% Compute tuning metrics of natural images:
for ses in tqdm(sessions,desc='Computing tuning metrics for each session'): 
    ses.celldata['tuning_SNR']                          = compute_tuning_SNR(ses)
    ses.celldata['corr_half'],ses.celldata['rel_half']  = compute_splithalf_reliability(ses)
    ses.celldata['sparseness']          = compute_sparseness(ses.respmat)
    ses.celldata['selectivity_index']   = compute_selectivity_index(ses.respmat)
    ses.celldata['fano_factor']         = compute_fano_factor(ses.respmat)
    ses.celldata['gini_coefficient']    = compute_gini_coefficient(ses.respmat)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)

# #%% Add how neurons are coupled to the population rate: 
# for ses in tqdm(sessions,desc='Computing population coupling for each session'):
#     resp        = stats.zscore(ses.respmat.T,axis=0)
#     poprate     = np.mean(resp, axis=1)
#     # popcoupling = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(N)]

#     ses.celldata['pop_coupling']   = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(len(ses.celldata))]

#%% 
nActBins = 10

knn_dec = np.full((nActBins,nSessions),np.nan)

for ises,ses in enumerate(sessions):

    resp    = ses.respmat.T
    istim   = np.array(ses.trialdata['ImageNumber'])
    nimg    = istim.max() + 1 # these are blank stims (exclude them)

    # mean center each neuron
    resp -= resp.mean(axis=0)
    resp = resp / (resp.std(axis=0) + 1e-6)

    # compute population response
    poprate     = np.mean(resp, axis=1)

    ### sanity check - decent signal variance ?
    # split stimuli into two repeats
    NN = resp.shape[1]
    sresp = np.zeros((2, nimg, NN), np.float64)
    inan = np.zeros((nimg,)).astype(bool)
    spopresp = np.zeros((nimg,))
    for n in range(nimg):
        ist = (istim==n).nonzero()[0]
        i1 = ist[:int(ist.size/2)]
        i2 = ist[int(ist.size/2):]
        # check if two repeats of stim
        if np.logical_or(i2.size < 1, i1.size < 1):
            inan[n] = 1
        else:
            sresp[0, n, :] = resp[i1, :].mean(axis=0)
            sresp[1, n, :] = resp[i2, :].mean(axis=0)
            spopresp[n] = poprate[[i1,i2]].mean()
    
    # remove image responses without two repeats
    sresp = sresp[:,~inan,:]
    spopresp = spopresp[~inan]

    ### KNN decoding
    # 1 nearest neighbor decoder    
    # (mean already subtracted)
    # sresp = snorm
    cc  = sresp[0] @ sresp[1].T
    cc /= (sresp[0]**2).sum()
    cc /= (sresp[1]**2).sum()
    nstims = sresp.shape[1]
    acc = (cc.argmax(axis=1)==np.arange(0,nstims,1,int)).mean()
    print('Session %d/%d decoding accuracy: %2.3f' % (ises+1,nSessions,acc))

    binedges        = np.percentile(spopresp,np.linspace(0,100,nActBins+1))
    bincenters      = (binedges[1:]+binedges[:-1])/2

    for iap in range(nActBins):
        idx_T       = np.all((spopresp >= binedges[iap],
                            spopresp <= binedges[iap+1]), axis=0)
        
        cc  = sresp[0,idx_T,:] @ sresp[1,idx_T,:].T
        cc /= (sresp[0,idx_T,:]**2).sum()
        cc /= (sresp[1,idx_T,:]**2).sum()
        nstims = np.sum(idx_T)
        acc = (cc.argmax(axis=1)==np.arange(0,nstims,1,int)).mean()
        # print('decoding accuracy: %2.3f'%acc)
        knn_dec[iap,ises] = acc

#%% Plot the accuracy as a function of the population activity
fig,ax = plt.subplots(1,1,figsize=(3.3,3))
for i in range(nSessions):
    ax.plot(np.arange(1,nActBins+1),knn_dec[:,i],c='k',linewidth=0.5,alpha=0.5)

# shaded_error(np.arange(1,nActBins+1),np.nanmean(knn_dec,axis=1),np.nanstd(knn_dec,axis=1)/np.sqrt(nSessions))
shaded_error(np.arange(1,nActBins+1),knn_dec.T,error='sem',ax=ax,linewidth=3)
ax.set_xlabel('Population activity bins')
ax.set_ylabel('Decoding accuracy')
ax.set_title('KNN Natural Image Decoding',fontsize=11)
ax.set_ylim([0,1])
ax.set_xticks(np.arange(1,nActBins+1),np.arange(1,nActBins+1))
ax.axhline(1/2400/nActBins, color='grey', linewidth=2, linestyle='--')
ax.text(1,1/2400/nActBins+0.02,'Chance',color='k',fontsize=8)
sns.despine(fig=fig, top=True, right=True,offset=3)
fig.tight_layout()
my_savefig(fig,savedir,'KNN_decoding_ActBins_%dsessions' % (nSessions), formats = ['png'])

#%% 





#%% 
maxnoiselevel = 20

nActBins                = 6
ncouplingbins           = 5
clrs_popcoupling        = sns.color_palette('magma',ncouplingbins)

knn_dec                 = np.full((nActBins,ncouplingbins,nSessions),np.nan)

for ises,ses in enumerate(sessions):
    binedges_popcoupling    = np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,ncouplingbins+1))

    resp    = ses.respmat.T
    istim   = np.array(ses.trialdata['ImageNumber'])
    nimg    = istim.max() + 1 # these are blank stims (exclude them)

    # mean center each neuron
    resp -= resp.mean(axis=0)
    resp = resp / (resp.std(axis=0) + 1e-6)

    # compute population response
    poprate     = np.mean(resp, axis=1)

    ### sanity check - decent signal variance ?
    # split stimuli into two repeats
    NN = resp.shape[1]
    sresp = np.zeros((2, nimg, NN), np.float64)
    inan = np.zeros((nimg,)).astype(bool)
    spopresp = np.zeros((nimg,))
    for n in range(nimg):
        ist = (istim==n).nonzero()[0]
        i1 = ist[:int(ist.size/2)]
        i2 = ist[int(ist.size/2):]
        # check if two repeats of stim
        if np.logical_or(i2.size < 1, i1.size < 1):
            inan[n] = 1
        else:
            sresp[0, n, :] = resp[i1, :].mean(axis=0)
            sresp[1, n, :] = resp[i2, :].mean(axis=0)
            spopresp[n] = poprate[[i1,i2]].mean()
    
    # remove image responses without two repeats
    sresp       = sresp[:,~inan,:]
    spopresp    = spopresp[~inan]

    ### KNN decoding
    # 1 nearest neighbor decoder    
    # (mean already subtracted)
    # sresp = snorm
    cc  = sresp[0] @ sresp[1].T
    cc /= (sresp[0]**2).sum()
    cc /= (sresp[1]**2).sum()
    nstims = sresp.shape[1]
    acc = (cc.argmax(axis=1)==np.arange(0,nstims,1,int)).mean()
    print('Session %d/%d decoding accuracy with all neurons, all trials: %2.3f' % (ises+1,nSessions,acc))

    binedges        = np.percentile(spopresp,np.linspace(0,100,nActBins+1))
    bincenters      = (binedges[1:]+binedges[:-1])/2

    for icp in range(ncouplingbins):
        # idx_N = np.all((ses.celldata['pop_coupling'] >= binedges_popcoupling[icp],
        #                 ses.celldata['pop_coupling'] <= binedges_popcoupling[icp+1]), axis=0)
        
        idx_N = np.all((
                        ses.celldata['pop_coupling'] >= binedges_popcoupling[icp],
                        ses.celldata['pop_coupling'] <= binedges_popcoupling[icp+1],
                        ses.celldata['noise_level'] < maxnoiselevel,
                        ses.celldata['roi_name'] == 'V1'
                        ), axis=0)
                
        for iap in range(nActBins):
            idx_T       = np.all((spopresp >= binedges[iap],
                                spopresp <= binedges[iap+1]), axis=0)
            
            cc  = sresp[np.ix_([0],idx_T,idx_N)].squeeze() @ sresp[np.ix_([1],idx_T,idx_N)].squeeze().T
            cc /= (sresp[np.ix_([0],idx_T,idx_N)].squeeze()**2).sum()
            cc /= (sresp[np.ix_([1],idx_T,idx_N)].squeeze()**2).sum()

            # cc  = sresp[0,idx_T,:] @ sresp[1,idx_T,:].T
            # cc /= (sresp[0,idx_T,:]**2).sum()
            # cc /= (sresp[1,idx_T,:]**2).sum()
            nstims = np.sum(idx_T)
            acc = (cc.argmax(axis=1)==np.arange(0,nstims,1,int)).mean()
            # print('decoding accuracy: %2.3f'%acc)
            # knn_dec[iap,ises] = acc
            knn_dec[iap,icp,ises] = acc

#%% Plot the accuracy as a function of the population activity
fig,ax = plt.subplots(1,1,figsize=(4.3,3))
# for i in range(nSessions):
#     ax.plot(np.arange(1,nActBins+1),knn_dec[:,i],c='k',linewidth=0.5,alpha=0.5)
handles = []
for icp in range(ncouplingbins):
    # ax.plot(np.arange(1,nActBins+1),knn_dec[:,icp,:],c='k',linewidth=0.5,alpha=0.5)

    # shaded_error(np.arange(1,nActBins+1),np.nanmean(knn_dec,axis=1),np.nanstd(knn_dec,axis=1)/np.sqrt(nSessions))
    handles.append(shaded_error(np.arange(1,nActBins+1),knn_dec[:,icp,:].T,error='sem',ax=ax,
                 linewidth=3,color=clrs_popcoupling[icp]))
ax.set_xlabel('Population activity bins')
ax.set_ylabel('Decoding accuracy')
ax.set_title('KNN Natural Image Decoding',fontsize=11)
# ax.set_ylim([0,1])
ax.set_ylim([0,0.4])
ax.set_xticks(np.arange(1,nActBins+1),np.arange(1,nActBins+1))
ax.axhline(1/2400/nActBins, color='grey', linewidth=2, linestyle='--')
ax.text(1,1/2400/nActBins+0.02,'Chance',color='k',fontsize=8)
ax.legend(handles,['0-20%','20-40%','40-60%','60-80%','80-100%'],
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
                    reverse=True,fontsize=8,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')

sns.despine(fig=fig, top=True, right=True,offset=3)
fig.tight_layout()
my_savefig(fig,savedir,'KNN_decoding_ActBins_CouplingBins_%dsessions' % (nSessions), formats = ['png'])

#%% Repeated measures ANOVA on knn_dec
df = data=pd.DataFrame({'knn_dec': knn_dec.ravel(), 
                          'animal': np.tile(range(nSessions), nActBins*ncouplingbins),
                          'activity_bin': np.repeat(range(nActBins), ncouplingbins*nSessions), 
                          'coupling_bin': np.tile(np.repeat(range(ncouplingbins), nSessions), nActBins)})

df.head(25)
# Conduct the repeated measures ANOVA
print(AnovaRM(df, depvar='knn_dec',subject='animal', within=['activity_bin', 'coupling_bin']).fit())





#%% 

#    # #     # #     #    ######  #######  #####  ####### ######  ### #     #  #####  
#   #  ##    # ##    #    #     # #       #     # #     # #     #  #  ##    # #     # 
#  #   # #   # # #   #    #     # #       #       #     # #     #  #  # #   # #       
###    #  #  # #  #  #    #     # #####   #       #     # #     #  #  #  #  # #  #### 
#  #   #   # # #   # #    #     # #       #       #     # #     #  #  #   # # #     # 
#   #  #    ## #    ##    #     # #       #     # #     # #     #  #  #    ## #     # 
#    # #     # #     #    ######  #######  #####  ####### ######  ### #     #  #####  

#%% 
# Do KNN decoding with choristers and soloists, very easy! Cool! Is there a large V1 dataset with natural images with spiking data? Yes Paolo Papale data. But that is MUA no?
# Do KNN category decoding? 
# Reconstruct the image with choristers and soloists, which one better? How about binning sparsity?

sesidx = 11
resp = sessions[sesidx].respmat.T
np.shape(resp)
istim = np.array(sessions[sesidx].trialdata['ImageNumber'])
nimg = istim.max() # these are blank stims (exclude them)

# mean center each neuron
resp -= resp.mean(axis=0)
resp = resp / (resp.std(axis=0) + 1e-6)

### sanity check - decent signal variance ?
# split stimuli into two repeats
NN = resp.shape[1]
sresp = np.zeros((2, nimg, NN), np.float64)
inan = np.zeros((nimg,)).astype(bool)
for n in range(nimg):
    ist = (istim==n).nonzero()[0]
    i1 = ist[:int(ist.size/2)]
    i2 = ist[int(ist.size/2):]
    # check if two repeats of stim
    if np.logical_or(i2.size < 1, i1.size < 1):
        inan[n] = 1
    else:
        sresp[0, n, :] = resp[i1, :].mean(axis=0)
        sresp[1, n, :] = resp[i2, :].mean(axis=0)
        
# normalize the responses across images
# Subtract that mean, and divide by std — this makes the response of each neuron
# for each repeat zero-mean and unit-variance across stimuli.
# So now, for each neuron, the normalized responses to different stimuli 
# are on the same scale in both repeats.
snorm = sresp - sresp.mean(axis=1)[:,np.newaxis,:]
snorm = snorm / (snorm.std(axis=1)[:,np.newaxis,:] + 1e-6)

#Get the correlation of each neuron's response across repeats
cc = (snorm[0].T @ snorm[1]) / sresp.shape[1]
#print the mean correlation coeff:
print('fraction of signal variance: %2.3f'%np.diag(cc).mean())

cc = sresp[0] @ sresp[1].T

cc = sresp[0] @ sresp[1].T
cc /= (sresp[0]**2).sum()
cc /= (sresp[1]**2).sum()
nstims = sresp.shape[1]
print('decoding accuracy: %2.3f'%(cc.argmax(axis=1)==np.arange(0,nstims,1,int)).mean())

#%% 

def plot_knn_binnedvar(sessions,sesidx,varlabel,nbins=5):
    resp = sessions[sesidx].respmat.T
    istim = np.array(sessions[sesidx].trialdata['ImageNumber'])
    # nimg = istim.max() + 1 # these are blank stims (exclude them)
    nimg = len(np.unique(istim))

    # mean center each neuron
    resp -= resp.mean(axis=0)
    resp = resp / (resp.std(axis=0) + 1e-6)

    ### sanity check - decent signal variance ?
    # split stimuli into two repeats
    NN = resp.shape[1]
    sresp = np.zeros((2, nimg, NN), np.float64)
    inan = np.zeros((nimg,)).astype(bool)
    # for n in range(nimg):
    for n,ni in enumerate(np.unique(istim)): # loop over images
        ist = (istim==ni).nonzero()[0]
        i1 = ist[:int(ist.size/2)]
        i2 = ist[int(ist.size/2):]
        # check if two repeats of stim
        if np.logical_or(i2.size < 1, i1.size < 1):
            inan[n] = 1
        else:
            sresp[0, n, :] = resp[i1, :].mean(axis=0)
            sresp[1, n, :] = resp[i2, :].mean(axis=0)
    sresp = sresp[:,~inan,:]

    binedges = np.nanpercentile(sessions[sesidx].celldata[varlabel],np.linspace(0,100,nbins+1))
    bincenters  = (binedges[:-1]+binedges[1:])/2
    knnperf = np.empty(nbins)
    for ibin in range(nbins):
        idx_N = (sessions[sesidx].celldata[varlabel] >= binedges[ibin]) & (sessions[sesidx].celldata[varlabel] < binedges[ibin+1])
        cc = sresp[0][:,idx_N] @ sresp[1][:,idx_N].T
        # cc = sresp[0] @ sresp[1].T
        cc /= (sresp[0,:,idx_N]**2).sum()
        cc /= (sresp[1,:,idx_N]**2).sum()

        knnperf[ibin] = (cc.argmax(axis=1)==np.arange(0,nimg,1,int)).mean()
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    ax.plot(bincenters,knnperf,color='k',linewidth=2)
    ax.set_xlabel(varlabel)
    # ax.set_xticks(binedges[:-1])
    ax.set_ylabel('KNN decoding accuracy')
    ax.set_ylim([0,ax.get_ylim()[1]])
    sns.despine(offset=3,top=True,right=True,trim=True)

# plot_knn_binnedvar(sessions,sesidx,varlabel='gini_coefficient',nbins=5)
# plot_knn_binnedvar(sessions,sesidx,varlabel='skew',nbins=5)

#%%
sesidx = 3
numeric_cols = sessions[sesidx].celldata.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plot_knn_binnedvar(sessions,sesidx,varlabel=col,nbins=5)

#%% Add how neurons are coupled to the population rate: 
for ses in tqdm(sessions,desc='Computing tuning metrics for each session'):
    resp = zscore(ses.respmat.T,axis=0)
    poprate = np.mean(resp, axis=1)
    # popcoupling = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(N)]

    ses.celldata['pop_coupling']                          = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(len(ses.celldata))]

plot_knn_binnedvar(sessions,10,varlabel='pop_coupling',nbins=5)

for ises in range(nSessions):
    sessions[ises].celldata['pop_coupling'][sessions[ises].celldata['pop_coupling']>0.3] = np.nan
    plot_knn_binnedvar(sessions,ises,varlabel='pop_coupling',nbins=10)

# plot_knn_binnedvar(sessions,11,varlabel='RF_R2',nbins=5)

#%% 
var1 = 'pop_coupling'
var2 = 'tuning_SNR'
# var2 = 'iscell_prob'
# var2 = 'npix_soma'
# var2 = 'radius'
# var2 = 'skew'
# var2 = 'event_rate'

plt.scatter(sessions[sesidx].celldata[var1],sessions[sesidx].celldata[var2],c='k',alpha=0.5,s=5)
plt.xlabel(var1)
plt.ylabel(var2)
from scipy.stats import linregress

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(sessions[sesidx].celldata[var1],sessions[sesidx].celldata[var2])

# Plot regression line
xs = np.array([sessions[sesidx].celldata[var1].min(),sessions[sesidx].celldata[var1].max()])
ys = slope * xs + intercept
plt.plot(xs,ys,'r')
# plt.axhline(y=1, color='k', linestyle='--', linewidth=1)
# Plot r value and p value
plt.text(0.05,0.95,'r=%1.2f, p=%1.2e' % (r_value,p_value),transform=plt.gca().transAxes)


#%%
sesidx = 11
numeric_cols = ['rf_sx_RRR','rf_sy_RRR','rf_r2_RRR','RF_R2']
for col in numeric_cols:
    plot_knn_binnedvar(sessions,sesidx,varlabel=col,nbins=10)

#%%
sessions[sesidx].celldata['rf_sx_RRR'] = np.clip(sessions[sesidx].celldata['rf_sx_RRR'],0,100)
sessions[sesidx].celldata['rf_sy_RRR'] = np.clip(sessions[sesidx].celldata['rf_sy_RRR'],0,100)

#%%
sesidx = 11
numeric_cols = ['rf_sx_RRR','rf_sy_RRR','rf_r2_RRR','RF_R2']
for col in numeric_cols:
    plot_knn_binnedvar(sessions,sesidx,varlabel=col,nbins=5)


