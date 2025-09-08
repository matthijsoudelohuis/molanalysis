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
from scipy.stats import linregress

from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import *
from utils.corr_lib import compute_signal_noise_correlation
from utils.gain_lib import *
from utils.imagelib import * 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\')


#%% ############################## Some individual sessions: ###################
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

#%% Load proper data and compute average trial responses:                      
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

#%% Plot the response across individual trials for some example neurons
# Color the response by the population rate
sesidx      = 0
ses         = sessions[sesidx]
resp        = stats.zscore(ses.respmat.T,axis=0)
poprate     = np.mean(resp, axis=1)

idx_N       = np.where((ses.celldata['pop_coupling'] > 0.3) & (ses.celldata['tuning_SNR']>0.3))[0]

idx_N       = np.random.choice(idx_N,8,replace=False)

im_repeats  = ses.trialdata['ImageNumber'].value_counts()[:100].index.to_numpy()

im_repeats = np.random.choice(im_repeats,20,replace=False)

poprate     -= np.min(poprate)
poprate     /= np.max(poprate)

fig,axes = plt.subplots(2,4,figsize=(15,5))
for iN,N in enumerate(idx_N):
    ax = axes.flatten()[iN]

    for iIM,IM in enumerate(im_repeats):
        idx_T = ses.trialdata['ImageNumber']==IM
        ax.scatter(np.ones(np.sum(idx_T))*iIM,resp[idx_T,N],c=poprate[idx_T],
                   s=(poprate[idx_T]-0.05)*50,cmap='crest',vmin=0.1,vmax=0.3)
sns.despine(fig=fig, top=True, right=True)
fig.tight_layout()


#%%
#     # ####### ######  ####### #          ####### ######  ####### ####### 
##   ## #     # #     # #       #          #       #     # #       #       
# # # # #     # #     # #       #          #       #     # #       #       
#  #  # #     # #     # #####   #          #####   ######  #####   #####   
#     # #     # #     # #       #          #       #   #   #       #       
#     # #     # #     # #       #          #       #    #  #       #       
#     # ####### ######  ####### #######    #       #     # ####### ####### 

# The idea here is to see whether for the same images the response is higher during trials with 
# high population rate or low population rate, and whether this shows characteristics of multiplicative or 
# additive variability. 

#%% 

# sameIM_poprate              = np.full((100,10),np.nan)
# sameIM_response             = np.full((N,100,2),np.nan)
# sameIM_reconR2_popcoupling  = np.full((nSessions,100,10,ncouplingbins),np.nan)

for ses in tqdm(sessions,desc='Computing affine modulation for each session (model free)'): 
    ses.poprate = np.mean(zscore(sessions[sesidx].respmat.T,axis=0), axis=1)

    #initialize arrays:
    ses.celldata['aff_alpha_imreps'] = np.nan
    ses.celldata['aff_beta_imreps']  = np.nan
    ses.celldata['aff_r2_imreps']    = np.nan

    N                       = len(ses.celldata)
    sameIM_response          = np.full((N,100,2),np.nan)

    #Identify all trials with images that are repeated 10 times in the session:
    im_repeats  = ses.trialdata['ImageNumber'].value_counts()[ses.trialdata['ImageNumber'].value_counts()==10].index.to_numpy()
    for iIM,IM in enumerate(im_repeats):

        idx_T                   = ses.trialdata['ImageNumber'] == IM
        idx_T_low               = np.logical_and(idx_T, poprate < np.percentile(poprate[idx_T],50))
        idx_T_high               = np.logical_and(idx_T, poprate > np.percentile(poprate[idx_T],50))
        assert(np.sum(idx_T_low)==np.sum(idx_T_high)==5),'Not 5 trials per image'
        sameIM_response[:,iIM,0] = np.nanmean(ses.respmat[:,idx_T_low], axis=1)
        sameIM_response[:,iIM,1] = np.nanmean(ses.respmat[:,idx_T_high], axis=1)
    
    for iN in range(N):
    # for iN in range(10):
        xdata   = sameIM_response[iN,:,0]
        ydata   = sameIM_response[iN,:,1]
        
        b       = linregress(xdata,ydata)

        ses.celldata.loc[iN,'aff_alpha_imreps'] = b[0]
        ses.celldata.loc[iN,'aff_beta_imreps']  = b[1]
        ses.celldata.loc[iN,'aff_r2_imreps']    = b[2]**2

#%% Get good unmodulated cell: 
idx_examples = np.all((ses.celldata['aff_r2_imreps']>0.7,
                       ses.celldata['aff_alpha_imreps']<1.1,
                       ses.celldata['aff_alpha_imreps']>0.8,
                       ses.celldata['aff_beta_imreps']<10),axis=0)

print(ses.celldata['cell_id'][idx_examples])

example_neuron      = np.random.choice(ses.celldata['cell_id'][idx_examples])

example_neuron      = 'LPE13959_2025_02_24_3_0310'
example_neuron      = 'LPE13959_2025_02_24_1_0284'
example_neuron      = 'LPE13959_2025_02_24_0_0382'

#%% Get good multiplicatively modulated cells: 
idx_examples = np.all((ses.celldata['aff_r2_imreps']>0.9,
                       ses.celldata['aff_alpha_imreps']>1.5,
                       ses.celldata['aff_beta_imreps']<50),axis=0)

print(ses.celldata['cell_id'][idx_examples])

example_neuron      = np.random.choice(ses.celldata['cell_id'][idx_examples])
example_neuron      = 'LPE13959_2025_02_24_3_0120'

#%% Get good additively modulated cells: 
idx_examples = np.all((ses.celldata['aff_r2_imreps']>0.2,
                    #    ses.celldata['aff_alpha_imreps']<1.2,
                       ses.celldata['aff_beta_imreps']>25),axis=0)

print(ses.celldata['cell_id'][idx_examples])

example_neuron      = np.random.choice(ses.celldata['cell_id'][idx_examples])
# example_neuron      = 'LPE13959_2025_02_24_7_0190'
# example_neuron      = 'LPE13959_2025_02_24_7_0106'

#%%
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
ex_sesid            = 'LPE13959_2025_02_24'

ises            = np.where(np.array(sessiondata['session_id']==ex_sesid))[0][0]
ineuron         = np.where(np.array(sessions[ises].celldata['cell_id'])==example_neuron)[0][0]

sameIM_response = np.full((100,2),np.nan)

#Identify all trials with images that are repeated 10 times in the session:
im_repeats  = ses.trialdata['ImageNumber'].value_counts()[ses.trialdata['ImageNumber'].value_counts()==10].index.to_numpy()
for iIM,IM in enumerate(im_repeats):
    idx_T                   = ses.trialdata['ImageNumber'] == IM
    idx_T_low               = np.logical_and(idx_T, poprate < np.percentile(poprate[idx_T],50))
    idx_T_high               = np.logical_and(idx_T, poprate > np.percentile(poprate[idx_T],50))
    assert(np.sum(idx_T_low)==np.sum(idx_T_high)==5),'Not 5 trials per image'
    sameIM_response[iIM,0] = np.nanmean(ses.respmat[ineuron,idx_T_low])
    sameIM_response[iIM,1] = np.nanmean(ses.respmat[ineuron,idx_T_high])

xdata   = sameIM_response[:,0]
ydata   = sameIM_response[:,1]

b       = linregress(xdata,ydata)

fig,axes = plt.subplots(1,1,figsize=(3,3),sharey='row')
ax = axes
sns.regplot(x=xdata,y=ydata,color='b',
            ax=ax,scatter=True,marker='o',
            scatter_kws={'alpha':0.5, 's':20, 'edgecolors':'black'},
            line_kws={'color':'b', 'ls':'-', 'linewidth':3})
ax.plot([0,10000],[0,10000],'k',ls='--',linewidth=1)
ax.set_xlim(np.percentile(sameIM_response,[0,100])*1.1)
ax.set_ylim(np.percentile(sameIM_response,[0,100])*1.1)
ax.set_xlabel('Low Pop. Rate')
ax.set_ylabel('High Pop. Rate')
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,savedir,'Example_cell_affine_naturalimages_%s' % (example_neuron), formats = ['png'])

#%% Concatenate data:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


#%% Show distribution of multiplicative and additive gain across sessions:
# percthr = 50
# bins = np.logspace(np.log10(0.1),np.log10(50),50)
bins = np.linspace(0,4,50)
fig,axes = plt.subplots(1,2,figsize=(6,3))
ax = axes[0]
idx_N = np.all((celldata['noise_level']<20,
                    # celldata['roi_name']=='V1',
                    # celldata['tuning_SNR']>0.2,
                    celldata['aff_r2_imreps']>0.2,
                    # celldata['pop_coupling']<np.percentile(celldata['pop_coupling'],100-percthr),
                    # celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],100-percthr),
                    # ((celldata['VM_r2_low'] + celldata['VM_r2_high'])/2)>r2_thr,
                    ),axis=0)
sns.histplot(celldata['aff_alpha_imreps'][idx_N],color='green',element="step",stat="probability", 
             common_norm=False,alpha=0.2,ax=ax,bins=bins)
ax.axvline(1,color='k',ls='--')
ax.plot(np.nanmean(celldata['aff_alpha_imreps'][idx_N]),0.1,markersize=10,
        color='green',marker='v')
ax.set_title('Multiplicative gain')
ax.text(0.6,0.6,'N = %d\nneurons' % (np.sum(idx_N)),transform=ax.transAxes)

bins = np.linspace(0,100,50)
ax = axes[1]
sns.histplot(celldata['aff_beta_imreps'][idx_N],color='blue',element="step",stat="probability", 
             common_norm=False,alpha=0.2,ax=ax,bins=bins)
ax.set_title('Additive offset')
plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=5,ax=ax)
my_savefig(fig,savedir,'IM_affine_imreps_naturalimages_%d_sessions' % (nSessions),formats=['png'])


#%% 
idx_N = np.all((celldata['noise_level']<20,
                    ),axis=0)
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
# sns.scatterplot(data=celldata,x='pop_coupling',y='aff_alpha_imreps',ax=ax,marker='.',color='black',alpha=0.1)
sns.histplot(data=celldata[idx_N],x='pop_coupling',ax=ax,color='green')
ax.set_xlim(np.percentile(celldata['pop_coupling'],[1,99]))
plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=5)
my_savefig(fig,savedir,'IM_popcoupling_hist_%d_sessions' % (nSessions),formats=['png'])


#%% 
idx_N = np.all((celldata['noise_level']<20,
                    # celldata['roi_name']=='V1',
                    celldata['aff_r2_imreps']>0.5,
                    ),axis=0)
fig,axes = plt.subplots(1,2,figsize=(6,3))
ax = axes[0]
# sns.scatterplot(data=celldata,x='pop_coupling',y='aff_alpha_imreps',ax=ax,marker='.',color='black',alpha=0.1)
sns.regplot(data=celldata[idx_N],x='pop_coupling',y='aff_alpha_imreps',ax=ax,marker='.',
            scatter_kws={'alpha':0.5, 's':25, 'edgecolors':'white'},
            color='green',robust=False)
ax = axes[1]
sns.regplot(data=celldata[idx_N],x='pop_coupling',y='aff_beta_imreps',ax=ax,marker='.',
            scatter_kws={'alpha':0.5, 's':25, 'edgecolors':'white'},
            color='purple',robust=False)
plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=5)
my_savefig(fig,savedir,'IM_affine_naturalimages_vs_popcoupling_%d_sessions' % (nSessions),formats=['png'])

#%% 






#%%
#     # ####### ######  ####### #          ######     #     #####  ####### ######  
##   ## #     # #     # #       #          #     #   # #   #     # #       #     # 
# # # # #     # #     # #       #          #     #  #   #  #       #       #     # 
#  #  # #     # #     # #####   #          ######  #     #  #####  #####   #     # 
#     # #     # #     # #       #          #     # #######       # #       #     # 
#     # #     # #     # #       #          #     # #     # #     # #       #     # 
#     # ####### ######  ####### #######    ######  #     #  #####  ####### ######  

#%% 
# Fit linear RF and predict Y_hat independent of population activity
# Then bin trials based on the predicted response, and split by population rate for each bin.
# Fit affine model on the binned data.

#%%  

#%% On the trial to trial response: RRR to get RF
#NOTES:
nsub    = 3 #without subsampling really slow, i.e. nsub=1
lam     = 0.05

for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting linear RF for each session'):
    #normalize the response for each neuron:
    resp            = zscore(ses.respmat.T, axis=0)

    IMdata          = natimgdata[:,:,ses.trialdata['ImageNumber']]

    # cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=0.05,nranks=100,nsub=nsub)
    ses.cRF,Y_hat   = linear_RF_cv(resp, IMdata, lam=lam, nsub=nsub)

    ses.Y           = resp.T
    ses.Y_hat       = Y_hat.T

    ses.celldata['RF_R2'] = r2_score(resp,Y_hat,multioutput='raw_values')
    print('RF R2: %0.2f' % (ses.celldata['RF_R2'].mean()))

#%% 
radius = 500
sessions = compute_pairwise_anatomical_distance(sessions)

for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting affine model based on linear RF for each session'):
    
    ses.celldata['aff_r2_rffull']      = np.nan
    ses.celldata['aff_alpha_rffull']   = np.nan
    ses.celldata['aff_beta_rffull']    = np.nan
    ses.celldata['aff_offset_rffull']    = np.nan

    # ses.celldata['aff_r2_rfsplit']      = np.nan
    # ses.celldata['aff_alpha_rfsplit']   = np.nan
    # ses.celldata['aff_beta_rfsplit']    = np.nan
    # ses.celldata['aff_offset_rfsplit']    = np.nan

    Y           = ses.Y

    T           = ses.Y_hat

    N           = ses.respmat.shape[0]

    Y_hat       = np.full_like(ses.respmat, np.nan)

    for iN in range(N):
    # for iN in range(10):
        idx_N       = ses.distmat_xyz[iN,:] < radius
        idx_N[iN]   = False
        r           = np.nanmean(ses.respmat[idx_N,:], axis=0)
        # r           = np.nanmean(ses.respmat[idx_N,:], axis=0)
    
        y           = Y[iN,:]
        x           = T[iN,:]
        
        if np.isnan(r).all():
            # modelcoefs[modelversions.index(modelversion), iN, :] = np.nan
            # model_R2[modelversions.index(modelversion), iN] = np.nan
            # Y_hat[iN,:,modelversions.index(modelversion)] = np.nan
            continue
        # Construct the design matrix
        A = np.vstack([r * x, r, np.ones_like(y)]).T

        # Perform linear regression using least squares
        coefs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

        # Store the coefficients
        [ses.celldata.loc[iN,'aff_alpha_rffull'], ses.celldata.loc[iN,'aff_beta_rffull'], 
            ses.celldata.loc[iN,'aff_offset_rffull']] = coefs

        # Compute R^2 value
        y_pred = A @ coefs
        ses.celldata.loc[iN,'aff_r2_rffull'] = r2_score(y, y_pred)



#%% 
radius = 500
sessions = compute_pairwise_anatomical_distance(sessions)

for ises, ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting affine model based on linear RF for each session'):

    ses.celldata['aff_r2_rfsplit']      = np.nan
    ses.celldata['aff_alpha_rfsplit']   = np.nan
    ses.celldata['aff_beta_rfsplit']    = np.nan
    ses.celldata['aff_offset_rfsplit']    = np.nan

    # valuedata are the correlation values, these are going to be binned
    vdata           = ses.respmat[iN,:]

    #First 2D binning: x is elevation, y is azimuth, 
    xdata               = ses.Y_hat[iN,:]
    ydata               = ses.poprate
    
    binedges_2d = np.percentile
    #Take the sum of the correlations in each bin:
    bin_2d   = binned_statistic_2d(x=xdata, y=ydata, values=vdata,bins=binedges_2d, statistic='mean')[0]
                           

    Y           = ses.Y

    T           = ses.Y_hat

    N           = ses.respmat.shape[0]

    Y_hat       = np.full_like(ses.respmat, np.nan)

    for iN in range(N):
    # for iN in range(10):
        idx_N       = ses.distmat_xyz[iN,:] < radius
        idx_N[iN]   = False
        r           = np.nanmean(ses.respmat[idx_N,:], axis=0)
        # r           = np.nanmean(ses.respmat[idx_N,:], axis=0)
    
        y           = Y[iN,:]
        x           = T[iN,:]
        
        if np.isnan(r).all():
            # modelcoefs[modelversions.index(modelversion), iN, :] = np.nan
            # model_R2[modelversions.index(modelversion), iN] = np.nan
            # Y_hat[iN,:,modelversions.index(modelversion)] = np.nan
            continue
        # Construct the design matrix
        A = np.vstack([r * x, r, np.ones_like(y)]).T

        # Perform linear regression using least squares
        coefs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

        # Store the coefficients
        [ses.celldata.loc[iN,'aff_alpha_rffull'], ses.celldata.loc[iN,'aff_beta_rffull'], 
            ses.celldata.loc[iN,'aff_offset_rffull']] = coefs

        # Compute R^2 value
        y_pred = A @ coefs
        ses.celldata.loc[iN,'aff_r2_rffull'] = r2_score(y, y_pred)
    
#%% Concatenate data:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Show distribution of multiplicative and additive gain across sessions:
# percthr = 50
# bins = np.logspace(np.log10(0.1),np.log10(50),50)
r2_thr = 0.1
bins = np.linspace(0,4,50)
fig,axes = plt.subplots(1,2,figsize=(6,3))
ax = axes[0]
idx_N = np.all((celldata['noise_level']<20,
                    # celldata['roi_name']=='V1',
                    # celldata['tuning_SNR']>0.2,
                    celldata['aff_r2_rffull']>r2_thr,
                    # celldata['pop_coupling']<np.percentile(celldata['pop_coupling'],100-percthr),
                    # celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],100-percthr),
                    # ((celldata['VM_r2_low'] + celldata['VM_r2_high'])/2)>r2_thr,
                    ),axis=0)
sns.histplot(celldata['aff_alpha_imreps'][idx_N],color='green',element="step",stat="probability", 
             common_norm=False,alpha=0.2,ax=ax,bins=bins)
ax.axvline(1,color='k',ls='--')
ax.plot(np.nanmean(celldata['aff_alpha_imreps'][idx_N]),0.1,markersize=10,
        color='green',marker='v')
ax.set_title('Multiplicative gain')
ax.text(0.6,0.6,'N = %d\nneurons' % (np.sum(idx_N)),transform=ax.transAxes)

bins = np.linspace(0,100,50)
ax = axes[1]
sns.histplot(celldata['aff_beta_imreps'][idx_N],color='blue',element="step",stat="probability", 
             common_norm=False,alpha=0.2,ax=ax,bins=bins)
ax.set_title('Additive offset')
plt.tight_layout()
sns.despine(fig=fig,trim=True,offset=5,ax=ax)
my_savefig(fig,savedir,'IM_affine_imreps_naturalimages_%d_sessions' % (nSessions),formats=['png'])


