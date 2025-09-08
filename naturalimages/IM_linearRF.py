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
from sklearn import preprocessing
from tqdm import tqdm
from scipy.stats import zscore
from scipy import stats
from sklearn.metrics import r2_score
from statannotations.Annotator import Annotator

from preprocessing.locate_rf import *
from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.imagelib import * 
from utils.tuning import *
from utils.RRRlib import *
from utils.corr_lib import compute_signal_noise_correlation
from utils.gain_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\')

#%%
# TODO:
# compute noise correlations for the sessions with 100 images repeated 10 times
# loop over sessions with all neurons included
# find regression coeff for each session with enough labeled cells
# do linear RF fit for each session and with finding optimal lambda
# fit RF for a subset of trials and reconstruct with held out test trial, 
# compute R^2 of reconstruction for test data
# do RF 2D gaussian fit for each session
# 

# Insights so far: 
# variability in sessions how well RF fits are
# reconstructions and linear RFs are much better in V1, can reconstruct images much better
# PM and PM labeled is poor
# dividing by population mean does not help a lot for fitting RF
# best reconstruction with optimal state: intermediate population size, medium pupil etc.




#%% ################################################
session_list        = np.array([['LPE11086_2023_12_16']])
session_list        = np.array([['LPE13959_2025_02_24']])

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['IM'],only_session_id=session_list)
sessions,nSessions   = filter_sessions(protocols = ['IM'],min_lab_cells_V1=50,min_lab_cells_PM=50)
sessions,nSessions   = filter_sessions(protocols = ['IM'],min_cells=1)

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    # sessions[ises].load_respmat(calciumversion='dF',keepraw=False)
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

#%% ### Load the natural images:
# natimgdata = load_natural_images(onlyright=False)
natimgdata = load_natural_images(onlyright=True)

#%% Compute tuning metrics:
for ses in sessions: 
    ses.respmean,imageids = mean_resp_image(ses)

#%% Compute tuning metrics of natural images:
for ses in tqdm(sessions,desc='Computing tuning metrics for each session'): 
    ses.celldata['tuning_SNR']                          = compute_tuning_SNR(ses)
    ses.celldata['corr_half'],ses.celldata['rel_half']  = compute_splithalf_reliability(ses)
    ses.celldata['sparseness']          = compute_sparseness(ses.respmat)
    ses.celldata['selectivity_index']   = compute_selectivity_index(ses.respmat)
    ses.celldata['fano_factor']         = compute_fano_factor(ses.respmat)
    ses.celldata['gini_coefficient']    = compute_gini_coefficient(ses.respmat)

#%%










#%% On the trial to trial response: RRR to get RF
sesidx  = 0
nsub    = 4
resp    = sessions[sesidx].respmat.T

#remove gain modulation by the population rate:
resp = resp / np.mean(resp, axis=1, keepdims=True)

#normalize the response for each neuron to the maximum:
resp = resp / np.max(resp, axis=0)
# resp = zscore(resp, axis=0)
# resp = resp / np.percentile(resp, 90,axis=0)

IMdata = natimgdata[:,:,sessions[sesidx].trialdata['ImageNumber']]

# cRF,Y_hat = lowrank_RF(resp, IMdata,lam=0.5,nranks=50,nsub=nsub)

cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=0.1,nranks=25,nsub=nsub)

#%% Show the difference in the distribution of the responses for predicted vs true data:
plt.hist(Y_hat.flatten(), bins=100, color='red', alpha=0.5)
plt.hist(resp.flatten(), bins=100, color='blue', alpha=0.5)
plt.savefig(os.path.join(savedir,'LowRank_RF_FittedResponses_%s.png' % (sessions[sesidx].sessiondata['session_id'][0])),format='png',dpi=300,bbox_inches='tight')

#%% Fit the cRF with a 2D gaussian:
sessions[sesidx].celldata = fit_2dgauss_cRF(cRF, nsub=nsub,celldata=sessions[sesidx].celldata)

#%% Compute EV of the cRF:
print('Fraction of variance explained: %1.3f' % EV(resp,Y_hat))

sessions[sesidx].celldata['RF_R2'] = r2_score(resp,Y_hat,multioutput='raw_values')
print('Average per neuron fraction of variance explained: %1.3f' % np.mean(sessions[sesidx].celldata['RF_R2']))

#%% Add area labels:
# sessions[sesidx].celldata['arealabel'] = sessions[sesidx].celldata['roi_name'] + sessions[sesidx].celldata['labeled']

#%% Show example neurons for different populations:
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
narealabels     = len(arealabels)
nN              = 6 #number of example neurons to show

fig,axes = plt.subplots(nN,narealabels,figsize=(3*narealabels,1*nN))
for ial,arealabel in enumerate(arealabels):
    idx_neurons = np.where(np.all((sessions[sesidx].celldata['arealabel']==arealabel,
                                   ),axis=0))[0]
    # idx_neurons = idx_neurons[np.argsort(sessions[sesidx].celldata['tuning_SNR'][idx_neurons])][-nN:]
    # idx_neurons = idx_neurons[np.argsort(-sessions[sesidx].celldata['tuning_SNR'][idx_neurons])][:nN]
    idx_neurons = idx_neurons[np.argsort(-sessions[sesidx].celldata['RF_R2'][idx_neurons])][:nN]

    for i,iN in enumerate(idx_neurons):
        ax = axes[i,ial]
        lim = np.max(np.abs(cRF[:,:,iN]))*1.2
        ax.imshow(cRF[:,:,iN],cmap='bwr',vmin=-lim,
                            vmax=lim)
        ax.plot(sessions[sesidx].celldata['rf_az_RRR'][iN]/nsub,
                sessions[sesidx].celldata['rf_el_RRR'][iN]/nsub,'k+',markersize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.set_title(arealabel + sessions[sesidx].celldata['cell_id'][iN],fontsize=8)
fig.savefig(os.path.join(savedir,'ReducedRank_FitRF_example_neurons_arealabel_%s.png' % sessions[sesidx].sessiondata['session_id'][0]),format='png',dpi=300,bbox_inches='tight')

#%% Compute pairwise correlations matrix of the cRF:
cRF_reshape             = np.reshape(cRF, (np.shape(cRF)[0]*np.shape(cRF)[1],np.shape(cRF)[2]))
sessions[sesidx].RF_corrmat  = np.corrcoef(cRF_reshape.T)

np.fill_diagonal(sessions[sesidx].RF_corrmat,np.nan)

#%% Sanity check:
# get five entries that have high values, not along the diagonal, show RF for the two neurons
corrmat_triu = np.triu(sessions[sesidx].RF_corrmat,k=1)

idx_neurons = sessions[sesidx].celldata['RF_R2']<0.025

corrmat_triu[idx_neurons,:] = 0

idx_corr = np.unravel_index(np.argsort(corrmat_triu, axis=None)[-5:], corrmat_triu.shape)

fig,ax = plt.subplots(5,2,figsize=(5,4),sharex=True,sharey=True)
# for i,idx in enumerate(idx_corr):
for i,idx in enumerate(zip(idx_corr[0],idx_corr[1])):
    lim = np.max(np.abs(cRF[:,:,idx[0]]))*1.3
    ax[i,0].imshow(cRF[:,:,idx[0]],cmap='bwr',vmin=-lim,vmax=lim)
    ax[i,0].set_xticks([])
    ax[i,0].set_yticks([])
    ax[i,0].set_axis_off()
    ax[i,0].set_title(sessions[sesidx].celldata['cell_id'][idx[0]],fontsize=8)
    ax[i,1].imshow(cRF[:,:,idx[1]],cmap='bwr',vmin=-lim,vmax=lim)
    ax[i,1].set_xticks([])
    ax[i,1].set_yticks([])
    ax[i,1].set_axis_off()
    ax[i,1].set_title(sessions[sesidx].celldata['cell_id'][idx[1]],fontsize=8)
    ax[i,0].text(0.9,0.6,'RF corr: %.2f' % corrmat_triu[idx],fontsize=8,transform=ax[i,0].transAxes)
fig.tight_layout()
fig.savefig(os.path.join(savedir,'ReducedRank_FitRF_example_RF_corr_%s.png' % sessions[sesidx].sessiondata['session_id'][0]),format='png',dpi=300,bbox_inches='tight')

#%% 

sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,remove_method=None)

# for ises in range(nSessions):
sessions[sesidx].resp_corr = np.corrcoef(sessions[sesidx].respmat)
# sessions[sesidx].resp_corr = np.corrcoef(resp)
np.fill_diagonal(sessions[sesidx].resp_corr,np.nan)

sessions[sesidx].resid_corr = np.corrcoef(resp.T - Y_hat.T)
np.fill_diagonal(sessions[sesidx].resid_corr,np.nan)

#%% Plot the relationship between similarity of RFs and similarity of responses:
#Expecting positive correlation of course

arealabelpairs = [np.array(['V1unl-V1unl',
                'V1unl-V1lab',
                'V1lab-V1lab']),
                  np.array(['PMunl-PMunl',
                'PMunl-PMlab',
                'PMlab-PMlab']),
                  np.array(['V1unl-PMunl',
                'V1unl-PMlab',
                'V1lab-PMunl',
                'V1lab-PMlab'])]

axtitles        = ['V1-V1','PM-PM','V1-PM']
min_relhalf     = 0
min_tuning_SNR  = 0.1
min_RF_R2       = 0.0

for datatype in ['sig_corr','resid_corr','resp_corr']:
    fig,axes = plt.subplots(1,3,figsize=(9,3),sharex=True,sharey=True)

    for iset,arealabelpair in enumerate(arealabelpairs):
        ax = axes[iset]
        for ial,alpair in enumerate(arealabelpair):

            al_source = alpair.split('-')[0]
            al_target = alpair.split('-')[1]

            idx_X = np.where(np.all((sessions[sesidx].celldata['arealabel']==al_source,
                                        sessions[sesidx].celldata['tuning_SNR']>min_tuning_SNR,
                                        sessions[sesidx].celldata['rel_half']>min_relhalf,
                                        sessions[sesidx].celldata['RF_R2']>min_RF_R2,
                                        sessions[sesidx].celldata['noise_level']<100),axis=0))[0]

            idx_Y = np.where(np.all((sessions[sesidx].celldata['arealabel']==al_target,
                                        sessions[sesidx].celldata['tuning_SNR']>min_tuning_SNR,
                                        sessions[sesidx].celldata['rel_half']>min_relhalf,
                                        sessions[sesidx].celldata['RF_R2']>min_RF_R2,
                                        sessions[sesidx].celldata['noise_level']<100),axis=0))[0]
            
            xdata = sessions[sesidx].RF_corrmat[np.ix_(idx_X,idx_Y)].flatten()
            # xdata = sessions[sesidx].sig_corr[np.ix_(idx_X,idx_Y)].flatten()
            ydata = getattr(sessions[sesidx],datatype)[np.ix_(idx_X,idx_Y)].flatten()
            
            ax.scatter(xdata,ydata,s=3,marker='.',c=get_clr_area_labelpairs([alpair]),alpha=0.5)
            
            notnan = ~np.isnan(ydata) & ~np.isnan(xdata)
            xdata = xdata[notnan]
            ydata = ydata[notnan]        

            slope, intercept, r_value, p_value, std_err = stats.linregress(xdata,ydata)
            
            x = np.linspace(np.min(xdata),np.max(xdata),100)
            y = slope*x + intercept
            
            ax.plot(x,y,c=get_clr_area_labelpairs([alpair]),lw=1)

        leg = ax.legend(arealabelpair,frameon=False,fontsize=9,loc='upper left')
        for i,t in enumerate(leg.texts):
            t.set_color(get_clr_area_labelpairs([arealabelpair[i]]))
        for handle in leg.legendHandles:
            handle.set_visible(False)
            
        ax.set_title(axtitles[iset],fontsize=12)
        if iset==0:
            ax.set_ylabel('Similarity of responses\n(%s)' % datatype,fontsize=10)
        if iset==1:
            ax.set_xlabel('Similarity of RFs',fontsize=10)

    ax.set_xlim([-1,1])
    ax.set_ylim([-0.2,0.6])
    sns.despine(offset=3,top=True,right=True)
    fig.tight_layout()
    plt.savefig(os.path.join(savedir,'ReducedRank_FitRF_%s_%s.png' % (datatype,sessions[sesidx].sessiondata['session_id'][0])),format='png',dpi=300,bbox_inches='tight')

#%%

plt.hist(sessions[sesidx].celldata['RF_R2'], bins=100, color='blue', alpha=0.5)

#%% Show correlations as a function of delta RF position
arealabelpairs = ['V1unl-V1unl',
                'V1unl-V1lab',
                'V1lab-V1lab',
                'PMunl-PMunl',
                'PMunl-PMlab',
                'PMlab-PMlab',
                'V1unl-PMunl',
                'V1unl-PMlab',
                'V1lab-PMunl',
                'V1lab-PMlab']

axordering = np.array([0,3,6,1,4,7,2,5,8,11])

min_relhalf     = 0
min_tuning_SNR  = 0.1
min_RF_R2       = 0

datatype       = 'resp_corr'
# datatype       = 'sig_corr'
# datatype       = 'resid_corr'

#Binning parameters 1D distance
binresolution   = 5
binlim          = 60
binedges_dist   = np.arange(-binresolution/2,binlim,binresolution)+binresolution/2 
binsdRF = binedges_dist[:-1]+binresolution/2 
nBins           = len(binsdRF)

bin_dist        = np.zeros((nSessions,nBins,len(arealabelpairs)))
bin_dist_count  = np.zeros((nSessions,nBins,len(arealabelpairs)))

rf_type         = 'RRR'
celldata        = sessions[sesidx].celldata
el              = celldata['rf_el_' + rf_type].to_numpy()
az              = celldata['rf_az_' + rf_type].to_numpy()

delta_el        = el[:,None] - el[None,:]
delta_az        = az[:,None] - az[None,:]

delta_rf        = np.sqrt(delta_az**2 + delta_el**2)

nquantiles      = 5
clrs_quantiles = sns.color_palette('magma',nquantiles)

fig,axes = plt.subplots(4,3,figsize=(6,7.5),sharex=True,sharey=True)
# ax = axes[iset]

for ial,alpair in enumerate(arealabelpairs):
    ax = axes.flatten()[axordering[ial]]
    al_source = alpair.split('-')[0]
    al_target = alpair.split('-')[1]

    idx_X = np.all((sessions[sesidx].celldata['arealabel']==al_source,
                            sessions[sesidx].celldata['tuning_SNR']>min_tuning_SNR,
                            sessions[sesidx].celldata['rel_half']>min_relhalf,
                            sessions[sesidx].celldata['RF_R2']>min_RF_R2,
                            sessions[sesidx].celldata['noise_level']<100),axis=0)

    idx_Y = np.all((sessions[sesidx].celldata['arealabel']==al_target,
                                sessions[sesidx].celldata['tuning_SNR']>min_tuning_SNR,
                                sessions[sesidx].celldata['rel_half']>min_relhalf,
                                sessions[sesidx].celldata['RF_R2']>min_RF_R2,
                                sessions[sesidx].celldata['noise_level']<100),axis=0)
    
    xdata = delta_rf[np.ix_(idx_X,idx_Y)].flatten()
    # xdata = sessions[sesidx].RF_corrmat[np.ix_(idx_X,idx_Y)].flatten()
    # xdata = sessions[sesidx].sig_corr[np.ix_(idx_X,idx_Y)].flatten()
    ydata = getattr(sessions[sesidx],datatype)[np.ix_(idx_X,idx_Y)].flatten()

    ax.scatter(xdata,ydata,s=2,marker='.',c=get_clr_area_labelpairs([alpair]),alpha=0.25)

    if len(ydata)>100:
        #Now 1D, so only by deltarf:
        bin_dist = binned_statistic(x=xdata,values=ydata,statistic=np.nanmean, bins=binedges_dist)[0]
        bin_dist_count = np.histogram(xdata,bins=binedges_dist)[0]

    ax.plot(binsdRF,bin_dist,color='k',linewidth=2)
    
    #show quantiles of correlations
    quantiles = np.linspace(0,1,nquantiles+2)[1:-1]
    quantiles = np.array([1,5,50,95,99])
    datatoplot = np.empty((nBins,nquantiles))
    for ibin,binlim in enumerate(binedges_dist[:-1]+binresolution/2):
        idx_quantile = np.all((xdata>binedges_dist[ibin],xdata<=binedges_dist[ibin+1]),axis=0)
        datatoplot[ibin,:] = np.nanpercentile(ydata[idx_quantile],quantiles)
    for iq in range(nquantiles):
        ax.plot(binsdRF,datatoplot[:,iq],color=clrs_quantiles[iq],linewidth=2)

    # #For N quantiles of similarity in RFs
    # quantiles = np.linspace(0,1,nquantiles+1)
    # zdata = sessions[sesidx].RF_corrmat
    # zquantiles = np.nanpercentile(zdata,quantiles*100)

    # for iq in range(nquantiles):
    #     idx_quantile = np.all((
    #                         zdata>zquantiles[iq],
    #                         zdata<zquantiles[iq+1],
    #                         np.outer(idx_X,idx_Y)),axis=0)

    #     xdata = delta_rf[idx_quantile].flatten()
    #     vdata = getattr(sessions[sesidx],datatype)[idx_quantile].flatten()

    #     if len(vdata)>100:
    #         bin_dist = binned_statistic(x=xdata,values=vdata,statistic=np.nanmean, bins=binedges_dist)[0]
    #         bin_dist_count = np.histogram(xdata,bins=binedges_dist)[0]

    #     ax.plot(binsdRF,bin_dist,color=clrs_quantiles[iq],linewidth=2)

    ax.set_title(alpair,fontsize=8,color=get_clr_area_labelpairs([alpair]))

ax.set_xlim([0,binlim])
ax.set_ylim([-0.2,0.6])
# ax.set_ylim([-0.02,0.05])
sns.despine(fig=fig, top=True, right=True,offset=3)
fig.tight_layout()
fig.savefig(os.path.join(savedir,'RRR_FitRF_corr_dRF_%s_%s.png' % (datatype,sessions[sesidx].sessiondata['session_id'][0])),format='png',dpi=300,bbox_inches='tight')






#%% On the trial to trial response: RRR to get RF

#NOTES:
# For the reconstruction it worked really well to zscore responses
# Then fit the data with lam=0.05, nranks=50, nsub=3
# Later update: reduced rank is not necessary, is only limiting, nsub2 is better but 
# slower. Lam depends on df/deconv and needs to be optimized with crossval. Furthermore 
# lambda biases towards low or high frequency reconstruction. 

nsub    = 3 #without subsampling really slow, i.e. nsub=1
lam     = 0.05

for ises, ses in enumerate(sessions):
    print(ses.session_id)
    resp    = ses.respmat.T

    K,N     = np.shape(resp)

    #normalize the response for each neuron to the maximum:
    resp        = zscore(resp, axis=0)

    # dividing by poprate:
    # resp = resp / np.mean(resp, axis=0, keepdims=True)

    IMdata      = natimgdata[:,:,ses.trialdata['ImageNumber']]

    # cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=0.05,nranks=100,nsub=nsub)
    ses.cRF,Y_hat   = linear_RF_cv(resp, IMdata, lam=lam, nsub=nsub)

    RF_R2 = r2_score(resp,Y_hat,multioutput='raw_values')
    ses.celldata['RF_R2'] = RF_R2
    print('RF R2: %0.2f' % (RF_R2.mean()))

    #Compute pairwise correlations matrix of the cRF:
    cRF_reshape             = np.reshape(ses.cRF, (np.shape(ses.cRF)[0]*np.shape(ses.cRF)[1],np.shape(ses.cRF)[2]))
    ses.RF_corrmat  = np.corrcoef(cRF_reshape.T)

    np.fill_diagonal(ses.RF_corrmat,np.nan)

    ses.resp_corr = np.corrcoef(ses.respmat)
    # sessions[sesidx].resp_corr = np.corrcoef(resp)
    np.fill_diagonal(ses.resp_corr,np.nan)

    ses.resid_corr = np.corrcoef(resp.T - Y_hat.T)
    np.fill_diagonal(ses.resid_corr,np.nan)

#%% 

#%% the relationship between similarity of RFs and similarity of responses:
#Expecting positive correlation of course

arealabelpairs = np.array(['V1unl-V1unl',
                'V1unl-V1lab',
                'V1lab-V1lab',
                'PMunl-PMunl',
                'PMunl-PMlab',
                'PMlab-PMlab',
                'V1unl-PMunl',
                'V1unl-PMlab',
                'V1lab-PMunl',
                'V1lab-PMlab'])

min_relhalf     = 0
min_tuning_SNR  = 0.1
min_RF_R2       = 0

datatypes = ['sig_corr','resid_corr','resp_corr']

RFcorr      = np.full((nSessions,len(arealabelpairs)),np.nan)
slopes      = np.full((nSessions,len(datatypes),len(arealabelpairs)),np.nan)
intercepts  = np.full((nSessions,len(datatypes),len(arealabelpairs)),np.nan)

for ises, ses in enumerate(sessions):
    for ial,alpair in enumerate(arealabelpairs):

        al_source = alpair.split('-')[0]
        al_target = alpair.split('-')[1]

        idx_X = np.where(np.all((sessions[ises].celldata['arealabel']==al_source,
                                    sessions[ises].celldata['tuning_SNR']>min_tuning_SNR,
                                    sessions[ises].celldata['rel_half']>min_relhalf,
                                    sessions[ises].celldata['RF_R2']>min_RF_R2,
                                    sessions[ises].celldata['noise_level']<20),axis=0))[0]

        idx_Y = np.where(np.all((sessions[ises].celldata['arealabel']==al_target,
                                    sessions[ises].celldata['tuning_SNR']>min_tuning_SNR,
                                    sessions[ises].celldata['rel_half']>min_relhalf,
                                    sessions[ises].celldata['RF_R2']>min_RF_R2,
                                    sessions[ises].celldata['noise_level']<20),axis=0))[0]
        
        if len(idx_X) <= 5 or len(idx_Y) <= 5:
            # slopes[ises,idt,ial] = np.nan
            continue

        xdata = sessions[ises].RF_corrmat[np.ix_(idx_X,idx_Y)].flatten()
        # RFcorr[ises,ial]        = np.nanmean(xdata)
        RFcorr[ises,ial]        = np.nanmean(np.abs(xdata))

        for idt,datatype in enumerate(datatypes):
            xdata = sessions[ises].RF_corrmat[np.ix_(idx_X,idx_Y)].flatten()

            ydata = getattr(sessions[ises],datatype)[np.ix_(idx_X,idx_Y)].flatten()
            
            notnan = ~np.isnan(ydata) & ~np.isnan(xdata)
            xdata = xdata[notnan]
            ydata = ydata[notnan]        

            slope, intercept, r_value, p_value, std_err = stats.linregress(xdata,ydata)
            
            x = np.linspace(np.min(xdata),np.max(xdata),100)
            y = slope*x + intercept

            slopes[ises,idt,ial]        = slope
            intercepts[ises,idt,ial]    = intercept



#%% 
pairs = [('V1unl-V1unl', 'V1unl-V1lab'),
         ('V1unl-V1unl', 'V1lab-V1lab'),
         ('PMunl-PMunl', 'PMunl-PMlab'),
         ('PMunl-PMunl', 'PMlab-PMlab'),
         ('V1unl-PMunl', 'V1lab-PMlab'),
         ('V1unl-PMunl', 'V1lab-PMunl'),
         ('V1unl-PMunl', 'V1unl-PMlab'),
        #  ('V1lab-PMunl', 'V1unl-PMlab'),
        #  ('V1lab-PMunl', 'V1lab-PMlab')
        #  ('V1unl-PMlab', 'V1lab-PMlab')
         ]

clrs_area_labelpairs = get_clr_area_labelpairs(arealabelpairs)

fig,axes = plt.subplots(1,len(datatypes),figsize=(len(datatypes)*3,3),sharey=True)
for idt,datatype in enumerate(datatypes):
    ax = axes[idt]
    df              = pd.DataFrame(data=slopes[:,idt,:],columns=arealabelpairs)
    sns.barplot(data=df,ax=ax,capsize=0,linewidth=0.5,palette=clrs_area_labelpairs)
   
    df              = df.dropna() 
    pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]]
    annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
    annotator.configure(test='t-test_paired', text_format='star', loc='inside',
                        line_height=0.05,line_offset_to_group=0.15,text_offset=0, 
                        line_width=0.25,comparisons_correction=None,verbose=False,
                        correction_format='replace',fontsize=8)
    annotator.apply_and_annotate()

    ax.set_xticks(range(len(arealabelpairs)))
    ax.set_xticklabels(arealabelpairs,rotation=90)

    ax.set_title(datatype)
    ax.set_ylabel('Slope')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
for ax in axes:
    ax.set_xticklabels(arealabelpairs,rotation=90,fontsize=6)

my_savefig(fig,savedir,'LinearRF_PairwiseCorrelations_Arealabelpairs_Slopes_%d' % nSessions,formats=['png'])


#%% 
fig,axes = plt.subplots(1,1,figsize=(3,3),sharey=True)
ax = axes

df              = pd.DataFrame(data=RFcorr,columns=arealabelpairs)
sns.barplot(data=df,ax=ax,capsize=0,linewidth=1,palette=clrs_area_labelpairs)

#Stats:
df              = df.dropna() 
pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]]
annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
annotator.configure(test='t-test_paired', text_format='star', loc='inside',
                    line_height=0.05,line_offset_to_group=0.05,text_offset=0, 
                    line_width=0.25,comparisons_correction=None,verbose=False,
                    correction_format='replace',fontsize=8)
annotator.apply_and_annotate()

ax.set_xticks(range(len(arealabelpairs)))
ax.set_xticklabels(arealabelpairs,rotation=90)

ax.set_ylabel('Receptive Field correlation')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
# for ax in axes:
ax.set_xticklabels(arealabelpairs,rotation=90,fontsize=6)

my_savefig(fig,savedir,'LinearRF_PairwiseCorrelation_Arealabelpairs_%d' % nSessions,formats=['png'])










#%% Do some optimizations: lambda,kfold,rank



#%% On the trial to trial response: RRR to get RF
sesidx  = 1
nsub    = 3
resp    = sessions[sesidx].respmat.T
#remove gain modulation by the population rate:
# resp = resp / np.mean(resp, axis=1, keepdims=True)
#normalize the response for each neuron to the maximum:
resp = zscore(resp, axis=0, nan_policy='omit')
# resp = resp / np.percentile(resp, 90,axis=0)

# dividing by poprate:
resp = resp / (np.mean(resp, axis=0, keepdims=True)+1e-8)

IMdata = natimgdata[:,:,sessions[sesidx].trialdata['ImageNumber']]

#%% Optimize lambda:
lams        = np.array([0,0.01,0.1,1,2,10])
lams        = np.logspace(-2,1,10)
EV_lams     = np.full((len(lams)),np.nan)

for i, lam in enumerate(lams):
    # cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=lam,nranks=50,nsub=nsub)
    cRF,Y_hat = linear_RF_cv(resp, IMdata,lam=lam,nsub=nsub)
    
    # EV_lams[i] = EV(resp,Y_hat)
    # EV_lams[i] = r2_score(resp,Y_hat)
    # EV_lams[i] = 1 - np.var(resp - Y_hat)/np.var(resp)
    EV_lams[i] = np.mean(r2_score(resp,Y_hat,multioutput='raw_values'))

fig, ax = plt.subplots()
plt.plot(lams,EV_lams)
plt.xticks(lams)
ax.set_xscale('log')
ax.set_ylim([-0.1,0.2])

#%% Rank increases EV, but flattens out around 25
ranks        = np.array([1,5,25,50,100,200])
EV_ranks     = np.full((len(ranks)),np.nan)

for i, rank in enumerate(ranks):
    cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=0.1,nranks=rank,nsub=nsub)
    # EV_ranks[i] = EV(resp,Y_hat)
    # EV_ranks[i] = r2_score(resp,Y_hat)
    # EV_lams[i] = 1 - np.var(resp - Y_hat)/np.var(resp)
    EV_ranks[i] = np.mean(r2_score(resp,Y_hat,multioutput='raw_values'))

plt.plot(ranks,EV_ranks)

#%% Is better at nsub 4, than 2 or 3
subs        = np.array([2,3,4,5])
EV_nsubs     = np.full((len(subs)),np.nan)

for i, nsub in enumerate(subs):
    cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=0.1,nranks=100,nsub=nsub,kfold=2)
    # EV_nsubs[i] = EV(resp,Y_hat)
    EV_nsubs[i] = np.mean(r2_score(resp,Y_hat,multioutput='raw_values'))

plt.plot(subs,EV_nsubs)





#%% On the trial to trial response: RRR to get RF

#NOTES:
# For the reconstruction it worked really well to divide by population rate, zscore
# Then fit the data with lam=0.05, nsub=3
# Later update: reduced rank is not necessary, is only limiting, nsub2 is better but 
# slower. Lam depends on df/deconv and needs to be optimized with crossval. Furthermore 
# lambda biases towards low or high frequency reconstruction. 

sesidx  = 11
print(sessions[sesidx].sessiondata['session_id'][0])
nsub    = 3 #without subsampling really slow, i.e. nsub=1
resp    = sessions[sesidx].respmat.T
lam     = 0.05

K,N     = np.shape(resp)

#remove gain modulation by the population rate:
# poprate = np.mean(resp, axis=1)
# popcoupling = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(N)]
# resp  = resp / np.outer(poprate,popcoupling)
#alternatively just by dividing by poprate:
# resp = resp / np.mean(resp, axis=0, keepdims=True)

#normalize the response for each neuron to the maximum:
resp = zscore(resp, axis=0)

IMdata = natimgdata[:,:,sessions[sesidx].trialdata['ImageNumber']]

# cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=0.05,nranks=100,nsub=nsub)
cRF,Y_hat = linear_RF_cv(resp, IMdata,lam=lam,nsub=nsub)

RF_R2 = r2_score(resp,Y_hat,multioutput='raw_values')
sessions[sesidx].celldata['RF_R2'] = RF_R2
plt.hist(RF_R2,bins=100)
print(np.mean(RF_R2))

#%% Show the difference in the distribution of the responses for predicted vs true data:
plt.hist(Y_hat.flatten(), bins=np.arange(-1,1,0.01), color='red', alpha=0.5)
plt.hist(resp.flatten(), bins=np.arange(-1,1,0.01), color='blue', alpha=0.5)

#%% Show example neurons for different populations:
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
narealabels     = len(arealabels)
nN              = 6 #number of example neurons to show

fig,axes = plt.subplots(nN,narealabels,figsize=(3*narealabels,1*nN))
for ial,arealabel in enumerate(arealabels):
    idx_neurons = np.where(np.all((sessions[sesidx].celldata['arealabel']==arealabel,
                                   ),axis=0))[0]
    idx_neurons = idx_neurons[np.argsort(-sessions[sesidx].celldata['RF_R2'][idx_neurons])][:nN]

    for i,iN in enumerate(idx_neurons):
        ax = axes[i,ial]
        lim = np.max(np.abs(cRF[:,:,iN]))*1.2
        ax.imshow(cRF[:,:,iN],cmap='bwr',vmin=-lim,
                            vmax=lim)
        # ax.plot(sessions[sesidx].celldata['rf_az_RRR'][iN]/nsub,
                # sessions[sesidx].celldata['rf_el_RRR'][iN]/nsub,'k+',markersize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.set_title(arealabel + sessions[sesidx].celldata['cell_id'][iN],fontsize=8)

fig.tight_layout()
my_savefig(fig,savedir,sessions[sesidx].sessiondata['session_id'][0] + '_RFs')

#%% Reconstruct images from the RF:

idx_N = RF_R2>0.01
idx_N = RF_R2>0.0

# arealabel = 'V1lab'
# arealabel = 'V1unl'
# arealabel = 'PMunl'
# arealabel = 'PMlab'
# idx_N = sessions[sesidx].celldata['arealabel']==arealabel
# print(np.sum(idx_N))

IMdata              = natimgdata[:,:,sessions[sesidx].trialdata['ImageNumber']]
IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images

resp_F              = copy.deepcopy(resp)
# plt.hist(resp_F.flatten(),bins=np.arange(-0.5,5.5,0.01),color='blue',alpha=0.5)

resp_F = np.clip(resp_F,np.percentile(resp_F,0),np.percentile(resp_F,99.99))
IMdata_hat          = np.tensordot(cRF[:,:,idx_N],resp_F[:,idx_N],axes=[2,1])

#%% Average the reconstructions for each image:
unique_images   = np.unique(sessions[sesidx].trialdata['ImageNumber'])
nUnImages       = len(unique_images)

Lx              = IMdata.shape[0]
Ly              = IMdata.shape[1]
IMdata_mean     = np.zeros((Lx,Ly,nUnImages))
IMdata_hat_mean = np.zeros((Lx,Ly,nUnImages))

for iIM,IM in enumerate(unique_images):
    IMdata_mean[:,:,iIM]        = np.mean(IMdata[:,:,sessions[sesidx].trialdata['ImageNumber']==IM],axis=2)
    IMdata_hat_mean[:,:,iIM]    = np.mean(IMdata_hat[:,:,sessions[sesidx].trialdata['ImageNumber']==IM],axis=2)

R2_IM   = np.zeros(nUnImages)
for im in range(nUnImages):
    R2_IM[im] = np.corrcoef(IMdata_mean[:,:,im].flatten(),IMdata_hat_mean[:,:,im].flatten())[0,1]
    # R2_IM[im] = np.corrcoef(IMdata_mean[:,:,im].flatten(),IMdata_hat_mean[:,:,im].flatten())[0,1]**2
plt.hist(R2_IM,bins=100)
print(np.mean(R2_IM))

#%% 
neximages = 8
eximages = np.argsort(-R2_IM)[:neximages]
# eximages = np.argsort(R2_IM)[:neximages]
fig,axes = plt.subplots(neximages,2,figsize=(3,neximages))
for iIM,IM in enumerate(eximages):
    ax = axes[iIM,0]
    ax.imshow(IMdata_mean[:,:,IM],cmap='gray',aspect=1)
    ax.axis('off')
    if iIM==0: 
        ax.set_title('True Image')
    
    # reconstruction = np.tensordot(cRF,resp[im,:],axes=[2,0])
    ax = axes[iIM,1]
    ax.imshow(IMdata_hat_mean[:,:,IM],cmap='gray',aspect=1)
    ax.axis('off')
    if iIM==0: 
        ax.set_title('Reconstruction')
plt.savefig(os.path.join(savedir,'BestReconstructions_%s_%s.png' % (arealabel,sessions[sesidx].sessiondata['session_id'][0])),format='png',dpi=300,bbox_inches='tight')

#%% 
IMdata_mean_hp = np.zeros(IMdata_mean.shape)
for im in range(nUnImages):
    IMdata_mean_hp[:,:,im] = IMdata_mean[:,:,im]-ndimage.gaussian_filter(IMdata_mean[:,:,im],sigma=2.5)

fig,axes = plt.subplots(neximages,2,figsize=(3,neximages))
for iIM,IM in enumerate(eximages):
    ax = axes[iIM,0]
    ax.imshow(IMdata_mean[:,:,IM],cmap='gray',aspect=1)
    ax.axis('off')
    ax = axes[iIM,1]
    ax.imshow(IMdata_mean_hp[:,:,IM],cmap='gray',aspect=1)
    ax.axis('off')

#%% 
R2_IM_hp   = np.zeros(nUnImages)
R2_IM   = np.zeros(nUnImages)
for im in range(nUnImages):
    R2_IM_hp[im] = np.corrcoef(IMdata_mean_hp[:,:,im].flatten(),IMdata_hat_mean[:,:,im].flatten())[0,1]
    R2_IM[im] = np.corrcoef(IMdata_mean[:,:,im].flatten(),IMdata_hat_mean[:,:,im].flatten())[0,1]
    # R2_IM[im] = np.corrcoef(IMdata_mean[:,:,im].flatten(),IMdata_hat_mean[:,:,im].flatten())[0,1]**2
plt.hist(R2_IM,bins=100,alpha=0.5)
plt.hist(R2_IM_hp,bins=100,alpha=0.5)
print(np.mean(R2_IM))
plt.legend(['No HP','HP'])


#%% Show the reconstruction of the same images with different populations: 








#%% Get the correlation between the origial and reconstructed images per trial:
IMdata              = natimgdata[:,:,sessions[sesidx].trialdata['ImageNumber']]
IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images

R2_pertrial   = np.zeros(5600)
for im in range(5600):
    R2_pertrial[im] = np.corrcoef(IMdata[:,:,im].flatten(),IMdata_hat[:,:,im].flatten())[0,1]
plt.hist(R2_pertrial,bins=100)

eximages = np.argsort(-R2_pertrial)[:neximages]

#%% 
fig,axes = plt.subplots(neximages,2,figsize=(3,neximages))
for iIM,IM in enumerate(eximages):
    ax = axes[iIM,0]
    # ax.imshow(IMdata_mean_hp[:,:,IM],cmap='gray',aspect=1)
    ax.imshow(IMdata[:,:,IM],cmap='gray',aspect=1)
    ax.axis('off')
    if iIM==0: 
        ax.set_title('True Image')
    
    # reconstruction = np.tensordot(cRF,resp[im,:],axes=[2,0])
    ax = axes[iIM,1]
    ax.imshow(IMdata_hat[:,:,IM],cmap='gray',aspect=1)
    ax.axis('off')
    if iIM==0: 
        ax.set_title('Reconstruction')

#%% Plot the R2_pertrial over the courrse of the whole session, i.e. per trial:
fig = plt.figure(figsize=(12,3))
# plt.plot(R2_pertrial,color='k',linewidth=0.5)
plt.plot(R2_pertrial,marker='.',markersize=1,color='k',linewidth=0.1)
plt.xlabel('Trial #')
plt.xticks([0,2800,5600])
plt.xlim([0,5600])
plt.ylim([0,0.8])
plt.ylabel('R2 per trial')
plt.title('Reconstruction')
sns.despine(offset=3,top=True,right=True,trim=True)
plt.savefig(os.path.join(savedir,'R2_pertrial_session_%s.png' % (sessions[sesidx].sessiondata['session_id'][0])),format='png',dpi=300,bbox_inches='tight')

#%% Is reconstruction correlated to different variables: 
from scipy.stats import pearsonr, binned_statistic

poprate = np.mean(zscore(sessions[sesidx].respmat.T,axis=0), axis=1)

varlabels = np.array(['trial number','poprate','runspeed','pupil area','video ME'])
varcolors = ['b','g','r','c','m']
vardata = np.stack((sessions[sesidx].trialdata['TrialNumber'],
                        poprate,
                        sessions[sesidx].respmat_runspeed,
                        sessions[sesidx].respmat_pupilarea,
                        sessions[sesidx].respmat_videome,))

fig,axes = plt.subplots(1,len(varlabels),figsize=(12,3),sharey=True)

for ivar,var in enumerate(varlabels):
    xdata = vardata[ivar,:]
    ax = axes[ivar]
    # ax.scatter(xdata,R2_pertrial,c=varcolors[ivar],s=2,alpha=0.5)
    gdat = binned_statistic(xdata,R2_pertrial,bins=10)
    binmeans = (gdat.bin_edges[1:]+gdat.bin_edges[:-1])/2
    ax.plot(binmeans,gdat.statistic,color='k',linewidth=2)
    ax.scatter(xdata,R2_pertrial,c=varcolors[ivar],s=2,alpha=0.5)
    r,p = pearsonr(xdata,R2_pertrial)

    ax.text(0.05,0.95,'r=%.2f, p=%.2e' % (r,p),transform=ax.transAxes)
    ax.set_xlabel(var)
    ax.set_ylim([0,1])
    ax.set_xlim(np.percentile(xdata,0.2),np.percentile(xdata,99.8))
    # ax_nticks(ax,5)
    if ivar==0: 
        ax.set_ylabel('R2 per trial')

fig.tight_layout()
sns.despine(offset=3,top=True,right=True,trim=True)
my_savefig(fig,savedir,'R2_pertrial_vs_variables_%s.png' % (sessions[sesidx].sessiondata['session_id'][0]))

#%% 
sessions[sesidx].celldata = fit_2dgauss_cRF(cRF, nsub=nsub,celldata=sessions[sesidx].celldata)



#%% 

 #####     #    ### #     #    ######  ####### 
#     #   # #    #  ##    #    #     # #       
#        #   #   #  # #   #    #     # #       
#  #### #     #  #  #  #  #    ######  #####   
#     # #######  #  #   # #    #   #   #       
#     # #     #  #  #    ##    #    #  #       
 #####  #     # ### #     #    #     # #       

#%% ################################################
session_list        = np.array([['LPE11086_2023_12_16']])
session_list        = np.array([['LPE13959_2025_02_24']])
session_list        = np.array([['LPE10885_2023_10_20']])

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['IM'],only_session_id=session_list)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_cells=1)

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)

#%% 


#%% On the trial to trial response: linear regression to get RF
sesidx  = 0
print(sessions[sesidx].session_id)
nsub    = 3 #without subsampling really slow, i.e. nsub=1
resp    = sessions[sesidx].respmat.T
lam     = 0.05
lam     = 0.2

K,N     = np.shape(resp)

#normalize the response for each neuron to the maximum:
resp = zscore(resp, axis=0)

IMdata = natimgdata[:,:,sessions[sesidx].trialdata['ImageNumber']]

# cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=0.05,nranks=100,nsub=nsub)
cRF,Y_hat = linear_RF_cv(resp, IMdata,lam=lam,nsub=nsub)

RF_R2 = r2_score(resp,Y_hat,multioutput='raw_values')
sessions[sesidx].celldata['RF_R2'] = RF_R2
plt.hist(RF_R2,bins=100)
print(np.mean(RF_R2))

#%% 
sessions[sesidx].celldata = fit_2dgauss_cRF(cRF, nsub, sessions[sesidx].celldata)


#%% Show some linear RF estimates for neurons with different coupling quantiles: 
nexamplecells           = 6
npopcouplingquantiles   = 5

quantiles               =  np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,npopcouplingquantiles+1))

fig,axes = plt.subplots(nexamplecells,npopcouplingquantiles,figsize=(npopcouplingquantiles*1.5,nexamplecells*0.8),sharey=True,sharex=True)
for iqrpopcoupling in range(npopcouplingquantiles):
    
    idx_N = np.where(np.all((ses.celldata['pop_coupling'] > quantiles[iqrpopcoupling], 
                             ses.celldata['pop_coupling'] < quantiles[iqrpopcoupling+1],
                             ses.celldata['RF_R2'] > 0.05
                             ), axis=0))[0]
    excells = np.random.choice(idx_N,nexamplecells,replace=False)

    for iN,N in enumerate(excells):
        # in range(nexamplecells):
        ax = axes[iN,iqrpopcoupling]
        lim = np.max(np.abs(cRF[:,:,N]))*1.2
        ax.imshow(cRF[:,:,N],cmap='bwr',vmin=-lim,
                            vmax=lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        # ax.imshow(cRF[:,:,icell+iqrpopcoupling*nexamplecells],vmin=-0.5,vmax=0.5,cmap='RdBu')

plt.suptitle('RFs for neurons with different population coupling levels',fontsize=11)
plt.tight_layout()
my_savefig(fig,savedir,'RFs_by_popcoupling_%s' % sessions[sesidx].session_id, formats = ['png'])

#%% 
fields = np.array(['RF_R2','rf_az_RRR','rf_el_RRR','rf_sx_RRR','rf_sy_RRR','rf_r2_RRR'])
nfields = len(fields)

idx_N = ses.celldata['RF_R2'] > 0.025

fig,axes = plt.subplots(2,nfields//2,figsize=(3*nfields/2,2.5*nfields/3))
axes = axes.flatten()
for ifield in range(nfields):
    ax = axes[ifield]
    x = ses.celldata['pop_coupling'][idx_N]
    y = ses.celldata[fields[ifield]][idx_N]
    ax.scatter(x,y,s=0.5)
    ax.set_xlabel('pop coupling')
    ax.set_ylabel(fields[ifield])
    ax.set_xlim(np.nanpercentile(x,[0.1,99.5]))
    ax.set_ylim(np.nanpercentile(y,[0.1,99.5]))
    ax.text(0.7,0.9,'r = %.2f' % (ses.celldata[[fields[ifield],'pop_coupling']].corr().to_numpy()[0,1]),
            transform=ax.transAxes)
sns.despine(trim=False,top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'PopCouplingBy_RF_params_%s' % sessions[sesidx].session_id, formats = ['png'])

#%% 
x = ses.celldata['pop_coupling']
y = ses.celldata['rf_sx_RRR'] * ses.celldata['rf_sy_RRR']
idx_N = (ses.celldata['RF_R2'] > 0.025) & (y<500)
idx_N = np.all((ses.celldata['RF_R2'] > 0.025,
                y<500,
                ses.celldata['roi_name'] == 'V1'
                ),axis=0)
# idx_N = (ses.celldata['RF_R2'] > 0.025) & (ses.celldata['roi_name'] == 'V1')
# idx_N = (ses.celldata['rf_r2_RRR'] > 0.6) & (y<500)

# y = ses.celldata['rf_sx_RRR']
# idx_N = (ses.celldata['RF_R2'] > 0.025) & (y<25)
# idx_N = (ses.celldata['rf_r2_RRR'] > 0.5) & (y<40)

fig,axes = plt.subplots(1,1,figsize=(3,2.5))
ax = axes
x = x[idx_N]
y = y[idx_N]
mask = ~np.isnan(x) & ~np.isnan(y)
x = np.array(x[mask])
y = np.array(y[mask])
ax.scatter(x,y,s=0.5)
ax.set_xlabel('pop coupling')
ax.set_ylabel('RF size')


# define the model
model = HuberRegressor()
X = x.reshape((len(x), 1))

model.fit(X, y)
ypred = model.predict(X)
xaxis = arange(X.min(), X.max(), 0.01)
yaxis = model.predict(xaxis.reshape((len(xaxis), 1)))
plt.plot(xaxis, yaxis, color='r')
r2 = r2_score(y,ypred)
r = np.sqrt(r2)
# r = r2

# results = evaluate_model(X, y, model)
# print('Mean MAE: %.3f (%.3f)' % (mean(results), std(results)))
# plot the line of best fit
# plot_best_fit(X, y, model)

# ax.set_xlim(np.nanpercentile(x,[0.1,99]))
# ax.set_ylim(np.nanpercentile(y,[0,99]))
ax.text(0.7,0.9,'r = %.2f' % r,
            transform=ax.transAxes)
sns.despine(trim=False,top=True,right=True,offset=3)
plt.tight_layout()
# my_savefig(fig,savedir,'PopCouplingBy_RF_params_%s' % sessions[sesidx].session_id, formats = ['png'])


#%% 





#%% On the trial to trial response: linear regression to get RF
#But separate based on trials with different population rate
sesidx  = 0
print(sessions[sesidx].session_id)
nsub    = 3 #without subsampling really slow, i.e. nsub=1
resp    = sessions[sesidx].respmat.T
lam     = 0.05
lam     = 0.1
lam     = 0.3

K,N     = np.shape(resp)

#normalize the response for each neuron:
resp = zscore(resp, axis=0)

#remove gain modulation by the population rate:
poprate = np.mean(resp, axis=1)

IMdata = natimgdata[:,:,sessions[sesidx].trialdata['ImageNumber']]

nquantiles = 3
quantiles = np.percentile(poprate,np.linspace(0,100,nquantiles+1))

cRF,Y_hat = linear_RF_cv(resp[:10,:], IMdata[:,:,:10],lam=lam,nsub=nsub)

cRF_quantiles       = np.empty((nquantiles,)+np.shape(cRF))
RF_R2_quantiles     = np.empty((nquantiles,N))
for iqr in range(nquantiles):
    idx_T = np.where(np.all((poprate > quantiles[iqr], 
                             poprate < quantiles[iqr+1]), axis=0))[0]
    print(len(idx_T))
    
    cRF_quantiles[iqr,:,:,:],Y_hat = linear_RF_cv(resp[idx_T,:], IMdata[:,:,idx_T],lam=lam,nsub=nsub)
    
    RF_R2_quantiles[iqr,:] = r2_score(resp[idx_T,:],Y_hat,multioutput='raw_values')

print(np.mean(RF_R2_quantiles))


#%% Show some linear RF estimates for the same neurons with different population rates:

nexamplecells           = 6

idx_N = np.where(np.nanmean(RF_R2_quantiles, axis=0) > 0.1)[0]

# idx_N = np.where(np.all((np.nanmean(RF_R2_quantiles, axis=0) > 0.05,
                        # ses.celldata['pop_coupling']>0.2), axis=0))[0]

excells = np.random.choice(idx_N,nexamplecells,replace=False)

fig,axes = plt.subplots(nexamplecells,nquantiles,figsize=(nquantiles*1.5,nexamplecells*0.8),sharey=True,sharex=True)
for iqr in range(nquantiles):
    for iN,N in enumerate(excells):
        # in range(nexamplecells):
        ax = axes[iN,iqr]
        lim = np.max(np.abs(cRF_quantiles[iqr,:,:,N]))*1.2
        ax.imshow(cRF_quantiles[iqr,:,:,N],cmap='bwr',vmin=-lim,
                            vmax=lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        # ax.imshow(cRF[:,:,icell+iqrpopcoupling*nexamplecells],vmin=-0.5,vmax=0.5,cmap='RdBu')
plt.suptitle('RFs estimated from trials with different population rates\n (same neurons)',fontsize=11)
plt.tight_layout()
my_savefig(fig,savedir,'RFs_by_poprate_%s' % sessions[sesidx].session_id, formats = ['png'])

#%% 
fig,axes = plt.subplots(1,1,figsize=(3,3),sharey=True,sharex=True)
ax = axes
for iqr in range(nquantiles):
    ax.scatter(np.random.randn(len(RF_R2_quantiles[iqr,:]))*0.1+iqr,RF_R2_quantiles[iqr,:],
               marker='.',color='k',alpha=0.2)
    ax.errorbar(iqr,np.mean(RF_R2_quantiles[iqr,:]),
                yerr=np.std(RF_R2_quantiles[iqr,:]),linewidth=2,marker='o',markerfacecolor='w')   
ax.set_xticks(np.arange(nquantiles))
ax.set_xlabel('Pop rate quantiles (trial set)')
ax.set_ylabel('RF R2')
        # ax.imshow(cRF[:,:,icell+iqrpopcoupling*nexamplecells],vmin=-0.5,vmax=0.5,cmap='RdBu')
# plt.suptitle('RFs estimated from trials with different population rates\n (same neurons)',fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'RFs_quant_by_poprate_%s' % sessions[sesidx].session_id, formats = ['png'])




#%% On the trial to trial response: linear regression to get RF



#%% On the trial to trial response: RRR to get RF

#NOTES:
# For the reconstruction it worked really well to zscore responses
# Then fit the data with lam=0.05, nranks=50, nsub=3
# Later update: reduced rank is not necessary, is only limiting, nsub2 is better but 
# slower. Lam depends on df/deconv and needs to be optimized with crossval. Furthermore 
# lambda biases towards low or high frequency reconstruction. 

nsub                = 3 #without subsampling really slow, i.e. nsub=1
lam                 = 0.1
ncouplingbins       = 5
ReconR2             = np.empty((nSessions,5600))
ReconR2_popcoupling = np.empty((nSessions,5600,ncouplingbins))
maxnoiselevel       = 20

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting linear RF for each session'):
    binedges_popcoupling    = np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,ncouplingbins+1))

    resp    = ses.respmat.T

    K,N     = np.shape(resp)

    #normalize the response for each neuron to the maximum:
    resp            = zscore(resp, axis=0)

    IMdata          = natimgdata[:,:,ses.trialdata['ImageNumber']]

    # ses.cRF,Y_hat   = linear_RF_cv(resp, IMdata, lam=lam, nsub=nsub)
    ses.cRF,Y_hat   = linear_RF(resp, IMdata, lam=lam, nsub=nsub)
    
    IMdata          = IMdata[::nsub, ::nsub, :]         #subsample the natural images

    # RF_R2 = r2_score(resp,Y_hat,multioutput='raw_values')
    # ses.celldata['RF_R2'] = RF_R2
    # print('RF R2: %0.2f' % (RF_R2.mean()))

    # Reconstruct images from the RF:
    # idx_N = RF_R2>0.01
    idx_N = np.ones(N,dtype=bool)

    resp_F              = copy.deepcopy(resp)
    resp_F              = np.clip(resp_F,np.percentile(resp_F,0),np.percentile(resp_F,99.99))
    IMdata_hat          = np.tensordot(ses.cRF[:,:,idx_N],resp_F[:,idx_N],axes=[2,1])

    for im in range(5600):
        ReconR2[ises,im] = np.corrcoef(IMdata[:,:,im].flatten(),IMdata_hat[:,:,im].flatten())[0,1]
    
    for icp in range(ncouplingbins):
        # idx_N = np.all((ses.celldata['pop_coupling'] >= binedges_popcoupling[icp],
        #                 ses.celldata['pop_coupling'] <= binedges_popcoupling[icp+1]), axis=0)
        
        idx_N = np.all((
                        ses.celldata['pop_coupling'] >= binedges_popcoupling[icp],
                        ses.celldata['pop_coupling'] <= binedges_popcoupling[icp+1],
                        # ses.celldata['noise_level'] < maxnoiselevel,
                        ses.celldata['roi_name'] == 'V1'
                        ), axis=0)

        IMdata_hat          = np.tensordot(ses.cRF[:,:,idx_N],resp_F[:,idx_N],axes=[2,1])

        for im in range(5600):
            ReconR2_popcoupling[ises,im,icp] = np.corrcoef(IMdata[:,:,im].flatten(),IMdata_hat[:,:,im].flatten())[0,1]


#%% 
for ises,ses in tqdm(enumerate(sessions),desc='Computing population rate for each session'):
    resp    = ses.respmat.T
    #normalize the response for each neuron to the maximum:
    resp            = zscore(resp, axis=0)
    ses.poprate = np.mean(resp, axis=1)

#%% Show the reconstruction as a function of population rate: 

fig,axes = plt.subplots(4,4,figsize=(10,10))
axes = axes.flatten()
for ises,ses in enumerate(sessions):
    ax = axes[ises]
    ax.scatter(ses.poprate,ReconR2[ises,:],color='black',alpha=0.5,s=1)
    ax.set_title('%s' % ses.session_id)

#%%
sameIM_poprate              = np.full((nSessions,100,10),np.nan)
sameIM_reconR2              = np.full((nSessions,100,10),np.nan)
sameIM_reconR2_popcoupling  = np.full((nSessions,100,10,ncouplingbins),np.nan)

for ises,ses in enumerate(sessions):
    #Identify all trials with images that are repeated 10 times in the session:
    im_repeats  = ses.trialdata['ImageNumber'].value_counts()[ses.trialdata['ImageNumber'].value_counts()==10].index.to_numpy()
    
    for iIM,IM in enumerate(im_repeats):
        idx_T = ses.trialdata['ImageNumber'] == IM
        sameIM_poprate[ises,iIM,:] = ses.poprate[idx_T]
        sameIM_reconR2[ises,iIM,:] = ReconR2[ises,idx_T]
        sameIM_reconR2_popcoupling[ises,iIM,:] = ReconR2_popcoupling[ises,idx_T,:]

#%% 
clrs_popcoupling    = sns.color_palette('magma',ncouplingbins)

coeflabels = ['slope','intercept','rvalue','pvalue']
coefdata = np.full((nSessions,ncouplingbins,len(coeflabels)),np.nan)

# fig,axes    = plt.subplots(4,4,figsize=(10,10),sharex=True,sharey=True)
fig,axes    = plt.subplots(1,5,figsize=(13,3),sharex=True,sharey=True)
axes        = axes.flatten()
idx_ses = np.where(np.any(~np.isnan(sameIM_reconR2),axis=(1,2)))[0]
handles     = np.empty(ncouplingbins,dtype=object)
    
for iax,ises in enumerate(idx_ses):
    ax = axes[iax]
    if np.any(~np.isnan(sameIM_reconR2[ises,:,:])):
        for icp in range(ncouplingbins):
            # for iIM in range(100):
            #     ax.scatter(sameIM_poprate[ises,iIM,:],sameIM_reconR2_popcoupling[ises,iIM,:,icp],
            #                color=clrs_popcoupling[icp],alpha=0.5,s=1)

            xdata = sameIM_poprate[ises,:,:]
            ydata = sameIM_reconR2_popcoupling[ises,:,:,icp]

            xdata = zscore(xdata,axis=1,nan_policy='omit')

            xdata = xdata.flatten()
            ydata = ydata.flatten()

            notnan = ~np.isnan(ydata) & ~np.isnan(xdata)
            xdata = xdata[notnan]
            ydata = ydata[notnan] 

            # xdata = np.clip(xdata,np.percentile(xdata,1),np.percentile(xdata,99))
            notextreme = xdata < np.percentile(xdata,99)
            xdata = xdata[notextreme]
            ydata = ydata[notextreme]

            ax.scatter(xdata,ydata,s=3,marker='.',c=clrs_popcoupling[icp],alpha=0.5)

            slope, intercept, r_value, p_value, std_err = stats.linregress(xdata,ydata)
            coefdata[ises,icp,:] = [slope,intercept,r_value,p_value]
            x = np.linspace(np.min(xdata),np.max(xdata),100)
            y = slope*x + intercept
            
            handles[icp] = ax.plot(x,y,c=clrs_popcoupling[icp],lw=2)[0]
        ax.set_title('%s' % sessions[ises].session_id)
        if iax == 0: 
            ax.set_ylabel('Recon. R2')
        if iax == len(idx_ses)//2:
            ax.set_xlabel('Population rate (z-scored within image repetitions)')
        if iax==len(idx_ses)-1:
            ax.legend(handles,['0-20%','20-40%','40-60%','60-80%','80-100%'],
                    reverse=True,fontsize=8,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')

plt.tight_layout()
sns.despine(top=True,right=True,offset=3,trim=False)
my_savefig(fig,savedir,'Scatter_ReconR2_vs_poprate_sameIM_%dsessions' % (nSessions),formats=['png'])

#%% Show the coefficients of the linear regression fits across sessions and coupling bins
df = pd.DataFrame(coefdata.reshape((nSessions*ncouplingbins,len(coeflabels))),
                  columns=coeflabels)
df['popcoupling'] = np.tile(np.arange(ncouplingbins),nSessions)

fig,axes    = plt.subplots(1,4,figsize=(9,3),sharex=True)
ax = axes
for icoef,coef in enumerate(coeflabels):
    ax = axes[icoef]
    sns.pointplot(data=df,x='popcoupling',y=coef,color=clrs_popcoupling[icp],ax=ax,
                  errorbar=('ci', 95),join=False,palette=clrs_popcoupling)
    ax.set_xlabel('Pop. Coupling')
    ax.set_ylabel(coef)
    # ax.set_ylim([0,1])
plt.tight_layout()
sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'ReconR2_vs_popcoupling_sameIM_%dsessions' % (nSessions),formats=['png'])

#%% Compute correlation between population rate and reconstruction R2 for each image
# to select a good example


#%% 
coeflabels = ['slope','intercept','rvalue','pvalue']
coefdata = np.full((nSessions,100,ncouplingbins,len(coeflabels)),np.nan)
   
for ises,ses in enumerate(sessions):
    if np.any(~np.isnan(sameIM_reconR2[ises,:,:])):
        for icp in range(ncouplingbins):
            for iIM in range(100):
                xdata = sameIM_poprate[ises,iIM,:]
                ydata = sameIM_reconR2_popcoupling[ises,iIM,:,icp]

                slope, intercept, r_value, p_value, std_err = stats.linregress(xdata,ydata)
                coefdata[ises,iIM,icp,:] = [slope,intercept,r_value,p_value]

#%% Find a session and image with good R2 and with significant improvement for high coupling units
# and not for low coupling units

meanR2          = np.mean(sameIM_reconR2_popcoupling,axis=(2,3))
corrdiff        = coefdata[:,:,4,2] - coefdata[:,:,0,2]

idx = np.where(np.all((
            meanR2>np.nanpercentile(meanR2,80),
            corrdiff>np.nanpercentile(corrdiff,70),
                ),axis=0))

#%% 
ncouplingbins       = 5
maxnoiselevel       = 20

exampleID           = 1

ises                = idx[0][exampleID]
iIM                 = idx[1][exampleID]

ses                     = sessions[ises]
binedges_popcoupling    = np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,ncouplingbins+1))

resp                = ses.respmat.T

K,N                 = np.shape(resp)

#normalize the response for each neuron to the maximum:
resp            = zscore(resp, axis=0)

IMdata          = natimgdata[:,:,ses.trialdata['ImageNumber']]
IMdata          = IMdata[::nsub, ::nsub, :]         #subsample the natural images

# Reconstruct images from the RF:
resp_F              = copy.deepcopy(resp)
resp_F              = np.clip(resp_F,np.percentile(resp_F,0),np.percentile(resp_F,99.99))

#%% 

im_repeats  = ses.trialdata['ImageNumber'].value_counts()[ses.trialdata['ImageNumber'].value_counts()==10].index.to_numpy()
IM = im_repeats[iIM]
idx_T = np.where(ses.trialdata['ImageNumber'] == IM)[0]
assert(len(idx_T) == 10), 'There should be 10 trials for this image'

idx_T = idx_T[np.argsort(ses.poprate[idx_T])]

fig,axes = plt.subplots(1,1,figsize=(3,3))
ax              = axes
IMdata          = natimgdata[:,:,ses.trialdata['ImageNumber']]
ax.imshow(IMdata[:,:,idx_T[0]],cmap='gray')
ax.set_title('Image %d' % IM)
ax.set_xticks([])
ax.set_yticks([])
my_savefig(fig,savedir,'ExampleImage_%d' % (IM),formats=['png'])

fig,axes = plt.subplots(ncouplingbins,10,figsize=(10,5),sharex=True,sharey=True)
for icp in range(ncouplingbins):
    idx_N = np.all((
                    ses.celldata['pop_coupling'] >= binedges_popcoupling[icp],
                    ses.celldata['pop_coupling'] <= binedges_popcoupling[icp+1],
                    ses.celldata['noise_level'] < maxnoiselevel,
                    ses.celldata['roi_name'] == 'V1'
                    ), axis=0)

    IMdata_hat      = np.tensordot(ses.cRF[:,:,idx_N],resp_F[:,idx_N],axes=[2,1])

    for imrep in range(10):
        ax = axes[icp,imrep]
        # ax.imshow(IMdata[:,:,idx_T[imrep]],cmap='gray',vmin=0,vmax=1)
        # ax.imshow(IMdata_hat[:,:,idx_T[imrep]],cmap='gray',vmin=0,vmax=256)
        ax.imshow(IMdata_hat[:,:,idx_T[imrep]],cmap='gray',
                  vmin=-np.percentile(IMdata_hat[:,:,idx_T[imrep]],99),
                  vmax=np.percentile(IMdata_hat[:,:,idx_T[imrep]],99))
        # ax.imshow(IMdata_hat[:,:,idx_T[imrep]],cmap='gray',vmin=0,vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
my_savefig(fig,savedir,'ExampleImage_couplingbins_reconstruction_%d_%s' % (IM,ses.session_id),formats=['png'])

#%% 


