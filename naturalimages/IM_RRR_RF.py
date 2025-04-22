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
from scipy.sparse.linalg import svds
from scipy import stats
from sklearn.metrics import r2_score

from preprocessing.locate_rf import *
from loaddata.session_info import filter_sessions,load_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.imagelib import load_natural_images #
from utils.tuning import *
from utils.RRRlib import *
from utils.corr_lib import compute_signal_noise_correlation

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\')

#%%
# TODO:
# compute noise correlations for the sessions with 100 images repeated 10 times
# loop over sessions with all neurons included
# find regression coeff for each session with enough labeled cells

#%% ################################################
session_list        = np.array([['LPE11086','2023_12_16']])

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['IM'],only_session_id=session_list)
sessions,nSessions   = filter_sessions(protocols = ['IM'],min_lab_cells_V1=50,min_lab_cells_PM=50)

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

#%% ### Load the natural images:
natimgdata = load_natural_images(onlyright=False)
# natimgdata = load_natural_images(onlyright=True)

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

def lowrank_RF(Y, IMdata, lam=0.05,nranks=25,nsub=3):
    """
    Compute a linear low-rank approximation of the receptive field (RF) from the natural image responses

    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nranks : int, default=25
        The number of ranks to keep in the low-rank approximation
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The low-rank approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """
    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    N                   = np.shape(Y)[1]        # N is the number of neurons

    B_hat               = LM(Y, X, lam=lam)     #fit multiple linear regression (with ridge penalty)

    U, s, V             = svds(B_hat, k=nranks) #truncated singular value decomposition of the coefficients

    B_hat_rrr           = U @ np.diag(s) @ V    #reconstruct the coefficients from low rank

    Y_hat               = X @ B_hat_rrr         #predict the trial to trial response from the low rank coefficients

    cRF                 = np.reshape(B_hat_rrr, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

def lowrank_RF_cv(Y, IMdata, lam=0.05,nranks=25,nsub=3,kfold=2):
    """
    Compute a linear low-rank approximation of the receptive field (RF) from the natural image responses
    Crossvalidated version
    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nranks : int, default=25
        The number of ranks to keep in the low-rank approximation
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The low-rank approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """

    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    K,N                 = np.shape(Y)        # K is the number of images, N is the number of neurons
    Y_hat               = np.full((K,N),np.nan)

    kf = KFold(n_splits=kfold, shuffle=True)

    B_hat_rrr_folds = np.full((Ly*Lx,N,kfold),np.nan)

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        B_hat               = LM(Y_train, X_train, lam=lam)     #fit multiple linear regression (with ridge penalty)

        U, s, V             = svds(B_hat, k=nranks) #truncated singular value decomposition of the coefficients

        B_hat_rrr           = U @ np.diag(s) @ V    #reconstruct the coefficients from low rank

        Y_hat[test_index,:] = X_test @ B_hat_rrr         #predict the trial to trial response from the low rank coefficients
        
        B_hat_rrr_folds[:,:,i] = B_hat_rrr

    B_hat_rrr   = np.nanmean(B_hat_rrr_folds, axis=2)

    cRF         = np.reshape(B_hat_rrr, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat


#%% Fit each cRF with a 2D gaussian:
def fit_2dgauss_cRF(cRF, nsub,celldata):
    N = np.shape(cRF)[2]

    celldata[['rf_az_RRR', 'rf_el_RRR', 'rf_sx_RRR', 'rf_sy_RRR', 'rf_r2_RRR']] = np.nan
    
    for n in tqdm(range(N),total=N,desc='Fitting 2D gauss to RF'):	
        rfdata = np.abs(cRF[:, :, n])
        gaussian_sigma = 1
        rfdata = gaussian_filter(rfdata,sigma=[gaussian_sigma,gaussian_sigma])

        try:
            popt,pcov,r2,z_fit = fit_2d_gaussian(rfdata)

            celldata.loc[n,'rf_az_RRR']   = popt[0]*nsub
            celldata.loc[n,'rf_el_RRR']   = popt[1]*nsub
            celldata.loc[n,'rf_sx_RRR']   = popt[2]*nsub
            celldata.loc[n,'rf_sy_RRR']   = popt[3]*nsub
            celldata.loc[n,'rf_r2_RRR']   = r2
        except:
            pass
    return celldata

#%% On the trial to trial response: RRR to get RF
sesidx  = 4
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

#%% Fit the cRF with a 2D gaussian:
sessions[sesidx].celldata = fit_2dgauss_cRF(cRF, nsub=nsub,celldata=sessions[sesidx].celldata)

# sessions[sesidx].celldata['arealabel'] = sessions[sesidx].celldata['roi_name'] + sessions[sesidx].celldata['labeled']

#%% Show example neurons for different populations:
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
narealabels     = len(arealabels)
nN              = 6 #number of example neurons to show

fig,axes = plt.subplots(nN,narealabels,figsize=(3*narealabels,1*nN))
for ial,arealabel in enumerate(arealabels):
    idx_neurons = np.where(np.all((sessions[sesidx].celldata['arealabel']==arealabel,
                                   sessions[sesidx].celldata['noise_level']<100),axis=0))[0]

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


#%% Do some optimizations: lambda,kfold,rank

#%% Optimize lambda:
lams        = np.array([0.1,1,2,5,10])
EV_lams     = np.full((len(lams)),np.nan)

for i, lam in enumerate(lams):
    cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=lam,nranks=25,nsub=nsub)
    EV_lams[i] = EV(resp,Y_hat)
    # EV_lams[i] = r2_score(resp,Y_hat)

plt.plot(lams,EV_lams)

#%% Rank increases EV, but flattens out around 25
ranks        = np.array([5,25,50,100])
EV_ranks     = np.full((len(ranks)),np.nan)

for i, rank in enumerate(ranks):
    cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=2,nranks=rank,nsub=nsub)
    # EV_ranks[i] = EV(resp,Y_hat)
    EV_ranks[i] = r2_score(resp,Y_hat)

plt.plot(ranks,EV_ranks)

#%% Is better at nsub 4, than 2 or 3
subs        = np.array([2,3,4,5])
EV_nsubs     = np.full((len(subs)),np.nan)

for i, nsub in enumerate(subs):
    cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=1,nranks=25,nsub=nsub,kfold=2)
    EV_nsubs[i] = EV(resp,Y_hat)

plt.plot(subs,EV_nsubs)






