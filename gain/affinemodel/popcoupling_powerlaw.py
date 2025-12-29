#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.stats import vonmises
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import linregress,binned_statistic
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator

os.chdir('c:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\')

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list        = np.array([['LPE12223_2024_06_10']])
# session_list        = np.array([['LPE11086_2024_01_05']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=calciumversion)
    
#%%
sessions = compute_pairwise_anatomical_distance(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions,version='radius_500')


#%% Modulation of tuned response for different coupled neurons: 
nPopCouplingBins    = 5
nPopRateBins        = 5
ises                = 0
binedges_popcoupling = np.percentile(sessions[ises].celldata['pop_coupling'],np.linspace(0,100,nPopCouplingBins+1))

# respmat = zscore(sessions[ises].respmat,axis=1)
respmat = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

poprate             = np.nanmean(respmat, axis=0)
# poprate             = np.nanmean(zscore(sessions[ises].respmat.T, axis=0),axis=1)
binedges_poprate    = np.percentile(poprate,np.linspace(0,100,nPopRateBins+1))

stims               = sessions[ises].trialdata['Orientation'].to_numpy()
ustim               = np.unique(stims)
nstim               = len(ustim)


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
        meandata[:,iPopRateBin,istim] = np.mean(respmat[:,idx_T],axis=1)
        stddata[:,iPopRateBin,istim] = np.std(respmat[:,idx_T],axis=1)

    # sm = np.roll(sm,shift=-prefori,axis=1)
    for n in range(N):
        meandata[n,iPopRateBin,:] = np.roll(meandata[n,iPopRateBin,:],-prefori[n])
        stddata[n,iPopRateBin,:] = np.roll(stddata[n,iPopRateBin,:],-prefori[n])

#%% 
clrs_popcoupling    = sns.color_palette('viridis',nPopRateBins)

fig,axes = plt.subplots(1,nPopCouplingBins,figsize=(15,2.5),sharey=True,sharex=True)
for iPopCouplingBin in range(nPopCouplingBins):
    ax = axes[iPopCouplingBin]
    idx_popcoupling = np.all((
                            sessions[ises].celldata['gOSI']>0.4,
                            # sessions[ises].celldata['roi_name']=='V1',
                            sessions[ises].celldata['pop_coupling']>binedges_popcoupling[iPopCouplingBin],
                            sessions[ises].celldata['pop_coupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)
                            # sessions[ises].celldata['pop_coupling']>binedges_popcoupling[iPopCouplingBin],
                            # sessions[ises].celldata['pop_coupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)

    for iPopRateBin in range(nPopRateBins):
        ax.plot(np.mean(meandata[idx_popcoupling,iPopRateBin,:],axis=0),color=clrs_popcoupling[iPopRateBin],
                linewidth=2)
    ax.set_xticks(np.arange(0,len(ustim),2),labels=ustim[::2],fontsize=7)
    # ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
    ax.set_xlabel('Orientation',fontsize=9)
    # ax.set_ylabel('Neuron',fontsize=9)
    ax.tick_params(axis='x', labelrotation=45)
sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)

#%%
# Problematic: population rate bin is defined not by the selection, but based on the above which is all neurons together, etc.

#%% Is modulation by population rate dependent on activity levels,
# more multiplicative for larger activity levels

# ci              = 95 #bootstrapped confidence interval
# nboots          = 250 #number of bootstrap samples
percspacing     = 2 #bins chosen to have approx equal number of points
# percentiles     = np.arange(0,100+percspacing,percspacing).astype('float')
# percentiles[percentiles==100] = 99.75 #avoid issues with max value
# bins            = np.nanpercentile(meandata,percentiles)
# bins            = bins[bins>0] #remove duplicate bins at 0
# bincenters      = (bins[:-1]+bins[1:])/2 #get bin centers

fig,axes = plt.subplots(1,nPopCouplingBins,figsize=(nPopCouplingBins*3,3),sharex=True,sharey=True)

for iPopCouplingBin in range(nPopCouplingBins):
    ax = axes[iPopCouplingBin]
    idx_N =    np.all((
                            # sessions[ises].celldata['gOSI']>0.4,
                            sessions[ises].celldata['noise_level']<20,
                            # sessions[ises].celldata['roi_name']=='V1'
                            # sessions[ises].celldata['roi_name']=='PM'
                            ),axis=0)
    
    binedges_popcoupling = np.percentile(sessions[ises].celldata['pop_coupling'][idx_N],np.linspace(0,100,nPopCouplingBins+1))

    idx_popcoupling = np.all((
                            idx_N,
                            sessions[ises].celldata['pop_coupling']>binedges_popcoupling[iPopCouplingBin],
                            sessions[ises].celldata['pop_coupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)
                            # sessions[ises].celldata['pop_coupling']>binedges_popcoupling[iPopCouplingBin],
                            # sessions[ises].celldata['pop_coupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)
    
    percentiles     = np.arange(0,100+percspacing,percspacing).astype('float')
    percentiles[percentiles==100] = 99.75 #avoid issues with max value
    bins            = np.nanpercentile(meandata[idx_popcoupling,:,:],percentiles)
    bins            = bins[bins>0] #remove duplicate bins at 0
    bincenters      = (bins[:-1]+bins[1:])/2 #get bin centers
    
    for iPopRateBin in range(nPopRateBins):
        xdata = meandata[idx_popcoupling,0,:]
        # xdata = np.nanmean(meandata[idx_popcoupling,:,:],axis=1)
        ydata = meandata[idx_popcoupling,iPopRateBin,:] - meandata[idx_popcoupling,0,:]
        # ydata = meandata[idx_popcoupling,iPopRateBin,:] - meandata[idx_popcoupling,2,:]

        xdata.flatten()
        ydata.flatten()

        idx_notnan = np.logical_and(~np.isnan(xdata),~np.isnan(ydata))

        xdata = xdata[idx_notnan]
        ydata = ydata[idx_notnan]
        ymeandata = binned_statistic(xdata, ydata, statistic='mean', 
                            bins=bins)[0]
        
        # ax.plot(bincenters,ymeandata,color=clrs_popcoupling[iPopRateBin],marker='o',linestyle='None',markersize=4)
        ax.plot(bincenters,ymeandata,color=clrs_popcoupling[iPopRateBin],linestyle='-',
                marker='o',markersize=4,linewidth=2)
        ax.set_title('Pop. Coupling %d/%d' % (iPopCouplingBin+1,nPopCouplingBins))
        if iPopCouplingBin==nPopCouplingBins-1:
            ax.legend(['0-20%','20-40%','40-60%','60-80%','80-100%'],
                    reverse=True,fontsize=7,frameon=False,title='pop. rate',bbox_to_anchor=(1.05,1), loc='upper left')
    if iPopCouplingBin==0:
        ax.set_ylabel('Modulation (rel. to low activity)')
    if iPopCouplingBin == np.round(nPopCouplingBins/2):
        ax.set_xlabel('Mean evoked activity')
    ax.axhline(0,color='grey',ls='--',linewidth=1)
ax.set_xlim([0,bincenters[-1]*1.1])
ax_nticks(ax,3)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Evoked_Activity_vs_Modulation_%dGRsessions_V1' % (nSessions))
my_savefig(fig,savedir,'Evoked_Activity_vs_Modulation_%dGRsessions_PM' % (nSessions))











#%% #########################################################################################
ises        = 0
ses         = sessions[ises]

Y           = zscore(ses.respmat, axis=1)

T           = copy.deepcopy(Y)

trial_ori   = ses.trialdata['Orientation']
oris        = np.sort(trial_ori.unique())

## Compute tuned response:
for ori in oris:
    ori_idx     = np.where(ses.trialdata['Orientation']==ori)[0]
    temp        = np.mean(Y[:,ses.trialdata['Orientation']==ori],axis=1)
    T[:,ori_idx] = np.repeat(temp[:, np.newaxis], len(ori_idx), axis=1)

#%% 
N               = ses.respmat.shape[0]

modelversions = ['all','area','plane','radius_50','radius_100','radius_500','radius_1000']
modelversions = ['random','runspeed','videome','all','area','plane','radius_50','radius_100','radius_500','radius_1000']
nmodels         = len(modelversions)
modelcoefs      = np.full((nmodels,N,3),np.nan)
model_R2        = np.full((nmodels,N),np.nan)

Y_hat           = np.full((ses.respmat.shape[0],ses.respmat.shape[1],nmodels),np.nan)

for modelversion in modelversions:
    print(modelversion)
    
    for iN in range(N):
        if modelversion == 'all':
            r = np.mean(ses.respmat[np.setdiff1d(np.arange(N),iN),:], axis=0)
            # r = poprate
        elif modelversion == 'area':
            idx_N = ses.celldata['roi_name'] == ses.celldata['roi_name'][iN]
            idx_N[iN] = False
            r = np.nanmean(ses.respmat[idx_N,:], axis=0)
        elif modelversion == 'plane':
            # r = poprate_planes[ses.celldata['plane_idx'][iN]]
            idx_N = ses.celldata['plane_idx'] == ses.celldata['plane_idx'][iN]
            idx_N[iN] = False
            r = np.nanmean(ses.respmat[idx_N,:], axis=0)
        elif modelversion == 'radius_50':
            idx_N = ses.distmat_xyz[iN,:] < 50
            idx_N[iN] = False
            r = np.nanmean(ses.respmat[idx_N,:], axis=0)
        elif modelversion == 'radius_100':
            idx_N = ses.distmat_xyz[iN,:] < 100
            idx_N[iN] = False
            r = np.nanmean(ses.respmat[idx_N,:], axis=0)
        elif modelversion == 'radius_500':
            idx_N = ses.distmat_xyz[iN,:] < 500
            idx_N[iN] = False
            r = np.nanmean(ses.respmat[idx_N,:], axis=0)
        elif modelversion == 'radius_1000':
            idx_N = ses.distmat_xyz[iN,:] < 1000
            idx_N[iN] = False
            r = np.nanmean(ses.respmat[idx_N,:], axis=0)
        elif modelversion == 'random':
            r = np.random.randn(1,ses.respmat.shape[1])
        elif modelversion == 'runspeed':
            r = ses.respmat_runspeed
        elif modelversion == 'videome':
            r = ses.respmat_videome

        y = Y[iN,:]
        x = T[iN,:]
        
        if np.isnan(r).all():
            modelcoefs[modelversions.index(modelversion), iN, :] = np.nan
            model_R2[modelversions.index(modelversion), iN] = np.nan
            Y_hat[iN,:,modelversions.index(modelversion)] = np.nan
            continue
        # Construct the design matrix
        A = np.vstack([r * x, r, np.ones_like(y)]).T

        # Perform linear regression using least squares
        coefs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

        # Store the coefficients
        modelcoefs[modelversions.index(modelversion), iN, :] = coefs

        # Compute R^2 value
        y_pred = A @ coefs
        model_R2[modelversions.index(modelversion), iN] = r2_score(y, y_pred)

        Y_hat[iN,:,modelversions.index(modelversion)] = y_pred

#%%
idx_N = ses.celldata['tuning_var'] > 0.01
# idx_N = ses.celldata['OSI'] > 0.5

fig,axes = plt.subplots(1,2+nmodels,figsize=(10+5*nmodels,5))
axes[0].imshow(Y[idx_N,:],aspect='auto',vmin=-0.5,vmax=0.5)
axes[1].imshow(T[idx_N,:],aspect='auto',vmin=-0.5,vmax=0.5)
axes[0].set_title('Raw')
axes[1].set_title('Tuned')

for i in range(nmodels):
    axes[i+2].imshow(Y_hat[idx_N,:,i],aspect='auto',vmin=-0.5,vmax=0.5)
    axes[i+2].set_title(modelversions[i])
my_savefig(fig,savedir,'Heatmap_AffineModel_SingleNeuron_GR_%ssession' % ses.session_id,formats=['png'])

#%% 
idx_N = ses.celldata['tuning_var'] > 0.01

clrs = sns.color_palette('colorblind',nmodels)
fig,ax = plt.subplots(1,1,figsize=(nmodels*0.8,3))
sns.violinplot(data=model_R2[:,idx_N].T,ax=ax,palette=clrs,inner="box",scale='width',linewidth=1,cut=0)
for r2 in np.linspace(0,1,6):
    ax.axhline(r2,color='black',linestyle='--',alpha=0.5,zorder=-1)
ax.set_ylabel('R2')
ax.set_title('R2 values for the different models')
ax.set_xticks(range(nmodels))
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
ax.set_xticklabels([v if i != np.argmax(np.nanmean(model_R2[:,idx_N],axis=1)) else '*'+v for i,v in enumerate(modelversions)],
                   rotation=45,ha='right',fontsize=9)
my_savefig(fig,savedir,'R2_AffineModel_SingleNeuron_GR_%ssession' % ses.session_id,formats=['png'])

print('Mean R2 for:')
for i in range(nmodels):
    idx_N = ses.celldata['tuning_var'] > 0.025
    print('%s: %.2f' % (modelversions[i],np.nanmean(model_R2[i,idx_N])))
    # print('%s: %.2f' % (modelversions[i],np.nanmean(model_R2[i,:])))
    # print('Median R2 for %s: %.2f' % (modelversions[i],np.nanmedian(model_R2[i,:])))


#%% 
def fitAffine_GR_singleneuron_full(sessions,radius=500):
    for ses in tqdm(sessions,desc='Fitting Single Neuron Affine Model',total=len(sessions)):

        ses.celldata['aff_r2_grfull'] = np.nan
        ses.celldata['aff_alpha_grfull'] = np.nan
        ses.celldata['aff_beta_grfull'] = np.nan
        ses.celldata['aff_offset_grfull'] = np.nan

        Y           = zscore(ses.respmat, axis=1)

        T           = copy.deepcopy(Y)

        trial_ori   = ses.trialdata['Orientation']
        oris        = np.sort(trial_ori.unique())

        ## Compute tuned response:
        for ori in oris:
            ori_idx     = np.where(ses.trialdata['Orientation']==ori)[0]
            temp        = np.mean(Y[:,ses.trialdata['Orientation']==ori],axis=1)
            T[:,ori_idx] = np.repeat(temp[:, np.newaxis], len(ori_idx), axis=1)

        N = ses.respmat.shape[0]

        Y_hat           = np.full_like(ses.respmat, np.nan)

        for iN in range(N):
            idx_N       = ses.distmat_xyz[iN,:] < radius
            idx_N[iN]   = False
            r           = np.nanmean(ses.respmat[idx_N,:], axis=0)
        
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
            [ses.celldata.loc[iN,'aff_alpha_grfull'], ses.celldata.loc[iN,'aff_beta_grfull'], 
                ses.celldata.loc[iN,'aff_offset_grfull']] = coefs

            # Compute R^2 value
            y_pred = A @ coefs
            ses.celldata.loc[iN,'aff_r2_grfull'] = r2_score(y, y_pred)

            # Y_hat[iN,:,modelversions.index(modelversion)] = y_pred
    return sessions

#%% 
def fitAffine_GR_singleneuron_split(sessions,radius=500,perc=50):
    for ses in tqdm(sessions,desc='Fitting Single Neuron Affine Model',total=len(sessions)):

        ses.celldata['aff_r2_grsplit'] = np.nan
        ses.celldata['aff_alpha_grsplit'] = np.nan
        ses.celldata['aff_beta_grsplit'] = np.nan

        trial_ori   = ses.trialdata['Orientation']
        oris        = np.sort(trial_ori.unique())

        N = ses.respmat.shape[0]

        Y_hat           = np.full_like(ses.respmat, np.nan)

        r = np.nanmean(zscore(ses.respmat, axis=1), axis=0)
# 
        for iN in range(N):
            # idx_N       = ses.distmat_xyz[iN,:] < radius
            # idx_N[iN]   = False
            # r           = np.nanmean(ses.respmat[idx_N,:], axis=0)
            
            if np.isnan(r).all():
                # modelcoefs[modelversions.index(modelversion), iN, :] = np.nan
                # model_R2[modelversions.index(modelversion), iN] = np.nan
                # Y_hat[iN,:,modelversions.index(modelversion)] = np.nan
                continue

            idx_low    = r<=np.percentile(r,perc)
            idx_high   = r>np.percentile(r,100-perc)

            meanresp    = np.empty([len(oris),2])
            for i,ori in enumerate(oris):
                meanresp[i,0] = np.nanmean(ses.respmat[iN,np.logical_and(ses.trialdata['Orientation']==ori,idx_low)])
                meanresp[i,1] = np.nanmean(ses.respmat[iN,np.logical_and(ses.trialdata['Orientation']==ori,idx_high)])
                
            # meanresp_pref          = meanresp.copy()
            # for n in range(N):
            #     meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            #     meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

            # normalize by peak response during still trials
            tempmin,tempmax = meanresp[:,0].min(axis=0,keepdims=True),meanresp[:,0].max(axis=0,keepdims=True)
            meanresp[:,0] = (meanresp[:,0] - tempmin) / (tempmax - tempmin)
            meanresp[:,1] = (meanresp[:,1] - tempmin) / (tempmax - tempmin)
            
            b = linregress(meanresp[:,0],meanresp[:,1])

            # Store the coefficients
            [ses.celldata.loc[iN,'aff_alpha_grsplit'], ses.celldata.loc[iN,'aff_beta_grsplit'], 
                ses.celldata.loc[iN,'aff_r2_grsplit']] = b[:3]
            
    return sessions

#%%

sessions = fitAffine_GR_singleneuron_full(sessions,radius=500)

#%%

sessions = fitAffine_GR_singleneuron_split(sessions,radius=500,perc=50)


#%% 
fig,axes = plt.subplots(1,2,figsize=(8,4))
sns.histplot(data=sessions[ises].celldata,x='aff_alpha_grsplit',color='green',element="step",
             common_norm=False,ax=axes[0],stat="density",hue='arealabel')
sns.histplot(data=sessions[ises].celldata,x='aff_beta_grsplit',color='blue',element="step",
             common_norm=False,ax=axes[1],stat="density",hue='arealabel')
axes[0].set_title('Mult')
axes[1].set_title('Add')
my_savefig(fig,savedir,'AffineModelCoefHist_SingleNeuron_GR_%ssession' % sessions[ises].session_id,formats=['png'])

#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
