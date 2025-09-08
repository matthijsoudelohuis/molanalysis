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

os.chdir('e:\\Python\\molanalysis')

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list        = np.array([['LPE12223_2024_06_10']])
# session_list        = np.array([['LPE11086_2024_01_05']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%%  Load data properly:        
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)
    
#%%
sessions = compute_pairwise_anatomical_distance(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions)

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
N = ses.respmat.shape[0]

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

def fitAffine_singleneuron(sessions,radius=500):
    
    for ses in tqdm(sessions,desc='Fitting Single Neuron Affine Model',total=len(sessions)):

        ses.celldata['Affine_R2'] = np.nan
        ses.celldata['Affine_Mult'] = np.nan
        ses.celldata['Affine_Add'] = np.nan
        ses.celldata['Affine_Inter'] = np.nan

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
            [ses.celldata.loc[iN,'Affine_Mult'], ses.celldata.loc[iN,'Affine_Add'], 
                ses.celldata.loc[iN,'Affine_Inter']] = coefs

            # Compute R^2 value
            y_pred = A @ coefs
            ses.celldata.loc[iN,'Affine_R2'] = r2_score(y, y_pred)

            # Y_hat[iN,:,modelversions.index(modelversion)] = y_pred
    return sessions

#%%

sessions = fitAffine_singleneuron(sessions,radius=500)


#%% 
fig,axes = plt.subplots(1,2,figsize=(8,4))
sns.histplot(data=sessions[ises].celldata,x='Affine_Mult',color='green',element="step",
             common_norm=False,ax=axes[0],stat="density",hue='arealabel')
sns.histplot(data=sessions[ises].celldata,x='Affine_Add',color='blue',element="step",
             common_norm=False,ax=axes[1],stat="density",hue='arealabel')
axes[0].set_title('Mult')
axes[1].set_title('Add')
my_savefig(fig,savedir,'AffineModelCoefHist_SingleNeuron_GR_%ssession' % sessions[ises].session_id,formats=['png'])

#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Fit linear mixed effects model on celldata ['Affine_Mult'] with 'session_id' as 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator
from scipy import stats

#%% 
arealabels = ['V1unl','V1lab','PMunl','PMlab']
fig,axes = plt.subplots(2,2,figsize=(6,6))
   
arealabelpairs = [['V1unl','V1lab'],['PMunl','PMlab']]

for ialp,alp in enumerate(arealabelpairs):
    for imod,mod in enumerate(['Mult','Add']):
        # idx_N = 
        idx_N = np.logical_and(celldata['arealabel'].isin(alp),celldata['noise_level']<20)
        sns.histplot(data=celldata[idx_N],x='Affine_%s' % mod,element="step",common_norm=False,ax=axes[ialp,imod],
                    stat="density",hue='arealabel',hue_order=alp,palette=get_clr_area_labeled(alp))

        # Fit linear mixed effects model on celldata ['Affine_Mult'] with 'session_id' as 
        model = smf.mixedlm("Affine_%s ~ arealabel" % mod, data=celldata[idx_N], groups=celldata["session_id"][idx_N])
        result = model.fit(reml=False)
        # print(result.summary())
        # pval = result.pvalues[1]
        pval = result.pvalues[1]
        print('P-value %s (%s): %1.5f' % ('_'.join(arealabels),'Multiplicative',pval))
        axes[ialp,imod].text(0.75,0.5,'p=%1.4f' % (pval),ha='center',va='center',transform=axes[ialp,imod].transAxes)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,savedir,'AffineModel_Labeled_CoefHist_SingleNeuron_GR_%d' % nSessions,formats=['png'])

#%% 
fig,ax = plt.subplots(1,1,figsize=(4,4))
sns.regplot(data=celldata,x='Affine_Mult',y='Affine_Add',ax=ax,color='green',
            scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'})

#%% 
nbins = 25
yvars = ['Affine_Mult','Affine_Add']
# xvars = ['depth','noise_level','OSI','tuning_var','gOSI','pop_coupling','pref_ori','event_rate']
xvars = ['depth','noise_level','OSI','pop_coupling','pref_ori','event_rate']
celldata['noise_level'] = np.clip(celldata['noise_level'],0,100)
fig,axes = plt.subplots(len(yvars),len(xvars),figsize=(len(xvars)*3,len(yvars)*3))
for iy,yvar in enumerate(yvars):
    for ix,xvar in enumerate(xvars):
        # sns.regplot(data=celldata,x=xvar,y=yvar,ax=axes[iy,ix],color='black', ci=None,
                    # scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'})
        sns.regplot(data=celldata,x=xvar,y=yvar,ax=axes[iy,ix],color='blue', ci=95,
                    # x_bins=np.linspace(np.nanpercentile(celldata[xvar],1),np.nanpercentile(celldata[xvar],99), nbins),
                    # x_bins=np.linspace(np.nanmin(celldata[xvar]),np.nanmax(celldata[xvar]), nbins),
                    x_bins=np.nanpercentile(celldata[xvar], np.linspace(0, 100, nbins)),
                    scatter_kws={'s': 5, 'alpha':0.5,'edgecolor': 'blue'})
                    # scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'})
my_savefig(fig,savedir,'AffineModel_VarCorrs_GR_%d' % nSessions,formats=['png'])

# sns.regplot(data=celldata,x='depth',y='Affine_Mult',ax=axes[0],color='green',
#             scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'})

# sns.histplot(data=celldata[celldata['arealabel'].isin(arealabels)],x='Affine_Add',element="step",common_norm=False,ax=axes[1,0],
#              stat="density",hue='arealabel',hue_order=arealabels,palette=get_clr_area_labeled(arealabels))
# model = smf.mixedlm("Affine_%s ~ arealabel" % 'Add', data=celldata[celldata['arealabel'].isin(arealabels)], groups=celldata["session_id"][celldata['arealabel'].isin(arealabels)])
# result = model.fit(reml=False)
# # print(result.summary())
# pval = result.pvalues[1]
# print('P-value %s (%s): %1.5f' % ('_'.join(arealabels),'Additive',pval))

# # axes[0,0].set_title('Multiplicative')
# # axes[1,0].set_title('Add')

# arealabels = ['PMunl','PMlab']
# sns.histplot(data=celldata[celldata['arealabel'].isin(arealabels)],x='Affine_Mult',element="step",common_norm=False,ax=axes[0,1],
#              stat="density",hue='arealabel',hue_order=arealabels,palette=get_clr_area_labeled(arealabels))
# model = smf.mixedlm("Affine_Mult ~ arealabel", data=celldata[celldata['arealabel'].isin(arealabels)], groups=celldata["session_id"][celldata['arealabel'].isin(arealabels)])
# result = model.fit(reml=False)
# # print(result.summary())
# pval = result.pvalues[1]
# print('P-value %s (%s): %1.5f' % ('_'.join(arealabels),'Multiplicative',pval))

# sns.histplot(data=celldata[celldata['arealabel'].isin(arealabels)],x='Affine_Add',element="step",common_norm=False,ax=axes[1,1],
#              stat="density",hue='arealabel',hue_order=arealabels,palette=get_clr_area_labeled(arealabels))
# model = smf.mixedlm("Affine_Add ~ arealabel", data=celldata[celldata['arealabel'].isin(arealabels)], groups=celldata["session_id"][celldata['arealabel'].isin(arealabels)])
# result = model.fit(reml=False)
# # print(result.summary())
# pval = result.pvalues[1]
# print('P-value %s (%s): %1.5f' % ('_'.join(arealabels),'Additive',pval))

# axes[0].set_title('Mult')
# axes[1].set_title('Add')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)

# my_savefig(fig,savedir,'AffineModel_SingleNeuron_GR_%ssession' % ses.session_id,formats=['png'])






