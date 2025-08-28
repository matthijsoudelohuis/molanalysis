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

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list        = np.array([['LPE12223_2024_06_10']])
session_list        = np.array([['LPE11086_2024_01_05']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#   Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%%
sessions = compute_pairwise_anatomical_distance(sessions)
sessions = compute_tuning_wrapper(sessions)

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
modelversions = ['random','all','area','plane','radius_50','radius_100','radius_500','radius_1000']
nmodels         = len(modelversions)
modelcoefs      = np.full((nmodels,N,3),np.nan)
model_R2        = np.full((nmodels,N),np.nan)

Y_hat           = np.full((ses.respmat.shape[0],ses.respmat.shape[1],nmodels),np.nan)
poprate         = np.nanmean(ses.respmat,axis=0)

poprate_areas = {'V1':np.nanmean(ses.respmat[ses.celldata['roi_name']=='V1',:],axis=0),
                 'PM':np.nanmean(ses.respmat[ses.celldata['roi_name']=='PM',:],axis=0),
                 'AL':np.nanmean(ses.respmat[ses.celldata['roi_name']=='AL',:],axis=0),
                 'RSP':np.nanmean(ses.respmat[ses.celldata['roi_name']=='RSP',:],axis=0)}
poprate_planes = {i: np.nanmean(ses.respmat[ses.celldata['plane_idx']==i,:],axis=0) for i in range(8)}

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
