#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
import statsmodels.formula.api as smf
from scipy import stats

os.chdir('c:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance,value_matching
from utils.plot_lib import * #get all the fixed color schemes
from preprocessing.preprocesslib import assign_layer,assign_layer2
from utils.rf_lib import filter_nearlabeled
from utils.RRRlib import regress_out_behavior_modulation

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\Affine_FF_vs_FB\\')

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_noiselevel=True)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_noiselevel=True)

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

#%%
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions)

#%%
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=50)


#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 
    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 

#%%
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


#%%
def fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True):
    nPredictors = X.shape[1]
    predcat = np.array(np.repeat(['Stim','Mult','Add'],(1,nPredictors,nPredictors)))

    cvR2_models = np.full((4),np.nan)
    cvR2_preds  = np.full((4,nPredictors),np.nan)

    # Construct the design matrix
    A                           = np.column_stack([S[:,None],X * S[:,None], X])
    coefs, residuals, rank, s   = np.linalg.lstsq(A, y, rcond=None)    # Perform linear regression using least squares

    cvR2_models[0] = r2_score(y, A[:,np.isin(predcat,'Stim')] @ coefs[np.isin(predcat,'Stim')])
    cvR2_models[1] = r2_score(y, A[:,np.isin(predcat,['Stim','Mult'])] @ coefs[np.isin(predcat,['Stim','Mult'])])
    cvR2_models[2] = r2_score(y, A[:,np.isin(predcat,['Stim','Add'])] @ coefs[np.isin(predcat,['Stim','Add'])])
    cvR2_models[3] = r2_score(y, A @ coefs)

    for ipred in range(nPredictors):
        A_shuf          = A.copy()
        idx_pred_mult = ipred + 1
        idx_pred_add  = ipred + 1 + nPredictors

        A_shuf[:,idx_pred_mult] = np.random.permutation(A_shuf[:,idx_pred_mult])
        B       = np.linalg.lstsq(A_shuf, y, rcond=None)[0]    # Perform linear regression using least squares
        cvR2_preds[1,ipred] = cvR2_models[3] - r2_score(y,A_shuf @ B)
        
        A_shuf = A.copy()
        A_shuf[:,idx_pred_add] = np.random.permutation(A_shuf[:,idx_pred_add])
        B       = np.linalg.lstsq(A_shuf, y, rcond=None)[0]    # Perform linear regression using least squares
        cvR2_preds[2,ipred] = cvR2_models[3] - r2_score(y,A_shuf @ B)

        A_shuf[:,[idx_pred_mult,idx_pred_add]] = np.random.permutation(A_shuf[:,[idx_pred_mult,idx_pred_add]])
        B       = np.linalg.lstsq(A_shuf, y, rcond=None)[0]    # Perform linear regression using least squares
        cvR2_preds[3,ipred] = cvR2_models[3] - r2_score(y,A_shuf @ B)

    return cvR2_models,cvR2_preds

#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 

arealabelpairs  = [
                    'V1lab-PMunlL2/3',
                    # 'V1lab-PMlabL2/3',
                    # 'V1lab-PMunlL5',
                    # 'V1lab-PMlabL5',
                    'PMlab-V1unlL2/3',
                    # 'PMlab-V1labL2/3',
                    ]

# arealabelpairs  = [
#                     'V1lab-ALunlL2/3',
#                     # 'V1lab-PMlabL2/3',
#                     # 'V1lab-PMunlL5',
#                     # 'V1lab-PMlabL5',
#                     'PMlab-ALunlL2/3',
#                     # 'PMlab-V1labL2/3',
#                     ]
dirlabels               = np.array(['FF','FB'])
narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
nCells                  = len(celldata)

minnneurons             = 10

nbehavPCs               = 5
nPredictors             = nbehavPCs + 1 # +1 for mean pop activity
predlabels              = np.array([f'Behav_PC{i}' for i in range(nbehavPCs)] + ['MeanPopAct'])
AffModels               = np.array(['Stim','Mult','Add','Both'])
nAffModels              = len(AffModels)

#initialize storage variables
cvR2_affine             = np.full((narealabelpairs,nAffModels,nCells),np.nan)
cvR2_preds              = np.full((narealabelpairs,nAffModels,nPredictors,nCells),np.nan)

pca                     = PCA(n_components=nbehavPCs)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing affine modulation models'):
    #construct behavioral design matrix
    X       = np.stack((sessions[ises].respmat_videome,
                    sessions[ises].respmat_runspeed,
                    sessions[ises].respmat_pupilarea,
                    sessions[ises].respmat_pupilareaderiv,
                    sessions[ises].respmat_pupilx,
                    sessions[ises].respmat_pupily),axis=1)
    X       = np.column_stack((X,sessions[ises].respmat_videopc[:30,:].T))
    X       = zscore(X,axis=0,nan_policy='omit')
    si      = SimpleImputer() #impute missing values
    X       = si.fit_transform(X)
    X_p     = pca.fit_transform(X) #reduce dimensionality

    # zscore neural responses
    respdata        = zscore(sessions[ises].respmat, axis=1)
    #Get mean response per orientation (to predict trial by trial responses and multiplicative modulation)
    meanresp    = np.full_like(respdata,np.nan)
    trial_ori   = sessions[ises].trialdata['Orientation']
    for i,ori in enumerate(trial_ori.unique()):
        idx_T              = trial_ori == ori
        meanresp[:,idx_T] = np.nanmean(respdata[:,idx_T],axis=1)[:,None]

    # Fit affine modulation model for each arealabel pair (e.g. V1lab to PMunl for FF)
    # Compute mean population activity in area 1 (e.g. V1lab)
    # compute R2 of predicting responses in neurons in area 2 (e.g. PMunl) using 
    # behavioral PCs + mean pop activity in area 1
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]

        idx_N2              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1])[0]

        if len(idx_N1) < minnneurons:
            continue
        
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        # meanpopact          = np.nanmean(zscore(respdata[idx_N1,:],axis=1),axis=0)

        for iN,N in enumerate(idx_N2):
            y       = respdata[N,:]
            X       = np.column_stack((X_p,meanpopact))
            S       = meanresp[N,:]

            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True)

            idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][N])

            cvR2_affine[ialp,:,idx_ses] = tempcvR2_models
            cvR2_preds[ialp,:,:,idx_ses] = tempcvR2_preds

#%%
clrs_arealabelpairs = ['green','purple']
legendlabels        = ['FF','FB']

#%% Show overall model performance: 
idx_N =    np.all((
            # celldata['gOSI']>0.5,
            celldata['gOSI']>0,
            # celldata['nearby'],
                ),axis=0)

fig,axes = plt.subplots(1,2,figsize=(5,2.5)) 
ax = axes[0]
ax.plot(np.arange(nAffModels),np.nanmean(cvR2_affine[:,:,idx_N],axis=(0,2)),marker=None,
        linewidth=1.3,color='k')
ax.scatter(np.arange(nAffModels),np.nanmean(cvR2_affine[:,:,idx_N],axis=(0,2)),s=50,
           c=sns.color_palette('magma',nAffModels))
ax.set_ylim([0,my_ceil(np.nanmean(cvR2_affine[:,-1,idx_N]),2)])
ax.set_ylabel('Performance R2')
ax_nticks(ax, 5)
ax.set_xticks(np.arange(nAffModels),labels=AffModels)

ax = axes[1]
ymean   = np.nanmean(cvR2_preds[np.ix_(range(narealabelpairs),[-1],range(nPredictors),idx_N)],axis=(0,1,3))
yerr    = np.nanstd(cvR2_preds[np.ix_(range(narealabelpairs),[-1],range(nPredictors),idx_N)],axis=(0,1,3)) / np.sqrt(np.sum(idx_N))*10
ax.errorbar(np.arange(nPredictors),ymean,yerr,linestyle='', color='k',marker='o',
            linewidth=2)
# ax.plot(np.arange(nPredictors),ymean,marker=None,
#         linewidth=1.3,color='k')
# shaded_error(np.arange(nPredictors),ymean,yerr,color='black',alpha=0.2,ax=ax)
ax.set_ylim([0,0.04])
ax.set_ylim([0,my_ceil(np.nanmean(cvR2_preds[:,-1,0,idx_N]),2)])
ax.set_ylabel('delta R2')
ax_nticks(ax, 5)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticks(np.arange(nPredictors),labels=predlabels,rotation=45,ha='right')
my_savefig(fig,savedir,'AffineModel_R2_MultAddSep_PredictorsOverall_%dsessions' % (nSessions), formats = ['png'])

#%% Show overall model performance as histograms:
fig,axes = plt.subplots(1,4,figsize=(12,3),sharey=True,sharex=True)
for imodel in range(nAffModels):
    ax = axes[imodel]
    for ialp,alp in enumerate(arealabelpairs):
        sns.histplot(cvR2_affine[ialp,imodel,:],bins=np.linspace(-0.1,1,50),element='step',stat='probability',
                 color=clrs_arealabelpairs[ialp],fill=False,ax=ax)
    ax.set_xlabel('R2')
    ax.set_title(AffModels[imodel])
    ax.legend(legendlabels,fontsize=9,frameon=False)
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
my_savefig(fig,savedir,'AffineModel_Hist_FF_FB_R2_%dsessions' % (nSessions), formats = ['png'])

#%% Show overall model performance as mean R2:
idx_N =    np.all((
            # celldata['gOSI']>0.5,
            celldata['gOSI']>0,
            # celldata['nearby'],
                ),axis=0)

fig,axes = plt.subplots(1,1,figsize=(2.5,2.5)) 
ax = axes
handles = []
for ialp,alp in enumerate(arealabelpairs):
    handles.append(ax.plot(np.arange(nAffModels),np.nanmean(cvR2_affine[ialp,:,idx_N],axis=0),marker=None,
            linestyle=['-','--'][ialp],linewidth=1.3,color='k')[0])
    ax.scatter(np.arange(nAffModels),np.nanmean(cvR2_affine[ialp,:,idx_N],axis=(0)),s=50,
            c=sns.color_palette('magma',nAffModels))
ax.set_ylim([0,my_ceil(np.nanmean(cvR2_affine[1,-1,idx_N]),2)])
ax.set_ylabel('Performance R2')
ax.legend(handles,legendlabels,fontsize=9,frameon=False,loc='lower right')
ax_nticks(ax, 5)
ax.set_xticks(np.arange(nAffModels),labels=AffModels)
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'AffineModel_R2_FF_vs_FB_%dsessions' % (nSessions), formats = ['png'])


#%% 




#%% 
bins = np.linspace(-0.01,0.1,100)
fig,axes = plt.subplots(nPredictors,2,figsize=(5,nPredictors*2),sharey=False,sharex=True)
for imodel,model in enumerate([1,2]):
    for ipred in range(nPredictors):
        ax = axes[ipred,imodel]
        for ialp,alp in enumerate(arealabelpairs):
            idx_N =    idx_N =  np.all((
                    celldata['gOSI']>0.5,
                    # celldata['nearby'],
                     ),axis=0)
            sns.histplot(np.clip(cvR2_preds[ialp,model,ipred,idx_N],bins[0],bins[-1]),bins=bins,element='step',stat='probability',
                     color=clrs_arealabelpairs[ialp],fill=False,ax=ax)
        if imodel == 0:
            ax.set_ylabel(predlabels[ipred])
        if ipred == nPredictors-1:
            ax.set_xlabel('delta R2')
        if ipred ==0:
            ax.set_title(AffModels[model])
        # if ipred ==0:
        ax.legend(legendlabels,fontsize=9,frameon=False)
        # ax.set_xscale('log')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
fig,axes = plt.subplots(nPredictors,2,figsize=(4,nPredictors*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    for ipred in range(nPredictors):
        ax = axes[ipred,imodel]
        idx_N =    idx_N =  np.all((
                celldata['gOSI']>0,
                # celldata['gOSI']>0.2,
                # celldata['nearby'],
                    ),axis=0)
        ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)
        yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(idx_N)/2)
        ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
        ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                    linewidth=4)
        ax.set_xticks([0,1],labels=dirlabels)
        h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
                            cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
        p = p * narealabelpairs
        add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
        ax_nticks(ax, 3)
        if imodel == 0:
            ax.set_ylabel('delta R2')
        if ipred ==0:
            ax.set_title(AffModels[model])
        # ax.legend(legendlabels,fontsize=9,frameon=False)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Affinemodulation_R2_Allpredictors_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% 
ipred = -1
fig,axes = plt.subplots(1,2,figsize=(3,1*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    # for ipred in range(nPredictors):
    ax = axes[imodel]
    idx_N =    idx_N =  np.all((
            celldata['gOSI']>0,
            # celldata['gOSI']>0.2,
            # celldata['nearby'],
                ),axis=0)
    ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)
    yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(idx_N)/2)
    ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=4)
    ax.set_xticks([0,1],labels=dirlabels)
    h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
                        cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
    p = p * narealabelpairs
    add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
    ax_nticks(ax, 3)
    if imodel == 0:
        ax.set_ylabel('delta R2')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% Show the same figure but for behavioral predictors only
arealabels = ['PM','V1']
ipred = np.arange(1,nbehavPCs)
fig,axes = plt.subplots(1,2,figsize=(3,1*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    ax = axes[imodel]
    idx_N =    idx_N =  np.all((
            celldata['gOSI']>0,
            # celldata['gOSI']>0.2,
            # celldata['nearby'],
                ),axis=0)
    ymean = np.nanmean(cvR2_preds[np.ix_(range(narealabelpairs),[model],ipred,np.where(idx_N)[0])],axis=(1,3))
    ymean = np.nansum(ymean,axis=1)
    yerror = np.nanstd(cvR2_preds[np.ix_(range(narealabelpairs),[model],ipred,np.where(idx_N)[0])],axis=(1,2,3)) / np.sqrt(np.sum(idx_N)/10)
    ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=4)
    ax.set_xticks([0,1],labels=arealabels)
    xdata = np.nanmean(cvR2_preds[np.ix_([0],[model],ipred,np.where(idx_N)[0])],axis=(1,2)).squeeze()
    ydata = np.nanmean(cvR2_preds[np.ix_([1],[model],ipred,np.where(idx_N)[0])],axis=(1,2)).squeeze()
    h,p = stats.ttest_ind(xdata,ydata,nan_policy='omit')
    p = p * narealabelpairs
    add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
    ax_nticks(ax, 3)
    if imodel == 0:
        ax.set_ylabel('delta R2')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Behavior_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% ## Show that the amount of multiplicative and additive modulation varies with orientation selectivity
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
x = celldata['gOSI']
sns.regplot(x=x,y=cvR2_preds[0,1,-1,:],x_ci='sd',scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[0]},
            line_kws={'color': clrs_arealabelpairs[0],'linewidth':3,'linestyle':'-'},ax=axes[0])
sns.regplot(x=x,y=cvR2_preds[1,1,-1,:],ci=99,scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[1]},
            line_kws={'color': clrs_arealabelpairs[1],'linewidth':3,'linestyle':'-'},ax=axes[0])
axes[0].set_ylim(np.nanpercentile(cvR2_preds[:,1,-1,:],[0.5,99]))
axes[0].set_title('Multiplicative')

axes[0].set_ylabel('Delta R2')
axes[0].set_xlim([0,1])
ax_nticks(axes[0], 3)

sns.regplot(x=x,y=cvR2_preds[0,2,-1,:],ci=99,scatter_kws={'s':2,'alpha':0.2,'color': clrs_arealabelpairs[0]},
            line_kws={'color': clrs_arealabelpairs[0],'linewidth':3,'linestyle':'-'},ax=axes[1])
sns.regplot(x=x,y=cvR2_preds[1,2,-1,:],ci=99,scatter_kws={'s':2,'alpha':0.2,'color': clrs_arealabelpairs[1]},
            line_kws={'color': clrs_arealabelpairs[1],'linewidth':3,'linestyle':'-'},ax=axes[1])
axes[1].set_ylim(np.nanpercentile(cvR2_preds[:,2,-1,:],[0.5,99]))
axes[1].set_title('Additive')
axes[1].set_xlim([0,1])
ax_nticks(axes[1], 3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_affineR2_control_gOSI_GR%dsessions' % (nSessions), formats = ['png'])

#%% Test if the multiplitivate and additive effect persists even if you control for orientation tuning strength
idx_predictor = -1
FF = np.any(~np.isnan(cvR2_preds[0,:,:,:]),axis=(0,1))
FB = np.any(~np.isnan(cvR2_preds[1,:,:,:]),axis=(0,1))
arealabel = np.repeat('None',nCells) 
arealabel[np.where(FF & ~FB)[0]] = 'FF'
arealabel[np.where(~FF & FB)[0]] = 'FB'

df = pd.DataFrame({'gOSI':    celldata['gOSI'],
                   'OSI':    celldata['OSI'],
                   'session_id': celldata['session_id'],
                   'Mult':  np.nanmean(cvR2_preds[:,1,idx_predictor,:],axis=0),
                   'Add':    np.nanmean(cvR2_preds[:,2,idx_predictor,:],axis=0),
                   'AreaLabel': arealabel,
                   })
df.dropna(inplace=True)

#%% Test multiplicative effect 
model     = smf.mixedlm("Mult ~ C(AreaLabel, Treatment('FF'))", data=df,groups=df["session_id"])
result    = model.fit(reml=False)
print(result.summary())

#%% Test Add effect 
model     = smf.mixedlm("Add ~ C(AreaLabel, Treatment('FF'))", data=df,groups=df["session_id"])
result    = model.fit(reml=False)
print(result.summary())

#%% Test mixed effects linear model:
# Does gOSI explain variance in multiplicative and additive modulation beyond area label?
# Add session id as random effect
# the * denotes interaction term as well as testing for main effects
# Reference level is FF. A positive value indicates that FB has higher modulation than FF
# Now the effect is captured as a positive interaction effect: FB multiplicatively modulates V1 cells
# more strongly if they are more orientation selective (higher gOSI)
model     = smf.mixedlm("Mult ~  C(AreaLabel, Treatment('FF')) * gOSI", data=df,groups=df["session_id"])
result_alpha    = model.fit(reml=False)
print(result_alpha.summary())

#%% test for additive modulation
model     = smf.mixedlm("Add ~  C(AreaLabel, Treatment('FF')) * gOSI", data=df,groups=df["session_id"])
result_beta    = model.fit(reml=False)
print(result_beta.summary())

#%%
fig, axes = plt.subplots(2,2,figsize=(15,5))
for itab in range(2):
    ax = axes[itab,0]
    ax.axis('off')
    ax.axis('tight')
    if itab == 0: 
        ax.set_title('Multiplicative')
    ax.table(cellText=result_alpha.summary().tables[itab].values.tolist(),
             rowLabels=result_alpha.summary().tables[itab].index.tolist(),
             colLabels=result_alpha.summary().tables[itab].columns.tolist(),
             loc="center",fontsize=8)

for itab in range(2):
    ax = axes[itab,1]
    ax.axis('off')
    ax.axis('tight')
    if itab == 0: 
        ax.set_title('Additive')
    ax.table(cellText=result_beta.summary().tables[itab].values.tolist(),
             rowLabels=result_beta.summary().tables[itab].index.tolist(),
             colLabels=result_beta.summary().tables[itab].columns.tolist(),
             loc="center",fontsize=8)
fig.tight_layout()
my_savefig(fig,savedir,'AffineModel_control_gOSI_table_GR_%dsessions' % (nSessions),formats=['png'])


# 0.007  0.011
# 0.005  0.008

#%% ## Show that the amount of multiplicative and additive modulation varies with orientation selectivity
# for behavioral variables
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
x = celldata['gOSI']
idx_predictor = np.arange(1,nbehavPCs)
ydata = np.nanmean(cvR2_preds[0,1,idx_predictor,:],axis=(0))
sns.regplot(x=x,y=ydata,x_ci='sd',scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[0]},
            line_kws={'color': clrs_arealabelpairs[0],'linewidth':3,'linestyle':'-'},ax=axes[0])
ydata = np.nanmean(cvR2_preds[1,1,idx_predictor,:],axis=(0))
sns.regplot(x=x,y=ydata,ci=99,scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[1]},
            line_kws={'color': clrs_arealabelpairs[1],'linewidth':3,'linestyle':'-'},ax=axes[0])
axes[0].set_ylim(np.nanpercentile(ydata,[0.5,98]))
axes[0].set_title('Multiplicative')
axes[0].set_ylabel('Delta R2')
axes[0].set_xlim([0,1])
ax_nticks(axes[0], 3)
ydata = np.nanmean(cvR2_preds[0,2,idx_predictor,:],axis=(0))
sns.regplot(x=x,y=ydata,x_ci='sd',scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[0]},
            line_kws={'color': clrs_arealabelpairs[0],'linewidth':3,'linestyle':'-'},ax=axes[1])
ydata = np.nanmean(cvR2_preds[1,2,idx_predictor,:],axis=(0))
sns.regplot(x=x,y=ydata,ci=99,scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[1]},
            line_kws={'color': clrs_arealabelpairs[1],'linewidth':3,'linestyle':'-'},ax=axes[1])
axes[1].set_ylim(np.nanpercentile(cvR2_preds[:,2,idx_predictor,:],[0.5,98]))
axes[1].set_title('Additive')
axes[1].set_xlim([0,1])
ax_nticks(axes[1], 3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Behav_affineR2_control_gOSI_GR%dsessions' % (nSessions), formats = ['png'])

#%% Test if the multiplitivate and additive effect persists even if you control for orientation tuning strength
idx_predictor = np.arange(1,nbehavPCs)
FF = np.any(~np.isnan(cvR2_preds[0,:,:,:]),axis=(0,1))
FB = np.any(~np.isnan(cvR2_preds[1,:,:,:]),axis=(0,1))
arealabel = np.repeat('None',nCells) 
arealabel[np.where(FF & ~FB)[0]] = 'FF'
arealabel[np.where(~FF & FB)[0]] = 'FB'

df = pd.DataFrame({'gOSI':    celldata['gOSI'],
                   'session_id': celldata['session_id'],
                   'Mult':  np.nanmean(cvR2_preds[:,1,idx_predictor,:],axis=(0,1)),
                   'Add':    np.nanmean(cvR2_preds[:,2,idx_predictor,:],axis=(0,1)),
                   'AreaLabel': arealabel,
                   })
df.dropna(inplace=True)

#%% Test mixed effects linear model for multiplicative modulation:
# Does gOSI explain variance in multiplicative and additive modulation for behavioral modulation differently?
model     = smf.mixedlm("Mult ~  C(AreaLabel, Treatment('FF')) * gOSI", data=df,groups=df["session_id"])
result    = model.fit(reml=False)
print(result.summary())
# 0.000  0.002

#%% test for additive modulation
model     = smf.mixedlm("Add ~  C(AreaLabel, Treatment('FF')) * gOSI", data=df,groups=df["session_id"])
result    = model.fit(reml=False)
print(result.summary())


# %%
