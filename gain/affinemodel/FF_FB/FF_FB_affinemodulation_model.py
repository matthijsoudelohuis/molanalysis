#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

os.chdir('e:\\Python\\molanalysis')

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

#%% Concatenate all cells:
celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
nCells                  = len(celldata)

#%%
def fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True):
    nPredictors = X.shape[1]
    predcat = np.array(np.repeat(['Stim','Mult','Add'],(1,nPredictors,nPredictors)))

    cvR2_models = np.full((4),np.nan)
    cvR2_preds  = np.full((4,nPredictors),np.nan)

    # Construct the design matrix
    A                           = np.column_stack([S[:,None],X * S[:,None], X])
    A                           = zscore(A,axis=0,nan_policy='omit')
    coefs, residuals, rank, s   = np.linalg.lstsq(A, y, rcond=None)    # Perform linear regression using least squares

    cvR2_models[0] = r2_score(y, A[:,np.isin(predcat,'Stim')] @ coefs[np.isin(predcat,'Stim')])
    cvR2_models[1] = r2_score(y, A[:,np.isin(predcat,['Stim','Mult'])] @ coefs[np.isin(predcat,['Stim','Mult'])])
    cvR2_models[2] = r2_score(y, A[:,np.isin(predcat,['Stim','Add'])] @ coefs[np.isin(predcat,['Stim','Add'])])
    cvR2_models[3] = r2_score(y, A @ coefs)

    for ipred in range(nPredictors):
        idx_pred_mult = ipred + 1
        idx_pred_add  = ipred + 1 + nPredictors

        A_shuf          = A.copy()
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

#%% Parameters for model fitting:
minnneurons             = 10
maxnoiselevel           = 20 
nbehavPCs               = 5
nvideoPCs               = 10

#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

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
    X       = np.column_stack((X,sessions[ises].respmat_videopc[:nvideoPCs,:].T))
    X       = zscore(X,axis=0,nan_policy='omit')
    si      = SimpleImputer() #impute missing values
    X       = si.fit_transform(X)
    X_p     = pca.fit_transform(X) #reduce dimensionality

    # zscore neural responses
    respdata        = zscore(sessions[ises].respmat, axis=1)
    # respdata        = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

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
        # idx_source_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        # idx_source_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]
        idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    # sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]
        idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    # sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]

        subsampleneurons = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
        idx_source_N1 = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
        idx_source_N2 = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

        idx_target              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_source_N1) < minnneurons or len(idx_source_N2) < minnneurons:
            continue
        
        meanpopact_N1          = np.nanmean(respdata[idx_source_N1,:],axis=0)
        meanpopact_N2          = np.nanmean(respdata[idx_source_N2,:],axis=0)

        for iN,N in enumerate(idx_target):
            y       = respdata[N,:]
            # X       = np.column_stack((X_p,meanpopact_N1,meanpopact_N2))
            X       = np.column_stack((X_p,meanpopact_N1))
            S       = meanresp[N,:]
            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True)

            idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][N])

            cvR2_affine[ialp,:,idx_ses] = tempcvR2_models
            cvR2_preds[ialp,:,:,idx_ses] = tempcvR2_preds

#%% 
clrs_arealabelpairs = ['green','purple']
legendlabels        = ['FF','FB']
legendlabels        = ['FF\n(V1->PM)','FB\n(PM->V1)']

#%% Show overall model performance: 
fig,axes = plt.subplots(1,2,figsize=(3,2.5),sharey=True,sharex=True) 
for ialp in range(narealabelpairs):
    ax = axes[ialp]
    ax.plot(np.arange(2),np.nanmean(cvR2_affine[ialp,[0,3],:],axis=(1)),marker=None,linewidth=1.3,color='k')
    ax.scatter(np.arange(2),np.nanmean(cvR2_affine[ialp,[0,3],:],axis=1),s=50,c=sns.color_palette('magma',2))
    ax.set_ylim([0,my_ceil(np.nanmean(cvR2_affine[1,-1,:])*1.1,2)])
    ax_nticks(ax, 3)
    ax.set_xticks(np.arange(2),labels=['Stimulus','Affine'])
    ax.set_title(legendlabels[ialp])
axes[0].set_ylabel('Performance R2')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'AffineModel_R2_StimvsAffine_Overall_%dsessions' % (nSessions), formats = ['png'])

#%% Show overall model performance: 
fig,axes = plt.subplots(1,2,figsize=(5,2.5)) 
ax = axes[0]
ax.plot(np.arange(nAffModels),np.nanmean(cvR2_affine,axis=(0,2)),marker=None,
        linewidth=1.3,color='k')
ax.scatter(np.arange(nAffModels),np.nanmean(cvR2_affine,axis=(0,2)),s=50,
           c=sns.color_palette('magma',nAffModels))
ax.set_ylim([0,my_ceil(np.nanmean(cvR2_affine[:,-1,:]),2)])
ax.set_ylabel('Performance R2')
ax_nticks(ax, 5)
ax.set_xticks(np.arange(nAffModels),labels=AffModels)

ax = axes[1]
ymean   = np.nanmean(cvR2_preds[:,-1,:,:],axis=(0,2))
yerr    = np.nanstd(cvR2_preds[:,-1,:,:],axis=(0,2)) / np.sqrt(np.sum(~np.isnan(cvR2_preds[0,-1,:,:]),axis=1))*5
ax.errorbar(np.arange(nPredictors),ymean,yerr,linestyle='', color='k',marker='o',
            linewidth=2)
# ax.plot(np.arange(nPredictors),ymean,marker=None,
#         linewidth=1.3,color='k')
# shaded_error(np.arange(nPredictors),ymean,yerr,color='black',alpha=0.2,ax=ax)
ax.set_ylim([0,0.04])
ax.set_ylim([0,my_ceil(np.nanmean(cvR2_preds[:,-1,0,:]),2)])
ax.set_ylabel(u'$\Delta R^2$')
ax_nticks(ax, 5)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticks(np.arange(nPredictors),labels=predlabels,rotation=45,ha='right')
# my_savefig(fig,savedir,'AffineModel_R2_MultAddSep_PredictorsOverall_%dsessions' % (nSessions), formats = ['png'])

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
# my_savefig(fig,savedir,'AffineModel_Hist_FF_FB_R2_%dsessions' % (nSessions), formats = ['png'])

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
# my_savefig(fig,savedir,'AffineModel_R2_FF_vs_FB_%dsessions' % (nSessions), formats = ['png'])


#%% 


#%% 
fig,axes = plt.subplots(nPredictors,2,figsize=(4,nPredictors*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    for ipred in range(nPredictors):
        ax = axes[ipred,imodel]
        idx_N =  np.all((
                celldata['gOSI']>0,
                # celldata['gOSI']>0.2,
                # celldata['nearby'],
                    ),axis=0)
        ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)
        # yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(~np.isnan(cvR2_preds[:,model,ipred,idx_N]),axis=1))
        
        # confidence interval
        d = cvR2_preds[0,model,ipred,idx_N]
        yerror1 = np.nanmean(d) - stats.t.interval(0.99, df=len(d)-1, loc=np.nanmean(d), scale=np.nanstd(d, ddof=1) / np.sqrt(len(d)))
        d = cvR2_preds[1,model,ipred,idx_N]
        yerror2 = np.nanmean(d) - stats.t.interval(0.99, df=len(d)-1, loc=np.nanmean(d), scale=np.nanstd(d, ddof=1) / np.sqrt(len(d)))
        yerror = np.abs(np.array([yerror1,yerror2]).T)
        
        ax.bar([0,1],height=ymean,yerr=0,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
        ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                    linewidth=4)
        ax.set_xticks([0,1],labels=legendlabels)
        h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
                            cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
        p = p * narealabelpairs
        add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
        ax_nticks(ax, 3)
        if imodel == 0:
            ax.set_ylabel(u'$\Delta R^2$')
        if ipred ==0:
            ax.set_title(AffModels[model])
        # ax.legend(legendlabels,fontsize=9,frameon=False)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Affinemodulation_R2_Allpredictors_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% Show for the interarea modulation only:
ipred = -1
fig,axes = plt.subplots(1,2,figsize=(3,1*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    # for ipred in range(nPredictors):
    ax = axes[imodel]
    idx_N = np.all((
            celldata['gOSI']>0,
            # celldata['nearby'],
            # celldata['gOSI']>0.2,
            # celldata['nearby'],
                ),axis=0)
    ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)

    # confidence interval
    d = cvR2_preds[0,model,ipred,idx_N]
    yerror1 = np.nanmean(d) - stats.t.interval(0.99, df=len(d)-1, loc=np.nanmean(d), scale=np.nanstd(d, ddof=1) / np.sqrt(len(d)))
    d = cvR2_preds[1,model,ipred,idx_N]
    yerror2 = np.nanmean(d) - stats.t.interval(0.99, df=len(d)-1, loc=np.nanmean(d), scale=np.nanstd(d, ddof=1) / np.sqrt(len(d)))
    yerror = np.abs(np.array([yerror1,yerror2]).T)
    
    # yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(idx_N)/2)
    ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=4)
    ax.set_xticks([0,1],labels=legendlabels)
    h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
                        cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
    p = p * narealabelpairs
    add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
    ax_nticks(ax, 3)
    if imodel == 0:
        ax.set_ylabel(u'$\Delta R^2$')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% make a violinplot instead of a barplot
legendlabels        = ['FF','FB']
ipred               = -1
upperclipvalue      = my_ceil(np.nanpercentile(cvR2_preds[:,1,-1,:],95),2)
df = pd.DataFrame({'deltaR2': [], 'direction': [], 'modulation': []})
for ialp,arealabel in enumerate(legendlabels):
    for imodulation,modulation in enumerate(['Mult','Add']):
        df = pd.concat((df,pd.DataFrame({'deltaR2': cvR2_preds[ialp,imodulation+1,ipred,:], 'direction': np.repeat(arealabel,nCells), 'modulation': np.repeat(modulation,nCells)})))
df.dropna(inplace=True)
df['deltaR2'].clip(lower=-0.0025,upper=upperclipvalue,inplace=True)
fig,ax = plt.subplots(1,1,figsize=(4,4))
sns.violinplot(data=df, x="modulation", y="deltaR2", hue="direction", hue_order=legendlabels, 
               linewidth=1,palette=clrs_arealabelpairs,split=True, inner="quart",ax=ax)
ax.axhline(upperclipvalue,linestyle='--',color='k',linewidth=0.5)
ax.text(0.8,upperclipvalue+0.0005,'clip')
ax_nticks(ax, 7)
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_Affinemodulation_dR2_violinplot_%dsessions' % (nSessions), formats = ['png'])

#%% Is the amount of multiplicative and additive modulation related across neurons?

#%% Show for the interarea modulation only:
ipred = -1
fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharey='row',sharex=True)
for ialp,alp in enumerate(legendlabels): #Mult and Add
    ax = axes[ialp]
    idx_N = np.all((
            celldata['gOSI']>0,
                ),axis=0)
    x = cvR2_preds[ialp,1,ipred,idx_N]
    y = cvR2_preds[ialp,2,ipred,idx_N]

    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    sns.scatterplot(x=x,y=y,s=5,alpha=0.2,color=clrs_arealabelpairs[ialp],ax=ax)
    
    # sns.regplot(x=x,y=y,x_ci='sd',
    #             scatter_kws={'s':2,'alpha':0.2,'color': clrs_arealabelpairs[ialp]},ax=ax)
    ax.set_xlim(np.nanpercentile([x],[0,99.8]))
    ax.set_ylim(np.nanpercentile([y],[0,99.8]))
    ax_nticks(ax, 3)
    ax.set_xlabel(u'Multiplicative $\Delta R^2$')
    ax.set_ylabel(u'Additive  $\Delta R^2$')
    ax.set_title(legendlabels[ialp],fontsize=11)
    ax.text(0.8,0.8,'r = %.2f' % (stats.pearsonr(x,y)[0]),transform=ax.transAxes,fontsize=9) #print(stats.pearsonr(x,y)[0])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Corr_Mult_Add_uniqueR2_FF_FB_GR%dsessions' % (nSessions), formats = ['png'])




#%% Show the same figure but for behavioral predictors only
arealabels = ['PM','V1']
ipred = np.arange(1,nbehavPCs)
fig,axes = plt.subplots(1,2,figsize=(3,1*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    ax = axes[imodel]
    idx_N =  np.all((
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
        ax.set_ylabel(u'$\Delta R^2$')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Behavior_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% Is the effect present in all the sessions?
ipred = -1
datamat = np.full((narealabelpairs,2,nSessions),np.nan)
ncellmat = np.full((narealabelpairs,2,nSessions),np.nan)
for ises in range(nSessions):
    # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'])
    idx_ses =  np.all((
            np.isin(celldata['session_id'],sessions[ises].celldata['session_id']),
            # celldata['gOSI']>0.5,
            # celldata['gOSI']>0.2,
            # ~celldata['nearby'],
                ),axis=0)
    for ialp in range(narealabelpairs):
        ntargetcells = np.sum(~np.isnan(cvR2_preds[ialp,imodtype+1,ipred,idx_ses]))
        for imodtype in range(2):
            ncellmat[ialp,imodtype,ises] = ntargetcells
            datamat[ialp,imodtype,ises] = np.nanmean(cvR2_preds[ialp,imodtype+1,ipred,idx_ses])
            # datamat[ialp,imodtype,ises] = np.nanmedian(cvR2_preds[ialp,imodtype+1,ipred,idx_ses])

#%% Showing for different sessions and normalizing for mean for each session
#Normalizing for the mean across modulation types
datamatnorm = datamat
datamatnorm = datamat - np.mean(datamat,axis=(1),keepdims=True)
datamatnorm = datamatnorm + np.mean(datamat,axis=(0,1),keepdims=True)
datamatnorm[:,:,8] = np.nan

fig,axes = plt.subplots(1,2,figsize=(3,1.5*2),sharey='row',sharex=True)
ax = axes[0]
sns.lineplot(datamatnorm[:,0,:],linewidth=1,ax=ax,legend=False,linestyle='-',
                palette=sns.color_palette("dark",nSessions))
p = stats.ttest_rel(datamatnorm[0,0,:],datamatnorm[1,0,:],nan_policy='omit')[1]
# p = stats.ttest_rel(datamat[0,0,:],datamat[1,0,:],nan_policy='omit')[1]
add_stat_annotation(ax, 0.2, 0.8,0.014, p, h=0)
ax.set_title('Multiplicative')
ax.set_xticks([0,1],labels=legendlabels)
ax.set_ylabel(u'$\Delta R^2$ (norm)')
ax.set_ylim([0,my_ceil(np.nanmax(datamatnorm)*1.1,3)])
ax_nticks(ax, 5)
ax = axes[1]
sns.lineplot(datamatnorm[:,1,:],linewidth=1,ax=ax,legend=False,linestyle='-',
                palette=sns.color_palette("dark",nSessions))
p = stats.ttest_rel(datamatnorm[0,1,:],datamatnorm[1,1,:],nan_policy='omit')[1]
add_stat_annotation(ax, 0.2, 0.8,0.014, p, h=0)
ax.set_title('Additive')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_affinemodulation_sessionplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% Is the variability explained by behavior and population rate in the same neurons?
ibehavpred = 0
ipopactpred = -1
fig,axes = plt.subplots(2,2,figsize=(5,5),sharey=True,sharex=True)
for ialp in range(narealabelpairs):
    for imodtype in range(2):
        ax = axes[ialp,imodtype]
        # x = cvR2_preds[ialp,imodtype+1,ibehavpred,:]
        x = np.nansum(cvR2_preds[ialp,imodtype+1,range(nbehavPCs),:],axis=0)
        # x = np.nansum(cvR2_preds[ialp,imodtype+1,[0,1],:],axis=0)
        # y = np.nansum(cvR2_preds[ialp,imodtype+1,[2,3,4],:],axis=0)
        
        y = cvR2_preds[ialp,imodtype+1,ipopactpred,:]
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        sns.scatterplot(x=x,y=y,s=5,alpha=0.2,color=clrs_arealabelpairs[ialp],ax=ax)
       
        # sns.regplot(x=x,y=y,x_ci='sd',
        #             scatter_kws={'s':2,'alpha':0.2,'color': clrs_arealabelpairs[ialp]},ax=ax)
        ax.set_xlim(np.nanpercentile([x],[0,99.8]))
        ax.set_ylim(np.nanpercentile([y],[0,99.8]))
        ax_nticks(ax, 3)
        ax.set_xlabel(u'Behavior ΔR2')
        ax.set_ylabel(u'Cross area modulation ΔR2')
        ax.set_title(AffModels[imodtype+1] + ' ' + legendlabels[ialp],fontsize=11)
        ax.text(0.8,0.8,'r = %.2f' % (stats.pearsonr(x,y)[0]),transform=ax.transAxes,fontsize=9) #print(stats.pearsonr(x,y)[0])
        # ax.text(0.8,0.8,'r = %.2f' % (stats.spearmanr(x,y)[0]),transform=ax.transAxes,fontsize=9) #print(stats.pearsonr(x,y)[0])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'AffMod_Corr_Behavior_Poprate_GR%dsessions' % (nSessions), formats = ['png'])

#%% This is not the case for behavioral variables amongst each other:
corrs = []
for ialp in range(narealabelpairs):
    for imodtype in range(2):
        x = np.nansum(cvR2_preds[ialp,imodtype+1,range(nbehavPCs//2),:],axis=0)
        y = np.nansum(cvR2_preds[ialp,imodtype+1,range(nbehavPCs//2,nbehavPCs),:],axis=0)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        corrs.append(stats.pearsonr(x,y)[0])
print('Correlation between behavior variables: %.2f' % np.nanmean(corrs))

#%% Show that there is a gOSI difference: 
print('mean gOSI: V1: %1.2f +- %1.2f, PM: %1.2f +- %1.2f' % (np.nanmean(celldata['gOSI'][celldata['arealayerlabel'] == 'V1unlL2/3']),
                                             np.nanstd(celldata['gOSI'][celldata['arealayerlabel'] == 'V1unlL2/3']),
                                               np.nanmean(celldata['gOSI'][celldata['arealayerlabel'] == 'PMunlL2/3']),
                                               np.nanstd(celldata['gOSI'][celldata['arealayerlabel'] == 'PMunlL2/3'])))

#%% Show that the amount of multiplicative and additive modulation varies with orientation selectivity
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ipred = 1
x = celldata['gOSI']
sns.regplot(x=x,y=cvR2_preds[0,1,ipred,:],x_ci='sd',scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[0]},
            line_kws={'color': clrs_arealabelpairs[0],'linewidth':3,'linestyle':'-'},ax=axes[0])
sns.regplot(x=x,y=cvR2_preds[1,1,ipred,:],ci=99,scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[1]},
            line_kws={'color': clrs_arealabelpairs[1],'linewidth':3,'linestyle':'-'},ax=axes[0])
axes[0].set_ylim(np.nanpercentile(cvR2_preds[:,1,ipred,:],[0.5,99]))
axes[0].set_title('Multiplicative')

axes[0].set_ylabel(u'$\Delta R^2$')
axes[0].set_xlim([0,1])
ax_nticks(axes[0], 3)

sns.regplot(x=x,y=cvR2_preds[0,2,ipred,:],ci=99,scatter_kws={'s':2,'alpha':0.2,'color': clrs_arealabelpairs[0]},
            line_kws={'color': clrs_arealabelpairs[0],'linewidth':3,'linestyle':'-'},ax=axes[1])
sns.regplot(x=x,y=cvR2_preds[1,2,ipred,:],ci=99,scatter_kws={'s':2,'alpha':0.2,'color': clrs_arealabelpairs[1]},
            line_kws={'color': clrs_arealabelpairs[1],'linewidth':3,'linestyle':'-'},ax=axes[1])
axes[1].set_ylim(np.nanpercentile(cvR2_preds[:,2,ipred,:],[0.5,99]))
axes[1].set_title('Additive')
axes[1].set_xlim([0,1])
ax_nticks(axes[1], 3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affineR2_control_gOSI_GR%dsessions' % (nSessions), formats = ['png'])

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
# my_savefig(fig,savedir,'AffineModel_control_gOSI_table_GR_%dsessions' % (nSessions),formats=['png'])

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
# my_savefig(fig,savedir,'Behav_affineR2_control_gOSI_GR%dsessions' % (nSessions), formats = ['png'])

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
 #####  ####### ### #       #          ####### ######  ###    #    #        #####  
#     #    #     #  #       #             #    #     #  #    # #   #       #     # 
#          #     #  #       #             #    #     #  #   #   #  #       #       
 #####     #     #  #       #             #    ######   #  #     # #        #####  
      #    #     #  #       #             #    #   #    #  ####### #             # 
#     #    #     #  #       #             #    #    #   #  #     # #       #     # 
 #####     #    ### ####### #######       #    #     # ### #     # #######  #####  

#%% 
# respmat_videoME = []
# for ises in range(nSessions):   
#     respmat_videoME.append(list(sessions[ises].respmat_videome))
maxvideome              = 0.2
maxrunspeed             = 0.5

respmat_videoME = np.array([])
respmat_runspeed = np.array([])
for ises in range(nSessions):

    # respmat_videoME = np.append(respmat_videoME,sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome))
    respmat_videoME = np.append(respmat_videoME,sessions[ises].respmat_videome/np.nanpercentile(sessions[ises].respmat_videome,100))
    respmat_runspeed = np.append(respmat_runspeed,sessions[ises].respmat_runspeed)

# sns.histplot(respmat_videoME,bins=np.linspace(0,np.nanpercentile(respmat_videoME,99),50),element='step',stat='probability',fill=False)
# sns.histplot(respmat_runspeed,bins=np.linspace(-1,np.nanpercentile(respmat_runspeed,99),100),element='step',stat='probability',fill=False)

idx_T_still = np.logical_and(respmat_videoME < maxvideome,
                            respmat_runspeed < maxrunspeed)
fig,ax = plt.subplots(1,1,figsize=(3,3))

sns.scatterplot(ax=ax,x=respmat_videoME,y=respmat_runspeed,alpha=0.25,s=4,hue=idx_T_still,
                palette=['grey','black'],legend=False)
# ax.legend(['Still','Moving'],fontsize=9,frameon=False)
ax.text(0.6,0.8,'n = %d/%d \nstill trials \n(%.1f%%)' % (np.sum(idx_T_still),len(idx_T_still),np.sum(idx_T_still)/len(idx_T_still)*100),transform=ax.transAxes,fontsize=9)
ax.set_xlabel('Video ME (norm.)')
ax.set_ylabel('Running speed (cm/s)')
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'StillTrials_Selection',formats = ['png'])


#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
arealabelpairs  = [
                    'V1lab-PMunlL2/3',
                    # 'V1lab-PMlabL2/3',
                    # 'V1lab-PMunlL5',
                    # 'V1lab-PMlabL5',
                    'PMlab-V1unlL2/3',
                    # 'PMlab-V1labL2/3',
                    ]

arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    # 'V1lab-PMlabL2/3',
                    # 'V1lab-PMunlL5',
                    # 'V1lab-PMlabL5',
                    'PMlab-PMunl-V1unlL2/3',
                    # 'PMlab-V1labL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
nCells                  = len(celldata)

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

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    # zscore neural responses
    respdata        = zscore(sessions[ises].respmat, axis=1)
    #Get mean response per orientation (to predict trial by trial responses and multiplicative modulation)
    meanresp    = np.full_like(respdata,np.nan)
    trial_ori   = sessions[ises].trialdata['Orientation']
    for i,ori in enumerate(trial_ori.unique()):
        idx_T              = trial_ori == ori
        # idx_T              = np.logical_and(idx_T_still,trial_ori == ori)
        meanresp[:,idx_T] = np.nanmean(respdata[:,idx_T],axis=1)[:,None]

    # Fit affine modulation model for each arealabel pair (e.g. V1lab to PMunl for FF)
    # Compute mean population activity in area 1 (e.g. V1lab)
    # compute R2 of predicting responses in neurons in area 2 (e.g. PMunl) using 
    # behavioral PCs + mean pop activity in area 1
    for ialp,alp in enumerate(arealabelpairs):
        idx_source_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        idx_source_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_target              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        subsampleneurons = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
        idx_source_N1 = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
        idx_source_N2 = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

        if len(idx_source_N1) < minnneurons:
            continue
        
        meanpopact_N1          = np.nanmean(respdata[idx_source_N1,:],axis=0)
        meanpopact_N2          = np.nanmean(respdata[idx_source_N2,:],axis=0)
        # meanpopact          = np.nanmean(zscore(respdata[idx_N1,:],axis=1),axis=0)

        for iN,N in enumerate(idx_target):
            y       = respdata[N,:]
            X       = np.column_stack((X_p,meanpopact_N1))
            # X       = np.column_stack((X_p,meanpopact_N1,meanpopact_N2))
            # X       = np.column_stack((X_p,meanpopact))
            S       = meanresp[N,:]

            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y[idx_T_still],X[idx_T_still],S[idx_T_still],kfold=5,subtract_shuffle=True)

            idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][N])

            cvR2_affine[ialp,:,idx_ses] = tempcvR2_models
            cvR2_preds[ialp,:,:,idx_ses] = tempcvR2_preds

#%% 
ipred = -1
fig,axes = plt.subplots(1,2,figsize=(3,1*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    # for ipred in range(nPredictors):
    ax = axes[imodel]
    idx_N = np.all((
            celldata['gOSI']>0,
            # celldata['nearby'],
                ),axis=0)
    ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)
    yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(idx_N)/2)
    ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=4)
    ax.set_xticks([0,1],labels=legendlabels)
    h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
                        cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
    p = p * narealabelpairs
    add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
    ax_nticks(ax, 3)
    if imodel == 0:
        ax.set_ylabel(u'$\Delta R^2$')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'StillTrials_FF_FB_affinemodulation_dR2_%dsessions' % (nSessions), formats = ['png'])



#%% 
######     #    ####### ### #######    #          #    ######        # #     # #     # #       
#     #   # #      #     #  #     #    #         # #   #     #      #  #     # ##    # #       
#     #  #   #     #     #  #     #    #        #   #  #     #     #   #     # # #   # #       
######  #     #    #     #  #     #    #       #     # ######     #    #     # #  #  # #       
#   #   #######    #     #  #     #    #       ####### #     #   #     #     # #   # # #       
#    #  #     #    #     #  #     #    #       #     # #     #  #      #     # #    ## #       
#     # #     #    #    ### #######    ####### #     # ######  #        #####  #     # ####### 

#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

minnneurons             = 20
narealabelpairs         = len(arealabelpairs)

AffModels               = np.array(['Stim','Mult','Add','Both'])
nAffModels              = len(AffModels)

#initialize storage variables
nPredictors             = nbehavPCs + 1 # +1 for mean pop activity
predlabels_orig         = np.array([f'Behav_PC{i}' for i in range(nbehavPCs)] + ['LabPopAct'])
cvR2_preds_orig         = np.full((narealabelpairs,nAffModels,nPredictors,nCells),np.nan)
predlabels_diff         = np.array([f'Behav_PC{i}' for i in range(nbehavPCs)] + ['DiffPopAct'])
cvR2_preds_diff         = np.full((narealabelpairs,nAffModels,nPredictors,nCells),np.nan)
nPredictors             = nbehavPCs + 2 # +1 for mean pop activity
predlabels_both         = np.array([f'Behav_PC{i}' for i in range(nbehavPCs)] + ['LabPopAct','UnlPopAct'])
cvR2_preds_both         = np.full((narealabelpairs,nAffModels,nPredictors,nCells),np.nan)

pca                     = PCA(n_components=nbehavPCs)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing affine modulation models'):
    #construct behavioral design matrix
    X       = np.stack((sessions[ises].respmat_videome,
                    sessions[ises].respmat_runspeed,
                    sessions[ises].respmat_pupilarea,
                    sessions[ises].respmat_pupilareaderiv,
                    sessions[ises].respmat_pupilx,
                    sessions[ises].respmat_pupily),axis=1)
    X       = np.column_stack((X,sessions[ises].respmat_videopc[:nvideoPCs,:].T))
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
    # behavioral PCs + mean pop activity in area 1, or diff or both etc.
    for ialp,alp in enumerate(arealabelpairs):
        # idx_source_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        # idx_source_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]
        idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    # sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]
        idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    # sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]

        subsampleneurons    = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
        idx_source_N1       = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
        idx_source_N2       = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

        idx_target          = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_source_N1) < minnneurons or len(idx_source_N2) < minnneurons:
            continue
        
        meanpopact_N1          = np.nanmean(respdata[idx_source_N1,:],axis=0)
        meanpopact_N2          = np.nanmean(respdata[idx_source_N2,:],axis=0)

        for iN,N in enumerate(idx_target):
            idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][N])
            y       = respdata[N,:]
            X       = np.column_stack((X_p,meanpopact_N1))
            S       = meanresp[N,:]
            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True)
            cvR2_preds_orig[ialp,:,:,idx_ses] = tempcvR2_preds

            X       = np.column_stack((X_p,meanpopact_N1 - meanpopact_N2))
            S       = meanresp[N,:]
            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True)
            cvR2_preds_diff[ialp,:,:,idx_ses] = tempcvR2_preds

            X       = np.column_stack((X_p,meanpopact_N1,meanpopact_N2))
            S       = meanresp[N,:]
            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True)
            cvR2_preds_both[ialp,:,:,idx_ses] = tempcvR2_preds

#%% 
fig,axes = plt.subplots(1,2,figsize=(6,2.5))
ax=axes[0]
ax.plot(meanpopact_N1,color='red',label=legendlabels[0],linewidth=0.1,alpha=1)
ax.plot(meanpopact_N2,color='grey',label=legendlabels[1],linewidth=0.1,alpha=1)
ax.set_xlabel('Trials')
ax.set_ylabel('Population activity')
ax.legend(['PMLab','PMUnl'],fontsize=9,frameon=False)
ax = axes[1]
# ax.scatter(meanpopact_N2,meanpopact_N1,color='k',s=0.1)
ax.scatter(meanpopact_N2,meanpopact_N1,c=meanpopact_N1 - meanpopact_N2,s=0.2,cmap='coolwarm',vmin=-0.5,vmax=0.5)
ax.text(0.1,0.8,'High FB',transform=ax.transAxes)
ax.text(0.7,0.2,'Low FB',transform=ax.transAxes)
ax.plot([-1,1],[-1,1],color='k',linewidth=0.2)
ax.set_xlim(np.percentile(meanpopact_N1,[0,99.5]))
ax.set_ylim(np.percentile(meanpopact_N1,[0,99.5]))
ax_nticks(ax, 3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=1)
# my_savefig(fig,savedir,'Diff_LabUnl_GR%dsessions' % (nSessions), formats = ['png'])

#%% 
ipred = nbehavPCs
# modelnames = ['Original','Diff','Both']
modelnames = ['Orig','Both','Diff']
data_models = np.full((3,nCells),np.nan)
data_models[0,:] = np.nanmean(cvR2_preds_orig[:,:,ipred,:], axis=(0,1))
data_models[1,:] = np.nanmean(cvR2_preds_both[:,:,ipred,:], axis=(0,1))
data_models[2,:] = np.nanmean(cvR2_preds_diff[:,:,ipred,:], axis=(0,1))

fig,ax = plt.subplots(1,1,figsize=(2.5,2.5))
sns.lineplot(x=range(3),y=np.nanmean(data_models,axis=1),color='k',alpha=0.5,ax=ax,marker='o',markersize=10)
for i in range(3):
    pval = stats.wilcoxon(data_models[i,:],nan_policy='omit')[1]
    ax.text(i,np.nanmean(data_models[i,:]),get_sig_asterisks(pval))
# sns.lineplot(data_models,color='k',alpha=0.5,ax=ax)
# sns.barplot(data_models,color='k',alpha=0.5)
ax.set_xticks(range(3),modelnames)
ax.set_ylabel(u'$\Delta R^2$')
ax.set_ylim([0,0.01])
ax_nticks(ax, 3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=1)
my_savefig(fig,savedir,'ModelPerf_Labpop_Affine_FF_FB_GR%dsessions' % (nSessions), formats = ['png'])


#%% 
data = cvR2_preds_orig
# data = cvR2_preds_diff
data = cvR2_preds_both

ipred = nbehavPCs
fig,axes = plt.subplots(1,2,figsize=(3,1*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    ax = axes[imodel]
    idx_N = np.all((
            celldata['gOSI']>0,
            # celldata['gOSI']>0.5,
            # celldata['nearby'],
                ),axis=0)
    ymean = np.nanmean(data[:,model,ipred,idx_N],axis=1)
    # yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(idx_N)/2)
    yerror = np.nanstd(data[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(~np.isnan(data[:,model,ipred,idx_N]),axis=1)) * 3

    ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=4)
    ax.set_xticks([0,1],labels=legendlabels)
    h,p = stats.ttest_ind(data[0,model,ipred,idx_N],
                        data[1,model,ipred,idx_N],nan_policy='omit')
    p = p * narealabelpairs
    add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
    ax_nticks(ax, 3)
    if imodel == 0:
        ax.set_ylabel(u'$\Delta R^2$')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_RateDiff_affinemodulation_dR2_%dsessions' % (nSessions), formats = ['png'])
my_savefig(fig,savedir,'FF_FB_RateBoth_affinemodulation_dR2_%dsessions' % (nSessions), formats = ['png'])

#%% make a violinplot instead of a barplot
legendlabels        = ['FF','FB']
upperclipvalue      = 0.01
ipred               = -1
df = pd.DataFrame({'deltaR2': [], 'direction': [], 'modulation': []})
for ialp,arealabel in enumerate(legendlabels):
    for imodulation,modulation in enumerate(['Mult','Add']):
        df = pd.concat((df,pd.DataFrame({'deltaR2': cvR2_preds_diff[ialp,imodulation+1,ipred,:], 'direction': np.repeat(arealabel,nCells), 'modulation': np.repeat(modulation,nCells)})))
df.dropna(inplace=True)
df['deltaR2'].clip(lower=-0.0025,upper=upperclipvalue,inplace=True)
fig,ax = plt.subplots(1,1,figsize=(4,4))
sns.violinplot(data=df, x="modulation", y="deltaR2", hue="direction", hue_order=legendlabels, 
               linewidth=1,palette=clrs_arealabelpairs,split=True, inner="quart",ax=ax)
ax.axhline(upperclipvalue,linestyle='--',color='k',linewidth=0.5)
ax.text(0.8,upperclipvalue+0.0005,'clip')
ax_nticks(ax, 7)
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_RateDiff_affinemodulation_dR2_violinplot_%dsessions' % (nSessions), formats = ['png'])



#%% 
#          #    #     # ####### ######   #####  
#         # #    #   #  #       #     # #     # 
#        #   #    # #   #       #     # #       
#       #     #    #    #####   ######   #####  
#       #######    #    #       #   #         # 
#       #     #    #    #       #    #  #     # 
####### #     #    #    ####### #     #  #####  



#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 
    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 

#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 

arealabelpairs          = [
                            'PMlabL2/3-PMunlL2/3-V1unlL2/3',
                            'PMlabL5-PMunlL5-V1unlL2/3',
                            ]
narealabelpairs         = len(arealabelpairs)
legendlabels            = np.array(['FBL23','FBL5'])

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
nCells                  = len(celldata)

minnneurons             = 10

nbehavPCs               = 5
nvideoPCs               = 5
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
    X       = np.column_stack((X,sessions[ises].respmat_videopc[:nvideoPCs,:].T))
    X       = zscore(X,axis=0,nan_policy='omit')
    si      = SimpleImputer() #impute missing values
    X       = si.fit_transform(X)
    X_p     = pca.fit_transform(X) #reduce dimensionality

    # zscore neural responses
    respdata        = zscore(sessions[ises].respmat, axis=1)
    # respdata        = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

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
        idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0],
                                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]
        idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]

        subsampleneurons = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
        idx_source_N1 = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
        idx_source_N2 = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

        idx_target              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_source_N1) < minnneurons or len(idx_source_N2) < minnneurons:
            continue
        
        meanpopact_N1          = np.nanmean(respdata[idx_source_N1,:],axis=0)
        meanpopact_N2          = np.nanmean(respdata[idx_source_N2,:],axis=0)

        for iN,N in enumerate(idx_target):
            y       = respdata[N,:]
            # X       = np.column_stack((X_p,meanpopact_N1,meanpopact_N2))
            X       = np.column_stack((X_p,meanpopact_N1))
            # X       = np.column_stack((X_p,meanpopact_N1/meanpopact_N2))
            # X       = np.column_stack((X_p,meanpopact_N1 - meanpopact_N2,meanpopact_N1/meanpopact_N2))
            # X       = np.column_stack((X_p,meanpopact_N1 - meanpopact_N2))
            S       = meanresp[N,:]
            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True)

            idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][N])

            cvR2_affine[ialp,:,idx_ses] = tempcvR2_models
            cvR2_preds[ialp,:,:,idx_ses] = tempcvR2_preds

#%% 
ipred = -1
fig,axes = plt.subplots(1,2,figsize=(3,1*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    # for ipred in range(nPredictors):
    ax = axes[imodel]
    idx_N = np.all((
            celldata['gOSI']>0.3,
            # celldata['nearby'],
                ),axis=0)
    ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)
    yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(idx_N)/2)
    ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=4)
    ax.text(0,ymean.max()/10,'n=%d' % np.sum(~np.isnan(cvR2_preds[0,model,ipred,idx_N])),fontsize=6,ha='center')
    ax.text(1,ymean.max()/10,'n=%d' % np.sum(~np.isnan(cvR2_preds[1,model,ipred,idx_N])),fontsize=6,ha='center')
    ax.set_xticks([0,1],labels=legendlabels)
    h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
                        cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
    p = p * narealabelpairs
    add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
    ax_nticks(ax, 3)
    if imodel == 0:
        ax.set_ylabel(u'$\Delta R^2$')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FB_PMLayers_V1_affinemodulation_dR2_%dsessions' % (nSessions), formats = ['png'])


#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'V1lab-V1unl-PMunlL5',
                    'PMlab-PMunl-V1unlL2/3',
                    'PMlab-PMunl-V1unlL5',
                    ]
legendlabels     = ['FF - PML2/3',
                    'FF - PML5',
                    'FB - V1L2/3',
                    'FB - V1L5',
                    ]
clrs_arealabelpairs = ['green','green','purple','purple']

narealabelpairs         = len(arealabelpairs)

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
    X       = np.column_stack((X,sessions[ises].respmat_videopc[:nvideoPCs,:].T))
    X       = zscore(X,axis=0,nan_policy='omit')
    si      = SimpleImputer() #impute missing values
    X       = si.fit_transform(X)
    X_p     = pca.fit_transform(X) #reduce dimensionality

    # zscore neural responses
    respdata        = zscore(sessions[ises].respmat, axis=1)
    # respdata        = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

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
        # idx_source_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        # idx_source_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]
        idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    # sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]
        idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    # sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]

        subsampleneurons = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
        idx_source_N1 = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
        idx_source_N2 = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

        idx_target              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_source_N1) < minnneurons or len(idx_source_N2) < minnneurons:
            continue
        
        meanpopact_N1          = np.nanmean(respdata[idx_source_N1,:],axis=0)
        meanpopact_N2          = np.nanmean(respdata[idx_source_N2,:],axis=0)

        for iN,N in enumerate(idx_target):
            y       = respdata[N,:]
            # X       = np.column_stack((X_p,meanpopact_N1,meanpopact_N2))
            X       = np.column_stack((X_p,meanpopact_N1))
            S       = meanresp[N,:]
            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True)

            idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][N])

            cvR2_affine[ialp,:,idx_ses] = tempcvR2_models
            cvR2_preds[ialp,:,:,idx_ses] = tempcvR2_preds

#%% 
ipred = -1
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    # for ipred in range(nPredictors):
    ax = axes[imodel]
    idx_N = np.all((
            celldata['gOSI']>0,
            # celldata['nearby'],
                ),axis=0)
    ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)
    yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(idx_N)/2)
    ax.bar(range(4),height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar(range(4),y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=4)
    ax.text(0,ymean.max()/10,'n=%d' % np.sum(~np.isnan(cvR2_preds[0,model,ipred,idx_N])),fontsize=6,ha='center')
    ax.text(1,ymean.max()/10,'n=%d' % np.sum(~np.isnan(cvR2_preds[1,model,ipred,idx_N])),fontsize=6,ha='center')
    ax.text(2,ymean.max()/10,'n=%d' % np.sum(~np.isnan(cvR2_preds[2,model,ipred,idx_N])),fontsize=6,ha='center')
    ax.text(3,ymean.max()/10,'n=%d' % np.sum(~np.isnan(cvR2_preds[3,model,ipred,idx_N])),fontsize=6,ha='center')
    ax.set_xticks(range(4),labels=legendlabels,fontsize=8,rotation=45)
    # h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
    #                     cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
    # p = p * narealabelpairs
    # add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
    # ax_nticks(ax, 3)
    if imodel == 0:
        ax.set_ylabel(u'$\Delta R^2$')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_TargetLayers_affinemodulation_dR2_%dsessions' % (nSessions), formats = ['png'])

