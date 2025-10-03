
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import statsmodels.formula.api as smf

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings,plot_PCA_gratings_3D_traces
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.psth import compute_tensor
from scipy.stats import zscore
from utils.plot_lib import shaded_error
from utils.regress_lib import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from utils.rf_lib import filter_nearlabeled

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE12223','2024_06_10']])

# load sessions lazy: 
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,filter_areas=['V1','PM'])
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list,filter_areas=['V1'])

#   Load proper data and compute average trial responses:                      
sessions[0].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_areas=['V1','PM'])

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)


#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)
                                # calciumversion='deconv',keepraw=True)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)

#%%
sessions = fitAffine_GR_singleneuron_split(sessions,radius=500,perc=50)


#%% Decoding performance as a function of population rate for differently coupled neurons
nActBins = 5
nVarBins = 5

kfold = 5
lam = 1
model_name = 'LOGR'
scoring_type = 'accuracy_score'
# scoring_type = 'balanced_accuracy_score'

cellvar = 'pop_coupling'
# cellvar = 'aff_alpha_grsplit'
cellvar = 'aff_beta_grsplit'

error_cv = np.full((nSessions,nActBins,nVarBins),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Decoding stimulus ori across sessions',total=nSessions):
    ori_ses                 = ses.trialdata['Orientation']

    data                    = zscore(ses.respmat, axis=1)

    poprate                 = np.nanmean(data,axis=0)
    popratequantiles        = np.percentile(poprate,range(0,101,100//nActBins))

    varquantiles            = np.percentile(ses.celldata[cellvar],range(0,101,100//nVarBins))

    if lam is None:
        y = ori_ses
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
        X = data.T
        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
        lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

    for ivarbin in range(nVarBins):
        idx_N     = np.where(np.all((
                            ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20,
                            ses.celldata[cellvar]>varquantiles[ivarbin],
                            ses.celldata[cellvar]<=varquantiles[ivarbin+1]),axis=0))[0]

        for iqrpoprate in range(len(popratequantiles)-1):
            idx_T = np.all((poprate>popratequantiles[iqrpoprate],
                            poprate<=popratequantiles[iqrpoprate+1]),axis=0)

            X = data[np.ix_(idx_N,idx_T)].T
            y = ori_ses[idx_T]

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array

            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            # error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name='LDA',kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=False)
            error_cv[ises,iqrpoprate,ivarbin],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                            lam=lam,norm_out=False,subtract_shuffle=False)

#%% Plot error as a function of population rate and for different populations
clrs = sns.color_palette('viridis',nVarBins)

fig,ax = plt.subplots(1,1,figsize=(3,3))
for ivarbin in range(nVarBins):
    # shaded_error(np.arange(nActBins)+1,error_cv[:,:,iqrpopcoupling],error='sem',ax=ax,color=clrs_popcoupling[iqrpopcoupling])
    ax.plot(np.arange(nActBins)+1,np.nanmean(error_cv[:,:,ivarbin],axis=0),
            color=clrs[ivarbin],linewidth=2)
ax.set_xlabel('Population rate (quantile)')
ax.set_ylabel('Decoding accuracy\n (crossval Log. Regression)')
ax.set_ylim([0,1])
ax.set_xticks(np.arange(nActBins)+1)
ax.axhline(y=1/len(np.unique(ses.trialdata['Orientation'])), color='grey', linestyle='--', linewidth=1)
ax.text(0.5,0.15,'Chance',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='grey')
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
ax.legend(['Weak','','Intermediate','','Strong'],
          reverse=True,fontsize=7,frameon=False,title='%s bins' % cellvar,bbox_to_anchor=(0.9,1), loc='upper left')

sns.despine(fig=fig,trim=True,top=True,right=True)

my_savefig(fig,savedir,'Decoding_Ori_LOGR_ActBins_%s_%dsessions' % (cellvar,nSessions), formats = ['png'])



#%% 
lam = 1
model_name = 'SVR'
scoring_type = 'circular_abs_error'
# scoring_type = 'mean_squared_error'
error_cv,_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                            lam=lam,norm_out=False,subtract_shuffle=False)

model_name = 'Ridge'
scoring_type = 'circular_abs_error'

lam = 0.5
error_cv,_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                lam=lam,norm_out=False,subtract_shuffle=False)
print(error_cv)

#%%
for ises,ses in enumerate(sessions):
    ses.celldata['near_labeled'] = filter_nearlabeled(ses,radius=50)

#%% Decoding performance as a function of population rate for differently coupled neurons
nActBins        = 5
kfold           = 5
lam             = 1
model_name      = 'LOGR'
scoring_type    = 'accuracy_score'
# scoring_type = 'balanced_accuracy_score'

# model_name = 'SVR'
# model_name = 'Ridge'

# scoring_type = 'circular_abs_error'

arealabels      = np.array(['V1unl','V1lab','PMunl','PMlab'])
narealabels     = len(arealabels)
nsampleneurons  = 25
nmodelfits      = 20

error_cv        = np.full((narealabels,nActBins,nSessions,nmodelfits),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Decoding stimulus ori across sessions',total=nSessions):
    ori_ses                 = ses.trialdata['Orientation']
    ori_ses                 = np.mod(ori_ses,180)

    data                    = zscore(ses.respmat, axis=1)

    poprate                 = np.nanmean(data,axis=0)
    popratequantiles        = np.percentile(poprate,range(0,101,100//nActBins))

    if lam is None:
        y = ori_ses
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
        X = data.T
        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
        lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

    for ial,al in enumerate(arealabels):
        idx_N              = np.all((sessions[ises].celldata['arealabel']==al,
                            # sessions[ises].celldata['noise_level']<20,
                            sessions[ises].celldata['near_labeled'],
                            sessions[ises].celldata['OSI']>np.percentile(sessions[ises].celldata['OSI'],25),
                            ),axis=0)

        if np.sum(idx_N) < nsampleneurons:
            continue
        
        for imf in range(nmodelfits):
            idx_N_sub = np.random.choice(np.where(idx_N)[0],nsampleneurons,replace=False)
            for iqrpoprate in range(len(popratequantiles)-1):
                idx_T = np.all((poprate>popratequantiles[iqrpoprate],
                                            poprate<=popratequantiles[iqrpoprate+1]),axis=0)

                X = data[np.ix_(idx_N_sub,idx_T)].T
                y = ori_ses[idx_T]

                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array

                X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                # error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name='LDA',kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=False)
                error_cv[ial,iqrpoprate,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)


#%% Plot error as a function of population rate and for different populations with coupling
# clrs = sns.color_palette('colorblind',n_colors=nPopCouplingBins)
clrs_arealabeled = get_clr_area_labeled(arealabels)

fig,ax = plt.subplots(1,1,figsize=(3,3))
handles = []
for ial,al in enumerate(arealabels):
    handles.append(shaded_error(np.arange(nActBins)+1,np.nanmean(error_cv[ial,:,:,:],axis=2).T,error='sem',ax=ax,
                 color=clrs_arealabeled[ial]))
    # ax.plot(np.arange(nActBins)+1,np.nanmean(error_cv[ial,:,:,:],axis=(1,2)),
            # color=clrs_arealabeled[ial],linewidth=2)
ax.set_xlabel('Population rate (quantile)')
ax.set_ylabel('Decoding accuracy\n (crossval Log. Regression)')
# ax.set_ylim([0,.8])
ax.set_xticks(np.arange(nActBins)+1)
ax.axhline(y=1/len(np.unique(ori_ses)), color='grey', linestyle='--', linewidth=1)
ax.text(0.5,1/len(np.unique(ori_ses))+0.07,'Chance',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='grey')
ax.legend(handles,arealabels,fontsize=7,frameon=False,bbox_to_anchor=(0.92,0.6), loc='center left')

plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True)
# my_savefig(fig,savedir,'Decoding_Dir_LOGR_ActBins_LabeledPops_%dsessions' % nSessions, formats = ['png'])
my_savefig(fig,savedir,'Decoding_Ori_LOGR_ActBins_LabeledPops_%dsessions' % nSessions, formats = ['png'])

#%% Plot error as a function of population rate and for different populations with coupling
# clrs = sns.color_palette('colorblind',n_colors=nPopCouplingBins)
clrs_arealabeled = get_clr_area_labeled(arealabels)

fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=True,sharey=True)
handles = []
ax = axes[0]
datatoplot_V1 = np.nanmean(error_cv[1,:,:,:] - error_cv[0,:,:,:],axis=2).T
# datatoplot_V1 -= np.nanmean(datatoplot_V1,axis=1,keepdims=True)
# datatoplot = np.nanmean(error_cv[1,:,:,:] / error_cv[0,:,:,:],axis=2).T
# datatoplot = np.nanmean(error_cv[0,:,:,:] / error_cv[1,:,:,:],axis=2).T
handles.append(shaded_error(np.arange(nActBins)+1,datatoplot_V1,error='sem',ax=ax,
                color='red'))
ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel('Population rate (quantile)')
ax.set_ylabel('Decoding accuracy\n (lab-unl)')


ax = axes[1]
datatoplot_PM = np.nanmean(error_cv[3,:,:,:] - error_cv[2,:,:,:],axis=2).T
# datatoplot_PM -= np.nanmean(datatoplot_PM,axis=1,keepdims=True)
handles.append(shaded_error(np.arange(nActBins)+1,datatoplot_PM,error='sem',ax=ax,
                color='red'))
ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)

# ax.set_ylim([-0.05,0.05])
# ax.set_ylim([-0.06,0.06])
ax.set_xticks(np.arange(nActBins)+1)
# ax.legend(handles,arealabels,fontsize=7,frameon=False,bbox_to_anchor=(0.92,0.6), loc='center left')
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True)
# my_savefig(fig,savedir,'Decoding_Dir_LOGR_ActBins_LabeledPops_%dsessions' % nSessions, formats = ['png'])
my_savefig(fig,savedir,'Decoding_Ori_LOGR_ActBins_LabeledPops_Diff_%dsessions' % nSessions, formats = ['png'])


df_V1 = pd.DataFrame({'perf': datatoplot_V1.flatten(),
                   'act': np.tile(np.arange(nActBins)+1,nSessions),
                   'session_id': np.repeat(np.arange(nSessions),nActBins)})
df_V1['area'] = 'V1'
df_PM = pd.DataFrame({'perf': datatoplot_PM.flatten(),
                   'act': np.tile(np.arange(nActBins)+1,nSessions),
                   'session_id': np.repeat(np.arange(nSessions),nActBins)})
df_PM['area'] = 'PM'
df = pd.concat([df_V1,df_PM])
df.dropna(inplace=True)

# model     = smf.mixedlm("perf ~ act + C(area)", data=df,groups=df["session_id"])
model     = smf.mixedlm("perf ~ act + area + act:area", data=df,groups=df["session_id"])
model     = smf.mixedlm("perf ~ act * area", data=df,groups=df["session_id"])
# model     = smf.mixedlm("perf ~ act:area", data=df,groups=df["session_id"])
result    = model.fit(reml=False)
print(result.summary())

