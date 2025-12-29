
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import zscore
from scipy.optimize import curve_fit

os.chdir('c:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive
from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

savedir =  os.path.join(get_local_drive(),'OneDrive\\Fellowships & Grants\\2025 VENI\\2025 Application\\Full Proposal\\')

#%% #############################################################################

sessions,nSessions   = filter_sessions(protocols = 'GR')


#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion='deconv')

#%%
sessions = compute_pairwise_anatomical_distance(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions,version='radius_500')

#%% Filter out cells that are close to a labeled cell:
for ises,ses in enumerate(sessions):
    ses.celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=30)

#%% Concatenate cell data:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% How are neurons are coupled to the population rate of different areas:
arealabels = np.array([['V1unl','V1lab'],['PMunl','PMlab']])
clrs_area = get_clr_labeled()
fig,axes = plt.subplots(1,2,figsize=(4,2))
bins = np.linspace(-0.05,0.5,200)
for ixarea,xarea in enumerate(arealabels):
    ax = axes[ixarea]
    print(xarea)
    # print(ixarea)
    for ilabeled,labeled in enumerate(['unl','lab']):
        idx_N = np.all((celldata['arealabel']==xarea[ilabeled],
                        # celldata['noise_level']<20,
                        celldata['gOSI']>0.4,
                        # celldata['nearby'],
                        ),axis=0)
        sns.histplot(
            celldata['pop_coupling'][idx_N],bins=bins,
            stat='density',element='step',fill=False,
            color=clrs_area[ilabeled],cumulative=True,ax=ax)


# 
# %% 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_areas=['V1'])

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion='deconv')



#%%
sessions = compute_pairwise_anatomical_distance(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions,version='radius_500')


#%% fit von mises
def vonmises(x,amp,loc,scale):
    return amp * np.exp( (np.cos(x-loc) - 1) / (2 * scale**2) )

def double_vonmises_pi_constrained(x,amp1,amp2,loc,scale,offset):
    return amp1 * np.exp( (np.cos(x-loc) - 1) / (2 * scale**2) ) + amp2 * np.exp( (np.cos(x-loc-np.pi) - 1) / (2 * scale**2) ) + offset











#%% Modulation of tuned response for different coupled neurons: 

ises                = 1
nPopCouplingBins    = 20
nPopRateBins        = 2
binedges_popcoupling = np.percentile(sessions[ises].celldata['pop_coupling'],np.linspace(0,100,nPopCouplingBins+1))

# respmat           = zscore(sessions[ises].respmat,axis=1)
respmat             = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

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
        meandata[n,iPopRateBin,:] = np.roll(meandata[n,iPopRateBin,:],-prefori[n]+4)
        stddata[n,iPopRateBin,:] = np.roll(stddata[n,iPopRateBin,:],-prefori[n]+4)

# clrs_popcoupling    = sns.color_palette('viridis',nPopRateBins)
clrs_popcoupling    = np.array(['#FFADB5','#B60500'])
orilabels = np.array(['-90','pref','+90','+180'])
subplotlabels = np.array(['Soloists','Choristers'])
linestyles = ['--','-']
# fig,axes = plt.subplots(1,nPopCouplingBins,figsize=(12,2.5),sharey=True,sharex=True)
fig,axes = plt.subplots(1,2,figsize=(5,2),sharey=True,sharex=True)
for iPopCouplingBin,PopCouplingBin in enumerate([0,nPopCouplingBins-1]):
    ax = axes[iPopCouplingBin]
    idx_popcoupling = np.all((
                            sessions[ises].celldata['gOSI']>0.4,
                            # sessions[ises].celldata['roi_name']=='V1',
                            sessions[ises].celldata['pop_coupling']>binedges_popcoupling[PopCouplingBin],
                            sessions[ises].celldata['pop_coupling']<=binedges_popcoupling[PopCouplingBin+1]),axis=0)

    for iPopRateBin in range(nPopRateBins):
        # ax.plot(ustim,np.mean(meandata[idx_popcoupling,iPopRateBin,:],axis=0),color=clrs_popcoupling[iPopRateBin],lw=0)
        ax.plot(ustim,np.mean(meandata[idx_popcoupling,iPopRateBin,:],axis=0),color=clrs_popcoupling[iPopRateBin],
                marker='.',markersize=10,lw=0)
    
        xdata  = np.radians(ustim)
        allmean = np.mean(meandata[idx_popcoupling,iPopRateBin,:],axis=0)
        popt_low, pcov = curve_fit(double_vonmises_pi_constrained, xdata, allmean,p0=[1,1,xdata[np.argmax(allmean)],0.25,0])
        xfit = np.linspace(0,2*math.pi,100)
        yfit = double_vonmises_pi_constrained(xfit, *popt_low)
        ax.plot(np.degrees(xfit),yfit,color=clrs_popcoupling[iPopRateBin],linestyle=linestyles[iPopCouplingBin],linewidth=0.8)
    #High activity in other area
    ax.set_ylim([0,my_ceil(np.max(allmean),1)])
    ax.set_xticks(ustim[::4],labels=orilabels,fontsize=7)
    # ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
    ax.set_xlabel('Stimulus direction',fontsize=9)
    # ax.set_ylabel('Neuron',fontsize=9)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(subplotlabels[iPopCouplingBin])
sns.despine(fig=fig, top=True, right=True, offset=1,trim=False)
# my_savefig(fig=fig,savedir=savedir,filename='VENI_populationcoupling_gain')

#%%




#%% Decoding performance as a function of population rate for differently coupled neurons
nActBins = 5
nPopCouplingBins = 10
kfold = 5
lam = 1
model_name = 'LOGR'
scoring_type = 'accuracy_score'
# scoring_type = 'balanced_accuracy_score'

error_cv = np.full((nSessions,nActBins,nPopCouplingBins),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Decoding stimulus ori across sessions',total=nSessions):
    ori_ses                 = ses.trialdata['Orientation']

    data                    = zscore(ses.respmat, axis=1)

    # data             = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    # poprate             = np.nanmean(data, axis=0)
    poprate                 = np.nanmean(data,axis=0)
    popratequantiles        = np.percentile(poprate,range(0,101,100//nActBins))

    N                       = np.shape(data)[0]
    popcoupling             = sessions[ises].celldata['pop_coupling'].to_numpy()
    if np.sum(np.isnan(popcoupling))>0:
        popcoupling         = np.array([np.corrcoef(data[i,:],poprate)[0,1] for i in range(N)])
    popcouplingquantiles    = np.percentile(popcoupling,range(0,101,100//nPopCouplingBins))
    # popcouplingquantiles    = np.percentile(popcoupling[sessions[ises].celldata['gOSI'].to_numpy()>0.4],range(0,101,100//nPopCouplingBins))

    # binedges    = np.percentile(poprate,np.linspace(0,100,nActBins+1))
    # bincenters  = (binedges[1:]+binedges[:-1])/2

    if lam is None:
        y = ori_ses
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
        X = data.T
        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
        lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

    for iqrpopcoupling in range(len(popcouplingquantiles)-1):
        idx_N     = np.where(np.all((
                                # sessions[ises].celldata['gOSI']>0.4,
                                popcoupling>popcouplingquantiles[iqrpopcoupling],
                                popcoupling<=popcouplingquantiles[iqrpopcoupling+1]),axis=0))[0]

    # for iap in range(nActBins):
        for iqrpoprate in range(len(popratequantiles)-1):
            idx_T = np.all((poprate>popratequantiles[iqrpoprate],
                                        poprate<=popratequantiles[iqrpoprate+1]),axis=0)

            X = data[np.ix_(idx_N,idx_T)].T
            y = ori_ses[idx_T]

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array

            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            # error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name='LDA',kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=False)
            error_cv[ises,iqrpoprate,iqrpopcoupling],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                            lam=lam,norm_out=False,subtract_shuffle=False)

#%% Plot error as a function of population rate and for different populations with coupling
# clrs = sns.color_palette('colorblind',n_colors=nPopCouplingBins)
# clrs_popcoupling = sns.color_palette('viridis',nPopCouplingBins)

clrs_popcoupling    = np.array(['#FFADB5','#B60500'])
# orilabels = np.array(['-90','pref','+90','+180'])
subplotlabels = np.array(['Soloists','Choristers'])
linestyles = ['--','-']

fig,ax = plt.subplots(1,1,figsize=(2,3))
# fig,axes = plt.subplots(1,2,figsize=(5,2),sharey=True,sharex=True)
for iPopCouplingBin,PopCouplingBin in enumerate([0,nPopCouplingBins-1]):
    
    for iqrPopRate,PopRateBin in enumerate([0,nActBins-1]):
        # print('Decoding accuracy for PopCoupling bin %d and PopRate bin %d: %.2f' % (PopCouplingBin,PopRateBin,
        #         np.nanmean(error_cv[:,PopRateBin,PopCouplingBin])))

        tempdata = error_cv[:,PopRateBin,iPopCouplingBin] - error_cv[:,0,iPopCouplingBin]
        tempdata = error_cv[:,PopRateBin,iPopCouplingBin]

        ax.plot(iqrPopRate+iPopCouplingBin*2,np.nanmean(tempdata,axis=0),
                color=clrs_popcoupling[iqrPopRate],marker='o',markersize=6,linestyle='')
        ax.errorbar(iqrPopRate+iPopCouplingBin*2,
                    np.nanmean(tempdata,axis=0),
                    yerr=np.nanstd(tempdata,axis=0) / np.sqrt(nSessions),
                    color=clrs_popcoupling[iqrPopRate],linewidth=1,linestyle='none',capsize=3)
    tempdata = error_cv[:,[0,nActBins-1],iPopCouplingBin] - error_cv[:,0,iPopCouplingBin][:,None]
    tempdata = error_cv[:,[0,nActBins-1],iPopCouplingBin]
    # ax.plot(np.array([0,1])+iPopCouplingBin*2,tempdata.T,
            # color='k',linewidth=1,linestyle=linestyles[iPopCouplingBin])
            # (ax, x,y,pos=[0.2,0.1],fontsize=8):
    add_paired_ttest_results(ax,tempdata[:,0],tempdata[:,1],pos=np.array([0.25+iPopCouplingBin*0.5,0.8]),fontsize=8)
    ax.plot(np.array([0,1])+iPopCouplingBin*2,np.nanmean(tempdata,axis=0),
            color='k',linewidth=1,linestyle=linestyles[iPopCouplingBin])

ax.set_xlabel('Population rate (quantile)')
ax.set_ylabel('Decoding accuracy\n (crossval Log. Regression)')
# ax.set_ylim([0,1])
# ax.set_xticks(np.arange(nActBins)+1)
ax.set_xticks([0,1],['Low','High'])
ax.set_xticks(range(4),['Low','High','Low','High'])

# ax.axhline(y=1/len(np.unique(ses.trialdata['Orientation'])), color='grey', linestyle='--', linewidth=1)
# ax.text(0.5,0.15,'Chance',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='grey')
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
ax.legend(subplotlabels,title='Pop. coupling bins',loc='best',frameon=False)
# ax.legend(np.arange,title='Pop. coupling bins',loc='best',frameon=False)
# ax.legend(['mean+-sem\nn=%d sessions' % nSessions],loc='center right',frameon=False)
sns.despine(fig=fig,trim=True,top=True,right=True)

my_savefig(fig,savedir,'Decoding_Low_High_ActBins_PopCoupling_%dsessions' % nSessions)


#%% Plot error as a function of population rate and for different populations with coupling
# # clrs = sns.color_palette('colorblind',n_colors=nPopCouplingBins)
# # clrs_popcoupling = sns.color_palette('viridis',nPopCouplingBins)

# clrs_popcoupling    = np.array(['#FFADB5','#B60500'])
# # orilabels = np.array(['-90','pref','+90','+180'])
# subplotlabels = np.array(['Soloists','Choristers'])
# linestyles = ['--','-']

# fig,ax = plt.subplots(1,1,figsize=(3,3))
# # fig,axes = plt.subplots(1,2,figsize=(5,2),sharey=True,sharex=True)
# for iPopCouplingBin,PopCouplingBin in enumerate([0,nPopCouplingBins-1]):

# # for iqrpopcoupling in range(len(popcouplingquantiles)-1):

#     ax.plot(np.arange(nActBins)+1,np.nanmean(error_cv[:,:,iPopCouplingBin],axis=0),
#             color=clrs_popcoupling[iPopCouplingBin],linewidth=2,linestyle=linestyles[iPopCouplingBin])
#     ax.errorbar(np.arange(nActBins)+1,
#                 np.nanmean(error_cv[:,:,iPopCouplingBin],axis=0),
#                 yerr=np.nanstd(error_cv[:,:,iPopCouplingBin],axis=0) / np.sqrt(nSessions),
#                 color=clrs_popcoupling[iPopCouplingBin],linewidth=1,linestyle='none',capsize=3)
    
#     # ax.plot(np.arange(nActBins)+1,np.nanmean(error_cv[:,:,iPopCouplingBin],axis=0),
#     #         color=clrs_popcoupling[iPopCouplingBin],linewidth=2,linestyle=linestyles[iPopCouplingBin])
#     # ax.errorbar(np.arange(nActBins)+1,
#     #             np.nanmean(error_cv[:,:,iPopCouplingBin],axis=0),
#     #             yerr=np.nanstd(error_cv[:,:,iPopCouplingBin],axis=0),
#     #             color=clrs_popcoupling[iPopCouplingBin],linewidth=1,linestyle='none',capsize=3)

# ax.set_xlabel('Population rate (quantile)')
# ax.set_ylabel('Decoding accuracy\n (crossval Log. Regression)')
# ax.set_ylim([0,1])
# # ax.set_xticks(np.arange(nActBins)+1)
# # ax.set_xticks([0,1],['Low','High'])

# ax.axhline(y=1/len(np.unique(ses.trialdata['Orientation'])), color='grey', linestyle='--', linewidth=1)
# ax.text(0.5,0.15,'Chance',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='grey')
# # ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
# ax.legend(subplotlabels,title='Pop. coupling bins',loc='best',frameon=False)
# # ax.legend(np.arange,title='Pop. coupling bins',loc='best',frameon=False)
# # ax.legend(['mean+-sem\nn=%d sessions' % nSessions],loc='center right',frameon=False)
# sns.despine(fig=fig,trim=True,top=True,right=True)

# # my_savefig(fig,savedir,'Decoding_Ori_LOGR_ActBins_PopCoupling_%dsessions' % nSessions, formats = ['png'])
# # my_savefig(fig,savedir,'Decoding_Low_High_ActBins_PopCoupling_%dsessions' % nSessions, formats = ['png'])

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

