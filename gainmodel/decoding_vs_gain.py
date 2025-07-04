
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings,plot_PCA_gratings_3D_traces
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.psth import compute_tensor
from scipy.stats import zscore
from utils.plot_lib import shaded_error
from utils.regress_lib import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_areas=['V1'])

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=True)

#%%  Load data properly:                      
## Parameters for temporal binning
t_pre       = -0.75    #pre s
t_post      = 2     #post s

for ises in range(nSessions):
    # Construct time tensor: 3D 'matrix' of K trials by N neurons by T time bins
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                    t_pre, t_post,method='nearby')


#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% Make the 3D figure:
ises = 0
fig = plot_PCA_gratings_3D(sessions[ises],thr_tuning=0.05,plotgainaxis=True)
axes = fig.get_axes()
axes[0].view_init(elev=-30, azim=25, roll=40)
axes[0].set_xlim([-2,35])
axes[0].set_ylim([-2,35])
for ax in axes:
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Get rid of colored axes planes, remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

plt.tight_layout()
# fig.savefig(os.path.join(savedir,'Example_Cone_3D_V1_PM_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% 

#%% Show histogram of gain model weights:
ises = 0
data                = sessions[ises].respmat
poprate             = np.nanmean(data,axis=0)
gain_trials         = poprate - np.nanmean(data,axis=None)
gain_weights        = np.array([np.corrcoef(poprate,data[n,:])[0,1] for n in range(data.shape[0])])

#%% 
fig,axes = plt.subplots(1,2,figsize=(6,3))
axes[0].hist(gain_trials,bins=25,color='grey')
axes[0].set_title('Trial gain')
axes[0].set_xlabel('Pop. rate - mean pop. rate')
axes[1].hist(gain_weights,bins=25,color='grey')
axes[1].set_title('Neuron gain ')
axes[1].set_xlabel('Correlation neuron rate to pop. rate')
plt.tight_layout()

# #%% Z-score the calciumdata: 
# for i in range(nSessions):
#     sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

# #%%  Load data properly:                      
# ## Parameters for temporal binning
# t_pre       = -2    #pre s
# t_post      = 3     #post s

# for ises in range(nSessions):
#     t_resp_start = 0
#     t_resp_stop = 1

#     # ## Construct trial response matrix:  N neurons by K trials
#     # sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
#     #                                 t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,method='mean',subtr_baseline=False, label = "response matrix")

#     # Construct time tensor: 3D 'matrix' of K trials by N neurons by T time bins
#     [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
#                                     t_pre, t_post,method='nearby')

#%% 
import matplotlib.animation as animation

#%% 
ises = 5
ises = 9
# ises = 0
fig = plot_PCA_gratings_3D_traces(sessions[ises],t_axis,thr_tuning=0.00,plotgainaxis=True,export_animation=False)
axes = fig.get_axes()
# for ax in axes:
    # ax.view_init(elev=-45, azim=25, roll=10)
# axes[0].view_init(azim=25)
fig.savefig(os.path.join(savedir,'Example_Cone_3D_V1_traces_%s' % sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

# print("Making animation")
# rot_animation = animation.FuncAnimation(
#     fig, rotate, frames=np.arange(0, 364, 4), interval=100)
# rot_animation.save(os.path.join(savedir, 'rotation_%s.gif' % sessions[ises].sessiondata['session_id'][0]), dpi=80, writer='imagemagick')

#%% 
ises = 0

fig = plot_PCA_gratings_3D_traces(sessions[ises],t_axis,thr_tuning=0.00,plotgainaxis=True,export_animation=False)



#%% 

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_areas=['V1'])

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='deconv',keepraw=False)
    sessions[ises].load_tensor(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

t_axis = sessions[ises].t_axis
#%% 
# Does the standard deviation of the population response on the decoding axis scale linearly with the mean?

# NOT FINISHED!!! 
# nActBins = 10
# kfold = 5
# lam = None
# lam = 1
# model_name = 'LOGR'
# model_name = 'LDA'

# oris = np.sort(pd.Series.unique(sessions[0].trialdata['Orientation']))
# noris = len(oris)

# decax_sd    = np.full((nSessions,nActBins,noris),np.nan)
# decax_mean  = np.full((nSessions,nActBins,noris),np.nan)

# for ises,ses in tqdm(enumerate(sessions),desc='Decoding stimulus ori across sessions',total=nSessions):
#     idx_N           = np.ones(len(ses.celldata)).astype(bool)

#     data            = zscore(ses.respmat, axis=1)
#     # data                = zscore(ses.respmat[idx, :], axis=1)
#     poprate         = np.nanmean(data,axis=0)

#     binedges        = np.percentile(poprate,np.linspace(0,100,nActBins+1))
#     bincenters      = (binedges[1:]+binedges[:-1])/2

#     if lam is None:
#         y = ses.trialdata['Orientation']
#         label_encoder = LabelEncoder()
#         y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
#         X = data.T
#         X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
#         lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

#     for iap in range(nActBins):
#         for iori,ori in enumerate(oris):
#             idx_T = (ses.trialdata['Orientation'] == ori)
#             idx_T_ortho = (ses.trialdata['Orientation'] != ori)

#             idx_T = (poprate >= binedges[iap]) & (poprate <= binedges[iap+1])
#             X = data[np.ix_(idx_N,idx_T)].T
#             y = ses.trialdata['Orientation'][idx_T]

#             label_encoder   = LabelEncoder()
#             y               = label_encoder.fit_transform(y.ravel())  # Convert to 1D array

#             X,y,_           = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

#             # error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name='LDA',kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=False)
#             error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=False)


#%% Decoding performance as a function of population rate: 
nActBins = 10
kfold = 5
lam = None
# lam = 1
model_name = 'LOGR'
scoring_type = 'accuracy_score'
# scoring_type = 'balanced_accuracy_score'

# model_name = 'Ridge'
# model_name = 'SVR'
# scoring_type = 'circular_abs_error'

error_cv = np.full((nSessions,nActBins),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Decoding stimulus ori across sessions',total=nSessions):
    idx_N               = np.ones(len(ses.celldata)).astype(bool)

    data                = zscore(ses.respmat, axis=1)
    # data                = zscore(ses.respmat[idx, :], axis=1)
    poprate             = np.nanmean(data,axis=0)

    # tensor_zsc = copy.deepcopy(ses.tensor)
    # tensor_zsc -= np.mean(tensor_zsc, axis=(1,2), keepdims=True)
    # tensor_zsc /= np.std(tensor_zsc, axis=(1,2), keepdims=True)

    # idx_B       = t_axis<=0 #get baseline mean activity 
    # poprate     = np.nanmean(tensor_zsc[:,:,idx_B], axis=(0,2))

    binedges    = np.percentile(poprate,np.linspace(0,100,nActBins+1))
    bincenters  = (binedges[1:]+binedges[:-1])/2
    # ori_ses     = np.mod(ses.trialdata['Orientation'],180)
    ori_ses     = ses.trialdata['Orientation']

    if lam is None:
        y = ori_ses
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
        X = data.T
        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
        lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

    for iap in range(nActBins):
        idx_T = (poprate >= binedges[iap]) & (poprate <= binedges[iap+1])
        X = data[np.ix_(idx_N,idx_T)].T
        y = ori_ses[idx_T]

        # label_encoder = LabelEncoder()
        # y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array

        X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

        # error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name='LDA',kfold=kfold,lam=lam,norm_out=False,subtract_shuffle=False)
        error_cv[ises,iap],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                        lam=lam,norm_out=False,subtract_shuffle=False)

#%% Correlation between baseline activity level and response:

fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
corrdata = np.full((nSessions),np.nan)

for ises,ses in enumerate(sessions):
# for ises,ses in enumerate([sessions[0]]):

    # tensor_zsc = copy.deepcopy(ses.tensor)
    # tensor_zsc -= np.mean(tensor_zsc, axis=(1,2), keepdims=True)
    # tensor_zsc /= np.std(tensor_zsc, axis=(1,2), keepdims=True)

    idx_B           = t_axis<=0 #get baseline mean activity 
    # idx_B           = (t_axis>-0.5) & (t_axis<0) #get response mean activity 
    # baselinerate     = np.nanmean(tensor_zsc[:,:,idx_B], axis=(0,2))
    baselinerate     = np.nanmean(ses.tensor[:,:,idx_B], axis=(0,2))

    idx_R           = (t_axis>0) & (t_axis<0.75) #get response mean activity 
    # responserate     = np.nanmean(tensor_zsc[:,:,idx_R], axis=(0,2))
    responserate     = np.nanmean(ses.tensor[:,:,idx_R], axis=(0,2))
    
    corrdata[ises] = np.corrcoef(baselinerate,responserate)[0,1]
    # ax.scatter(baselinerate, responserate, s=4, alpha=0.8)


#%% Plot performance as a function of population rate: 
fig,ax = plt.subplots(1,1,figsize=(3,3))
# ax.plot(np.arange(nActBins),error_cv.mean(axis=0))
shaded_error(np.arange(nActBins)+1,error_cv,error='sem',ax=ax)
ax.set_xlabel('Population rate (quantile)')
# ax.set_ylabel('Decoding accuracy\n (crossval LDA)')
ax.set_ylabel('Decoding accuracy\n (crossval Log. Regression)')
ax.set_ylim([0,1])
ax.set_xticks(np.arange(nActBins)+1)
ax.axhline(y=1/len(np.unique(np.mod(ses.trialdata['Orientation'],180))), color='grey', linestyle='--', linewidth=1)
ax.text(0.5,0.15,'Chance',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='grey')
ax.legend(['mean+-sem\nn=%d sessions' % nSessions],loc='center right',frameon=False)
sns.despine(fig=fig,trim=True,top=True,right=True)

my_savefig(fig,savedir,'Decoding_Orientation_LOGR_ActBins_baseline_%d' % nSessions, formats = ['png'])
# fig.savefig(os.path.join(savedir,'Decoding_Orientation_LDA_ActBins_%d' % nSessions + '.png'), format = 'png')


#%% Plot error as a function of population rate: 
fig,ax = plt.subplots(1,1,figsize=(3,3))
# ax.plot(np.arange(nActBins),error_cv.mean(axis=0))
shaded_error(np.arange(nActBins)+1,error_cv,error='sem',ax=ax)
ax.set_xlabel('Population rate (quantile)')
# ax.set_ylabel('Decoding accuracy\n (crossval LDA)')
ax.set_ylabel('Decoding error\n (crossval)')
# ax.set_ylim([0,25])
ax.set_xticks(np.arange(nActBins)+1)
ax.axhline(y=180/2, color='grey', linestyle='--', linewidth=1)
ax.set_ylim([15,30])
ax.legend(['mean+-sem\nn=%d sessions' % nSessions],loc='best',frameon=False)
sns.despine(fig=fig,trim=True,top=True,right=True)

my_savefig(fig,savedir,'Ori_Decoding_Error_%s_ActBins_%dsessions' % (model_name,nSessions), formats = ['png'])
# fig.savefig(os.path.join(savedir,'Decoding_Orientation_LDA_ActBins_%d' % nSessions + '.png'), format = 'png')

#%% Decoding performance as a function of population rate for differently coupled neurons
nActBins = 10
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

    poprate                 = np.nanmean(data,axis=0)
    popratequantiles        = np.percentile(poprate,range(0,101,100//nActBins))

    N                       = np.shape(data)[0]
    popcoupling             = [np.corrcoef(data[i,:],poprate)[0,1] for i in range(N)]
    popcouplingquantiles    = np.percentile(popcoupling,range(0,101,100//nPopCouplingBins))

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
        idx_N     = np.where(np.all((popcoupling>popcouplingquantiles[iqrpopcoupling],
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
clrs_popcoupling = sns.color_palette('viridis',nPopCouplingBins)

fig,ax = plt.subplots(1,1,figsize=(3,3))
for iqrpopcoupling in range(len(popcouplingquantiles)-1):
    # shaded_error(np.arange(nActBins)+1,error_cv[:,:,iqrpopcoupling],error='sem',ax=ax,color=clrs_popcoupling[iqrpopcoupling])
    ax.plot(np.arange(nActBins)+1,np.nanmean(error_cv[:,:,iqrpopcoupling],axis=0),
            color=clrs_popcoupling[iqrpopcoupling],linewidth=2)
ax.set_xlabel('Population rate (quantile)')
ax.set_ylabel('Decoding accuracy\n (crossval Log. Regression)')
ax.set_ylim([0,1])
ax.set_xticks(np.arange(nActBins)+1)
ax.axhline(y=1/len(np.unique(ses.trialdata['Orientation'])), color='grey', linestyle='--', linewidth=1)
ax.text(0.5,0.15,'Chance',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='grey')
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
ax.legend(['Weak/Negative','','','','','Intermediate','','','','Strong'],
          reverse=True,fontsize=7,frameon=False,title='pop. coupling bins',bbox_to_anchor=(0.9,1), loc='upper left')

# ax.legend(np.arange,title='Pop. coupling bins',loc='best',frameon=False)
# ax.legend(['mean+-sem\nn=%d sessions' % nSessions],loc='center right',frameon=False)
sns.despine(fig=fig,trim=True,top=True,right=True)

my_savefig(fig,savedir,'Decoding_Ori_LOGR_ActBins_PopCoupling_%dsessions' % nSessions, formats = ['png'])


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

#######    #    #     # #######    #######    #     #####  ####### ####### ######  
#         # #   ##    # #     #    #         # #   #     #    #    #     # #     # 
#        #   #  # #   # #     #    #        #   #  #          #    #     # #     # 
#####   #     # #  #  # #     #    #####   #     # #          #    #     # ######  
#       ####### #   # # #     #    #       ####### #          #    #     # #   #   
#       #     # #    ## #     #    #       #     # #     #    #    #     # #    #  
#       #     # #     # #######    #       #     #  #####     #    ####### #     # 

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_areas=['V1'])

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='deconv',keepraw=False)
    sessions[ises].load_tensor(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='dF',keepraw=False)

t_axis = sessions[ises].t_axis

#%% 
idx_resp = (t_axis>=0.5) & (t_axis<=1.5)
for ses in sessions:
    ses.respmat = np.nanmean(ses.tensor[:,:,idx_resp], axis=2)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% Fano Factor over time: 
celldata        = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
N               = len(celldata)

ntimebins       = len(t_axis)
fano_data       = np.full((nActBins,ntimebins,N),np.nan)

for ises,ses in enumerate(sessions):
    idx_ses         = np.where(celldata['session_id']==ses.sessiondata['session_id'][0])[0]
    data            = zscore(ses.respmat, axis=1)
    # poprate         = np.nanmean(data,axis=0)
    poprate         = np.nanmean(ses.tensor,axis=(0,2))

    binedges        = np.percentile(poprate,np.linspace(0,100,nActBins+1))
    bincenters      = (binedges[1:]+binedges[:-1])/2

    N_ses           = len(ses.celldata)
    uoris           = np.unique(ses.trialdata['Orientation'])

    #Per orientation:
    # fano_temp = np.full((nActBins,ntimebins,N_ses,len(uoris)),np.nan)
    # for iap in range(nActBins):
    #     for iori,ori in enumerate(uoris):
    #         idx_T       = np.all((poprate >= binedges[iap],
    #                               poprate <= binedges[iap+1],
    #                               ses.trialdata['Orientation'] == ori), axis=0)
    #         X       = ses.tensor[:,idx_T,:]
    #         fano_temp[iap,:,:,iori] = np.transpose(np.nanmean(X,axis=1) / np.nanstd(X,axis=1))
    # fano_data[:,:,idx_ses] = np.nanmean(fano_temp, axis=3)

    #All trials:
    fano_temp = np.full((nActBins,ntimebins,N_ses),np.nan)
    for iap in range(nActBins):
        idx_T       = np.all((poprate >= binedges[iap],
                                poprate <= binedges[iap+1]), axis=0)
        X           = ses.tensor[:,idx_T,:]
        fano_data[iap,:,idx_ses] = np.nanmean(X,axis=1) / np.nanstd(X,axis=1)

#%% Fano Factor over time: 
clrs_actbins = sns.color_palette('magma',nActBins+2)
min_tuned = 0.0
idx_N = np.where(celldata['tuning_var']>min_tuned)[0]
# idx_N = np.where(celldata['tuning_var']>0.025)[0]
# idx_N = np.where(celldata['roi_name']=='V1')[0]
# idx_N = celldata['tuning_var']>0.1
# idx_N = celldata['tuning_var']>0.1

fano_mean = np.nanmean(fano_data[:,:,idx_N],axis=2)
fig,ax = plt.subplots(1,1,figsize=(4,3))
for iap in range(nActBins):
    
    ax.plot(t_axis,fano_mean[iap,:],alpha=1,color=clrs_actbins[iap],linewidth=1.5)
# ax.plot(t_axis,fano_data[:,:,idx_N].mean(axis=2).T)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Fano Factor')
ax.axhline(1,ls='--',color='k',alpha=0.5)
ax.set_ylim([0.9,1.2])
# ax.set_ylim([0.8,1.5])
# ax.set_ylim([1.15,1.45])
ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
          reverse=True,fontsize=7,frameon=False,title='pop. rate bins',bbox_to_anchor=(0.9,1), loc='upper left')
sns.despine(fig=fig,trim=True,offset=3,top=True,right=True)
plt.tight_layout()
my_savefig(fig,savedir,'FanoFactor_ActBins_baseline_%dsessions_%1.3ftuning' % (nSessions,min_tuned), formats = ['png'])
























#%%

ises = 0
ses = sessions[ises]


# def plot_PCA_gratings_3D_traces(ses, t_axis,
size='runspeed'
thr_tuning=0
# n_single_trials=10):

########### PCA on trial-averaged responses ############
######### plot result as scatter by orientation ########

ori = ses.trialdata['Orientation']
oris = np.sort(pd.Series.unique(ses.trialdata['Orientation']))

ori_ind = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

pal = sns.color_palette('husl', len(oris))
pal = np.tile(sns.color_palette('husl', int(len(oris)/2)), (2, 1))
if size == 'runspeed':
    sizes = (ses.respmat_runspeed - np.percentile(ses.respmat_runspeed, 5)) / \
        (np.percentile(ses.respmat_runspeed, 95) -
            np.percentile(ses.respmat_runspeed, 5))
elif size == 'videome':
    sizes = (ses.respmat_videome - np.percentile(ses.respmat_videome, 5)) / \
        (np.percentile(ses.respmat_videome, 95) -
            np.percentile(ses.respmat_videome, 5))


# zscore for each neuron across trial responses
# respmat_zsc = ses.respmat[idx, :]

# tensor Z:
# tensor_zsc = ses.tensor[idx, :, :]
# tensor_zsc = zscore(ses.tensor[idx, :, :], axis=(1,2))
tensor_zsc = copy.deepcopy(ses.tensor)
# tensor_zsc = zscore(ses.tensor, axis=(1,2))
tensor_zsc -= np.mean(tensor_zsc, axis=(1,2), keepdims=True)
tensor_zsc /= np.std(tensor_zsc, axis=(1,2), keepdims=True)

idx_B = (t_axis>=0) & (t_axis<=1)
respmat_zsc = np.nanmean(tensor_zsc[:,:,idx_B], axis=2)
data                = respmat_zsc

lam = 0.0
## LDA on grating stimuli:
model = LDA(n_components=2,solver='eigen', shrinkage=np.clip(lam,0,1))

idx_N = np.ones(len(ses.celldata)).astype(bool)
idx_T = np.ones(len(ses.trialdata)).astype(bool)
# idx_T = (poprate >= binedges[iap]) & (poprate <= binedges[iap+1])
X = data[np.ix_(idx_N,idx_T)].T
y = np.mod(ses.trialdata['Orientation'][idx_T],180) #ses.trialdata['Orientation'][idx_T]
# y = ses.trialdata['Orientation'][idx_T]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array

X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

# Train a classification model on the training data with regularization
LDAproj = model.fit_transform(X, y).T

# weights[ifold,:] = model.coef_ #deprecated, estimate weights from all data, not cv
# construct PCA object with specified number of components
# pca = PCA(n_components=3)
# fit pca to response matrix (n_samples by n_features)
# Xp = pca.fit_transform(respmat_zsc.T).T
# dimensionality is now reduced from N by K to ncomp by K

data                = respmat_zsc
poprate             = np.nanmean(respmat_zsc,axis=0)

idx_B = t_axis<=0
poprate = np.nanmean(tensor_zsc[:,:,idx_B], axis=(0,2))


# poprate             = np.nanmean(ses.respmat,axis=0)
gain_weights        = np.array([np.corrcoef(poprate,data[n,:])[0,1] for n in range(data.shape[0])])
# gain_trials         = poprate - np.nanmean(data,axis=None)
    # g = np.outer(np.percentile(gain_trials,[0,100]),gain_weights)
    # g = np.outer([-2,10],gain_weights)
    # g = np.outer(np.percentile(gain_trials,[0,100])*np.percentile(poprate,[0,100]),gain_weights)
    # Xg = pca.transform(g).T
# gain_weights = np.ones(data.shape[0])
GAINproj = np.dot(respmat_zsc.T,gain_weights)

# data_re     = np.reshape(tensor_zsc,(tensor_zsc.shape[0],-1))
# Xt          = pca.transform(data_re.T).T
# Xt          = np.reshape(Xt,(Xt.shape[0],tensor_zsc.shape[1],tensor_zsc.shape[2]))

# data                = respmat_zsc
# poprate             = np.nanmean(data,axis=0)

#%% 
nActBins = 1
binedges = np.percentile(poprate,np.linspace(0,100,nActBins+1))

fig = plt.figure(figsize=(nActBins*5,5))
for iap  in range(nActBins):
        # fig,axes = plt.figure(1, len(areas), figsize=[len(areas)*3, 3])
    ax = fig.add_subplot(1, nActBins, iap+1, projection='3d')

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        idx_T = np.all((np.array(ori) == t_type,
                        poprate >= binedges[iap],
                        poprate <= binedges[iap+1]
                        ),axis=0)
        # get all data points for this ori along first PC or projection pairs
        # x = LDAproj[0, ori_ind[t]]
        # y = LDAproj[1, ori_ind[t]]  # and the second
        # z = GAINproj[ori_ind[t]]  # and the third,the population gain axis

        x = LDAproj[0, idx_T]
        y = LDAproj[1, idx_T]  # and the second
        z = GAINproj[idx_T]  # and the third,the population gain axis

        # each trial is one dot
        ax.scatter(x, y, z, color=pal[t], s=sizes[idx_T]*6, alpha=0.8)
        # ax.scatter(x, y, z, color='k', s=2, alpha=0.5)

    ax.set_xlabel('LDA 1')  # give labels to axes
    ax.set_ylabel('LDA 2')
    ax.set_zlabel('Gain axis')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.grid(True)
    ax.set_facecolor('white')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_title('Pop Act Bin %d' % iap)

    # Get rid of colored axes planes, remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    limpercs = [1, 99]
    # limpercs = [0, 98]
    # ax.set_xlim(np.percentile(Xp[0, :], limpercs))
    # ax.set_ylim(np.percentile(Xp[1, :], limpercs))
    # ax.set_zlim(np.percentile(Xp[2, :], limpercs))

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

#%% 
nActBins = 10
binedges = np.percentile(poprate,np.linspace(0,100,nActBins+1))

fig,axes = plt.subplots(1,nActBins,figsize=(nActBins*5,5),sharex=True,sharey=True)

for iap  in range(nActBins):
        # fig,axes = plt.figure(1, len(areas), figsize=[len(areas)*3, 3])
    ax = axes[iap]

    # plot orientation separately with diff colors
    for t, t_type in enumerate(oris):
        idx_T = np.all((np.array(ori) == t_type,
                        poprate >= binedges[iap],
                        poprate <= binedges[iap+1]
                        ),axis=0)
        # get all data points for this ori along first PC or projection pairs
        # x = LDAproj[0, ori_ind[t]]
        # y = LDAproj[1, ori_ind[t]]  # and the second
        # z = GAINproj[ori_ind[t]]  # and the third,the population gain axis

        x = LDAproj[0, idx_T]
        y = LDAproj[1, idx_T]  # and the second
        # z = GAINproj[idx_T]  # and the third,the population gain axis

        # each trial is one dot
        ax.scatter(x, y, color=pal[t], s=4, alpha=0.8)
        # ax.scatter(x, y, z, color='k', s=2, alpha=0.5)
        # ax.scatter(x, y, z,marker='o')     #each trial is one dot

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Pop Act Bin %d' % iap)
