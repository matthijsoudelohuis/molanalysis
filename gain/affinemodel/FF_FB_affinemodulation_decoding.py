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
from scipy.stats import linregress
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from statsmodels.stats.anova import AnovaRM

os.chdir('e:\\Python\\molanalysis')

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance,value_matching
from utils.plot_lib import * #get all the fixed color schemes
from preprocessing.preprocesslib import assign_layer,assign_layer2
from utils.rf_lib import filter_nearlabeled
from utils.RRRlib import regress_out_behavior_modulation
from utils.regress_lib import *

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain\\Affine_FF_vs_FB'

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

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
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

#%%
# sessions = compute_pairwise_anatomical_distance(sessions)
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

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)


# #%% 
# arealayers = np.array(['V1L2/3','PML2/3','PML5'])
# narealayers = len(arealayers)
# maxnoiselevel = 20
# celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
# clrs = ['black','red']
# fig,axes = plt.subplots(1,narealayers,figsize=(narealayers*3,3),sharey=True,sharex=True)

# for ial,al in enumerate(arealayers):
#     ax = axes[ial]
#     idx_N              = np.all((
#                                 celldata['noise_level']<maxnoiselevel,
#                                 celldata['arealayer']==al,
#                                 celldata['nearby'],
#                                     ),axis=0)

#     sns.histplot(data=celldata[idx_N],x='pop_coupling',hue='arealayerlabel',color='green',element="step",stat="density", 
#                 common_norm=False,fill=False,bins=np.linspace(-0.3,1,1000),cumulative=True,ax=ax,palette=clrs,legend=False)
#     ax.set_title(al)
# sns.despine(fig=fig, top=True, right=True,offset=0)


#%% For every session remove behavior related variability:
rank_behavout = 3
# maxnoiselevel = 20

#%%
for ises in range(nSessions):

    # Convert response_matrix and orientations_vector to numpy arrays
    # response_matrix         = np.array(response_matrix)
    conditions_vector       = np.array(sessions[ises].trialdata['stimCond'])
    conditions              = np.sort(np.unique(conditions_vector))
    C                       = len(conditions)

    resp_mean       = sessions[ises].respmat.copy()
    # resp_res        = sessions[ises].respmat.copy()

    for iC,cond in enumerate(conditions):
        tempmean                            = np.nanmean(sessions[ises].respmat[:,conditions_vector==cond],axis=1)
        # resp_mean[:,iC]                     = tempmean
        resp_mean[:,conditions_vector==cond] = tempmean[:,np.newaxis]

    Y               = sessions[ises].respmat.copy()
    Y               = Y - resp_mean
    [Y_orig,Y_hat,Y_out,rank,ev] = regress_out_behavior_modulation(sessions[ises],X=None,Y=Y.T,
                                nvideoPCs = 30,rank=rank_behavout,lam=0,perCond=True,kfold = 5)
    # print(ev)
    sessions[ises].respmat_behavout = Y_out.T + resp_mean

# plt.imshow(sessions[ises].respmat,cmap='RdBu_r',vmin=0,vmax=100)
# plt.imshow(sessions[ises].respmat_behavout,cmap='RdBu_r',vmin=0,vmax=100)

#%% Show the distribution of mean activity of the labeled populatoin and the selection of the
# extreme percentiles: 
ises = 0
arealabelpairs  = ['V1lab','PMlab']
clrs_arealabelpairs = ['green','purple']
legendlabels        = ['FF','FB']

respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
maxrunspeed = 0.5
maxvideome = 0.2
perc = 25

#%% 

######  #######  #####  ####### ######  ### #     #  #####  
#     # #       #     # #     # #     #  #  ##    # #     # 
#     # #       #       #     # #     #  #  # #   # #       
#     # #####   #       #     # #     #  #  #  #  # #  #### 
#     # #       #       #     # #     #  #  #   # # #     # 
#     # #       #     # #     # #     #  #  #    ## #     # 
######  #######  #####  ####### ######  ### #     #  #####  


#%% Decoding orientation when the FF or FB activity is high or low:

arealabelpairs  = [
                    'V1lab-PMunlL2/3',
                    # 'V1lab-V1unl-PMlabL2/3',
                    # 'V1lab-V1unl-PMunlL5',
                    # 'V1lab-V1unl-PMlabL5',
                    'PMlab-V1unlL2/3',
                    # 'PMlab-PMunl-V1labL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

perc                = 25
minnneurons         = 10
maxnoiselevel       = 20

nmodelfits          = 25
nsampleneurons      = 50
kfold               = 5
# lam               = None
lam                 = 0.05
model_name          = 'SVM'
scoring_type        = 'accuracy_score'

stilltrialsonly     = True
error_cv            = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Decoding across sessions'):
# for ises in tqdm([9,10],total=nSessions,desc='Decoding across sessions'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    # respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # data            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # data            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    data            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # data            = sessions[ises].respmat

    ori_ses         = sessions[ises].trialdata['Orientation']

    if stilltrialsonly:
        idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                    sessions[ises].respmat_runspeed < maxrunspeed)
    else:
        idx_T_still = np.ones(K,dtype=bool)
   
    # idx_nearby          = filter_nearlabeled(sessions[ises],radius=50)

    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        # idx_N2              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
        #                               sessions[ises].celldata['noise_level']<maxnoiselevel,
        #                               ),axis=0))[0]

        idx_ses             = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][0])
        idx_N2              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                    #   corrsig_cells[ialp,idx_ses]==1,
                                    #   corrsig_cells[ialp,idx_ses]==-1,
                                    #   corrsig_cells[ialp,idx_ses]==0,
                                    #   np.any(corrsig_cells[:,idx_ses]!=1,axis=0),
                                    #   np.any(corrsig_cells[:,idx_ses]==1,axis=0),
                                    #   np.any(data_gainregress[idx_ses,:,2] > 0.5,axis=1),
                                    #   sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        if len(idx_N1) < minnneurons or len(idx_N2) < nsampleneurons:
            # print(f'Not enough neurons in {alp} for session {ises}')
            # print(f'N1: {len(idx_N1)}, N2: {len(idx_N2)}')
            continue
        
        if lam is None:
            y = ori_ses
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
            X = data.T
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

        # meanpopact          = np.nanmean(zscore(data[idx_N1,:],axis=1),axis=0)
        meanpopact          = np.nanmean(data[idx_N1,:],axis=0)
        
        # meanpopact2          = np.nanmean(data[idx_N2,:],axis=0)
        # plt.scatter(meanpopact,meanpopact2)
        # plt.plot([0,1],[0,1],transform=ax.transAxes)

        # compute index for trials with low and high activity in the other labeled pop
        idx_K1            = np.array([]).astype(int)
        idx_K2            = np.array([]).astype(int)

        oris                = np.unique(ori_ses)

        for i,ori in enumerate(oris):
            idx_T = np.logical_and(ori_ses == ori,idx_T_still)
            idx_K1 = np.concatenate([idx_K1,
                                     np.where(np.all((ori_ses == ori,
                                                      idx_T_still,
                                                      meanpopact < np.nanpercentile(meanpopact[idx_T],perc)),axis=0))[0]])
            idx_K2 = np.concatenate([idx_K2,
                                    np.where(np.all((ori_ses == ori,
                                                    idx_T_still,
                                                    meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)),axis=0))[0]])
            
        for imf in range(nmodelfits):
            idx_N_sub = np.random.choice(idx_N2,nsampleneurons,replace=False)
            X = data[np.ix_(idx_N_sub,idx_K1)].T
            y = ori_ses[idx_K1]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # C
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            error_cv[ialp,0,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)
            
            X = data[np.ix_(idx_N_sub,idx_K2)].T
            y = ori_ses[idx_K2]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # C
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            error_cv[ialp,1,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)
         

#%%
axtitles = np.array(['Feedforward','Feedback']) 
fig,axes = plt.subplots(1,2,figsize=(5,3),sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    data = np.nanmean(error_cv[ialp,:,:,:],axis=2)
    sns.stripplot(data.T,palette=['grey','red'],alpha=0.5,ax=ax,jitter=0.1)
    sns.barplot(data.T,palette=['grey','red'],alpha=0.5,ax=ax)
    sns.lineplot(data,palette=sns.color_palette(['grey'],nSessions),alpha=0.5,ax=ax,legend=False,
                linewidth=2,linestyle='-')

    ax.axhline(1/16,linestyle='--',color='k',alpha=0.5)
    ax.text(0.5,1/16+0.05,'Chance',fontsize=10,ha='center',va='center')
    ax.set_ylim([0,1])
    ax.set_title(axtitles[ialp],fontsize=12,)
    ax.set_xticks([0,1],labels=['Low','High'])
    if ialp == 0:
        ax.set_ylabel('Decoding Performance')
    else:
        ax.set_ylabel('')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

#Statistics:
testdata    = np.nanmean(error_cv,axis=3) #average over modelfits
idx_ses  = ~np.any(np.isnan(testdata),axis=(0,1))
df = pd.DataFrame({'perf': testdata.flatten(),
                   'act': np.repeat(np.tile(np.arange(2),2),nSessions),
                   'area': np.repeat(np.arange(2),2*nSessions),
                   'session_id': np.tile(np.arange(nSessions),2*2)
                   })
df.dropna(inplace=True)
df = df[df['session_id'].isin(np.where(idx_ses)[0])]

# Conduct the repeated measures ANOVA
aov = AnovaRM(data=df,
              depvar='perf',
              subject='session_id',
              within=['act', 'area'])
res = aov.fit()
print(res.summary())

my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_StillOnly_%dsessions' % nSessions, formats = ['png'])
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_SigP_%dsessions' % nSessions, formats = ['png'])
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_SigN_%dsessions' % nSessions, formats = ['png'])

#%% 

#Comments: the interaction effect between FF and FB on decoding 
# is there if in the first part the deconv/F0 is chosen for meanpopact in population 1
# but then in the second half for the decoding you work with respmat original
# This is confounding with the behavioral data again. So need to fix this... 


