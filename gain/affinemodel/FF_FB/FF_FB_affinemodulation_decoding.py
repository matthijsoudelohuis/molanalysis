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

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain\\Affine_FF_vs_FB\\Decoding'

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
sessions = compute_tuning_wrapper(sessions)

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

#%% For every session remove behavior related variability:
# rank_behavout = 3
# # maxnoiselevel = 20

# #%%
# for ises in range(nSessions):

#     # Convert response_matrix and orientations_vector to numpy arrays
#     # response_matrix         = np.array(response_matrix)
#     conditions_vector       = np.array(sessions[ises].trialdata['stimCond'])
#     conditions              = np.sort(np.unique(conditions_vector))
#     C                       = len(conditions)

#     resp_mean       = sessions[ises].respmat.copy()
#     # resp_res        = sessions[ises].respmat.copy()

#     for iC,cond in enumerate(conditions):
#         tempmean                            = np.nanmean(sessions[ises].respmat[:,conditions_vector==cond],axis=1)
#         # resp_mean[:,iC]                     = tempmean
#         resp_mean[:,conditions_vector==cond] = tempmean[:,np.newaxis]

#     Y               = sessions[ises].respmat.copy()
#     Y               = Y - resp_mean
#     [Y_orig,Y_hat,Y_out,rank,ev] = regress_out_behavior_modulation(sessions[ises],X=None,Y=Y.T,
#                                 nvideoPCs = 30,rank=rank_behavout,lam=0,perCond=True,kfold = 5)
#     # print(ev)
#     sessions[ises].respmat_behavout = Y_out.T + resp_mean

# plt.imshow(sessions[ises].respmat,cmap='RdBu_r',vmin=0,vmax=100)
# plt.imshow(sessions[ises].respmat_behavout,cmap='RdBu_r',vmin=0,vmax=100)

#%% #%% Show tuning curve when activityin the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
perc                    = 25

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5
alphathr                = 0.001 #threshold for correlation with cross area rate

minnneurons             = 10
maxnoiselevel           = 20
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
        idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
        idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue
        
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)

        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp[idx_N3]

# Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
N = len(celldata)
data_gainregress = np.full((N,narealabelpairs,3),np.nan)
for iN in tqdm(range(N),total=N,desc='Fitting gain for each neuron'):
    for ialp,alp in enumerate(arealabelpairs):
        xdata = mean_resp_split[ialp,:,0,iN]
        ydata = mean_resp_split[ialp,:,1,iN]
        data_gainregress[iN,ialp,:] = linregress(xdata,ydata)[:3]


#%% Show tuning curve when activity in the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
perc                    = 25

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5
alphathr                = 0.001 #threshold for correlation with cross area rate

minnneurons             = 10
maxnoiselevel           = 20
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells),np.nan)

#Regression output:
nboots                  = 0
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

#Correlation output:
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
        idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
        idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue
        
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        #Ratio:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        #Difference:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        # meanpopact          = np.nanmean(respdata[idx_N2,:],axis=0) - np.nanmean(respdata[idx_N1,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        errorresp           = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            errorresp[:,i,0]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K1)))
            errorresp[:,i,1]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K2)))

        meanresp = meanresp - np.nanmin(meanresp[:,:,:],axis=(1,2),keepdims=True)

        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp[idx_N3]
        error_resp_split[ialp,:,:,idx_ses] = errorresp[idx_N3]

        regressdata          = np.full((N,3),np.nan)
        regress_sig          = np.full((N,2),0)
        for n in range(N):
            xdata = meanresp[n,:,0]
            ydata = meanresp[n,:,1]
            regressdata[n,:] = linregress(xdata,ydata)[:3]
        params_regress[idx_ses,ialp,:] = regressdata[idx_N3]

        if nboots:
            bootregressdata  = np.full((N,nboots,3),np.nan)
            bootregress_sig  = np.full((N,2),0)
            for iboot in range(nboots):
                meanrespboot            = np.empty([N,len(oris),2])
                for i,ori in enumerate(oris):
                    idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
                    idx_K1              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
                    idx_K2              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
                    meanrespboot[:,i,0]     = np.nanmean(respdata[:,idx_K1],axis=1)
                    meanrespboot[:,i,1]     = np.nanmean(respdata[:,idx_K2],axis=1)
                for n in range(N):
                    bootregressdata[n,iboot,:] = linregress(meanrespboot[n,:,0],meanrespboot[n,:,1])[:3]

            bootregress_sig[regressdata[:,0]>np.percentile(bootregressdata[:,:,0],97.5,axis=1),0] = 1
            bootregress_sig[regressdata[:,0]<np.percentile(bootregressdata[:,:,0],2.5,axis=1),0] = -1
            bootregress_sig[regressdata[:,1]>np.percentile(bootregressdata[:,:,1],97.5,axis=1),1] = 1
            bootregress_sig[regressdata[:,1]<np.percentile(bootregressdata[:,:,1],2.5,axis=1),1] = -1

            sig_params_regress[idx_ses,ialp,:] = bootregress_sig[idx_N3]

        #Aligned:
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        mean_resp_split_aligned[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<alphathr) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig


#%% Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))


#%%

######  #######  #####  ####### ######  ### #     #  #####  
#     # #       #     # #     # #     #  #  ##    # #     # 
#     # #       #       #     # #     #  #  # #   # #       
#     # #####   #       #     # #     #  #  #  #  # #  #### 
#     # #       #       #     # #     #  #  #   # # #     # 
#     # #       #     # #     # #     #  #  #    ## #     # 
######  #######  #####  ####### ######  ### #     #  #####  

#%% parameters:
kfold               = 5
lam                 = 0.5
model_name          = 'SVM'
scoring_type        = 'accuracy_score'
stilltrialsonly     = True
minnneurons         = 10
maxnoiselevel       = 20
nmodelfits          = 25

# model_name          = 'LogisticRegression'

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5

#%% Baseline decoding: 

#%% Decoding orientation when the FF or FB activity is high or low:
arealabelpairs          = [
                            'PMunlL2/3',
                            'V1unlL2/3',
                            ]
clrs_arealabelpairs     = ['purple','green']
legendlabels            = ['PM','V1']

narealabelpairs         = len(arealabelpairs)

# popsizes              = np.array([2,5,10,20,50,100,200,500])
popsizes                = np.array([2,5,10,20,50,100,200])
# popsizes              = np.array([5,10])
npopsizes               = len(popsizes)

error_cv                = np.full((npopsizes,narealabelpairs,nSessions,nmodelfits),np.nan)

# for ises in tqdm(range(nSessions),total=nSessions,desc='Decoding across sessions'):
for ises in tqdm([0,1],total=nSessions,desc='Decoding across sessions'):
    data            = sessions[ises].respmat
    ori_ses         = sessions[ises].trialdata['Orientation']

    if stilltrialsonly:
        idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                    sessions[ises].respmat_runspeed < maxrunspeed)
    else:
        idx_T_still = np.ones(K,dtype=bool)
    idx_T = idx_T_still

    for ialp,alp in enumerate(arealabelpairs):
        idx_ses         = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][0])

        idx_N           =  np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp,
                                        #   rangeresp[idx_ses]>.04,
                                          sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        ),axis=0))[0]

        for ipop,popsize in enumerate(popsizes):
            if len(idx_N)>popsize:

                for imf in range(nmodelfits):
                    idx_N_sub = np.random.choice(idx_N,popsize,replace=False)
                    X = data[np.ix_(idx_N_sub,idx_T)].T
                    y = ori_ses[idx_T]
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y.ravel())  # C
                    X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                    error_cv[ipop,ialp,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                        lam=lam,norm_out=False,subtract_shuffle=False)
                
#%% Plot decoding performance for different number of V1 and PM neurons

fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=True,sharey=True)
for ialp,alp in enumerate(arealabelpairs):

    ax = axes[ialp]
    ax.scatter(popsizes, np.nanmean(error_cv[:,ialp,:,:],axis=(1,2)), marker='o', color=clrs_arealabelpairs[ialp])
    ax.plot(popsizes,np.nanmean(error_cv[:,ialp,:,:],axis=(2)),color=clrs_arealabelpairs[ialp], linewidth=0.5)
    shaded_error(popsizes,np.nanmean(error_cv[:,ialp,:,:],axis=(2)).T,center='mean',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)
    ax.set_ylim([0,1])
    # ax.set_xlim([0,popsizes[-1]+1])
    ax.set_xticks(popsizes)
    ax.axhline(1/16,linestyle='--',color='k',alpha=0.5)
    ax.text(popsizes[-1],1/16+0.05,'Chance',fontsize=10,ha='right',va='center')

    ax.set_xlabel('Population size')
    ax.set_ylabel('Decoding Performance')
    ax.set_title(legendlabels[ialp],fontsize=12)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Decoding_Perf_PopSize_V1PM_%dsessions' % nSessions)


#%% Decoding orientation when the FF or FB activity is high or low:
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]
narealabelpairs         = len(arealabelpairs)

perc                = 25
maxnoiselevel       = 20
nsampleneurons      = 50

error_cv            = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Decoding across sessions'):
# for ises in tqdm([9,10],total=nSessions,desc='Decoding across sessions'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

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
        
        for imf in range(nmodelfits):
 
            idx_N1              = np.where(np.all((
                                        sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        ),axis=0))[0]
            
            idx_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        ),axis=0))[0]

            idx_ses             = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][0])
            idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                        #   np.any(corrsig_cells[:,idx_ses]!=1,axis=0),
                                          rangeresp[idx_ses]>.04,
                                        #   np.any(corrsig_cells[:,idx_ses]==1,axis=0),
                                          sessions[ises].celldata['noise_level']<maxnoiselevel,

                                        #   sig_params_regress[idx_ses,ialp,0]==-1,
                                        #   sig_params_regress[idx_ses,ialp,0]!=0,
                                        #   sig_params_regress[idx_ses,ialp,0]!=1,
                                        #   sig_params_regress[idx_ses,ialp,1]==1,
                                        #   sig_params_regress[idx_ses,ialp,0]==1,
                                        #   sig_params_regress[idx_ses,ialp,1]!=1,
                                        ),axis=0))[0]

            subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
            idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
            idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

            if len(idx_N1) < minnneurons or len(idx_N2) < minnneurons or len(idx_N3) < nsampleneurons:
                # print(f'Not enough neurons in {alp} for session {ises}')
                # print(f'N1: {len(idx_N1)}, N2: {len(idx_N2)}')
                continue

            if lam is None:
                idx_N_sub = np.random.choice(idx_N3,nsampleneurons,replace=False)
                y = ori_ses
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
                # X = data.T
                X = data[idx_N_sub,:].T
                X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
                lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

            meanpopact          = np.nanmean(data[idx_N1,:],axis=0)
            # meanpopact          = np.nanmean(data[idx_N1,:],axis=0) - np.nanmean(data[idx_N2,:],axis=0)
            # meanpopact          = np.nanmean(data[idx_N1,:],axis=0) / np.nanmean(data[idx_N2,:],axis=0)

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
            
            idx_N_sub = np.random.choice(idx_N3,nsampleneurons,replace=False)
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

#%% Decoding performance increase with increased FF or FB activity:  
clrs_arealabelpairs     = ['green','purple']
legendlabels            = ['FF','FB']

palettes = [['#00E028',[0,1,0]],[[0,1,0],[0,1,0]]]
palettes = [['#7BE087','#00E028'],['#DA78E6','#8300E6']]

axtitles = np.array(['Feedforward','Feedback']) 
fig,axes = plt.subplots(1,2,figsize=(5,3),sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    data = np.nanmean(error_cv[ialp,:,:,:],axis=2)
    sns.stripplot(data.T,palette=palettes[ialp],alpha=0.5,ax=ax,jitter=0.1)
    sns.barplot(data.T,palette=palettes[ialp],alpha=0.5,ax=ax)
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

# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_StillOnly_OnlyMult_%dsessions' % nSessions, formats = ['png'])
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_StillOnly_NonMult_%dsessions' % nSessions, formats = ['png'])
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_StillOnly_%dsessions' % nSessions, formats = ['png'])
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_SigP_%dsessions' % nSessions, formats = ['png'])
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_SigN_%dsessions' % nSessions, formats = ['png'])

#%% Normalized plot:
# extreme percentiles: 
clrs_arealabelpairs     = ['green','purple']
legendlabels            = ['FF','FB']
xoffset = 1.5
normalize = True

axtitles = np.array(['Feedforward','Feedback']) 
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
ialp = 0 
data = np.nanmean(error_cv[ialp,:,:,:],axis=2)
if normalize:
    data -= data[0,:][None,:]
ax.plot(np.tile([0,1],(nSessions,1)).T,data,color='grey',ls='-',linewidth=1)
ax.errorbar(x=[0,1],y=np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(nSessions),color='k',ls='None',linewidth=2)
ax.plot(np.array([0,1]),np.nanmean(data,axis=1),color='black',ls='-',linewidth=2)

ialp = 1
data = np.nanmean(error_cv[ialp,:,:,:],axis=2)
if normalize:
    data -= data[0,:][None,:]
ax.plot(np.tile([0,1],(nSessions,1)).T+xoffset,data,color='grey',ls='-',linewidth=1)
ax.errorbar(x=np.array([0,1])+xoffset,y=np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(nSessions),
            color='k',ls='None',linewidth=2)
ax.plot(np.array([0,1])+xoffset,np.nanmean(data,axis=1),color='black',ls='-',linewidth=2)
if not normalize:
    ax.axhline(1/16,linestyle='--',color='k',alpha=0.5)
    ax.text(0.5,1/16+0.05,'Chance',fontsize=10,ha='center',va='center')
    ax.set_ylim([0,1])
else:
    ax.set_ylim([-0.05,0.2])

ax.set_ylabel('Decoding Performance')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticks(np.concatenate((np.arange(2),np.arange(2)+xoffset)),labels=['FF Low','FF High', 'FB Low','FB High'],fontsize=8)

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

restable = res.anova_table
testlabel = ['Activity','Pathway','Interaction']
for i in range(3):
    ax.text(0.5,0.95-0.07*i,'%s%s: F(%d,%d)=%1.2f, p=%1.2e' % (get_sig_asterisks(restable['Pr > F'][i]),testlabel[i],restable['Num DF'][i],restable['Den DF'][i],restable['F Value'][i],restable['Pr > F'][i])
            ,transform=plt.gca().transAxes,fontsize=7,ha='center',va='center')
# 
my_savefig(fig,savedir,'Decoding_Minimalistic_Ori_FF_FB_LowHigh_StillOnly_%dsessions' % nSessions)
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_StillOnly_Normalized_%dsessions' % nSessions, formats = ['png'])

#%% Relating decoding performance increase to multiplicative gain increases: 

arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]
narealabelpairs         = len(arealabelpairs)

perc                = 25
maxnoiselevel       = 20
nsampleneurons      = 10
nmodelfits          = 20

error_cv            = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)
params_meansub      = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)
# add_fact            = np.full((narealabelpairs,nSessions,nmodelfits),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Decoding across sessions'):
# for ises in tqdm([9,10],total=nSessions,desc='Decoding across sessions'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

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
        
        for imf in range(nmodelfits):
 
            idx_N1              = np.where(np.all((
                                        sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        ),axis=0))[0]
            
            idx_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        ),axis=0))[0]

            idx_ses             = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][0])
            idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                        #   np.any(corrsig_cells[:,idx_ses]!=1,axis=0),
                                          rangeresp[idx_ses]>.04,
                                        #   np.any(corrsig_cells[:,idx_ses]==1,axis=0),
                                          sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        ),axis=0))[0]

            subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
            idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
            idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

            if len(idx_N1) < minnneurons or len(idx_N2) < minnneurons or len(idx_N3) < nsampleneurons:
                # print(f'Not enough neurons in {alp} for session {ises}')
                # print(f'N1: {len(idx_N1)}, N2: {len(idx_N2)}')
                continue

            meanpopact          = np.nanmean(data[idx_N1,:],axis=0)
            # meanpopact          = np.nanmean(data[idx_N1,:],axis=0) - np.nanmean(data[idx_N2,:],axis=0)
            # meanpopact          = np.nanmean(data[idx_N1,:],axis=0) / np.nanmean(data[idx_N2,:],axis=0)

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
            
            idx_N_sub = np.random.choice(idx_N3,nsampleneurons,replace=False)
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

            params_regress_ses = params_regress[idx_ses]
            params_meansub[ialp,0,ises,imf] = np.nanmean(params_regress_ses[idx_N_sub,ialp,0])
            params_meansub[ialp,1,ises,imf] = np.nanmean(params_regress_ses[idx_N_sub,ialp,1])
            # params_regress          = np.full((nCells,narealabelpairs,3),np.nan)

#%% Plot increase in decoding performance for different number of V1 and PM neurons
fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=False,sharey=True)

for iparam,param in enumerate(['Mult','Add']):
    ax = axes[iparam]
    for ialp,alp in enumerate(arealabelpairs):
        xdata = params_meansub[ialp,iparam,:,:].flatten()
        # ydata = np.diff(error_cv[:,ialp,:,:],axis=1).flatten()
        ydata = error_cv[ialp,1,:,:] - error_cv[ialp,0,:,:]
        ydata = ydata.flatten()
        # ax.scatter(xdata,ydata, marker='o', color=clrs_arealabelpairs[ialp])
        # ax.scatter(xdata,ydata, marker='.', color=clrs_arealabelpairs[ialp])
        sns.regplot(x=xdata,y=ydata,ax=ax,scatter=True,ci=95,color=clrs_arealabelpairs[ialp],
                    scatter_kws={'s':8},line_kws={'color':clrs_arealabelpairs[ialp]})
    ax.axhline(0,linestyle='--',color='k',alpha=0.5)
    # ax.plot(popsizes,np.nanmean(error_cv[:,ialp,:,:],axis=(2)),color=clrs_arealabelpairs[ialp], linewidth=0.5)
    # shaded_error(popsizes,np.nanmean(error_cv[:,ialp,:,:],axis=(2)).T,center='mean',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)

    ax.text(0.1,0.9,legendlabels[0],fontsize=12,ha='center',va='center',transform=ax.transAxes,color=clrs_arealabelpairs[0])
    ax.text(0.3,0.9,legendlabels[1],fontsize=12,ha='center',va='center',transform=ax.transAxes,color=clrs_arealabelpairs[1])
    ax.set_xlabel('Ensemble mean %s factor' % param)
    if iparam == 0: 
        ax.set_ylabel('Increase Decoding Performance')
    ax.set_title(param,fontsize=12)
    # ax.set_title(legendlabels[ialp],fontsize=12)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'DeltaDecoding_Mult_Add_Ensembles_%dsessions' % nSessions)

