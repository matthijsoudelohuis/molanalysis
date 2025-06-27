# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *
from utils.corr_lib import *
from utils.tuning import compute_tuning_wrapper
from utils.RRRlib import *
from utils.regress_lib import *
from preprocessing.preprocesslib import assign_layer

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\Labeling')

#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% 
areas       = ['V1','PM','AL','RSP']
nareas      = len(areas)

# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion=calciumversion,keepraw=False)
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,keepraw=False)

t_axis = sessions[0].t_axis

#%% Test wrapper function: (should be fast, just one model fit etc. EV around 0.05-0.07)
nsampleneurons  = 20
nranks          = 25
nmodelfits      = 1 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions),np.nan)
optim_rank      = np.full((nSessions),np.nan)
idx_resp        = np.where((t_axis>=0) & (t_axis<=1))[0]

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    #on tensor during the response:
    X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)].reshape(len(idx_areax),-1).T
    Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)].reshape(len(idx_areay),-1).T

    #on averaged response:
    # X                   = np.nanmean(sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)],axis=2).T
    # Y                   = np.nanmean(sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)],axis=2).T

    R2_cv[ises],optim_rank[ises],_      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))

#%% 


#%% 

######  ######  ######     #          #    ######     #     #  #####     #     # #     # #       
#     # #     # #     #    #         # #   #     #    #     # #     #    #     # ##    # #       
#     # #     # #     #    #        #   #  #     #    #     # #          #     # # #   # #       
######  ######  ######     #       #     # ######     #     #  #####     #     # #  #  # #       
#   #   #   #   #   #      #       ####### #     #     #   #        #    #     # #   # # #       
#    #  #    #  #    #     #       #     # #     #      # #   #     #    #     # #    ## #       
#     # #     # #     #    ####### #     # ######        #     #####      #####  #     # ####### 

#%

from utils.pair_lib import value_matching

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons

# arealabelpairs  = ['V1unl-V1unl',
#                     'V1unl-V1lab',
#                     'V1lab-V1lab',
#                     'PMunl-PMunl',
#                     'PMunl-PMlab',
#                     'PMlab-PMlab',
#                     'V1unl-PMunl',
#                     'V1unl-PMlab',
#                     'V1lab-PMunl',
#                     'V1lab-PMlab',
#                     'PMunl-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1unl',
#                     'PMlab-V1lab']

# arealabelpairs  = [
#                     'V1unl-PMunl', #Feedforward pairs
#                     'V1lab-PMunl',
#                     'V1unl-PMlab',
#                     'V1lab-PMlab',

#                     'PMunl-V1unl', #Feedback pairs
#                     'PMlab-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1lab']

arealabelpairs  = [ #Feedforward selectivity
                    'V1unl-PMunl', #Feedforward pairs
                    'V1lab-PMunl',
                    'V1unl-ALunl', #to AL
                    'V1lab-ALunl',
                    'V1unl-RSPunl', #to RSP
                    'V1lab-RSPunl']

# arealabelpairs  = [ #Feedback selectivity
#                     'PMunl-V1unl', #Feedback pairs
#                     'PMlab-V1unl',
#                     'PMunl-ALunl', #to AL
#                     'PMlab-ALunl',
#                     'PMunl-RSPunl',#to RSP
#                     'PMlab-RSPunl',
#                     ]

# arealabelpairs  = ['V1unl-PMunl',
#                     'V1lab-PMunl',
#                     'V1unl-PMlab',
#                     'V1lab-PMlab',
#                     'PMunl-V1unl',
#                     'PMlab-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1lab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nranks              = 20
nmodelfits          = 50 #number of times new neurons are resampled 
kfold               = 5
maxnoiselevel       = 20
# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
# idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
idx_resp            = np.where((t_axis>=0.5) & (t_axis<=1.5))[0]
ntimebins           = len(idx_resp)

R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions),np.nan)
R2_ranks            = np.full((narealabelpairs,nSessions,nranks,nmodelfits,kfold),np.nan)

filter_nearby       = True
# filter_nearby       = False

valuematching       = None
# valuematching       = 'noise_level'
# valuematching       = 'event_rate'
# valuematching       = 'skew'
# valuematching       = 'meanF'
nmatchbins          = 10
minsampleneurons    = 10

timeaverage         = True
# perOri              = False
sub_mean            = True
sub_PC1             = False
# timeaverage         = False

version             = '%s_%s%s'%('mean' if timeaverage else 'tensor','resid' if sub_mean else 'orig','_PC1sub' if sub_PC1 else '')

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    # idx_T               = ses.trialdata['stimCond']==0

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    # allpops             = np.array([i.split('-') for i in arealabelpairs]).flatten()
    # nsampleneurons      = np.min([np.sum((ses.celldata['arealabel']==i) & (ses.celldata['noise_level']<maxnoiselevel)) for i in allpops])
    allpops             = np.array([i.split('-') for i in arealabelpairs])
    sourcepops,targetpops          = allpops[:,0],allpops[:,1]
    nsampleneurons      = np.min(np.concatenate(([np.sum(np.all((ses.celldata['arealabel']==i,
                                                 ses.celldata['noise_level']<maxnoiselevel,
                                                 idx_nearby),axis=0)) for i in sourcepops],
                                    [np.sum(np.all((ses.celldata['arealabel']==i,
                                                 ses.celldata['noise_level']<maxnoiselevel
                                                 ),axis=0)) for i in targetpops])))
    #take the smallest sample size

    tensor              = copy.deepcopy(ses.tensor)

    tensor              -= np.nanmean(tensor,axis=(1,2),keepdims=True)
    tensor              /= np.nanstd(tensor,axis=(1,2),keepdims=True)

    #subtract mean response per stimulus condition (e.g. grating direction):
    if sub_mean:
        stim                = ses.trialdata['stimCond']
        stimconds           = np.sort(stim.unique())
        for stimcond in stimconds:
            tensor[:,stim==stimcond,:] -= np.nanmean(tensor[:,stim==stimcond,:],axis=1,keepdims=True)

    # Subtract PC1:         #Remove low rank prediction from data per stimulus (accounts for mult. gain for example)
    if sub_PC1: 
        stim                = ses.trialdata['stimCond']
        stimconds           = np.sort(stim.unique())
        for stimcond in stimconds:
            N,K,T       = np.shape(tensor)
            data        = tensor[:,stim==stimcond,:].reshape(N,-1).T
            pca         = PCA(n_components=1)
            data_hat    = pca.inverse_transform(pca.fit_transform(data))
            tensor[:,stim==stimcond,:] -= data_hat.T.reshape(N,np.sum(stim==stimcond),T)

            # data        = Y[:,stim==stimcond,:].reshape(NY,-1).T
            # # pca         = PCA(n_components=1)
            # data_hat    = pca.inverse_transform(pca.fit_transform(data))
            # Y[:,stim==stimcond,:] -= data_hat.T.reshape(NY,np.sum(stim==stimcond),ntimebins)

    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby
                                ),axis=0))[0]
    
        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        if valuematching is not None: #balance populations by some metric from celldata:
            #Get value to match from celldata:
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_areax,idx_areay))
            group       = np.concatenate((np.zeros(len(idx_areax)),np.ones(len(idx_areay))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_areax   = np.intersect1d(idx_areax,idx_sub) #recover subset from idx_joint
            idx_areay   = np.intersect1d(idx_areay,idx_sub)

        #get data from tensor during the response:
        X                   = tensor[np.ix_(idx_areax,idx_T,idx_resp)]
        Y                   = tensor[np.ix_(idx_areay,idx_T,idx_resp)]

        if timeaverage:
            X = np.nanmean(X,axis=2,keepdims=True)
            Y = np.nanmean(Y,axis=2,keepdims=True)

        X                   = X.reshape(len(idx_areax),-1).T
        Y                   = Y.reshape(len(idx_areay),-1).T

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue
        R2_cv[iapl,ises],optim_rank[iapl,ises],R2_ranks[iapl,ises,:,:,:]  = RRR_wrapper(Y, X, 
                        nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
        #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
    del tensor


#%% Plot the R2 performance and number of dimensions per area pair
# fig         = plot_RRR_R2_arealabels(R2_cv,optim_rank,R2_ranks,arealabelpairs,clrs_arealabelpairs)
# my_savefig(fig,savedir,'RRR_cvR2_RegressOutBehavior_V1PM_LabUnl_%dsessions' % nSessions)
# fig         = plot_RRR_R2_arealabels(R2_cv[:2],optim_rank[:2],R2_ranks[:2],arealabelpairs[:2],clrs_arealabelpairs[:2])
# fig         = plot_RRR_R2_arealabels(R2_cv[2:4],optim_rank[2:4],R2_ranks[2:4],arealabelpairs[2:4],clrs_arealabelpairs[2:4])
# fig         = plot_RRR_R2_arealabels(R2_cv[[1,3]],optim_rank[[1,3]],R2_ranks[[0,3]],arealabelpairs[[0,3]],clrs_arealabelpairs[2:4])

normalize   = False
#Same target population:
for idx in np.array([[0,1],[2,3],[4,5],[6,7]]):
    clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
    # fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs,normalize=normalize)
    fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs,normalize=normalize)
    my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

for idx in np.array([[0,1],[2,3],[4,5],[6,7]]):
    mean,sd = np.nanmean(R2_cv[idx[1]] / R2_cv[idx[0]])*100-100,np.nanstd(R2_cv[idx[1]] / R2_cv[idx[0]])
    print('%s vs %s: %2.1f %% +/- %2.1f' % (arealabelpairs[idx[1]],arealabelpairs[idx[0]],mean,sd))

#Different target population:
for idx in np.array([[0,3],[4,7]]):
    clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
    fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs,normalize=normalize)
    my_savefig(fig,savedir,'RRR_cvR2_diffTarget_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

for idx in np.array([[0,3],[4,7]]):
    mean,sd = np.nanmean(R2_cv[idx[1]] / R2_cv[idx[0]])*100-100,np.nanstd(R2_cv[idx[1]] / R2_cv[idx[0]])
    print('%s vs %s: %2.1f %% +/- %2.1f' % (arealabelpairs[idx[1]],arealabelpairs[idx[0]],mean,sd))

# normalize   = False
# fig         = plot_RRR_R2_arealabels(R2_cv[:4],optim_rank[:4],R2_ranks[:4],arealabelpairs[:4],clrs_arealabelpairs[:4],normalize=normalize)
# my_savefig(fig,savedir,'RRR_cvR2_V1PM_LabUnl_%s_%dsessions' % (version,nSessions))
# fig         = plot_RRR_R2_arealabels(R2_cv[4:],optim_rank[4:],R2_ranks[4:],arealabelpairs[4:],clrs_arealabelpairs[4:],normalize=normalize)
# my_savefig(fig,savedir,'RRR_cvR2_PMV1_LabUnl_%s_%dsessions' % (version,nSessions))

#%% AL and RSP sessions only:
normalize   = False
for idx in np.array([[0,1],[2,3],[4,5]]):
    clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
    fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs,normalize=normalize)
    # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%d_ALRSP_sessions' % (arealabelpairs[idx[1]],version,nSessions))

for idx in np.array([[0,1],[2,3],[4,5]]):
    mean,sd = np.nanmean(R2_cv[idx[1]] / R2_cv[idx[0]])*100-100,np.nanstd(R2_cv[idx[1]] / R2_cv[idx[0]])
    print('%s vs %s: %2.1f %% +/- %2.1f' % (arealabelpairs[idx[1]],arealabelpairs[idx[0]],mean,sd))


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1unl-PMunl',
                    'V1lab-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMlab',
                    'PMunl-V1unl',
                    'PMlab-V1unl',
                    'PMunl-V1lab',
                    'PMlab-V1lab',
                    'V1unl-ALunl',
                    'V1lab-ALunl',
                    'PMunl-ALunl',
                    'PMlab-ALunl',
                    'V1unl-RSPunl',
                    'V1lab-RSPunl',
                    'PMunl-RSPunl',
                    'PMlab-RSPunl',
                    ]

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nranks              = 20
nmodelfits          = 1 #number of times new neurons are resampled 
kfold               = 5
maxnoiselevel       = 20
# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
# idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
idx_resp            = np.where((t_axis>=0.5) & (t_axis<=1.5))[0]
# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
ntimebins           = len(idx_resp)

R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions),np.nan)
R2_ranks            = np.full((narealabelpairs,nSessions,nranks,nmodelfits,kfold),np.nan)

filter_nearby       = True

minsampleneurons    = 10

timeaverage         = True
# perOri              = False
sub_mean            = True
sub_PC1             = False
# timeaverage         = False

version             = '%s_%s%s'%('mean' if timeaverage else 'tensor','resid' if sub_mean else 'orig','_PC1sub' if sub_PC1 else '')

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    # idx_T               = ses.trialdata['stimCond']==0

    allpops             = np.array([i.split('-') for i in arealabelpairs]).flatten()
    nsampleneurons      = np.min([np.sum((ses.celldata['arealabel']==i) & (ses.celldata['noise_level']<maxnoiselevel)) for i in allpops])
    #take the smallest sample size

    tensor              = copy.deepcopy(ses.tensor)

    tensor              -= np.nanmean(tensor,axis=(1,2),keepdims=True)
    tensor              /= np.nanstd(tensor,axis=(1,2),keepdims=True)

    #subtract mean response per stimulus condition (e.g. grating direction):
    if sub_mean:
        stim                = ses.trialdata['stimCond']
        stimconds           = np.sort(stim.unique())
        for stimcond in stimconds:
            tensor[:,stim==stimcond,:] -= np.nanmean(tensor[:,stim==stimcond,:],axis=1,keepdims=True)

    # Subtract PC1:         #Remove low rank prediction from data per stimulus (accounts for mult. gain for example)
    if sub_PC1: 
        stim                = ses.trialdata['stimCond']
        stimconds           = np.sort(stim.unique())
        for stimcond in stimconds:
            N,K,T       = np.shape(tensor)
            data        = tensor[:,stim==stimcond,:].reshape(N,-1).T
            pca         = PCA(n_components=1)
            data_hat    = pca.inverse_transform(pca.fit_transform(data))
            tensor[:,stim==stimcond,:] -= data_hat.T.reshape(N,np.sum(stim==stimcond),T)

    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=25)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    
        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.split(np.random.permutation(idx_areax), 2)

        if valuematching is not None: #balance populations by some metric from celldata:
            #Get value to match from celldata:
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_areax,idx_areay))
            group       = np.concatenate((np.zeros(len(idx_areax)),np.ones(len(idx_areay))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_areax   = np.intersect1d(idx_areax,idx_sub) #recover subset from idx_joint
            idx_areay   = np.intersect1d(idx_areay,idx_sub)

        #get data from tensor during the response:
        X                   = tensor[np.ix_(idx_areax,idx_T,idx_resp)]
        Y                   = tensor[np.ix_(idx_areay,idx_T,idx_resp)]

        if timeaverage:
            X = np.nanmean(X,axis=2,keepdims=True)
            Y = np.nanmean(Y,axis=2,keepdims=True)

        X                   = X.reshape(len(idx_areax),-1).T
        Y                   = Y.reshape(len(idx_areay),-1).T

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue
        R2_cv[iapl,ises],optim_rank[iapl,ises],R2_ranks[iapl,ises,:,:,:]  = RRR_wrapper(Y, X, 
                        nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
        #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
    del tensor
