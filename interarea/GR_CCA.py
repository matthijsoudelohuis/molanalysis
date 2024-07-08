# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math, os
from loaddata.get_data_folder import get_local_drive
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import PCA_gratings,PCA_gratings_3D
from utils.plot_lib import shaded_error
from utils.CCAlib import CCA_sample_2areas_v3

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\CCA\\GR\\')

##############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_19']])
# sessions            = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
                                    # load_calciumdata=True, load_videodata=False, calciumversion='deconv')
sessions            = filter_sessions(protocols = ['GR'],load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')
nSessions = len(sessions)

# load_respmat: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=False, 
                                    load_calciumdata=False, load_videodata=False, calciumversion='deconv')
# sessions,nSessions   = filter_sessions(protocols = ['GR'],load_behaviordata=True, 
                                    
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    
    if 'pupil_area' in sessions[ises].videodata:
        sessions[ises].videodata['pupil_area']    = medfilt(sessions[ises].videodata['pupil_area'] , kernel_size=25)
    if 'motionenergy' in sessions[ises].videodata:
        sessions[ises].videodata['motionenergy']  = medfilt(sessions[ises].videodata['motionenergy'] , kernel_size=25)
    sessions[ises].behaviordata['runspeed']   = medfilt(sessions[ises].behaviordata['runspeed'] , kernel_size=51)

    ##############################################################################
    ## Construct trial response matrix:  N neurons by K trials
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

    sessions[ises].respmat_runspeed = compute_respmat(sessions[ises].behaviordata['runspeed'],
                                                      sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
                                                    t_resp_start=0,t_resp_stop=1,method='mean')

    if 'motionenergy' in sessions[ises].videodata:
        sessions[ises].respmat_videome = compute_respmat(sessions[ises].videodata['motionenergy'],
                                                    sessions[ises].videodata['timestamps'],sessions[ises].trialdata['tOnset'],
                                                    t_resp_start=0,t_resp_stop=1,method='mean')

    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')

############################ Compute tuning metrics: ###################################

for ises in range(nSessions):
    sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='OSI')
    sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='gOSI')
    sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='tuning_var')

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%%  

sesidx = 0
ori    = 90

[N,K]           = np.shape(sessions[sesidx].respmat) #get dimensions of response matrix

oris            = np.sort(sessions[sesidx].trialdata['Orientation'].unique())
ori_counts      = sessions[sesidx].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
assert(len(ori_counts) == 16 or len(ori_counts) == 8)
# resp_meanori    = np.empty([N,len(oris)])

idx_V1 = sessions[sesidx].celldata['roi_name']=='V1'
idx_PM = sessions[sesidx].celldata['roi_name']=='PM'

ori_idx             = sessions[sesidx].trialdata['Orientation']==ori

resp_meanori        = np.nanmean(sessions[sesidx].respmat[:,ori_idx],axis=1,keepdims=True)
resp_res            = sessions[sesidx].respmat[:,ori_idx] - resp_meanori

##   split into area 1 and area 2:
DATA1               = resp_res[idx_V1,:]
DATA2               = resp_res[idx_PM,:]

DATA1_z         = zscore(DATA1,axis=1) # zscore for each neuron across trial responses
DATA2_z         = zscore(DATA2,axis=1) # zscore for each neuron across trial responses

pca             = PCA(n_components=15) #construct PCA object with specified number of components
Xp_1            = pca.fit_transform(DATA1_z.T).T #fit pca to response matrix (n_samples by n_features)
Xp_2            = pca.fit_transform(DATA2_z.T).T #fit pca to response matrix (n_samples by n_features)

plt.subplots(figsize=(3,3))
plt.scatter(Xp_1[0,:], Xp_2[0,:],s=10,color=sns.color_palette('husl',8)[4])
plt.xlabel('PCA 1 (V1)')
plt.ylabel('PCA 1 (PM)')
plt.text(5,40,'r=%1.2f' % np.corrcoef(Xp_1[0,:],Xp_2[0,:], rowvar = False)[0,1],fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'PCA_corr_example' + '.png'), format = 'png')


#%% 

nOris = 16
corr_test = np.zeros((nSessions,nOris))
corr_train = np.zeros((nSessions,nOris))
for ises in range(nSessions):
    # get signal correlations:
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
    ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)

    idx_V1 = sessions[ises].celldata['roi_name']=='V1'
    idx_PM = sessions[ises].celldata['roi_name']=='PM'

    for i,ori in enumerate(oris): # loop over orientations 
        ori_idx             = sessions[ises].trialdata['Orientation']==ori
        resp_meanori        = np.nanmean(sessions[ises].respmat[:,ori_idx],axis=1,keepdims=True)
        resp_res            = sessions[ises].respmat[:,ori_idx] - resp_meanori
        
        ## Split data into area 1 and area 2:
        DATA1               = resp_res[idx_V1,:]
        DATA2               = resp_res[idx_PM,:]

        corr_test[ises,i],corr_train[ises,i] = CCA_sample_2areas_v3(DATA1,DATA2,resamples=5,kFold=5,prePCA=25)

fig,ax = plt.subplots(figsize=(3,3))
shaded_error(ax,oris,corr_test,error='std',color='blue')
ax.set_ylim([0,1])
ax.set_xlabel('Orientation')
ax.set_ylabel('First canonical correlation')
ax.set_xticks(oris[::2])
ax.set_xticklabels(oris[::2].astype('int'),rotation = 45)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'CCA1_Gratings_%dsessions' % nSessions  + '.png'), format = 'png')

#%% 
plt.figure()
plt.scatter(Xp_1[0,:], Xp_2[0,:],s=10,color=sns.color_palette('husl',1))

np.corrcoef(Xp_1[0,:],Xp_2[0,:], rowvar = False)[0,1]

np.corrcoef(Xp_1[0,:],sessions[ises].respmat_runspeed[ori_idx], rowvar = False)[0,1]

np.corrcoef(Xp_1[0,:],sessions[ises].respmat_videome[ori_idx], rowvar = False)[0,1]

model = CCA(n_components = 1,scale = False, max_iter = 1000)

model.fit(Xp_1.T,Xp_2.T)

X_c, Y_c = model.transform(Xp_1.T,Xp_2.T)
corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]

plt.figure()
plt.scatter(X_c, Y_c)


from sklearn.cross_decomposition import CCA

nK = 200
X = DATA1[np.ix_(np.random.choice(N1,N1,replace=False),np.random.choice(K,nK,replace=False))]
Y = DATA2[np.ix_(np.random.choice(N2,N2,replace=False),np.random.choice(K,nK,replace=False))]

X = X.T
Y = Y.T

X = Xp_1.T
Y = Xp_2.T

corr_train = []
corr_test = []

model = CCA(n_components = 1,scale = False, max_iter = 1000)
from sklearn.model_selection import KFold
kFold = 5

#Implementing cross validation
kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)

for train_index, test_index in kf.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    Y_train , Y_test = Y[train_index,:],Y[test_index,:]

    model.fit(X_train,Y_train)

    # Compute and store canonical correlations for the first pair
    X_c, Y_c = model.transform(X_train,Y_train)
    corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
    corr_train.append(corr)

    X_c, Y_c = model.transform(X_test,Y_test)
    corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
    corr_test.append(corr)

