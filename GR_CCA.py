# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math, os
try:
    os.chdir('t:\\Python\\molanalysis\\')
except:
    os.chdir('e:\\Python\\molanalysis\\')
os.chdir('c:\\Python\\molanalysis\\')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import PCA_gratings,PCA_gratings_3D
from utils.plot_lib import shaded_error

savedir = 'C:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\'

##############################################################################
# session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_19']])
# sessions            = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
                                    # load_calciumdata=True, load_videodata=False, calciumversion='deconv')
sessions            = filter_sessions(protocols = ['GR'],load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')
nSessions = len(sessions)

##############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
# t_pre       = -1    #pre s
# t_post      = 1     #post s
# binsize     = 0.2   #temporal binsize in s

# for i in range(nSessions):
#     [sessions[i].tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
#                                  t_pre, t_post, binsize,method='interp_lin')

#Compute average response per trial:
for ises in range(nSessions):
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)
    # delattr(sessions[ises],'calciumdata')

    #hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
    temp = pd.DataFrame(np.reshape(np.array(sessions[ises].behaviordata['runspeed']),(len(sessions[ises].behaviordata['runspeed']),1)))
    sessions[ises].respmat_runspeed = compute_respmat(temp, sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,method='mean')
    sessions[ises].respmat_runspeed = np.squeeze(sessions[ises].respmat_runspeed)

    #hacky way to create dataframe of the video motion with F x 1 with F number of samples:
    temp = pd.DataFrame(np.reshape(np.array(sessions[ises].videodata['motionenergy']),(len(sessions[ises].videodata['motionenergy']),1)))
    sessions[ises].respmat_videome = compute_respmat(temp, sessions[ises].videodata['timestamps'], sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,method='mean')
    sessions[ises].respmat_videome = np.squeeze(sessions[ises].respmat_videome)

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


from utils.CCAwrappers import CCA_sample_2areas,CCA_sample_2areas_v2


sesidx = 0
#%%  


from sklearn.decomposition import PCA
from scipy.stats import zscore


for ises in range(nSessions):
    # get signal correlations:
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
    ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)
    # resp_meanori    = np.empty([N,len(oris)])

    idx_V1 = sessions[ises].celldata['roi_name']=='V1'
    idx_PM = sessions[ises].celldata['roi_name']=='PM'

    for i,ori in enumerate(oris):
        # resp_meanori[:,i] = np.nanmean(sessions[ises].respmat[:,sessions[ises].trialdata['Orientation']==ori],axis=1)
        ori_idx             = sessions[ises].trialdata['Orientation']==ori

        resp_meanori        = np.nanmean(sessions[ises].respmat[:,ori_idx],axis=1,keepdims=True)
        resp_res            = sessions[ises].respmat[:,ori_idx] - resp_meanori
        
        ##   split into area 1 and area 2:
        DATA1               = resp_res[idx_V1,:]
        DATA2               = resp_res[idx_PM,:]

        # DATA1 = tensor_res[np.ix_(idx_V1,idx_time,range(np.shape(tensor_res)[2]))]
        # DATA2 = tensor_res[np.ix_(idx_PM,idx_time,range(np.shape(tensor_res)[2]))]

        # Define neural data parameters
        N1,K        = np.shape(DATA1)
        N2          = np.shape(DATA2)[0]

        minN        = np.min((N1,N2)) #find common minimum number of neurons recorded


        DATA1_z = zscore(DATA1,axis=1) # zscore for each neuron across trial responses
        DATA2_z = zscore(DATA2,axis=1) # zscore for each neuron across trial responses

        pca             = PCA(n_components=15) #construct PCA object with specified number of components
        Xp_1          = pca.fit_transform(DATA1_z.T).T #fit pca to response matrix (n_samples by n_features)
        #dimensionality is now reduced from N by K to ncomp by K

        # pca         = PCA(n_components=15) #construct PCA object with specified number of components
        Xp_2          = pca.fit_transform(DATA2_z.T).T #fit pca to response matrix (n_samples by n_features)
        #dimensionality is now reduced from N by K to ncomp by K

        # DATA1 = np.expand_dims(DATA1, axis=1)
        # DATA2 = np.expand_dims(DATA2, axis=1)

        CCA_sample_2areas(DATA1,DATA2,nN=N,nK=160,resamples=5,kFold=5,prePCA=25)


plt.figure()
plt.scatter(Xp_1[0,:], Xp_2[0,:],s=10,color=sns.color_palette('husl',1))


np.corrcoef(Xp_1[0,:],Xp_2[0,:], rowvar = False)[0,1]

np.corrcoef(Xp_1[0,:],sessions[ises].respmat_runspeed[ori_idx], rowvar = False)[0,1]

np.corrcoef(Xp_1[0,:],sessions[ises].respmat_videome[ori_idx], rowvar = False)[0,1]


model = CCA(n_components = 1,scale = False, max_iter = 1000)

model.fit(Xp_1.T,Xp_2.T)

X_c, Y_c = model.transform(Xp_1,Xp_2)
corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]

plt.figure()
plt.scatter(Xp_1[0,:], Xp_2[0,:],c=corr)


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

