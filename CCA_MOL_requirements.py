"""
@author: Matthijs oude Lohuis
Champalimaud 2023

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.cross_decomposition import CCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold
from scipy.stats import zscore

from loaddata.session_info import load_sessions
from utils.psth import compute_tensor

plt.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "Arial" 

##################################################
session_list        = np.array([['LPE09830','2023_04_12']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,
                                    load_behaviordata=False, load_calciumdata=True, load_videodata=False, calciumversion='dF')

##############################################################################
## Construct tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
[tensor,t_axis]     = compute_tensor(zscore(sessions[0].calciumdata,axis=0),sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='nearby')

tensor              = tensor.transpose((0,2,1))
[N,T,allK]          = np.shape(tensor) #get dimensions of tensor

oris        = sorted(sessions[0].trialdata['Orientation'].unique())
ori_counts  = sessions[0].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
assert(len(ori_counts) == 16 or len(ori_counts) == 8)
assert(np.all(ori_counts == 200) or np.all(ori_counts == 400))


#%% Compute residuals:
tensor_res = tensor.copy()
for ori in oris:
    ori_idx     = np.where(sessions[0].trialdata['Orientation']==ori)[0]
    temp        = np.mean(tensor_res[:,:,ori_idx],axis=2)
    tensor_res[:,:,ori_idx] = tensor_res[:,:,ori_idx] - np.repeat(temp[:, :, np.newaxis], len(ori_idx), axis=2)


#%% 
tensor_ori      = np.zeros([N,T,len(oris),np.max(ori_counts)])
tensor_mean     = np.zeros([N,T,len(oris),np.max(ori_counts)])
tensor_ori_res  = np.zeros([N,T,len(oris),np.max(ori_counts)])
for iori,ori in enumerate(oris):
    ori_idx                     = np.where(sessions[0].trialdata['Orientation']==ori)[0]
    tensor_ori[:,:,iori,:]      = tensor[:,:,ori_idx]
    temp                        = np.mean(tensor_ori[:,:,iori,:],axis=2)
    tensor_mean[:,:,iori,:]     = np.repeat(temp[:, :, np.newaxis], len(ori_idx), axis=2)
    tensor_ori_res[:,:,iori,:]  = tensor_ori[:,:,iori,:] - tensor_mean[:,:,iori,:]

#%%  Show residuals:

for example_neuron in range(1000,1100):

    fig_1,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(8,5))
    # orig = np.reshape(tensor_ori[example_neuron,:,:,:],(T,-1),order='F')
    origtoplot = tensor_ori[example_neuron,:,:,:].transpose((0,2,1))
    origtoplot = np.reshape(origtoplot,(T,-1),order='F')
    ax1.imshow(origtoplot.T, aspect='auto',extent=[-1,2.17,3200,0],vmin=-0.5, vmax=2)

    meantoplot = tensor_mean[example_neuron,:,:,:].transpose((0,2,1))
    meantoplot = np.reshape(meantoplot,(T,-1),order='F')
    ax2.imshow(meantoplot.T, aspect='auto',extent=[-1,2.17,3200,0],vmin=-0.5, vmax=2)

    residtoplot = tensor_ori_res[example_neuron,:,:,:].transpose((0,2,1))
    residtoplot = np.reshape(residtoplot,(T,-1),order='F')
    ax3.imshow(residtoplot.T, aspect='auto',extent=[-1,2.17,3200,0],vmin=-0.5, vmax=2)
                                                       
# ax.set_xlabel('#Trials')
# ax.set_ylabel('Corr CCA Dim 1')
# ax.legend(['Train', 'Test','Train_PCA25', 'Test_PCA25'])



#%%  

## split into area 1 and area 2:
idx_V1 = np.where(sessions[0].celldata['roi_name']=='V1')[0]
idx_PM = np.where(sessions[0].celldata['roi_name']=='PM')[0]

# Time selection:
idx_time = np.logical_and(t_axis>=0,t_axis<=1)
# idx_time = np.logical_and(t_axis>=0,t_axis<=0.2)

DATA1 = tensor_res[np.ix_(idx_V1,idx_time,range(np.shape(tensor_res)[2]))]
DATA2 = tensor_res[np.ix_(idx_PM,idx_time,range(np.shape(tensor_res)[2]))]

# Define neural data parameters
N1,T,K      = np.shape(DATA1)
N2          = np.shape(DATA2)[0]

minN        = np.min((N1,N2)) #find common minimum number of neurons recorded


# Vary levels of regularization?

def CCA_sample_2areas(DATA1,DATA2,nN,nK,resamples=5,kFold=5,prePCA=None):
    # Data format: 
    #  DATA1 is the source data (number of source neurons x number of time points x number of trials)
    #  DATA2 is the target data (number of target neurons x number of time points x number of trials)
    N1,T,K = np.shape(DATA1)
    N2 = np.shape(DATA2)[0]
    
    corr_train = []
    corr_test = []
    
    for iRS in np.arange(resamples):
        X = DATA1[np.ix_(np.random.choice(N1,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]
        Y = DATA2[np.ix_(np.random.choice(N2,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]

        # X2 = np.reshape(X,(nN,-1),order='F').T #concatenate time bins from each trial and transpose (samples by features now)
        # plt.figure()
        # plt.imshow(X[:,:,0])
        # plt.figure()
        # # plt.imshow(X2[0::nK,:].T)
        # plt.imshow(X2[:5,:].T)

        X = np.reshape(X,(nN,-1),order='F').T #concatenate time bins from each trial and transpose (samples by features now)
        Y = np.reshape(Y,(nN,-1),order='F').T

        X = zscore(X,axis=0)  #Z score activity for each neuron
        Y = zscore(Y,axis=0)

        if prePCA and nN>prePCA:
            pca         = PCA(n_components=prePCA)
            X           = pca.fit_transform(X)
            Y           = pca.fit_transform(Y)

        model = CCA(n_components = 1,scale = False, max_iter = 1000)

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
        
    corr_train  = np.mean(corr_train)
    corr_test   = np.mean(corr_test)

    return corr_test,corr_train


def CCA_sample_2areas_v2(DATA1,DATA2,nN,nK,resamples=5,kFold=5,prePCA=None):
    # Data format: 
    #  DATA1 is the source data (number of source neurons x number of time points x number of trials)
    #  DATA2 is the target data (number of target neurons x number of time points x number of trials)
    N1,T,K = np.shape(DATA1)
    N2 = np.shape(DATA2)[0]
    
    corr_train = []
    corr_test = []
    
    for iRS in np.arange(resamples):
        X = DATA1[np.ix_(np.random.choice(N1,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]
        Y = DATA2[np.ix_(np.random.choice(N2,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]

        X = np.reshape(X,(nN,-1),order='F').T #concatenate time bins from each trial and transpose (samples by features now)
        Y = np.reshape(Y,(nN,-1),order='F').T 

        X = zscore(X,axis=0)  #Z score activity for each neuron
        Y = zscore(Y,axis=0)

        if prePCA and nN>prePCA:
            pca         = PCA(n_components=prePCA)
            X           = pca.fit_transform(X)
            Y           = pca.fit_transform(Y)

        model = CCA(n_components = 1,scale = True, max_iter = 1000)

        #Implementing cross validation
        kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)
        
        for train_index, test_index in kf.split(np.arange(nK)):
            
            train_index_bins                = np.full((T,nK),False) #init false array
            test_index_bins                 = np.full((T,nK),False)
            train_index_bins[:,train_index] = True #set all time bins of train trials to true
            test_index_bins[:,test_index]   = True
            train_index_bins                = np.reshape(train_index_bins,-1,order='F') #concatenate time bins from each trial and transpose (samples by features now)
            test_index_bins                 = np.reshape(test_index_bins,-1,order='F')

            #  plt.figure()
            #  plt.plot(train_index_bins[:100])
            #  plt.plot(test_index_bins[:100])
            X_train , X_test = X[train_index_bins,:],X[test_index_bins,:]
            Y_train , Y_test = Y[train_index_bins,:],Y[test_index_bins,:]
            
            model.fit(X_train,Y_train)

            # Compute and store canonical correlations for the first pair
            X_c, Y_c = model.transform(X_train,Y_train)
            corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
            corr_train.append(corr)

            X_c, Y_c = model.transform(X_test,Y_test)
            corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
            corr_test.append(corr)
        
    corr_train  = np.mean(corr_train)
    corr_test   = np.mean(corr_test)

    return corr_test,corr_train

#%% Apply CCA for different numbers of neurons in the two areas: 

# Intialize variables to store CCA results
# (first canonical pair)

nNeurons_samples    = [1,2,5,10,20,50,100,200,500,750,1000,1500,2000,5000]

nResamples          = 5
kFold               = 5

# With all trials, at how many neurons do you get optimal crossvalidated prediction?
CCA_nNeurons_test   = np.zeros((len(nNeurons_samples)))
CCA_nNeurons_test.fill(np.nan)
CCA_nNeurons_train  = np.zeros((len(nNeurons_samples)))
CCA_nNeurons_train.fill(np.nan)

nK = 3200
for inNs,nN in enumerate(nNeurons_samples):
    if nN<N1 and nN<N2: #only if #sampled neurons is lower than smallest number of recorded neurons 
        print(nN)
        # [CCA_nNeurons_test[inNs],CCA_nNeurons_train[inNs]] = CCA_sample_2areas(DATA1,DATA2,nN,nK,nResamples,kFold)
        [CCA_nNeurons_test[inNs],CCA_nNeurons_train[inNs]] = CCA_sample_2areas_v2(DATA1,DATA2,nN,nK,nResamples,kFold)

fig_1 = plt.figure(figsize=(8,5))
plt.plot(nNeurons_samples,CCA_nNeurons_train,color='r')
plt.plot(nNeurons_samples,CCA_nNeurons_test,color='b')
plt.xlabel('#Neurons')
plt.ylabel('Corr CCA Dim 1')
plt.legend(['Train', 'Test'])

#%% With PCA dim reduc first:
CCA_nNeurons_wPCA_test   = np.zeros((len(nNeurons_samples)))
CCA_nNeurons_wPCA_test.fill(np.nan)
CCA_nNeurons_wPCA_train  = np.zeros((len(nNeurons_samples)))
CCA_nNeurons_wPCA_train.fill(np.nan)

nK = 3200
for inNs,nN in enumerate(nNeurons_samples):
    if nN<N1 and nN<N2: #only if number of to be sampled neurons is lower than smalles number of neurons 
        print(nN)
        [CCA_nNeurons_wPCA_test[inNs],CCA_nNeurons_wPCA_train[inNs]] = CCA_sample_2areas(DATA1,DATA2,nN,nK,nResamples,kFold,prePCA=True)

fig_1,ax = plt.subplots(figsize=(8,5))
ax.plot(nNeurons_samples,CCA_nNeurons_train,color='r')
ax.plot(nNeurons_samples,CCA_nNeurons_test,color='b')
ax.plot(nNeurons_samples,CCA_nNeurons_wPCA_train,color='r',linestyle=':')
ax.plot(nNeurons_samples,CCA_nNeurons_wPCA_test,color='b',linestyle=':')
ax.set_xlabel('#Neurons')
ax.set_ylabel('Corr CCA Dim 1')
ax.legend(['Train', 'Test','Train_PCA25', 'Test_PCA25'])


#%% With PCA dim reduc and various numbers of PCs:

nPCs_samples = [1,2,5,10,20,50,100,200,500]

CCA_nPCs_test   = np.zeros((len(nPCs_samples)))
CCA_nPCs_test.fill(np.nan)
CCA_nPCs_train  = np.zeros((len(nPCs_samples)))
CCA_nPCs_train.fill(np.nan)

nK = 3200
nN = 500
for inPCs,nPC in enumerate(nPCs_samples):
    print(nPC)
    [CCA_nPCs_test[inPCs],CCA_nPCs_train[inPCs]] = CCA_sample_2areas(DATA1,DATA2,nN,nK,nResamples,kFold,prePCA=nPC)

fig_1,ax = plt.subplots(figsize=(8,5))
ax.plot(nPCs_samples,CCA_nPCs_train,color='r',linestyle=':')
ax.plot(nPCs_samples,CCA_nPCs_test,color='b',linestyle=':')
ax.set_xlabel('#PCs')
ax.set_ylabel('Corr CCA Dim 1')
ax.legend(['Train', 'Test'])


#%% For different numbers of trials:
nTrials_samples     = [5,10,20,50,100,200,500,1000,2000,3200]
# nTrials_samples     = [5,10,20,50,100,200]

CCA_nTrials_test    = np.zeros((len(nTrials_samples)))
CCA_nTrials_test.fill(np.nan)
CCA_nTrials_train   = np.zeros((len(nTrials_samples)))
CCA_nTrials_train.fill(np.nan)

nN = 100
for inKs,nK in enumerate(nTrials_samples):
    if nK<K: #only if number of to be sampled trials is lower than number of trials in this session
        print(nK)
        # [CCA_nTrials_test[inKs],CCA_nTrials_train[inKs]] = CCA_sample_2areas(DATA1,DATA2,nN,nK,nResamples,kFold)
        [CCA_nTrials_test[inKs],CCA_nTrials_train[inKs]] = CCA_sample_2areas_v2(DATA1,DATA2,nN,nK,nResamples,kFold,prePCA=False)

fig_1 = plt.figure(figsize=(8,5))
plt.plot(nTrials_samples,CCA_nTrials_train,color='r')
plt.plot(nTrials_samples,CCA_nTrials_test,color='b')
plt.xlabel('#Trials')
plt.ylabel('Corr CCA Dim 1')
plt.legend(['Train', 'Test'])

#%% With PCA dim reduc first:
CCA_nTrials_wPCA_test   = np.zeros((len(nTrials_samples)))
CCA_nTrials_wPCA_test.fill(np.nan)
CCA_nTrials_wPCA_train  = np.zeros((len(nTrials_samples)))
CCA_nTrials_wPCA_train.fill(np.nan)

nN = 500
for inKs,nK in enumerate(nTrials_samples):
    if nK<K: #only if number of to be sampled trials is lower than number of trials in this session
        print(nK)
        [CCA_nTrials_wPCA_test[inKs],CCA_nTrials_wPCA_train[inKs]] = CCA_sample_2areas(DATA1,DATA2,nN,nK,nResamples,kFold,prePCA=True)

fig_1,ax = plt.subplots(figsize=(8,5))
ax.plot(nTrials_samples,CCA_nTrials_train,color='r')
ax.plot(nTrials_samples,CCA_nTrials_test,color='b')
ax.plot(nTrials_samples,CCA_nTrials_wPCA_train,color='r',linestyle=':')
ax.plot(nTrials_samples,CCA_nTrials_wPCA_test,color='b',linestyle=':')
ax.set_xlabel('#Trials')
ax.set_ylabel('Corr CCA Dim 1')
ax.legend(['Train', 'Test','Train_PCA25', 'Test_PCA25'])

#%% For both combinations:

nNeurons_samples    = [5,10,20,50,100,200,500,750,1000]
nTrials_samples     = [5,10,20,50,100,200,500,750,1000,2000,3200]

CCA_mVar_test    = np.zeros((len(nTrials_samples),len(nNeurons_samples)))
CCA_mVar_test.fill(np.nan)
CCA_mVar_train   = np.zeros((len(nTrials_samples),len(nNeurons_samples)))
CCA_mVar_train.fill(np.nan)

for inNs,nN in enumerate(nNeurons_samples):
    if nN<N1 and nN<N2: #only if number of to be sampled neurons is lower than smalles number of neurons 
        for inKs,nK in enumerate(nTrials_samples):
            if nK<K: #only if number of to be sampled trials is lower than number of trials
                print(nK,'-',nN)
                X = DATA1[np.ix_(np.random.choice(N1,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]
                Y = DATA2[np.ix_(np.random.choice(N2,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]

                X = np.reshape(X,(nN,-1)).T #concatenate time bins from each trial and transpose (samples by features now)
                Y = np.reshape(Y,(nN,-1)).T

                X = zscore(X,axis=0)  #Z score activity for each neuron
                Y = zscore(Y,axis=0)

                model = CCA(n_components = 1,scale = False, max_iter = 1000)

                #Implementing cross validation
                k = 5
                kf = KFold(n_splits=k, random_state=None,shuffle=True)
                
                corr_train = []
                corr_test = []
                
                for train_index , test_index in kf.split(X):
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
                    
                CCA_mVar_train[inKs,inNs] = sum(corr_train)/k
                CCA_mVar_test[inKs,inNs] = sum(corr_test)/k

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Arial" 

fig_1,(ax1,ax2) = plt.subplots(1,2,figsize=(8,5))
ax1.imshow(CCA_mVar_train,interpolation='none',cmap='Reds',aspect='auto',origin='lower')
ax2.imshow(CCA_mVar_test,interpolation='none',cmap='Reds',aspect='auto',origin='lower')
ax1.set(xticks=range(len(nNeurons_samples)), xticklabels=nNeurons_samples,
        yticks=range(len(nTrials_samples)), yticklabels=nTrials_samples)
ax2.set(xticks=range(len(nNeurons_samples)), xticklabels=nNeurons_samples,
        yticks=range(len(nTrials_samples)), yticklabels=nTrials_samples)
ax1.set_xlabel('#Neurons')
ax1.set_ylabel('#Trials')
ax2.set_xlabel('#Neurons')
ax2.set_ylabel('#Trials')
ax1.set_title('Train')
ax2.set_title('Test')
               
#%% Within versus across area dimensionality: 
#  
def CCA_sample_2areas(DATA1,DATA2,nN,nK,resamples=5,kFold=5,prePCA=False):
    N1,T,K = np.shape(DATA1)
    N2 = np.shape(DATA2)[0]
    
    corr_train = []
    corr_test = []
    
    for iRS in np.arange(resamples):
        X = DATA1[np.ix_(np.random.choice(N1,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]
        Y = DATA2[np.ix_(np.random.choice(N2,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]

        X = np.reshape(X,(nN,-1)).T #concatenate time bins from each trial and transpose (samples by features now)
        Y = np.reshape(Y,(nN,-1)).T

        X = zscore(X,axis=0)  #Z score activity for each neuron
        Y = zscore(Y,axis=0)

        if prePCA and nN>25:
            pca         = PCA(n_components=25)
            X           = pca.fit_transform(X)

        model = CCA(n_components = 1,scale = False, max_iter = 1000)

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
        
    corr_train  = np.mean(corr_train)
    corr_test   = np.mean(corr_test)
    # corr_train  = sum(corr_train)/(kFold*resamples)
    # corr_test   = sum(corr_test)/(kFold*resamples)

    return corr_test,corr_train

# nK = 3200
# for inNs,nN in enumerate(nNeurons_samples):
#     if nN<N1 and nN<N2: #only if number of to be sampled neurons is lower than smalles number of neurons 
#         print(nN)
        
#         X = DATA1[np.ix_(np.random.choice(N1,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]
#         Y = DATA2[np.ix_(np.random.choice(N2,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]

#         [corrtrain,corrtest] = CCA_wrapper(X,Y,nN,nK,resamples=5,kFold=5)

#         X = np.reshape(X,(nN,-1)).T #concatenate time bins from each trial and transpose (samples by features now)
#         Y = np.reshape(Y,(nN,-1)).T

#         X = zscore(X,axis=0)  #Z score activity for each neuron
#         Y = zscore(Y,axis=0)

#         model = CCA(n_components = 1,scale = False, max_iter = 1000)

#         #Implementing cross validation
#         k   = 5
#         kf  = KFold(n_splits=k, random_state=None,shuffle=True)
        
#         corr_train = []
#         corr_test = []
        
#         for train_index, test_index in kf.split(X):
#         # for train_index, test_index in kf.split(np.arange(K)):
#             X_train , X_test = X[train_index,:],X[test_index,:]
#             Y_train , Y_test = Y[train_index,:],Y[test_index,:]
            
#             model.fit(X_train,Y_train)

#             # Compute and store canonical correlations for the first pair
#             X_c, Y_c = model.transform(X_train,Y_train)
#             corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
#             corr_train.append(corr)

#             X_c, Y_c = model.transform(X_test,Y_test)
#             corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
#             corr_test.append(corr)
            
#         CCA_nNeurons_train[inNs] = sum(corr_train)/k
#         CCA_nNeurons_test[inNs] = sum(corr_test)/k