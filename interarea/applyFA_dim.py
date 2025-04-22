# import h5py
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy.io
from scipy.sparse import csr_matrix
from scipy import stats
import random
import math
import matplotlib.font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sklearn.metrics

#%% DEFINE FUNCTION THAT APPLIES FA TO V1-V2 DATASET

def apply_FA(V1_data, t_axis, bin_width, subset):
    
    # print('ANALYZING DATA FROM SESSION  ' + sess)
        
    # common_path = '/media/storage2/joana/PhD/V1-V2_preproc/'

    # full_path_V1 = common_path + 'V1_spikes_ori_' + sess + '.npy'

    # V1_data = np.load(full_path_V1)
    
    print('Define data parameters')

    n_V1 = np.shape(V1_data)[0]
    time = np.shape(V1_data)[1]
    n_trials_ori = np.shape(V1_data)[2]
    n_ori = np.shape(V1_data)[3]

    stim_start_bin = np.where(t_axis>=0)[0][0]
    stim_time = np.arange(0,stim_start_bin)
    spont_time = np.arange(stim_start_bin,time)

    stim = stim_start_bin
    spont = len(t_axis) - stim_start_bin
    # n_trials_all = n_trials_ori * n_ori

    print('Split data between stimulus and spontaneous periods')
    
    if len(subset) == 0:
        
        V1_stim = V1_data[:n_V1,stim_time,:,:]
        V1_spont = V1_data[:n_V1,spont_time,:,:]
        mean_activity_V1_stim = np.sum(np.mean(V1_stim[:,:,:], axis = 2), axis = 1)/(stim/1000)
        mean_activity_V1_spont = np.sum(np.mean(V1_spont[:,:,:], axis = 2), axis = 1)/(spont/1000)  
        del V1_data
    
    if len(subset) > 0:
        
        n_V1 = len(subset)
        V1_data = V1_data[subset,:,:,:]
        V1_stim = V1_data[:,stim_time,:,:]
        V1_spont = V1_data[:,spont_time,:,:]
        mean_activity_V1_stim = np.sum(np.mean(V1_stim[:,:,:], axis = 2), axis = 1)/(stim/1000)
        mean_activity_V1_spont = np.sum(np.mean(V1_spont[:,:,:], axis = 2), axis = 1)/(spont/1000)   
        del V1_data 
    
    print('Compute residuals for stimulus and spontaneous periods')
    
        # V1

    V1_stim_res = np.zeros(np.shape(V1_stim))
    psth_V1 = np.zeros((n_V1, stim, n_ori))

    for j in range(n_V1):
        
        # print(j)
        
        psth = np.mean(V1_stim[j,:,:,:], axis = 1)
        psth_V1[j,:,:] = psth
        
        for k in range(stim):
                   
            V1_stim_res[j,k,:,:] = V1_stim[j,k,:,:] - psth_V1[j,k,:]
            
    V1_spont_res = np.zeros(np.shape(V1_spont))
    psth_V1_spont = np.zeros((n_V1, spont, n_ori))

    for j in range(n_V1):
        
        # print(j)
        
        psth = np.mean(V1_spont[j,:,:,:], axis = 1)
        psth_V1_spont[j,:,:] = psth
        
        for k in range(spont):
                   
            V1_spont_res[j,k,:,:] = V1_spont[j,k,:,:] - psth_V1_spont[j,k,:]    
                 
    # Compute mean residual activities 
    mean_activity_V1_stim_res = np.mean(np.mean(V1_stim_res[:,:,:], axis = 2), axis = 1)
    mean_activity_V1_spont_res = np.mean(np.mean(V1_spont_res[:,:,:], axis = 2), axis = 1)

    # mean_activity_V1_stim_res = np.sum(np.mean(V1_stim_res[:,:,:], axis = 2), axis = 1)/(stim/1000)
    # mean_activity_V1_spont_res = np.sum(np.mean(V1_spont_res[:,:,:], axis = 2), axis = 1)/(spont/1000)
    
    del V1_stim     
    del V1_spont
        
    print('Bin data for stimulus and spontaneous periods')

    # Stimulus period

    # bin_width = 100
    n_bins_stim = math.floor(stim/bin_width)
    bin_time_stim = np.arange(0,stim,bin_width)
    
    V1_stim_bin = np.zeros((n_V1,n_bins_stim,n_trials_ori,n_ori))
    
    for i in range(n_bins_stim):
        
        # print(i)
        
        ind_init = bin_time_stim[i]
        ind_end = ind_init + bin_width
        
        V1_stim_bin[:,i,:,:] = np.sum(V1_stim_res[:,ind_init:ind_end,:,:],axis = 1)
    
    # Spontaneous period
        
    n_bins_spont = math.floor(spont/bin_width)
    bin_time_spont = np.arange(0,spont + bin_width,bin_width)    
        
    V1_spont_bin = np.zeros((n_V1,n_bins_spont,n_trials_ori,n_ori))
    
    for i in range(n_bins_spont):
        
        # print(i)
        
        ind_init = bin_time_spont[i]
        ind_end = ind_init + bin_width
        
        V1_spont_bin[:,i,:,:] = np.sum(V1_spont_res[:,ind_init:ind_end,:,:],axis = 1)
        
    del V1_stim_res
    del V1_spont_res

    print('Concatenate trials for each orientation')

    V1_stim_final = np.reshape(V1_stim_bin,(n_V1,n_bins_stim * n_trials_ori,n_ori), order = 'F')
    V1_spont_final = np.reshape(V1_spont_bin,(n_V1,n_bins_spont * n_trials_ori,n_ori), order = 'F')

    del V1_stim_bin
    del V1_spont_bin
    
    print('Apply FA to V1 data for a range of dimensions')

    n = 20
    n_components_V1 = np.arange(1,n + 1)

    cv_folds = 10
    fa_V1_stim = FactorAnalysis()
    fa_V1_spont = FactorAnalysis()

    fa_scores_V1_stim = np.zeros((n,cv_folds,n_ori))
    fa_scores_V1_spont = np.zeros((n,cv_folds,n_ori))
    
    cv = KFold(n_splits = cv_folds, shuffle = False, random_state = None)

    for i in range(n_ori):
        
        print(i)
        
        data_stim = V1_stim_final[:,:,i].T
        data_spont = V1_spont_final[:,:,i].T
        
        for j in range(n):
            
            n_components = n_components_V1[j]
            
            # print(n_components)
            
            k = 0
            
            for train_ix, test_ix in cv.split(data_stim):
                
                # Split data
                
                data_train_stim = data_stim[train_ix,:]
                data_test_stim = data_stim[test_ix,:]
                
                data_train_spont = data_spont[train_ix,:]
                data_test_spont = data_spont[test_ix,:]
                
                fa_V1_stim = FactorAnalysis()
                fa_V1_spont = FactorAnalysis()
                
                fa_V1_stim.n_components = n_components
                fa_V1_stim.fit(data_train_stim)
                score_stim = fa_V1_stim.score(data_test_stim)            
                fa_scores_V1_stim[j,k,i] = score_stim
    
                fa_V1_spont.n_components = n_components
                fa_V1_spont.fit(data_train_spont)
                score_spont = fa_V1_spont.score(data_test_spont)            
                fa_scores_V1_spont[j,k,i] = score_spont            
                            
                k += 1                    
                
    print('Find number of FA dimensions that maximizes the cross-validated LL for V1 data')       

    fa_scores_V1_stim = np.squeeze(fa_scores_V1_stim)
    fa_scores_V1_stim_mean = np.mean(fa_scores_V1_stim, axis = 1)
    
    fa_scores_V1_spont = np.squeeze(fa_scores_V1_spont)
    fa_scores_V1_spont_mean = np.mean(fa_scores_V1_spont, axis = 1)
    
    peak_dim_V1_stim = np.zeros(n_ori)
    peak_dim_V1_spont = np.zeros(n_ori)
    
    for i in range(n_ori):
        
        max_LL_V1_stim = np.max(fa_scores_V1_stim_mean[:,i])
        ind_peak_dim_V1_stim = np.where(fa_scores_V1_stim_mean[:,i] == max_LL_V1_stim)[0][0]
        peak_dim_V1_stim[i] = n_components_V1[ind_peak_dim_V1_stim]
        
        max_LL_V1_spont = np.max(fa_scores_V1_spont_mean[:,i])
        ind_peak_dim_V1_spont = np.where(fa_scores_V1_spont_mean[:,i] == max_LL_V1_spont)[0][0]
        peak_dim_V1_spont[i] = n_components_V1[ind_peak_dim_V1_spont]

    # Fit an FA model with that number of dimensions
    
    factors_V1_stim = []
    noise_V1_stim = []
    factors_V1_spont = []
    noise_V1_spont = []
    
    for i in range(n_ori):
        
        # print(i)
        
        fa_V1_stim = FactorAnalysis()
        fa_V1_spont = FactorAnalysis()
    
        fa_V1_stim.n_components = int(peak_dim_V1_stim[i])
        fa_V1_spont.n_components = int(peak_dim_V1_spont[i])
        
        fa_V1_stim.fit(V1_stim_final[:,:,i].T)
        factors_V1_stim.append(fa_V1_stim.components_)
        noise_V1_stim.append(fa_V1_stim.noise_variance_)
        
        fa_V1_spont.fit(V1_spont_final[:,:,i].T)
        factors_V1_spont.append(fa_V1_spont.components_)
        noise_V1_spont.append(fa_V1_spont.noise_variance_)
        
    # Compute shared covariance matrices and their eigendecomposition
    
    cum_var_V1_stim_all = []
    cum_var_V1_spont_all = []
    
    n_factors_V1_stim = np.zeros(n_ori)
    n_factors_V1_spont = np.zeros(n_ori)
    
    for i in range(n_ori):
    
        shared_cov_V1_stim = np.matmul(factors_V1_stim[i].T,factors_V1_stim[i])
        shared_cov_V1_spont = np.matmul(factors_V1_spont[i].T,factors_V1_spont[i])
    
        eigenval_V1_stim, _ = np.linalg.eig(shared_cov_V1_stim)
        eigenval_V1_spont, _ = np.linalg.eig(shared_cov_V1_spont)
        
        eigenval_V1_stim = np.sort(np.real(eigenval_V1_stim))
        eigenval_V1_stim = np.flip(eigenval_V1_stim)
        
        eigenval_V1_spont = np.sort(np.real(eigenval_V1_spont))
        eigenval_V1_spont = np.flip(eigenval_V1_spont)
            
        cum_var_V1_stim = np.cumsum(eigenval_V1_stim)/np.sum(eigenval_V1_stim)
        cum_var_V1_spont = np.cumsum(eigenval_V1_spont)/np.sum(eigenval_V1_spont)
        
        cum_var_V1_stim_all.append(cum_var_V1_stim)
        cum_var_V1_spont_all.append(cum_var_V1_spont)
        
        n_factors_V1_stim[i]  = np.where(np.squeeze(cum_var_V1_stim) > 0.95)[0][0] + 1 
        n_factors_V1_spont[i] = np.where(np.squeeze(cum_var_V1_spont) > 0.95)[0][0] + 1
    
        # Summing 1 because indices start at 0
        
    print('Apply FA model with optimal dimensionality to V1 and V2 data')
    
    fa_V1_stim_all = []
    fa_V1_spont_all = []
    fa_V2_stim_all = []
    fa_V2_spont_all = []
    
    for i in range(n_ori):
        
        fa_V1_stim = FactorAnalysis()
        fa_V1_spont = FactorAnalysis()
        fa_V2_stim = FactorAnalysis()
        fa_V2_spont = FactorAnalysis()
        
        n_factors_V1_stim_ori = int(n_factors_V1_stim[i])
        n_factors_V1_spont_ori = int(n_factors_V1_spont[i])

        fa_V1_stim.n_components = n_factors_V1_stim_ori
        fa_V1_spont.n_components = n_factors_V1_spont_ori
        
        fa_V1_stim.fit(V1_stim_final[:,:,i].T)
        fa_V1_spont.fit(V1_spont_final[:,:,i].T)
        
        fa_V1_stim_all.append(fa_V1_stim)
        fa_V1_spont_all.append(fa_V1_spont)
        fa_V2_stim_all.append(fa_V2_stim)
        fa_V2_spont_all.append(fa_V2_spont)
            
    return [fa_scores_V1_stim, fa_scores_V1_spont,
           n_factors_V1_stim, n_factors_V1_spont, 
           fa_V1_stim_all, fa_V1_spont_all,
           mean_activity_V1_stim, mean_activity_V1_spont, 
           mean_activity_V1_stim_res, mean_activity_V1_spont_res]
