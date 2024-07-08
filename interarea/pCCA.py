#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  2 16:52:28 2023

@author: joana
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.cross_decomposition import CCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% DEFINE FUNCTION FOR PROBABILISTIC CCA

def probCCA(X, Y, A, B, r, n_pairs):
    
    n1 = np.shape(X)[1]
    
    data_all = np.concatenate((X,Y), axis = 1)
    
    S = np.cov(data_all.T)
    S1 = S[:n1,:n1]
    S2 = S[n1:,n1:]
            
    if n_pairs == 1:
        
        R = r
        
    else:
    
        R = np.diag(r)
    
    M1 = R**(1/2)
    M2 = M1
    
    if n_pairs == 1:
        
        W1 = np.matmul(S1,A) * M1
        W2 = np.matmul(S2,B) * M2
        
    else:
    
        W1 = np.matmul(np.matmul(S1, A), M1)
        W2 = np.matmul(np.matmul(S2, B), M2)
    
    Psi1 = S1 - np.matmul(W1,W1.T)
    Psi2 = S2 - np.matmul(W2,W2.T)

    return [W1, W2, Psi1, Psi2, A, B]

def CompSubsCorr(X, Y, W1, W2, n_pairs, n_V1, n_V2, epsilon):
    
    S1 = np.cov(X.T)
    S2 = np.cov(Y.T)
    S12 = np.cov(X.T, Y.T)    
    
    if n_V1 == 1 and n_V2 > 1:
        
        inv_S1 = S1**(-1)
        inv_S2 = np.linalg.inv(S2 + epsilon * np.eye(n_V2))
        
        A = S1 * W1
        B = np.matmul(inv_S2,W2)
        
        X_hat = X * A
        Y_hat = np.matmul(Y,B)
        
    elif n_V2 == 1 and n_V1 > 1:
        
        inv_S2 = S2**(-1)
        inv_S1 = np.linalg.inv(S1 + epsilon * np.eye(n_V1))
        
        A = np.matmul(inv_S1,W1)
        B = S2 * W2
        
        X_hat = np.matmul(X,A)
        Y_hat = Y * B
        
    else:
        
        inv_S1 = np.linalg.inv(S1 + epsilon * np.eye(n_V1))
        inv_S2 = np.linalg.inv(S2 + epsilon * np.eye(n_V2))
    
        A = np.matmul(inv_S1,W1)
        B = np.matmul(inv_S2,W2)
    
        X_hat = np.matmul(X,A)
        Y_hat = np.matmul(Y,B)
    
    # X_hat_zc = zscore(X_hat)
    # Y_hat_zc = zscore(Y_hat)
        
    if n_pairs == 1:
        
        r_hat = np.corrcoef(X_hat, Y_hat, rowvar = False)[0,1]
        
    else:   
        
        cca = CCA(n_components = n_pairs, scale = False)
        cca.fit(X_hat,Y_hat)
            
        X_c, Y_c = cca.transform(X_hat,Y_hat)
    
        r_hat = np.corrcoef(X_c,Y_c, rowvar = False)[0,1]
    
    return r_hat, S12 

#%% INVESTIGATE STABILITY OF CCA DIMS (NO CV)

# Data format: 
    
    #  X is the source data (number of source neurons x number of time points x number of trials)
    #  Y is the target data (number of target neurons x number of time points x number of trials)

# Define neural data parameters

n_steps = 100
n_V1 = 159
n_V2 = 24
n_total = n_V1 + n_V2

# Define temporal parameters

window_len = 20
epoch_init = np.arange(0, n_steps - window_len, window_len)
epoch_end = epoch_init + window_len   
n_epochs = len(epoch_init)

# Initialize variables to store results

n_pairs = 1
corr_hat = np.zeros((n_epochs, n_epochs))
corr_original = np.zeros((n_epochs, n_epochs))
corr_test = np.zeros((n_epochs, n_epochs))
corr_fit = np.zeros((n_epochs, n_epochs))
norm_corr = np.zeros((n_epochs, n_epochs))

cov_all = np.zeros((n_total, n_total, n_epochs))

W1_fit_all = np.zeros((n_V1,n_epochs))
W2_fit_all = np.zeros((n_V2,n_epochs))

A_fit_all = np.zeros((n_V1, n_epochs))
B_fit_all = np.zeros((n_V2, n_epochs))

epsilon = 0

for i in range(n_epochs):
    
    print(i)
    
    ind_init_fit = epoch_init[i]
    ind_end_fit = epoch_end[i]
       
    X_epoch_fit_window = X[:,ind_init_fit:ind_end_fit,:]            
    Y_epoch_fit_window = Y[:,ind_init_fit:ind_end_fit,:]
    
    X_epoch_fit = np.reshape(X_epoch_fit_window,(n_V1,window_len * n_trials), order = 'F').T
    Y_epoch_fit = np.reshape(Y_epoch_fit_window,(n_V2,window_len * n_trials), order = 'F').T
    
    # Apply CCA
                    
    cca_fit = CCA(n_components = n_pairs, scale = False)
    cca_fit.fit(X_epoch_fit,Y_epoch_fit)
    
    X_c_fit, Y_c_fit = cca_fit.transform(X_epoch_fit,Y_epoch_fit)
    
    r_fit = np.corrcoef(X_c_fit,Y_c_fit, rowvar = False)[0,1]
    
    A_fit = cca_fit.x_weights_
    B_fit = cca_fit.y_weights_  
            
    X_c_fit_check = np.matmul(X_epoch_fit,A_fit)
    Y_c_fit_check = np.matmul(Y_epoch_fit,B_fit)
                    
    # Apply probabilistic CCA
    
    [W1_fit, W2_fit, Psi1_fit, Psi2_fit, A_fit, B_fit] = probCCA(X_epoch_fit, Y_epoch_fit, A_fit, B_fit, r_fit, n_pairs)
    
    W1_fit_all[:,i] = np.reshape(W1_fit,n_neurons_1)
    W2_fit_all[:,i] = np.reshape(W2_fit,n_neurons_2)
    
    A_fit_all[:,i] = np.reshape(A_fit,n_neurons_1)
    B_fit_all[:,i] = np.reshape(B_fit,n_neurons_2)
    
    for j in range(n_epochs):
        
        print(j)
        
        ind_init_test = epoch_init[j]
        ind_end_test = epoch_end[j]
                
        X_epoch_test_window = X[:,ind_init_test:ind_end_test,:]         
        Y_epoch_test_window = Y[:,ind_init_test:ind_end_test,:]
    
        X_epoch_test = np.reshape(X_epoch_test_window,(n_V1,window_len * n_trials), order = 'F').T
        Y_epoch_test = np.reshape(Y_epoch_test_window,(n_V2,window_len * n_trials), order = 'F').T

        # Apply CCA
        
        cca_test = CCA(n_components = n_pairs, scale = False)
        cca_test.fit(X_epoch_test,Y_epoch_test)
        
        X_c_test, Y_c_test = cca_test.transform(X_epoch_test,Y_epoch_test)
        
        r_test = np.corrcoef(X_c_test,Y_c_test, rowvar = False)[0,1]
            
        A_test = cca_test.x_weights_
        B_test = cca_test.y_weights_ 
        
        # Apply probabilistic CCA
        
        [W1_test, W2_test, Psi1_test, Psi2_test, A_test, B_test] = probCCA(X_epoch_test, Y_epoch_test, A_test, B_test, r_test, n_pairs)
        
        # Compute subspace correlation
        
        W1_fit_i = W1_fit_all[:,i]
        W2_fit_i = W2_fit_all[:,i]
        
        r_hat, S12 = CompSubsCorr(X_epoch_test, Y_epoch_test, W1_fit_i, W2_fit_i, n_pairs, n_V1, n_V2, epsilon)
        
        r_original, _ = CompSubsCorr(X_epoch_test, Y_epoch_test, W1_test, W2_test, n_pairs, n_V1, n_V2, epsilon)
        
        #  Save canonical correlations and normalized correlations
        
        corr_fit[i,j] = r_fit
        
        corr_hat[i,j] = r_hat
        
        corr_original[i,j] = r_original
        
        corr_test[i,j] = r_test
        
        norm_corr[i,j] = r_hat/r_original  
        
        # Save covariance matrices
        
        cov_all[:,:,j] = S12
        
#%% PLOT (STABILITY OF CCA DIMS)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plot = plt.imshow(norm_corr) 
plt.xticks([])
plt.yticks([])
ax.set_xlabel('Time Used for Correlation')
ax.set_ylabel('Time Used for Fitting')
ax.set_aspect('equal')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.box(on = None)
cbar = fig.colorbar(plot, cax = cax)
cbar.set_label('Normalized correlation')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plot = plt.imshow(np.abs(norm_corr), vmin = 0, vmax = 1) 
plt.xticks([])
plt.yticks([])
ax.set_xlabel('Time Used for Correlation')
ax.set_ylabel('Time Used for Fitting')
ax.set_aspect('equal')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.box(on = None)
cbar = fig.colorbar(plot, cax = cax)
cbar.set_label('Normalized correlation')
