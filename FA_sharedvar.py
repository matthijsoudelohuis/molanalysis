#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:20:44 2023

@author: joana
"""

#%% IMPORT RELEVANT PACKAGES

# import h5py
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

# Plotting definitions

plt.rcParams.update({'font.size':16})
plt.rcParams['font.family'] = 'Arial'

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

#%% LOAD OUTPUTS OF FA ANALYSIS FOR ALL DATASETS

sess1 = '105l001p16'
sess2 = '106r001p26'
sess3 = '106r002p70'
sess4 = '107l002p67'
sess5 = '107l003p143' 

sess = sess1
common_path = '/media/storage2/joana/PhD/V1-V2_preproc/'

file_name = 'FA_output_subset_' + sess
full_path = '/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/' + file_name + '_subset.npy'

FA_output = np.load(full_path, allow_pickle = True)

# Outputs of FA analysis pipeline
# 1. fa_scores_V1_stim
# 2. fa_scores_V1_spont
# 3. n_factors_V1_stim
# 4. n_factors_V1_spont 
# 5. fa_V1_stim
# 6. fa_V1_spont 

fa_scores_V1_stim = FA_output[0]
fa_scores_V1_spont = FA_output[1]

n_factors_V1_stim = FA_output[3]
n_factors_V1_spont = FA_output[4]

fa_V1_stim = FA_output[5]
fa_V1_spont = FA_output[6]

# Define data parameters for one particular session

# n_V1 = np.shape(V1_data)[0]
# n_V2 = np.shape(V2_data)[0]
# time = np.shape(V1_data)[1]
# stim_time = np.arange(0,1280)
# spont_time = np.arange(1280,time)
# stim = 1280
# spont = 1500
# n_trials_ori = np.shape(V1_data)[2]
# n_ori = np.shape(V1_data)[3]
# n_trials_all = n_trials_ori * n_ori

n_ori = np.shape(fa_scores_V1_spont)[2]

#%% Plot FA scores for V1 and V2 for one particular session

# Colors for V1: steelblue (dark) and lightblue (light)
# Colors for V2: darkred (dark) and indianread (light)

n = np.shape(fa_scores_V1_stim)[0]
n_components_V1 = np.arange(1,n + 1)
cv_folds = 10

    # V1

fa_scores_V1_stim_mean = np.mean(fa_scores_V1_stim, axis = 1)
fa_scores_V1_stim_std = scipy.stats.sem(fa_scores_V1_stim, axis = 1)

fa_scores_V1_spont_mean = np.mean(fa_scores_V1_spont, axis = 1)
fa_scores_V1_spont_std = scipy.stats.sem(fa_scores_V1_spont, axis = 1)

    # V1 stimulus period for all orientations

fig = plt.figure(figsize=(12, 8))

plt.suptitle('V1 stimulus period - session ' + sess)

for i in range(n_ori):
    
    fig.add_subplot(2,4,i + 1) 
        
    for j in range(n):    
   
        plt.scatter(np.ones(cv_folds) * n_components_V1[j], fa_scores_V1_stim[j,:,i], color = 'steelblue', alpha = 0.3, s = 3)        
        plt.scatter(n_components_V1[j], fa_scores_V1_stim_mean[j,i], color = 'steelblue', s = 10)
        
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if i == 0 or i == 4:
        plt.xlabel('Number of components')
        plt.ylabel('Cross-validated log-likelihood')

plt.tight_layout()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/FA_LL_V1_stimulus_' + sess + '.svg', format = 'svg')

    # V1 spontaneous period for all orientations

fig = plt.figure(figsize=(12, 8))

plt.suptitle('V1 spontaneous period - session' + sess)

for i in range(n_ori):
    
    fig.add_subplot(2,4,i + 1) 
        
    for j in range(n):    
   
        plt.scatter(np.ones(cv_folds) * n_components_V1[j], fa_scores_V1_spont[j,:,i], color = 'steelblue', alpha = 0.3, s = 3)        
        plt.scatter(n_components_V1[j], fa_scores_V1_spont_mean[j,i], color = 'steelblue', s = 10 )

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if i == 0 or i == 4:
        plt.xlabel('Number of components')
        plt.ylabel('Cross-validated log-likelihood')
        
plt.tight_layout()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/FA_LL_V1_spontaneous_' + sess + '.svg', format = 'svg')
        
#%% Scatter plots of V1 and V2 dimensionalities

max_dim_V1 = np.max(np.concatenate((n_factors_V1_stim,n_factors_V1_spont))) + 1
min_dim_V1 = np.min(np.concatenate((n_factors_V1_stim,n_factors_V1_spont))) -1

# Jitter dimensions to avoid overlap in plots

n_factors_V1_stim_jit = n_factors_V1_stim + np.random.normal(0,0.1,8)
n_factors_V1_spont_jit = n_factors_V1_spont + np.random.normal(0,0.1,8)

fig = plt.figure(figsize=(12, 8))

plt.suptitle('Session ' + sess)

fig.add_subplot(1,2,1)

plt.scatter(n_factors_V1_stim_jit, n_factors_V1_spont_jit, s = 20, color = 'steelblue')
ax = plt.gca()
plt.plot(np.arange(min_dim_V1, max_dim_V1),np.arange(min_dim_V1, max_dim_V1), color = 'gray', linestyle = '--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim([min_dim_V1,max_dim_V1])
plt.ylim([min_dim_V1,max_dim_V1])
ax.set_aspect('equal')
plt.xlabel('# FA factors stimulus')
plt.ylabel('# FA factors spontaneous')
plt.title('V1')
  
#%% Compute percent shared variance for each neuron during stimulus and spontaneous periods

n_V1 = np.shape(fa_V1_stim[0].components_)[1]

    # V1

shared_variance_V1_stim = np.zeros((n_V1,n_ori))
private_variance_V1_stim = np.zeros((n_V1, n_ori))
pshared_variance_V1_stim = np.zeros((n_V1, n_ori))

shared_variance_V1_spont = np.zeros((n_V1,n_ori))
private_variance_V1_spont = np.zeros((n_V1, n_ori))
pshared_variance_V1_spont = np.zeros((n_V1, n_ori))

for i in range(n_ori):
    
    print(i)
    
    V1_stim_components_ori = fa_V1_stim[i].components_
    V1_stim_noise_ori = fa_V1_stim[i].noise_variance_
    
    V1_spont_components_ori = fa_V1_spont[i].components_
    V1_spont_noise_ori = fa_V1_spont[i].noise_variance_
    
    for j in range(n_V1):
    
        shared_var_stim = np.dot(V1_stim_components_ori[:,j], V1_stim_components_ori[:,j])
        private_var_stim = V1_stim_noise_ori[j]
        
        shared_variance_V1_stim[j,i] = shared_var_stim
        private_variance_V1_stim[j,i] = private_var_stim
        pshared_variance_V1_stim[j,i] = shared_var_stim/(shared_var_stim + private_var_stim) * 100
        
        shared_var_spont = np.dot(V1_spont_components_ori[:,j], V1_spont_components_ori[:,j])
        private_var_spont = V1_spont_noise_ori[j]
        
        shared_variance_V1_spont[j,i] = shared_var_spont
        private_variance_V1_spont[j,i] = private_var_spont
        pshared_variance_V1_spont[j,i] = shared_var_spont/(shared_var_spont + private_var_spont) * 100
    

# Plots for V1 for all orientations

for i in range(n_ori):
    
    fig = plt.figure(figsize=(12, 8))

    plt.suptitle('V1 orientation ' + str(i + 1) + '  - session ' + sess)
    
    fig.add_subplot(2,1,1) 
    ax = plt.gca()               
    ax.bar(range(1,n_V1 + 1), shared_variance_V1_stim[:,i], width = 1, label = 'shared', color = 'steelblue', ec = 'mediumblue')
    ax.bar(range(1,n_V1 + 1), private_variance_V1_stim[:,i], bottom = shared_variance_V1_stim[:,i], width = 1, label = 'private', color = 'gray', ec = 'black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = 'upper right')
    ax.set_title('Stimulus period')
    ax.set_ylabel('Variance')
    
    fig.add_subplot(2,1,2) 
    ax = plt.gca()               
    ax.bar(range(1,n_V1 + 1), shared_variance_V1_spont[:,i], width = 1, label = 'shared', color = 'plum', ec = 'mediumorchid')
    ax.bar(range(1,n_V1 + 1), private_variance_V1_spont[:,i], bottom = shared_variance_V1_spont[:,i], width = 1, label = 'private', color = 'gray', ec = 'black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = 'upper right')
    ax.set_title('Spontaneous period')
    ax.set_ylabel('Variance')
    ax.set_xlabel('Neuron')
    
# Compute percent shared variance for each orientation

    # V1

mean_pshared_variance_V1_stim = np.mean(pshared_variance_V1_stim, axis = 0)
mean_pshared_variance_V1_spont = np.mean(pshared_variance_V1_spont, axis = 0)

# Save outputs

np.save('/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/mean_pshared_variance_neurons_V1_stim_' + sess + '.npy', mean_pshared_variance_V1_stim, allow_pickle = True)  
np.save('/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/mean_pshared_variance_neurons_V1_spont_' + sess + '.npy', mean_pshared_variance_V1_spont, allow_pickle = True)  

print('SAVED')

#%% Compute average percent shared variance across all orientations

mean_pshared_variance_V1_stim = np.mean(pshared_variance_V1_stim, axis = 1)
mean_pshared_variance_V1_spont = np.mean(pshared_variance_V1_spont, axis = 1)

min_V1 = np.min(np.concatenate((mean_pshared_variance_V1_stim,mean_pshared_variance_V1_spont)))
max_V1 = np.max(np.concatenate((mean_pshared_variance_V1_stim,mean_pshared_variance_V1_spont))) + 10


    # V1

fig = plt.figure(figsize=(12, 8))

plt.suptitle('V1 all orientations - session ' + sess)

fig.add_subplot(2,1,1) 
ax = plt.gca()               
ax.bar(range(1,n_V1 + 1), mean_pshared_variance_V1_stim, width = 1, label = 'shared', color = 'steelblue', ec = 'mediumblue')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0,max_V1])
ax.set_title('Stimulus period')
ax.set_ylabel('% Shared variance')

fig.add_subplot(2,1,2) 
ax = plt.gca()               
ax.bar(range(1,n_V1 + 1), mean_pshared_variance_V1_spont, width = 1, label = 'shared', color = 'plum', ec = 'mediumorchid')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0,max_V1])
ax.set_title('Spontaneous period')
ax.set_ylabel('% Shared variance')
ax.set_xlabel('Neuron')   

#%% Compute percent variance explained by each mode

shared_variance_V1_stim_mode_all = []
shared_variance_V1_spont_mode_all = []

shared_variance_V2_stim_mode_all = []
shared_variance_V2_spont_mode_all = []

for i in range(n_ori):
    
    print(i)
    
    # V1
    
    V1_stim_components_ori = fa_V1_stim[i].components_
    V1_spont_components_ori = fa_V1_spont[i].components_
    
    shared_cov_V1_stim = np.matmul(V1_stim_components_ori,V1_stim_components_ori.T)
    shared_cov_V1_spont = np.matmul(V1_spont_components_ori,V1_spont_components_ori.T)
    
    eigenval_V1_stim, _ = np.linalg.eig(shared_cov_V1_stim)
    eigenval_V1_spont, _ = np.linalg.eig(shared_cov_V1_spont)
    
    eigenval_V1_stim = np.sort(np.real(eigenval_V1_stim))
    eigenval_V1_stim = np.flip(eigenval_V1_stim)
    
    eigenval_V1_spont = np.sort(np.real(eigenval_V1_spont))
    eigenval_V1_spont = np.flip(eigenval_V1_spont)
    
    shared_variance_V1_stim_mode = eigenval_V1_stim/np.sum(eigenval_V1_stim) * 100
    shared_variance_V1_spont_mode = eigenval_V1_spont/np.sum(eigenval_V1_spont) * 100
    
    shared_variance_V1_stim_mode_all.append(shared_variance_V1_stim_mode)
    shared_variance_V1_spont_mode_all.append(shared_variance_V1_spont_mode)
        
    # Plots V1
    
fig = plt.figure(figsize=(12, 8))

plt.suptitle('V1 all orientations - session ' + sess)

for i in range(n_ori):
    
    fig.add_subplot(3,4,i + 1) 
    plt.scatter(np.arange(len(shared_variance_V1_stim_mode_all[i])), shared_variance_V1_stim_mode_all[i], color = 'steelblue', s = 20, label = 'Stimulus')    
    plt.scatter(np.arange(len(shared_variance_V1_spont_mode_all[i])), shared_variance_V1_spont_mode_all[i], color = 'grey', s = 20, label = 'Spontaneous')  
        
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if i == 0 or i == 4:
        ax.set_xlabel('FA factors')
        ax.set_ylabel('% Shared variance')
                
    if i == 7:
        plt.legend(loc ='upper right')
    
plt.tight_layout()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/Percent_shared_variance_mode_V1_' + sess + '.svg', format = 'svg')
             
# Save outputs

np.save('/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/pshared_variance_mode_V1_stim_' + sess + '.npy', shared_variance_V1_stim_mode_all, allow_pickle = True)  
np.save('/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/pshared_variance_mode_V1_spont_' + sess + '.npy', shared_variance_V1_spont_mode_all, allow_pickle = True)  

print('SAVED')
    
#%% Compute angles between the first FA mode computed during stimulus and during spontaneous periods 

angle_fa1_V1_stim_spont_all = np.zeros(n_ori)
angle_fa1_V2_stim_spont_all = np.zeros(n_ori)


for i in range(n_ori):
    
    print(i)
    
    # V1
    
    V1_stim_components_ori = fa_V1_stim[i].components_
    V1_spont_components_ori = fa_V1_spont[i].components_     
    
    fa1_V1_stim = V1_stim_components_ori[0,:]
    fa1_V1_spont = V1_spont_components_ori[0,:]
    
    fa2_V1_stim = V1_stim_components_ori[1,:]
    fa2_V1_spont = V1_spont_components_ori[1,:]
    
    fa1_V1_stim_norm = fa1_V1_stim/np.linalg.norm(fa1_V1_stim)
    fa1_V1_spont_norm = fa1_V1_spont/np.linalg.norm(fa1_V1_spont)

    dotprod_fa1_V1_stim_spont = np.dot(fa1_V1_stim_norm, fa1_V1_spont_norm)
    angle_fa1_V1_stim_spont = np.arccos(dotprod_fa1_V1_stim_spont) * 180/np.pi # angle in degrees
    
    angle_fa1_V1_stim_spont_all[i] = angle_fa1_V1_stim_spont
    
    fa2_V1_stim_norm = fa2_V1_stim/np.linalg.norm(fa2_V1_stim)
    fa2_V1_spont_norm = fa2_V1_spont/np.linalg.norm(fa2_V1_spont)

    dotprod_fa2_V1_stim_spont = np.dot(fa2_V1_stim_norm, fa2_V1_spont_norm)
    angle_fa2_V1_stim_spont = np.arccos(dotprod_fa2_V1_stim_spont) * 180/np.pi # angle in degrees
    
# Save outputs

np.save('/media/storage2/joana/PhD/V1-V2_analysis/FA_analysis/angle_fa1_V1_stim_spont_' + sess + '.npy', angle_fa1_V1_stim_spont_all, allow_pickle = True)    
print('SAVED')    


 