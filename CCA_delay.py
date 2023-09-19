#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:18:37 2023

@author: joana
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.cross_decomposition import CCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%  

# Data format: 
    
    #  X is the source data (number of source neurons x number of time points x number of trials)
    #  Y is the target data (number of target neurons x number of time points x number of trials)

# Define neural data parameters

n_neurons_1 = 159
n_neurons_2 = 24
n_steps = 1000
n_trials = 3200

# Define temporal parameters

max_delay = 100
delay_step = 1
delays = np.arange(- max_delay,max_delay + 1,delay_step)
n_delays = len(delays)

time_step = 10
n_time_steps = int(n_steps/time_step)

# Intialize variables to store CCA results

    # Correlation of the first canonical pair
CCA_corr_1 = np.zeros((n_time_steps,n_delays))
CCA_corr_1[:] = np.nan

    # Canonical dimensions of the first canonical pair
CCA_dim1 = np.zeros((n_time_steps,n_delays,n_neurons_1))
CCA_dim2 = np.zeros((n_time_steps,n_delays,n_neurons_2))

#%% Apply CCA (using Python's built in function)

for i in range(n_time_steps):
        
    for j in range(n_delays):
        
        delay = delays[j]
    
        # Define temporal window
        
        ind_init_X = i * time_step
        ind_end_X = ind_init_X + time_step
        
        ind_init_Y = i * time_step + delay
        ind_end_Y = ind_init_Y + time_step
                    
        if (ind_init_X) >=0  and (ind_init_Y) >= 0 and (ind_end_X) <= n_steps and (ind_end_Y) <= n_steps:
            
            # Select temporal window
            
            X_CCA = X[:,ind_init_X:ind_end_X,:]
            X_CCA = np.reshape(X_CCA, (n_neurons_1, time_step * n_trials), order = 'F').T
                            
            Y_CCA = Y[:,ind_init_Y:ind_end_Y,:]
            Y_CCA = np.reshape(Y_CCA, (n_neurons_2, time_step * n_trials), order = 'F').T
                
            # Apply CCA
            
            cca = CCA(n_components = 1,scale = False, max_iter = 1000)
            cca.fit(X_CCA,Y_CCA)
            
            X_c, Y_c = cca.transform(X_CCA,Y_CCA)
            
            # Compute and store canonical correlations for the first pair
            
            corr_1 = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
            
            CCA_corr_1[n_time_steps - i - 1,j] = corr_1
            
            # Compute and store canonical dimensions for the second pair
            
            a = cca.x_weights_
            b = cca.y_weights_   
                        
            CCA_dim1[n_time_steps - i - 1,j,:] = np.reshape(a[:,0],n_neurons_1)        
            CCA_dim2[n_time_steps - i - 1,j,:] = np.reshape(b[:,0],n_neurons_2)
                        
#%% PLOT THE CCA MAP (1ST PAIR)

plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Arial" 

# Plotting parameters

max_delay = delays[-1]    
ind_nodelay = np.where(delays == 0)[0][0]    

early_evoked = 140
late_evoked = 400
spontaneous = 800

fig_1 = plt.figure()

    # Plot full CCA map

fig_1.add_subplot(1,2,1)       
            
xaxis = np.arange(0,2 * max_delay + 50, 50)
xaxis_labels = np.arange(- max_delay,max_delay + 50,50)
  
plot = plt.imshow(CCA_corr_1[:,:],cmap = 'viridis', aspect='auto') 
plt.xticks(xaxis,xaxis_labels)
plt.yticks([])
plt.xlabel('Delay')
plt.ylabel('Time from trial onset')  
cbar = fig_1.colorbar(plot)
cbar.set_label('Population correlation (first can. pair)', rotation = 270, labelpad = 30)

ax = plt.gca() 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   

early_ind = np.int(n_time_steps - early_evoked/time_step)
late_ind = np.int(n_time_steps - late_evoked/time_step)
spont_ind = np.int(n_time_steps - spontaneous/time_step)

plt.axhline(y=early_ind,color='darkorange')
plt.axhline(y=late_ind,color='forestgreen')
plt.axhline(y=spont_ind,color='purple')
      
    # Plot CCA values as a function of delay at fixed time points

fig_1.add_subplot(1,2,2)   

ax = plt.gca()

plt.plot(CCA_corr_1[early_ind,:],label = 'early', color='darkorange')
plt.plot(CCA_corr_1[late_ind,:],label = 'late', color = 'forestgreen')
plt.plot(CCA_corr_1[spont_ind,:],label = 'spontaneous', color = 'purple')
plt.axvline(x=ind_nodelay,color = 'gray', linestyle = '--')
   
ax.yaxis.tick_right()
plt.xticks(xaxis,xaxis_labels)
ax.yaxis.set_label_position("right")
ax = plt.gca() 
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False) 
plt.xlabel('Delay')

