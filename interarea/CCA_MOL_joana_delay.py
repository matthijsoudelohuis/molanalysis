#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:18:37 2023

@author: joana
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.cross_decomposition import CCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

from loaddata.session_info import load_sessions
from utils.psth import compute_tensor

##################################################
session_list        = np.array([['LPE09830','2023_04_12']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,
                                    load_behaviordata=False, load_calciumdata=True, load_videodata=False, calciumversion='deconv')

#Get n neurons from V1 and from PM:
n                   = 500
V1_selec            = np.random.choice(np.where(sessions[0].celldata['roi_name']=='V1')[0],n)
PM_selec            = np.random.choice(np.where(sessions[0].celldata['roi_name']=='PM')[0],n)
sessions[0].calciumdata     = sessions[0].calciumdata.iloc[:,np.concatenate((V1_selec,PM_selec))]
sessions[0].celldata        = sessions[0].celldata.iloc[np.concatenate((V1_selec,PM_selec)),:]

##############################################################################
## Construct tensor: 3D 'matrix' of K trials by N neurons by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
[tensor,t_axis]     = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
                                 t_pre, t_post, binsize,method='interp_lin')

tensor              = tensor.transpose((1,2,0))
# [N,T,allK]          = np.shape(tensor) #get dimensions of tensor

oris        = sorted(sessions[0].trialdata['Orientation'].unique())
ori_counts  = sessions[0].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
assert(len(ori_counts) == 16 or len(ori_counts) == 8)
assert(np.all(ori_counts == 200) or np.all(ori_counts == 400))

#%% Compute residuals:
tensor_res = tensor.copy()
for ori in oris:
    ori_idx = np.where(sessions[0].trialdata['Orientation']==ori)[0]
    temp = np.mean(tensor_res[:,:,ori_idx],axis=2)
    tensor_res[:,:,ori_idx] = tensor_res[:,:,ori_idx] - np.repeat(temp[:, :, np.newaxis], len(ori_idx), axis=2)


#%%  

## split into area 1 and area 2:
idx_V1 = np.where(sessions[0].celldata['roi_name']=='V1')[0]
idx_PM = np.where(sessions[0].celldata['roi_name']=='PM')[0]

# Data format: 
    
    #  X is the source data (number of source neurons x number of time points x number of trials)
    #  Y is the target data (number of target neurons x number of time points x number of trials)

# X = tensor[idx_V1,:,:]
# Y = tensor[idx_PM,:,:]

X = tensor_res[idx_V1,:,:]
Y = tensor_res[idx_PM,:,:]

# Define neural data parameters
N1,T,K      = np.shape(X)
N2          = np.shape(Y)[0]

# Define temporal parameters
max_delay = 5
delay_step = 1
delays = np.arange(- max_delay,max_delay + 1,delay_step)
n_delays = len(delays)

time_step = 1
n_time_steps = int(T/time_step)

# Intialize variables to store CCA results

# Correlation of the first canonical pair
CCA_corr_1 = np.zeros((n_time_steps,n_delays))
CCA_corr_1[:] = np.nan

# Canonical dimensions of the first canonical pair
CCA_dim1 = np.zeros((n_time_steps,n_delays,N1))
CCA_dim2 = np.zeros((n_time_steps,n_delays,N2))

#%% Apply CCA (using Python's built in function)

for i in range(n_time_steps):
        
    for j in range(n_delays):
        
        delay = delays[j]
    
        # Define temporal window
        
        ind_init_X = i * time_step
        ind_end_X = ind_init_X + time_step
        
        ind_init_Y = i * time_step + delay
        ind_end_Y = ind_init_Y + time_step
                    
        if (ind_init_X) >=0  and (ind_init_Y) >= 0 and (ind_end_X) <= T and (ind_end_Y) <= T:
            
            # Select temporal window
            
            X_CCA = X[:,ind_init_X:ind_end_X,:]
            X_CCA = np.reshape(X_CCA, (N1, time_step * K), order = 'F').T
                            
            Y_CCA = Y[:,ind_init_Y:ind_end_Y,:]
            Y_CCA = np.reshape(Y_CCA, (N2, time_step * K), order = 'F').T
                
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
                        
            CCA_dim1[n_time_steps - i - 1,j,:] = np.reshape(a[:,0],N1)        
            CCA_dim2[n_time_steps - i - 1,j,:] = np.reshape(b[:,0],N2)
                        
#%% PLOT THE CCA MAP (1ST PAIR)

plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Arial" 

# Plotting parameters

max_delay = delays[-1]    
ind_nodelay = np.where(delays == 0)[0][0]    

early_evoked = np.where(t_axis>=0)[0][0]
late_evoked =  np.where(t_axis>=0.9)[0][0]
spontaneous =  np.where(t_axis>=-0.5)[0][0]

fig_1 = plt.figure(figsize=(15,7))

###### Plot full CCA map

fig_1.add_subplot(1,2,1)       

# xaxis = np.arange(0,2 * max_delay + 50, 50)
# xaxis_labels = np.arange(- max_delay,max_delay + 50,50)

xaxis = np.arange(0,max_delay*2)
xaxis_labels = np.round(np.arange(-max_delay,max_delay) * np.mean(np.diff(t_axis)),1)

plot = plt.imshow(CCA_corr_1[:,:],cmap = 'viridis', aspect='auto') 
plt.xticks(xaxis,xaxis_labels,rotation = 45)
plt.yticks(np.arange(len(t_axis)),np.round(np.flip(t_axis),1),rotation = 45)
plt.xlabel('Delay')
plt.ylabel('Time from trial onset')  
cbar = fig_1.colorbar(plot)
cbar.set_label('Population correlation (first can. pair)', rotation = 270, labelpad = 30)

ax = plt.gca() 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   

early_ind = int(n_time_steps - early_evoked/time_step)
late_ind = int(n_time_steps - late_evoked/time_step)
spont_ind = int(n_time_steps - spontaneous/time_step)

plt.axhline(y=early_ind,color='darkorange')
plt.axhline(y=late_ind,color='forestgreen')
plt.axhline(y=spont_ind,color='purple')
      

##### Plot CCA values as a function of delay at fixed time points

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


X = tensor_res[idx_V1,:,:]
Y = tensor_res[idx_PM,:,:]

X_CCA = np.reshape(X,(N1,T*K))
Y_CCA = np.reshape(Y,(N2,T*K))

cca = CCA(n_components = 2,scale = False, max_iter = 1000)
cca.fit(X_CCA,Y_CCA)

X_CCA = X[:,:,:]
X_CCA = np.reshape(X_CCA, (N1, T * K), order = 'F').T

Y_CCA = Y[:,:,:]
Y_CCA = np.reshape(Y_CCA, (N2, T * K), order = 'F').T

cca = CCA(n_components = 2,scale = False, max_iter = 1000)
cca.fit(X_CCA,Y_CCA)

X_c, Y_c = cca.transform(X_CCA,Y_CCA)

# Compute and store canonical correlations for the first pair
corr_1 = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
corr_2 = np.corrcoef(X_c[:,1],Y_c[:,1], rowvar = False)[0,1]

# Compute and store canonical dimensions for the second pair
CCA_weights_V1 = cca.x_weights_
CCA_weights_PM = cca.y_weights_   

plt.figure(figsize=(8,6))
# sns.barplot(x=np.arange(N1),y=sorted(CCA_dim1),color='k')
x = np.arange(N1)
y = np.array(sorted(CCA_weights_V1[:,0]))
plt.plot(x, y, lw=2,color='k')
plt.fill_between(x, 0, y, where=y>0, interpolate=True,color='k')
plt.fill_between(x, 0, y, where=y<0, interpolate=True,color='k')
plt.xticks([0,N1/2,N1])
plt.ylabel('Weights V1 CCA Dim 1')

plt.figure(figsize=(8,6))
sns.scatterplot(x=CCA_weights_V1[:,0],y=CCA_weights_V1[:,1])
plt.xlabel('Weights V1 CCA Dim 1')
plt.ylabel('Weights V1 CCA Dim 2')

sessions[0].celldata['CCA_weights_dim1'] = np.concatenate((CCA_weights_V1[:,0],CCA_weights_PM[:,0]),axis=0)
sessions[0].celldata['CCA_weights_dim2'] = np.concatenate((CCA_weights_V1[:,1],CCA_weights_PM[:,1]),axis=0)

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,10))

dfV1 = sessions[0].celldata.loc[sessions[0].celldata['roi_name'] == 'V1']
dfPM = sessions[0].celldata.loc[sessions[0].celldata['roi_name'] == 'PM']

dfV1['CCA_weights_dim1'] = np.abs(dfV1['CCA_weights_dim1'])
dfV1['CCA_weights_dim2'] = np.abs(dfV1['CCA_weights_dim1'])

sns.barplot(ax = ax1,data=dfV1,x='redcell',y='CCA_weights_dim1',hue='redcell',palette='Pastel2')
ax1.set_ylabel('absolute CCA weights dim 1')
ax1.set_title('V1')
ax1.set_xlabel('')

sns.barplot(ax = ax2,data=dfPM,x='redcell',y='CCA_weights_dim1',hue='redcell',palette='Pastel2')
ax2.set_ylabel('absolute CCA weights dim 1')
ax2.set_title('PM')
ax3.set_xlabel('')

sns.barplot(ax = ax3,data=dfV1,x='redcell',y='CCA_weights_dim2',hue='redcell',palette='Pastel2')
ax3.set_xlabel('tdTomato labeled')
ax3.set_ylabel('absolute CCA weights dim 2')

sns.barplot(ax = ax4,data=dfPM,x='redcell',y='CCA_weights_dim2',hue='redcell',palette='Pastel2')
ax4.set_xlabel('tdTomato labeled')
ax4.set_ylabel('absolute CCA weights dim 2')








