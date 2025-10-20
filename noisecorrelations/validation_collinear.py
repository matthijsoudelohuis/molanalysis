# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic,binned_statistic_2d

from loaddata.session_info import filter_sessions,load_sessions
from utils.corr_lib import *
from utils.tuning import compute_tuning_wrapper,ori_remapping
from preprocessing.preprocesslib import assign_layer,assign_layer2

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PairwiseCorrelations\\Collinear\\')

#%% #############################################################################
session_list        = np.array([['LPE09665_2023_03_21',
                                 'LPE10919_2023_11_06']])

sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list)

#%%  Load data properly:      
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

#%%
sessions = ori_remapping(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% 
for ises in range(len(sessions)):
    dpref = np.subtract.outer(sessions[ises].celldata['pref_ori'].to_numpy(),sessions[ises].celldata['pref_ori'].to_numpy())
    sessions[ises].delta_pref = np.min(np.array([np.abs(dpref+360),np.abs(dpref-0),np.abs(dpref-360)]),axis=0)

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%%  #assign arealayerlabel
for ises in range(nSessions):   
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 
    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 

#%%
from sklearn.decomposition import FactorAnalysis as FA

areas = ['V1','PM']
n_components = 20
fa = FA(n_components=n_components)

# comps = np.array([0,1,2,3,4,5,6,7,8,9])
comps = np.arange(1,n_components)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Computing noise correlations'):
    
    [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
    if sessions[ises].sessiondata['protocol'][0]=='GR':
        resp_meanori,respmat_res        = mean_resp_gr(sessions[ises])
    elif sessions[ises].sessiondata['protocol'][0]=='GN':
        resp_meanori,respmat_res        = mean_resp_gn(sessions[ises])

    stims        = np.sort(sessions[ises].trialdata['stimCond'].unique())
    trial_stim   = sessions[ises].trialdata['stimCond']
    noise_corr  = np.empty((N,N,len(stims)))  
    noise_cov   = np.empty((N,N,len(stims))) 
    # nc_behavout = np.empty((N,N,len(stims)))  

    for i,stim in enumerate(stims):
        data                = zscore(respmat_res[:,trial_stim==stim],axis=1)

        noise_corr[:,:,i]   = np.corrcoef(data)

        for iarea,area in enumerate(areas):
            idx_N               = ses.celldata['roi_name']==area

            fa.fit(data[idx_N,:].T)
            data_T              = fa.transform(data[idx_N,:].T)
            data[idx_N,:]       = np.dot(data_T[:,comps], fa.components_[comps,:]).T        # Reconstruct data
        
        noise_cov[:,:,i]  = np.cov(data)

    sessions[ises].noise_corr       = np.mean(noise_corr,axis=2)
    sessions[ises].noise_cov        = np.mean(noise_cov,axis=2)
    # sessions[ises].nc_behavout      = np.mean(nc_behavout,axis=2)

#%% #########################################################################################

#%% Loop over all delta preferred orientations and store mean correlations as well as distribution of pos and neg correlations:
# areapairs           = ['V1-PM','PM-V1']
# layerpairs          = ' '
# layerpairs          = ['L2/3-L2/3', 'L2/3-L5']'
# projpairs           = 'unl-unl'

areapair           ='V1-PM'
# areapair           ='V1-V1'
layerpair         = ' '
# projpair          = 'unl-unl'
projpair          = ' '

ises                = 1
centerori           = 0
deltaori            = 0
rotate_prefori      = True

rf_type             = 'F'
rf_type             = 'Fsmooth'
corr_type           = 'noise_corr'
# corr_type           = 'noise_cov'

noise_thr           = 100
r2_thr              = 0.1
binresolution       = 5
tuned_thr           = 75
min_dist            = 15
binlim              = 60

#Binning parameters 2D:
binedges_2d     = np.arange(-binlim,binlim,binresolution)+binresolution/2 
bincenters_2d   = binedges_2d[:-1]+binresolution/2 
nBins           = len(bincenters_2d)

celldata        = copy.deepcopy(sessions[ises].celldata)
corrdata        = getattr(sessions[ises],corr_type).copy()

el              = celldata['rf_el_' + rf_type].to_numpy()
az              = celldata['rf_az_' + rf_type].to_numpy()

delta_el        = el[:,None] - el[None,:]
delta_az        = az[:,None] - az[None,:]

delta_rf        = np.sqrt(delta_az**2 + delta_el**2)

# plt.hist(el.flatten(),bins=np.linspace(-100,100,100))
# plt.hist(az.flatten(),bins=np.linspace(-100,100,100))

# Careful definitions:
# delta_az is source neurons azimuth minus target neurons azimuth position:
# plt.imshow(delta_az[:10,:10],vmin=-20,vmax=20,cmap='bwr')
# entry delta_az[0,1] being positive means target neuron RF is to the right of source neuron
# entry delta_el[0,1] being positive means target neuron RF is above source neuron
# To rotate azimuth and elevation to relative to the preferred orientation of the source neuron
# means that for a neuron with preferred orientation 45 deg all delta az and delta el of paired neruons
# will rotate 45 deg, such that now delta azimuth and delta elevation is relative to the angle 
# of pref ori of the source neuron

# Rotate delta azimuth and delta elevation to the pref ori of the source neuron
# delta_az is source neurons
if rotate_prefori: 
    for iN in range(len(celldata)):
        ori_rots            = 360 - np.tile(celldata['pref_ori'][iN],len(celldata))
        angle_vec           = np.vstack((delta_el[iN,:], delta_az[iN,:]))
        angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots) 
        delta_el[iN,:]      = angle_vec_rot[0,:]
        delta_az[iN,:]      = angle_vec_rot[1,:]

    delta_rf         = np.sqrt(delta_az**2 + delta_el**2)
    # angle_rf         = np.mod(np.arctan2(delta_az,delta_el)+np.pi/2,np.pi*2)
    # angle_rf         = np.mod(angle_rf+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
    # plt.hist(angle_rf.flatten())

# plt.scatter(angle_rf_b[celldata['pref_ori']==90,:].flatten(),angle_rf[celldata['pref_ori']==90,:].flatten())

rffilter        = np.meshgrid(celldata['rf_r2_' + rf_type]> r2_thr,celldata['rf_r2_'  + rf_type] > r2_thr)
rffilter        = np.logical_and(rffilter[0],rffilter[1])

signalfilter    = np.meshgrid(celldata['noise_level']<noise_thr,celldata['noise_level']<noise_thr)
signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

if tuned_thr:
    if tuned_thr<1:
        tuningfilter    = np.meshgrid(celldata['tuning_var']>tuned_thr,celldata['tuning_var']>tuned_thr)
    elif tuned_thr>1:
        tuningfilter    = np.meshgrid(celldata['gOSI']>np.percentile(celldata['gOSI'],100-tuned_thr),
                                        celldata['gOSI']>np.percentile(celldata['gOSI'],100-tuned_thr))
    tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
else: 
    tuningfilter    = np.ones(np.shape(rffilter))

centerorifilter = np.tile(celldata['pref_ori']== centerori,(len(celldata),1)).T

nanfilter       = np.all((~np.isnan(corrdata),~np.isnan(delta_rf)),axis=0)

proxfilter      = ~(sessions[ises].distmat_xy<min_dist)

if deltaori is not None:
    if isinstance(deltaori,(float,int)):
        deltaori = np.array([deltaori,deltaori])
    if np.shape(deltaori) == (1,):
        deltaori = np.tile(deltaori,2)
    assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
    delta_pref = sessions[ises].delta_pref.copy()
    # delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
    # delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
    deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
                            delta_pref <= deltaori[1]),axis=0)
else:
    deltaorifilter = np.ones(np.shape(rffilter)).astype(bool)

areafilter      = filter_2d_areapair(sessions[ises],areapair)

layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

projfilter      = filter_2d_projpair(sessions[ises],projpair)
#Combine all filters into a single filter:
cellfilter      = np.all((rffilter,signalfilter,tuningfilter,areafilter,
                    layerfilter,projfilter,proxfilter,nanfilter,centerorifilter,
                    deltaorifilter),axis=0)
# valuedata are the correlation values, these are going to be binned
vdata               = corrdata[cellfilter].flatten()
#First 2D binning: x is elevation, y is azimuth, 
xdata               = delta_el[cellfilter].flatten()
ydata               = delta_az[cellfilter].flatten()

bin_2d   = binned_statistic_2d(x=xdata, y=ydata, values=vdata,bins=binedges_2d, statistic='sum')[0]

# Count how many correlation observations are in each bin:
bin_2d_count  = np.histogram2d(x=xdata,y=ydata,bins=binedges_2d)[0]

bin_2d = bin_2d / bin_2d_count


fig,axes = plt.subplots(2,1,figsize =(4,8))
ax=axes[0]
ax.scatter(ydata,xdata,s=5,c=vdata,cmap='magma',vmin=-0.1,vmax=0.25)
# ax.scatter(xdata,ydata,s=5,c=vdata,cmap='magma',vmin=-0.1,vmax=0.25)
ax.set_xlim(-binlim,binlim)
ax.set_ylim(-binlim,binlim)
ax.set_xlabel('Azimuth')
ax.set_ylabel('Elevation')
ax.set_title('%s - %d deg' % (areapair,centerori))

ax=axes[1]
delta_az,delta_el   = np.meshgrid(bincenters_2d,bincenters_2d)

ax.pcolor(delta_az,delta_el,bin_2d,vmin=np.nanpercentile(bin_2d,5),
          vmax=np.nanpercentile(bin_2d,95),cmap='magma')
ax.set_facecolor('grey')

filename = '%s_%ddeg_%s_rotated%d_%s' % (areapair,centerori,corr_type,rotate_prefori,
                                         sessions[ises].session_id)
my_savefig(fig,savedir,filename,formats=['png'])

