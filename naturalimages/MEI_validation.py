# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Imports
# Import general libs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
os.chdir('e:\\Python\\molanalysis')

# os.chdir('../')  # set working directory to the root of the git repo

# Import personal lib funcs
from loaddata.session_info import load_sessions
from utils.plot_lib import *  # get all the fixed color schemes
from utils.imagelib import load_natural_images
from loaddata.get_data_folder import get_local_drive
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.rf_lib import *
from utils.tuning import mean_resp_image

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Images\\MEI\\Validation\\')

#%% 





#%% Load IM and MV session ################################################
session_list = np.array([['LPE10885', '2023_10_20'],
                         ['LPE10885', '2023_10_20']])
# Load sessions lazy: (no calciumdata, behaviordata etc.,)
sessions, nSessions = load_sessions(protocol='IM', session_list=session_list)

#%% Load data
for ises in range(nSessions):    # Load proper data and compute average trial responses:
    sessions[ises].load_respmat(calciumversion='deconv', keepraw=False)

#%% Compute the mean response to each of the natural images: 
for ises in range(nSessions):    
    sessions[ises].respmean,sessions[ises].imageids = mean_resp_image(sessions[ises])

#%% ###########################################################
common_images = np.intersect1d(sessions[0].imageids,sessions[1].imageids)
nCommonImages = len(common_images)

ref_cell_ids = np.unique(sessions[1].celldata['ref_cell_id'][sessions[1].celldata['ref_cell_id']!=''])

nMatchingCells = len(ref_cell_ids)

data = np.empty((nMatchingCells,nCommonImages,2))

_,idx_N,_ = np.intersect1d(sessions[0].celldata['cell_id'],ref_cell_ids)
data[:,:,0] = sessions[0].respmean[np.ix_(idx_N,common_images)]

_,idx_N,_ = np.intersect1d(sessions[1].celldata['ref_cell_id'],ref_cell_ids)
data[:,:,1] = sessions[1].respmean[np.ix_(idx_N,common_images)]


#%% Plot correlation between population vector responses for each image across sessions: 

popcorrmat = np.corrcoef(data[:,:,0],data[:,:,1])


#%% Plot also the correlation to the repeated image presentation of individual neurons across sessions
neuroncorrmat = np.corrcoef(data[:,:,0].T,data[:,:,1].T)

#%% 
confusion_matrix = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(confusion_matrix)
plt.imshow(confusion_matrix,interpolation='none',cmap='magma',vmin=0,vmax=1)
cb = plt.colorbar(shrink=0.4,ticks=[0,1])
cb.set_label(label='Normalized response',labelpad=-30)
plt.xticks([0,nMEI_cells-1], [1,nMEI_cells])
plt.yticks([0,nMEI_cells-1], [1,nMEI_cells])
plt.xlabel('Cell id')
plt.ylabel('MEI id')
plt.title('Confusion matrix of MEI responses')
plt.tight_layout()




#%% Load MV session ################################################
session_list = np.array([['LPE10885', '2023_10_20']])
# Load sessions lazy: (no calciumdata, behaviordata etc.,)
sessions, nSessions = load_sessions(protocol='IM', session_list=session_list)

#%% Load data
for ises in range(nSessions):    # Load proper data and compute average trial responses:
    sessions[ises].load_respmat(calciumversion='deconv', keepraw=False)

#%% Just for debugging the script:
sessions[0].trialdata['meiNumber'] = (sessions[0].trialdata['ImageNumber']/14).astype(int)

sessions[0].trialdata.loc[sessions[0].trialdata['meiNumber']>100, 'meiNumber'] = ''
# sessions[0].trialdata.loc[sessions[0].trialdata['meiNumber']>100, 'mei_cell_id'] = ''
sessions[0].trialdata['mei_cell_id'] = ''

uMEIs = np.unique(sessions[0].trialdata['meiNumber'][sessions[0].trialdata['meiNumber']!=''])

for i in uMEIs:
    idx = sessions[0].trialdata['meiNumber']==i
    sessions[0].trialdata.loc[idx, 'mei_cell_id'] = 'LPE10885_2023_10_20_%d_%04d' % (np.random.randint(9),np.random.randint(9999))

old_mei_cell_ids    = np.unique(sessions[0].trialdata['mei_cell_id'][sessions[0].trialdata['mei_cell_id']!=''])

cell_id_map         = dict(zip(old_mei_cell_ids,np.random.choice(sessions[0].celldata['cell_id'],90)))
#!!!! get cell id map from load file preprocessing!!! 

new_mei_cell_ids = np.empty(np.shape(old_mei_cell_ids),dtype=object)

for iold,old_id in enumerate(old_mei_cell_ids):
    if old_id in cell_id_map.keys():
        new_mei_cell_ids[iold] = cell_id_map[old_id]

#Which cells exist in the new session:
idx_overlap                     = ~pd.isna(new_mei_cell_ids)
old_mei_cell_ids_overlap        = old_mei_cell_ids[idx_overlap]
new_mei_cell_ids_overlap        = new_mei_cell_ids[idx_overlap]

# nMEI_cells             = len(old_mei_cell_ids)
nMEI_cells              = len(new_mei_cell_ids_overlap)

len(np.unique(new_mei_cell_ids[new_mei_cell_ids!=None]))

#%% ###########################################################
# #Get the mean response to the MEIs and to the natural images
N           = len(sessions[0].celldata)
uMEIs = np.unique(sessions[0].trialdata['meiNumber'][sessions[0].trialdata['meiNumber']!=''])
nMEIs       = len(uMEIs)

mei_mean = np.empty((N,nMEIs))

for imei in range(nMEIs):
    mei_mean[:,imei] = np.mean(sessions[0].respmat[:,sessions[0].trialdata['meiNumber'] == uMEIs[imei]],axis=1)

uIMs       = np.unique(sessions[0].trialdata['ImageNumber'])
nIMs       = len(uIMs)

natimg_mean = np.empty((N,nIMs))
for iim in range(nIMs):
    natimg_mean[:,iim] = np.mean(sessions[0].respmat[:,sessions[0].trialdata['ImageNumber'] == uIMs[iim]],axis=1)

#%% ###########################################################
# #Get the mapping of the neurons 

# cell_id_map = dict(zip(old_mei_cell_ids,np.random.choice(sessions[0].celldata['cell_id'],90)))

###### load mapping of old cell id to new cell id
# old_cell_ids = np.load(os.path.join(savedir,'old_cell_ids.npy'))
# output something like: 
# cell_id_map = dict(zip(old_mei_cell_ids,new_mei_cell_ids))

#%% ###########################################################
# Show the confusion matrix of the MEI response for each originally intended cell
confusion_matrix        = np.zeros((nMEI_cells,nMEI_cells))

for imei,icell_id in enumerate(new_mei_cell_ids_overlap):
    for jmei,jcell_id in enumerate(new_mei_cell_ids_overlap):
        confusion_matrix[imei,jmei] = mei_mean[sessions[0].celldata['cell_id'] == icell_id,jmei]

#%% Plot confusion matrix: 

confusion_matrix = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(confusion_matrix)
plt.imshow(confusion_matrix,interpolation='none',cmap='magma',vmin=0,vmax=1)
cb = plt.colorbar(shrink=0.4,ticks=[0,1])
cb.set_label(label='Normalized response',labelpad=-30)
plt.xticks([0,nMEI_cells-1], [1,nMEI_cells])
plt.yticks([0,nMEI_cells-1], [1,nMEI_cells])
plt.xlabel('Cell id')
plt.ylabel('MEI id')
plt.title('Confusion matrix of MEI responses')
plt.tight_layout()

#%% 
percentiles = np.empty((nMEI_cells,))

for jmei in range(nMEI_cells):
    percentiles[jmei] = np.percentile(confusion_matrix[:,jmei], confusion_matrix[jmei,jmei]*100)

#%% Plot the percentile placement for different subpopulations of neurons: 

sessions[0].celldata['labeled'] = ['lab' if x==1 else 'unl' for x in sessions[0].celldata['redcell']]
sessions[0].celldata['arealabel'] = sessions[0].celldata['roi_name'] + sessions[0].celldata['labeled']
arealabels = np.unique(sessions[0].celldata['arealabel'])
arealabels = ['V1unl','V1lab','PMunl','PMlab']

clrs_arealabels = get_clr_area_labeled(arealabels)
sessions[0].celldata['mei_percentile'] = np.nan
sessions[0].celldata.loc[sessions[0].celldata['cell_id'].isin(new_mei_cell_ids_overlap),'mei_percentile'] = percentiles

fig,ax = plt.subplots(1,1,figsize=(3,3))
df = sessions[0].celldata[~np.isnan(sessions[0].celldata['mei_percentile'])]
# sns.barplot(data = df,y='mei_percentile',x='arealabel',palette=clrs_arealabels)
sns.stripplot(data = df,y='mei_percentile',x='arealabel',palette=clrs_arealabels,s=5,order=arealabels)
# sns.scatterplot(data = df,y='mei_percentile',hue='arealabel',palette=clrs_arealabels)
plt.ylabel('Response percentile')
