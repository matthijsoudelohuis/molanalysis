# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
# from scipy.stats import binned_statistic
from loaddata.session_info import filter_sessions,load_sessions
from natsort import natsorted 
from scipy import ndimage 

from preprocessing.preprocesslib import proc_RF, align_timestamps


clusterthr = 10
# this is sum of negative log p values of clustered RF responses
# e.g. array = [0.001,0.0001,0.05,0.05,0.02] #pvalues of cluster of RF on or off responses
# np.sum(-np.log10(array))

def find_clusters(array,clusterthr=10):
    array_p = array<0.05
    labelling, label_count = ndimage.label(array_p == True) #find clusters of significance
    cluster_p = np.zeros(label_count)
    for k in range(label_count): #loop over clusters and compute summed significance (negative log)
        cluster_p[k] = np.sum(-np.log10(array[labelling==(k+1)]))
    x = y = np.nan
    ones = np.ones_like(array, dtype=int)
    if np.sum(cluster_p>clusterthr)==1 and np.max(cluster_p)>clusterthr: #if only one cluster above clusterthr then store:
        y,x = ndimage.center_of_mass(ones, labels=labelling, index=np.argmax(cluster_p)+1)
        
    return x,y

rawdatadir          = 'W:\\Users\\Matthijs\\Rawdata\\'

# animal_ids          = ['LPE09665','LPE09830','NSH07422','NSH07429'] #If empty than all animals in folder will be processed
# animal_ids          = [] #If empty than all animals in folder will be processed
# date_filter         = []

animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed
date_filter         = ['2023_04_10']

vec_elevation       = [] #
vec_azimuth         = [] #

## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = [f.name for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]

for animal_id in animal_ids: #for each animal

    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 
    
    if any(date_filter): #If dates specified, then process only those:
        sessiondates = [x for x in sessiondates if x in date_filter]

    for sessiondate in sessiondates: #for each of the sessions for this animal
        
        sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)

        sessiondata = pd.DataFrame({'protocol': ['RF']})
        sessiondata['animal_id'] = 'LPE09830'
        sessiondata['sessiondate'] = '2023_04_10'

        suite2p_folder = os.path.join(sesfolder,"suite2p")

        plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
        # load ops of plane0:
        ops                = np.load(os.path.join(plane_folders[0], 'ops.npy'), allow_pickle=True).item()

        ## Get trigger data to align timestamps:
        filenames         = os.listdir(os.path.join(sesfolder,'RF','Behavior'))
        triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
        triggerdata       = pd.read_csv(os.path.join(sesfolder,'RF','Behavior',triggerdata_file[0]),skiprows=2).to_numpy()


        [ts_master, protocol_frame_idx_master] = align_timestamps(sessiondata, ops, triggerdata)

        RF_on_azim      = np.array([])
        RF_on_elev      = np.array([])
        RF_off_azim     = np.array([])
        RF_off_elev     = np.array([]) 

        # for iplane,plane_folder in enumerate(plane_folders):
        for iplane,plane_folder in enumerate(plane_folders[:1]):
            print('processing plane %s / %s' % (iplane+1,ops['nplanes']))

            iscell              = np.load(os.path.join(plane_folder, 'iscell.npy'))

            [ts_plane, protocol_frame_idx_plane] = align_timestamps(sessiondata, ops, triggerdata)
            ts_plane = np.squeeze(ts_plane)

            ##################### load suite2p activity outputs:
            # F     = np.load(os.path.join(plane_folder, 'F.npy'), allow_pickle=True)
            # F     = F[:,protocol_frame_idx_plane==1].transpose()
            spks    = np.load(os.path.join(plane_folder, 'spks.npy'), allow_pickle=True)
            spks    = spks[:,protocol_frame_idx_plane==1].transpose()

            # if imaging was aborted during scanning of a volume, later planes have less frames
            # Compensate by duplicating last value
            if np.shape(spks)[0]==len(ts_master):
                pass       #do nothing, shapes match
            elif np.shape(spks)[0]==len(ts_master)-1: #copy last timestamp of array
                # F           = np.vstack((F, np.tile(F[[-1],:], 1)))
                spks        = np.vstack((spks, np.tile(spks[[-1],:], 1)))
                spks        = np.hstack((spks, np.tile(spks[[-1],:], 1)))
            else:
                print("Problem with timestamps and imaging frames")

            # Get the receptive field stimuli shown and their timestamps
            grid_array,RF_timestamps = proc_RF(rawdatadir,sessiondata)
            
            ### get parameters
            [xGrid , yGrid , nGrids] = np.shape(grid_array)

            N               = spks.shape[1]

            t_resp_start    = 0        #pre s
            t_resp_stop     = 0.5      #post s
            t_base_start    = -2       #pre s
            t_base_stop     = 0        #post s

            rfmaps_on        = np.empty([xGrid,yGrid,N])
            rfmaps_off       = np.empty([xGrid,yGrid,N])

            rfmaps_on_p      = np.empty([xGrid,yGrid,N])
            rfmaps_off_p     = np.empty([xGrid,yGrid,N])

            for n in range(N):
                print(f"\rComputing RF for neuron {n+1} / {N}",end='\r')
                resps = np.empty(nGrids)
                for g in range(nGrids):

                    temp = np.logical_and(ts_plane > RF_timestamps[g]+t_resp_start,ts_plane < RF_timestamps[g]+t_resp_stop)
                    resp = spks[temp,n].mean()
                    temp = np.logical_and(ts_plane > RF_timestamps[g]+t_base_start,ts_plane < RF_timestamps[g]+t_base_stop)
                    base = spks[temp,n].mean()
                
                    resps[g] = np.max([resp-base,0])
                    # resps[g] = resp-base

                for i in range(xGrid):
                    for j in range(yGrid):
                        rfmaps_on[i,j,n] = np.mean(resps[grid_array[i,j,:]==1])
                        rfmaps_off[i,j,n] = np.mean(resps[grid_array[i,j,:]==-1])
                        
                        rfmaps_on_p[i,j,n] = st.ttest_ind(resps[grid_array[i,j,:]==1],resps[grid_array[i,j,:] == 0])[1]
                        rfmaps_off_p[i,j,n] = st.ttest_ind(resps[grid_array[i,j,:]==-1],resps[grid_array[i,j,:] == 0])[1]

            plane_rfcenter_on_x      = np.empty(N)
            plane_rfcenter_on_y      = np.empty(N)
            plane_rfcenter_off_x      = np.empty(N)
            plane_rfcenter_off_y      = np.empty(N)

            for n in range(N):
                plane_rfcenter_on_x[n],plane_rfcenter_on_y[n] = find_clusters(rfmaps_on_p[:,:,n],clusterthr=clusterthr)
                plane_rfcenter_off_x[n],plane_rfcenter_off_y[n] = find_clusters(rfmaps_off_p[:,:,n],clusterthr=clusterthr)

            RF_on_azim      = np.vstack((RF_on_azim,plane_rfcenter_on_x))
            RF_on_elev      = np.array([])
            RF_off_azim     = np.array([])
            RF_off_elev     = np.array([]) 







example_cells = np.where(iscell==1)[0][50:75]

Tot         = len(example_cells)*2
Rows        = int(np.floor(np.sqrt(Tot)))
Cols        = Tot // Rows # Compute Rows required
if Tot % Rows != 0: #If one additional row is necessary -> add one:
    Cols += 1
Position = range(1,Tot + 1) # Create a Position index

fig = plt.figure(figsize=[18, 9])
for i,n in enumerate(example_cells):
    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[i*2])
    ax.imshow(-np.log10(rfmaps_on_p[:,:,n]),cmap='Reds',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
    ax.scatter(rfcenter_on_x[n],rfcenter_on_y[n],marker='+',c='w',s=10)
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title("%d,ON" % n)
    
    ax = fig.add_subplot(Rows,Cols,Position[i*2 + 1])
    ax.imshow(-np.log10(rfmaps_off_p[:,:,n]),cmap='Blues',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
    ax.scatter(rfcenter_off_x[n],rfcenter_off_y[n],marker='+',c='w',s=10)

    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title("%d,OFF" % n)
  
plt.tight_layout(rect=[0, 0, 1, 1])



# ### get parameters
# [xGrid , yGrid , nGrids] = np.shape(grid_array)

# # N               = celldata.shape[0]

# t_resp_start     = 0        #pre s
# t_resp_stop      = 0.3        #post s
# t_base_start     = -2       #pre s
# t_base_stop      = 0        #post s

# rfmaps          = np.zeros([xGrid,yGrid,N])

# ### Compute RF maps: (method 1)
# for n in range(N):
#     print(f"\rComputing RF for neuron {n+1} / {N}")

#     for g in range(nGrids):
#         temp = np.logical_and(ts_F > RF_timestamps[g]+t_resp_start,ts_F < RF_timestamps[g]+t_resp_stop)
#         resp = calciumdata.iloc[temp,n].mean()
#         temp = np.logical_and(ts_F > RF_timestamps[g]+t_base_start,ts_F < RF_timestamps[g]+t_base_stop)
#         base = calciumdata.iloc[temp,n].mean()
        
#         # rfmaps[:,:,n] = rfmaps[:,:,n] + (resp-base) * grid_array[:,:,g]
#         rfmaps[:,:,n] = np.nansum(np.dstack((rfmaps[:,:,n],np.max([resp-base,0]) * grid_array[:,:,g])),2)
#         # rfmaps[:,:,n] = np.nansum(np.dstack((rfmaps[:,:,n],np.max([resp-base,0]) * grid_array[:,:,g])),2)


#### Zscored version:
rfmaps_z          = np.zeros([xGrid,yGrid,N])

for n in range(N):
    print(f"\rZscoring RF for neuron {n+1} / {N}")
    rfmaps_z[:,:,n] = st.zscore(rfmaps[:,:,n],axis=None)

## Show example cell RF maps:
example_cells = [0,24,285,335,377,496,417,551,430,543,696,689,617,612,924] #V1
example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

example_cells = range(900,1000)

example_cells = range(0,100) 

Tot         = len(example_cells)
Rows        = int(np.floor(np.sqrt(Tot)))
Cols        = Tot // Rows # Compute Rows required
if Tot % Rows != 0: #If one additional row is necessary -> add one:
    Cols += 1
Position = range(1,Tot + 1) # Create a Position index

fig = plt.figure(figsize=[18, 9])
for i,n in enumerate(example_cells):
    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[i])
    ax.imshow(rfmaps[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps[:,:,n])),vmax=np.max(abs(rfmaps[:,:,n])))
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title(n)
  
plt.tight_layout(rect=[0, 0, 1, 1])


#### 
fig, axes = plt.subplots(7, 13, figsize=[17, 8])
for i in range(np.shape(axes)[0]):
    for j in range(np.shape(axes)[1]):
        n = i*np.shape(axes)[1] + j
        ax = axes[i,j]
        # ax.imshow(rfmaps[:,:,n],cmap='gray')
        ax.imshow(rfmaps[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps[:,:,n])),vmax=np.max(abs(rfmaps[:,:,n])))
        ax.set_axis_off()
        ax.set_aspect('auto')
        ax.set_title(n)

### 
plt.close('all')

####################### Population Receptive Field

depths,ind  = np.unique(celldata['depth'], return_index=True)
depths      = depths[np.argsort(ind)]
areas       = ['V1','V1','V1','V1','PM','PM','PM','PM']

Rows        = 2
Cols        = 4 
Position    = range(1,8 + 1) # Create a Position index

# fig, axes = plt.subplots(2, 4, figsize=[17, 8])
fig = plt.figure()

for iplane,depth in enumerate(depths):
    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[iplane])
    idx = celldata['depth']==depth
    # popmap = np.nanmean(abs(rfmaps_z[:,:,idx]),axis=2)
    popmap = np.nanmean(abs(rfmaps[:,:,idx]),axis=2)
    ax.imshow(popmap,cmap='OrRd')
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title(areas[iplane])
    
plt.tight_layout(rect=[0, 0, 1, 1])

#######################################
### Compute RF maps: (method 2)

rfmaps_on        = np.empty([xGrid,yGrid,N])
rfmaps_off       = np.empty([xGrid,yGrid,N])

rfmaps_on_p      = np.empty([xGrid,yGrid,N])
rfmaps_off_p     = np.empty([xGrid,yGrid,N])


for n in range(N):
    print(f"\rComputing RF for neuron {n+1} / {N}")
    
    resps = np.empty(nGrids)
    for g in range(nGrids):

        temp = np.logical_and(ts_F > RF_timestamps[g]+t_resp_start,ts_F < RF_timestamps[g]+t_resp_stop)
        resp = calciumdata.iloc[temp,n].mean()
        temp = np.logical_and(ts_F > RF_timestamps[g]+t_base_start,ts_F < RF_timestamps[g]+t_base_stop)
        base = calciumdata.iloc[temp,n].mean()
    
        # resps[g] = np.max([resp-base,0])
        resps[g] = resp-base

    # temp_resps = np.empty([xGrid,yGrid,50])

    for i in range(xGrid):
        for j in range(yGrid):
            rfmaps_on[i,j,n] = np.mean(resps[grid_array[i,j,:]==1])
            rfmaps_off[i,j,n] = np.mean(resps[grid_array[i,j,:]==-1])
            
            
            rfmaps_on_p[i,j,n] = st.ttest_ind(resps[grid_array[i,j,:]==1],resps[grid_array[i,j,:] == 0])[1]
            rfmaps_off_p[i,j,n] = st.ttest_ind(resps[grid_array[i,j,:]==-1],resps[grid_array[i,j,:] == 0])[1]
                    

            # rfmaps_on_p[i,j,n] = sum(rfmaps_on[i,j,n] > resps[grid_array[i,j,:]==0]) / sum(grid_array[i,j,:]==0)
            # rfmaps_off_p[i,j,n] = sum(rfmaps_off[i,j,n] > resps[grid_array[i,j,:]==0]) / sum(grid_array[i,j,:]==0)
                    


print("Black squares: mean %2.1f +- %2.1f" % (np.mean(np.sum(grid_array[:,:,:]==1,axis=2).flatten()),
                                              np.std(np.sum(grid_array[:,:,:]==-1,axis=2).flatten())))
print("White squares: mean %2.1f +- %2.1f\n" % (np.mean(np.sum(grid_array[:,:,:]==1,axis=2).flatten()),
                                                np.std(np.sum(grid_array[:,:,:]==1,axis=2).flatten())))  


# rfmaps_on_p = 1 - rfmaps_on_p
# rfmaps_off_p = 1 - rfmaps_off_p

## Show example cell RF maps:
# example_cells = [0,24,285,335,377,496,417,551,430,543,696,689,617,612,924] #V1
# example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

# example_cells = range(900,1000)

# example_cells = [0,9,17,18,24,27,29,42,44,45,54,56,57,69,72,82,83,89,90,94,96,98] #V1
# example_cells = [1250,1257,1414,1415,1417,1423,1551,1559,2006,1925,1972,2178,1666] #PM

# example_cells = range(0,20)

example_cells = np.where(iscell==1)[0][50:75]

Tot         = len(example_cells)*2
Rows        = int(np.floor(np.sqrt(Tot)))
Cols        = Tot // Rows # Compute Rows required
if Tot % Rows != 0: #If one additional row is necessary -> add one:
    Cols += 1
Position = range(1,Tot + 1) # Create a Position index

fig = plt.figure(figsize=[18, 9])
for i,n in enumerate(example_cells):
    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[i*2])
    ax.imshow(-np.log10(rfmaps_on_p[:,:,n]),cmap='Reds',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))

    # ax.imshow(rfmaps_on[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps_on[:,:,n])),vmax=np.max(abs(rfmaps_on[:,:,n])))
    # ax.imshow(rfmaps_off[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps_off[:,:,n])),vmax=np.max(abs(rfmaps_off[:,:,n])))
    
    # ax.imshow(-np.log10(rfmaps_on_p[:,:,n]),cmap='Reds',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
    # ax.imshow(-np.log10(rfmaps_off_p[:,:,n]),cmap='Blues',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))

    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title("%d,ON" % n)
    
    ax = fig.add_subplot(Rows,Cols,Position[i*2 + 1])
        # ax.imshow(-np.log10(rfmaps_off_p[:,:,n]),cmap='Blues',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
    ax.imshow(rfmaps_off_p[:,:,n]<0.001,cmap='Blues',vmin=0,vmax=1)

    # img = np.dstack((rfmaps_on_p[:,:,n],np.ones(np.shape(rfmaps_off_p[:,:,n])),rfmaps_off_p[:,:,n]))
    # ax.imshow(-np.log10(img),vmin=-np.log10(0.05),vmax=-np.log10(0.00001))

    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title("%d,OFF" % n)
  
plt.tight_layout(rect=[0, 0, 1, 1])

## POP map
depths,ind  = np.unique(celldata['depth'], return_index=True)
depths      = depths[np.argsort(ind)]
areas       = ['V1','V1','V1','V1','PM','PM','PM','PM']

Rows        = 2
Cols        = 4 
Position    = range(1,8 + 1) # Create a Position index

# fig, axes = plt.subplots(2, 4, figsize=[17, 8])
fig = plt.figure(figsize=[9, 3])

for iplane,depth in enumerate(depths):
    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[iplane])
    idx = celldata['depth']==depth
    
    # popmap = np.sum(np.logical_or(rfmaps_on_p[:,:,idx] <0.001, rfmaps_off_p[:,:,idx] < 0.001),axis=2) / np.sum(idx)
    popmap = np.sum(np.logical_or(rfmaps_on_p[:,:,idx] <0.01, rfmaps_off_p[:,:,idx] < 0.01),axis=2) / np.sum(idx)
    IM = ax.imshow(popmap,cmap='PuRd',vmin=0,vmax=0.25)
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.set_title(areas[iplane])
    fig.colorbar(IM, ax=ax)

    
plt.tight_layout(rect=[0, 0, 1, 1])


## old code to find optimal response window size:
n = 24
n = 0
t_base_start     = -0.5     #pre s
t_base_stop      = 0        #post s

fig, axes = plt.subplots(4, 4, figsize=[17, 8], sharey='row')

t_starts = np.array([0, 0.1, 0.2, 0.3])
t_stops = np.array([0.4,0.5,0.6,0.7])
  
# for i,t_resp_start in np.array([0, 0.1, 0.2, 0.3]):
for i,t_resp_start in enumerate(t_starts):
    for j,t_resp_stop in enumerate(t_stops):
        rfmap = np.zeros([xGrid,yGrid])

        for g in range(nGrids):
            temp = np.logical_and(ts_F >= RF_timestamps[g]+t_resp_start,ts_F <= RF_timestamps[g]+t_resp_stop)
            resp = calciumdata.iloc[temp,n].mean()
            temp = np.logical_and(ts_F >= RF_timestamps[g]+t_base_start,ts_F <= RF_timestamps[g]+t_base_stop)
            base = calciumdata.iloc[temp,n].mean()

            # base = 0

            # rfmap = np.nansum(rfmap,np.max([resp-base,0]) * grid_array[:,:,g])
            rfmap = np.nansum(np.dstack((rfmap,np.max([resp-base,0]) * grid_array[:,:,g])),2)
            # rfmap = rfmap + (resp-base) * grid_array[:,:,g]

        ax = axes[i,j]
        ax.imshow(rfmap,cmap='gray',vmin=-np.max(abs(rfmap)),vmax=np.max(abs(rfmap)))
        # ax.imshow(rfmap,cmap='gray',vmin=-30000,vmax=30000)

