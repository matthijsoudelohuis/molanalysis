# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025
Locate receptive field following squared checkerboard noise for individual neurons
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
# from scipy.stats import binned_statistic
from loaddata.session_info import filter_sessions,load_sessions
from natsort import natsorted 
from scipy import ndimage 
from scipy.stats import combine_pvalues #Fisher's method to combine significance of multiple p-values

from preprocessing.preprocesslib import align_timestamps

## Mapping of RF on/off blocks to elevation and azimuth:
vec_elevation       = [-16.7,50.2] #bottom and top of screen displays
vec_azimuth         = [-135,135] #left and right of screen displays

t_resp_start        = 0.1        #pre s
t_resp_stop         = 0.6      #post s #this one is updated based on protocol of 5 or 10 degrees
t_base_start        = -2       #pre s
t_base_stop         = 0        #post s

def find_largest_cluster(array_p,minblocks=2): #filters clusters of adjacent significant blocks
    # minblocks   = minimum number of adjacent blocks with significant response

    array_p_thr = array_p<0.05
    labeling, label_count = ndimage.label(array_p_thr) #find clusters of significance
    
    for k in range(label_count): #remove all singleton clusters:
        if np.sum(labeling==(k+1))<minblocks:
            labeling[labeling == (k+1)] = 0
    
    #find the largest cluster based on histogram of nd image label indices:
    largest_cluster = []
    if np.any(labeling):
        largest_cluster = np.argmax(np.histogram(labeling,range(label_count+2))[0][1:])+1

    cluster         = np.isin(labeling,largest_cluster) #keep only largest cluster
    cluster_p       = combine_pvalues(array_p[np.isin(labeling,largest_cluster)])[1] #get combined p-value, Fisher's test

    return cluster,cluster_p

def filter_clusters(array,clusterthr=10,minblocks=2): #filters clusters of adjacent significant blocks
    # minblocks   = minimum number of adjacent blocks with significant response
    # clusterthr  =  is sum of negative log p values of clustered RF responses
    # e.g. array = [0.001,0.0001,0.05,0.05,0.02] #pvalues of cluster of RF on or off responses
    # np.sum(-np.log10(array))

    array_p = array<0.05
    labeling, label_count = ndimage.label(array_p == True) #find clusters of significance
    
    for k in range(label_count): #remove all singleton clusters:
        if np.sum(labeling==(k+1))<minblocks:
            labeling[labeling == (k+1)] = 0

    clusters_p = np.zeros(label_count)
    for k in range(label_count): #loop over clusters and compute summed significance (negative log)
        clusters_p[k] = np.sum(-np.log10(array[labeling==(k+1)]))

    cluster    = np.isin(labeling,np.argmax(clusters_p>clusterthr)+1) #keep only clusters with combined significance
    cluster_p  = clusters_p[np.argmax(clusters_p>clusterthr)] #keep only clusters with combined significance
    
    return cluster,cluster_p

def com_clusters(clusters): #computes center of mass and size of sig RF clusters
    x = y = size = np.nan
    if np.any(clusters):
        ones = np.ones_like(clusters, dtype=int)
        y,x = ndimage.center_of_mass(ones, labels=clusters, index=True)
        size = np.sum(clusters) * 5.16**2
    return x,y,size

def proc_RF(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    
    filenames       = os.listdir(sesfolder)
    
    log_file        = list(filter(lambda a: 'log' in a, filenames)) #find the trialdata file
    
    #RF_log.bin
    #The vector saved is long GridSize(1)xGridSize(2)x(RunTime/Duration)
    #where RunTime is the total display time of the Bonsai programme.
    #The file format is .binary data with int8 data format
    with open(os.path.join(sesfolder,log_file[0]) , 'rb') as fid:
        grid_array = np.fromfile(fid, np.int8)
    
    if np.mod(len(grid_array),13*52)==0:
        xGrid           = 52
        yGrid           = 13
        nGrids          = 1800
    elif np.mod(len(grid_array),7*26)==0:
        xGrid           = 26
        yGrid           = 7
        nGrids          = 1200
    else: 
        print('unknown dimensions of grid array in RF protocol')
    
    nGrids_emp = int(len(grid_array)/xGrid/yGrid)
    if nGrids_emp != nGrids:
        if np.isclose(len(grid_array)/xGrid/yGrid,nGrids,atol=1):
            nGrids          = nGrids_emp
            print('\n####### One grid too many or too few.... Correcting for it.\n')
        else:
            print('\n####### Problem with number of grids in receptive field mapping\n')

    grid_array                      = np.reshape(grid_array, [nGrids,xGrid,yGrid])
    grid_array                      = np.transpose(grid_array, [1,2,0])
    grid_array = np.rot90(grid_array, k=1, axes=(0,1))
    
    grid_array[grid_array==-1]       = 1
    grid_array[grid_array==0]       = -1
    grid_array[grid_array==-128]    = 0
    
    # fig, ax = plt.subplots(figsize=(7, 3))
    # ax.imshow(grid_array[:,:,0], aspect='auto',cmap='gray')
    # ax.imshow(grid_array[:,:,-1], aspect='auto',cmap='gray')
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file

    if not len(trialdata_file)==0 and os.path.exists(os.path.join(sesfolder,trialdata_file[0])):
        trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
        RF_timestamps   = trialdata.iloc[:,1].to_numpy()

    else: ## Get trigger data to align ts_master:
        print('Interpolating timestamps because trigger data is missing for the receptive field stimuli')
        triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
        triggerdata       = pd.read_csv(os.path.join(sesfolder,triggerdata_file[0]),skiprows=2).to_numpy()
        
        #rework from last timestamp: triggerdata[1,-1]
        RF_timestamps = np.linspace(triggerdata[-1,1]-nGrids*0.5, triggerdata[-1,1], num=nGrids, endpoint=True)
        RF_timestamps = RF_timestamps + 1.2 #specific offset for LPE09665 - 2023_03_14

    assert np.shape(grid_array)[2]==len(RF_timestamps),'number of timestamps does not match number of grids presented'

    return grid_array,RF_timestamps

def locate_rf_session(rawdatadir,animal_id,sessiondate,signals=['F','Fneu'],showFig=True):
    if isinstance(signals, str):
        signals =  [signals]

    sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)

    sessiondata = pd.DataFrame({'protocol': ['RF']})
    sessiondata['animal_id'] = animal_id
    sessiondata['sessiondate'] = sessiondate
    sessiondata['fs'] = 5.317

    suite2p_folder  = os.path.join(sesfolder,"suite2p")
    rf_folder       = os.path.join(sesfolder,'RF','Behavior')

    if os.path.exists(suite2p_folder) and os.path.exists(rf_folder): 
        plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
        # load ops of plane0:
        ops                = np.load(os.path.join(plane_folders[0], 'ops.npy'), allow_pickle=True).item()

        ## Get trigger data to align timestamps:
        filenames         = os.listdir(rf_folder)
        triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
        triggerdata       = pd.read_csv(os.path.join(rf_folder,triggerdata_file[0]),skiprows=1).to_numpy()

        [ts_master, protocol_frame_idx_master] = align_timestamps(sessiondata, ops, triggerdata)

        # Get the receptive field stimuli shown and their timestamps
        grid_array,RF_timestamps = proc_RF(rawdatadir,sessiondata)

        ### get parameters
        [xGrid , yGrid , nGrids] = np.shape(grid_array)
        
        t_resp_stop         = np.diff(RF_timestamps).mean() + 0.1
        
        # for iplane,plane_folder in enumerate(plane_folders):
        for iplane,plane_folder in enumerate(plane_folders[:1]):
            print('\n Processing plane %s / %s \n' % (iplane+1,ops['nplanes']))
            for signal in signals:

                iscell             = np.load(os.path.join(plane_folder, 'iscell.npy'))
                ops                = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()
                
                [ts_plane, protocol_frame_idx_plane] = align_timestamps(sessiondata, ops, triggerdata)
                ts_plane = np.squeeze(ts_plane) #make 1-dimensional

                ##################### load suite2p activity output:
                if signal=='Fneu':
                    sig    = np.load(os.path.join(plane_folder, 'Fneu.npy'), allow_pickle=True)
                elif signal=='spks':
                    sig     = np.load(os.path.join(plane_folder, 'spks.npy'), allow_pickle=True)
                elif signal=='F':
                    sig     = np.load(os.path.join(plane_folder, 'F.npy'), allow_pickle=True)
                elif signal=='Favg':

                    #Get locations of cells:
                    stat               = np.load(os.path.join(plane_folder, 'stat.npy'), allow_pickle=True)
                    xloc = yloc = np.zeros(len(stat))
                    for k in range(len(stat)):
                        xloc[k],yloc[k] = stat[k]['med']
                    distmatxy = np.sqrt((xloc[:,None] - xloc[None,:])**2 + (yloc[:,None] - yloc[None,:])**2)
                    #Average the activity of neurons in the same location (within 50 um):
                    Fneu    = np.load(os.path.join(plane_folder, 'Fneu.npy'), allow_pickle=True)
                    sig = Fneu.copy()
                    for iN in range(sig.shape[0]):
                        sig[iN,:] = Fneu[distmatxy[iN,:]<50,:].mean(0)

                sig    = sig[:,protocol_frame_idx_plane==1].transpose()

                # For debugging sample only first 20 neurons: 
                iscell = iscell[:100,:]
                sig = sig[:,:100]
                
                N               = sig.shape[1]

                rfmaps_on       = np.empty([xGrid,yGrid,N])
                rfmaps_off       = np.empty([xGrid,yGrid,N])

                rfmaps_on_p      = np.empty([xGrid,yGrid,N])
                rfmaps_off_p     = np.empty([xGrid,yGrid,N])

                for n in range(N):
                    print(f"\rComputing RF on {signal} for neuron {n+1} / {N}",end='\r')
                    resps = np.empty(nGrids)
                    for g in range(nGrids):

                        temp = np.logical_and(ts_plane > RF_timestamps[g]+t_resp_start,ts_plane < RF_timestamps[g]+t_resp_stop)
                        resp = sig[temp,n].mean()
                        temp = np.logical_and(ts_plane > RF_timestamps[g]+t_base_start,ts_plane < RF_timestamps[g]+t_base_stop)
                        base = sig[temp,n].mean()
                    
                        resps[g] = np.max([resp-base,0])
                        # resps[g] = resp-base

                    for i in range(xGrid):
                        for j in range(yGrid):
                            rfmaps_on[i,j,n] = np.mean(resps[grid_array[i,j,:]==1])
                            rfmaps_off[i,j,n] = np.mean(resps[grid_array[i,j,:]==-1])
                            
                            rfmaps_on_p[i,j,n] = st.ranksums(resps[grid_array[i,j,:]==1],resps[grid_array[i,j,:] == 0])[1]
                            rfmaps_off_p[i,j,n] = st.ranksums(resps[grid_array[i,j,:]==-1],resps[grid_array[i,j,:] == 0])[1]

                            # rfmaps_on_p[i,j,n] = st.ttest_ind(resps[grid_array[i,j,:]==1],resps[grid_array[i,j,:] == 0])[1]
                            # rfmaps_off_p[i,j,n] = st.ttest_ind(resps[grid_array[i,j,:]==-1],resps[grid_array[i,j,:] == 0])[1]

                RF_x            = np.empty(N)
                RF_y            = np.empty(N)
                RF_size         = np.empty(N)
                RF_p            = np.ones(N)

                rfmaps_on_p_filt  = rfmaps_on_p.copy() #this is only for visualization purposes
                rfmaps_off_p_filt = rfmaps_off_p.copy()

                for n in range(N):

                    # clusters_on     = filter_clusters(rfmaps_on_p[:,:,n],clusterthr=clusterthr,minblocks=minblocks)
                    # clusters_off    = filter_clusters(rfmaps_off_p[:,:,n],clusterthr=clusterthr,minblocks=minblocks)
                    
                    # clusters_on,clusters_on_p     = filter_clusters(rfmaps_on_p[:,:,n],clusterthr=clusterthr,minblocks=minblocks)
                    # clusters_off,clusters_off_p    = filter_clusters(rfmaps_off_p[:,:,n],clusterthr=clusterthr,minblocks=minblocks)
                    
                    clusters_on,clusters_on_p     = find_largest_cluster(rfmaps_on_p[:,:,n],minblocks=2)
                    clusters_off,clusters_off_p    = find_largest_cluster(rfmaps_off_p[:,:,n],minblocks=2)
                    
                    #temporary calc of center of mass for each separately:
                    #if far apart then ignore this RF fit:
                    RF_x_on,RF_y_on     = com_clusters(clusters_on)[:2]
                    RF_x_off,RF_y_off   = com_clusters(clusters_off)[:2]
                    deltaRF = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip((RF_x_on,RF_y_on), (RF_x_off,RF_y_off))))
                    if deltaRF > 10:
                        clusters_on = clusters_off = np.tile(False,(xGrid,yGrid))

                    rfmaps_on_p_filt[~clusters_on,n]    = 1 #set to 1 for all noncluster p values
                    rfmaps_off_p_filt[~clusters_off,n]  = 1
                    
                    clusters        = np.logical_or(clusters_on,clusters_off)
                    
                    RF_x[n],RF_y[n],RF_size[n] = com_clusters(clusters)
                    # RF_p[n] = np.nansum((clusters_on_p,clusters_off_p))
                    RF_p[n] = combine_pvalues((clusters_on_p,clusters_off_p))[1] #get combined p-value, Fisher's test

                #convert x and y values in grid space to azimuth and elevation:
                RF_azim = RF_x/yGrid * np.diff(vec_azimuth) + vec_azimuth[0]
                RF_elev = RF_y/xGrid * np.diff(vec_elevation) + vec_elevation[0]

                # np.save(os.path.join(plane_folder,'RF.npy'),np.vstack((RF_azim,RF_elev,RF_size,RF_p)))
                df = pd.DataFrame(data=np.column_stack((RF_azim,RF_elev,RF_size,RF_p)),columns=['RF_azim','RF_elev','RF_size','RF_p'])
                np.save(os.path.join(plane_folder,'RF_%s.npy' % signal),df)

                if showFig: 
                    # example_cells = np.where(np.logical_and(iscell[:,0]==1,RF_size>200))[0][:20] #get first twenty good cells
                    example_cells = np.where(np.logical_and(iscell[:,0]==1,RF_p<0.001))[0][:20] #get first twenty good cells

                    Tot         = len(example_cells)
                    if Tot: 
                        Rows        = int(np.floor(np.sqrt(Tot)))
                        Cols        = Tot // Rows # Compute Rows required
                        if Tot % Rows != 0: #If one additional row is necessary -> add one:
                            Cols += 1
                        Position = range(1,Tot + 1) # Create a Position index

                        fig = plt.figure(figsize=[18, 9])
                        for i,n in enumerate(example_cells):
                            # add every single subplot to the figure with a for loop
                            ax = fig.add_subplot(Rows,Cols,Position[i])

                            data = np.ones((xGrid,yGrid,3))

                            R = np.divide(-np.log10(rfmaps_on_p_filt[:,:,n]),-np.log10(0.000001)) #red intensity
                            B = np.divide(-np.log10(rfmaps_off_p_filt[:,:,n]),-np.log10(0.000001)) #blue intenstiy
                            data[:,:,0] = 1 - B #red intensity is one minus blue
                            data[:,:,1] = 1 - B - R #green channel is one minus blue and red
                            data[:,:,2] = 1 - R #blue channel is one minus red

                            data[data<0] = 0 #lower trim to zero

                            ax.imshow(data)
                            ax.scatter(RF_x[n],RF_y[n],marker='+',c='k',s=35) #show RF center location
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_aspect('auto')
                            ax.set_title("%d" % n)
                        
                        plt.tight_layout(rect=[0, 0, 1, 1])
                        fig.savefig(os.path.join(plane_folder,'RF_Plane%d_%s.jpg' % (iplane,signal)),dpi=600)
    return 





# example_cells = np.where(iscell==1)[0][50:75]
# # example_cells = 74

# Tot         = len(example_cells)*2
# Rows        = int(np.floor(np.sqrt(Tot)))
# Cols        = Tot // Rows # Compute Rows required
# if Tot % Rows != 0: #If one additional row is necessary -> add one:
#     Cols += 1
# Position = range(1,Tot + 1) # Create a Position index

# fig = plt.figure(figsize=[18, 9])
# for i,n in enumerate(example_cells):
#     # add every single subplot to the figure with a for loop
#     ax = fig.add_subplot(Rows,Cols,Position[i*2])
#     # ax.imshow(-np.log10(rfmaps_on_p[:,:,n]),cmap='Reds',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
#     ax.imshow(-np.log10(rfmaps_on_p_filt[:,:,n]),cmap='Reds',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
#     ax.scatter(plane_rf_x[n],plane_rf_y[n],marker='+',c='w',s=10)
#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     ax.set_title("%d,ON" % n)
    
#     ax = fig.add_subplot(Rows,Cols,Position[i*2 + 1])
#     # ax.imshow(-np.log10(rfmaps_off_p[:,:,n]),cmap='Blues',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
#     ax.imshow(-np.log10(rfmaps_off_p_filt[:,:,n]),cmap='Blues',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
#     ax.scatter(plane_rf_x[n],plane_rf_y[n],marker='+',c='w',s=10)

#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     ax.set_title("%d,OFF" % n)
  
# plt.tight_layout(rect=[0, 0, 1, 1])



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


# #### Zscored version:
# rfmaps_z          = np.zeros([xGrid,yGrid,N])

# for n in range(N):
#     print(f"\rZscoring RF for neuron {n+1} / {N}")
#     rfmaps_z[:,:,n] = st.zscore(rfmaps[:,:,n],axis=None)

# ## Show example cell RF maps:
# example_cells = [0,24,285,335,377,496,417,551,430,543,696,689,617,612,924] #V1
# example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

# example_cells = range(900,1000)

# example_cells = range(0,100) 

# Tot         = len(example_cells)
# Rows        = int(np.floor(np.sqrt(Tot)))
# Cols        = Tot // Rows # Compute Rows required
# if Tot % Rows != 0: #If one additional row is necessary -> add one:
#     Cols += 1
# Position = range(1,Tot + 1) # Create a Position index

# fig = plt.figure(figsize=[18, 9])
# for i,n in enumerate(example_cells):
#     # add every single subplot to the figure with a for loop
#     ax = fig.add_subplot(Rows,Cols,Position[i])
#     ax.imshow(rfmaps[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps[:,:,n])),vmax=np.max(abs(rfmaps[:,:,n])))
#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     ax.set_title(n)
  
# plt.tight_layout(rect=[0, 0, 1, 1])


# #### 
# fig, axes = plt.subplots(7, 13, figsize=[17, 8])
# for i in range(np.shape(axes)[0]):
#     for j in range(np.shape(axes)[1]):
#         n = i*np.shape(axes)[1] + j
#         ax = axes[i,j]
#         # ax.imshow(rfmaps[:,:,n],cmap='gray')
#         ax.imshow(rfmaps[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps[:,:,n])),vmax=np.max(abs(rfmaps[:,:,n])))
#         ax.set_axis_off()
#         ax.set_aspect('auto')
#         ax.set_title(n)

# ### 
# plt.close('all')

# ####################### Population Receptive Field

# depths,ind  = np.unique(celldata['depth'], return_index=True)
# depths      = depths[np.argsort(ind)]
# areas       = ['V1','V1','V1','V1','PM','PM','PM','PM']

# Rows        = 2
# Cols        = 4 
# Position    = range(1,8 + 1) # Create a Position index

# # fig, axes = plt.subplots(2, 4, figsize=[17, 8])
# fig = plt.figure()

# for iplane,depth in enumerate(depths):
#     # add every single subplot to the figure with a for loop
#     ax = fig.add_subplot(Rows,Cols,Position[iplane])
#     idx = celldata['depth']==depth
#     # popmap = np.nanmean(abs(rfmaps_z[:,:,idx]),axis=2)
#     popmap = np.nanmean(abs(rfmaps[:,:,idx]),axis=2)
#     ax.imshow(popmap,cmap='OrRd')
#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     ax.set_title(areas[iplane])
    
# plt.tight_layout(rect=[0, 0, 1, 1])

# #######################################
# ### Compute RF maps: (method 2)

# rfmaps_on        = np.empty([xGrid,yGrid,N])
# rfmaps_off       = np.empty([xGrid,yGrid,N])

# rfmaps_on_p      = np.empty([xGrid,yGrid,N])
# rfmaps_off_p     = np.empty([xGrid,yGrid,N])


# for n in range(N):
#     print(f"\rComputing RF for neuron {n+1} / {N}")
    
#     resps = np.empty(nGrids)
#     for g in range(nGrids):

#         temp = np.logical_and(ts_F > RF_timestamps[g]+t_resp_start,ts_F < RF_timestamps[g]+t_resp_stop)
#         resp = calciumdata.iloc[temp,n].mean()
#         temp = np.logical_and(ts_F > RF_timestamps[g]+t_base_start,ts_F < RF_timestamps[g]+t_base_stop)
#         base = calciumdata.iloc[temp,n].mean()
    
#         # resps[g] = np.max([resp-base,0])
#         resps[g] = resp-base

#     # temp_resps = np.empty([xGrid,yGrid,50])

#     for i in range(xGrid):
#         for j in range(yGrid):
#             rfmaps_on[i,j,n] = np.mean(resps[grid_array[i,j,:]==1])
#             rfmaps_off[i,j,n] = np.mean(resps[grid_array[i,j,:]==-1])
            
            
#             rfmaps_on_p[i,j,n] = st.ttest_ind(resps[grid_array[i,j,:]==1],resps[grid_array[i,j,:] == 0])[1]
#             rfmaps_off_p[i,j,n] = st.ttest_ind(resps[grid_array[i,j,:]==-1],resps[grid_array[i,j,:] == 0])[1]
                    

#             # rfmaps_on_p[i,j,n] = sum(rfmaps_on[i,j,n] > resps[grid_array[i,j,:]==0]) / sum(grid_array[i,j,:]==0)
#             # rfmaps_off_p[i,j,n] = sum(rfmaps_off[i,j,n] > resps[grid_array[i,j,:]==0]) / sum(grid_array[i,j,:]==0)
                    


# print("Black squares: mean %2.1f +- %2.1f" % (np.mean(np.sum(grid_array[:,:,:]==1,axis=2).flatten()),
#                                               np.std(np.sum(grid_array[:,:,:]==-1,axis=2).flatten())))
# print("White squares: mean %2.1f +- %2.1f\n" % (np.mean(np.sum(grid_array[:,:,:]==1,axis=2).flatten()),
#                                                 np.std(np.sum(grid_array[:,:,:]==1,axis=2).flatten())))  


# # rfmaps_on_p = 1 - rfmaps_on_p
# # rfmaps_off_p = 1 - rfmaps_off_p

# ## Show example cell RF maps:
# # example_cells = [0,24,285,335,377,496,417,551,430,543,696,689,617,612,924] #V1
# # example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM

# # example_cells = range(900,1000)

# # example_cells = [0,9,17,18,24,27,29,42,44,45,54,56,57,69,72,82,83,89,90,94,96,98] #V1
# # example_cells = [1250,1257,1414,1415,1417,1423,1551,1559,2006,1925,1972,2178,1666] #PM

# # example_cells = range(0,20)

# example_cells = np.where(iscell==1)[0][50:75]

# Tot         = len(example_cells)*2
# Rows        = int(np.floor(np.sqrt(Tot)))
# Cols        = Tot // Rows # Compute Rows required
# if Tot % Rows != 0: #If one additional row is necessary -> add one:
#     Cols += 1
# Position = range(1,Tot + 1) # Create a Position index

# fig = plt.figure(figsize=[18, 9])
# for i,n in enumerate(example_cells):
#     # add every single subplot to the figure with a for loop
#     ax = fig.add_subplot(Rows,Cols,Position[i*2])
#     ax.imshow(-np.log10(rfmaps_on_p[:,:,n]),cmap='Reds',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))

#     # ax.imshow(rfmaps_on[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps_on[:,:,n])),vmax=np.max(abs(rfmaps_on[:,:,n])))
#     # ax.imshow(rfmaps_off[:,:,n],cmap='gray',vmin=-np.max(abs(rfmaps_off[:,:,n])),vmax=np.max(abs(rfmaps_off[:,:,n])))
    
#     # ax.imshow(-np.log10(rfmaps_on_p[:,:,n]),cmap='Reds',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
#     # ax.imshow(-np.log10(rfmaps_off_p[:,:,n]),cmap='Blues',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))

#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     ax.set_title("%d,ON" % n)
    
#     ax = fig.add_subplot(Rows,Cols,Position[i*2 + 1])
#         # ax.imshow(-np.log10(rfmaps_off_p[:,:,n]),cmap='Blues',vmin=-np.log10(0.05),vmax=-np.log10(0.00001))
#     ax.imshow(rfmaps_off_p[:,:,n]<0.001,cmap='Blues',vmin=0,vmax=1)

#     # img = np.dstack((rfmaps_on_p[:,:,n],np.ones(np.shape(rfmaps_off_p[:,:,n])),rfmaps_off_p[:,:,n]))
#     # ax.imshow(-np.log10(img),vmin=-np.log10(0.05),vmax=-np.log10(0.00001))

#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     ax.set_title("%d,OFF" % n)
  
# plt.tight_layout(rect=[0, 0, 1, 1])

# ## POP map
# depths,ind  = np.unique(celldata['depth'], return_index=True)
# depths      = depths[np.argsort(ind)]
# areas       = ['V1','V1','V1','V1','PM','PM','PM','PM']

# Rows        = 2
# Cols        = 4 
# Position    = range(1,8 + 1) # Create a Position index

# # fig, axes = plt.subplots(2, 4, figsize=[17, 8])
# fig = plt.figure(figsize=[9, 3])

# for iplane,depth in enumerate(depths):
#     # add every single subplot to the figure with a for loop
#     ax = fig.add_subplot(Rows,Cols,Position[iplane])
#     idx = celldata['depth']==depth
    
#     # popmap = np.sum(np.logical_or(rfmaps_on_p[:,:,idx] <0.001, rfmaps_off_p[:,:,idx] < 0.001),axis=2) / np.sum(idx)
#     popmap = np.sum(np.logical_or(rfmaps_on_p[:,:,idx] <0.01, rfmaps_off_p[:,:,idx] < 0.01),axis=2) / np.sum(idx)
#     IM = ax.imshow(popmap,cmap='PuRd',vmin=0,vmax=0.25)
#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     ax.set_title(areas[iplane])
#     fig.colorbar(IM, ax=ax)

    
# plt.tight_layout(rect=[0, 0, 1, 1])


# ## old code to find optimal response window size:
# n = 24
# n = 0
# t_base_start     = -0.5     #pre s
# t_base_stop      = 0        #post s

# fig, axes = plt.subplots(4, 4, figsize=[17, 8], sharey='row')

# t_starts = np.array([0, 0.1, 0.2, 0.3])
# t_stops = np.array([0.4,0.5,0.6,0.7])
  
# # for i,t_resp_start in np.array([0, 0.1, 0.2, 0.3]):
# for i,t_resp_start in enumerate(t_starts):
#     for j,t_resp_stop in enumerate(t_stops):
#         rfmap = np.zeros([xGrid,yGrid])

#         for g in range(nGrids):
#             temp = np.logical_and(ts_F >= RF_timestamps[g]+t_resp_start,ts_F <= RF_timestamps[g]+t_resp_stop)
#             resp = calciumdata.iloc[temp,n].mean()
#             temp = np.logical_and(ts_F >= RF_timestamps[g]+t_base_start,ts_F <= RF_timestamps[g]+t_base_stop)
#             base = calciumdata.iloc[temp,n].mean()

#             # base = 0

#             # rfmap = np.nansum(rfmap,np.max([resp-base,0]) * grid_array[:,:,g])
#             rfmap = np.nansum(np.dstack((rfmap,np.max([resp-base,0]) * grid_array[:,:,g])),2)
#             # rfmap = rfmap + (resp-base) * grid_array[:,:,g]

#         ax = axes[i,j]
#         ax.imshow(rfmap,cmap='gray',vmin=-np.max(abs(rfmap)),vmax=np.max(abs(rfmap)))
#         # ax.imshow(rfmap,cmap='gray',vmin=-30000,vmax=30000)

