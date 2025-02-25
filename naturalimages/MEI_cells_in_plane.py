# %% # Imports
# -*- coding: utf-8 -*-
"""
This script visualizes the neurons selected for MEI in vivo validation
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

# Import general libs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('../')  # set working directory to the root of the git repo

# Import personal lib funcs
from loaddata.session_info import filter_sessions, load_sessions
from labeling.tdTom_labeling_cellpose import *
from natsort import natsorted
from utils.imagelib import im_norm,im_norm8,im_log,im_sqrt

#%% Set the session:
rawdatadir      = "G:\\RawData\\"
animal_id       = 'LPE12385' #If empty than all animals in folder will be processed
sessiondate     = '2024_06_15'

rawdatadir      = "M:\\RawData\\"
animal_id       = 'LPE12385' #If empty than all animals in folder will be processed
sessiondate     = '2024_06_15'
# protocol        = 'MV'

sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)

#################################################
session_list        = np.array([[animal_id,sessiondate]])
sessions,nSessions  = load_sessions(protocol = 'DN',session_list=session_list,load_behaviordata=False, 
                                    load_calciumdata=False, load_videodata=False, calciumversion='deconv')

cell_ids = sessions[0].celldata['cell_id'].sample(n=150).tolist()

cell_ids_df = pd.DataFrame({'cell_id':cell_ids})
cell_ids_df[['animal_id','year','month','day','plane','idx']] = cell_ids_df['cell_id'].str.split('_',expand=True)

# load selected cells:
cells = pd.read_csv(os.path.join(sesfolder,'selected_cells.csv'))


# %%
# !!!! ADD title subplot which area it is, and color outline whether cells are labeled or not

#%% 

suite2p_folder  = os.path.join(sesfolder,"suite2p")
assert os.path.exists(suite2p_folder), 'suite2p folder not found'

plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
nplanes = len(plane_folders)
#Show labeling results in green and red image:
fig, axes = plt.subplots(2, nplanes,figsize=(nplanes*2,2*2))

for iplane,plane_folder in enumerate(plane_folders):
    ops         = np.load(os.path.join(plane_folder,'ops.npy'), allow_pickle=True).item()
    stats       = np.load(os.path.join(plane_folder,'stat.npy'), allow_pickle=True)
    iscell      = np.load(os.path.join(plane_folder,'iscell.npy'), allow_pickle=True)

    # # Get mean green GCaMP image: 
    mimg = ops['meanImg']
    mimg = im_norm8(mimg,min=1,max=99) #scale between 0 and 255
    
    # From cell masks create outlines:
    masks_suite2p = np.zeros((512,512), np.float32)
    for i,s in enumerate(stats):
        masks_suite2p[s['ypix'],s['xpix']] = i+1

    outl_green = get_outlines(masks_suite2p)

    ismeicell = np.empty(len(iscell),dtype=bool)
    ismeicell[:] = False
    # ismeicell[cell_ids_df['plane'].astype(int) == iplane] = True
    idx = cell_ids_df['idx'][cell_ids_df['plane'].astype(int) == iplane].astype(int).to_numpy()
    ismeicell[idx] = True
    # ismeicell = cell_ids_df['idx'][cell_ids_df['plane'].astype(int) == iplane].astype(int).to_numpy()

    # Get max projection GCaMP image: 
    mimg = np.zeros([512,512])
    mimg[ops['yrange'][0]:ops['yrange'][1],
    ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']
    mimg = im_norm8(mimg,min=1,max=99) #scale between 0 and 255

    mimg2 = ops['meanImg_chan2'] #get red channel image from ops
    mimg2 = im_norm(mimg2,min=0.5,max=99.9) #scale between 0 and 255
    mimg2 = im_sqrt(mimg2) #log transform to enhance weakly expressing cells
    mimg2 = im_norm(mimg2,min=0,max=100) #scale between 0 and 255

   
    ax = axes[0,iplane]
    im1 = np.dstack((np.zeros(np.shape(mimg)),mimg,np.zeros(np.shape(mimg)))).astype(np.uint8)
    ax.imshow(im1,vmin=0,vmax=255)
    ax.set_axis_off()
    ax.set_aspect('auto')
    # ax.set_title('GCaMP', fontsize=16, color='green', fontweight='bold',loc='center')
    ax.set_title('Plane %d' % iplane, fontsize=10, color='k', fontweight='bold',loc='center')

    for i,o in enumerate(outl_green):
        if ismeicell[i]: #show only good cells
            ax.plot(o[:,0], o[:,1], color='w',linewidth=0.6)

    ax = axes[1,iplane]
    im2 = np.dstack((mimg2,np.zeros(np.shape(mimg2)),np.zeros(np.shape(mimg2)))).astype(np.uint8)
    ax.imshow(im2,vmin=0,vmax=255)
    ax.set_axis_off()
    ax.set_aspect('auto')
    # ax.set_title('tdTomato', fontsize=16, color='red', fontweight='bold',loc='center')

    for i,o in enumerate(outl_green):
        if ismeicell[i]: #show only good cells
            ax.plot(o[:,0], o[:,1], color='w',linewidth=0.6)

plt.suptitle('MEI cells - %s - %s' % (animal_id,sessiondate), fontsize=13, color='k', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 1])

