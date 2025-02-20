# %% # Imports
# -*- coding: utf-8 -*-
"""
This script visualizes the neurons selected for MEI in vivo validation
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Import general libs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import suite2p
import scipy


#%% Set the followup session:
rawdatadir_mei      = "E:\\RawData\\"
animal_id_mei       = 'LPE13959' #If empty than all animals in folder will be processed
sessiondate_mei     = '2025_02_19'

meidir          = os.path.join(rawdatadir_mei,animal_id_mei,sessiondate_mei)
meidir_suite2p  = os.path.join(meidir,"suite2p")
assert os.path.exists(meidir_suite2p), 'suite2p folder not found'

fname = os.path.join(meidir_suite2p,'plane1','Fmatch.mat')

data = scipy.io.loadmat(fname,squeeze_me=True)
data = scipy.io.loadmat(fname)

g = data['roiMatchData']['mapping'][0][0]

np.shape(g)

np.max(g,axis=0)

h = data['roiMatchData']['refImage'][0][0]
plt.imshow(h)

fig, axes = plt.subplots(1,2)
h = data['roiMatchData']['rois'][0][0][0][0]['meanFrame'][0][0]
axes[0].imshow(h)
h = data['roiMatchData']['rois'][0][0][0][1]['meanFrame'][0][0]
axes[1].imshow(h)

plt.show()



#Old code, superfluous because of ROIMatchPub

# #%% Import general libs
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import suite2p

# os.chdir('../')  # set working directory to the root of the git repo

# # Import personal lib funcs
# from loaddata.session_info import filter_sessions, load_sessions
# from labeling.tdTom_labeling_cellpose import *
# from natsort import natsorted
# from utils.imagelib import im_norm,im_norm8,im_log,im_sqrt

# #%% Set the reference session:
# rawdatadir_ref      = "M:\\RawData\\"
# animal_id_ref       = 'LPE12385' #If empty than all animals in folder will be processed
# sessiondate_ref     = '2024_06_15'

# refdir          = os.path.join(rawdatadir_ref,animal_id_ref,sessiondate_ref)
# refdir_suite2p  = os.path.join(refdir,"suite2p")
# assert os.path.exists(refdir_suite2p), 'suite2p folder not found'

# #%% Set the followup session:
# rawdatadir_mei      = "M:\\RawData\\"
# animal_id_mei       = 'LPE12385' #If empty than all animals in folder will be processed
# sessiondate_mei     = '2024_06_16'

# meidir          = os.path.join(rawdatadir_mei,animal_id_mei,sessiondate_mei)
# meidir_suite2p  = os.path.join(meidir,"suite2p")
# assert os.path.exists(meidir_suite2p), 'suite2p folder not found'

# #%% 

# plane_folders_ref = natsorted([f.path for f in os.scandir(refdir_suite2p) if f.is_dir() and f.name[:5]=='plane'])
# plane_folders_mei = natsorted([f.path for f in os.scandir(meidir_suite2p) if f.is_dir() and f.name[:5]=='plane'])
# nplanes = len(plane_folders_ref)

# #%% 
# data_overlap    = [[] for _ in range(8)]
# data_centroid   = [[] for _ in range(8)]
# cell_ids_ref    = [[] for _ in range(8)]
# cell_ids_mei    = [[] for _ in range(8)]

# for iplane,(ref_plane_folder,mei_plane_folder) in enumerate(zip(plane_folders_ref,plane_folders_mei)):
#     print(iplane)
#     # ops         = np.load(os.path.join(ref_plane_folder,'ops.npy'), allow_pickle=True).item()
#     stats_ref   = np.load(os.path.join(ref_plane_folder,'stat.npy'), allow_pickle=True)[:100]
#     stats_mei   = np.load(os.path.join(mei_plane_folder,'stat.npy'), allow_pickle=True)[:200]

#     temp_overlap   = np.empty((len(stats_ref),len(stats_mei)))
#     temp_centroid  = np.empty((len(stats_ref),len(stats_mei)))

#     for iref,s_ref in enumerate(stats_ref):
#         for imei,s_mei in enumerate(stats_mei):
            
#             pix_overlap = np.sum(np.logical_and(np.isin(s_ref['ypix'],s_mei['ypix']),
#                                   np.isin(s_ref['xpix'],s_mei['xpix'])))
#             temp_overlap[iref,imei] = pix_overlap / np.mean((s_ref['npix'],s_mei['npix']))
#             temp_centroid[iref,imei] = np.sqrt((s_ref['med'][0] - s_mei['med'][0])**2
#                                                + (s_ref['med'][1] - s_mei['med'][1])**2)
#     data_overlap[iplane] = temp_overlap
#     data_centroid[iplane] = temp_centroid / 512 * 600

#     cell_ids_ref[iplane]           = np.array([animal_id_ref + '_' + sessiondate_ref + '_' + '%s' % iplane + '_' + '%04.0f' % k for k in range(0,len(stats_ref))])
#     cell_ids_mei[iplane]           = np.array([animal_id_mei + '_' + sessiondate_mei + '_' + '%s' % iplane + '_' + '%04.0f' % k for k in range(0,len(stats_mei))])

# #%% Flatten data:
# data_overlap_flat = np.concatenate([data_overlap[i].flatten() for i in range(nplanes)])
# data_centroid_flat = np.concatenate([data_centroid[i].flatten() for i in range(nplanes)])

# #%% Show histograms:

# fig, axes = plt.subplots(1, 3,figsize=(6,2))

# axes[0].hist(data_overlap_flat[data_centroid_flat < 12],bins=50)
# axes[0].set_xlabel('Footprint Overlap')
# axes[0].set_ylabel('Count')

# axes[1].hist(data_centroid_flat[data_centroid_flat < 12],bins=50)
# axes[1].set_xlabel('Centroid Distance (μm)')

# sns.kdeplot(x=data_centroid_flat[data_centroid_flat < 12],
#             y=data_overlap_flat[data_centroid_flat < 12],ax=axes[2],thresh=0,fill=True,cmap="mako", cbar=False)
# # sns.kdeplot(x=data_centroid_flat,|
# #             y=data_overlap_flat,ax=axes[2],thresh=0,fill=True,cmap="mako", cbar=False)
# axes[2].set_xlabel('Centroid Distance (μm)')
# axes[2].set_ylabel('Overlap Fraction')
# axes[2].set_ylim([0,1])
# axes[2].set_xlim([0,12])
# axes[2].set_yticks([0,1])
# axes[2].set_xticks([0,6,12])

# plt.tight_layout()


# #%% Identify same cells:

# thr_overlap     = 0.8
# thr_centroid    = 6

# cell_id_map = np.empty((0,2),dtype='<U26')

# for iplane in range(nplanes): #for each plane identify overlapping cells:
#     idx             = np.all((data_overlap[iplane]>thr_overlap,data_centroid[iplane]<thr_centroid),axis=0)
#     idx_x,idx_y     = np.where(idx) #get ref and mei indices
#     temp            = np.vstack((cell_ids_ref[iplane][idx_x],cell_ids_mei[iplane][idx_y])).T
#     cell_id_map     = np.concatenate((cell_id_map,temp),axis=0) #add to running list of same cells

# #%% Show same cells across days:
# fig, axes = plt.subplots(2, nplanes,figsize=(nplanes*2,2*2))

# for iplane,plane_folder in enumerate(plane_folders_ref):
#     ops         = np.load(os.path.join(plane_folder,'ops.npy'), allow_pickle=True).item()
#     stats       = np.load(os.path.join(plane_folder,'stat.npy'), allow_pickle=True)
#     iscell      = np.load(os.path.join(plane_folder,'iscell.npy'), allow_pickle=True)

#     # # Get mean green GCaMP image: 
#     mimg = ops['meanImg']
#     mimg = im_norm8(mimg,min=1,max=99) #scale between 0 and 255
    
#     # From cell masks create outlines:
#     masks_suite2p = np.zeros((512,512), np.float32)
#     for i,s in enumerate(stats):
#         masks_suite2p[s['ypix'],s['xpix']] = i+1

#     outl_green = get_outlines(masks_suite2p)

#     ismeicell = np.empty(len(iscell),dtype=bool)
#     ismeicell[:] = False
#     # ismeicell[cell_ids_df['plane'].astype(int) == iplane] = True
#     idx = cell_ids_df['idx'][cell_ids_df['plane'].astype(int) == iplane].astype(int).to_numpy()
#     ismeicell[idx] = True
#     # ismeicell = cell_ids_df['idx'][cell_ids_df['plane'].astype(int) == iplane].astype(int).to_numpy()

#     # Get max projection GCaMP image: 
#     mimg = np.zeros([512,512])
#     mimg[ops['yrange'][0]:ops['yrange'][1],
#     ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']
#     mimg = im_norm8(mimg,min=1,max=99) #scale between 0 and 255

#     mimg2 = ops['meanImg_chan2'] #get red channel image from ops
#     mimg2 = im_norm(mimg2,min=0.5,max=99.9) #scale between 0 and 255
#     mimg2 = im_sqrt(mimg2) #log transform to enhance weakly expressing cells
#     mimg2 = im_norm(mimg2,min=0,max=100) #scale between 0 and 255

   
#     ax = axes[0,iplane]
#     im1 = np.dstack((np.zeros(np.shape(mimg)),mimg,np.zeros(np.shape(mimg)))).astype(np.uint8)
#     ax.imshow(im1,vmin=0,vmax=255)
#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     # ax.set_title('GCaMP', fontsize=16, color='green', fontweight='bold',loc='center')
#     ax.set_title('Plane %d' % iplane, fontsize=10, color='k', fontweight='bold',loc='center')

#     for i,o in enumerate(outl_green):
#         if ismeicell[i]: #show only good cells
#             ax.plot(o[:,0], o[:,1], color='w',linewidth=0.6)

#     ax = axes[1,iplane]
#     im2 = np.dstack((mimg2,np.zeros(np.shape(mimg2)),np.zeros(np.shape(mimg2)))).astype(np.uint8)
#     ax.imshow(im2,vmin=0,vmax=255)
#     ax.set_axis_off()
#     ax.set_aspect('auto')
#     # ax.set_title('tdTomato', fontsize=16, color='red', fontweight='bold',loc='center')

#     for i,o in enumerate(outl_green):
#         if ismeicell[i]: #show only good cells
#             ax.plot(o[:,0], o[:,1], color='w',linewidth=0.6)

# plt.suptitle('MEI cells - %s - %s' % (animal_id,sessiondate), fontsize=13, color='k', fontweight='bold')
# plt.tight_layout(rect=[0, 0, 1, 1])

