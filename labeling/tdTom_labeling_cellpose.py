"""
Analyzes labeling of tdtomato expressing cells using cellpose software (Pachitariu & Stringer)
optimized for green + red channel mesoscopic 2p Ca2+ imaging recordings
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread
from PIL import ImageColor
from natsort import natsorted 

from cellpose import utils, io

import pandas as pd
import seaborn as sns
from suite2p.extraction import extract, masks
from suite2p.detection.chan2detect import detect,correct_bleedthrough

from utils.imagelib import im_norm,im_norm8,im_log,im_sqrt

from PIL import Image

def proc_labeling_plane(iplane,plane_folder,showcells=True,overlap_threshold=0.5):
    stats       = np.load(os.path.join(plane_folder,'stat.npy'), allow_pickle=True)
    ops         = np.load(os.path.join(plane_folder,'ops.npy'), allow_pickle=True).item()
    iscell      = np.load(os.path.join(plane_folder,'iscell.npy'), allow_pickle=True)

    Nsuite2pcells      = np.shape(stats)[0]

    # From cell masks create outlines:
    masks_suite2p = np.zeros((512,512), np.float32)
    for i,s in enumerate(stats):
        masks_suite2p[s['ypix'],s['xpix']] = i+1

    # load red cell roi information from cellpose gui:
    filenames       = os.listdir(plane_folder)
    cellpose_file   = list(filter(lambda a: 'seg.npy' in a, filenames)) #find the files
    redcell_seg     = np.load(os.path.join(plane_folder,cellpose_file[0]), allow_pickle=True).item()
    masks_cp_red    = redcell_seg['masks']
    
    Ncellpose_redcells      = len(np.unique(masks_cp_red))-1

    # Compute fraction of suite2p ROI that is red labeled
    frac_of_ROI_red = np.empty(Nsuite2pcells)
    for i,s in enumerate(stats):
        frac_of_ROI_red[i] = np.sum(masks_cp_red[s['ypix'],s['xpix']] != 0) / len(s['ypix'])

    # Compute fraction of labeled cellpose soma that falls inside suite2p ROI
    # Suite2p ROIs are generally larger and capture calcium activity of membrane and AIS
    # If full tdTomato labeled soma falls inside ROI then labeled. upper measure however could 
    # have lower value if large extra ROI taken as calcium dependent ROI
    frac_red_in_ROI     = np.zeros(Nsuite2pcells)
    npix_redsomas   = np.histogram(masks_cp_red,
                        bins=np.arange(Ncellpose_redcells+1).astype(int))[0]
    if Ncellpose_redcells:
        for i,s in enumerate(stats):
            c = np.histogram(masks_cp_red[s['ypix'],s['xpix']],
                        bins=np.arange(Ncellpose_redcells+1).astype(int))[0]
            c[0] = 0 #set overlap with non-ROI pixels to zero
            frac_red_in_ROI[i] = np.max(c/npix_redsomas) #get maximum overlap (red cell with largest fraction inside suite2p ROI)

    # redcell             = redcell_overlap > overlap_threshold
    redcell             = frac_red_in_ROI > overlap_threshold
    redcell_cellpose    = np.vstack((redcell,frac_of_ROI_red,frac_red_in_ROI))

    np.save(os.path.join(plane_folder,'redcell_cellpose.npy'),redcell_cellpose)

    if showcells:
         # Get mean green GCaMP image: 
        mimg = ops['meanImg']
        mimg = im_norm8(mimg,min=1,max=99) #scale between 0 and 255
        
        # Get max projection GCaMP image: 
        # mimg = np.zeros([512,512])
        # mimg[ops['yrange'][0]:ops['yrange'][1],
        # ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']
        # mimg = im_norm8(mimg,min=1,max=99) #scale between 0 and 255

        ## Get red image:
        mimg2 = ops['meanImg_chan2'] #get red channel image from ops
        mimg2 = im_norm(mimg2,min=0.5,max=99.5) #scale between percentiles

        # mimg2 = im_log(mimg2) #log transform to enhance weakly expressing cells
        mimg2 = im_sqrt(mimg2) #square root transform to enhance weakly expressing cells

        mimg2 = im_norm(mimg2,min=0,max=100) #scale between 0 and 255
        
        #scale between 0 and 255
        # mimg2       = im_log(mimg2) #log transform to enhance weakly expressing cells
        # mimg2       = im_norm(mimg2,min=0,max=100).astype(np.uint8) #scale between 0 and 255

        #Show labeling results in green and red image:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,6))

        im1 = np.dstack((np.zeros(np.shape(mimg)),mimg,np.zeros(np.shape(mimg)))).astype(np.uint8)
        ax1.imshow(im1,vmin=0,vmax=255)

        im2 = np.dstack((mimg2,np.zeros(np.shape(mimg2)),np.zeros(np.shape(mimg2)))).astype(np.uint8)
        ax2.imshow(im2,vmin=0,vmax=255)

        im3 = np.dstack((mimg2,mimg,np.zeros(np.shape(mimg2)))).astype(np.uint8)
        ax3.imshow(im3,vmin=0,vmax=255)

        outl_green = utils.outlines_list(masks_suite2p)
        #Filter good cells for visualization laters: 
        outl_green = np.array(outl_green)[iscell[:,0]==1]
        
        red_filtered = redcell_cellpose[1,iscell[:,0]==1]

        outl_red = utils.outlines_list(masks_cp_red)
        
        for i,o in enumerate(outl_green):
            if iscell[i,0]: #show only good cells
                ax1.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
                # ax2.plot(o[:,0], o[:,1], color='g',linewidth=0.6)
                if red_filtered[i]:
                    ax3.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
                # else: 
                    # ax3.plot(o[:,0], o[:,1], color='w',linewidth=0.6)

        for o in outl_red:
            # ax1.plot(o[:,0], o[:,1], color='r',linewidth=0.6)
            ax2.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
            # ax3.plot(o[:,0], o[:,1], color='y',linewidth=0.6)
            # ax3.plot(o[:,0], o[:,1], color='#ffe601',linewidth=0.6)
        
        ax1.set_axis_off()
        ax1.set_aspect('auto')
        ax1.set_title('Suite2p cells (GCaMP)', fontsize=12, color='black', fontweight='bold',loc='center')
        ax2.set_axis_off()
        ax2.set_aspect('auto')
        ax2.set_title('tdTomato cells (tdTomato)', fontsize=12, color='black', fontweight='bold',loc='center')
        ax3.set_axis_off()
        ax3.set_aspect('auto')
        ax3.set_title('Labeled cells (Merge)', fontsize=12, color='black', fontweight='bold',loc='center')

        plt.tight_layout(rect=[0, 0, 1, 1])

        fig.savefig(os.path.join(plane_folder,'labeling_Plane%d.jpg' % iplane),dpi=600)

    return 

def proc_labeling_session(rawdatadir,animal_id,sessiondate,showcells=True):
    sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)

    suite2p_folder  = os.path.join(sesfolder,"suite2p")

    assert os.path.exists(suite2p_folder), 'suite2p folders not found'
    
    plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])

    for iplane,plane_folder in enumerate(plane_folders):
        # print(iplane)
        proc_labeling_plane(iplane,plane_folder,showcells=showcells)
        

def gen_red_images(rawdatadir,animal_id,sessiondate):

    sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)

    suite2p_folder  = os.path.join(sesfolder,"suite2p")

    assert os.path.exists(suite2p_folder), 'suite2p folders not found'
    
    plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])

    for iplane,plane_folder in enumerate(plane_folders):
    
        ops = np.load(os.path.join(plane_folder,'ops.npy'), allow_pickle=True).item()
        
        mimg2 = ops['meanImg_chan2'] #get red channel image from ops
        # mimg2 = im_norm(mimg2,min=2.5,max=100) #scale between 0 and 255
        mimg2 = im_norm(mimg2,min=0.5,max=99.5) #scale between 0 and 255

        # mimg2 = im_log(mimg2) #log transform to enhance weakly expressing cells

        mimg2 = im_sqrt(mimg2) #log transform to enhance weakly expressing cells

        mimg2 = im_norm(mimg2,min=0,max=100) #scale between 0 and 255

        # mimg2 = np.dstack((mimg2,np.zeros(np.shape(mimg2)),np.zeros(np.shape(mimg2))))
        mimg2 = np.dstack((mimg2,np.zeros(np.shape(mimg2)),np.zeros(np.shape(mimg2))))

        img = Image.fromarray(mimg2.astype(np.uint8))

        img.save(os.path.join(plane_folder,'redim_plane%d.png' % iplane))

def plotseq_labeling_plane(plane_folder,savedir,showcells=True,overlap_threshold=0.5):
    stats       = np.load(os.path.join(plane_folder,'stat.npy'), allow_pickle=True)
    ops         = np.load(os.path.join(plane_folder,'ops.npy'), allow_pickle=True).item()
    iscell      = np.load(os.path.join(plane_folder,'iscell.npy'), allow_pickle=True)

    Nsuite2pcells      = np.shape(stats)[0]

    # From cell masks create outlines:
    masks_suite2p = np.zeros((512,512), np.float32)
    for i,s in enumerate(stats):
        masks_suite2p[s['ypix'],s['xpix']] = i+1

    # load red cell roi information from cellpose gui:
    filenames       = os.listdir(plane_folder)
    cellpose_file   = list(filter(lambda a: 'seg.npy' in a, filenames)) #find the files
    redcell_seg     = np.load(os.path.join(plane_folder,cellpose_file[0]), allow_pickle=True).item()
    masks_cp_red    = redcell_seg['masks']
    
    # Compute overlap of red labeled cell bodies with suite2p cell bodies
    redcell_overlap = np.empty(Nsuite2pcells)
    for i in range(Nsuite2pcells):    # Compute overlap in masks:
        redcell_overlap[i] = np.sum(masks_cp_red[masks_suite2p==i+1] != 0) / np.sum(masks_suite2p ==i+1)
        # if mask_overlap_green_with_red[i]>0:
            # mask_overlap_red_with_green[np.unique(masks_cp_red[masks_suite2p==i+1])[1]-1] = overlap

    redcell             = redcell_overlap > overlap_threshold
    redcell_cellpose    = np.vstack((redcell_overlap,redcell))

    # # Get mean green GCaMP image: 
    # mimg = ops['meanImg']
    # mimg = im_norm8(mimg,min=1,max=99) #scale between 0 and 255
    
    # Get max projection GCaMP image: 
    mimg = np.zeros([512,512])
    mimg[ops['yrange'][0]:ops['yrange'][1],
    ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']
    mimg = im_norm8(mimg,min=1,max=99) #scale between 0 and 255

    ## Get red image:
    mimg2 = ops['meanImg_chan2'] #get red channel image from ops
    # mimg2 = im_norm(mimg2,min=2.5,max=100) #scale between 0 and 255
    mimg2 = im_norm(mimg2,min=0.5,max=99.5) #scale between 0 and 255

    mimg2 = im_sqrt(mimg2) #square root transform to enhance weakly expressing cells

    mimg2 = im_norm(mimg2,min=0,max=100) #scale between 0 and 255

    # clr_rchan = np.array(ImageColor.getcolor('#ff0040', "RGB")) / 255
    # clr_gchan = np.array(ImageColor.getcolor('#00ffbf', "RGB")) / 255

    ######
    # rchan = (mimg2 - np.percentile(mimg2,lowprc)) / np.percentile(mimg2 - np.percentile(mimg2,lowprc),uppprc)
    # gchan = (mimg - np.percentile(mimg,lowprc)) / np.percentile(mimg - np.percentile(mimg,lowprc),uppprc)
    # bchan = np.zeros(np.shape(mimg))

    #Show labeling results in green and red image:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,6))

    im1 = np.dstack((np.zeros(np.shape(mimg)),mimg,np.zeros(np.shape(mimg)))).astype(np.uint8)
    ax1.imshow(im1,vmin=0,vmax=255)

    im2 = np.dstack((mimg2,np.zeros(np.shape(mimg2)),np.zeros(np.shape(mimg2)))).astype(np.uint8)
    ax2.imshow(im2,vmin=0,vmax=255)

    im3 = np.dstack((mimg2,mimg,np.zeros(np.shape(mimg2)))).astype(np.uint8)
    ax3.imshow(im3,vmin=0,vmax=255)

    ax1.set_axis_off()
    ax1.set_aspect('auto')
    ax1.set_title('GCaMP', fontsize=16, color='green', fontweight='bold',loc='center')
    ax2.set_axis_off()
    ax2.set_aspect('auto')
    ax2.set_title('tdTomato', fontsize=16, color='red', fontweight='bold',loc='center')
    ax3.set_axis_off()
    ax3.set_aspect('auto')
    ax3.set_title('Merge', fontsize=16, color='black', fontweight='bold',loc='center')
    plt.tight_layout(rect=[0, 0, 1, 1])

    fig.savefig(os.path.join(savedir,'1.png'))

    if showcells:
        outl_green = utils.outlines_list(masks_suite2p)
        #Filter good cells for visualization laters: 
        outl_green = np.array(outl_green)[iscell[:,0]==1]
        
        red_filtered = redcell_cellpose[1,iscell[:,0]==1]

        outl_red = utils.outlines_list(masks_cp_red)
        
        for i,o in enumerate(outl_green):
            if iscell[i,0]: #show only good cells
                ax1.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
                # if red_filtered[i]:
                #     ax3.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
                
        fig.savefig(os.path.join(savedir,'2.png'))

        for o in outl_red:
            ax2.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
        fig.savefig(os.path.join(savedir,'3.png'))

        for i,o in enumerate(outl_green):
            if iscell[i,0]: #show only good cells
                # ax1.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
                if red_filtered[i]:
                    ax3.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
                
        fig.savefig(os.path.join(savedir,'4.png'))


    return 
#piece of code to analyze how many red cells were labeled etc. overlap with suite2p bladiebla
    # nOnlyRedCells   = np.sum(mask_overlap_red_with_green==0)
    # nOverlapCells   = np.sum(mask_overlap_green_with_red>0)
    # nOnlyGreenCells = Nsuite2pcells - nOverlapCells
    # nOnlyRedCells   = Ncellpose_redcells - nOverlapCells

    # nTotalCells     = nOnlyGreenCells + nOnlyRedCells + nOverlapCells

    # df = pd.DataFrame()
    # df['suite2p']       = np.concatenate((np.full((Nsuite2pcells), True), np.full((nOnlyRedCells), False)))
    # df['cellpose_red']  = np.concatenate((np.full((nOnlyGreenCells), False), np.full((Ncellpose_redcells), True)))
    # df['overlap']       = np.zeros(nTotalCells)
    # df['overlap'][np.logical_and(df['suite2p'],df['cellpose_red'])] = mask_overlap_green_with_red[mask_overlap_green_with_red>0]

# deprecated version with labeling by cellpose algorithm not GUI:

# chan                = [[1,0]] # grayscale=0, R=1, G=2, B=3 # channels = [cytoplasm, nucleus]
# diam                = 12
# overlap_threshold   = 0.2
# model_type='cyto' or 'nuclei' or 'cyto2'
# model = models.Cellpose(model_type='cyto')
# model_red = models.CellposeModel(pretrained_model = 'T:\\Python\\cellpose\\testdir\\models\\MOL_20230814_redcells')
# model_green = models.Cellpose(model_type='cyto')


# def proc_labeling_plane(plane_folder,show_plane=False,showcells=True):
#     stats       = np.load(os.path.join(plane_folder,'stat.npy'), allow_pickle=True)
#     ops         = np.load(os.path.join(plane_folder,'ops.npy'), allow_pickle=True).item()
#     iscell      = np.load(os.path.join(plane_folder,'iscell.npy'), allow_pickle=True)
#     # redcell     = np.load(os.path.join(plane_folder,'redcell.npy'), allow_pickle=True)

#     Nsuite2pcells      = np.shape(stats)[0]

#     # From cell masks create outlines:
#     masks_suite2p = np.zeros((512,512), np.float32)
#     for i,s in enumerate(stats):
#         masks_suite2p[s['ypix'],s['xpix']] = i+1

#     mimg = ops['meanImg']
#     # mimg = np.zeros([512,512])
#     # mimg[ops['yrange'][0]:ops['yrange'][1],
#         # ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']

#     mimg2 = ops['meanImg_chan2']

#     mimg2 = im_norm(mimg2,min=2.5,max=99) #scale between 0 and 255

#     mimg2_log = im_log(mimg2) #log transform to enhance weakly expressing cells

#     mimg2 = im_norm(mimg2,min=0,max=100) #scale between 0 and 255

#     # img_red = np.dstack((mimg2,np.zeros(np.shape(mimg2)),np.zeros(np.shape(mimg2))))

#     # plt.figure()
#     # plt.imshow(mimg2)
#     # img_green = np.zeros((512, 512, 3), dtype=np.uint8)
#     # img_green[:,:,1] = normalize8(mimg)

#     # masks_green, flows, styles, diams = model_green.eval(img_green, diameter=diam)
#     # outl_green = utils.outlines_list(masks_green)

#     img_red = np.zeros((512, 512, 3), dtype=np.uint8)
#     img_red[:,:,0] = normalize8(mimg2)

#     masks_cp_red, flows, styles = model_red.eval(img_red, diameter=diam, channels=chan)
#     Ncellpose_redcells      = len(np.unique(masks_cp_red))-1

#     redcell_overlap = np.empty(Nsuite2pcells)
#     for i in range(Nsuite2pcells):    # Compute overlap in masks:
#         redcell_overlap[i] = np.sum(masks_cp_red[masks_suite2p==i+1] != 0) / np.sum(masks_suite2p ==i+1)
#         # if mask_overlap_green_with_red[i]>0:
#             # mask_overlap_red_with_green[np.unique(masks_cp_red[masks_suite2p==i+1])[1]-1] = overlap

#     redcell             = redcell_overlap > overlap_threshold

#     redcell_cellpose    = np.vstack((redcell_overlap,redcell))

#     # df_green = pd.DataFrame({'overlap': mask_overlap_green_with_red})

#     # df_red  = pd.DataFrame({'overlap': mask_overlap_red_with_green})

#     clr_rchan = np.array(ImageColor.getcolor('#ff0040', "RGB")) / 255
#     clr_gchan = np.array(ImageColor.getcolor('#00ffbf', "RGB")) / 255

#     # nOnlyRedCells   = np.sum(mask_overlap_red_with_green==0)
#     # nOverlapCells   = np.sum(mask_overlap_green_with_red>0)
#     # nOnlyGreenCells = Nsuite2pcells - nOverlapCells
#     # nOnlyRedCells   = Ncellpose_redcells - nOverlapCells

#     # nTotalCells     = nOnlyGreenCells + nOnlyRedCells + nOverlapCells

#     # df = pd.DataFrame()
#     # df['suite2p']       = np.concatenate((np.full((Nsuite2pcells), True), np.full((nOnlyRedCells), False)))
#     # df['cellpose_red']  = np.concatenate((np.full((nOnlyGreenCells), False), np.full((Ncellpose_redcells), True)))
#     # df['overlap']       = np.zeros(nTotalCells)
#     # df['overlap'][np.logical_and(df['suite2p'],df['cellpose_red'])] = mask_overlap_green_with_red[mask_overlap_green_with_red>0]

#     if show_plane:
#         ######
#         lowprc = 1
#         uppprc = 99
#         rchan = (mimg2 - np.percentile(mimg2,lowprc)) / np.percentile(mimg2 - np.percentile(mimg2,lowprc),uppprc)
#         gchan = (mimg - np.percentile(mimg,lowprc)) / np.percentile(mimg - np.percentile(mimg,lowprc),uppprc)
#         bchan = np.zeros(np.shape(mimg))

#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,6))

#         ax1.imshow(gchan,cmap='gray',vmin=np.percentile(gchan,lowprc),vmax=np.percentile(gchan,uppprc))
#         ax2.imshow(rchan,cmap='gray',vmin=np.percentile(rchan,lowprc),vmax=np.percentile(rchan,uppprc))
        
#         im3 = rchan[:,:,np.newaxis] * clr_rchan + gchan[:,:,np.newaxis] * clr_gchan
#         # im3 = rchan[:,:,np.newaxis] * clr_rchan + gchan[:,:,np.newaxis] * clr_gchan
#         ax3.imshow(im3)

#         if showcells:
#             outl_green = utils.outlines_list(masks_suite2p)
#             #Filter good cells for visualization laters: 
#             # outl_green = np.array(outl_green)[iscell[:,0]==1]

#             outl_red = utils.outlines_list(masks_cp_red)
            
#             for i,o in enumerate(outl_green):
#                 if iscell[i,0]: #show only good cells
#                     ax1.plot(o[:,0], o[:,1], color='g',linewidth=0.6)
#                     # ax2.plot(o[:,0], o[:,1], color='g',linewidth=0.6)
#                     # ax3.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
#                     if redcell[i]:
#                         ax3.plot(o[:,0], o[:,1], color='#011aff',linewidth=0.6)

#             for o in outl_red:
#                 # ax1.plot(o[:,0], o[:,1], color='r',linewidth=0.6)
#                 ax2.plot(o[:,0], o[:,1], color='r',linewidth=0.6)
#                 # ax3.plot(o[:,0], o[:,1], color='y',linewidth=0.6)
#                 # ax3.plot(o[:,0], o[:,1], color='#ffe601',linewidth=0.6)
        
#         ax1.set_axis_off()
#         ax1.set_aspect('auto')
#         ax1.set_title('Suite2p cells (GCaMP)', fontsize=12, color='black', fontweight='bold',loc='center')
#         ax2.set_axis_off()
#         ax2.set_aspect('auto')
#         ax2.set_title('tdTomato cells (tdTomato)', fontsize=12, color='black', fontweight='bold',loc='center')
#         ax3.set_axis_off()
#         ax3.set_aspect('auto')
#         ax3.set_title('Labeled cells (Merge)', fontsize=12, color='black', fontweight='bold',loc='center')

#         plt.tight_layout(rect=[0, 0, 1, 1])

#     return redcell_cellpose,fig

# def proc_labeling_plane(plane_folder,show_plane=False,showcells=True):
#     stats       = np.load(os.path.join(plane_folder,'stat.npy'), allow_pickle=True)
#     ops         = np.load(os.path.join(plane_folder,'ops.npy'), allow_pickle=True).item()
#     iscell      = np.load(os.path.join(plane_folder,'iscell.npy'), allow_pickle=True)
#     redcell     = np.load(os.path.join(plane_folder,'redcell.npy'), allow_pickle=True)

#     #Filter good cells: 
#     stats       = stats[iscell[:,0]==1]
#     redcell     = redcell[iscell[:,0]==1,:]
#     # iscell    = iscell[iscell[:,0]==1]

#     Nsuite2pcells      = np.shape(redcell)[0]

#     # From cell masks create outlines:
#     masks_suite2p = np.zeros((512,512), np.float32)
#     for i,s in enumerate(stats):
#         masks_suite2p[s['ypix'],s['xpix']] = i+1
#     outl_green = utils.outlines_list(masks_suite2p)

#     # mimg = ops['meanImg']
#     mimg = np.zeros([512,512])
#     mimg[ops['yrange'][0]:ops['yrange'][1],
#         ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']

#     mimg2 = ops['meanImg_chan2']
#     # mimg2 = ops['meanImg_chan2_corrected']

#     # img_green = np.zeros((512, 512, 3), dtype=np.uint8)
#     # img_green[:,:,1] = normalize8(mimg)

#     # masks_green, flows, styles, diams = model_green.eval(img_green, diameter=diam)
#     # outl_green = utils.outlines_list(masks_green)

#     img_red = np.zeros((512, 512, 3), dtype=np.uint8)
#     img_red[:,:,0] = normalize8(mimg2)

#     masks_cp_red, flows, styles = model_red.eval(img_red, diameter=diam, channels=chan)
#     outl_red = utils.outlines_list(masks_cp_red)
#     Ncellpose_redcells      = np.shape(outl_red)[0]

#     mask_overlap_green_with_red = np.empty(Nsuite2pcells)
#     for i in range(Nsuite2pcells):    # Compute overlap in masks:
#         overlap = np.sum(masks_cp_red[masks_suite2p==i+1] != 0) / np.sum(masks_suite2p ==i+1)
#         mask_overlap_green_with_red[i] = overlap
#         # if mask_overlap_green_with_red[i]>0:
#             # mask_overlap_red_with_green[np.unique(masks_cp_red[masks_suite2p==i+1])[1]-1] = overlap

#     mask_overlap_red_with_green = np.zeros(Ncellpose_redcells)
#     for i in range(Ncellpose_redcells):    # Compute overlap in masks:
#         overlap = np.sum(masks_suite2p[masks_cp_red==i+1] != 0) / np.sum(masks_cp_red ==i+1)
#         mask_overlap_red_with_green[i] = overlap

#     df_green = pd.DataFrame({'overlap': mask_overlap_green_with_red})

#     df_red  = pd.DataFrame({'overlap': mask_overlap_red_with_green})

#     clr_rchan = np.array(ImageColor.getcolor('#ff0040', "RGB")) / 255
#     clr_gchan = np.array(ImageColor.getcolor('#00ffbf', "RGB")) / 255

#     # nOnlyRedCells   = np.sum(mask_overlap_red_with_green==0)
#     # nOverlapCells   = np.sum(mask_overlap_green_with_red>0)
#     # nOnlyGreenCells = Nsuite2pcells - nOverlapCells
#     # nOnlyRedCells   = Ncellpose_redcells - nOverlapCells

#     # nTotalCells     = nOnlyGreenCells + nOnlyRedCells + nOverlapCells

#     # df = pd.DataFrame()
#     # df['suite2p']       = np.concatenate((np.full((Nsuite2pcells), True), np.full((nOnlyRedCells), False)))
#     # df['cellpose_red']  = np.concatenate((np.full((nOnlyGreenCells), False), np.full((Ncellpose_redcells), True)))
#     # df['overlap']       = np.zeros(nTotalCells)
#     # df['overlap'][np.logical_and(df['suite2p'],df['cellpose_red'])] = mask_overlap_green_with_red[mask_overlap_green_with_red>0]

#     if show_plane:
#         ######
#         lowprc = 1
#         uppprc = 99
#         rchan = (mimg2 - np.percentile(mimg2,lowprc)) / np.percentile(mimg2 - np.percentile(mimg2,lowprc),uppprc)
#         gchan = (mimg - np.percentile(mimg,lowprc)) / np.percentile(mimg - np.percentile(mimg,lowprc),uppprc)
#         bchan = np.zeros(np.shape(mimg))

#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,6))

#         ax1.imshow(gchan,cmap='gray',vmin=np.percentile(gchan,lowprc),vmax=np.percentile(gchan,uppprc))
#         ax2.imshow(rchan,cmap='gray',vmin=np.percentile(rchan,lowprc),vmax=np.percentile(rchan,uppprc))
        
#         im3 = rchan[:,:,np.newaxis] * clr_rchan + gchan[:,:,np.newaxis] * clr_gchan
#         ax3.imshow(im3)

#         if showcells:
#             for o in outl_green:
#                 ax1.plot(o[:,0], o[:,1], color='g',linewidth=0.6)
#                 ax2.plot(o[:,0], o[:,1], color='g',linewidth=0.6)
#                 ax3.plot(o[:,0], o[:,1], color='w',linewidth=0.6)
#                 ax3.plot(o[:,0], o[:,1], color='#011aff',linewidth=0.6)

#             for o in outl_red:
#                 ax1.plot(o[:,0], o[:,1], color='r',linewidth=0.6)
#                 ax2.plot(o[:,0], o[:,1], color='r',linewidth=0.6)
#                 # ax3.plot(o[:,0], o[:,1], color='y',linewidth=0.6)
#                 ax3.plot(o[:,0], o[:,1], color='#ffe601',linewidth=0.6)

#         ax1.set_axis_off()
#         ax1.set_aspect('auto')
#         ax1.set_title('GCaMP', fontsize=12, color='black', fontweight='bold',loc='center')
#         ax2.set_axis_off()
#         ax2.set_aspect('auto')
#         ax2.set_title('tdTomato', fontsize=12, color='black', fontweight='bold',loc='center')
#         ax3.set_axis_off()
#         ax3.set_aspect('auto')
#         ax3.set_title('Merge', fontsize=12, color='black', fontweight='bold',loc='center')

#         plt.tight_layout(rect=[0, 0, 1, 1])

#     np.save('redcell_cellpose')

#     return df_green,df_red,fig




# direc = 'X:\\RawData\\LPE09829\\2023_03_31\\suite2p\\'
# direc = 'X:\\RawData\\LPE09830\\2023_04_10\\suite2p\\'
# direc = 'O:\\RawData\\LPE09830\\2023_04_12\\suite2p\\'
# direc = 'O:\\RawData\\LPE09665\\2023_03_15\\suite2p\\'
# direc = 'X:\\RawData\\LPE09667\\2023_03_30\\suite2p\\'
# direc = 'H:\\RawData\\LPE09667\\2023_03_30\\suite2p\\'

# from matplotlib_venn import venn2
# import matplotlib.pyplot as plt

# overlap_threshold = 0.2

# # Use the venn2 function
# G = np.sum(df_green['overlap']<=overlap_threshold)
# GR = np.sum(df_green['overlap']>overlap_threshold)
# R = np.sum(df_red['overlap']<=overlap_threshold)
# venn2(subsets = (G, R, GR), set_labels = ('Suite2p', 'cellpose'))

# fig = plt.figure(figsize=[5, 4])
# sns.histplot(data=df_green, x="overlap",stat='probability',binwidth=0.025)
# plt.ylim([0,0.02])
# plt.xlabel("Overlap gcamp with tdtomato")

# fig = plt.figure(figsize=[5, 4])
# sns.histplot(data=df_red, x="overlap",stat='probability',binwidth=0.025)
# plt.ylim([0,0.15])
# plt.xlabel("Overlap tdtomato with gcamp")

# # Use the venn2 function
# G = np.sum(np.logical_and(df['suite2p'],~df['cellpose_red']))
# GR = np.sum(np.logical_and(df['suite2p'],df['cellpose_red']))
# R = np.sum(np.logical_and(~df['suite2p'],df['cellpose_red']))
# venn2(subsets = (G, R, GR), set_labels = ('Suite2p', 'cellpose'))

# # df2 = df.copy()

# # df2.loc[df2.index[np.logical_and(df2['overlap']<0.1,df2['suite2p'])],'cellpose_red'] = False


# fig = plt.figure(figsize=[5, 4])
# sns.histplot(data=df, x="overlap",stat='probability',binwidth=0.025)
# plt.ylim([0,0.02])

# overlap_threshold = 0.2
# # Use the venn2 function
# G = np.sum(np.logical_and(df['suite2p'],df['overlap']<overlap_threshold))
# GR = np.sum(df['overlap']>overlap_threshold)
# R = np.sum(np.logical_and(df['cellpose_red'],df['overlap']<overlap_threshold))
# venn2(subsets = (G, R, GR), set_labels = ('Suite2p', 'cellpose'))


# x, batch_size=8, channels=None, channel_axis=None, 
#              z_axis=None, normalize=True, invert=False, 
#              rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=False, 
#              augment=False, tile=True, tile_overlap=0.1,
#              resample=True, interp=True,
#              flow_threshold=0.4, cellprob_threshold=0.0,
#              compute_masks=True, min_size=15, stitch_threshold=0.0, progress=None,  
#              loop_run=False, model_loaded=False):

# masks, flows, styles, diams = model.eval(img, diameter=diam, channels=chan)

# masks, flows, styles, diams = model.eval(img_numpy, diameter=diam, channels=chan)

# save results so you can load in gui
# io.masks_flows_to_seg(imgs, masks, flows, diams, filename, channels)




    #####Compute intensity ratio (code taken from Suite2p):
            #redstats = intensity_ratio(ops, stats)

    # Ly, Lx = ops['Ly'], ops['Lx']
    # cell_pix = masks.create_cell_pix(stats, Ly=ops['Ly'], Lx=ops['Lx'])
    # cell_masks0 = [masks.create_cell_mask(stat, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap']) for stat in stats]
    # neuropil_ipix = masks.create_neuropil_masks(
    #     ypixs=[stat['ypix'] for stat in stats],
    #     xpixs=[stat['xpix'] for stat in stats],
    #     cell_pix=cell_pix,
    #     inner_neuropil_radius=ops['inner_neuropil_radius'],
    #     min_neuropil_pixels=ops['min_neuropil_pixels'],
    # )
    # cell_masks = np.zeros((len(stats), Ly * Lx), np.float32)
    # neuropil_masks = np.zeros((len(stats), Ly * Lx), np.float32)
    # for cell_mask, cell_mask0, neuropil_mask, neuropil_mask0 in zip(cell_masks, cell_masks0, neuropil_masks, neuropil_ipix):
    #     cell_mask[cell_mask0[0]] = cell_mask0[1]
    #     neuropil_mask[neuropil_mask0.astype(np.int64)] = 1. / len(neuropil_mask0)




        # redcells        = np.where(np.logical_and(iscell[:,0],redcell))[0]
        # if any(redcells):
        #     nmsk_red        = np.reshape(cell_masks[redcells,:],[len(redcells),512,512])
        #     nmsk_red        = np.max(nmsk_red,axis=0) > 0
        #     # ax2.imshow(nmsk_red,cmap='Reds',alpha=nmsk_red/np.max(nmsk_red)*0.5)
        #     # ax3.imshow(nmsk_red,cmap='Reds',alpha=nmsk_red/np.max(nmsk_red)*0.5)

        #     ax1.scatter(x[redcells],y[redcells],s=25,facecolors='none', edgecolors='r',linewidths=0.6)
        #     ax2.scatter(x[redcells],y[redcells],s=25,facecolors='none', edgecolors='r',linewidths=0.6)
        #     ax3.scatter(x[redcells],y[redcells],s=25,facecolors='none', edgecolors='w',linewidths=0.6)
        #     # ax3.arrow(x[redcells],y[redcells],10,10,color='yellow',head_starts_at_zero=True)
        #     ax3.quiver(x[redcells]+2,y[redcells]-2,-4,-4,color='yellow',width=0.007,headlength=4,headwidth=2,pivot='tip')

        # notredcells     = np.where(np.logical_and(iscell[:,0],np.logical_not(redcell)))[0]
        # if any(notredcells):
        #     nmsk_notred     = np.reshape(cell_masks[notredcells,:],[len(notredcells),512,512])
        #     nmsk_notred     = np.max(nmsk_notred,axis=0) > 0

        #     # ax2.imshow(nmsk_notred,cmap='Greens',alpha=nmsk_notred/np.max(nmsk_notred)*0.5)
        #     # ax3.imshow(nmsk_notred,cmap='Greens',alpha=nmsk_notred/np.max(nmsk_notred)*0.5)

        #     ax1.scatter(x[notredcells],y[notredcells],s=25,facecolors='none', edgecolors='g',linewidths=0.4)
        #     ax2.scatter(x[notredcells],y[notredcells],s=25,facecolors='none', edgecolors='g',linewidths=0.4)
        #     ax3.scatter(x[notredcells],y[notredcells],s=25,facecolors='none', edgecolors='w',linewidths=0.4)




 # inpix = cell_masks @ mimg2.flatten()
    # extpix = neuropil_masks @ mimg2.flatten()
    # # extpix = np.mean(extpix)
    # inpix = np.maximum(1e-3, inpix)
    # redprob = inpix / (inpix + extpix)
    # redcell = redprob > ops['chan2_thres']

    # redcell = inpix > 130