

import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread

from cellpose import utils, io

import pandas as pd
import seaborn as sns
from suite2p.extraction import extract, masks
from suite2p.detection.chan2detect import detect,correct_bleedthrough

from PIL import Image

chan = [[1,0]] # grayscale=0, R=1, G=2, B=3 # channels = [cytoplasm, nucleus]
diam = 12

# model_type='cyto' or 'nuclei' or 'cyto2'
# model = models.Cellpose(model_type='cyto')
model_red = models.CellposeModel(pretrained_model = 'T:\\Python\\cellpose\\testdir\\models\\MOL_20230814_redcells')

model_green = models.Cellpose(model_type='cyto')

def normalize8(I):
    # mn = I.min()
    # mx = I.max()

    mn = np.percentile(I,0.5)
    mx = np.percentile(I,99.5)

    mx -= mn

    I = ((I - mn)/mx) * 255
    I[I<0] =0
    I[I>255] = 255
    
    return I.astype(np.uint8)

def proc_labeling_plane(direc_folder,show_plane=False):
    stats = np.load(os.path.join(direc_folder,'stat.npy'), allow_pickle=True)
    ops =  np.load(os.path.join(direc_folder,'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(direc_folder,'iscell.npy'), allow_pickle=True)
    redcell = np.load(os.path.join(direc_folder,'redcell.npy'), allow_pickle=True)

    #Filter good cells: 
    stats = stats[iscell[:,0]==1]
    redcell = redcell[iscell[:,0]==1,:]
    # iscell = iscell[iscell[:,0]==1]

    Ncells = np.shape(redcell)[0]

    # From cell masks create outlines:
    masks_suite2p = np.zeros((512,512), np.float32)
    for i,s in enumerate(stats):
        masks_suite2p[s['ypix'],s['xpix']] = i+1
    outl_green = utils.outlines_list(masks_suite2p)



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

    mimg = ops['meanImg']
    # mimg = np.zeros([512,512])
    # mimg[ops['yrange'][0]:ops['yrange'][1],
        # ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']

    mimg2 = ops['meanImg_chan2']
    # mimg2 = ops['meanImg_chan2_corrected']
    # mimg2 = correct_bleedthrough(Ly, Lx, 3, mimg, mimg2)


    # img_green = np.zeros((512, 512, 3), dtype=np.uint8)
    # img_green[:,:,1] = normalize8(mimg)

    # masks_green, flows, styles, diams = model_green.eval(img_green, diameter=diam)
    # outl_green = utils.outlines_list(masks_green)

    img_red = np.zeros((512, 512, 3), dtype=np.uint8)
    img_red[:,:,0] = normalize8(mimg2)

    masks_cp_red, flows, styles = model_red.eval(img_red, diameter=diam, channels=chan)
    outl_red = utils.outlines_list(masks_cp_red)

    mask_overlap = np.empty(Ncells)
    # Compute overlap in masks:
    for i in range(Ncells):
        # mask_overlap[i] = np.sum(masks_cp_red[masks_suite2p==i+1] != 0) / np.sum()
        mask_overlap[i] = np.sum(masks_cp_red[masks_suite2p==i+1] != 0) / np.sum(masks_suite2p ==i+1)
        # print(np.sum(masks_cp_red[masks_suite2p==i+1] != 0))
    
    mask_overlap = mask_overlap.round(2)
    # inpix = cell_masks @ mimg2.flatten()
    # extpix = neuropil_masks @ mimg2.flatten()
    # # extpix = np.mean(extpix)
    # inpix = np.maximum(1e-3, inpix)
    # redprob = inpix / (inpix + extpix)
    # redcell = redprob > ops['chan2_thres']

    # redcell = inpix > 130

    df = pd.DataFrame()
    # df['inpix'] = inpix
    # df['extpix'] = extpix
    df['redprob'] = mask_overlap
    df['redcell'] = mask_overlap > 0.1

    if show_plane:
        ######
        lowprc = 1
        uppprc = 99
        rchan = (mimg2 - np.percentile(mimg2,lowprc)) / np.percentile(mimg2 - np.percentile(mimg2,lowprc),uppprc)
        gchan = (mimg - np.percentile(mimg,lowprc)) / np.percentile(mimg - np.percentile(mimg,lowprc),uppprc)
        bchan = np.zeros(np.shape(mimg))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,6))

        ax1.imshow(gchan,cmap='gray',vmin=np.percentile(gchan,lowprc),vmax=np.percentile(gchan,uppprc))
        ax2.imshow(rchan,cmap='gray',vmin=np.percentile(rchan,lowprc),vmax=np.percentile(rchan,uppprc))
        ax3.imshow(np.dstack((rchan,gchan,bchan)))
        # ax3.imshow(np.dstack((rchan,bchan,bchan)))
        # ax3.imshow(np.dstack((rchan,bchan,bchan)),cmap='gray',vmin=np.percentile(mimg2,3),vmax=np.percentile(mimg2,99))

        # x =  np.array([stats[i]['med'][1] for i in range(Ncells)])
        # y =  np.array([stats[i]['med'][0] for i in range(Ncells)])

        # plot image with outlines overlaid in white

        # plt.figure(figsize=(12,12))
        # plt.imshow(ops['meanImgE'])
        # plt.imshow(ops['max_proj'])

        for o in outl_green:
            ax1.plot(o[:,0], o[:,1], color='g',linewidth=0.6)
            ax2.plot(o[:,0], o[:,1], color='g',linewidth=0.6)
            ax3.plot(o[:,0], o[:,1], color='w',linewidth=0.6)

        for o in outl_red:
            ax1.plot(o[:,0], o[:,1], color='r',linewidth=0.6)
            ax2.plot(o[:,0], o[:,1], color='r',linewidth=0.6)
            ax3.plot(o[:,0], o[:,1], color='y',linewidth=0.6)
        


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

        ax1.set_axis_off()
        ax1.set_aspect('auto')
        ax1.set_title('GCaMP', fontsize=12, color='black', fontweight='bold',loc='center')
        ax2.set_axis_off()
        ax2.set_aspect('auto')
        ax2.set_title('tdTomato', fontsize=12, color='black', fontweight='bold',loc='center')
        ax3.set_axis_off()
        ax3.set_aspect('auto')
        ax3.set_title('Merge', fontsize=12, color='black', fontweight='bold',loc='center')

        plt.tight_layout(rect=[0, 0, 1, 1])

        # fig.savefig('labeling.jpg',dpi=600)


    return mimg, mimg2, df


direc = 'X:\\RawData\\LPE09829\\2023_03_30\\suite2p\\'

direc = 'O:\\RawData\\LPE09665\\2023_03_14\\suite2p\\'

for iplane in range(8):

    mimg,mimg2,tempdf = proc_labeling_plane(os.path.join(direc,"plane%s" % iplane),show_plane=True)
    tempdf['iplane'] = iplane
    if iplane == 0:
        df = tempdf
    else:
        df = df.append(tempdf)



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



# plt.imshow(outlines)
# plt.imshow(img,vmin=0,vmax=255)


# save results so you can load in gui
io.masks_flows_to_seg(imgs, masks, flows, diams, filename, channels)

# redstats = {'redcell': redcell[iscell[:,0]==1],'redcellprob': redprob[iscell[:,0]==1]}
# redstats = pd.DataFrame(data=redstats)

fig = plt.figure(figsize=[5, 4])
sns.histplot(data=df, x="redprob",hue='redcell',stat='count',binwidth=0.025)

sns.scatterplot(data=df, y="redprob",x='inpix',hue='redcell')
plt.xscale('log')
sns.scatterplot(data=df, y="extpix",x='inpix',hue='redcell')

sns.histplot(data=df,x='inpix', stat='count',hue='redcell',log_scale=True)