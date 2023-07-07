# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:16:24 2023

@author: USER
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from suite2p.extraction import extract, masks
from suite2p.detection.chan2detect import detect,correct_bleedthrough

direc = 'X:\\RawData\\LPE09829\\2023_03_30\\suite2p\\combined'
direc = 'X:\\RawData\\LPE09829\\2023_03_30\\suite2p\\plane5'
direc = 'X:\\RawData\\LPE09829\\2023_03_30\\suite2p\\plane0'

direc = 'X:\\RawData\\LPE09830\\2023_04_10\\suite2p\\plane0'
# direc = 'X:\\RawData\\LPE09830\\2023_04_10\\suite2p\\plane4'

# direc = 'X:\\RawData\\LPE09665\\2023_03_14\\suite2p\\plane4'


os.chdir(direc)

# F = np.load('F.npy', allow_pickle=True)
# Fneu = np.load('Fneu.npy', allow_pickle=True)
# spks = np.load('spks.npy', allow_pickle=True)
stats = np.load('stat.npy', allow_pickle=True)
ops =  np.load('ops.npy', allow_pickle=True).item()
iscell = np.load('iscell.npy', allow_pickle=True)
redcell = np.load('redcell.npy', allow_pickle=True)

# [ops, redstats] = detect(ops, stats)

Ncells = np.shape(redcell)[0]

#####Compute intensity ratio (code taken from Suite2p):
        #redstats = intensity_ratio(ops, stats)

Ly, Lx = ops['Ly'], ops['Lx']
cell_pix = masks.create_cell_pix(stats, Ly=ops['Ly'], Lx=ops['Lx'])
cell_masks0 = [masks.create_cell_mask(stat, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap']) for stat in stats]
neuropil_ipix = masks.create_neuropil_masks(
    ypixs=[stat['ypix'] for stat in stats],
    xpixs=[stat['xpix'] for stat in stats],
    cell_pix=cell_pix,
    inner_neuropil_radius=ops['inner_neuropil_radius'],
    min_neuropil_pixels=ops['min_neuropil_pixels'],
)
cell_masks = np.zeros((len(stats), Ly * Lx), np.float32)
neuropil_masks = np.zeros((len(stats), Ly * Lx), np.float32)
for cell_mask, cell_mask0, neuropil_mask, neuropil_mask0 in zip(cell_masks, cell_masks0, neuropil_masks, neuropil_ipix):
    cell_mask[cell_mask0[0]] = cell_mask0[1]
    neuropil_mask[neuropil_mask0.astype(np.int64)] = 1. / len(neuropil_mask0)


# mimg = ops['max_proj']
mimg = np.zeros([512,512])
mimg[ops['yrange'][0]:ops['yrange'][1],
     ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']

mimg = ops['meanImg']
mimg2 = ops['meanImg_chan2']
mimg2 = ops['meanImg_chan2_corrected']

# mimg2 = correct_bleedthrough(Ly, Lx, 3, mimg, mimg2)

inpix = cell_masks @ mimg2.flatten()
extpix = neuropil_masks @ mimg2.flatten()
inpix = np.maximum(1e-3, inpix)
redprob = inpix / (inpix + extpix)
redcell = redprob > ops['chan2_thres']

################## Show full plane figure:

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2,figsize=(9,8))

ax1.imshow(mimg,cmap='gray',vmin=np.percentile(mimg,3),vmax=np.percentile(mimg,99))
ax2.imshow(mimg2,cmap='gray',vmin=np.percentile(mimg2,3),vmax=np.percentile(mimg2,99))

ax1.set_axis_off()
ax1.set_aspect('auto')
ax1.set_title('GCaMP', fontsize=12, color='black', fontweight='bold',loc='center')
ax2.set_axis_off()
ax2.set_aspect('auto')
ax2.set_title('tdTomato', fontsize=12, color='black', fontweight='bold',loc='center')

ax3.imshow(mimg,cmap='gray',vmin=np.percentile(mimg,3),vmax=np.percentile(mimg,99))
ax4.imshow(mimg2,cmap='gray',vmin=np.percentile(mimg2,3),vmax=np.percentile(mimg2,99))

redcells        = np.where(np.logical_and(iscell[:,0],redcell))[0]
notredcells     = np.where(np.logical_and(iscell[:,0],np.logical_not(redcell)))[0]

nmsk_red = np.reshape(cell_masks[redcells,:],[len(redcells),512,512])
nmsk_notred = np.reshape(cell_masks[notredcells,:],[len(notredcells),512,512])

nmsk_red = np.max(nmsk_red,axis=0) > 0
nmsk_notred = np.max(nmsk_notred,axis=0) > 0

ax3.imshow(nmsk_red,cmap='Reds',alpha=nmsk_red/np.max(nmsk_red)*0.6)
ax4.imshow(nmsk_red,cmap='Reds',alpha=nmsk_red/np.max(nmsk_red)*0.6)

ax3.imshow(nmsk_notred,cmap='Blues',alpha=nmsk_notred/np.max(nmsk_notred)*0.6)
ax4.imshow(nmsk_notred,cmap='Blues',alpha=nmsk_notred/np.max(nmsk_notred)*0.6)

ax3.set_axis_off()
ax3.set_aspect('auto')
# ax3.set_title('GCaMP', fontsize=12, color='black', fontweight='bold',loc='center')
ax4.set_axis_off()
ax4.set_aspect('auto')
# ax4.set_title('tdTomato', fontsize=12, color='black', fontweight='bold',loc='center')
plt.tight_layout(rect=[0, 0, 1, 1])

####### Show close ups of example cells:

# example_cells = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110]

# example_cells = range(Ncells)

# example_cells = np.where(np.logical_and(np.logical_and(iscell[:,0],redprob>0.63),redprob<0.7))[0]

example_cells = np.where(np.logical_and(iscell[:,0],np.logical_or(redprob<np.percentile(redprob,7),redprob>np.percentile(redprob,93))))[0]

#Sort based on redcell probability:
arr1inds                = redprob[example_cells].argsort()
example_cells           = example_cells[arr1inds]

Tot         = len(example_cells)*2
Rows        = int(np.floor(np.sqrt(Tot)))
Cols        = Tot // Rows # Compute Rows required
if Tot % Rows != 0: #If one additional row is necessary -> add one:
    Cols += 1
# Position = range(1,Tot + 1) # Create a Position index
Position = range(1,Tot + 1) # Create a Position index

fig = plt.figure(figsize=[12, 8])
for i,n in enumerate(example_cells):
    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[i*2])
    
    nmsk = np.reshape(cell_masks[n,:],[512,512])
    npil = np.reshape(neuropil_masks[n,:],[512,512])
    
    # ax.imshow(mimg,cmap='Greens',vmin=np.percentile(mimg,2),vmax=np.percentile(mimg,99))
    ax.imshow(mimg,cmap='gray',vmin=np.percentile(mimg,3),vmax=np.percentile(mimg,99))
    ax.imshow(npil,cmap='Greens',alpha=npil/np.max(npil)*0.4)
    ax.imshow(nmsk,cmap='Blues',alpha=nmsk/np.max(nmsk)*0.8)
    
    ax.set_axis_off()
    ax.set_aspect('auto')
    # ax.set_title(n, fontsize=2, color='black', fontweight='bold',loc='center',x=0.5, y=1.1);
    ax.set_title(n, fontsize=10, color='black', fontweight='bold',loc='center');

    [x,y] = np.where(npil>0)
    ax.set_ylim([np.min(x)-2,np.max(x)+2])
    ax.set_xlim([np.min(y)-2,np.max(y)+2])
    
    ax = fig.add_subplot(Rows,Cols,Position[i*2 + 1])
        
    ax.imshow(mimg2,cmap='gray',vmin=np.percentile(mimg2,3),vmax=np.percentile(mimg2,99))
    ax.imshow(npil,cmap='Greens',alpha=npil/np.max(npil)*0.4)
    ax.imshow(nmsk,cmap='Blues',alpha=nmsk/np.max(nmsk)*0.8)

    ax.set_axis_off()
    ax.set_aspect('auto')
    if redcell[n]:
        ax.set_title("%0.2f, %s" % (redprob[n],redcell[n]), fontsize=10, color='green', fontweight='bold',loc='center');
    else:
        ax.set_title("%0.2f, %s" % (redprob[n],redcell[n]), fontsize=10, color='red', fontweight='bold',loc='center');

    [x,y] = np.where(npil>0)
    ax.set_ylim([np.min(x)-2,np.max(x)+2])
    ax.set_xlim([np.min(y)-2,np.max(y)+2])
    
plt.tight_layout(rect=[0, 0, 1, 1])

############### Show threshold of intensity ratio:
# redprob = inpix / (inpix + extpix)
# redcell = redprob > ops['chan2_thres']

redstats = {'redcell': redcell[iscell[:,0]==1],'redcellprob': redprob[iscell[:,0]==1]}
redstats = pd.DataFrame(data=redstats)



fig = plt.figure(figsize=[5, 4])
sns.histplot(data=redstats, x="redcellprob",hue='redcell',stat='count',binwidth=0.025)


