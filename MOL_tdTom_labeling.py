# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:55:14 2023

@author: USER
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from suite2p.extraction import extract, masks

direc = 'X:\\RawData\\NSH07422\\2022_12_9\\VR\\Imaging\\ROI_1\\suite2p\\plane0'

os.chdir(direc)

F = np.load('F.npy', allow_pickle=True)
Fneu = np.load('Fneu.npy', allow_pickle=True)
spks = np.load('spks.npy', allow_pickle=True)
stat = np.load('stat.npy', allow_pickle=True)
ops =  np.load('ops.npy', allow_pickle=True).item()
iscell = np.load('iscell.npy', allow_pickle=True)

ops = extract.enhanced_mean_image_chan2(ops)

###################################################
## To show the different images constructed from the tiff during suite2p analyses
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(10,6.5))

ax1.imshow(ops['meanImg'])
ax1.set_title('Mean Image Chan 1')

ax2.imshow(ops['meanImg'])
ax2.set_title('Mean Corrected Image Chan 1')

ax3.imshow(ops['meanImgE'])
ax3.set_title('Enhanced Image Chan 1')

ax4.imshow(ops['meanImg_chan2'])
ax4.set_title('Mean Image Chan 2')

ax5.imshow(ops['meanImg_chan2_corrected'])
ax5.set_title('Corrected Image Chan 2')

ax6.imshow(ops['meanImgE_chan2'])
ax6.set_title('Enhanced Image Chan 2')

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

### 
start = 100
stop = 200
ax1.set_xlim([start,stop])
ax1.set_ylim([start,stop])
ax2.set_xlim([start,stop])
ax2.set_ylim([start,stop])
ax3.set_xlim([start,stop])
ax3.set_ylim([start,stop])
ax4.set_xlim([start,stop])
ax4.set_ylim([start,stop])
ax5.set_xlim([start,stop])
ax5.set_ylim([start,stop])
ax6.set_xlim([start,stop])
ax6.set_ylim([start,stop])

###################################################
## 
mean_chan0 = ops['meanImgE']
mean_chan1 = ops['meanImgE_chan2']
# mean_chan0 = ops['meanImg']
# mean_chan1 = ops['meanImg_chan2']

ncells = np.shape(F)[0]
im = np.zeros((ops['Ly'], ops['Lx']))

maskF_chan0  = np.zeros([ncells,1],dtype=np.float32)
maskF_chan1  = np.zeros([ncells,1],dtype=np.float32)

for n in range(0,ncells):
    if iscell[n,0]==1:
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        im[ypix,xpix] = n+1
        maskF_chan0[n] = np.average(mean_chan0[ypix,xpix])
        maskF_chan1[n] = np.average(mean_chan1[ypix,xpix])
        # maskF_chan0[n] = np.max(mean_chan0[ypix,xpix])
        # maskF_chan1[n] = np.max(mean_chan1[ypix,xpix])

im[im == 0.0] = np.nan

####
## Show 
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(10,6.5))

ax1.imshow(mean_chan0,vmin = np.nanmin(mean_chan0),vmax = np.nanmax(mean_chan0))
ax1.set_title('Enhanced Image Chan 1')

ax2.imshow(im,cmap='Set1')
temp = mean_chan0.copy()
temp[np.isnan(im)] = 0
ax2.set_title('ROIs')

ax3.imshow(temp,vmin = np.nanmin(mean_chan0),vmax = np.nanmax(mean_chan0))
ax3.set_title('Masked Image Chan 1')

ax4.imshow(mean_chan1,vmin = np.nanmin(mean_chan1),vmax = np.nanmax(mean_chan1))
ax4.set_title('Enhanced Image Chan 2')

ax5.imshow(im,cmap='Set1')
ax5.set_title('ROIs')

temp = mean_chan1.copy()
temp[np.isnan(im)] = 0
ax6.imshow(temp,vmin = np.nanmin(mean_chan1),vmax = np.nanmax(mean_chan1))
ax6.set_title('Masked Image Chan 2')

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

### for close up:
xstart = 100
xstop = 200
ystart = 100
ystop = 200
ax1.set_xlim([xstart,xstop])
ax1.set_ylim([ystart,ystop])
ax2.set_xlim([xstart,xstop])
ax2.set_ylim([ystart,ystop])
ax3.set_xlim([xstart,xstop])
ax3.set_ylim([ystart,ystop])
ax4.set_xlim([xstart,xstop])
ax4.set_ylim([ystart,ystop])
ax5.set_xlim([xstart,xstop])
ax5.set_ylim([ystart,ystop])
ax6.set_xlim([xstart,xstop])
ax6.set_ylim([ystart,ystop])

##########################################


fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(9,4))
ax1.scatter(maskF_chan0[iscell[:,0]==1],maskF_chan1[iscell[:,0]==1],s=10,c=[[0.2,0.6,0.7]])
temp = [maskF_chan0[iscell[:,0]==1],maskF_chan1[iscell[:,0]==1]]
ax1.set_xlabel('GCaMP Fluorescence')
ax1.set_ylabel('tdTomato Fluorescence')

# the histogram of the data
n, bins, patches = ax2.hist(maskF_chan1[iscell[:,0]==1], 50,
                            density = False, 
                            facecolor ='g', 
                            alpha = 0.75)

threshold = 0.75
threshold = 1300

ax2.axvline(threshold, color='r', linestyle='dashed')  
ax2.text(threshold*1.05, 8, 'Fluorescence Cutoff', fontsize=8)

tdTom_labeled = maskF_chan1>threshold

frac_labeled = tdTom_labeled 

n_labeled = np.logical_and(tdTom_labeled.transpose(), iscell[:,0]==1).sum()

n_unlabeled = np.logical_and(tdTom_labeled.transpose()==False, iscell[:,0]==1).sum()

txt1 = "Labeled {lab}/{tot}".format(lab = n_labeled, tot = n_unlabeled + n_labeled)
ax2.text(threshold*1.05, 6,txt1, fontsize=8)
ax2.set_xlabel('tdTomato Fluorescence')
ax2.set_xlabel('#Neurons')

########





