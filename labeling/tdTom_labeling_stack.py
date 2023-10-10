"""
Analyzes stack of GCaMP and tdtomato expressing cells using cellpose software (Pachitariu & Stringer)
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import ImageColor
from PIL import Image
from ScanImageTiffReader import ScanImageTiffReader as imread
from pathlib import Path
from matplotlib.patches import Rectangle

from cellpose import models
from cellpose import utils, io
# from cellpose.io import imread

# import matplotlib as mpl
# mpl.rcParams["figure.facecolor"] = 'w'
# mpl.rcParams["axes.facecolor"]  = 'w'
# mpl.rcParams["savefig.facecolor"]  = 'w'

# from suite2p.extraction import extract, masks
# from suite2p.detection.chan2detect import detect,correct_bleedthrough

savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Neural - Labeling\\Stack\\'

chan = [[1,0]] # grayscale=0, R=1, G=2, B=3 # channels = [cytoplasm, nucleus]
diam = 12

# model_type='cyto' or 'nuclei' or 'cyto2'
# model = models.Cellpose(model_type='cyto')
model_red = models.CellposeModel(pretrained_model = 'T:\\Python\\cellpose\\testdir\\models\\MOL_20230814_redcells')

model_green = models.Cellpose(model_type='cyto')


def normalize8(I):
    mn = I.min()
    mx = I.max()

    # mn = np.percentile(I,0.5)
    # mx = np.percentile(I,99.5)

    mx -= mn

    I = ((I - mn)/mx) * 255
    I[I<0] =0
    I[I>255] = 255
    
    return I.astype(np.uint8)



direc = 'X:\\RawData\\LPE09829\\2023_03_31\\suite2p\\'


direc = 'W:\\Users\\Matthijs\\Rawdata\\LPE10192\\2023_10_10\\'

nslices         = len(os.listdir(os.path.join(direc,'STACK_PM')))
slicedepths     = np.linspace(0,10*(nslices-1),nslices)
nchannels       = 2 #whether image has both red and green channel acquisition (PMT)

### V1 stack loading:
greenstack_V1  = np.empty([512,512,nslices])
redstack_V1    = np.empty([512,512,nslices])
for i,x in enumerate(os.listdir(os.path.join(direc,'STACK_V1'))):
    if x.endswith(".tif"):
            fname = Path(os.path.join(direc,'STACK_V1',x))
            reader = imread(str(fname)) # amazing - this librarty needs str
            Data = reader.data()
            greenstack_V1[:,:,i] = np.average(Data[0::2,:,:], axis=0)
            redstack_V1[:,:,i] = np.average(Data[1::2,:,:], axis=0)

### PM stack loading:
greenstack_PM  = np.empty([512,512,nslices])
redstack_PM    = np.empty([512,512,nslices])

for i,x in enumerate(os.listdir(os.path.join(direc,'STACK_PM'))):
    if x.endswith(".tif"):
            fname = Path(os.path.join(direc,'STACK_PM',x))
            reader = imread(str(fname)) # amazing - this librarty needs str
            Data = reader.data()
            greenstack_PM[:,:,i] = np.average(Data[0::2,:,:], axis=0)
            redstack_PM[:,:,i] = np.average(Data[1::2,:,:], axis=0)



chan = [[1,0]] # grayscale=0, R=1, G=2, B=3 # channels = [cytoplasm, nucleus]
diam = 12

# model_type='cyto' or 'nuclei' or 'cyto2'
model_red = models.CellposeModel(pretrained_model = 'T:\\Python\\cellpose\\testdir\\models\\MOL_20230814_redcells')


nTdTomCells_V1       = np.zeros(nslices)
nTdTomCells_PM       = np.zeros(nslices)

### Get number of tdTomato labeled cells (using cellpose):
for i in range(nslices): 
    print(i)
    img_red_V1 = np.zeros((512, 512, 3), dtype=np.uint8)
    img_red_V1[:,:,0] = normalize8(redstack_V1[:,:,i])

    masks_cp_red, flows, styles = model_red.eval(img_red_V1, diameter=diam, channels=chan)
    nTdTomCells_V1[i]      = len(np.unique(masks_cp_red))-1 #zero is counted as unique

    img_red_PM = np.zeros((512, 512, 3), dtype=np.uint8)
    img_red_PM[:,:,0] = normalize8(redstack_PM[:,:,i])

    masks_cp_red, flows, styles = model_red.eval(img_red_PM, diameter=diam, channels=chan)
    nTdTomCells_PM[i]      = len(np.unique(masks_cp_red))-1 #zero is counted as unique


### Get the mean fluorescence for each plane:
meanF2_V1       = [np.mean(redstack_V1[:,:,i]) for i in range(nslices)]
meanF1_V1       = [np.mean(greenstack_V1[:,:,i]) for i in range(nslices)]

meanF2_PM       = [np.mean(redstack_PM[:,:,i]) for i in range(nslices)]
meanF1_PM       = [np.mean(greenstack_PM[:,:,i]) for i in range(nslices)]


### Figure with depth profile: 
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,7))
ax1.plot(meanF2_V1,slicedepths,c='darkviolet')
ax1.plot(meanF2_PM,slicedepths,c='lightseagreen')
ax1.invert_yaxis()
ax1.set_ylabel('Cortical Depth')
ax1.set_xlabel('Fluorescence')
# ax1.legend(['V1','PM'],frameon=False)
ax1.set_title('Fluorescence')

ax2.plot(nTdTomCells_V1,slicedepths,c='darkviolet')
ax2.plot(nTdTomCells_PM,slicedepths,c='lightseagreen')
ax2.invert_yaxis()
ax2.set_ylabel('')
ax2.set_xlabel('#Labeled cells')
ax2.legend(['V1','PM'],frameon=False)
ax2.set_title('#Labeled cells')
plt.tight_layout()


##### + Show example planes alongside it:

explanes_depths = [70,180,320,450]
vmin = 0
vmax = 200

fig,axes = plt.subplots(4,4,figsize=(9,7))
ax1 = plt.subplot(141)

ax1.plot(meanF2_V1,slicedepths,c='darkviolet')
ax1.plot(meanF2_PM,slicedepths,c='lightseagreen')
ax1.invert_yaxis()
ax1.set_ylabel('Cortical Depth')
ax1.set_xlabel('Fluorescence')
# ax1.legend(['V1','PM'],frameon=False)
ax1.set_title('Fluorescence')
for d in explanes_depths:
    ax1.axhline(d,color='k',linestyle=':',linewidth=1)

ax2 = plt.subplot(142)
ax2.plot(nTdTomCells_V1,slicedepths,c='darkviolet')
ax2.plot(nTdTomCells_PM,slicedepths,c='lightseagreen')
ax2.invert_yaxis()
ax2.set_ylabel('')
ax2.set_xlabel('#Labeled cells')
ax2.legend(['V1','PM'],frameon=False)
ax2.set_title('#Labeled cells')

for d in explanes_depths:
    ax2.axhline(d,color='k',linestyle=':',linewidth=1)

for i,d in enumerate(explanes_depths):
    ax = plt.subplot(4,4,i*4+3)
    ax.imshow(redstack_V1[:,:,slicedepths==d],cmap='gray',vmin=vmin,vmax=vmax)
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.add_patch(Rectangle((0,0),512,512,edgecolor = 'darkviolet',
             linewidth = 3, fill=False))
    if i==0: 
         ax.set_title('V1')

    ax = plt.subplot(4,4,i*4+4)
    ax.imshow(redstack_PM[:,:,slicedepths==d],cmap='gray',vmin=vmin,vmax=vmax)
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.add_patch(Rectangle((0,0),512,512,edgecolor = 'lightseagreen',
            linewidth = 3, fill=False))
    if i==0: 
         ax.set_title('PM')
plt.tight_layout()


# greenstackre = greenstack.reshape(512,512,350,2)
# greenstackre = np.average(greenstackre,axis=3)
# greenstackre = np.transpose(greenstackre, (2, 0, 1))

# redstackre = redstack.reshape(512,512,350,2)
# redstackre = np.average(redstackre,axis=3)
# redstackre = np.transpose(redstackre, (2, 0, 1))

# plt.imshow(redstackre[253,:,:])
# plt.show()

# outpath = outputdirec + 'greenstack_ROI2_avg.tif'
# fH = open(outpath,'wb') #as fH:
# tifffile.imwrite(fH,greenstackre.astype('int16'), bigtiff=True)

# outpath = outputdirec + 'redstack_ROI2_avg.tif'
# fH = open(outpath,'wb') #as fH:
# tifffile.imwrite(fH,redstackre.astype('int16'), bigtiff=True)

