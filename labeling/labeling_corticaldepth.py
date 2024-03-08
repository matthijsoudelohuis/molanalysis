"""
Analyzes stack of GCaMP and tdtomato expressing cells using cellpose software (Pachitariu & Stringer)
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ScanImageTiffReader import ScanImageTiffReader as imread
from pathlib import Path
from matplotlib.patches import Rectangle
from cellpose import models
from utils.imagelib import im_norm8
from sklearn.preprocessing import minmax_scale
from utils.plotting_style import *

## Directories: 
rawdatadir      = 'F:\\Stacks\\'
savedir         = 'T:\\OneDrive\\PostDoc\\Figures\\Neural - Labeling\\Stack\\'

## Pretrained models to label tdtomato expressing cells:
model_red       = models.CellposeModel(pretrained_model = 'T:\\Python\\cellpose\\redlib_tiff\\trainingdata\\models\\redcell_20231107')
# model_type='cyto' or 'nuclei' or 'cyto2'
# model_red = models.CellposeModel(pretrained_model = 'T:\\Python\\cellpose\\testdir\\models\\MOL_20230814_redcells')
# model_green     = models.Cellpose(model_type='cyto')

def get_stack_data(direc,model=model_red):

    nslices         = len(os.listdir(direc))
    assert nslices==75, 'wrong number of slices'

    diam            = 12
    chan            = [[1,0]] # grayscale=0, R=1, G=2, B=3 # channels = [cytoplasm, nucleus]
    nchannels       = 2 #whether image has both red and green channel acquisition (PMT)

    ### Stack loading:
    greenstack  = np.empty([512,512,nslices])
    redstack    = np.empty([512,512,nslices])
    for i,x in enumerate(os.listdir(direc)):
        print(f"Averaging frames for slice {i+1}",end='\r')
        if x.endswith(".tif"):
                fname               = Path(os.path.join(direc,x))
                reader              = imread(str(fname))
                Data                = reader.data()
                greenstack[:,:,i]   = np.average(Data[0::2,:,:], axis=0)
                redstack[:,:,i]     = np.average(Data[1::2,:,:], axis=0)

    # nTdTomCells_V1       = np.zeros(nslices)
    # nTdTomCells_PM       = np.zeros(nslices)
    nTdTomCells       = np.zeros(nslices)
    print('\n')
    ### Get number of tdTomato labeled cells (using cellpose):
    for i in range(nslices):
        print(f"Labeling cells for slice {i+1}",end='\r')
        img_red = np.zeros((512, 512, 3), dtype=np.uint8)
        img_red[:,:,0] = im_norm8(redstack[:,:,i])

        masks_cp_red, flows, styles = model.eval(img_red, diameter=diam, channels=chan)
        nTdTomCells[i]      = len(np.unique(masks_cp_red))-1 #zero is counted as unique

    ### Get the mean fluorescence for each plane:
    meanF2       = [np.mean(redstack[:,:,i]) for i in range(nslices)]
    # meanF1       = [np.mean(greenstack[:,:,i]) for i in range(nslices)]

    data = np.vstack((meanF2,nTdTomCells))

    return data

################################ #################################
##  Loop over all selected animals and folders
animal_ids      = [f.name for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]
nanimals        = len(animal_ids)

nslices         = 75
slicedepths     = np.linspace(0,10*(nslices-1),nslices)
dataV1          = np.empty((2,nslices,nanimals)) 
dataPM          = np.empty((2,nslices,nanimals)) 
# X=2 (fluo and cells), Y=2 areas, V1 and PM, 
# Z=number of slices 75, Z= number of animals 

# dataV1           = np.random.rand(2,nslices,nanimals) 
# dataPM          = np.random.rand(2,nslices,nanimals) 

for iA,animal_id in enumerate(animal_ids): #for each animal

    animaldir   = os.path.join(rawdatadir,animal_id) 
    assert len(os.listdir(animaldir))==1, 'multiple stacks found per animal'
    sesdir      = os.path.join(animaldir,os.listdir(animaldir)[0])

    dataV1[:,:,iA] = get_stack_data(os.path.join(sesdir,'STACK_V1'))
    dataPM[:,:,iA] = get_stack_data(os.path.join(sesdir,'STACK_PM'))

np.save(os.path.join(rawdatadir,'stackdata_%danimals.npy' % nanimals),(dataV1,dataPM))

## Load previously processed stack data: 
(dataV1,dataPM) = np.load(os.path.join(rawdatadir,'stackdata_%danimals.npy' % nanimals))

###################################################################
######################   Make figures: ############################


dataV1_mean     = np.nanmean(dataV1,axis=2)
dataV1_err      = np.nanstd(dataV1,axis=2) / np.sqrt(nanimals)
dataPM_mean     = np.nanmean(dataPM,axis=2)
dataPM_err      = np.nanstd(dataPM,axis=2) / np.sqrt(nanimals)

dataV1_norm = dataV1.copy()
dataV1_norm[0,:,:] = minmax_scale(dataV1_norm[0,:,:], feature_range=(0, 1), axis=0, copy=True)
dataPM_norm = dataPM.copy()
dataPM_norm[0,:,:] = minmax_scale(dataPM_norm[0,:,:], feature_range=(0, 1), axis=0, copy=True)

dataV1_norm_mean     = np.nanmean(dataV1_norm,axis=2)
dataV1_norm_err      = np.nanstd(dataV1_norm,axis=2) / np.sqrt(nanimals)
dataPM_norm_mean     = np.nanmean(dataPM_norm,axis=2)
dataPM_norm_err      = np.nanstd(dataPM_norm,axis=2) / np.sqrt(nanimals)

clrs_areas = get_clr_areas(['V1','PM'])

### Figure with depth profile: 
fig,(ax1,ax2)   = plt.subplots(1,2,figsize=(6,7))
ax1.plot(dataV1_norm_mean[0,:],slicedepths,c=clrs_areas[0],linewidth=2)
ax1.plot(dataPM_norm_mean[0,:],slicedepths,c=clrs_areas[1],linewidth=2)
ax1.plot(dataV1_norm[0,:,:],slicedepths,c=clrs_areas[0],linewidth=0.5)
ax1.plot(dataPM_norm[0,:,:],slicedepths,c=clrs_areas[1],linewidth=0.5)
ax1.invert_yaxis()
ax1.set_ylabel('Cortical Depth')
ax1.set_xlabel('Fluorescence')
# ax1.legend(['V1','PM'],frameon=False)
ax1.set_title('Fluorescence')

ax2.plot(dataV1_mean[1,:],slicedepths,c=clrs_areas[0],linewidth=2)
ax2.plot(dataPM_mean[1,:],slicedepths,c=clrs_areas[1],linewidth=2)
ax2.plot(dataV1[1,:,:],slicedepths,c=clrs_areas[0],linewidth=0.5)
ax2.plot(dataPM[1,:,:],slicedepths,c=clrs_areas[1],linewidth=0.5)
ax2.invert_yaxis()
ax2.set_ylabel('')
ax2.set_xlabel('#Labeled cells')
ax2.legend(['V1','PM'],frameon=False)
ax2.set_title('#Labeled cells')
plt.tight_layout()


##### + Show example planes alongside it:
exanimal            = 'LPE10192'
# exanimal            = 'LPE10885'
explanes_depths     = [70,180,320,450]
vmin                = 0
vmax                = 200

fig,axes = plt.subplots(4,4,figsize=(9,7))
ax1 = plt.subplot(141)
ax1.plot(dataV1_norm_mean[0,:],slicedepths,c=clrs_areas[0],linewidth=2)
ax1.plot(dataPM_norm_mean[0,:],slicedepths,c=clrs_areas[1],linewidth=2)
ax1.plot(dataV1_norm[0,:,:],slicedepths,c=clrs_areas[0],linewidth=0.5)
ax1.plot(dataPM_norm[0,:,:],slicedepths,c=clrs_areas[1],linewidth=0.5)
ax1.invert_yaxis()
ax1.set_ylabel('Cortical Depth')
ax1.set_xlabel('Fluorescence')
# ax1.legend(['V1','PM'],frameon=False)
ax1.set_title('Fluorescence')
for d in explanes_depths:
    ax1.axhline(d,color='k',linestyle=':',linewidth=1)

ax2 = plt.subplot(142)
ax2.plot(dataV1_mean[1,:],slicedepths,c=clrs_areas[0],linewidth=2)
ax2.plot(dataPM_mean[1,:],slicedepths,c=clrs_areas[1],linewidth=2)
ax2.plot(dataV1[1,:,:],slicedepths,c=clrs_areas[0],linewidth=0.5)
ax2.plot(dataPM[1,:,:],slicedepths,c=clrs_areas[1],linewidth=0.5)
ax2.invert_yaxis()
ax2.set_ylabel('')
ax2.set_xlabel('#Labeled cells')
ax2.legend(['V1','PM'],frameon=False)
ax2.set_title('#Labeled cells')

for d in explanes_depths:
    ax2.axhline(d,color='k',linestyle=':',linewidth=1)

for i,d in enumerate(explanes_depths):
    ax = plt.subplot(4,4,i*4+3)

    direc_V1    = os.path.join(os.path.join(rawdatadir,exanimal),
                            os.listdir(os.path.join(rawdatadir,exanimal))[0],
                            'STACK_V1')
    fname       = os.listdir(direc_V1)[np.floor(d/10).astype('int32')] #get file name corresponding to ex slice depth
    
    reader              = imread(str(os.path.join(direc_V1,fname)))
    Data                = reader.data()
    imdata              = np.average(Data[1::2,:,:], axis=0)

    ax.imshow(imdata,cmap='gray',vmin=vmin,vmax=vmax)
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.add_patch(Rectangle((0,0),512,512,edgecolor = clrs_areas[0],
             linewidth = 3, fill=False))
    if i==0: 
         ax.set_title('V1')

    direc_PM    = os.path.join(os.path.join(rawdatadir,exanimal),
                            os.listdir(os.path.join(rawdatadir,exanimal))[0],
                            'STACK_PM')
    fname       = os.listdir(direc_PM)[np.floor(d/10).astype('int32')] 
    
    reader              = imread(str(os.path.join(direc_PM,fname)))
    Data                = reader.data()
    imdata              = np.average(Data[1::2,:,:], axis=0)

    ax = plt.subplot(4,4,i*4+4)
    ax.imshow(imdata,cmap='gray',vmin=vmin,vmax=vmax)
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.add_patch(Rectangle((0,0),512,512,edgecolor =clrs_areas[1],
            linewidth = 3, fill=False))
    if i==0: 
         ax.set_title('PM')
plt.tight_layout()

fig.savefig(os.path.join(savedir,'Labeling_Depth_%danimals_example_planes.png' % nanimals))





# # direc = 'W:\\Users\\Matthijs\\Rawdata\\LPE10192\\2023_10_10\\'

# nslices         = len(os.listdir(os.path.join(direc,'STACK_PM')))
# slicedepths     = np.linspace(0,10*(nslices-1),nslices)
# nchannels       = 2 #whether image has both red and green channel acquisition (PMT)

# ### V1 stack loading:
# greenstack_V1  = np.empty([512,512,nslices])
# redstack_V1    = np.empty([512,512,nslices])
# for i,x in enumerate(os.listdir(os.path.join(direc,'STACK_V1'))):
#     if x.endswith(".tif"):
#             fname = Path(os.path.join(direc,'STACK_V1',x))
#             reader = imread(str(fname)) # amazing - this librarty needs str
#             Data = reader.data()
#             greenstack_V1[:,:,i] = np.average(Data[0::2,:,:], axis=0)
#             redstack_V1[:,:,i] = np.average(Data[1::2,:,:], axis=0)

# ### PM stack loading:
# greenstack_PM  = np.empty([512,512,nslices])
# redstack_PM    = np.empty([512,512,nslices])

# for i,x in enumerate(os.listdir(os.path.join(direc,'STACK_PM'))):
#     if x.endswith(".tif"):
#             fname = Path(os.path.join(direc,'STACK_PM',x))
#             reader = imread(str(fname)) # amazing - this librarty needs str
#             Data = reader.data()
#             greenstack_PM[:,:,i] = np.average(Data[0::2,:,:], axis=0)
#             redstack_PM[:,:,i] = np.average(Data[1::2,:,:], axis=0)



# chan = [[1,0]] # grayscale=0, R=1, G=2, B=3 # channels = [cytoplasm, nucleus]
# diam = 12

# # model_type='cyto' or 'nuclei' or 'cyto2'
# model_red = models.CellposeModel(pretrained_model = 'T:\\Python\\cellpose\\testdir\\models\\MOL_20230814_redcells')


# nTdTomCells_V1       = np.zeros(nslices)
# nTdTomCells_PM       = np.zeros(nslices)

# ### Get number of tdTomato labeled cells (using cellpose):
# for i in range(nslices): 
#     print(i)
#     img_red_V1 = np.zeros((512, 512, 3), dtype=np.uint8)
#     img_red_V1[:,:,0] = normalize8(redstack_V1[:,:,i])

#     masks_cp_red, flows, styles = model_red.eval(img_red_V1, diameter=diam, channels=chan)
#     nTdTomCells_V1[i]      = len(np.unique(masks_cp_red))-1 #zero is counted as unique

#     img_red_PM = np.zeros((512, 512, 3), dtype=np.uint8)
#     img_red_PM[:,:,0] = normalize8(redstack_PM[:,:,i])

#     masks_cp_red, flows, styles = model_red.eval(img_red_PM, diameter=diam, channels=chan)
#     nTdTomCells_PM[i]      = len(np.unique(masks_cp_red))-1 #zero is counted as unique


# ### Get the mean fluorescence for each plane:
# meanF2_V1       = [np.mean(redstack_V1[:,:,i]) for i in range(nslices)]
# meanF1_V1       = [np.mean(greenstack_V1[:,:,i]) for i in range(nslices)]

# meanF2_PM       = [np.mean(redstack_PM[:,:,i]) for i in range(nslices)]
# meanF1_PM       = [np.mean(greenstack_PM[:,:,i]) for i in range(nslices)]