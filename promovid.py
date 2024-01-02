# -*- coding: utf-8 -*-
"""
Script to generate a video of mesoscopic 2p Ca2+ imaging recordings
alongside facial videography in virtual reality
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os, shutil
import numpy as np
# import suite2p
from suite2p.io.binary import BinaryFile
import pandas as pd
# from labeling.label_lib import bleedthrough_correction
from utils.imagelib import im_norm8
from preprocessing.preprocesslib import align_timestamps

from ScanImageTiffReader import ScanImageTiffReader as imread

os.chdir('T:\\Python\\molanalysis\\')
import tifffile
from utils.twoplib import split_mROIs
import cv2
import matplotlib.pyplot as plt
# import imageio



#### Parameters #####

savedir         = "T:\\OneDrive\\PostDoc\\Figures\\WindowVideo\\"

# rawdatadir      = 'W:\\Users\\Matthijs\\Rawdata\\LPE10885\\2023_10_12\\OV_lowSR_highTR\\'
animal_id       = 'LPE10919'
sessiondate     = '2023_11_13'
protocol        = 'GN'

rawdatadir      = 'J:\\RawData\\LPE10919\\2023_11_13\\'
procdatadir     = 'V:\\Procdata\\GN\\LPE10919\\2023_11_13'
# file_OV_green   = 'V:\\Procdata\\OV\\LPE11086_2023_11_21_green.tif'
# file_OV_red     = 'V:\\Procdata\\OV\\LPE11086_2023_11_21_red.tif'

file_OV_green   = 'V:\\Procdata\\OV\\LPE11081_2023_11_21_green.tif'
file_OV_red     = 'V:\\Procdata\\OV\\LPE11081_2023_11_21_red.tif'

ex_plane1       = 0 #V1
ex_plane2       = 6 #PM

savefilename    = 'LPE10919_promovid'


t_start         = 16568902.900032 #timestamp of start of video
fps             = 30 #frames per second for the movie
nframes         = 30 #number of frames from timestamp

### Init the video data structures:
data_window     = np.empty((3976, 4014)) #size = 376, 360
data_plane1     = np.empty((512, 512, nframes))
data_plane2     = np.empty((512, 512, nframes))
data_vr         = np.empty((1024,512, nframes))
# data_face       = np.empty((512,512, nframes))

data_face       = np.empty((960,1280, nframes))

## Get the facial video data:
sesfolder       = os.path.join(rawdatadir,'GN','Behavior')
filenames       = os.listdir(sesfolder)
avi_file        = list(filter(lambda a: '.avi' in a, filenames)) #find the trialdata file
csv_file        = list(filter(lambda a: 'cameracsv' in a, filenames)) #find the trialdata file
csvdata         = pd.read_csv(os.path.join(sesfolder,csv_file[0]))
ts              = csvdata['Item2'].to_numpy()

cap             = cv2.VideoCapture(os.path.join(sesfolder,avi_file[0]))

frame_number    = np.where(ts>t_start)[0][0]

assert cap.get(cv2.CAP_PROP_FPS)==30, 'video not 30 frames per second' 
assert cap.get(cv2.CAP_PROP_FRAME_COUNT)>frame_number+nframes,'requested frame exceeds frame count'

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number) # optional

for i in range(nframes):
    success, image = cap.read()
    data_face[:,:,i] = image[:,:,0]



### Get the overview window: 
reader = imread(str(file_OV_green)) # amazing - this librarty needs str
greendata = im_norm8(reader.data(),min=35,max=99.5)
reader = imread(str(file_OV_red)) # amazing - this librarty needs str
reddata = im_norm8(reader.data(),min=1,max=99.5)

data_window = np.dstack((reddata,greendata,np.zeros(np.shape(greendata)))).astype(np.uint8)

plt.figure()
plt.imshow(data_window)

### Get the calcium imaging data: 

ex_plane1       = 0 #V1
ex_plane2       = 6 #PM

cadata = pd.read_csv(os.path.join(procdatadir,'deconvdata.csv'))

sessiondata = pd.DataFrame({'protocol': [protocol]})
sessiondata['animal_id'] = animal_id
sessiondata['sessiondate'] = sessiondate

 ## Get trigger data to align timestamps:
filenames         = os.listdir(sesfolder)
triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
triggerdata       = pd.read_csv(os.path.join(sesfolder,triggerdata_file[0]),skiprows=2).to_numpy()


####################### Plane 1: #################################
ops = np.load(os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane1,'ops.npy'),allow_pickle=True).item()
[ts_master, protocol_frame_idx_master] = align_timestamps(sessiondata, ops, triggerdata)

fps_imaging     = 5.357
firstframe      = np.where(ts_master > t_start)[0][0]
framestoload    = np.arange(firstframe,firstframe + nframes / fps * fps_imaging).astype(np.int64)
ts_frames       = ts_master[framestoload]

file_chan1 = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane1,'data.bin')
file_chan2 = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane1,'data_chan2.bin')

with BinaryFile(read_filename=file_chan1,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2:
    data_green      = f1.ix(indices=framestoload)
    data_red        = f2.ix(indices=framestoload)

data_green  = im_norm8(data_green)
data_red    = im_norm8(data_red)

data_plane1 = np.stack((data_red,data_green,np.zeros(np.shape(data_green))),axis=3).astype(np.uint8)

####################### Plane 2: #################################

ops = np.load(os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'ops.npy'),allow_pickle=True).item()
[ts_master, protocol_frame_idx_master] = align_timestamps(sessiondata, ops, triggerdata)

fps_imaging     = 5.357
firstframe      = np.where(ts_master > t_start)[0][0]
framestoload    = np.arange(firstframe,firstframe + nframes / fps * fps_imaging).astype(np.int64)
ts_frames       = ts_master[framestoload]

file_chan1 = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'data.bin')
file_chan2 = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'data_chan2.bin')

with BinaryFile(read_filename=file_chan1,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2:
    data_green      = f1.ix(indices=framestoload)
    data_red        = f2.ix(indices=framestoload)

data_green  = im_norm8(data_green)
data_red    = im_norm8(data_red)

data_plane2 = np.stack((data_red,data_green,np.zeros(np.shape(data_green))),axis=3).astype(np.uint8)


# from scipy.interpolate import interp2d

ix = 150
iy = 200
lenxy = 768
data_face2 = data_face[ix:ix+lenxy,iy:iy+lenxy,:]

#####   

fig,axes = plt.subplots(2,3,figsize=(12,8))
fig.set_facecolor('black')

axes = []

axes.append(plt.subplot(2,3,(1,2)))
### VR here
axes[0].set_axis_off()
# ax1.set_aspect('equal')

axes.append(plt.subplot(233))
axes[1].imshow(data_face2[:,:,0],cmap='gray')
# axes[1].set_axis_off()


### VR here
axes.append(plt.subplot(234))
axes[2].imshow(data_window)
# ax3.set_axis_off()
# ax3.set_aspect('equal')
# 
axes.append(plt.subplot(235))
axes[3].imshow(data_plane1[0,:,:,:])
# ax4.set_axis_off()
# ax4.set_aspect('equal')

axes.append(plt.subplot(236))
axes[4].imshow(data_plane2[0,:,:,:])
# ax5.set_axis_off()
# ax5.set_aspect('equal')

[ax.set_axis_off() for ax in axes]

plt.subplots_adjust(wspace=0, hspace=0)


for d in explanes_depths:
    ax2.axhline(d,color='k',linestyle=':',linewidth=1)

for i,d in enumerate(explanes_depths):
    ax = plt.subplot(4,4,i*4+3)


## Make a mp4 video of it:
totalframes = np.shape(viddata)[0]
# out = cv2.VideoWriter(os.path.join(savedir,savefilename +  '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
out = cv2.VideoWriter(os.path.join(savedir,savefilename +  '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size), True)
# out = cv2.VideoWriter(os.path.join(savedir,savefilenameavi), cv2.VideoWriter_fourcc('P','I','M','1'), fps, (size[1], size[0]), False)
for ifr in range(totalframes):
    data = np.dstack((np.zeros((size[1],size[0])),viddata[ifr,:,:],np.zeros((size[1],size[0])))).astype(np.uint8)
    out.write(data)
    # out.write(viddata[ifr,:,:])
out.release()


# Make a gif out of it: 
def seqtogif(viddata,savedir,savefilename,fps=20):
    nframes = np.shape(viddata)[0]
    plt.figure()
    filenames = []
    for ifr in range(np.min([nframes,200])):
    
        data = np.dstack((np.zeros(size),viddata[ifr,:,:],np.zeros(size))).astype(np.uint8)
        plt.figure()
        plt.imshow(data,vmin=0, vmax=255)
        plt.axis('off')
        plt.tight_layout()

        # create file name and append it to a list
        filename = f'{ifr}.png'
        filenames.append(os.path.join(savedir, filename))
        
        # save frame
        plt.savefig(os.path.join(savedir, filename))
        plt.close()
        
    # Load each file into a list
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    # Save them as frames into a gif 
    exportname = os.path.join(savedir, savefilename)
    imageio.mimsave(exportname, frames, 'GIF', fps=fps)
    
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    
    return


seqtogif(viddata,savedir,savefilename + '.gif',fps=20)


#move original to subdir and rename corrected to data.bin to be read by suite2p for detection:
for iplane in np.arange(ops['nplanes']):
    planefolder = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane)
    file_chan1       = os.path.join(planefolder,'data.bin')
    file_chan1_corr   = os.path.join(planefolder,'data_corr.bin')
    
    os.mkdir(os.path.join(planefolder,'orig'))

    shutil.move(os.path.join(planefolder,file_chan1),os.path.join(planefolder,'orig'))
    
    os.rename(os.path.join(planefolder,file_chan1_corr), os.path.join(planefolder,file_chan1))