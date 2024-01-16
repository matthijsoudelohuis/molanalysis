# -*- coding: utf-8 -*-
"""
Script to generate a video of mesoscopic 2p Ca2+ imaging recordings
alongside facial videography in virtual reality
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os, shutil
os.chdir('T:\\Python\\molanalysis\\')

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from suite2p.io.binary import BinaryFile
from utils.imagelib import im_norm8
from preprocessing.preprocesslib import align_timestamps
from ScanImageTiffReader import ScanImageTiffReader as imread


#### Parameters #####

savedir         = "T:\\OneDrive\\PostDoc\\Figures\\PromoVideo\\"

animal_id       = 'LPE10919'
sessiondate     = '2023_11_13'
protocol        = 'GN'
savefilename    = '%s_promovid' % animal_id

rawdatadir      = 'J:\\RawData\\LPE10919\\2023_11_13\\'
procdatadir     = 'V:\\Procdata\\GN\\LPE10919\\2023_11_13'
# file_OV_green   = 'V:\\Procdata\\OV\\LPE11086_2023_11_21_green.tif'
# file_OV_red     = 'V:\\Procdata\\OV\\LPE11086_2023_11_21_red.tif'

file_OV_green   = 'V:\\Procdata\\OV\\LPE11081_2023_11_21_green.tif'
file_OV_red     = 'V:\\Procdata\\OV\\LPE11081_2023_11_21_red.tif'

ex_plane1       = 0 #V1
ex_plane2       = 6 #PM

boxpos1         = [2400,1800] #location of ROI 1 in overview window
boxpos2         = [2950,1350]
npix_box_inOV   = np.round(512 * 512/568) #because width of scan area in overview window is not the same

ix              = 150 #cropping of facial video data
iy              = 200
lenxy           = 768 #size of video crop

t_start         = 16568902.900032 #timestamp of start of video
fps             = 30 #frames per second for the movie
nframes         = 30 #number of frames from timestamp
ts_vid          = np.linspace(t_start,t_start+nframes/fps,nframes)

# vidsize         = [1920,1080] #size of the video (16:9 aspect ratio)
vidsize         = [1200,800] #size of the output video

### Init the video data structures:
data_window     = np.empty((3976, 4014)) #size = 376, 360
data_plane1     = np.empty((512, 512, nframes))
data_plane2     = np.empty((512, 512, nframes))
data_vr         = np.empty((1024,512, nframes))
data_face       = np.empty((lenxy,lenxy, nframes))

################################################################
#################### Get the facial video data: ################

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
    # data_face = data_face[ix:ix+lenxy,iy:iy+lenxy,:]

    data_face[:,:,i] = image[ix:ix+lenxy,iy:iy+lenxy,0]


# data_face = data_face[ix:ix+lenxy,iy:iy+lenxy,:]

###################################################################
#################### Get the overview window data: ################

reader = imread(str(file_OV_green)) # amazing - this librarty needs str
greendata = im_norm8(reader.data(),min=35,max=99.5)
reader = imread(str(file_OV_red)) # amazing - this librarty needs str
reddata = im_norm8(reader.data(),min=1,max=99.5)

data_window = np.dstack((reddata,greendata,np.zeros(np.shape(greendata)))).astype(np.uint8)

fig,ax = plt.subplots()
plt.imshow(data_window)
ax.add_patch(plt.Rectangle(boxpos1,512,512,alpha=1,
                           facecolor='none',linewidth=1,edgecolor='white'))
ax.add_patch(plt.Rectangle(boxpos2,512,512,alpha=1,
                           facecolor='none',linewidth=1,edgecolor='white'))

####################################################################
###################### Get the calcium imaging data: #################### 

sessiondata = pd.DataFrame({'protocol': [protocol]})
sessiondata['animal_id'] = animal_id
sessiondata['sessiondate'] = sessiondate

 ## Get trigger data to align timestamps:
filenames         = os.listdir(sesfolder)
triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
triggerdata       = pd.read_csv(os.path.join(sesfolder,triggerdata_file[0]),skiprows=2).to_numpy()

###################################################################
####################### Plane 1: ##################################

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

data_green  = im_norm8(data_green,min=1,max=99)
data_red    = im_norm8(data_red,min=1,max=99)

data_plane1 = np.stack((data_red,data_green,np.zeros(np.shape(data_green))),axis=3).astype(np.uint8)

###################################################################
####################### Plane 2: #################################

ops = np.load(os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'ops.npy'),allow_pickle=True).item()
[ts_master, protocol_frame_idx_master] = align_timestamps(sessiondata, ops, triggerdata)

fps_imaging     = 5.357
firstframe      = np.where(ts_master > t_start)[0][0]
framestoload    = np.arange(firstframe,firstframe + nframes / fps * fps_imaging).astype(np.int64)
ts_imaging       = ts_master[framestoload]

file_chan1 = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'data.bin')
file_chan2 = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'data_chan2.bin')

with BinaryFile(read_filename=file_chan1,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2:
    data_green      = f1.ix(indices=framestoload)
    data_red        = f2.ix(indices=framestoload)

data_green  = im_norm8(data_green,min=1,max=99)
data_red    = im_norm8(data_red,min=1,max=99)

data_plane2 = np.stack((data_red,data_green,np.zeros(np.shape(data_green))),axis=3).astype(np.uint8)


#########################################################################
################# Virtual Reality Rendering Video ######################


#########################################################################
##################### Make one frame of the video ############################

iF = 0

fig,axes = plt.subplots(2,3,figsize=(12,8))
fig.set_facecolor('black')

axes = []


axes.append(plt.subplot(231))
axes[0].imshow(data_face[:,:,iF],cmap='gray')

axes.append(plt.subplot(2,3,(2,3)))
### VR here
axes[1].set_axis_off()

### VR here
axes.append(plt.subplot(234))
axes[2].imshow(data_window)
axes[2].add_patch(plt.Rectangle(boxpos1,npix_box_inOV,npix_box_inOV,alpha=1,
                           facecolor='none',linewidth=1,edgecolor='white'))
axes[2].add_patch(plt.Rectangle(boxpos2,npix_box_inOV,npix_box_inOV,alpha=1,
                           facecolor='none',linewidth=1,edgecolor='white'))

axes.append(plt.subplot(235))
axes[3].imshow(data_plane1[iF,:,:,:])

axes.append(plt.subplot(236))
axes[4].imshow(data_plane2[iF,:,:,:])

[ax.set_axis_off() for ax in axes]

plt.subplots_adjust(wspace=0, hspace=0)


#########################################################################
########################### Make the video: ##############################


## Make a mp4 video of it:
out = cv2.VideoWriter(os.path.join(savedir,savefilename +  '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (vidsize), True)
# out = cv2.VideoWriter(os.path.join(savedir,savefilenameavi), cv2.VideoWriter_fourcc('P','I','M','1'), fps, (size[1], size[0]), False)
for iF in range(nframes):

    fig,axes = plt.subplots(2,3,figsize=(12,8))
    fig.set_facecolor('black')

    axes = []

    axes.append(plt.subplot(231))
    axes[0].imshow(data_face[:,:,iF],cmap='gray')

    axes.append(plt.subplot(2,3,(2,3)))
    ### VR here
    axes[1].set_axis_off()

    ### VR here
    axes.append(plt.subplot(234))
    axes[2].imshow(data_window)
    axes[2].add_patch(plt.Rectangle(boxpos1,npix_box_inOV,npix_box_inOV,alpha=1,
                            facecolor='none',linewidth=1,edgecolor='white'))
    axes[2].add_patch(plt.Rectangle(boxpos2,npix_box_inOV,npix_box_inOV,alpha=1,
                            facecolor='none',linewidth=1,edgecolor='white'))

    iF_imaging = np.where(ts_imaging>ts_vid[iF])[0][0]
    axes.append(plt.subplot(235))
    axes[3].imshow(data_plane1[iF_imaging,:,:,:])

    axes.append(plt.subplot(236))
    axes[4].imshow(data_plane2[iF_imaging,:,:,:])

    # Make sure axes fill the entire figure:
    [ax.set_axis_off() for ax in axes]
    [ax.margins(0) for ax in axes]
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.tight_layout(pad=0)

    #Draw the canvas, important for saving it in the next step:
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imdata = imdata.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.figure()
    # plt.imshow(imdata)
    # plt.margins(0,0)
    out.write(np.flip(imdata, axis=-1) )
    plt.close(fig)


out.release()

# # Make a gif out of it: 
# def seqtogif(viddata,savedir,savefilename,fps=20):
#     nframes = np.shape(viddata)[0]
#     plt.figure()
#     filenames = []
#     for ifr in range(np.min([nframes,200])):
    
#         data = np.dstack((np.zeros(size),viddata[ifr,:,:],np.zeros(size))).astype(np.uint8)
#         plt.figure()
#         plt.imshow(data,vmin=0, vmax=255)
#         plt.axis('off')
#         plt.tight_layout()

#         # create file name and append it to a list
#         filename = f'{ifr}.png'
#         filenames.append(os.path.join(savedir, filename))
        
#         # save frame
#         plt.savefig(os.path.join(savedir, filename))
#         plt.close()
        
#     # Load each file into a list
#     frames = []
#     for filename in filenames:
#         frames.append(imageio.imread(filename))

#     # Save them as frames into a gif 
#     exportname = os.path.join(savedir, savefilename)
#     imageio.mimsave(exportname, frames, 'GIF', fps=fps)
    
#     # Remove files
#     for filename in set(filenames):
#         os.remove(filename)
    
#     return


# seqtogif(viddata,savedir,savefilename + '.gif',fps=30)
