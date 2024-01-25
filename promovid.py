# -*- coding: utf-8 -*-
"""
Script to generate an example video of mesoscopic 2p Ca2+ imaging recordings
alongside facial videography and stimuli
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

animal_id       = 'LPE11086'
sessiondate     = '2023_12_16'
protocol        = 'IM'
savefilename    = '%s_promovid' % animal_id

rawdatadir      = os.path.join('I:\\RawData',animal_id,sessiondate)
# procdatadir     = 'V:\\Procdata\\GN\\LPE11086\\2023_11_13'
# file_OV_green   = 'V:\\Procdata\\OV\\LPE11086_2023_11_21_green.tif'
# file_OV_red     = 'V:\\Procdata\\OV\\LPE11086_2023_11_21_red.tif'

file_OV_green   = 'V:\\Procdata\\OV\\LPE11086_2023_11_21_green.tif'
file_OV_red     = 'V:\\Procdata\\OV\\LPE11086_2023_11_21_red.tif'

ex_plane1       = 0 #V1
ex_plane2       = 5 #PM

# boxpos1         = [2400,1800] #location of ROI 1 in overview window
# boxpos2         = [2950,1350]
boxpos1         = [1600,2350] #location of ROI 1 (PM) in overview window
boxpos2         = [2700,1060]

npix_box_inOV   = np.round(512 * 512/568) #because width of scan area in overview window is not the same

ix              = 150 #cropping of facial video data
iy              = 200
lenxy           = 768 #size of video crop

fps             = 30 #frames per second for the movie
speedup         = 3 #how many times realtime
viddur          = 10 #number of seconds long the video is going to be (not how long it covers)
nframes         = int(np.ceil(viddur * fps))
# t_start         = 19430450.042016 #timestamp of start of video
t_start         = 19430473.554016 #timestamp of start of video
ts_vid          = np.linspace(t_start,t_start+speedup*viddur,nframes)
t_end           = ts_vid[-1] #timestamp of start of video

vidsize         = [1200,800] #size of the output video

################################################################
#################### Get the facial video data: ################

sesfolder       = os.path.join(rawdatadir,protocol,'Behavior')
filenames       = os.listdir(sesfolder)
avi_file        = list(filter(lambda a: '.avi' in a, filenames)) #find the trialdata file
csv_file        = list(filter(lambda a: 'cameracsv' in a, filenames)) #find the trialdata file
csvdata         = pd.read_csv(os.path.join(sesfolder,csv_file[0]))
ts_face         = csvdata['Item2'].to_numpy()
data_face       = np.empty((lenxy,lenxy, nframes))

cap             = cv2.VideoCapture(os.path.join(sesfolder,avi_file[0]))

assert cap.get(cv2.CAP_PROP_FPS)==30, 'video not 30 frames per second' 

for i,ts in enumerate(ts_vid):
    frame_number    = np.where(ts_face>ts)[0][0]
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT)>frame_number+nframes,'requested frame exceeds frame count'
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number) # optional
    success, image = cap.read()
    data_face[:,:,i] = image[ix:ix+lenxy,iy:iy+lenxy,0]

###################################################################
#################### Get the overview window data: ################

reader      = imread(str(file_OV_green)) 
greendata   = im_norm8(reader.data(),min=35,max=99.5)
reader      = imread(str(file_OV_red)) 
reddata     = im_norm8(reader.data(),min=5,max=98.5)

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

ops = np.load(os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane1,'ops.npy'),allow_pickle=True).item()
[ts_master, protocol_frame_idx_master] = align_timestamps(sessiondata, ops, triggerdata)

framestoload = np.empty(nframes).astype(int)
for i,ts in enumerate(ts_vid):
    framestoload[i] = np.argmin(abs(ts_master-ts))

###################################################################
####################### Plane 1: ##################################

# file_chan1      = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane1,'data.bin')
# file_chan2      = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane1,'data_chan2.bin')

# with BinaryFile(read_filename=file_chan1,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2:
#     data_green      = f1.ix(indices=framestoload)
#     data_red        = f2.ix(indices=framestoload)

# data_green  = im_norm8(data_green,min=1,max=99)
# data_red    = im_norm8(data_red,min=15,max=98)

# data_green = data_green / 1.5

# data_plane1 = np.stack((data_red,data_green,np.zeros(np.shape(data_green))),axis=3).astype(np.uint8)

file_chan1      = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane1,'data.bin')

with BinaryFile(read_filename=file_chan1,Ly=512, Lx=512) as f1:
    data_green      = f1.ix(indices=framestoload)

data_green  = im_norm8(data_green,min=1,max=99)
# data_green  = data_green / 1.2

ops = np.load(os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane1,'ops.npy'),allow_pickle=True).item()
data_red    = im_norm8(ops['meanImg_chan2'],min=5,max=98)
data_red    = np.repeat(data_red[np.newaxis, :, :], nframes, axis=0)
# data_red    = data_red / 1.1

data_plane1 = np.stack((data_red,data_green,np.zeros(np.shape(data_green))),axis=3).astype(np.uint8)

plt.figure()
plt.imshow(data_plane1[0,:,:])

###################################################################
####################### Plane 2: #################################

file_chan1 = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'data.bin')
# file_chan2 = os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'data_chan2.bin')

with BinaryFile(read_filename=file_chan1,Ly=512, Lx=512) as f1:
# BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2:
    data_green      = f1.ix(indices=framestoload)
    # data_red        = f2.ix(indices=framestoload)

data_green  = im_norm8(data_green,min=1,max=99)
# data_green  = data_green / 1.5

ops = np.load(os.path.join(rawdatadir,'suite2p','plane%d' % ex_plane2,'ops.npy'),allow_pickle=True).item()
data_red    = im_norm8(ops['meanImg_chan2'],min=5,max=98)
data_red    = np.repeat(data_red[np.newaxis, :, :], nframes, axis=0)

data_plane2 = np.stack((data_red,data_green,np.zeros(np.shape(data_green))),axis=3).astype(np.uint8)

plt.figure()
plt.imshow(data_plane2[0,:,:])

#########################################################################
########################### Get Stimuli  ################################

import scipy.io as sio
mat_fname = 't:\Bonsai\lab-leopoldo-solene-vr\Matlab\images_natimg2800_all.mat'
mat_contents = sio.loadmat(mat_fname)
natimgdata = mat_contents['imgs']

## Get trial data file to show stimuli alongside neural data: 
filenames         = os.listdir(sesfolder)
trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)

data_stimuli    = np.empty((68,180, nframes)).astype('uint8')
data_stimuli.fill(128)

for i,ts in enumerate(ts_vid):
    temp = np.where(np.logical_and(ts>trialdata['tOnset'],ts<trialdata['tOffset']))[0]
    if temp.size > 0:
        data_stimuli[:,:,i] = natimgdata[:,:180,trialdata.iloc[temp,trialdata.columns.get_loc('ImageNumber')]].squeeze()

#########################################################################
##################### Make one frame of the video ############################

iF = 0

fig,axes = plt.subplots(2,3,figsize=(12,8))
fig.set_facecolor('black')

axes = []

axes.append(plt.subplot(231))
axes[0].imshow(data_face[:,:,iF],cmap='gray')
axes[0].text(290,730,'3x normal speed',color='w')

axes.append(plt.subplot(2,3,(2,3)))
### VR here
axes[1].set_axis_off()
axes[1].imshow(data_stimuli[:,:,iF],cmap='gray',vmin=0,vmax=255)

### VR here
axes.append(plt.subplot(234))
axes[2].imshow(data_window)
axes[2].add_patch(plt.Rectangle(boxpos1,npix_box_inOV,npix_box_inOV,alpha=1,
                           facecolor='none',linewidth=1,edgecolor='white'))
axes[2].add_patch(plt.Rectangle(boxpos2,npix_box_inOV,npix_box_inOV,alpha=1,
                           facecolor='none',linewidth=1,edgecolor='white'))
axes[2].text(boxpos1[0]+100,boxpos1[1]-60,'PM',color='w')
axes[2].text(boxpos2[0]+100,boxpos2[1]-60,'V1',color='w')

axes.append(plt.subplot(235))
axes[3].imshow(data_plane1[iF,:,:,:])
axes[3].text(150,30,'Example plane PM',color='w')

axes.append(plt.subplot(236))
axes[4].imshow(data_plane2[iF,:,:,:])
axes[4].text(150,30,'Example plane V1',color='w')

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
    axes[0].text(400,750,'3x normal speed',color='w')

    axes.append(plt.subplot(2,3,(2,3)))
    ### VR here
    axes[1].set_axis_off()
    # plt.figure()
    # axes[1].imshow(natimgdata[:,:90,15],cmap='gray')
    axes[1].imshow(data_stimuli[:,:,iF],cmap='gray',vmin=0,vmax=255)

    ### VR here
    axes.append(plt.subplot(234))
    axes[2].imshow(data_window)
    axes[2].add_patch(plt.Rectangle(boxpos1,npix_box_inOV,npix_box_inOV,alpha=1,
                            facecolor='none',linewidth=1,edgecolor='white'))
    axes[2].add_patch(plt.Rectangle(boxpos2,npix_box_inOV,npix_box_inOV,alpha=1,
                            facecolor='none',linewidth=1,edgecolor='white'))
    axes[2].text(boxpos1[0]+100,boxpos1[1]-60,'PM',color='w')
    axes[2].text(boxpos2[0]+100,boxpos2[1]-60,'V1',color='w')

    # iF_imaging = np.where(ts_imaging>ts_vid[iF])[0][0]
    axes.append(plt.subplot(235))
    axes[3].imshow(data_plane1[iF,:,:,:])
    axes[3].text(150,30,'Example plane PM',color='w')

    axes.append(plt.subplot(236))
    axes[4].imshow(data_plane2[iF,:,:,:])
    axes[4].text(150,30,'Example plane V1',color='w')

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
