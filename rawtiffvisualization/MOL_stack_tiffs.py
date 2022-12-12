import os
import numpy as np
import twoplib
from twoplib import *
from matplotlib import pyplot as plt

direc  = 'V:/Rawdata/PILOTS/20221116_NSH07435_ExpressionWindow/ZstackROI_2/'

outputdirec = 'V:/Rawdata/PILOTS/20221116_NSH07435_ExpressionWindow/'

nchannels   = 2 #whether image has both red and green channel acquisition (PMT)
greenstack  = np.empty([512,512,650])
redstack    = np.empty([512,512,650])

greenstack  = np.empty([512,512,700])
redstack    = np.empty([512,512,700])

# for x in os.listdir(direc):
for i,x in enumerate(os.listdir(direc)):
    if x.endswith(".tif"):
            # mROI_data, meta = split_mROIs(os.path.join(direc,x))
            # nROIs = len(mROI_data)
            i
            fname = Path(os.path.join(direc,x))
            reader = imread(str(fname)) # amazing - this librarty needs str
            Data = reader.data()
            greenstack[:,:,i] = np.average(Data[0::2,:,:], axis=0)
            redstack[:,:,i] = np.average(Data[1::2,:,:], axis=0)

            # cmax        = np.max(c[1::2,:,:], axis=0)
            # redframe    = np.stack([redframe,cmax],axis=2).max(axis=2)
            # redframe    = np.concatenate([redframe,c[1::2,:,:]],axis=0)
            # redframe    = np.max(redframe, axis=0)

greenstackre = greenstack.reshape(512,512,350,2)
greenstackre = np.average(greenstackre,axis=3)
greenstackre = np.transpose(greenstackre, (2, 0, 1))

redstackre = redstack.reshape(512,512,350,2)
redstackre = np.average(redstackre,axis=3)
redstackre = np.transpose(redstackre, (2, 0, 1))

plt.imshow(redstackre[253,:,:])
plt.show()

outpath = outputdirec + 'greenstack_ROI2_avg.tif'
fH = open(outpath,'wb') #as fH:
tifffile.imwrite(fH,greenstackre.astype('int16'), bigtiff=True)

outpath = outputdirec + 'redstack_ROI2_avg.tif'
fH = open(outpath,'wb') #as fH:
tifffile.imwrite(fH,redstackre.astype('int16'), bigtiff=True)
