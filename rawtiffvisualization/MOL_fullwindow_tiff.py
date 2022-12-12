import os
import numpy as np
import twoplib
from twoplib import *
from matplotlib import pyplot as plt

# direc  = 'V:/Rawdata/PILOTS/20221108_NSH0429_Spontaneous4ROI/Window2/'
# direc  = 'V:/Rawdata/PILOTS/20221108_NSH0429_Spontaneous4ROI/Window2test/'
direc  = 'V:/Rawdata/PILOTS/20221116_NSH07432_ExpressionWindow/Window/'

# outputdirec = 'V:/Rawdata/PILOTS/20221108_NSH0429_Spontaneous4ROI/'
outputdirec = 'V:/Rawdata/PILOTS/20221116_NSH07432_ExpressionWindow/'

nchannels = 2 #whether image has both red and green channel acquisition (PMT)
greenframe = np.empty([4014,3976])
redframe = np.empty([4014,3976])

for x in os.listdir(direc):
    if x.endswith(".tif"):
            mROI_data, meta = split_mROIs(os.path.join(direc,x))
            nROIs = len(mROI_data)
            c           = np.concatenate(mROI_data[:], axis=2)  # ValueError: all the input arrays must have same number of dimensions 
            cmax        = np.max(c[0::2,:,:], axis=0)
            greenframe  = np.stack([greenframe,cmax],axis=2).max(axis=2)
            # greenframe  = np.max(greenframe, axis=0)
            cmax        = np.max(c[1::2,:,:], axis=0)
            redframe    = np.stack([redframe,cmax],axis=2).max(axis=2)
            # redframe    = np.concatenate([redframe,c[1::2,:,:]],axis=0)
            # redframe    = np.max(redframe, axis=0)
          
outpath = outputdirec + 'NSH07429_greenwindow_max.tif'
fH = open(outpath,'wb') #as fH:
tifffile.imwrite(fH,greenframe.astype('int16'), bigtiff=True)

outpath = outputdirec + 'NSH07429_redwindow_max.tif'
fH = open(outpath,'wb') #as fH:
tifffile.imwrite(fH,redframe.astype('int16'), bigtiff=True)
