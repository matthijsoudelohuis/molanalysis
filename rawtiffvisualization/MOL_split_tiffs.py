
import os
import twoplib
from twoplib import *

direc  = 'V:/Rawdata/PILOTS/20221103_NSH0429_windowtiltpilot/Imagingpilot/'

direc  = 'V:/Rawdata/PILOTS/20221108_NSH0429_Spontaneous4ROI/Spontaneous/'

direc  = 'V:/Rawdata/PILOTS/20221108_NSH0429_Spontaneous4ROI/Window/'

direc = 'X:\\RawData\\VR\\NSH07422\\2022_12_09\\OV'
direc = 'X:\\RawData\\VR\\NSH07422\\2022_12_09\\VR'


# todo:
#     -save .json and meta only for the first file perhaps
#     -integrate into automatic pipeline of suite2p processing

    
for x in os.listdir(direc):
    if x.endswith(".tif"):
        # Prints only text file present in My Folder
                # print(x)
                # [] = split_mROIs(fname):
            split_and_save_mROIs(os.path.join(direc,x))


# split_and_save_mROIs(fpath)
