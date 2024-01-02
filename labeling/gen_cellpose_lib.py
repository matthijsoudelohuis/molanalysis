# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

generate cellpose image library from all the suite2p motion registered average images
of recording sessions found in the rawdatadirs
"""

import os
import numpy as np
from natsort import natsorted 
from PIL import Image
from utils.imagelib import im_norm8

# os.chdir('T:\\Python\\molanalysis\\')

rawdatadir      = "X:\\Rawdata\\"
procdatadir     = "T:\\Python\\cellpose\\"

## Loop over all animals and folders
animal_ids = [f.name for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]

for animal_id in animal_ids: #for each animal

    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id))
    
    for sessiondate in sessiondates: #for each of the sessions for this animal
        
        suite2pfolder       = os.path.join(rawdatadir,animal_id,sessiondate,'suite2p')
        
        if os.path.exists(suite2pfolder):

            plane_folders = natsorted([f.path for f in os.scandir(suite2pfolder) if f.is_dir() and f.name[:5]=='plane'])

            for iplane, plane_folder in enumerate(plane_folders):
                # load ops of plane0:
                ops                = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()

                img_numpy = np.zeros((512, 512, 3), dtype=np.uint8)
                img_numpy[:,:,1] = im_norm8(ops['meanImg'],min=0.5,max=99.5)

                img = Image.fromarray(img_numpy, "RGB")

                # Save the Numpy array as Image
                image_filename = "_".join([animal_id, sessiondate,str(iplane),'green.tiff'])
                img.save(os.path.join(procdatadir,'greenlib_tiff',image_filename))

                img_numpy = np.zeros((512, 512, 3), dtype=np.uint8)
                # img_numpy[:,:,0] = ops['meanImg_chan2']
                
                mimg2 = ops['meanImg_chan2']
                mimg2 = np.log(mimg2 - np.min(mimg2))

                img_numpy[:,:,0] = im_norm8(mimg2,min=0.5,max=99.5)

                img = Image.fromarray(img_numpy, "RGB")

                image_filename = "_".join([animal_id, sessiondate,str(iplane),'red.tiff'])
                img.save(os.path.join(procdatadir,'redlib_tiff',image_filename))

print(f'\n\nPreprocessing Completed')



