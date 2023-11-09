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

# os.chdir('T:\\Python\\molanalysis\\')

rawdatadir      = "X:\\Rawdata\\"
procdatadir     = "T:\\Python\\cellpose\\"


## Loop over all selected animals and folders
# animal_ids = os.listdir(rawdatadir)

# animal_dirs = [f.path for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]

def normalize8(I):
    # mn = I.min()
    # mx = I.max()

    mn = np.percentile(I,0.5)
    mx = np.percentile(I,99.5)

    mx -= mn

    I = ((I - mn)/mx) * 255
    I[I<0] =0
    I[I>255] = 255
    
    return I.astype(np.uint8)

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
                # img_numpy[:,:,1] = int((ops['meanImg'] / np.max(ops['meanImg']))*256)
                img_numpy[:,:,1] = normalize8(ops['meanImg'])

                img = Image.fromarray(img_numpy, "RGB")

                # Save the Numpy array as Image
                image_filename = "_".join([animal_id, sessiondate,str(iplane),'green.tiff'])
                img.save(os.path.join(procdatadir,'greenlib_tiff',image_filename))

                img_numpy = np.zeros((512, 512, 3), dtype=np.uint8)
                # img_numpy[:,:,0] = ops['meanImg_chan2']
                
                mimg2 = ops['meanImg_chan2']
                mimg2 = np.log(mimg2 - np.min(mimg2))

                img_numpy[:,:,0] = normalize8(mimg2)

                img = Image.fromarray(img_numpy, "RGB")

                # Save the Numpy array as Image
                # image_filename = "_".join([animal_id, sessiondate,str(iplane),'red.jpeg'])
                # img.save(os.path.join(procdatadir,'redlib_tiff',image_filename))

                image_filename = "_".join([animal_id, sessiondate,str(iplane),'red.tiff'])
                img.save(os.path.join(procdatadir,'redlib_tiff',image_filename))

                # #Save sessiondata:
                # sessiondata.to_csv(os.path.join(outdir,"sessiondata.csv"), sep=',')

print(f'\n\nPreprocessing Completed')



