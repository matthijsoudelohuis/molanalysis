"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script makes an average image of the mesoview data for each session
2Pram Mesoscope data

"""

import os
os.chdir('e:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive,get_data_folder
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))
import numpy as np
import tifffile
from utils.twoplib import split_mROIs
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from labeling.label_lib import bleedthrough_correction, estimate_correc_coeff
import time

rawdatadir      = "F:\\Mesoviews\\"
# rawdatadir      = "W:\\Users\\Matthijs\\Rawdata\\PILOTS\\"
outputdir           = os.path.join(get_data_folder(),"OV")

animal_ids          = [] #If empty than all animals in folder will be processed
animal_ids          = ['LPE10919'] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE09830','LPE09831'] #If empty than all animals in folder will be processed

cmred = LinearSegmentedColormap.from_list(
        "Custom", [(0, 0, 0), (1, 0, 0)], N=100)
cmgreen = LinearSegmentedColormap.from_list(
        "Custom", [(0, 0, 0), (0, 1, 0)], N=100)


## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = os.listdir(rawdatadir)

for animal_id in animal_ids: #for each animal
    
    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 

    for sessiondate in sessiondates: #for each of the sessions for this animal
        
        sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)
        
        ovfolder        = os.path.join(sesfolder,'OV')

        if os.path.exists(ovfolder):
            greenframe = np.empty([0,0])
            redframe = np.empty([0,0])
            
            for x in os.listdir(ovfolder):
                if x.endswith(".tif"):
                    mROI_data, meta = split_mROIs(os.path.join(ovfolder,x))

                    # for iframe in range(np.shape(mROI_data[0])[0])
                    c           = np.concatenate(mROI_data[:], axis=2) #reshape to full ROI (size frames by xpix by ypix)

                    if np.shape(greenframe)[0]==0:
                        greenframe  = np.empty(np.shape(c)[1:])
                        redframe    = np.empty(np.shape(c)[1:])

                    cmax        = np.max(c[0::2,:,:], axis=0)
                    greenframe  = np.stack([greenframe,cmax],axis=2).max(axis=2)

                    cmax        = np.max(c[1::2,:,:], axis=0)
                    redframe    = np.stack([redframe,cmax],axis=2).max(axis=2)
                    del mROI_data,c,cmax #free up memory
                    time.sleep(0.5) #for memory management
            
            # coeff       = estimate_correc_coeff(greenframe,redframe)

            # greenframe  = bleedthrough_correction(greenframe,redframe,coeff=0)

            outpath = os.path.join(outputdir,animal_id + '_' + sessiondate + '_green.tif')
            fH = open(outpath,'wb') #as fH:
            tifffile.imwrite(fH,greenframe.astype('int16'), bigtiff=True)

            outpath = os.path.join(outputdir,animal_id + '_' + sessiondate + '_red.tif')
            fH = open(outpath,'wb') #as fH:
            tifffile.imwrite(fH,redframe.astype('int16'), bigtiff=True)
            
            if not os.path.exists(os.path.join(outputdir,animal_id)):
                os.makedirs(os.path.join(outputdir,animal_id))

            fig1, ax1 = plt.subplots(figsize=(10,10))
            ax1.imshow(greenframe,cmap=cmgreen,vmin=np.percentile(greenframe,1),vmax=np.percentile(greenframe,99))
            plt.axis('off')
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
            plt.tight_layout()
            # Save the full figure...
            outpath = os.path.join(outputdir,animal_id,animal_id + '_' + sessiondate + '_green.png')
            fig1.savefig(outpath, bbox_inches='tight', pad_inches=0)

            fig2, ax2 = plt.subplots(figsize=(10,10))
            ax2.imshow(redframe,cmap=cmred,vmin=np.percentile(redframe,15),vmax=np.percentile(redframe,99))
            plt.axis('off')
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
            plt.tight_layout()
            # Save the full figure...
            outpath = os.path.join(outputdir,animal_id,animal_id + '_' + sessiondate + '_red.png')
            fig2.savefig(outpath, bbox_inches='tight', pad_inches=0)
