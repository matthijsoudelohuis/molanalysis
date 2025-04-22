"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script makes an average image of the mesoview data for each session
2Pram Mesoscope data

"""
#%% Import packages
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
from utils.imagelib import im_norm,im_norm8,im_log,im_sqrt

#%% Set parameters
rawdatadir      = "F:\\Mesoviews\\"
# rawdatadir      = "W:\\Users\\Matthijs\\Rawdata\\PILOTS\\"
outputdir           = os.path.join(get_data_folder(),"OV")

animal_ids          = [] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE10919'] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE09830','LPE09831'] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE11086'] #If empty than all animals in folder will be processed

cmred = LinearSegmentedColormap.from_list(
        "Custom", [(0, 0, 0), (1, 0, 0)], N=100)
cmgreen = LinearSegmentedColormap.from_list(
        "Custom", [(0, 0, 0), (0, 1, 0)], N=100)

lowprc = 0.5 #scaling minimum percentile
uppprc = 99 #scaling maximum percentile

# clr_rchan = np.array(ImageColor.getcolor('#ff0040', "RGB")) / 255
# clr_gchan = np.array(ImageColor.getcolor('#00ffbf', "RGB")) / 255
clr_rchan = [1,0,0]
clr_gchan = [0,1,0]

#%% Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = os.listdir(rawdatadir)

for animal_id in animal_ids: #for each animal
    
    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 

    for sessiondate in sessiondates: #for each of the sessions for this animal
        
        sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)
        
        ovfolder        = os.path.join(sesfolder,'OV')

        if os.path.exists(ovfolder):
            mimg = np.empty([0,0])
            mimg2 = np.empty([0,0])
            
            for x in os.listdir(ovfolder):
                if x.endswith(".tif"):
                    mROI_data, meta = split_mROIs(os.path.join(ovfolder,x))

                    # for iframe in range(np.shape(mROI_data[0])[0])
                    c           = np.concatenate(mROI_data[:], axis=2) #reshape to full ROI (size frames by xpix by ypix)

                    if np.shape(mimg)[0]==0:
                        mimg  = np.empty(np.shape(c)[1:])
                        mimg2    = np.empty(np.shape(c)[1:])

                    cmax        = np.max(c[0::2,:,:], axis=0)
                    mimg  = np.stack([mimg,cmax],axis=2).max(axis=2)

                    cmax        = np.max(c[1::2,:,:], axis=0)
                    mimg2    = np.stack([mimg2,cmax],axis=2).max(axis=2)
                    del mROI_data,c,cmax #free up memory
                    time.sleep(0.5) #for memory management
            
            mimg    = im_norm8(mimg,min=lowprc,max=99.8) #scale between 0 and 255

            mimg2   = im_norm(mimg2,min=lowprc,max=uppprc) #scale between 0 and 255
            mimg2   = im_sqrt(mimg2) #square root transform to enhance weakly expressing cells
            mimg2   = im_norm(mimg2,min=50,max=100) #scale between 0 and 255

            # coeff       = estimate_correc_coeff(mimg,mimg2)
            # mimg  = bleedthrough_correction(mimg,mimg2)

            if not os.path.exists(os.path.join(outputdir,animal_id)):
                os.makedirs(os.path.join(outputdir,animal_id))

            outpath = os.path.join(outputdir,animal_id,animal_id + '_' + sessiondate + '_green.tif')
            fH = open(outpath,'wb') #as fH:
            tifffile.imwrite(fH,mimg.astype('int16'), bigtiff=True)

            outpath = os.path.join(outputdir,animal_id,animal_id + '_' + sessiondate + '_red.tif')
            fH = open(outpath,'wb') #as fH:
            tifffile.imwrite(fH,mimg2.astype('int16'), bigtiff=True)
            
            rchan = (mimg2 - np.min(mimg2)) / (np.max(mimg2) - np.min(mimg2))
            gchan = (mimg - np.min(mimg)) / (np.max(mimg) - np.min(mimg))

            fig, axes = plt.subplots(1,3,figsize=(30,10))

            # axes[0].imshow(mimg,cmap=cmgreen,vmin=0,vmax=1)
            axes[0].imshow(gchan,cmap=cmgreen,vmin=0,vmax=1)
           
            axes[1].imshow(rchan,cmap=cmred,vmin=0,vmax=1)

            im3 = rchan[:,:,np.newaxis] * clr_rchan + gchan[:,:,np.newaxis] * clr_gchan

            axes[2].imshow(im3,vmin=0,vmax=1)

            for ax in axes.flatten():
                ax.set_axis_off()
                ax.set_aspect('auto')
            fig.suptitle(f'{animal_id}')
            plt.tight_layout(rect=[0, 0, 1, 1])

            # Save the full figure...
            outpath = os.path.join(outputdir,animal_id,animal_id + '_' + sessiondate + '_merge.png')
            fig.savefig(outpath, bbox_inches='tight', pad_inches=0)
            outpath = os.path.join(outputdir,animal_id,animal_id + '_' + sessiondate + '_merge.pdf')
            fig.savefig(outpath, bbox_inches='tight', pad_inches=0)
