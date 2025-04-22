"""
Analyzes stack of GCaMP and tdtomato expressing cells using cellpose software (Pachitariu & Stringer)
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

#%% Import packages
import os
import numpy as np
os.chdir('e:\\Python\\molanalysis\\')
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted 
from PIL import ImageColor

from loaddata.get_data_folder import *
from utils.imagelib import im_norm,im_norm8,im_log,im_sqrt

#%% ###################################################################

##### Show mean planes for each session and save in original dir:
protocols           = ['GR','IM','GN']
animal_ids          = [] #If empty than all animals in folder will be processed
date_filter         = []
# animal_ids          = ['LPE11086'] #If empty than all animals in folder will be processed
# date_filter         = ['2023_12_16']

clr_rchan = np.array(ImageColor.getcolor('#ff0040', "RGB")) / 255
clr_gchan = np.array(ImageColor.getcolor('#00ffbf', "RGB")) / 255
clr_rchan = [1,0,0]
clr_gchan = [0,1,0]

lowprc = 0.5 #scaling minimum percentile
uppprc = 99 #scaling maximum percentile

## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids =  get_animals_protocol(protocols)
    # [f.name for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]

for animal_id in tqdm(animal_ids, desc="Processing animal", total=len(animal_ids)): #for each animal

    rawdatadir = get_rawdata_drive(animal_id,protocols)

    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id))
    sessiondates = [x for x in sessiondates if os.path.isdir(os.path.join(rawdatadir,animal_id,x))]

    if any(date_filter): #If dates specified, then process only those:
        sessiondates = [x for x in sessiondates if x in date_filter]

    for sessiondate in sessiondates: #for each of the sessions for this animal
        suite2pfolder       = os.path.join(rawdatadir,animal_id,sessiondate,'suite2p')
        if os.path.exists(suite2pfolder):
            plane_folders = natsorted([f.path for f in os.scandir(suite2pfolder) if f.is_dir() and f.name[:5]=='plane'])

            fig, axes = plt.subplots(8, 3,figsize=(3*2,8*2))
            for iplane, plane_folder in enumerate(plane_folders):

                # load ops of plane0:
                ops                = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()

                #standard mean image:
                # mimg = ops['meanImg'] 
                #max projection:
                mimg = np.zeros([512,512])
                mimg[ops['yrange'][0]:ops['yrange'][1],
                    ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']

                mimg = im_norm8(mimg,min=lowprc,max=uppprc) #scale between 0 and 255

                ## Get red image:
                mimg2 = ops['meanImg_chan2'] #get red channel image from ops

                mimg2 = im_norm(mimg2,min=lowprc,max=uppprc) #scale between 0 and 255
                mimg2 = im_sqrt(mimg2) #square root transform to enhance weakly expressing cells
                mimg2 = im_norm(mimg2,min=5,max=100) #scale between 0 and 255

                ######

                rchan = (mimg2 - np.min(mimg2)) / (np.max(mimg2) - np.min(mimg2))
                gchan = (mimg - np.min(mimg)) / (np.max(mimg) - np.min(mimg))
                
                bchan = np.zeros(np.shape(mimg))

                axes[iplane,0].imshow(gchan,cmap='gray',vmin=0,vmax=1)
                axes[iplane,1].imshow(rchan,cmap='gray',vmin=0,vmax=1)
        
                im3 = rchan[:,:,np.newaxis] * clr_rchan + gchan[:,:,np.newaxis] * clr_gchan
        
                axes[iplane,2].imshow(im3)

        for ax in axes.flatten():
            ax.set_axis_off()
            ax.set_aspect('auto')
        fig.suptitle(f'{animal_id} - {sessiondate}')
        plt.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(os.path.join(rawdatadir,animal_id,sessiondate,'PlaneOverview_%s_%s.png' % (animal_id,sessiondate)), 
                    dpi=300,format = 'png')       
        fig.savefig(os.path.join(rawdatadir,animal_id,sessiondate,'PlaneOverview_%s_%s.pdf' % (animal_id,sessiondate)), 
                    dpi=300,format = 'pdf')

print(f'\n\nPreprocessing Completed')
