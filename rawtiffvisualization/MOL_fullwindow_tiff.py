import os
import numpy as np
import tifffile
from twoplib import split_mROIs
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

rawdatadir      = "X:\\Rawdata\\"
rawdatadir      = "W:\\Users\\Matthijs\\Rawdata\\"
outputdir      = "V:\\Procdata\\OV\\"

animal_ids          = ['LPE09665'] #If empty than all animals in folder will be processed
animal_ids          = ['LPE09830','LPE09831'] #If empty than all animals in folder will be processed
animal_ids          = ['NSH07422'] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE09833'] #If empty than all animals in folder will be processed

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
                    
                    # c           = np.concatenate(mROI_data[1:3][1,:,:], axis=2) #reshape to full ROI (size frames by xpix by ypix)
                    # c           = np.concatenate(mROI_data[1:3], axis=2) #reshape to full ROI (size frames by xpix by ypix)

                    
                    if np.shape(greenframe)[0]==0:
                        greenframe  = np.empty(np.shape(c)[1:])
                        redframe    = np.empty(np.shape(c)[1:])

                    cmax        = np.max(c[0::2,:,:], axis=0)
                    greenframe  = np.stack([greenframe,cmax],axis=2).max(axis=2)

                    cmax        = np.max(c[1::2,:,:], axis=0)
                    redframe    = np.stack([redframe,cmax],axis=2).max(axis=2)

                    
            outpath = os.path.join(outputdir,animal_id + '_' + sessiondate + '_green.tif')
            fH = open(outpath,'wb') #as fH:
            tifffile.imwrite(fH,greenframe.astype('int16'), bigtiff=True)

            outpath = os.path.join(outputdir,animal_id + '_' + sessiondate + '_red.tif')
            fH = open(outpath,'wb') #as fH:
            tifffile.imwrite(fH,redframe.astype('int16'), bigtiff=True)
            
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,6.5))

            ax1.imshow(greenframe,cmap=cmgreen,vmin=np.percentile(greenframe,1),vmax=np.percentile(greenframe,98))
            ax1.set_title('Mean Image Chan 1')
            ax2.imshow(redframe,cmap=cmred,vmin=np.percentile(redframe,15),vmax=np.percentile(redframe,99))
            ax2.set_title('Mean Image Chan 2')

            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
            plt.tight_layout()
            
            # Save the full figure...
            outpath = os.path.join(outputdir,animal_id + '_' + sessiondate + '.png')

            fig.savefig(outpath)

# animal_id = animal_ids[0] #for each animal
# sessiondate = sessiondates[1] #for each animal
# x = os.listdir(ovfolder)[0]
# nchannels = 2 #whether image has both red and green channel acquisition (PMT)
# greenframe = np.empty([4014,3976])
# redframe = np.empty([4014,3976])

# for x in os.listdir(direc):
#     if x.endswith(".tif"):
#             mROI_data, meta = split_mROIs(os.path.join(direc,x))
#             nROIs = len(mROI_data)
#             c           = np.concatenate(mROI_data[:], axis=2)  # ValueError: all the input arrays must have same number of dimensions 
#             cmax        = np.max(c[0::2,:,:], axis=0)
#             greenframe  = np.stack([greenframe,cmax],axis=2).max(axis=2)
#             # greenframe  = np.max(greenframe, axis=0)
#             cmax        = np.max(c[1::2,:,:], axis=0)
#             redframe    = np.stack([redframe,cmax],axis=2).max(axis=2)
#             # redframe    = np.concatenate([redframe,c[1::2,:,:]],axis=0)
#             # redframe    = np.max(redframe, axis=0)
          
# outpath = outputdirec + 'NSH07429_greenwindow_max.tif'
# fH = open(outpath,'wb') #as fH:
# tifffile.imwrite(fH,greenframe.astype('int16'), bigtiff=True)

# outpath = outputdirec + 'NSH07429_redwindow_max.tif'
# fH = open(outpath,'wb') #as fH:
# tifffile.imwrite(fH,redframe.astype('int16'), bigtiff=True)



