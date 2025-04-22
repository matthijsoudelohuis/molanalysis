import os
os.chdir('T:\\Python\\molanalysis\\')
import numpy as np
import tifffile
from utils.twoplib import split_mROIs
import cv2
import matplotlib.pyplot as plt
import imageio

savedir  = "T:\\OneDrive\\PostDoc\\Figures\\WindowVideo\\"

rawdatadir = 'W:\\Users\\Matthijs\\Rawdata\\LPE10885\\2023_10_12\\OV_lowSR_highTR\\'
savefilename = 'LPE10885_OV_highTR'
size = 376, 360
fps = 15

rawdatadir = 'W:\\Users\\Matthijs\\Rawdata\\LPE10885\\2023_10_12\\OV\\'
savefilename = 'LPE10885_OV_slowTR'
size = 3976, 4014
fps = 5


viddata = np.zeros((0,size[1],size[0]))

for x in os.listdir(rawdatadir):
    if x.endswith(".tif"):
        mROI_data, meta = split_mROIs(os.path.join(rawdatadir,x))

        c           = np.concatenate(mROI_data[:], axis=2) #reshape to full ROI (size frames by xpix by ypix)
        
        viddata = np.vstack((viddata,c[0::2,:,:])) #Keep only green channel data

viddata = viddata - np.percentile(viddata,15)
viddata[viddata<0] = 0
viddata = viddata / np.percentile(viddata,95) * 255
viddata[viddata>255] = 255
viddata = np.uint8(viddata)

## Make a mp4 video of it:
totalframes = np.shape(viddata)[0]
# out = cv2.VideoWriter(os.path.join(savedir,savefilename +  '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
out = cv2.VideoWriter(os.path.join(savedir,savefilename +  '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size), True)
# out = cv2.VideoWriter(os.path.join(savedir,savefilenameavi), cv2.VideoWriter_fourcc('P','I','M','1'), fps, (size[1], size[0]), False)
for ifr in range(totalframes):
    data = np.dstack((np.zeros((size[1],size[0])),viddata[ifr,:,:],np.zeros((size[1],size[0])))).astype(np.uint8)
    out.write(data)
    # out.write(viddata[ifr,:,:])
out.release()


# Make a gif out of it: 
def seqtogif(viddata,savedir,savefilename,fps=20):
    nframes = np.shape(viddata)[0]
    plt.figure()
    filenames = []
    for ifr in range(np.min([nframes,200])):
    
        data = np.dstack((np.zeros(size),viddata[ifr,:,:],np.zeros(size))).astype(np.uint8)
        plt.figure()
        plt.imshow(data,vmin=0, vmax=255)
        plt.axis('off')
        plt.tight_layout()

        # create file name and append it to a list
        filename = f'{ifr}.png'
        filenames.append(os.path.join(savedir, filename))
        
        # save frame
        plt.savefig(os.path.join(savedir, filename))
        plt.close()
        
    # Load each file into a list
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    # Save them as frames into a gif 
    exportname = os.path.join(savedir, savefilename)
    imageio.mimsave(exportname, frames, 'GIF', fps=fps)
    
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    
    return


seqtogif(viddata,savedir,savefilename + '.gif',fps=20)
