# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:50:43 2023

@author: USER
"""

# import sys, os
# from pathlib import Path
# import json

# https://pypi.org/project/scanimage-tiff-reader/
from ScanImageTiffReader import ScanImageTiffReader as imread
import matplotlib.pyplot as plt
import numpy as np
import os 
import imageio


fname = 'V:\\Rawdata\\PILOTS\\20230308_LPE09832_tdTomato_Bleedthrough\\Gain1_0.6_Gain2_0.6\\LPE09832_SP_00001_00001.tif'
fname = 'V:\\Rawdata\\PILOTS\\20230308_LPE09832_tdTomato_Bleedthrough\\Gain1_0.7_Gain2_0.5\\LPE09832_SP_00001_00001.tif'
fname = 'V:\\Rawdata\\PILOTS\\20230308_LPE09832_tdTomato_Bleedthrough\\CloseUp_zoom5x_Gain0.6_0.6\\LPE09832_SP_00001_00001.tif'

directory = 'V:\\Rawdata\\PILOTS\\20230308_LPE09832_tdTomato_Bleedthrough\\Gain1_0.6_Gain2_0.6\\'
directory = 'V:\\Rawdata\\PILOTS\\20230308_LPE09832_tdTomato_Bleedthrough\\CloseUp_zoom5x_Gain0.6_0.6\\'
directory = 'V:\\Rawdata\\PILOTS\\20230308_LPE09832_tdTomato_Bleedthrough\\Gain1_0.8_Gain2_0.6\\'


# directory = 'C:\\TempData\\LPE09665\\2023_03_14\\GR\\Imaging\\'

#####################################################################################

data_green      = np.empty([0,512,512])
data_red        = np.empty([0,512,512])

# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        reader = imread(f)
        Data = reader.data()
        data_green = np.append(data_green, Data[0::2,:,:],axis=0)
        data_red = np.append(data_red, Data[1::2,:,:],axis=0)
        

## Show 
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17,5))

greenchanim = np.average(data_green,axis=0)
ax1.imshow(greenchanim,vmin=0, vmax=1000)
ax1.set_title('Chan 1')
ax1.set_axis_off()

redchanim = np.average(data_red,axis=0)
ax2.imshow(redchanim,vmin=0, vmax=10000)
ax2.set_title('Chan 2')
ax2.set_axis_off()

greenchan = greenchanim.reshape(1,512*512)[0]
redchan = redchanim.reshape(1,512*512)[0]

ax3.scatter(redchan,greenchan,0.02)
ax3.set_xlabel('Chan 2')
ax3.set_ylabel('Chan 1')

# Fit linear regression via least squares with numpy.polyfit
b, a = np.polyfit(redchan, greenchan, deg=1)

xseq = np.linspace(-15000, 32000, num=32000)
# Plot regression line
ax3.plot(xseq, a + b * xseq, color="k", lw=1.5);

ax3.set_xlim([-8000,32000])
ax3.set_ylim([-800,10000])

txt1 = "Coefficient is %1.4f" % b

plt.text(-500, 1200,txt1, fontsize=12)

#####################################################################################

directory = 'V:\\Rawdata\\PILOTS\\20230308_LPE09832_tdTomato_Bleedthrough\\NSH07429_2023_03_12_Gain0.6_Gain2_0.6\\'

data_green      = np.empty([0,512,512])
data_red        = np.empty([0,512,512])

# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        reader = imread(f)
        Data = reader.data()
        data_green = np.append(data_green, Data[2::16,:,:],axis=0)
        data_red = np.append(data_red, Data[3::16,:,:],axis=0)


## Show 
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17,5))

greenchanim = np.average(data_green,axis=0)
ax1.imshow(greenchanim,vmin=0, vmax=1000)
ax1.set_title('Chan 1')
ax1.set_axis_off()

redchanim = np.average(data_red,axis=0)
ax2.imshow(redchanim,vmin=0, vmax=10000)
ax2.set_title('Chan 2')
ax2.set_axis_off()

greenchan = greenchanim.reshape(1,512*512)[0]
redchan = redchanim.reshape(1,512*512)[0]

ax3.scatter(redchan,greenchan,0.02)
# ax3.scatter(data_green.flatten(),data_red.flatten(),0.02)
ax3.set_xlabel('Chan 2')
ax3.set_ylabel('Chan 1')

# Plot regression line
ax3.set_xlim([-2000,32000])
ax3.set_ylim([-500,8000])

xseq = np.linspace(-15000, 32000, num=32000)

ax3.plot(xseq, 0.0668 * xseq, color="k", lw=1.5);
txt1 = "%1.4f (from control animal)" % 0.0668
plt.text(10000, 0,txt1, fontsize=10)

b, a = np.polyfit(redchan, greenchan, deg=1)
txt1 = "%1.4f" % b
ax3.plot(xseq, a + b * xseq, color="r", lw=1.5);
plt.text(10000, 4000,txt1, fontsize=10,color="r")

#####################################################################################


greendirectory = 'V:\\Rawdata\\PILOTS\\20230322_LPE09665_tdTomato_correction_temp\\chan0\\'
reddirectory = 'V:\\Rawdata\\PILOTS\\20230322_LPE09665_tdTomato_correction_temp\\chan1\\'

data_green      = np.empty([0,512,512])
data_red        = np.empty([0,512,512])

# iterate over files in
# that directory
for filename in os.listdir(greendirectory):
    f = os.path.join(greendirectory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        reader = imread(f)
        Data = reader.data()
        data_green = np.append(data_green,Data,axis=0)
    
for filename in os.listdir(reddirectory):
    f = os.path.join(reddirectory, filename)
            # checking if it is a file
    if os.path.isfile(f):
        print(f)
        reader = imread(f)
        Data = reader.data()
        data_red = np.append(data_red,Data,axis=0)

## Show 
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17,5))
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(10,6.5))

greenchanim = np.average(data_green,axis=0)
ax1.imshow(greenchanim,vmin=-200, vmax=6000)
ax1.set_title('Chan 1')
ax1.set_axis_off()

redchanim = np.average(data_red,axis=0)
ax2.imshow(redchanim,vmin=-200, vmax=6000)
ax2.set_title('Chan 2')
ax2.set_axis_off()

greenchan = greenchanim.reshape(1,512*512)[0]
redchan = redchanim.reshape(1,512*512)[0]

ax3.scatter(redchan,greenchan,0.02)
ax3.set_xlabel('Chan 2')
ax3.set_ylabel('Chan 1')

# Fit linear regression via least squares with numpy.polyfit
b, a = np.polyfit(redchan, greenchan, deg=1)

xseq = np.linspace(-15000, 32000, num=32000)
# Plot regression line
ax3.plot(xseq, a + b * xseq, color="k", lw=1.5);

ax3.set_xlim([-800,10000])
ax3.set_ylim([-800,10000])

txt1 = "Coefficient is %1.4f" % b

ax3.text(2500,1000,txt1, fontsize=9)

#Correction:
data_green_corr = data_green - b * data_red

greenchanim = np.average(data_green_corr,axis=0)
ax4.imshow(greenchanim,vmin=-200, vmax=6000)
ax4.set_title('Chan 1')
ax4.set_axis_off()

redchanim = np.average(data_red,axis=0)
ax5.imshow(redchanim,vmin=-200, vmax=6000)
ax5.set_title('Chan 2')
ax5.set_axis_off()

greenchan = greenchanim.reshape(1,512*512)[0]
redchan = redchanim.reshape(1,512*512)[0]

ax6.scatter(redchan,greenchan,0.02)
ax6.set_xlabel('Chan 2')
ax6.set_ylabel('Chan 1')

# Fit linear regression via least squares with numpy.polyfit
b2, a2 = np.polyfit(redchan, greenchan, deg=1)

xseq = np.linspace(-15000, 32000, num=32000)
# Plot regression line
ax6.plot(xseq, a2 + b2 * xseq, color="k", lw=1.5);

ax6.set_xlim([-800,10000])
ax6.set_ylim([-800,10000])

txt1 = "Coefficient is %1.4f" % b2

ax6.text(2500,1000,txt1, fontsize=9)


###################### Make gif:



def seqtogif(data,savedir,savefilename):
    nframes = np.shape(data)[0]

    filenames = []
    for i in range(np.min([nframes,1000])):
    # for i in range(20):
    
        plt.imshow(data[i,:,:],vmin=-200, vmax=8000)
        plt.axis('off')
        
        # create file name and append it to a list
        filename = f'{i}.png'
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
    imageio.mimsave(exportname, frames, 'GIF', fps=30)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    
    return
    


savedir = 'T:\\OneDrive\\PostDoc\\Figures\\Imaging - tdTomatoCorrection\\GIF_correction\\'

seqtogif(data_green,savedir,savefilename = 'green_orig.gif')
seqtogif(data_green_corr,savedir,savefilename = 'green_corr.gif')

temp = np.repeat(np.average(data_red,axis=0)[np.newaxis,:, :], np.shape(data_green)[0], axis=0)
data_green_corr_v2 = data_green - b * temp
seqtogif(data_green_corr_v2,savedir,savefilename = 'green_corr_v2.gif')


# reader = imread(fname)
# Data = reader.data()

# ## Show 
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17,5))

# greenchanim = np.average(Data[0::2,:,:],axis=0)
# ax1.imshow(greenchanim)
# ax1.set_title('Chan 1')

# redchanim = np.average(Data[1::2,:,:],axis=0)
# ax2.imshow(redchanim)
# ax2.set_title('Chan 2')

# greenchan = greenchanim.reshape(1,512*512)[0]
# redchan = redchanim.reshape(1,512*512)[0]

# ax3.scatter(redchan,greenchan,0.01)


# # greenchan = Data[0,:,:].reshape(1,512*512)[0]
# # redchan = Data[1,:,:].reshape(1,512*512)[0]

# ### 
# fig, ax = plt.subplots(figsize = (9, 9))

# # Add scatterplot
# ax.scatter(redchan, greenchan, s=2, alpha=0.7, edgecolors="k")

# # Fit linear regression via least squares with numpy.polyfit
# b, a = np.polyfit(redchan, greenchan, deg=1)

# xseq = np.linspace(-15000, 32000, num=32000)
# # Plot regression line
# ax.plot(xseq, a + b * xseq, color="k", lw=2.5);

# ax.set_xlim([-10000,32000])
# ax.set_ylim([-10000,32000])

# print(b)