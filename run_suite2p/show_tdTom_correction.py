# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:50:43 2023

@author: USER
"""

from ScanImageTiffReader import ScanImageTiffReader as imread
import matplotlib.pyplot as plt
import numpy as np
import os 
from suite2p.detection.chan2detect import correct_bleedthrough
import pandas as pd
import seaborn as sns
from scipy.stats import zscore

# import imageio


#####################################################################################
def load_data(directory,nplanes): 
    data_green      = np.empty([0,512,512])
    data_red        = np.empty([0,512,512])
    maxtifs         = 20
    
    # iterate over files in that directory
    randfiles = os.listdir(directory)
    randfiles = np.random.choice(randfiles,maxtifs)
    for filename in randfiles:
        f = os.path.join(directory, filename)
        
        if f.endswith(".tif"): # checking if it is a tiff file
            print(f)
            reader      = imread(f)
            Data        = reader.data()
            data_green  = np.append(data_green, Data[0::(2*nplanes),:,:],axis=0)
            data_red    = np.append(data_red, Data[1::(2*nplanes),:,:],axis=0)
    
    return data_green,data_red

## plotting func:
def plot_correction(im1,im2,im1_corr):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17,5))
    
    ax1.imshow(im1,vmin=np.percentile(im1,1), vmax=np.percentile(im1,99))
    ax1.set_title('Chan 1')
    ax1.set_axis_off()
    
    ax2.imshow(im2,vmin=np.percentile(im2,1), vmax=np.percentile(im2,99))
    ax2.set_title('Chan 2')
    ax2.set_axis_off()
        
    ax3.imshow(im1_corr,vmin=np.percentile(im1,1), vmax=np.percentile(im1,99))
    ax3.set_title('Chan 2')
    ax3.set_axis_off()

## correlation func:
def plot_correlation(im1,im2):
    ## Show linear correction:
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
        
    ax1.scatter(im2.flatten(),im1.flatten(),0.02)
    ax1.set_xlabel('Chan 2')
    ax1.set_ylabel('Chan 1')
    
    # Fit linear regression via least squares with numpy.polyfit
    b, a = np.polyfit(im2.flatten(),im1.flatten(), deg=1)
    
    xseq = np.linspace(-15000, 32000, num=32000)
    # Plot regression line
    ax1.plot(xseq, a + b * xseq, color="k", lw=1.5);
    
    ax1.set_xlim(np.percentile(im2.flatten()*1.05,(0,100)))
    ax1.set_ylim(np.percentile(im1.flatten()*1.05,(0,100)))
    
    txt1 = "Coefficient is %1.4f" % b
    
    plt.text(-500, 1200,txt1, fontsize=12)


##########################################

directory = 'O:\\RawData\\LPE10192\\2023_05_04\\SP\\Imaging\\'
[data_green,data_red] = load_data(directory,nplanes=4)

directory = 'X:\\RawData\\LPE09665\\2023_03_14\\GR\\Imaging\\'
[data_green,data_red] = load_data(directory,nplanes=8)

directory = 'X:\\RawData\\LPE09667\\2023_03_29\\VR\\Imaging\\'
[data_green,data_red] = load_data(directory,nplanes=8)

directory = 'X:\\RawData\\NSH07422\\2023_03_14\\IM\\Imaging\\'
[data_green,data_red] = load_data(directory,nplanes=8)

directory = 'V:\\Rawdata\\PILOTS\\20230308_LPE09832_tdTomato_Bleedthrough\\Gain1_0.6_Gain2_0.6\\'
[data_green,data_red] = load_data(directory,nplanes=1)

# directory = 'C:\\TempData\\LPE09665\\2023_03_14\\GR\\Imaging\\'

greenchanim     = np.average(data_green,axis=0)
redchanim       = np.average(data_red,axis=0)

# greenchanim = np.average(data_green[:,256:,:256],axis=0)
# redchanim = np.average(data_red[:,256:,:256],axis=0)

b, a = np.polyfit(redchanim.flatten(), greenchanim.flatten(), deg=1)

plot_correlation(greenchanim,redchanim)
plot_correction(greenchanim,redchanim,greenchanim - b * redchanim)

plot_correction(greenchanim,redchanim,greenchanim - 0.068 * redchanim) #0.6 0.6
plot_correction(greenchanim,redchanim,greenchanim - 0.32 * redchanim) #0.6 0.5
plot_correction(greenchanim,redchanim,greenchanim - 1.54 * redchanim) #0.6 0.4

# plt.plot(np.log([0.068,0.32,1.54]))
######################################

# subtract bleedthrough of green into red channel
# non-rigid regression with nblks x nblks pieces
nblks = 1
img1 = greenchanim.copy()
img2 = redchanim.copy()
greenchanim_corrected = correct_bleedthrough(512, 512, nblks, img2, img1)

plot_correction(greenchanim,redchanim,greenchanim_corrected)

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

#######     

#TODO:
# -Loop over planes, because why not
# -do for multiple sessions, not just example session
#- 

dir_corrected = 'X:\\RawData\\LPE09667\\2023_03_29\\suite2p_corrected\\'
dir_uncorrected = 'X:\\RawData\\LPE09667\\2023_03_29\\suite2p_uncorrected\\'

# iplane = 5
# # Load the data of this plane:
# F_corr       = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'F.npy'))
# F2_corr      = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'F_chan2.npy'))
# Fneu_corr    = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'Fneu.npy'))
# spks_corr    = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'spks.npy'))
# ops_corr     = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'ops.npy'), allow_pickle=True).item()
# iscell_corr  = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'iscell.npy'))
# stat_corr    = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'stat.npy'), allow_pickle=True)
# redcell_corr = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'redcell.npy'))

# dir_corrected = 'V:\\RawData\\tdTom_correction\\LPE09667\\2023_03_29\\suite2p_corrected\\'
# dir_uncorrected = 'V:\\RawData\\tdTom_correction\\LPE09667\\2023_03_29\\suite2p_uncorrected\\'
# # Load the data of this plane:
# F_corr       = np.load(os.path.join(dir_corrected,'F.npy'))
# F2_corr      = np.load(os.path.join(dir_corrected,'F_chan2.npy'))
# Fneu_corr    = np.load(os.path.join(dir_corrected,'Fneu.npy'))
# spks_corr    = np.load(os.path.join(dir_corrected,'spks.npy'))
# ops_corr     = np.load(os.path.join(dir_corrected,'ops.npy'), allow_pickle=True).item()
# iscell_corr  = np.load(os.path.join(dir_corrected,'iscell.npy'))
# stat_corr    = np.load(os.path.join(dir_corrected,'stat.npy'), allow_pickle=True)
# redcell_corr = np.load(os.path.join(dir_corrected,'redcell.npy'))

F_corr  = np.array([])
F2_corr = np.array([])
redcell_corr = np.array([])

F_uncorr  = np.array([])
Fmax_uncorr  = np.array([])
F2_uncorr = np.array([])
redcell_uncorr = np.array([])

for iplane in range(8):
    iscell       = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'iscell.npy'))

    F           = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'F.npy'))
    F_corr      = np.append(F_corr,np.mean(F[iscell[:,0]==1],axis=1))

    F2          = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'F_chan2.npy'))
    F2_corr     = np.append(F2_corr,np.mean(F2[iscell[:,0]==1],axis=1))

    redcell      = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'redcell.npy'))
    redcell_corr = np.append(redcell_corr,redcell[iscell[:,0]==1,0])

    #Uncorrected run:
    iscell       = np.load(os.path.join(dir_uncorrected,'plane%d' %iplane,'iscell.npy'))

    F           = np.load(os.path.join(dir_uncorrected,'plane%d' %iplane,'F.npy'))
    F_uncorr    = np.append(F_uncorr,np.mean(F[iscell[:,0]==1],axis=1))
    Fmax_uncorr = np.append(Fmax_uncorr,np.max(F[iscell[:,0]==1],axis=1))

    F2          = np.load(os.path.join(dir_uncorrected,'plane%d' %iplane,'F_chan2.npy'))
    F2_uncorr   = np.append(F2_uncorr,np.mean(F2[iscell[:,0]==1],axis=1))

    redcell       = np.load(os.path.join(dir_uncorrected,'plane%d' %iplane,'redcell.npy'))
    redcell_uncorr = np.append(redcell_uncorr,redcell[iscell[:,0]==1,0])

df_corr = pd.DataFrame({'F_corr': F_corr, 'F2_corr' : F2_corr, 'redcell_corr' : redcell_corr})
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(5,3))
sns.barplot(data=df_corr,y='F_corr',x='redcell_corr',ax=ax1)

sns.barplot(data=df_corr,y='F2_corr',x='redcell_corr',ax=ax2)

df_uncorr = pd.DataFrame({'F_uncorr': F_uncorr, 'F2_uncorr' : F2_uncorr, 'redcell_uncorr' : redcell_uncorr})
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(5,3))
sns.barplot(data=df_uncorr,y='F_uncorr',x='redcell_uncorr',ax=ax1)

sns.barplot(data=df_uncorr,y='F2_uncorr',x='redcell_uncorr',ax=ax2)

## How much of the fluorescence of labeled cells is due to red bleedthrough?

# What is the fluorescence in the red channel of the labeled cells?
# F2_uncorr[redcell_uncorr==1]
# What is the fluorescence in the green channel of the labeled cells?
# F_uncorr[redcell_uncorr==1]
# What is bleedthrough (with pmt1 0.6 gain and pmt2 0.4 gain)?
# F2_uncorr[redcell_uncorr==1] * 1.54

frac_bleedthrough_uncorrected   = (F2_uncorr * 1.54) / F_uncorr * 100
frac_bleedthrough_uncorrected   = (F2_uncorr * 1.54) / Fmax_uncorr * 100
frac_bleedthrough_corrected     = (F2_corr * 1.54) / F_corr * 100

# frac_bleedthrough_uncorrected = (F2_uncorr * 1.54) / F_uncorr * 100

# frac_bleedthrough_uncorrected = F2_corr * 1.54 / (F_corr + F2_corr * 1.54) * 100


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(5,3))
# sns.scatterplot(x=F_uncorr[redcell_uncorr==1],y=frac_bleedthrough_uncorrected[redcell_uncorr==1],ax=ax1)
sns.scatterplot(x=F_uncorr,y=frac_bleedthrough_uncorrected,hue=redcell_uncorr,ax=ax1,size=5)
# sns.scatterplot(x=F_corr,y=frac_bleedthrough_uncorrected,hue=redcell_corr,ax=ax1,size=5)
# sns.scatterplot(x=F_uncorr,y=frac_bleedthrough_uncorrected,ax=ax1,size=5)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
# sns.scatterplot(x=F_corr[redcell_corr==1],y=frac_bleedthrough_corrected[redcell_corr==1],ax=ax2)
sns.scatterplot(x=F_corr,y=frac_bleedthrough_corrected,hue=redcell_corr,size=5,ax=ax2)
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)







# np.mean(F2_corr[redcell_corr[:,0]==1,:],axis=1)

F_corr_tracemean = np.mean(F_corr,axis=1)
F2_corr_tracemean = np.mean(F2_corr,axis=1)

df_corr = pd.DataFrame({'F_corr': F_corr_tracemean, 'F2_corr' : F2_corr_tracemean, 'redcell_corr' : redcell_corr[:,0]})

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(5,3))
sns.barplot(data=df_corr,y='F_corr',x='redcell_corr',ax=ax1)

sns.barplot(data=df_corr,y='F2_corr',x='redcell_corr',ax=ax2)

F2_diff = np.abs(np.diff(F2,n=1,axis=1))

F2_diff_mean = np.mean(F2_diff,axis=0)

fig = plt.subplots(1,1,figsize=(15,5))
plt.plot(F2_diff_mean,linewidth=0.1)

F2_diff = np.abs(np.diff(F2,n=1,axis=1))

F2_diff_mean = np.mean(F2_diff,axis=0)

fig = plt.subplots(1,1,figsize=(15,5))
plt.plot(F2_diff_mean,linewidth=0.1)



# F2   = [np.vstack(np.load(os.path.join(dir_corrected,'plane%d' %iplane,'F_chan2.npy')) for iplane in range(8))]

###### F2 trace over time: 
# Store as proxy for movement 
iplane = 0
nframes = np.shape(np.load(os.path.join(dir_corrected,'plane%d' %iplane,'F_chan2.npy')))[1]
F2 = np.empty((0,nframes))

for iplane in range(8):
    F           = np.load(os.path.join(dir_corrected,'plane%d' %iplane,'F_chan2.npy'))

    # if imaging was aborted during scanning of a volume, later planes have less frames
    # if so, compensate by duplicating value last frame
    if np.shape(F)[1]==nframes:
        pass       #do nothing, shapes match
    elif np.shape(F)[1]==nframes-1: #copy last timestamp of array
        F           = np.hstack((F, np.tile(F[:,[-1]], 1)))

    F2          = np.vstack((F2,F))

## tdTomato fluorescence change across planes is correlated:
g   = np.abs(zscore(F2,axis=1))
h   = zscore(np.mean(g,axis=0))

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,4))
ax1.imshow(g,aspect='auto',vmin=0,vmax=2)
ax2.plot(h,linewidth=0.2)
ax2.set_xlim([0,np.shape(F2)[1]])

# plt.imshow(F2,aspect='auto',vmin=np.percentile(F2,5),vmax=np.percentile(F2,95))

# plt.figure(figsize=(10,3))
# h  = zscore(np.mean(F2,axis=0))
# plt.plot(h,linewidth=0.2)

# plt.plot(h[:1000],linewidth=0.2)



