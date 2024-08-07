
import numpy as np
import matplotlib.pyplot as plt
import os 
from ScanImageTiffReader import ScanImageTiffReader as imread

def extrema_np(arr):
    return np.min(arr),np.max(arr)

def bleedthrough_correction(data_green,data_red,coeff=None,gain1=0.6,gain2=0.4):
    # Regression with pre-established values:
    if coeff is None:
        coeff = get_gain_coeff(gain1,gain2)

    offset              = np.percentile(data_red.flatten(),5)
    data_green_corr    = data_green - coeff * (data_red-offset)

    return data_green_corr

## 
def plot_correction_images(greenchanim,redchanim):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(10,6.5))

    ax1.imshow(greenchanim,vmin=np.percentile(greenchanim,5), vmax=np.percentile(greenchanim,99)*1.3)
    ax1.set_title('Chan 1')
    ax1.set_axis_off()

    ax2.imshow(redchanim,vmin=np.percentile(redchanim,5), vmax=np.percentile(redchanim,99)*1.3)
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
    ax3.plot(xseq, a + b * xseq, color="r", lw=1.5)   # Plot regression line

    txt1 = "Fit coefficient is %1.4f" % b
    ax3.text(np.percentile(redchanim,40),np.percentile(greenchanim,5),txt1, fontsize=9)

    # regression through pre-established values:
    b = 1.54
    a = np.percentile(redchanim.flatten(),5)
    xseq = np.linspace(-15000, 32000, num=32000)
    ax3.plot(xseq, a + b * xseq, color="k", lw=1.5)   # Plot regression line

    ax3.set_xlim(extrema_np(redchanim))
    ax3.set_ylim(extrema_np(greenchanim))

    #Correction:
    greenchanim_corr = greenchanim - b * (redchanim-a)

    ax4.imshow(greenchanim_corr,vmin=np.percentile(greenchanim,5), vmax=np.percentile(greenchanim,99)*1.3)
    ax4.set_title('Chan 1')
    ax4.set_axis_off()

    ax5.imshow(redchanim,vmin=np.percentile(redchanim,5), vmax=np.percentile(redchanim,99)*1.3)
    ax5.set_title('Chan 2')
    ax5.set_axis_off()

    greenchan = greenchanim_corr.reshape(1,512*512)[0]
    redchan = redchanim.reshape(1,512*512)[0]

    ax6.scatter(redchan,greenchan,0.02)
    ax6.set_xlabel('Chan 2')
    ax6.set_ylabel('Chan 1')

    ax6.set_xlim(extrema_np(redchanim))
    ax6.set_ylim(extrema_np(greenchanim))

    return

# regression through pre-established values:
    # b = 1.54
    # a = np.percentile(redchanim.flatten(),5)
    # xseq = np.linspace(-15000, 32000, num=32000)
    # ax3.plot(xseq, a + b * xseq, color="k", lw=1.5)   # Plot regression line

    # ax3.set_xlim(extrema_np(redchanim))
    # ax3.set_ylim(extrema_np(greenchanim))

# ###### correction coefficient for red into green:
# coeff = 1.54 #for 0.6 and 0.4 combination of PMT gains
# coeff = 0.32 #for 0.6 and 0.5 combination of PMT gains
# coeff = 0.068 #for 0.6 and 0.6 combination of PMT gains
    
def show_gain_coeff():
    diff = np.array([-0.2,-0.1,0,0.1,0.2])
    corr = np.array([0.003,0.015,0.0668,0.32,1.54])

    b, a = np.polyfit(diff[2:], np.log10(corr[2:]), deg=1)

    corr_pred = 10**(b*diff+a)

    fig = plt.figure()
    plt.plot(diff,corr)
    plt.scatter(diff,corr,s=20,color='r')
    plt.yscale('log')
    plt.scatter(diff,corr_pred,s=20,color='b')
    return

def get_gain_coeff(gain1,gain2):
    diff = gain1-gain2
    b = 6.813721291804585
    a = -1.1755564086364871
    return 10**(b*diff+a)

def estimate_correc_coeff(greendata,reddata):

    # Fit linear regression via least squares with numpy.polyfit
    coeff, offset = np.polyfit(reddata.flatten(),greendata.flatten(), deg=1)

    # offset              = np.percentile(reddata.flatten(),5)
    # greendata_corr    = greendata - coeff * (reddata-offset)

    return coeff

#####################################################################################
def load_tiffs(directory,nplanes): 
    data_green      = np.empty([0,nplanes,512,512])
    data_red        = np.empty([0,nplanes,512,512])
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
            data_green  = np.append(data_green, np.expand_dims(Data[0:nplanes*2:2,:,:],axis=0),axis=0)
            data_red    = np.append(data_red, np.expand_dims(Data[1:nplanes*2:2,:,:],axis=0),axis=0)
    
    return data_green,data_red

#####################################################################################
def load_tiffs_plane(directory,nplanes,iplane=0): 
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
            data_green  = np.append(data_green, Data[iplane*2::(2*nplanes),:,:],axis=0)
            data_red    = np.append(data_red, Data[iplane*2+1::(2*nplanes),:,:],axis=0)
    
    return data_green,data_red

## plotting func:
def plot_bleedthrough_correction(im1,im2,im1_corr):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17,5))
    
    ax1.imshow(im1,vmin=np.percentile(im1,1), vmax=np.percentile(im1,99))
    ax1.set_title('Chan 1')
    ax1.set_axis_off()
    
    ax2.imshow(im2,vmin=np.percentile(im2,1), vmax=np.percentile(im2,99))
    ax2.set_title('Chan 2')
    ax2.set_axis_off()
        
    ax3.imshow(im1_corr,vmin=np.percentile(im1,1), vmax=np.percentile(im1,99))
    ax3.set_title('Chan 1 Corr')
    ax3.set_axis_off()

## correlation func:
def plot_corr_redgreen(im1,im2):
    ## Show linear correction:
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
        
    ax1.scatter(im2.flatten(),im1.flatten(),0.02)
    ax1.set_xlabel('Chan 2')
    ax1.set_ylabel('Chan 1')
    
    # Fit linear regression via least squares with numpy.polyfit
    b, a = np.polyfit(im2.flatten(),im1.flatten(), deg=1)
    
    xseq = np.linspace(-15000, 32000, num=32000)
    # Plot regression line
    ax1.plot(xseq, a + b * xseq, color="k", lw=1.5)
    
    ax1.set_xlim(np.percentile(im2.flatten()*1.05,(0,100)))
    ax1.set_ylim(np.percentile(im1.flatten()*1.05,(0,100)))
    
    txt1 = "Coefficient is %1.4f" % b
    
    plt.text(-500, 1200,txt1, fontsize=12)

