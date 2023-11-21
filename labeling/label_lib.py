
import numpy as np
import matplotlib.pyplot as plt
import os 

def bleedthrough_correction(greenchanim,redchanim,gain1=0.6,gain2=0.4):
    
    # Regression with pre-established values:
    b                   = 1.54
    a                   = np.percentile(redchanim.flatten(),5)
    greenchanim_corr    = greenchanim - b * (redchanim-a)

    return greenchanim_corr


def extrema_np(arr):
    return np.min(arr),np.max(arr)

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
    b = 1.54
    a = np.percentile(redchanim.flatten(),5)
    xseq = np.linspace(-15000, 32000, num=32000)
    ax3.plot(xseq, a + b * xseq, color="k", lw=1.5)   # Plot regression line

    ax3.set_xlim(extrema_np(redchanim))
    ax3.set_ylim(extrema_np(greenchanim))

###### correction coefficient for red into green:
coeff = 1.54 #for 0.6 and 0.4 combination of PMT gains
# coeff = 0.32 #for 0.6 and 0.5 combination of PMT gains
# coeff = 0.068 #for 0.6 and 0.6 combination of PMT gains

diff = np.array([-0.2,-0.1,0,0.1,0.2])
corr = np.array([0.02,0.05,0.0668,0.32,1.54])

b, a = np.polyfit(diff[2:], np.log10(corr[2:]), deg=1)

corr_pred = 10**(b*diff+a)

fig = plt.figure()
plt.plot(diff,corr)
plt.scatter(diff,corr,s=20,color='r')
plt.yscale('log')
plt.scatter(diff,corr_pred,s=20,color='b')