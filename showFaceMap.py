import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import zscore

facemapfile = "W:\\Users\\Matthijs\\Rawdata\\LPE09829\\2023_03_29\\VR\\Behavior\\VR_LPE09829_camera_2023-03-29T15_32_29_proc.npy"

facemapfile = 'E:\\OneDrive\\PostDoc\\Analysis\\PupilMotSVD\\testvid\\RF_NSH07422_camera_2023-03-15T10_11_29_proc.npy'

proc = np.load(facemapfile,allow_pickle=True).item()

fig = plt.subplots(1,7,figsize=(14, 2)) 
ax = plt.subplot(1, 7, 1)
plt.imshow(proc['avgframe_reshape'],cmap='gray',aspect='auto')
plt.title("average frame",fontsize=12)
plt.axis("off")

motSVDidx = [proc['rois'][i]['rtype'] for i in range(2)].index('motion SVD')
ax.add_patch(Rectangle((proc['rois'][motSVDidx]['xrange'][0]/proc['sbin'], proc['rois'][motSVDidx]['yrange'][0]/proc['sbin']),
                       np.ptp(proc['rois'][motSVDidx]['xrange'])/proc['sbin'], np.ptp(proc['rois'][motSVDidx]['yrange']/proc['sbin']), 
             edgecolor = 'pink',
             linewidth = 1,
             linestyle=':',
             fill=False))

ax = plt.subplot(1, 7, 2)
plt.imshow(proc['avgmotion_reshape'], vmin=0, vmax=10,cmap='Reds',aspect='auto')
plt.title('average motion',fontsize=12)
plt.axis("off")

for i in range(5):
    plt.subplot(1, 7, i + 3)
    plt.imshow(proc['motMask_reshape'][1][:, :, i] / proc['motMask_reshape'][1][:, :, i].std(), vmin=-2, vmax=2,cmap='bwr',aspect='auto')
    plt.axis("off")
    plt.title('video PC %d' % i,fontsize=12)
plt.tight_layout()

## if you want to show something for the pupil:
# pupilidx = [proc['rois'][i]['rtype'] for i in range(2)].index('Pupil')
data    = proc['pupil'][0]

selecwindow = [3000,4500]

fig = plt.subplots(2,1,figsize=(8, 2))
plt.subplot(211)
plt.plot(data['area_smooth'][:])
plt.xlim(selecwindow)
plt.ylabel('Area \n (px2)')
plt.xticks([], [])
plt.ylim([100,400])

plt.subplot(212)
plt.plot(zscore(data['com'][:,0]),'r')
plt.plot(zscore(data['com'][:,1]),'b')
plt.xlim(selecwindow)
plt.xlabel('Frame number')
plt.ylabel('Pupil Position \n (z-scored)')
plt.legend(['Ypos', 'Xpos'],loc='upper left')
plt.ylim([-3,5])
