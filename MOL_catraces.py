
import os
import matplotlib.pyplot as plt
import numpy as np
from suite2p.extraction import dcnv

direc = 'V:\\Rawdata\\PILOTS\\20221108_NSH07429_Spontaneous4ROI\\Spontaneous\\suite2p\\plane0'


os.chdir(direc)

F = np.load('F.npy')
Fneu = np.load('Fneu.npy')
ops = np.load('ops.npy', allow_pickle=True).item()
dF = F.copy() - 0.7*Fneu
dF = dcnv.preprocess(dF, ops['baseline'], ops['win_baseline'], 
                                   ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])


iscell = np.load('iscell.npy')

ncells = sum(iscell[:,0])

dFselec = dF[iscell[:,0]==1,:]

plt.figure()
plt.plot(dFselec)


spks = np.load('spks.npy')
spksselec = spks[iscell[:,0]==1,:]

####
# Calculate the range of values in each row
row_range = np.ptp(spksselec, axis=1)
# Subtract the minimum value from each element in the row
spksselec = spksselec - np.min(spksselec, axis=1)[:, np.newaxis]
# Divide by the range of values in each row
spksselec = spksselec / row_range[:, np.newaxis]

spksselec = spksselec +
nframes = np.shape(spksselec)[1]

temp = np.repeat(range(ncells.astype('Int32')), nframes, axis=1)

# Print the normalized array
print(arr)

plt.figure()
plt.plot(spksselec)

####################
stat = np.load('stat.npy', allow_pickle=True)
ops = np.load('ops.npy', allow_pickle=True).item()

im = np.zeros((ops['Ly'], ops['Lx']))

  
for n, j in enumerate(iscell[:,0]):
    if j:
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        im[ypix,xpix] = n+1

plt.imshow(im)
plt.show()

