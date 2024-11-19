

#%% Import libs
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from natsort import natsorted 

os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import *

from labeling.tdTom_labeling_cellpose import proc_labeling_plane
from loaddata.get_data_folder import get_rawdata_drive
from preprocessing.preprocesslib import align_timestamps
from scipy.signal import butter, filtfilt

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Labeling\\CalciumTracesComparison\\'

#%% Session info:
animal_id          = 'LPE09830' #If empty than all animals in folder will be processed
sessiondate        = '2023_04_12'

animal_id          = 'LPE12013' #If empty than all animals in folder will be processed
sessiondate        = '2024_05_02'

animal_id          = 'LPE10919' #If empty than all animals in folder will be processed
sessiondate        = '2023_11_06'

animal_id          = 'LPE11622' #If empty than all animals in folder will be processed
sessiondate        = '2024_03_28'


# Sessions with the biggest difference in absolute pairwise corrleations (dF/F) between labeled and unlabeled cells
# LPE11622_2024_03_28
# LPE11495_2024_02_28
# LPE09665_2023_03_21


# animal_ids          = ['LPE09665', 'LPE11495', 'LPE11998', 'LPE12013'] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE10885'] #If empty than all animals in folder will be processed

protocols           = 'RF'

rawdatadir          = get_rawdata_drive([animal_id],protocols=protocols)
sesfolder           = os.path.join(rawdatadir,animal_id,sessiondate)
suite2p_folder      = os.path.join(sesfolder,"suite2p")

#%% 
signal = 'F'
# signal = 'spks'
# signal = 'F_chan2'

if os.path.exists(suite2p_folder): 
    plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
    fig,axes = plt.subplots(4,2,figsize=(9,9),sharex=True)

    for iplane,plane_folder in enumerate(plane_folders):
        iscell       = np.load(os.path.join(plane_folder, 'iscell.npy'))
        data         = np.load(os.path.join(plane_folder, '%s.npy' % signal), allow_pickle=True)

        if os.path.exists(os.path.join(plane_folder,'redim_plane%d_seg.npy' %iplane)):
            redcell_seg         = np.load(os.path.join(plane_folder,'redim_plane%d_seg.npy' %iplane), allow_pickle=True).item()
            masks_cp_red        = redcell_seg['masks']
            Nredcells_plane     = len(np.unique(masks_cp_red))-1 # number of labeled cells overall, minus 1 because 0 for all nonlabeled pixels
            redcell             = proc_labeling_plane(iplane,plane_folder,showcells=False,overlap_threshold=0.5)
        
        idx_unl     = np.logical_and(redcell[:,0]==0,iscell[:,0]==1)
        idx_lab     = np.logical_and(redcell[:,0]==1,iscell[:,0]==1)
        temp        = np.vstack((data[idx_unl,:],data[idx_lab,:]))

        # ax          = axes[iplane//2,iplane%2]
        ax          = axes[iplane%4,iplane//4]
        ax.imshow(temp,aspect='auto',vmin=np.percentile(data,10),vmax=np.percentile(data,90))
        ax.axhline(np.sum(idx_unl),color='r',linewidth=2)
        ax.set_xticks([])
    plt.tight_layout()
fig.savefig(os.path.join(savedir,'%s_allplanes_%s_%s.png' % (signal,animal_id,sessiondate)))
    


#%% 
# signal = 'F'
signal = 'spks'
signal = 'Fneu'
# signal = 'F_chan2'

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

if os.path.exists(suite2p_folder): 
    plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
    fig,axes = plt.subplots(4,2,figsize=(9,9),sharex=True)

    for iplane,plane_folder in enumerate(plane_folders):
        iscell       = np.load(os.path.join(plane_folder, 'iscell.npy'))
        data         = np.load(os.path.join(plane_folder, '%s.npy' % signal), allow_pickle=True)

        if os.path.exists(os.path.join(plane_folder,'redim_plane%d_seg.npy' %iplane)):
            redcell_seg         = np.load(os.path.join(plane_folder,'redim_plane%d_seg.npy' %iplane), allow_pickle=True).item()
            masks_cp_red        = redcell_seg['masks']
            Nredcells_plane     = len(np.unique(masks_cp_red))-1 # number of labeled cells overall, minus 1 because 0 for all nonlabeled pixels
            redcell             = proc_labeling_plane(iplane,plane_folder,showcells=False,overlap_threshold=0.5)
        
        Fs          = 5.317

        b, a        = butter_lowpass(0.1, Fs)

        temp_unl     = data[np.logical_and(redcell[:,0]==0,iscell[:,0]==1),:].mean(axis=0)
        temp_lab     = data[np.logical_and(redcell[:,0]==1,iscell[:,0]==1),:].mean(axis=0)

        temp_unl     = filtfilt(b, a, temp_unl)
        temp_lab     = filtfilt(b, a, temp_lab)

        ax          = axes[iplane%4,iplane//4]
        ax.plot(temp_unl,color='k',linewidth=0.25)
        ax.plot(temp_lab,color='r',linewidth=0.25)
        ax.set_xticks([])
    plt.tight_layout()
fig.savefig(os.path.join(savedir,'%s_mean_allplanes_%s_%s.png' % (signal,animal_id,sessiondate)))
    



#%%


#%% 
signal = 'F_chan2'

if os.path.exists(suite2p_folder): 
    plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
    fig,axes = plt.subplots(4,2,figsize=(9,9),sharex=True)

    for iplane,plane_folder in enumerate(plane_folders):
        iscell       = np.load(os.path.join(plane_folder, 'iscell.npy'))
        data         = np.load(os.path.join(plane_folder, '%s.npy' % signal), allow_pickle=True)
        print(np.max(data))
        # print('%d cells with Fchan2 exceeding max')

