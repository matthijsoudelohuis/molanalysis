
import copy
import numpy as np
from tqdm import tqdm
import pandas as pd


def my_shuffle(data,method='random',axis=0):
    data = copy.deepcopy(data)
    if method == 'random':
        if axis == 0:
            for icol in range(data.shape[1]):
                data[:,icol] = np.random.permutation(data[:,icol])
        elif axis == 1:
            for irow in range(data.shape[0]):
                data[irow,:] = np.random.permutation(data[irow,:])
    elif method == 'circular':
        if axis == 0:
            for icol in range(data.shape[1]):
                data[:,icol] = np.roll(data[:,icol],shift=np.random.randint(0,data.shape[0]))
        elif axis == 1:
            for irow in range(data.shape[0]):
                data[irow,:] = np.roll(data[irow,:],shift=np.random.randint(0,data.shape[1])) 
    else:
        raise ValueError('method should be "random" or "circular"')
    return data

def corr_shuffle(sessions,method='random'):
    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing shuffled noise correlations: '):
        if hasattr(sessions[ises],'respmat'):
            data                                = my_shuffle(sessions[ises].respmat,axis=1,method=method)
            sessions[ises].corr_shuffle         = np.corrcoef(data)
            [N,K]                               = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            np.fill_diagonal(sessions[ises].corr_shuffle,np.nan)
    return sessions

def my_shuffleRF(az,el):


    source_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
    target_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
    delta_el        = source_el[:,None] - target_el[None,:]

    source_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
    target_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
    delta_az        = source_az[:,None] - target_az[None,:]

    delta_rf        = np.sqrt(delta_az**2 + delta_el**2)
    angle_rf        = np.mod(np.arctan2(delta_el,delta_az)-np.pi,np.pi*2)
    angle_rf        = np.mod(angle_rf+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
    
    return az,el
