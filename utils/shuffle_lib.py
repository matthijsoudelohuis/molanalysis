
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
