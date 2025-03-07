
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
        elif axis is None:
            rng = np.random.default_rng()
            orig_size = data.shape
            data = np.random.permutation(data.ravel()).reshape(orig_size)

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

def my_shuffle_celldata_joint(x,y,area):
    x = np.array(x)
    y = np.array(y)
    for uarea in np.unique(area):
        idx_area = area == uarea
        idx_shuffle = np.random.permutation(sum(idx_area))
        x[idx_area], y[idx_area] = x[idx_area][idx_shuffle], y[idx_area][idx_shuffle]
    return x,y

def my_shuffle_celldata(celldata,shufflefield,keep_roi_name=True):
    if keep_roi_name:
       celldata[shufflefield] = celldata.groupby('roi_name')[shufflefield].transform(lambda x: x.sample(frac=1).values)
    else:
       celldata[shufflefield] = celldata[shufflefield].sample(frac=1).values
    return celldata


