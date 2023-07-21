# -*- coding: utf-8 -*-
"""
This script contains some processing function to align activity to certain timestamps and compute psths
This is both possible for 2D and 3D version, i.e. keep activity over time alinged to event to obtain 3D tensor
Or average across a time window to compute a single response scalar per trial to obtain a 2D response matrix
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import numpy as np
from scipy.stats import binned_statistic
from scipy.interpolate import CubicSpline

def compute_tensor(data,ts_F,ts_T,t_pre=-1,t_post=2,binsize=0.2,method='interpolate'):
    """
    This function constructs a tensor: a 3D 'matrix' of K trials by N neurons by T time bins
    It needs a 2D matrix of activity across neurons, the timestamps of this data (ts_F)
    It further needs the timestamps (ts_T) to align to (the trials) and the parameters for 
    temporal binning to construct a time axis. The function returns the tensor and the time axis. 
    The neuron and trial information is kept outside of the function
    """
    
    assert np.shape(data)[0] > np.shape(data)[1], 'the data matrix appears to have more neurons than timepoints'
    assert np.shape(data)[0] == np.shape(ts_F)[0], 'the amount of datapoints does not seem to match the timestamps'
    
    binedges    = np.arange(t_pre-binsize/2,t_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(t_pre,t_post+binsize,binsize)
    
    N           = np.shape(data)[1]
    K           = len(ts_T)
    T           = len(bincenters)
    
        # N = 10 #for debug only

    tensor      = np.empty([K,N,T])
    
    for n in range(N):
        print(f"\rComputing tensor for neuron {n+1} / {N}",end='\r')
        for k in range(K):
            if method=='binmean':
                tensor[k,n,:]       = binned_statistic(ts_F-ts_T[k],data.iloc[:,n], statistic='mean', bins=binedges)[0]
            
            elif method == 'interp_lin':
                tensor[k,n,:]       = np.interp(bincenters, ts_F-ts_T[k], data.iloc[:,n])
                
            elif method == 'interp_cub':
                spl = CubicSpline(ts_F-ts_T[k], data.iloc[:,n])
                spl(bincenters)
                tensor[k,n,:]       = spl(bincenters)

            else:
                print('method to bin is unknown')
                tensor = None
                bincenters = None
                return tensor,bincenters
            
    return tensor,bincenters


def compute_tensor_space(data,ts_F,ts_T,zpos_F,trialnum_F,s_pre=-100,s_post=100,binsize=5,method='interpolate'):
    """
    This function constructs a tensor: a 3D 'matrix' of K trials by N neurons by S spatial bins
    It needs a 2D matrix of activity across neurons, the timestamps of this data (ts_F)
    It further needs the z-position of the animal in the linear VR track (zpos_F) in centimeters at calcium frame times
    and the spatial position to to align to (ts_T, e.g. stimulus start location per trial)
    IT further needs the parameters for temporal binning to construct a time axis. 
    The function returns the tensor and the spatial bin axis. 
    The neuron and trial information is kept outside of the function
    """
    
    assert np.shape(data)[0] > np.shape(data)[1], 'the data matrix appears to have more neurons than timepoints'
    assert np.shape(data)[0] == np.shape(ts_F)[0], 'the amount of datapoints does not seem to match the timestamps'
    
    binedges    = np.arange(s_pre-binsize/2,s_post+binsize+binsize/2,binsize)
    bincenters  = np.arange(s_pre,s_post+binsize,binsize)
    
    N           = np.shape(data)[1]
    K           = len(ts_T)
    S           = len(bincenters)
    
    # N = 10 #for debug only

    tensor      = np.empty([K,N,S])
    
    for n in range(N):
        print(f"\rComputing tensor for neuron {n+1} / {N}",end='\r')
        for k in range(K):
            if method=='binmean':
                # tensor[k,n,:]       = binned_statistic(ts_F-ts_T[k],data.iloc[:,n], statistic='mean', bins=binedges)[0]

                idx = trialnum_F==k+1
                tensor[k,n,:] = binned_statistic(zpos_F[idx]-ts_T[k],data[idx,n], statistic='mean', bins=binedges)[0]

            
            elif method == 'interp_lin':
                
                tensor[k,n,:]       = np.interp(bincenters, ts_F-ts_T[k], data.iloc[:,n])
                
            elif method == 'interp_cub':
                spl = CubicSpline(ts_F-ts_T[k], data.iloc[:,n])
                spl(bincenters)
                tensor[k,n,:]       = spl(bincenters)

            else:
                print('method to bin is unknown')
                tensor = None
                bincenters = None
                return tensor,bincenters
            
    return tensor,bincenters

def compute_respmat(data,ts_F,ts_T,t_resp_start=0,t_resp_stop=1,
                    t_base_start=-1,t_base_stop=0,subtr_baseline=False,method='mean'):
    """
    This function constructs a 2D matrix of K trials by N neurons
    It needs a 2D matrix of activity across neurons, the timestamps of this data (ts_F)
    It further needs the timestamps (ts_T) to align to (the trials) and the response window
    Different ways of measuring the response can be specified such as 'mean','max'
    The neuron and trial information is kept outside of the function
    """
    
    assert np.shape(data)[0] > np.shape(data)[1], 'the data matrix appears to have more neurons than timepoints'
    assert np.shape(data)[0] == np.shape(ts_F)[0], 'the amount of datapoints does not seem to match the timestamps'
    
    N               = np.shape(data)[1] #get number of neurons from shape of datamatrix (number of columns)
    K               = len(ts_T)         #get number of trials from the number of timestamps given 
    
    respmat         = np.empty([K,N])   #init output matrix

    for k in range(K): #loop across trials, for every trial, slice through activity matrix and compute response across neurons:
        print(f"\rComputing response for trial {k+1} / {K}",end='\r')
        respmat[k,:]      = data[np.logical_and(ts_F>ts_T[k]+t_resp_start,ts_F<ts_T[k]+t_resp_stop)].to_numpy().mean(axis=0)

        if subtr_baseline: #subtract baseline activity if requested:
            base                = data[np.logical_and(ts_F>ts_T[k]+t_base_start,ts_F<ts_T[k]+t_base_stop)].to_numpy().mean(axis=0)
            respmat[k,:]        = np.subtract(respmat[k,:],base)
    
    return respmat

def compute_respmat_space(data,ts_F,ts_T,t_resp_start=0,t_resp_stop=1,
                    t_base_start=-1,t_base_stop=0,subtr_baseline=False,method='mean'):
    """
    This function constructs a 2D matrix of K trials by N neurons
    It needs a 2D matrix of activity across neurons, the timestamps of this data (ts_F)
    It further needs the timestamps (ts_T) to align to (the trials) and the response window
    Different ways of measuring the response can be specified such as 'mean','max'
    The neuron and trial information is kept outside of the function
    """
    
    assert np.shape(data)[0] > np.shape(data)[1], 'the data matrix appears to have more neurons than timepoints'
    assert np.shape(data)[0] == np.shape(ts_F)[0], 'the amount of datapoints does not seem to match the timestamps'
    
    N               = np.shape(data)[1] #get number of neurons from shape of datamatrix (number of columns)
    K               = len(ts_T)         #get number of trials from the number of timestamps given 
    
    respmat         = np.empty([K,N])   #init output matrix

    for k in range(K): #loop across trials, for every trial, slice through activity matrix and compute response across neurons:
        print(f"\rComputing response for trial {k+1} / {K}",end='\r')
        respmat[k,:]      = data[np.logical_and(ts_F>ts_T[k]+t_resp_start,ts_F<ts_T[k]+t_resp_stop)].to_numpy().mean(axis=0)

        if subtr_baseline: #subtract baseline activity if requested:
            base                = data[np.logical_and(ts_F>ts_T[k]+t_base_start,ts_F<ts_T[k]+t_base_stop)].to_numpy().mean(axis=0)
            respmat[k,:]        = np.subtract(respmat[k,:],base)
    
    return respmat
