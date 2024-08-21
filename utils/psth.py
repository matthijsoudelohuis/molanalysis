# -*- coding: utf-8 -*-
"""
This script contains some processing function to align activity to certain timestamps and compute psths
This is both possible for 2D and 3D version, i.e. keep activity over time alinged to event to obtain 3D tensor
Or average across a time window to compute a single response scalar per trial to obtain a 2D response matrix
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from scipy.interpolate import CubicSpline

"""
 ####### ####### #     #  #####  ####### ######  
    #    #       ##    # #     # #     # #     # 
    #    #       # #   # #       #     # #     # 
    #    #####   #  #  #  #####  #     # ######  
    #    #       #   # #       # #     # #   #   
    #    #       #    ## #     # #     # #    #  
    #    ####### #     #  #####  ####### #     # 
"""


def compute_tensor(data, ts_F, ts_T, t_pre=-1, t_post=2, binsize=0.2, method='interpolate'):
    """
    This function constructs a tensor: a 3D 'matrix' of N neurons by K trials by T time bins
    It needs a 2D matrix of activity across neurons, the timestamps of this data (ts_F)
    It further needs the timestamps (ts_T) to align to (the trials) and the parameters for 
    temporal binning to construct a time axis. The function returns the tensor and the time axis. 
    The neuron and trial information is kept outside of the function
    """

    assert np.shape(data)[0] > np.shape(
        data)[1], 'the data matrix appears to have more neurons than timepoints'
    assert np.shape(data)[0] == np.shape(
        ts_F)[0], 'the amount of datapoints does not seem to match the timestamps'

    if method == 'nearby':
        print('Binsize parameter ignored - set by imaging frame rate in nearby method\n')

        binsize = np.mean(np.diff(ts_F))

        binedges = np.arange(t_pre-binsize/2, t_post +
                             binsize+binsize/2, binsize)
        bincenters = np.arange(t_pre, t_post+binsize, binsize)

        N = np.shape(data)[1]
        K = len(ts_T)
        T = len(bincenters)

        tensor = np.empty([N, K, T])

        for k in range(K):
            print(f"\rComputing tensor for trial {k+1} / {K}", end='\r')
            firstframe = np.where(ts_F > ts_T[k] + t_pre - binsize/2)[0][0]
            tensor[:, k, :] = data.iloc[firstframe:firstframe+T, :].T
    else:
        binedges = np.arange(t_pre-binsize/2, t_post +
                             binsize+binsize/2, binsize)
        bincenters = np.arange(t_pre, t_post+binsize, binsize)

        N = np.shape(data)[1]
        K = len(ts_T)
        T = len(bincenters)

        # N = 10 #for debug only

        tensor = np.empty([N, K, T])

        for n in range(N):
            print(f"\rComputing tensor for neuron {n+1} / {N}", end='\r')
            for k in range(K):
                if method == 'binmean':
                    tensor[n, k, :] = binned_statistic(
                        ts_F-ts_T[k], data.iloc[:, n], statistic='mean', bins=binedges)[0]

                elif method == 'interp_lin':
                    tensor[n, k, :] = np.interp(
                        bincenters, ts_F-ts_T[k], data.iloc[:, n])

                elif method == 'interp_cub':
                    spl = CubicSpline(ts_F-ts_T[k], data.iloc[:, n])
                    spl(bincenters)
                    tensor[n, k, :] = spl(bincenters)

                else:
                    print('method to bin is unknown')
                    tensor = None
                    bincenters = None
                    return tensor, bincenters

    return tensor, bincenters


# data = sessions[i].calciumdata
# ts_F = sessions[i].ts_F
# z_T = sessions[i].trialdata['stimStart']
# zpos_F = sessions[i].zpos_F
# trialnum_F = sessions[i].trialnum_F
# method='binmean'


def compute_tensor_space(data, ts_F, z_T, zpos_F, trialnum_F, s_pre=-100, s_post=100, binsize=5, method='interpolate'):
    """
    This function constructs a tensor: a 3D 'matrix' of N neurons by K trials by S spatial bins
    It needs a 2D matrix of activity across neurons, the timestamps of this data (ts_F)
    It further needs the z-position of the animal in the linear VR track (zpos_F) in centimeters at calcium frame times
    and the spatial position to to align to (ts_T, e.g. stimulus start location per trial)
    IT further needs the parameters for temporal binning to construct a time axis. 
    The function returns the tensor and the spatial bin axis. 
    The neuron and trial information is kept outside of the function
    """

    assert np.shape(data)[0] > np.shape(
        data)[1], 'the data matrix appears to have more neurons than timepoints'
    assert np.shape(data)[0] == np.shape(
        ts_F)[0], 'the amount of datapoints does not seem to match the timestamps'

    # if isinstance(data, pd.DataFrame): data = data.to_numpy()

    binedges = np.arange(s_pre-binsize/2, s_post+binsize+binsize/2, binsize)
    bincenters = np.arange(s_pre, s_post+binsize, binsize)

    N = np.shape(data)[1]
    K = len(z_T)
    S = len(bincenters)

    # N = 10 #for debug only

    tensor = np.empty([N, K, S])

    for k in range(K):
        print(f"\rComputing tensor for trial  {k+1} / {K}", end='\r')
        # idx_trial = trialnum_F==k+1
        for s, (bin_start, bin_end) in enumerate(zip(binedges[:-1], binedges[1:])):
            # idx_bin = bin_start <= zpos_F[idx]-z_T[k] < bin_end
            # idx = np.all((idx_trial,zpos_F-z_T[k] >= bin_start,zpos_F-z_T[k] < bin_end),axis=0)
            idx = np.all((zpos_F-z_T[k] >= bin_start,
                         zpos_F-z_T[k] < bin_end), axis=0)
            tensor[:, k, s] = np.nanmean(data.iloc[idx, :], axis=0)

    if method != 'binmean':
        print('method to bin is unknown')

    # for n in range(N):
    #     print(f"\rComputing tensor for neuron {n+1} / {N}",end='\r')
    #     for k in range(K):
    #         if method=='binmean':
    #             # tensor[k,n,:]       = binned_statistic(ts_F-ts_T[k],data.iloc[:,n], statistic='mean', bins=binedges)[0]

    #             idx = trialnum_F==k+1
    #             tensor[n,k,:] = binned_statistic(zpos_F[idx]-z_T[k],data.iloc[idx,n], statistic='mean', bins=binedges)[0]
    #             # tensor[n,k,:] = binned_statistic(zpos_F[idx]-ts_T[k],data[idx,n], statistic='mean', bins=binedges)[0]

    #         elif method == 'interp_lin':
    #             idx = trialnum_F==k+1
    #             tensor[n,k,:]       = np.interp(bincenters, zpos_F[idx]-z_T[k], data.iloc[idx,n],left=np.nan,right=np.nan)

    #         elif method == 'interp_cub':
    #             print('method to bin is not recommended for spatial tensor due to small changes in space with large calcium fluctuations')

    #             idx = trialnum_F==k+1
    #             x = zpos_F[idx]-z_T[k]
    #             y = data.iloc[idx,n]
    #             x_idx = np.argsort(x)
    #             x  = x[x_idx]
    #             y  = y[x_idx]
    #             spl = CubicSpline(x,y,extrapolate=False)
    #             tensor[n,k,:]       = spl(bincenters)

    #         else:
    #             print('method to bin is unknown')
    #             tensor = None
    #             bincenters = None
    #             return tensor,bincenters

    return tensor, bincenters


"""
 ######  #######  #####  ######     #     #    #    ####### ######  ### #     # 
 #     # #       #     # #     #    ##   ##   # #      #    #     #  #   #   #  
 #     # #       #       #     #    # # # #  #   #     #    #     #  #    # #   
 ######  #####    #####  ######     #  #  # #     #    #    ######   #     #    
 #   #   #             # #          #     # #######    #    #   #    #    # #   
 #    #  #       #     # #          #     # #     #    #    #    #   #   #   #  
 #     # #######  #####  #          #     # #     #    #    #     # ### #     # 
                                                  
"""


def compute_respmat(data, ts_F, ts_T, t_resp_start=0, t_resp_stop=1,
                    t_base_start=-1, t_base_stop=0, subtr_baseline=False, method='mean'):
    """
    This function constructs a 2D matrix of N neurons by K trials
    It needs a 2D matrix of activity across neurons, the timestamps of this data (ts_F)
    It further needs the timestamps (ts_T) to align to (the trials) and the response window
    Different ways of measuring the response can be specified such as 'mean','max'
    The neuron and trial information is kept outside of the function
    """
    data = np.array(data)
    ts_F = np.array(ts_F)
    ts_T = np.array(ts_T)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    assert np.shape(data)[0] > np.shape(
        data)[1], 'the data matrix appears to have more neurons than timepoints'
    assert np.shape(data)[0] == np.shape(
        ts_F)[0], 'the amount of datapoints does not seem to match the timestamps'

    # get number of neurons from shape of datamatrix (number of columns)
    N = np.shape(data)[1]
    K = len(ts_T)  # get number of trials from the number of timestamps given

    respmat = np.empty([N, K])  # init output matrix
    # print(f"\n")
    # loop across trials, for every trial, slice through activity matrix and compute response across neurons:
    for k in range(K):
        print(f"\rComputing average response for trial {k+1} / {K}", end='\r')
        respmat[:, k] = data[np.logical_and(
            ts_F > ts_T[k]+t_resp_start, ts_F < ts_T[k]+t_resp_stop)].mean(axis=0)

        if subtr_baseline:  # subtract baseline activity if requested:
            base = data[np.logical_and(
                ts_F > ts_T[k]+t_base_start, ts_F < ts_T[k]+t_base_stop)].mean(axis=0)
            respmat[:, k] = np.subtract(respmat[:, k], base)

    return np.squeeze(respmat)


def compute_respmat_space(data, ts_F, z_T, zpos_F, trialnum_F, s_resp_start=0, s_resp_stop=20,
                          s_base_start=-80, s_base_stop=-60, subtr_baseline=False, method='mean'):
    """
    This function constructs a 2D matrix of N neurons by K trials
    It needs a 2D matrix of activity across neurons, the timestamps of this data (ts_F)
    It further needs the spatial position (z_T) to align to (e.g. stimulus position in trials)
    and the response window start and stop positions.
    Different ways of measuring the response can be specified such as 'mean','max'
    The neuron and trial information is kept outside of the function
    """
    data = np.array(data)
    ts_F = np.array(ts_F)
    z_T = np.array(z_T)
    zpos_F = np.array(zpos_F)
    trialnum_F = np.array(trialnum_F)

    assert np.shape(data)[0] > np.shape(
        data)[1], 'the data matrix appears to have more neurons than timepoints'
    assert np.shape(data)[0] == np.shape(
        ts_F)[0], 'the amount of datapoints does not seem to match the timestamps'

    # get number of neurons from shape of datamatrix (number of columns)
    N = np.shape(data)[1]
    K = len(z_T)  # get number of trials from the number of timestamps given

    respmat = np.empty([N, K])  # init output matrix

    # loop across trials, for every trial, slice through activity matrix and compute response across neurons:
    for k in range(K):
        print(f"\rComputing response for trial {k+1} / {K}", end='\r')
        idx_K = trialnum_F == k+1
        idx_S = np.logical_and(
            zpos_F-z_T[k] > s_resp_start, zpos_F-z_T[k] < s_resp_stop)

        respmat[:, k] = data[np.logical_and(idx_K, idx_S), :].mean(axis=0)

        if subtr_baseline:  # subtract baseline activity if requested:
            idx_S_base = np.logical_and(
                zpos_F-z_T[k] > s_base_start, zpos_F-z_T[k] < s_base_stop)
            base = data[np.logical_and(idx_K, idx_S_base), :].mean(axis=0)
            respmat[:, k] = np.subtract(respmat[:, k], base)

    return respmat

# data = temp
# ts_F = sessions[i].behaviordata['ts']
# z_T = sessions[i].trialdata['stimStart']
# zpos_F = sessions[i].behaviordata['zpos']
# trialnum_F = sessions[i].behaviordata['trialNumber']

# s_resp_start=0
# s_resp_stop=20
# method='mean'
# subtr_baseline=False

# data = sessions[i].calciumdata
# ts_F = sessions[i].ts_F
# z_T = sessions[i].trialdata['stimStart']
# zpos_F = sessions[i].zpos_F
# trialnum_F = sessions[i].trialnum_F

# s_resp_start=0
# s_resp_stop=20
# method='mean'
# subtr_baseline=False


def construct_behav_matrix_ts_F(ses, nvideoPCs=30):
    Slabels = []
    S = np.empty((len(ses.ts_F), 0))
    S = np.hstack((S, np.expand_dims(np.interp(ses.ts_F.to_numpy(
    ), ses.behaviordata['ts'].to_numpy(), ses.behaviordata['runspeed'].to_numpy()), axis=1)))
    Slabels.append('runspeed')

    # fields = ['pupil_area','motionenergy']
    fields = ['pupil_area']
    [fields.append('videoPC_' + '%s' % k) for k in range(0, nvideoPCs)]

    for field in fields:
        S = np.hstack((S, np.expand_dims(np.interp(ses.ts_F.to_numpy(
        ), ses.videodata['timestamps'].to_numpy(), ses.videodata[field].to_numpy()), axis=1)))
        Slabels.append(field)

    return S, Slabels
