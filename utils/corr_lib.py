"""
This script contains functions to compute noise correlations
on simultaneously acquired calcium imaging data with mesoscope
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic,binned_statistic_2d
from skimage.measure import block_reduce
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.plot_lib import shaded_error
from utils.plotting_style import * #get all the fixed color schemes
from utils.psth import mean_resp_gn,mean_resp_image,mean_resp_gr
from utils.rf_lib import filter_nearlabeled

def compute_trace_correlation(sessions,uppertriangular=True,binwidth=1):
    nSessions = len(sessions)
    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing trace correlations: '):
    
        avg_nframes     = int(np.round(sessions[ises].sessiondata['fs'][0] * binwidth))

        arr_reduced     = block_reduce(sessions[ises].calciumdata.T, block_size=(1,avg_nframes), func=np.mean, cval=np.mean(sessions[ises].calciumdata.T))

        sessions[ises].trace_corr                   = np.corrcoef(arr_reduced)
        N           = np.shape(sessions[ises].calciumdata)[1] #get dimensions of response matrix

        idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
        
        if uppertriangular:
            sessions[ises].trace_corr[idx_triu] = np.nan
        else:
            np.fill_diagonal(sessions[ises].trace_corr,np.nan)

        assert np.all(sessions[ises].trace_corr[~idx_triu] > -1)
        assert np.all(sessions[ises].trace_corr[~idx_triu] < 1)
    return sessions    

def compute_signal_noise_correlation(sessions,uppertriangular=True):
    # computing the pairwise correlation of activity that is shared due to mean response (signal correlation)
    # or residual to any stimuli in GR and GN protocols (noise correlation).

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing signal correlations: '):
        if sessions[ises].sessiondata['protocol'][0]=='IM':
            [respmean,imageids]         = mean_resp_image(sessions[ises])
            [N,K]                       = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            sessions[ises].sig_corr     = np.corrcoef(respmean)

            if uppertriangular:
                idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
                sessions[ises].sig_corr[idx_triu] = np.nan

        elif sessions[ises].sessiondata['protocol'][0]=='GR':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(sessions[ises].trialdata['Orientation'].unique())
            resp_meanori,respmat_res        = mean_resp_gr(sessions[ises])
            prefori                         = oris[np.argmax(resp_meanori,axis=1)]
            # delta_pref                      = np.subtract.outer(prefori, prefori)
            sessions[ises].delta_pref       = np.abs(np.mod(np.subtract.outer(prefori, prefori),180))
            
            sessions[ises].sig_corr         = np.corrcoef(resp_meanori)
            sessions[ises].noise_corr       = np.corrcoef(respmat_res)

            idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
            if uppertriangular:
                sessions[ises].noise_corr[idx_triu] = np.nan
                sessions[ises].sig_corr[idx_triu] = np.nan
                sessions[ises].delta_pref[idx_triu] = np.nan
            else: #set only autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)
                np.fill_diagonal(sessions[ises].delta_pref,np.nan)
                np.fill_diagonal(sessions[ises].noise_corr,np.nan)

            assert np.all(sessions[ises].sig_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].sig_corr[~idx_triu] < 1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] < 1)
        
        elif sessions[ises].sessiondata['protocol'][0]=='GN':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(pd.Series.unique(sessions[ises].trialdata['centerOrientation']))
            speeds                          = np.sort(pd.Series.unique(sessions[ises].trialdata['centerSpeed']))
            resp_meanori,respmat_res        = mean_resp_gn(sessions[ises])
            prefori, prefspeed              = np.unravel_index(resp_meanori.reshape(N,-1).argmax(axis=1), (len(oris), len(speeds)))
            sessions[ises].prefori          = oris[prefori]
            sessions[ises].prefspeed        = speeds[prefspeed]

            sessions[ises].sig_corr         = np.corrcoef(resp_meanori.reshape(N,len(oris)*len(speeds)))
            sessions[ises].noise_corr       = np.corrcoef(respmat_res)

            idx_triu = np.tri(N,N,k=0)==1   #index upper triangular part
            if uppertriangular:
                sessions[ises].sig_corr[idx_triu] = np.nan
                sessions[ises].noise_corr[idx_triu] = np.nan
            else: #set autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)
                np.fill_diagonal(sessions[ises].noise_corr,np.nan)

            assert np.all(sessions[ises].sig_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].sig_corr[~idx_triu] < 1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] < 1)
        # else, do nothing, skipping protocol other than GR, GN, and IM'

    return sessions

def compute_pairwise_metrics(sessions):
    sessions = compute_pairwise_anatomical_distance(sessions)
    sessions = compute_pairwise_delta_rf(sessions)
    return sessions

def compute_pairwise_anatomical_distance(sessions):

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing pairwise anatomical distance for each session: '):
        N           = len(sessions[ises].celldata) #get dimensions of response matrix

        ## Compute euclidean distance matrix based on soma center:
        sessions[ises].distmat_xyz     = np.zeros((N,N))
        sessions[ises].distmat_xy      = np.zeros((N,N))

        x = sessions[ises].celldata['xloc'].to_numpy()
        y = sessions[ises].celldata['yloc'].to_numpy()
        z = sessions[ises].celldata['depth'].to_numpy()
        b = np.array((x,y,z))
        for i in range(N):
            a = np.array((x[i],y[i],z[i]))
            sessions[ises].distmat_xyz[i,:] = np.linalg.norm(a[:,np.newaxis]-b,axis=0)
            sessions[ises].distmat_xy[i,:] = np.linalg.norm(a[:2,np.newaxis]-b[:2,:],axis=0)

        for area in ['V1','PM','AL','RSP']: #set all interarea pairs to nan:
            sessions[ises].distmat_xy[np.ix_(sessions[ises].celldata['roi_name']==area,sessions[ises].celldata['roi_name']!=area)] = np.nan
            sessions[ises].distmat_xyz[np.ix_(sessions[ises].celldata['roi_name']==area,sessions[ises].celldata['roi_name']!=area)] = np.nan
            # sessions[ises].distmat_xyz[~np.logical_or(sessions[ises].areamat=='V1-V1',sessions[ises].areamat=='PM-PM')] = np.nan

        idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
        sessions[ises].distmat_xyz[idx_triu] = np.nan
        sessions[ises].distmat_xy[idx_triu] = np.nan

    return sessions

def compute_pairwise_delta_rf(sessions,rf_type='F'):

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing pairwise delta receptive field for each session: '):
        N           = len(sessions[ises].celldata) #get dimensions of response matrix

        ## Compute euclidean distance matrix based on receptive field:
        sessions[ises].distmat_rf      = np.full((N,N),np.NaN)

        if 'rf_az_' + rf_type in sessions[ises].celldata:
            rfaz = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
            rfel = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()

            d = np.array((rfaz,rfel))

            for i in range(N):
                c = np.array((rfaz[i],rfel[i]))
                sessions[ises].distmat_rf[i,:] = np.linalg.norm(c[:,np.newaxis]-d,axis=0)

    return sessions

def plot_delta_rf_across_sessions(sessions,areapairs):
    clrs_areapairs = get_clr_area_pairs(areapairs)
    binedges    = np.arange(-5,150,5) 
    nbins       = len(binedges)-1
    # binmean     = np.full((len(sessions),len(areapairs),nbins),np.nan)

    fig,axes = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*3,3))
    for ipair,areapair in enumerate(areapairs):
        for ses in sessions:
                        
            # Define function to filter neuronpairs based on area combination
            areafilter = filter_2d_areapair(ses,areapair)
            nanfilter  = ~np.isnan(ses.distmat_rf)
            cellfilter = np.logical_and(areafilter,nanfilter)
            sns.histplot(data=ses.distmat_rf[cellfilter].flatten(),bins=binedges,ax=axes[ipair],color=clrs_areapairs[ipair],
                         alpha=0.5,fill=False,stat='percent',element='step')
        axes[ipair].set_title(areapair)
            # axes[ipair].hist(ses.distmat_rf[cellfilter],bins=binedges,color=clrs_areapairs[ipair],alpha=0.5)
    return fig



def noisecorr_rfmap(sessions,corr_type='noise_corr',binresolution=5,rotate_prefori=False,rotate_deltaprefori=False,
                    thr_tuned=0,thr_rf_p=1,rf_type='F'):
    # Computes the average noise correlation depending on the difference in receptive field between the two neurons
    # binresolution determines spatial bins in degrees visual angle
    # If rotate_prefori=True then the delta RF is rotated depending on the preferred orientation of the source neuron 
    # This means that the output axis are now collinear vs orthogonal instead of azimuth and elevation
    
    if rotate_prefori:
        binrange        = np.array([[-135, 135],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
    else: 
        binrange        = np.array([[-50, 50],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
  
    noiseRFmat          = np.zeros(nBins)
    countsRFmat         = np.zeros(nBins)

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D noise corr histograms maps: '):
        
        if 'rf_az_' + rf_type in sessions[ises].celldata:
            idx_source = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF
            idx_target = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF

            if thr_tuned:
                idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>thr_tuned)
                idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>thr_tuned)
            
            if thr_rf_p<1:
                if 'rf_p' in sessions[ises].celldata:
                    idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
                    idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
                    
            corrdata = getattr(sessions[ises], corr_type)

            [noiseRFmat_ses,countsRFmat_ses] = compute_NC_map(sourcecells = sessions[ises].celldata[idx_source].reset_index(drop=True),
                                                    targetcells = sessions[ises].celldata[idx_target].reset_index(drop=True),
                                                    NC_data = corrdata[np.ix_(idx_source, idx_target)],
                                                    nBins=nBins,binrange=binrange,
                                                    rotate_deltaprefori=rotate_deltaprefori, rotate_prefori=rotate_prefori)
            noiseRFmat      += noiseRFmat_ses
            countsRFmat     += countsRFmat_ses

    # divide the total summed noise correlations by the number of counts in that bin to get the mean:
    noiseRFmat_mean = noiseRFmat / countsRFmat 
    
    return noiseRFmat_mean,countsRFmat,binrange

def noisecorr_rfmap_perori(sessions,corr_type='noise_corr',binresolution=5,rotate_prefori=False,rotate_deltaprefori=False,
                           thr_tuned=0,thr_rf_p=1,rf_type='F'):
    """
    Computes the average noise correlation depending on 
    azimuth and elevation
        Parameters:
    sessions (list of Session objects)
    binresolution (int, default=5)
    rotate_prefori (bool, default=False)
    rotate_deltaprefori (bool, default=False)
    thr_tuned (float, default=0)
    thr_rf_p (float, default=1)
    corr_type (str, default='distmat_rf')
        Type of correlation data to use. Can be one of:
            - 'noise_corr'
            - 'trace_corr'
            - 'sig_corr'
    
    Returns:
    noiseRFmat_mean, countsRFmat, binrange
    """
    # Computes the average noise correlation depending on the difference in receptive field between the two neurons
    # binresolution determines spatial bins in degrees visual angle
    # If rotate_prefori=True then the delta RF is rotated depending on the preferred orientation of the source neuron 
    # This means that the output axis are now collinear vs orthogonal instead of azimuth and elevation
    
    oris = np.sort(sessions[0].trialdata['Orientation'].unique())
    nOris = len(oris)

    if rotate_prefori:
        binrange        = np.array([[-135, 135],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
    else: 
        binrange        = np.array([[-50, 50],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
  
    noiseRFmat          = np.zeros((nOris,*nBins))
    countsRFmat         = np.zeros((nOris,*nBins))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D noise corr histograms maps: '):
        if 'rf_az_' + rf_type in sessions[ises].celldata:
            for iOri,Ori in enumerate(oris):

                idx_source = np.logical_and(~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]),
                                            sessions[ises].celldata['pref_ori']==Ori)#get all neurons with RF
                # idx_target = ~np.isnan(sessions[ises].celldata['rf_az_Fneu'],
                #                              sessions[ises].celldata['pref_ori']==Ori) #get all neurons with RF
                idx_target = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type],
                                            sessions[ises].celldata['pref_ori'].between(Ori-30, Ori+30)) #get all neurons with RF

                if thr_tuned:
                    idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>thr_tuned)
                    idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>thr_tuned)
                    
                if thr_rf_p<1:
                    idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_Fneu']<thr_rf_p)
                    idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_Fneu']<thr_rf_p)
                
                corrdata = getattr(sessions[ises], corr_type)

                [noiseRFmat_ses,countsRFmat_ses] = compute_NC_map(sourcecells = sessions[ises].celldata[idx_source].reset_index(drop=True),
                                                        targetcells = sessions[ises].celldata[idx_target].reset_index(drop=True),
                                                        NC_data = corrdata[np.ix_(idx_source, idx_target)],
                                                        nBins=nBins,binrange=binrange,
                                                        rotate_prefori=rotate_prefori)
                noiseRFmat[iOri,:,:]      += noiseRFmat_ses
                countsRFmat[iOri,:,:]     += countsRFmat_ses

    # divide the total summed noise correlations by the number of counts in that bin to get the mean:
    noiseRFmat_mean = noiseRFmat / countsRFmat 
    
    return noiseRFmat_mean,countsRFmat,binrange


def noisecorr_rfmap_areas(sessions,corr_type='noise_corr',binresolution=5,rotate_prefori=False,
                            rotate_deltaprefori=False,thr_tuned=0,thr_rf_p=1,rf_type='F'):

    areas               = ['V1','PM']

    if rotate_prefori:
        binrange        = np.array([[-135, 135],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
    else: 
        binrange        = np.array([[-50, 50],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
  
    noiseRFmat          = np.zeros((2,2,*nBins))
    countsRFmat         = np.zeros((2,2,*nBins))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D noise corr histograms maps: '):
        if 'rf_az_' + rf_type in sessions[ises].celldata and hasattr(sessions[ises], corr_type):
            for ixArea,xArea in enumerate(areas):
                for iyArea,yArea in enumerate(areas):

                    idx_source = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF
                    idx_target = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF

                    if thr_tuned:
                        idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>thr_tuned)
                        idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>thr_tuned)
                    
                    if thr_rf_p<1:
                        idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
                        idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
                    
                    idx_source = np.logical_and(idx_source,sessions[ises].celldata['roi_name']==xArea)
                    idx_target = np.logical_and(idx_target,sessions[ises].celldata['roi_name']==yArea)

                    corrdata = getattr(sessions[ises], corr_type)

                    [noiseRFmat_temp,countsRFmat_temp] = compute_NC_map(sourcecells = sessions[ises].celldata[idx_source].reset_index(drop=True),
                                                            targetcells = sessions[ises].celldata[idx_target].reset_index(drop=True),
                                                            NC_data = corrdata[np.ix_(idx_source, idx_target)],
                                                            nBins=nBins,binrange=binrange,
                                                            rotate_deltaprefori=rotate_deltaprefori, rotate_prefori=rotate_prefori)
                    noiseRFmat[ixArea,iyArea,:,:]  += noiseRFmat_temp
                    countsRFmat[ixArea,iyArea,:,:] += countsRFmat_temp

    # divide the total summed noise correlations by the number of counts in that bin to get the mean:
    noiseRFmat_mean = noiseRFmat / countsRFmat 
    
    return noiseRFmat_mean,countsRFmat,binrange


def noisecorr_rfmap_areas_projections(sessions,corr_type='noise_corr',binresolution=5,rotate_prefori=True,
                            rotate_deltaprefori=False,thr_tuned=0,thr_rf_p=1,rf_type='F'):

    areas               = ['V1','PM']
    redcells            = [0,1]
    redcelllabels       = ['unl','lab']
    legendlabels        = np.empty((4,4),dtype='object')

    if rotate_prefori:
        binrange        = np.array([[-135, 135],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
    else: 
        binrange        = np.array([[-50, 50],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
  
    # noiseRFmat          = np.zeros((2,2,2,2,*nBins))
    # countsRFmat         = np.zeros((2,2,2,2,*nBins))
    
    noiseRFmat          = np.zeros((4,4,*nBins))
    countsRFmat         = np.zeros((4,4,*nBins))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D noise corr histograms maps: '):
        if 'rf_az_' + rf_type in sessions[ises].celldata:
            for ixArea,xArea in enumerate(areas):
                for iyArea,yArea in enumerate(areas):
                    for ixRed,xRed in enumerate(redcells):
                        for iyRed,yRed in enumerate(redcells):
                            
                            idx_source = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF
                            idx_target = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF

                            idx_source = np.logical_and(idx_source,sessions[ises].celldata['roi_name']==xArea)
                            idx_target = np.logical_and(idx_target,sessions[ises].celldata['roi_name']==yArea)

                            idx_source = np.logical_and(idx_source,sessions[ises].celldata['redcell']==xRed)
                            idx_target = np.logical_and(idx_target,sessions[ises].celldata['redcell']==yRed)

                            if thr_tuned:
                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>thr_tuned)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>thr_tuned)
                            
                            if thr_rf_p<1:
                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
                    
                            corrdata = getattr(sessions[ises], corr_type)

                            [noiseRFmat_temp,countsRFmat_temp] = compute_NC_map(sourcecells = sessions[ises].celldata[idx_source].reset_index(drop=True),
                                                                    targetcells = sessions[ises].celldata[idx_target].reset_index(drop=True),
                                                                    NC_data = corrdata[np.ix_(idx_source, idx_target)],
                                                                    nBins=nBins,binrange=binrange,rf_type=rf_type,
                                                                    rotate_deltaprefori=rotate_deltaprefori, rotate_prefori=rotate_prefori)
                            
                            noiseRFmat[ixArea*2 + ixRed,iyArea*2 + iyRed,:,:]  += noiseRFmat_temp
                            countsRFmat[ixArea*2 + ixRed,iyArea*2 + iyRed,:,:] += countsRFmat_temp
                            
                            legendlabels[ixArea*2 + ixRed,iyArea*2 + iyRed]  = areas[ixArea] + redcelllabels[ixRed] + '-' + areas[iyArea] + redcelllabels[iyRed]

    # divide the total summed noise correlations by the number of counts in that bin to get the mean:
    noiseRFmat_mean = noiseRFmat / countsRFmat 

    return noiseRFmat_mean,countsRFmat,binrange,legendlabels

def compute_NC_map(sourcecells,targetcells,NC_data,nBins,binrange,
                   rotate_deltaprefori=False,rotate_prefori=False,rf_type='F'):

    noiseRFmat          = np.zeros(nBins)
    countsRFmat         = np.zeros(nBins)

    for iN in range(len(sourcecells)):
        delta_el    = targetcells['rf_el_' + rf_type] - sourcecells['rf_el_' + rf_type][iN]
        delta_az    = targetcells['rf_az_' + rf_type] - sourcecells['rf_az_' + rf_type][iN]
        angle_vec   = np.vstack((delta_el, delta_az))

        if rotate_deltaprefori:
            ori_rots    = targetcells['pref_ori'] - sourcecells['pref_ori'][iN]
            angle_vec   = apply_ori_rot(angle_vec,ori_rots)
        
        if rotate_prefori:
            ori_rots   = np.tile(sourcecells['pref_ori'][iN],len(targetcells))
            angle_vec  = apply_ori_rot(angle_vec,ori_rots)

        noiseRFmat       = noiseRFmat + binned_statistic_2d(x=angle_vec[0,:],y=angle_vec[1,:],
                        values = NC_data[iN, :],
                        bins=nBins,range=binrange,statistic='sum')[0]
        
        countsRFmat      = countsRFmat + np.histogram2d(x=angle_vec[0,:],y=angle_vec[1,:],
                        bins=nBins,range=binrange)[0]
            
    return noiseRFmat,countsRFmat

def apply_ori_rot(angle_vec,ori_rots):
    oris = np.sort(np.unique(ori_rots))
    rotation_matrix_oris = np.empty((2,2,len(oris)))
    for iori,ori in enumerate(oris):
        c, s = np.cos(np.radians(ori)), np.sin(np.radians(ori))
        rotation_matrix_oris[:,:,iori] = np.array(((c, -s), (s, c)))

    for iori,ori in enumerate(oris):
        ori_diff = np.mod(ori_rots,360)
        idx_ori = ori_diff ==ori

        angle_vec[:,idx_ori] = rotation_matrix_oris[:,:,iori] @ angle_vec[:,idx_ori]

    return angle_vec


def compute_noisecorr_rfmap_v2(sessions,binresolution=5,rotate_prefori=False,splitareas=False,splitlabeled=False):
    # Computes the average noise correlation depending on the difference in receptive field between the two neurons
    # binresolution determines spatial bins in degrees visual angle
    # If rotate_prefori=True then the delta RF is rotated depending on their 
    # This means that the output axis are now collinear vs orthogonal instead of azimuth and elevation
    
    if rotate_prefori:
        binrange        = np.array([[-135, 135],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
    else: 
        binrange        = np.array([[-50, 50],[-135, 135]])
        nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
    
    celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

    if splitareas is not None:
        areas = np.sort(np.unique(celldata['roi_name']))[::-1]

    if splitlabeled is not None:
        redcells            = [0,1]
        redcelllabels       = ['unl','lab']
    
    # legendlabels        = np.empty((4,4),dtype='object')
    # noiseRFmat          = np.zeros((4,4,*nBins))
    # countsRFmat         = np.zeros((4,4,*nBins))

    rotate_prefori = True

    noiseRFmat          = np.zeros(nBins)
    countsRFmat         = np.zeros(nBins)

    if rotate_prefori:
        oris            = np.sort(sessions[0].trialdata['Orientation'].unique())
        rotation_matrix_oris = np.empty((2,2,len(oris)))
        for iori,ori in enumerate(oris):
            c, s = np.cos(np.radians(ori)), np.sin(np.radians(ori))
            rotation_matrix_oris[:,:,iori] = np.array(((c, -s), (s, c)))


    for ises in range(len(sessions)):
        print('computing 2d receptive field hist of noise correlations for session %d / %d' % (ises+1,len(sessions)))
        nNeurons    = len(sessions[ises].celldata) #number of neurons in this session
        idx_RF      = ~np.isnan(sessions[ises].celldata['rf_az_Fneu']) #get all neurons with RF

        # for iN in range(nNeurons):
        for iN in range(100):
        # for iN in range(100):
            if idx_RF[iN]:
                idx = np.logical_and(idx_RF, range(nNeurons) != iN)

                delta_el = sessions[ises].celldata['rf_el_Fneu'] - sessions[ises].celldata['rf_el_Fneu'][iN]
                delta_az = sessions[ises].celldata['rf_az_Fneu'] - sessions[ises].celldata['rf_az_Fneu'][iN]

                angle_vec = np.vstack((delta_el, delta_az))
                if rotate_prefori:
                    for iori,ori in enumerate(oris):
                        ori_diff = np.mod(sessions[ises].celldata['pref_ori'] - sessions[ises].celldata['pref_ori'][iN],360)
                        idx_ori = ori_diff ==ori

                        angle_vec[:,idx_ori] = rotation_matrix_oris[:,:,iori] @ angle_vec[:,idx_ori]

                noiseRFmat       = noiseRFmat + binned_statistic_2d(x=angle_vec[0,idx],y=angle_vec[1,idx],
                                values = sessions[ises].noise_corr[iN, idx],
                                bins=nBins,range=binrange,statistic='sum')[0]
                
                countsRFmat      = countsRFmat + np.histogram2d(x=angle_vec[0,idx],y=angle_vec[1,idx],
                                bins=nBins,range=binrange)[0]
    
    # divide the total summed noise correlations by the number of counts in that bin to get the mean:
    noiseRFmat_mean = noiseRFmat / countsRFmat 
    
    return noiseRFmat_mean,countsRFmat,binrange

def bin_corr_distance(sessions,areapairs,corr_type='trace_corr',normalize=False):
    binedges = np.arange(0,1000,20) 
    nbins= len(binedges)-1
    binmean = np.full((len(sessions),len(areapairs),nbins),np.nan)
    for ises in tqdm(range(len(sessions)),desc= 'Computing trace correlations per pairwise distance: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            # corrdata[corrdata>0] = np.nan
            for iap,areapair in enumerate(areapairs):
                areafilter      = filter_2d_areapair(sessions[ises],areapair)
                nanfilter       = ~np.isnan(corrdata)
                cellfilter      = np.all((areafilter,nanfilter),axis=0)
                binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_xyz[cellfilter].flatten(),
                                                    values=corrdata[cellfilter].flatten(),
                                                    statistic='mean', bins=binedges)[0]
            
    if normalize: # subtract mean NC from every session:
        binmean = binmean - np.nanmean(binmean[:,:,binedges[:-1]<600],axis=2,keepdims=True)

    return binmean,binedges

def plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type):
    sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    protocols = np.unique(sessiondata['protocol'])
    clrs_areapairs = get_clr_area_pairs(areapairs)
    fig,axes = plt.subplots(1,len(protocols),figsize=(4*len(protocols),4))
    handles = []
    for iprot,protocol in enumerate(protocols):
        sesidx = np.where(sessiondata['protocol']== protocol)[0]
        if len(protocols)>1:
            ax = axes[iprot]
        else:
            ax = axes

        for iap,areapair in enumerate(areapairs):
            for ises in sesidx:
                ax.plot(binedges[:-1],binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
            handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[sesidx,iap,:].squeeze(),error='sem',color=clrs_areapairs[iap]))
            # plt.savefig(os.path.join(savedir,'NoiseCorr_distRF_RegressOut_' + areapair + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

        ax.legend(handles,areapairs,loc='upper right',frameon=False)	
        ax.set_xlabel('Anatomical distance ($\mu$m)')
        ax.set_ylabel('Correlation')
        ax.set_xlim([10,600])
        ax.set_title('%s (%s)' % (corr_type,protocol))
        # ax.set_ylim([-0.01,0.04])
        ax.set_ylim([0,0.09])
        ax.set_aspect('auto')
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    return fig


def bin_corr_distance(sessions,areapairs,corr_type='trace_corr',normalize=False):
    binedges = np.arange(0,1000,10) 
    nbins= len(binedges)-1
    binmean = np.full((len(sessions),len(areapairs),nbins),np.nan)
    for ises in tqdm(range(len(sessions)),desc= 'Binning correlations by pairwise anatomical distance: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            # corrdata[corrdata>0] = np.nan
            for iap,areapair in enumerate(areapairs):
                areafilter      = filter_2d_areapair(sessions[ises],areapair)
                nanfilter       = ~np.isnan(corrdata)
                cellfilter      = np.all((areafilter,nanfilter),axis=0)
                binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_xyz[cellfilter].flatten(),
                                                    values=corrdata[cellfilter].flatten(),
                                                    statistic='mean', bins=binedges)[0]
            
    if normalize: # subtract mean NC from every session:
        binmean = binmean - np.nanmean(binmean[:,:,binedges[:-1]<600],axis=2,keepdims=True)

    return binmean,binedges

def bin_corr_deltarf_areapairs(sessions,areapairs,corr_type='trace_corr',normalize=False,rf_type = 'F',sig_thr = 0.001):
    binedges    = np.arange(0,120,2.5) 
    nbins       = len(binedges)-1
    binmean     = np.full((len(sessions),len(areapairs),nbins),np.nan)
    bincount     = np.full((len(sessions),len(areapairs),nbins),np.nan)
    
    for ises in tqdm(range(len(sessions)),desc= 'Binning correlations by delta receptive field: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            if 'rf_p_F' in sessions[ises].celldata:
                for iap,areapair in enumerate(areapairs):
                    signalfilter    = np.meshgrid(sessions[ises].celldata['rf_p_' + rf_type]<sig_thr,sessions[ises].celldata['rf_p_'  + rf_type]<sig_thr)
                    signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])
                    
                    areafilter      = filter_2d_areapair(sessions[ises],areapair)
                    nanfilter       = ~np.isnan(corrdata)
                    proxfilter      = ~(sessions[ises].distmat_xy<20)
                    
                    cellfilter      = np.all((signalfilter,areafilter,proxfilter,nanfilter),axis=0)
                    binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_rf[cellfilter].flatten(),
                                                        values=corrdata[cellfilter].flatten(),
                                                        statistic='mean', bins=binedges)[0]
                    bincount[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_rf[cellfilter].flatten(),
                                                        values=corrdata[cellfilter].flatten(),
                                                        statistic='count', bins=binedges)[0]
                    
    binmean[bincount<25] = np.nan
    if normalize: # subtract mean correlation from every session:
        binmean = binmean - np.nanmean(binmean[:,:,binedges[:-1]<100],axis=2,keepdims=True)

    return binmean,binedges


def plot_bin_corr_deltarf_protocols(sessions,binmean,binedges,areapairs,corr_type,normalize=False):
    sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    protocols = np.unique(sessiondata['protocol'])
    clrs_areapairs = get_clr_area_pairs(areapairs)

    fig,axes = plt.subplots(1,len(protocols),figsize=(4*len(protocols),4))
    handles = []
    for iprot,protocol in enumerate(protocols):
        sesidx = np.where(sessiondata['protocol']== protocol)[0]
        if len(protocols)>1:
            ax = axes[iprot]
        else:
            ax = axes

        for iap,areapair in enumerate(areapairs):
            for ises in sesidx:
                ax.plot(binedges[:-1],binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
            handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[sesidx,iap,:].squeeze(),center='mean',error='sem',color=clrs_areapairs[iap]))

        ax.legend(handles,areapairs,loc='upper right',frameon=False)	
        ax.set_xlabel('Delta RF')
        ax.set_ylabel('Correlation')
        ax.set_xlim([-2,100])
        ax.set_title('%s (%s)' % (corr_type,protocol))
        if normalize:
            ax.set_ylim([-0.015,0.05])
        else: 
            ax.set_ylim([0,0.12])
        ax.set_aspect('auto')
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    return fig

# Define function to filter neuronpairs based on area combination
def filter_2d_areapair(ses,areapair):
    area1,area2 = areapair.split('-')
    areafilter1 = np.meshgrid(ses.celldata['roi_name']==area1,ses.celldata['roi_name']==area2)
    areafilter1 = np.logical_and(areafilter1[0],areafilter1[1])
    areafilter2 = np.meshgrid(ses.celldata['roi_name']==area1,ses.celldata['roi_name']==area2)
    areafilter2 = np.logical_and(areafilter2[0],areafilter2[1])

    return np.logical_or(areafilter1,areafilter2)

def mean_corr_areas_labeling(sessions,corr_type='trace_corr',absolute=False,minNcells=10):
    areas               = ['V1','PM']
    redcells            = [0,1]
    redcelllabels       = ['unl','lab']
    legendlabels        = np.empty((4,4),dtype='object')

    noisemat            = np.full((4,4,len(sessions)),np.nan)

    for ises in tqdm(range(len(sessions)),desc='Averaging %s across sessions' % corr_type):
        idx_nearfilter = filter_nearlabeled(sessions[ises],radius=50)
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
        
            if absolute:
                corrdata = np.abs(corrdata)

            for ixArea,xArea in enumerate(areas):
                for iyArea,yArea in enumerate(areas):
                    for ixRed,xRed in enumerate(redcells):
                        for iyRed,yRed in enumerate(redcells):

                                idx_source = sessions[ises].celldata['roi_name']==xArea
                                idx_target = sessions[ises].celldata['roi_name']==yArea

                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['redcell']==xRed)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['redcell']==yRed)

                                idx_source = np.logical_and(idx_source,idx_nearfilter)
                                idx_target = np.logical_and(idx_target,idx_nearfilter)

                                if np.sum(idx_source)>minNcells and np.sum(idx_target)>minNcells:	
                                    noisemat[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(corrdata[np.ix_(idx_source, idx_target)])
                                
                                legendlabels[ixArea*2 + ixRed,iyArea*2 + iyRed]  = areas[ixArea] + redcelllabels[ixRed] + '-' + areas[iyArea] + redcelllabels[iyRed]

    # assuming legendlabels is a 4x4 array
    legendlabels_upper_tri = legendlabels[np.triu_indices(4, k=0)]

    # assuming noisemat is a 4x4xnSessions array
    upper_tri_indices = np.triu_indices(4, k=0)
    noisemat_upper_tri = noisemat[upper_tri_indices[0], upper_tri_indices[1], :]

    df = pd.DataFrame(data=noisemat_upper_tri.T,columns=legendlabels_upper_tri)

    colorder = [0,1,4,7,8,9,2,3,5,6]
    legendlabels_upper_tri = legendlabels_upper_tri[colorder]
    df = df[legendlabels_upper_tri]

    return df
