"""
This script contains functions to compute noise correlations
on simultaneously acquired calcium imaging data with mesoscope
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

## Import libs:
import os
import copy
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic,binned_statistic_2d
from skimage.measure import block_reduce
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.plot_lib import *
from utils.plotting_style import * #get all the fixed color schemes
from utils.tuning import mean_resp_gn,mean_resp_gr,mean_resp_image 
from utils.rf_lib import filter_nearlabeled
from utils.pair_lib import *
from sklearn.decomposition import PCA
from statannotations.Annotator import Annotator
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from utils.dimreduc_lib import remove_dim
from scipy.signal import detrend
from utils.gain_lib import pop_rate_gain_model
import itertools
import scipy.stats as ss
from scipy.optimize import curve_fit
from scipy.stats import linregress


 #####  ####### #     # ######  #     # ####### #######     #####  ####### ######  ######  
#     # #     # ##   ## #     # #     #    #    #          #     # #     # #     # #     # 
#       #     # # # # # #     # #     #    #    #          #       #     # #     # #     # 
#       #     # #  #  # ######  #     #    #    #####      #       #     # ######  ######  
#       #     # #     # #       #     #    #    #          #       #     # #   #   #   #   
#     # #     # #     # #       #     #    #    #          #     # #     # #    #  #    #  
 #####  ####### #     # #        #####     #    #######     #####  ####### #     # #     # 

def compute_trace_correlation(sessions,uppertriangular=True,binwidth=1):
    """
    Compute the trace correlation between the calcium traces of all neurons in a session
    Trace correlation is computed by taking the mean of the fluorescence traces over a specified time window (binwidth)
    Parameters
    sessions : Session
        list of Session objects
    uppertriangular : bool
        if set to True, only upper triangular part of the correlation matrix is computed
    binwidth : float
        time window over which to compute the mean of the fluorescence trace
    Returns sessions
    """
    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing trace correlations: '):
    
        avg_nframes     = int(np.round(sessions[ises].sessiondata['fs'][0] * binwidth))

        if avg_nframes > 1:
            arr_reduced     = block_reduce(sessions[ises].calciumdata.T, block_size=(1,avg_nframes), func=np.mean, cval=np.mean(sessions[ises].calciumdata.T))
        else:
            arr_reduced     = sessions[ises].calciumdata.T.to_numpy()

        sessions[ises].trace_corr                   = np.corrcoef(arr_reduced)

        N           = np.shape(sessions[ises].calciumdata)[1] #get dimensions of response matrix

        idx_triu    = np.tri(N,N,k=0)==1 #index only upper triangular part
        
        if uppertriangular:
            sessions[ises].trace_corr[idx_triu] = np.nan
        else:
            np.fill_diagonal(sessions[ises].trace_corr,np.nan)

        assert np.all(sessions[ises].trace_corr[~idx_triu] > -1)
        assert np.all(sessions[ises].trace_corr[~idx_triu] < 1)
    return sessions    

def compute_signal_noise_correlation(sessions,uppertriangular=True,filter_stationary=False,remove_method=None,remove_rank=0):
    # computing the pairwise correlation of activity that is shared due to mean response (signal correlation)
    # or residual to any stimuli in GR and GN protocols (noise correlation).

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing signal and noise correlations: '):
        if sessions[ises].sessiondata['protocol'][0]=='IM':
            [respmean,imageids]         = mean_resp_image(sessions[ises])
            [N,K]                       = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            sessions[ises].sig_corr     = np.corrcoef(respmean)

            if uppertriangular:
                idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
                sessions[ises].sig_corr[idx_triu] = np.nan
            else: #set only autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)

        elif sessions[ises].sessiondata['protocol'][0]=='GR':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(sessions[ises].trialdata['Orientation'].unique())
            trialfilter                     = sessions[ises].respmat_runspeed<2 if filter_stationary else np.ones(K,bool)
            resp_meanori,respmat_res        = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
            prefori                         = oris[np.argmax(resp_meanori,axis=1)]

            sessions[ises].delta_pref       = np.abs(np.mod(np.subtract.outer(prefori, prefori),180))
            
            # Compute signal correlations on all trials: 
            # sessions[ises].sig_corr         = np.corrcoef(resp_meanori)

            #Compute signal correlation on separate halfs of trials:
            trialfilter                     = np.random.choice([True,False],size=(K),p=[0.5,0.5])
            resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
            resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=~trialfilter)
            sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_meanori1, resp_meanori2)[:N, N:] +
                                                np.corrcoef(resp_meanori2, resp_meanori1)[:N, N:])

            if remove_method is not None:
                if remove_method in ['PCA','FA','RRR']:

                    assert remove_rank > 0, 'remove_rank must be > 0'	
                    
                    trial_ori   = sessions[ises].trialdata['Orientation']
                    respmat_res = copy.deepcopy(sessions[ises].respmat)
                    respmat_res = zscore(respmat_res,axis=1)
                    
                    # for iarea,area in enumerate(sessions[ises].celldata['roi_name'].unique()):
                    #     idx = sessions[ises].celldata['roi_name'] == area
                    #     data = respmat_res[idx,:]

                        # data_hat = remove_dim(data,remove_method,remove_rank)

                    #     #Remove low rank prediction from data:
                    #     respmat_res[idx,:] = data - data_hat
                    
                    for i,ori in enumerate(oris):
                        data = respmat_res[:,trial_ori==ori]
                        
                        data_hat = remove_dim(data,remove_method,remove_rank)
                        
                        #Remove low rank prediction from data:
                        respmat_res[:,trial_ori==ori] = data - data_hat
                elif remove_method == 'GM':
                    stimuli         = np.array(sessions[ises].trialdata['stimCond'])
                    data_hat        = pop_rate_gain_model(sessions[ises].respmat, stimuli)
                    respmat_res     = sessions[ises].respmat - data_hat

            # Compute noise correlations from residuals:
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
            trialfilter                     = sessions[ises].respmat_runspeed<2 if filter_stationary else np.ones(K,bool)
            resp_mean,respmat_res           = mean_resp_gn(sessions[ises],trialfilter)
            prefori, prefspeed              = np.unravel_index(resp_mean.reshape(N,-1).argmax(axis=1), (len(oris), len(speeds)))
            sessions[ises].prefori          = oris[prefori]
            sessions[ises].prefspeed        = speeds[prefspeed]

            # Compute signal correlations on all trials: 
            # sessions[ises].sig_corr         = np.corrcoef(resp_mean.reshape(N,len(oris)*len(speeds)))
            
            #Compute signal correlation on separate halfs of trials:
            trialfilter                     = np.random.choice([True,False],size=(K),p=[0.5,0.5])
            resp_mean1,_                    = mean_resp_gn(sessions[ises],trialfilter = trialfilter)
            resp_mean2,_                    = mean_resp_gn(sessions[ises],trialfilter = ~trialfilter)
            sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_mean1, resp_mean2)[:N, N:] +
                                                np.corrcoef(resp_mean2, resp_mean1)[:N, N:])

            if remove_method is not None:
                if remove_method in ['PCA','FA','RRR']:
                    assert remove_rank > 0, 'remove_rank must be > 0'	
                    respmat_res = copy.deepcopy(sessions[ises].respmat)
                    respmat_res = zscore(respmat_res,axis=1)

                    trial_ori   = sessions[ises].trialdata['centerOrientation']
                    trial_spd   = sessions[ises].trialdata['centerSpeed']
                    for iO,ori in enumerate(oris):
                        for iS,speed in enumerate(speeds):
                            idx_trial = np.logical_and(trial_ori==ori,trial_spd==speed)
                            data = respmat_res[:,idx_trial]
                            data_hat = remove_dim(data,remove_method,remove_rank)
                            #Remove low rank prediction from data:
                            respmat_res[:,idx_trial] = data - data_hat
                elif remove_method == 'GM':
                    stimuli         = np.array(sessions[ises].trialdata['stimCond'])
                    data_hat        = pop_rate_gain_model(sessions[ises].respmat, stimuli)
                    respmat_res     = sessions[ises].respmat - data_hat

            # Detrend the data:
            # respmat_res = detrend(respmat_res,axis=1)

            #Compute noise correlations from residuals:
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

#     # ###  #####  #######     #####  ####### ######  ######  
#     #  #  #     #    #       #     # #     # #     # #     # 
#     #  #  #          #       #       #     # #     # #     # 
#######  #   #####     #       #       #     # ######  ######  
#     #  #        #    #       #       #     # #   #   #   #   
#     #  #  #     #    #       #     # #     # #    #  #    #  
#     # ###  #####     #        #####  ####### #     # #     # 

def hist_corr_areas_labeling(sessions,corr_type='trace_corr',filternear=True,minNcells=10, 
                        areapairs=' ',layerpairs=' ',projpairs=' ',noise_thr=100,valuematching=None,
                        zscore=False,binres=0.01):
    # areas               = ['V1','PM']
    # redcells            = [0,1]
    # redcelllabels       = ['unl','lab']
    # legendlabels        = np.empty((4,4),dtype='object')

    binedges            = np.arange(-1,1,binres)
    bincenters          = binedges[:-1] + binres/2
    nbins               = len(bincenters)

    if zscore:
        binedges            = np.arange(-5,5,binres)
        bincenters          = binedges[:-1] + binres/2
        nbins               = len(bincenters)

    histcorr           = np.full((nbins,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    meancorr           = np.full((len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    varcorr            = np.full((len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    fraccorr           = np.full((2,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)

    for ises in tqdm(range(len(sessions)),desc='Averaging %s across sessions' % corr_type):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            if valuematching is not None:
                #Get value to match from celldata:
                values  = sessions[ises].celldata[valuematching].to_numpy()

                #For both areas match the values between labeled and unlabeled cells
                idx_V1      = sessions[ises].celldata['roi_name']=='V1'
                idx_PM      = sessions[ises].celldata['roi_name']=='PM'
                group       = sessions[ises].celldata['redcell'].to_numpy()
                idx_sub_V1  = value_matching(np.where(idx_V1)[0],group[idx_V1],values[idx_V1],bins=20,showFig=False)
                idx_sub_PM  = value_matching(np.where(idx_PM)[0],group[idx_PM],values[idx_PM],bins=20,showFig=False)
                
                # matchfilter2d  = np.isin(sessions[ises].celldata.index[:,None], np.concatenate([idx_sub_V1,idx_sub_PM])[None,:])
                # matchfilter    = np.logical_and(matchfilter2d,matchfilter2d.T)

                matchfilter1d = np.zeros(len(sessions[ises].celldata)).astype(bool)
                matchfilter1d[idx_sub_V1] = True
                matchfilter1d[idx_sub_PM] = True

                matchfilter    = np.meshgrid(matchfilter1d,matchfilter1d)
                matchfilter    = np.logical_and(matchfilter[0],matchfilter[1])

            else: 
                matchfilter = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            if filternear:
                nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                nearfilter      = np.meshgrid(nearfilter,nearfilter)
                nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
            else: 
                nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            if zscore:
                corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)
            
            rf_type = 'Fsmo4oth'
            if 'rf_r2_' + rf_type in sessions[ises].celldata:
                source_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                target_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                delta_el        = source_el[:,None] - target_el[None,:]

                source_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                target_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                delta_az        = source_az[:,None] - target_az[None,:]

                delta_rf        = np.sqrt(delta_az**2 + delta_el**2)
                rffilter        = delta_rf<50
            else: 
                rffilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            for iap,areapair in enumerate(areapairs):
                for ilp,layerpair in enumerate(layerpairs):
                    for ipp,projpair in enumerate(projpairs):
                        signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
                        signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                        areafilter      = filter_2d_areapair(sessions[ises],areapair)

                        layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                        projfilter      = filter_2d_projpair(sessions[ises],projpair)

                        nanfilter       = ~np.isnan(corrdata)

                        proxfilter      = ~(sessions[ises].distmat_xy<10)

                        cellfilter      = np.all((signalfilter,areafilter,layerfilter,matchfilter,
                                                projfilter,proxfilter,nanfilter,nearfilter,rffilter),axis=0)

                        if np.sum(np.any(cellfilter,axis=0))>minNcells and np.sum(np.any(cellfilter,axis=1))>minNcells:
                            # if ipp==3:
                                # print(np.sum(cellfilter))
                            data      = corrdata[cellfilter].flatten()

                            histcorr[:,ises,iap,ilp,ipp]    = np.histogram(data,bins=binedges,density=True)[0]
                            meancorr[ises,iap,ilp,ipp]      = np.nanmean(data)
                            varcorr[ises,iap,ilp,ipp]       = np.nanstd(data)

                            if corr_type == 'trace_corr':
                                n = len(sessions[ises].ts_F)
                            elif corr_type in ['noise_corr','sig_corr']:
                                n = np.shape(sessions[ises].respmat)[1]

                            sigcorrdata = corrdata.copy()
                            sigcorrdata = filter_corr_p(sigcorrdata,n,p_thr=0.01)
                            fraccorr[0,ises,iap,ilp,ipp]       = np.sum(np.logical_and(cellfilter,sigcorrdata>0)) / np.sum(cellfilter)
                            fraccorr[1,ises,iap,ilp,ipp]       = np.sum(np.logical_and(cellfilter,sigcorrdata<0)) / np.sum(cellfilter)

    return bincenters,histcorr,meancorr,varcorr,fraccorr


def filter_corr_p(r,n,p_thr=0.01):
    """Filter out non-significant correlations in a correlation matrix.
    Parameters
    r : array
        Correlation matrix.
    n : int
        Number of datapoints.
    p_thr : float, optional
        Threshold for significant correlations. Default is 0.01.
    Returns
    r : array
        Correlation matrix with non-significant correlations set to nan.
    """
    t           = np.clip(r * np.sqrt((n-2)/(1-r*r)),a_min=-30,a_max=30)#convert correlation to t-statistic
    p           = ss.t.pdf(t, n-2) #convert to p-value using pdf of t-distribution and deg of freedom
    r[p>p_thr]  = np.nan #set all nonsignificant to nan
    # plt.scatter(r.flatten(),p.flatten())
    return r

#     # #######    #    #     #     #####  ####### ######  ######  
##   ## #         # #   ##    #    #     # #     # #     # #     # 
# # # # #        #   #  # #   #    #       #     # #     # #     # 
#  #  # #####   #     # #  #  #    #       #     # ######  ######  
#     # #       ####### #   # #    #       #     # #   #   #   #   
#     # #       #     # #    ##    #     # #     # #    #  #    #  
#     # ####### #     # #     #     #####  ####### #     # #     # 

def mean_corr_areas_labeling(sessions,corr_type='trace_corr',absolute=False,
                             filternear=True,filtersign=None,minNcells=10):
    areas               = ['V1','PM']
    redcells            = [0,1]
    redcelllabels       = ['unl','lab']
    legendlabels        = np.empty((4,4),dtype='object')

    meancorr            = np.full((4,4,len(sessions)),np.nan)
    fraccorr            = np.full((4,4,len(sessions)),np.nan)

    for ises in tqdm(range(len(sessions)),desc='Averaging %s across sessions' % corr_type):
        idx_nearfilter = filter_nearlabeled(sessions[ises],radius=50)
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            
            if filtersign == 'neg':
                corrdata[corrdata>0] = np.nan
            
            if filtersign =='pos':
                corrdata[corrdata<0] = np.nan

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

                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['noise_level']<20)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['noise_level']<20)

                                # if 'rf_p_F' in sessions[ises].celldata:
                                #     idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_F']<0.001)
                                    # idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_F']<0.001)

                                # if 'tuning_var' in sessions[ises].celldata:
                                #     idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>0.05)
                                #     idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>0.05)

                                if filternear:
                                    idx_source = np.logical_and(idx_source,idx_nearfilter)
                                    idx_target = np.logical_and(idx_target,idx_nearfilter)

                                if np.sum(idx_source)>minNcells and np.sum(idx_target)>minNcells:	
                                    meancorr[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(corrdata[np.ix_(idx_source, idx_target)])
                                    fraccorr[ixArea*2 + ixRed,iyArea*2 + iyRed,ises] = (
                                        np.sum(~np.isnan(corrdata[np.ix_(idx_source, idx_target)])) /
                                        corrdata[np.ix_(idx_source, idx_target)].size
                                    )

                                legendlabels[ixArea*2 + ixRed,iyArea*2 + iyRed]  = areas[ixArea] + redcelllabels[ixRed] + '-' + areas[iyArea] + redcelllabels[iyRed]

    # assuming meancorr and legeldlabels are 4x4xnSessions array
    upper_tri_indices           = np.triu_indices(4, k=0)
    meancorr_upper_tri          = meancorr[upper_tri_indices[0], upper_tri_indices[1], :]
    fraccorr_upper_tri          = fraccorr[upper_tri_indices[0], upper_tri_indices[1], :]
    
    # assuming legendlabels is a 4x4 array
    # legendlabels_upper_tri      = legendlabels[np.triu_indices(4, k=0)]
    legendlabels_upper_tri      = legendlabels[upper_tri_indices[0], upper_tri_indices[1]]

    df_mean                     = pd.DataFrame(data=meancorr_upper_tri.T,columns=legendlabels_upper_tri)
    df_frac                     = pd.DataFrame(data=fraccorr_upper_tri.T,columns=legendlabels_upper_tri)

    colorder                    = [0,1,4,7,8,9,2,3,5,6]
    legendlabels_upper_tri      = legendlabels_upper_tri[colorder]
    df_mean                     = df_mean[legendlabels_upper_tri]
    df_frac                     = df_frac[legendlabels_upper_tri]

    return df_mean,df_frac

######  ### #     #    #     # #######    #    #     #           #     # #     # 
#     #  #  ##    #    ##   ## #         # #   ##    #    #####   #   #   #   #  
#     #  #  # #   #    # # # # #        #   #  # #   #    #    #   # #     # #   
######   #  #  #  #    #  #  # #####   #     # #  #  #    #    #    #       #    
#     #  #  #   # #    #     # #       ####### #   # #    #    #   # #      #    
#     #  #  #    ##    #     # #       #     # #    ##    #    #  #   #     #    
######  ### #     #    #     # ####### #     # #     #    #####  #     #    #    

def bin_corr_deltaxy(sessions,method='mean',areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='F',
                    rotate_prefori=False,deltaori=None,noise_thr=1,onlysameplane=False,
                    binresolution=5,tuned_thr=0,absolute=False,normalize=False,dsi_thr=0,
                    filtersign=None,corr_thr=0.05):
    """
    Binning pairwise correlations as a function of pairwise delta x and y position.
    - Sessions are binned by areapairs, layerpairs, and projpairs.
    - Returns binmean,bincount,binedges

    Parameters
    ----------
    sessions : list
        list of sessions
    areapairs : list (if ' ' then all areapairs are used)
        list of areapairs
    layerpairs : list  (if ' ' then all layerpairs are used)
        list of layerpairs
    projpairs : list  (if ' ' then all projpairs are used)
        list of projpairs
    corr_type : str, optional
        type of correlation to use, by default 'trace_corr'
    normalize : bool, optional
        whether to normalize correlations to the mean correlation at distances < 60 um, by default False
    sig_thr : float, optional
        significance threshold for including cells in the analysis, by default 0.001
    """

    #Binning parameters 2D:
    binlim          = 600
    binedges_2d     = np.arange(-binlim,binlim,binresolution)+binresolution/2 
    bincenters_2d   = binedges_2d[:-1]+binresolution/2 
    nBins           = len(bincenters_2d)

    bin_2d          = np.zeros((nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_2d_count    = np.zeros((nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D distance
    binlim          = 600
    binedges_dist   = np.arange(0,binlim,binresolution)+binresolution/2 
    bincenters_dist = binedges_dist[:-1]+binresolution/2 
    nBins           = len(bincenters_dist)

    bin_dist        = np.zeros((nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_dist_count  = np.zeros((nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D angle
    polarbinres         = 45
    centerthr           = [15,15,15]
    binedges_angle      = np.deg2rad(np.arange(0-polarbinres/2,360,step=polarbinres))
    bincenters_angle    = binedges_angle[:-1]+np.deg2rad(polarbinres/2)
    npolarbins          = len(bincenters_angle)

    bin_angle_cent      = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_cent_count = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    bin_angle_surr      = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_surr_count = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()

            delta_x        = sessions[ises].celldata['xloc'].to_numpy()[:,None] - sessions[ises].celldata['xloc'].to_numpy()[None,:]
            delta_y        = sessions[ises].celldata['yloc'].to_numpy()[:,None] - sessions[ises].celldata['yloc'].to_numpy()[None,:]
            delta_xy        = np.sqrt(delta_x**2 + delta_y**2)
            angle_xy         = np.mod(np.arctan2(delta_x,delta_y)-np.pi,np.pi*2)
            angle_xy         = np.mod(angle_xy+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
            
            if absolute == True:
                corrdata = np.abs(corrdata)

            if normalize == True:
                corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)

            if method=='mean':
                if filtersign == 'neg':
                    corrsignfilter              = corrdata < 0
                elif filtersign =='pos':
                    corrsignfilter              = corrdata > 0
                else:
                    corrsignfilter = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)
            elif method=='frac':
                corrsignfilter = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)
                if filtersign == 'neg':
                    fracsignfilter              = corrdata < np.nanpercentile(corrdata,(corr_thr*100))
                elif filtersign =='pos':
                    fracsignfilter              = corrdata > np.nanpercentile(corrdata,(100-corr_thr*100))
                else:
                    raise ValueError('filtersign must be either pos or neg if metohd==frac is chosen')
            else: 
                raise ValueError('invalid method to apply to bins')

            # Rotate delta azimuth and delta elevation to the pref ori of the source neuron
            # delta_az is source neurons
            if rotate_prefori: 
                for iN in range(len(sessions[ises].celldata)):
                    ori_rots            = sessions[ises].celldata['pref_ori'][iN]
                    ori_rots            = np.tile(sessions[ises].celldata['pref_ori'][iN],len(sessions[ises].celldata))
                    angle_vec           = np.vstack((delta_x[iN,:], delta_y[iN,:]))
                    angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots + 90) #90 degrees is added to make collinear horizontal
                    delta_x[iN,:]      = angle_vec_rot[0,:]
                    delta_y[iN,:]      = angle_vec_rot[1,:]

            if onlysameplane:
                planefilter    = np.meshgrid(sessions[ises].celldata['plane_idx'],sessions[ises].celldata['plane_idx'])
                planefilter    = planefilter[0] == planefilter[1]
            else:
                planefilter    = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            for iap,areapair in enumerate(areapairs):
                for ilp,layerpair in enumerate(layerpairs):
                    for ipp,projpair in enumerate(projpairs):
                        signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
                        signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                        if tuned_thr:
                            tuningfilter    = np.meshgrid(sessions[ises].celldata['tuning_var']>tuned_thr,sessions[ises].celldata['tuning_var']>tuned_thr)
                            tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
                        else: 
                            tuningfilter    = np.ones(np.shape(signalfilter))

                        areafilter      = filter_2d_areapair(sessions[ises],areapair)

                        layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                        projfilter      = filter_2d_projpair(sessions[ises],projpair)

                        nanfilter       = np.all((~np.isnan(corrdata),~np.isnan(delta_xy)),axis=0)

                        if deltaori is not None:
                            if isinstance(deltaori,(float,int)):
                                deltaori = np.array([deltaori,deltaori])
                            if np.shape(deltaori) == (1,):
                                deltaori = np.tile(deltaori,2)
                            assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
                            delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
                            delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
                            deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
                                                    delta_pref <= deltaori[1]),axis=0)
                        else:
                            deltaorifilter = np.ones(np.shape(signalfilter)).astype(bool)

                        if dsi_thr:
                            dsi_filter = np.meshgrid(sessions[ises].celldata['DSI']>dsi_thr,sessions[ises].celldata['DSI']>dsi_thr)
                            dsi_filter = np.logical_and(dsi_filter[0],dsi_filter[1])
                        else:
                            dsi_filter = np.ones(np.shape(signalfilter)).astype(bool)

                        #Combine all filters into a single filter:
                        cellfilter      = np.all((signalfilter,tuningfilter,areafilter,corrsignfilter,
                                            layerfilter,projfilter,nanfilter,deltaorifilter,dsi_filter),axis=0)

                        if np.any(cellfilter):
                            # valuedata are the correlation values, these are going to be binned
                            vdata           = corrdata[cellfilter].flatten()

                            #First 2D binning: x is elevation, y is azimuth, 
                            xdata               = delta_x[cellfilter].flatten()
                            ydata               = delta_y[cellfilter].flatten()
                            
                            #Take the sum of the correlations in each bin:
                            if method == 'mean': 
                                bin_2d[:,:,iap,ilp,ipp]   += binned_statistic_2d(x=xdata, y=ydata, values=vdata,bins=binedges_2d, statistic='sum')[0]
                            elif method == 'frac':
                                bin_2d[:,:,iap,ilp,ipp]   += np.histogram2d(x=delta_x[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), 
                                        y=delta_y[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), bins=binedges_2d)[0]                                       

                            # Count how many correlation observations are in each bin:
                            bin_2d_count[:,:,iap,ilp,ipp]  += np.histogram2d(x=xdata,y=ydata,bins=binedges_2d)[0]

                            #Now 1D, so only by deltarf:
                            xdata           = delta_xy[cellfilter].flatten()
                            if method == 'mean': 
                                bin_dist[:,iap,ilp,ipp] += binned_statistic(x=xdata,values=vdata,statistic='sum', bins=binedges_dist)[0]
                            elif method == 'frac':
                                bin_dist[:,iap,ilp,ipp] += np.histogram(delta_xy[np.all((cellfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_dist)[0]
                            bin_dist_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_dist)[0]

                            #Now polar binning:
                            tempfilter      = np.all((cellfilter,delta_xy<centerthr[iap]),axis=0)
                            vdata           = corrdata[tempfilter].flatten()
                            xdata           = angle_xy[tempfilter].flatten() #x is angle of rf difference

                            if method == 'mean': 
                                if np.any(tempfilter):
                                    bin_angle_cent[:,iap,ilp,ipp]  += binned_statistic(x=xdata,values=vdata,
                                                                statistic='sum',bins=binedges_angle)[0]
                            elif method == 'frac':
                                bin_angle_cent[:,iap,ilp,ipp] += np.histogram(angle_xy[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                            bin_angle_cent_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_angle)[0]
                            
                            tempfilter      = np.all((cellfilter,delta_xy>centerthr[iap]),axis=0)
                            vdata           = corrdata[tempfilter].flatten()
                            xdata           = angle_xy[tempfilter].flatten() #x is angle of rf difference
                            
                            if method == 'mean': 
                                if np.any(tempfilter):
                                    bin_angle_surr[:,iap,ilp,ipp]  += binned_statistic(x=xdata,values=vdata,
                                                                statistic='sum',bins=binedges_angle)[0]
                            elif method == 'frac':
                                bin_angle_surr[:,iap,ilp,ipp] += np.histogram(angle_xy[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                            bin_angle_surr_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_angle)[0]
        
    # divide the total summed correlations by the number of counts in that bin to get the mean:
    bin_2d = bin_2d / bin_2d_count
    bin_dist = bin_dist / bin_dist_count
    bin_angle_cent = bin_angle_cent / bin_angle_cent_count
    bin_angle_surr = bin_angle_surr / bin_angle_surr_count

    return bincenters_2d,bin_2d,bin_2d_count,bin_dist,bin_dist_count,bincenters_dist,bin_angle_cent,bin_angle_cent_count,bin_angle_surr,bin_angle_surr_count,bincenters_angle


def bin_corr_distance(sessions,areapairs,corr_type='trace_corr',normalize=False):
    binedges = np.arange(0,1000,20) 
    nbins= len(binedges)-1
    binmean = np.full((len(sessions),len(areapairs),nbins),np.nan)
    for ises in tqdm(range(len(sessions)),desc= 'Computing pairwise correlations across antom. distance: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            # corrdata[corrdata<0] = np.nan
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
    if len(areapairs)==1:
        clrs_areapairs = [clrs_areapairs]
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

######  ### #     #    ######  ####### #       #######    #       ######  ####### 
#     #  #  ##    #    #     # #       #          #      # #      #     # #       
#     #  #  # #   #    #     # #       #          #     #   #     #     # #       
######   #  #  #  #    #     # #####   #          #    #     #    ######  #####   
#     #  #  #   # #    #     # #       #          #    #######    #   #   #       
#     #  #  #    ##    #     # #       #          #    #     #    #    #  #       
######  ### #     #    ######  ####### #######    #    #     #    #     # #       


def bin_corr_deltarf(sessions,method='mean',areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='Fsmooth',
                    r2_thr=0.2,noise_thr=100,filternear=False,binresolution=5,tuned_thr=0,absolute=False,
                    normalize=False,dsi_thr=0,min_dist=15,filtersign=None,corr_thr=0.05,
                    rotate_prefori=False,deltaori=None,centerori=None,surroundori=None):
    """
    Binning pairwise correlations as a function of pairwise delta azimuth and elevation.
    - Sessions are binned by areapairs, layerpairs, and projpairs.
    - Returns binmean,bincount,binedges

    Parameters
    ----------
    sessions : list
        list of sessions
    areapairs : list (if ' ' then all areapairs are used)
        list of areapairs
    layerpairs : list  (if ' ' then all layerpairs are used)
        list of layerpairs
    projpairs : list  (if ' ' then all projpairs are used)
        list of projpairs
    corr_type : str, optional
        type of correlation to use, by default 'trace_corr'
    normalize : bool, optional
        whether to normalize correlations to the mean correlation at distances < 60 um, by default False
    rf_type : str, optional
        type of receptive field to use, by default 'F'
    """

    #Binning parameters 2D:
    binlim          = 75
    binedges_2d     = np.arange(-binlim,binlim,binresolution)+binresolution/2 
    bincenters_2d   = binedges_2d[:-1]+binresolution/2 
    nBins           = len(bincenters_2d)

    bin_2d          = np.zeros((nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_2d_count    = np.zeros((nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D distance
    binlim          = 75
    binedges_dist   = np.arange(-binresolution/2,binlim,binresolution)+binresolution/2 
    bincenters_dist = binedges_dist[:-1]+binresolution/2 
    nBins           = len(bincenters_dist)

    bin_dist        = np.zeros((nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_dist_count  = np.zeros((nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D angle
    polarbinres         = 22.5
    binedges_angle      = np.deg2rad(np.arange(0-polarbinres/2,360,step=polarbinres))
    bincenters_angle    = binedges_angle[:-1]+np.deg2rad(polarbinres/2)
    npolarbins          = len(bincenters_angle)

    centerthr           = [15,15,15,15]
    bin_angle_cent      = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_cent_count = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    bin_angle_surr      = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_surr_count = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            if 'rf_r2_' + rf_type in sessions[ises].celldata:

                source_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                target_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                delta_el        = source_el[:,None] - target_el[None,:]

                source_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                target_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                delta_az        = source_az[:,None] - target_az[None,:]

                delta_rf        = np.sqrt(delta_az**2 + delta_el**2)
                angle_rf        = np.mod(np.arctan2(delta_el,delta_az)-np.pi,np.pi*2)
                angle_rf        = np.mod(angle_rf+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
                
                # Careful definitions:
                # delta_az is source neurons azimuth minus target neurons azimuth position:
                # plt.imshow(delta_az[:10,:10],vmin=-20,vmax=20,cmap='bwr')
                # entry delta_az[0,1] being positive means target neuron RF is to the right of source neuron
                # entry delta_el[0,1] being positive means target neuron RF is above source neuron
                # To rotate azimuth and elevation to relative to the preferred orientation of the source neuron
                # means that for a neuron with preferred orientation 45 deg all delta az and delta el of paired neruons
                # will rotate 45 deg, such that now delta azimuth and delta elevation is relative to the angle 
                # of pref ori of the source neuron 

                if absolute:
                    corrdata = np.abs(corrdata)

                if normalize:
                    corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)

                if corr_type == 'trace_corr':
                    n = len(sessions[ises].ts_F)
                elif corr_type in ['noise_corr','sig_corr']:
                    n = np.shape(sessions[ises].respmat)[1]
                sigcorrdata = corrdata.copy()

                if method=='mean':
                    if filtersign == 'neg':
                        # corrsignfilter              = corrdata < -0.1
                        # corrsignfilter              = corrdata < np.nanpercentile(corrdata,(corr_thr*100))
                        corrsignfilter              = filter_corr_p(sigcorrdata,n,p_thr=corr_thr) < 0
                    elif filtersign =='pos':
                        # corrsignfilter              = corrdata > 0.3
                        # corrsignfilter              = corrdata > np.nanpercentile(corrdata,(100-corr_thr*100))
                        corrsignfilter              = filter_corr_p(sigcorrdata,n,p_thr=corr_thr) > 0
                    else:
                        corrsignfilter = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)
                elif method=='frac':
                    corrsignfilter = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)
                    if filtersign == 'neg':
                        # fracsignfilter              = corrdata < np.nanpercentile(corrdata,(corr_thr*100))
                        # fracsignfilter              = corrdata < -0.15
                        fracsignfilter              = filter_corr_p(sigcorrdata,n,p_thr=corr_thr) < 0
                    elif filtersign =='pos':
                        # fracsignfilter              = corrdata > np.nanpercentile(corrdata,(100-corr_thr*100))
                        # fracsignfilter              = corrdata > 0.3
                        fracsignfilter              = filter_corr_p(sigcorrdata,n,p_thr=corr_thr) > 0
                    else:
                        raise ValueError('filtersign must be either pos or neg if metohd==frac is chosen')
                else: 
                    raise ValueError('invalid method to apply to bins')

                if filternear:
                    nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                    nearfilter      = np.meshgrid(nearfilter,nearfilter)
                    nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
                else: 
                    nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

                # Rotate delta azimuth and delta elevation to the pref ori of the source neuron
                # delta_az is source neurons
                if rotate_prefori: 
                    for iN in range(len(sessions[ises].celldata)):
                        # ori_rots            = sessions[ises].celldata['pref_ori'][iN]
                        ori_rots            = np.tile(sessions[ises].celldata['pref_ori'][iN],len(sessions[ises].celldata))
                        angle_vec           = np.vstack((delta_el[iN,:], delta_az[iN,:]))
                        angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots) 
                        # angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots + 90) #90 degrees is added to make collinear horizontal, incorrect
                        delta_el[iN,:]      = angle_vec_rot[0,:]
                        delta_az[iN,:]      = angle_vec_rot[1,:]

                    delta_rf         = np.sqrt(delta_az**2 + delta_el**2)
                    angle_rf         = np.mod(np.arctan2(delta_el,delta_az)-np.pi,np.pi*2)
                    angle_rf         = np.mod(angle_rf+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
                    # plt.hist(angle_rf.flatten())

                # plt.scatter(angle_rf_b[sessions[ises].celldata['pref_ori']==90,:].flatten(),angle_rf[sessions[ises].celldata['pref_ori']==90,:].flatten())

                rffilter        = np.meshgrid(sessions[ises].celldata['rf_r2_' + rf_type]> r2_thr,sessions[ises].celldata['rf_r2_'  + rf_type] > r2_thr)
                rffilter        = np.logical_and(rffilter[0],rffilter[1])
                
                signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
                signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                if tuned_thr:
                    tuningfilter    = np.meshgrid(sessions[ises].celldata['tuning_var']>tuned_thr,sessions[ises].celldata['tuning_var']>tuned_thr)
                    tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
                else: 
                    tuningfilter    = np.ones(np.shape(rffilter))

                nanfilter       = np.all((~np.isnan(corrdata),~np.isnan(delta_rf)),axis=0)

                proxfilter      = ~(sessions[ises].distmat_xy<min_dist)

                assert sum([deltaori is not None, centerori is not None, surroundori is not None]) <= 1, 'at maximum one of deltaori, centerori, or surroundori can be not None'
                
                if centerori is not None:
                    centerorifilter = np.tile(sessions[ises].celldata['pref_ori']== centerori,(len(sessions[ises].celldata),1)).T
                else:
                    centerorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if surroundori is not None:
                    surroundorifilter = np.tile(sessions[ises].celldata['pref_ori']== surroundori,(len(sessions[ises].celldata),1))
                else:
                    surroundorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if deltaori is not None:
                    if isinstance(deltaori,(float,int)):
                        deltaori = np.array([deltaori,deltaori])
                    if np.shape(deltaori) == (1,):
                        deltaori = np.tile(deltaori,2)
                    assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
                    delta_pref = sessions[ises].delta_pref 
                    # delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
                    # delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
                    deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
                                            delta_pref <= deltaori[1]),axis=0)
                else:
                    deltaorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if dsi_thr:
                    dsi_filter = np.meshgrid(sessions[ises].celldata['DSI']>dsi_thr,sessions[ises].celldata['DSI']>dsi_thr)
                    dsi_filter = np.logical_and(dsi_filter[0],dsi_filter[1])
                else:
                    dsi_filter = np.ones(np.shape(rffilter)).astype(bool)

                for iap,areapair in enumerate(areapairs):
                    for ilp,layerpair in enumerate(layerpairs):
                        for ipp,projpair in enumerate(projpairs):

                            areafilter      = filter_2d_areapair(sessions[ises],areapair)

                            layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                            projfilter      = filter_2d_projpair(sessions[ises],projpair)

                            #Combine all filters into a single filter:
                            cellfilter      = np.all((rffilter,signalfilter,tuningfilter,areafilter,nearfilter,corrsignfilter,
                                                layerfilter,projfilter,proxfilter,nanfilter,
                                                deltaorifilter,dsi_filter,centerorifilter,surroundorifilter),axis=0)

                            if np.any(cellfilter):
                                # valuedata are the correlation values, these are going to be binned
                                vdata               = corrdata[cellfilter].flatten()

                                #First 2D binning: x is elevation, y is azimuth, 
                                xdata               = delta_el[cellfilter].flatten()
                                ydata               = delta_az[cellfilter].flatten()
                                
                                #Take the sum of the correlations in each bin:
                                if method == 'mean': 
                                    bin_2d[:,:,iap,ilp,ipp]   += binned_statistic_2d(x=xdata, y=ydata, values=vdata,bins=binedges_2d, statistic='sum')[0]
                                elif method == 'frac':
                                    bin_2d[:,:,iap,ilp,ipp]   += np.histogram2d(x=delta_el[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), 
                                            y=delta_az[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), bins=binedges_2d)[0]                                       

                                # Count how many correlation observations are in each bin:
                                bin_2d_count[:,:,iap,ilp,ipp]  += np.histogram2d(x=xdata,y=ydata,bins=binedges_2d)[0]

                                #Now 1D, so only by deltarf:
                                xdata           = delta_rf[cellfilter].flatten()
                                if method == 'mean': 
                                    bin_dist[:,iap,ilp,ipp] += binned_statistic(x=xdata,values=vdata,statistic='sum', bins=binedges_dist)[0]
                                elif method == 'frac':
                                    bin_dist[:,iap,ilp,ipp] += np.histogram(delta_rf[np.all((cellfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_dist)[0]
                                bin_dist_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_dist)[0]

                                #Now polar binning:
                                tempfilter      = np.all((cellfilter,delta_rf<centerthr[iap]),axis=0)
                                vdata           = corrdata[tempfilter].flatten()
                                xdata           = angle_rf[tempfilter].flatten() #x is angle of rf difference

                                if method == 'mean': 
                                    if np.any(tempfilter):
                                        bin_angle_cent[:,iap,ilp,ipp]  += binned_statistic(x=xdata,values=vdata,
                                                                    statistic='sum',bins=binedges_angle)[0]
                                elif method == 'frac':
                                    bin_angle_cent[:,iap,ilp,ipp] += np.histogram(angle_rf[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                                bin_angle_cent_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_angle)[0]
                                
                                tempfilter      = np.all((cellfilter,delta_rf>centerthr[iap]),axis=0)
                                vdata           = corrdata[tempfilter].flatten()
                                xdata           = angle_rf[tempfilter].flatten() #x is angle of rf difference
                                
                                if method == 'mean': 
                                    if np.any(tempfilter):
                                        bin_angle_surr[:,iap,ilp,ipp]  += binned_statistic(x=xdata,values=vdata,
                                                                    statistic='sum',bins=binedges_angle)[0]
                                elif method == 'frac':
                                    bin_angle_surr[:,iap,ilp,ipp] += np.histogram(angle_rf[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                                bin_angle_surr_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_angle)[0]
        
    # divide the total summed correlations by the number of counts in that bin to get the mean:
    bin_2d = bin_2d / bin_2d_count
    bin_dist = bin_dist / bin_dist_count
    bin_angle_cent = bin_angle_cent / bin_angle_cent_count
    bin_angle_surr = bin_angle_surr / bin_angle_surr_count

    return bincenters_2d,bin_2d,bin_2d_count,bin_dist,bin_dist_count,bincenters_dist,bin_angle_cent,bin_angle_cent_count,bin_angle_surr,bin_angle_surr_count,bincenters_angle



######  #       ####### #######    ######  ####### #       #######    #       ######  ####### 
#     # #       #     #    #       #     # #       #          #      # #      #     # #       
#     # #       #     #    #       #     # #       #          #     #   #     #     # #       
######  #       #     #    #       #     # #####   #          #    #     #    ######  #####   
#       #       #     #    #       #     # #       #          #    #######    #   #   #       
#       #       #     #    #       #     # #       #          #    #     #    #    #  #       
#       ####### #######    #       ######  ####### #######    #    #     #    #     # #       


def plot_corr_radial_tuning_areas(bincenters_dist,bin_dist_count,bin_dist_mean,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    if np.max(bincenters_dist)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        xylim               = 65
        dim12label = 'RF (\N{DEGREE SIGN})'

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    fig,ax    = plt.subplots(1,1,figsize=(3,3))
    ilp = 0
    ipp = 0
    handles = []
    for iap,areapair in enumerate(areapairs):
        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.3
        handles.append(shaded_error(x=bincenters_dist,y=bin_dist_mean[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
                        ax = ax,color=clrs_areapairs[iap],label=areapair))
        bindata = bin_dist_mean[:,iap,ilp,ipp]
        xdata = bincenters_dist[(~np.isnan(bindata)) & (bincenters_dist<=60)]
        ydata = bindata[(~np.isnan(bindata)) & (bincenters_dist<=60)]
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(xdata,ydata)
            ax.plot(xdata, intercept + slope*xdata,linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} linfit',linewidth=1)
            
            # popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[ydata[-1]-ydata[0], ydata[-1]-ydata[0], ydata[-1]],bounds=(-10, 10))
            # ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} fit',linewidth=1)
        except:
            print('curve_fit failed for %s' % (areapair))
            continue
        
    ax.legend(handles=handles,labels=areapairs,frameon=False)
    ax.set_xlim([0,xylim])
    ax.set_ylim([my_floor(np.min(bin_dist_mean)*0.65,3),my_ceil(np.max(bin_dist_mean)*1.1,3)])
    ax.set_xlabel(u'Δ %s' % dim12label)   
    # ax.set_title('%s\n Joint' % (areapair),c=clrs_areapairs[iap])
    ax.set_ylabel('Correlation')

    plt.tight_layout()
    return fig

def plot_corr_radial_tuning_projs(bincenters_dist,bin_dist_count,bin_dist_mean,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    if np.max(bincenters_dist)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        xylim               = 70
        dim12label = 'RF (\N{DEGREE SIGN})'

    areapair = 'V1-PM'
    areaprojpairs = projpairs.copy()
    for ipp,projpair in enumerate(projpairs):
        areaprojpairs[ipp]       = areapair.split('-')[0] + projpair.split('-')[0] + '-' + areapair.split('-')[1] + projpair.split('-')[1]

    clrs_projpairs      = get_clr_labelpairs(projpairs)
    if len(projpairs)==1:
        clrs_projpairs =[clrs_projpairs]

    fig,ax    = plt.subplots(1,1,figsize=(4,3))
    ilp = 0
    iap = 0
    handles = []
    for ipp,projpair in enumerate(projpairs):
        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.4
        handles.append(shaded_error(x=bincenters_dist,y=bin_dist_mean[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
                        ax = ax,color=clrs_projpairs[ipp],label=projpair))
        bindata = bin_dist_mean[:,iap,ilp,ipp]
        xdata = bincenters_dist[(~np.isnan(bindata)) & (bincenters_dist<=60)]
        ydata = bindata[(~np.isnan(bindata)) & (bincenters_dist<=60)]
        # try:
        #     popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[0.02, 0, 0.02],bounds=(-10, 10))
        #     ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_projpairs[ipp],label=f'{areapair} fit',linewidth=1)
        # except:
        #     print('curve_fit failed for %s' % (projpair))
        #     continue
    ax.legend(handles=handles,labels=areaprojpairs,frameon=False,bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=7)
    ax.set_xlim([0,xylim])
    ax.set_xlabel(u'Δ %s' % dim12label)   
    # ax.set_title('%s\n Joint' % (areapair),c=clrs_projpairs[iap])
    ax.set_ylabel('Correlation')

    plt.tight_layout(rect=(0,0,1,1))
    return fig



def plot_corr_radial_tuning_dori(bincenters_dist,bin_dist_count,bin_dist_data,deltaoris,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.3
    
    if np.max(bincenters_dist)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        xylim               = 65
        dim12label = 'RF (\N{DEGREE SIGN})'

    ndeltaoris = len(deltaoris)
    clrs_deltaoris      = get_clr_deltaoris(deltaoris)

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    # fig,axes    = plt.subplots(len(areapairs),ndeltaoris,figsize=(len(areapairs)*3,ndeltaoris*3))
    fig,axes    = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*3,3))
    ilp = 0
    ipp = 0
    for iap,areapair in enumerate(areapairs):
        ax = axes[iap]
        handles = []
        for idOri,dOri in enumerate(deltaoris):
            handles.append(shaded_error(x=bincenters_dist,y=bin_dist_data[idOri,:,iap,ilp,ipp],yerror=bin_dist_error[idOri,:,iap,ilp,ipp],
                            ax = ax,color=clrs_deltaoris[idOri],label=areapair))
                            # ax = ax,color=clrs_areapairs[iap],label=areapair))
            bindata = bin_dist_data[idOri,:,iap,ilp,ipp]
            xdata = bincenters_dist[(~np.isnan(bindata)) & (bincenters_dist<=60)]
            ydata = bindata[(~np.isnan(bindata)) & (bincenters_dist<=60)]
            # try:
            #     slope, intercept, r_value, p_value, std_err = linregress(xdata,ydata)
            #     # ax.plot(xdata, intercept + slope*xdata,linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} linfit',linewidth=1)
                
            #     # popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[ydata[-1]-ydata[0], ydata[-1]-ydata[0], ydata[-1]],bounds=(-10, 10))
            #     # ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} fit',linewidth=1)
            # except:
            #     print('curve_fit failed for %s' % (areapair))
            #     continue
        
        ax.legend(handles=handles,labels=[str(x) for x in deltaoris],frameon=False,ncol=3,fontsize=6)
        ax.set_xlim([0,xylim])
        ax.set_ylim([my_floor(np.min(bin_dist_data)*0.65,3),my_ceil(np.max(bin_dist_data)*1.1,3)])
        ax.set_xlabel(u'Δ %s' % dim12label)   
        ax.set_title('%s' % (areapair),c=clrs_areapairs[iap])
        ax.set_ylabel('Correlation')

    plt.tight_layout()
    return fig

def plot_mean_frac_corr_areas(bincenters_2d,bin_2d_count,bin_2d_mean,bin_2d_posf,bin_2d_negf,
                            bincenters_dist,bin_dist_count,bin_dist_mean,bin_dist_posf,bin_dist_negf,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    delta_x,delta_y   = np.meshgrid(bincenters_2d,bincenters_2d)

    min_counts          = 200

    if np.max(bincenters_2d)>100:
        xylim               = 250
        gaussian_sigma      = 3
        dim1label = 'X (um)'
        dim2label = 'Y (um)'
        dim12label = 'XY (um)'
    else:
        xylim               = 70
        gaussian_sigma      = 2
        dim1label = ' Azimuth (\N{DEGREE SIGN})'
        dim2label = ' Elevation (\N{DEGREE SIGN})'
        dim12label = 'RF (\N{DEGREE SIGN})'

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    cmaps = ['hot','Reds_r','Blues_r']
    idata_labels = ['Mean','Pos','Neg']
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    fig,axes    = plt.subplots(len(areapairs),4,figsize=(12,len(areapairs)*3))
    ilp = 0
    ipp = 0
    for iap,areapair in enumerate(areapairs):
        for idata,data in enumerate([bin_2d_mean,bin_2d_posf,bin_2d_negf]):
            ax = axes[iap,idata]
    # for ilp,layerpair in enumerate(layerpairs):
        # for ipp,projpair in enumerate(projpairs):
            data                                            = copy.deepcopy(data[:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bin_2d_count[:,:,iap,ilp,ipp]<min_counts]     = np.nan
            ax.pcolor(delta_x,delta_y,data,vmin=np.nanpercentile(data,20),vmax=np.nanpercentile(data,99),cmap=cmaps[idata])
            ax.set_facecolor('grey')
            ax.set_title('%s\n%s' % (areapair, idata_labels[idata]),c=clrs_areapairs[iap])
            ax.set_xlim([-xylim,xylim])
            ax.set_ylim([-xylim,xylim])
            ax.set_xlabel(u'Δ %s' % dim1label)
            ax.set_ylabel(u'Δ %s' % dim2label)
        ax = axes[iap,3]

        ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('fraction', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)

        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.3
        data_pos_error = np.sqrt(bin_dist_posf*(1-bin_dist_posf)/bin_dist_count) * 2.576 #99% CI
        data_neg_error = np.sqrt(bin_dist_negf*(1-bin_dist_negf)/bin_dist_count) * 2.576 #99% CI
        
        shaded_error(x=bincenters_dist,y=bin_dist_mean[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
                    ax = ax,color='k',label='mean')
        shaded_error(x=bincenters_dist,y=bin_dist_posf[:,iap,ilp,ipp],yerror=data_pos_error[:,iap,ilp,ipp],
                    ax = ax2,color='r',label='pos')
        shaded_error(x=bincenters_dist,y=bin_dist_negf[:,iap,ilp,ipp],yerror=data_neg_error[:,iap,ilp,ipp],
                    ax = ax2,color='b',label='neg')
        ax.legend(frameon=False)
        ax.set_xlim([0,xylim])
        ax.set_xlabel(u'Δ %s' % dim12label)   
        ax.set_title('%s\n Joint' % (areapair),c=clrs_areapairs[iap])
        ax.set_ylabel('correlation')

    plt.tight_layout()
    return fig

def plot_mean_frac_corr_projs(bincenters_2d,bin_2d_count,bin_2d_mean,bin_2d_posf,bin_2d_negf,
                            bincenters_dist,bin_dist_count,bin_dist_mean,bin_dist_posf,bin_dist_negf,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    delta_x,delta_y   = np.meshgrid(bincenters_2d,bincenters_2d)

    min_counts          = 200

    if np.max(bincenters_2d)>100:
        xylim               = 250
        gaussian_sigma      = 3
        dim1label = 'X (um)'
        dim2label = 'Y (um)'
        dim12label = 'XY (um)'
    else:
        xylim               = 70
        gaussian_sigma      = 2
        dim1label = ' Azimuth (\N{DEGREE SIGN})'
        dim2label = ' Elevation (\N{DEGREE SIGN})'
        dim12label = 'RF (\N{DEGREE SIGN})'


    clrs_projpairs      = get_clr_labelpairs(projpairs)
    if len(projpairs)==1:
        clrs_projpairs =[clrs_projpairs]
    cmaps = ['hot','Reds_r','Blues_r']
    idata_labels = ['Mean','Pos','Neg']
   

    fig,axes    = plt.subplots(len(projpairs),4,figsize=(12,len(projpairs)*3))
    ilp = 0
    iap = 0
    # for iap,areapair in enumerate(areapairs):
    for ipp,projpair in enumerate(projpairs):
        for idata,data in enumerate([bin_2d_mean,bin_2d_posf,bin_2d_negf]):
            ax = axes[ipp,idata]
    # for ilp,layerpair in enumerate(layerpairs):
            data                                            = copy.deepcopy(data[:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bin_2d_count[:,:,iap,ilp,ipp]<min_counts]     = np.nan
            ax.pcolor(delta_x,delta_y,data,vmin=np.nanpercentile(data,20),vmax=np.nanpercentile(data,99),cmap=cmaps[idata])

            # ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,10),vmax=np.nanpercentile(data,95),cmap="crest")
            ax.set_facecolor('grey')
            ax.set_title('%s\n%s' % (projpair, idata_labels[idata]),c=clrs_projpairs[ipp])
            ax.set_xlim([-xylim,xylim])
            ax.set_ylim([-xylim,xylim])
            ax.set_xlabel(u'Δ %s' % dim1label)
            ax.set_ylabel(u'Δ %s' % dim2label)
        ax = axes[ipp,3]

        ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('fraction', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)

        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.3
        data_pos_error = np.sqrt(bin_dist_posf*(1-bin_dist_posf)/bin_dist_count) * 2.576 #99% CI
        data_neg_error = np.sqrt(bin_dist_negf*(1-bin_dist_negf)/bin_dist_count) * 2.576 #99% CI
        
        shaded_error(x=bincenters_dist,y=bin_dist_mean[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
                    ax = ax,color='k',label='mean')
        shaded_error(x=bincenters_dist,y=bin_dist_posf[:,iap,ilp,ipp],yerror=data_pos_error[:,iap,ilp,ipp],
                    ax = ax2,color='r',label='pos')
        shaded_error(x=bincenters_dist,y=bin_dist_negf[:,iap,ilp,ipp],yerror=data_neg_error[:,iap,ilp,ipp],
                    ax = ax2,color='b',label='neg')

        # ax.plot(bincenters_dist,bin_dist_mean[:,iap,ilp,ipp],color='k',label='mean')
        # ax.plot(bincenters_dist,bin_dist_posf[:,iap,ilp,ipp],color='r',label='pos')
        # ax.plot(bincenters_dist,bin_dist_negf[:,iap,ilp,ipp],color='b',label='neg')
        ax.legend(frameon=False)
        ax.set_xlim([0,xylim])
        ax.set_xlabel(u'Δ %s' % dim12label)   
        ax.set_title('%s\n Joint' % (projpair),c=clrs_projpairs[ipp])
        ax.set_ylabel('correlation')
    plt.tight_layout()
    return fig



def plot_mean_corr_layers(bincenters_dist,bin_dist_count,bin_dist_mean,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    if np.max(bincenters_dist)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        xylim               = 70
        dim12label = 'RF (\N{DEGREE SIGN})'

    areapair = 'V1-PM'
    arealayerpairs = layerpairs.copy()
    for ilp,layerpair in enumerate(layerpairs):
        arealayerpairs[ilp]       = areapair.split('-')[0] + layerpair.split('-')[0] + '-' + areapair.split('-')[1] + layerpair.split('-')[1]

    clrs_layerpairs      = get_clr_layerpairs(layerpairs)
    if len(layerpairs)==1:
        clrs_layerpairs =[clrs_layerpairs]

    fig,ax    = plt.subplots(1,1,figsize=(4,3))
    ipp = 0
    iap = 0
    handles = []
    for ilp,layerpair in enumerate(layerpairs):
        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.4
        handles.append(shaded_error(x=bincenters_dist,y=bin_dist_mean[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
                        ax = ax,color=clrs_layerpairs[ilp],label=layerpair))
        bindata = bin_dist_mean[:,iap,ilp,ipp]
        xdata = bincenters_dist[(~np.isnan(bindata)) & (bincenters_dist<=60)]
        ydata = bindata[(~np.isnan(bindata)) & (bincenters_dist<=60)]
        # try:
        #     popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[0.02, 0, 0.02],bounds=(-10, 10))
        #     ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_layerpairs[ipp],label=f'{areapair} fit',linewidth=1)
        # except:
        #     print('curve_fit failed for %s' % (layerpair))
        #     continue
    ax.legend(handles=handles,labels=arealayerpairs,frameon=False,bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=7)
    ax.set_xlim([0,xylim])
    ax.set_xlabel(u'Δ %s' % dim12label)   
    # ax.set_title('%s\n Joint' % (areapair),c=clrs_layerpairs[iap])
    ax.set_ylabel('Correlation')

    plt.tight_layout(rect=(0,0,1,1))
    return fig


def plot_2D_mean_corr(bin_2d,bin_2d_count,bincenters_2d,areapairs=' ',layerpairs=' ',projpairs=' ',
                      gaussian_sigma=0.8,centerthr=[15,15,15],min_counts=50,cmap='hot'):
    #Definitions of azimuth, elevation and delta RF 2D space:
    # delta_az,delta_el   = np.meshgrid(bincenters_2d,bincenters_2d)
    delta_el,delta_az   = np.meshgrid(bincenters_2d,bincenters_2d)
    # angle_rf        = np.mod(np.arctan2(delta_el,delta_az)-np.pi,np.pi*2)

    deglim              = 60
    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    fig,axes    = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
    if len(projpairs)==1 and len(areapairs)==1:
        axes = np.array([axes])
    axes = axes.reshape(len(projpairs),len(areapairs))

    for iap,areapair in enumerate(areapairs):
        for ilp,layerpair in enumerate(layerpairs):
            for ipp,projpair in enumerate(projpairs):
                ax                                              = axes[ipp,iap]
                data                                            = copy.deepcopy(bin_2d[:,:,iap,ilp,ipp])
                data[np.isnan(data)]                            = np.nanmean(data)
                data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
                data[bin_2d_count[:,:,iap,ilp,ipp]<min_counts]     = np.nan

                # ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,10),vmax=np.nanpercentile(data,95),cmap=cmap)
                ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(bin_2d[:,:,iap,:,:],25),vmax=np.nanpercentile(bin_2d[:,:,iap,:,:],80),cmap=cmap)
                # ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,10),vmax=np.nanpercentile(data,95),cmap=cmap)
                ax.set_facecolor('grey')
                ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
                ax.set_xlim([-deglim,deglim])
                ax.set_ylim([-deglim,deglim])
                ax.set_xlabel(u'Δ deg Orthogonal')
                ax.set_ylabel(u'Δ deg Collinear')
                circle=plt.Circle((0,0),centerthr[iap], color='g', fill=False,linestyle='--',linewidth=1)
                ax.add_patch(circle)

    plt.tight_layout()
    return fig


def plot_2D_mean_corr_dori(bin_2d,bin_2d_count,bincenters_2d,deltaoris,areapairs=' ',layerpairs=' ',projpairs=' ',
                      gaussian_sigma=0.8,centerthr=[15,15,15],min_counts=50,cmap='hot'):
    #Definitions of azimuth, elevation and delta RF 2D space:
    delta_el,delta_az   = np.meshgrid(bincenters_2d,bincenters_2d)

    deglim              = 60
    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    ndeltaoris          = len(deltaoris)
    # fig,axes    = plt.subplots(ndeltaoris,len(areapairs),figsize=(len(areapairs)*2,ndeltaoris*2))
    fig,axes    = plt.subplots(len(areapairs),ndeltaoris,figsize=(ndeltaoris*2,len(areapairs)*2))
    ilp = 0
    ipp = 0 
    for iap,areapair in enumerate(areapairs):
        for idOri,deltaori in enumerate(deltaoris):
            ax                                              = axes[iap,idOri]
            data                                            = copy.deepcopy(bin_2d[idOri,:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bin_2d_count[idOri,:,:,iap,ilp,ipp]<min_counts]     = np.nan

            ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95),cmap=cmap)
            # ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(bin_2d,10),vmax=np.nanpercentile(bin_2d,95),cmap=cmap)
            ax.set_facecolor('grey')
            ax.set_title('%s-%s deg' % (areapair, deltaori),c=clrs_areapairs[iap],fontsize=10)
            ax.set_xlim([-deglim,deglim])
            ax.set_ylim([-deglim,deglim])
            ax.set_xlabel(u'Δ deg Orthogonal')
            ax.set_ylabel(u'Δ deg Collinear')
            circle=plt.Circle((0,0),centerthr[iap], color='g', fill=False,linestyle='--',linewidth=1)
            ax.add_patch(circle)

    plt.tight_layout()
    return fig

def plot_2D_mean_corr_projs_dori(bin_2d,bin_2d_count,bincenters_2d,deltaoris,areapairs=' ',layerpairs=' ',projpairs=' ',
                      gaussian_sigma=0.8,centerthr=[15,15,15],min_counts=50,cmap='hot'):
    #Definitions of azimuth, elevation and delta RF 2D space:
    delta_el,delta_az   = np.meshgrid(bincenters_2d,bincenters_2d)

    deglim              = 60
    clrs_projpairs = get_clr_labelpairs(projpairs)

    ndeltaoris          = len(deltaoris)
    fig,axes    = plt.subplots(len(projpairs),ndeltaoris,figsize=(ndeltaoris*2,len(projpairs)*2))
    ilp = 0
    iap = 0 
    for ipp,projpair in enumerate(projpairs):
        for idOri,deltaori in enumerate(deltaoris):
            ax                                              = axes[ipp,idOri]
            data                                            = copy.deepcopy(bin_2d[idOri,:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bin_2d_count[idOri,:,:,iap,ilp,ipp]<min_counts]     = np.nan

            # ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95),cmap=cmap)
            ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(bin_2d[idOri,:,:,:,:,:],15),vmax=np.nanpercentile(bin_2d[idOri,:,:,iap,ilp,:],85),cmap=cmap)
            ax.set_facecolor('grey')
            ax.set_title('%s-%s deg' % (projpair, deltaori),c=clrs_projpairs[ipp],fontsize=10)
            ax.set_xlim([-deglim,deglim])
            ax.set_ylim([-deglim,deglim])
            ax.set_xlabel(u'Δ deg Orthogonal')
            ax.set_ylabel(u'Δ deg Collinear')
            circle=plt.Circle((0,0),centerthr[iap], color='g', fill=False,linestyle='--',linewidth=1)
            ax.add_patch(circle)

    plt.tight_layout()
    return fig


def plot_csi_deltaori_areas(csi_mean,csi_pos,csi_neg,deltaoris,areapairs):
    fig,axes = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*2.5,2),sharex=True,sharey=True)
    clrs_areapairs = get_clr_area_pairs(areapairs)
    ilp = 0
    ipp = 0

    for iap,areapair in enumerate(areapairs):
        ax = axes[iap]
        ax.plot(deltaoris,csi_mean[:,iap,ilp,ipp],label='mean',color='k')
        ax.plot(deltaoris,csi_pos[:,iap,ilp,ipp],label='pos',color='r',linestyle='-')
        ax.plot(deltaoris,csi_neg[:,iap,ilp,ipp],label='neg',color='b',linestyle='-')
        ax.set_xticks(deltaoris[::2])
        ax.set_xlabel('Delta Ori (deg)')
        if iap==0:
            ax.set_ylabel('Angular CSI')
        # ax.set_ylim([-1,1])
        ax.set_ylim([-0.5,0.5])
        # ax.set_xticks(deltaoris[::2])
        ax.axhline(0,linestyle='--',color='k',linewidth=1)
        l = ax.legend(frameon=False,loc='upper right',fontsize=7,ncol=3,handlelength=0,handletextpad=0)
        for i,text in enumerate(l.get_texts()):
            text.set_color(ax.lines[i].get_color())
        ax.set_title(areapair,fontsize=11,color=clrs_areapairs[iap])
        plt.tight_layout()
    return fig



def plot_csi_deltaori_projs(csi_mean,csi_pos,csi_neg,deltaoris,projpairs):
    fig,axes = plt.subplots(1,len(projpairs),figsize=(len(projpairs)*2.5,2),sharex=True,sharey=True)
    iap = 0
    ilp = 0
    clrs_projpairs = get_clr_labelpairs(projpairs)

    for ipp,projpair in enumerate(projpairs):
        ax = axes[ipp]
        ax.plot(deltaoris,csi_mean[:,iap,ilp,ipp],label='mean',color='k')
        ax.plot(deltaoris,csi_pos[:,iap,ilp,ipp],label='pos',color='r',linestyle='-')
        ax.plot(deltaoris,csi_neg[:,iap,ilp,ipp],label='neg',color='b',linestyle='-')
        ax.set_xticks(deltaoris[::2])
        ax.set_xlabel('Delta Ori (deg)')
        if ipp==0:
            ax.set_ylabel('Angular CSI')
        # ax.set_ylim([-1,1])
        ax.set_ylim([-0.5,0.5])
        # ax.set_xticks(deltaoris[::2])
        ax.axhline(0,linestyle='--',color='k',linewidth=1)
        l = ax.legend(frameon=False,loc='upper right',fontsize=7,ncol=3,handlelength=0,handletextpad=0)
        for i,text in enumerate(l.get_texts()):
            text.set_color(ax.lines[i].get_color())
        ax.set_title(projpair,fontsize=11,color=clrs_projpairs[ipp])
        plt.tight_layout()
    return fig


# def plot_bin_corr_deltarf_flex(sessions,binmean,binpos,areapairs=' ',layerpairs=' ',projpairs=' ',
#                                corr_type='trace_corr',normalize=False):
#     # sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
#     # protocols = np.unique(sessiondata['protocol'])

#     # clrs_areapairs = get_clr_area_pairs(areapairs)
#     # clrs_projpairs = get_clr_labelpairs(projpairs)

#     if projpairs==' ':
#         clrs_projpairs = 'k'
#     else:
#         clrs_projpairs = get_clr_labelpairs(projpairs)

#     fig,axes = plt.subplots(len(areapairs),len(layerpairs),figsize=(3*len(layerpairs),3*len(areapairs)),sharex=True,sharey=True)
#     axes = axes.reshape(len(areapairs),len(layerpairs))
#     for iap,areapair in enumerate(areapairs):
#         for ilp,layerpair in enumerate(layerpairs):
#             ax = axes[iap,ilp]
#             handles = []

#             for ipp,projpair in enumerate(projpairs):
#                 for ises in range(len(sessions)):
#                     ax.plot(binpos,binmean[:,ises,iap,ilp,ipp].squeeze(),linewidth=0.15,color=clrs_projpairs[ipp])
#                 handles.append(shaded_error(ax=ax,x=binpos,y=binmean[:,:,iap,ilp,ipp].squeeze().T,center='mean',error='sem',color=clrs_projpairs[ipp]))

#             ax.legend(handles,projpairs,loc='upper right',frameon=False)	
#             ax.set_xlabel('Delta RF')
#             ax.set_ylabel('Correlation')
#             ax.set_xlim([-2,60])
#             ax.set_title('%s\n%s' % (areapair, layerpair))
#             # if normalize:
#             #     ax.set_ylim([-0.015,0.05])
#             # else: 
#                 # ax.set_ylim([0,0.2])
#             ax.set_aspect('auto')
#             ax.tick_params(axis='both', which='major', labelsize=8)

#     plt.tight_layout()
#     return fig

def plot_1D_corr_areas(binmean,bincounts,bincenters,areapairs=' ',layerpairs=' ',projpairs=' ',
                            min_counts = 50):

    clrs_areapairs  = get_clr_area_pairs(areapairs)

    binedges        = np.arange(0,70,5)
    bin1dcenters    = binedges[:-1] + 5/2
    handles         = []
    labels          = []

    delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
    deltarf             = np.sqrt(delta_az**2 + delta_el**2)

    ilp = 0
    ipp = 0   

    fig,axes = plt.subplots(1,1,figsize=(3,3))
    ax              = axes
    for iap,areapair in enumerate(areapairs):
        ax          = ax
        rfdata      = deltarf.flatten()
        corrdata    = binmean[:,:,iap,ilp,ipp].flatten()
        countdata   = bincounts[:,:,iap,ilp,ipp].flatten()
        nanfilter   = ~np.isnan(rfdata) & ~np.isnan(corrdata) & (countdata>min_counts)
        corrdata    = corrdata[nanfilter]
        rfdata      = rfdata[nanfilter]
        countdata   = countdata[nanfilter]
        
        if np.any(rfdata):
            bindata     = binned_statistic(x=rfdata,
                                        values= corrdata,
                                        statistic='mean', bins=binedges)[0]
            # bindata_co   = np.histogram(rfdata,bins=binedges)[0]
            # bindata_se   = binned_statistic(x=rfdata,
            #                             values= corrdata,
            #                             statistic='std', bins=binedges)[0] / np.sqrt(bindata_co)
            
            bindata_co = binned_statistic(x=rfdata,
                                        values= countdata,
                                    statistic='sum',bins=binedges)[0]
            bindata_se = np.full(bindata.shape,0.07) / bindata_co**0.3
            # polardata_err = np.full(polardata.shape,np.nanstd(getattr(sessions[ises],corr_type))) / polardata_counts**0.3

            xdata = bin1dcenters[(~np.isnan(bindata)) & (bin1dcenters<60)]
            ydata = bindata[(~np.isnan(bindata)) & (bin1dcenters<60)]
            handles.append(shaded_error(ax,x=bin1dcenters,y=bindata,yerror = bindata_se,color=clrs_areapairs[iap]))
            labels.append(f'{areapair}')           
            try:
                popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[0.2, 4, 0.11],bounds=(-5, 5))
                ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} fit',linewidth=1)
            except:
                print('curve_fit failed for %s' % (areapair))
                continue
    
    ax.set_xlim([0,50])
    yl = ax.get_ylim()
    if np.mean(yl)<0:
        ax.set_ylim([my_ceil(yl[0],2),my_floor(yl[1],2)])
    else:
        ax.set_ylim([my_floor(yl[0],2),my_ceil(yl[1],2)])
    yl = ax.get_ylim()
    ax.set_yticks(ticks=[yl[0],(yl[0]+yl[1])/2,yl[1]])
    ax.set_xlabel(u'Δ RF')
    ax.set_ylabel(u'Correlation')
    ax.legend(handles=handles,labels=labels,loc='lower right',frameon=False,fontsize=7,ncol=2)
    fig.tight_layout()
    return fig

def plot_1D_corr_areas_projs(binmean,bincounts,bincenters,
                            areapairs=' ',layerpairs=' ',projpairs=' ',
                            min_counts = 50):

    clrs_projpairs  = get_clr_labelpairs(projpairs)
    clrs_areapairs = get_clr_area_pairs(areapairs)

    binedges        = np.arange(0,70,5)
    bin1dcenters    = binedges[:-1] + 5/2

    delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
    deltarf             = np.sqrt(delta_az**2 + delta_el**2)

    ilp = 0
    ipp = 0   
    
    fig,axes        = plt.subplots(1,len(areapairs),figsize=(9,3))

    for iap,areapair in enumerate(areapairs):
        ax          = axes[iap]
        handles     = []
        labels      = []
        ilp = 0
        for ipp,projpair in enumerate(projpairs):
            ax          = ax
            rfdata      = deltarf.flatten()
            corrdata    = binmean[:,:,iap,ilp,ipp].flatten()
            countdata   = bincounts[:,:,iap,ilp,ipp].flatten()
            nanfilter   = ~np.isnan(rfdata) & ~np.isnan(corrdata) & (countdata>min_counts)
            corrdata    = corrdata[nanfilter]
            rfdata      = rfdata[nanfilter]
            countdata   = countdata[nanfilter]
            
            if np.any(rfdata):
                bindata     = binned_statistic(x=rfdata,
                                            values= corrdata,
                                            statistic='mean', bins=binedges)[0]
                # bindata_co   = np.histogram(rfdata,bins=binedges)[0]
                # bindata_se   = binned_statistic(x=rfdata,
                #                             values= corrdata,
                #                             statistic='std', bins=binedges)[0] / np.sqrt(bindata_co)
                
                bindata_co = binned_statistic(x=rfdata,
                                        values= countdata,
                                    statistic='sum',bins=binedges)[0]
                # bindata_se = np.full(bindata.shape,0.09) / bindata_co**0.25
                bindata_se = np.full(bindata.shape,0.09) / bindata_co**0.3

                xdata = binedges[:-1][(~np.isnan(bindata)) & (binedges[:-1]<60)]
                ydata = bindata[(~np.isnan(bindata)) & (binedges[:-1]<60)]
                handles.append(shaded_error(ax,x=binedges[:-1],y=bindata,yerror = bindata_se,color=clrs_projpairs[ipp]))
                labels.append(f'{areapair}\n{projpair}')
                try:
                    popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[0.2, 4, 0.11],bounds=(-5, 5))
                    ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_projpairs[ipp],label=f'{areapair} fit',linewidth=1)
                except:
                    print('curve_fit failed for %s, %s' % (areapair,projpair))
                    continue

        ax.set_xlim([0,50])
        yl = ax.get_ylim()
        if np.mean(yl)<0:
            ax.set_ylim([my_ceil(yl[0],2),my_floor(yl[1],2)])
        else:
            ax.set_ylim([my_floor(yl[0],2),my_ceil(yl[1],2)])
        yl = ax.get_ylim()
        ax.set_yticks(ticks=[yl[0],(yl[0]+yl[1])/2,yl[1]])

        ax.set_xlabel(u'Δ RF')
        ax.set_ylabel(u'Correlation')
        ax.legend(handles=handles,labels=labels,loc='lower right',frameon=False,fontsize=7,ncol=2)
        ax.set_title('%s' % (areapair),c=clrs_areapairs[iap])
    fig.tight_layout()
    return fig


def plot_corr_angular_tuning(sessions,bin_angle_data,bin_angle_count,
            bincenters_angle,areapairs,layerpairs,projpairs):
    bin_angle_err = np.full(bin_angle_count.shape,0.08) / bin_angle_count**0.35

    # Make the figure:
    deglim      = 2*np.pi
    fig,axes    = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3),sharey=True)
    if len(projpairs)==1 and len(areapairs)==1:
        axes = np.array([axes])
    axes = axes.reshape(len(projpairs),len(areapairs))
    clrs_labelpairs     = get_clr_labelpairs(projpairs)
    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]
    if len(projpairs)==1:
        clrs_labelpairs = [clrs_labelpairs]

    for iap,areapair in enumerate(areapairs):
        for ilp,layerpair in enumerate(layerpairs):
            for ipp,projpair in enumerate(projpairs):
                handles = []
                ax                                          = axes[ipp,iap]
                handles.append(shaded_error(ax=ax,x=bincenters_angle,y=bin_angle_data[:,iap,ilp,ipp],
                    yerror=bin_angle_err[:,iap,ilp,ipp],color=clrs_labelpairs[ipp]))
    
                ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
                ax.set_xlim([0,deglim])
                # ax.set_ylim([0.04,0.1])
                ax.set_xticks(np.arange(0,2*np.pi,step = np.deg2rad(45)),labels=np.arange(0,360,step = 45))
                ax.set_xlabel(u'Angle (deg)')
                ax.set_ylabel(u'Correlation')
                ax.legend(handles=handles,labels=projpairs,frameon=False,fontsize=8,loc='upper right')

    plt.tight_layout()
    return fig



def plot_corr_angular_tuning_dori(bin_angle_oris,bin_angle_count_oris,
            bincenters_angle,deltaoris,areapairs,layerpairs,projpairs):
    bin_angle_err = np.full(bin_angle_count_oris.shape,0.08) / bin_angle_count_oris**0.25

    # Make the figure:
    deglim              = 2*np.pi
    ndeltaoris          = len(deltaoris)
    fig,axes    = plt.subplots(len(areapairs),len(deltaoris),figsize=(len(deltaoris)*1.5,len(areapairs)*1),
                               sharex=True,sharey=True)
    ilp = 0
    ipp = 0
    
    clrs_deltaoris      = get_clr_deltaoris(deltaoris)
    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    for iap,areapair in enumerate(areapairs):
        for idOri,deltaori in enumerate(deltaoris):
            ax                                              = axes[iap,idOri]
            handles = []
            handles.append(shaded_error(ax=ax,x=bincenters_angle,y=bin_angle_oris[idOri,:,iap,ilp,ipp],
                    yerror=bin_angle_err[idOri,:,iap,ilp,ipp],color=clrs_deltaoris[idOri]))

            if iap==0:
                ax.set_title(u'Δ Pref = %d\N{DEGREE SIGN}' % (deltaori),c=clrs_deltaoris[idOri])
            if idOri==0:
                ax.set_ylabel('%s' % (areapair),c=clrs_areapairs[iap])
            else:     
                ax.set_ylabel('')

            ax.set_xlim([0,deglim])
            ax.set_xticks(np.arange(0,2*np.pi,step = np.deg2rad(45)),labels=np.arange(0,360,step = 45),fontsize=7)
            if idOri==np.floor(len(deltaoris)/2) and iap==len(areapairs):
                ax.set_xlabel(u'Angular surround bin (\N{DEGREE SIGN})')
            # if iap==len(areapairs):
                # ax.set_xlabel(u'Angle (deg)')
    plt.tight_layout()
    return fig


def plot_corr_angular_tuning_projs_dori(bin_angle_oris,bin_angle_count_oris,
            bincenters_angle,deltaoris,areapairs,layerpairs,projpairs):
    bin_angle_err = np.full(bin_angle_count_oris.shape,0.08) / bin_angle_count_oris**0.35

    # Make the figure:
    deglim              = 2*np.pi
    ndeltaoris          = len(deltaoris)
    fig,axes    = plt.subplots(len(projpairs),len(deltaoris),figsize=(len(deltaoris)*1.5,len(projpairs)*1),
                               sharex=True,sharey=True)
    ilp = 0
    iap = 0 

    clrs_projpairs      = get_clr_labelpairs(projpairs)
    clrs_deltaoris      = get_clr_deltaoris(deltaoris)

    for ipp,projpair in enumerate(projpairs):
        for idOri,deltaori in enumerate(deltaoris):
            ax                                              = axes[ipp,idOri]
            handles = []
            handles.append(shaded_error(ax=ax,x=bincenters_angle,y=bin_angle_oris[idOri,:,iap,ilp,ipp],
                    yerror=bin_angle_err[idOri,:,iap,ilp,ipp],color=clrs_deltaoris[idOri]))

            if iap==0:
                ax.set_title(u'Δ Pref = %d\N{DEGREE SIGN}' % (deltaori),c=clrs_deltaoris[idOri])
            if idOri==0:
                ax.set_ylabel('%s' % (projpair),c=clrs_projpairs[ipp])
            else:     
                ax.set_ylabel('')

            ax.set_xlim([0,deglim])
            ax.set_xticks(np.arange(0,2*np.pi,step = np.deg2rad(45)),labels=np.arange(0,360,step = 45),fontsize=7)
            if idOri==np.floor(len(deltaoris)/2) and iap==len(areapairs):
                ax.set_xlabel(u'Angular surround bin (\N{DEGREE SIGN})')
    plt.tight_layout()
    return fig


def plot_center_surround_corr_areas(binmean,bincenters,centerthr=15,areapairs=' ',layerpairs=' ',projpairs=' '):
    clrs_areapairs = get_clr_area_pairs(areapairs)

    data        = np.zeros((3,*np.shape(binmean)[2:]))
    data_ci     = np.zeros((3,*np.shape(binmean)[2:],2))
    ilp         = 0
    ipp         = 0

    delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
    deltarf             = np.sqrt(delta_az**2 + delta_el**2)

    fig,axes = plt.subplots(1,2,figsize=(4,3))
    ax = axes[0]
    for iap,areapair in enumerate(areapairs):
        centerdata             = binmean[np.abs(deltarf)<centerthr,iap,ilp,ipp]
        surrounddata           = binmean[(np.abs(deltarf)>=centerthr)&(np.abs(deltarf)<=50),iap,ilp,ipp]

        # centercounts           = binmean[np.abs(deltarf)<centerthr,iap,ilp,ipp]
        # surroundcounts         = binmean[(np.abs(deltarf)>=centerthr)&(np.abs(deltarf)<=50),iap,ilp,ipp]

        data[0,iap,ilp,ipp]    = np.nanmean(centerdata,axis=0)
        data[1,iap,ilp,ipp]    = np.nanmean(surrounddata,axis=0)
        data[2,iap,ilp,ipp]    = data[0,iap,ilp,ipp]/data[1,iap,ilp,ipp]
        data[2,iap,ilp,ipp]    = np.clip(data[2,iap,ilp,ipp],a_min=-1,a_max=3)

        ax.plot(np.array([1,2])+iap*0.15, [data[0,iap,ilp,ipp],data[1,iap,ilp,ipp]],
                                color=clrs_areapairs[iap],linewidth=1,alpha=0.5)
        
        # bindata_co = binned_statistic(x=rfdata,
        #                                 values= countdata,
        #                             statistic='sum',bins=binedges)[0]
        #         # bindata_se = np.full(bindata.shape,0.09) / bindata_co**0.25
        #         bindata_se = np.full(bindata.shape,0.09) / bindata_co**0.3

        data_ci[0,iap,ilp,ipp,:]  = stats.bootstrap((centerdata,),np.nanmean,n_resamples=1000,confidence_level=0.99).confidence_interval[:2]
        data_ci[1,iap,ilp,ipp,:]  = stats.bootstrap((surrounddata,),np.nanmean,n_resamples=1000,confidence_level=0.99).confidence_interval[:2]
        data_ci[2,iap,ilp,ipp,:]  = data_ci[0,iap,ilp,ipp,:] / np.flipud(data_ci[1,iap,ilp,ipp,:])

        ax.errorbar(1+iap*0.15,data[0,iap,ilp,ipp],data_ci[0,iap,ilp,ipp,1]-data[0,iap,ilp,ipp],marker='s',
                        color=clrs_areapairs[iap])
        ax.errorbar(2+iap*0.15,data[1,iap,ilp,ipp],data_ci[1,iap,ilp,ipp,1]-data[1,iap,ilp,ipp],marker='s',
                        color=clrs_areapairs[iap])
        # ax.errorbar(np.array([1,2]), [data[0,iap,ilp,ipp],data[1,iap,ilp,ipp]],
        #                         color=clrs_areapairs[iap],linewidth=1,alpha=0.5)
        # data[1,:,iap,ilp,ipp]    = np.nanmean(binmean[(np.abs(bincenters)>=centerthr)&(np.abs(bincenters)<=50),:,iap,ilp,ipp],axis=0)
       
        # data[0,:,iap,ilp,ipp]    = np.nanmean(binmean[np.abs(bincenters)<centerthr,:,iap,ilp,ipp],axis=0)
        # data[1,:,iap,ilp,ipp]    = np.nanmean(binmean[(np.abs(bincenters)>=centerthr)&(np.abs(bincenters)<=50),:,iap,ilp,ipp],axis=0)
        # data[2,:,iap,ilp,ipp]    = data[0,:,iap,ilp,ipp]/data[1,:,iap,ilp,ipp]
        # data[2,:,iap,ilp,ipp]    = np.clip(data[2,:,iap,ilp,ipp],a_min=-1,a_max=3)
        # ax.plot(np.array([1,2]), [data[0,:,iap,ilp,ipp],data[1,:,iap,ilp,ipp]],
        #                         color=clrs_areapairs[iap],linewidth=1,alpha=0.5)
    
    # ax.legend(projpairs,loc='upper right',frameon=False)	
    ax.set_xlabel('')
    ax.set_ylabel('Correlation')
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Center','Surround'])
    ax.set_aspect('auto')
    ax.tick_params(axis='both', which='major', labelsize=8)

    ax = axes[1]
    for iap,areapair in enumerate(areapairs):
        ax.scatter(iap-0.15,data[2,iap,ilp,ipp],s=8,color=clrs_areapairs[iap],marker='o',label=areapair)
        ax.errorbar(iap-0.15,data[2,iap,ilp,ipp],data_ci[2,iap,ilp,ipp,1]-data[2,iap,ilp,ipp],marker='s',
                        color=clrs_areapairs[iap])
        ax.set_ylabel('C/S Ratio')
    ax.set_xlabel('')
    ax.set_xticks(np.arange(len(areapairs)))
    ax.set_xticklabels(areapairs)
    ax.set_title('')
    ax.set_aspect('auto')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.axhline(1,linestyle='--',color='k')

    plt.tight_layout()
    return fig

def plot_center_surround_corr_areas_projs(binmean,binedges,centerthr=15,areapairs=' ',layerpairs=' ',projpairs=' '):
    clrs_areapairs = get_clr_area_pairs(areapairs)
    clrs_projpairs = get_clr_labelpairs(projpairs)

    data        = np.full((3,*np.shape(binmean)[2:]),np.nan)
    data_ci     = np.full((3,*np.shape(binmean)[2:],2),np.nan)
    # data        = np.zeros((3,*np.shape(binmean)[2:]))
    # data_ci     = np.zeros((3,*np.shape(binmean)[2:],2))
    ilp         = 0

    delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
    deltarf             = np.sqrt(delta_az**2 + delta_el**2)

    fig,axes = plt.subplots(len(areapairs),2,figsize=(4,7),sharey='col')
    for iap,areapair in enumerate(areapairs):
        ax = axes[iap,0]
        for ipp,projpair in enumerate(projpairs):

            centerdata             = binmean[np.abs(deltarf)<centerthr,iap,ilp,ipp]
            surrounddata           = binmean[(np.abs(deltarf)>=centerthr)&(np.abs(deltarf)<=50),iap,ilp,ipp]

            data[0,iap,ilp,ipp]    = np.nanmean(centerdata,axis=0)
            data[1,iap,ilp,ipp]    = np.nanmean(surrounddata,axis=0)
            data[2,iap,ilp,ipp]    = data[0,iap,ilp,ipp]/data[1,iap,ilp,ipp]
            data[2,iap,ilp,ipp]    = np.clip(data[2,iap,ilp,ipp],a_min=-1,a_max=3)

            ax.plot(np.array([1,2])+ipp*0.15, [data[0,iap,ilp,ipp],data[1,iap,ilp,ipp]],
                                    color=clrs_projpairs[ipp],linewidth=1,alpha=0.5)
            
            data_ci[0,iap,ilp,ipp,:]  = stats.bootstrap((centerdata,),np.nanmean,n_resamples=1000,confidence_level=0.99).confidence_interval[:2]
            data_ci[1,iap,ilp,ipp,:]  = stats.bootstrap((surrounddata,),np.nanmean,n_resamples=1000,confidence_level=0.99).confidence_interval[:2]
            data_ci[2,iap,ilp,ipp,:]  = data_ci[0,iap,ilp,ipp,:] / np.flipud(data_ci[1,iap,ilp,ipp,:])

            ax.errorbar(1+ipp*0.15,data[0,iap,ilp,ipp],data_ci[0,iap,ilp,ipp,1]-data[0,iap,ilp,ipp],marker='s',
                            color=clrs_projpairs[ipp])
            ax.errorbar(2+ipp*0.15,data[1,iap,ilp,ipp],data_ci[1,iap,ilp,ipp,1]-data[1,iap,ilp,ipp],marker='s',
                            color=clrs_projpairs[ipp])

            # ax.errorbar(np.array([1,2]), [data[0,iap,ilp,ipp],data[1,iap,ilp,ipp]],
            #                         color=clrs_areapairs[iap],linewidth=1,alpha=0.5)
            # data[1,:,iap,ilp,ipp]    = np.nanmean(binmean[(np.abs(bincenters)>=centerthr)&(np.abs(bincenters)<=50),:,iap,ilp,ipp],axis=0)
        
            # data[0,:,iap,ilp,ipp]    = np.nanmean(binmean[np.abs(bincenters)<centerthr,:,iap,ilp,ipp],axis=0)
            # data[1,:,iap,ilp,ipp]    = np.nanmean(binmean[(np.abs(bincenters)>=centerthr)&(np.abs(bincenters)<=50),:,iap,ilp,ipp],axis=0)
            # data[2,:,iap,ilp,ipp]    = data[0,:,iap,ilp,ipp]/data[1,:,iap,ilp,ipp]
            # data[2,:,iap,ilp,ipp]    = np.clip(data[2,:,iap,ilp,ipp],a_min=-1,a_max=3)
            # ax.plot(np.array([1,2]), [data[0,:,iap,ilp,ipp],data[1,:,iap,ilp,ipp]],
            #                         color=clrs_areapairs[iap],linewidth=1,alpha=0.5)
    
        data_ci[data_ci<0]         = 5
        # pairs = np.array([['unl-unl', 'unl-lab'],
        #                     ['unl-unl', 'lab-unl'],
        #                     ['unl-unl', 'lab-lab'],
        #                     ['unl-lab', 'lab-unl'],
        #                     ['unl-lab', 'lab-lab'],
        #                     ['lab-lab', 'lab-unl']], dtype='<U7')

        # df                  = pd.DataFrame(data=data[2,iap,ilp,:],columns=projpairs)
        # df                  = df.dropna() 

        # # pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]]
        # annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
        # annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=0.05,text_offset=0, 
        #                     line_width=0.25,comparisons_correction=None,verbose=False,
        #                     correction_format='replace',fontsize=5)
        # annotator.apply_and_annotate()

        # ax.legend(projpairs,loc='upper right',frameon=False)	
        ax.set_xlabel('')
        ax.set_ylabel('Correlation')
        ax.set_xticks([1,2])
        ax.set_xticklabels(['Center','Surround'])
        ax.set_aspect('auto')
        ax.tick_params(axis='both', which='major', labelsize=8)

        for iap,areapair in enumerate(areapairs):
            ax = axes[iap,1]
            for ipp,projpair in enumerate(projpairs):
                # ax.scatter(ipp-0.15,data[2,iap,ilp,ipp],s=8,color=clrs_projpairs[ipp],marker='o',label=areapair)
                if not np.isnan(data[2,iap,ilp,ipp]):
                    ax.errorbar(ipp-0.15,data[2,iap,ilp,ipp],data_ci[2,iap,ilp,ipp,1]-data[2,iap,ilp,ipp],marker='s',
                                    color=clrs_projpairs[ipp])
                    ax.set_ylabel('C/S Ratio')
            
            ax.set_xlabel('')
            if iap==0: 
                ax.set_ylabel('Ratio Center/Surround')
            ax.set_xticks(np.arange(len(projpairs)))
            ax.set_xticklabels(projpairs)
            ax.set_title('')
            ax.set_aspect('auto')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.axhline(1,linestyle='--',color='k')

    plt.tight_layout()
    return fig
    

def bin_2d_rangecorr_deltarf(sessions,areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='F',
                            r2_thr = 0.2,noise_thr=100,filternear=False,binres_rf=5,binres_corr=0.01,min_dist=15):
    """
    Pairwise correlations binned across range of values and as a function of pairwise delta azimuth and elevation.
    - Sessions are binned by areapairs, layerpairs, and projpairs.
    - Returns binmean,bincount,bincenters_rf,bincenters_corr

    Parameters
    ----------
    sessions : list
        list of sessions
    areapairs : list (if ' ' then all areapairs are used)
        list of areapairs
    layerpairs : list  (if ' ' then all layerpairs are used)
        list of layerpairs
    projpairs : list  (if ' ' then all projpairs are used)
        list of projpairs
    corr_type : str, optional
        type of correlation to use, by default 'trace_corr'
    normalize : bool, optional
        whether to normalize correlations to the mean correlation at distances < 60 um, by default False
    rf_type : str, optional
        type of receptive field to use, by default 'F'
    sig_thr : float, optional
        significance threshold for including cells in the analysis, by default 0.001
    """
    #Binning        parameters:
    binlim          = 100
    binedges_rf     = np.arange(0,binlim+binres_rf,binres_rf)-binres_rf/2 
    bincenters_rf   = binedges_rf[:-1]+binres_rf/2 
    nBins_rf        = len(bincenters_rf)

    #Binning        parameters:
    binlim          = 1
    binedges_corr   = np.arange(-binlim,binlim,binres_corr)+binres_corr/2 
    bincenters_corr = binedges_corr[:-1]+binres_corr/2 
    nBins_corr      = len(bincenters_corr)

    # binmean         = np.zeros((nBins_rf,nBins_corr,len(areapairs),len(layerpairs),len(projpairs)))
    bincount        = np.zeros((nBins_rf,nBins_corr,len(areapairs),len(layerpairs),len(projpairs)))

    # binmean     = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs),len(sessions)))
    # bincount    = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs),len(sessions)))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            if 'rf_r2_' + rf_type in sessions[ises].celldata:
                source_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                target_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                delta_el        = source_el[:,None] - target_el[None,:]

                source_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                target_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                delta_az        = source_az[:,None] - target_az[None,:]

                delta_rf        = np.sqrt(delta_el**2 + delta_az**2)

                # delta_rf        = sessions[ises].distmat_xy

                if filternear:
                    nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                    nearfilter      = np.meshgrid(nearfilter,nearfilter)
                    nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
                else: 
                    nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

                for iap,areapair in enumerate(areapairs):
                    for ilp,layerpair in enumerate(layerpairs):
                        for ipp,projpair in enumerate(projpairs):
                            rffilter        = np.meshgrid(sessions[ises].celldata['rf_r2_' + rf_type]>r2_thr,sessions[ises].celldata['rf_r2_'  + rf_type]> r2_thr)
                            rffilter        = np.logical_and(rffilter[0],rffilter[1])
                            
                            signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<=noise_thr,sessions[ises].celldata['noise_level']<=noise_thr)
                            signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                            areafilter      = filter_2d_areapair(sessions[ises],areapair)

                            layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                            projfilter      = filter_2d_projpair(sessions[ises],projpair)

                            nanfilter       = ~np.isnan(corrdata)

                            proxfilter      = ~(sessions[ises].distmat_xy<min_dist)
                            
                            #Combine all filters into a single filter:
                            cellfilter      = np.all((rffilter,signalfilter,areafilter,nearfilter,
                                                layerfilter,projfilter,proxfilter,nanfilter),axis=0)

                            if np.any(cellfilter):
                                
                                xdata               = delta_rf[cellfilter].flatten()
                                ydata               = corrdata[cellfilter].flatten()

                                tempfilter          = ~np.isnan(xdata) & ~np.isnan(ydata)
                                xdata               = xdata[tempfilter]
                                ydata               = ydata[tempfilter]
                                
                                # #Take the sum of the correlations in each bin:
                                # binmean[:,:,iap,ilp,ipp]   += binned_statistic_2d(x=xdata, y=ydata, values=vdata,
                                #                                                     bins=binedges, statistic='sum')[0]
                                
                                # Count how many correlation observations are in each bin:
                                bincount[:,:,iap,ilp,ipp]  += np.histogram2d(x=xdata,y=ydata,bins=[binedges_rf,binedges_corr])[0]

    return bincount,bincenters_rf,bincenters_corr

def plot_2D_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,areapairs=' ',layerpairs=' ',projpairs=' ',gaussian_sigma = 0):

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    clrs_projpairs      = get_clr_labelpairs(projpairs)
    clrs_layerpairs     = get_clr_layerpairs(layerpairs)

    assert bincounts.shape[:2] == (len(bincenters_rf),len(bincenters_corr)), "bincounts should have shape (%d,%d), but has shape %s" % (len(bincenters_rf),len(bincenters_corr),bincounts.shape)

    X,Y          = np.meshgrid(bincenters_rf,bincenters_corr)
    # X,Y          = np.meshgrid(bincenters_corr,bincenters_rf)

    data = copy.deepcopy(bincounts)
    normalize_rf = True
    if normalize_rf:
        #normalizing across the rf dimension, 
        # i.e. divide by sum across all correlation bins for each delta RF bin
        # this accounts for unequal distribution of delta RFs across pairs
        # data   = data/np.nansum(data,axis=(0,1),keepdims=True) 
        data   = data/np.nansum(data,axis=1,keepdims=True) 
        # This normalizes to the mean across all delta RFs, and layers and projections
        # I.e. the result is that values >1 mean more correlations for this correlation value and delta RF
        # than other delta RFs and other projection identities.
        # data   = data/np.nanmean(data,axis=(0,3,4),keepdims=True)
        # This normalizes to the mean across all delta RFs, and layers and projections
        data   = data/np.nanmean(data,axis=(0),keepdims=True)
    data[np.isnan(data)] = 1

    # min_counts = 0
    # Make the figure:
    fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
    if len(projpairs)==1 and len(areapairs)==1:
        axes = np.array([axes])
    axes = axes.reshape(len(projpairs),len(areapairs))

    for iap,areapair in enumerate(areapairs):
        for ilp,layerpair in enumerate(layerpairs):
            for ipp,projpair in enumerate(projpairs):
                ax                                              = axes[ipp,iap]
                
                data_cat                                        = data[:,:,iap,ilp,ipp].T
                if gaussian_sigma: 
                    data_cat[np.isnan(data_cat)]                = np.nanmean(data_cat)
                    data_cat                                    = gaussian_filter(data_cat,sigma=[gaussian_sigma,gaussian_sigma])
                # data_cat[bincounts[:,:,iap,ilp,ipp].T<min_counts]     = np.nan

                # ax.pcolor(X,Y,data_cat,vmin=np.nanpercentile(data_cat,2),vmax=np.nanpercentile(data_cat,99),cmap="seismic")
                # ax.pcolor(X,Y,data_cat,cmap="hot")
                axim = ax.pcolor(X,Y,data_cat,vmin=-1,vmax=3,cmap="seismic")
                # ax.pcolor(X,Y,np.log10(1+data_cat),vmin=np.nanpercentile(np.log10(1+data_cat),5),vmax=np.nanpercentile(np.log10(1+data_cat),95),cmap="hot")
                
                ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
                # ax.set_xlim([0,600])
                ax.set_xlim([0,70])
                ax.set_ylim([-0.75,1])
                ax.set_xlabel(u'Δ RF')
                ax.set_ylabel(u'Correlation')
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.95, 0.8, 0.025, 0.15])
    fig.colorbar(axim, cax=cbar_ax, ticks=[-1,1,3])
    cbar_ax.set_ylabel('Fold change')
    return fig


def bin_1d_fraccorr_deltarf(sessions,areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='F',
                            r2_thr=0.2,noise_thr=100,filternear=False,binres_rf=5,corr_thr=0.1,min_dist=0):
    """
    Pairwise correlations binned across range of values and as a function of pairwise delta azimuth and elevation.
    - Sessions are binned by areapairs, layerpairs, and projpairs.
    - Returns binmean,bincount,bincenters_rf,bincenters_corr

    Parameters
    ----------
    sessions : list
        list of sessions
    areapairs : list (if ' ' then all areapairs are used)
        list of areapairs
    layerpairs : list  (if ' ' then all layerpairs are used)
        list of layerpairs
    projpairs : list  (if ' ' then all projpairs are used)
        list of projpairs
    corr_type : str, optional
        type of correlation to use, by default 'trace_corr'
    normalize : bool, optional
        whether to normalize correlations to the mean correlation at distances < 60 um, by default False
    rf_type : str, optional
        type of receptive field to use, by default 'F'
    sig_thr : float, optional
        significance threshold for including cells in the analysis, by default 0.001
    """
    #Binning        parameters:
    binlim          = 100
    binedges_rf     = np.arange(0,binlim,binres_rf)-binres_rf/2 
    bincenters_rf   = binedges_rf[:-1]+binres_rf/2
    binedges_rf[-1] = 1000
    nBins_rf        = len(bincenters_rf)

    binpos          = np.zeros((nBins_rf,len(areapairs),len(layerpairs),len(projpairs)))
    binneg          = np.zeros((nBins_rf,len(areapairs),len(layerpairs),len(projpairs)))
    bincounts       = np.zeros((nBins_rf,len(areapairs),len(layerpairs),len(projpairs)))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 1D corr histograms maps: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()

            pos_thr = np.nanpercentile(corrdata,(100-corr_thr*100))
            neg_thr = np.nanpercentile(corrdata,(corr_thr*100))

            # pos_thr = 0.2
            # neg_thr = -0.075

            if 'rf_r2_' + rf_type in sessions[ises].celldata:

                source_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                target_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                delta_el        = source_el[:,None] - target_el[None,:]

                source_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                target_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                delta_az        = source_az[:,None] - target_az[None,:]

                delta_rf        = np.sqrt(delta_el**2 + delta_az**2)

                if filternear:
                    nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                    nearfilter      = np.meshgrid(nearfilter,nearfilter)
                    nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
                else: 
                    nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

                for iap,areapair in enumerate(areapairs):
                    for ilp,layerpair in enumerate(layerpairs):
                        for ipp,projpair in enumerate(projpairs):
                            rffilter        = np.meshgrid(sessions[ises].celldata['rf_r2_' + rf_type]>r2_thr,sessions[ises].celldata['rf_r2_'  + rf_type]>r2_thr)
                            rffilter        = np.logical_and(rffilter[0],rffilter[1])
                            
                            signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<=noise_thr,sessions[ises].celldata['noise_level']<=noise_thr)
                            signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                            # signalfilter    = np.meshgrid(sessions[ises].celldata['tuning_var']>0.05,sessions[ises].celldata['tuning_var']>0.05)
                            # signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                            areafilter      = filter_2d_areapair(sessions[ises],areapair)

                            layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                            projfilter      = filter_2d_projpair(sessions[ises],projpair)

                            nanfilter       = ~np.isnan(corrdata)

                            proxfilter      = ~(sessions[ises].distmat_xy<min_dist)
                            
                            #Combine all filters into a single filter:
                            cellfilter      = np.all((rffilter,signalfilter,areafilter,nearfilter,
                                                layerfilter,projfilter,proxfilter,nanfilter),axis=0)
                            
                            counts          = np.histogram(delta_rf[cellfilter],bins=binedges_rf)[0]
                            
                            poscounts       = np.histogram(delta_rf[np.all((cellfilter,corrdata>pos_thr),axis=0)],bins=binedges_rf)[0]
                            negcounts       = np.histogram(delta_rf[np.all((cellfilter,corrdata<neg_thr),axis=0)],bins=binedges_rf)[0]

                            binpos[:,iap,ilp,ipp] += poscounts # / np.sum(np.all((cellfilter,corrdata>corr_thr),axis=0))
                            binneg[:,iap,ilp,ipp] += negcounts #/ np.sum(np.all((cellfilter,corrdata<-corr_thr),axis=0))
                            bincounts[:,iap,ilp,ipp] += counts

    return bincounts,binpos,binneg,bincenters_rf

def plot_1D_fraccorr(bincounts,binpos,binneg,bincenters_rf,normalize_rf=True,mincounts=50,
            areapairs=' ',layerpairs=' ',projpairs=' '):
    """
    Plot the fraction of pairs with positive/negative correlation as a function of
    Delta RF, for each combination of area pair, layer pair, and projection pair.

    Parameters
    ----------
    bincounts : array
        2D array of shape (nBins,nBins) with the number of pairs of cells in each bin
    binpos : array
        2D array of shape (nBins,nBins) with the number of pairs of cells with positive
        correlation in each bin
    binneg : array
        2D array of shape (nBins,nBins) with the number of pairs of cells with negative
        correlation in each bin
    bincenters_rf : array
        1D array with the centers of the bins for the Delta RF axis
    normalize_rf : bool, optional
        If True, normalize the fraction of pairs with positive/negative correlation
        by the total number of pairs in each bin. If False, plot the absolute counts
        instead. Default is True.
    areapairs : list of str, optional
        List of area pairs to plot. If not provided, all area pairs are plotted.
    layerpairs : list of str, optional
        List of layer pairs to plot. If not provided, all layer pairs are plotted.
    projpairs : list of str, optional
        List of projection pairs to plot. If not provided, all projection pairs are plotted.
    """

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    clrs_projpairs      = get_clr_labelpairs(projpairs)
    clrs_layerpairs     = get_clr_layerpairs(layerpairs)

    assert bincounts.shape == binpos.shape, "bincounts and binpos should have the same shape, but bincounts has shape %s and binpos has shape %s" % (bincounts.shape,binpos.shape)

    data_pos = copy.deepcopy(binpos)
    data_neg = copy.deepcopy(binneg)

    data_pos = data_pos/bincounts
    data_neg = data_neg/bincounts

    # data_pos = data_pos/np.nansum(bincounts,axis=0,keepdims=True)
    # data_neg = data_neg/np.nansum(bincounts,axis=0,keepdims=True)

    data_pos_error = np.sqrt(data_pos*(1-data_pos)/bincounts) * 1.960 #95% CI
    data_neg_error = np.sqrt(data_neg*(1-data_neg)/bincounts) * 1.960 #95% CI

    data_pos[bincounts<mincounts] = np.nan # binfrac[bincounts]
    data_neg[bincounts<mincounts] = np.nan # binfrac[bincounts]

    # data_pos[bincounts<mincounts] = 0 # binfrac[bincounts]
    # data_neg[bincounts<mincounts] = 0 # binfrac[bincounts]

    data_pos_error[bincounts<mincounts] = 0 # binfrac[bincounts]
    data_neg_error[bincounts<mincounts] = 0 # binfrac[bincounts]

    if normalize_rf:
        data_pos_error                                  = data_pos_error / np.nanmean(data_pos,axis=0,keepdims=True)
        data_neg_error                                  = data_neg_error / np.nanmean(data_neg,axis=0,keepdims=True)
        
        data_pos                                        = data_pos / np.nanmean(data_pos,axis=0,keepdims=True)
        data_neg                                        = data_neg / np.nanmean(data_neg,axis=0,keepdims=True)

    # Make the figure:
    fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3),sharex=True,sharey='col')
    if len(projpairs)==1 and len(areapairs)==1:
        axes = np.array([axes])
    axes = axes.reshape(len(projpairs),len(areapairs))

    for iap,areapair in enumerate(areapairs):
        for ilp,layerpair in enumerate(layerpairs):
            for ipp,projpair in enumerate(projpairs):
                ax                                              = axes[ipp,iap]

                shaded_error(ax=ax,x=bincenters_rf,y=data_pos[:,iap,ilp,ipp],yerror=data_pos_error[:,iap,ilp,ipp],color='r')
                shaded_error(ax=ax,x=bincenters_rf,y=data_neg[:,iap,ilp,ipp],yerror=data_neg_error[:,iap,ilp,ipp],color='b')

                com_pos = np.average(bincenters_rf, weights=np.nan_to_num(data_pos[:,iap,ilp,ipp]))
                com_neg = np.average(bincenters_rf, weights=np.nan_to_num(data_neg[:,iap,ilp,ipp]))

                # ax.plot(com_pos,1.5,'o',color='r')
                # ax.plot(com_neg,1.5,'o',color='b')
                # ax.plot(bincenters_rf,data_pos[:,iap,ilp,ipp],color='r')
                # ax.plot(bincenters_rf,-data_neg[:,iap,ilp,ipp],color='b')

                ax.axhline(1,color='k',lw=1,ls=':')
                # ax.axhline(-1,color='k',lw=1,ls=':')
                ax.axhline(0,color='k',lw=1,ls='-')
                # ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
                ax.set_title('%s\n%s *%dK pairs' % (areapair, projpair,np.sum(bincounts[:,iap,ilp,ipp]/1000)),c=clrs_areapairs[iap])
                
                # ax.set_xlim([0,600])
                ax.set_xlim([0,bincenters_rf[-1]])
                # ax.set_ylim([-2.2,2.2])
                # if normalize_rf:
                # ax.set_ylim([my_floor(np.min(-data_neg[:,iap,:,:])*1.2,1),my_ceil(np.max(data_pos[:,iap,:,:])*1.2,1)])
                ax.set_ylim([0,my_ceil(np.nanmax([np.nanmax(data_pos[:,iap,:,:]),np.nanmax(data_neg[:,iap,:,:])])*1.2,1)])
                # ax.set_ylim([0,my_ceil(np.max(np.concatenate((data_pos[:,iap,:,:],data_neg[:,iap,:,:])))*1.2,1)])
                # else:
                    # ax.set_ylim([-0.05,0.05])

                ax.set_xlabel(u'Δ RF')
                ax.set_ylabel(u'P corr / P all cells')
    plt.tight_layout()
    return fig


 #####  ######     #     #    #    ######   #####  
#     # #     #    ##   ##   # #   #     # #     # 
      # #     #    # # # #  #   #  #     # #       
 #####  #     #    #  #  # #     # ######   #####  
#       #     #    #     # ####### #             # 
#       #     #    #     # #     # #       #     # 
####### ######     #     # #     # #        #####  


# def bin_2d_corr_deltarf(sessions,areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='F',
#                             rotate_prefori=False,deltaori=None,sig_thr = 0.001,noise_thr=1,filternear=False,
#                             binresolution=5,tuned_thr=0,absolute=False,normalize=False,dsi_thr=0,
#                             min_dist=15,filtersign=None):
#     """
#     Average pairwise correlations as a function of pairwise delta azimuth and elevation.
#     - Sessions are binned by areapairs, layerpairs, and projpairs.
#     - Returns binmean,bincount,binedges

#     Parameters
#     ----------
#     sessions : list
#         list of sessions
#     areapairs : list (if ' ' then all areapairs are used)
#         list of areapairs
#     layerpairs : list  (if ' ' then all layerpairs are used)
#         list of layerpairs
#     projpairs : list  (if ' ' then all projpairs are used)
#         list of projpairs
#     corr_type : str, optional
#         type of correlation to use, by default 'trace_corr'
#     normalize : bool, optional
#         whether to normalize correlations to the mean correlation at distances < 60 um, by default False
#     rf_type : str, optional
#         type of receptive field to use, by default 'F'
#     sig_thr : float, optional
#         significance threshold for including cells in the analysis, by default 0.001
#     """
#     #Binning        parameters:
#     binlim          = 75
#     binedges        = np.arange(-binlim,binlim,binresolution)+binresolution/2 
#     bincenters      = binedges[:-1]+binresolution/2 
#     nBins           = [len(binedges)-1,len(binedges)-1]

#     binmean         = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs)))
#     bincount        = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs)))

#     # binmean     = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs),len(sessions)))
#     # bincount    = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs),len(sessions)))

#     for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):
#         if hasattr(sessions[ises],corr_type):
#             corrdata = getattr(sessions[ises],corr_type).copy()
#             if 'rf_p_' + rf_type in sessions[ises].celldata:

#                 source_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
#                 target_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
#                 delta_el        = source_el[:,None] - target_el[None,:]

#                 source_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
#                 target_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
#                 delta_az        = source_az[:,None] - target_az[None,:]

#                 # Careful definitions:
#                 # delta_az is source neurons azimuth minus target neurons azimuth position:
#                 # plt.imshow(delta_az[:10,:10],vmin=-20,vmax=20,cmap='bwr')
#                 # entry delta_az[0,1] being positive means target neuron RF is to the right of source neuron
#                 # entry delta_el[0,1] being positive means target neuron RF is above source neuron
#                 # To rotate azimuth and elevation to relative to the preferred orientation of the source neuron
#                 # means that for a neuron with preferred orientation 45 deg all delta az and delta el of paired neruons
#                 # will rotate 45 deg, such that now delta azimuth and delta elevation is relative to the angle 
#                 # of pref ori of the source neuron 
                
#                 if absolute == True:
#                     corrdata = np.abs(corrdata)

#                 if normalize == True:
#                     corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)

#                 if filtersign == 'neg':
#                     corrdata[corrdata>0] = np.nan
                
#                 if filtersign =='pos':
#                     corrdata[corrdata<0] = np.nan

#                 if filternear:
#                     nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
#                     nearfilter      = np.meshgrid(nearfilter,nearfilter)
#                     nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
#                 else: 
#                     nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

#                 # Rotate delta azimuth and delta elevation to the pref ori of the source neuron
#                 # delta_az is source neurons
#                 if rotate_prefori: 
#                     for iN in range(len(sessions[ises].celldata)):
#                         ori_rots            = sessions[ises].celldata['pref_ori'][iN]
#                         ori_rots            = np.tile(sessions[ises].celldata['pref_ori'][iN],len(sessions[ises].celldata))
#                         angle_vec           = np.vstack((delta_el[iN,:], delta_az[iN,:]))
#                         angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots + 90) #90 degrees is added to make collinear horizontal
#                         delta_el[iN,:]      = angle_vec_rot[0,:]
#                         delta_az[iN,:]      = angle_vec_rot[1,:]

#                 for iap,areapair in enumerate(areapairs):
#                     for ilp,layerpair in enumerate(layerpairs):
#                         for ipp,projpair in enumerate(projpairs):
#                             rffilter        = np.meshgrid(sessions[ises].celldata['rf_p_' + rf_type]<sig_thr,sessions[ises].celldata['rf_p_'  + rf_type]<sig_thr)
#                             rffilter        = np.logical_and(rffilter[0],rffilter[1])
                            
#                             signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
#                             signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

#                             if tuned_thr:
#                                 tuningfilter    = np.meshgrid(sessions[ises].celldata['tuning_var']>tuned_thr,sessions[ises].celldata['tuning_var']>tuned_thr)
#                                 tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
#                             else: 
#                                 tuningfilter    = np.ones(np.shape(rffilter))

#                             areafilter      = filter_2d_areapair(sessions[ises],areapair)

#                             layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

#                             projfilter      = filter_2d_projpair(sessions[ises],projpair)

#                             nanfilter       = ~np.isnan(corrdata)

#                             proxfilter      = ~(sessions[ises].distmat_xy<min_dist)
                            
#                             if deltaori is not None:
#                                 assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
#                                 delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
#                                 delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
#                                 deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
#                                                         delta_pref <= deltaori[1]),axis=0)
#                             else:
#                                 deltaorifilter = np.ones(np.shape(rffilter))

#                             if dsi_thr:
#                                 dsi_filter = np.meshgrid(sessions[ises].celldata['DSI']>dsi_thr,sessions[ises].celldata['DSI']>dsi_thr)
#                                 dsi_filter = np.logical_and(dsi_filter[0],dsi_filter[1])
#                             else:
#                                 dsi_filter = np.ones(np.shape(rffilter))

#                             #Combine all filters into a single filter:
#                             cellfilter      = np.all((rffilter,signalfilter,tuningfilter,areafilter,nearfilter,
#                                                 layerfilter,projfilter,proxfilter,nanfilter,deltaorifilter,dsi_filter),axis=0)

#                             if np.any(cellfilter):
                                
#                                 xdata               = delta_el[cellfilter].flatten()
#                                 ydata               = delta_az[cellfilter].flatten()
#                                 vdata               = corrdata[cellfilter].flatten()

#                                 tempfilter          = ~np.isnan(xdata) & ~np.isnan(ydata) & ~np.isnan(vdata)
#                                 xdata               = xdata[tempfilter]
#                                 ydata               = ydata[tempfilter]
#                                 vdata               = vdata[tempfilter]
                                
#                                 #Take the sum of the correlations in each bin:
#                                 binmean[:,:,iap,ilp,ipp]   += binned_statistic_2d(x=xdata, y=ydata, values=vdata,
#                                                                                     bins=binedges, statistic='sum')[0]
                                
#                                 # Count how many correlation observations are in each bin:
#                                 bincount[:,:,iap,ilp,ipp]  += np.histogram2d(x=xdata,y=ydata,bins=binedges)[0]

#                                 # binmean[:,:,iap,ilp,ipp,ises]   += binmean_temp
#                                 # bincount[:,:,iap,ilp,ipp,ises]  += bincount_temp

#     # import scipy.stats as st 

#     # # Confidence Interval = x(+/-)t*(s/√n)
#     # # create 95% confidence interval
#     # numsamples  = np.unique(bincount[~np.isnan(bincount)])
#     # binci       = np.empty((*np.shape(binmean),2))
    
#     # for ns in numsamples: #ns = 10
#     #     st.t.interval(alpha=0.95, df=len(ns)-1, 
#     #             loc=np.nanmean(binmean[:]), 
#     #             scale=st.sem(binmean[:])) 
    
#     # divide the total summed correlations by the number of counts in that bin to get the mean:
#     binmean = binmean / bincount

#     return binmean,bincount,bincenters

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


def plot_noise_corr_deltaori(ses):
    fig,ax = plt.subplots(1,1,figsize=(5,5))

    tuning_perc_labels = np.linspace(0,100,11)
    tuning_percentiles  = np.percentile(ses.celldata['tuning_var'],tuning_perc_labels)
    clrs_percentiles    = sns.color_palette('inferno', len(tuning_percentiles))

    for ip in range(len(tuning_percentiles)-1):

        filter_tuning = np.logical_and(tuning_percentiles[ip] <= ses.celldata['tuning_var'],
                                ses.celldata['tuning_var'] <= tuning_percentiles[ip+1])

        df = pd.DataFrame({'NoiseCorrelation':  ses.noise_corr[filter_tuning,:].flatten(),
                        'DeltaPrefOri':  ses.delta_pref[filter_tuning,:].flatten()}).dropna(how='all')

        deltapreforis = np.sort(df['DeltaPrefOri'].unique())
        histdata            = df.groupby(['DeltaPrefOri'], as_index=False)['NoiseCorrelation'].mean()

        plt.plot(histdata['DeltaPrefOri'], 
                histdata['NoiseCorrelation'],
                color=clrs_percentiles[ip])
        
    plt.xlabel('Delta Pref. Ori')
    plt.ylabel('NoiseCorrelation')
            
    plt.legend(tuning_perc_labels[1:],fontsize=9,loc='best')
    plt.tight_layout()
    return fig



# def bin_corr_deltarf_old(sessions,areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='trace_corr',noise_thr=1,
#                      filtersign=None,normalize=False,rf_type = 'F',sig_thr = 0.001,
#                      binres=5,mincount=25,absolute=False,filternear=False):
#     """
#     Compute pairwise correlations as a function of pairwise delta receptive field.
    
#     Parameters
#     ----------
#     sessions : list
#         list of sessions
#     areapairs : list (if ' ' then all areapairs are used)
#         list of areapairs
#     layerpairs : list  (if ' ' then all layerpairs are used)
#         list of layerpairs
#     projpairs : list  (if ' ' then all projpairs are used)
#         list of projpairs
#     corr_type : str, optional
#         type of correlation to use, by default 'trace_corr'
#     normalize : bool, optional
#         whether to normalize correlations to the mean correlation at distances < 60 um, by default False
#     rf_type : str, optional
#         type of receptive field to use, by default 'F'
#     sig_thr : float, optional
#         significance threshold for including cells in the analysis, by default 0.001
#     mincount : int, optional
#         minimum number of cell pairs required in a bin, by default 25
#     absolute : bool, optional
#         whether to take the absolute value of the correlations, by default False
    
#     Returns
#     -------
#     binmean : 5D array
#         mean correlation for each bin (nbins x nSessions x nAreapairs x nLayerpairs x nProjpairs)
#     binpos : 1D array
#         bin positions
#     """

#     if binres == 'centersurround':
#         binedges    = np.array([0,15,50])
#     else: 
#         assert isinstance(binres,int), 'binres type error'
#         binedges    = np.arange(0,120,binres)

#     binpos      = [np.mean(binedges[i:i+2]) for i in range(0, len(binedges)-1)]# find the mean of consecutive bins 
#     nbins       = len(binpos)
#     binmean     = np.full((nbins,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
#     bincount    = np.full((nbins,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    
#     for ises in tqdm(range(len(sessions)),desc= 'Binning correlations by delta receptive field: '):
#         if hasattr(sessions[ises],corr_type):
#             corrdata = getattr(sessions[ises],corr_type).copy()

#             if absolute == True:
#                 corrdata = np.abs(corrdata)

#             if filtersign == 'neg':
#                 corrdata[corrdata>0] = np.nan
            
#             if filtersign =='pos':
#                 corrdata[corrdata<0] = np.nan

#             if filternear:
#                 nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
#                 nearfilter      = np.meshgrid(nearfilter,nearfilter)
#                 nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
#             else: 
#                 nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

#             if 'rf_p_' + rf_type in sessions[ises].celldata:
#                 delta_rf        = np.linalg.norm(sessions[ises].celldata[['rf_az_' + rf_type,'rf_el_' + rf_type]].to_numpy()[None,:] - sessions[ises].celldata[['rf_az_' + rf_type,'rf_el_' + rf_type]].to_numpy()[:,None],axis=2)

#                 for iap,areapair in enumerate(areapairs):
#                     for ilp,layerpair in enumerate(layerpairs):
#                         for ipp,projpair in enumerate(projpairs):
#                             rffilter    = np.meshgrid(sessions[ises].celldata['rf_p_' + rf_type]<sig_thr,sessions[ises].celldata['rf_p_'  + rf_type]<sig_thr)
#                             rffilter    = np.logical_and(rffilter[0],rffilter[1])
                            
#                             signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
#                             signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

#                             areafilter      = filter_2d_areapair(sessions[ises],areapair)

#                             layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

#                             projfilter      = filter_2d_projpair(sessions[ises],projpair)

#                             nanfilter       = ~np.isnan(corrdata)

#                             proxfilter      = ~(sessions[ises].distmat_xy<15)

#                             cellfilter      = np.all((rffilter,signalfilter,areafilter,layerfilter,
#                                                     projfilter,proxfilter,nanfilter,nearfilter),axis=0)
                            
#                             if np.any(cellfilter):
#                                 xdata           = delta_rf[cellfilter].flatten()
#                                 # xdata           = sessions[ises].distmat_rf[cellfilter].flatten()
                                
#                                 vdata      = corrdata[cellfilter].flatten()
#                                 tempfilter      = ~np.isnan(xdata) & ~np.isnan(vdata)
#                                 xdata           = xdata[tempfilter]
#                                 vdata      = vdata[tempfilter]
                                
#                                 binmean[:,ises,iap,ilp,ipp] = binned_statistic(x=xdata,
#                                                                                 values=vdata,
#                                                                                 statistic='mean', bins=binedges)[0]
#                                 bincount[:,ises,iap,ilp,ipp] = binned_statistic(x=xdata,
#                                                                                 values=vdata,
#                                                                                 statistic='count', bins=binedges)[0]
                        
#     binmean[bincount<mincount] = np.nan
#     if normalize: # subtract mean correlation from every session:
#         binmean = binmean - np.nanmean(binmean[binedges[:-1]<60,:,:,:,:],axis=0,keepdims=True)

#     # binmean = binmean.squeeze()
#     return binmean,binpos

# def compute_NC_map(sourcecells,targetcells,NC_data,nBins,binrange,
#                    rotate_prefori=False,rf_type='F'):

#     noiseRFmat          = np.zeros(nBins)
#     countsRFmat         = np.zeros(nBins)

#     for iN in range(len(sourcecells)):
#         delta_el    = targetcells['rf_el_' + rf_type] - sourcecells['rf_el_' + rf_type][iN]
#         delta_az    = targetcells['rf_az_' + rf_type] - sourcecells['rf_az_' + rf_type][iN]
#         angle_vec   = np.vstack((delta_el, delta_az))

#         # if rotate_deltaprefori:
#         #     ori_rots    = targetcells['pref_ori'] - sourcecells['pref_ori'][iN]
#         #     angle_vec   = apply_ori_rot(angle_vec,ori_rots)
        
#         if rotate_prefori:
#             ori_rots   = np.tile(sourcecells['pref_ori'][iN],len(targetcells))
#             angle_vec  = apply_ori_rot(angle_vec,ori_rots)

#         idx_notnan      = ~np.isnan(angle_vec[0,:]) & ~np.isnan(angle_vec[1,:]) & ~np.isnan(NC_data[iN, :])
#         noiseRFmat       = noiseRFmat + binned_statistic_2d(x=angle_vec[0,idx_notnan],y=angle_vec[1,idx_notnan],
#                         values = NC_data[iN, idx_notnan],
#                         bins=nBins,range=binrange,statistic='sum')[0]
        
#         countsRFmat      = countsRFmat + np.histogram2d(x=angle_vec[0,idx_notnan],y=angle_vec[1,idx_notnan],
#                         bins=nBins,range=binrange)[0]
            
#     return noiseRFmat,countsRFmat

# def compute_noisecorr_rfmap_v2(sessions,binresolution=5,rotate_prefori=False,splitareas=False,splitlabeled=False):
#     # Computes the average noise correlation depending on the difference in receptive field between the two neurons
#     # binresolution determines spatial bins in degrees visual angle
#     # If rotate_prefori=True then the delta RF is rotated depending on their 
#     # This means that the output axis are now collinear vs orthogonal instead of azimuth and elevation
    
#     if rotate_prefori:
#         binrange        = np.array([[-135, 135],[-135, 135]])
#         nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
#     else: 
#         binrange        = np.array([[-50, 50],[-135, 135]])
#         nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
    
#     celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#     if splitareas is not None:
#         areas = np.sort(np.unique(celldata['roi_name']))[::-1]

#     if splitlabeled is not None:
#         redcells            = [0,1]
#         redcelllabels       = ['unl','lab']
    
#     # legendlabels        = np.empty((4,4),dtype='object')
#     # noiseRFmat          = np.zeros((4,4,*nBins))
#     # countsRFmat         = np.zeros((4,4,*nBins))

#     rotate_prefori = True

#     noiseRFmat          = np.zeros(nBins)
#     countsRFmat         = np.zeros(nBins)

#     if rotate_prefori:
#         oris            = np.sort(sessions[0].trialdata['Orientation'].unique())
#         rotation_matrix_oris = np.empty((2,2,len(oris)))
#         for iori,ori in enumerate(oris):
#             c, s = np.cos(np.radians(ori)), np.sin(np.radians(ori))
#             rotation_matrix_oris[:,:,iori] = np.array(((c, -s), (s, c)))


#     for ises in range(len(sessions)):
#         print('computing 2d receptive field hist of noise correlations for session %d / %d' % (ises+1,len(sessions)))
#         nNeurons    = len(sessions[ises].celldata) #number of neurons in this session
#         idx_RF      = ~np.isnan(sessions[ises].celldata['rf_az_Fneu']) #get all neurons with RF

#         # for iN in range(nNeurons):
#         for iN in range(100):
#         # for iN in range(100):
#             if idx_RF[iN]:
#                 idx = np.logical_and(idx_RF, range(nNeurons) != iN)

#                 delta_el = sessions[ises].celldata['rf_el_Fneu'] - sessions[ises].celldata['rf_el_Fneu'][iN]
#                 delta_az = sessions[ises].celldata['rf_az_Fneu'] - sessions[ises].celldata['rf_az_Fneu'][iN]

#                 angle_vec = np.vstack((delta_el, delta_az))
#                 if rotate_prefori:
#                     for iori,ori in enumerate(oris):
#                         ori_diff = np.mod(sessions[ises].celldata['pref_ori'] - sessions[ises].celldata['pref_ori'][iN],360)
#                         idx_ori = ori_diff ==ori

#                         angle_vec[:,idx_ori] = rotation_matrix_oris[:,:,iori] @ angle_vec[:,idx_ori]

#                 noiseRFmat       = noiseRFmat + binned_statistic_2d(x=angle_vec[0,idx],y=angle_vec[1,idx],
#                                 values = sessions[ises].noise_corr[iN, idx],
#                                 bins=nBins,range=binrange,statistic='sum')[0]
                
#                 countsRFmat      = countsRFmat + np.histogram2d(x=angle_vec[0,idx],y=angle_vec[1,idx],
#                                 bins=nBins,range=binrange)[0]
    
#     # divide the total summed noise correlations by the number of counts in that bin to get the mean:
#     noiseRFmat_mean = noiseRFmat / countsRFmat 
    
#     return noiseRFmat_mean,countsRFmat,binrange




# def noisecorr_rfmap_areas(sessions,corr_type='noise_corr',binresolution=5,rotate_prefori=False,
#                             rotate_deltaprefori=False,thr_tuned=0,thr_rf_p=1,rf_type='F'):

#     areas               = ['V1','PM']

#     if rotate_prefori:
#         binrange        = np.array([[-135, 135],[-135, 135]])
#         nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
#     else: 
#         binrange        = np.array([[-50, 50],[-135, 135]])
#         nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
  
#     noiseRFmat          = np.zeros((2,2,*nBins))
#     countsRFmat         = np.zeros((2,2,*nBins))

#     for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D noise corr histograms maps: '):
#         if 'rf_az_' + rf_type in sessions[ises].celldata and hasattr(sessions[ises], corr_type):
#             for ixArea,xArea in enumerate(areas):
#                 for iyArea,yArea in enumerate(areas):

#                     idx_source = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF
#                     idx_target = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF

#                     if thr_tuned:
#                         idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>thr_tuned)
#                         idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>thr_tuned)
                    
#                     if thr_rf_p<1:
#                         idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
#                         idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
                    
#                     idx_source = np.logical_and(idx_source,sessions[ises].celldata['roi_name']==xArea)
#                     idx_target = np.logical_and(idx_target,sessions[ises].celldata['roi_name']==yArea)

#                     corrdata = getattr(sessions[ises], corr_type)

#                     [noiseRFmat_temp,countsRFmat_temp] = compute_NC_map(sourcecells = sessions[ises].celldata[idx_source].reset_index(drop=True),
#                                                             targetcells = sessions[ises].celldata[idx_target].reset_index(drop=True),
#                                                             NC_data = corrdata[np.ix_(idx_source, idx_target)],
#                                                             nBins=nBins,binrange=binrange,
#                                                             rotate_deltaprefori=rotate_deltaprefori, rotate_prefori=rotate_prefori)
#                     noiseRFmat[ixArea,iyArea,:,:]  += noiseRFmat_temp
#                     countsRFmat[ixArea,iyArea,:,:] += countsRFmat_temp

#     # divide the total summed noise correlations by the number of counts in that bin to get the mean:
#     noiseRFmat_mean = noiseRFmat / countsRFmat 
    
#     return noiseRFmat_mean,countsRFmat,binrange

# def noisecorr_rfmap_perori(sessions,corr_type='noise_corr',binresolution=5,rotate_prefori=False,rotate_deltaprefori=False,
#                            thr_tuned=0,thr_rf_p=1,rf_type='F'):
#     """
#     Computes the average noise correlation depending on 
#     azimuth and elevation
#         Parameters:
#     sessions (list of Session objects)
#     binresolution (int, default=5)
#     rotate_prefori (bool, default=False)
#     rotate_deltaprefori (bool, default=False)
#     thr_tuned (float, default=0)
#     thr_rf_p (float, default=1)
#     corr_type (str, default='distmat_rf')
#         Type of correlation data to use. Can be one of:
#             - 'noise_corr'
#             - 'trace_corr'
#             - 'sig_corr'
    
#     Returns:
#     noiseRFmat_mean, countsRFmat, binrange
#     """
#     # Computes the average noise correlation depending on the difference in receptive field between the two neurons
#     # binresolution determines spatial bins in degrees visual angle
#     # If rotate_prefori=True then the delta RF is rotated depending on the preferred orientation of the source neuron 
#     # This means that the output axis are now collinear vs orthogonal instead of azimuth and elevation
    
#     oris = np.sort(sessions[0].trialdata['Orientation'].unique())
#     nOris = len(oris)

#     if rotate_prefori:
#         binrange        = np.array([[-135, 135],[-135, 135]])
#         nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
#     else: 
#         binrange        = np.array([[-50, 50],[-135, 135]])
#         nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
  
#     noiseRFmat          = np.zeros((nOris,*nBins))
#     countsRFmat         = np.zeros((nOris,*nBins))

#     for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D noise corr histograms maps: '):
#         if 'rf_az_' + rf_type in sessions[ises].celldata:
#             for iOri,Ori in enumerate(oris):

#                 idx_source = np.logical_and(~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]),
#                                             sessions[ises].celldata['pref_ori']==Ori)#get all neurons with RF
#                 idx_target = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type],
#                                             sessions[ises].celldata['pref_ori'].between(Ori-30, Ori+30)) #get all neurons with RF

#                 if thr_tuned:
#                     idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>thr_tuned)
#                     idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>thr_tuned)
                    
#                 if thr_rf_p<1:
#                     idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_Fneu']<thr_rf_p)
#                     idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_Fneu']<thr_rf_p)
                
#                 corrdata = getattr(sessions[ises], corr_type)

#                 [noiseRFmat_ses,countsRFmat_ses] = compute_NC_map(sourcecells = sessions[ises].celldata[idx_source].reset_index(drop=True),
#                                                         targetcells = sessions[ises].celldata[idx_target].reset_index(drop=True),
#                                                         NC_data = corrdata[np.ix_(idx_source, idx_target)],
#                                                         nBins=nBins,binrange=binrange,
#                                                         rotate_prefori=rotate_prefori)
#                 noiseRFmat[iOri,:,:]      += noiseRFmat_ses
#                 countsRFmat[iOri,:,:]     += countsRFmat_ses

#     # divide the total summed noise correlations by the number of counts in that bin to get the mean:
#     noiseRFmat_mean = noiseRFmat / countsRFmat 
    
#     return noiseRFmat_mean,countsRFmat,binrange


# def noisecorr_rfmap(sessions,corr_type='noise_corr',binresolution=5,rotate_prefori=False,rotate_deltaprefori=False,
#                     thr_tuned=0,thr_rf_p=1,rf_type='F'):
#     # Computes the average noise correlation depending on the difference in receptive field between the two neurons
#     # binresolution determines spatial bins in degrees visual angle
#     # If rotate_prefori=True then the delta RF is rotated depending on the preferred orientation of the source neuron 
#     # This means that the output axis are now collinear vs orthogonal instead of azimuth and elevation
    
#     if rotate_prefori:
#         binrange        = np.array([[-135, 135],[-135, 135]])
#         nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
#     else: 
#         binrange        = np.array([[-50, 50],[-135, 135]])
#         nBins           = np.array([(binrange[0,1] - binrange[0,0]) / binresolution,(binrange[1,1] - binrange[1,0]) / binresolution]).astype(int)
  
#     noiseRFmat          = np.zeros(nBins)
#     countsRFmat         = np.zeros(nBins)

#     for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D noise corr histograms maps: '):
        
#         if 'rf_az_' + rf_type in sessions[ises].celldata:
#             idx_source = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF
#             idx_target = ~np.isnan(sessions[ises].celldata['rf_az_' + rf_type]) #get all neurons with RF

#             if thr_tuned:
#                 idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>thr_tuned)
#                 idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>thr_tuned)
            
#             if thr_rf_p<1:
#                 if 'rf_p' in sessions[ises].celldata:
#                     idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
#                     idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_' + rf_type]<thr_rf_p)
                    
#             corrdata = getattr(sessions[ises], corr_type)

#             [noiseRFmat_ses,countsRFmat_ses] = compute_NC_map(sourcecells = sessions[ises].celldata[idx_source].reset_index(drop=True),
#                                                     targetcells = sessions[ises].celldata[idx_target].reset_index(drop=True),
#                                                     NC_data = corrdata[np.ix_(idx_source, idx_target)],
#                                                     nBins=nBins,binrange=binrange,
#                                                     rotate_deltaprefori=rotate_deltaprefori, rotate_prefori=rotate_prefori)
#             noiseRFmat      += noiseRFmat_ses
#             countsRFmat     += countsRFmat_ses

#     # divide the total summed noise correlations by the number of counts in that bin to get the mean:
#     noiseRFmat_mean = noiseRFmat / countsRFmat 
    
#     return noiseRFmat_mean,countsRFmat,binrange



# idx_source      = np.any(cellfilter,axis=1)
#                                 idx_target      = np.any(cellfilter,axis=0)

#                                 if deltaori is not None:
#                                     assert np.shape(deltaori) == (1,),'deltaori must be a 1 array'
#                                     oris    = np.sort(sessions[ises].trialdata['Orientation'].unique())


#                                     angle_vec   = np.vstack((delta_el, delta_az))
#                                     if rotate_prefori:
#                                         xdata               = delta_el.copy()
#                                         ydata               = delta_az.copy()
#                                         for iN in range(len(sessions[ises].celldata)):

#                                             #Rotate pref ori
#                                             ori_rots            = sessions[ises].celldata['pref_ori'][iN]
#                                             ori_rots            = np.tile(sessions[ises].celldata['pref_ori'][iN],len(sessions[ises].celldata))
#                                             angle_vec           = np.vstack((delta_el[iN,:], delta_az[iN,:]))
#                                             angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots)
#                                             xdata[iN,:]         = angle_vec_rot[0,:]
#                                             ydata[iN,:]         = angle_vec_rot[1,:]
                                            
#                                     np.mod(sessions[ises].delta_pref,90)<deltaori
#                                     for iOri,Ori in enumerate(oris):

#                                         idx_source_ori = np.all((idx_source,sessions[ises].celldata['pref_ori']==Ori),axis=0) 
#                                         idx_target_ori = np.all((idx_target,
#                                                                 np.mod(sessions[ises].celldata['pref_ori']+deltaori[0],180)>=np.mod(Ori,180),
#                                                                 np.mod(sessions[ises].celldata['pref_ori']+deltaori[1],180)<=np.mod(Ori,180)),axis=0) 





#                                         [binmean_temp,bincount_temp] = compute_NC_map(sourcecells = sessions[ises].celldata[idx_source_ori].reset_index(drop=True),
#                                                                             targetcells = sessions[ises].celldata[idx_target_ori].reset_index(drop=True),
#                                                                             NC_data = corrdata[np.ix_(idx_source_ori, idx_target_ori)],
#                                                                             nBins=nBins,binrange=binrange,rf_type=rf_type,
#                                                                             rotate_prefori=rotate_prefori)
                                        
#                                         binmean[:,:,iap,ilp,ipp]   += binmean_temp
#                                         bincount[:,:,iap,ilp,ipp]  += bincount_temp

# def plot_bin_corr_deltarf_protocols(sessions,binmean,binedges,areapairs,corr_type,normalize=False):
#     sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
#     protocols = np.unique(sessiondata['protocol'])
#     clrs_areapairs = get_clr_area_pairs(areapairs)

#     fig,axes = plt.subplots(1,len(protocols),figsize=(4*len(protocols),4))
#     handles = []
#     for iprot,protocol in enumerate(protocols):
#         sesidx = np.where(sessiondata['protocol']== protocol)[0]
#         if len(protocols)>1:
#             ax = axes[iprot]
#         else:
#             ax = axes

#         for iap,areapair in enumerate(areapairs):
#             for ises in sesidx:
#                 ax.plot(binedges[:-1],binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
#             handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[sesidx,iap,:].squeeze(),center='mean',error='sem',color=clrs_areapairs[iap]))

#         ax.legend(handles,areapairs,loc='upper right',frameon=False)	
#         ax.set_xlabel('Delta RF')
#         ax.set_ylabel('Correlation')
#         ax.set_xlim([-2,100])
#         ax.set_title('%s (%s)' % (corr_type,protocol))
#         if normalize:
#             ax.set_ylim([-0.015,0.05])
#         else: 
#             ax.set_ylim([0,0.12])
#         ax.set_aspect('auto')
#         ax.tick_params(axis='both', which='major', labelsize=8)

#     plt.tight_layout()
#     return fig
