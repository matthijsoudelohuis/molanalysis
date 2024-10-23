"""
This script contains functions to compute noise correlations
on simultaneously acquired calcium imaging data with mesoscope
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

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


 #####  ####### ######  ######  ####### #          #    ####### ### ####### #     #  #####  
#     # #     # #     # #     # #       #         # #      #     #  #     # ##    # #     # 
#       #     # #     # #     # #       #        #   #     #     #  #     # # #   # #       
#       #     # ######  ######  #####   #       #     #    #     #  #     # #  #  #  #####  
#       #     # #   #   #   #   #       #       #######    #     #  #     # #   # #       # 
#     # #     # #    #  #    #  #       #       #     #    #     #  #     # #    ## #     # 
 #####  ####### #     # #     # ####### ####### #     #    #    ### ####### #     #  #####  
import itertools
import scipy.stats as ss

def compute_trace_correlation(sessions,uppertriangular=True,binwidth=1,filtersig=False):
    nSessions = len(sessions)
    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing trace correlations: '):
    
        avg_nframes     = int(np.round(sessions[ises].sessiondata['fs'][0] * binwidth))

        arr_reduced     = block_reduce(sessions[ises].calciumdata.T, block_size=(1,avg_nframes), func=np.mean, cval=np.mean(sessions[ises].calciumdata.T))

        # pca = PCA(n_components=50)
        # pca.fit(arr_reduced.T)
        # arr_reduced = pca.transform(arr_reduced.T)
        # arr_reduced = pca.inverse_transform(arr_reduced).T

        sessions[ises].trace_corr                   = np.corrcoef(arr_reduced)

        N           = np.shape(sessions[ises].calciumdata)[1] #get dimensions of response matrix

        idx_triu    = np.tri(N,N,k=0)==1 #index only upper triangular part
        
        if uppertriangular:
            sessions[ises].trace_corr[idx_triu] = np.nan
        else:
            np.fill_diagonal(sessions[ises].trace_corr,np.nan)

        if filtersig: #set all nonsignificant to nan:
            sessions[ises].trace_corr = filter_corr_p(sessions[ises].trace_corr,
                                                        np.shape(arr_reduced)[1],p_thr=0.01)

        if not filtersig:
            assert np.all(sessions[ises].trace_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].trace_corr[~idx_triu] < 1)
    return sessions    

def compute_signal_noise_correlation(sessions,uppertriangular=True,filter_stationary=True,remove_method=None,remove_rank=0):
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
            resp_meanori,respmat_res        = mean_resp_gr(sessions[ises],filter_stationary=filter_stationary)
            prefori                         = oris[np.argmax(resp_meanori,axis=1)]
            # delta_pref                      = np.subtract.outer(prefori, prefori)
            sessions[ises].delta_pref       = np.abs(np.mod(np.subtract.outer(prefori, prefori),180))
            

            sessions[ises].sig_corr         = np.corrcoef(resp_meanori)
            if remove_method is not None:
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

                # fig,axes = plt.subplots(1,3,figsize=(9,3))
                # axes[0].imshow(data,aspect='auto',vmin=-0.5,vmax=0.5)
                # axes[0].set_title('Data')
                # axes[1].imshow(data_hat,aspect='auto',vmin=-0.5,vmax=0.5)
                # axes[1].set_title('Data_hat')
                # axes[2].imshow(data-data_hat,aspect='auto',vmin=-0.5,vmax=0.5)
                # axes[2].set_title('Data - Data_hat')
                # plt.suptitle('%s (rank %d)' % (remove_method,remove_rank))
                # for ax in axes:
                #     ax.set_xticks([])
                #     ax.set_yticks([])
                #     ax.set_xlabel('Trials')
                #     ax.set_ylabel('Neurons')
                # plt.tight_layout()

            # Detrend the data:
            # respmat_res = detrend(respmat_res,axis=1)

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
            resp_meanori,respmat_res        = mean_resp_gn(sessions[ises])
            prefori, prefspeed              = np.unravel_index(resp_meanori.reshape(N,-1).argmax(axis=1), (len(oris), len(speeds)))
            sessions[ises].prefori          = oris[prefori]
            sessions[ises].prefspeed        = speeds[prefspeed]

            sessions[ises].sig_corr         = np.corrcoef(resp_meanori.reshape(N,len(oris)*len(speeds)))
            
            if remove_method is not None:
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

def filter_corr_p(r,n,p_thr=0.01):
    # r           = correlation matrix
    # p_thr       = threshold for significant correlations
    # n           = number of datapoints
    t           = np.clip(r * np.sqrt((n-2)/(1-r*r)),a_min=-30,a_max=30)#convert correlation to t-statistic
    p           = ss.t.pdf(t, n-2) #convert to p-value using pdf of t-distribution and deg of freedom
    r[p>p_thr]  = np.nan #set all nonsignificant to nan
    # plt.scatter(r.flatten(),p.flatten())
    return r


def hist_corr_areas_labeling(sessions,corr_type='trace_corr',filternear=True,minNcells=10, 
                        areapairs=' ',layerpairs=' ',projpairs=' ',noise_thr=1,valuematching=None,
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

    for ises in tqdm(range(len(sessions)),desc='Averaging %s across sessions' % corr_type):
        idx_nearfilter = filter_nearlabeled(sessions[ises],radius=50)
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

            for iap,areapair in enumerate(areapairs):
                for ilp,layerpair in enumerate(layerpairs):
                    for ipp,projpair in enumerate(projpairs):
                        signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
                        signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                        areafilter      = filter_2d_areapair(sessions[ises],areapair)

                        layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                        projfilter      = filter_2d_projpair(sessions[ises],projpair)

                        nanfilter       = ~np.isnan(corrdata)

                        proxfilter      = ~(sessions[ises].distmat_xy<15)

                        cellfilter      = np.all((signalfilter,areafilter,layerfilter,matchfilter,
                                                projfilter,proxfilter,nanfilter,nearfilter),axis=0)
                        
                        if np.sum(np.any(cellfilter,axis=0))>minNcells and np.sum(np.any(cellfilter,axis=1))>minNcells:
                            
                            data      = corrdata[cellfilter].flatten()
                            data      = data[~np.isnan(data)]

                            histcorr[:,ises,iap,ilp,ipp]    = np.histogram(data,bins=binedges,density=True)[0]
                            meancorr[ises,iap,ilp,ipp]      = np.nanmean(data)
                            varcorr[ises,iap,ilp,ipp]       = np.nanstd(data)

    return bincenters,histcorr,meancorr,varcorr


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

                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['noise_level']<0.2)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['noise_level']<0.2)

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
                                    fraccorr[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.sum(~np.isnan(corrdata[np.ix_(idx_source, idx_target)])) / corrdata[np.ix_(idx_source, idx_target)].size

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

   #    #     #    #    #######    ######  ###  #####  #######    #    #     #  #####  ####### 
  # #   ##    #   # #      #       #     #  #  #     #    #      # #   ##    # #     # #       
 #   #  # #   #  #   #     #       #     #  #  #          #     #   #  # #   # #       #       
#     # #  #  # #     #    #       #     #  #   #####     #    #     # #  #  # #       #####   
####### #   # # #######    #       #     #  #        #    #    ####### #   # # #       #       
#     # #    ## #     #    #       #     #  #  #     #    #    #     # #    ## #     # #       
#     # #     # #     #    #       ######  ###  #####     #    #     # #     #  #####  ####### 

def bin_corr_distance(sessions,areapairs,corr_type='trace_corr',normalize=False):
    binedges = np.arange(0,1000,20) 
    nbins= len(binedges)-1
    binmean = np.full((len(sessions),len(areapairs),nbins),np.nan)
    for ises in tqdm(range(len(sessions)),desc= 'Computing pairwise correlations across antom. distance: '):
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

   #   ######     ######  #######  #####     ####### ### ####### #       ######  
  ##   #     #    #     # #       #     #    #        #  #       #       #     # 
 # #   #     #    #     # #       #          #        #  #       #       #     # 
   #   #     #    ######  #####   #          #####    #  #####   #       #     # 
   #   #     #    #   #   #       #          #        #  #       #       #     # 
   #   #     #    #    #  #       #     #    #        #  #       #       #     # 
 ##### ######     #     # #######  #####     #       ### ####### ####### ######  


def bin_corr_deltarf(sessions,areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='trace_corr',noise_thr=1,
                     filtersign=None,normalize=False,rf_type = 'F',sig_thr = 0.001,
                     binres=5,mincount=25,absolute=False,filternear=False):
    """
    Compute pairwise correlations as a function of pairwise delta receptive field.
    
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
    mincount : int, optional
        minimum number of cell pairs required in a bin, by default 25
    absolute : bool, optional
        whether to take the absolute value of the correlations, by default False
    
    Returns
    -------
    binmean : 5D array
        mean correlation for each bin (nbins x nSessions x nAreapairs x nLayerpairs x nProjpairs)
    binpos : 1D array
        bin positions
    """

    if binres == 'centersurround':
        binedges    = np.array([0,15,50])
    else: 
        assert isinstance(binres,int), 'binres type error'
        binedges    = np.arange(0,120,binres)

    binpos      = [np.mean(binedges[i:i+2]) for i in range(0, len(binedges)-1)]# find the mean of consecutive bins 
    nbins       = len(binpos)
    binmean     = np.full((nbins,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    bincount    = np.full((nbins,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    
    for ises in tqdm(range(len(sessions)),desc= 'Binning correlations by delta receptive field: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()

            if absolute == True:
                corrdata = np.abs(corrdata)

            if filtersign == 'neg':
                corrdata[corrdata>0] = np.nan
            
            if filtersign =='pos':
                corrdata[corrdata<0] = np.nan

            if filternear:
                nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                nearfilter      = np.meshgrid(nearfilter,nearfilter)
                nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
            else: 
                nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            if 'rf_p_' + rf_type in sessions[ises].celldata:
                delta_rf        = np.linalg.norm(sessions[ises].celldata[['rf_az_' + rf_type,'rf_el_' + rf_type]].to_numpy()[None,:] - sessions[ises].celldata[['rf_az_' + rf_type,'rf_el_' + rf_type]].to_numpy()[:,None],axis=2)

                for iap,areapair in enumerate(areapairs):
                    for ilp,layerpair in enumerate(layerpairs):
                        for ipp,projpair in enumerate(projpairs):
                            rffilter    = np.meshgrid(sessions[ises].celldata['rf_p_' + rf_type]<sig_thr,sessions[ises].celldata['rf_p_'  + rf_type]<sig_thr)
                            rffilter    = np.logical_and(rffilter[0],rffilter[1])
                            
                            signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
                            signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                            areafilter      = filter_2d_areapair(sessions[ises],areapair)

                            layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                            projfilter      = filter_2d_projpair(sessions[ises],projpair)

                            nanfilter       = ~np.isnan(corrdata)

                            proxfilter      = ~(sessions[ises].distmat_xy<15)

                            cellfilter      = np.all((rffilter,signalfilter,areafilter,layerfilter,
                                                    projfilter,proxfilter,nanfilter,nearfilter),axis=0)
                            
                            if np.any(cellfilter):
                                xdata           = delta_rf[cellfilter].flatten()
                                # xdata           = sessions[ises].distmat_rf[cellfilter].flatten()
                                
                                valueddata      = corrdata[cellfilter].flatten()
                                tempfilter      = ~np.isnan(xdata) & ~np.isnan(valueddata)
                                xdata           = xdata[tempfilter]
                                valueddata      = valueddata[tempfilter]
                                
                                binmean[:,ises,iap,ilp,ipp] = binned_statistic(x=xdata,
                                                                                values=valueddata,
                                                                                statistic='mean', bins=binedges)[0]
                                bincount[:,ises,iap,ilp,ipp] = binned_statistic(x=xdata,
                                                                                values=valueddata,
                                                                                statistic='count', bins=binedges)[0]
                        
    binmean[bincount<mincount] = np.nan
    if normalize: # subtract mean correlation from every session:
        binmean = binmean - np.nanmean(binmean[binedges[:-1]<60,:,:,:,:],axis=0,keepdims=True)

    # binmean = binmean.squeeze()
    return binmean,binpos

def plot_bin_corr_deltarf_flex(sessions,binmean,binpos,areapairs=' ',layerpairs=' ',projpairs=' ',
                               corr_type='trace_corr',normalize=False):
    # sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    # protocols = np.unique(sessiondata['protocol'])

    # clrs_areapairs = get_clr_area_pairs(areapairs)
    # clrs_projpairs = get_clr_labelpairs(projpairs)

    if projpairs==' ':
        clrs_projpairs = 'k'
    else:
        clrs_projpairs = get_clr_labelpairs(projpairs)

    fig,axes = plt.subplots(len(areapairs),len(layerpairs),figsize=(3*len(layerpairs),3*len(areapairs)),sharex=True,sharey=True)
    axes = axes.reshape(len(areapairs),len(layerpairs))
    for iap,areapair in enumerate(areapairs):
        for ilp,layerpair in enumerate(layerpairs):
            ax = axes[iap,ilp]
            handles = []

            for ipp,projpair in enumerate(projpairs):
                for ises in range(len(sessions)):
                    ax.plot(binpos,binmean[:,ises,iap,ilp,ipp].squeeze(),linewidth=0.15,color=clrs_projpairs[ipp])
                handles.append(shaded_error(ax=ax,x=binpos,y=binmean[:,:,iap,ilp,ipp].squeeze().T,center='mean',error='sem',color=clrs_projpairs[ipp]))

            ax.legend(handles,projpairs,loc='upper right',frameon=False)	
            ax.set_xlabel('Delta RF')
            ax.set_ylabel('Correlation')
            ax.set_xlim([-2,60])
            ax.set_title('%s\n%s' % (areapair, layerpair))
            # if normalize:
            #     ax.set_ylim([-0.015,0.05])
            # else: 
                # ax.set_ylim([0,0.2])
            ax.set_aspect('auto')
            ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    return fig

def plot_2D_corr_map(binmean,bincounts,bincenters,areapairs=' ',layerpairs=' ',projpairs=' ',min_counts = 50,gaussian_sigma = 0):

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    clrs_projpairs      = get_clr_labelpairs(projpairs)
    clrs_layerpairs     = get_clr_layerpairs(layerpairs)

    # Make the figure:
    delta_az,delta_el   = np.meshgrid(bincenters,bincenters)
    deltarf             = np.sqrt(delta_az**2 + delta_el**2)

    # Make the figure:
    fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
    if len(projpairs)==1 and len(areapairs)==1:
        axes = np.array([axes])
    axes = axes.reshape(len(projpairs),len(areapairs))

    for iap,areapair in enumerate(areapairs):
        for ilp,layerpair in enumerate(layerpairs):
            for ipp,projpair in enumerate(projpairs):
                ax                                              = axes[ipp,iap]
                
                data                                            = copy.deepcopy(binmean[:,:,iap,ilp,ipp])
                if gaussian_sigma: 
                    data[np.isnan(data)]                            = np.nanmean(data)
                    data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
                data[bincounts[:,:,iap,ilp,ipp]<min_counts]     = np.nan

                ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95),cmap="hot")
                # ax.imshow(binmean[:,:,iap,ilp,ipp],vmin=np.nanpercentile(binmean[:,:,iap,ilp,ipp],5),
                #                     vmax=np.nanpercentile(binmean[:,:,iap,ilp,ipp],99.9),cmap="hot",interpolation="none",extent=np.flipud(binrange).flatten())
                ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
                ax.set_xlim([-50,50])
                ax.set_ylim([-50,50])
                ax.set_xlabel(u'Δ Azimuth')
                ax.set_ylabel(u'Δ Elevation')
    plt.tight_layout()
    return fig

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

            # data = binmean[:,:,iap,ilp,ipp].copy()
            # data[deltarf>centerthr[iap]] = np.nan
            # polardata[:,0,iap,ilp,ipp] = binned_statistic(x=anglerf[~np.isnan(data)],
            #                         values=data[~np.isnan(data)],
            #                         statistic='mean',bins=polarbinedges)[0]
            
            # polardata_counts[:,0,iap,ilp,ipp] = binned_statistic(x=anglerf[~np.isnan(data)],
            #                         values=data[~np.isnan(data)],
            #                         statistic='sum',bins=polarbinedges)[0]

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
                            sig_thr = 0.001,noise_thr=1,filternear=False,binres_rf=5,binres_corr=0.01,min_dist=15):
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
    binlim          = 75
    binedges_rf     = np.arange(0,binlim,binres_rf)+binres_rf/2 
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
            if 'rf_p_' + rf_type in sessions[ises].celldata:

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
                            rffilter        = np.meshgrid(sessions[ises].celldata['rf_p_' + rf_type]<=sig_thr,sessions[ises].celldata['rf_p_'  + rf_type]<=sig_thr)
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


def plot_perc_rangecorr_map(bincounts,bincenters_rf,bincenters_corr,areapairs=' ',layerpairs=' ',projpairs=' '):

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    clrs_projpairs      = get_clr_labelpairs(projpairs)
    clrs_layerpairs     = get_clr_layerpairs(layerpairs)

    assert bincounts.shape[:2] == (len(bincenters_rf),len(bincenters_corr)), "bincounts should have shape (%d,%d), but has shape %s" % (len(bincenters_rf),len(bincenters_corr),bincounts.shape)

    # X,Y          = np.meshgrid(bincenters_rf,bincenters_corr)
    # X,Y          = np.meshgrid(bincenters_corr,bincenters_rf)

    prctiles = np.array([1,5,10,25,50,75,90,95,99]) / 100
    prctiles = np.arange(0,1,0.02)[1:]
    clrs_prctiles = plt.cm.seismic(np.linspace(0,1,len(prctiles)))

    # Make the figure:
    fig,axes = plt.subplots(len(projpairs),len(areapairs),figsize=(len(areapairs)*3,len(projpairs)*3))
    if len(projpairs)==1 and len(areapairs)==1:
        axes = np.array([axes])
    axes = axes.reshape(len(projpairs),len(areapairs))

    for iap,areapair in enumerate(areapairs):
        for ilp,layerpair in enumerate(layerpairs):
            for ipp,projpair in enumerate(projpairs):
                data                                            = bincounts[:,:,iap,ilp,ipp]
                ax                                              = axes[ipp,iap]
                for iprc,prctile in enumerate(prctiles):
                    percdata = np.zeros(len(bincenters_rf))

                    for irf in range(len(bincenters_rf)):
                        if np.any(data[irf,:]):
                            percdata[irf] = bincenters_corr[np.where((np.cumsum(data[irf,:])/np.sum(data[irf,:]))>prctile)[0][1]]
                    
                    ax.plot(bincenters_rf,percdata,color=clrs_prctiles[iprc,:],label=prctile)

                ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
                # ax.set_xlim([0,50])
                # ax.set_ylim([-0.3,0.45])
                ax.set_xlabel(u'Δ RF')
                ax.set_ylabel(u'Correlation')
    plt.tight_layout()
    return fig

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
        data   = data/np.nansum(data,axis=1,keepdims=True)

        # data   = data/np.nanmean(data[:,:,:,:,0][...,np.newaxis],axis=(4),keepdims=True)[...,0,np.newaxis]
        data   = data/np.nanmean(data,axis=(0),keepdims=True)
        # data   = data/np.nanmean(data,axis=(0,3,4),keepdims=True)
        # data   = data/np.nanmean(data[:,:,:,:,0][...,np.newaxis],axis=(0),keepdims=True)

    min_counts = 0
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
                data_cat[bincounts[:,:,iap,ilp,ipp].T<=min_counts]     = np.nan

                # ax.pcolor(X,Y,data_cat,vmin=np.nanpercentile(data_cat,2),vmax=np.nanpercentile(data_cat,99),cmap="seismic")
                # ax.pcolor(X,Y,data_cat,cmap="hot")
                ax.pcolor(X,Y,data_cat,vmin=-1,vmax=3,cmap="seismic")
                # ax.pcolor(X,Y,np.log10(1+data_cat),vmin=np.nanpercentile(np.log10(1+data_cat),5),vmax=np.nanpercentile(np.log10(1+data_cat),95),cmap="hot")
                
                ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
                # ax.set_xlim([0,600])
                ax.set_xlim([0,50])
                ax.set_ylim([-0.75,1])
                ax.set_xlabel(u'Δ RF')
                ax.set_ylabel(u'Correlation')
    plt.tight_layout()
    return fig


def bin_1d_fraccorr_deltarf(sessions,areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='F',
                            sig_thr = 0.001,noise_thr=1,filternear=False,binres_rf=5,corr_thr=0.1,min_dist=0):
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
    binedges_rf     = np.arange(0,binlim,binres_rf)+binres_rf/2 
    bincenters_rf   = binedges_rf[:-1]+binres_rf/2 
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
            # neg_thr = -0.1

            if 'rf_p_' + rf_type in sessions[ises].celldata:

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
                            rffilter        = np.meshgrid(sessions[ises].celldata['rf_p_' + rf_type]<=sig_thr,sessions[ises].celldata['rf_p_'  + rf_type]<=sig_thr)
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
                shaded_error(ax=ax,x=bincenters_rf,y=-data_neg[:,iap,ilp,ipp],yerror=data_neg_error[:,iap,ilp,ipp],color='b')

                com_pos = np.average(bincenters_rf, weights=np.nan_to_num(binpos[:,iap,ilp,ipp]))
                com_neg = np.average(bincenters_rf, weights=np.nan_to_num(binneg[:,iap,ilp,ipp]))

                ax.plot(com_pos,1.5,'o',color='r')
                ax.plot(com_neg,-1.5,'o',color='b')
                # ax.plot(bincenters_rf,data_pos[:,iap,ilp,ipp],color='r')
                # ax.plot(bincenters_rf,-data_neg[:,iap,ilp,ipp],color='b')

                ax.axhline(1,color='k',lw=1,ls=':')
                ax.axhline(-1,color='k',lw=1,ls=':')
                ax.axhline(0,color='k',lw=1,ls='-')
                ax.set_title('%s\n%s' % (areapair, projpair),c=clrs_areapairs[iap])
                # ax.set_xlim([0,600])
                ax.set_xlim([0,75])
                # ax.set_ylim([-2.2,2.2])
                ax.set_ylim([-1.65,1.65])
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

    # [binmean,bincounts,bincenters_rf,bincenters_corr] = bin_2d_rangecorr_deltarf(sessions_subset,areapairs=areapairs,layerpairs=' ',projpairs=projpairs,
    #                             corr_type=corr_type,binres_rf=binres_rf,binres_corr=binres_corr,rf_type=rf_type,normalize=False,
    #                             sig_thr = 0.001,filternear=filternear,filtersign=filtersign)

def bin_2d_corr_deltarf(sessions,areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='F',
                            rotate_prefori=False,deltaori=None,sig_thr = 0.001,noise_thr=1,filternear=False,
                            binresolution=5,tuned_thr=0,absolute=False,normalize=False,dsi_thr=0,
                            min_dist=15,filtersign=None):
    """
    Average pairwise correlations as a function of pairwise delta azimuth and elevation.
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
    sig_thr : float, optional
        significance threshold for including cells in the analysis, by default 0.001
    """
    #Binning        parameters:
    binlim          = 75
    binedges        = np.arange(-binlim,binlim,binresolution)+binresolution/2 
    bincenters      = binedges[:-1]+binresolution/2 
    nBins           = [len(binedges)-1,len(binedges)-1]

    binmean         = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bincount        = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs)))

    # binmean     = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs),len(sessions)))
    # bincount    = np.zeros((*nBins,len(areapairs),len(layerpairs),len(projpairs),len(sessions)))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            if 'rf_p_' + rf_type in sessions[ises].celldata:

                source_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                target_el       = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                delta_el        = source_el[:,None] - target_el[None,:]

                source_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                target_az       = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                delta_az        = source_az[:,None] - target_az[None,:]

                # Careful definitions:
                # delta_az is source neurons azimuth minus target neurons azimuth position:
                # plt.imshow(delta_az[:10,:10],vmin=-20,vmax=20,cmap='bwr')
                # entry delta_az[0,1] being positive means target neuron RF is to the right of source neuron
                # entry delta_el[0,1] being positive means target neuron RF is above source neuron
                # To rotate azimuth and elevation to relative to the preferred orientation of the source neuron
                # means that for a neuron with preferred orientation 45 deg all delta az and delta el of paired neruons
                # will rotate 45 deg, such that now delta azimuth and delta elevation is relative to the angle 
                # of pref ori of the source neuron 
                
                if absolute == True:
                    corrdata = np.abs(corrdata)

                if normalize == True:
                    corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)

                if filtersign == 'neg':
                    corrdata[corrdata>0] = np.nan
                
                if filtersign =='pos':
                    corrdata[corrdata<0] = np.nan

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
                        ori_rots            = sessions[ises].celldata['pref_ori'][iN]
                        ori_rots            = np.tile(sessions[ises].celldata['pref_ori'][iN],len(sessions[ises].celldata))
                        angle_vec           = np.vstack((delta_el[iN,:], delta_az[iN,:]))
                        angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots + 90) #90 degrees is added to make collinear horizontal
                        delta_el[iN,:]      = angle_vec_rot[0,:]
                        delta_az[iN,:]      = angle_vec_rot[1,:]

                for iap,areapair in enumerate(areapairs):
                    for ilp,layerpair in enumerate(layerpairs):
                        for ipp,projpair in enumerate(projpairs):
                            rffilter        = np.meshgrid(sessions[ises].celldata['rf_p_' + rf_type]<sig_thr,sessions[ises].celldata['rf_p_'  + rf_type]<sig_thr)
                            rffilter        = np.logical_and(rffilter[0],rffilter[1])
                            
                            signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
                            signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                            if tuned_thr:
                                tuningfilter    = np.meshgrid(sessions[ises].celldata['tuning_var']>tuned_thr,sessions[ises].celldata['tuning_var']>tuned_thr)
                                tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
                            else: 
                                tuningfilter    = np.ones(np.shape(rffilter))

                            areafilter      = filter_2d_areapair(sessions[ises],areapair)

                            layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                            projfilter      = filter_2d_projpair(sessions[ises],projpair)

                            nanfilter       = ~np.isnan(corrdata)

                            proxfilter      = ~(sessions[ises].distmat_xy<min_dist)
                            
                            if deltaori is not None:
                                assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
                                delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
                                delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
                                deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
                                                        delta_pref <= deltaori[1]),axis=0)
                            else:
                                deltaorifilter = np.ones(np.shape(rffilter))

                            if dsi_thr:
                                dsi_filter = np.meshgrid(sessions[ises].celldata['DSI']>dsi_thr,sessions[ises].celldata['DSI']>dsi_thr)
                                dsi_filter = np.logical_and(dsi_filter[0],dsi_filter[1])
                            else:
                                dsi_filter = np.ones(np.shape(rffilter))

                            #Combine all filters into a single filter:
                            cellfilter      = np.all((rffilter,signalfilter,tuningfilter,areafilter,nearfilter,
                                                layerfilter,projfilter,proxfilter,nanfilter,deltaorifilter,dsi_filter),axis=0)

                            if np.any(cellfilter):
                                
                                xdata               = delta_el[cellfilter].flatten()
                                ydata               = delta_az[cellfilter].flatten()
                                vdata               = corrdata[cellfilter].flatten()

                                tempfilter          = ~np.isnan(xdata) & ~np.isnan(ydata) & ~np.isnan(vdata)
                                xdata               = xdata[tempfilter]
                                ydata               = ydata[tempfilter]
                                vdata               = vdata[tempfilter]
                                
                                #Take the sum of the correlations in each bin:
                                binmean[:,:,iap,ilp,ipp]   += binned_statistic_2d(x=xdata, y=ydata, values=vdata,
                                                                                    bins=binedges, statistic='sum')[0]
                                
                                # Count how many correlation observations are in each bin:
                                bincount[:,:,iap,ilp,ipp]  += np.histogram2d(x=xdata,y=ydata,bins=binedges)[0]

                                # binmean[:,:,iap,ilp,ipp,ises]   += binmean_temp
                                # bincount[:,:,iap,ilp,ipp,ises]  += bincount_temp

    # import scipy.stats as st 

    # # Confidence Interval = x(+/-)t*(s/√n)
    # # create 95% confidence interval
    # numsamples  = np.unique(bincount[~np.isnan(bincount)])
    # binci       = np.empty((*np.shape(binmean),2))
    
    # for ns in numsamples: #ns = 10
    #     st.t.interval(alpha=0.95, df=len(ns)-1, 
    #             loc=np.nanmean(binmean[:]), 
    #             scale=st.sem(binmean[:])) 
    
    # divide the total summed correlations by the number of counts in that bin to get the mean:
    binmean = binmean / bincount

    return binmean,bincount,bincenters

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
