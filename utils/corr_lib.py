
"""
This script contains functions to compute noise correlations
on simultaneously acquired calcium imaging data with mesoscope
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import os
import numpy as np
import pandas as pd

def compute_noise_correlation(sessions):
    nSessions = len(sessions)

    sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    if np.all(sessiondata['protocol']=='GR'):
        for ises in range(nSessions):
            # get signal correlations:
            [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

            oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
            ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
            assert(len(ori_counts) == 16 or len(ori_counts) == 8)
            resp_meanori    = np.empty([N,len(oris)])

            for i,ori in enumerate(oris):
                resp_meanori[:,i] = np.nanmean(sessions[ises].respmat[:,sessions[ises].trialdata['Orientation']==ori],axis=1)

            sessions[ises].sig_corr                 = np.corrcoef(resp_meanori)

            prefori                         = oris[np.argmax(resp_meanori,axis=1)]
            sessions[ises].delta_pref       = np.abs(np.subtract.outer(prefori, prefori))

            respmat_res                     = sessions[ises].respmat.copy()

            ## Compute residuals:
            for ori in oris:
                ori_idx     = np.where(sessions[ises].trialdata['Orientation']==ori)[0]
                temp        = np.mean(respmat_res[:,ori_idx],axis=1)
                respmat_res[:,ori_idx] = respmat_res[:,ori_idx] - np.repeat(temp[:, np.newaxis], len(ori_idx), axis=1)

            sessions[ises].noise_corr                   = np.corrcoef(respmat_res)
            
            # sessions[ises].sig_corr[np.eye(N)==1]   = np.nan
            # sessions[ises].noise_corr[np.eye(N)==1]     = np.nan

            idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
            sessions[ises].sig_corr[idx_triu] = np.nan
            sessions[ises].noise_corr[idx_triu] = np.nan
            sessions[ises].delta_pref[idx_triu] = np.nan

            assert np.all(sessions[ises].sig_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].sig_corr[~idx_triu] < 1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] < 1)
    else: 
        print('not yet implemented noise corr for other protocols than GR')
    return sessions


def mean_resp_image(ses):
    nNeurons = np.shape(ses.respmat)[0]
    images = np.unique(ses.trialdata['ImageNumber'])
    respmean = np.empty((nNeurons,len(images)))
    for im in images:
        respmean[:,im] = np.mean(ses.respmat[:,ses.trialdata['ImageNumber']==im],axis=1)
    return respmean


def compute_signal_correlation(sessions):
    nSessions = len(sessions)

    sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
    if np.all(sessiondata['protocol']=='IM'):
         for ises in range(nSessions):
            respmean        = mean_resp_image(sessions[ises])
            [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

            sessions[ises].sig_corr                   = np.corrcoef(respmean)

            idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part

            sessions[ises].sig_corr[idx_triu] = np.nan

    else: 
        print('not yet implemented signal corr for other protocols than IM')

    return sessions

def compute_pairwise_metrics(sessions):

    for ises in range(len(sessions)):
        [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

        ## Compute euclidean distance matrix based on soma center:
        sessions[ises].distmat_xyz     = np.zeros((N,N))
        sessions[ises].distmat_xy      = np.zeros((N,N))
        sessions[ises].distmat_rf      = np.zeros((N,N))
        sessions[ises].areamat         = np.empty((N,N),dtype=object)
        sessions[ises].labelmat        = np.empty((N,N),dtype=object)

        x,y,z = sessions[ises].celldata['xloc'],sessions[ises].celldata['yloc'],sessions[ises].celldata['depth']
        b = np.array((x,y,z))
        for i in range(N):
            print(f"\rComputing pairwise distances for neuron {i+1} / {N}",end='\r')
            a = np.array((x[i],y[i],z[i]))
            sessions[ises].distmat_xyz[i,:] = np.linalg.norm(a[:,np.newaxis]-b,axis=0)
            sessions[ises].distmat_xy[i,:] = np.linalg.norm(a[:2,np.newaxis]-b[:2,:],axis=0)

        if 'rf_azimuth' in sessions[ises].celldata:
            rfaz,rfel = sessions[ises].celldata['rf_azimuth'],sessions[ises].celldata['rf_elevation']
            d = np.array((rfaz,rfel))

            for i in range(N):
                c = np.array((rfaz[i],rfel[i]))
                sessions[ises].distmat_rf[i,:] = np.linalg.norm(c[:,np.newaxis]-d,axis=0)

        g = np.meshgrid(sessions[ises].celldata['roi_name'],sessions[ises].celldata['roi_name'])
        sessions[ises].areamat = g[0] + '-' + g[1]
        sessions[ises].areamat[sessions[ises].areamat=='PM-V1'] = 'V1-PM' #fix order for combinations

        g = np.meshgrid(sessions[ises].celldata['redcell'].astype(int).astype(str).to_numpy(),
                        sessions[ises].celldata['redcell'].astype(int).astype(str).to_numpy())
        sessions[ises].labelmat = g[0] + '-' + g[1] 
        sessions[ises].labelmat[sessions[ises].labelmat=='1-0'] = '0-1' #fix order for combinations

        idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
        sessions[ises].distmat_xyz[idx_triu] = np.nan
        sessions[ises].distmat_xy[idx_triu] = np.nan
        sessions[ises].distmat_rf[idx_triu] = np.nan
        sessions[ises].areamat[idx_triu] = np.nan
        sessions[ises].labelmat[idx_triu] = np.nan
    return sessions