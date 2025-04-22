# -*- coding: utf-8 -*-
"""
Set of functions that combine properties of cell pairs to create 2D relationship matrices
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic,binned_statistic_2d
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_pairwise_metrics(sessions):
    sessions = compute_pairwise_anatomical_distance(sessions)
    sessions = compute_pairwise_delta_rf(sessions)
    return sessions

def compute_pairwise_anatomical_distance(sessions):

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing pairwise anatomical distance for each session: '):
        ## Compute euclidean distance matrix based on soma center:
        
        N                              = len(sessions[ises].celldata) #get dimensions of response matrix
        sessions[ises].distmat_xyz     = np.zeros((N,N)) #init output arrays
        sessions[ises].distmat_xy      = np.zeros((N,N))

        x = sessions[ises].celldata['xloc'].to_numpy()
        y = sessions[ises].celldata['yloc'].to_numpy()
        z = sessions[ises].celldata['depth'].to_numpy()
        b = np.array((x,y,z)) #store this vector for fast linalg norm computation
        for i in range(N): #compute distance from each neuron to all others:
            a = np.array((x[i],y[i],z[i]))
            sessions[ises].distmat_xyz[i,:] = np.linalg.norm(a[:,np.newaxis]-b,axis=0)
            sessions[ises].distmat_xy[i,:] = np.linalg.norm(a[:2,np.newaxis]-b[:2,:],axis=0)

        for area in ['V1','PM','AL','RSP']: #set all interarea pairs to nan:
            sessions[ises].distmat_xy[np.ix_(sessions[ises].celldata['roi_name']==area,sessions[ises].celldata['roi_name']!=area)] = np.nan
            sessions[ises].distmat_xyz[np.ix_(sessions[ises].celldata['roi_name']==area,sessions[ises].celldata['roi_name']!=area)] = np.nan

        # idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
        # sessions[ises].distmat_xyz[idx_triu] = np.nan
        # sessions[ises].distmat_xy[idx_triu] = np.nan

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

# Define function to filter neuronpairs based on area combination
def filter_2d_areapair(ses,areapair):
    #first entry of the areapair determines x, second entry determines y
    if areapair == ' ':
        return np.full(np.shape(ses.distmat_xy),True)
    area1,area2 = areapair.split('-')
    assert 'roi_name' in ses.celldata, "Error: 'roi_name' is not in ses.celldata."
    assert np.isin([area1,area2],ses.celldata['roi_name'].unique()).all(), \
        f"Error: one of {area1} or {area2} is not in ses.celldata['roi_name']. Unique labels are {ses.celldata['roi_name'].unique()}"
    return np.outer(ses.celldata['roi_name']==area1, ses.celldata['roi_name']==area2)

# Define function to filter neuronpairs based on layer combination
def filter_2d_layerpair(ses,layerpair):
    #first entry of the layerpair determines x, second entry determines y
    if layerpair == ' ':
        return np.full(np.shape(ses.distmat_xy),True)
    layer1,layer2 = layerpair.split('-')
    assert 'layer' in ses.celldata, "Error: 'layer' is not in ses.celldata."
    # assert np.isin([layer1,layer2],ses.celldata['layer'].unique()).all(), \
    #     f"Error: one of {layer1} or {layer2} is not in ses.celldata['layer']. Unique labels are {ses.celldata['layer'].unique()}"
    return np.outer(ses.celldata['layer']==layer1, ses.celldata['layer']==layer2)

# Define function to filter neuronpairs based on projection combination
def filter_2d_projpair(ses,projpair):
    #first entry of the projpair determines x, second entry determines y
    if projpair == ' ':
        return np.full(np.shape(ses.distmat_xy),True)
    proj1,proj2 = projpair.split('-')
    assert 'labeled' in ses.celldata, "Error: 'labeled' is not in ses.celldata. "
    # assert np.isin([proj1,proj2],ses.celldata['labeled'].unique()).all(), \
        # f"Error: one of {proj1} or {proj2} is not in ses.celldata['labeled']. Unique labels are {ses.celldata['labeled'].unique()}"
    return np.outer(ses.celldata['labeled']==proj1, ses.celldata['labeled']==proj2)


def value_matching(idx,group,values,bins=20,showFig=False):
    """
    Subsample from the other groups to make the distribution of values across groups match the group with the least counts overall.

    Parameters
    ----------
    idx : numpy array of indices
        Vector with indices of original data (e.g. neurons [56,57,58,62,70,134,etc.])
    group : numpy array
        Vector with group identity (e.g. area [1,1,1,2,2,2,etc.])
    values : numpy array
        Vector with values (e.g. correlation [0.1,0.2,0.3,0.4,0.5,etc.])
    bins : int
        Number of bins to divide the distribution in
    showFig : bool
        If True, make a plot where on the left subplot the original distributions are shown in counts and on the left the subsampled distributions after the matching.

    Returns
    -------
    idx_subsampled : numpy array
        Indices of the subsampled elements
    """
    # first identify the group with the least counts overall
    group_counts = np.array([np.sum(group==g) for g in np.unique(group)])
    least_group = np.unique(group)[np.argmin(group_counts)]

    # make a histogram of the values of the group with the least counts
    hist,bin_edges = np.histogram(values[group==least_group],bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    # go over the other groups and subsample without replacement from each of the groups the same number of values in each bin
    idx_subsampled = []
    for g in np.unique(group):
        if g != least_group:
            for i in range(len(bin_centers)):
                bin_group_idx = np.all((group==g,values>=bin_edges[i],values<bin_edges[i+1]),axis=0)
                if np.sum(bin_group_idx) > hist[i]:
                    idx_subsampled.extend(np.random.choice(np.where(bin_group_idx)[0],hist[i],replace=False))
                else:
                    idx_subsampled.extend(np.where(bin_group_idx)[0])
    idx_subsampled.extend(np.where(group==least_group)[0])
    
    if showFig:
        values_new = values[idx_subsampled]
        group_new = group[idx_subsampled]
        fig,ax = plt.subplots(1,2,sharey=True)
        ax[0].set_title('Original distributions')
        for g in np.unique(group):  
            ax[0].hist(values[group==g],bins=bin_edges,label=g,alpha=0.5,histtype='step')  
        ax[0].legend()

        ax[1].set_title('Subsampled distributions')
        for g in np.unique(group):
            ax[1].hist(values_new[group_new==g],bins=bin_edges,label=g,alpha=0.5,histtype='step')
        ax[1].legend()

    return np.array(idx[idx_subsampled])

# idx = np.arange(1000)+1000
# group = np.random.choice([0,1,2],p=[0.1,0.45,0.45],size=1000)
# values = np.random.rand(1000)
# idx_subsampled = value_matching(idx,group,values,showFig=True)


# # Define function to filter neuronpairs based on area combination
# def filter_2d_areapair(ses,areapair):
#     if  areapair == ' ':
#         return np.full(np.shape(ses.distmat_xy),True)
#     area1,area2 = areapair.split('-')
#     areafilter1 = np.meshgrid(ses.celldata['roi_name']==area1,ses.celldata['roi_name']==area2)
#     areafilter1 = np.logical_and(areafilter1[0],areafilter1[1])
#     areafilter2 = np.meshgrid(ses.celldata['roi_name']==area1,ses.celldata['roi_name']==area2)
#     areafilter2 = np.logical_and(areafilter2[0],areafilter2[1])

#     return np.logical_or(areafilter1,areafilter2)

# # Define function to filter neuronpairs based on area combination
# def filter_2d_layerpair(ses,layerpair):
#     if  layerpair == ' ':
#         return np.full(np.shape(ses.distmat_xy),True)
#     layer1,layer2 = layerpair.split('-')
#     layerfilter1 = np.meshgrid(ses.celldata['layer']==layer1,ses.celldata['layer']==layer2)
#     layerfilter1 = np.logical_and(layerfilter1[0],layerfilter1[1])
#     # layerfilter2 = np.meshgrid(ses.celldata['layer']==layer1,ses.celldata['layer']==layer2)
#     # layerfilter2 = np.logical_and(layerfilter2[0],layerfilter2[1])

#     return np.logical_or(layerfilter1,layerfilter2)
