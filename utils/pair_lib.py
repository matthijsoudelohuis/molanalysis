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
from skimage.measure import block_reduce
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
    if  areapair == ' ':
        return np.full(np.shape(ses.distmat_xy),True)
    area1,area2 = areapair.split('-')
    areafilter = np.meshgrid(ses.celldata['roi_name']==area1,ses.celldata['roi_name']==area2)
    return np.logical_and(areafilter[0],areafilter[1])

# Define function to filter neuronpairs based on area combination
def filter_2d_layerpair(ses,layerpair):
    if layerpair == ' ':
        return np.full(np.shape(ses.distmat_xy),True)
    layer1,layer2 = layerpair.split('-')
    layerfilter = np.meshgrid(ses.celldata['layer']==layer1,ses.celldata['layer']==layer2)
    return np.logical_and(layerfilter[0],layerfilter[1])

# Define function to filter neuronpairs based on area combination
def filter_2d_projpair(ses,projpair):
    if projpair == ' ':
        return np.full(np.shape(ses.distmat_xy),True)
    proj1,proj2 = projpair.split('-')
    projfilter = np.meshgrid(ses.celldata['labeled']==proj1,ses.celldata['labeled']==proj2)
    return np.logical_and(projfilter[0],projfilter[1])

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