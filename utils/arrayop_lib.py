# -*- coding: utf-8 -*-
"""
This script contains some processing function on numpy arrays
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import numpy as np


def nanweightedaverage(A,weights,axis):
    """
    Compute the weighted average of the values in A along the given axis.
    If a value in A is NaN, it is ignored (i.e. not counted in the sum of weights)

    Parameters
    ----------
    A : array_like
        The array to average
    weights : array_like
        The weights to use for the average
    axis : int
        The axis to average over

    Returns
    -------
    average : array_like
        The weighted average of the values in A along the given axis
    """
    with np.errstate(invalid='ignore'):
        return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)



    # mask = ~np.isnan(A)
    # weights_masked = weights.copy()
    # weights_masked[~mask] = 0
    # return np.nansum(A*weights_masked,axis=axis)/weights_masked.sum(axis=axis)