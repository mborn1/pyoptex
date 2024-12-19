"""
Module for utility functions related to computational formulas.
"""
import numba
import numpy as np


@numba.njit
def outer_integral(arr):
    """
    Computes the integral of the outer products of the array rows 
    using the Monte-Carlo approximation, up to the volume factor.
    This is a simple average of the outer products.

    Parameters
    ----------
    arr : np.array(2d)
        The array
    
    Returns
    -------
    out : np.array(2d)
        The integral of the outer product, up to the volume factor.
    """
    out = np.zeros((arr.shape[-1], arr.shape[-1]))
    for i in range(arr.shape[0]):
        out += np.expand_dims(arr[i], 1) @ np.expand_dims(arr[i], 0)
    return out / arr.shape[0]
