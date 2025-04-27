"""
Module for utility functions related to the design matrices.
"""
import numpy as np

from ._design_cy import *

def create_default_coords(effect_type):
    """
    Defines the default possible coordinates per effect type. 
    A continuous variable has [-1, 0, 1], a categorical variable 
    is an array from 1 to the number of categorical levels.

    Parameters
    ----------
    effect_type : int
        The type of the effect. 1 indicates continuous, 
        higher indicates categorical with that number of levels.
    
    Returns
    -------
    coords : np.array(1d, 1)
        The default possible coordinates for the factor. Each row
        represents a coordinate.
    """
    if effect_type == 1:
        return np.array([-1, 0, 1], dtype=np.float64).reshape(-1, 1)
    else:
        return np.arange(effect_type, dtype=np.float64).reshape(-1, 1)

def obs_var_from_Zs(Zs, N, ratios=None, include_error=True):
    """
    Computes the observation covariance matrix from the different groupings.
    Computed as V = I + sum(ratio * Zi Zi.T) (where Zi is the expanded grouping
    matrix).
    For example [0, 0, 1, 1] is represented by [[1, 0], [1, 0], [0, 1], [0, 1]].

    Parameters
    ----------
    Zs : tuple(np.array(1d) or None)
        The tuple of grouping matrices. Can include Nones which are ignored.
    N : int
        The number of runs. Necessary in case no random groups are present.
    ratios : np.array(1d)
        The variance ratios of the different groups compared to the variance of
        the random errors.
    include_error : bool
        Whether to include the random errors or not.
    
    Returns
    -------
    V : np.array(2d)
        The observation covariance matrix.
    """
    if include_error:
        V = np.eye(N)
    else:
        V = np.zeros((N, N))

    if ratios is None:
        ratios = np.ones(len(Zs))
        
    Zs = [np.eye(Zi[-1]+1)[Zi] for Zi in Zs if Zi is not None]
    return V + sum(ratios[i] * Zs[i] @ Zs[i].T for i in range(len(Zs)))


