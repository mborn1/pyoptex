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
        
    return V + sum(
        r * (Zi_expanded @ Zi_expanded.T)
        for r, Zi_expanded in (
            (r, np.eye(Zi[-1] + 1)[Zi])
            for r, Zi in zip(ratios, Zs)
            if Zi is not None
        )
    )

def x2fx(Yenc, modelenc):
    """
    Create the model matrix from the design matrix
    and model specification.

    Parameters
    ----------
    Yenc : np.ndarray(2d)
        The encoded design matrix.
    modelenc : np.ndarray(2d)
        The encoded model, specified as in MATLAB.

    Returns
    -------
    Xenc : np.ndarray(2d)
        The model matrix
    """
    return x2fx_cython_impl(np.ascontiguousarray(Yenc), np.ascontiguousarray(modelenc, dtype=np.int64))

def force_Zi_asc(Zi):
    """
    Force ascending groups. In other words [0, 0, 2, 1, 1, 1]
    is transformed to [0, 0, 1, 2, 2, 2].

    Parameters
    ----------
    Zi : np.array(1d)
        The current grouping matrix
    
    Returns
    -------
    Zi : np.array(1d)
        The grouping matrix with ascending groups
    """
    return force_Zi_asc_cython_impl(np.ascontiguousarray(Zi, dtype=np.int64))

def encode_design(Y, effect_types, coords=None):
    """
    Encode the design according to the effect types.
    Each categorical factor is encoded using
    effect-encoding, unless the coordinates are specified.

    It is the inverse of :py:func:`decode_design <pyoptex.utils.design.decode_design>`

    Parameters
    ----------
    Y : np.array(2d)
        The current design matrix.
    effect_types : np.array(1d) 
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).
    coords : None or list[np.ndarray]
        The possible coordinates for each factor. 

    Returns
    -------
    Yenc : np.array(2d)
        The encoded design-matrix 
    """
    return encode_design_cython_impl(np.ascontiguousarray(Y), np.ascontiguousarray(effect_types, dtype=np.int64), coords)

def decode_design(Yenc, effect_types, coords=None):
    """
    Decode the design according to the effect types.
    Each categorical factor is decoded from
    effect-encoding, unless the coordinates are specified.

    It is the inverse of :py:func:`encode_design <pyoptex.utils.design.encode_design>`

    Parameters
    ----------
    Y : np.array(2d)
        The effect-encoded design matrix.
    effect_types : np.array(1d) 
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).
    coords: None or list[np.ndarray]
        Coordinates to be used for decoding the categorical variables.

    Returns
    -------
    Ydec : np.array(2d)
        The decoded design-matrix 
    """
    return decode_design_cython_impl(np.ascontiguousarray(Yenc), np.ascontiguousarray(effect_types, dtype=np.int64), coords)