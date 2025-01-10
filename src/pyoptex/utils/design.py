"""
Module for utility functions related to the design matrices.
"""
import numba
import numpy as np

from .numba import numba_all_axis2, numba_take_advanced


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

################################################

@numba.njit
def x2fx(Yenc, modelenc):
    """
    Create the model matrix from the design matrix and model specification.
    This specification is the same as MATLAB's.
    A model is specified as a matrix with each term being a row. The elements
    in each row specify the power of the factor.
    E.g.

    * The intercept is [0, 0, ..., 0]
    * A main factor is [1, 0, ..., 0]
    * A two-factor interaction is [1, 1, 0, ..., 0]
    * A quadratic term is [2, 0, ..., 0]

    Parameters
    ----------
    Yenc : np.array(2d)
        The encoded design matrix.
    modelenc : np.array(2d)
        The encoded model, specified as in MATLAB.

    Returns
    -------
    X : np.array(2d)
        The model matrix
    """
    Xenc = np.zeros((*Yenc.shape[:-1], modelenc.shape[0]))
    for i, term in enumerate(modelenc):
        p = np.ones(Yenc.shape[:-1])
        for j in range(modelenc.shape[1]):
            if term[j] != 0:
                if term[j] == 1:
                    p *= Yenc[..., j]
                else:
                    p *= Yenc[..., j] ** term[j]
        Xenc[..., i] = p
    return Xenc

@numba.njit
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
    # Initialization
    c_asc = 0
    c = Zi[0]

    # Loop over each element
    Zi[0] = c_asc
    for i in range(1, len(Zi)):
        # If a different group number, update state
        if Zi[i] != c:
            c_asc += 1
            c = Zi[i]
        
        # Set ascending
        Zi[i] = c_asc

    return Zi

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

################################################

@numba.njit
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
    coords : None or :py:class:`numba.typed.List` (np.array(2d))
        The possible coordinates for each factor. 

    Returns
    -------
    Yenc : np.array(2d)
        The encoded design-matrix 
    """
    # Compute amount of columns per factor
    cols = np.where(effect_types > 1, effect_types - 1, effect_types)

    # Initialize encoding
    ncols = np.sum(cols)
    Yenc = np.zeros((*Y.shape[:-1], ncols))

    start = 0
    # Loop over factors
    for i in range(effect_types.size):
        if effect_types[i] == 1:
            # Continuous factor: copy
            Yenc[..., start] = Y[..., i]
            start += 1
        else:
            # Categorical factor: effect encode
            if coords is None:
                eye = np.concatenate((np.eye(cols[i]), -np.ones((1, cols[i]))))
            else:
                eye = coords[i]
            Yenc[..., start:start+cols[i]] = numba_take_advanced(eye, Y[..., i].astype(np.int_))
            start += cols[i]

    return Yenc

@numba.njit
def decode_design(Y, effect_types, coords=None):
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
    coords: None or :py:class:`numba.typed.List` (np.array(2d))
        Coordinates to be used for decoding the categorical variables.

    Returns
    -------
    Ydec : np.array(2d)
        The decoded design-matrix 
    """
    # Initialize dencoding
    Ydec = np.zeros((*Y.shape[:-1], effect_types.size))

    # Loop over all factors
    start = 0
    for i in range(effect_types.size):
        if effect_types[i] == 1:
            Ydec[..., i] = Y[..., start]
            start += 1
        else:
            ncols = effect_types[i] - 1
            if coords is None:
                Ydec[..., i] = np.where(Y[..., start] == -1, ncols, np.argmax(Y[..., start:start+ncols], axis=-1))
            else:
                Ydec[..., i] = np.argmax(numba_all_axis2(np.expand_dims(coords[i], 1) == Y[..., start:start+ncols]).astype(np.int8), axis=0)
            start += ncols

    return Ydec

