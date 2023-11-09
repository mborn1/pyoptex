import numba
import numpy as np

from ...utils.numba import numba_take_advanced, numba_all_axis2

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
    Y : np.array
        The design matrix. It should be 2D
    model : np.array
        The model, specified as in MATLAB.

    Returns
    -------
    X : np.array
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
    Force ascending groups. In other words [0, 0, 2, 2, 1, 1]
    is transformed to [0, 0, 1, 1, 2, 2].

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

def obs_var_from_Zs(Zs, N, ratios=None):
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
        The number of runs.
    ratios : np.array(1d)
        The variance ratios of the different groups compared to the variance of epsilon.
    
    Returns
    -------
    V : np.array(2d)
        The observation covariance matrix.
    """
    V = np.eye(N)
    if ratios is None:
        ratios = np.ones(len(Zs))
    Zs = [np.eye(Zi[-1]+1)[Zi] for Zi in Zs if Zi is not None]
    return V + sum(ratios[i] * Zs[i] @ Zs[i].T for i in range(len(Zs)))

################################################

@numba.njit
def encode_design(Y, effect_types):
    """
    Encode the design according to the effect types.
    Each categorical factor is encoded using
    effect-encoding.

    Parameters
    ----------
    Y : np.ndarray(2d)
        The current design matrix
    effect_types : np.ndarray(1d) 
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).

    Returns
    -------
    Yenc : np.ndarray(2d)
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
            eye = np.concatenate((np.eye(cols[i]), -np.ones((1, cols[i]))))
            Yenc[..., start:start+cols[i]] = numba_take_advanced(eye, Y[..., i].astype(np.int64))
            start += cols[i]

    return Yenc

@numba.njit
def decode_design(Y, effect_types, coords=None):
    """
    Decode the design according to the effect types.
    Each categorical factor is decoded from
    effect-encoding. The expected input is an
    effect-encoded design matrix.

    It is the inverse of :py:func:`encode_design`

    Parameters
    ----------
    Y : np.ndarray(2d)
        The current, effect-encoded design matrix.
    effect_types : np.ndarray(1d) 
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).
    coords: List(np.array(2d))
        Coordinates to be used for decoding the categorical variables.

    Returns
    -------
    Ydec : np.ndarray(2d)
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


