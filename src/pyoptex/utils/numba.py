"""
Module for numba equivalents of some unsupported numpy functions.
"""

import numba
import numpy as np


@numba.njit
def numba_diff(x):
    """
    Numba compatible implementation of np.diff(...) for
    1d arrays.

    Parameters
    ----------
    x : np.array(1d)
        The input array

    Returns
    -------
    out : np.array(1d)
        The results of np.diff(...)
    """
    return x[1:] - x[:-1]

@numba.njit
def numba_diff_axis0(x):
    """
    Numba compatible implementation of np.diff(..., axis=0) for
    2d arrays.

    Parameters
    ----------
    x : np.array(2d)
        The input array

    Returns
    -------
    out : np.array(2d)
        The results of np.diff(..., axis=0)
    """
    diff = np.zeros((x.shape[0] - 1, x.shape[1]))
    for i in range(x.shape[1]):
        diff[:, i] = x[1:, i] - x[:-1, i]
    return diff

@numba.njit
def numba_any_axis1(x):
    """
    Numba compatible implementation of np.any(..., axis=1) for
    2d arrays.

    Parameters
    ----------
    x : np.array(2d)
        The input array

    Returns
    -------
    out : np.array(1d, bool)
        The results of np.any(..., axis=1)
    """
    res = np.zeros(x.shape[0], dtype=np.bool_)
    for i in range(x.shape[1]):
        res = np.logical_or(res, x[:, i])
    return res

@numba.njit
def numba_all_axis1(x):
    """
    Numba compatible implementation of np.all(..., axis=1) for
    2d arrays.

    Parameters
    ----------
    x : np.array(2d)
        The input array

    Returns
    -------
    out : np.array(1d, bool)
        The results of np.all(..., axis=1)
    """
    out = np.ones(x.shape[0], dtype=np.bool_)
    for i in range(x.shape[0]):
        out[i] = np.all(x[i, :])
    return out

@numba.njit
def numba_all_axis2(x):
    """
    Numba compatible implementation of np.all(..., axis=2) for
    3d arrays.

    Parameters
    ----------
    x : np.array(3d)
        The input array

    Returns
    -------
    out : np.array(2d, bool)
        The results of np.all(..., axis=2)
    """
    res = np.ones((x.shape[0], x.shape[1]), dtype=np.bool_)
    for i in range(x.shape[2]):
        res = np.logical_and(res, x[:, :, i])
    return res

@numba.njit
def numba_delete_axis0(x, pos):
    """
    Numba compatible implementation of np.delete(..., axis=0) for
    2d arrays.

    Parameters
    ----------
    x : np.array(2d)
        The input array

    Returns
    -------
    out : np.array(2d)
        The results of np.delete(..., axis=0)
    """
    mask = np.ones(x.shape[0], dtype=np.bool_)
    mask[pos] = False
    return np.copy(x[mask])

@numba.njit
def numba_insert(x, pos, value):
    """
    Numba compatible implementation of np.insert(...) for
    2d arrays.

    Parameters
    ----------
    x : np.array(2d)
        The input array

    Returns
    -------
    out : np.array(2d)
        The results of np.insert(...)
    """
    """Numba version of np.insert"""
    a = np.zeros(x.size + 1, dtype=x.dtype)
    a[:pos] = x[:pos]
    a[pos] = value
    a[pos+1:] = x[pos:]
    return a

@numba.njit
def numba_insert_axis0(x, pos, value):
    """
    Numba compatible implementation of np.insert(..., axis=0) for
    2d arrays.

    Parameters
    ----------
    x : np.array(2d)
        The input array

    Returns
    -------
    out : np.array(2d)
        The results of np.insert(..., axis=0)
    """
    a = np.zeros((x.shape[0] + 1, x.shape[1]), dtype=x.dtype)
    a[:pos] = x[:pos]
    a[pos] = value
    a[pos+1:] = x[pos:]
    return a

@numba.njit
def numba_take_advanced(arr, idx, out=None):
    """
    Numba compatible implementation of the advanced
    indexing scheme in numpy.
    """
    # Reshape indices
    shape = idx.shape
    idx = idx.flatten()

    # Initialize out
    if out is None:
        out = np.zeros((idx.size, *arr.shape[1:]), dtype=arr.dtype)

    # Fill result
    for i in range(idx.size):
        out[i] = arr[idx[i]]

    # Reshape and return
    return out.reshape((*shape, *arr.shape[1:]))

@numba.njit
def numba_choice_bool_axis0(valids):
    """
    For each row in valids, chooses a random index of the true
    elements in that row. For example, if valids
    is [[True, False], [True, True]], the first element of out
    must be 0 as there is no other options, the second element
    has a 50% chance to be zero, and a 50% chance to be one.
    If all elements are False, -1 is returned.

    Parameters
    ----------
    valids : np.array(2d)
        A 2d-boolean matrix.

    Returns
    -------
    out : np.array(1d)
        An integer array with the randomly chosen indices.
    """
    out = np.zeros(len(valids), dtype=np.int_)
    for i in range(len(valids)):
        idx = np.flatnonzero(valids[i])
        if idx.size == 0:
            out[i] = -1
        else:
            out[i] = np.random.choice(idx)
    return out

@numba.njit
def numba_int2bool(arr, size):
    """
    Converts an ndarray of integers to a boolean representation.
    The input array has size (..., N), the output array has
    size (..., `size`), where '...' represent the same shape.

    For examples:

    * An array [0, 1] and size 3 will be converted to
      [True, True, False].
    * An array [[0, 1], [2, 3]] and size 5 will be converted
      to [[True, True, False, False, False], [False, False, True, True, False]].

    .. note::
        Every element in arr must be strictly smaller than size.

    Parameters
    ----------
    arr : np.array(nd)
        Any nd-array with integers smaller than size.
    size : int
        The size of the last dimension in the output array. All
        elements in `arr` must be strictly smaller than this number.
    
    Returns
    -------
    out : np.array(nd)
        An nd-array with booleans. The last dimension is equal
        to `size`, the other dimensions are all but the last
        dimension of `arr`
    """

    # Store the original shape
    original_shape = arr.shape

    # Reshape existing array to keep only last dimension
    n = np.prod(np.array(arr.shape[:-1]))
    arr = arr.reshape(n, arr.shape[-1])

    # Create the output array
    out = np.zeros((n, size), dtype=np.bool_)

    # Convert to boolean
    for i in range(arr.shape[0]):
        out[i, arr[i]] = True

    # Return the reshaped array
    return out.reshape(*original_shape[:-1], size)
