import numba
import numpy as np

@numba.njit
def numba_diff(x):
    """Numba version of np.diff"""
    return x[1:] - x[:-1]

@numba.njit
def numba_diff_axis0(x):
    """Numba version of np.diff along axis 0"""
    diff = np.zeros((x.shape[0] - 1, x.shape[1]))
    for i in range(x.shape[1]):
        diff[:, i] = x[1:, i] - x[:-1, i]
    return diff

@numba.njit
def numba_any_axis1(x):
    """Numba version of np.any along axis 1"""
    res = np.zeros(x.shape[0], dtype=np.bool_)
    for i in range(x.shape[1]):
        res = np.logical_or(res, x[:, i])
    return res

@numba.njit
def numba_all_axis2(x):
    """Numba version of np.all along axis 2"""
    res = np.ones((x.shape[0], x.shape[1]), dtype=np.bool_)
    for i in range(x.shape[2]):
        res = np.logical_and(res, x[:, :, i])
    return res

@numba.njit
def numba_delete_axis0(x, pos):
    """Numba version of np.delete along axis 0"""
    mask = np.ones(x.shape[0], dtype=np.bool_)
    mask[pos] = False
    return np.copy(x[mask])

@numba.njit
def numba_insert(x, pos, value):
    """Numba version of np.insert"""
    a = np.zeros(x.size + 1, dtype=x.dtype)
    a[:pos] = x[:pos]
    a[pos] = value
    a[pos+1:] = x[pos:]
    return a

@numba.njit
def numba_insert_axis0(x, pos, value):
    """Numba version of np.insert along axis 0"""
    a = np.zeros((x.shape[0] + 1, x.shape[1]), dtype=x.dtype)
    a[:pos] = x[:pos]
    a[pos] = value
    a[pos+1:] = x[pos:]
    return a

@numba.njit
def numba_take_advanced(arr, idx, out=None):
    """Numba version of numpy advanced indexing"""
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
