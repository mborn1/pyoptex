"""
Module for utility functions related to computational formulas.
"""

import multiprocessing

import numpy as np
from scipy.ndimage import uniform_filter1d

from pyoptex.utils._comp_cy import choice_bool, int2bool_cython_impl, outer_integral_cython_impl  # noqa: F401


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
    return outer_integral_cython_impl(np.ascontiguousarray(arr))


def int2bool(arr, size):
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

    # Convert to boolean
    out = int2bool_cython_impl(arr, n, size, arr.shape[1])

    # Return the reshaped array
    return out.reshape(*original_shape[:-1], size)


def timeout(func, *args, timeout=1, default=None):
    """
    Sets a timeout on a function by using a ThreadPool with
    one thread. If the function did not complete
    before the timeout, the default value is returned.

    Parameters
    ----------
    func : func
        The function to run.
    args : iterable
        The arguments to pass to the function.
    timeout : int
        The timeout in seconds.
    default : obj
        Any object to be returned if the function does not
        complete in time.

    Returns
    -------
    result : obj
        The result from the function or default if not completed in time.
    """
    p = multiprocessing.pool.ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        return default


def find_knee(metric, min_distance=0.05, lo=0.01, hi=0.90):
    """
    Find the knee point in a monotonically increasing array using
    the maximum perpendicular distance from the diagonal.

    Parameters
    ----------
    metric : np.array(1d)
        A sorted (ascending) array of values.
    min_distance : float
        Minimum normalized distance from the diagonal to consider
        the knee meaningful. If the curve is approximately linear
        (max distance below this threshold), returns 0.
    lo : float
        Lower bound of the search range as a fraction of the array
        length. The knee is never placed before this point, ensuring
        at least (1 - lo) of the data is always retained.
    hi : float
        Upper bound of the search range as a fraction of the array
        length. The knee is never placed past this point, ensuring
        at least (1 - hi) of the data is always retained.

    Returns
    -------
    idx : int
        The index of the knee point.
    """
    y = np.asarray(metric, dtype=float)
    n = len(y)
    if n < 10:
        return 0

    smooth_window = max(3, n // 100)
    y = uniform_filter1d(y, size=smooth_window)

    x = np.linspace(0, 1, n)
    y_range = y[-1] - y[0]

    if y_range < 1e-12:
        return 0

    y_norm = (y - y[0]) / y_range
    dx = 1.0
    dy = y_norm[-1] - y_norm[0]
    line_len = np.sqrt(dx**2 + dy**2)
    dist = np.abs(dy * x - dx * y_norm + y_norm[0]) / line_len

    lo = max(1, int(n * lo))
    hi = int(n * hi)
    best_idx = lo + np.argmax(dist[lo:hi])

    # If the curve is approximately linear, there is no real knee
    if dist[best_idx] < min_distance:
        return 0
    
    return best_idx
