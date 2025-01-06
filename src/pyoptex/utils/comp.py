"""
Module for utility functions related to computational formulas.
"""
import multiprocessing
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
