import numba
import numpy as np

@numba.njit
def outer_integral(arr):
    """
    Computes the integral of the outer products of the array rows (simple average)

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    arr : np.array(2d)
        The array
    
    Returns
    -------
    out : np.array(2d)
        The integral of the outer product
    """
    out = np.zeros((arr.shape[-1], arr.shape[-1]))
    for i in range(arr.shape[0]):
        out += np.expand_dims(arr[i], 1) @ np.expand_dims(arr[i], 0)
    return out / arr.shape[0]
