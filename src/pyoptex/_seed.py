import numba
import numpy as np


def set_seed(n):
    """
    Sets the seed of the program for both numpy and numba.

    Parameters
    ----------
    n : int
        The seed.
    """
    np.random.seed(n)
    @numba.njit
    def _set_seed(value):
        np.random.seed(value)
    _set_seed(n)