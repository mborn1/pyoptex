import numba
import numpy as np
from ._seed_cy import set_seed_cy

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
    set_seed_cy(n)