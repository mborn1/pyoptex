import numpy as np
import numba

def set_seed(n):
    np.random.seed(42)
    @numba.njit
    def _set_seed(value):
        np.random.seed(value)
    _set_seed(42)