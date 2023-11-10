import numpy as np
import numba
from numba.experimental import jitclass
from numba import types

from .utils import obs_var
from .formulas import det_update_UD, inv_update_UD_no_P

# TODO: covariances
# TODO: multiple ratios (bayesian)
# TODO: priors and augmentation
# TODO: plot_sizes = lowest first: note in documentation!

class Dopt:
    """
    The D-optimality criterion.
    Computes the geometric mean in case multiple Vinv are provided.
    """
    def __init__(self):
        self.c = None
        self.Vinv = None
        self.Minv = None

    def init(self, Y, X, params):
        # Compute information matrix
        self.Vinv = params.Vinv
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def update(self, Y, X, update):
        # Compute change in determinant
        du = 1
        for i in range(len(self.Minv)):
            _du, _ = det_update_UD(update.U[i], update.D[i], self.Minv[i])
            du *= _du
        du = np.power(du, 1/(X.shape[1] * len(self.Vinv)))

        # Return update as addition
        return (du - 1) * update.old_metric

    def accepted(self, Y, X, update):
        # Update inv(M)
        for i in range(len(self.Minv)):
            self.Minv[i] -= inv_update_UD_no_P(update.U[i], update.D[i], self.Minv[i])

    def call(self, Y, X):
        M = X.T @ self.Vinv @ X
        return np.power(
            np.product(np.maximum(np.linalg.det(M), 0)), 
            1/(X.shape[1] * len(self.Vinv))
        )
