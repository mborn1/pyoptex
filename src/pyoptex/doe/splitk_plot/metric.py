import numpy as np
import numba
from numba.experimental import jitclass
from numba import types

from .utils import obs_var
from .formulas import det_update_UD, inv_update_UD

# TODO: priors and augmentation
# TODO: add I and A optimality
# TODO: covariances
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
        self.P = None

    def init(self, Y, X, params):
        # Compute information matrix
        self.Vinv = params.Vinv
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def update(self, Y, X, update):
        # Compute change in determinant
        du, self.P = det_update_UD(update.U, update.D, self.Minv)
        du = np.power(np.prod(du), 1/(X.shape[1] * len(self.Vinv)))

        # Return update as addition
        return (du - 1) * update.old_metric

    def accepted(self, Y, X, update):
        # Update Minv
        self.Minv -= inv_update_UD(update.U, update.D, self.Minv, self.P)

    def call(self, Y, X):
        M = X.T @ self.Vinv @ X
        return np.power(
            np.product(np.maximum(np.linalg.det(M), 0)), 
            1/(X.shape[1] * len(self.Vinv))
        )
