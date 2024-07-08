import numpy as np
import numba
from numba.experimental import jitclass
from numba import types

from .utils import obs_var
from .init import init_random
from .formulas import det_update_UD, inv_update_UD, inv_update_UD_no_P
from ..utils.comp import outer_integral

class Dopt:
    """
    The D-optimality criterion.
    Computes the geometric mean in case multiple Vinv are provided.

    Attributes
    ----------
    Minv : np.array(3d)
        The inverse of the information matrix. This is used as a cache.
    P : np.array(2d)
        The P-matrix in the update formula. This is used as a cache.
    """
    def __init__(self):
        self.Minv = None
        self.P = None

    def preinit(self, params):
        pass

    def init(self, Y, X, params):
        if params.compute_update:
            # Compute information matrix
            M = X.T @ params.Vinv @ X
            self.Minv = np.linalg.inv(M)

    def update(self, Y, X, params, update):
        if params.compute_update:
            # Compute change in determinant
            du, self.P = det_update_UD(update.U, update.D, self.Minv)
            du = np.power(np.prod(du), 1/(X.shape[1] * len(self.Minv)))

            # Return update as addition
            metric_update = (du - 1) * update.old_metric

        else:
            # Compute from scratch
            new_metric = self.call(Y, X, params)
            metric_update = new_metric - update.old_metric

        return metric_update

    def accepted(self, Y, X, params, update):
        # Update Minv
        if params.compute_update:
            self.Minv -= inv_update_UD(update.U, update.D, self.Minv, self.P)

    def call(self, Y, X, params):
        M = X.T @ params.Vinv @ X
        return np.power(
            np.product(np.maximum(np.linalg.det(M), 0)), 
            1/(X.shape[1] * len(params.Vinv))
        )

class Aopt:
    """
    The A-optimality criterion.
    Computes the average trace if multiple Vinv are provided.

    Attributes
    ----------
    Minv : np.array(3d)
        The inverse of the information matrix. This is used as a cache.
    Mup : np.array(3d)
        The update to the inverse of the information matrix. This is used as a cache.
    """
    def __init__(self):
        self.Minv = None
        self.Mup = None

    def preinit(self, params):
        pass

    def init(self, Y, X, params):
        if params.compute_update:
            # Compute information matrix
            M = X.T @ params.Vinv @ X
            self.Minv = np.linalg.inv(M)

    def update(self, Y, X, params, update):
        if params.compute_update:
            # Compute update to Minv
            try:
                self.Mup = inv_update_UD_no_P(update.U, update.D, self.Minv)
            except np.linalg.LinAlgError as e:
                # Infeasible design
                return -np.inf

            # Compute update to metric (double negation with update)
            metric_update = np.mean(np.trace(self.Mup, axis1=-2, axis2=-1))

            # Numerical instability (negative trace of variances)
            if metric_update > -update.old_metric:
                metric_update = -np.inf
        
        else:
            # Compute from scratch
            new_metric = self.call(Y, X, params)
            metric_update = new_metric - update.old_metric

        return metric_update

    def accepted(self, Y, X, params, update):
        # Update Minv
        if params.compute_update:
            self.Minv -= self.Mup

    def call(self, Y, X, params):
        # Compute covariates
        M = X.T @ params.Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        if np.linalg.matrix_rank(X) >= X.shape[1]:
            # Compute average trace
            trace = np.mean(np.trace(np.linalg.inv(M), axis1=-2, axis2=-1))

            # Invert for minimization
            return -trace
        return -np.inf

class Iopt:
    """
    The I-optimality criterion.
    Computes the average (average) prediction variance if multiple Vinv are provided.

    Attributes
    ----------
    moments : np.array(2d)
        The moments matrix.
    samples : np.array(2d)
        The covariate expanded samples for the moments matrix.
    intdx : float
        The integral over the input space for normalization.
    n : int
        The number of samples.
    Minv : np.array(3d)
        The inverse of the information matrix. Used as a cache.
    Mup : np.array(3d)
        The update to the inverse of the information matrix. Used as a cache.
    """
    def __init__(self, n=10000):
        self.moments = None
        self.intdx = None
        self.n = n

        self.Minv = None
        self.Mup = None

    def preinit(self, params):
        # Create the random samples
        samples = init_random(params, self.n, complete=True)
        self.samples = params.Y2X(samples)

        # Compute moments matrix and normalization factor
        self.moments = outer_integral(self.samples)  # Correct up to volume factor (Monte Carlo integration), can be ignored
        self.intdx = 2**np.sum(params.effect_types == 1) \
                    * np.prod(params.effect_types[params.effect_types > 1], initial=1)

    def init(self, Y, X, params):
        if params.compute_update:
            # Compute information matrix
            M = X.T @ params.Vinv @ X
            self.Minv = np.linalg.inv(M)

    def update(self, Y, X, params, update):
        if params.compute_update:
            # Compute update to Minv
            try:
                self.Mup = inv_update_UD_no_P(update.U, update.D, self.Minv)
            except np.linalg.LinAlgError as e:
                # Infeasible design
                return -np.inf

            # Compute update to metric (double negation with update)
            metric_update = np.mean(np.sum(self.Mup * self.moments.T, axis=(1, 2))) / self.intdx

            # Numerical instability (negative variance)
            if metric_update > -update.old_metric:
                metric_update = -np.inf

        else:
            # Compute from scratch
            new_metric = self.call(Y, X, params)
            metric_update = new_metric - update.old_metric

        return metric_update

    def accepted(self, Y, X, params, update):
        # Update Minv
        if params.compute_update:
            self.Minv -= self.Mup

    def call(self, Y, X, params):
        # Apply covariates
        M = X.T @ params.Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        print(X.shape, np.linalg.matrix_rank(X))
        if np.linalg.matrix_rank(X) >= X.shape[1]:
            # Compute average trace (normalized)
            trace = np.mean(np.trace(np.linalg.solve(
                M, 
                np.broadcast_to(self.moments, (params.Vinv.shape[0], *self.moments.shape))
            ), axis1=-2, axis2=-1)) / self.intdx

            # Invert for minimization
            return -trace 
        return -np.inf 


