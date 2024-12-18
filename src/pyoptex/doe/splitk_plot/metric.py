import numpy as np
import numba
from numba.experimental import jitclass
from numba import types
import warnings

from .utils import obs_var
from .cov import no_cov
from .init import init_random
from .formulas import det_update_UD, inv_update_UD, inv_update_UD_no_P, compute_update_UD
from ..utils.comp import outer_integral

class Metric:

    def preinit(self, params):
        pass

    def _init(self, Y, X, params):
        raise NotImplementedError('Must implement an _init function when using update formulas')

    def init(self, Y, X, params):
        if params.compute_update:
            self._init(Y, X, params)

    def _update(self, Y, X, params, update):
        raise NotImplementedError('Must implement an _update function when using update formulas')

    def update(self, Y, X, params, update):
        if params.compute_update:
            # Use update formulas
            return self._update(Y, X, params, update)

        else:
            # Compute from scratch
            new_metric = self.call(Y, X, params)
            metric_update = new_metric - update.old_metric

        return metric_update

    def _accepted(self, Y, X, params, update):
        raise NotImplementedError('Must implement an _accepted function when using update formulas')

    def accepted(self, Y, X, params, update):
        if params.compute_update:
            return self._accepted(Y, X, params, update)

    def call(self, Y, X, params):
        raise NotImplementedError('Must implement a call function')

class Dopt(Metric):
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
    def __init__(self, cov=None):
        self.cov = cov or no_cov
        self.Minv = None
        self.P = None
        self.U = None
        self.D = None

    def call(self, Y, X, params):
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X

        # Compute D-optimality
        return np.power(
            np.product(np.maximum(np.linalg.det(M), 0)), 
            1/(X.shape[1] * len(params.Vinv))
        )

    def _init(self, Y, X, params):
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def _update(self, Y, X, params, update):
        # Covariate expansion
        _, X = self.cov(Y, X) 
        _, Xi_old = self.cov(
            np.broadcast_to(update.old_coord, (len(update.Xi_old), len(update.old_coord))), 
            update.Xi_old,
            subset=update.runs
        )

        # Compute U, D update
        self.U, self.D = compute_update_UD(
            update.level, update.grp, Xi_old, X,
            params.plot_sizes, params.c, params.thetas, params.thetas_inv
        )

        # Compute change in determinant
        du, self.P = det_update_UD(self.U, self.D, self.Minv)
        if du > 0:
            # Compute power
            duu = np.power(np.prod(du), 1/(X.shape[1] * len(self.Minv)))

            # Return update as addition
            metric_update = (duu - 1) * update.old_metric
        else:
            metric_update = -update.old_metric

        return metric_update

    def _accepted(self, Y, X, params, update):
        # Update Minv
        try:
            self.Minv -= inv_update_UD(self.U, self.D, self.Minv, self.P)
        except np.linalg.LinAlgError as e:
            warnings.warn('Update formulas are very unstable for this problem, try rerunning without update formulas', RuntimeWarning)
            raise e
 
class Aopt(Metric):
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

    def call(self, Y, X, params):
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        if np.linalg.matrix_rank(X) >= X.shape[1]:
            # Compute average trace
            trace = np.mean(np.trace(np.linalg.inv(M), axis1=-2, axis2=-1))

            # Invert for minimization
            return -trace
        return -np.inf

    def _init(self, Y, X, params):
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def _update(self, Y, X, params, update):
        # Covariate expansion
        _, X = self.cov(Y, X)
        _, Xi_old = self.cov(
            np.broadcast_to(update.old_coord, (len(update.Xi_old), len(update.old_coord))), 
            update.Xi_old,
            subset=update.runs
        )

        # Compute U, D update
        U, D = compute_update_UD(
            update.level, update.grp, Xi_old, X,
            params.plot_sizes, params.c, params.thetas, params.thetas_inv
        )

        # Compute update to Minv
        try:
            self.Mup = inv_update_UD_no_P(U, D, self.Minv)
        except np.linalg.LinAlgError as e:
            # Infeasible design
            return -np.inf

        # Compute update to metric (double negation with update)
        metric_update = np.mean(np.trace(self.Mup, axis1=-2, axis2=-1))

        # Numerical instability (negative trace of variances)
        if metric_update > -update.old_metric:
            metric_update = -np.inf

        return metric_update

    def _accepted(self, Y, X, params, update):
        # Update Minv
        self.Minv -= self.Mup

class Iopt(Metric):
    """
    The I-optimality criterion.
    Computes the average (average) prediction variance if multiple Vinv are provided.

    Attributes
    ----------
    moments : np.array(2d)
        The moments matrix.
    samples : np.array(2d)
        The covariate expanded samples for the moments matrix.
    n : int
        The number of samples.
    Minv : np.array(3d)
        The inverse of the information matrix. Used as a cache.
    Mup : np.array(3d)
        The update to the inverse of the information matrix. Used as a cache.
    """
    def __init__(self, n=10000, cov=None, complete=True):
        self.complete = complete
        self.cov = cov or no_cov
        self.moments = None
        self.n = n
        self.Minv = None
        self.Mup = None

    def preinit(self, params):
        # Create the random samples
        samples = init_random(params, self.n, complete=self.complete)
        self.samples = params.fn.Y2X(samples)

        # Expand covariates
        _, self.samples = self.cov(samples, self.samples, random=True)

        # Compute moments matrix and normalization factor
        self.moments = outer_integral(self.samples)  # Correct up to volume factor (Monte Carlo integration), can be ignored

    def call(self, Y, X, params):
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Apply covariates
        M = X.T @ params.Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        if np.linalg.matrix_rank(X) >= X.shape[1]:
            # Compute average trace (normalized)
            trace = np.mean(np.trace(np.linalg.solve(
                M, 
                np.broadcast_to(self.moments, (params.Vinv.shape[0], *self.moments.shape))
            ), axis1=-2, axis2=-1))

            # Invert for minimization
            return -trace 
        return -np.inf 

    def _init(self, Y, X, params):
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def _update(self, Y, X, params, update):
        # Covariate expansion
        _, X = self.cov(Y, X)
        _, Xi_old = self.cov(
            np.broadcast_to(update.old_coord, (len(update.Xi_old), len(update.old_coord))), 
            update.Xi_old,
            subset=update.runs
        )

        # Compute U, D update
        U, D = compute_update_UD(
            update.level, update.grp, Xi_old, X,
            params.plot_sizes, params.c, params.thetas, params.thetas_inv
        )

        # Compute update to Minv
        try:
            self.Mup = inv_update_UD_no_P(U, D, self.Minv)
        except np.linalg.LinAlgError as e:
            # Infeasible design
            return -np.inf

        # Compute update to metric (double negation with update)
        metric_update = np.mean(np.sum(self.Mup * self.moments.T, axis=(1, 2)))

        # Numerical instability (negative variance)
        if metric_update > -update.old_metric:
            metric_update = -np.inf

        return metric_update

    def _accepted(self, Y, X, params, update):
        # Update Minv
        self.Minv -= self.Mup

