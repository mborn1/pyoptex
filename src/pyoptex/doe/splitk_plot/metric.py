"""
Module for all metrics of the split^k-plot algorithm
"""

import warnings

import numpy as np

from ..utils.comp import outer_integral
from .cov import no_cov
from .formulas import (compute_update_UD, det_update_UD, inv_update_UD,
                       inv_update_UD_no_P)
from .init import init_random


class Metric:
    """
    The base class for a metric

    Attributes
    ----------
    cov : func(Y, X)
        A function computing the covariate parameters
        and potential extra random effects.
    """
    def __init__(self, cov=None):
        """
        Creates the metric

        Parameters
        ----------
        cov : func(Y, X)
            The covariance function
        """
        self.cov = cov or no_cov

    def preinit(self, params):
        """
        Pre-initializes the metric

        Parameters
        ----------
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        """
        pass

    def _init(self, Y, X, params):
        """
        Internal function to initialize the metric when
        using update formulas.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        """
        pass

    def init(self, Y, X, params):
        """
        Initializes the metric for each random
        initialization of the coordinate-exchange algorithm.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        """
        if params.compute_update:
            self._init(Y, X, params)

    def _update(self, Y, X, params, update):
        """
        Computes the update to the metric according to
        `update`. The update to the metric is of the
        form :math:`m_{new} = m_{old} + up`. This is
        only called for when applying update formulas.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.

        Returns
        -------
        up : float
            The update to the metric.
        """
        # Compute from scratch
        new_metric = self.call(Y, X, params)
        metric_update = new_metric - update.old_metric
        return metric_update

    def update(self, Y, X, params, update):
        """
        Computes the update to the metric according to
        `update`. The update to the metric is of the
        form :math:`m_{new} = m_{old} + up`.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.

        Returns
        -------
        up : float
            The update to the metric.
        """
        if params.compute_update:
            # Use update formulas
            return self._update(Y, X, params, update)

        else:
            # Compute from scratch
            new_metric = self.call(Y, X, params)
            metric_update = new_metric - update.old_metric

        return metric_update

    def _accepted(self, Y, X, params, update):
        """
        Updates the internal state when the updated
        design was accepted (and therefore better).
        Only called when considering update formulas.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.
        """
        pass

    def accepted(self, Y, X, params, update):
        """
        Updates the internal state when the updated
        design was accepted (and therefore better).

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.
        """
        if params.compute_update:
            return self._accepted(Y, X, params, update)

    def call(self, Y, X, params):
        """
        Computes the criterion for the provided
        design and model matrices.

        .. note::
            The metric is maximized in the algorithm,
            so the in case of minimization, the negative
            value should be returned.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix.
        X : np.array(2d)
            The updated model matrix.
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        
        Returns
        -------
        metric : float
            The result metric (to be maximized).
        """
        raise NotImplementedError('Must implement a call function')

class Dopt(Metric):
    """
    The D-optimality criterion.
    Computes the geometric mean in case multiple Vinv are provided.

    Attributes
    ----------
    cov : func(Y, X)
        A function computing the covariate parameters
        and potential extra random effects.
    Minv : np.array(3d)
        The inverses of the information matrices.
    P : np.array(2d)
        The P-matrix in the update formula.
    U : np.array(2d)
        The U-matrix in the update formula.
    D : np.array(2d)
        The D-matrix in the update formula.
    """
    def __init__(self, cov=None):
        """
        Creates the metric

        Parameters
        ----------
        cov : func(Y, X)
            The covariance function
        """
        super().__init__(cov)
        self.Minv = None
        self.P = None
        self.U = None
        self.D = None

    def call(self, Y, X, params):
        """
        Computes the D-optimality criterion.
        Computes the geometric mean in case multiple Vinv are provided.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix.
        X : np.array(2d)
            The updated model matrix.
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        
        Returns
        -------
        metric : float
            The D-optimality criterion value.
        """
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
        """
        Internal function to initialize the metric when
        using update formulas.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        """
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def _update(self, Y, X, params, update):
        """
        Computes the update to the metric according to
        `update`. The update to the metric is of the
        form :math:`m_{new} = m_{old} + up`.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.

        Returns
        -------
        up : float
            The update to the metric.
        """
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
        """
        Updates the internal Minv attribute
        according to the last computed update.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.
        """
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
    cov : func(Y, X)
        A function computing the covariate parameters
        and potential extra random effects.
    Minv : np.array(3d)
        The inverses of the information matrices.
    Mup : np.array(3d)
        The update for the inverse information matrix.
    """
    def __init__(self, W=None, cov=None):
        """
        Creates the metric

        Parameters
        ----------
        W : None or np.array(1d)
            The weights for computing A-optimality.
        cov : func(Y, X)
            The covariance function.
        """
        super().__init__(cov)
        self.W = W
        self.Minv = None
        self.Mup = None

    def call(self, Y, X, params):
        """
        Computes the A-optimality criterion.
        Computes the average trace if multiple Vinv are provided.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix.
        X : np.array(2d)
            The updated model matrix.
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        
        Returns
        -------
        metric : float
            The negative of the A-optimality criterion value.
        """
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        if np.linalg.matrix_rank(X) >= X.shape[1]:
            # Extrace variances
            Minv = np.linalg.inv(M)
            diag = np.array([np.diag(m) for m in Minv])

            # Weight
            if self.W is not None:
                diag *= self.W

            # Compute average
            trace = np.mean(np.sum(diag, axis=-1))

            # Invert for minimization
            return -trace
        return -np.inf

    def _init(self, Y, X, params):
        """
        Internal function to initialize the metric when
        using update formulas.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        """
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def _update(self, Y, X, params, update):
        """
        Computes the update to the metric according to
        `update`. The update to the metric is of the
        form :math:`m_{new} = m_{old} + up`.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.

        Returns
        -------
        up : float
            The update to the metric.
        """
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
        
        # Extrace variances
        diag = np.array([np.diag(m) for m in self.Mup])

        # Weight
        if self.W is not None:
            diag *= self.W

        # Compute average
        metric_update = np.mean(np.sum(diag, axis=-1))

        # Numerical instability (negative trace of variances)
        if metric_update > -update.old_metric:
            metric_update = -np.inf

        return metric_update

    def _accepted(self, Y, X, params, update):
        """
        Updates the internal Minv attribute
        according to the last computed update.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.
        """
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
        """
        Creates the metric

        Parameters
        ----------
        n : int
            The number of samples to compute the moments matrix.
        cov : func(Y, X)
            The covariance function
        complete : bool
            Whether to only use the coordinates or completely
            randomly initialize the samples to generate the
            moments matrix.
        """
        super().__init__(cov)
        self.complete = complete
        self.moments = None
        self.n = n
        self.Minv = None
        self.Mup = None

    def preinit(self, params):
        """
        Pre-initializes the metric

        Parameters
        ----------
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        """
        # Create the random samples
        samples = init_random(params, self.n, complete=self.complete)
        self.samples = params.fn.Y2X(samples)

        # Expand covariates
        _, self.samples = self.cov(samples, self.samples, random=True)

        # Compute moments matrix and normalization factor
        self.moments = outer_integral(self.samples)  # Correct up to volume factor (Monte Carlo integration), can be ignored

    def call(self, Y, X, params):
        """
        Computes the I-optimality criterion.
        Computes the average (average) prediction variance if 
        multiple Vinv are provided.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix.
        X : np.array(2d)
            The updated model matrix.
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        
        Returns
        -------
        metric : float
            The negative of the I-optimality criterion value.
        """
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
        """
        Internal function to initialize the metric when
        using update formulas.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        """
        # Covariate expansion
        _, X = self.cov(Y, X)

        # Compute information matrix
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def _update(self, Y, X, params, update):
        """
        Computes the update to the metric according to
        `update`. The update to the metric is of the
        form :math:`m_{new} = m_{old} + up`.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.

        Returns
        -------
        up : float
            The update to the metric.
        """
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
        """
        Updates the internal Minv attribute
        according to the last computed update.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix
        X : np.array(2d)
            The updated model matrix
        params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
            The optimization parameters.
        update : :py:class:`pyoptex.doe.splitk_plot.utils.Update`
            The update being applied to the state.
        """
        # Update Minv
        self.Minv -= self.Mup

