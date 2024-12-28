"""
Module for all metrics of the split^k-plot algorithm
"""

import warnings

import numpy as np

from ..metric import (
    Dopt as Dopto, 
    Aopt as Aopto, 
    Iopt as Iopto,
    Aliasing as Aliasingo,
)
from .formulas import (compute_update_UD, det_update_UD, inv_update_UD,
                       inv_update_UD_no_P)


class SplitkPlotMetricMixin:
    """
    The base mixin class for a splitk_plot metric.
    To be used in multiple inheritance together with
    :py:class:`Metric <pyoptex.doe.fixed_structure.metric.Metric>` as
    `class MyCustomMetric(SplitkPlotMetricMixin, Metric)`.
    """

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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        """
        if params.compute_update:
            return self._init(Y, X, params)
        return super().init(Y, X, params)

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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
            The update being applied to the state.
        """
        if params.compute_update:
            return self._accepted(Y, X, params, update)


class Dopt(SplitkPlotMetricMixin, Dopto):
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
            The update being applied to the state.
        """
        # Update Minv
        try:
            self.Minv -= inv_update_UD(self.U, self.D, self.Minv, self.P)
        except np.linalg.LinAlgError as e:
            warnings.warn('Update formulas are very unstable for this problem, try rerunning without update formulas', RuntimeWarning)
            raise e
 
class Aopt(SplitkPlotMetricMixin, Aopto):
    """
    The A-optimality criterion.
    Computes the average trace if multiple Vinv are provided.

    Attributes
    ----------
    cov : func(Y, X)
        A function computing the covariate parameters
        and potential extra random effects.
    W : None or np.array(1d)
        The weights for computing A-optimality.
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
        super().__init__(W, cov)
        self.Minv = None
        self.Mup = None

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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
            The update being applied to the state.
        """
        # Update Minv
        self.Minv -= self.Mup

class Iopt(SplitkPlotMetricMixin, Iopto):
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
        super().__init__(n, cov, complete)
        self.Minv = None
        self.Mup = None

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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
            The optimization parameters.
        update : :py:class:`Update <pyoptex.doe.fixed_structure.splitk_plot.utils.Update>`
            The update being applied to the state.
        """
        # Update Minv
        self.Minv -= self.Mup

class Aliasing(SplitkPlotMetricMixin, Aliasingo):
    """
    The sum of squares criterion for the weighted alias matrix.
    Computes the mean in case multiple Vinv are provided.

    The `effects` indices from the model matrix are aliased
    against `alias` indices from the model matrix.

    Attributes
    ----------
    cov : func(Y, X, Zs, Vinv, costs)
        A function computing the covariate parameters
        and potential extra random effects.
    W : np.array(2d)
        A potential weighting matrix for the elements in aliasing matrix A.
    effects : np.array(1d)
        The indices of the effects in the model matrix to alias from.
    alias : np.array(1d)
        The indices of the effects in the model matrix to alias to.
    """
    pass
