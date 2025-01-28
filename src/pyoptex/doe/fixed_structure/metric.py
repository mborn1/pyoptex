"""
Module for all metrics of fixed structure designs.
"""

import numpy as np

from ...utils.comp import outer_integral
from .cov import no_cov
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
            The optimization parameters.
        """
        pass

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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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
    """

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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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
            np.prod(np.maximum(np.linalg.det(M), 0)),
            1/(X.shape[1] * len(params.Vinv))
        )
 
class Aopt(Metric):
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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

class Iopt(Metric):
    """
    The I-optimality criterion.
    Computes the average (average) prediction variance if multiple Vinv are provided.

    Attributes
    ----------
    cov : func(Y, X)
        A function computing the covariate parameters
        and potential extra random effects.
    moments : np.array(2d)
        The moments matrix.
    samples : np.array(2d)
        The covariate expanded samples for the moments matrix.
    n : int
        The number of samples.
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

    def preinit(self, params):
        """
        Pre-initializes the metric

        Parameters
        ----------
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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

class Aliasing(Metric):
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
    def __init__(self, effects, alias, cov=None, W=None):
        """
        Creates the metric

        Parameters
        ----------
        effects : np.array(1d)
            The indices of the effects in the model matrix to alias from.
        alias : np.array(1d)
            The indices of the effects in the model matrix to alias to.
        cov : func(Y, X)
            The covariance function
        W : np.array(1d)
            The weights for the aliasing matrix.
        """
        super().__init__(cov)
        self.W = W
        self.effects = effects
        self.alias = alias

    def call(self, Y, X, params):
        """
        Computes the aliasing criterion.
        Computes the average (average) prediction variance if 
        multiple Vinv are provided.

        Parameters
        ----------
        Y : np.array(2d)
            The updated design matrix.
        X : np.array(2d)
            The updated model matrix.
        params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
            The optimization parameters.
        
        Returns
        -------
        metric : float
            The negative of the aliasing criterion value.
        """
        # Compute covariates
        _, X = self.cov(Y, X)

        # Compute aliasing matrix
        Xeff = X[:, self.effects]
        Xa = X[:, self.alias]
        A = np.linalg.solve(Xeff.T @ params.Vinv @ Xeff, Xeff.T @ params.Vinv) @ Xa

        # Multiply by weights
        if self.W is not None:
            A *= self.W

        # Compute mean of SS
        return -np.power(
            np.mean(np.sum(np.square(A), axis=(-1, -2))), 
            1/(X.shape[1] * len(params.Vinv))
        )

