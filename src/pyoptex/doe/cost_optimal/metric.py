"""
Module for all metrics of the cost optimal designs.
"""

import numpy as np

from ...utils.comp import outer_integral
from .cov import no_cov
from .init import init


class Metric:
    """
    The base class for a metric

    Attributes
    ----------
    cov : func(Y, X, Zs, Vinv, costs)
        A function computing the covariate parameters
        and potential extra random effects.
    """
    def __init__(self, cov=None):
        """
        Creates the metric

        Parameters
        ----------
        cov : func(Y, X, Zs, Vinv, costs)
            The covariance function
        """
        self.cov = cov or no_cov

    def init(self, params):
        """
        Initializes the metric before optimization.

        Parameters
        ----------
        params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
            The simulation parameters
        """
        pass

    def call(self, Y, X, Zs, Vinv, costs):
        """
        Computes the metric for a given design.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        Zs : list(np.array(1d))
            The grouping matrices
        Vinv : np.array(3d)
            The inverses of the multiple covariance matrices for each
            set of a-priori variance ratios.
        costs : list(np.array(1d), float, np.array(1d))
            The list of different costs.

        Returns
        -------
        metric : float
            The value of the criterion.
        """
        raise NotImplementedError('Must implement a call function')

class Dopt(Metric):
    """
    The D-optimality criterion.
    Computes the geometric mean in case multiple Vinv are provided.

    Attributes
    ----------
    cov : func(Y, X, Zs, Vinv, costs)
        A function computing the covariate parameters
        and potential extra random effects.
    """
    def call(self, Y, X, Zs, Vinv, costs):
        """
        Computes the D-optimality criterion for a given design.
        Computes the geometric mean in case multiple Vinv are provided.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        Zs : list(np.array(1d))
            The grouping matrices
        Vinv : np.array(3d)
            The inverses of the multiple covariance matrices for each
            set of a-priori variance ratios.
        costs : list(np.array(1d), float, np.array(1d))
            The list of different costs.

        Returns
        -------
        metric : float
            The D-optimality criterion.
        """
        # Compute covariates
        _, X, _, Vinv = self.cov(Y, X, Zs, Vinv, costs)
        M = X.T @ Vinv @ X

        # Compute geometric mean of determinants
        return np.power(
            np.prod(np.maximum(np.linalg.det(M), 0)), 
            1/(X.shape[1] * len(Vinv))
        )

class Aopt(Metric):
    """
    The A-optimality criterion.
    Computes the average trace if multiple Vinv are provided.

    Attributes
    ----------
    cov : func(Y, X, Zs, Vinv, costs)
        A function computing the covariate parameters
        and potential extra random effects.
    W : np.array(1d)
        A weights matrix for the trace of the inverse of the information matrix.
    """
    def __init__(self, cov=None, W=None):
        """
        Creates the metric

        Parameters
        ----------
        cov : func(Y, X, Zs, Vinv, costs)
            The covariance function
        W : np.array(1d)
            The weights for the trace of the inverse 
            of the information matrix.
        """
        super().__init__(cov)
        self.W = W

    def call(self, Y, X, Zs, Vinv, costs):
        """
        Computes the A-optimality criterion for a given design.
        Computes the average trace if multiple Vinv are provided.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        Zs : list(np.array(1d))
            The grouping matrices
        Vinv : np.array(3d)
            The inverses of the multiple covariance matrices for each
            set of a-priori variance ratios.
        costs : list(np.array(1d), float, np.array(1d))
            The list of different costs.

        Returns
        -------
        metric : float
            The negative of the A-optimality criterion.
        """
        # Compute covariates
        _, X, _, Vinv = self.cov(Y, X, Zs, Vinv, costs)
        M = X.T @ Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        if np.linalg.matrix_rank(M[0]) >= M.shape[1]:
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
    Computes the average (average) prediction variance
    if multiple Vinv are provided.

    .. note::
        The covariance function is called by passing 
        random=True for initialization. The function 
        should not use grouping or costs in this case.

    Attributes
    ----------
    cov : func(Y, X, Zs, Vinv, costs)
        A function computing the covariate parameters
        and potential extra random effects.
    moments : np.array(2d)
        The moments matrix.
    samples : np.array(2d)
        The covariate expanded samples for the moments matrix.
    n : int
        The number of samples.
    complete : bool
        Whether to initialize the samples between -1 and 1, 
        or from the given coordinates.
    """
    def __init__(self, n=10000, cov=None, complete=True):
        """
        Creates the metric

        Parameters
        ----------
        n : int
            The number of samples for computing the moments
            matrix.
        cov : func(Y, X, Zs, Vinv, costs)
            The covariance function
        complete : bool
            Whether to use the fixed coordinates or initialize
            the moments matrix from completely random samples.
        """
        super().__init__(cov)
        self.moments = None
        self.samples = None
        self.n = n
        self.complete = complete
        self.initialized_ = False

    def init(self, params):
        """
        Initializes the I-optimal metric if not yet
        initialized.

        Parameters
        ----------
        params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
            The simulation parameters
        """
        if not self.initialized_:
            # Create the random samples
            samples = init(params, self.n, complete=self.complete)
            self.samples = params.fn.Y2X(samples)

            # Add random covariates
            _, self.samples, _, _ = self.cov(
                samples, self.samples, None, None, None, random=True
            )

            # Compute moments matrix and normalization factor
            # Correct up to volume factor (Monte Carlo integration), can be ignored
            self.moments = outer_integral(self.samples)  

            # Sets the initialized_ parameter
            self.initialized_ = True

    def call(self, Y, X, Zs, Vinv, costs):
        """
        Computes the I-optimal metric for a given design.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        Zs : list(np.array(1d))
            The grouping matrices
        Vinv : np.array(3d)
            The inverses of the multiple covariance matrices for each
            set of a-priori variance ratios.
        costs : list(np.array(1d), float, np.array(1d))
            The list of different costs.

        Returns
        -------
        metric : float
            The negative of the I-optimality criterion.
        """
        # Apply covariates
        _, X, _, Vinv = self.cov(Y, X, Zs, Vinv, costs)
        M = X.T @ Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        if np.linalg.matrix_rank(M[0]) >= M.shape[1]:
            # Compute average trace (normalized)
            trace = np.mean(np.trace(np.linalg.solve(
                M, 
                np.broadcast_to(
                    self.moments, 
                    (Vinv.shape[0], *self.moments.shape)
                )
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
        cov : func(Y, X, Zs, Vinv, costs)
            The covariance function
        W : np.array(1d)
            The weights for the aliasing matrix.
        """
        super().__init__(cov)
        self.W = W
        self.effects = effects
        self.alias = alias

    def call(self, Y, X, Zs, Vinv, costs):
        """
        Computes the aliasing criterion for a given design.
        Computes the average trace if multiple Vinv are provided.

        Parameters
        ----------
        Y : np.array(2d)
            The design matrix
        X : np.array(2d)
            The model matrix
        Zs : list(np.array(1d))
            The grouping matrices
        Vinv : np.array(3d)
            The inverses of the multiple covariance matrices for each
            set of a-priori variance ratios.
        costs : list(np.array(1d), float, np.array(1d))
            The list of different costs.

        Returns
        -------
        metric : float
            The negative of the aliasing criterion.
        """
        # Compute covariates
        _, X, _, Vinv = self.cov(Y, X, Zs, Vinv, costs)

        # Compute aliasing matrix
        Xeff = X[:, self.effects]
        Xa = X[:, self.alias]
        A = np.linalg.solve(Xeff.T @ Vinv @ Xeff, Xeff.T @ Vinv) @ Xa

        # Multiply by weights
        if self.W is not None:
            A *= self.W

        # Compute mean of SS
        return -np.power(
            np.mean(np.sum(np.square(A), axis=(-1, -2))), 
            1/(X.shape[1] * len(Vinv))
        )
