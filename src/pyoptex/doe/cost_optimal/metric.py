import numpy as np
from math import prod

from .init import init
from .cov import no_cov
from ..utils.comp import outer_integral

class Dopt:
    """
    The D-optimality criterion.
    Computes the geometric mean in case multiple Vinv are provided.

    Attributes:
    cov : func
        A function computing the covariate parameters and potential extra random effects.
    """
    def __init__(self, cov=None):
        self.cov = cov or no_cov

    def init(self, params):
        pass

    def call(self, Y, X, Zs, Vinv, costs):
        # Compute covariates
        _, X, _, Vinv = self.cov(Y, X, Zs, Vinv, costs)
        M = X.T @ Vinv @ X

        # Compute geometric mean of determinants
        return np.power(
            np.product(np.maximum(np.linalg.det(M), 0)), 
            1/(X.shape[1] * len(Vinv))
        )

class Aopt:
    """
    The A-optimality criterion.
    Computes the average trace if multiple Vinv are provided.

    Attributes:
    cov : func
        A function computing the covariate parameters and potential extra random effects.
    """
    def __init__(self, cov=None, W=None):
        self.cov = cov or no_cov
        self.W = W

    def init(self, params):
        pass

    def call(self, Y, X, Zs, Vinv, costs):
        # Compute covariates
        _, X, _, Vinv = self.cov(Y, X, Zs, Vinv, costs)
        M = X.T @ Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        if np.linalg.matrix_rank(X) == X.shape[1]:
            # Extrace variances
            diag = np.diag(np.linalg.inv(M), axis1=-2, axis2=-1)

            # Weight
            if self.W is not None:
                diag *= self.W

            # Compute average
            trace = np.mean(np.sum(diag, axis=-1))

            # Invert for minimization
            return -trace
        return -np.inf

class Iopt:
    """
    The I-optimality criterion.
    Computes the average (average) prediction variance if multiple Vinv are provided.

    .. note::
        The covariance function is called by passing random=True for initialization. The
        function should not use grouping or costs in this case.

    Attributes:
    cov : func
        A function computing the covariate parameters and potential extra random effects.
    moments : np.array(2d)
        The moments matrix.
    samples : np.array(2d)
        The covariate expanded samples for the moments matrix.
    intdx : float
        The integral over the input space for normalization.
    n : int
        The number of samples.
    """
    def __init__(self, n=10000, cov=None):
        self.cov = cov or no_cov
        self.moments = None
        self.intdx = None
        self.n = n

    def init(self, params):
        # Create the random samples
        samples = init(params, self.n, complete=True)
        self.samples = params.Y2X(samples)

        # Add random covariates
        _, self.samples, _, _ = self.cov(samples, self.samples, None, None, None, random=True)

        # Compute moments matrix and normalization factor
        self.moments = outer_integral(self.samples)  # Correct up to volume factor (Monte Carlo integration), can be ignored
        self.intdx = 2**np.sum(params.effect_types == 1) \
                    * np.prod(params.effect_types[params.effect_types > 1], initial=1)

    def call(self, Y, X, Zs, Vinv, costs):
        # Apply covariates
        _, X, _, Vinv = self.cov(Y, X, Zs, Vinv, costs)
        M = X.T @ Vinv @ X

        # Check if invertible (more stable than relying on inverse)
        if np.linalg.matrix_rank(X) == X.shape[1]:
            # Compute average trace (normalized)
            trace = np.mean(np.trace(np.linalg.solve(
                M, 
                np.broadcast_to(self.moments, (Vinv.shape[0], *self.moments.shape))
            ), axis1=-2, axis2=-1)) / self.intdx

            # Invert for minimization
            return -trace 
        return -np.inf

class SumSquares:
    """
    The sum of squares criterion.
    Computes the geometric mean in case multiple Vinv are provided.

    Attributes:
    cov : func
        A function computing the covariate parameters and potential extra random effects.
    W : np.array
        A potential weighting matrix for the elements in M.
    """
    def __init__(self, cov=None, W=None, inv=True):
        self.cov = cov or no_cov
        self.W = W
        self.inv = inv

    def init(self, params):
        pass

    def call(self, Y, X, Zs, Vinv, costs):
        # Compute covariates
        _, X, _, Vinv = self.cov(Y, X, Zs, Vinv, costs)
        M = X.T @ Vinv @ X

        # Invert M
        if self.inv:
            M = np.linalg.inv(M)

        # Multiply by weights
        if self.W is not None:
            M *= W

        # Compute geometric mean of determinants
        return np.power(
            np.product(np.sum(np.square(M), axis=[-1, -2])), 
            1/(X.shape[1] * len(Vinv))
        )
